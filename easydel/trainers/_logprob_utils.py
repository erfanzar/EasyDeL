# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import typing as tp

import jax
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array


def resolve_lmhead_projection_module(model: tp.Any) -> tp.Any | None:
    """Resolve the module that owns the LM-head projection helpers.

    Many EasyDeL model wrappers expose a ``compute_lm_logits`` method that
    projects hidden states into vocabulary-sized logits.  Depending on the
    model hierarchy the method may live on the top-level model object itself
    or on an inner ``model.model`` sub-module.  This helper walks up to two
    levels of nesting to find the first object that exposes the method.

    This is used by the chunked scoring utilities
    (``compute_sequence_scores_from_hidden_states`` and
    ``compute_per_token_logps_and_entropies_from_hidden_states``) so that
    they can project hidden states through the LM head without materializing
    the full forward pass again.

    Args:
        model: An EasyDeL model instance (or any object that may expose
            ``compute_lm_logits`` directly or via a ``.model`` attribute).

    Returns:
        The module that owns ``compute_lm_logits``, or ``None`` if neither
        the model nor its ``.model`` attribute exposes the method.
    """

    if hasattr(model, "compute_lm_logits"):
        return model
    if hasattr(model, "model") and hasattr(model.model, "compute_lm_logits"):
        return model.model
    return None


def resolve_lmhead_chunksize(model: tp.Any) -> int | None:
    """Return the configured LM-head token chunk size when headless scoring is supported.

    The chunk size controls how many tokens along the *sequence* dimension are
    projected through the LM head in a single step.  Smaller values reduce
    peak memory at the cost of more loop iterations.  The value is read from
    the model's ``config.lmhead_chunksize`` attribute.

    The function first checks that the model actually supports headless
    scoring (via :func:`resolve_lmhead_projection_module`).  If the model
    does not expose ``compute_lm_logits``, or if the configured chunk size
    is ``None`` or non-positive, the function returns ``None`` to signal
    that chunked headless scoring should not be used.

    Args:
        model: An EasyDeL model instance.  The config is looked up on the
            top-level model first, falling back to the inner projection
            module.

    Returns:
        A positive ``int`` chunk size if headless scoring is available and
        a valid chunk size is configured, otherwise ``None``.
    """

    projection_model = resolve_lmhead_projection_module(model)
    if projection_model is None:
        return None
    config_holder = model if hasattr(model, "config") else projection_model
    chunk_size = config_holder.config.lmhead_chunksize
    if chunk_size is None:
        return None
    chunk_size = int(chunk_size)
    return chunk_size if chunk_size > 0 else None


def compute_token_logps_and_entropies_chunked(
    logits: Array,
    targets: Array,
    *,
    return_entropy: bool,
    chunk_size: int | None,
) -> tuple[Array, Array | None]:
    """Compute per-token log-probabilities (and optional entropies) without materializing a full vocab-sized log-softmax.

    For large vocabularies the standard ``log_softmax -> gather`` pattern
    requires allocating a ``[batch, seq_len, vocab_size]`` tensor of
    log-probabilities, which can be prohibitively expensive.  This function
    avoids that by chunking across the **vocabulary** dimension and using a
    numerically stable three-pass streaming algorithm:

    1. **Max pass** -- iterates over vocabulary chunks to compute
       ``logit_max = max_v logits[..., v]`` for numerical stability, using
       ``jax.lax.fori_loop`` so that only one chunk is live at a time.
    2. **Sum-exp pass** -- iterates again to compute
       ``exp_sum = sum_v exp(logits[..., v] - logit_max)``, yielding
       ``log_z = log(exp_sum) + logit_max`` (the log-partition function).
       The target-token log-probability is then
       ``logits[target] - log_z`` obtained via a single ``take_along_axis``.
    3. **Entropy pass** (optional) -- a third iteration computes the
       expected logit ``E_p[logit] = sum_v p_v * logit_v`` from which
       entropy is recovered as ``H = log_z - E_p[logit]``.

    Each inner step is wrapped with ``jax.checkpoint`` to allow XLA to
    rematerialize rather than keep all chunks resident during
    back-propagation.

    This function operates purely on pre-computed logits.  For an
    end-to-end variant that starts from hidden states and projects through
    the LM head, see :func:`compute_per_token_logps_and_entropies_from_hidden_states`.

    Args:
        logits: Float array of shape ``[..., vocab_size]`` containing raw
            (unnormalized) logits.  Typically ``[batch, seq_len, vocab_size]``.
        targets: Integer array of shape ``[...]`` (matching all but the last
            dimension of *logits*) holding the token indices whose
            log-probabilities should be gathered.
        return_entropy: If ``True``, the third pass is executed and per-token
            entropies are returned.  When ``False`` the entropy pass is
            skipped entirely and ``None`` is returned in its place.
        chunk_size: The number of vocabulary entries to process per chunk.
            If less than or equal to zero, or greater than or equal to
            ``vocab_size``, the function falls back to the dense
            ``log_softmax -> gather`` path.

    Returns:
        A tuple ``(token_log_probs, entropies)`` where:

        - **token_log_probs** -- float32 array of shape ``[...]`` containing
          the log-probability of each target token.
        - **entropies** -- float32 array of the same shape containing the
          Shannon entropy ``H(p)`` of the softmax distribution at each
          position, or ``None`` when *return_entropy* is ``False``.
    """

    vocab_size = logits.shape[-1]
    if chunk_size is None or chunk_size <= 0 or chunk_size >= vocab_size:
        log_probs = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
        token_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1).astype(jnp.float32)
        token_log_probs = jnp.squeeze(token_log_probs, axis=-1)
        if not return_entropy:
            return token_log_probs, None
        probs = jnp.exp(log_probs)
        entropies = -jnp.sum(probs * log_probs, axis=-1)
        return token_log_probs, entropies
    chunk_size = int(chunk_size)
    num_full_chunks = vocab_size // chunk_size
    tail = vocab_size - num_full_chunks * chunk_size

    def _max_step(start: int, size: int, running_max: Array) -> Array:
        chunk = lax.dynamic_slice_in_dim(logits, start, size, axis=-1).astype(jnp.float32)
        return jnp.maximum(running_max, jnp.max(chunk, axis=-1))

    _max_step = jax.checkpoint(_max_step, prevent_cse=False, static_argnums=(1,))

    def max_body(i: int, running_max: Array) -> Array:
        return _max_step(i * chunk_size, chunk_size, running_max)

    logit_max = jnp.full(logits.shape[:-1], -jnp.inf, dtype=jnp.float32)
    if num_full_chunks > 0:
        logit_max = lax.fori_loop(0, num_full_chunks, max_body, logit_max)
    if tail:
        logit_max = _max_step(num_full_chunks * chunk_size, tail, logit_max)

    def _sum_step(start: int, size: int, running_sum: Array) -> Array:
        chunk = lax.dynamic_slice_in_dim(logits, start, size, axis=-1).astype(jnp.float32)
        return running_sum + jnp.sum(jnp.exp(chunk - logit_max[..., None]), axis=-1)

    _sum_step = jax.checkpoint(_sum_step, prevent_cse=False, static_argnums=(1,))

    def sum_body(i: int, running_sum: Array) -> Array:
        return _sum_step(i * chunk_size, chunk_size, running_sum)

    exp_sum = jnp.zeros_like(logit_max)
    if num_full_chunks > 0:
        exp_sum = lax.fori_loop(0, num_full_chunks, sum_body, exp_sum)
    if tail:
        exp_sum = _sum_step(num_full_chunks * chunk_size, tail, exp_sum)

    log_z = jnp.log(exp_sum) + logit_max
    target_logits = jnp.take_along_axis(logits, targets[..., None], axis=-1).astype(jnp.float32)
    token_log_probs = jnp.squeeze(target_logits, axis=-1) - log_z

    if not return_entropy:
        return token_log_probs, None

    def _entropy_step(start: int, size: int, expected_logits: Array) -> Array:
        chunk = lax.dynamic_slice_in_dim(logits, start, size, axis=-1).astype(jnp.float32)
        probs = jnp.exp(chunk - log_z[..., None])
        return expected_logits + jnp.sum(probs * chunk, axis=-1)

    _entropy_step = jax.checkpoint(_entropy_step, prevent_cse=False, static_argnums=(1,))

    def entropy_body(i: int, expected_logits: Array) -> Array:
        return _entropy_step(i * chunk_size, chunk_size, expected_logits)

    expected_logits = jnp.zeros_like(log_z)
    if num_full_chunks > 0:
        expected_logits = lax.fori_loop(0, num_full_chunks, entropy_body, expected_logits)
    if tail:
        expected_logits = _entropy_step(num_full_chunks * chunk_size, tail, expected_logits)

    entropies = log_z - expected_logits
    return token_log_probs, entropies


def compute_sequence_scores_from_hidden_states(
    model: tp.Any,
    hidden_states: Array,
    labels: Array,
    loss_mask: Array,
    *,
    token_chunk_size: int,
    vocab_chunk_size: int | None,
    return_correct_counts: bool = False,
) -> tuple[Array, Array, Array] | tuple[Array, Array, Array, Array]:
    """Project hidden states through the LM head chunk-by-chunk and accumulate masked sequence-level scores.

    This is the primary memory-efficient scoring path used by preference
    optimization trainers (DPO, ORPO, CPO, etc.) that need per-sequence
    log-probability sums but do *not* need per-token detail.  Rather than
    running the full LM head over the entire sequence at once -- which
    would allocate a ``[batch, seq_len, vocab_size]`` logit tensor -- this
    function slices the **sequence/token** dimension into chunks of
    ``token_chunk_size`` tokens and processes each chunk independently:

    1. Slice the hidden states, labels, and loss mask for the current
       chunk along the sequence axis.
    2. Project the hidden-state chunk through the LM head
       (``compute_lm_logits``, optionally preceded by
       ``prepare_lm_head_inputs`` if the model defines it).
    3. Compute per-token log-probabilities within the chunk via
       :func:`compute_token_logps_and_entropies_chunked`, which itself
       chunks across the *vocabulary* dimension for further savings.
    4. Accumulate three per-sequence (shape ``[batch]``) running totals:

       - **log-prob sum** -- ``sum of logp[t]`` for tokens where
         ``loss_mask`` is true.
       - **logit sum** -- ``sum of sum_v logits[t, v]`` for masked tokens
         (useful for auxiliary regularisation terms).
       - **token count** -- number of masked tokens per sequence.

    All inner steps are wrapped with ``jax.checkpoint`` so that XLA can
    trade compute for memory during back-propagation.  A
    ``jax.lax.fori_loop`` drives the iteration over full-sized chunks, with
    a possible tail chunk handled separately.

    Args:
        model: An EasyDeL model that exposes ``compute_lm_logits`` (directly
            or via its ``.model`` attribute).  Resolved internally through
            :func:`resolve_lmhead_projection_module`.
        hidden_states: Float array of shape ``[batch, seq_len, hidden_dim]``
            -- the last hidden states produced by the model backbone.
        labels: Integer array of shape ``[batch, seq_len]`` holding target
            token ids used to gather log-probabilities.
        loss_mask: Boolean (or 0/1 integer) array of shape
            ``[batch, seq_len]``.  Only positions where the mask is true
            contribute to the accumulated sums.
        token_chunk_size: Number of tokens along the sequence axis to
            project in a single step.  Clamped to ``[1, seq_len]``.
        vocab_chunk_size: Vocabulary chunk size forwarded to
            :func:`compute_token_logps_and_entropies_chunked` for the inner
            vocabulary-dimension chunking.

    Returns:
        A 3-tuple or 4-tuple of float32 arrays, each of shape ``[batch]``:

        - **logp_sums** -- per-sequence sum of masked token log-probabilities.
        - **logit_sums** -- per-sequence sum of masked token logit sums
          (each token contributes ``sum_v logits[v]``).
        - **token_counts** -- per-sequence count of masked tokens.
        - **correct_counts** -- optional per-sequence count of masked tokens
          whose argmax prediction matches the label.

    Raises:
        TypeError: If the model does not expose ``compute_lm_logits``.
    """

    projection_model = resolve_lmhead_projection_module(model)
    if projection_model is None:
        raise TypeError(f"{type(model).__name__} does not expose `compute_lm_logits` for headless scoring.")

    batch_size, seq_len = labels.shape
    token_chunk_size = max(1, min(int(token_chunk_size), int(seq_len)))

    # Obtain the trace-safe projection callable once, before entering the
    # fori_loop / checkpoint region.  This avoids calling SpecTrax-module
    # forward methods (which may carry nn.remat wrappers) from inside
    # nested JAX traced regions, preventing TraceContextError.
    _lm_head_fn = (
        projection_model.make_lm_head_fn()
        if hasattr(projection_model, "make_lm_head_fn")
        else projection_model.compute_lm_logits
    )
    _has_prepare = hasattr(projection_model, "prepare_lm_head_inputs")

    def _project_chunk(chunk_hidden_states: Array) -> Array:
        if _has_prepare:
            chunk_hidden_states = projection_model.prepare_lm_head_inputs(chunk_hidden_states)
        return _lm_head_fn(chunk_hidden_states)

    _project_chunk = jax.checkpoint(_project_chunk, prevent_cse=False)

    def _chunk_contributions(
        chunk_hidden_states: Array,
        chunk_labels: Array,
        chunk_loss_mask: Array,
    ) -> tuple[Array, Array, Array] | tuple[Array, Array, Array, Array]:
        chunk_logits = _project_chunk(chunk_hidden_states)
        chunk_logps, _ = compute_token_logps_and_entropies_chunked(
            chunk_logits,
            chunk_labels,
            return_entropy=False,
            chunk_size=vocab_chunk_size,
        )
        chunk_mask_f = chunk_loss_mask.astype(jnp.float32)
        chunk_logit_sums = jnp.sum(chunk_logits.astype(jnp.float32), axis=-1)
        chunk_values = (
            jnp.sum(jnp.where(chunk_loss_mask, chunk_logps, 0.0), axis=-1),
            jnp.sum(chunk_logit_sums * chunk_mask_f, axis=-1),
            jnp.sum(chunk_mask_f, axis=-1),
        )
        if not return_correct_counts:
            return chunk_values
        chunk_predictions = jnp.argmax(chunk_logits, axis=-1).astype(chunk_labels.dtype)
        chunk_correct_counts = jnp.sum(
            chunk_mask_f * (chunk_predictions == chunk_labels).astype(jnp.float32),
            axis=-1,
        )
        return (*chunk_values, chunk_correct_counts)

    _chunk_contributions = jax.checkpoint(_chunk_contributions, prevent_cse=False)

    base_carry = (
        jnp.zeros((batch_size,), dtype=jnp.float32),
        jnp.zeros((batch_size,), dtype=jnp.float32),
        jnp.zeros((batch_size,), dtype=jnp.float32),
    )
    carry = (
        (
            *base_carry,
            jnp.zeros((batch_size,), dtype=jnp.float32),
        )
        if return_correct_counts
        else base_carry
    )
    num_full_chunks = seq_len // token_chunk_size
    tail = seq_len - num_full_chunks * token_chunk_size

    def _accumulate_chunk(
        start: int,
        size: int,
        current: tuple[Array, Array, Array] | tuple[Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array] | tuple[Array, Array, Array, Array]:
        chunk_hidden_states = lax.dynamic_slice_in_dim(hidden_states, start, size, axis=1)
        chunk_labels = lax.dynamic_slice_in_dim(labels, start, size, axis=1)
        chunk_loss_mask = lax.dynamic_slice_in_dim(loss_mask, start, size, axis=1)
        chunk_contributions = _chunk_contributions(
            chunk_hidden_states,
            chunk_labels,
            chunk_loss_mask,
        )
        if return_correct_counts:
            chunk_logps, chunk_logit_sums, chunk_token_counts, chunk_correct_counts = chunk_contributions
            return (
                current[0] + chunk_logps,
                current[1] + chunk_logit_sums,
                current[2] + chunk_token_counts,
                current[3] + chunk_correct_counts,
            )
        chunk_logps, chunk_logit_sums, chunk_token_counts = chunk_contributions
        return (
            current[0] + chunk_logps,
            current[1] + chunk_logit_sums,
            current[2] + chunk_token_counts,
        )

    def _full_body(
        i: int,
        current: tuple[Array, Array, Array] | tuple[Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array] | tuple[Array, Array, Array, Array]:
        return _accumulate_chunk(i * token_chunk_size, token_chunk_size, current)

    if num_full_chunks > 0:
        carry = lax.fori_loop(0, num_full_chunks, _full_body, carry)
    if tail:
        carry = _accumulate_chunk(num_full_chunks * token_chunk_size, tail, carry)
    return carry


def compute_per_token_logps_and_entropies_from_hidden_states(
    model: tp.Any,
    hidden_states: Array,
    targets: Array,
    *,
    token_chunk_size: int,
    vocab_chunk_size: int | None,
    return_entropy: bool,
) -> tuple[Array, Array | None]:
    """Project hidden states through the LM head chunk-by-chunk and return per-token log-probabilities and optional entropies.

    This function is the per-token counterpart of
    :func:`compute_sequence_scores_from_hidden_states`.  Instead of
    accumulating sequence-level sums it writes per-token results back into
    full-length arrays using ``jax.lax.dynamic_update_slice``, making it
    suitable for trainers that require fine-grained token-level scores
    (e.g. GRPO, PPO, or any method that applies per-token weighting or
    masking downstream).

    Memory efficiency is achieved through two levels of chunking:

    * **Sequence-dimension chunking** (controlled by *token_chunk_size*) --
      only a slice of the hidden states is projected through the LM head at
      a time, so the peak ``[batch, chunk, vocab_size]`` logit tensor is
      bounded.
    * **Vocabulary-dimension chunking** (controlled by *vocab_chunk_size*,
      forwarded to :func:`compute_token_logps_and_entropies_chunked`) --
      within each token chunk the log-softmax is computed without
      materializing a dense ``[batch, chunk, vocab_size]`` log-prob tensor.

    The iteration pattern mirrors
    :func:`compute_sequence_scores_from_hidden_states`: a
    ``jax.lax.fori_loop`` over full-sized chunks followed by an optional
    tail chunk, with ``jax.checkpoint`` on the inner projection and scoring
    steps.

    Args:
        model: An EasyDeL model that exposes ``compute_lm_logits`` (directly
            or via its ``.model`` attribute).  Resolved internally through
            :func:`resolve_lmhead_projection_module`.
        hidden_states: Float array of shape ``[batch, seq_len, hidden_dim]``
            -- the last hidden states from the model backbone.
        targets: Integer array of shape ``[batch, seq_len]`` holding the
            target token ids whose log-probabilities are gathered.
        token_chunk_size: Number of tokens along the sequence axis to
            project in a single step.  Clamped to ``[1, seq_len]``.
        vocab_chunk_size: Vocabulary chunk size forwarded to
            :func:`compute_token_logps_and_entropies_chunked`.
        return_entropy: If ``True``, Shannon entropies are computed and
            returned alongside log-probabilities.  When ``False`` the
            entropy computation is skipped and ``None`` is returned.

    Returns:
        A tuple ``(token_logps, entropies)`` where:

        - **token_logps** -- float32 array of shape ``[batch, seq_len]``
          with the log-probability of each target token.
        - **entropies** -- float32 array of the same shape with per-token
          Shannon entropies, or ``None`` when *return_entropy* is ``False``.

    Raises:
        TypeError: If the model does not expose ``compute_lm_logits``.
    """

    projection_model = resolve_lmhead_projection_module(model)
    if projection_model is None:
        raise TypeError(f"{type(model).__name__} does not expose `compute_lm_logits` for headless scoring.")

    batch_size, seq_len = targets.shape
    token_chunk_size = max(1, min(int(token_chunk_size), int(seq_len)))

    _lm_head_fn = (
        projection_model.make_lm_head_fn()
        if hasattr(projection_model, "make_lm_head_fn")
        else projection_model.compute_lm_logits
    )
    _has_prepare = hasattr(projection_model, "prepare_lm_head_inputs")

    def _project_chunk(chunk_hidden_states: Array) -> Array:
        if _has_prepare:
            chunk_hidden_states = projection_model.prepare_lm_head_inputs(chunk_hidden_states)
        return _lm_head_fn(chunk_hidden_states)

    _project_chunk = jax.checkpoint(_project_chunk, prevent_cse=False)

    def _chunk_contributions(chunk_hidden_states: Array, chunk_targets: Array) -> tuple[Array, Array | None]:
        chunk_logits = _project_chunk(chunk_hidden_states)
        return compute_token_logps_and_entropies_chunked(
            chunk_logits,
            chunk_targets,
            return_entropy=return_entropy,
            chunk_size=vocab_chunk_size,
        )

    _chunk_contributions = jax.checkpoint(_chunk_contributions, prevent_cse=False)

    token_logps = jnp.zeros((batch_size, seq_len), dtype=jnp.float32)
    entropies = jnp.zeros((batch_size, seq_len), dtype=jnp.float32)
    num_full_chunks = seq_len // token_chunk_size
    tail = seq_len - num_full_chunks * token_chunk_size

    def _accumulate_chunk(
        start: int,
        size: int,
        current: tuple[Array, Array],
    ) -> tuple[Array, Array]:
        chunk_hidden_states = lax.dynamic_slice_in_dim(hidden_states, start, size, axis=1)
        chunk_targets = lax.dynamic_slice_in_dim(targets, start, size, axis=1)
        chunk_logps, chunk_entropies = _chunk_contributions(chunk_hidden_states, chunk_targets)
        next_logps = lax.dynamic_update_slice(current[0], chunk_logps, (0, start))
        if not return_entropy:
            return next_logps, current[1]
        next_entropies = lax.dynamic_update_slice(current[1], chunk_entropies, (0, start))
        return next_logps, next_entropies

    def _full_body(i: int, current: tuple[Array, Array]) -> tuple[Array, Array]:
        return _accumulate_chunk(i * token_chunk_size, token_chunk_size, current)

    carry = (token_logps, entropies)
    if num_full_chunks > 0:
        carry = lax.fori_loop(0, num_full_chunks, _full_body, carry)
    if tail:
        carry = _accumulate_chunk(num_full_chunks * token_chunk_size, tail, carry)
    return carry[0], carry[1] if return_entropy else None
