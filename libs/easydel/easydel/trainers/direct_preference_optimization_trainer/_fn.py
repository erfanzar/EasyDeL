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

"""Internal functions for Direct Preference Optimization training.

This module contains the core computational functions used by the DPO trainer,
including various loss functions, forward pass implementations, and training/evaluation
step functions. These functions are designed to work with JAX/spectrax models and support
distributed training through JAX's sharding capabilities.

The module implements multiple DPO loss variants as described in various papers:
- Standard DPO (sigmoid loss)
- IPO (Identity Preference Optimization)
- Hinge loss variant
- Robust DPO with label smoothing
- BCO (Binary Cross-entropy Optimization)
- APO (Anchored Preference Optimization)
- And several other experimental variants

All functions are JIT-compilable for optimal performance on TPU/GPU hardware.
"""

import typing as tp

import jax
import spectrax as spx
from jax import lax
from jax import numpy as jnp
from jax.nn import log_sigmoid as logsigmoid
from jax.nn import relu, sigmoid
from jax.sharding import PartitionSpec
from jaxtyping import Array
from spectrax import with_sharding_constraint

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.loss_utils import LossConfig, LossMetrics

from .._logprob_utils import (
    compute_token_logps_and_entropies_chunked,
    resolve_lmhead_chunksize,
    resolve_lmhead_projection_module,
    sum_logits_vocab_chunked,
)
from .._shared import apply_paired_truncation, gather_multimodal_kwargs
from .._shared import concatenated_inputs as concatenated_inputs
from ..training_utils import (
    ScheduledLossAdapter,
    _scheduled_terminal_stage_rank,
    _slice_batch_for_scheduled_step,
    bind_scheduled_module,
    constrain_scheduled_batch,
    filter_kwargs_for_callable,
    make_assertions_and_get_sizes,
    make_scheduled_module_binder,
    minibatch_call,
    prepare_scheduled_reference_outputs,
    register_scheduled_loss_adapter,
    sanitize_model_call_kwargs,
    scheduled_loss_cache_key,
    update_metrics,
    update_state_respectfully,
)
from .dpo_config import LOSS_FN_VARIANTS

_VALID_LOSS_TYPES: frozenset[str] = frozenset(
    {
        "sigmoid",
        "hinge",
        "ipo",
        "exo_pair",
        "nca_pair",
        "robust",
        "bco_pair",
        "sppo_hard",
        "aot",
        "aot_pair",
        "aot_unpaired",
        "apo_zero",
        "apo_down",
        "discopop",
        "sft",
        "sigmoid_norm",
    }
)


def _compute_wpo_log_denom(logits: Array, chunk_size: int | None) -> Array:
    """Compute ``log(sum_v p_v^2)`` stably in float32 without materializing ``2 * logits``.

    Uses the shift-invariant identity ``log(sum_v exp(2*(z_v - m))) - 2 * log(sum_v exp(z_v - m))``
    where ``m = max_v z_v``. When ``chunk_size`` is positive and smaller than the vocabulary size,
    streams across vocabulary chunks using ``jax.lax.fori_loop``.
    """
    vocab_size = logits.shape[-1]
    if chunk_size is None or chunk_size <= 0 or chunk_size >= vocab_size:
        logits_f32 = logits.astype(jnp.float32)
        logit_max = jnp.max(logits_f32, axis=-1, keepdims=True)
        exp_shifted = jnp.exp(logits_f32 - logit_max)
        s1 = jnp.sum(exp_shifted, axis=-1)
        s2 = jnp.sum(jnp.square(exp_shifted), axis=-1)
        return jnp.log(s2) - 2.0 * jnp.log(s1)

    chunk_size = int(chunk_size)
    num_full_chunks = vocab_size // chunk_size
    tail = vocab_size - num_full_chunks * chunk_size

    def _max_step(start: int, size: int, running_max: Array) -> Array:
        chunk = lax.dynamic_slice_in_dim(logits, start, size, axis=-1).astype(jnp.float32)
        return jnp.maximum(running_max, jnp.max(chunk, axis=-1))

    _max_step = jax.checkpoint(_max_step, prevent_cse=False, static_argnums=(1,))

    logit_max = jnp.full(logits.shape[:-1], -jnp.inf, dtype=jnp.float32)
    if num_full_chunks > 0:
        logit_max = lax.fori_loop(0, num_full_chunks, lambda i, m: _max_step(i * chunk_size, chunk_size, m), logit_max)
    if tail:
        logit_max = _max_step(num_full_chunks * chunk_size, tail, logit_max)

    def _sum_step(start: int, size: int, carry: tuple[Array, Array]) -> tuple[Array, Array]:
        chunk = lax.dynamic_slice_in_dim(logits, start, size, axis=-1).astype(jnp.float32)
        exp_shifted = jnp.exp(chunk - logit_max[..., None])
        return (
            carry[0] + jnp.sum(exp_shifted, axis=-1),
            carry[1] + jnp.sum(jnp.square(exp_shifted), axis=-1),
        )

    _sum_step = jax.checkpoint(_sum_step, prevent_cse=False, static_argnums=(1,))

    carry = (jnp.zeros_like(logit_max), jnp.zeros_like(logit_max))
    if num_full_chunks > 0:
        carry = lax.fori_loop(0, num_full_chunks, lambda i, c: _sum_step(i * chunk_size, chunk_size, c), carry)
    if tail:
        carry = _sum_step(num_full_chunks * chunk_size, tail, carry)
    s1, s2 = carry
    return jnp.log(s2) - 2.0 * jnp.log(s1)


def _compute_token_logps_chunked(
    logits: Array,
    targets: Array,
    *,
    chunk_size: int | None,
) -> Array:
    """Thin wrapper for backwards compatibility.

    Forwards to the shared :func:`compute_token_logps_and_entropies_chunked`
    in ``trainers/_logprob_utils.py`` and discards the entropy slot since DPO
    does not consume per-token entropies.
    """
    log_probs, _ = compute_token_logps_and_entropies_chunked(
        logits,
        targets,
        return_entropy=False,
        chunk_size=chunk_size,
    )
    return log_probs


def _ld_logp_weights(loss_mask: Array, num_examples: int, ld_alpha: float | None) -> Array:
    """Return per-token LD-DPO weights for concatenated chosen/rejected rows."""
    if ld_alpha is None:
        return jnp.ones_like(loss_mask, dtype=jnp.float32)

    completion_pos = jnp.cumsum(loss_mask.astype(jnp.int32), axis=1)
    completion_lengths = loss_mask.sum(axis=1).astype(jnp.int32)
    chosen_lengths = completion_lengths[:num_examples]
    rejected_lengths = completion_lengths[num_examples:]
    shared_lengths = jnp.minimum(chosen_lengths, rejected_lengths)
    shared_lengths = jnp.concatenate([shared_lengths, shared_lengths], axis=0)
    shared_mask = loss_mask & (completion_pos > 0) & (completion_pos <= shared_lengths[:, None])
    tail_mask = loss_mask & (completion_pos > shared_lengths[:, None])
    return shared_mask.astype(jnp.float32) + float(ld_alpha) * tail_mask.astype(jnp.float32)


def _compute_dpo_outputs_from_hidden_states(
    model: tp.Any,
    hidden_states: Array,
    labels: Array,
    loss_mask: Array,
    *,
    num_examples: int,
    chunk_size: int,
    logprob_vocab_chunk_size: int | None,
    loss_type: LOSS_FN_VARIANTS,
    use_weighting: bool = False,
    ld_alpha: float | None = None,
    vocab_shard_stage: int | None = None,
    return_mean_logits: bool = True,
) -> dict[str, Array]:
    """Project DPO hidden states through the LM head chunk-by-chunk across the sequence dimension.

    Instead of computing logits for the entire sequence at once (which would
    require ``O(batch * seq * vocab)`` memory), this function slices the
    sequence into fixed-size chunks and, for each chunk:

    1. Projects hidden states to vocabulary logits via the model's
       ``compute_lm_logits`` (optionally preceded by ``prepare_lm_head_inputs``).
    2. Computes per-token log-probabilities with
       :func:`_compute_token_logps_chunked`, further chunking along the
       vocabulary axis to keep memory bounded.
    3. Accumulates the masked per-example log-probability sums and the
       weighted logit summary statistics (sum of logits and token counts
       for chosen and rejected halves of the batch).

    The batch is assumed to be structured so that the first ``num_examples``
    rows correspond to *chosen* completions and the remaining rows to
    *rejected* completions (as produced by ``concatenated_inputs``).

    The accumulated log-probabilities are always returned as sequence
    sums. Losses that need length-normalized scores (IPO, sigmoid_norm)
    use the returned token counts during loss reduction.

    Both the per-chunk projection and the per-chunk contribution helpers are
    wrapped with ``jax.checkpoint`` to trade compute for memory during the
    backward pass.

    Args:
        model: The language model instance.  Must expose ``compute_lm_logits``
            and, optionally, ``prepare_lm_head_inputs``.
        hidden_states: Float array of shape ``(batch, seq_len, hidden_dim)``
            produced by the model's body (without the final LM head).
        labels: Integer array of shape ``(batch, seq_len)`` with target token
            ids.
        loss_mask: Boolean or float array of shape ``(batch, seq_len)``
            indicating which positions contribute to the loss.
        num_examples: Number of *chosen* examples in the batch (the first
            ``num_examples`` rows).  The remaining rows are treated as
            rejected examples.
        chunk_size: Maximum number of sequence positions to project through
            the LM head in a single chunk.
        logprob_vocab_chunk_size: Vocabulary-dimension chunk size forwarded to
            :func:`_compute_token_logps_chunked` for the inner log-prob
            computation.
        loss_type: The DPO loss variant in use. Retained as a static
            compatibility knob for existing compiled call sites.
        use_weighting: Whether to accumulate WPO per-example weights.
        ld_alpha: Optional length-debiased DPO tail-token weight. When
            set, log-probs after the shared chosen/rejected completion
            length are multiplied by this value before summation.

    Returns:
        A dictionary with the following keys:

        - ``"chosen_logps"`` -- Float array of shape ``(num_examples,)`` with
          the summed log-probabilities for the chosen completions.
        - ``"rejected_logps"`` -- Float array of shape ``(num_examples,)`` for
          the rejected completions.
        - ``"mean_chosen_logits"`` -- Scalar float: the mean logit value
          across all loss-bearing chosen tokens (a lightweight summary
          replacing the full logit tensor).
        - ``"mean_rejected_logits"`` -- Scalar float: the corresponding mean
          for rejected tokens.
    """

    batch_size, seq_len = labels.shape
    chunk_size = max(1, min(int(chunk_size), int(seq_len)))

    projection_model = resolve_lmhead_projection_module(model)
    if projection_model is None:
        raise TypeError(f"{type(model).__name__} does not expose `compute_lm_logits` for headless scoring.")

    _lm_head_fn = (
        projection_model.make_lm_head_fn(vocab_shard_stage=vocab_shard_stage)
        if hasattr(projection_model, "make_lm_head_fn")
        else projection_model.compute_lm_logits
    )
    _has_prepare = hasattr(projection_model, "prepare_lm_head_inputs")

    def _project_chunk(chunk_hidden_states: Array) -> Array:
        """Project a sequence-axis hidden-state chunk through the LM head.

        Args:
            chunk_hidden_states: ``[batch, chunk_seq, hidden_dim]`` slice
                of hidden states.

        Returns:
            The corresponding ``[batch, chunk_seq, vocab_size]`` logits.
        """
        if _has_prepare:
            chunk_hidden_states = projection_model.prepare_lm_head_inputs(chunk_hidden_states)
        return _lm_head_fn(chunk_hidden_states)

    _project_chunk = jax.checkpoint(_project_chunk, prevent_cse=False)

    def _chunk_contributions(
        chunk_hidden_states: Array,
        chunk_labels: Array,
        chunk_loss_mask: Array,
        chunk_logp_weights: Array,
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        """Compute per-sequence per-chunk DPO log-prob and logit accumulators.

        Args:
            chunk_hidden_states: Hidden-state slice for the current
                chunk.
            chunk_labels: Label slice with the same sequence range.
            chunk_loss_mask: Boolean loss mask slice.
            chunk_logp_weights: Per-token multipliers for LD-DPO.

        Returns:
            A 6-tuple with masked log-prob sums, WPO log-prob sums,
            chosen / rejected logit mean sums, and chosen / rejected token
            counts for the current chunk.
        """
        chunk_logits = _project_chunk(chunk_hidden_states)
        chunk_logps = _compute_token_logps_chunked(
            chunk_logits,
            chunk_labels,
            chunk_size=logprob_vocab_chunk_size,
        )
        chunk_loss_mask_bool = chunk_loss_mask.astype(bool)
        masked_logps = jnp.where(chunk_loss_mask_bool, chunk_logps * chunk_logp_weights, 0.0)
        if use_weighting:
            log_denom = _compute_wpo_log_denom(chunk_logits, logprob_vocab_chunk_size)
            wpo_logps = jnp.where(chunk_loss_mask_bool, chunk_logps - log_denom, 0.0)
        else:
            wpo_logps = jnp.zeros_like(masked_logps)
        if return_mean_logits:
            vocab_size = float(chunk_logits.shape[-1])
            chunk_token_logit_means = (
                sum_logits_vocab_chunked(chunk_logits, chunk_size=logprob_vocab_chunk_size) / vocab_size
            )
            chosen_mask_bool = chunk_loss_mask_bool[:num_examples]
            rejected_mask_bool = chunk_loss_mask_bool[num_examples:]
            chosen_logit_sum = jnp.sum(jnp.where(chosen_mask_bool, chunk_token_logit_means[:num_examples], 0.0))
            rejected_logit_sum = jnp.sum(jnp.where(rejected_mask_bool, chunk_token_logit_means[num_examples:], 0.0))
            chosen_count = jnp.sum(chosen_mask_bool.astype(jnp.float32))
            rejected_count = jnp.sum(rejected_mask_bool.astype(jnp.float32))
        else:
            chosen_logit_sum = jnp.array(0.0, dtype=jnp.float32)
            rejected_logit_sum = jnp.array(0.0, dtype=jnp.float32)
            chosen_count = jnp.array(0.0, dtype=jnp.float32)
            rejected_count = jnp.array(0.0, dtype=jnp.float32)
        return (
            masked_logps.sum(axis=-1),
            wpo_logps.sum(axis=-1),
            chosen_logit_sum,
            rejected_logit_sum,
            chosen_count,
            rejected_count,
        )

    _chunk_contributions = jax.checkpoint(_chunk_contributions, prevent_cse=False)

    zero_logps = jnp.zeros((batch_size,), dtype=jnp.float32)
    zero_scalar = jnp.array(0.0, dtype=jnp.float32)
    logp_weights = _ld_logp_weights(loss_mask, num_examples, ld_alpha).astype(jnp.float32)

    def _accumulate_chunk(
        start: int,
        size: int,
        carry: tuple[Array, Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        """Add the contributions of a sequence-chunk into the running totals.

        Args:
            start: Start index along the sequence axis.
            size: Number of tokens in this chunk.
            carry: Running ``(logp_sums, wpo_logp_sums, chosen_logit_sum,
                rejected_logit_sum, chosen_count, rejected_count)``.

        Returns:
            The updated carry with the chunk's contributions added.
        """
        chunk_hidden_states = lax.dynamic_slice_in_dim(hidden_states, start, size, axis=1)
        chunk_labels = lax.dynamic_slice_in_dim(labels, start, size, axis=1)
        chunk_loss_mask = lax.dynamic_slice_in_dim(loss_mask, start, size, axis=1)
        chunk_logp_weights = lax.dynamic_slice_in_dim(logp_weights, start, size, axis=1)
        (
            chunk_logp_sums,
            chunk_wpo_logp_sums,
            chosen_logit_sum,
            rejected_logit_sum,
            chosen_denom,
            rejected_denom,
        ) = _chunk_contributions(chunk_hidden_states, chunk_labels, chunk_loss_mask, chunk_logp_weights)
        return (
            carry[0] + chunk_logp_sums,
            carry[1] + chunk_wpo_logp_sums,
            carry[2] + chosen_logit_sum,
            carry[3] + rejected_logit_sum,
            carry[4] + chosen_denom,
            carry[5] + rejected_denom,
        )

    num_full_chunks = seq_len // chunk_size
    tail = seq_len - num_full_chunks * chunk_size
    carry = (zero_logps, zero_logps, zero_scalar, zero_scalar, zero_scalar, zero_scalar)

    def _full_body(
        i: int,
        inner_carry: tuple[Array, Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        """``fori_loop`` body that processes the ``i``-th full-sized sequence chunk.

        Args:
            i: Chunk index in ``[0, num_full_chunks)``.
            inner_carry: Current accumulator tuple.

        Returns:
            The updated accumulator tuple.
        """
        return _accumulate_chunk(i * chunk_size, chunk_size, inner_carry)

    if num_full_chunks > 0:
        carry = lax.fori_loop(0, num_full_chunks, _full_body, carry)
    if tail:
        carry = _accumulate_chunk(num_full_chunks * chunk_size, tail, carry)

    all_logps, all_wpo_logps, chosen_logit_sum, rejected_logit_sum, chosen_denom, rejected_denom = carry
    lengths = jnp.maximum(loss_mask.sum(axis=-1).astype(jnp.float32), 1.0)
    wpo_weights = jnp.exp(all_wpo_logps / lengths)

    result = {
        "chosen_logps": all_logps[:num_examples],
        "rejected_logps": all_logps[num_examples:],
        "chosen_lengths": lengths[:num_examples],
        "rejected_lengths": lengths[num_examples:],
        "wpo_weights": wpo_weights[:num_examples] * wpo_weights[num_examples:],
    }
    if return_mean_logits:
        result["mean_chosen_logits"] = chosen_logit_sum / jnp.maximum(chosen_denom, 1.0)
        result["mean_rejected_logits"] = rejected_logit_sum / jnp.maximum(rejected_denom, 1.0)
    return result


def _get_reference_logps_from_batch(batch: dict[str, tp.Any]) -> tuple[tp.Any | None, tp.Any | None]:
    """Read reference log-prob columns from either the canonical or legacy keys.

    DPO batches may carry precomputed reference log-probabilities under
    one of two naming conventions: the canonical ``ref_chosen_logps`` /
    ``ref_rejected_logps`` or the legacy ``reference_chosen_log_probs`` /
    ``reference_rejected_log_probs``. This helper checks both so the
    downstream training and evaluation steps stay agnostic to the
    convention used by the dataset.

    Args:
        batch: Preference minibatch dictionary.

    Returns:
        ``(ref_chosen_logps, ref_rejected_logps)``; either value is
        ``None`` when the batch does not provide that column under
        either convention.
    """
    ref_chosen_logps = batch.get("ref_chosen_logps")
    if ref_chosen_logps is None:
        ref_chosen_logps = batch.get("reference_chosen_log_probs")

    ref_rejected_logps = batch.get("ref_rejected_logps")
    if ref_rejected_logps is None:
        ref_rejected_logps = batch.get("reference_rejected_log_probs")

    return ref_chosen_logps, ref_rejected_logps


def _as_loss_type_tuple(loss_type: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Normalize one or more DPO loss names to an immutable tuple."""
    if isinstance(loss_type, str):
        return (loss_type,)
    return tuple(loss_type)


def _as_loss_weight_tuple(loss_weights: tuple[float, ...] | list[float] | None, loss_count: int) -> tuple[float, ...]:
    """Normalize optional per-loss weights and validate their count."""
    if loss_weights is None:
        return (1.0,) * loss_count
    weights = tuple(float(weight) for weight in loss_weights)
    if len(weights) != loss_count:
        raise ValueError(
            "`loss_weights` must have the same length as `loss_type`; "
            f"got {len(weights)} weights for {loss_count} loss types."
        )
    return weights


def _compute_f_divergence_scores(
    chosen_logratios: Array,
    rejected_logratios: Array,
    *,
    f_divergence_type: str,
    f_alpha_divergence_coef: float,
) -> tuple[Array, Array]:
    """Apply the shifted f-DPO score transform ``f'(r) - f'(1)`` to chosen/rejected log ratios.

    Shifting by ``f'(1)`` keeps every divergence centered at ``0`` when
    ``pi == pi_ref`` (preserving pairwise differences ``chosen_scores - rejected_scores``
    identically while fixing single-branch objectives such as ``bco_pair``, ``nca_pair``,
    ``sppo_hard``, ``apo_zero``, and ``apo_down``). Exponential terms are clamped below
    float32 overflow so extreme full-sequence log-ratios remain finite.
    """
    if f_divergence_type == "reverse_kl":
        return chosen_logratios, rejected_logratios
    if f_divergence_type == "forward_kl":
        chosen_scores = -jnp.expm1(jnp.minimum(-chosen_logratios, 80.0))
        rejected_scores = -jnp.expm1(jnp.minimum(-rejected_logratios, 80.0))
        return chosen_scores, rejected_scores
    if f_divergence_type == "js_divergence":
        log2 = jnp.log(jnp.asarray(2.0, dtype=chosen_logratios.dtype))
        return jax.nn.log_sigmoid(chosen_logratios) + log2, jax.nn.log_sigmoid(rejected_logratios) + log2
    if f_divergence_type == "alpha_divergence":
        if abs(float(f_alpha_divergence_coef) - 1.0) < 1e-6:
            return chosen_logratios, rejected_logratios
        scale = float(f_alpha_divergence_coef) - 1.0
        coef = 1.0 / scale
        chosen_scores = jnp.expm1(jnp.minimum(scale * chosen_logratios, 80.0)) * coef
        rejected_scores = jnp.expm1(jnp.minimum(scale * rejected_logratios, 80.0)) * coef
        return chosen_scores, rejected_scores
    raise ValueError(f"Unknown f_divergence_type: {f_divergence_type}")


def _sort_valid_prefix(values: Array, valid: Array) -> Array:
    """Sort valid rows ascending into ``[0, num_valid)`` without injecting ``inf`` into differentiated values."""
    sort_keys = jnp.where(valid, jax.lax.stop_gradient(values), jnp.inf)
    order = jnp.argsort(sort_keys, axis=0)
    return jnp.take_along_axis(values, order, axis=0)


def compute_dpo_losses(
    chosen_logps: Array,
    rejected_logps: Array,
    ref_chosen_logps: Array,
    ref_rejected_logps: Array,
    *,
    beta: float,
    label_smoothing: float,
    loss_type: str | tuple[str, ...] | list[str],
    loss_weights: tuple[float, ...] | list[float] | None = None,
    f_divergence_type: str = "reverse_kl",
    f_alpha_divergence_coef: float = 0.5,
    discopop_tau: float = 0.05,
    chosen_lengths: Array | None = None,
    rejected_lengths: Array | None = None,
    wpo_weights: Array | None = None,
    rpo_alpha: float | None = None,
    pair_valid: Array | None = None,
) -> Array:
    """Compute weighted single- or multi-objective DPO losses."""
    loss_types = _as_loss_type_tuple(loss_type)
    weights = _as_loss_weight_tuple(loss_weights, len(loss_types))
    chosen_lengths = jnp.ones_like(chosen_logps) if chosen_lengths is None else jnp.maximum(chosen_lengths, 1.0)
    rejected_lengths = jnp.ones_like(rejected_logps) if rejected_lengths is None else jnp.maximum(rejected_lengths, 1.0)
    valid_mask = None if pair_valid is None else (pair_valid > 0)

    chosen_logratios = chosen_logps - ref_chosen_logps
    rejected_logratios = rejected_logps - ref_rejected_logps
    chosen_scores, rejected_scores = _compute_f_divergence_scores(
        chosen_logratios,
        rejected_logratios,
        f_divergence_type=f_divergence_type,
        f_alpha_divergence_coef=f_alpha_divergence_coef,
    )
    delta_score = chosen_scores - rejected_scores
    losses = jnp.zeros_like(chosen_logps)

    for single_loss_type, loss_weight in zip(loss_types, weights, strict=True):
        if single_loss_type == "sigmoid":
            per_sequence_loss = -(
                jax.nn.log_sigmoid(beta * delta_score) * (1 - label_smoothing)
                + jax.nn.log_sigmoid(-beta * delta_score) * label_smoothing
            )
        elif single_loss_type == "hinge":
            per_sequence_loss = relu(1 - beta * delta_score)
        elif single_loss_type == "ipo":
            ipo_delta = chosen_scores / chosen_lengths - rejected_scores / rejected_lengths
            per_sequence_loss = (ipo_delta - 1 / (2 * beta)) ** 2
        elif single_loss_type == "exo_pair":
            epsilon = jnp.asarray(label_smoothing, dtype=chosen_logps.dtype)
            per_sequence_loss = sigmoid(beta * delta_score) * (
                logsigmoid(beta * delta_score) - jnp.log1p(-epsilon)
            ) + sigmoid(-beta * delta_score) * (logsigmoid(-beta * delta_score) - jnp.log(epsilon))
        elif single_loss_type == "nca_pair":
            chosen_rewards = beta * chosen_scores
            rejected_rewards = beta * rejected_scores
            per_sequence_loss = -(
                logsigmoid(chosen_rewards) + 0.5 * logsigmoid(-chosen_rewards) + 0.5 * logsigmoid(-rejected_rewards)
            )
        elif single_loss_type == "robust":
            clean_loss_term = -(1 - label_smoothing) * logsigmoid(beta * delta_score)
            flipped_loss_term = -label_smoothing * logsigmoid(-beta * delta_score)
            per_sequence_loss = (clean_loss_term - flipped_loss_term) / (1 - 2 * label_smoothing)
        elif single_loss_type == "bco_pair":
            chosen_rewards = beta * chosen_scores
            rejected_rewards = beta * rejected_scores
            per_sequence_loss = -logsigmoid(chosen_rewards) - logsigmoid(-rejected_rewards)
        elif single_loss_type == "sppo_hard":
            per_sequence_loss = (chosen_scores - 0.5 / beta) ** 2 + (rejected_scores + 0.5 / beta) ** 2
        elif single_loss_type == "aot":
            if valid_mask is not None:
                num_valid = jnp.sum(valid_mask.astype(jnp.int32))
                sorted_valid = jnp.arange(chosen_logps.shape[0]) < num_valid
                if f_divergence_type == "reverse_kl":
                    logratios_sorted = _sort_valid_prefix(chosen_logps - rejected_logps, valid_mask)
                    ref_logratios_sorted = _sort_valid_prefix(ref_chosen_logps - ref_rejected_logps, valid_mask)
                    delta = jnp.where(sorted_valid, logratios_sorted - ref_logratios_sorted, 0.0)
                else:
                    delta = jnp.where(
                        sorted_valid,
                        _sort_valid_prefix(chosen_scores, valid_mask) - _sort_valid_prefix(rejected_scores, valid_mask),
                        0.0,
                    )
                sorted_loss = jnp.where(
                    sorted_valid,
                    -logsigmoid(beta * delta) * (1 - label_smoothing) - logsigmoid(-beta * delta) * label_smoothing,
                    0.0,
                )
                valid_slot = jnp.clip(
                    jnp.cumsum(valid_mask.astype(jnp.int32), axis=0) - 1,
                    0,
                    max(chosen_logps.shape[0] - 1, 0),
                )
                per_sequence_loss = jnp.where(valid_mask, jnp.take_along_axis(sorted_loss, valid_slot, axis=0), 0.0)
            else:
                if f_divergence_type == "reverse_kl":
                    logratios_sorted = jnp.sort(chosen_logps - rejected_logps, axis=0)
                    ref_logratios_sorted = jnp.sort(ref_chosen_logps - ref_rejected_logps, axis=0)
                    delta = logratios_sorted - ref_logratios_sorted
                else:
                    delta = jnp.sort(chosen_scores, axis=0) - jnp.sort(rejected_scores, axis=0)
                per_sequence_loss = (
                    -logsigmoid(beta * delta) * (1 - label_smoothing) - logsigmoid(-beta * delta) * label_smoothing
                )
        elif single_loss_type in {"aot_pair", "aot_unpaired"}:
            if valid_mask is not None:
                num_valid = jnp.sum(valid_mask.astype(jnp.int32))
                sorted_valid = jnp.arange(chosen_logps.shape[0]) < num_valid
                chosen_scores_sorted = _sort_valid_prefix(chosen_scores, valid_mask)
                rejected_scores_sorted = _sort_valid_prefix(rejected_scores, valid_mask)
                delta = jnp.where(sorted_valid, chosen_scores_sorted - rejected_scores_sorted, 0.0)
                sorted_loss = jnp.where(
                    sorted_valid,
                    -logsigmoid(beta * delta) * (1 - label_smoothing) - logsigmoid(-beta * delta) * label_smoothing,
                    0.0,
                )
                valid_slot = jnp.clip(
                    jnp.cumsum(valid_mask.astype(jnp.int32), axis=0) - 1,
                    0,
                    max(chosen_logps.shape[0] - 1, 0),
                )
                per_sequence_loss = jnp.where(valid_mask, jnp.take_along_axis(sorted_loss, valid_slot, axis=0), 0.0)
            else:
                chosen_scores_sorted = jnp.sort(chosen_scores, axis=0)
                rejected_scores_sorted = jnp.sort(rejected_scores, axis=0)
                delta = chosen_scores_sorted - rejected_scores_sorted
                per_sequence_loss = (
                    -logsigmoid(beta * delta) * (1 - label_smoothing) - logsigmoid(-beta * delta) * label_smoothing
                )
        elif single_loss_type == "apo_zero":
            per_sequence_loss = 1 - sigmoid(beta * chosen_scores) + sigmoid(beta * rejected_scores)
        elif single_loss_type == "apo_down":
            per_sequence_loss = sigmoid(beta * chosen_scores) + (1 - sigmoid(beta * delta_score))
        elif single_loss_type == "discopop":
            logits = beta * delta_score
            inv_tau = 1.0 / discopop_tau
            one_minus_modulation = sigmoid(-logits * inv_tau)
            exp_term = jnp.exp(jnp.minimum(-logits + logsigmoid(logits * inv_tau), 80.0))
            per_sequence_loss = -logsigmoid(logits) * one_minus_modulation + exp_term
        elif single_loss_type == "sft":
            if valid_mask is not None:
                valid_chosen_logps = jnp.where(valid_mask, chosen_logps, 0.0)
                valid_chosen_lengths = jnp.where(valid_mask, chosen_lengths, 0.0)
                sft_loss = -jnp.sum(valid_chosen_logps) / jnp.maximum(jnp.sum(valid_chosen_lengths), 1.0)
                per_sequence_loss = jnp.where(valid_mask, sft_loss, 0.0)
            else:
                sft_loss = -jnp.sum(chosen_logps) / jnp.maximum(jnp.sum(chosen_lengths), 1.0)
                per_sequence_loss = jnp.broadcast_to(sft_loss, chosen_logps.shape)
        elif single_loss_type == "sigmoid_norm":
            delta = chosen_scores / chosen_lengths - rejected_scores / rejected_lengths
            per_sequence_loss = -logsigmoid(beta * delta)
        else:
            raise ValueError(f"given loss_type({single_loss_type}) is not valid")

        if wpo_weights is not None:
            per_sequence_loss = per_sequence_loss * jax.lax.stop_gradient(wpo_weights)
        losses = losses + per_sequence_loss * float(loss_weight)

    if rpo_alpha is not None and rpo_alpha > 0.0:
        rpo_term = float(rpo_alpha) * (-chosen_logps / chosen_lengths)
        if valid_mask is not None:
            rpo_term = jnp.where(valid_mask, rpo_term, 0.0)
        losses = losses + rpo_term

    return losses


def get_loss_function(
    loss_type: LOSS_FN_VARIANTS,
    beta: float,
    label_smoothing: float | int,
):
    """Resolve the DPO-family loss closure for a given variant.

    All variants share the same calling convention -- they consume the
    summed sequence log-probabilities of policy and reference models on
    the chosen/rejected halves of a preference pair and return a
    per-example loss tensor that the trainer averages. The variants
    differ in how they shape the policy-vs-reference log-ratio
    ``(log pi(y_w|x) - log pi(y_l|x)) - (log pi_ref(y_w|x) - log pi_ref(y_l|x))``
    into a scalar penalty:

    * ``sigmoid`` -- the canonical DPO objective from Rafailov et al. 2023:
      negative log-sigmoid of the temperature-scaled log-ratio with optional
      smoothing toward the conservative DPO (cDPO) variant.
    * ``ipo`` -- Identity Preference Optimization (Azar et al. 2024); a
      squared loss that targets ``logits == 1/(2*beta)`` and avoids the
      saturation pathology of the sigmoid form. Requires
      length-normalized logps (handled upstream).
    * ``hinge`` -- max-margin loss ``relu(1 - beta * logits)``.
    * ``robust`` -- noise-aware DPO (Chowdhury et al. 2024) that rescales
      smoothed sigmoid by ``1/(1 - 2*label_smoothing)``.
    * ``exo_pair`` -- Exact Preference Optimization (EXO).
    * ``nca_pair`` -- pair-based Noise-Contrastive Alignment.
    * ``bco_pair``, ``sppo_hard``, ``aot``/``aot_pair``,
      ``apo_zero``/``apo_down`` -- additional preference objectives;
      see each closure for the exact functional form.

    The returned callable is pure, jit-compatible, and ignores extra
    keyword arguments so callers can pass training-time auxiliaries
    (e.g. discopop temperature) uniformly.

    Args:
        loss_type: Variant key. See :data:`LOSS_FN_VARIANTS`.
        beta: Inverse-temperature on the policy-vs-reference log-ratio
            (the DPO ``beta``). Larger values penalise deviations from
            the reference model more aggressively.
        label_smoothing: cDPO-style smoothing factor in ``[0, 0.5)``.
            For variants that ignore smoothing this argument is dropped.

    Returns:
        A pure function with signature
        ``(chosen_logps, rejected_logps, ref_chosen_logps,
        ref_rejected_logps, beta, label_smoothing, **kwargs) -> Array``
        producing the per-example loss tensor.

    Raises:
        ValueError: If ``loss_type`` is not one of the registered
            variants.
    """

    loss_types = _as_loss_type_tuple(loss_type)
    if not loss_types:
        raise ValueError("`loss_type` must contain at least one DPO loss variant.")
    for single_loss_type in loss_types:
        if single_loss_type not in _VALID_LOSS_TYPES:
            raise ValueError(f"given loss_type({single_loss_type}) is not valid")

    def _combined_loss(
        chosen_logps: Array,
        rejected_logps: Array,
        ref_chosen_logps: Array,
        ref_rejected_logps: Array,
        beta: float,
        label_smoothing: float,
        **kwargs,
    ) -> Array:
        """Compute the configured DPO-family objective through the shared path."""
        return compute_dpo_losses(
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta=beta,
            label_smoothing=label_smoothing,
            loss_type=loss_types,
            loss_weights=kwargs.get("loss_weights"),
            f_divergence_type=kwargs.get("f_divergence_type", "reverse_kl"),
            f_alpha_divergence_coef=kwargs.get("f_alpha_divergence_coef", 0.5),
            discopop_tau=kwargs.get("discopop_tau", 0.05),
            chosen_lengths=kwargs.get("chosen_lengths"),
            rejected_lengths=kwargs.get("rejected_lengths"),
            wpo_weights=kwargs.get("wpo_weights"),
            rpo_alpha=kwargs.get("rpo_alpha"),
            pair_valid=kwargs.get("pair_valid"),
        )

    return _combined_loss


def concatenated_forward(
    model: EasyDeLBaseModule,
    batch: dict[str, list | Array],
    is_encoder_decoder: bool,
    label_pad_token_id: int,
    padding_value: int,
    max_length: int | None = None,
    truncation_mode: str = "keep_end",
    aux_loss_enabled: bool = False,
    loss_type: str = "sigmoid",
    logprob_vocab_chunk_size: int | None = None,
    vocab_shard_stage: int | None = None,
    use_weighting: bool = False,
    ld_alpha: float | None = None,
    return_mean_logits: bool = True,
) -> dict[str, Array]:
    """Run the model on concatenated chosen/rejected inputs and produce DPO log-probs.

    Workflow:

    1. Concatenate chosen and rejected halves with
       :func:`concatenated_inputs` so they can be processed by a
       single forward pass.
    2. Build the model call kwargs differently for encoder-decoder vs
       decoder-only models. Decoder-only models additionally apply
       paired truncation via :func:`apply_paired_truncation` so the
       prompt-and-completion concatenation respects ``max_length``.
    3. When the LM head can be chunked (via ``apply_lm_head=False`` +
       ``last_hidden_state``), defer logit materialisation to
       :func:`_compute_dpo_outputs_from_hidden_states` to avoid the
       ``O(B * L * V)`` peak logit memory.
    4. Otherwise compute per-token log-probabilities directly through
       :func:`_compute_token_logps_chunked` and sum them per example.
    5. For ``loss_type == "ipo"``, length-normalise the per-example
       log-probabilities as required by the IPO objective.
    6. Compute per-half mean logits for diagnostic logging (without
       materialising the full logits tensor when the chunked path is
       used).

    Args:
        model: Module to run -- typically a policy or reference model.
        batch: Preference batch with at least
            ``prompt_input_ids`` / ``chosen_input_ids`` /
            ``rejected_input_ids`` and their attention masks.
        is_encoder_decoder: Whether the model is encoder-decoder
            (changes how labels and the forward kwargs are built).
        label_pad_token_id: Token id used to mark padded label
            positions (encoder-decoder path).
        padding_value: Padding token id passed to
            :func:`concatenated_inputs`.
        max_length: Total sequence length cap for the decoder-only
            path. ``None`` disables truncation.
        truncation_mode: ``"keep_end"`` or ``"keep_start"``.
        aux_loss_enabled: When ``True``, request and forward the
            model's ``aux_loss`` field (e.g. MoE load-balancing).
        loss_type: DPO variant key; only ``"ipo"`` triggers
            length-normalisation here.
        logprob_vocab_chunk_size: Vocab-axis chunk size for
            :func:`compute_token_logps_and_entropies_chunked`. ``None``
            disables chunking.
        vocab_shard_stage: Optional MPMD pipeline stage rank used when
            sharding the LM head; forwarded to ``make_lm_head_fn``.
        use_weighting: Whether to compute WPO example weights.
        ld_alpha: Optional length-debiased DPO tail-token weight.

    Returns:
        Dictionary with:

        * ``chosen_logps`` / ``rejected_logps`` -- per-example summed
          (or length-normalised) log-probabilities for each half.
        * ``mean_chosen_logits`` / ``mean_rejected_logits`` -- scalar
          diagnostic averages over loss-bearing tokens.
        * ``aux_loss`` -- forwarded from the model output when
          ``aux_loss_enabled`` and present.

    Raises:
        TypeError: If the model is invoked with ``apply_lm_head=False``
            but does not return ``last_hidden_state``, or returns
            neither logits nor a hidden state on the standard path.
    """
    num_examples = batch["prompt_input_ids"].shape[0]
    concatenated_batch = concatenated_inputs(batch=batch, padding_value=padding_value)

    model_kwargs = gather_multimodal_kwargs(concatenated_batch, aux_loss_enabled=aux_loss_enabled)

    prompt_input_ids = concatenated_batch["prompt_input_ids"]
    prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
    completion_input_ids = concatenated_batch["completion_input_ids"]
    completion_attention_mask = concatenated_batch["completion_attention_mask"]
    # Optional explicit loss mask (from ``chosen_loss_mask``/``rejected_loss_mask``) selects which attended
    # completion tokens enter the log-prob sums; without it every attended completion token is scored.
    completion_loss_mask = concatenated_batch.get("completion_loss_mask")
    if completion_loss_mask is None:
        completion_scoring_mask = completion_attention_mask
    else:
        completion_scoring_mask = completion_attention_mask * completion_loss_mask.astype(
            completion_attention_mask.dtype
        )
    lmhead_chunksize = None

    if is_encoder_decoder:
        # For encoder-decoder models, use completion inputs as labels.
        labels = completion_input_ids
        labels = jnp.where(
            completion_attention_mask == 0,
            label_pad_token_id,
            completion_input_ids,
        )
        call_kwargs = {
            "input_ids": prompt_input_ids,
            "attention_mask": prompt_attention_mask,
            "labels": labels,
            **model_kwargs,
        }
        call_kwargs = filter_kwargs_for_callable(getattr(model, "forward", model), call_kwargs)
        call_kwargs = sanitize_model_call_kwargs(call_kwargs)
        outputs = model(**call_kwargs)
        logits = outputs.logits
        loss_mask = completion_scoring_mask.astype(bool)
    else:
        # For decoder-only models, concatenate prompt and completion.
        input_ids = jnp.concatenate(
            [prompt_input_ids, completion_input_ids],
            axis=1,
        )
        attention_mask = jnp.concatenate(
            [prompt_attention_mask, completion_attention_mask],
            axis=1,
        )
        loss_mask = jnp.concatenate(
            [
                jnp.zeros_like(prompt_attention_mask),
                completion_scoring_mask,
            ],
            axis=1,
        )
        input_ids, attention_mask, loss_mask = apply_paired_truncation(
            input_ids,
            attention_mask,
            loss_mask,
            max_length=max_length,
            truncation_mode=truncation_mode,
        )
        lmhead_chunksize = resolve_lmhead_chunksize(model)
        call_kwargs = {
            **model_kwargs,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if lmhead_chunksize is not None:
            call_kwargs["apply_lm_head"] = False
        call_kwargs = filter_kwargs_for_callable(getattr(model, "forward", model), call_kwargs)
        call_kwargs = sanitize_model_call_kwargs(call_kwargs)
        outputs = model(**call_kwargs)
        logits = outputs.logits
        labels = jnp.roll(input_ids, shift=-1, axis=1)
        loss_mask = jnp.roll(loss_mask, shift=-1, axis=1).astype("bool").at[:, -1].set(False)

    if logits is not None and logits.shape[:2] != labels.shape[:2]:
        seq_len = labels.shape[1]
        logits = logits[:, -seq_len:]

    labels = jnp.where(loss_mask, labels, 0)
    if not is_encoder_decoder and logits is None and lmhead_chunksize is not None:
        hidden_states = outputs.last_hidden_state
        if hidden_states is None:
            raise TypeError(
                f"{type(model).__name__} was called with `apply_lm_head=False` but did not return `last_hidden_state`."
            )
        if hidden_states.shape[:2] != labels.shape[:2]:
            hidden_states = hidden_states[:, -labels.shape[1] :, :]
        output = _compute_dpo_outputs_from_hidden_states(
            model=model,
            hidden_states=hidden_states,
            labels=labels,
            loss_mask=loss_mask,
            num_examples=num_examples,
            chunk_size=lmhead_chunksize,
            logprob_vocab_chunk_size=logprob_vocab_chunk_size,
            loss_type=loss_type,
            use_weighting=use_weighting,
            ld_alpha=ld_alpha,
            vocab_shard_stage=vocab_shard_stage,
            return_mean_logits=return_mean_logits,
        )
    else:
        if logits is None:
            raise TypeError(f"{type(model).__name__} did not return logits.")
        gathered_logps = _compute_token_logps_chunked(
            logits,
            labels,
            chunk_size=logprob_vocab_chunk_size,
        )
        logp_weights = _ld_logp_weights(loss_mask, num_examples, ld_alpha).astype(gathered_logps.dtype)
        per_token_logps = jnp.where(loss_mask, gathered_logps * logp_weights, 0.0)
        all_logps = per_token_logps.sum(-1)

        lengths = jnp.maximum(loss_mask.sum(-1).astype(jnp.float32), 1.0)
        output = {}
        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]
        output["chosen_lengths"] = lengths[:num_examples]
        output["rejected_lengths"] = lengths[num_examples:]
        if use_weighting:
            log_denom = _compute_wpo_log_denom(logits, logprob_vocab_chunk_size)
            aligned_logps = jnp.where(loss_mask, gathered_logps - log_denom, 0.0)
            weights = jnp.exp(aligned_logps.sum(axis=-1) / lengths)
            output["wpo_weights"] = weights[:num_examples] * weights[num_examples:]

        if return_mean_logits:
            vocab_size = float(logits.shape[-1])
            token_logit_means = sum_logits_vocab_chunked(logits, chunk_size=logprob_vocab_chunk_size) / vocab_size
            chosen_denom = jnp.maximum(jnp.sum(loss_mask[:num_examples].astype(jnp.float32)), 1.0)
            rejected_denom = jnp.maximum(jnp.sum(loss_mask[num_examples:].astype(jnp.float32)), 1.0)
            output["mean_chosen_logits"] = (
                jnp.where(loss_mask[:num_examples], token_logit_means[:num_examples], 0.0).sum() / chosen_denom
            )
            output["mean_rejected_logits"] = (
                jnp.where(loss_mask[num_examples:], token_logit_means[num_examples:], 0.0).sum() / rejected_denom
            )

    if completion_loss_mask is not None:
        # A pair whose chosen or rejected side scores no token (e.g. a placeholder row kept to hold the
        # batch shape fixed) carries no preference signal; the loss reductions exclude it.
        scored_tokens = loss_mask.astype(jnp.int32).sum(-1)
        output["pair_valid"] = ((scored_tokens[:num_examples] > 0) & (scored_tokens[num_examples:] > 0)).astype(
            jnp.float32
        )
    aux_loss = getattr(outputs, "aux_loss", None)
    if aux_loss_enabled and aux_loss is not None:
        output["aux_loss"] = aux_loss
    return output


def _reduce_pair_losses(losses: Array, model_output: dict) -> Array:
    """Mean of the per-pair DPO losses.

    Without explicit loss masks this is ``losses.mean()``. With them, only pairs that score at
    least one token on both sides (``pair_valid``) enter the mean, so placeholder rows add neither
    a constant to the reported loss nor dilute the gradient.

    Args:
        losses: Per-pair losses, shape ``[num_pairs]``.
        model_output: Output of :func:`concatenated_forward`.

    Returns:
        Scalar loss.
    """
    pair_valid = model_output.get("pair_valid")
    if pair_valid is None:
        return losses.mean()
    valid = pair_valid > 0
    return jnp.sum(jnp.where(valid, losses, 0.0)) / jnp.maximum(jnp.sum(pair_valid), 1.0)


def _resolve_dpo_forward_mask_config(
    concatenated_forward_fn: tp.Callable | None,
) -> tuple[bool, int | None, str]:
    """Extract static mask/truncation knobs bound onto ``concatenated_forward_fn``."""
    fn: tp.Any = concatenated_forward_fn
    seen: set[int] = set()
    while fn is not None and id(fn) not in seen:
        seen.add(id(fn))
        keywords = getattr(fn, "keywords", None)
        if isinstance(keywords, dict) and any(
            k in keywords for k in ("is_encoder_decoder", "max_length", "truncation_mode")
        ):
            return (
                bool(keywords.get("is_encoder_decoder", False)),
                keywords.get("max_length", None),
                str(keywords.get("truncation_mode", "keep_end")),
            )
        fn = getattr(fn, "__wrapped__", None) or getattr(fn, "func", None)
    return False, None, "keep_end"


def _compute_dpo_microbatch_weight(
    minibatch: dict,
    *,
    is_encoder_decoder: bool,
    max_length: int | None,
    truncation_mode: str,
    weight_by_chosen_tokens: bool = False,
) -> Array:
    """Compute the valid-pair (or valid-chosen-token for pure SFT) weight of a DPO microbatch from masks alone.

    Mirrors the exact padding, paired truncation, and causal-shift mask construction in
    :func:`concatenated_forward` without running a model forward pass, so ``minibatch_call`` and
    scheduled MPMD accumulation combine microbatches proportionally to valid pair counts.
    """
    from ..utils import pad_to_length

    num_examples = minibatch["prompt_input_ids"].shape[0]
    chosen_attn = minibatch["chosen_attention_mask"]
    rejected_attn = minibatch["rejected_attention_mask"]
    chosen_loss = minibatch.get("chosen_loss_mask")
    rejected_loss = minibatch.get("rejected_loss_mask")
    if chosen_loss is None:
        chosen_loss = jnp.ones_like(chosen_attn)
    if rejected_loss is None:
        rejected_loss = jnp.ones_like(rejected_attn)

    max_comp_len = max(chosen_attn.shape[1], rejected_attn.shape[1])
    comp_attn = jnp.concatenate(
        (
            pad_to_length(chosen_attn, max_comp_len, pad_value=0),
            pad_to_length(rejected_attn, max_comp_len, pad_value=0),
        ),
        axis=0,
    )
    comp_loss = jnp.concatenate(
        (
            pad_to_length(jnp.asarray(chosen_loss), max_comp_len, pad_value=0),
            pad_to_length(jnp.asarray(rejected_loss), max_comp_len, pad_value=0),
        ),
        axis=0,
    )
    comp_scoring = comp_attn * comp_loss.astype(comp_attn.dtype)

    if is_encoder_decoder:
        loss_mask = comp_scoring.astype(jnp.bool_)
    else:
        prompt_attn = jnp.concatenate(
            [minibatch["prompt_attention_mask"], minibatch["prompt_attention_mask"]],
            axis=0,
        )
        loss_mask = jnp.concatenate([jnp.zeros_like(prompt_attn), comp_scoring], axis=1)
        (loss_mask,) = apply_paired_truncation(
            loss_mask,
            max_length=max_length,
            truncation_mode=truncation_mode,
        )
        loss_mask = jnp.roll(loss_mask, shift=-1, axis=1).astype(jnp.bool_).at[:, -1].set(False)

    scored_tokens = loss_mask.astype(jnp.int32).sum(-1)
    chosen_scored = scored_tokens[:num_examples]
    rejected_scored = scored_tokens[num_examples:]
    pair_valid = ((chosen_scored > 0) & (rejected_scored > 0)).astype(jnp.float32)
    if weight_by_chosen_tokens:
        return jnp.sum(chosen_scored.astype(jnp.float32) * pair_valid)
    return jnp.sum(pair_valid)


def _make_dpo_microbatch_weight_fn(
    batch: dict,
    *,
    concatenated_forward_fn: tp.Callable | None,
    loss_type: str | tuple[str, ...] | list[str],
    use_weighting: bool,
    rpo_alpha: float | None,
) -> tp.Callable[[tp.Any, dict], Array] | None:
    """Build a valid-pair microbatch weight function when explicit preference loss masks are present."""
    if batch.get("chosen_loss_mask") is None and batch.get("rejected_loss_mask") is None:
        return None
    is_encoder_decoder, max_length, truncation_mode = _resolve_dpo_forward_mask_config(concatenated_forward_fn)
    loss_types = _as_loss_type_tuple(loss_type)
    weight_by_chosen_tokens = (
        loss_types == ("sft",)
        and not use_weighting
        and not (rpo_alpha is not None and float(rpo_alpha) > 0.0)
    )

    def _weight_fn(_tree: tp.Any, minibatch: dict) -> Array:
        return _compute_dpo_microbatch_weight(
            minibatch,
            is_encoder_decoder=is_encoder_decoder,
            max_length=max_length,
            truncation_mode=truncation_mode,
            weight_by_chosen_tokens=weight_by_chosen_tokens,
        )

    return _weight_fn


def training_step(
    state: EasyDeLState,
    batch: dict,
    reference_state: EasyDeLState | None,
    learning_rate_fn: tp.Callable,
    concatenated_forward: tp.Callable,
    beta: float = 0.1,
    label_smoothing: float = 0,
    loss_type: LOSS_FN_VARIANTS | tuple[str, ...] = "sigmoid",
    loss_weights: tuple[float, ...] | None = None,
    f_divergence_type: str = "reverse_kl",
    f_alpha_divergence_coef: float = 0.5,
    use_weighting: bool = False,
    discopop_tau: float = 0.05,
    ld_alpha: float | None = None,
    rpo_alpha: float | None = None,
    reference_free: bool = False,
    loss_config: LossConfig | None = None,
    partition_spec: PartitionSpec | None = None,
    gradient_accumulation_steps: int = 1,
    straight_through_emulator: tp.Callable[[tp.Any], tp.Any] | None = None,
) -> tuple[EasyDeLState, LossMetrics]:
    """Run one DPO training step (forward, loss, backward, optimizer update).

    The DPO objective is a maximum-likelihood-on-preferences surrogate
    (Rafailov et al. 2023): given a preference pair ``(x, y_w, y_l)``,
    the policy is updated so its log-ratio against a *frozen* reference
    increases on the preferred completion and decreases on the
    dispreferred one. This function executes one such update:

    1. Validate batch shapes and resolve the gradient-accumulation
       minibatch size.
    2. If not ``reference_free``, fetch ``ref_chosen_logps`` and
       ``ref_rejected_logps`` either from the batch (if precomputed by
       :class:`DPOPreprocessTransform`) or by running ``reference_state``
       through ``concatenated_forward`` outside the gradient trace
       (avoids ``nn.remat`` retrace conflicts when the reference model
       checkpoints).
    3. Run ``minibatch_call`` over the local batch to accumulate the
       value-and-grad of the inner :func:`calculate_loss` closure.
    4. Update the optimizer state via ``update_state_respectfully``
       (NaN-aware).

    Args:
        state: Current policy ``EasyDeLState`` (graphdef + graphstate +
            optimizer state). Differentiation target.
        batch: Preference minibatch carrying paired
            ``prompt_input_ids``, ``chosen_input_ids``,
            ``rejected_input_ids`` and their attention masks (and
            optionally precomputed ``ref_chosen_logps``/``ref_rejected_logps``).
        reference_state: Frozen reference-model state used when the
            batch does not already carry reference logps. Ignored when
            ``reference_free`` is set.
        learning_rate_fn: Schedule mapping ``state.step -> lr``.
        concatenated_forward: Forward closure built by the trainer that
            packs chosen/rejected through one model call and returns
            ``{"chosen_logps", "rejected_logps", "mean_*_logits"}``.
        beta: DPO inverse-temperature on the log-ratio differential.
        label_smoothing: cDPO smoothing factor (``[0, 0.5)``).
        loss_type: Variant key passed through :func:`get_loss_function`.
        reference_free: If ``True``, replaces reference logps with zeros
            (PPO-style implicit-reward baseline).
        loss_config: Optional ``LossConfig`` controlling NaN handling
            inside ``update_state_respectfully``.
        partition_spec: Sharding spec applied to the input batch under
            the model's mesh.
        gradient_accumulation_steps: Number of gradient-accumulation
            sub-steps; the batch must be divisible by this.
        straight_through_emulator: Optional STE callable that rewrites
            the policy graphstate before the forward pass (used by QAT).

    Returns:
        ``(new_state, metrics)`` where ``metrics`` is a ``LossMetrics``
        with the mean DPO loss, per-example ``chosen_rewards`` /
        ``rejected_rewards`` (``beta * (logp - logp_ref)``, stop-gradient'd),
        and the standard learning-rate / gradient-norm fields.
    """
    scope_root = "easydel/trainer/dpo/train_step"
    with jax.named_scope(scope_root + "/prepare_batch"):
        _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
            batch=batch,
            gradient_accumulation_steps=gradient_accumulation_steps,
            batch_partition_spec=partition_spec,
        )

        batch = with_sharding_constraint(batch, partition_spec, mesh=state.model.mesh, ignore_mpmd=True)
    if not reference_free:
        with jax.named_scope(scope_root + "/reference_forward"):
            # Pre-compute reference logps outside jax.value_and_grad to avoid
            # nn.remat trace-level conflicts when the reference model uses
            # gradient checkpointing inside the grad trace. Microbatch under
            # gradient accumulation so the reference forward stays within the
            # same peak memory envelope as the policy microbatch forward.
            ref_chosen_logps, ref_rejected_logps = _get_reference_logps_from_batch(batch)
            if ref_chosen_logps is None or ref_rejected_logps is None:
                rfm = state.model if reference_state is None else reference_state.model
                num_accum_steps = _batch_size // minibatch_size
                if num_accum_steps > 1:
                    start_indices = jnp.arange(num_accum_steps, dtype=jnp.int32) * minibatch_size

                    def _ref_microbatch(_, start_index):
                        mb = _slice_batch_for_scheduled_step(
                            batch,
                            _batch_size,
                            start_index,
                            minibatch_size,
                            dynamic=True,
                        )
                        mb_out = jax.lax.stop_gradient(concatenated_forward(rfm, mb, ld_alpha=ld_alpha))
                        return None, (mb_out["chosen_logps"], mb_out["rejected_logps"])

                    _, (mb_chosen, mb_rejected) = lax.scan(_ref_microbatch, None, start_indices, length=num_accum_steps)
                    ref_chosen_logps = mb_chosen.reshape((_batch_size,))
                    ref_rejected_logps = mb_rejected.reshape((_batch_size,))
                else:
                    ref_out = jax.lax.stop_gradient(concatenated_forward(rfm, batch, ld_alpha=ld_alpha))
                    ref_chosen_logps = ref_out["chosen_logps"]
                    ref_rejected_logps = ref_out["rejected_logps"]

            if "ref_chosen_logps" not in batch or "ref_rejected_logps" not in batch:
                batch = {
                    **batch,
                    "ref_chosen_logps": ref_chosen_logps,
                    "ref_rejected_logps": ref_rejected_logps,
                }

    def calculate_loss(tree: spx.State, call_batch):
        """Compute the DPO loss + metrics for a single minibatch.

        Steps inside the value-and-grad trace:

        1. Optionally rewrite ``tree`` through ``straight_through_emulator``
           (QAT path).
        2. Merge ``tree`` with the captured ``state.graphdef`` to materialize
           a callable model module.
        3. Run ``concatenated_forward`` to obtain
           ``{"chosen_logps", "rejected_logps", ...}`` for the policy.
        4. Substitute zero reference logps when ``reference_free``,
           otherwise read precomputed (and stop-gradient'd) reference
           logps from ``call_batch``.
        5. Apply ``_loss_func`` (a closure of ``beta`` /
           ``label_smoothing`` / ``loss_type``) to the four logp arrays.
        6. Add ``aux_loss`` from the model output if present (load
           balancing / MoE auxiliaries are forwarded straight through).

        Reference logps are always wrapped in ``stop_gradient`` so the
        optimizer can never leak gradients into the reference model.

        Args:
            tree: Policy graphstate to differentiate against.
            call_batch: Per-microbatch slice of the outer ``batch``.

        Returns:
            ``(loss_scalar, LossMetrics)`` where ``loss_scalar`` is the
            mean per-example DPO loss and ``LossMetrics`` carries
            stop-gradient'd ``chosen_rewards`` / ``rejected_rewards`` for
            logging.
        """
        with jax.named_scope(scope_root + "/loss_fn"):
            if straight_through_emulator is not None:
                with jax.named_scope(scope_root + "/loss_fn/straight_through_emulator"):
                    tree = straight_through_emulator(tree)
            with jax.named_scope(scope_root + "/loss_fn/merge_state"):
                module = state.merge(tree=tree)

            with jax.named_scope(scope_root + "/loss_fn/policy_forward"):
                model_output = concatenated_forward(module, call_batch, ld_alpha=ld_alpha)

            chosen_logps = model_output["chosen_logps"]
            rejected_logps = model_output["rejected_logps"]
            with jax.named_scope(scope_root + "/loss_fn/resolve_reference_logps"):
                if reference_free:
                    ref_chosen_logps = jnp.zeros_like(chosen_logps)
                    ref_rejected_logps = jnp.zeros_like(rejected_logps)
                else:
                    ref_chosen_logps = jax.lax.stop_gradient(call_batch["ref_chosen_logps"])
                    ref_rejected_logps = jax.lax.stop_gradient(call_batch["ref_rejected_logps"])
            with jax.named_scope(scope_root + "/loss_fn/compute_dpo_loss"):
                losses = compute_dpo_losses(
                    chosen_logps,
                    rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta=beta,
                    label_smoothing=label_smoothing,
                    loss_type=loss_type,
                    loss_weights=loss_weights,
                    f_divergence_type=f_divergence_type,
                    f_alpha_divergence_coef=f_alpha_divergence_coef,
                    discopop_tau=discopop_tau,
                    chosen_lengths=model_output.get("chosen_lengths"),
                    rejected_lengths=model_output.get("rejected_lengths"),
                    wpo_weights=model_output.get("wpo_weights") if use_weighting else None,
                    rpo_alpha=rpo_alpha,
                    pair_valid=model_output.get("pair_valid"),
                )

            with jax.named_scope(scope_root + "/loss_fn/rewards_and_metrics"):
                chosen_rewards = beta * jax.lax.stop_gradient(chosen_logps - ref_chosen_logps)
                rejected_rewards = beta * jax.lax.stop_gradient(rejected_logps - ref_rejected_logps)
                aux_loss = model_output.get("aux_loss")
                if aux_loss is not None:
                    losses = losses + aux_loss

                metrics = LossMetrics(
                    loss=_reduce_pair_losses(losses, model_output),
                    rejected_rewards=rejected_rewards,
                    chosen_rewards=chosen_rewards,
                )
            return metrics.loss, metrics

    dpo_microbatch_weight_fn = (
        _make_dpo_microbatch_weight_fn(
            batch,
            concatenated_forward_fn=concatenated_forward,
            loss_type=loss_type,
            use_weighting=use_weighting,
            rpo_alpha=rpo_alpha,
        )
        if gradient_accumulation_steps > 1
        else None
    )

    with jax.named_scope(scope_root + "/grad_and_minibatch"):
        gradients, metrics = minibatch_call(
            state=state,
            batch=batch,
            minibatch_size=minibatch_size,
            grad_fn=jax.value_and_grad(calculate_loss, has_aux=True),
            microbatch_weight_fn=dpo_microbatch_weight_fn,
        )

    with jax.named_scope(scope_root + "/update_state"):
        metrics = update_metrics(
            metrics=metrics,
            learning_rate_fn=learning_rate_fn,
            step=state.step,
            gradients=gradients,
        )
        state = update_state_respectfully(
            state=state,
            gradients=gradients,
            loss_config=loss_config,
            metrics=metrics,
        )
    return (state, metrics)


def _prepare_dpo_scheduled_batch(call) -> dict[str, tp.Any]:
    """Inject reference chosen/rejected log-probabilities into ``call.batch``.

    When the batch already supplies reference logps (via either the
    canonical or legacy column names), it is returned untouched.  In
    reference-free mode the trainer skips this hook altogether (handled
    inside the loss closure).

    Args:
        call: The :class:`ScheduledStepCall` being prepared.

    Returns:
        A copy of ``call.batch`` with ``ref_chosen_logps`` and
        ``ref_rejected_logps`` populated.
    """
    batch = dict(call.batch)
    ref_chosen_logps, ref_rejected_logps = _get_reference_logps_from_batch(batch)
    if ref_chosen_logps is not None and ref_rejected_logps is not None:
        return batch

    return prepare_scheduled_reference_outputs(
        call,
        reference_state_field="reference_state",
        forward_field="concatenated_forward",
        output_to_batch={
            "chosen_logps": "ref_chosen_logps",
            "rejected_logps": "ref_rejected_logps",
        },
        forward_kwargs={"ld_alpha": call.get("ld_alpha")} if call.get("ld_alpha") is not None else None,
        skip_field="reference_free",
        missing_error="DPO scheduled MPMD training requires reference_state and concatenated_forward.",
    )


def _dpo_scheduled_loss_cache_key(call) -> tuple[tp.Any, ...]:
    """Build a cache key for the DPO scheduled-loss compilation.

    Args:
        call: The current :class:`ScheduledStepCall`.

    Returns:
        A tuple covering DPO knobs that influence compilation
        (``beta``, ``label_smoothing``, ``loss_type``,
        ``reference_free``, partition spec, and forward fn / quantizer
        identities).
    """
    base_key = scheduled_loss_cache_key(
        call,
        value_fields=(
            "beta",
            "label_smoothing",
            "loss_type",
            "loss_weights",
            "f_divergence_type",
            "f_alpha_divergence_coef",
            "use_weighting",
            "discopop_tau",
            "ld_alpha",
            "rpo_alpha",
            "reference_free",
            "partition_spec",
        ),
        object_fields=("concatenated_forward", "straight_through_emulator"),
    )
    has_explicit_loss_mask = (call.batch.get("chosen_loss_mask") is not None) or (
        call.batch.get("rejected_loss_mask") is not None
    )
    return (*base_key, ("has_explicit_loss_mask", has_explicit_loss_mask))


def _make_dpo_scheduled_weight(call):
    """Build a valid-pair microbatch weight callable for scheduled MPMD DPO accumulation."""
    return _make_dpo_microbatch_weight_fn(
        call.batch,
        concatenated_forward_fn=call.get("concatenated_forward"),
        loss_type=call.get("loss_type", "sigmoid"),
        use_weighting=bool(call.get("use_weighting", False)),
        rpo_alpha=call.get("rpo_alpha"),
    )


def _make_dpo_scheduled_loss(call):
    """Build a SpectraX-scheduled DPO scalar-loss closure for ``call``.

    Args:
        call: The :class:`ScheduledStepCall` providing forward fn,
            ``beta``, ``label_smoothing``, loss-type, etc.

    Returns:
        A closure ``loss_fn(tree, batch) -> scalar`` ready for
        :func:`spx.sxvalue_and_grad`.
    """
    concatenated_forward_fn = call.get("concatenated_forward")
    beta = call.get("beta", 0.1)
    label_smoothing = call.get("label_smoothing", 0)
    loss_type = call.get("loss_type", "sigmoid")
    loss_weights = call.get("loss_weights")
    f_divergence_type = call.get("f_divergence_type", "reverse_kl")
    f_alpha_divergence_coef = call.get("f_alpha_divergence_coef", 0.5)
    use_weighting = bool(call.get("use_weighting", False))
    discopop_tau = call.get("discopop_tau", 0.05)
    ld_alpha = call.get("ld_alpha")
    rpo_alpha = call.get("rpo_alpha")
    reference_free = bool(call.get("reference_free", False))
    partition_spec = call.get("partition_spec")
    bind_module = make_scheduled_module_binder(call, binder_fn=bind_scheduled_module)
    schedule = call.schedule

    def scheduled_loss(tree: spx.State, batch: dict[str, tp.Any]):
        """Compute the scalar DPO loss inside the SpectraX scheduled VJP.

        Args:
            tree: Policy graphstate to differentiate against.
            batch: Minibatch dict carrying preference triples and (when
                not reference-free) precomputed reference logps.

        Returns:
            The mean DPO loss with optional aux-loss term.

        Raises:
            RuntimeError: If reference logps are missing while the
                trainer is not in reference-free mode.
        """
        with jax.named_scope("easydel/trainer/dpo/scheduled_loss"):
            with jax.named_scope("easydel/trainer/dpo/scheduled_loss/bind_module"):
                module = bind_module(tree)
                batch = constrain_scheduled_batch(module, batch, partition_spec)
                _terminal_rank = _scheduled_terminal_stage_rank(module, schedule)
            with jax.named_scope("easydel/trainer/dpo/scheduled_loss/policy_forward"):
                model_output = concatenated_forward_fn(
                    module, batch, vocab_shard_stage=_terminal_rank, ld_alpha=ld_alpha
                )

            chosen_logps = model_output["chosen_logps"]
            rejected_logps = model_output["rejected_logps"]
            with jax.named_scope("easydel/trainer/dpo/scheduled_loss/resolve_reference_logps"):
                if reference_free:
                    ref_chosen_logps = jnp.zeros_like(chosen_logps)
                    ref_rejected_logps = jnp.zeros_like(rejected_logps)
                else:
                    ref_chosen_logps, ref_rejected_logps = _get_reference_logps_from_batch(batch)
                    if ref_chosen_logps is None or ref_rejected_logps is None:
                        raise RuntimeError(
                            "DPO scheduled MPMD loss requires precomputed reference log-probs in the batch."
                        )
                    ref_chosen_logps = jax.lax.stop_gradient(ref_chosen_logps)
                    ref_rejected_logps = jax.lax.stop_gradient(ref_rejected_logps)

            with jax.named_scope("easydel/trainer/dpo/scheduled_loss/compute_dpo_loss"):
                losses = compute_dpo_losses(
                    chosen_logps,
                    rejected_logps,
                    ref_chosen_logps,
                    ref_rejected_logps,
                    beta=beta,
                    label_smoothing=label_smoothing,
                    loss_type=loss_type,
                    loss_weights=loss_weights,
                    f_divergence_type=f_divergence_type,
                    f_alpha_divergence_coef=f_alpha_divergence_coef,
                    discopop_tau=discopop_tau,
                    chosen_lengths=model_output.get("chosen_lengths"),
                    rejected_lengths=model_output.get("rejected_lengths"),
                    wpo_weights=model_output.get("wpo_weights") if use_weighting else None,
                    rpo_alpha=rpo_alpha,
                    pair_valid=model_output.get("pair_valid"),
                )
                aux_loss = model_output.get("aux_loss")
                if aux_loss is not None:
                    losses = losses + aux_loss
            return _reduce_pair_losses(losses, model_output)

    return scheduled_loss


register_scheduled_loss_adapter(
    step_fn=training_step,
    adapter=ScheduledLossAdapter(
        name="dpo",
        make_loss=_make_dpo_scheduled_loss,
        make_cache_key=_dpo_scheduled_loss_cache_key,
        prepare_batch=_prepare_dpo_scheduled_batch,
        make_microbatch_weight=_make_dpo_scheduled_weight,
    ),
)


def evaluation_step(
    state: EasyDeLState,
    batch: dict,
    reference_state: EasyDeLState | None,
    concatenated_forward: tp.Callable,
    beta: float = 0.1,
    label_smoothing: float = 0,
    loss_type: LOSS_FN_VARIANTS | tuple[str, ...] = "sigmoid",
    loss_weights: tuple[float, ...] | None = None,
    f_divergence_type: str = "reverse_kl",
    f_alpha_divergence_coef: float = 0.5,
    use_weighting: bool = False,
    discopop_tau: float = 0.05,
    ld_alpha: float | None = None,
    rpo_alpha: float | None = None,
    reference_free: bool = False,
    partition_spec: PartitionSpec | None = None,
) -> LossMetrics:
    """Run one DPO evaluation step (forward only, no parameter update).

    Computes the same loss and reward metrics as :func:`training_step`
    but without gradient accumulation or optimizer interaction. When
    ``reference_state`` is ``None`` and the batch does not carry
    precomputed reference logps, the policy itself stands in as the
    reference; this is mainly a convenience for sanity checks (the
    reported loss is then trivially zero on the canonical sigmoid
    variant).

    Args:
        state: Current policy state to evaluate.
        batch: Preference minibatch (same structure as
            :func:`training_step`).
        reference_state: Optional reference state. If ``None`` and the
            batch lacks precomputed reference logps, the policy stands
            in as its own reference (purely diagnostic).
        concatenated_forward: Forward closure built by the trainer.
        beta: DPO inverse-temperature.
        label_smoothing: cDPO smoothing factor.
        loss_type: Variant key.
        reference_free: When ``True``, reference logps are zeroed out
            and the loss reduces to a policy-only objective.
        partition_spec: Sharding spec applied to the input batch.

    Returns:
        ``LossMetrics`` with ``loss``, ``chosen_rewards``, and
        ``rejected_rewards`` populated.
    """
    eval_scope = "easydel/trainer/dpo/eval_step"
    with jax.named_scope(eval_scope + "/prepare_batch"):
        *_, partition_spec = make_assertions_and_get_sizes(
            batch=batch,
            gradient_accumulation_steps=1,
            batch_partition_spec=partition_spec,
        )

        batch = with_sharding_constraint(batch, partition_spec, mesh=state.model.mesh, ignore_mpmd=True)

    def calculate_loss(tree: spx.State):
        """Compute DPO eval metrics on the captured ``batch``.

        Mirrors :func:`training_step.calculate_loss` minus the STE /
        gradient plumbing: merges the policy ``tree`` with the captured
        graphdef, runs ``concatenated_forward``, resolves reference
        logps (precomputed, reference state forward, or fallback to the
        policy itself), evaluates ``_loss_func``, and wraps the result
        in a ``LossMetrics`` object with ``chosen_rewards`` /
        ``rejected_rewards`` (no stop-gradient is needed at eval time).

        Args:
            tree: Policy graphstate to evaluate against.

        Returns:
            ``LossMetrics`` populated with the mean loss and the
            implicit reward arrays.
        """
        with jax.named_scope(eval_scope + "/loss_fn"):
            with jax.named_scope(eval_scope + "/loss_fn/policy_forward"):
                model_output = concatenated_forward(state.merge(tree), batch, ld_alpha=ld_alpha)
            chosen_logps = model_output["chosen_logps"]
            rejected_logps = model_output["rejected_logps"]

            with jax.named_scope(eval_scope + "/loss_fn/resolve_reference_logps"):
                if reference_free:
                    ref_chosen_for_loss = jnp.zeros_like(chosen_logps)
                    ref_rejected_for_loss = jnp.zeros_like(rejected_logps)
                else:
                    ref_chosen_logps, ref_rejected_logps = _get_reference_logps_from_batch(batch)
                    if ref_chosen_logps is None or ref_rejected_logps is None:
                        ref_model = state.model if reference_state is None else reference_state.model
                        ref_output = concatenated_forward(ref_model, batch, ld_alpha=ld_alpha)
                        ref_chosen_logps = ref_output["chosen_logps"]
                        ref_rejected_logps = ref_output["rejected_logps"]
                    ref_chosen_for_loss = ref_chosen_logps
                    ref_rejected_for_loss = ref_rejected_logps

            with jax.named_scope(eval_scope + "/loss_fn/compute_dpo_loss"):
                losses = compute_dpo_losses(
                    chosen_logps,
                    rejected_logps,
                    ref_chosen_for_loss,
                    ref_rejected_for_loss,
                    beta=beta,
                    label_smoothing=label_smoothing,
                    loss_type=loss_type,
                    loss_weights=loss_weights,
                    f_divergence_type=f_divergence_type,
                    f_alpha_divergence_coef=f_alpha_divergence_coef,
                    discopop_tau=discopop_tau,
                    chosen_lengths=model_output.get("chosen_lengths"),
                    rejected_lengths=model_output.get("rejected_lengths"),
                    wpo_weights=model_output.get("wpo_weights") if use_weighting else None,
                    rpo_alpha=rpo_alpha,
                    pair_valid=model_output.get("pair_valid"),
                )

            with jax.named_scope(eval_scope + "/loss_fn/rewards_and_metrics"):
                chosen_rewards = beta * (chosen_logps - ref_chosen_for_loss)
                rejected_rewards = beta * (rejected_logps - ref_rejected_for_loss)

                metrics = LossMetrics(
                    loss=_reduce_pair_losses(losses, model_output),
                    rejected_rewards=rejected_rewards,
                    chosen_rewards=chosen_rewards,
                )
            return metrics

    with jax.named_scope(eval_scope + "/eval_call"):
        metrics = calculate_loss(state.graphstate)
    return metrics
