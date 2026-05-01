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

from .._logprob_utils import compute_token_logps_and_entropies_chunked, resolve_lmhead_chunksize
from .._shared import apply_paired_truncation, gather_multimodal_kwargs
from ..training_utils import (
    ScheduledLossAdapter,
    bind_scheduled_module,
    constrain_scheduled_batch,
    filter_kwargs_for_callable,
    make_assertions_and_get_sizes,
    minibatch_call,
    prepare_scheduled_reference_outputs,
    register_scheduled_loss_adapter,
    sanitize_model_call_kwargs,
    scheduled_loss_cache_key,
    update_metrics,
    update_state_respectfully,
)
from ..utils import pad_to_length
from .dpo_config import LOSS_FN_VARIANTS


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


# Backwards-compatible alias retained for any in-tree references; new call
# sites should use :func:`resolve_lmhead_chunksize` from ``_logprob_utils``
# directly.
_resolve_dpo_lmhead_chunksize = resolve_lmhead_chunksize


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

    When ``loss_type`` is ``"ipo"``, the accumulated log-probabilities are
    normalized by the number of loss-bearing tokens per example (as required
    by the IPO objective).

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
        loss_type: The DPO loss variant in use (e.g. ``"sigmoid"``,
            ``"ipo"``).  Only ``"ipo"`` triggers per-example length
            normalization of the returned log-probabilities.

    Returns:
        A dictionary with the following keys:

        - ``"chosen_logps"`` -- Float array of shape ``(num_examples,)`` with
          the summed (or length-normalized for IPO) log-probabilities for the
          chosen completions.
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

    _lm_head_fn = model.make_lm_head_fn() if hasattr(model, "make_lm_head_fn") else model.compute_lm_logits
    _has_prepare = hasattr(model, "prepare_lm_head_inputs")

    def _project_chunk(chunk_hidden_states: Array) -> Array:
        """Project a sequence-axis hidden-state chunk through the LM head.

        Args:
            chunk_hidden_states: ``[batch, chunk_seq, hidden_dim]`` slice
                of hidden states.

        Returns:
            The corresponding ``[batch, chunk_seq, vocab_size]`` logits.
        """
        if _has_prepare:
            chunk_hidden_states = model.prepare_lm_head_inputs(chunk_hidden_states)
        return _lm_head_fn(chunk_hidden_states)

    _project_chunk = jax.checkpoint(_project_chunk, prevent_cse=False)

    def _chunk_contributions(
        chunk_hidden_states: Array,
        chunk_labels: Array,
        chunk_loss_mask: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Compute per-sequence per-chunk DPO log-prob and logit accumulators.

        Args:
            chunk_hidden_states: Hidden-state slice for the current
                chunk.
            chunk_labels: Label slice with the same sequence range.
            chunk_loss_mask: Boolean loss mask slice.

        Returns:
            A 5-tuple with masked log-prob sums, chosen / rejected
            logit sums, and chosen / rejected token counts for the
            current chunk.
        """
        chunk_logits = _project_chunk(chunk_hidden_states)
        chunk_logps = _compute_token_logps_chunked(
            chunk_logits,
            chunk_labels,
            chunk_size=logprob_vocab_chunk_size,
        )
        masked_logps = jnp.where(chunk_loss_mask, chunk_logps, 0.0)
        chunk_token_logit_sums = chunk_logits.astype(jnp.float32).sum(axis=-1)
        chosen_mask = chunk_loss_mask[:num_examples].astype(jnp.float32)
        rejected_mask = chunk_loss_mask[num_examples:].astype(jnp.float32)
        return (
            masked_logps.sum(axis=-1),
            jnp.sum(chunk_token_logit_sums[:num_examples] * chosen_mask),
            jnp.sum(chunk_token_logit_sums[num_examples:] * rejected_mask),
            jnp.sum(chosen_mask),
            jnp.sum(rejected_mask),
        )

    _chunk_contributions = jax.checkpoint(_chunk_contributions, prevent_cse=False)

    zero_logps = jnp.zeros((batch_size,), dtype=jnp.float32)
    zero_scalar = jnp.array(0.0, dtype=jnp.float32)

    def _accumulate_chunk(
        start: int,
        size: int,
        carry: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array]:
        """Add the contributions of a sequence-chunk into the running totals.

        Args:
            start: Start index along the sequence axis.
            size: Number of tokens in this chunk.
            carry: Running ``(logp_sums, chosen_logit_sum,
                rejected_logit_sum, chosen_count, rejected_count)``.

        Returns:
            The updated carry with the chunk's contributions added.
        """
        chunk_hidden_states = lax.dynamic_slice_in_dim(hidden_states, start, size, axis=1)
        chunk_labels = lax.dynamic_slice_in_dim(labels, start, size, axis=1)
        chunk_loss_mask = lax.dynamic_slice_in_dim(loss_mask, start, size, axis=1)
        (
            chunk_logp_sums,
            chosen_logit_sum,
            rejected_logit_sum,
            chosen_denom,
            rejected_denom,
        ) = _chunk_contributions(chunk_hidden_states, chunk_labels, chunk_loss_mask)
        return (
            carry[0] + chunk_logp_sums,
            carry[1] + chosen_logit_sum,
            carry[2] + rejected_logit_sum,
            carry[3] + chosen_denom,
            carry[4] + rejected_denom,
        )

    num_full_chunks = seq_len // chunk_size
    tail = seq_len - num_full_chunks * chunk_size
    carry = (zero_logps, zero_scalar, zero_scalar, zero_scalar, zero_scalar)

    def _full_body(
        i: int,
        inner_carry: tuple[Array, Array, Array, Array, Array],
    ) -> tuple[Array, Array, Array, Array, Array]:
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

    all_logps, chosen_logit_sum, rejected_logit_sum, chosen_denom, rejected_denom = carry
    if loss_type == "ipo":
        all_logps = all_logps / jnp.maximum(loss_mask.sum(axis=-1), 1)

    return {
        "chosen_logps": all_logps[:num_examples],
        "rejected_logps": all_logps[num_examples:],
        "mean_chosen_logits": chosen_logit_sum / jnp.maximum(chosen_denom, 1.0),
        "mean_rejected_logits": rejected_logit_sum / jnp.maximum(rejected_denom, 1.0),
    }


def _get_reference_logps_from_batch(batch: dict[str, tp.Any]) -> tuple[tp.Any | None, tp.Any | None]:
    """Read reference log-prob columns from either the canonical or legacy keys."""
    ref_chosen_logps = batch.get("ref_chosen_logps")
    if ref_chosen_logps is None:
        ref_chosen_logps = batch.get("reference_chosen_log_probs")

    ref_rejected_logps = batch.get("ref_rejected_logps")
    if ref_rejected_logps is None:
        ref_rejected_logps = batch.get("reference_rejected_log_probs")

    return ref_chosen_logps, ref_rejected_logps


def concatenated_inputs(
    batch: dict[str, list | Array],
    padding_value: int,
) -> dict[str, Array]:
    """
    Concatenates chosen and rejected examples from the batch, and pads the inputs to a uniform length.

    This function is used to merge paired inputs (e.g. chosen vs. rejected examples)
    so that the model can process them in one forward pass. It concatenates the prompt inputs,
    attention masks, and (if present) image-related arrays. The completion inputs (and their attention masks)
    are padded to the length of the longest completion among the chosen and rejected examples.

    Args:
        batch (tp.Dict[str, tp.Union[tp.List, Array]]):
            A dictionary containing the batch of data. Expected keys include:
            - "prompt_input_ids", "prompt_attention_mask"
            - "chosen_input_ids", "rejected_input_ids"
            - "chosen_attention_mask", "rejected_attention_mask"
            Optionally, keys like "pixel_values", "pixel_attention_mask", and "image_sizes" may be present.
        padding_value (int): The padding value to use when padding completion inputs.

    Returns:
        tp.Dict[str, Array]: A dictionary with concatenated arrays under keys such as:
            - "prompt_input_ids", "prompt_attention_mask"
            - "completion_input_ids", "completion_attention_mask"
            and optionally image-related keys.
    """
    output = {}
    # Concatenate the prompt-related arrays (duplicated for chosen and rejected).
    output["prompt_input_ids"] = jnp.concatenate(
        [batch["prompt_input_ids"], batch["prompt_input_ids"]],
        axis=0,
    )
    output["prompt_attention_mask"] = jnp.concatenate(
        [batch["prompt_attention_mask"], batch["prompt_attention_mask"]],
        axis=0,
    )
    if "pixel_values" in batch:
        output["pixel_values"] = jnp.concatenate(
            [batch["pixel_values"], batch["pixel_values"]],
            axis=0,
        )
    if "pixel_attention_mask" in batch:
        output["pixel_attention_mask"] = jnp.concatenate(
            [batch["pixel_attention_mask"], batch["pixel_attention_mask"]],
            axis=0,
        )
    if "image_sizes" in batch:
        output["image_sizes"] = jnp.concatenate(
            [batch["image_sizes"], batch["image_sizes"]],
            axis=0,
        )

    # Determine maximum length for the completion inputs.
    max_completion_length = max(
        batch["chosen_input_ids"].shape[1],
        batch["rejected_input_ids"].shape[1],
    )
    # Pad chosen and rejected completion input IDs to the same length and concatenate them.
    output["completion_input_ids"] = jnp.concatenate(
        (
            pad_to_length(
                batch["chosen_input_ids"],
                max_completion_length,
                pad_value=padding_value,
            ),
            pad_to_length(
                batch["rejected_input_ids"],
                max_completion_length,
                pad_value=padding_value,
            ),
        ),
    )
    # Similarly pad and concatenate the attention masks.
    output["completion_attention_mask"] = jnp.concatenate(
        (
            pad_to_length(
                batch["chosen_attention_mask"],
                max_completion_length,
                pad_value=0,
            ),
            pad_to_length(
                batch["rejected_attention_mask"],
                max_completion_length,
                pad_value=0,
            ),
        ),
    )

    return output


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

    def _base_dpo_loss(
        chosen_logps: Array,
        rejected_logps: Array,
        ref_chosen_logps: Array,
        ref_rejected_logps: Array,
        beta: float,
        label_smoothing: float,
        **kwargs,
    ) -> tuple[Array, Array, Array]:
        """Compute the policy/reference log-ratio differential at the heart of DPO.

        Given the four per-example summed log-probabilities, this helper
        returns ``logits = logratios - ref_logratios`` (also known as the
        DPO implicit reward differential) along with the two intermediate
        log-ratios so callers can reuse them. ``logits`` is what every
        DPO variant feeds into its scalar shaping function (sigmoid,
        squared, hinge, ...).

        Args:
            chosen_logps: ``[batch]`` summed log-prob of the chosen
                completion under the *policy*.
            rejected_logps: ``[batch]`` for the rejected completion
                under the policy.
            ref_chosen_logps: Same as ``chosen_logps`` but under the
                frozen reference model.
            ref_rejected_logps: Same as ``rejected_logps`` but under
                the reference model.
            beta: Unused in this base helper; accepted for signature
                uniformity with the variant closures.
            label_smoothing: Unused here.
            **kwargs: Ignored.

        Returns:
            ``(logits, logratios, ref_logratios)`` where each entry has
            shape ``[batch]`` and ``logits = logratios - ref_logratios``.
        """
        logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = logratios - ref_logratios
        return logits, logratios, ref_logratios

    def _sigmoid_dpo_loss(
        chosen_logps: Array,
        rejected_logps: Array,
        ref_chosen_logps: Array,
        ref_rejected_logps: Array,
        beta: float,
        label_smoothing: float,
        **kwargs,
    ) -> Array:
        """Compute the canonical DPO loss (Rafailov et al. 2023).

        Returns ``-(1 - eps) * logsigmoid(beta * h) - eps * logsigmoid(-beta * h)``
        where ``h`` is the policy-vs-reference log-ratio differential
        from :func:`_base_dpo_loss` and ``eps = label_smoothing`` is the
        cDPO smoothing knob (Mitchell 2023). With ``eps = 0`` this
        reduces to the original DPO objective.

        Returns:
            ``[batch]`` per-example loss tensor.
        """
        logits, _, _ = _base_dpo_loss(
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta,
            label_smoothing,
        )
        return -(
            jax.nn.log_sigmoid(beta * logits) * (1 - label_smoothing)
            + jax.nn.log_sigmoid(-beta * logits) * label_smoothing
        )

    def _nca_pair_dpo_loss(
        chosen_logps: Array,
        rejected_logps: Array,
        ref_chosen_logps: Array,
        ref_rejected_logps: Array,
        beta: float,
        label_smoothing: float,
        **kwargs,
    ) -> Array:
        """Compute the pair NCA-style DPO loss (Chen et al. 2024).

        Combines a positive-likelihood term on the chosen reward with
        symmetric noise-contrastive penalties on both halves:
        ``-logsigmoid(r_w) - 0.5 * logsigmoid(-r_w) - 0.5 * logsigmoid(-r_l)``
        where ``r_w/r_l = beta * (logp - logp_ref)`` are the implicit
        DPO rewards. Ignores ``label_smoothing``.

        Returns:
            ``[batch]`` per-example loss tensor.
        """
        chosen_rewards = (chosen_logps - ref_chosen_logps) * beta
        rejected_rewards = (rejected_logps - ref_rejected_logps) * beta
        return -(
            jax.nn.log_sigmoid(chosen_rewards)
            + 0.5 * jax.nn.log_sigmoid(-chosen_rewards)
            + 0.5 * jax.nn.log_sigmoid(-rejected_rewards)
        )

    def _aot_dpo_loss(
        chosen_logps: Array,
        rejected_logps: Array,
        ref_chosen_logps: Array,
        ref_rejected_logps: Array,
        beta: float,
        label_smoothing: float,
        **kwargs,
    ) -> Array:
        """Compute the AOT (Alignment via Optimal Transport) DPO loss.

        Sorts both the policy and reference log-ratios across the batch
        axis and applies the standard sigmoid DPO objective on the
        rank-aligned differences. This reframes preference matching as
        a 1-D optimal-transport problem along the batch.

        Returns:
            ``[batch]`` per-example loss tensor (over the sorted permutation).
        """
        logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logratios_sorted = jnp.sort(logratios, axis=0)
        ref_logratios_sorted = jnp.sort(ref_logratios, axis=0)
        delta = logratios_sorted - ref_logratios_sorted
        return -(
            jax.nn.log_sigmoid(beta * delta) * (1 - label_smoothing)
            + jax.nn.log_sigmoid(-beta * delta) * label_smoothing
        )

    def _discopop_dpo_loss(
        chosen_logps: Array,
        rejected_logps: Array,
        ref_chosen_logps: Array,
        ref_rejected_logps: Array,
        beta: float,
        label_smoothing: float,
        discopop_tau: float = 1.0,
        **kwargs,
    ) -> Array:
        """Compute the DiscoPOP discovered-preference loss.

        Smoothly interpolates between the standard logistic DPO loss
        ``-logsigmoid(beta * h)`` and an exponential ``exp(-beta * h)``
        penalty. The mixing weight is itself a sigmoid of
        ``beta * h / discopop_tau``, so larger margins gradually shift
        the loss surface from logistic toward exponential.

        Args:
            discopop_tau: Temperature controlling how sharply the
                exponential branch turns on as the margin grows.

        Returns:
            ``[batch]`` per-example loss tensor.
        """
        logits, _, _ = _base_dpo_loss(
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta,
            label_smoothing,
        )
        logits = logits * beta
        log_ratio_modulation = jax.nn.sigmoid(logits / discopop_tau)
        logistic_component = -jax.nn.log_sigmoid(logits)
        exp_component = jnp.exp(-logits)
        return logistic_component * (1 - log_ratio_modulation) + exp_component * log_ratio_modulation

    def _hinge_dpo_loss(
        chosen_logps: Array,
        rejected_logps: Array,
        ref_chosen_logps: Array,
        ref_rejected_logps: Array,
        beta: float,
        label_smoothing: float,
    ) -> Array:
        """Compute the hinge variant of DPO (max-margin alignment).

        Drops the logistic shaping in favour of ``relu(1 - beta * h)``,
        which produces a hard zero penalty once the policy has a
        sufficient margin (``beta * h > 1``) over the reference. Useful
        when the gradient saturation of the sigmoid form is undesirable.
        Ignores ``label_smoothing``.

        Returns:
            ``[batch]`` per-example loss tensor.
        """
        logits = (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
        return relu(1 - beta * logits)

    def _ipo_dpo_loss(
        chosen_logps: Array,
        rejected_logps: Array,
        ref_chosen_logps: Array,
        ref_rejected_logps: Array,
        beta: float,
        label_smoothing: float,
    ) -> Array:
        """Compute the Identity Preference Optimization (IPO) loss.

        From Azar et al. 2024, IPO replaces DPO's logistic shaping with
        the squared error ``(h - 1/(2*beta))**2``, where ``h`` is the
        log-ratio differential. This avoids the early-saturation
        pathology of sigmoid-DPO and keeps gradients well-conditioned
        for arbitrarily separable preferences. The caller is expected
        to pass *length-normalized* logps (handled by the surrounding
        forward when ``loss_type=="ipo"``). Ignores ``label_smoothing``.

        Returns:
            ``[batch]`` per-example squared-error loss.
        """
        logits = (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
        return (logits - 1 / (2 * beta)) ** 2

    def _kto_pair_dpo_loss(
        chosen_logps: Array,
        rejected_logps: Array,
        ref_chosen_logps: Array,
        ref_rejected_logps: Array,
        beta: float,
        label_smoothing: float,
    ) -> Array:
        """Compute the paired-data KTO surrogate (Kahneman-Tversky-style).

        Falls back to the same logistic form as the canonical sigmoid
        DPO with cDPO smoothing, but is exposed as ``"kto"`` for
        configurations that want a paired analogue of unpaired KTO.

        Returns:
            ``[batch]`` per-example loss tensor.
        """
        logits = (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
        return -logsigmoid(beta * logits) * (1 - label_smoothing) - logsigmoid(-beta * logits) * label_smoothing

    def _robust_dpo_loss(
        chosen_logps: Array,
        rejected_logps: Array,
        ref_chosen_logps: Array,
        ref_rejected_logps: Array,
        beta: float,
        label_smoothing: float,
    ) -> Array:
        """Compute the noise-robust DPO loss (Chowdhury et al. 2024).

        Like cDPO this assumes the labels are flipped with probability
        ``label_smoothing``, but rescales the smoothed sigmoid loss by
        ``1 / (1 - 2 * label_smoothing)`` to recover an unbiased
        estimator of the clean-preference DPO objective. Requires
        ``label_smoothing < 0.5``.

        Returns:
            ``[batch]`` per-example loss tensor.
        """
        logits = (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
        return (-logsigmoid(beta * logits) * (1 - label_smoothing) + logsigmoid(-beta * logits) * label_smoothing) / (
            1 - 2 * label_smoothing
        )

    def _exo_pair_dpo_loss(
        chosen_logps: Array,
        rejected_logps: Array,
        ref_chosen_logps: Array,
        ref_rejected_logps: Array,
        beta: float,
        label_smoothing: float,
    ) -> Array:
        """Compute the EXO (Efficient eXact Optimization) pair loss.

        Implements the cross-entropy between the policy-induced
        Bradley-Terry distribution and a smoothed target, yielding
        ``sigmoid(z) * (logsigmoid(z) - log(1 - eps)) +
        sigmoid(-z) * (logsigmoid(-z) - log(eps))`` with
        ``z = beta * h`` and ``eps`` clipped at 1e-3 for numerical
        safety. Recovers the clean-preference KL when ``eps -> 0``.

        Returns:
            ``[batch]`` per-example loss tensor.
        """
        import math

        logits = (chosen_logps - rejected_logps) - (ref_chosen_logps - ref_rejected_logps)
        label_smoothing = jnp.maximum(label_smoothing, 1e-3)
        return sigmoid(beta * logits) * (logsigmoid(beta * logits) - math.log(1 - label_smoothing)) + sigmoid(
            -beta * logits
        ) * (logsigmoid(-beta * logits) - math.log(label_smoothing))

    def _bco_pair_dpo_loss(
        chosen_logps: Array,
        rejected_logps: Array,
        ref_chosen_logps: Array,
        ref_rejected_logps: Array,
        beta: float,
        label_smoothing: float,
    ) -> Array:
        """Compute the Binary Classifier Optimization (BCO) pair loss.

        Trains an implicit reward classifier whose decision boundary is
        the in-batch mean reward ``delta``. Each chosen example is
        pushed above ``delta`` and each rejected example below it via
        ``-logsigmoid(beta * r_w - delta) - logsigmoid(-(beta * r_l - delta))``
        where ``r_* = logp_* - logp_ref_*``. ``delta`` is recomputed
        per-batch from the concatenation of chosen/rejected rewards.

        Returns:
            ``[batch]`` per-example loss tensor.
        """
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        chosen_rewards = beta * chosen_logratios
        rejected_rewards = beta * rejected_logratios
        delta = jnp.mean(jnp.concatenate([chosen_rewards, rejected_rewards]))
        return -logsigmoid((beta * chosen_logratios) - delta) - logsigmoid(-(beta * rejected_logratios - delta))

    def _sppo_hard_dpo_loss(
        chosen_logps: Array,
        rejected_logps: Array,
        ref_chosen_logps: Array,
        ref_rejected_logps: Array,
        beta: float,
        label_smoothing: float,
    ) -> Array:
        """Compute the hard-target SPPO loss (Wu et al. 2024).

        Targets per-side rewards directly: drives ``beta * r_w`` toward
        ``+0.5`` and ``beta * r_l`` toward ``-0.5`` with squared
        penalties ``(r_w - 0.5/beta)**2 + (r_l + 0.5/beta)**2``. Unlike
        sigmoid DPO this trains both halves to symmetric targets rather
        than only the margin.

        Returns:
            ``[batch]`` per-example loss tensor.
        """
        a = chosen_logps - ref_chosen_logps
        b = rejected_logps - ref_rejected_logps
        return (a - 0.5 / beta) ** 2 + (b + 0.5 / beta) ** 2

    def _aot_pair_dpo_loss(
        chosen_logps: Array,
        rejected_logps: Array,
        ref_chosen_logps: Array,
        ref_rejected_logps: Array,
        beta: float,
        label_smoothing: float,
    ) -> Array:
        """Compute the paired AOT (Alignment via Optimal Transport) loss.

        Variant of :func:`_aot_dpo_loss` that sorts the *per-side*
        rewards independently before pairing: chosen rewards sorted
        ascending against rejected rewards sorted ascending, then the
        smoothed sigmoid DPO objective is applied to the rank-aligned
        differences. Reframes preference matching as 1-D OT between the
        two reward distributions.

        Returns:
            ``[batch]`` per-example loss tensor.
        """
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        chosen_logratios_sorted = jnp.sort(chosen_logratios, axis=0)
        rejected_logratios_sorted = jnp.sort(rejected_logratios, axis=0)
        delta = chosen_logratios_sorted - rejected_logratios_sorted
        return -logsigmoid(beta * delta) * (1 - label_smoothing) - logsigmoid(-beta * delta) * label_smoothing

    def _aot_dpo_loss(
        chosen_logps: Array,
        rejected_logps: Array,
        ref_chosen_logps: Array,
        ref_rejected_logps: Array,
        beta: float,
        label_smoothing: float,
    ) -> Array:
        """Compute the unpaired AOT (Alignment via Optimal Transport) loss.

        Distinct from :func:`_aot_pair_dpo_loss`: here the policy *log
        ratios* and the reference *log ratios* are each sorted across
        the batch, and the smoothed sigmoid DPO objective is applied to
        their rank-aligned differences. Used when only the marginal
        distributions of margins (not paired chosen/rejected) need to
        align.

        Returns:
            ``[batch]`` per-example loss tensor.
        """
        logratios = chosen_logps - rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logratios_sorted = jnp.sort(logratios, axis=0)
        ref_logratios_sorted = jnp.sort(ref_logratios, axis=0)
        delta = logratios_sorted - ref_logratios_sorted
        return -logsigmoid(beta * delta) * (1 - label_smoothing) - logsigmoid(-beta * delta) * label_smoothing

    def _apo_zero_dpo_loss(
        chosen_logps: Array,
        rejected_logps: Array,
        ref_chosen_logps: Array,
        ref_rejected_logps: Array,
        beta: float,
        label_smoothing: float,
    ) -> Array:
        """Compute the APO-zero loss (D'Oosterlinck et al. 2024).

        Pushes the policy *up* on chosen and *down* on rejected
        independently of the reference margin:
        ``(1 - sigmoid(beta * r_w)) + sigmoid(beta * r_l)``.
        Recommended when the model is *worse* than the reference and
        you want to drag the chosen reward up before relying on the
        margin signal.

        Returns:
            ``[batch]`` per-example loss tensor.
        """
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        losses_chosen = 1 - sigmoid(beta * chosen_logratios)
        losses_rejected = sigmoid(beta * rejected_logratios)
        return losses_chosen + losses_rejected

    def _apo_down_dpo_loss(
        chosen_logps: Array,
        rejected_logps: Array,
        ref_chosen_logps: Array,
        ref_rejected_logps: Array,
        beta: float,
        label_smoothing: float,
    ) -> Array:
        """Compute the APO-down loss (D'Oosterlinck et al. 2024).

        Pulls the chosen reward *down* slightly while still penalising
        the rejected side using the *margin*:
        ``sigmoid(beta * r_w) + (1 - sigmoid(beta * (r_w - r_l)))``.
        Recommended when the model already exceeds the reference on
        chosen completions and you want to keep it close to the
        reference rather than diverge further.

        Returns:
            ``[batch]`` per-example loss tensor.
        """
        chosen_logratios = chosen_logps - ref_chosen_logps
        rejected_logratios = rejected_logps - ref_rejected_logps
        losses_chosen = sigmoid(beta * chosen_logratios)
        losses_rejected = 1 - sigmoid(beta * (chosen_logratios - rejected_logratios))
        return losses_chosen + losses_rejected

    # Map loss_type strings to corresponding loss function implementations.
    loss_function = {
        "ipo": _ipo_dpo_loss,
        "kto": _kto_pair_dpo_loss,
        "hinge": _hinge_dpo_loss,
        "sigmoid": _sigmoid_dpo_loss,
        "robust": _robust_dpo_loss,
        "exo_pair": _exo_pair_dpo_loss,
        "bco_pair": _bco_pair_dpo_loss,
        "sppo_hard": _sppo_hard_dpo_loss,
        "nca_pair": _nca_pair_dpo_loss,
        "aot_pair": _aot_pair_dpo_loss,
        "aot": _aot_dpo_loss,
        "apo_zero": _apo_zero_dpo_loss,
        "apo_down": _apo_down_dpo_loss,
        "discopop": _discopop_dpo_loss,
    }.get(loss_type, None)
    if loss_function is None:
        raise ValueError(f"given loss_type({loss_type}) is not valid")
    return loss_function


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
) -> dict[str, Array]:
    """
    Runs the model on concatenated chosen/rejected inputs for efficiency.

    This function first concatenates inputs (using the `concatenated_inputs` function) and then runs
    a forward pass through the model. It handles both encoder-decoder and decoder-only architectures,
    applies truncation if required, and computes per-token log probabilities.

    Args:
        model (EasyDeLBaseModule): The model to run.
        batch (tp.Dict[str, tp.Union[tp.List, Array]]): The input batch of data.
        is_encoder_decoder (bool): Flag indicating whether the model is an encoder-decoder.
        label_pad_token_id (int): Token id used to mark padded tokens in the labels.
        padding_value (int): Padding value for inputs.
        max_length (int | None, optional): Maximum sequence length for truncation. Defaults to None.
        truncation_mode (str, optional): Truncation strategy ("keep_end" or "keep_start"). Defaults to "keep_end".
        aux_loss_enabled (bool, optional): If True, enables auxiliary loss computation. Defaults to False.
        loss_type (str, optional): The type of loss function to be used. Defaults to "sigmoid".

    Returns:
        tp.Dict[str, Array]: A dictionary containing:
            - "chosen_logps": Log probabilities for chosen examples.
            - "rejected_logps": Log probabilities for rejected examples.
            - "mean_chosen_logits": Mean logits over tokens for chosen examples.
            - "mean_rejected_logits": Mean logits over tokens for rejected examples.
            Optionally, if `aux_loss_enabled` is True and the model output contains "aux_loss",
            it is included in the output dictionary.
    """
    num_examples = batch["prompt_input_ids"].shape[0]
    concatenated_batch = concatenated_inputs(batch=batch, padding_value=padding_value)

    model_kwargs = gather_multimodal_kwargs(concatenated_batch, aux_loss_enabled=aux_loss_enabled)

    prompt_input_ids = concatenated_batch["prompt_input_ids"]
    prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
    completion_input_ids = concatenated_batch["completion_input_ids"]
    completion_attention_mask = concatenated_batch["completion_attention_mask"]

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
        loss_mask = completion_attention_mask.astype(bool)
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
                completion_attention_mask,
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
        lmhead_chunksize = _resolve_dpo_lmhead_chunksize(model)
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
        loss_mask = jnp.roll(loss_mask, shift=-1, axis=1).astype("bool")

    # Adjust logits shape if necessary.
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
        )
    else:
        gathered_logps = _compute_token_logps_chunked(
            logits,
            labels,
            chunk_size=logprob_vocab_chunk_size,
        )
        per_token_logps = jnp.roll(
            jnp.where(loss_mask, gathered_logps, 0.0),
            shift=1,
            axis=1,
        )
        all_logps = per_token_logps.sum(-1)

        # Special handling for "ipo" loss type.
        if loss_type == "ipo":
            all_logps = all_logps / loss_mask.sum(-1)
        output = {}
        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]

        chosen_token_logit_sums = logits[:num_examples].sum(axis=-1)
        rejected_token_logit_sums = logits[num_examples:].sum(axis=-1)
        chosen_denom = jnp.maximum(jnp.sum(loss_mask[:num_examples]), 1)
        rejected_denom = jnp.maximum(jnp.sum(loss_mask[num_examples:]), 1)
        mean_chosen_logits = jnp.where(loss_mask[:num_examples], chosen_token_logit_sums, 0.0).sum() / chosen_denom
        mean_rejected_logits = jnp.where(loss_mask[num_examples:], rejected_token_logit_sums, 0.0).sum() / rejected_denom
        output["mean_chosen_logits"] = mean_chosen_logits
        output["mean_rejected_logits"] = mean_rejected_logits

    if aux_loss_enabled and hasattr(outputs, "aux_loss"):
        output["aux_loss"] = outputs.aux_loss
    return output


def training_step(
    state: EasyDeLState,
    batch: dict,
    reference_state: EasyDeLState,
    learning_rate_fn: tp.Callable,
    concatenated_forward: tp.Callable,
    beta: float = 0.1,
    label_smoothing: float = 0,
    loss_type: LOSS_FN_VARIANTS = "sigmoid",
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
    _batch_size, minibatch_size, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_partition_spec=partition_spec,
    )

    batch = with_sharding_constraint(batch, partition_spec, mesh=state.model.mesh, ignore_mpmd=True)
    _loss_func = get_loss_function(
        loss_type=loss_type,
        beta=beta,
        label_smoothing=label_smoothing,
    )

    if not reference_free:
        # Pre-compute reference logps outside jax.value_and_grad to avoid
        # nn.remat trace-level conflicts when the reference model uses
        # gradient checkpointing inside the grad trace.
        ref_chosen_logps, ref_rejected_logps = _get_reference_logps_from_batch(batch)
        if ref_chosen_logps is None or ref_rejected_logps is None:
            rfm = reference_state.model
            rfm.eval()
            ref_out = jax.lax.stop_gradient(concatenated_forward(rfm, batch))
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
        if straight_through_emulator is not None:
            tree = straight_through_emulator(tree)
        module = state.merge(tree=tree)

        model_output = concatenated_forward(module, call_batch)

        chosen_logps = model_output["chosen_logps"]
        rejected_logps = model_output["rejected_logps"]
        if reference_free:
            ref_chosen_logps = jnp.zeros_like(chosen_logps)
            ref_rejected_logps = jnp.zeros_like(rejected_logps)
        else:
            ref_chosen_logps = jax.lax.stop_gradient(call_batch["ref_chosen_logps"])
            ref_rejected_logps = jax.lax.stop_gradient(call_batch["ref_rejected_logps"])
        losses = _loss_func(
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta,
            label_smoothing,
        )

        chosen_rewards = beta * jax.lax.stop_gradient(chosen_logps - ref_chosen_logps)
        rejected_rewards = beta * jax.lax.stop_gradient(rejected_logps - ref_rejected_logps)
        if "aux_loss" in model_output:
            losses += model_output["aux_loss"]

        metrics = LossMetrics(
            loss=losses.mean(),
            rejected_rewards=rejected_rewards,
            chosen_rewards=chosen_rewards,
        )
        return metrics.loss, metrics

    gradients, metrics = minibatch_call(
        state=state,
        batch=batch,
        minibatch_size=minibatch_size,
        grad_fn=jax.value_and_grad(calculate_loss, has_aux=True),
    )

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
    return scheduled_loss_cache_key(
        call,
        value_fields=("beta", "label_smoothing", "loss_type", "reference_free", "partition_spec"),
        object_fields=("concatenated_forward", "straight_through_emulator"),
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
    reference_free = bool(call.get("reference_free", False))
    partition_spec = call.get("partition_spec")
    loss_func = get_loss_function(
        loss_type=loss_type,
        beta=beta,
        label_smoothing=label_smoothing,
    )

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
        module = bind_scheduled_module(call, tree)
        batch = constrain_scheduled_batch(module, batch, partition_spec)
        model_output = concatenated_forward_fn(module, batch)

        chosen_logps = model_output["chosen_logps"]
        rejected_logps = model_output["rejected_logps"]
        if reference_free:
            ref_chosen_logps = jnp.zeros_like(chosen_logps)
            ref_rejected_logps = jnp.zeros_like(rejected_logps)
        else:
            ref_chosen_logps, ref_rejected_logps = _get_reference_logps_from_batch(batch)
            if ref_chosen_logps is None or ref_rejected_logps is None:
                raise RuntimeError("DPO scheduled MPMD loss requires precomputed reference log-probs in the batch.")
            ref_chosen_logps = jax.lax.stop_gradient(ref_chosen_logps)
            ref_rejected_logps = jax.lax.stop_gradient(ref_rejected_logps)

        losses = loss_func(
            chosen_logps,
            rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            beta,
            label_smoothing,
        )
        if "aux_loss" in model_output:
            losses += model_output["aux_loss"]
        return losses.mean()

    return scheduled_loss


register_scheduled_loss_adapter(
    training_step,
    ScheduledLossAdapter(
        name="dpo",
        make_loss=_make_dpo_scheduled_loss,
        make_cache_key=_dpo_scheduled_loss_cache_key,
        prepare_batch=_prepare_dpo_scheduled_batch,
    ),
)


def evaluation_step(
    state: EasyDeLState,
    batch: dict,
    reference_state: EasyDeLState | None,
    concatenated_forward: tp.Callable,
    beta: float = 0.1,
    label_smoothing: float = 0,
    loss_type: LOSS_FN_VARIANTS = "sigmoid",
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
    *_, partition_spec = make_assertions_and_get_sizes(
        batch=batch,
        gradient_accumulation_steps=1,
        batch_partition_spec=partition_spec,
    )

    batch = with_sharding_constraint(batch, partition_spec, mesh=state.model.mesh, ignore_mpmd=True)
    _loss_func = get_loss_function(
        loss_type=loss_type,
        beta=beta,
        label_smoothing=label_smoothing,
    )

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
        model_output = concatenated_forward(state.merge(tree), batch)
        chosen_logps = model_output["chosen_logps"]
        rejected_logps = model_output["rejected_logps"]

        if reference_free:
            ref_chosen_for_loss = jnp.zeros_like(chosen_logps)
            ref_rejected_for_loss = jnp.zeros_like(rejected_logps)
        else:
            ref_chosen_logps, ref_rejected_logps = _get_reference_logps_from_batch(batch)
            if ref_chosen_logps is None or ref_rejected_logps is None:
                ref_model = state.model if reference_state is None else reference_state.model
                ref_output = concatenated_forward(ref_model, batch)
                ref_chosen_logps = ref_output["chosen_logps"]
                ref_rejected_logps = ref_output["rejected_logps"]
            ref_chosen_for_loss = ref_chosen_logps
            ref_rejected_for_loss = ref_rejected_logps

        losses = _loss_func(
            chosen_logps,
            rejected_logps,
            ref_chosen_for_loss,
            ref_rejected_for_loss,
            beta,
            label_smoothing,
        )

        chosen_rewards = beta * (chosen_logps - ref_chosen_for_loss)
        rejected_rewards = beta * (rejected_logps - ref_rejected_for_loss)

        metrics = LossMetrics(
            loss=losses.mean(),
            rejected_rewards=rejected_rewards,
            chosen_rewards=chosen_rewards,
        )
        return metrics

    metrics = calculate_loss(state.graphstate)
    return metrics
