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
        if _has_prepare:
            chunk_hidden_states = model.prepare_lm_head_inputs(chunk_hidden_states)
        return _lm_head_fn(chunk_hidden_states)

    _project_chunk = jax.checkpoint(_project_chunk, prevent_cse=False)

    def _chunk_contributions(
        chunk_hidden_states: Array,
        chunk_labels: Array,
        chunk_loss_mask: Array,
    ) -> tuple[Array, Array, Array, Array, Array]:
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
    """
    Returns a loss function based on the specified loss type.

    This function maps a given loss type (e.g., "sigmoid", "hinge", "ipo", etc.)
    to a corresponding loss function implementation that computes the DPO (Direct Preference Optimization) loss.

    Args:
        loss_type (LOSS_FN_VARIANTS): The type of loss function to return.
        beta (float): A scaling factor applied to the loss computation.
        label_smoothing (tp.Union[float, int]): A value for label smoothing used in some loss functions.

    Returns:
        A callable loss function that accepts arguments:
            (chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps, beta, label_smoothing, **kwargs)
        and returns the computed loss.
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
        """
        Base computation for DPO loss.

        Computes the log ratios between chosen and rejected log probabilities, and similarly for reference values.

        Args:
            chosen_logps (Array): Log probabilities for chosen examples.
            rejected_logps (Array): Log probabilities for rejected examples.
            ref_chosen_logps (Array): Reference log probabilities for chosen examples.
            ref_rejected_logps (Array): Reference log probabilities for rejected examples.
            beta (float): Scaling factor.
            label_smoothing (float): Label smoothing factor.
            **kwargs: Additional arguments (ignored).

        Returns:
            A tuple of (logits, logratios, ref_logratios) where:
                logits = logratios - ref_logratios.
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
        """
        Computes the DPO loss using a sigmoid-based formulation.

        Args:
            chosen_logps, rejected_logps, ref_chosen_logps, ref_rejected_logps (Array):
                Log probabilities for chosen/rejected examples and their reference values.
            beta (float): Scaling factor.
            label_smoothing (float): Label smoothing factor.
            **kwargs: Additional arguments (ignored).

        Returns:
            The computed loss as a negative weighted log sigmoid.
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
        """
        Computes the DPO loss using an NCA pair formulation.

        Args:
            (Same as above.)

        Returns:
            The computed loss based on the NCA pair loss formulation.
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
        """
        Computes the DPO loss using the AOT (All Ordered Terms) loss formulation.

        This loss function sorts the log ratios and compares them with the sorted reference log ratios.

        Args:
            (Same as above.)

        Returns:
            The computed loss based on sorted differences.
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
        """
        Computes the DPO loss using a Discopo-based modulation.

        Args:
            discopop_tau (float): Temperature parameter for modulation.
            (Other arguments are as described above.)

        Returns:
            The computed loss with a logistic and exponential modulation.
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
        """
        Computes the hinge loss version of the DPO loss.

        Args:
            (Same as above.)

        Returns:
            The hinge loss computed as the ReLU of (1 - beta * logits).
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
        """
        Computes the IPO loss variant of the DPO loss.

        Args:
            (Same as above.)

        Returns:
            A squared loss computed from the logits with a bias term.
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
        """
        Computes the KTO pair loss variant.

        Args:
            (Same as above.)

        Returns:
            The loss computed using the log-sigmoid function.
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
        """
        Computes a robust variant of the DPO loss.

        Args:
            (Same as above.)

        Returns:
            The loss computed with an adjustment that involves dividing by (1 - 2 * label_smoothing).
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
        """
        Computes the exo-pair variant of the DPO loss.

        Args:
            (Same as above.)

        Returns:
            The computed loss combining sigmoid and log-sigmoid terms with label smoothing.
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
        """
        Computes the BCO pair variant of the DPO loss.

        Args:
            (Same as above.)

        Returns:
            The loss computed from the log-ratios of chosen and rejected rewards.
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
        """
        Computes the SPO PPO hard variant of the DPO loss.

        Args:
            (Same as above.)

        Returns:
            A squared loss combining the differences for chosen and rejected examples.
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
        """
        Computes the AOT pair variant of the DPO loss.

        Args:
            (Same as above.)

        Returns:
            The loss computed from the sorted differences between chosen and rejected log ratios.
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
        """
        Computes the AOT variant of the DPO loss.

        This is similar to _aot_pair_dpo_loss but may be used when the pair version is not required.

        Args:
            (Same as above.)

        Returns:
            The computed loss based on the differences of sorted log ratios.
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
        """
        Computes the APO zero variant of the DPO loss.

        Args:
            (Same as above.)

        Returns:
            The computed loss based on the sigmoid of the log ratios.
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
        """
        Computes the APO down variant of the DPO loss.

        Args:
            (Same as above.)

        Returns:
            The computed loss based on an alternative weighting of the chosen and rejected log ratios.
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
    """
    Performs a single training step.

    This function computes gradients via minibatch processing over the input batch,
    calculates the loss using a specified loss function, updates the model state,
    and returns the updated state along with loss metrics.

    Args:
        state (EasyDeLState): The current model state.
        batch (dict): Input batch data.
        reference_state (EasyDeLState): A reference model state used for computing reference log probabilities.
        learning_rate_fn (tp.Callable): Function to compute the learning rate.
        concatenated_forward (tp.Callable): Function to perform a forward pass on concatenated inputs.
        beta (float, optional): Scaling factor for loss computation. Defaults to 0.1.
        label_smoothing (float, optional): Label smoothing factor. Defaults to 0.
        loss_type (LOSS_FN_VARIANTS, optional): Type of loss function to use. Defaults to "sigmoid".
        ref_precalculated (bool, optional): If True, uses precalculated reference log probabilities from the batch.
            Defaults to True.
        loss_config (tp.Optional[LossConfig], optional): Additional configuration for loss. Defaults to None.
        partition_spec (tp.Optional[PartitionSpec], optional): Partitioning specification for sharding the batch.
            Defaults to None.
        gradient_accumulation_steps (int, optional): Number of steps for gradient accumulation. Defaults to 1.

    Returns:
        tp.Tuple[EasyDeLState, LossMetrics]: A tuple containing the updated model state and the loss metrics.
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
        """
        Inner function to compute loss and metrics for a given minibatch.

        Args:
            tree (spx.State): The current model graph state.
            call_batch (dict): A minibatch of data.

        Returns:
            A tuple (loss, metrics) where loss is a scalar and metrics is a LossMetrics instance.
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
    return scheduled_loss_cache_key(
        call,
        value_fields=("beta", "label_smoothing", "loss_type", "reference_free", "partition_spec"),
        object_fields=("concatenated_forward", "straight_through_emulator"),
    )


def _make_dpo_scheduled_loss(call):
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
    """
    Performs a single evaluation step.

    This function computes loss metrics for the input batch using the provided model state.
    It can optionally use a reference state to compute reference log probabilities.

    Args:
        state (EasyDeLState): The current model state.
        batch (dict): Input batch data.
        concatenated_forward (tp.Callable): Function to perform a forward pass on concatenated inputs.
        reference_state (EasyDeLState, optional): A reference model state. Defaults to None.
        beta (float, optional): Scaling factor for loss computation. Defaults to 0.1.
        label_smoothing (float, optional): Label smoothing factor. Defaults to 0.
        loss_type (LOSS_FN_VARIANTS, optional): Type of loss function to use. Defaults to "sigmoid".
        reference_free (bool, optional): If True, ignores reference log probabilities. Defaults to False.
        partition_spec (tp.Optional[PartitionSpec], optional): Partitioning specification for sharding the batch.
            Defaults to None.

    Returns:
        LossMetrics: The computed loss metrics.
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
        """
        Inner function to compute loss metrics for evaluation.

        Args:
            tree (spx.State): The current model graph state.

        Returns:
            LossMetrics: The computed loss metrics.
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
