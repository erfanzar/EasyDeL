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

"""Qwen3Next model implementation for EasyDeL.

This module implements the Qwen3Next hybrid attention architecture, which combines:
- Full attention layers with sigmoid gating and partial RoPE
- Linear attention layers using GatedDeltaNet
- MoE FFN with routed and shared experts

The hybrid attention approach allows for efficient long-context processing
with linear complexity in linear attention layers while maintaining
expressive power through full attention at regular intervals.
"""

import typing
from functools import partial

import jax
import jax.numpy as jnp
from eformer import common_types
from eformer.common_types import Replicated
from eformer.escale import (
    PartitionAxis,
    PartitionManager,
    apply_logical_sharding,
    get_corrected_named_sharding,
    with_sharding_constraint,
)
from ejkernel.types import MaskInfo  # pyright: ignore[reportMissingTypeStubs]
from flax import nnx as nn
from jax.ad_checkpoint import checkpoint_name
from jaxtyping import Array, Bool, Float, Int

from easydel.caching import (
    HybridCache,
    LinearCacheView,
    LinearMetadata,
    OperationsMetadata,
    RaggedPagesCache,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    TransformerCache,
    TransformerCacheView,
    TransformerMetadata,
)
from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.factory import TaskType, register_module
from easydel.infra.loss_utils import auxiliary_load_balancing_loss_func
from easydel.infra.modeling_outputs import (
    AttentionLayerOutput,
    DecoderLayerOutput,
    MoeCausalLMOutput,
    MoeModelOutput,
)
from easydel.infra.utils import ACT2FN, auto_remat
from easydel.layers import (
    BaseMoeModule,
    ColumnParallelLinear,
    ColumnParallelMoELinear,
    Embed,
    MoeLoadBalancingStrategy,
    MoeRoutingStrategy,
    RMSNormGated,
    RowParallelLinear,
    RowParallelMoELinear,
)
from easydel.layers.attention import UnifiedAttention
from easydel.layers.linear_attention import (
    apply_conv_with_state,
    apply_manual_depthwise_conv,
    apply_mask_to_padding_states,
    shift_conv_state_left,
)
from easydel.layers.norms import lowfloats
from easydel.modules._base import BaseCausalLMModule
from easydel.operations import OperationMetadata
from easydel.operations.kernels import GatedDeltaRuleOp, GatedDeltaRuleOutput

from .qwen3_next_configuration import Qwen3NextConfig


def l2norm_decode(
    x: Array,
    *,
    axis: int = -1,
    eps: float = 1e-6,
) -> Array:
    """Match ejkernel's decode-time L2 normalization."""
    inv_norm = jax.lax.rsqrt(jnp.sum(x * x, axis=axis, keepdims=True) + eps)
    return x * inv_norm


def _preserve_array_sharding(
    value: Array,
    *,
    partition_manager: PartitionManager | None,
    partition_axis: PartitionAxis | None,
) -> Array:
    """Apply recurrent-state sharding even when grouped decode bypasses the op kernel."""
    if partition_manager is None or partition_axis is None:
        return value

    spec = partition_manager.resolve(
        axes=[common_types.BATCH, common_types.HEAD, common_types.EMPTY, common_types.EMPTY],
        mode=common_types.MODE_PREFILL,
        shape=value.shape,
    )
    sharding = get_corrected_named_sharding(tuple(value.shape), spec, raise_mesh_error=False)
    return with_sharding_constraint(value, sharding)


def apply_grouped_single_step_gdr(
    query: Float[Array, "batch 1 num_k_heads head_dim"],
    key: Float[Array, "batch 1 num_k_heads head_dim"],
    value: Float[Array, "batch 1 num_v_heads value_dim"],
    beta: Float[Array, "batch 1 num_v_heads"],
    decay: Float[Array, "batch 1 num_v_heads"] | None,
    recurrent_state: Float[Array, "batch num_v_heads head_dim value_dim"],
    *,
    gdr_op: GatedDeltaRuleOp | None = None,
    use_qk_l2norm: bool = True,
) -> tuple[
    Float[Array, "batch 1 num_v_heads value_dim"],
    Float[Array, "batch num_v_heads head_dim value_dim"],
]:
    """Decode-only GDR step for Qwen's repeated-q/k head layout.

    Qwen linear attention stores one recurrent state per value head, but the
    query/key heads are shared across groups of value heads. This helper keeps
    the recurrent-state layout unchanged while avoiding materializing repeated
    q/k heads during single-token decode.

    On TPU, dispatches to a fused Pallas kernel that performs the entire
    grouped state update in VMEM without materializing the 5D intermediate.
    """
    batch_size, _, num_k_heads, _head_dim = query.shape
    num_v_heads = value.shape[2]
    if num_v_heads % num_k_heads != 0:
        raise ValueError(f"num_v_heads ({num_v_heads}) must be divisible by num_k_heads ({num_k_heads})")
    expand_ratio = num_v_heads // num_k_heads

    if use_qk_l2norm:
        query = l2norm_decode(query, axis=-1, eps=1e-6)
        key = l2norm_decode(key, axis=-1, eps=1e-6)

    input_dtype = query.dtype
    # Squeeze seq_len=1 and reshape for the kernel interface
    q_2d = query[:, 0, :, :]  # [batch, num_k_heads, head_dim]
    k_2d = key[:, 0, :, :]
    v_grouped = value[:, 0, :, :].reshape(batch_size, num_k_heads, expand_ratio, -1)
    beta_grouped = beta[:, 0, :].reshape(batch_size, num_k_heads, expand_ratio)

    decay_grouped = decay[:, 0, :].reshape(batch_size, num_k_heads, expand_ratio) if decay is not None else None

    if gdr_op is not None:
        output, new_state = gdr_op.grouped_gdr_decode(
            query=q_2d,
            key=k_2d,
            value=v_grouped,
            beta=beta_grouped,
            decay=decay_grouped,
            recurrent_state=recurrent_state,
        )
    else:
        output, new_state = GatedDeltaRuleOp.grouped_gdr_decode_jax(
            query=q_2d,
            key=k_2d,
            value=v_grouped,
            beta=beta_grouped,
            decay=decay_grouped,
            recurrent_state=recurrent_state,
        )

    return (
        output.reshape(batch_size, 1, num_v_heads, -1).astype(input_dtype),
        new_state,
    )


def _apply_qwen3_next_packed_slow_updates(
    *,
    conv_states: Float[Array, "num_slots conv_dim d_conv"],
    recurrent_states: Float[Array, "num_slots num_v_heads head_dim value_dim"],
    conv_input: Float[Array, "batch seq_len conv_dim"],
    beta: Float[Array, "batch seq_len num_v_heads"],
    decay: Float[Array, "batch seq_len num_v_heads"],
    kernel: Float[Array, "conv_dim d_conv"],
    query_start_loc,
    num_requests,
    key_dim: int,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    expand_ratio: int,
    conv_output_dtype: jnp.dtype,
    gdr_op: GatedDeltaRuleOp,
) -> tuple:
    """Reference packed decode path -- processes tokens one-by-one via fori_loop.

    Iterates over all active tokens in a packed batch, updating the
    per-slot convolution and recurrent states in-place and collecting the
    gated-delta-rule output for each token.

    Args:
        conv_states: Per-slot 1-D convolution states, shape
            ``(num_slots, conv_dim, d_conv)``.
        recurrent_states: Per-slot recurrent (GDR) states, shape
            ``(num_slots, num_v_heads, head_dim, value_dim)``.
        conv_input: Packed convolution input, shape
            ``(batch, seq_len, conv_dim)``.
        beta: GDR beta coefficients, shape
            ``(batch, seq_len, num_v_heads)``.
        decay: GDR decay coefficients, shape
            ``(batch, seq_len, num_v_heads)``.
        kernel: Depthwise convolution kernel, shape
            ``(conv_dim, d_conv)``.
        query_start_loc: Cumulative token offsets per request, length
            ``num_requests + 1``.
        num_requests: Number of active requests in the batch.
        key_dim: Total key projection dimension.
        num_k_heads: Number of key heads.
        head_k_dim: Per-head key dimension.
        num_v_heads: Number of value heads.
        head_v_dim: Per-head value dimension.
        expand_ratio: GDR group expansion ratio; values ``> 1`` use grouped
            single-step GDR.
        conv_output_dtype: Dtype for the depthwise convolution output.
        gdr_op: Gated delta rule operator instance.

    Returns:
        A ``(conv_states, recurrent_states, token_outputs)`` tuple with
        updated slot states and the per-token output tensor of shape
        ``(seq_len, num_v_heads, head_v_dim)``.
    """
    seq_len = conv_input.shape[1]
    max_req_idx = query_start_loc.shape[0] - 1
    total_tokens = query_start_loc[jnp.clip(num_requests, 0, max_req_idx)]

    token_positions = jnp.arange(seq_len, dtype=jnp.int32)
    token_slots = jnp.searchsorted(query_start_loc[1:], token_positions, side="right")
    token_slots = jnp.clip(token_slots, 0, conv_states.shape[0] - 1)
    token_active = token_positions < total_tokens

    token_outputs = jnp.zeros((seq_len, num_v_heads, head_v_dim), dtype=jnp.float32)

    def _body(idx: int, carry):
        """Process a single token position inside the ``fori_loop``.

        Looks up the slot for token *idx*, conditionally runs convolution
        and GDR state updates when the token is active, and writes the
        output back into the carry tensors.

        Args:
            idx: Flat token index within the packed batch.
            carry: A ``(conv_states, recurrent_states, token_outputs)``
                tuple of mutable state arrays.

        Returns:
            Updated ``(conv_states, recurrent_states, token_outputs)`` carry.
        """
        conv_states_c, recurrent_states_c, token_outputs_c = carry
        slot = token_slots[idx]
        is_active = token_active[idx]

        def _update_states(inner_carry):
            """Run conv + GDR update for one active token and write outputs.

            Args:
                inner_carry: Same shape as the outer carry tuple.

            Returns:
                Updated carry with the new conv/recurrent states and output
                for this token written in-place.
            """
            conv_states_i, recurrent_states_i, token_outputs_i = inner_carry

            conv_state_i = jax.lax.dynamic_slice_in_dim(conv_states_i, slot, 1, axis=0)
            recurrent_state_i = jax.lax.dynamic_slice_in_dim(recurrent_states_i, slot, 1, axis=0)
            conv_token = jax.lax.dynamic_slice_in_dim(conv_input, idx, 1, axis=1)[:, 0, :]

            conv_state_i = shift_conv_state_left(conv_state_i, conv_token.astype(conv_state_i.dtype))

            conv_output_i = apply_manual_depthwise_conv(
                conv_state_i,
                kernel,
                output_dtype=conv_output_dtype,
            )

            query_i = conv_output_i[:, :key_dim].reshape(1, 1, num_k_heads, head_k_dim)
            key_i = conv_output_i[:, key_dim : key_dim * 2].reshape(1, 1, num_k_heads, head_k_dim)
            value_i = conv_output_i[:, key_dim * 2 :].reshape(1, 1, num_v_heads, head_v_dim)

            beta_i = beta[0, idx].reshape(1, 1, num_v_heads)
            decay_i = decay[0, idx].reshape(1, 1, num_v_heads)

            if expand_ratio > 1:
                out_i, new_rec_i = apply_grouped_single_step_gdr(
                    query=query_i,
                    key=key_i,
                    value=value_i,
                    beta=beta_i,
                    decay=decay_i,
                    recurrent_state=recurrent_state_i,
                    gdr_op=gdr_op,
                )
            else:
                gdr_out_i: GatedDeltaRuleOutput = gdr_op(
                    query=query_i,
                    key=key_i,
                    value=value_i,
                    beta=beta_i,
                    decay=decay_i,
                    recurrent_state=recurrent_state_i,
                )
                out_i = gdr_out_i.attention_outputs
                new_rec_i = gdr_out_i.recurrent_state

            conv_states_i = jax.lax.dynamic_update_slice_in_dim(conv_states_i, conv_state_i, slot, axis=0)
            recurrent_states_i = jax.lax.dynamic_update_slice_in_dim(
                recurrent_states_i,
                new_rec_i.astype(recurrent_states_i.dtype),
                slot,
                axis=0,
            )

            attn_token = out_i[0, 0, :, :].astype(token_outputs_i.dtype)
            token_outputs_i = token_outputs_i.at[idx].set(attn_token)
            return conv_states_i, recurrent_states_i, token_outputs_i

        return jax.lax.cond(
            is_active,
            _update_states,
            lambda x: x,
            (conv_states_c, recurrent_states_c, token_outputs_c),
        )

    return jax.lax.fori_loop(
        0,
        seq_len,
        _body,
        (conv_states, recurrent_states, token_outputs),
    )


QWEN3_NEXT_PACKED_PREFILL_BATCH_CAP = 8


def _apply_qwen3_next_depthwise_conv_sequence(
    inputs: Float[Array, "batch seq_len conv_dim"],
    kernel: Float[Array, "conv_dim d_conv"],
    *,
    output_dtype: jnp.dtype,
) -> Float[Array, "batch seq_len conv_dim"]:
    """Run a causal depthwise 1-D convolution over a dense sequence batch.

    Applies a per-channel (depthwise) causal convolution using
    ``jax.lax.conv_general_dilated`` with left-padding of ``d_conv - 1`` so
    that each output position depends only on current and prior input
    positions. The convolution is followed by SiLU activation.

    This is used during the prefill phase of packed batching, where each
    request's tokens are laid out contiguously and can be processed as a
    standard sequence convolution rather than the incremental shift-and-convolve
    used in single-token decode.

    Args:
        inputs: Dense input sequences, shape ``[batch, seq_len, conv_dim]``.
        kernel: Depthwise conv kernel, shape ``[conv_dim, d_conv]``. The kernel
            is transposed internally to match the ``("NWC", "WIO", "NWC")``
            layout expected by ``conv_general_dilated``.
        output_dtype: Desired dtype for the output tensor.

    Returns:
        Convolution output with SiLU activation applied,
        shape ``[batch, seq_len, conv_dim]``.
    """
    conv_kernel = kernel.T[:, None, :]
    conv_output = jax.lax.conv_general_dilated(
        inputs.astype(jnp.float32),
        conv_kernel.astype(jnp.float32),
        window_strides=(1,),
        padding=((kernel.shape[1] - 1, 0),),
        dimension_numbers=("NWC", "WIO", "NWC"),
        feature_group_count=inputs.shape[-1],
    )
    return jax.nn.silu(conv_output).astype(output_dtype)


def _scatter_qwen3_next_selected_updates(
    base: Array,
    slots: Int[Array, "selected_slots"],
    valid: Bool[Array, "selected_slots"],
    updates: Array,
) -> Array:
    """Scatter selected row updates into a base array via sequential ``dynamic_update_slice``.

    Given a set of ``updates`` indexed by ``slots``, this function writes each
    valid update row into the corresponding slot position of ``base``. Invalid
    entries (where ``valid[i]`` is False) leave the base unchanged at that slot.

    Implementation uses ``jax.lax.fori_loop`` with per-slot
    ``dynamic_update_slice_in_dim`` calls, which translates to per-slot DMA
    writes on TPU. This is more efficient than the alternative dense approach
    of building a one-hot index matrix and performing a full tensordot scatter,
    which would incur O(chunk_size * num_slots * D) HBM traffic versus the
    O(chunk_size * D) traffic of the sequential approach.

    Args:
        base: The target array to scatter into, shape ``[num_slots, ...]``.
        slots: Integer indices indicating which row of ``base`` each update
            should be written to, shape ``[selected_slots]``.
        valid: Boolean mask indicating which updates are active,
            shape ``[selected_slots]``. When ``valid[i]`` is False, the
            existing row in ``base`` at ``slots[i]`` is preserved.
        updates: The update values to scatter, shape ``[selected_slots, ...]``.
            Must have the same trailing dimensions as ``base``.

    Returns:
        A new array with the same shape and dtype as ``base``, with valid
        updates written at the specified slot positions.
    """
    num_updates = slots.shape[0]

    def _scatter_one(i, acc):
        slot = slots[i]
        is_valid = valid[i]
        update_row = jax.lax.dynamic_index_in_dim(updates, i, axis=0, keepdims=False)
        existing_row = jax.lax.dynamic_index_in_dim(acc, slot, axis=0, keepdims=False)
        row = jnp.where(is_valid, update_row.astype(acc.dtype), existing_row)
        return jax.lax.dynamic_update_slice_in_dim(acc, row[None], slot, axis=0)

    return jax.lax.fori_loop(0, num_updates, _scatter_one, base)


def _finalize_qwen3_next_conv_state_from_combined(
    combined_inputs: Float[Array, "batch total_seq_len conv_dim"],
    total_lengths: Int[Array, "batch"],
    *,
    d_conv: int,
    output_dtype: jnp.dtype,
) -> Float[Array, "batch conv_dim d_conv"]:
    """Extract the trailing conv-state window from combined (prefix + new tokens) sequences.

    After running a full causal depthwise convolution over the concatenation of
    the old conv state prefix and new input tokens, the conv state for
    subsequent decode steps is the last ``d_conv`` positions of that combined
    sequence. This function extracts that trailing window for each slot using
    ``jax.lax.dynamic_slice``, vectorized over the batch axis via ``jax.vmap``.

    The dynamic_slice approach produces contiguous DMA reads per slot on TPU,
    which is more efficient than a ``take_along_axis`` gather that would
    require non-contiguous index lookups.

    The output is transposed from ``[d_conv, conv_dim]`` to
    ``[conv_dim, d_conv]`` to match the conv-state layout convention used
    throughout the codebase.

    Args:
        combined_inputs: Concatenated old-state prefix and new input tokens,
            shape ``[batch, total_seq_len, conv_dim]``. ``total_seq_len``
            equals ``d_conv + request_seq_len``.
        total_lengths: The effective length of each slot's combined sequence,
            shape ``[batch]``. Used to compute the start index of the trailing
            window as ``clip(total_lengths - d_conv, 0, ...)``.
        d_conv: The convolution kernel width (number of positions to retain).
        output_dtype: Desired dtype for the output conv state.

    Returns:
        Extracted conv states, shape ``[batch, conv_dim, d_conv]``.
    """
    conv_dim = combined_inputs.shape[2]
    max_seq = combined_inputs.shape[1]

    def _extract_one(row, length):
        start = jnp.clip(length - d_conv, 0, max_seq - d_conv)
        window = jax.lax.dynamic_slice(row, (start, 0), (d_conv, conv_dim))
        return window.T.astype(output_dtype)  # [conv_dim, d_conv]

    return jax.vmap(_extract_one)(combined_inputs, total_lengths)


def _apply_qwen3_next_packed_updates_unified(
    *,
    conv_states: Float[Array, "num_slots conv_dim d_conv"],
    recurrent_states: Float[Array, "num_slots num_v_heads head_dim value_dim"],
    conv_input: Float[Array, "batch seq_len conv_dim"],
    beta: Float[Array, "batch seq_len num_v_heads"],
    decay: Float[Array, "batch seq_len num_v_heads"],
    kernel: Float[Array, "conv_dim d_conv"],
    query_start_loc: Int[Array, "num_slots_plus_1"],
    num_requests: Int[Array, ""],
    key_dim: int,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    expand_ratio: int,
    conv_output_dtype: jnp.dtype,
    gdr_op: GatedDeltaRuleOp,
) -> tuple[
    Float[Array, "num_slots conv_dim d_conv"],
    Float[Array, "num_slots num_v_heads head_dim value_dim"],
    Float[Array, "seq_len num_v_heads head_v_dim"],
]:
    """Unified packed update implementation with a single-token fast lane and scan-based prefill.

    This is the production implementation for packed-batch inference in
    Qwen3Next's linear-attention layers. It handles mixed decode (single-token)
    and prefill (multi-token) requests within a single packed batch, optimizing
    each case separately.

    **Single-token fast lane**: Slots with exactly one scheduled token are
    detected via ``single_slot_mask`` and processed in a single batched
    operation (no loop). The conv state is updated using
    ``GatedDeltaRuleOp.fused_conv_decode``, and the GDR recurrent update is
    applied in parallel across all single-token slots. This path is guarded
    by ``jax.lax.cond`` so it is skipped entirely when there are no
    single-token slots.

    **Scan-based prefill**: Multi-token slots are grouped into chunks of
    ``QWEN3_NEXT_PACKED_PREFILL_BATCH_CAP`` and processed via
    ``jax.lax.scan`` (rather than ``fori_loop``), which enables XLA to
    overlap computation and memory transfers more effectively. Each chunk:
    - Gathers per-slot tokens and prepends the conv-state prefix.
    - Runs a full causal depthwise convolution over the combined sequence.
    - Applies the chunked GDR recurrent update.
    - Collects updated conv and recurrent states as scan outputs.

    After the scan, all prefill state updates are scattered back to the
    slot arrays in a single pass using ``_scatter_qwen3_next_selected_updates``.

    Unlike the legacy implementation, this function does NOT carry the full
    state arrays through the scan body, reducing loop-carry overhead and
    improving TPU performance.

    Args:
        conv_states: Per-slot conv state, shape ``[num_slots, conv_dim, d_conv]``.
        recurrent_states: Per-slot recurrent state,
            shape ``[num_slots, num_v_heads, head_dim, value_dim]``.
        conv_input: Packed input, shape ``[batch, seq_len, conv_dim]``.
        beta: Gating coefficients, shape ``[batch, seq_len, num_v_heads]``.
        decay: Decay factors, shape ``[batch, seq_len, num_v_heads]``.
        kernel: Depthwise conv kernel, shape ``[conv_dim, d_conv]``.
        query_start_loc: Cumulative token offsets, length ``num_slots + 1``.
        num_requests: Scalar number of active requests.
        key_dim: Total key dimension.
        num_k_heads: Number of key/query heads.
        head_k_dim: Per-head key dimension.
        num_v_heads: Number of value heads.
        head_v_dim: Per-head value dimension.
        expand_ratio: Value-head expansion ratio.
        conv_output_dtype: Dtype for conv outputs.
        gdr_op: GatedDeltaRuleOp instance.

    Returns:
        A tuple of (updated conv_states, updated recurrent_states,
        token_outputs of shape ``[seq_len, num_v_heads, head_v_dim]``).
    """
    seq_len = conv_input.shape[1]
    d_conv = kernel.shape[1]
    num_slots = min(conv_states.shape[0], query_start_loc.shape[0] - 1)
    prefill_chunk_size = QWEN3_NEXT_PACKED_PREFILL_BATCH_CAP
    num_prefill_chunks = max((num_slots + prefill_chunk_size - 1) // prefill_chunk_size, 1)
    slot_ids = jnp.arange(num_slots, dtype=jnp.int32)
    starts = query_start_loc[:num_slots]
    scheduled_tokens = query_start_loc[1 : num_slots + 1] - query_start_loc[:num_slots]
    active_slots = (slot_ids < num_requests) & (scheduled_tokens > 0)
    single_slot_mask = active_slots & (scheduled_tokens == 1)
    multi_slot_mask = active_slots & (scheduled_tokens > 1)
    token_outputs = jnp.zeros((seq_len, num_v_heads, head_v_dim), dtype=jnp.float32)

    if num_slots == 0:
        return conv_states, recurrent_states, token_outputs

    prefix_conv_states = conv_states[:num_slots]
    prefix_recurrent_states = recurrent_states[:num_slots]
    safe_single_indices = jnp.clip(starts, 0, seq_len - 1)
    single_tokens = conv_input[0, safe_single_indices, :]

    def _apply_single_token_fast_lane(operand):
        token_outputs_i, conv_states_i, recurrent_states_i = operand
        shifted_conv_states, conv_output = GatedDeltaRuleOp.fused_conv_decode(
            conv_state=conv_states_i,
            new_tokens=single_tokens,
            kernel=kernel,
            output_dtype=conv_output_dtype,
        )
        query_single = conv_output[:, :key_dim].reshape(num_slots, 1, num_k_heads, head_k_dim)
        key_single = conv_output[:, key_dim : key_dim * 2].reshape(num_slots, 1, num_k_heads, head_k_dim)
        value_single = conv_output[:, key_dim * 2 :].reshape(num_slots, 1, num_v_heads, head_v_dim)
        beta_single = beta[0, safe_single_indices].reshape(num_slots, 1, num_v_heads)
        decay_single = decay[0, safe_single_indices].reshape(num_slots, 1, num_v_heads)

        single_query_mask = single_slot_mask[:, None, None, None]
        single_token_mask = single_slot_mask[:, None, None]
        query_single = jnp.where(single_query_mask, query_single, 0)
        key_single = jnp.where(single_query_mask, key_single, 0)
        value_single = jnp.where(single_query_mask, value_single, 0)
        beta_single = jnp.where(single_token_mask, beta_single, 0)
        decay_single = jnp.where(single_token_mask, decay_single, 0)

        if expand_ratio > 1:
            single_output, single_recurrent = apply_grouped_single_step_gdr(
                query=query_single,
                key=key_single,
                value=value_single,
                beta=beta_single,
                decay=decay_single,
                recurrent_state=recurrent_states_i,
                gdr_op=gdr_op,
            )
        else:
            gdr_single: GatedDeltaRuleOutput = gdr_op(
                query=query_single,
                key=key_single,
                value=value_single,
                beta=beta_single,
                decay=decay_single,
                recurrent_state=recurrent_states_i,
            )
            single_output = gdr_single.attention_outputs
            single_recurrent = gdr_single.recurrent_state

        conv_states_i = jnp.where(single_slot_mask[:, None, None], shifted_conv_states, conv_states_i)
        recurrent_states_i = jnp.where(
            single_slot_mask[:, None, None, None],
            single_recurrent.astype(recurrent_states_i.dtype),
            recurrent_states_i,
        )
        token_outputs_i = token_outputs_i.at[safe_single_indices].add(
            jnp.where(single_token_mask, single_output[:, 0].astype(token_outputs_i.dtype), 0)
        )
        return token_outputs_i, conv_states_i, recurrent_states_i

    token_outputs, base_conv_states, base_recurrent_states = jax.lax.cond(
        jnp.any(single_slot_mask),
        _apply_single_token_fast_lane,
        lambda operand: operand,
        (token_outputs, prefix_conv_states, prefix_recurrent_states),
    )

    packed_slots = jnp.where(
        multi_slot_mask,
        size=num_prefill_chunks * prefill_chunk_size,
        fill_value=0,
    )[0].reshape(num_prefill_chunks, prefill_chunk_size)
    packed_valid = (
        jnp.arange(num_prefill_chunks * prefill_chunk_size, dtype=jnp.int32) < jnp.sum(multi_slot_mask.astype(jnp.int32))
    ).reshape(num_prefill_chunks, prefill_chunk_size)
    prefill_offsets = jnp.arange(seq_len, dtype=jnp.int32)[None, :]

    empty_conv_updates = jnp.zeros(
        (prefill_chunk_size, prefix_conv_states.shape[1], d_conv),
        dtype=prefix_conv_states.dtype,
    )
    empty_recurrent_updates = jnp.zeros(
        (prefill_chunk_size, *prefix_recurrent_states.shape[1:]),
        dtype=prefix_recurrent_states.dtype,
    )

    def _prefill_chunk_step(
        token_outputs_i: Float[Array, "seq_len num_v_heads head_v_dim"],
        scan_inputs: tuple[Int[Array, "chunk"], Bool[Array, "chunk"]],
    ) -> tuple[
        Float[Array, "seq_len num_v_heads head_v_dim"],
        tuple[
            Float[Array, "chunk conv_dim d_conv"],
            Float[Array, "chunk num_v_heads head_dim value_dim"],
            Int[Array, "chunk"],
            Bool[Array, "chunk"],
        ],
    ]:
        chunk_slots, chunk_valid = scan_inputs

        def _apply_chunk(operand):
            token_outputs_j, chunk_slots_j, chunk_valid_j = operand
            chunk_starts = starts[chunk_slots_j]
            chunk_lengths = scheduled_tokens[chunk_slots_j]
            chunk_token_mask = chunk_valid_j[:, None] & (prefill_offsets < chunk_lengths[:, None])

            chunk_token_indices = chunk_starts[:, None] + prefill_offsets
            safe_chunk_indices = jnp.clip(chunk_token_indices, 0, seq_len - 1)
            chunk_inputs = conv_input[0, safe_chunk_indices, :]
            chunk_inputs = jnp.where(chunk_token_mask[:, :, None], chunk_inputs, 0)

            chunk_prefix = base_conv_states[chunk_slots_j].transpose(0, 2, 1)
            combined_inputs = jnp.concatenate([chunk_prefix, chunk_inputs], axis=1)
            conv_output = _apply_qwen3_next_depthwise_conv_sequence(
                combined_inputs,
                kernel,
                output_dtype=conv_output_dtype,
            )[:, d_conv:, :]

            query_prefill = conv_output[:, :, :key_dim].reshape(
                prefill_chunk_size,
                seq_len,
                num_k_heads,
                head_k_dim,
            )
            key_prefill = conv_output[:, :, key_dim : key_dim * 2].reshape(
                prefill_chunk_size,
                seq_len,
                num_k_heads,
                head_k_dim,
            )
            value_prefill = conv_output[:, :, key_dim * 2 :].reshape(
                prefill_chunk_size,
                seq_len,
                num_v_heads,
                head_v_dim,
            )

            beta_prefill = beta[0, safe_chunk_indices]
            decay_prefill = decay[0, safe_chunk_indices]

            prefill_query_mask = chunk_token_mask[:, :, None, None]
            prefill_beta_mask = chunk_token_mask[:, :, None]
            if expand_ratio > 1:
                query_prefill = jnp.repeat(query_prefill, expand_ratio, axis=2)
                key_prefill = jnp.repeat(key_prefill, expand_ratio, axis=2)
            query_prefill = jnp.where(prefill_query_mask, query_prefill, 0)
            key_prefill = jnp.where(prefill_query_mask, key_prefill, 0)
            value_prefill = jnp.where(prefill_query_mask, value_prefill, 0)
            beta_prefill = jnp.where(prefill_beta_mask, beta_prefill, 0)
            decay_prefill = jnp.where(prefill_beta_mask, decay_prefill, 0)

            chunk_recurrent_states = base_recurrent_states[chunk_slots_j]
            gdr_prefill: GatedDeltaRuleOutput = gdr_op(
                query=query_prefill,
                key=key_prefill,
                value=value_prefill,
                beta=beta_prefill,
                decay=decay_prefill,
                recurrent_state=chunk_recurrent_states,
            )
            prefill_outputs = jnp.where(
                prefill_query_mask,
                gdr_prefill.attention_outputs.astype(token_outputs_j.dtype),
                0,
            )
            token_outputs_j = token_outputs_j.at[safe_chunk_indices.reshape(-1)].add(
                prefill_outputs.reshape(-1, num_v_heads, head_v_dim)
            )

            updated_chunk_conv_states = _finalize_qwen3_next_conv_state_from_combined(
                combined_inputs,
                chunk_lengths + d_conv,
                d_conv=d_conv,
                output_dtype=base_conv_states.dtype,
            )
            updated_chunk_recurrent_states = gdr_prefill.recurrent_state.astype(base_recurrent_states.dtype)
            return token_outputs_j, (
                updated_chunk_conv_states,
                updated_chunk_recurrent_states,
                chunk_slots_j,
                chunk_valid_j,
            )

        def _skip_chunk(operand):
            token_outputs_j, chunk_slots_j, chunk_valid_j = operand
            return token_outputs_j, (
                empty_conv_updates,
                empty_recurrent_updates,
                chunk_slots_j,
                chunk_valid_j,
            )

        return jax.lax.cond(
            jnp.any(chunk_valid),
            _apply_chunk,
            _skip_chunk,
            (token_outputs_i, chunk_slots, chunk_valid),
        )

    def _run_prefill_scan(operand):
        token_outputs_i, conv_states_i, recurrent_states_i = operand
        token_outputs_i, scan_outputs = jax.lax.scan(
            _prefill_chunk_step,
            token_outputs_i,
            (packed_slots, packed_valid),
        )

        chunk_conv_updates, chunk_recurrent_updates, chunk_slots, chunk_valid = scan_outputs
        flat_slots = chunk_slots.reshape(-1)
        flat_valid = chunk_valid.reshape(-1)
        conv_states_i = _scatter_qwen3_next_selected_updates(
            conv_states_i,
            flat_slots,
            flat_valid,
            chunk_conv_updates.reshape(-1, *chunk_conv_updates.shape[2:]),
        )
        recurrent_states_i = _scatter_qwen3_next_selected_updates(
            recurrent_states_i,
            flat_slots,
            flat_valid,
            chunk_recurrent_updates.reshape(-1, *chunk_recurrent_updates.shape[2:]),
        )
        return token_outputs_i, conv_states_i, recurrent_states_i

    token_outputs, final_prefix_conv_states, final_prefix_recurrent_states = jax.lax.cond(
        jnp.any(multi_slot_mask),
        _run_prefill_scan,
        lambda operand: operand,
        (token_outputs, base_conv_states, base_recurrent_states),
    )

    out_conv_states = conv_states.at[:num_slots].set(final_prefix_conv_states)
    out_recurrent_states = recurrent_states.at[:num_slots].set(final_prefix_recurrent_states)

    return out_conv_states, out_recurrent_states, token_outputs


def _apply_qwen3_next_packed_updates(
    *,
    conv_states: Float[Array, "num_slots conv_dim d_conv"],
    recurrent_states: Float[Array, "num_slots num_v_heads head_dim value_dim"],
    conv_input: Float[Array, "batch seq_len conv_dim"],
    beta: Float[Array, "batch seq_len num_v_heads"],
    decay: Float[Array, "batch seq_len num_v_heads"],
    kernel: Float[Array, "conv_dim d_conv"],
    query_start_loc: Int[Array, "num_slots_plus_1"],
    num_requests: Int[Array, ""],
    key_dim: int,
    num_k_heads: int,
    head_k_dim: int,
    num_v_heads: int,
    head_v_dim: int,
    expand_ratio: int,
    conv_output_dtype: jnp.dtype,
    gdr_op: GatedDeltaRuleOp,
) -> tuple[
    Float[Array, "num_slots conv_dim d_conv"],
    Float[Array, "num_slots num_v_heads head_dim value_dim"],
    Float[Array, "seq_len num_v_heads head_v_dim"],
]:
    """Dispatch function that selects the packed update implementation.

    This is the public entry point called by ``Qwen3NextLinearAttention`` during
    packed-batch (continuous batching) inference. It delegates to
    ``_apply_qwen3_next_packed_updates_unified``, which provides the
    production-quality implementation with a single-token fast lane and
    scan-based prefill processing.

    An alternative legacy implementation
    (``_apply_qwen3_next_packed_updates_legacy``) and a slow reference
    implementation (``_apply_qwen3_next_packed_slow_updates``) exist for
    benchmarking and correctness verification but are not called from this
    dispatch point.

    All arguments are forwarded directly to the underlying implementation.
    See ``_apply_qwen3_next_packed_updates_unified`` for detailed argument
    and return value documentation.

    Args:
        conv_states: Per-slot conv state, shape ``[num_slots, conv_dim, d_conv]``.
        recurrent_states: Per-slot recurrent state,
            shape ``[num_slots, num_v_heads, head_dim, value_dim]``.
        conv_input: Packed input, shape ``[batch, seq_len, conv_dim]``.
        beta: Gating coefficients, shape ``[batch, seq_len, num_v_heads]``.
        decay: Decay factors, shape ``[batch, seq_len, num_v_heads]``.
        kernel: Depthwise conv kernel, shape ``[conv_dim, d_conv]``.
        query_start_loc: Cumulative token offsets, length ``num_slots + 1``.
        num_requests: Scalar number of active requests.
        key_dim: Total key dimension.
        num_k_heads: Number of key/query heads.
        head_k_dim: Per-head key dimension.
        num_v_heads: Number of value heads.
        head_v_dim: Per-head value dimension.
        expand_ratio: Value-head expansion ratio.
        conv_output_dtype: Dtype for conv outputs.
        gdr_op: GatedDeltaRuleOp instance.

    Returns:
        A tuple of (updated conv_states, updated recurrent_states,
        token_outputs of shape ``[seq_len, num_v_heads, head_v_dim]``).
    """
    return _apply_qwen3_next_packed_updates_unified(
        conv_states=conv_states,
        recurrent_states=recurrent_states,
        conv_input=conv_input,
        beta=beta,
        decay=decay,
        kernel=kernel,
        query_start_loc=query_start_loc,
        num_requests=num_requests,
        key_dim=key_dim,
        num_k_heads=num_k_heads,
        head_k_dim=head_k_dim,
        num_v_heads=num_v_heads,
        head_v_dim=head_v_dim,
        expand_ratio=expand_ratio,
        conv_output_dtype=conv_output_dtype,
        gdr_op=gdr_op,
    )


class Qwen3NextRMSNorm(nn.Module):
    """RMSNorm for Qwen3Next with (1 + weight) scaling formula.

    Qwen3Next uses a modified RMSNorm where the weight is centered at 1:
    output = (1 + weight) * RMSNorm(x)

    This allows initializing weight to zeros while having identity scaling.

    Attributes:
        hidden_size: Dimension of the input features.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Qwen3NextRMSNorm layer.

        Args:
            hidden_size (int): Dimension of the input features.
            eps (float, optional): Small constant for numerical stability. Defaults to 1e-6.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            rngs (nn.Rngs): Random number generator state.
        """
        self.hidden_size = hidden_size
        self.eps = eps
        self.dtype = dtype
        self.param_dtype = param_dtype

        self.kernel = nn.Param(jnp.zeros((hidden_size,), dtype=param_dtype))

    def craft_sharding(self, *, partition_manager=None, **_kwargs) -> dict[str, object]:
        """Return sharding specs for normalization parameters."""
        return {"kernel": Replicated}

    def _norm(self, x):
        """Compute RMS normalization.

        Args:
            x: Input tensor to normalize.

        Returns:
            RMS-normalized tensor.
        """
        return x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)

    def __call__(self, hidden_states: Float[Array, "... hidden_size"]) -> Float[Array, "... hidden_size"]:
        """
        Apply RMSNorm with (1 + weight) formula.

        Args:
            hidden_states: Input tensor to normalize.

        Returns:
            Normalized tensor.
        """
        org_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        output = self._norm(hidden_states)
        output = output * (1.0 + self.kernel.value.astype(jnp.float32))
        return output.astype(org_dtype)


class Qwen3NextMLP(nn.Module):
    """Qwen3Next dense MLP module.

    Standard gated MLP with SiLU activation, used for layers
    that don't use MoE.

    Attributes:
        config: Model configuration.
        dtype: Computation data type.
        param_dtype: Parameter data type.
        precision: JAX precision setting.
    """

    def __init__(
        self,
        config: Qwen3NextConfig,
        intermediate_size: int | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Qwen3Next MLP block.

        Args:
            config (Qwen3NextConfig): Model configuration with MLP parameters.
            intermediate_size (int | None, optional): Override intermediate dimension.
                Uses config.intermediate_size if None. Defaults to None.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        column_parallel_linear = partial(
            ColumnParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        row_parallel_linear = partial(
            RowParallelLinear,
            dtype=dtype,
            param_dtype=param_dtype,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
            precision=precision,
            rngs=rngs,
        )
        self.gate_proj = column_parallel_linear(config.hidden_size, intermediate_size, rngs=rngs)
        self.down_proj = row_parallel_linear(intermediate_size, config.hidden_size, rngs=rngs)
        self.up_proj = column_parallel_linear(config.hidden_size, intermediate_size, rngs=rngs)
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> jnp.ndarray:
        """Apply SwiGLU feedforward transformation.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim].

        Returns:
            Transformed hidden states [batch, seq_len, hidden_dim].
        """
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        gate = checkpoint_name(self.act_fn(self.gate_proj(hidden_states)), "mlp_gate")
        up = checkpoint_name(self.up_proj(hidden_states), "mlp_up")
        hidden_states = checkpoint_name(self.down_proj(gate * up), "mlp_down")
        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        return checkpoint_name(hidden_states, "mlp_output")


class Qwen3NextMLPStack(nn.Module):
    """Qwen3Next MoE MLP using parallel MoE linear layers.

    Implements the expert MLPs for the Mixture of Experts architecture,
    using column and row parallel linear layers for efficient expert computation.

    Attributes:
        config: Model configuration.
        gate_proj: Column-parallel gate projection for all experts.
        down_proj: Row-parallel down projection for all experts.
        up_proj: Column-parallel up projection for all experts.
        act_fn: Activation function (SiLU).
    """

    reform_param: typing.ClassVar = {
        "gate_up_proj$": {
            "splits": [
                {
                    "name": "gate_proj.kernel",
                    "spliter": lambda x: x[:, : x.shape[1] // 2, :].swapaxes(-1, -2),
                },
                {
                    "name": "up_proj.kernel",
                    "spliter": lambda x: x[:, x.shape[1] // 2 :, :].swapaxes(-1, -2),
                },
            ],
            "inverse_spliter": lambda torch, gate, up: torch.cat(
                (gate.transpose(-1, -2), up.transpose(-1, -2)),
                dim=1,
            ),
        },
        "down_proj$": {
            "splits": [{"name": "down_proj.kernel", "spliter": lambda x: x.swapaxes(-1, -2)}],
            "inverse_spliter": lambda x: x.swapaxes(-1, -2),
        },
    }

    def __init__(
        self,
        config: Qwen3NextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Qwen3Next MoE MLP stack.

        Args:
            config (Qwen3NextConfig): Model configuration with MoE parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__()
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.gate_proj = ColumnParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.hidden_size,
            out_features=config.moe_intermediate_size,
            rngs=rngs,
            kernel_init=nn.initializers.normal(),
            use_bias=False,
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.down_proj = RowParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.moe_intermediate_size,
            out_features=config.hidden_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.normal(),
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.up_proj = ColumnParallelMoELinear(
            num_experts=config.num_experts,
            in_features=config.hidden_size,
            out_features=config.moe_intermediate_size,
            rngs=rngs,
            use_bias=False,
            kernel_init=nn.initializers.normal(),
            partition_manager=config.partition_manager,
            use_expert_tensor_mode=config.use_expert_tensor_mode,
            dtype=dtype,
            param_dtype=param_dtype,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        group_sizes: Array,
        sorted_experts: Array | None = None,
    ) -> Array:
        """Apply MoE MLP transformation with SwiGLU activation.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim].
            group_sizes: Sizes of token groups per expert.
            sorted_experts: Optional sorted expert indices for routing.

        Returns:
            Transformed hidden states [batch, seq_len, hidden_dim].
        """
        return self.down_proj(
            self.act_fn(self.gate_proj(hidden_states, group_sizes, sorted_experts))
            * self.up_proj(hidden_states, group_sizes, sorted_experts),
            group_sizes,
            sorted_experts,
        )


class Qwen3NextSparseMoeBlock(BaseMoeModule):
    """Sparse Mixture of Experts block for Qwen3Next.

    Routes input to selected experts and combines outputs.
    Includes optional shared expert that processes all tokens.

    Attributes:
        config: Model configuration.
        gate: Router linear layer.
        experts: Stack of expert MLPs.
        shared_expert: Optional shared expert MLP.
    """

    def __init__(
        self,
        config: Qwen3NextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Qwen3Next Sparse MoE block.

        Args:
            config (Qwen3NextConfig): Model configuration with MoE parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision for operations.
                Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            n_routed_experts=config.num_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            lbl_coef=None,
            rzl_coef=None,
            routing_strategy=MoeRoutingStrategy.TOP_K if config.norm_topk_prob else MoeRoutingStrategy.TOP_K_NDIV,
            load_balancing_strategy=MoeLoadBalancingStrategy.STANDARD,
        )
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.gate = ColumnParallelLinear(
            config.hidden_size,
            config.num_experts,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=nn.initializers.normal(config.initializer_range),
        )

        self.experts = Qwen3NextMLPStack(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.shared_expert = Qwen3NextMLP(
            config=config,
            intermediate_size=config.shared_expert_intermediate_size,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.shared_expert_gate = ColumnParallelLinear(
            config.hidden_size,
            1,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=nn.initializers.normal(config.initializer_range),
        )

    def __call__(self, hidden_states: Float[Array, "batch seq_len hidden_dim"]) -> tuple[Array, Array]:
        """Route inputs through selected experts and combine outputs.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_dim].

        Returns:
            Tuple of (output tensor, router logits) where output includes
            both routed expert outputs and shared expert contribution.
        """
        out, router_logits = self.moe_call(
            hidden_state=hidden_states,
            gate_layer=self.gate,
            expert_layer=self.experts,
            wi_kernel=self.experts.gate_proj.kernel.value,
            wu_kernel=self.experts.up_proj.kernel.value,
            wd_kernel=self.experts.down_proj.kernel.value,
            act_fn=self.experts.act_fn,
        )

        shared_out = self.shared_expert(hidden_states)
        needs_upcast = self.dtype in lowfloats
        gate_input = self.shared_expert_gate(hidden_states)
        if needs_upcast:
            gate = jax.nn.sigmoid(gate_input.astype(jnp.float32))
            shared_out = shared_out.astype(jnp.float32) * gate
            out = (out.astype(jnp.float32) + shared_out).astype(self.dtype)
        else:
            gate = jax.nn.sigmoid(gate_input)
            shared_out = shared_out * gate
            out = out + shared_out
        return checkpoint_name(out, "moe_expert_output"), checkpoint_name(router_logits, "moe_router_logits")


class Qwen3NextFullAttention(UnifiedAttention):
    """Qwen3Next full attention layer with sigmoid gating and partial RoPE.

    Features:
    - q_proj outputs 2x dimension (query + gate), matching HF structure
    - Per-head RMSNorm on Q/K
    - Sigmoid gating applied to attention output

    HuggingFace-compatible structure:
    - q_proj: [hidden_size -> num_heads * head_dim * 2] (query + gate concatenated)
    - k_proj, v_proj, o_proj: Standard projections
    - q_norm, k_norm: Per-head RMSNorm

    Attributes:
        config: Model configuration.
        layer_idx: Index of this layer.
    """

    def __init__(
        self,
        config: Qwen3NextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initialize Qwen3Next full attention layer.

        Args:
            config (Qwen3NextConfig): Model configuration with attention parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the model.
        """
        super().__init__(
            config,
            dtype,
            param_dtype,
            precision,
            rngs=rngs,
            layer_idx=layer_idx,
            attention_type="standard",
            causal=True,
            sliding_window=None,
            use_qk_norm=True,
        )
        self.layer_idx = layer_idx

    def _create_q_proj(
        self,
        config,
        dtype,
        param_dtype,
        precision,
        rngs,
    ):
        """Create query projection with 2x output for query + gate.

        HuggingFace Qwen3Next uses q_proj that outputs doubled dimension,
        which is then split into query states and gate values.
        """
        return ColumnParallelLinear(
            config.hidden_size,
            self.num_heads * self.head_dim * 2,  # 2x for query + gate
            rngs=rngs,
            use_bias=getattr(config, "attention_bias", False),
            dtype=dtype,
            param_dtype=param_dtype,
            kernel_init=jax.nn.initializers.normal(getattr(config, "initializer_range", 0.02)),
            precision=precision,
        )

    def _create_q_norm(self, config, dtype, param_dtype, rngs):
        """Use Qwen3Next RMSNorm (1 + weight) for query normalization."""
        return Qwen3NextRMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def _create_k_norm(self, config, dtype, param_dtype, rngs):
        """Use Qwen3Next RMSNorm (1 + weight) for key normalization."""
        return Qwen3NextRMSNorm(
            self.head_dim,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def _postprocess_qkv(self, query_states, key_states, value_states):
        """Apply Q/K normalization after computing query, key, and value projections.

        Args:
            query_states: Query tensor from projection layer.
            key_states: Key tensor from projection layer.
            value_states: Value tensor from projection layer.

        Returns:
            Tuple of normalized query, normalized key, and value tensors.
        """
        query_states = self.query_normalization(query_states)
        key_states = self.key_normalization(key_states)
        return query_states, key_states, value_states

    def forward(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | LinearCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesCacheView | OperationsMetadata | None = None,
        output_attentions: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
        alibi: Float[Array, "batch_or_1 heads qseq_len_or_1 kvseq_len_or_1"] | None = None,
    ) -> AttentionLayerOutput:
        """Forward pass for Qwen3Next full attention with gated query.

        The q_proj outputs 2x dimension (query + gate), which is split and the gate
        is applied to attention output via sigmoid activation.

        Args:
            hidden_states (Array): Input tensor of shape (batch, seq_len, hidden_dim).
            mask_info (MaskInfo | None): Attention mask information including causal masks.
            position_ids (Array): Position indices for tokens, shape (batch, seq_len).
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode) for optimization.
            cache_view (TransformerCacheView | RaggedPagesCacheView | LinearCacheView | None, optional):
                Cache view for KV caching. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesCacheView | OperationsMetadata | None, optional):
                Cache metadata. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies. Defaults to None.
            alibi (Array | None, optional): ALiBi attention biases. Defaults to None.

        Returns:
            AttentionLayerOutput: Contains attention output, optional weights, and cache view.
        """
        batch_size = hidden_states.shape[0]
        sequence_length = hidden_states.shape[1]

        q_proj_output = checkpoint_name(self.q_proj(hidden_states), "attn_query")
        q_proj_output = apply_logical_sharding(
            q_proj_output,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        q_proj_output = q_proj_output.reshape(batch_size, sequence_length, -1, self.head_dim * 2)
        query_states, gate = jnp.split(q_proj_output, 2, axis=-1)

        key_states = checkpoint_name(self.k_proj(hidden_states), "attn_key")
        value_states = checkpoint_name(self.v_proj(hidden_states), "attn_value")
        key_states = apply_logical_sharding(
            key_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        value_states = apply_logical_sharding(
            value_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        key_states = key_states.reshape(batch_size, sequence_length, -1, self.head_dim)
        value_states = value_states.reshape(batch_size, sequence_length, -1, self.head_dim)

        query_states, key_states, value_states = self._postprocess_qkv(query_states, key_states, value_states)

        query_states, key_states, value_states = self.apply_qkv_shardings(query_states, key_states, value_states)

        query_states, key_states = self._apply_rotary(query_states, key_states, position_ids, frequencies)

        (
            key_states,
            value_states,
            mask_info,
            init_attention_bias,
            cache_view,
            cache_metadata,
        ) = self.concatenate(
            query=query_states,
            key=key_states,
            value=value_states,
            cache_view=cache_view,
            cache_metadata=cache_metadata,
            mask_info=mask_info,
            sliding_window=self.sliding_window,
        )

        softmax_aux = getattr(self, "sinks", getattr(self, "softmax_aux", None))
        softmax_aux = getattr(softmax_aux, "value", softmax_aux)

        attentions: AttentionLayerOutput = self.attention_performer.forward(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            mode=mode,
            bias=None,
            cache_metadata=cache_metadata,
            cache_view=cache_view,
            init_bias=init_attention_bias,
            mask_info=mask_info,
            causal=self.causal,
            sliding_window=self.sliding_window,
            softmax_aux=softmax_aux,
        )

        if attentions.cache_view is not None:
            cache_view = attentions.cache_view

        attn_output = attentions.attention_outputs

        # Re-apply QKV sharding so attn_output matches the gate's partition
        # (which retains "sp" on the sequence axis from HiddenStateSharding).
        # Without this, blocksparse's shard_map strips "sp" from the sequence
        # dimension, causing GSPMD to insert f32 all-gathers during backward.
        attn_output = apply_logical_sharding(
            attn_output,
            dynamic_axes=common_types.AttnQSharding,
            partition_manager=self.config.partition_manager,
        )

        if attn_output.dtype in lowfloats or gate.dtype in lowfloats:
            attn_output_dtype = attn_output.dtype
            attn_output = attn_output.astype(jnp.float32) * jax.nn.sigmoid(gate.astype(jnp.float32))
            attn_output = attn_output.astype(attn_output_dtype)
        else:
            attn_output = attn_output * jax.nn.sigmoid(gate)

        attn_output = self._merge_heads(attn_output)
        attn_output = self.shard_attention_prod(attn_output)
        attn_output = self.o_proj(attn_output)

        return AttentionLayerOutput(
            attention_output=attn_output,
            attention_weight=attentions.attention_weights if output_attentions else None,
            cache_view=cache_view,
        )


class Qwen3NextLinearAttention(nn.Module):
    """Qwen3Next linear attention layer using GatedDeltaNet.

    Implements linear attention with:
    - Causal 1D convolution for local context
    - Gated delta rule recurrence for global context
    - Learnable decay for state forgetting (A_log parameter)
    - Mamba-style dt_bias for time discretization

    HuggingFace-compatible parameter naming:
    - Packed mode (Qwen3-Next): in_proj_qkvz and in_proj_ba
    - Split mode (Qwen3.5): in_proj_qkv, in_proj_z, in_proj_b, and in_proj_a
    - A_log: Log of decay matrix A
    - dt_bias: Time discretization bias
    - conv1d: Causal convolution
    - norm: Gated RMSNorm
    - out_proj: Output projection

    Attributes:
        config: Model configuration.
        layer_idx: Index of this layer.
    """

    reform_param: typing.ClassVar = {
        "conv1d.weight$": {
            "splits": [{"name": "conv1d.kernel", "spliter": lambda x: x.permute(2, 1, 0)}],
            "inverse_spliter": lambda torch, kernel: kernel.permute(2, 1, 0),
        },
    }

    def __init__(
        self,
        config: Qwen3NextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initialize Qwen3Next linear attention layer.

        Args:
            config (Qwen3NextConfig): Model configuration with linear attention parameters.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the model.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.layer_idx = layer_idx

        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim

        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        self.uses_split_proj = bool(getattr(config, "linear_attention_separate_proj", False))
        if self.uses_split_proj:
            self.in_proj_qkv = ColumnParallelLinear(
                config.hidden_size,
                self.key_dim * 2 + self.value_dim,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                kernel_init=jax.nn.initializers.normal(config.initializer_range),
            )
            self.in_proj_z = ColumnParallelLinear(
                config.hidden_size,
                self.value_dim,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                kernel_init=jax.nn.initializers.normal(config.initializer_range),
            )
            self.in_proj_b = ColumnParallelLinear(
                config.hidden_size,
                self.num_v_heads,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                kernel_init=jax.nn.initializers.normal(config.initializer_range),
            )
            self.in_proj_a = ColumnParallelLinear(
                config.hidden_size,
                self.num_v_heads,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                kernel_init=jax.nn.initializers.normal(config.initializer_range),
            )
        else:
            qkvz_dim = self.key_dim * 2 + self.value_dim * 2
            self.in_proj_qkvz = ColumnParallelLinear(
                config.hidden_size,
                qkvz_dim,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                kernel_init=jax.nn.initializers.normal(config.initializer_range),
            )

            ba_dim = self.num_v_heads * 2
            self.in_proj_ba = ColumnParallelLinear(
                config.hidden_size,
                ba_dim,
                use_bias=False,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                kernel_init=jax.nn.initializers.normal(config.initializer_range),
            )

        self.out_proj = RowParallelLinear(
            self.value_dim,
            config.hidden_size,
            use_bias=False,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            kernel_init=jax.nn.initializers.normal(config.initializer_range),
        )

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

        self.conv1d = nn.Conv(
            in_features=self.conv_dim,
            out_features=self.conv_dim,
            kernel_size=(config.linear_conv_kernel_dim,),
            feature_group_count=self.conv_dim,
            padding=((config.linear_conv_kernel_dim - 1, 0),),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
            use_bias=False,
        )

        self.A_log = nn.Param(
            jnp.log(
                jax.random.uniform(
                    rngs.params(),
                    (self.num_v_heads,),
                    dtype=param_dtype,
                    minval=1.0,
                    maxval=16.0,
                )
            )
        )

        self.dt_bias = nn.Param(jnp.ones((self.num_v_heads,), dtype=param_dtype))

        metadata = OperationMetadata(
            runtime_dtype=self.dtype,
            runtime_softmax_dtype=jnp.float32,
            base_config=self.config,
        )
        self.gdr_op = GatedDeltaRuleOp(metadata)

    def craft_sharding(self, *, partition_manager=None, **_kwargs) -> dict[str, object]:
        """Return sharding specifications for non-standard parameters.

        Marks A_log and dt_bias as replicated across all devices since they are
        small per-head parameters that do not benefit from sharding.

        Args:
            partition_manager: Partition manager (unused, for interface compatibility).
            **_kwargs: Additional keyword arguments (unused).

        Returns:
            dict[str, object]: Mapping of parameter names to sharding specifications.
        """
        return {"A_log": Replicated, "dt_bias": Replicated}

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: Float[Array, "batch seq proj_dim"],
        mixed_ba: Float[Array, "batch seq ba_dim"],
    ):
        """Reorder QKV from grouped layout to separate tensors.

        HuggingFace organizes the projection output in a grouped layout:
        [q_h0, k_h0, v_h0, z_h0, q_h1, k_h1, v_h1, z_h1, ...]

        This method unpacks that layout into separate Q, K, V, Z tensors.

        Args:
            mixed_qkvz: Projected QKVZ tensor in grouped layout.
            mixed_ba: Projected beta/alpha tensor.

        Returns:
            Tuple of (query, key, value, z, beta, alpha) tensors.
        """
        batch, seq, _ = mixed_qkvz.shape

        expand_ratio = self.num_v_heads // self.num_k_heads
        per_head_dim = 2 * self.head_k_dim + 2 * self.head_v_dim * expand_ratio
        mixed_qkvz = mixed_qkvz.reshape(batch, seq, self.num_k_heads, per_head_dim)

        split_sizes = [
            self.head_k_dim,
            self.head_k_dim,
            expand_ratio * self.head_v_dim,
            expand_ratio * self.head_v_dim,
        ]
        query, key, value, z = jnp.split(
            mixed_qkvz,
            [split_sizes[0], split_sizes[0] + split_sizes[1], split_sizes[0] + split_sizes[1] + split_sizes[2]],
            axis=-1,
        )

        value = value.reshape(batch, seq, self.num_v_heads, self.head_v_dim)
        z = z.reshape(batch, seq, self.num_v_heads, self.head_v_dim)

        ba_per_head = 2 * expand_ratio
        mixed_ba = mixed_ba.reshape(batch, seq, self.num_k_heads, ba_per_head)
        beta, alpha = jnp.split(mixed_ba, [expand_ratio], axis=-1)
        beta = beta.reshape(batch, seq, self.num_v_heads)
        alpha = alpha.reshape(batch, seq, self.num_v_heads)

        return query, key, value, z, beta, alpha

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo | None,
        cache_view: LinearCacheView | None = None,
        cache_metadata: LinearMetadata | None = None,
    ) -> DecoderLayerOutput:
        """Forward pass through the linear attention layer.

        Uses GatedDeltaNet for efficient linear complexity attention with:
        - Causal 1D convolution for local context
        - Delta rule recurrence for global context
        - Gated output normalization

        Args:
            hidden_states (Array): Input tensor of shape (batch, seq_len, hidden_dim).
            mask_info (MaskInfo): Mask information for padding handling.
            cache_view (LinearCacheView | None, optional): Cache view for incremental
                decoding with conv and recurrent states. Defaults to None.
            cache_metadata (LinearMetadata | None, optional): Cache metadata. Defaults to None.

        Returns:
            AttentionLayerOutput: Contains attention output and updated cache view.
        """
        if mask_info is not None:
            q_mask: Array | None = typing.cast("Array | None", mask_info.q_attention_mask)
            if q_mask is not None and q_mask.shape[1] != hidden_states.shape[1]:
                q_mask = q_mask[:, : hidden_states.shape[1]]
            hidden_states = apply_mask_to_padding_states(hidden_states, q_mask)

        batch_size, seq_len, _ = hidden_states.shape
        is_inference = seq_len == 1 and cache_view is not None
        expand_ratio = self.num_v_heads // self.num_k_heads

        if self.uses_split_proj:
            projected_qkv = self.in_proj_qkv(hidden_states)
            z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
            beta = self.in_proj_b(hidden_states)
            alpha = self.in_proj_a(hidden_states)
            conv_input = projected_qkv
        else:
            projected_qkvz = self.in_proj_qkvz(hidden_states)
            projected_ba = self.in_proj_ba(hidden_states)
            query, key, value, z, beta, alpha = self.fix_query_key_value_ordering(projected_qkvz, projected_ba)
            query_flat = query.reshape(batch_size, seq_len, -1)
            key_flat = key.reshape(batch_size, seq_len, -1)
            value_flat = value.reshape(batch_size, seq_len, -1)
            conv_input = jnp.concatenate([query_flat, key_flat, value_flat], axis=-1)
        # conv_input: [batch, seq_len, conv_dim]

        A = -jnp.exp(self.A_log.value.astype(jnp.float32))
        alpha_biased = alpha.astype(jnp.float32) + self.dt_bias.value.astype(jnp.float32)
        decay = A[None, None, :] * jax.nn.softplus(alpha_biased)
        beta = jax.nn.sigmoid(beta)

        new_conv_state = None

        packed_query_start_loc = getattr(cache_metadata, "query_start_loc", None) if cache_metadata is not None else None
        packed_num_seqs = getattr(cache_metadata, "num_seqs", None) if cache_metadata is not None else None
        use_packed_state_updates = (
            cache_view is not None
            and cache_view.conv_state is not None
            and cache_view.recurrent_state is not None
            and batch_size == 1
            and packed_query_start_loc is not None
            and packed_num_seqs is not None
        )

        if use_packed_state_updates:
            conv_states = cache_view.conv_state
            recurrent_states = cache_view.recurrent_state
            # bf16 is the smallest dtype that preserves silu precision for conv outputs.
            conv_output_dtype = jnp.bfloat16 if self.dtype in lowfloats else self.dtype

            query_start_loc = jnp.asarray(packed_query_start_loc, dtype=jnp.int32)
            num_seqs_arr = jnp.asarray(packed_num_seqs, dtype=jnp.int32).reshape(-1)
            num_requests = num_seqs_arr[0]

            kernel = self.conv1d.kernel.value  # [kernel_size, 1, conv_dim]
            kernel = jnp.squeeze(kernel, axis=1).T  # [conv_dim, kernel_size]

            conv_states, recurrent_states, token_outputs = _apply_qwen3_next_packed_updates(
                conv_states=conv_states,
                recurrent_states=recurrent_states,
                conv_input=conv_input,
                beta=beta,
                decay=decay,
                kernel=kernel,
                query_start_loc=query_start_loc,
                num_requests=num_requests,
                key_dim=self.key_dim,
                num_k_heads=self.num_k_heads,
                head_k_dim=self.head_k_dim,
                num_v_heads=self.num_v_heads,
                head_v_dim=self.head_v_dim,
                expand_ratio=expand_ratio,
                conv_output_dtype=conv_output_dtype,
                gdr_op=self.gdr_op,
            )

            output = token_outputs[None, ...]
            new_cache_view = cache_view.replace(
                conv_state=conv_states,
                recurrent_state=recurrent_states,
            )

        elif cache_view is not None:
            conv_output_dtype = jnp.bfloat16 if self.dtype in lowfloats else self.dtype
            conv_output, new_conv_state = apply_conv_with_state(
                conv_input,
                self.conv1d,
                cache_view.conv_state,
                is_inference=is_inference,
                d_conv=self.config.linear_conv_kernel_dim,
                output_dtype=conv_output_dtype,
                reuse_partial_state=False,
            )
        else:
            # Training/prefill without cache: use the full convolution directly.
            conv_output = jax.nn.silu(self.conv1d(conv_input))

        if use_packed_state_updates:
            output = self.norm(output, z)
            output = output.reshape(batch_size, seq_len, -1)
            output = self.out_proj(output)

            return AttentionLayerOutput(attention_output=output, attention_weight=None, cache_view=new_cache_view)

        conv_query = conv_output[:, :, : self.key_dim]
        conv_key = conv_output[:, :, self.key_dim : self.key_dim * 2]
        conv_value = conv_output[:, :, self.key_dim * 2 :]

        query = conv_query.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = conv_key.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = conv_value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)

        query = apply_logical_sharding(
            query,
            dynamic_axes=common_types.AttnQSharding,
            partition_manager=self.config.partition_manager,
        )
        key = apply_logical_sharding(
            key,
            dynamic_axes=common_types.AttnKVSharding,
            partition_manager=self.config.partition_manager,
        )
        value = apply_logical_sharding(
            value,
            dynamic_axes=common_types.AttnKVSharding,
            partition_manager=self.config.partition_manager,
        )

        recurrent_state = None
        if cache_view is not None and cache_view.recurrent_state is not None:
            recurrent_state = cache_view.recurrent_state

        if expand_ratio > 1:
            grouped_decode = is_inference and recurrent_state is not None
        else:
            grouped_decode = False

        if expand_ratio > 1 and not grouped_decode:
            query = jnp.repeat(query, expand_ratio, axis=2)
            key = jnp.repeat(key, expand_ratio, axis=2)

        if not use_packed_state_updates:
            if grouped_decode:
                output, new_recurrent_state = apply_grouped_single_step_gdr(
                    query=query,
                    key=key,
                    value=value,
                    beta=beta,
                    decay=decay,
                    recurrent_state=recurrent_state,
                    gdr_op=self.gdr_op,
                )
                new_recurrent_state = _preserve_array_sharding(
                    new_recurrent_state,
                    partition_manager=self.config.partition_manager,
                    partition_axis=self.config.partition_axis,
                )
            else:
                gdr_output: GatedDeltaRuleOutput = self.gdr_op(
                    query=query,
                    key=key,
                    value=value,
                    beta=beta,
                    decay=decay,
                    conv_state=None,  # conv_state is handled separately above
                    recurrent_state=recurrent_state,
                )
                output = gdr_output.attention_outputs
                new_recurrent_state = gdr_output.recurrent_state

        output = self.norm(output, z)
        output = output.reshape(batch_size, seq_len, -1)
        output = apply_logical_sharding(
            output,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        output = self.out_proj(output)

        new_cache_view = cache_view
        if cache_view is not None and not use_packed_state_updates:
            new_cache_view = cache_view.replace(
                conv_state=new_conv_state if new_conv_state is not None else cache_view.conv_state,
                recurrent_state=new_recurrent_state,
            )

        return AttentionLayerOutput(
            attention_output=output,
            attention_weight=None,
            cache_view=new_cache_view,
        )  # pyright: ignore[reportReturnType]


class Qwen3NextDecoderLayer(nn.Module):
    """Qwen3Next transformer decoder layer.

    Combines either full or linear attention with MoE or dense MLP,
    based on layer configuration.

    Attributes:
        config: Model configuration.
        layer_idx: Index of this layer.
        is_full_attention: Whether this layer uses full attention.
        is_moe: Whether this layer uses MoE FFN.
    """

    def __init__(
        self,
        config: Qwen3NextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
        layer_idx: int,
    ):
        """Initialize Qwen3Next decoder layer.

        Args:
            config (Qwen3NextConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
            layer_idx (int): Index of this layer in the model.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.layer_idx = layer_idx

        self.is_full_attention = config.is_full_attention_layer(layer_idx)
        self.is_moe = config.is_moe_layer(layer_idx)

        if self.is_full_attention:
            self.self_attn = Qwen3NextFullAttention(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                layer_idx=layer_idx,
            )
        else:
            self.linear_attn = Qwen3NextLinearAttention(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
                layer_idx=layer_idx,
            )

        if self.is_moe:
            self.mlp = Qwen3NextSparseMoeBlock(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )
        else:
            self.mlp = Qwen3NextMLP(
                config=config,
                dtype=dtype,
                param_dtype=param_dtype,
                precision=precision,
                rngs=rngs,
            )

        self.input_layernorm = Qwen3NextRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        self.post_attention_layernorm = Qwen3NextRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        hidden_states: Float[Array, "batch seq_len hidden_dim"],
        mask_info: MaskInfo,
        position_ids: Int[Array, "batch seq_len"],
        mode: common_types.RUNTIME_MODE_TYPES,  # type:ignore
        cache_view: TransformerCacheView | RaggedPagesCacheView | LinearCacheView | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesCacheView | OperationsMetadata | None = None,
        output_attentions: bool = False,
        output_router_logits: bool = False,
        frequencies: Float[Array, "seq_len head_dim"] | None = None,
    ) -> DecoderLayerOutput:
        """Forward pass through the decoder layer.

        Applies pre-normalization architecture with either full or linear attention,
        followed by either MoE or dense MLP based on layer configuration.

        Args:
            hidden_states (Array): Input tensor of shape (batch, seq_len, hidden_dim).
            mask_info (MaskInfo): Attention mask information including causal masks.
            position_ids (Array): Position indices for tokens, shape (batch, seq_len).
            mode (RUNTIME_MODE_TYPES): Runtime mode (train, decode) for optimization.
            cache_view (TransformerCacheView | RaggedPagesCacheView | LinearCacheView | None, optional):
                Cache view for KV or recurrent state caching. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesCacheView | OperationsMetadata | None, optional):
                Cache metadata. Defaults to None.
            output_attentions (bool, optional): Whether to return attention weights. Defaults to False.
            output_router_logits (bool, optional): Whether to return MoE router logits. Defaults to False.
            frequencies (Array | None, optional): Precomputed RoPE frequencies. Defaults to None.

        Returns:
            DecoderLayerOutput: Contains hidden states, optional attention weights,
                optional router logits, and cache view.
        """
        normed_hidden = self.input_layernorm(hidden_states)

        if self.is_full_attention:
            attn_outputs = self.self_attn(
                normed_hidden,
                mask_info,
                position_ids,
                mode,
                cache_view,
                cache_metadata,
                output_attentions,
                frequencies,
            )
        else:
            attn_outputs = self.linear_attn(
                normed_hidden,
                mask_info,
                cache_view,
                cache_metadata,
            )

        attn_output = attn_outputs.attention_output
        attn_weight = attn_outputs.attention_weight
        cache_view = attn_outputs.cache_view
        hidden_states = checkpoint_name(hidden_states + attn_output, "residual")

        feed_forward_input = self.post_attention_layernorm(hidden_states)

        feed_forward_output = self.mlp(feed_forward_input)

        router_logits = None
        if self.is_moe:
            feed_forward_output, router_logits = feed_forward_output

        hidden_states = checkpoint_name(hidden_states + feed_forward_output, "residual")

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )

        return DecoderLayerOutput(
            hidden_states=checkpoint_name(hidden_states, "layer_output"),
            attention_weight=attn_weight if output_attentions else None,
            router_logits=router_logits if output_router_logits else None,
            cache_view=cache_view,
        )


@register_module(TaskType.BASE_MODULE, config=Qwen3NextConfig, model_type="qwen3_next")
class Qwen3NextModel(EasyDeLBaseModule):
    """Qwen3Next base transformer model.

    Implements the core transformer architecture with hybrid attention
    (alternating between full and linear attention) and MoE FFN layers.

    Attributes:
        config: Model configuration.
        embed_tokens: Token embedding layer.
        layers: List of decoder layers.
        norm: Final layer normalization.
    """

    def __init__(
        self,
        config: Qwen3NextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Qwen3Next base model.

        Args:
            config (Qwen3NextConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
        )

        self.embed_tokens = Embed(
            config.vocab_size,
            config.hidden_size,
            embedding_init=jax.nn.initializers.normal(stddev=config.initializer_range),
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )
        remat_layer_block = auto_remat(
            Qwen3NextDecoderLayer,
            policy=config.gradient_checkpointing,
            save_names=config.gradient_checkpointing_targets,
            exclude_names=config.gradient_checkpointing_targets,
        )
        self.layers = nn.List(
            [
                remat_layer_block(
                    config=config,
                    layer_idx=layer_idx,
                    dtype=dtype,
                    param_dtype=param_dtype,
                    precision=precision,
                    rngs=rngs,
                )
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3NextRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
            param_dtype=param_dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
    ) -> MoeModelOutput:
        """Forward pass through the Qwen3Next base model.

        Processes input tokens through embedding, all decoder layers with hybrid
        attention (full and linear), optional MoE layers, and final normalization.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch, seq_len).
                Must be provided if inputs_embeds is None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch, seq_len, hidden_size). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on
                padding tokens, shape (batch, seq_len). Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention
                operations. Defaults to None.
            position_ids (Array | None, optional): Position indices for each token, shape
                (batch, seq_len). Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights
                from all layers. Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states
                from all layers. Defaults to None.
            output_router_logits (bool | None, optional): Whether to return MoE router
                logits from all MoE layers. Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode) for
                optimizations. Auto-detected if None. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache with precomputed states for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.

        Returns:
            MoeModelOutput: Contains last_hidden_state, optional hidden_states, optional
                attentions, updated past_key_values, and optional router_logits.

        Raises:
            ValueError: If both input_ids and inputs_embeds are provided or both are None.
            AssertionError: If sequence_length exceeds max_position_embeddings.
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = checkpoint_name(self.embed_tokens(input_ids.astype("i4")), "embeddings")

        sequence_length = inputs_embeds.shape[1]
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        all_router_logits = () if output_router_logits else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None
        assert sequence_length <= self.config.max_position_embeddings, (
            f"Maximum Position Embedding Reached ! "
            f"(Expected <= {self.config.max_position_embeddings} got {sequence_length})"
        )
        mask_info = MaskInfo.dynamic_init(
            mask_info=mask_info,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        if position_ids is None:
            position_ids = mask_info.q_position_ids

        hidden_states = inputs_embeds
        if mode is None:
            mode = (
                common_types.MODE_DECODE
                if sequence_length == 1 and past_key_values is not None
                else common_types.MODE_TRAIN
            )
        if past_key_values is None:
            past_key_values = HybridCache.init_empty(len(self.layers))

        hidden_states = apply_logical_sharding(
            hidden_states,
            dynamic_axes=common_types.HiddenStateSharding,
            partition_manager=self.config.partition_manager,
        )
        for idx, block in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = block(
                hidden_states=hidden_states,
                mask_info=mask_info,
                position_ids=position_ids,
                mode=mode,
                cache_view=past_key_values.views[idx],
                cache_metadata=cache_metadata,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                frequencies=self.frequencies,
            )
            hidden_states = layer_outputs.hidden_states
            if output_attentions:
                all_attentions += (layer_outputs.attention_weight,)

            past_key_values[idx] = layer_outputs.cache_view
            if output_router_logits and layer_outputs.router_logits is not None:
                all_router_logits += (layer_outputs.router_logits,)

        hidden_states = checkpoint_name(self.norm(hidden_states), "model_output")

        return MoeModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            past_key_values=past_key_values,
            router_logits=all_router_logits,
        )

    def get_encoder(self) -> None:
        """Get the encoder component of the model.

        Raises:
            NotImplementedError: Qwen3Next is a decoder-only model.
        """
        raise NotImplementedError("This is a decoder-only model and does not have an encoder.")

    def get_decoder(self) -> "Qwen3NextModel":
        """Get the decoder component of the model.

        Returns:
            The model itself as Qwen3Next is a decoder-only architecture.
        """
        return self

    def get_lm_head(self) -> None:
        """Get the language model head.

        Raises:
            NotImplementedError: Base model does not have a language model head.
        """
        raise NotImplementedError("The base model does not have a language model head.")

    def get_embedding(self) -> Embed:
        """Get the embedding layer of the model.

        Returns:
            The token embedding layer.
        """
        return self.embed_tokens


@register_module(TaskType.CAUSAL_LM, config=Qwen3NextConfig, model_type="qwen3_next")
class Qwen3NextForCausalLM(BaseCausalLMModule[Qwen3NextModel, Qwen3NextConfig]):  # type: ignore
    """Qwen3Next model with a causal language modeling head.

    Extends the base Qwen3NextModel with a linear output layer for
    next-token prediction.

    Attributes:
        model: Base Qwen3NextModel.
        lm_head: Linear projection to vocabulary.
    """

    _task_type = TaskType.CAUSAL_LM
    _model_type = "qwen3_next"
    _config_class = Qwen3NextConfig

    def __init__(
        self,
        config: Qwen3NextConfig,
        dtype: jnp.dtype = jnp.bfloat16,
        param_dtype: jnp.dtype = jnp.bfloat16,
        precision: jax.lax.PrecisionLike = None,
        *,
        rngs: nn.Rngs,
    ):
        """Initialize Qwen3Next model for causal language modeling.

        Args:
            config (Qwen3NextConfig): Model configuration.
            dtype (jnp.dtype, optional): Data type for computation. Defaults to jnp.bfloat16.
            param_dtype (jnp.dtype, optional): Data type for parameters. Defaults to jnp.bfloat16.
            precision (jax.lax.PrecisionLike, optional): Numerical precision. Defaults to None.
            rngs (nn.Rngs): Random number generator state.
        """
        super().__init__(
            config=config,
            base_model_class=Qwen3NextModel,
            base_model_name="model",
            dtype=dtype,
            param_dtype=param_dtype,
            precision=precision,
            rngs=rngs,
            lm_head_bias=False,
            router_aux_loss_coef=getattr(config, "router_aux_loss_coef", None),
        )

    def __call__(
        self,
        input_ids: Int[Array, "batch seq_len"] | None = None,
        inputs_embeds: Float[Array, "batch seq_len hidden_dim"] | None = None,
        attention_mask: Bool[Array, "batch seq_len"] | None = None,
        mask_info: MaskInfo | None = None,
        position_ids: Int[Array, "batch seq_len"] | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        output_router_logits: bool | None = None,
        mode: common_types.RUNTIME_MODE_TYPES | None = None,  # type:ignore
        past_key_values: TransformerCache | RaggedPagesCache | HybridCache | None = None,
        cache_metadata: TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None = None,
        apply_lm_head: bool = True,
    ) -> MoeCausalLMOutput:
        """Forward pass for causal language modeling with MoE auxiliary loss.

        Args:
            input_ids (Array | None, optional): Input token IDs of shape (batch, seq_len).
                Must be provided if inputs_embeds is None.
            inputs_embeds (Array | None, optional): Pre-computed input embeddings of shape
                (batch, seq_len, hidden_size). Defaults to None.
            attention_mask (Array | None, optional): Boolean mask to avoid attention on
                padding tokens, shape (batch, seq_len). Defaults to None.
            mask_info (MaskInfo | None, optional): Advanced mask information for attention
                operations. Defaults to None.
            position_ids (Array | None, optional): Position indices for each token, shape
                (batch, seq_len). Defaults to None.
            output_attentions (bool | None, optional): Whether to return attention weights
                from all layers. Defaults to None.
            output_hidden_states (bool | None, optional): Whether to return hidden states
                from all layers. Defaults to None.
            output_router_logits (bool | None, optional): Whether to return MoE router
                logits from all MoE layers. Defaults to None.
            mode (RUNTIME_MODE_TYPES | None, optional): Runtime mode (train/decode) for
                optimizations. Auto-detected if None. Defaults to None.
            past_key_values (TransformerCache | RaggedPagesCache | HybridCache | None, optional):
                Cache with precomputed states for generation. Defaults to None.
            cache_metadata (TransformerMetadata | RaggedPagesMetadata | OperationsMetadata | None, optional):
                Metadata for cache management. Defaults to None.
            apply_lm_head (bool, optional): Whether to apply the language model head
                projection. Defaults to True.

        Returns:
            MoeCausalLMOutput: Contains logits, optional hidden_states, optional attentions,
                updated past_key_values, optional router_logits, and optional aux_loss.
        """
        return self.forward_moe(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            mask_info=mask_info,
            position_ids=position_ids,
            mode=mode,
            past_key_values=past_key_values,
            cache_metadata=cache_metadata,
            apply_lm_head=apply_lm_head,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            aux_loss_fn=self._compute_aux_loss,
        )

    def _compute_aux_loss(self, outputs, attention_mask):
        """Compute auxiliary load balancing loss from router logits.

        Uses the auxiliary load balancing loss function to encourage balanced
        token routing across experts.

        Args:
            outputs: Model outputs containing router_logits.
            attention_mask: Attention mask for valid tokens.

        Returns:
            Auxiliary loss value or None if no router logits are available.
        """
        if outputs.router_logits is None or len(outputs.router_logits) == 0:
            return None
        aux_loss = auxiliary_load_balancing_loss_func(
            gate_logits=outputs.router_logits,
            num_experts=self.config.num_experts,
            top_k=self.config.num_experts_per_tok,
            attention_mask=attention_mask,
        )
        return aux_loss + (aux_loss * self.config.router_aux_loss_coef)


__all__ = [
    "Qwen3NextConfig",
    "Qwen3NextForCausalLM",
    "Qwen3NextModel",
]
