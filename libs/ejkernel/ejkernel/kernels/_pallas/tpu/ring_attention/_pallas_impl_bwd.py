# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
# Copyright 2025 DeepMind Technologies Limited (modified from original tokamax implementation).
# (we dont use their splash impl as is, but modified our splash for ring attention)
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

"""Ring Attention implementation using Splash Attention kernels.

This module provides ring attention by wrapping JAX's splash attention kernels
with a ring communication topology for distributed attention computation across
multiple TPU devices.

Ring attention enables processing of very long sequences by distributing the
sequence across devices and using a ring communication pattern to share KV
pairs. Each device holds a shard of the sequence and iteratively receives
KV data from neighboring devices, computing local attention and accumulating
results using numerically stable online softmax.

Algorithm overview:
1. Each device starts with its local Q, K, V shards
2. In each iteration:
   - Compute local attention using Splash Attention kernel
   - Merge results with running output using log-sum-exp correction
   - Send K, V to next device in the ring (ppermute)
3. After N iterations (N = ring size), all Q have seen all K, V

Key features:
- Memory-efficient: Only local K, V shards in memory at once
- Numerically stable: Uses log-sum-exp trick for softmax accumulation
- Supports causal attention across device boundaries
- Compatible with Splash Attention's block-sparse optimizations
- Enforces ring context for distributed execution

Example:
    >>> from ejkernel.kernels._pallas.tpu.ring_attention import make_ring_attention
    >>> from jax.experimental.shard_map import shard_map
    >>>
    >>> kernel = make_ring_attention(mask, block_sizes=block_sizes, is_mqa=False)
    >>> sharded_fn = shard_map(kernel, mesh, in_specs=..., out_specs=...)
    >>> output = sharded_fn(q, k, v)
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, tree_util

from ..blocksparse_attention import _info as mask_info_lib
from ..blocksparse_attention import _kernel as splash_kernel
from ..blocksparse_attention import _masks as mask_lib

partial = functools.partial

RING_AXIS = "sp"

MaskInfo = mask_info_lib.MaskInfo
BlockSizes = splash_kernel.BlockSizes
MaskFunctionType = splash_kernel.MaskFunctionType
DEFAULT_MASK_VALUE = splash_kernel.DEFAULT_MASK_VALUE


class SegmentIds(NamedTuple):
    """Segment IDs for query and KV sequences in packed-sequence attention.

    Enables document/sequence boundary masking where tokens with different
    segment IDs are masked from attending to each other. This is essential
    for packed training where multiple documents share the same batch slot.

    Attributes:
        q: Query segment IDs [q_seq_len]. Each value identifies which
            document/sequence the query token belongs to.
        kv: KV segment IDs [kv_seq_len]. Each value identifies which
            document/sequence the key/value token belongs to.
    """

    q: jax.Array
    kv: jax.Array


def _update_out_and_lse(
    out: jax.Array,
    lse: jax.Array,
    block_out: jax.Array,
    block_lse: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Numerically stable update of attention output and log-sum-exp.

    Combines the current accumulated output with a new block's output using
    the log-sum-exp trick to maintain numerical stability. This enables
    incremental computation of softmax across multiple blocks.

    The update formula uses:
    - new_lse = lse + log(1 + exp(block_lse - lse))
    - new_out = out + sigmoid(block_lse - lse) * (block_out - out)

    Args:
        out: Current accumulated output [num_heads, q_seq_len, head_dim].
        lse: Current log-sum-exp [num_heads, q_seq_len].
        block_out: New block's attention output.
        block_lse: New block's log-sum-exp.

    Returns:
        Tuple of (updated_out, updated_lse) with merged results.
    """
    is_first = lse == -jnp.inf
    block_lse_expanded = block_lse[..., None]
    lse_expanded = lse[..., None]
    sigmoid_weight = jax.nn.sigmoid(block_lse_expanded - lse_expanded)
    new_out = out - sigmoid_weight * (out - block_out)
    new_lse = lse + jax.nn.softplus(block_lse - lse)
    new_out = jnp.where(is_first[..., None], block_out, new_out)
    new_lse = jnp.where(is_first, block_lse, new_lse)
    return new_out, new_lse


def _ring_attention_forward(
    fwd_mask_info: MaskInfo,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    mask_function: MaskFunctionType | None,
    logits_soft_cap: float | None,
    ring_axis: str = RING_AXIS,
    causal: bool = False,
    sliding_window: int | tuple[int, int] | None = None,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    """Forward pass for ring attention over distributed devices.

    Executes a ring communication pattern where each device computes
    local attention and passes its K, V to the next device. Results are
    accumulated using numerically stable log-sum-exp merging.

    Args:
        fwd_mask_info: Mask information for Splash Attention.
        q: Local query shard [num_heads, q_seq_len, head_dim].
        k: Local key shard [kv_seq_len, head_dim] for MQA or [num_heads, kv_seq_len, head_dim].
        v: Local value shard with same shape as k.
        segment_ids: Optional segment IDs for segment masking.
        sinks: Optional attention sink logits.
        mask_value: Value for masked positions.
        is_mqa: Whether using multi-query attention (shared KV heads).
        block_sizes: Tile sizes for Splash Attention kernel.
        mask_function: Optional custom mask function.
        logits_soft_cap: Optional soft cap for logits.
        ring_axis: Name of the axis for ring communication.
        causal: Whether to apply causal masking across ring iterations.

    Returns:
        Tuple of (output, (lse, lse)) where lse is the final log-sum-exp
        for each query position.
    """
    ring_axis_size = lax.psum(1, ring_axis)
    device_idx = lax.axis_index(ring_axis)

    num_heads = q.shape[0]
    q_seq_len = q.shape[1]
    kv_seq_len = k.shape[0] if is_mqa else k.shape[1]

    shift = partial(
        lax.ppermute,
        axis_name=ring_axis,
        perm=[(i, (i + 1) % ring_axis_size) for i in range(ring_axis_size)],
    )

    o_shape = q.shape
    o_init = jnp.zeros(o_shape, dtype=jnp.float32)
    lse_init = jnp.full((num_heads, q_seq_len), -jnp.inf, dtype=jnp.float32)

    splash_segment_ids = None
    if segment_ids is not None:
        splash_segment_ids = splash_kernel.SegmentIds(q=segment_ids.q, kv=segment_ids.kv)

    needs_position_offset = causal or sliding_window is not None or mask_function is not None
    base_q_sequence = jnp.arange(q_seq_len, dtype=jnp.int32)

    if sliding_window is not None:
        if isinstance(sliding_window, int):
            sw_left, sw_right = sliding_window, sliding_window
        else:
            sw_left, sw_right = sliding_window

    def ring_mask_fn(q_ids, kv_ids):
        """Dynamic mask function for ring rotation combining causal + sliding window."""
        mask = jnp.ones((q_ids.shape[0], kv_ids.shape[1]), dtype=jnp.bool_)
        if causal:
            mask = mask & (q_ids >= kv_ids)
        if sliding_window is not None:
            mask = mask & (q_ids - sw_left <= kv_ids) & (q_ids + sw_right >= kv_ids)
        return mask

    def body(carry, iteration):
        o_prev, lse_prev, k_current, v_current, kv_source_device = carry
        k_next = shift(k_current)
        v_next = shift(v_current)
        is_first_iteration = iteration == 0

        if needs_position_offset:
            offset = device_idx * q_seq_len - kv_source_device * kv_seq_len
            modified_q_sequence = base_q_sequence + offset
            fwd_mask_info_iter = MaskInfo(
                data_next=None,
                mask_next=None,
                block_mask=None,
                partial_mask_blocks=None,
                q_sequence=modified_q_sequence,
            )
            mask_function_iter = ring_mask_fn
        else:
            fwd_mask_info_iter = fwd_mask_info
            mask_function_iter = mask_function

        sinks_iter = None
        if sinks is not None:
            sinks_iter = jnp.where(is_first_iteration, sinks, jnp.full_like(sinks, -1e9))

        out_curr, residuals = splash_kernel._splash_attention_forward(
            fwd_mask_info=fwd_mask_info_iter,
            q=q,
            k=k_current,
            v=v_current,
            segment_ids=splash_segment_ids,
            sinks=sinks_iter,
            mask_value=mask_value,
            is_mqa=is_mqa,
            block_sizes=block_sizes,
            residual_checkpoint_name=None,
            mask_function=mask_function_iter,
            save_residuals=True,
            logits_soft_cap=logits_soft_cap,
        )

        lse_curr = residuals[0].astype(jnp.float32)
        out_curr = out_curr.astype(jnp.float32)
        o_next, lse_next = _update_out_and_lse(o_prev, lse_prev, out_curr, lse_curr)
        kv_source_next = (kv_source_device - 1) % ring_axis_size

        return (o_next, lse_next, k_next, v_next, kv_source_next), None

    initial_kv_source = device_idx.astype(jnp.int32)
    initial_carry = (o_init, lse_init, k, v, initial_kv_source)
    (o_final, lse_final, _, _, _), _ = lax.scan(
        body,
        initial_carry,
        xs=jnp.arange(0, ring_axis_size),
        length=ring_axis_size,
        unroll=True,
    )

    out = o_final.astype(q.dtype)
    return out, (lse_final, lse_final)


def _ring_attention_backward(
    res: tuple,
    do: jax.Array,
    *,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    mask_function: MaskFunctionType | None,
    logits_soft_cap: float | None,
    ring_axis: str,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
    """Backward pass for ring attention with distributed gradient accumulation.

    Computes gradients for Q, K, V by running the ring communication pattern
    in reverse. Each device accumulates local dQ and rotates dK, dV gradients
    around the ring. Uses Splash Attention's backward kernel for local
    gradient computation.

    Args:
        res: Residuals from forward pass containing (q, k, v, segment_ids,
            sinks, out, logsumexp, fwd_mask_info, dq_mask_info, dkv_mask_info).
        do: Gradient of loss w.r.t. attention output.
        mask_value: Value for masked positions.
        is_mqa: Whether using multi-query attention.
        block_sizes: Tile sizes for kernel execution.
        mask_function: Optional custom mask function.
        logits_soft_cap: Optional soft cap for logits.
        ring_axis: Axis name for ring communication.

    Returns:
        Tuple of (dq, dk, dv, dsinks) gradients.
    """
    q, k, v, segment_ids, sinks, out, logsumexp, _fwd_mask_info, dq_mask_info, dkv_mask_info = res
    do_main = do.astype(jnp.float32)

    ring_axis_size = lax.psum(1, ring_axis)

    shift = partial(
        lax.ppermute,
        axis_name=ring_axis,
        perm=[(i, (i + 1) % ring_axis_size) for i in range(ring_axis_size)],
    )

    dq_accum = jnp.zeros_like(q, dtype=jnp.float32)
    dk_accum = jnp.zeros_like(k, dtype=jnp.float32)
    dv_accum = jnp.zeros_like(v, dtype=jnp.float32)
    dsinks_accum = None
    if sinks is not None:
        dsinks_accum = jnp.zeros_like(sinks, dtype=jnp.float32)

    splash_segment_ids = None
    if segment_ids is not None:
        splash_segment_ids = splash_kernel.SegmentIds(q=segment_ids.q, kv=segment_ids.kv)

    def body(carry, _: int):
        dq_accum, dk_accum, dv_accum, k_cur, v_cur, dsinks = carry
        k_next = shift(k_cur)
        v_next = shift(v_cur)

        residuals_for_chunk = (
            q,
            k_cur,
            v_cur,
            splash_segment_ids,
            sinks,
            out,
            logsumexp,
            dq_mask_info,
            dkv_mask_info,
        )

        grads = splash_kernel._splash_attention_bwd(
            save_residuals=False,
            mask_value=mask_value,
            is_mqa=is_mqa,
            block_sizes=block_sizes,
            residual_checkpoint_name=None,
            mask_function=mask_function,
            logits_soft_cap=logits_soft_cap,
            interpret=False,
            res=residuals_for_chunk,
            do=do_main,
        )

        dq_i = grads[3].astype(jnp.float32)
        dk_i = grads[4].astype(jnp.float32)
        dv_i = grads[5].astype(jnp.float32)
        dsinks_i = grads[7]

        dv_accum = dv_accum + dv_i
        dv_next = shift(dv_accum)
        dk_accum = dk_accum + dk_i
        dk_next = shift(dk_accum)
        dq_accum = dq_accum + dq_i

        if dsinks is not None and dsinks_i is not None:
            dsinks = dsinks + dsinks_i.astype(jnp.float32)

        return (dq_accum, dk_next, dv_next, k_next, v_next, dsinks), None

    initial_carry = (dq_accum, dk_accum, dv_accum, k, v, dsinks_accum)
    (dq_final, dk_final, dv_final, _, _, dsinks_final), _ = lax.scan(
        body,
        initial_carry,
        xs=jnp.arange(0, ring_axis_size),
        length=ring_axis_size,
        unroll=True,
    )

    if sinks is not None and dsinks_final is not None:
        dsinks_final = jax.lax.psum(dsinks_final, axis_name=ring_axis)

    dq_final = dq_final.astype(q.dtype)
    dk_final = dk_final.astype(k.dtype)
    dv_final = dv_final.astype(v.dtype)

    return dq_final, dk_final, dv_final, dsinks_final


def _ring_attention_fwd_rule(
    fwd_mask_info: MaskInfo,
    dq_mask_info: MaskInfo | None,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    *,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    mask_function: MaskFunctionType | None,
    logits_soft_cap: float | None,
    ring_axis: str = RING_AXIS,
    causal: bool = False,
    sliding_window: int | tuple[int, int] | None = None,
) -> tuple[jax.Array, tuple]:
    """Custom VJP forward rule for ring attention.

    Runs the ring attention forward pass and saves all tensors needed
    for the backward pass as residuals.

    Args:
        fwd_mask_info: Mask information for forward Splash Attention.
        dq_mask_info: Optional mask info for dQ backward pass.
        dkv_mask_info: Optional mask info for dKV backward pass.
        q: Query tensor (local shard).
        k: Key tensor (local shard).
        v: Value tensor (local shard).
        segment_ids: Optional segment IDs for packed sequences.
        sinks: Optional attention sink logits.
        mask_value: Value for masked positions.
        is_mqa: Whether using multi-query attention.
        block_sizes: Tile sizes for kernel execution.
        mask_function: Optional custom mask function.
        logits_soft_cap: Optional soft cap for logits.
        ring_axis: Axis name for ring communication.
        causal: Whether to apply causal masking.

    Returns:
        Tuple of (output, residuals_for_backward).
    """
    out, (logsumexp, _) = _ring_attention_forward(
        fwd_mask_info,
        q,
        k,
        v,
        segment_ids,
        sinks=sinks,
        mask_value=mask_value,
        is_mqa=is_mqa,
        block_sizes=block_sizes,
        mask_function=mask_function,
        logits_soft_cap=logits_soft_cap,
        ring_axis=ring_axis,
        causal=causal,
        sliding_window=sliding_window,
    )
    residuals_for_bwd = (
        q,
        k,
        v,
        segment_ids,
        sinks,
        out,
        logsumexp,
        fwd_mask_info,
        dq_mask_info,
        dkv_mask_info,
    )
    return out, residuals_for_bwd


def _ring_attention_bwd_rule(
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    mask_function: MaskFunctionType | None,
    logits_soft_cap: float | None,
    ring_axis: str,
    res: tuple,
    do: jax.Array,
):
    """Custom VJP backward rule for ring attention.

    Dispatches to _ring_attention_backward and formats the gradient
    outputs to match the custom_vjp signature. Returns None for
    non-differentiable inputs (mask infos, segment_ids).

    Args:
        mask_value: Value for masked positions.
        is_mqa: Whether using multi-query attention.
        block_sizes: Tile sizes for kernel execution.
        mask_function: Optional custom mask function.
        logits_soft_cap: Optional soft cap for logits.
        ring_axis: Axis name for ring communication.
        res: Residuals saved from forward pass.
        do: Gradient of loss w.r.t. output.

    Returns:
        Tuple of gradients matching the forward signature.
    """
    dq, dk, dv, dsinks = _ring_attention_backward(
        res,
        do,
        mask_value=mask_value,
        is_mqa=is_mqa,
        block_sizes=block_sizes,
        mask_function=mask_function,
        logits_soft_cap=logits_soft_cap,
        ring_axis=ring_axis,
    )
    return (None, None, None, dq, dk, dv, None, dsinks)


@partial(
    jax.custom_vjp,
    nondiff_argnums=(8, 9, 10, 11, 12, 13, 14, 15),
)
def _ring_attention_custom(
    fwd_mask_info: MaskInfo,
    dq_mask_info: MaskInfo | None,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    mask_function: MaskFunctionType | None,
    logits_soft_cap: float | None,
    ring_axis: str = RING_AXIS,
    causal: bool = False,
    sliding_window: int | tuple[int, int] | None = None,
) -> jax.Array:
    """Ring attention with custom VJP for efficient gradient computation.

    Wraps the ring attention forward pass with a JAX custom_vjp to enable
    explicit control over backward pass gradient computation, ensuring
    proper ring communication during backpropagation.

    Args:
        fwd_mask_info: Mask information for forward pass.
        dq_mask_info: Optional mask info for dQ backward.
        dkv_mask_info: Optional mask info for dKV backward.
        q: Query tensor (local shard).
        k: Key tensor (local shard).
        v: Value tensor (local shard).
        segment_ids: Optional segment IDs.
        sinks: Optional attention sink logits.
        mask_value: Value for masked positions.
        is_mqa: Whether using multi-query attention.
        block_sizes: Tile sizes for kernel.
        mask_function: Optional custom mask function.
        logits_soft_cap: Optional soft cap for logits.
        ring_axis: Ring communication axis name.
        causal: Whether to apply causal masking.

    Returns:
        Attention output tensor.
    """
    out, _ = _ring_attention_forward(
        fwd_mask_info,
        q,
        k,
        v,
        segment_ids,
        sinks=sinks,
        mask_value=mask_value,
        is_mqa=is_mqa,
        block_sizes=block_sizes,
        mask_function=mask_function,
        logits_soft_cap=logits_soft_cap,
        ring_axis=ring_axis,
        causal=causal,
        sliding_window=sliding_window,
    )
    return out


def _ring_attention_custom_fwd(
    fwd_mask_info: MaskInfo,
    dq_mask_info: MaskInfo | None,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None,
    sinks: jax.Array | None,
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    mask_function: MaskFunctionType | None,
    logits_soft_cap: float | None,
    ring_axis: str = RING_AXIS,
    causal: bool = False,
    sliding_window: int | tuple[int, int] | None = None,
):
    """VJP forward implementation for _ring_attention_custom."""
    return _ring_attention_fwd_rule(
        fwd_mask_info,
        dq_mask_info,
        dkv_mask_info,
        q,
        k,
        v,
        segment_ids,
        sinks,
        mask_value=mask_value,
        is_mqa=is_mqa,
        block_sizes=block_sizes,
        mask_function=mask_function,
        logits_soft_cap=logits_soft_cap,
        ring_axis=ring_axis,
        causal=causal,
        sliding_window=sliding_window,
    )


def _ring_attention_custom_bwd(
    mask_value: float,
    is_mqa: bool,
    block_sizes: BlockSizes,
    mask_function: MaskFunctionType | None,
    logits_soft_cap: float | None,
    ring_axis: str,
    causal: bool,
    sliding_window: int | tuple[int, int] | None,
    res: tuple,
    do: jax.Array,
):
    """VJP backward implementation for _ring_attention_custom.

    Delegates to _ring_attention_bwd_rule to compute gradients using
    the ring communication pattern.
    """
    return _ring_attention_bwd_rule(
        mask_value=mask_value,
        is_mqa=is_mqa,
        block_sizes=block_sizes,
        mask_function=mask_function,
        logits_soft_cap=logits_soft_cap,
        ring_axis=ring_axis,
        res=res,
        do=do,
    )


_ring_attention_custom.defvjp(_ring_attention_custom_fwd, _ring_attention_custom_bwd)


def _has_axis(axis_name: str) -> bool:
    """Check whether a named axis exists in the current JAX context.

    Attempts a collective operation to determine if the axis name is
    defined (e.g., inside shard_map or pmap). Used to enforce that ring
    execution runs under the requested distributed axis.

    Args:
        axis_name: Name of the axis to check.

    Returns:
        True if the axis exists, False otherwise.
    """
    try:
        lax.psum(1, axis_name)
        return True
    except (NameError, ValueError):
        return False


@partial(
    jax.jit,
    static_argnames=[
        "is_mqa",
        "block_sizes",
        "mask_value",
        "mask_function",
        "logits_soft_cap",
        "ring_axis",
        "causal",
        "sliding_window",
    ],
)
def ring_splash_attention(
    fwd_mask_info: MaskInfo,
    dkv_mask_info: MaskInfo | None,
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    segment_ids: SegmentIds | None = None,
    sinks: jax.Array | None = None,
    *,
    is_mqa: bool,
    block_sizes: BlockSizes,
    mask_value: float = DEFAULT_MASK_VALUE,
    mask_function: MaskFunctionType | None = None,
    logits_soft_cap: float | None = None,
    ring_axis: str = RING_AXIS,
    causal: bool = False,
    sliding_window: int | tuple[int, int] | None = None,
) -> jax.Array:
    """Compute ring attention using Splash Attention kernels.

    Main entry point for ring attention computation. Requires a distributed
    context (``shard_map``/``pmap``) with the requested ring axis.

    Args:
        fwd_mask_info: Pre-computed mask info for forward pass.
        dkv_mask_info: Optional mask info for dKV backward pass.
        q: Query tensor, local shard if distributed.
        k: Key tensor, local shard if distributed.
        v: Value tensor, local shard if distributed.
        segment_ids: Optional segment IDs for packed sequence masking.
        sinks: Optional attention sink logits for StreamingLLM-style attention.
        is_mqa: Whether using multi-query attention (shared KV across heads).
        block_sizes: Tile sizes for kernel execution.
        mask_value: Value for masked attention positions.
        mask_function: Optional custom mask function.
        logits_soft_cap: Optional soft cap for attention logits.
        ring_axis: Axis name for ring communication (default: "sp").
        causal: Whether to apply causal masking.

    Returns:
        Attention output tensor with same shape as query.
    """
    dq_mask_info = fwd_mask_info if block_sizes.has_backward_blocks else None

    if not _has_axis(ring_axis):
        raise NotImplementedError(
            "ring_splash_attention requires execution under shard_map/pmap with "
            f"axis '{ring_axis}'. Fallback paths are disabled."
        )

    return _ring_attention_custom(
        fwd_mask_info,
        dq_mask_info,
        dkv_mask_info,
        q,
        k,
        v,
        segment_ids,
        sinks,
        mask_value,
        is_mqa,
        block_sizes,
        mask_function,
        logits_soft_cap,
        ring_axis,
        causal,
        sliding_window,
    )


@jax.tree_util.register_pytree_node_class
class RingSplashAttentionKernel:
    """Callable ring attention kernel with pre-computed mask information.

    A JAX-pytree-compatible wrapper around ring_splash_attention that
    stores pre-processed mask information and kernel configuration.
    Created by make_ring_attention() and can be used directly as a
    callable or passed through JAX transformations.

    Attributes:
        fwd_mask_info: Pre-computed mask info for forward pass.
        dkv_mask_info: Optional pre-computed mask info for dKV backward.
        ring_axis: Axis name for ring communication.
        kwargs: Additional keyword arguments passed to ring_splash_attention.
    """

    def __init__(
        self,
        fwd_mask_info: MaskInfo,
        dkv_mask_info: MaskInfo | None,
        ring_axis: str = RING_AXIS,
        **kwargs,
    ):
        """Initialize the ring splash attention kernel.

        Args:
            fwd_mask_info: Pre-computed mask information for forward pass.
            dkv_mask_info: Optional mask info for dKV backward pass.
            ring_axis: Axis name for ring communication (default: "sp").
            **kwargs: Additional arguments forwarded to ring_splash_attention
                (e.g., is_mqa, block_sizes, mask_value, logits_soft_cap).
        """
        self.fwd_mask_info = fwd_mask_info
        self.dkv_mask_info = dkv_mask_info
        self.ring_axis = ring_axis
        self.kwargs = kwargs

    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        segment_ids: SegmentIds | None = None,
        sinks: jax.Array | None = None,
    ) -> jax.Array:
        """Compute ring attention with the pre-configured mask and parameters.

        Args:
            q: Query tensor [num_heads, q_seq_len, head_dim].
            k: Key tensor [kv_seq_len, head_dim] (MQA) or [num_heads, kv_seq_len, head_dim].
            v: Value tensor with same shape as k.
            segment_ids: Optional segment IDs for packed-sequence masking.
            sinks: Optional attention sink logits.

        Returns:
            Attention output with same shape as q.
        """
        return ring_splash_attention(
            self.fwd_mask_info,
            self.dkv_mask_info,
            q,
            k,
            v,
            segment_ids=segment_ids,
            sinks=sinks,
            ring_axis=self.ring_axis,
            **self.kwargs,
        )

    def tree_flatten(self):
        """Flatten for JAX pytree serialization.

        Returns:
            Tuple of (children, aux_data) where children contains the
            mask info arrays and aux_data contains configuration.
        """
        children = (self.fwd_mask_info, self.dkv_mask_info)
        aux_data = {"ring_axis": self.ring_axis, **self.kwargs}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Reconstruct from JAX pytree serialization.

        Args:
            aux_data: Configuration dict with ring_axis and kwargs.
            children: Tuple of (fwd_mask_info, dkv_mask_info).

        Returns:
            Reconstructed RingSplashAttentionKernel instance.
        """
        fwd_mask_info, dkv_mask_info = children
        if isinstance(fwd_mask_info, tuple):
            fwd_mask_info = MaskInfo(*fwd_mask_info)
        if dkv_mask_info is not None and isinstance(dkv_mask_info, tuple):
            dkv_mask_info = MaskInfo(*dkv_mask_info)
        return cls(fwd_mask_info, dkv_mask_info, **aux_data)


def make_ring_attention(
    mask: np.ndarray | jax.Array | mask_lib.Mask,
    *,
    block_sizes: BlockSizes | None = None,
    is_mqa: bool = False,
    mask_value: float = DEFAULT_MASK_VALUE,
    logits_soft_cap: float | None = None,
    ring_axis: str = RING_AXIS,
    q_seq_shards: int = 1,
) -> RingSplashAttentionKernel:
    """Create a ring attention kernel from an attention mask.

    Factory function that preprocesses the mask and creates a callable
    kernel object for ring attention. The kernel can be used directly
    or wrapped in shard_map for distributed execution.

    Args:
        mask: Attention mask as numpy array, JAX array, or Mask object.
            Shape should be [q_seq_len, kv_seq_len].
        block_sizes: Optional tile sizes for kernel. Uses defaults if None.
        is_mqa: Whether using multi-query attention.
        mask_value: Value for masked positions in attention.
        logits_soft_cap: Optional soft cap for attention logits.
        ring_axis: Axis name for ring communication pattern.
        q_seq_shards: Number of shards for query sequence (for mask processing).

    Returns:
        RingSplashAttentionKernel that can be called with (q, k, v) tensors.

    Example:
        >>> mask = make_causal_mask((seq_len, seq_len))
        >>> kernel = make_ring_attention(mask, is_mqa=False)
        >>> output = kernel(q, k, v)
    """
    if len(mask.shape) != 2:
        raise ValueError(f"Expected 2D mask, got shape: {mask.shape}")

    if isinstance(mask, np.ndarray):
        mask = mask_lib.NumpyMask(mask)

    if block_sizes is None:
        block_sizes = BlockSizes.get_default()

    multi_head_mask = mask_lib.MultiHeadMask(masks=(mask,))

    fwd_mask_info, mask_function = mask_info_lib._process_mask(
        multi_head_mask,
        (block_sizes.block_q, block_sizes.block_kv),
        is_dkv=False,
        q_seq_shards=q_seq_shards,
    )
    fwd_mask_info = tree_util.tree_map(jnp.array, fwd_mask_info)

    dkv_mask_info = None
    if block_sizes.has_backward_blocks:
        dkv_mask_info, _ = mask_info_lib._process_mask(
            multi_head_mask,
            (block_sizes.block_q_dkv, block_sizes.block_kv_dkv),
            is_dkv=True,
        )
        dkv_mask_info = tree_util.tree_map(jnp.array, dkv_mask_info)

    return RingSplashAttentionKernel(
        fwd_mask_info,
        dkv_mask_info,
        ring_axis=ring_axis,
        is_mqa=is_mqa,
        block_sizes=block_sizes,
        mask_value=mask_value,
        mask_function=mask_function,
        logits_soft_cap=logits_soft_cap,
    )
