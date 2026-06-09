# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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


"""Flash Attention forward pass TPU kernel implementation using Pallas.

This module provides an optimized Flash Attention forward pass implementation
for Google TPUs using the Pallas kernel framework. The implementation uses
tiled matrix operations and online softmax computation to achieve
memory-efficient attention with O(N) memory complexity.

Flash Attention is an IO-aware exact attention algorithm that dramatically
reduces memory usage by never materializing the full attention matrix.
Instead, it computes attention block-by-block, maintaining running statistics
for numerically stable softmax computation.

Algorithm overview:
1. Split Q, K, V into blocks of configurable sizes
2. For each query block, iterate over key-value blocks
3. Compute block-wise attention scores and apply softmax incrementally
4. Accumulate weighted values using online softmax correction factors
5. Output final attention values with proper normalization

TPU-specific optimizations:
- Uses Pallas BlockSpecs for efficient memory prefetching
- Leverages TPU VMEM scratch space for intermediate results
- Supports parallel execution across batch, head, and sequence dimensions
- Optimized for TPU MXU (Matrix Multiply Unit) tile sizes

Supported features:
- Causal and non-causal attention patterns
- Attention bias tensors
- Segment IDs for packed sequences
- Sliding window attention
- Logits soft capping for numerical stability
- Configurable block sizes for performance tuning

Example:
    >>> from ejkernel.kernels._pallas.tpu.flash_attention import _flash_attention_impl
    >>> output, l, m = _flash_attention_impl(
    ...     q, k, v, ab=None, segment_ids=None,
    ...     save_residuals=True, causal=True,
    ...     softmax_scale=1.0/8.0,  # 1/sqrt(head_dim)
    ...     sliding_window=None, logits_soft_cap=None,
    ...     block_b=1, block_q=128, block_k_major=128, block_k=128
    ... )
"""

from __future__ import annotations

import functools
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ._utils import (
    DEFAULT_MASK_VALUE,
    MIN_BLOCK_SIZE,
    NUM_LANES,
    NUM_SUBLANES,
    TRANS_B_DIM_NUMBERS,
    _fwd_cost_estimate,
    _verify_block,
    below_or_on_diag,
)


def _flash_attention_fwd(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    softmax_scale,
    block_sizes,
    sliding_window,
    logits_soft_cap,
):
    """Forward pass entry point for Flash Attention with residual saving.

    This function wraps _flash_attention_impl to compute the forward pass
    and package the residuals needed for backward pass gradient computation.
    The residuals include the input tensors and intermediate softmax statistics.

    Args:
        q: Query tensor of shape [batch, num_heads, q_seq_len, head_dim].
        k: Key tensor of shape [batch, num_heads, kv_seq_len, head_dim].
        v: Value tensor of shape [batch, num_heads, kv_seq_len, head_dim].
        ab: Optional attention bias of shape [batch, num_heads, q_seq_len, kv_seq_len].
        segment_ids: Optional SegmentIds for packed sequence masking.
        save_residuals: Whether to save residuals for backward pass.
        causal: Whether to apply causal (autoregressive) masking.
        softmax_scale: Scaling factor for attention logits (typically 1/sqrt(head_dim)).
        block_sizes: BlockSizes instance specifying tile dimensions.
        sliding_window: Optional tuple (left, right) for sliding window attention.
        logits_soft_cap: Optional soft cap value for attention logits.

    Returns:
        Tuple of (output, residuals) where:
        - output: Attention output of shape [batch, num_heads, q_seq_len, head_dim]
        - residuals: Tuple containing (q, k, v, ab, segment_ids, output, l, m) for
          backward pass computation
    """
    o, l, m = _flash_attention_impl(
        q,
        k,
        v,
        ab,
        segment_ids,
        True,
        causal,
        softmax_scale,
        sliding_window,
        logits_soft_cap,
        block_sizes.block_b,
        block_sizes.block_q,
        block_sizes.block_k_major,
        block_sizes.block_k,
    )
    return o, (q, k, v, ab, segment_ids, o, l, m)


def _flash_attention_kernel(q_tile_ref, *args, **kwargs):
    """Pallas kernel dispatcher for Flash Attention forward pass.

    This kernel function dispatches to the appropriate implementation based on
    whether the KV sequence fits in a single step or requires multiple iterations.
    It iterates over the batch dimension within each tile.

    Args:
        q_tile_ref: Reference to the query tile in TPU memory.
        *args: Additional tile references (k, v, ab, segment_ids, outputs, scratch).
        **kwargs: Kernel configuration including block_k, kv_seq_len, causal, etc.

    Note:
        This function is called by Pallas for each grid position. The kernel
        automatically selects between single-step (when kv_seq_len == block_k)
        and multi-step implementations for optimal performance.
    """
    block_b = q_tile_ref.shape[0]

    if kwargs["block_k"] == kwargs["kv_seq_len"]:
        kernel = _flash_attention_kernel_single_batch_single_step
    else:
        kernel = _flash_attention_kernel_single_batch
    for batch_idx in range(block_b):
        kernel((batch_idx, 0), q_tile_ref, *args, **kwargs)


def _flash_attention_kernel_single_batch(
    batch_idx: tuple[int, ...],
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,
    o_tile_ref,
    l_ref,
    m_ref,
    m_scratch_ref,
    l_scratch_ref,
    acc_scratch_ref,
    *,
    causal,
    softmax_scale,
    sliding_window,
    logits_soft_cap,
    block_k,
    kv_seq_len,
    mask_value,
):
    """Multi-step Flash Attention kernel for a single batch element.

    This kernel computes attention for sequences where the KV length exceeds
    a single block, requiring multiple iterations over KV blocks with online
    softmax computation to maintain numerical stability.

    The algorithm maintains running statistics (m for max, l for sum) in scratch
    memory and incrementally updates the output accumulator as it processes
    each KV block.

    Args:
        batch_idx: Current batch index within the tile.
        q_tile_ref: Query tile reference [block_b, 1, block_q, head_dim].
        k_tile_ref: Key tile reference [block_b, 1, block_k_major, head_dim].
        v_tile_ref: Value tile reference [block_b, 1, block_k_major, head_dim].
        ab_tile_ref: Optional attention bias tile reference.
        q_segment_ids_tile_ref: Optional query segment IDs for packed sequences.
        kv_segment_ids_tile_ref: Optional KV segment IDs for packed sequences.
        o_tile_ref: Output tile reference to write attention results.
        l_ref: Reference to store softmax denominator (sum of exp).
        m_ref: Reference to store softmax numerator stability term (max).
        m_scratch_ref: VMEM scratch for running max values.
        l_scratch_ref: VMEM scratch for running sum values.
        acc_scratch_ref: VMEM scratch for output accumulation.
        causal: Whether to apply causal masking.
        softmax_scale: Scaling factor for QK^T scores.
        sliding_window: Optional (left, right) window sizes for local attention.
        logits_soft_cap: Optional soft capping for logits.
        block_k: Inner loop KV block size.
        kv_seq_len: Total KV sequence length.
        mask_value: Value to use for masked positions (typically large negative).
    """
    block_k_major = k_tile_ref.shape[2]
    block_q = q_tile_ref.shape[2]
    head_dim = q_tile_ref.shape[-1]

    kv_seq_idx = pl.program_id(3)

    @pl.when(kv_seq_idx == 0)
    def start_new_sequence():
        m_scratch_ref[batch_idx] = jnp.full(m_scratch_ref.shape[2:], -jnp.inf, jnp.float32)
        l_scratch_ref[batch_idx] = jnp.zeros(l_scratch_ref.shape[2:], jnp.float32)
        acc_scratch_ref[batch_idx] = jnp.zeros(acc_scratch_ref.shape[2:], jnp.float32)

    q_seq_idx = pl.program_id(2)
    if causal:
        should_run = below_or_on_diag(q_seq_idx, block_q, kv_seq_idx, block_k_major)
    else:
        should_run = True

    @pl.when(should_run)
    def run():
        @pl.loop(0, block_k_major, step=block_k, unroll=True)
        def _body(start_k):
            m_prev = m_scratch_ref[batch_idx]
            l_prev = l_scratch_ref[batch_idx]
            q = q_tile_ref[batch_idx]
            k = k_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]

            s = jax.lax.dot_general(q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)

            if ab_tile_ref is not None:
                ab = ab_tile_ref[(*batch_idx, pl.dslice(None), pl.dslice(start_k, block_k))].astype(jnp.float32)
                s += ab

            if softmax_scale != 1.0:
                s *= softmax_scale

            if logits_soft_cap is not None:
                s = logits_soft_cap * jnp.tanh(s / logits_soft_cap)

            mask = None
            if q_segment_ids_tile_ref is not None:
                repeats, rem = divmod(block_k, NUM_LANES)
                if rem:
                    raise NotImplementedError(f"kv block size must be a multiple of {NUM_LANES}")
                q_segment_ids = pltpu.repeat(q_segment_ids_tile_ref[batch_idx[0]], repeats, axis=1)
                kv_segment_ids = kv_segment_ids_tile_ref[batch_idx[0], :1, pl.dslice(start_k, block_k)]
                mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

            if causal:
                mask_shape = (block_q, block_k)
                row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
                row_ids += q_seq_idx * block_q
                col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
                col_ids += kv_seq_idx * block_k_major + start_k
                causal_mask = col_ids <= row_ids
                mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

            if sliding_window is not None:
                window_left, window_right = sliding_window
                mask_shape = (block_q, block_k)
                row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
                row_ids += q_seq_idx * block_q
                col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
                col_ids += kv_seq_idx * block_k_major + start_k
                window_mask = (col_ids >= (row_ids - window_left)) & (col_ids <= (row_ids + window_right))
                mask = window_mask if mask is None else jnp.logical_and(mask, window_mask)

            s = s if mask is None else s + jnp.where(mask, 0.0, mask_value)

            m_curr = jnp.max(s, axis=1)[:, None]
            m_next = jnp.maximum(m_prev, m_curr)

            block_k_repeats, rem = divmod(block_k, MIN_BLOCK_SIZE)
            if rem:
                raise NotImplementedError(f"{block_k=} should be a multiple of {MIN_BLOCK_SIZE}")
            p = jnp.exp(s - pltpu.repeat(m_next, block_k_repeats, 1))

            alpha = jnp.exp(m_prev - m_next)

            l_corr = alpha * l_prev

            l_next = jnp.sum(p, axis=1)[:, None] + l_corr

            head_dim_repeats, rem = divmod(head_dim, MIN_BLOCK_SIZE)

            def l_broadcast(l):
                return pltpu.repeat(l, head_dim_repeats, 1)

            if rem:
                if head_dim_repeats == 0:

                    def l_broadcast(l):
                        return l[:, :head_dim]

                else:
                    raise NotImplementedError(f"{head_dim=} should be a multiple of {MIN_BLOCK_SIZE} if larger")
            l_scratch_ref[batch_idx] = l_next
            m_scratch_ref[batch_idx] = m_next

            l_next_inv_safe = jnp.where(l_next == 0.0, 1.0, 1.0 / l_next)
            acc_scratch_ref[batch_idx] *= l_broadcast(l_corr * l_next_inv_safe)
            v = v_tile_ref[(*batch_idx, pl.dslice(start_k, block_k), slice(None))]
            o_curr = jax.lax.dot(p.astype(v.dtype), v, preferred_element_type=jnp.float32)
            acc_scratch_ref[batch_idx] += o_curr * l_broadcast(l_next_inv_safe)

    @pl.when(kv_seq_idx == (kv_seq_len // block_k_major) - 1)
    def store_output():
        o_tile_ref[batch_idx] = acc_scratch_ref[batch_idx].astype(o_tile_ref.dtype)
        if l_ref is not None:
            l_ref[batch_idx] = l_scratch_ref[batch_idx].astype(l_ref.dtype)
        if m_ref is not None:
            m_ref[batch_idx] = m_scratch_ref[batch_idx].astype(m_ref.dtype)


def _flash_attention_kernel_single_batch_single_step(
    batch_idx: tuple[int, ...],
    q_tile_ref,
    k_tile_ref,
    v_tile_ref,
    ab_tile_ref,
    q_segment_ids_tile_ref,
    kv_segment_ids_tile_ref,
    o_tile_ref,
    l_ref: Any | None = None,
    m_ref: Any | None = None,
    *,
    causal,
    softmax_scale,
    sliding_window,
    logits_soft_cap,
    block_k,
    kv_seq_len,
    mask_value,
):
    """Single-step Flash Attention kernel for short KV sequences.

    Optimized kernel path for when the entire KV sequence fits in a single
    block (kv_seq_len == block_k). This avoids the overhead of online softmax
    accumulation since all attention weights can be computed in one pass.

    Args:
        batch_idx: Current batch index within the tile.
        q_tile_ref: Query tile reference [block_b, 1, block_q, head_dim].
        k_tile_ref: Key tile reference [block_b, 1, block_k, head_dim].
        v_tile_ref: Value tile reference [block_b, 1, block_k, head_dim].
        ab_tile_ref: Optional attention bias tile reference.
        q_segment_ids_tile_ref: Optional query segment IDs.
        kv_segment_ids_tile_ref: Optional KV segment IDs.
        o_tile_ref: Output tile reference to write attention results.
        l_ref: Optional reference to store softmax denominator.
        m_ref: Optional reference to store softmax max values.
        causal: Whether to apply causal masking.
        softmax_scale: Scaling factor for QK^T scores.
        sliding_window: Optional (left, right) window sizes.
        logits_soft_cap: Optional soft capping for logits.
        block_k: KV block size (equals kv_seq_len in this path).
        kv_seq_len: Total KV sequence length.
        mask_value: Value for masked positions.
    """
    block_k_major = k_tile_ref.shape[2]
    block_q = q_tile_ref.shape[2]

    assert kv_seq_len == block_k_major == block_k

    q = q_tile_ref[batch_idx]
    k = k_tile_ref[batch_idx]
    s = jax.lax.dot_general(q, k, TRANS_B_DIM_NUMBERS, preferred_element_type=jnp.float32)

    if ab_tile_ref is not None:
        s += ab_tile_ref[batch_idx].astype(jnp.float32)
    if softmax_scale != 1.0:
        s *= softmax_scale

    if logits_soft_cap is not None:
        s = logits_soft_cap * jnp.tanh(s / logits_soft_cap)

    mask = None
    if q_segment_ids_tile_ref is not None:
        repeats, rem = divmod(block_k, NUM_LANES)
        if rem:
            raise NotImplementedError(f"kv block size must be a multiple of {NUM_LANES}")
        q_segment_ids = q_segment_ids_tile_ref[batch_idx[0]]
        q_segment_ids = pltpu.repeat(q_segment_ids, repeats, axis=1)
        kv_segment_ids = kv_segment_ids_tile_ref[batch_idx[0], :1]
        mask = jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

    if causal:
        q_seq_idx = pl.program_id(2)
        mask_shape = (block_q, block_k)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        row_ids += q_seq_idx * block_q
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        causal_mask = col_ids <= row_ids
        mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)

    if sliding_window is not None:
        window_left, window_right = sliding_window
        q_seq_idx = pl.program_id(2)
        mask_shape = (block_q, block_k)
        row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
        row_ids += q_seq_idx * block_q
        col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
        window_mask = (col_ids >= (row_ids - window_left)) & (col_ids <= (row_ids + window_right))
        mask = window_mask if mask is None else jnp.logical_and(mask, window_mask)
    s = s if mask is None else s + jnp.where(mask, 0.0, mask_value)

    m = jnp.max(s, axis=1)[:, None]
    p = jnp.exp(s - m)
    l = jnp.sum(p, axis=1)[:, None]
    p /= l

    if m_ref is not None:
        m_ref[batch_idx] = lax.broadcast_in_dim(m, m_ref.shape[2:], range(2))
    if l_ref is not None:
        l_ref[batch_idx] = lax.broadcast_in_dim(l, l_ref.shape[2:], range(2))

    v = v_tile_ref[batch_idx]
    o_tile_ref[batch_idx] = jax.lax.dot(p.astype(v.dtype), v, preferred_element_type=jnp.float32).astype(
        o_tile_ref.dtype
    )


def _flash_attention_impl(
    q,
    k,
    v,
    ab,
    segment_ids,
    save_residuals,
    causal,
    softmax_scale,
    sliding_window,
    logits_soft_cap,
    block_b,
    block_q,
    block_k_major,
    block_k,
):
    """Core Flash Attention implementation using Pallas TPU kernels.

    This function sets up and executes the Pallas kernel for Flash Attention
    computation. It handles grid configuration, BlockSpec definitions for
    memory prefetching, and scratch memory allocation for intermediate results.

    The implementation uses a 4D grid: (batch_blocks, num_heads, q_blocks, kv_blocks)
    with parallel execution over the first three dimensions and sequential
    iteration over the KV dimension.

    Args:
        q: Query tensor [batch, num_heads, q_seq_len, head_dim].
        k: Key tensor [batch, num_heads, kv_seq_len, head_dim].
        v: Value tensor [batch, num_heads, kv_seq_len, head_dim].
        ab: Optional attention bias [batch, num_heads, q_seq_len, kv_seq_len].
        segment_ids: Optional SegmentIds for packed sequence masking.
        save_residuals: Whether to return l (sum) and m (max) for backward pass.
        causal: Whether to apply causal attention masking.
        softmax_scale: Scaling factor for attention scores.
        sliding_window: Optional (left, right) tuple for sliding window attention.
        logits_soft_cap: Optional soft cap value for logits stability.
        block_b: Batch tile size.
        block_q: Query sequence tile size.
        block_k_major: Major KV sequence tile size for memory prefetching.
        block_k: Inner KV block size for computation.

    Returns:
        If save_residuals is True:
            Tuple of (output, l, m) where l and m are softmax statistics.
        Otherwise:
            Just the output tensor.

    Raises:
        ValueError: If block sizes don't properly divide sequence dimensions.
    """
    batch_size, num_heads, q_seq_len, head_dim = q.shape
    _, _, kv_seq_len, _ = k.shape
    _verify_block("block_q", "q_seq_len", block_q, q_seq_len, should_divide=False)
    _verify_block("block_k_major", "kv_seq_len", block_k_major, kv_seq_len)
    _verify_block("block_k", "kv_seq_len", block_k, kv_seq_len)
    _verify_block("block_b", "batch", block_b, batch_size, should_divide=False)

    grid = (
        pl.cdiv(batch_size, block_b),
        num_heads,
        pl.cdiv(q_seq_len, block_q),
        kv_seq_len // block_k_major,
    )

    def q_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    def kv_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        if causal:
            next_kv_index = lax.select(
                below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k_major),
                kv_seq_index,
                0,
            )
        else:
            next_kv_index = kv_seq_index
        return (batch_index, head_index, next_kv_index, 0)

    def ab_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
        if causal:
            should_run = below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k_major)

            next_q_index = lax.select(
                should_run,
                q_seq_index,
                lax.select(q_seq_index == (q_seq_len // block_q) - 1, 0, q_seq_index + 1),
            )
            next_kv_index = lax.select(should_run, kv_seq_index, 0)
        else:
            next_q_index = q_seq_index
            next_kv_index = kv_seq_index

        return (batch_index, head_index, next_q_index, next_kv_index)

    def o_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    def lm_index_map(batch_index, head_index, q_seq_index, _):
        return (batch_index, head_index, q_seq_index, 0)

    kernel = functools.partial(
        _flash_attention_kernel,
        causal=causal,
        mask_value=DEFAULT_MASK_VALUE,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        block_k=block_k,
        kv_seq_len=kv_seq_len,
    )
    out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
    out_shape = [out_shape]
    out_specs = [pl.BlockSpec((block_b, 1, block_q, head_dim), o_index_map)]

    if block_k != kv_seq_len:
        m_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
        l_scratch = pltpu.VMEM((block_b, 1, block_q, MIN_BLOCK_SIZE), jnp.float32)
        acc_scratch = pltpu.VMEM((block_b, 1, block_q, head_dim), jnp.float32)
        scratch_shapes = [m_scratch, l_scratch, acc_scratch]
    else:
        scratch_shapes = []

    if save_residuals:
        out_specs = [
            *out_specs,
            pl.BlockSpec((block_b, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
            pl.BlockSpec((block_b, 1, block_q, MIN_BLOCK_SIZE), lm_index_map),
        ]
        l = jax.ShapeDtypeStruct((batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32)
        m = jax.ShapeDtypeStruct((batch_size, num_heads, q_seq_len, MIN_BLOCK_SIZE), dtype=jnp.float32)
        out_shape = (*out_shape, l, m)
    else:
        out_specs = [*out_specs, None, None]
        out_shape = (*out_shape, None, None)

    ab_block_spec = pl.BlockSpec((block_b, 1, block_q, block_k_major), ab_index_map) if ab is not None else None

    q_segment_ids_spec = kv_segment_ids_spec = None
    q_segment_ids = kv_segment_ids = None
    if segment_ids is not None:

        def q_segment_ids_index_map(batch_index, head_index, q_seq_index, _):
            del head_index
            return (batch_index, q_seq_index, 0)

        def kv_segment_ids_index_map(batch_index, head_index, q_seq_index, kv_seq_index):
            del head_index
            if causal:
                next_kv_index = lax.select(
                    below_or_on_diag(q_seq_index, block_q, kv_seq_index, block_k_major),
                    kv_seq_index,
                    0,
                )
            else:
                next_kv_index = kv_seq_index
            return (batch_index, 0, next_kv_index)

        q_segment_ids_spec = pl.BlockSpec((block_b, block_q, NUM_LANES), q_segment_ids_index_map)
        kv_segment_ids_spec = pl.BlockSpec((block_b, NUM_SUBLANES, block_k_major), kv_segment_ids_index_map)

        q_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.q,
            (batch_size, q_seq_len, NUM_LANES),
            (
                0,
                1,
            ),
        )
        kv_segment_ids = jax.lax.broadcast_in_dim(
            segment_ids.kv,
            (batch_size, NUM_SUBLANES, kv_seq_len),
            (
                0,
                2,
            ),
        )

    in_specs = [
        pl.BlockSpec((block_b, 1, block_q, head_dim), q_index_map),
        pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
        pl.BlockSpec((block_b, 1, block_k_major, head_dim), kv_index_map),
        ab_block_spec,
        q_segment_ids_spec,
        kv_segment_ids_spec,
    ]

    o, *aux = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            grid=grid,
            in_specs=in_specs,
            out_specs=out_specs,
            scratch_shapes=scratch_shapes,
        ),
        out_shape=out_shape,
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=(
                "parallel",
                "parallel",
                "parallel",
                "arbitrary",
            )
        ),
        cost_estimate=_fwd_cost_estimate(
            q,
            k,
            v,
            ab,
            segment_ids,
            causal=causal,
            softmax_scale=softmax_scale,
            kernel_inputs_specs=(q, k, v, ab, q_segment_ids, kv_segment_ids),
            kernel_outputs_specs=out_shape,
        ),
    )(q, k, v, ab, q_segment_ids, kv_segment_ids)
    if save_residuals:
        l, m = (v[..., 0] for v in aux[-2:])
        return (o, l, m)
    else:
        return o
