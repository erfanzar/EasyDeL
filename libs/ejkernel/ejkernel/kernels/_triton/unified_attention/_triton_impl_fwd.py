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

"""Unified attention kernels supporting both prefill and decode phases.

This module implements a unified attention mechanism that handles both
prefill (processing multiple query tokens) and decode (single token
generation) phases with a shared kernel architecture.

Kernel Variants:
---------------
_unified_attention_2d:
    Standard 2D attention kernel for moderate sequence lengths.
    Processes query blocks against the full KV context using
    tiled matrix operations with online softmax.

_unified_attention_3d:
    Segmented 3D kernel for decode-heavy workloads with long contexts.
    Splits KV context into segments processed in parallel, then
    reduced to final output.

_reduce_segments:
    Reduction kernel for 3D attention. Combines segment-level
    outputs using numerically stable log-sum-exp weighted averaging.

Supported Features:
------------------
- Causal masking for autoregressive attention
- Sliding window attention for local context patterns
- ALiBi positional biases (alibi_slopes)
- Query-query biases (qq_bias) for specialized attention patterns
- Attention sinks (softmax_aux) for initial token importance
- Logit soft-capping for numerical stability
- Grouped-query attention (GQA) and multi-query attention (MQA)

Memory Layout:
-------------
Uses block-tabled KV cache with:
- Packed query tensor [total_tokens, num_q_heads, head_dim]
- Block-tabled KV cache [num_blocks, block_size, num_kv_heads, head_dim]
- Block tables mapping sequences to physical blocks

Selection Logic:
---------------
The implementation automatically selects between 2D and 3D kernels based on:
- Batch size and sequence lengths
- Whether in decode-only mode (single query per sequence)
- Configurable thresholds for optimal performance
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import triton
import triton.language as tl

from ejkernel.callib import triton_call


@triton.jit
def _cdiv(x, y):
    """Compute ceiling division of x by y in Triton device code."""
    return (x + y - 1) // y


@triton.jit
def _apply_softcap(scores, cap):
    """Apply logit soft-capping via stable tanh using exponentials.

    Computes ``cap * tanh(scores / cap)`` to bound attention logits, using
    the identity ``tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))``
    for numerical stability on GPU.

    Args:
        scores: Attention logit scores to be capped.
        cap: Soft-cap value controlling the maximum logit magnitude.

    Returns:
        Capped scores bounded to the range ``(-cap, +cap)``.
    """
    s = scores / cap
    p1 = tl.exp(s)
    p2 = tl.exp(-s)
    return cap * (p1 - p2) / (p1 + p2)


@triton.jit
def _find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    """Find the sequence index that owns a given query position via binary search.

    Performs a binary search over the ``query_start_len_ptr`` cumulative-offset
    array to locate which sequence owns the given ``target_idx``.

    When ``use_q_block_mode`` is True, the search operates on block-level
    indices: the cumulative start value is converted to a block index via
    ``val // BLOCK_Q + seq_idx`` to account for the inter-sequence block
    padding. When False, the raw token-level values are compared directly.

    Args:
        query_start_len_ptr: Pointer to cumulative query start offsets,
            shape ``(num_seqs + 1,)``.
        target_idx: The global query block index (block mode) or token
            index (token mode) to locate.
        num_seqs: Number of sequences in the batch.
        BLOCK_Q: Number of query tokens per block (compile-time constant).
        use_q_block_mode: If True, search in block-index space; if False,
            search in token-index space.

    Returns:
        The 0-based sequence index that contains ``target_idx``.
    """
    left: tl.int32 = 0  # type: ignore
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid if use_q_block_mode else val
        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1


@triton.jit
def _unified_attention_2d(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    sink_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    qq_bias_ptr,
    query_start_len_ptr,
    scale,
    softcap,
    block_table_stride,
    query_stride_0,
    query_stride_1,
    output_stride_0,
    output_stride_1,
    qq_bias_stride_0,
    stride_k_cache_0,
    stride_k_cache_1,
    stride_k_cache_2,
    stride_k_cache_3,
    stride_v_cache_0,
    stride_v_cache_1,
    stride_v_cache_2,
    stride_v_cache_3,
    num_seqs,
    out_ptr,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    USE_ALIBI_SLOPES: tl.constexpr,
    USE_QQ_BIAS: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Standard 2D unified attention kernel for prefill and moderate-length decode.

    Each program instance handles one (query_block, kv_head) pair. It iterates
    through the full KV context (up to the causal boundary) using tiled matrix
    operations and online softmax accumulation. Supports grouped-query attention
    (GQA) by interleaving multiple query heads within a single block.

    The kernel uses online softmax to compute attention in a single forward pass
    without materializing the full attention matrix. For each KV tile, it
    computes QK^T scores, applies optional modifiers (soft-cap, ALiBi,
    query-query bias, sliding window, causal mask), and updates the running
    softmax statistics (max, exp-sum) and accumulated output.

    Grid: ``(total_q_blocks, num_kv_heads)``

    Args:
        query_ptr: Query tensor pointer, shape ``(total_tokens, num_q_heads, head_dim)``.
        key_cache_ptr: Key cache pointer, shape ``(num_blocks, block_size, num_kv_heads, head_dim)``.
        value_cache_ptr: Value cache pointer, same shape as key cache.
        sink_ptr: Attention sink scores pointer, shape ``(num_q_heads,)``.
        block_tables_ptr: Block table pointer mapping sequences to physical blocks.
        seq_lens_ptr: Total sequence lengths (context + query) per sequence.
        alibi_slopes_ptr: ALiBi slope pointer, shape ``(num_q_heads,)``.
        qq_bias_ptr: Query-query bias matrix pointer.
        query_start_len_ptr: Cumulative query start offsets, shape ``(num_seqs + 1,)``.
        scale: Softmax scaling factor (typically ``1/sqrt(head_dim)``).
        softcap: Logit soft-capping value (0 to disable).
        block_table_stride: Stride between block table rows.
        query_stride_0, query_stride_1: Query tensor strides.
        output_stride_0, output_stride_1: Output tensor strides.
        qq_bias_stride_0: Row stride of the query-query bias matrix.
        stride_k_cache_0..3: Key cache strides for each dimension.
        stride_v_cache_0..3: Value cache strides for each dimension.
        num_seqs: Number of sequences in the batch.
        out_ptr: Output tensor pointer, same shape as queries.
        num_query_heads: Total number of query heads (compile-time constant).
        num_queries_per_kv: Number of query heads per KV head for GQA.
        BLOCK_SIZE: Physical KV cache block size.
        TILE_SIZE: Number of KV positions processed per tile iteration.
        HEAD_SIZE: Actual head dimension.
        HEAD_SIZE_PADDED: Padded head dimension (next power of 2).
        USE_ALIBI_SLOPES: Whether ALiBi positional biases are active.
        USE_QQ_BIAS: Whether query-query biases are active.
        USE_SOFTCAP: Whether logit soft-capping is active.
        USE_SINKS: Whether attention sinks are active.
        SLIDING_WINDOW: Sliding window size (0 to disable).
        BLOCK_Q: Number of KV-aligned query positions per block.
        BLOCK_M: Total query head slots per block (``BLOCK_Q * num_queries_per_kv``).
    """
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    num_seqs = num_seqs.to(tl.int32)
    seq_idx = _find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True)

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)

    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = query_offset_0[:, None] * query_stride_0 + query_offset_1[:, None] * query_stride_1 + offs_d[None, :]

    dim_mask = (offs_d < HEAD_SIZE).to(tl.int1)
    query_mask_0 = (query_pos < cur_batch_query_len).to(tl.int1)
    query_mask_1 = (query_offset_1 < num_query_heads).to(tl.int1)

    q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )
    if q.dtype != key_cache_ptr.dtype.element_ty:
        q = q.to(key_cache_ptr.dtype.element_ty)

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS:
        m_prev = tl.load(sink_ptr + query_offset_1, mask=query_mask_1, other=float("-inf")).to(tl.float32)
        l_prev = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    else:
        m_prev = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_prev = tl.full([BLOCK_M], 1.0, dtype=tl.float32)

    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    seq_len = tl.load(seq_lens_ptr + seq_idx).to(tl.int32)
    context_len = seq_len - cur_batch_query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0).to(tl.float32)

    if USE_QQ_BIAS:
        qq_bias_row_ptrs = qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0

    max_seq_prefix_len = (context_len + q_block_local_idx * BLOCK_Q + (BLOCK_M - 1) // num_queries_per_kv + 1).to(
        tl.int32
    )
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    num_tiles = _cdiv(max_seq_prefix_len, TILE_SIZE)

    tile_start = 0
    tile_end = num_tiles
    if SLIDING_WINDOW > 0:
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(qpos_lo + (BLOCK_M - 1) // num_queries_per_kv, cur_batch_query_len - 1)

        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    for j in range(tile_start, tile_end):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )
        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )

        k = tl.load(key_cache_ptr + k_offset, mask=dim_mask[:, None] & tile_mask[None, :], other=0.0)
        v = tl.load(value_cache_ptr + v_offset, mask=dim_mask[None, :] & tile_mask[:, None], other=0.0)

        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1

        scores = scale * tl.dot(q, k)

        if USE_SOFTCAP:
            scores = _apply_softcap(scores, softcap)

        scores = tl.where(query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, scores, float("-inf"))

        if SLIDING_WINDOW > 0:
            scores = tl.where(
                (context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW,
                scores,
                float("-inf"),
            )

        if USE_ALIBI_SLOPES:
            scores += alibi_slope[:, None] * (seq_offset - context_len).to(tl.float32)

        if USE_QQ_BIAS:
            key_rel_pos = seq_offset - context_len
            is_query_key = (key_rel_pos >= 0) & (key_rel_pos < qq_bias_stride_0)
            qqb = tl.load(qq_bias_row_ptrs + key_rel_pos[None, :], mask=is_query_key[None, :], other=0.0)
            scores += qqb.to(tl.float32)

        m_curr = tl.maximum(m_prev, tl.max(scores, axis=1))
        m_curr = tl.where(m_curr > float("-inf"), m_curr, 0.0)

        p = tl.exp(scores - m_curr[:, None])
        l_curr = tl.sum(p, axis=1)

        alpha = tl.exp(m_prev - m_curr)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

        l_prev = l_prev * alpha + l_curr
        m_prev = m_curr

    l_prev = tl.maximum(l_prev, 1e-6)
    acc = acc / l_prev[:, None]

    output_offset = (
        query_offset_0[:, None] * output_stride_0 + query_offset_1[:, None] * output_stride_1 + offs_d[None, :]
    )
    tl.store(out_ptr + output_offset, acc, mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None])


@triton.jit
def _unified_attention_3d(
    query_ptr,
    key_cache_ptr,
    value_cache_ptr,
    sink_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    alibi_slopes_ptr,
    qq_bias_ptr,
    query_start_len_ptr,
    scale,
    softcap,
    block_table_stride,
    query_stride_0,
    query_stride_1,
    qq_bias_stride_0,
    stride_k_cache_0,
    stride_k_cache_1,
    stride_k_cache_2,
    stride_k_cache_3,
    stride_v_cache_0,
    stride_v_cache_1,
    stride_v_cache_2,
    stride_v_cache_3,
    num_seqs,
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    num_query_heads: tl.constexpr,
    num_queries_per_kv: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    USE_ALIBI_SLOPES: tl.constexpr,
    USE_QQ_BIAS: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
):
    """Segmented 3D unified attention kernel for decode-heavy workloads.

    Extends the 2D kernel by splitting the KV context into parallel segments
    along the third grid dimension (``segm_idx``). Each program instance
    processes a subset of KV tiles for a given (query_block, kv_head, segment)
    triple, producing partial softmax statistics (output accumulator, max,
    exp-sum) that are later combined by ``_reduce_segments``.

    This approach achieves higher GPU occupancy for long-context decode
    scenarios where a single query attends to many KV positions: by splitting
    the KV range into ``NUM_SEGMENTS_PER_SEQ`` parallel segments, multiple
    thread blocks can work on the same query concurrently.

    Grid: ``(total_q_blocks, num_kv_heads, NUM_SEGMENTS_PER_SEQ)``

    Args:
        query_ptr: Query tensor pointer, shape ``(total_tokens, num_q_heads, head_dim)``.
        key_cache_ptr: Key cache pointer, shape ``(num_blocks, block_size, num_kv_heads, head_dim)``.
        value_cache_ptr: Value cache pointer, same shape as key cache.
        sink_ptr: Attention sink scores pointer (used only in segment 0).
        block_tables_ptr: Block table pointer mapping sequences to physical blocks.
        seq_lens_ptr: Total sequence lengths per sequence.
        alibi_slopes_ptr: ALiBi slope pointer, shape ``(num_q_heads,)``.
        qq_bias_ptr: Query-query bias matrix pointer.
        query_start_len_ptr: Cumulative query start offsets.
        scale: Softmax scaling factor.
        softcap: Logit soft-capping value.
        block_table_stride: Stride between block table rows.
        query_stride_0, query_stride_1: Query tensor strides.
        qq_bias_stride_0: Row stride of query-query bias.
        stride_k_cache_0..3: Key cache strides.
        stride_v_cache_0..3: Value cache strides.
        num_seqs: Number of sequences in the batch.
        segm_output_ptr: Segment output accumulator pointer,
            shape ``(total_tokens, num_q_heads, NUM_SEGMENTS_PER_SEQ, HEAD_SIZE_PADDED)``.
        segm_max_ptr: Per-segment running max pointer,
            shape ``(total_tokens, num_q_heads, NUM_SEGMENTS_PER_SEQ)``.
        segm_expsum_ptr: Per-segment exponential sum pointer (same shape as max).
        num_query_heads: Total number of query heads.
        num_queries_per_kv: Number of query heads per KV head.
        BLOCK_SIZE: Physical KV cache block size.
        TILE_SIZE: KV positions per tile iteration.
        HEAD_SIZE: Actual head dimension.
        HEAD_SIZE_PADDED: Padded head dimension.
        USE_ALIBI_SLOPES: Whether ALiBi biases are active.
        USE_QQ_BIAS: Whether query-query biases are active.
        USE_SOFTCAP: Whether logit soft-capping is active.
        USE_SINKS: Whether attention sinks are active (segment 0 only).
        SLIDING_WINDOW: Sliding window size (0 to disable).
        BLOCK_Q: KV-aligned query positions per block.
        BLOCK_M: Total query head slots per block.
        NUM_SEGMENTS_PER_SEQ: Number of parallel KV segments.
    """
    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)

    num_seqs = num_seqs.to(tl.int32)
    seq_idx = _find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True)

    q_block_start_idx = tl.load(query_start_len_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    seq_len = tl.load(seq_lens_ptr + seq_idx).to(tl.int32)
    tiles_per_segment = _cdiv(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = query_offset_0[:, None] * query_stride_0 + query_offset_1[:, None] * query_stride_1 + offs_d[None, :]

    dim_mask = (offs_d < HEAD_SIZE).to(tl.int1)
    query_mask_0 = (query_pos < cur_batch_query_len).to(tl.int1)
    query_mask_1 = (query_offset_1 < num_query_heads).to(tl.int1)

    q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
        other=0.0,
    )
    if q.dtype != key_cache_ptr.dtype.element_ty:
        q = q.to(key_cache_ptr.dtype.element_ty)

    block_table_offset = seq_idx * block_table_stride

    if USE_SINKS and segm_idx == 0:
        m_prev = tl.load(sink_ptr + query_offset_1, mask=query_mask_1, other=float("-inf")).to(tl.float32)
        l_prev = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    else:
        m_prev = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_prev = tl.full([BLOCK_M], 1.0, dtype=tl.float32)

    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    context_len = seq_len - cur_batch_query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0).to(tl.float32)

    if USE_QQ_BIAS:
        qq_bias_row_ptrs = qq_bias_ptr + query_pos[:, None] * qq_bias_stride_0

    max_seq_prefix_len = (context_len + q_block_local_idx * BLOCK_Q + (BLOCK_M - 1) // num_queries_per_kv + 1).to(
        tl.int32
    )
    max_seq_prefix_len = tl.minimum(max_seq_prefix_len, seq_len)
    num_tiles = _cdiv(max_seq_prefix_len, TILE_SIZE)

    segm_begin = segm_idx * tiles_per_segment
    segm_end = tl.minimum((segm_idx + 1) * tiles_per_segment, num_tiles)

    for j in range(segm_begin, segm_end):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        physical_block_idx = tl.load(block_tables_ptr + block_table_offset + seq_offset // BLOCK_SIZE).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx * stride_v_cache_2
            + offs_d[None, :] * stride_v_cache_3
            + (seq_offset % BLOCK_SIZE)[:, None] * stride_v_cache_1
        )
        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx * stride_k_cache_2
            + offs_d[:, None] * stride_k_cache_3
            + (seq_offset % BLOCK_SIZE)[None, :] * stride_k_cache_1
        )

        k = tl.load(key_cache_ptr + k_offset, mask=dim_mask[:, None] & tile_mask[None, :], other=0.0)
        v = tl.load(value_cache_ptr + v_offset, mask=dim_mask[None, :] & tile_mask[:, None], other=0.0)

        seq_mask = seq_offset[None, :] < context_len + query_pos[:, None] + 1
        scores = scale * tl.dot(q, k)

        if USE_SOFTCAP:
            scores = _apply_softcap(scores, softcap)

        scores = tl.where(query_mask_1[:, None] & query_mask_0[:, None] & seq_mask, scores, float("-inf"))

        if SLIDING_WINDOW > 0:
            scores = tl.where(
                (context_len + query_pos[:, None] - seq_offset) < SLIDING_WINDOW,
                scores,
                float("-inf"),
            )

        if USE_ALIBI_SLOPES:
            scores += alibi_slope[:, None] * (seq_offset - context_len).to(tl.float32)

        if USE_QQ_BIAS:
            key_rel_pos = seq_offset - context_len
            is_query_key = (key_rel_pos >= 0) & (key_rel_pos < qq_bias_stride_0)
            qqb = tl.load(qq_bias_row_ptrs + key_rel_pos[None, :], mask=is_query_key[None, :], other=0.0)
            scores += qqb.to(tl.float32)

        m_curr = tl.maximum(m_prev, tl.max(scores, axis=1))
        m_curr = tl.where(m_curr > float("-inf"), m_curr, 0.0)

        p = tl.exp(scores - m_curr[:, None])
        l_curr = tl.sum(p, axis=1)

        alpha = tl.exp(m_prev - m_curr)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)

        l_prev = l_prev * alpha + l_curr
        m_prev = m_curr

    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_offset_1[:, None] * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & query_mask_0[:, None] & query_mask_1[:, None],
    )

    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1 * NUM_SEGMENTS_PER_SEQ
        + segm_idx
    )
    tl.store(segm_max_ptr + segm_offset, m_prev, mask=query_mask_0 & query_mask_1)
    tl.store(segm_expsum_ptr + segm_offset, l_prev, mask=query_mask_0 & query_mask_1)


@triton.jit
def _reduce_segments(
    segm_output_ptr,
    segm_max_ptr,
    segm_expsum_ptr,
    seq_lens_ptr,
    query_start_len_ptr,
    num_seqs,
    output_stride_0,
    output_stride_1,
    out_ptr,
    num_query_heads: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
):
    """Reduce segment-level outputs from the 3D kernel into final attention output.

    Combines the partial softmax outputs produced by ``_unified_attention_3d``
    using numerically stable log-sum-exp weighted averaging. Each program
    instance handles one (query_token, query_head) pair, reading the partial
    outputs, max values, and exp-sums from all segments and producing the
    final normalized output.

    The reduction computes:
        overall_max = max(segm_max[s] for s in segments)
        rescaled_expsum[s] = segm_expsum[s] * exp(segm_max[s] - overall_max)
        overall_expsum = sum(rescaled_expsum)
        output = sum(segm_output[s] * exp(segm_max[s] - overall_max)) / overall_expsum

    Grid: ``(total_tokens, num_query_heads)``

    Args:
        segm_output_ptr: Segment output accumulator from 3D kernel,
            shape ``(total_tokens, num_q_heads, NUM_SEGMENTS_PER_SEQ, HEAD_SIZE_PADDED)``.
        segm_max_ptr: Per-segment max values,
            shape ``(total_tokens, num_q_heads, NUM_SEGMENTS_PER_SEQ)``.
        segm_expsum_ptr: Per-segment exp-sums (same shape as max).
        seq_lens_ptr: Total sequence lengths per sequence.
        query_start_len_ptr: Cumulative query start offsets.
        num_seqs: Number of sequences in the batch.
        output_stride_0, output_stride_1: Output tensor strides.
        out_ptr: Final output tensor pointer.
        num_query_heads: Total number of query heads.
        TILE_SIZE: KV tile size (used to compute active segment count).
        HEAD_SIZE: Actual head dimension.
        HEAD_SIZE_PADDED: Padded head dimension.
        BLOCK_Q: KV-aligned query positions per block.
        NUM_SEGMENTS_PER_SEQ: Maximum number of parallel segments.
    """
    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    num_seqs = num_seqs.to(tl.int32)
    seq_idx = _find_seq_idx(query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False)

    seq_len = tl.load(seq_lens_ptr + seq_idx).to(tl.int32)

    tiles_per_segment = _cdiv(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
    act_num_segments = _cdiv(seq_len, tiles_per_segment * TILE_SIZE)

    segm_mask = tl.arange(0, NUM_SEGMENTS_PER_SEQ) < tl.full([NUM_SEGMENTS_PER_SEQ], act_num_segments, tl.int32)
    dim_mask = (tl.arange(0, HEAD_SIZE_PADDED) < HEAD_SIZE).to(tl.int1)

    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx * NUM_SEGMENTS_PER_SEQ
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max)

    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * tl.exp(segm_max - overall_max)
    overall_expsum = tl.sum(segm_expsum)

    segm_output_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + tl.arange(0, NUM_SEGMENTS_PER_SEQ)[:, None] * HEAD_SIZE_PADDED
        + tl.arange(0, HEAD_SIZE_PADDED)[None, :]
    )
    segm_output = tl.load(segm_output_ptr + segm_output_offset, mask=segm_mask[:, None] & dim_mask[None, :], other=0.0)
    segm_output *= tl.exp(segm_max - overall_max)[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

    out_offset = query_token_idx * output_stride_0 + query_head_idx * output_stride_1 + tl.arange(0, HEAD_SIZE_PADDED)
    tl.store(out_ptr + out_offset, acc, mask=dim_mask)


def unified_attention_triton(
    *,
    queries: jax.Array,
    key_cache: jax.Array,
    value_cache: jax.Array,
    block_tables: jax.Array,
    kv_lens: jax.Array,
    query_start_loc: jax.Array,
    softmax_scale: float | None,
    causal: bool,
    sliding_window: int | None,
    logits_soft_cap: float | None,
    seq_threshold_3d: int | None,
    num_par_softmax_segments: int | None,
    alibi_slopes: jax.Array | None,
    qq_bias: jax.Array | None,
    softmax_aux: jax.Array | None,
    num_warps: int | None,
    num_stages: int | None,
) -> jax.Array:
    """Launch unified attention Triton kernels for prefill and decode phases.

    Automatically selects between the 2D kernel (standard) and the 3D
    segmented kernel (decode-optimized) based on batch characteristics.
    The 3D kernel is used when all sequences have a single query token
    (pure decode), the batch is small enough, and segmentation parameters
    are provided. Otherwise, the 2D kernel handles both prefill and decode.

    The function validates input shapes and types, computes strides for the
    block-tabled KV cache layout, and configures GQA interleaving within
    query blocks.

    Args:
        queries: Packed query tensor of shape
            ``(total_tokens, num_q_heads, head_dim)``.
        key_cache: Block-tabled key cache of shape
            ``(num_blocks, block_size, num_kv_heads, head_dim)``.
        value_cache: Block-tabled value cache, same shape as key_cache.
        block_tables: Mapping from (sequence, logical_block) to physical
            block index, shape ``(num_seqs, max_blocks_per_seq)``, int32.
        kv_lens: Total KV length (context + query) per sequence,
            shape ``(num_seqs,)``, int32.
        query_start_loc: Cumulative query start offsets,
            shape ``(num_seqs + 1,)``, int32.
        softmax_scale: Attention scaling factor. Defaults to
            ``1 / sqrt(head_dim)`` if None.
        causal: Must be True (non-causal not implemented).
        sliding_window: Sliding window size in tokens. None or 0 disables.
        logits_soft_cap: Logit soft-capping value. None or 0.0 disables.
        seq_threshold_3d: Maximum batch size for 3D kernel selection.
            None disables the 3D path.
        num_par_softmax_segments: Number of parallel KV segments for the
            3D kernel. None disables the 3D path.
        alibi_slopes: Per-head ALiBi slopes, shape ``(num_q_heads,)``.
            None disables ALiBi.
        qq_bias: Square query-query bias matrix. None disables.
        softmax_aux: Attention sink pre-scores, shape ``(num_q_heads,)``.
            None disables sinks.
        num_warps: Triton num_warps override. None uses default.
        num_stages: Triton num_stages override. None uses default.

    Returns:
        Output tensor of shape ``(total_tokens, num_q_heads, head_dim)``
        with the same dtype as ``queries``.

    Raises:
        NotImplementedError: If ``causal`` is False.
        ValueError: If input shapes or dtypes are invalid.
    """
    if not causal:
        raise NotImplementedError("unified_attention_triton only supports causal attention.")

    if queries.ndim != 3:
        raise ValueError("queries must be rank-3: [total_tokens, num_q_heads, head_dim]")
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("key_cache/value_cache must be rank-4: [num_blocks, block_size, num_kv_heads, head_dim]")
    if key_cache.shape != value_cache.shape:
        raise ValueError("key_cache and value_cache must have the same shape")

    if query_start_loc.dtype != jnp.int32 or kv_lens.dtype != jnp.int32 or block_tables.dtype != jnp.int32:
        raise ValueError("query_start_loc/kv_lens/block_tables must be int32")

    total_tokens, num_query_heads, head_size = map(int, queries.shape)
    _num_blocks, block_size, num_kv_heads, head_size_kv = map(int, key_cache.shape)
    if head_size_kv != head_size:
        raise ValueError("head_dim mismatch between queries and KV cache")
    if num_query_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads (GQA)")

    num_seqs, max_blocks_per_seq = map(int, block_tables.shape)
    if query_start_loc.shape[0] != num_seqs + 1 or kv_lens.shape[0] != num_seqs:
        raise ValueError("query_start_loc must be [num_seqs+1] and kv_lens must be [num_seqs]")

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_size)

    use_alibi = alibi_slopes is not None
    use_qq_bias = qq_bias is not None
    use_sinks = softmax_aux is not None

    if use_alibi:
        if alibi_slopes.shape != (num_query_heads,):
            raise ValueError("alibi_slopes must have shape (num_q_heads,)")
        alibi_slopes = alibi_slopes.astype(jnp.float32)
    else:
        alibi_slopes = jnp.zeros((1,), dtype=jnp.float32)

    if use_qq_bias:
        if qq_bias.ndim != 2 or qq_bias.shape[0] != qq_bias.shape[1]:
            raise ValueError("qq_bias must be square [num_query_tokens, num_query_tokens]")
        qq_bias = qq_bias.astype(jnp.float32)
        qq_bias_stride_0 = int(qq_bias.shape[1])
    else:
        qq_bias = jnp.zeros((1, 1), dtype=jnp.float32)
        qq_bias_stride_0 = 0

    if use_sinks:
        if softmax_aux.shape != (num_query_heads,):
            raise ValueError("softmax_aux must have shape (num_q_heads,)")
        softmax_aux = softmax_aux.astype(jnp.float32)
    else:
        softmax_aux = jnp.zeros((1,), dtype=jnp.float32)

    num_queries_per_kv = num_query_heads // num_kv_heads
    block_m = 16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    block_q = block_m // num_queries_per_kv

    total_num_q_blocks = total_tokens // block_q + num_seqs

    tile_size_prefill = 32
    tile_size_decode = 16

    head_size_padded = triton.next_power_of_2(head_size)

    q_stride_0 = num_query_heads * head_size
    q_stride_1 = head_size
    o_stride_0 = q_stride_0
    o_stride_1 = q_stride_1

    bt_stride = max_blocks_per_seq
    k_stride_0 = block_size * num_kv_heads * head_size
    k_stride_1 = num_kv_heads * head_size
    k_stride_2 = head_size
    k_stride_3 = 1
    v_stride_0 = k_stride_0
    v_stride_1 = k_stride_1
    v_stride_2 = k_stride_2
    v_stride_3 = 1

    if sliding_window is None:
        sliding_window_val = 0
    else:
        sliding_window_val = int(sliding_window)
        if sliding_window_val <= 0:
            sliding_window_val = 0

    if logits_soft_cap is None:
        logits_soft_cap_val = 0.0
    else:
        logits_soft_cap_val = float(logits_soft_cap)

    decode_only = total_tokens <= num_seqs
    use_2d = (
        seq_threshold_3d is None
        or num_par_softmax_segments is None
        or not decode_only
        or num_seqs > int(seq_threshold_3d)
    )

    metaparams_2d = dict(
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size_prefill,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        USE_ALIBI_SLOPES=bool(use_alibi),
        USE_QQ_BIAS=bool(use_qq_bias),
        USE_SOFTCAP=bool(logits_soft_cap_val > 0),
        USE_SINKS=bool(use_sinks),
        SLIDING_WINDOW=int(sliding_window_val),
        BLOCK_Q=int(block_q),
        BLOCK_M=int(block_m),
        num_warps=num_warps,
        num_stages=num_stages,
    )

    if use_2d:
        (out,) = triton_call(
            queries,
            key_cache,
            value_cache,
            softmax_aux,
            block_tables,
            kv_lens,
            alibi_slopes,
            qq_bias,
            query_start_loc,
            float(softmax_scale),
            float(logits_soft_cap_val),
            int(bt_stride),
            int(q_stride_0),
            int(q_stride_1),
            int(o_stride_0),
            int(o_stride_1),
            int(qq_bias_stride_0),
            int(k_stride_0),
            int(k_stride_1),
            int(k_stride_2),
            int(k_stride_3),
            int(v_stride_0),
            int(v_stride_1),
            int(v_stride_2),
            int(v_stride_3),
            int(num_seqs),
            kernel=_unified_attention_2d,
            out_shape=[jax.ShapeDtypeStruct(queries.shape, queries.dtype)],
            grid=lambda META: (int(total_num_q_blocks), int(num_kv_heads)),
            name="ejkernel::triton::unified_attention_2d",
            **metaparams_2d,
        )
        return out

    num_segments = int(num_par_softmax_segments)
    segm_out_shape = jax.ShapeDtypeStruct((total_tokens, num_query_heads, num_segments, head_size_padded), jnp.float32)
    segm_ml_shape = jax.ShapeDtypeStruct((total_tokens, num_query_heads, num_segments), jnp.float32)

    metaparams_3d = dict(
        num_query_heads=num_query_heads,
        num_queries_per_kv=num_queries_per_kv,
        BLOCK_SIZE=block_size,
        TILE_SIZE=tile_size_decode,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        USE_ALIBI_SLOPES=bool(use_alibi),
        USE_QQ_BIAS=bool(use_qq_bias),
        USE_SOFTCAP=bool(logits_soft_cap_val > 0),
        USE_SINKS=bool(use_sinks),
        SLIDING_WINDOW=int(sliding_window_val),
        BLOCK_Q=int(block_q),
        BLOCK_M=int(block_m),
        NUM_SEGMENTS_PER_SEQ=num_segments,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    segm_output, segm_max, segm_expsum = triton_call(
        queries,
        key_cache,
        value_cache,
        softmax_aux,
        block_tables,
        kv_lens,
        alibi_slopes,
        qq_bias,
        query_start_loc,
        float(softmax_scale),
        float(logits_soft_cap_val),
        int(bt_stride),
        int(q_stride_0),
        int(q_stride_1),
        int(qq_bias_stride_0),
        int(k_stride_0),
        int(k_stride_1),
        int(k_stride_2),
        int(k_stride_3),
        int(v_stride_0),
        int(v_stride_1),
        int(v_stride_2),
        int(v_stride_3),
        int(num_seqs),
        kernel=_unified_attention_3d,
        out_shape=(segm_out_shape, segm_ml_shape, segm_ml_shape),
        grid=lambda META: (int(total_num_q_blocks), int(num_kv_heads), int(num_segments)),
        name="ejkernel::triton::unified_attention_3d",
        **metaparams_3d,
    )

    metaparams_reduce = dict(
        num_query_heads=num_query_heads,
        TILE_SIZE=tile_size_decode,
        HEAD_SIZE=head_size,
        HEAD_SIZE_PADDED=head_size_padded,
        BLOCK_Q=int(block_q),
        NUM_SEGMENTS_PER_SEQ=num_segments,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    (out,) = triton_call(
        segm_output,
        segm_max,
        segm_expsum,
        kv_lens,
        query_start_loc,
        int(num_seqs),
        int(o_stride_0),
        int(o_stride_1),
        kernel=_reduce_segments,
        out_shape=[jax.ShapeDtypeStruct(queries.shape, queries.dtype)],
        grid=lambda META: (int(total_tokens), int(num_query_heads)),
        name="ejkernel::triton::unified_attention_reduce_segments",
        **metaparams_reduce,
    )
    return out
