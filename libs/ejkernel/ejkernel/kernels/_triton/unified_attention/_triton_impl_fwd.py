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
    Splits KV context into segments processed in parallel (skipping
    tiles outside the sliding window), then reduced to final output.

_reduce_segments:
    Reduction kernel for 3D attention. Combines segment-level
    outputs using numerically stable log-sum-exp weighted averaging.
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
    """Apply overflow-safe logit soft-capping via ``exp(-2|x|)`` tanh.

    Computes ``cap * tanh(scores / cap)`` without overflowing ``exp(s)`` for
    large ``|s| > 88.7`` (where naive ``(exp(s) - exp(-s)) / (exp(s) + exp(-s))``
    produces ``inf / inf = NaN``).
    """
    s = scores / cap
    ax = tl.abs(s)
    z = tl.exp(-2.0 * ax)
    t = (1.0 - z) / (1.0 + z)
    return cap * tl.where(s >= 0.0, t, -t)


@triton.jit
def _find_seq_idx(
    query_start_len_ptr,
    target_idx,
    num_seqs,
    BLOCK_Q: tl.constexpr,
    use_q_block_mode: tl.constexpr,
):
    """Find the sequence index that owns a given query position via binary search."""
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
    USE_FP32_IEEE: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """Standard 2D unified attention kernel for prefill and moderate-length decode."""
    LOG2E: tl.constexpr = 1.4426950408889634

    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    kv_head_idx64 = kv_head_idx.to(tl.int64)

    num_seqs = num_seqs.to(tl.int32)
    seq_idx = _find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True)

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx).to(tl.int32)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1).to(tl.int32)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    q_block_start_idx = cur_batch_in_all_start_index // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_d64 = offs_d.to(tl.int64)
    offs_t = tl.arange(0, TILE_SIZE)

    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0.to(tl.int64)[:, None] * query_stride_0
        + query_offset_1.to(tl.int64)[:, None] * query_stride_1
        + offs_d64[None, :]
    )

    dim_mask = (offs_d < HEAD_SIZE).to(tl.int1)
    query_mask_0 = (query_pos < cur_batch_query_len).to(tl.int1)
    query_mask_1 = (query_offset_1 < num_query_heads).to(tl.int1)
    row_valid = query_mask_0 & query_mask_1

    q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & row_valid[:, None],
        other=0.0,
    )
    if q.dtype != key_cache_ptr.dtype.element_ty:
        q = q.to(key_cache_ptr.dtype.element_ty)

    block_table_offset = seq_idx.to(tl.int64) * block_table_stride

    if USE_SINKS:
        m_prev = tl.load(sink_ptr + query_offset_1, mask=query_mask_1, other=float("-inf")).to(tl.float32)
        l_prev = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    else:
        m_prev = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)

    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    seq_len = tl.load(seq_lens_ptr + seq_idx).to(tl.int32)
    context_len = seq_len - cur_batch_query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0).to(tl.float32)

    if USE_QQ_BIAS:
        qq_bias_row_ptrs = qq_bias_ptr + query_pos.to(tl.int64)[:, None] * qq_bias_stride_0

    max_seq_prefix_len = (context_len + q_block_local_idx * BLOCK_Q + (BLOCK_M - 1) // num_queries_per_kv + 1).to(
        tl.int32
    )
    max_seq_prefix_len = tl.minimum(tl.maximum(max_seq_prefix_len, 0), tl.maximum(seq_len, 0))
    num_tiles = _cdiv(max_seq_prefix_len, TILE_SIZE)

    tile_start = 0
    tile_end = num_tiles
    if SLIDING_WINDOW > 0:
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(qpos_lo + (BLOCK_M - 1) // num_queries_per_kv, cur_batch_query_len - 1)

        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        tile_start = tl.minimum(tl.maximum(0, first_allowed_key // TILE_SIZE), num_tiles)
        tile_end = tl.minimum(tl.maximum(0, (last_allowed_key // TILE_SIZE) + 1), num_tiles)

    for j in range(tile_start, tile_end):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        logical_block = seq_offset // BLOCK_SIZE
        bt_mask = tile_mask & (logical_block < block_table_stride)
        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + logical_block.to(tl.int64),
            mask=bt_mask,
            other=0,
        ).to(tl.int64)
        within_block = (seq_offset % BLOCK_SIZE).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx64 * stride_v_cache_2
            + offs_d64[None, :] * stride_v_cache_3
            + within_block[:, None] * stride_v_cache_1
        )
        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx64 * stride_k_cache_2
            + offs_d64[:, None] * stride_k_cache_3
            + within_block[None, :] * stride_k_cache_1
        )

        k = tl.load(key_cache_ptr + k_offset, mask=dim_mask[:, None] & tile_mask[None, :], other=0.0)
        v = tl.load(value_cache_ptr + v_offset, mask=dim_mask[None, :] & tile_mask[:, None], other=0.0)

        if USE_FP32_IEEE:
            scores = scale * tl.dot(q, k, input_precision="ieee", out_dtype=tl.float32)
        else:
            scores = scale * tl.dot(q, k, out_dtype=tl.float32)

        if USE_SOFTCAP:
            scores = _apply_softcap(scores, softcap)

        if USE_ALIBI_SLOPES:
            scores += alibi_slope[:, None] * (seq_offset - context_len).to(tl.float32)[None, :]

        if USE_QQ_BIAS:
            key_rel_pos = seq_offset - context_len
            is_query_key = (key_rel_pos >= 0) & (key_rel_pos < qq_bias_stride_0)
            qqb = tl.load(
                qq_bias_row_ptrs + key_rel_pos.to(tl.int64)[None, :],
                mask=query_mask_0[:, None] & is_query_key[None, :],
                other=0.0,
            )
            scores += qqb.to(tl.float32)

        seq_mask = tile_mask[None, :] & (seq_offset[None, :] < context_len + query_pos[:, None] + 1)
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & ((context_len + query_pos[:, None] - seq_offset[None, :]) < SLIDING_WINDOW)

        valid_mask = row_valid[:, None] & seq_mask
        scores_masked = tl.where(valid_mask, scores, float("-inf"))

        m_blk = tl.max(scores_masked, axis=1)
        has_any = m_blk > float("-inf")
        m_curr = tl.where(has_any, tl.maximum(m_prev, m_blk), m_prev)

        alpha = tl.where(
            has_any & (m_prev > float("-inf")),
            tl.math.exp2((m_prev - m_curr) * LOG2E),
            tl.where(has_any, 0.0, 1.0),
        )
        p = tl.where(
            valid_mask,
            tl.math.exp2(tl.where(valid_mask, (scores - m_curr[:, None]) * LOG2E, float("-inf"))),
            0.0,
        )
        l_curr = tl.sum(p, axis=1)

        if USE_FP32_IEEE:
            pv = tl.dot(p.to(v.dtype), v, input_precision="ieee", out_dtype=tl.float32)
        else:
            pv = tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)

        acc = acc * alpha[:, None] + pv
        l_prev = l_prev * alpha + l_curr
        m_prev = m_curr

    l_safe = tl.where(l_prev > 0.0, l_prev, 1.0)
    acc = acc / l_safe[:, None]

    output_offset = (
        query_offset_0.to(tl.int64)[:, None] * output_stride_0
        + query_offset_1.to(tl.int64)[:, None] * output_stride_1
        + offs_d64[None, :]
    )
    tl.store(
        out_ptr + output_offset,
        acc.to(out_ptr.dtype.element_ty),
        mask=dim_mask[None, :] & row_valid[:, None],
    )


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
    USE_FP32_IEEE: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    NUM_SEGMENTS_PER_SEQ: tl.constexpr,
):
    """Segmented 3D unified attention kernel for decode-heavy workloads."""
    LOG2E: tl.constexpr = 1.4426950408889634

    q_block_global_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    segm_idx = tl.program_id(2)
    kv_head_idx64 = kv_head_idx.to(tl.int64)

    num_seqs = num_seqs.to(tl.int32)
    seq_idx = _find_seq_idx(query_start_len_ptr, q_block_global_idx, num_seqs, BLOCK_Q, True)

    cur_batch_in_all_start_index = tl.load(query_start_len_ptr + seq_idx).to(tl.int32)
    cur_batch_in_all_stop_index = tl.load(query_start_len_ptr + seq_idx + 1).to(tl.int32)
    cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index

    q_block_start_idx = cur_batch_in_all_start_index // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    if q_block_local_idx * BLOCK_Q >= cur_batch_query_len:
        return

    seq_len = tl.load(seq_lens_ptr + seq_idx).to(tl.int32)
    if seq_len <= 0:
        return
    tiles_per_segment = _cdiv(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE)
    if segm_idx * tiles_per_segment * TILE_SIZE >= seq_len:
        return

    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_d64 = offs_d.to(tl.int64)
    offs_t = tl.arange(0, TILE_SIZE)
    query_pos = q_block_local_idx * BLOCK_Q + offs_m // num_queries_per_kv

    query_offset_0 = cur_batch_in_all_start_index + query_pos
    query_offset_1 = kv_head_idx * num_queries_per_kv + offs_m % num_queries_per_kv
    query_offset = (
        query_offset_0.to(tl.int64)[:, None] * query_stride_0
        + query_offset_1.to(tl.int64)[:, None] * query_stride_1
        + offs_d64[None, :]
    )

    dim_mask = (offs_d < HEAD_SIZE).to(tl.int1)
    query_mask_0 = (query_pos < cur_batch_query_len).to(tl.int1)
    query_mask_1 = (query_offset_1 < num_query_heads).to(tl.int1)
    row_valid = query_mask_0 & query_mask_1

    q = tl.load(
        query_ptr + query_offset,
        mask=dim_mask[None, :] & row_valid[:, None],
        other=0.0,
    )
    if q.dtype != key_cache_ptr.dtype.element_ty:
        q = q.to(key_cache_ptr.dtype.element_ty)

    block_table_offset = seq_idx.to(tl.int64) * block_table_stride

    if USE_SINKS and segm_idx == 0:
        m_prev = tl.load(sink_ptr + query_offset_1, mask=query_mask_1, other=float("-inf")).to(tl.float32)
        l_prev = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    else:
        m_prev = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_prev = tl.zeros([BLOCK_M], dtype=tl.float32)

    acc = tl.zeros([BLOCK_M, HEAD_SIZE_PADDED], dtype=tl.float32)

    context_len = seq_len - cur_batch_query_len

    if USE_ALIBI_SLOPES:
        alibi_slope = tl.load(alibi_slopes_ptr + query_offset_1, mask=query_mask_1, other=0.0).to(tl.float32)

    if USE_QQ_BIAS:
        qq_bias_row_ptrs = qq_bias_ptr + query_pos.to(tl.int64)[:, None] * qq_bias_stride_0

    max_seq_prefix_len = (context_len + q_block_local_idx * BLOCK_Q + (BLOCK_M - 1) // num_queries_per_kv + 1).to(
        tl.int32
    )
    max_seq_prefix_len = tl.minimum(tl.maximum(max_seq_prefix_len, 0), seq_len)
    num_tiles = _cdiv(max_seq_prefix_len, TILE_SIZE)

    segm_begin = segm_idx * tiles_per_segment
    segm_end = tl.minimum((segm_idx + 1) * tiles_per_segment, num_tiles)

    if SLIDING_WINDOW > 0:
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi = tl.minimum(qpos_lo + (BLOCK_M - 1) // num_queries_per_kv, cur_batch_query_len - 1)
        first_allowed_key = context_len + qpos_lo - SLIDING_WINDOW + 1
        last_allowed_key = context_len + qpos_hi
        win_tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        win_tile_end = tl.minimum(tl.maximum(0, (last_allowed_key // TILE_SIZE) + 1), num_tiles)
        segm_begin = tl.maximum(segm_begin, win_tile_start)
        segm_end = tl.minimum(segm_end, win_tile_end)

    for j in range(segm_begin, segm_end):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        logical_block = seq_offset // BLOCK_SIZE
        bt_mask = tile_mask & (logical_block < block_table_stride)
        physical_block_idx = tl.load(
            block_tables_ptr + block_table_offset + logical_block.to(tl.int64),
            mask=bt_mask,
            other=0,
        ).to(tl.int64)
        within_block = (seq_offset % BLOCK_SIZE).to(tl.int64)

        v_offset = (
            physical_block_idx[:, None] * stride_v_cache_0
            + kv_head_idx64 * stride_v_cache_2
            + offs_d64[None, :] * stride_v_cache_3
            + within_block[:, None] * stride_v_cache_1
        )
        k_offset = (
            physical_block_idx[None, :] * stride_k_cache_0
            + kv_head_idx64 * stride_k_cache_2
            + offs_d64[:, None] * stride_k_cache_3
            + within_block[None, :] * stride_k_cache_1
        )

        k = tl.load(key_cache_ptr + k_offset, mask=dim_mask[:, None] & tile_mask[None, :], other=0.0)
        v = tl.load(value_cache_ptr + v_offset, mask=dim_mask[None, :] & tile_mask[:, None], other=0.0)

        if USE_FP32_IEEE:
            scores = scale * tl.dot(q, k, input_precision="ieee", out_dtype=tl.float32)
        else:
            scores = scale * tl.dot(q, k, out_dtype=tl.float32)

        if USE_SOFTCAP:
            scores = _apply_softcap(scores, softcap)

        if USE_ALIBI_SLOPES:
            scores += alibi_slope[:, None] * (seq_offset - context_len).to(tl.float32)[None, :]

        if USE_QQ_BIAS:
            key_rel_pos = seq_offset - context_len
            is_query_key = (key_rel_pos >= 0) & (key_rel_pos < qq_bias_stride_0)
            qqb = tl.load(
                qq_bias_row_ptrs + key_rel_pos.to(tl.int64)[None, :],
                mask=query_mask_0[:, None] & is_query_key[None, :],
                other=0.0,
            )
            scores += qqb.to(tl.float32)

        seq_mask = tile_mask[None, :] & (seq_offset[None, :] < context_len + query_pos[:, None] + 1)
        if SLIDING_WINDOW > 0:
            seq_mask = seq_mask & ((context_len + query_pos[:, None] - seq_offset[None, :]) < SLIDING_WINDOW)

        valid_mask = row_valid[:, None] & seq_mask
        scores_masked = tl.where(valid_mask, scores, float("-inf"))

        m_blk = tl.max(scores_masked, axis=1)
        has_any = m_blk > float("-inf")
        m_curr = tl.where(has_any, tl.maximum(m_prev, m_blk), m_prev)

        alpha = tl.where(
            has_any & (m_prev > float("-inf")),
            tl.math.exp2((m_prev - m_curr) * LOG2E),
            tl.where(has_any, 0.0, 1.0),
        )
        p = tl.where(
            valid_mask,
            tl.math.exp2(tl.where(valid_mask, (scores - m_curr[:, None]) * LOG2E, float("-inf"))),
            0.0,
        )
        l_curr = tl.sum(p, axis=1)

        if USE_FP32_IEEE:
            pv = tl.dot(p.to(v.dtype), v, input_precision="ieee", out_dtype=tl.float32)
        else:
            pv = tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)

        acc = acc * alpha[:, None] + pv
        l_prev = l_prev * alpha + l_curr
        m_prev = m_curr

    segm_output_offset = (
        query_offset_0[:, None].to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_offset_1[:, None].to(tl.int64) * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + segm_idx.to(tl.int64) * HEAD_SIZE_PADDED
        + offs_d64[None, :]
    )
    tl.store(
        segm_output_ptr + segm_output_offset,
        acc,
        mask=dim_mask[None, :] & row_valid[:, None],
    )

    segm_offset = (
        query_offset_0.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_offset_1.to(tl.int64) * NUM_SEGMENTS_PER_SEQ
        + segm_idx.to(tl.int64)
    )
    tl.store(segm_max_ptr + segm_offset, m_prev, mask=row_valid)
    tl.store(segm_expsum_ptr + segm_offset, l_prev, mask=row_valid)


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
    NUM_SEGMENTS_PADDED: tl.constexpr,
):
    """Reduce segment-level outputs from the 3D kernel into final attention output."""
    LOG2E: tl.constexpr = 1.4426950408889634

    query_token_idx = tl.program_id(0)
    query_head_idx = tl.program_id(1)

    num_seqs = num_seqs.to(tl.int32)
    seq_idx = _find_seq_idx(query_start_len_ptr, query_token_idx, num_seqs, BLOCK_Q, False)

    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_d64 = offs_d.to(tl.int64)
    dim_mask = (offs_d < HEAD_SIZE).to(tl.int1)
    out_offset = (
        query_token_idx.to(tl.int64) * output_stride_0
        + query_head_idx.to(tl.int64) * output_stride_1
        + offs_d64
    )

    seq_len = tl.load(seq_lens_ptr + seq_idx).to(tl.int32)
    if seq_len <= 0:
        tl.store(
            out_ptr + out_offset,
            tl.zeros([HEAD_SIZE_PADDED], dtype=tl.float32).to(out_ptr.dtype.element_ty),
            mask=dim_mask,
        )
        return

    tiles_per_segment = tl.maximum(_cdiv(seq_len, NUM_SEGMENTS_PER_SEQ * TILE_SIZE), 1)
    act_num_segments = tl.minimum(_cdiv(seq_len, tiles_per_segment * TILE_SIZE), NUM_SEGMENTS_PER_SEQ)

    offs_seg = tl.arange(0, NUM_SEGMENTS_PADDED)
    segm_mask = (offs_seg < act_num_segments) & (offs_seg < NUM_SEGMENTS_PER_SEQ)

    segm_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ)
        + query_head_idx.to(tl.int64) * NUM_SEGMENTS_PER_SEQ
        + offs_seg.to(tl.int64)
    )
    segm_max = tl.load(segm_max_ptr + segm_offset, mask=segm_mask, other=float("-inf"))
    overall_max = tl.max(segm_max, axis=0)

    valid_max = segm_mask & (segm_max > float("-inf")) & (overall_max > float("-inf"))
    segm_scale = tl.where(
        valid_max,
        tl.math.exp2(tl.where(valid_max, (segm_max - overall_max) * LOG2E, float("-inf"))),
        0.0,
    )

    segm_expsum = tl.load(segm_expsum_ptr + segm_offset, mask=segm_mask, other=0.0)
    segm_expsum = segm_expsum * segm_scale
    overall_expsum = tl.sum(segm_expsum, axis=0)

    segm_output_offset = (
        query_token_idx.to(tl.int64) * (num_query_heads * NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + query_head_idx.to(tl.int64) * (NUM_SEGMENTS_PER_SEQ * HEAD_SIZE_PADDED)
        + offs_seg.to(tl.int64)[:, None] * HEAD_SIZE_PADDED
        + offs_d64[None, :]
    )
    segm_output = tl.load(segm_output_ptr + segm_output_offset, mask=segm_mask[:, None] & dim_mask[None, :], other=0.0)
    segm_output = segm_output * segm_scale[:, None]
    acc_sum = tl.sum(segm_output, axis=0)
    acc = tl.where(overall_expsum > 0.0, acc_sum / tl.where(overall_expsum > 0.0, overall_expsum, 1.0), 0.0)

    tl.store(out_ptr + out_offset, acc.to(out_ptr.dtype.element_ty), mask=dim_mask)


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
    """Launch unified attention Triton kernels for prefill and decode phases."""
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

    head_size_padded = max(16, triton.next_power_of_2(head_size))
    use_fp32_ieee = key_cache.dtype == jnp.float32

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
        or int(num_par_softmax_segments) <= 1
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
        USE_FP32_IEEE=bool(use_fp32_ieee),
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
            zeroed_outputs=(0,),
            grid=lambda META: (int(total_num_q_blocks), int(num_kv_heads)),
            name="ejkernel::triton::unified_attention_2d",
            **metaparams_2d,
        )
        return out

    num_segments = int(num_par_softmax_segments)
    num_segments_padded = max(1, int(triton.next_power_of_2(num_segments)))
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
        USE_FP32_IEEE=bool(use_fp32_ieee),
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
        NUM_SEGMENTS_PADDED=num_segments_padded,
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
