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

"""Paged attention kernels with optional partitioning for Triton.

This module provides Triton kernels for paged attention computation, supporting
both single-pass and partitioned attention modes.
"""

import triton
import triton.language as tl


@triton.jit
def _tanh_safe(x):
    """Overflow-safe tanh(x) computed via exp(-2|x|)."""
    ax = tl.abs(x)
    z = tl.exp(-2.0 * ax)
    t = (1.0 - z) / (1.0 + z)
    return tl.where(x >= 0.0, t, -t)


@triton.jit
def _paged_attn_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    context_lens_ptr,
    block_tables_ptr,
    m_i_ptr,
    l_i_ptr,
    out_ptr,
    attn_scale,
    softcap,
    stride_bt0,
    stride_bt1,
    stride_q0,
    stride_q1,
    stride_q2,
    stride_kv0,
    stride_kv1,
    stride_kv2,
    stride_kv3,
    stride_o0,
    stride_o1,
    stride_o2,
    stride_o3,
    stride_o4,
    HEAD_SIZE: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    PADDED_QUERY_GROUP_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    USE_FP32_IEEE: tl.constexpr,
):
    """Paged attention Triton kernel with optional context partitioning."""
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    part_idx = tl.program_id(2)
    max_num_partitions = tl.num_programs(2)

    seq_idx64 = seq_idx.to(tl.int64)
    kv_head_idx64 = kv_head_idx.to(tl.int64)
    part_idx64 = part_idx.to(tl.int64)
    max_num_partitions64 = max_num_partitions.to(tl.int64)

    log2e: tl.constexpr = 1.4426950408889634

    USE_PARTITIONING = PARTITION_SIZE > 0
    context_len = tl.load(context_lens_ptr + seq_idx).to(tl.int32)
    if USE_PARTITIONING:
        context_start_idx = part_idx * PARTITION_SIZE
        if context_start_idx >= context_len:
            return
        context_end_idx = tl.minimum(context_start_idx + PARTITION_SIZE, context_len)
        context_range = context_end_idx - context_start_idx
        num_blocks = (context_range + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE
    else:
        context_end_idx = tl.maximum(context_len, 0)
        num_blocks = (context_end_idx + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE

    block_offset = tl.arange(0, KV_BLOCK_SIZE)
    head_offset = tl.arange(0, HEAD_SIZE)
    padding_group_offset = tl.arange(0, PADDED_QUERY_GROUP_SIZE)

    block_offset64 = block_offset.to(tl.int64)
    head_offset64 = head_offset.to(tl.int64)
    padding_group_offset64 = padding_group_offset.to(tl.int64)

    kv_offset = (
        kv_head_idx64 * stride_kv1
        + block_offset64[:, None] * stride_kv2
        + head_offset64[None, :] * stride_kv3
    )

    q_offset = (
        seq_idx64 * stride_q0
        + (kv_head_idx64 * QUERY_GROUP_SIZE + padding_group_offset64[:, None]) * stride_q1
        + head_offset64[None, :] * stride_q2
    )
    group_mask = padding_group_offset[:, None] < QUERY_GROUP_SIZE

    q = tl.load(q_ptr + q_offset, mask=group_mask, other=0.0)

    m_i = tl.full([PADDED_QUERY_GROUP_SIZE], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([PADDED_QUERY_GROUP_SIZE], dtype=tl.float32)
    acc = tl.zeros([PADDED_QUERY_GROUP_SIZE, HEAD_SIZE], dtype=tl.float32)

    if USE_PARTITIONING:
        num_prev_blocks = part_idx * (PARTITION_SIZE // KV_BLOCK_SIZE)
    else:
        num_prev_blocks = 0

    bt_row_base = seq_idx64 * stride_bt0
    for i in range(num_blocks):
        block_idx = num_prev_blocks + i
        block_number = tl.load(block_tables_ptr + bt_row_base + block_idx.to(tl.int64) * stride_bt1).to(tl.int64)

        kv_block_offset = block_number * stride_kv0 + kv_offset
        mask_offset = block_idx * KV_BLOCK_SIZE + block_offset
        # Mask against this partition's end, not the full context length, so a
        # tail block that straddles the partition boundary is not also counted
        # by the next partition (which would double-weight those positions).
        valid_k = mask_offset < context_end_idx
        kv_mask = valid_k[:, None]

        k = tl.load(k_cache_ptr + kv_block_offset, mask=kv_mask, other=0.0)

        if PADDED_QUERY_GROUP_SIZE == 1:
            qk = tl.sum(q.to(tl.float32)[:, None, :] * k.to(tl.float32)[None, :, :], axis=2)
        elif USE_FP32_IEEE:
            qk = tl.dot(q, tl.trans(k), input_precision="ieee", out_dtype=tl.float32)
        else:
            qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32)

        qk *= attn_scale
        if HAS_SOFTCAP:
            qk = softcap * _tanh_safe(qk / softcap)
        qk_masked = tl.where(valid_k[None, :], qk, float("-inf"))

        m_blk = tl.max(qk_masked, axis=1)
        has_any = m_blk > float("-inf")
        m_i_new = tl.where(has_any, tl.maximum(m_i, m_blk), m_i)

        alpha = tl.where(
            has_any & (m_i > float("-inf")),
            tl.math.exp2((m_i - m_i_new) * log2e),
            tl.where(has_any, 0.0, 1.0),
        )
        p = tl.where(
            valid_k[None, :],
            tl.math.exp2(tl.where(valid_k[None, :], (qk - m_i_new[:, None]) * log2e, float("-inf"))),
            0.0,
        )
        acc *= alpha[:, None]

        v = tl.load(v_cache_ptr + kv_block_offset, mask=kv_mask, other=0.0)

        if PADDED_QUERY_GROUP_SIZE == 1:
            acc += tl.sum(tl.trans(p)[:, :, None] * v.to(tl.float32)[:, None, :], axis=0)
        elif USE_FP32_IEEE:
            acc += tl.dot(p.to(v.dtype), v, input_precision="ieee", out_dtype=tl.float32)
        else:
            acc += tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)

        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_i_new

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    acc = acc / l_safe[:, None]

    if USE_PARTITIONING:
        part_offset = (
            (seq_idx64 * NUM_KV_HEADS + kv_head_idx64) * max_num_partitions64 * QUERY_GROUP_SIZE
            + part_idx64 * QUERY_GROUP_SIZE
            + padding_group_offset64
        )
        mask = padding_group_offset < QUERY_GROUP_SIZE
        tl.store(m_i_ptr + part_offset, m_i, mask=mask)
        tl.store(l_i_ptr + part_offset, l_i, mask=mask)

    out_offset = seq_idx64 * stride_o0
    if USE_PARTITIONING:
        out_offset += kv_head_idx64 * stride_o1
    else:
        out_offset += kv_head_idx64 * QUERY_GROUP_SIZE * stride_o1
    out_offset += (
        part_idx64 * stride_o2
        + padding_group_offset64[:, None] * stride_o3
        + head_offset64[None, :] * stride_o4
    )

    tl.store(out_ptr + out_offset, acc.to(out_ptr.dtype.element_ty), mask=group_mask)


@triton.jit
def _paged_attn_v2_reduce_kernel(
    m_i_ptr,
    l_i_ptr,
    tmp_out_ptr,
    context_lens_ptr,
    out_ptr,
    max_num_partitions,
    stride_o0,
    stride_o1,
    stride_o2,
    HEAD_SIZE: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    PADDED_QUERY_GROUP_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    PARTITION_SIZE: tl.constexpr,
    NUM_PARTITIONS: tl.constexpr,
):
    """Reduction kernel for partitioned paged attention.

    Supports non-power-of-2 ``QUERY_GROUP_SIZE`` via ``PADDED_QUERY_GROUP_SIZE``
    masking and guards ``context_len <= 0`` without division by zero.
    """
    seq_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    seq_idx64 = seq_idx.to(tl.int64)
    kv_head_idx64 = kv_head_idx.to(tl.int64)
    max_num_partitions64 = max_num_partitions.to(tl.int64)

    offs_g = tl.arange(0, PADDED_QUERY_GROUP_SIZE)
    offs_d = tl.arange(0, HEAD_SIZE)
    offs_g64 = offs_g.to(tl.int64)
    offs_d64 = offs_d.to(tl.int64)
    g_mask = offs_g < QUERY_GROUP_SIZE

    out_offset = (
        seq_idx64 * stride_o0
        + (kv_head_idx64 * QUERY_GROUP_SIZE + offs_g64[:, None]) * stride_o1
        + offs_d64[None, :] * stride_o2
    )

    context_len = tl.load(context_lens_ptr + seq_idx).to(tl.int32)
    if context_len <= 0:
        tl.store(
            out_ptr + out_offset,
            tl.zeros([PADDED_QUERY_GROUP_SIZE, HEAD_SIZE], dtype=tl.float32).to(out_ptr.dtype.element_ty),
            mask=g_mask[:, None],
        )
        return

    num_partitions = (context_len + PARTITION_SIZE - 1) // PARTITION_SIZE
    group_head_offset = offs_g64[:, None] * HEAD_SIZE + offs_d64[None, :]

    if num_partitions == 1:
        tmp_out_offset = (
            (seq_idx64 * NUM_KV_HEADS + kv_head_idx64) * max_num_partitions64 * QUERY_GROUP_SIZE * HEAD_SIZE
            + group_head_offset
        )
        tmp_out = tl.load(tmp_out_ptr + tmp_out_offset, mask=g_mask[:, None], other=0.0)
        tl.store(out_ptr + out_offset, tmp_out.to(out_ptr.dtype.element_ty), mask=g_mask[:, None])
        return

    offs_p = tl.arange(0, NUM_PARTITIONS)
    offs_p64 = offs_p.to(tl.int64)
    mask = (offs_p[:, None] < num_partitions) & g_mask[None, :]

    ml_offset = (
        (seq_idx64 * NUM_KV_HEADS + kv_head_idx64) * max_num_partitions64 * QUERY_GROUP_SIZE
        + offs_p64[:, None] * QUERY_GROUP_SIZE
        + offs_g64[None, :]
    )

    m_i = tl.load(m_i_ptr + ml_offset, mask=mask, other=float("-inf"))
    m = tl.max(m_i, axis=0)

    valid_m = mask & (m_i > float("-inf")) & (m[None, :] > float("-inf"))
    scale = tl.where(valid_m, tl.exp(tl.where(valid_m, m_i - m[None, :], float("-inf"))), 0.0)

    l_i = tl.load(l_i_ptr + ml_offset, mask=mask, other=0.0) * scale
    l = tl.sum(l_i, axis=0)
    l_safe = tl.where(l > 0.0, l, 1.0)

    r = l_i / l_safe[None, :]
    r = tl.reshape(r, (NUM_PARTITIONS, PADDED_QUERY_GROUP_SIZE, 1))

    tmp_out_offset = (
        (seq_idx64 * NUM_KV_HEADS + kv_head_idx64) * max_num_partitions64 * QUERY_GROUP_SIZE * HEAD_SIZE
        + offs_p64[:, None, None] * (QUERY_GROUP_SIZE * HEAD_SIZE)
        + offs_g64[None, :, None] * HEAD_SIZE
        + offs_d64[None, None, :]
    )

    tmp_out = tl.load(tmp_out_ptr + tmp_out_offset, mask=mask[:, :, None], other=0.0)
    out = tl.sum((tmp_out.to(tl.float32) * r), axis=0)

    tl.store(out_ptr + out_offset, out.to(out_ptr.dtype.element_ty), mask=g_mask[:, None])
