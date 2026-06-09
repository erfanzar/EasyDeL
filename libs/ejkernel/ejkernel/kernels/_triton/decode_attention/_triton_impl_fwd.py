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

"""Paged decode attention forward pass using Triton kernels.

This module implements a two-stage paged attention mechanism optimized for
decode-phase inference where each sequence has a single query token that
attends to its full KV context stored in a paged memory buffer.

Two-Stage Algorithm:
-------------------
Stage 1 (_decode_stage1):
    - Splits each sequence's KV context into NUM_KV_SPLITS partitions
    - Computes partial attention outputs and LSE values for each partition
    - Uses paged memory layout with block tables for efficient KV access
    - Supports grouped-query attention (GQA) and logit soft-capping

Stage 2 (_decode_stage2):
    - Combines partial results from all KV splits
    - Uses numerically stable log-sum-exp reduction
    - Produces final attention output and global LSE values

Paged Memory Layout:
-------------------
Keys and values are stored in a paged buffer where:
- Each page contains PAGE_SIZE consecutive tokens
- Block tables (req_to_tokens) map logical pages to physical page indices
- This allows efficient memory management for variable-length sequences

Key Features:
- Efficient for batched decode with heterogeneous sequence lengths
- Supports grouped-query attention (GQA/MQA)
- Optional logit soft-capping for numerical stability
- Parallel processing across KV splits for large contexts

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._triton.decode_attention import decode_attention_triton
    >>>
    >>> batch, num_heads, head_dim = 4, 12, 64
    >>> query = jnp.ones((batch, num_heads, head_dim))
    >>> key_buffer = jnp.ones((1024, 12, head_dim))  # Paged KV buffer
    >>> value_buffer = jnp.ones((1024, 12, head_dim))
    >>> req_to_tokens = jnp.zeros((batch, 16), dtype=jnp.int32)
    >>> seq_lens = jnp.array([128, 256, 512, 64], dtype=jnp.int32)
    >>>
    >>> output, lse = decode_attention_triton(
    ...     query=query,
    ...     key_buffer=key_buffer,
    ...     value_buffer=value_buffer,
    ...     req_to_tokens=req_to_tokens,
    ...     seq_lens=seq_lens,
    ...     softmax_scale=None,  # Defaults to 1/sqrt(head_dim)
    ...     num_kv_splits=8,
    ...     page_size=16,
    ...     logits_soft_cap=None,
    ...     num_warps=None,
    ...     num_stages=None,
    ... )
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import triton
import triton.language as tl

from ejkernel.callib import next_power_of_2, strides_from_shape, triton_call


@triton.jit
def _tanh(x):
    """Compute numerically stable tanh using sigmoid identity: tanh(x) = 2*sigmoid(2x) - 1."""
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _decode_stage1(
    q_ptr,
    k_ptr,
    v_ptr,
    req_to_tokens_ptr,
    seq_lens_ptr,
    partial_out_ptr,
    partial_lse_ptr,
    softmax_scale,
    stride_q0: tl.constexpr,
    stride_q1: tl.constexpr,
    stride_q2: tl.constexpr,
    stride_k0: tl.constexpr,
    stride_k1: tl.constexpr,
    stride_k2: tl.constexpr,
    stride_v0: tl.constexpr,
    stride_v1: tl.constexpr,
    stride_v2: tl.constexpr,
    stride_req0: tl.constexpr,
    stride_req1: tl.constexpr,
    stride_po0: tl.constexpr,
    stride_po1: tl.constexpr,
    stride_po2: tl.constexpr,
    stride_po3: tl.constexpr,
    stride_pl0: tl.constexpr,
    stride_pl1: tl.constexpr,
    stride_pl2: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    LOGIT_CAP: tl.constexpr,
):
    """Stage 1 Triton kernel: compute partial attention outputs per KV split.

    Each thread block processes one (batch, head, kv_split) combination. It loads
    the query, iterates over the assigned portion of the KV context using paged
    addressing via ``req_to_tokens``, and writes partial attention output and
    log-sum-exp values.

    Grid shape: ``(batch, num_q_heads, NUM_KV_SPLITS)``.

    Args:
        q_ptr: Query tensor pointer [batch, num_q_heads, head_dim]
        k_ptr: Key buffer pointer [total_tokens, num_kv_heads, head_dim]
        v_ptr: Value buffer pointer [total_tokens, num_kv_heads, head_dim]
        req_to_tokens_ptr: Page table pointer [batch, max_pages]
        seq_lens_ptr: Context lengths pointer [batch]
        partial_out_ptr: Output pointer for partial results
            [batch, num_q_heads, NUM_KV_SPLITS, head_dim]
        partial_lse_ptr: Output pointer for partial LSE values
            [batch, num_q_heads, NUM_KV_SPLITS]
        softmax_scale: Attention logit scaling factor
        stride_q0..q2: Query tensor strides (batch, head, head_dim)
        stride_k0..k2: Key buffer strides (token, head, head_dim)
        stride_v0..v2: Value buffer strides (token, head, head_dim)
        stride_req0..req1: Page table strides (batch, page)
        stride_po0..po3: Partial output strides
        stride_pl0..pl2: Partial LSE strides
        kv_group_num: Number of query heads per KV head (GQA ratio) (constexpr)
        BLOCK_DMODEL: Padded head dim (next power of 2 >= HEAD_DIM) (constexpr)
        BLOCK_N: KV tokens processed per inner loop iteration (constexpr)
        NUM_KV_SPLITS: Number of KV splits per sequence (constexpr)
        PAGE_SIZE: Tokens per memory page (constexpr)
        HEAD_DIM: Actual head dimension (constexpr)
        LOGIT_CAP: If > 0, applies tanh soft-cap: ``LOGIT_CAP * tanh(qk / LOGIT_CAP)``
            (constexpr)
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    split = tl.program_id(2)

    kv_head = h // kv_group_num

    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < HEAD_DIM

    b_i64 = b.to(tl.int64)
    h_i64 = h.to(tl.int64)
    kv_head_i64 = kv_head.to(tl.int64)
    offs_d_i64 = offs_d.to(tl.int64)

    q_off = b_i64 * stride_q0 + h_i64 * stride_q1 + offs_d_i64 * stride_q2
    q = tl.load(q_ptr + q_off, mask=mask_d, other=0.0).to(tl.float32)

    seq_len = tl.load(seq_lens_ptr + b).to(tl.int32)

    kv_len_per_split = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = kv_len_per_split * split
    split_end = tl.minimum(split_start + kv_len_per_split, seq_len)

    m = tl.full((), float("-inf"), dtype=tl.float32)
    l = tl.zeros((), dtype=tl.float32)
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)

    if split_end > split_start:
        for start_n in range(split_start, split_end, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < split_end

            page_idx = offs_n // PAGE_SIZE
            page_idx_i64 = page_idx.to(tl.int64)
            page_num = tl.load(
                req_to_tokens_ptr + b_i64 * stride_req0 + page_idx_i64 * stride_req1,
                mask=mask_n,
                other=0,
            ).to(tl.int32)
            kv_loc = page_num * PAGE_SIZE + (offs_n % PAGE_SIZE)
            kv_loc_i64 = kv_loc.to(tl.int64)

            k_off = kv_loc_i64[:, None] * stride_k0 + kv_head_i64 * stride_k1 + offs_d_i64[None, :] * stride_k2
            k = tl.load(k_ptr + k_off, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

            qk = tl.sum(q[None, :] * k, axis=1) * softmax_scale
            if LOGIT_CAP > 0:
                qk = LOGIT_CAP * _tanh(qk / LOGIT_CAP)
            qk = tl.where(mask_n, qk, float("-inf"))

            v_off = kv_loc_i64[:, None] * stride_v0 + kv_head_i64 * stride_v1 + offs_d_i64[None, :] * stride_v2
            v = tl.load(v_ptr + v_off, mask=mask_n[:, None] & mask_d[None, :], other=0.0).to(tl.float32)

            m_new = tl.maximum(m, tl.max(qk, axis=0))
            m_new = tl.where(m_new > float("-inf"), m_new, m)

            alpha = tl.exp(m - m_new)
            p = tl.exp(qk - m_new)

            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            l = l * alpha + tl.sum(p, axis=0)
            m = m_new

    out = tl.where(l > 0.0, acc / l, 0.0)
    lse = tl.where(l > 0.0, m + tl.log(l), float("-inf"))

    split_i64 = split.to(tl.int64)
    po_off = b_i64 * stride_po0 + h_i64 * stride_po1 + split_i64 * stride_po2 + offs_d_i64 * stride_po3
    tl.store(partial_out_ptr + po_off, out, mask=mask_d)
    pl_off = b_i64 * stride_pl0 + h_i64 * stride_pl1 + split_i64 * stride_pl2
    tl.store(partial_lse_ptr + pl_off, lse)


@triton.jit
def _decode_stage2(
    partial_out_ptr,
    partial_lse_ptr,
    out_ptr,
    lse_ptr,
    stride_po0: tl.constexpr,
    stride_po1: tl.constexpr,
    stride_po2: tl.constexpr,
    stride_po3: tl.constexpr,
    stride_pl0: tl.constexpr,
    stride_pl1: tl.constexpr,
    stride_pl2: tl.constexpr,
    stride_o0: tl.constexpr,
    stride_o1: tl.constexpr,
    stride_o2: tl.constexpr,
    stride_lse0: tl.constexpr,
    stride_lse1: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    """Stage 2 Triton kernel: combine partial results from all KV splits.

    Each thread block processes one (batch, head) combination. It loads the partial
    outputs and LSE values produced by Stage 1, performs numerically stable
    log-sum-exp reduction across all KV splits, and writes the final attention
    output and combined LSE.

    Grid shape: ``(batch, num_q_heads)``.

    Args:
        partial_out_ptr: Partial output pointer [batch, num_q_heads, NUM_KV_SPLITS, head_dim]
        partial_lse_ptr: Partial LSE pointer [batch, num_q_heads, NUM_KV_SPLITS]
        out_ptr: Final output pointer [batch, num_q_heads, head_dim]
        lse_ptr: Final LSE pointer [batch, num_q_heads]
        stride_po0..po3: Partial output strides (batch, head, split, head_dim)
        stride_pl0..pl2: Partial LSE strides (batch, head, split)
        stride_o0..o2: Output strides (batch, head, head_dim)
        stride_lse0..lse1: LSE strides (batch, head)
        BLOCK_DMODEL: Padded head dim (next power of 2 >= HEAD_DIM) (constexpr)
        NUM_KV_SPLITS: Number of KV splits to combine (constexpr)
        HEAD_DIM: Actual head dimension (constexpr)
    """
    b = tl.program_id(0)
    h = tl.program_id(1)

    b_i64 = b.to(tl.int64)
    h_i64 = h.to(tl.int64)

    offs_s = tl.arange(0, NUM_KV_SPLITS).to(tl.int64)
    lse_off = b_i64 * stride_pl0 + h_i64 * stride_pl1 + offs_s * stride_pl2
    lse_s = tl.load(partial_lse_ptr + lse_off, mask=offs_s < NUM_KV_SPLITS, other=float("-inf")).to(tl.float32)

    m = tl.max(lse_s, axis=0)
    has_any = m > float("-inf")
    w = tl.where(has_any, tl.exp(lse_s - m), 0.0)
    sum_w = tl.sum(w, axis=0)
    sum_w_safe = tl.where(sum_w > 0.0, sum_w, 1.0)
    lse_total = tl.where(has_any, m + tl.log(sum_w_safe), float("-inf"))

    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_d = offs_d < HEAD_DIM
    po_off = (
        b_i64 * stride_po0
        + h_i64 * stride_po1
        + offs_s[:, None] * stride_po2
        + offs_d[None, :].to(tl.int64) * stride_po3
    )
    mask_po = (offs_s[:, None] < NUM_KV_SPLITS) & mask_d[None, :]
    out_s = tl.load(partial_out_ptr + po_off, mask=mask_po, other=0.0).to(tl.float32)
    acc = tl.sum(out_s * w[:, None], axis=0)
    out = acc / sum_w_safe

    o_off = b_i64 * stride_o0 + h_i64 * stride_o1 + offs_d.to(tl.int64) * stride_o2
    tl.store(out_ptr + o_off, out, mask=mask_d)

    lse_store_off = b_i64 * stride_lse0 + h_i64 * stride_lse1
    tl.store(lse_ptr + lse_store_off, lse_total.to(tl.float32))


def decode_attention_triton(
    *,
    query: jax.Array,
    key_buffer: jax.Array,
    value_buffer: jax.Array,
    req_to_tokens: jax.Array,
    seq_lens: jax.Array,
    softmax_scale: float | None,
    num_kv_splits: int,
    page_size: int,
    logits_soft_cap: float | None,
    num_warps: int | None,
    num_stages: int | None,
) -> tuple[jax.Array, jax.Array]:
    """Triton implementation of paged decode attention with two-stage processing.

    Stage 1: Splits each sequence's KV context across num_kv_splits blocks and computes
    partial attention outputs and LSE values in parallel.

    Stage 2: Combines partial results using numerically stable log-sum-exp reduction.

    Args:
        query: Query vectors [batch, num_q_heads, head_dim]
        key_buffer: Paged key buffer [total_tokens, num_kv_heads, head_dim]
        value_buffer: Paged value buffer [total_tokens, num_kv_heads, head_dim]
        req_to_tokens: Page table mapping [batch, max_pages]
        seq_lens: Context length per sequence [batch]
        softmax_scale: Attention scaling factor (default: 1/√head_dim)
        num_kv_splits: Number of KV splits for parallelization
        page_size: Memory page size in tokens
        logits_soft_cap: Optional logit capping value
        num_warps: Number of Triton warps
        num_stages: Number of pipeline stages

    Returns:
        Tuple of (attention_output, lse):
            - attention_output: [batch, num_q_heads, head_dim]
            - lse: Log-sum-exp values [batch, num_q_heads] (float32)

    Raises:
        ValueError: If input shapes are inconsistent or invalid
        NotImplementedError: If head_dim > 256
    """
    if query.ndim != 3:
        raise ValueError("query must be [batch, num_q_heads, head_dim]")
    if key_buffer.ndim != 3 or value_buffer.ndim != 3:
        raise ValueError("key_buffer/value_buffer must be [total_tokens, num_kv_heads, head_dim]")
    if key_buffer.shape != value_buffer.shape:
        raise ValueError("key_buffer/value_buffer shape mismatch")
    if req_to_tokens.ndim != 2:
        raise ValueError("req_to_tokens must be [batch, max_pages]")
    if seq_lens.ndim != 1:
        raise ValueError("seq_lens must be [batch]")
    if req_to_tokens.dtype != jnp.int32 or seq_lens.dtype != jnp.int32:
        raise ValueError("req_to_tokens and seq_lens must be int32")

    batch, num_q_heads, head_dim = map(int, query.shape)
    total_tokens, num_kv_heads, head_dim_kv = map(int, key_buffer.shape)
    if head_dim_kv != head_dim:
        raise ValueError("head_dim mismatch between query and key/value buffers")
    if batch != int(req_to_tokens.shape[0]) or batch != int(seq_lens.shape[0]):
        raise ValueError("batch mismatch among query/req_to_tokens/seq_lens")

    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads (GQA)")
    kv_group_num = num_q_heads // num_kv_heads

    page_size = int(page_size)
    if page_size <= 0:
        raise ValueError("page_size must be > 0")

    if total_tokens % page_size != 0:
        raise ValueError("key/value buffer first dim must be a multiple of page_size")

    num_kv_splits = int(num_kv_splits)
    if num_kv_splits <= 0:
        raise ValueError("num_kv_splits must be > 0")

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    softmax_scale = float(softmax_scale)

    if logits_soft_cap is None:
        logits_soft_cap = 0.0
    logits_soft_cap = float(logits_soft_cap)

    block_d = int(next_power_of_2(head_dim))
    if block_d > 256:
        raise NotImplementedError("decode_attention only supports head_dim <= 256")
    block_n = 64

    stride_q0, stride_q1, stride_q2 = strides_from_shape(query.shape)
    stride_k0, stride_k1, stride_k2 = strides_from_shape(key_buffer.shape)
    stride_v0, stride_v1, stride_v2 = strides_from_shape(value_buffer.shape)
    stride_req0, stride_req1 = strides_from_shape(req_to_tokens.shape)

    partial_out_shape = jax.ShapeDtypeStruct((batch, num_q_heads, num_kv_splits, head_dim), dtype=jnp.float32)
    partial_lse_shape = jax.ShapeDtypeStruct((batch, num_q_heads, num_kv_splits), dtype=jnp.float32)
    stride_po0, stride_po1, stride_po2, stride_po3 = strides_from_shape(partial_out_shape.shape)
    stride_pl0, stride_pl1, stride_pl2 = strides_from_shape(partial_lse_shape.shape)

    def grid_stage1(meta):
        """Return the 3D grid dimensions for Stage 1: (batch, num_q_heads, num_kv_splits)."""
        return (batch, num_q_heads, num_kv_splits)

    partial_out, partial_lse = triton_call(
        query,
        key_buffer,
        value_buffer,
        req_to_tokens,
        seq_lens,
        kernel=_decode_stage1,
        out_shape=(partial_out_shape, partial_lse_shape),
        grid=grid_stage1,
        name="ejkernel::triton::decode_attention_stage1",
        num_warps=num_warps,
        num_stages=num_stages,
        softmax_scale=softmax_scale,
        stride_q0=stride_q0,
        stride_q1=stride_q1,
        stride_q2=stride_q2,
        stride_k0=stride_k0,
        stride_k1=stride_k1,
        stride_k2=stride_k2,
        stride_v0=stride_v0,
        stride_v1=stride_v1,
        stride_v2=stride_v2,
        stride_req0=stride_req0,
        stride_req1=stride_req1,
        stride_po0=stride_po0,
        stride_po1=stride_po1,
        stride_po2=stride_po2,
        stride_po3=stride_po3,
        stride_pl0=stride_pl0,
        stride_pl1=stride_pl1,
        stride_pl2=stride_pl2,
        kv_group_num=kv_group_num,
        BLOCK_DMODEL=block_d,
        BLOCK_N=block_n,
        NUM_KV_SPLITS=num_kv_splits,
        PAGE_SIZE=page_size,
        HEAD_DIM=head_dim,
        LOGIT_CAP=logits_soft_cap,
    )

    out_shape = jax.ShapeDtypeStruct((batch, num_q_heads, head_dim), dtype=query.dtype)
    lse_shape = jax.ShapeDtypeStruct((batch, num_q_heads), dtype=jnp.float32)
    stride_o0, stride_o1, stride_o2 = strides_from_shape(out_shape.shape)
    stride_lse0, stride_lse1 = strides_from_shape(lse_shape.shape)

    def grid_stage2(meta):
        """Return the 2D grid dimensions for Stage 2: (batch, num_q_heads)."""
        return (batch, num_q_heads)

    out, lse = triton_call(
        partial_out,
        partial_lse,
        kernel=_decode_stage2,
        out_shape=(out_shape, lse_shape),
        grid=grid_stage2,
        name="ejkernel::triton::decode_attention_stage2",
        num_warps=num_warps,
        num_stages=num_stages,
        stride_po0=stride_po0,
        stride_po1=stride_po1,
        stride_po2=stride_po2,
        stride_po3=stride_po3,
        stride_pl0=stride_pl0,
        stride_pl1=stride_pl1,
        stride_pl2=stride_pl2,
        stride_o0=stride_o0,
        stride_o1=stride_o1,
        stride_o2=stride_o2,
        stride_lse0=stride_lse0,
        stride_lse1=stride_lse1,
        BLOCK_DMODEL=block_d,
        NUM_KV_SPLITS=num_kv_splits,
        HEAD_DIM=head_dim,
    )

    return out, lse
