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

"""Ragged paged attention v2 with combined KV pages for Triton.

This module implements paged attention for variable-length sequences
where keys and values are stored in interleaved format within pages.
Supports both prefill and decode phases with packed query tensors.

Kernel Variants:
---------------
_ragged_paged_attn_fwd:
    Token-level kernel processing BLOCK_M queries at a time.
    Iterates over all sequences, computing attention for tokens
    belonging to each sequence based on cumulative query lengths.

_ragged_paged_attn_qblock_fwd:
    Query-block level kernel using binary search to find sequence
    boundaries. More efficient for workloads with many sequences.

Memory Layout:
-------------
- Queries: [total_tokens, num_q_heads, head_dim] (packed)
- KV Pages: [num_pages, page_size, 2*num_kv_heads, head_dim]
  Keys and values are interleaved: pages[..., 2*h, :] = keys,
                                   pages[..., 2*h+1, :] = values
- Block Tables: [num_seqs, pages_per_seq] mapping to physical pages
- Context Lens: [num_seqs] full KV lengths including context
- Query Start Loc: [num_seqs + 1] cumulative query positions

Key Features:
- Interleaved KV storage for better memory locality
- Causal masking with optional sliding window
- Logit soft-capping for numerical stability
- GQA/MQA support via NUM_REPEATS
- Optional LSE computation for gradient checkpointing

Functions:
- _ragged_paged_attn_fwd: Token-level forward kernel
- _ragged_paged_attn_qblock_fwd: Block-level forward kernel
- ragged_paged_attention_triton_call: Main entry point (token-level)
- ragged_paged_attention_triton_call_qblock: Block-level entry point
"""

import math

import jax
import triton
import triton.language as tl
from jax import numpy as jnp

from ejkernel.callib import triton_call


@triton.jit
def _ragged_paged_attn_fwd(
    q_ptr,
    kv_pages_ptr,
    block_tables_ptr,
    context_lens_ptr,
    cu_q_lens_ptr,
    softmax_scale,
    logits_soft_cap,
    total_tokens,
    num_seqs,
    num_q_heads,
    num_kv_heads,
    pages_per_seq,
    head_dim,
    total_tokens_rounded,
    window_left,
    window_right,
    q_stride_m,
    q_stride_h,
    q_stride_d,
    kv_stride_p,
    kv_stride_s,
    kv_stride_c,
    kv_stride_d,
    bt_stride_s,
    bt_stride_p,
    o_stride_m,
    o_stride_h,
    o_stride_d,
    lse_stride_h,
    lse_stride_m,
    o_ptr,
    lse_ptr,
    NUM_REPEATS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SLIDING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_NPAGES: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    MAX_NUM_SEQS: tl.constexpr,
    PAGES_PER_SEQ: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_SIZE_TILE: tl.constexpr,
    COMPUTE_LSE: tl.constexpr,
):
    """Token-level ragged paged attention forward kernel.

    Processes BLOCK_M query tokens at a time, iterating over all sequences
    and their KV pages. Uses streaming softmax for numerically stable
    attention computation without materializing full attention matrices.

    Each program instance handles one (query_block, head) pair and scans
    through all sequences, computing attention only for tokens belonging
    to the current query block.

    Args:
        q_ptr: Query tensor pointer, shape (total_tokens, num_q_heads, head_dim).
        kv_pages_ptr: KV pages pointer with interleaved K/V format.
        block_tables_ptr: Block table pointer mapping to physical pages.
        context_lens_ptr: Per-sequence KV context lengths.
        cu_q_lens_ptr: Cumulative query lengths for sequence boundaries.
        softmax_scale: Attention scaling factor.
        logits_soft_cap: Soft capping value for logits (0 if disabled).
        total_tokens: Total number of query tokens across all sequences.
        num_seqs: Number of sequences in the batch.
        num_q_heads: Number of query attention heads.
        num_kv_heads: Number of key-value attention heads.
        pages_per_seq: Maximum pages per sequence.
        head_dim: Head dimension size.
        total_tokens_rounded: Rounded total tokens for LSE storage.
        window_left, window_right: Sliding window bounds.
        *_stride_*: Tensor stride parameters.
        o_ptr: Output attention tensor pointer.
        lse_ptr: Log-sum-exp output pointer (for gradient checkpointing).
        NUM_REPEATS: GQA repeat factor (num_q_heads // num_kv_heads).
        IS_CAUSAL: Whether to apply causal masking.
        SLIDING: Whether sliding window is active.
        SOFTCAP: Whether logit soft capping is active.
        BLOCK_M: Query block size.
        BLOCK_NPAGES: Number of pages processed per inner iteration.
        BLOCK_DMODEL: Padded head dimension (power of 2).
        MAX_NUM_SEQS: Maximum number of sequences (loop bound).
        PAGES_PER_SEQ: Maximum pages per sequence (loop bound).
        PAGE_SIZE: Tokens per page.
        PAGE_SIZE_TILE: Padded page size for aligned access.
        COMPUTE_LSE: Whether to compute and store log-sum-exp values.
    """
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < total_tokens

    offs_d = tl.arange(0, BLOCK_DMODEL)
    d_mask = offs_d < head_dim

    q_ptrs = q_ptr + offs_m[:, None] * q_stride_m + pid_h * q_stride_h + offs_d[None, :] * q_stride_d
    q = tl.load(q_ptrs, mask=mask_m[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
    l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], tl.float32)

    kv_head = pid_h // NUM_REPEATS
    k_combined_idx = 2 * kv_head
    v_combined_idx = 2 * kv_head + 1

    for seq_idx in tl.static_range(0, MAX_NUM_SEQS):
        seq_active = seq_idx < num_seqs

        q_start = tl.load(cu_q_lens_ptr + seq_idx, mask=seq_active, other=0)
        q_end = tl.load(cu_q_lens_ptr + seq_idx + 1, mask=seq_active, other=0)
        kv_len = tl.load(context_lens_ptr + seq_idx, mask=seq_active, other=0)
        q_len = q_end - q_start

        row_mask = mask_m & seq_active & (offs_m >= q_start) & (offs_m < q_end)
        row_pos = offs_m - q_start
        row_idx = (kv_len - q_len) + row_pos

        if SLIDING:
            left_bound = tl.maximum(row_idx - window_left, 0)
            if IS_CAUSAL:
                right_bound = row_idx
            else:
                rb = tl.minimum(row_idx + window_right, tl.maximum(kv_len - 1, 0))
                right_bound = rb
        else:
            left_bound = tl.zeros_like(row_idx)
            if IS_CAUSAL:
                right_bound = row_idx
            else:
                right_bound = tl.full_like(row_idx, tl.maximum(kv_len - 1, 0))

        end_pages = (kv_len + PAGE_SIZE - 1) // PAGE_SIZE

        for p_blk in tl.static_range(0, PAGES_PER_SEQ, BLOCK_NPAGES):
            for j in tl.static_range(0, BLOCK_NPAGES):
                p_ind = p_blk + j
                page_active = (p_ind < end_pages) & seq_active

                page_id_ptr = block_tables_ptr + seq_idx * bt_stride_s + p_ind * bt_stride_p
                page_id = tl.load(page_id_ptr, mask=page_active, other=0)

                k_in_tile = tl.arange(0, PAGE_SIZE_TILE)
                k_mask_page = k_in_tile < PAGE_SIZE

                k_abs = p_ind * PAGE_SIZE + k_in_tile
                k_valid = k_abs < kv_len

                allowed = (k_abs[None, :] >= left_bound[:, None]) & (k_abs[None, :] <= right_bound[:, None])
                s_mask = row_mask[:, None] & page_active & k_valid[None, :] & allowed & k_mask_page[None, :]

                base_page = kv_pages_ptr + page_id * kv_stride_p
                k_ptrs = (
                    base_page
                    + k_in_tile[:, None] * kv_stride_s
                    + (k_combined_idx) * kv_stride_c
                    + offs_d[None, :] * kv_stride_d
                )
                v_ptrs = (
                    base_page
                    + k_in_tile[:, None] * kv_stride_s
                    + (v_combined_idx) * kv_stride_c
                    + offs_d[None, :] * kv_stride_d
                )

                k_tile = tl.load(
                    k_ptrs,
                    mask=(k_mask_page[:, None] & d_mask[None, :] & page_active),
                    other=0.0,
                ).to(tl.float32)
                v_tile = tl.load(
                    v_ptrs,
                    mask=(k_mask_page[:, None] & d_mask[None, :] & page_active),
                    other=0.0,
                ).to(tl.float32)

                qk = tl.dot(q, tl.trans(k_tile)) * softmax_scale

                if SOFTCAP:
                    x = qk / logits_soft_cap
                    ax = tl.where(x >= 0, x, -x)
                    z = tl.exp(-2.0 * ax)
                    tanh_x = tl.where(x >= 0, (1 - z) / (1 + z), -(1 - z) / (1 + z))
                    qk = logits_soft_cap * tanh_x

                neg_large = -1.0e30
                qk_masked = tl.where(s_mask, qk, neg_large)

                has_any_i32 = tl.max(s_mask.to(tl.int32), axis=1)
                has_any = has_any_i32 != 0

                m_curr = tl.max(qk_masked, axis=1)
                qk_minus_m = tl.where(has_any[:, None], qk_masked - m_curr[:, None], neg_large)
                s_curr = tl.exp(qk_minus_m)
                l_curr = tl.sum(s_curr, axis=1)
                qkv = tl.dot(s_curr, v_tile)

                m_next = tl.maximum(m_i, m_curr)
                m_next = tl.where(has_any, m_next, m_i)

                alpha = tl.where(has_any, tl.exp(m_i - m_next), 1.0)
                beta = tl.where(has_any, tl.exp(m_curr - m_next), 0.0)
                l_next = alpha * l_i + beta * l_curr

                o_num = (alpha[:, None] * l_i[:, None]) * acc + (beta[:, None] * qkv)
                den = tl.where(l_next[:, None] > 0, l_next[:, None], 1.0)
                o_new = o_num / den

                update_mask = row_mask & has_any
                m_i = tl.where(update_mask, m_next, m_i)
                l_i = tl.where(update_mask, l_next, l_i)
                acc = tl.where(update_mask[:, None], o_new, acc)

    o_ptrs = o_ptr + offs_m[:, None] * o_stride_m + pid_h * o_stride_h + offs_d[None, :] * o_stride_d
    tl.store(o_ptrs, acc, mask=mask_m[:, None] & d_mask[None, :])

    if COMPUTE_LSE:
        lse_vals = m_i + tl.log(l_i)
        lse_ptrs = lse_ptr + pid_h * lse_stride_h + offs_m * lse_stride_m
        lse_mask = offs_m < total_tokens_rounded
        tl.store(lse_ptrs, lse_vals, mask=lse_mask)


def _contig_strides_3(shape):
    """Compute contiguous strides for a 3D tensor shape (M, H, D)."""
    _M, H, D = shape
    return (H * D, D, 1)


def _contig_strides_4(shape):
    """Compute contiguous strides for a 4D tensor shape (P, S, C, D)."""
    _P, S, C, D = shape
    return (S * C * D, C * D, D, 1)


@triton.jit
def _cdiv_fn(x, y):
    """Compute ceiling division: ceil(x / y)."""
    return (x + y - 1) // y


@triton.jit
def _find_seq_idx(query_start_len_ptr, target_idx, num_seqs, BLOCK_Q: tl.constexpr):
    """Find the sequence index for a given global query block index using binary search.

    Maps a global query block index to the corresponding sequence index
    by searching through cumulative query lengths. This avoids iterating
    through all sequences to find which one owns a particular query block.

    Args:
        query_start_len_ptr: Pointer to cumulative query start positions.
        target_idx: Global query block index to locate.
        num_seqs: Total number of sequences.
        BLOCK_Q: Query block size (compile-time constant).

    Returns:
        Sequence index (0-based) that contains the target query block.
    """
    left = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = tl.load(query_start_len_ptr + mid)
        mid_val = val // BLOCK_Q + mid
        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1


@triton.jit
def _ragged_paged_attn_qblock_fwd(
    q_ptr,
    kv_pages_ptr,
    block_tables_ptr,
    seq_lens_ptr,
    cu_q_lens_ptr,
    softmax_scale,
    logits_soft_cap,
    num_seqs,
    num_q_heads,
    num_kv_heads,
    pages_per_seq,
    page_size,
    head_dim,
    q_stride_m,
    q_stride_h,
    q_stride_d,
    kv_stride_p,
    kv_stride_s,
    kv_stride_c,
    kv_stride_d,
    bt_stride_s,
    bt_stride_p,
    o_stride_m,
    o_stride_h,
    o_stride_d,
    o_ptr,
    NUM_REPEATS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SLIDING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    """Query-block level ragged paged attention forward kernel.

    Processes queries in fixed-size blocks (BLOCK_Q), using binary search
    to efficiently locate the sequence for each query block. More efficient
    than the token-level kernel for workloads with many sequences, as it
    avoids iterating through all sequences per program instance.

    Each program instance handles one (global_q_block, head) pair, first
    determining which sequence the block belongs to via _find_seq_idx,
    then computing attention over the relevant KV pages using FlashAttention-
    style online softmax.

    Args:
        q_ptr: Query tensor pointer, shape (total_tokens, num_q_heads, head_dim).
        kv_pages_ptr: KV pages pointer with interleaved K/V format.
        block_tables_ptr: Block table pointer mapping to physical pages.
        seq_lens_ptr: Per-sequence total lengths (context + query).
        cu_q_lens_ptr: Cumulative query lengths for sequence boundaries.
        softmax_scale: Attention scaling factor.
        logits_soft_cap: Soft capping value for logits (0 if disabled).
        num_seqs: Number of sequences in the batch.
        num_q_heads: Number of query attention heads.
        num_kv_heads: Number of key-value attention heads.
        pages_per_seq: Maximum pages per sequence.
        page_size: Tokens per page.
        head_dim: Head dimension size.
        *_stride_*: Tensor stride parameters.
        o_ptr: Output attention tensor pointer.
        NUM_REPEATS: GQA repeat factor (num_q_heads // num_kv_heads).
        IS_CAUSAL: Whether to apply causal masking.
        SLIDING: Whether sliding window is active.
        SOFTCAP: Whether logit soft capping is active.
        BLOCK_Q: Query block size.
        TILE_SIZE: KV tile size for inner loop.
        HEAD_SIZE_PADDED: Padded head dimension for aligned access.
    """
    q_block_global_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)

    seq_idx = _find_seq_idx(cu_q_lens_ptr, q_block_global_idx, num_seqs, BLOCK_Q)

    q_block_start_idx = tl.load(cu_q_lens_ptr + seq_idx) // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    q_seq_start = tl.load(cu_q_lens_ptr + seq_idx)
    q_seq_end = tl.load(cu_q_lens_ptr + seq_idx + 1)
    q_len = q_seq_end - q_seq_start

    if q_block_local_idx * BLOCK_Q >= q_len:
        return

    offs_m = tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)

    qpos = q_block_local_idx * BLOCK_Q + offs_m
    row_mask = qpos < q_len
    d_mask = offs_d < head_dim

    q_m_offset = (q_seq_start + qpos)[:, None] * q_stride_m
    q_h_offset = q_head_idx * q_stride_h
    q_d_offset = offs_d[None, :] * q_stride_d
    q_ptrs = q_ptr + q_m_offset + q_h_offset + q_d_offset
    Q = tl.load(q_ptrs, mask=row_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

    kv_head = q_head_idx // NUM_REPEATS
    k_combined_idx = 2 * kv_head
    v_combined_idx = 2 * kv_head + 1

    seq_len = tl.load(seq_lens_ptr + seq_idx)
    context_len = seq_len - q_len

    M = tl.full([BLOCK_Q], -float("inf"), tl.float32)
    L = tl.full([BLOCK_Q], 1.0, tl.float32)
    ACC = tl.zeros([BLOCK_Q, HEAD_SIZE_PADDED], tl.float32)

    qpos_hi = tl.minimum(qpos[-1], q_len - 1)
    max_seq_prefix_len = context_len + qpos_hi + 1

    num_tiles = _cdiv_fn(max_seq_prefix_len, TILE_SIZE)
    tile_start = 0
    tile_end = num_tiles

    if SLIDING:
        qpos_lo = q_block_local_idx * BLOCK_Q
        qpos_hi2 = tl.minimum(qpos_lo + BLOCK_Q - 1, q_len - 1)
        first_allowed_key = context_len + qpos_lo
        if IS_CAUSAL:
            last_allowed_key = context_len + qpos_hi2
        else:
            last_allowed_key = context_len + qpos_hi2

        tile_start = tl.maximum(0, first_allowed_key // TILE_SIZE)
        tile_end = tl.minimum((last_allowed_key // TILE_SIZE) + 1, num_tiles)

    for j in range(tile_start, tile_end):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        page_ids = tl.load(block_tables_ptr + seq_idx * bt_stride_s + (seq_offset // page_size), mask=tile_mask, other=0)
        page_ids = page_ids.to(tl.int32)

        k_in_page = (seq_offset % page_size).to(tl.int32)

        base_k = (
            kv_pages_ptr
            + page_ids[None, :] * kv_stride_p
            + k_in_page[None, :] * kv_stride_s
            + k_combined_idx * kv_stride_c
            + offs_d[:, None] * kv_stride_d
        )
        base_v = (
            kv_pages_ptr
            + page_ids[:, None] * kv_stride_p
            + k_in_page[:, None] * kv_stride_s
            + v_combined_idx * kv_stride_c
            + offs_d[None, :] * kv_stride_d
        )

        K = tl.load(base_k, mask=d_mask[:, None] & tile_mask[None, :], other=0.0).to(tl.float32)
        V = tl.load(base_v, mask=tile_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        S = tl.dot(Q, K) * softmax_scale

        if SOFTCAP:
            inv_cap = 1.0 / logits_soft_cap
            S = logits_soft_cap * tl.tanh(S * inv_cap)

        seq_mask = seq_offset[None, :] < (context_len + qpos[:, None] + 1)
        S = tl.where(row_mask[:, None] & tile_mask[None, :] & seq_mask, S, float("-inf"))

        m_j = tl.maximum(M, tl.max(S, axis=1))
        m_j = tl.where(m_j > float("-inf"), m_j, 0.0)

        P = tl.exp(S - m_j[:, None])
        l_j = tl.sum(P, axis=1)
        alpha = tl.exp(M - m_j)

        ACC = ACC * alpha[:, None]
        L = L * alpha + l_j
        M = m_j

        ACC += tl.dot(P, V)

    ACC = ACC / L[:, None]
    o_ptrs = o_ptr + (q_seq_start + qpos)[:, None] * o_stride_m + q_head_idx * o_stride_h + offs_d[None, :] * o_stride_d
    tl.store(o_ptrs, ACC, mask=row_mask[:, None] & d_mask[None, :])


def ragged_paged_attention_triton_call_qblock(
    queries: jax.Array,
    kv_pages: jax.Array,
    context_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    *,
    softmax_scale: float | None = None,
    logits_soft_cap: float | None = None,
    causal: bool = True,
    block_q: int = 64,
    tile_size: int = 32,
    num_warps: int | None = None,
    num_stages: int | None = None,
):
    """Launch the query-block level ragged paged attention Triton kernel.

    Uses binary search for sequence identification, making it more efficient
    for workloads with many sequences. Each kernel instance processes a
    fixed-size query block and locates its owning sequence dynamically.

    Args:
        queries: Packed queries of shape (total_tokens, num_q_heads, head_dim).
        kv_pages: Paged KV cache of shape (num_pages, page_size, 2*num_kv_heads, head_dim).
        context_lens: Per-sequence context lengths, shape (num_seqs,).
        block_tables: Page table of shape (num_seqs, pages_per_seq).
        query_start_loc: Cumulative query offsets, shape (num_seqs + 1,).
        softmax_scale: Attention scaling factor. Defaults to 1/sqrt(head_dim).
        logits_soft_cap: Optional logit soft capping value.
        causal: Whether to apply causal masking.
        block_q: Query block size for the kernel.
        tile_size: KV tile size for inner loop.
        num_warps: Optional Triton warps per program.
        num_stages: Optional Triton pipeline stages.

    Returns:
        Attention output of shape (total_tokens, num_q_heads, head_dim).
    """
    assert queries.ndim == 3 and kv_pages.ndim == 4
    assert queries.dtype in (jnp.float16, jnp.bfloat16) and kv_pages.dtype == queries.dtype
    assert context_lens.dtype == jnp.int32 and block_tables.dtype == jnp.int32 and query_start_loc.dtype == jnp.int32

    total_tokens, num_q_heads, head_dim = map(int, queries.shape)
    _num_pages, page_size, combined_kv_heads, head_dim_kv = map(int, kv_pages.shape)
    assert head_dim == head_dim_kv and combined_kv_heads % 2 == 0
    num_kv_heads = combined_kv_heads // 2
    assert num_q_heads % num_kv_heads == 0
    num_repeats = num_q_heads // num_kv_heads

    num_seqs, pages_per_seq = map(int, block_tables.shape)
    assert context_lens.shape[0] == num_seqs and query_start_loc.shape[0] == num_seqs + 1

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    softcap_flag = logits_soft_cap is not None
    logits_soft_cap_val = float(logits_soft_cap or 0.0)

    BLOCK_Q = int(block_q)
    TILE_SIZE = int(tile_size)
    HEAD_SIZE_PADDED = max(((head_dim + 15) // 16) * 16, 16)

    total_num_q_blocks = total_tokens // BLOCK_Q + num_seqs

    q_sm, q_sh, q_sd = _contig_strides_3(queries.shape)
    kv_sp, kv_ss, kv_sc, kv_sd = _contig_strides_4(kv_pages.shape)
    bt_ss, bt_sp = pages_per_seq, 1
    o_sm, o_sh, o_sd = _contig_strides_3(queries.shape)

    out_shape = [jax.ShapeDtypeStruct(queries.shape, queries.dtype)]

    metaparams = dict(
        NUM_REPEATS=num_repeats,
        IS_CAUSAL=bool(causal),
        SLIDING=True,
        SOFTCAP=bool(softcap_flag),
        BLOCK_Q=BLOCK_Q,
        TILE_SIZE=TILE_SIZE,
        HEAD_SIZE_PADDED=HEAD_SIZE_PADDED,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    (out,) = triton_call(
        queries,
        kv_pages,
        block_tables,
        context_lens,
        query_start_loc,
        float(softmax_scale),
        float(logits_soft_cap_val),
        int(num_seqs),
        int(num_q_heads),
        int(num_kv_heads),
        int(pages_per_seq),
        int(page_size),
        int(head_dim),
        int(q_sm),
        int(q_sh),
        int(q_sd),
        int(kv_sp),
        int(kv_ss),
        int(kv_sc),
        int(kv_sd),
        int(bt_ss),
        int(bt_sp),
        int(o_sm),
        int(o_sh),
        int(o_sd),
        kernel=_ragged_paged_attn_qblock_fwd,
        out_shape=out_shape,
        grid=lambda META: (int(total_num_q_blocks), num_q_heads),
        name="ejkernel::triton::ragged_page_attn_qblock",
        **metaparams,
    )
    return out


def ragged_paged_attention_triton_call(
    queries: jax.Array,
    kv_pages: jax.Array,
    context_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    *,
    softmax_scale: float | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    causal: bool = True,
    block_m: int = 128,
    block_npages: int = 2,
    num_warps: int | None = None,
    num_stages: int | None = None,
    compute_lse: bool = False,
):
    """Launch the token-level ragged paged attention Triton kernel.

    Processes packed variable-length queries against paged KV caches with
    interleaved key-value storage. Supports causal masking, sliding window,
    logit soft capping, and optional LSE computation.

    Args:
        queries: Packed queries of shape (total_tokens, num_q_heads, head_dim).
        kv_pages: Paged KV cache of shape (num_pages, page_size, 2*num_kv_heads, head_dim).
        context_lens: Per-sequence context lengths, shape (num_seqs,).
        block_tables: Page table of shape (num_seqs, pages_per_seq).
        query_start_loc: Cumulative query offsets, shape (num_seqs + 1,).
        softmax_scale: Attention scaling factor. Defaults to 1/sqrt(head_dim).
        sliding_window: Optional sliding window size as int or (left, right) tuple.
        logits_soft_cap: Optional logit soft capping value.
        causal: Whether to apply causal masking.
        block_m: Query block size.
        block_npages: Number of KV pages per inner iteration.
        num_warps: Optional Triton warps per program.
        num_stages: Optional Triton pipeline stages.
        compute_lse: Whether to compute log-sum-exp values for checkpointing.

    Returns:
        Attention output of shape (total_tokens, num_q_heads, head_dim).
    """
    assert queries.ndim == 3 and kv_pages.ndim == 4
    assert queries.dtype in (jnp.float16, jnp.bfloat16, jnp.float32) and kv_pages.dtype == queries.dtype
    assert context_lens.dtype == jnp.int32 and block_tables.dtype == jnp.int32 and query_start_loc.dtype == jnp.int32

    total_tokens, num_q_heads, head_dim = map(int, queries.shape)
    _num_pages, page_size, combined_kv_heads, head_dim_kv = map(int, kv_pages.shape)

    page_size_tile = ((page_size + 15) // 16) * 16
    assert head_dim == head_dim_kv and combined_kv_heads % 2 == 0
    num_kv_heads = combined_kv_heads // 2

    num_seqs, pages_per_seq = map(int, block_tables.shape)
    assert context_lens.shape[0] == num_seqs and query_start_loc.shape[0] == num_seqs + 1
    assert num_q_heads % num_kv_heads == 0
    num_repeats = num_q_heads // num_kv_heads

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    if sliding_window is None:
        window_left = 0
        window_right = 0
        sliding_flag = False
    else:
        if isinstance(sliding_window, int):
            window_left = int(sliding_window)
            window_right = 0 if causal else int(sliding_window)
        else:
            wl, wr = sliding_window
            window_left = int(wl)
            window_right = int(wr)
        assert window_left >= 0 and window_right >= 0
        sliding_flag = (window_left > 0) or (window_right > 0)

    if logits_soft_cap is None:
        logits_soft_cap_val = 0.0
        softcap_flag = False
    else:
        logits_soft_cap_val = float(logits_soft_cap)
        softcap_flag = True

    BLOCK_M = int(block_m)
    BLOCK_NPAGES = int(block_npages)
    BLOCK_DMODEL = max(triton.next_power_of_2(head_dim), 16)
    total_tokens_rounded = int(math.ceil(total_tokens / 128) * 128)

    q_sm, q_sh, q_sd = _contig_strides_3(queries.shape)
    kv_sp, kv_ss, kv_sc, kv_sd = _contig_strides_4(kv_pages.shape)
    bt_ss, bt_sp = pages_per_seq, 1
    o_sm, o_sh, o_sd = _contig_strides_3(queries.shape)
    lse_sh, lse_sm = total_tokens_rounded, 1

    out_shape = [
        jax.ShapeDtypeStruct(queries.shape, queries.dtype),
        jax.ShapeDtypeStruct((num_q_heads, total_tokens_rounded), jnp.float32),
    ]

    metaparams = dict(
        NUM_REPEATS=num_repeats,
        IS_CAUSAL=bool(causal),
        SLIDING=bool(sliding_flag),
        SOFTCAP=bool(softcap_flag),
        BLOCK_M=BLOCK_M,
        BLOCK_NPAGES=BLOCK_NPAGES,
        BLOCK_DMODEL=BLOCK_DMODEL,
        MAX_NUM_SEQS=num_seqs,
        PAGES_PER_SEQ=pages_per_seq,
        PAGE_SIZE=page_size,
        PAGE_SIZE_TILE=page_size_tile,
        COMPUTE_LSE=bool(compute_lse),
        num_warps=num_warps,
        num_stages=num_stages,
    )

    out, _lse = triton_call(
        queries,
        kv_pages,
        block_tables,
        context_lens,
        query_start_loc,
        float(softmax_scale),
        float(logits_soft_cap_val),
        int(total_tokens),
        int(num_seqs),
        int(num_q_heads),
        int(num_kv_heads),
        int(pages_per_seq),
        int(head_dim),
        int(total_tokens_rounded),
        int(window_left),
        int(window_right),
        int(q_sm),
        int(q_sh),
        int(q_sd),
        int(kv_sp),
        int(kv_ss),
        int(kv_sc),
        int(kv_sd),
        int(bt_ss),
        int(bt_sp),
        int(o_sm),
        int(o_sh),
        int(o_sd),
        int(lse_sh),
        int(lse_sm),
        kernel=_ragged_paged_attn_fwd,
        out_shape=out_shape,
        grid=lambda META: (triton.cdiv(total_tokens, META["BLOCK_M"]), num_q_heads),
        name="ejkernel::triton::ragged_page_attn",
        **metaparams,
    )
    return out
