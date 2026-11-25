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
    Uses runtime loops over active sequences (skipping non-overlapping
    sequences) and over active KV pages.

_ragged_paged_attn_qblock_fwd:
    Query-block level kernel using binary search to locate the owning
    sequence for each query block.

Memory Layout:
-------------
- Queries: [total_tokens, num_q_heads, head_dim] (packed)
- KV Pages: [num_pages, page_size, 2*num_kv_heads, head_dim]
  Keys and values are interleaved: pages[..., 2*h, :] = keys,
                                   pages[..., 2*h+1, :] = values
- Block Tables: [max_num_seqs, pages_per_seq] mapping to physical pages
- Context Lens: [max_num_seqs] full KV lengths including context
- Query Start Loc: [max_num_seqs + 1] cumulative query positions
"""

import math

import jax
import triton
import triton.language as tl
from jax import numpy as jnp

from ejkernel.callib import triton_call


@triton.jit
def _tanh_safe(x):
    """Overflow-safe tanh(x) computed via exp(-2|x|)."""
    ax = tl.abs(x)
    z = tl.exp(-2.0 * ax)
    t = (1.0 - z) / (1.0 + z)
    return tl.where(x >= 0.0, t, -t)


@triton.jit
def _ragged_paged_attn_fwd(
    q_ptr,
    kv_pages_ptr,
    block_tables_ptr,
    context_lens_ptr,
    cu_q_lens_ptr,
    num_seqs_ptr,
    sink_ptr,
    softmax_scale,
    logits_soft_cap,
    total_tokens,
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
    HAS_SINK: tl.constexpr,
    USE_FP32_IEEE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    MAX_NUM_SEQS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_SIZE_TILE: tl.constexpr,
    COMPUTE_LSE: tl.constexpr,
):
    """Token-level ragged paged attention forward kernel."""
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)

    block_q_start = pid_m * BLOCK_M
    block_q_end = tl.minimum(block_q_start + BLOCK_M, total_tokens)

    offs_m = block_q_start + tl.arange(0, BLOCK_M)
    mask_m = offs_m < total_tokens

    offs_d = tl.arange(0, BLOCK_DMODEL)
    d_mask = offs_d < head_dim

    q_ptrs = (
        q_ptr
        + offs_m.to(tl.int64)[:, None] * q_stride_m
        + pid_h.to(tl.int64) * q_stride_h
        + offs_d.to(tl.int64)[None, :] * q_stride_d
    )
    q = tl.load(q_ptrs, mask=mask_m[:, None] & d_mask[None, :], other=0.0)

    if HAS_SINK:
        sink_val = tl.load(sink_ptr + pid_h).to(tl.float32)
        m_i = tl.full([BLOCK_M], sink_val, tl.float32)
        l_i = tl.full([BLOCK_M], 1.0, tl.float32)
    else:
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], tl.float32)

    kv_head = pid_h // NUM_REPEATS
    k_combined_idx = (2 * kv_head).to(tl.int64)
    v_combined_idx = (2 * kv_head + 1).to(tl.int64)

    active_num_seqs = tl.minimum(tl.load(num_seqs_ptr).to(tl.int32), MAX_NUM_SEQS)
    k_in_tile = tl.arange(0, PAGE_SIZE_TILE)
    k_mask_page = k_in_tile < PAGE_SIZE
    k_in_tile64 = k_in_tile.to(tl.int64)
    offs_d64 = offs_d.to(tl.int64)

    for seq_idx in range(0, active_num_seqs):
        q_start = tl.load(cu_q_lens_ptr + seq_idx).to(tl.int32)
        q_end = tl.load(cu_q_lens_ptr + seq_idx + 1).to(tl.int32)

        if (q_end > block_q_start) & (q_start < block_q_end):
            kv_len = tl.load(context_lens_ptr + seq_idx).to(tl.int32)
            q_len = q_end - q_start

            row_mask = mask_m & (offs_m >= q_start) & (offs_m < q_end)
            row_pos = offs_m - q_start
            row_idx = (kv_len - q_len) + row_pos

            if SLIDING:
                left_bound = tl.maximum(row_idx - window_left, 0)
                if IS_CAUSAL:
                    right_bound = row_idx
                else:
                    right_bound = tl.minimum(row_idx + window_right, tl.maximum(kv_len - 1, 0))
            else:
                left_bound = tl.zeros_like(row_idx)
                if IS_CAUSAL:
                    right_bound = row_idx
                else:
                    right_bound = tl.full_like(row_idx, tl.maximum(kv_len - 1, 0))

            lo_pos = tl.maximum(block_q_start - q_start, 0)
            hi_pos = tl.minimum(block_q_end, q_end) - q_start - 1
            min_row_idx = (kv_len - q_len) + lo_pos
            max_row_idx = (kv_len - q_len) + hi_pos

            if IS_CAUSAL:
                max_kv_tokens = tl.minimum(max_row_idx + 1, kv_len)
            elif SLIDING:
                max_kv_tokens = tl.minimum(max_row_idx + window_right + 1, kv_len)
            else:
                max_kv_tokens = kv_len

            end_pages = tl.minimum((tl.maximum(max_kv_tokens, 0) + PAGE_SIZE - 1) // PAGE_SIZE, pages_per_seq)
            if SLIDING:
                min_key = tl.maximum(min_row_idx - window_left, 0)
                start_page = tl.minimum(min_key // PAGE_SIZE, end_pages)
            else:
                start_page = 0

            bt_row_base = seq_idx.to(tl.int64) * bt_stride_s

            for p_ind in range(start_page, end_pages):
                page_id = tl.load(block_tables_ptr + bt_row_base + p_ind.to(tl.int64) * bt_stride_p).to(tl.int64)

                k_abs = p_ind * PAGE_SIZE + k_in_tile
                k_valid = k_mask_page & (k_abs < kv_len)

                allowed = (k_abs[None, :] >= left_bound[:, None]) & (k_abs[None, :] <= right_bound[:, None])
                s_mask = row_mask[:, None] & k_valid[None, :] & allowed

                base_page = kv_pages_ptr + page_id * kv_stride_p
                k_ptrs = (
                    base_page
                    + k_in_tile64[:, None] * kv_stride_s
                    + k_combined_idx * kv_stride_c
                    + offs_d64[None, :] * kv_stride_d
                )
                v_ptrs = (
                    base_page
                    + k_in_tile64[:, None] * kv_stride_s
                    + v_combined_idx * kv_stride_c
                    + offs_d64[None, :] * kv_stride_d
                )

                kv_load_mask = k_valid[:, None] & d_mask[None, :]
                k_tile = tl.load(k_ptrs, mask=kv_load_mask, other=0.0)
                v_tile = tl.load(v_ptrs, mask=kv_load_mask, other=0.0)

                if USE_FP32_IEEE:
                    qk = tl.dot(q, tl.trans(k_tile), input_precision="ieee", out_dtype=tl.float32) * softmax_scale
                else:
                    qk = tl.dot(q, tl.trans(k_tile), out_dtype=tl.float32) * softmax_scale

                if SOFTCAP:
                    qk = logits_soft_cap * _tanh_safe(qk / logits_soft_cap)

                qk_masked = tl.where(s_mask, qk, float("-inf"))
                m_blk = tl.max(qk_masked, axis=1)
                has_any = m_blk > float("-inf")
                update_mask = row_mask & has_any

                m_next = tl.where(update_mask, tl.maximum(m_i, m_blk), m_i)
                alpha = tl.where(
                    update_mask & (m_i > float("-inf")),
                    tl.exp(m_i - m_next),
                    tl.where(update_mask, 0.0, 1.0),
                )
                p = tl.where(s_mask, tl.exp(tl.where(s_mask, qk - m_next[:, None], float("-inf"))), 0.0)

                if USE_FP32_IEEE:
                    qkv = tl.dot(p.to(v_tile.dtype), v_tile, input_precision="ieee", out_dtype=tl.float32)
                else:
                    qkv = tl.dot(p.to(v_tile.dtype), v_tile, out_dtype=tl.float32)

                l_next = alpha * l_i + tl.sum(p, axis=1)
                acc = tl.where(update_mask[:, None], acc * alpha[:, None] + qkv, acc)
                l_i = tl.where(update_mask, l_next, l_i)
                m_i = m_next

    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    out = acc / l_safe[:, None]

    o_ptrs = (
        o_ptr
        + offs_m.to(tl.int64)[:, None] * o_stride_m
        + pid_h.to(tl.int64) * o_stride_h
        + offs_d64[None, :] * o_stride_d
    )
    tl.store(o_ptrs, out.to(o_ptr.dtype.element_ty), mask=mask_m[:, None] & d_mask[None, :])

    if COMPUTE_LSE:
        lse_vals = tl.where(l_i > 0.0, m_i + tl.log(l_safe), float("-inf"))
        lse_ptrs = lse_ptr + pid_h.to(tl.int64) * lse_stride_h + offs_m.to(tl.int64) * lse_stride_m
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
    """Find the sequence index for a given global query block index using binary search."""
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
    num_seqs_ptr,
    sink_ptr,
    softmax_scale,
    logits_soft_cap,
    window_left,
    window_right,
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
    MAX_NUM_SEQS: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    SLIDING: tl.constexpr,
    SOFTCAP: tl.constexpr,
    HAS_SINK: tl.constexpr,
    USE_FP32_IEEE: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
):
    """Query-block level ragged paged attention forward kernel."""
    q_block_global_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)

    active_num_seqs = tl.minimum(tl.load(num_seqs_ptr).to(tl.int32), MAX_NUM_SEQS)
    if active_num_seqs <= 0:
        return

    seq_idx = _find_seq_idx(cu_q_lens_ptr, q_block_global_idx, active_num_seqs, BLOCK_Q)
    if (seq_idx < 0) | (seq_idx >= active_num_seqs):
        return

    q_seq_start = tl.load(cu_q_lens_ptr + seq_idx).to(tl.int32)
    q_seq_end = tl.load(cu_q_lens_ptr + seq_idx + 1).to(tl.int32)
    q_len = q_seq_end - q_seq_start

    q_block_start_idx = q_seq_start // BLOCK_Q + seq_idx
    q_block_local_idx = q_block_global_idx - q_block_start_idx

    if q_block_local_idx * BLOCK_Q >= q_len:
        return

    offs_m = tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, HEAD_SIZE_PADDED)
    offs_t = tl.arange(0, TILE_SIZE)

    qpos_lo = q_block_local_idx * BLOCK_Q
    qpos_hi = tl.minimum(qpos_lo + BLOCK_Q - 1, q_len - 1)
    qpos = qpos_lo + offs_m
    row_mask = qpos < q_len
    d_mask = offs_d < head_dim

    q_head64 = q_head_idx.to(tl.int64)
    offs_d64 = offs_d.to(tl.int64)
    q_m_offset = (q_seq_start + qpos).to(tl.int64)[:, None] * q_stride_m
    q_ptrs = q_ptr + q_m_offset + q_head64 * q_stride_h + offs_d64[None, :] * q_stride_d
    Q = tl.load(q_ptrs, mask=row_mask[:, None] & d_mask[None, :], other=0.0)

    kv_head = q_head_idx // NUM_REPEATS
    k_combined_idx = (2 * kv_head).to(tl.int64)
    v_combined_idx = (2 * kv_head + 1).to(tl.int64)

    seq_len = tl.load(seq_lens_ptr + seq_idx).to(tl.int32)
    context_len = seq_len - q_len

    if HAS_SINK:
        sink_val = tl.load(sink_ptr + q_head_idx).to(tl.float32)
        M = tl.full([BLOCK_Q], sink_val, tl.float32)
        L = tl.full([BLOCK_Q], 1.0, tl.float32)
    else:
        M = tl.full([BLOCK_Q], float("-inf"), tl.float32)
        L = tl.zeros([BLOCK_Q], tl.float32)
    ACC = tl.zeros([BLOCK_Q, HEAD_SIZE_PADDED], tl.float32)

    q_abs = context_len + qpos
    if IS_CAUSAL:
        max_seq_prefix_len = tl.minimum(context_len + qpos_hi + 1, seq_len)
    elif SLIDING:
        max_seq_prefix_len = tl.minimum(context_len + qpos_hi + window_right + 1, seq_len)
    else:
        max_seq_prefix_len = seq_len

    num_tiles = _cdiv_fn(tl.maximum(max_seq_prefix_len, 0), TILE_SIZE)
    tile_start = 0
    tile_end = num_tiles

    if SLIDING:
        first_allowed_key = tl.maximum(context_len + qpos_lo - window_left, 0)
        tile_start = tl.minimum(first_allowed_key // TILE_SIZE, num_tiles)

    bt_row_base = seq_idx.to(tl.int64) * bt_stride_s

    for j in range(tile_start, tile_end):
        seq_offset = j * TILE_SIZE + offs_t
        tile_mask = seq_offset < max_seq_prefix_len

        logical_page = seq_offset // page_size
        bt_mask = tile_mask & (logical_page < pages_per_seq)
        page_ids = tl.load(
            block_tables_ptr + bt_row_base + logical_page.to(tl.int64) * bt_stride_p,
            mask=bt_mask,
            other=0,
        ).to(tl.int64)
        k_in_page = (seq_offset % page_size).to(tl.int64)

        base_k = (
            kv_pages_ptr
            + page_ids[None, :] * kv_stride_p
            + k_in_page[None, :] * kv_stride_s
            + k_combined_idx * kv_stride_c
            + offs_d64[:, None] * kv_stride_d
        )
        base_v = (
            kv_pages_ptr
            + page_ids[:, None] * kv_stride_p
            + k_in_page[:, None] * kv_stride_s
            + v_combined_idx * kv_stride_c
            + offs_d64[None, :] * kv_stride_d
        )

        K = tl.load(base_k, mask=d_mask[:, None] & tile_mask[None, :], other=0.0)
        V = tl.load(base_v, mask=tile_mask[:, None] & d_mask[None, :], other=0.0)

        if USE_FP32_IEEE:
            S = tl.dot(Q, K, input_precision="ieee", out_dtype=tl.float32) * softmax_scale
        else:
            S = tl.dot(Q, K, out_dtype=tl.float32) * softmax_scale

        if SOFTCAP:
            S = logits_soft_cap * _tanh_safe(S / logits_soft_cap)

        if IS_CAUSAL:
            seq_mask = seq_offset[None, :] <= q_abs[:, None]
        else:
            seq_mask = seq_offset[None, :] < seq_len
        if SLIDING:
            left_bound = tl.maximum(q_abs - window_left, 0)
            seq_mask = seq_mask & (seq_offset[None, :] >= left_bound[:, None])
            if not IS_CAUSAL:
                right_bound = tl.minimum(q_abs + window_right, tl.maximum(seq_len - 1, 0))
                seq_mask = seq_mask & (seq_offset[None, :] <= right_bound[:, None])

        valid_mask = row_mask[:, None] & tile_mask[None, :] & seq_mask
        S_masked = tl.where(valid_mask, S, float("-inf"))

        m_blk = tl.max(S_masked, axis=1)
        has_any = m_blk > float("-inf")
        m_j = tl.where(has_any, tl.maximum(M, m_blk), M)
        alpha = tl.where(has_any & (M > float("-inf")), tl.exp(M - m_j), tl.where(has_any, 0.0, 1.0))

        P = tl.where(valid_mask, tl.exp(tl.where(valid_mask, S - m_j[:, None], float("-inf"))), 0.0)
        l_j = tl.sum(P, axis=1)

        if USE_FP32_IEEE:
            pv = tl.dot(P.to(V.dtype), V, input_precision="ieee", out_dtype=tl.float32)
        else:
            pv = tl.dot(P.to(V.dtype), V, out_dtype=tl.float32)

        ACC = ACC * alpha[:, None] + pv
        L = L * alpha + l_j
        M = m_j

    L_safe = tl.where(L > 0.0, L, 1.0)
    ACC = ACC / L_safe[:, None]
    o_ptrs = (
        o_ptr
        + (q_seq_start + qpos).to(tl.int64)[:, None] * o_stride_m
        + q_head64 * o_stride_h
        + offs_d64[None, :] * o_stride_d
    )
    tl.store(o_ptrs, ACC.to(o_ptr.dtype.element_ty), mask=row_mask[:, None] & d_mask[None, :])


def _parse_sliding_window(sliding_window: int | tuple[int, int] | None, causal: bool) -> tuple[int, int, bool]:
    """Normalize sliding_window specification into (window_left, window_right, sliding_flag)."""
    if sliding_window is None:
        return 0, 0, False
    if isinstance(sliding_window, int):
        window_left = max(int(sliding_window) - 1, 0)
        window_right = 0 if causal else window_left
        return window_left, window_right, True
    wl, wr = sliding_window
    window_left = int(wl)
    window_right = int(wr)
    assert window_left >= 0 and window_right >= 0
    return window_left, window_right, (window_left > 0) or (window_right > 0)


def ragged_paged_attention_triton_call_qblock(
    queries: jax.Array,
    kv_pages: jax.Array,
    context_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    num_seqs: jax.Array | int | None = None,
    *,
    softmax_scale: float | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: jax.Array | None = None,
    causal: bool = True,
    block_q: int = 64,
    tile_size: int = 32,
    num_warps: int | None = None,
    num_stages: int | None = None,
):
    """Launch the query-block level ragged paged attention Triton kernel."""
    assert queries.ndim == 3 and kv_pages.ndim == 4
    assert queries.dtype in (jnp.float16, jnp.bfloat16, jnp.float32) and kv_pages.dtype == queries.dtype
    assert context_lens.dtype == jnp.int32 and block_tables.dtype == jnp.int32 and query_start_loc.dtype == jnp.int32

    total_tokens, num_q_heads, head_dim = map(int, queries.shape)
    _num_pages, page_size, combined_kv_heads, head_dim_kv = map(int, kv_pages.shape)
    assert head_dim == head_dim_kv and combined_kv_heads % 2 == 0
    num_kv_heads = combined_kv_heads // 2
    assert num_q_heads % num_kv_heads == 0
    num_repeats = num_q_heads // num_kv_heads

    max_num_seqs, pages_per_seq = map(int, block_tables.shape)
    assert context_lens.shape[0] == max_num_seqs and query_start_loc.shape[0] == max_num_seqs + 1

    if num_seqs is None:
        num_seqs_arr = jnp.asarray([max_num_seqs], dtype=jnp.int32)
    else:
        num_seqs_arr = jnp.asarray(num_seqs, dtype=jnp.int32).reshape((1,))

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    window_left, window_right, sliding_flag = _parse_sliding_window(sliding_window, causal=causal)

    softcap_flag = logits_soft_cap is not None and float(logits_soft_cap) > 0.0
    logits_soft_cap_val = float(logits_soft_cap) if softcap_flag else 0.0

    has_sink = softmax_aux is not None
    if has_sink:
        if softmax_aux.shape != (num_q_heads,):
            raise ValueError(f"softmax_aux must have shape ({num_q_heads},), got {softmax_aux.shape}")
        sink_arr = softmax_aux.astype(jnp.float32)
    else:
        sink_arr = jnp.zeros((1,), dtype=jnp.float32)

    BLOCK_Q = max(triton.next_power_of_2(int(block_q)), 16)
    TILE_SIZE = max(triton.next_power_of_2(int(tile_size)), 16)
    HEAD_SIZE_PADDED = max(triton.next_power_of_2(head_dim), 16)

    total_num_q_blocks = total_tokens // BLOCK_Q + max_num_seqs

    q_sm, q_sh, q_sd = _contig_strides_3(queries.shape)
    kv_sp, kv_ss, kv_sc, kv_sd = _contig_strides_4(kv_pages.shape)
    bt_ss, bt_sp = pages_per_seq, 1
    o_sm, o_sh, o_sd = _contig_strides_3(queries.shape)

    out_shape = [jax.ShapeDtypeStruct(queries.shape, queries.dtype)]
    use_fp32_ieee = queries.dtype == jnp.float32

    metaparams = dict(
        NUM_REPEATS=num_repeats,
        MAX_NUM_SEQS=max_num_seqs,
        IS_CAUSAL=bool(causal),
        SLIDING=bool(sliding_flag),
        SOFTCAP=bool(softcap_flag),
        HAS_SINK=bool(has_sink),
        USE_FP32_IEEE=bool(use_fp32_ieee),
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
        num_seqs_arr,
        sink_arr,
        float(softmax_scale),
        float(logits_soft_cap_val),
        int(window_left),
        int(window_right),
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
        zeroed_outputs=(0,),
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
    num_seqs: jax.Array | int | None = None,
    *,
    softmax_scale: float | None = None,
    sliding_window: int | tuple[int, int] | None = None,
    logits_soft_cap: float | None = None,
    softmax_aux: jax.Array | None = None,
    causal: bool = True,
    block_m: int = 128,
    block_npages: int = 2,
    num_warps: int | None = None,
    num_stages: int | None = None,
    compute_lse: bool = False,
):
    """Launch the token-level ragged paged attention Triton kernel."""
    del block_npages
    assert queries.ndim == 3 and kv_pages.ndim == 4
    assert queries.dtype in (jnp.float16, jnp.bfloat16, jnp.float32) and kv_pages.dtype == queries.dtype
    assert context_lens.dtype == jnp.int32 and block_tables.dtype == jnp.int32 and query_start_loc.dtype == jnp.int32

    total_tokens, num_q_heads, head_dim = map(int, queries.shape)
    _num_pages, page_size, combined_kv_heads, head_dim_kv = map(int, kv_pages.shape)

    page_size_tile = max(triton.next_power_of_2(page_size), 16)
    assert head_dim == head_dim_kv and combined_kv_heads % 2 == 0
    num_kv_heads = combined_kv_heads // 2

    max_num_seqs, pages_per_seq = map(int, block_tables.shape)
    assert context_lens.shape[0] == max_num_seqs and query_start_loc.shape[0] == max_num_seqs + 1
    assert num_q_heads % num_kv_heads == 0
    num_repeats = num_q_heads // num_kv_heads

    if num_seqs is None:
        num_seqs_arr = jnp.asarray([max_num_seqs], dtype=jnp.int32)
    else:
        num_seqs_arr = jnp.asarray(num_seqs, dtype=jnp.int32).reshape((1,))

    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)

    window_left, window_right, sliding_flag = _parse_sliding_window(sliding_window, causal=causal)

    softcap_flag = logits_soft_cap is not None and float(logits_soft_cap) > 0.0
    logits_soft_cap_val = float(logits_soft_cap) if softcap_flag else 0.0

    has_sink = softmax_aux is not None
    if has_sink:
        if softmax_aux.shape != (num_q_heads,):
            raise ValueError(f"softmax_aux must have shape ({num_q_heads},), got {softmax_aux.shape}")
        sink_arr = softmax_aux.astype(jnp.float32)
    else:
        sink_arr = jnp.zeros((1,), dtype=jnp.float32)

    BLOCK_M = max(triton.next_power_of_2(int(block_m)), 16)
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
    use_fp32_ieee = queries.dtype == jnp.float32

    metaparams = dict(
        NUM_REPEATS=num_repeats,
        IS_CAUSAL=bool(causal),
        SLIDING=bool(sliding_flag),
        SOFTCAP=bool(softcap_flag),
        HAS_SINK=bool(has_sink),
        USE_FP32_IEEE=bool(use_fp32_ieee),
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=BLOCK_DMODEL,
        MAX_NUM_SEQS=max_num_seqs,
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
        num_seqs_arr,
        sink_arr,
        float(softmax_scale),
        float(logits_soft_cap_val),
        int(total_tokens),
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
