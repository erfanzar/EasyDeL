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

"""Ragged paged attention v3 with inline KV cache updates for Triton.

This module implements an advanced paged attention variant that performs
KV cache updates inline with attention computation using a 5D packed
KV cache format.

Core Algorithm:
--------------
1. Scatter only active tokens (strictly masked by ``distribution[2]``) into
   the paged KV cache using ``lax.GatherScatterMode.FILL_OR_DROP`` so padded
   tokens never corrupt live cache slots.
2. Compute causal attention over the updated cache with runtime page loops,
   native fp16/bf16 Tensor Core ``tl.dot`` with fp32 accumulation, overflow-safe
   ``exp(-2|x|)`` softcap tanh, int64 pointer offsets, and early exits for
   inactive sequences and out-of-range query blocks.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import triton
import triton.language as tl
from jax import lax

from ejkernel.callib import triton_call

DEFAULT_MASK_VALUE = -2.381976426469702e38


def _align_to(x: int, multiple: int) -> int:
    """Round up x to the nearest multiple of the given alignment value."""
    return ((int(x) + int(multiple) - 1) // int(multiple)) * int(multiple)


def _dtype_packing(dtype: jnp.dtype) -> int:
    """Determine the packing factor (elements per 32-bit word) for a given dtype."""
    bw = jnp.dtype(dtype).itemsize * 8
    if bw not in (16, 32):
        raise ValueError(f"Only 16/32-bit floats supported for packing, got {dtype} ({bw} bits).")
    return 32 // bw


def _merge_kv(
    keys: jax.Array,
    values: jax.Array,
    *,
    head_dim_padded: int,
    kv_pair_in_head_dim: bool,
) -> jax.Array:
    """Interleave and pack keys and values into the 4D cache slot format."""
    if keys.shape != values.shape or keys.dtype != values.dtype:
        raise ValueError("keys/values mismatch")
    total_tokens, num_kv_heads, head_dim = map(int, keys.shape)
    pack = _dtype_packing(keys.dtype)
    if kv_pair_in_head_dim:
        padded_heads = _align_to(num_kv_heads, pack)
        kv = jnp.pad(
            jnp.concatenate([keys, values], axis=-1),
            ((0, 0), (0, padded_heads - num_kv_heads), (0, 0)),
            constant_values=0,
        )
        return kv.reshape(total_tokens, padded_heads // pack, pack, head_dim * 2)

    num_kv_heads_x2 = _align_to(num_kv_heads * 2, pack)
    kv = jnp.pad(
        jnp.concatenate([keys, values], axis=-1).reshape(total_tokens, num_kv_heads * 2, head_dim),
        (
            (0, 0),
            (0, num_kv_heads_x2 - num_kv_heads * 2),
            (0, head_dim_padded - head_dim),
        ),
        constant_values=0,
    ).reshape(total_tokens, num_kv_heads_x2 // pack, pack, head_dim_padded)
    return kv


@triton.jit
def _tanh_safe(x):
    """Overflow-safe tanh(x) computed via exp(-2|x|)."""
    ax = tl.abs(x)
    z = tl.exp(-2.0 * ax)
    t = (1.0 - z) / (1.0 + z)
    return tl.where(x >= 0.0, t, -t)


@triton.jit
def _rpa_v3_attn_fwd(
    q_ptr,
    kv_ptr,
    block_tables_ptr,
    kv_lens_ptr,
    cu_q_lens_ptr,
    distribution_ptr,
    sink_ptr,
    softmax_scale,
    logits_soft_cap,
    sliding_window,
    q_scale,
    k_scale,
    v_scale,
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
    PAGES_PER_SEQ: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_NPAGES: tl.constexpr,
    KV_BLOCK_TOKENS: tl.constexpr,
    KV_BLOCK_TOKENS_PADDED: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    KV_PAIR_IN_HEAD_DIM: tl.constexpr,
    USE_FP32_IEEE: tl.constexpr,
    HAS_SINK: tl.constexpr,
    HAS_SLIDING: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    HAS_Q_SCALE: tl.constexpr,
    HAS_K_SCALE: tl.constexpr,
    HAS_V_SCALE: tl.constexpr,
):
    """Ragged paged attention v3 forward Triton kernel."""
    pid_qb = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    num_seqs = tl.load(distribution_ptr + 2).to(tl.int32)
    if pid_s >= num_seqs:
        return

    q_start = tl.load(cu_q_lens_ptr + pid_s).to(tl.int32)
    q_end = tl.load(cu_q_lens_ptr + pid_s + 1).to(tl.int32)
    q_len = q_end - q_start
    if pid_qb * BLOCK_M >= q_len:
        return

    kv_len = tl.load(kv_lens_ptr + pid_s).to(tl.int32)
    context_len = kv_len - q_len

    qpos_lo = pid_qb * BLOCK_M
    qpos_hi = tl.minimum(qpos_lo + BLOCK_M - 1, q_len - 1)
    offs_m = qpos_lo + tl.arange(0, BLOCK_M)
    row_mask = offs_m < q_len

    offs_d = tl.arange(0, BLOCK_DMODEL)
    offs_d64 = offs_d.to(tl.int64)
    d_mask = offs_d < head_dim

    pid_h64 = pid_h.to(tl.int64)
    q_tok64 = (q_start + offs_m).to(tl.int64)
    q_ptrs = q_ptr + q_tok64[:, None] * q_stride_m + pid_h64 * q_stride_h + offs_d64[None, :] * q_stride_d
    q = tl.load(q_ptrs, mask=row_mask[:, None] & d_mask[None, :], other=0.0)
    if HAS_Q_SCALE:
        q = (q.to(tl.float32) / q_scale).to(kv_ptr.dtype.element_ty)
    elif q.dtype != kv_ptr.dtype.element_ty:
        q = q.to(kv_ptr.dtype.element_ty)

    kv_head = pid_h // NUM_REPEATS
    kv_head64 = kv_head.to(tl.int64)
    if KV_PAIR_IN_HEAD_DIM:
        k_c_off = kv_head64 * kv_stride_c
        v_c_off = kv_head64 * kv_stride_c
        v_d_offs64 = (offs_d + head_dim).to(tl.int64) * kv_stride_d
    else:
        k_c_off = (2 * kv_head64) * kv_stride_c
        v_c_off = (2 * kv_head64 + 1) * kv_stride_c
        v_d_offs64 = offs_d64 * kv_stride_d
    k_d_offs64 = offs_d64 * kv_stride_d

    row_idx = tl.where(row_mask, context_len + offs_m, 0).to(tl.int32)
    if HAS_SLIDING:
        left_bound = tl.maximum(row_idx - sliding_window + 1, 0).to(tl.int32)
    else:
        left_bound = tl.zeros([BLOCK_M], tl.int32)

    if HAS_SINK:
        sink_val = tl.load(sink_ptr + pid_h).to(tl.float32)
        m = tl.full([BLOCK_M], sink_val, tl.float32)
        l = tl.full([BLOCK_M], 1.0, tl.float32)
    else:
        m = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], tl.float32)

    max_kv = tl.minimum(context_len + qpos_hi + 1, kv_len)
    end_pages = tl.minimum((tl.maximum(max_kv, 0) + PAGE_SIZE - 1) // PAGE_SIZE, PAGES_PER_SEQ)
    if HAS_SLIDING:
        min_key = tl.maximum(context_len + qpos_lo - sliding_window + 1, 0)
        start_page = tl.minimum(min_key // PAGE_SIZE, end_pages)
    else:
        start_page = 0

    offs_k = tl.arange(0, KV_BLOCK_TOKENS_PADDED).to(tl.int32)
    in_tile = offs_k < KV_BLOCK_TOKENS
    bt_row_base = pid_s.to(tl.int64) * bt_stride_s

    for page_base in range(start_page, end_pages, BLOCK_NPAGES):
        kv_pos = (page_base * PAGE_SIZE) + offs_k
        kv_valid = in_tile & (kv_pos < kv_len)

        page_idx = page_base + (offs_k // PAGE_SIZE)
        off_in_page = (offs_k % PAGE_SIZE).to(tl.int64)

        bt_mask = in_tile & (page_idx < PAGES_PER_SEQ) & kv_valid
        bt_ptrs = block_tables_ptr + bt_row_base + page_idx.to(tl.int64) * bt_stride_p
        page_id = tl.load(bt_ptrs, mask=bt_mask, other=0).to(tl.int64)

        base_page = kv_ptr + page_id[:, None] * kv_stride_p + off_in_page[:, None] * kv_stride_s
        k_ptrs = base_page + k_c_off + k_d_offs64[None, :]
        v_ptrs = base_page + v_c_off + v_d_offs64[None, :]

        kv_load_mask = kv_valid[:, None] & d_mask[None, :]
        k_block = tl.load(k_ptrs, mask=kv_load_mask, other=0.0)
        v_block = tl.load(v_ptrs, mask=kv_load_mask, other=0.0)

        if USE_FP32_IEEE:
            logits = tl.dot(q, tl.trans(k_block), input_precision="ieee", out_dtype=tl.float32) * softmax_scale
        else:
            logits = tl.dot(q, tl.trans(k_block), out_dtype=tl.float32) * softmax_scale
        if HAS_K_SCALE:
            logits = logits * k_scale
        if HAS_Q_SCALE:
            logits = logits * q_scale
        if HAS_SOFTCAP:
            logits = logits_soft_cap * _tanh_safe(logits / logits_soft_cap)

        allowed = (kv_pos[None, :] <= row_idx[:, None]) & (kv_pos[None, :] >= left_bound[:, None])
        mask = row_mask[:, None] & kv_valid[None, :] & allowed
        logits_masked = tl.where(mask, logits, float("-inf"))

        block_max = tl.max(logits_masked, axis=1)
        has_any = block_max > float("-inf")
        new_m = tl.where(has_any, tl.maximum(m, block_max), m)
        exp_m = tl.where(has_any & (m > float("-inf")), tl.exp(m - new_m), tl.where(has_any, 0.0, 1.0))

        exp_logits = tl.where(mask, tl.exp(tl.where(mask, logits - new_m[:, None], float("-inf"))), 0.0)
        l = exp_m * l + tl.sum(exp_logits, axis=1)

        if USE_FP32_IEEE:
            pv = tl.dot(exp_logits.to(v_block.dtype), v_block, input_precision="ieee", out_dtype=tl.float32)
        else:
            pv = tl.dot(exp_logits.to(v_block.dtype), v_block, out_dtype=tl.float32)

        acc = exp_m[:, None] * acc + pv
        m = new_m

    l_safe = tl.where(l > 0.0, l, 1.0)
    out = acc / l_safe[:, None]
    if HAS_V_SCALE:
        out = out * v_scale

    o_ptrs = o_ptr + q_tok64[:, None] * o_stride_m + pid_h64 * o_stride_h + offs_d64[None, :] * o_stride_d
    tl.store(o_ptrs, out.to(o_ptr.dtype.element_ty), mask=row_mask[:, None] & d_mask[None, :])


def _contig_strides_3(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    """Compute contiguous strides for a 3D tensor shape (M, H, D)."""
    _m, h, d = map(int, shape)
    return (h * d, d, 1)


def _contig_strides_4(shape: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Compute contiguous strides for a 4D tensor shape (P, S, C, D)."""
    _p, s, c, d = map(int, shape)
    return (s * c * d, c * d, d, 1)


def _kv_update_scatter(
    *,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    kv_pair_in_head_dim: bool,
) -> jax.Array:
    """Scatter only active key-value pairs into the paged KV cache.

    Padded tokens beyond ``query_start_loc[distribution[2]]`` (or belonging to
    inactive sequences ``s >= distribution[2]``) are mapped to an out-of-bounds
    slot and dropped via ``lax.GatherScatterMode.FILL_OR_DROP``, preventing any
    cache slot corruption.
    """
    total_tokens, _num_kv_heads, _head_dim = map(int, keys.shape)
    num_pages, page_size, _hx2_per_pack, _pack, head_dim_padded = map(int, kv_cache.shape)

    max_num_seqs = int(kv_lens.shape[0])
    pages_per_seq = int(block_tables.shape[0]) // max_num_seqs

    merged_kv = _merge_kv(
        keys,
        values,
        head_dim_padded=head_dim_padded,
        kv_pair_in_head_dim=kv_pair_in_head_dim,
    )
    kv_cache_flat = kv_cache.reshape(-1, *kv_cache.shape[2:])

    num_seqs = jnp.clip(distribution[2], jnp.int32(0), jnp.int32(max_num_seqs))
    active_tokens = query_start_loc[num_seqs]

    t_idx = jnp.arange(total_tokens, dtype=jnp.int32)
    s_idx = jnp.searchsorted(query_start_loc, t_idx, side="right") - jnp.int32(1)
    s_idx_clipped = jnp.clip(s_idx, jnp.int32(0), jnp.int32(max_num_seqs - 1))

    q_start = query_start_loc[s_idx_clipped]
    q_end = query_start_loc[s_idx_clipped + 1]
    q_len = q_end - q_start
    kv_len = kv_lens[s_idx_clipped]

    pos_in_new = t_idx - q_start
    pos_in_kv = (kv_len - q_len) + pos_in_new

    page_idx_in_seq = pos_in_kv // jnp.int32(page_size)
    page_offset = pos_in_kv - page_idx_in_seq * jnp.int32(page_size)
    flat_block_idx = s_idx_clipped * jnp.int32(pages_per_seq) + page_idx_in_seq
    flat_block_idx_safe = jnp.clip(flat_block_idx, jnp.int32(0), jnp.int32(int(block_tables.shape[0]) - 1))
    physical_page = block_tables[flat_block_idx_safe]

    token_valid = (
        (t_idx < active_tokens)
        & (s_idx >= jnp.int32(0))
        & (s_idx < num_seqs)
        & (pos_in_kv >= jnp.int32(0))
        & (pos_in_kv < kv_len)
        & (page_idx_in_seq >= jnp.int32(0))
        & (page_idx_in_seq < jnp.int32(pages_per_seq))
        & (physical_page >= jnp.int32(0))
        & (physical_page < jnp.int32(num_pages))
    )

    total_slots = jnp.int32(num_pages * page_size)
    scatter_row = jnp.where(
        token_valid,
        physical_page * jnp.int32(page_size) + page_offset,
        total_slots,
    )
    scatter_idx = scatter_row[:, None]

    dnums = lax.ScatterDimensionNumbers(
        update_window_dims=(1, 2, 3),
        inserted_window_dims=(0,),
        scatter_dims_to_operand_dims=(0,),
    )
    kv_cache_flat = lax.scatter(
        kv_cache_flat,
        scatter_idx,
        merged_kv,
        dnums,
        mode=lax.GatherScatterMode.FILL_OR_DROP,
    )
    return kv_cache_flat.reshape(kv_cache.shape)


def ragged_paged_attention_triton(
    queries: jax.Array,
    keys: jax.Array,
    values: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    softmax_aux: jax.Array | None = None,
    *,
    softmax_scale: float = 1.0,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
    num_warps: int | None = None,
    num_stages: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Execute ragged paged attention v3 with inline KV cache update."""
    del chunk_prefill_size, vmem_limit_bytes

    if softmax_scale is None:
        softmax_scale = queries.shape[-1] ** -0.5

    if queries.ndim != 3 or keys.ndim != 3 or values.ndim != 3:
        raise ValueError("queries/keys/values must be rank-3 arrays")
    if keys.shape != values.shape:
        raise ValueError("keys and values must have the same shape")
    if queries.shape[0] != keys.shape[0] or queries.shape[2] != keys.shape[2]:
        raise ValueError("queries/keys/values must agree on (total_tokens, head_dim)")
    if kv_cache.ndim != 5:
        raise ValueError("kv_cache must be rank-5")
    if kv_lens.dtype != jnp.int32 or block_tables.dtype != jnp.int32 or query_start_loc.dtype != jnp.int32:
        raise ValueError("kv_lens/block_tables/query_start_loc must be int32")
    if query_start_loc.shape != (kv_lens.shape[0] + 1,):
        raise ValueError("query_start_loc must have shape (max_num_seqs + 1,)")
    if distribution.dtype != jnp.int32 or distribution.shape != (3,):
        raise ValueError("distribution must be int32 with shape (3,)")
    if kv_cache.dtype != keys.dtype:
        raise ValueError("kv_cache dtype must match keys/values dtype for the Triton implementation")

    total_tokens, num_q_heads, head_dim = map(int, queries.shape)
    _num_pages, page_size, hx2_per_pack, pack, head_dim_padded = map(int, kv_cache.shape)
    max_num_seqs = int(kv_lens.shape[0])
    if int(block_tables.shape[0]) % max_num_seqs != 0:
        raise ValueError("block_tables length must be divisible by max_num_seqs")
    pages_per_seq = int(block_tables.shape[0]) // max_num_seqs

    num_kv_heads = int(keys.shape[1])
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads")
    num_repeats = num_q_heads // num_kv_heads

    combined_heads = hx2_per_pack * pack
    expected_pack = _dtype_packing(kv_cache.dtype)
    if pack != expected_pack:
        raise ValueError(f"kv_cache packing mismatch: got pack={pack}, expected={expected_pack}")
    if page_size % pack != 0:
        raise ValueError("page_size must be divisible by kv_packing")

    expected_h64 = _align_to(num_kv_heads, pack)
    expected_interleaved = _align_to(num_kv_heads * 2, pack)
    kv_pair_in_head_dim = head_dim == 64 and combined_heads == expected_h64
    if not kv_pair_in_head_dim and combined_heads != expected_interleaved:
        raise ValueError(
            f"kv_cache head packing mismatch: combined_heads={combined_heads}, "
            f"expected={expected_h64 if head_dim == 64 else expected_interleaved} "
            f"(num_kv_heads={num_kv_heads}, head_dim={head_dim}, pack={pack})"
        )
    if head_dim_padded < head_dim or head_dim_padded != _align_to(head_dim, 128):
        raise ValueError("kv_cache head_dim_padded must equal align_to(head_dim, 128)")

    kv_cache = _kv_update_scatter(
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        distribution=distribution,
        kv_pair_in_head_dim=kv_pair_in_head_dim,
    )

    kv_pages = kv_cache.reshape(_num_pages, page_size, combined_heads, head_dim_padded)

    raw_block_m = 128 if num_queries_per_block is None else int(num_queries_per_block)
    block_m = max(triton.next_power_of_2(raw_block_m), 16)

    raw_block_npages = 16 if num_kv_pages_per_block is None else int(num_kv_pages_per_block)
    block_npages = max(1, min(pages_per_seq, raw_block_npages, 32))

    block_dmodel = max(triton.next_power_of_2(head_dim), 16)
    kv_block_tokens = block_npages * page_size
    kv_block_tokens_padded = max(triton.next_power_of_2(kv_block_tokens), 16)

    if head_dim < 16:
        raise ValueError(f"ragged_page_attention_v3 (triton) requires head_dim>=16; got head_dim={head_dim}.")

    q_sm, q_sh, q_sd = _contig_strides_3(queries.shape)
    kv_sp, kv_ss, kv_sc, kv_sd = _contig_strides_4(kv_pages.shape)
    bt_ss, bt_sp = pages_per_seq, 1
    o_sm, o_sh, o_sd = _contig_strides_3(queries.shape)

    qblocks_max = math.ceil(total_tokens / block_m)

    has_sink = softmax_aux is not None
    if softmax_aux is None:
        softmax_aux_arr = jnp.zeros((1,), dtype=jnp.float32)
    else:
        if softmax_aux.shape != (num_q_heads,):
            raise ValueError("softmax_aux must have shape (num_q_heads,)")
        softmax_aux_arr = softmax_aux.astype(jnp.float32)

    has_sliding = sliding_window is not None
    sliding_window_val = int(sliding_window or 0)

    has_softcap = logits_soft_cap is not None and float(logits_soft_cap) > 0.0
    logits_soft_cap_val = float(logits_soft_cap) if has_softcap else 0.0

    has_q_scale = q_scale is not None
    q_scale_val = float(q_scale or 1.0)

    has_k_scale = k_scale is not None
    k_scale_val = float(k_scale or 1.0)

    has_v_scale = v_scale is not None
    v_scale_val = float(v_scale or 1.0)

    resolved_warps = 4 if num_warps is None else int(num_warps)
    resolved_stages = 2 if num_stages is None else int(num_stages)
    use_fp32_ieee = kv_cache.dtype == jnp.float32

    out_shape = [jax.ShapeDtypeStruct(queries.shape, queries.dtype)]
    metaparams = dict(
        NUM_REPEATS=num_repeats,
        PAGES_PER_SEQ=pages_per_seq,
        PAGE_SIZE=page_size,
        BLOCK_M=block_m,
        BLOCK_NPAGES=block_npages,
        KV_BLOCK_TOKENS=kv_block_tokens,
        KV_BLOCK_TOKENS_PADDED=kv_block_tokens_padded,
        BLOCK_DMODEL=block_dmodel,
        KV_PAIR_IN_HEAD_DIM=bool(kv_pair_in_head_dim),
        USE_FP32_IEEE=bool(use_fp32_ieee),
        HAS_SINK=bool(has_sink),
        HAS_SLIDING=bool(has_sliding),
        HAS_SOFTCAP=bool(has_softcap),
        HAS_Q_SCALE=bool(has_q_scale),
        HAS_K_SCALE=bool(has_k_scale),
        HAS_V_SCALE=bool(has_v_scale),
        num_warps=resolved_warps,
        num_stages=resolved_stages,
    )

    (out,) = triton_call(
        queries,
        kv_pages,
        block_tables,
        kv_lens,
        query_start_loc,
        distribution,
        softmax_aux_arr,
        float(softmax_scale),
        float(logits_soft_cap_val),
        int(sliding_window_val),
        float(q_scale_val),
        float(k_scale_val),
        float(v_scale_val),
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
        kernel=_rpa_v3_attn_fwd,
        out_shape=out_shape,
        zeroed_outputs=(0,),
        grid=lambda META: (qblocks_max, num_q_heads, max_num_seqs),
        name="ejkernel::triton::ragged_page_attention_v3_fwd",
        **metaparams,
    )
    return out, kv_cache
