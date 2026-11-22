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

"""Ragged paged attention v3 with inline KV cache updates.

This module implements an advanced paged attention variant that performs
KV cache updates inline with attention computation. It uses a 5D packed
KV cache format optimized for memory efficiency and dtype packing.

Core Algorithm:
--------------
1. Scatter new keys/values into the paged KV cache using block tables
2. Compute attention over the updated cache with causal masking
3. Support optional features: sliding window, attention sinks, FP8 scaling

Memory Layout:
-------------
KV Cache (5D): [num_pages, page_size, heads_per_pack, pack_factor, head_dim_padded]
  - Packs multiple KV heads for memory-efficient access
  - Supports FP16/BF16 with 2x packing or FP32 with 1x packing
  - Head dimension padded to 128 for aligned access

Block Tables: [max_num_seqs * pages_per_seq] (flattened)
  - Maps (sequence, page_index) to physical page numbers

Distribution: [3] integer tensor with layout [num_decode, num_prefill, num_seqs]
  - num_decode: number of decode (single-token) sequences at the front
  - num_prefill: number of short prefill sequences following the decode group
  - num_seqs: total number of active sequences (index 2 is read by the kernel)
  Note: the Triton kernel only reads index 2 (num_seqs) from this tensor.

Key Features:
- Inline KV cache scatter updates (donates cache buffer)
- FP8 quantization support with Q/K/V scales
- Sliding window attention with left-bound truncation
- Attention sink integration for persistent tokens
- Logit soft-capping for bounded attention scores

Helper Functions:
- _align_to: Round up to alignment multiple
- _dtype_packing: Get packing factor for dtype
- _merge_kv: Interleave and pack keys/values
- _kv_update_scatter: Scatter new KV into cache

Main Entry:
- ragged_paged_attention_triton: Full interface with cache update
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import triton
import triton.language as tl
from jax import lax

from ejkernel.callib import ejit, triton_call

DEFAULT_MASK_VALUE = -2.381976426469702e38


def _align_to(x: int, multiple: int) -> int:
    """Round up x to the nearest multiple of the given alignment value."""
    return ((int(x) + int(multiple) - 1) // int(multiple)) * int(multiple)


def _dtype_packing(dtype: jnp.dtype) -> int:
    """Determine the packing factor for a given dtype.

    Returns how many values of the given dtype fit into a 32-bit word.
    This is used for the packed KV cache format where multiple values
    are stored per 32-bit element.

    Args:
        dtype: JAX dtype to compute packing for.

    Returns:
        Packing factor (1 for float32, 2 for float16/bfloat16).

    Raises:
        ValueError: If dtype is not 16-bit or 32-bit.
    """
    bw = jnp.dtype(dtype).itemsize * 8
    if bw not in (16, 32):
        raise ValueError(f"Only 16/32-bit floats supported for packing, got {dtype} ({bw} bits).")
    return 32 // bw


def _merge_kv(keys: jax.Array, values: jax.Array, *, head_dim_padded: int) -> jax.Array:
    """Interleave and pack keys and values into the 4D cache format.

    Concatenates keys and values along the head dimension, pads to alignment,
    and reshapes into the packed format expected by the KV cache.

    Args:
        keys: Key tensor of shape (total_tokens, num_kv_heads, head_dim).
        values: Value tensor of shape (total_tokens, num_kv_heads, head_dim).
        head_dim_padded: Padded head dimension (aligned to 128).

    Returns:
        Merged KV tensor of shape (total_tokens, heads_per_pack, pack_factor, head_dim_padded).

    Raises:
        ValueError: If keys and values have mismatched shapes or dtypes.
    """
    if keys.shape != values.shape or keys.dtype != values.dtype:
        raise ValueError("keys/values mismatch")
    total_tokens, num_kv_heads, head_dim = map(int, keys.shape)
    pack = _dtype_packing(keys.dtype)
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
    MAX_NUM_SEQS: tl.constexpr,
    PAGES_PER_SEQ: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_NPAGES: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    MASK_VALUE: tl.constexpr,
    HAS_SINK: tl.constexpr,
    HAS_SLIDING: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
    HAS_Q_SCALE: tl.constexpr,
    HAS_K_SCALE: tl.constexpr,
    HAS_V_SCALE: tl.constexpr,
):
    """Ragged paged attention v3 forward Triton kernel.

    Computes attention over a 5D packed KV cache with inline support for
    FP8 scaling, sliding window, attention sinks, and logit soft capping.
    Each program instance processes one (query_block, head, sequence) tuple.

    The kernel reads from an already-updated KV cache (scatter performed
    in Python before launch) and produces the attention output. It uses
    streaming softmax for numerically stable computation.

    Args:
        q_ptr: Query tensor pointer, shape (total_tokens, num_q_heads, head_dim).
        kv_ptr: Packed KV pages pointer (4D reshaped from 5D cache).
        block_tables_ptr: Flattened block table pointer.
        kv_lens_ptr: Per-sequence KV lengths.
        cu_q_lens_ptr: Cumulative query start positions.
        distribution_ptr: Workload distribution [num_decode, num_prefill, num_seqs].
        sink_ptr: Attention sink values pointer.
        softmax_scale: Attention scaling factor.
        logits_soft_cap: Soft capping value (0 if disabled).
        sliding_window: Sliding window size (0 if disabled).
        q_scale, k_scale, v_scale: Optional FP8 quantization scales.
        head_dim: Head dimension size.
        *_stride_*: Tensor stride parameters.
        o_ptr: Output attention tensor pointer.
        NUM_REPEATS: GQA repeat factor.
        MAX_NUM_SEQS: Maximum sequences (grid dimension).
        PAGES_PER_SEQ: Maximum pages per sequence.
        PAGE_SIZE: Tokens per page.
        BLOCK_M: Query block size.
        BLOCK_NPAGES: KV pages per inner iteration.
        BLOCK_DMODEL: Padded head dimension (power of 2).
        MASK_VALUE: Value for masked positions.
        HAS_SINK: Whether attention sinks are active.
        HAS_SLIDING: Whether sliding window is active.
        HAS_SOFTCAP: Whether logit soft capping is active.
        HAS_Q_SCALE, HAS_K_SCALE, HAS_V_SCALE: Whether FP8 scales are active.
    """
    pid_qb = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    num_seqs = tl.load(distribution_ptr + 2).to(tl.int32)
    seq_active = pid_s < num_seqs

    q_start = tl.load(cu_q_lens_ptr + pid_s, mask=seq_active, other=0).to(tl.int32)
    q_end = tl.load(cu_q_lens_ptr + pid_s + 1, mask=seq_active, other=0).to(tl.int32)
    q_len = q_end - q_start
    kv_len = tl.load(kv_lens_ptr + pid_s, mask=seq_active, other=0).to(tl.int32)

    context_len = kv_len - q_len

    offs_m = pid_qb * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = seq_active & (offs_m < q_len)

    offs_d = tl.arange(0, BLOCK_DMODEL)
    d_mask = offs_d < head_dim

    q_ptrs = q_ptr + (q_start + offs_m)[:, None] * q_stride_m + pid_h * q_stride_h + offs_d[None, :] * q_stride_d
    q = tl.load(q_ptrs, mask=row_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    if HAS_Q_SCALE:
        q = q / q_scale

    kv_head = pid_h // NUM_REPEATS
    k_idx = (2 * kv_head).to(tl.int32)
    v_idx = (2 * kv_head + 1).to(tl.int32)

    row_idx = context_len + offs_m
    row_idx = tl.where(row_mask, row_idx, 0).to(tl.int32)

    if HAS_SLIDING:
        left_bound = tl.maximum(row_idx - sliding_window + 1, 0).to(tl.int32)
    else:
        left_bound = tl.zeros([BLOCK_M], tl.int32)

    if HAS_SINK:
        sink_val = tl.load(sink_ptr + pid_h).to(tl.float32)
        m = tl.full([BLOCK_M], sink_val, tl.float32)
        l = tl.full([BLOCK_M], 1.0, tl.float32)
    else:
        m = tl.full([BLOCK_M], -float("inf"), tl.float32)
        l = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], tl.float32)

    kv_block_tokens: tl.constexpr = BLOCK_NPAGES * PAGE_SIZE
    offs_k = tl.arange(0, kv_block_tokens).to(tl.int32)

    for page_base in tl.static_range(0, PAGES_PER_SEQ, BLOCK_NPAGES):
        kv_pos = (page_base * PAGE_SIZE) + offs_k

        kv_valid = kv_pos < kv_len

        page_idx = page_base + (offs_k // PAGE_SIZE)
        off_in_page = (offs_k % PAGE_SIZE).to(tl.int32)

        bt_ptrs = block_tables_ptr + pid_s * bt_stride_s + page_idx * bt_stride_p
        page_id = tl.load(bt_ptrs, mask=seq_active & (page_idx < PAGES_PER_SEQ), other=0).to(tl.int32)

        base_page = kv_ptr + page_id[:, None] * kv_stride_p + off_in_page[:, None] * kv_stride_s
        k_ptrs = base_page + k_idx * kv_stride_c + offs_d[None, :] * kv_stride_d
        v_ptrs = base_page + v_idx * kv_stride_c + offs_d[None, :] * kv_stride_d

        k_block = tl.load(k_ptrs, mask=kv_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
        v_block = tl.load(v_ptrs, mask=kv_valid[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

        logits = tl.dot(q, tl.trans(k_block)) * softmax_scale
        if HAS_K_SCALE:
            logits = logits * k_scale
        if HAS_Q_SCALE:
            logits = logits * q_scale
        if HAS_SOFTCAP:
            logits = logits_soft_cap * tl.tanh(logits / logits_soft_cap)

        allowed = (kv_pos[None, :] <= row_idx[:, None]) & (kv_pos[None, :] >= left_bound[:, None])
        mask = row_mask[:, None] & kv_valid[None, :] & allowed
        logits = tl.where(mask, logits, MASK_VALUE)

        has_any = tl.max(mask.to(tl.int32), axis=1) != 0

        block_max = tl.max(logits, axis=1)
        new_m = tl.maximum(m, block_max)
        new_m = tl.where(has_any | (l > 0), new_m, 0.0)

        exp_m = tl.exp(m - new_m)
        exp_logits = tl.exp(logits - new_m[:, None])
        exp_logits = tl.where(mask, exp_logits, 0.0)

        l = exp_m * l + tl.sum(exp_logits, axis=1)
        acc = exp_m[:, None] * acc + tl.dot(exp_logits, v_block)
        m = new_m

    l = tl.maximum(l, 1e-6)
    out = acc / l[:, None]
    if HAS_V_SCALE:
        out = out * v_scale

    o_ptrs = o_ptr + (q_start + offs_m)[:, None] * o_stride_m + pid_h * o_stride_h + offs_d[None, :] * o_stride_d
    tl.store(o_ptrs, out, mask=row_mask[:, None] & d_mask[None, :])


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
) -> jax.Array:
    """Scatter new key-value pairs into the paged KV cache.

    Inserts new keys and values into their correct positions in the paged
    cache using the block table mapping. For each new token, determines
    its target page and offset based on the sequence's KV length and
    query position, then performs an in-place scatter update.

    The function uses JAX's lax.scatter with CLIP mode for safe indexing,
    and jnp.searchsorted for efficient sequence-to-token mapping without
    Python loops.

    Args:
        keys: New key tensor of shape (total_tokens, num_kv_heads, head_dim).
        values: New value tensor of shape (total_tokens, num_kv_heads, head_dim).
        kv_cache: Existing paged KV cache (5D packed format).
        kv_lens: Per-sequence KV lengths, shape (max_num_seqs,).
        block_tables: Flattened block table, shape (max_num_seqs * pages_per_seq,).
        query_start_loc: Cumulative query offsets, shape (max_num_seqs + 1,).
        distribution: Workload distribution [num_decode, num_prefill, num_seqs].

    Returns:
        Updated KV cache with new keys and values inserted.
    """
    total_tokens, _num_kv_heads, _head_dim = map(int, keys.shape)
    num_pages, page_size, _hx2_per_pack, _pack, head_dim_padded = map(int, kv_cache.shape)
    del num_pages

    max_num_seqs = int(kv_lens.shape[0])
    pages_per_seq = int(block_tables.shape[0]) // max_num_seqs

    merged_kv = _merge_kv(keys, values, head_dim_padded=head_dim_padded)
    kv_cache_flat = kv_cache.reshape(-1, *kv_cache.shape[2:])

    t_idx = jnp.arange(total_tokens, dtype=jnp.int32)
    s_idx = jnp.searchsorted(query_start_loc, t_idx, side="right") - jnp.int32(1)
    s_idx = jnp.clip(s_idx, 0, max_num_seqs - 1)

    q_start = query_start_loc[s_idx]
    q_end = query_start_loc[s_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens[s_idx]

    pos_in_new = t_idx - q_start
    pos_in_kv = (kv_len - q_len) + pos_in_new

    page_idx_in_seq = pos_in_kv // jnp.int32(page_size)
    page_offset = pos_in_kv - page_idx_in_seq * jnp.int32(page_size)
    flat_block_idx = s_idx * jnp.int32(pages_per_seq) + page_idx_in_seq
    physical_page = block_tables[flat_block_idx]

    scatter_row = physical_page * jnp.int32(page_size) + page_offset
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
        mode=lax.GatherScatterMode.CLIP,
    )
    return kv_cache_flat.reshape(kv_cache.shape)


@ejit(
    static_argnames=(
        "softmax_scale",
        "sliding_window",
        "logits_soft_cap",
        "q_scale",
        "k_scale",
        "v_scale",
        "chunk_prefill_size",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
    ),
    donate_argnums=(3,),
)
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
) -> tuple[jax.Array, jax.Array]:
    """Execute ragged paged attention v3 with inline KV cache update.

    Performs two operations in sequence:
    1. Scatters new keys and values into the paged KV cache via
       ``_kv_update_scatter`` (using JAX ``lax.scatter`` with CLIP mode).
    2. Computes causal attention over the updated cache using the Triton
       kernel ``_rpa_v3_attn_fwd``.

    The Triton kernel grid is ``(ceil(total_tokens / BLOCK_M), num_q_heads, max_num_seqs)``.
    ``BLOCK_M`` defaults to 128 (overridable via ``num_queries_per_block``);
    ``BLOCK_NPAGES`` defaults to ``min(16, pages_per_seq, 32)``
    (overridable via ``num_kv_pages_per_block``).

    Args:
        queries: Query tensor, shape ``(total_tokens, num_q_heads, head_dim)``.
        keys: New key tensor, shape ``(total_tokens, num_kv_heads, head_dim)``.
        values: New value tensor, shape ``(total_tokens, num_kv_heads, head_dim)``.
        kv_cache: Paged KV cache in 5D packed format
            ``(num_pages, page_size, num_kv_heads_x2_per_pack, pack, head_dim_padded)``.
            The buffer is donated (mutated in-place via JAX buffer donation).
        kv_lens: Per-sequence KV lengths, shape ``(max_num_seqs,)``, int32.
        block_tables: Flattened block table, shape ``(max_num_seqs * pages_per_seq,)``,
            int32.
        query_start_loc: Cumulative query offsets, shape ``(max_num_seqs + 1,)``, int32.
        distribution: Workload distribution ``[num_decode, num_prefill, num_seqs]``,
            int32. Only element 2 (``num_seqs``) is read by the Triton kernel.
        softmax_aux: Optional attention sink values, shape ``(num_q_heads,)``.
            Pre-fills the running softmax max before the first KV tile is processed.
        softmax_scale: Attention scaling factor. Defaults to ``1/sqrt(head_dim)``.
        sliding_window: Optional sliding window size in tokens.
        logits_soft_cap: Optional logit soft capping value. Applied as
            ``logits_soft_cap * tanh(logits / logits_soft_cap)``.
        q_scale: Optional FP8 scale for queries (applied after loading).
        k_scale: Optional FP8 scale for keys (applied to QK dot product).
        v_scale: Optional FP8 scale for values (applied to output).
        chunk_prefill_size: Ignored; accepted for API compatibility (TPU-specific).
        num_kv_pages_per_block: KV pages per inner-loop block (``BLOCK_NPAGES``).
            Clamped to ``[1, min(pages_per_seq, 32)]``. Defaults to 16.
        num_queries_per_block: Query block size (``BLOCK_M``). Defaults to 128.
        vmem_limit_bytes: Ignored; accepted for API compatibility (TPU-specific).

    Returns:
        Tuple of ``(attention_output, updated_kv_cache)``:
            - ``attention_output``: shape ``(total_tokens, num_q_heads, head_dim)``.
            - ``updated_kv_cache``: same shape as input ``kv_cache``, with new
              keys/values scattered in.

    Raises:
        ValueError: If input shapes, dtypes, or dimensions are invalid.
    """
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

    kv_cache = _kv_update_scatter(
        keys=keys,
        values=values,
        kv_cache=kv_cache,
        kv_lens=kv_lens,
        block_tables=block_tables,
        query_start_loc=query_start_loc,
        distribution=distribution,
    )

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
    expected_combined = _align_to(num_kv_heads * 2, pack)
    if combined_heads != expected_combined:
        raise ValueError(
            f"kv_cache head packing mismatch: combined_heads={combined_heads}, expected={expected_combined} "
            f"(num_kv_heads={num_kv_heads}, pack={pack})"
        )
    if head_dim_padded < head_dim or head_dim_padded != _align_to(head_dim, 128):
        raise ValueError("kv_cache head_dim_padded must equal align_to(head_dim, 128)")
    kv_pages = kv_cache.reshape(_num_pages, page_size, combined_heads, head_dim_padded)

    if num_queries_per_block is None:
        block_m = 128
    else:
        block_m = int(num_queries_per_block)

    if num_kv_pages_per_block is None:
        block_npages = 16
    else:
        block_npages = int(num_kv_pages_per_block)
    block_npages = max(1, min(pages_per_seq, block_npages))
    block_npages = min(block_npages, 32)

    block_dmodel = max(triton.next_power_of_2(head_dim), 16)
    kv_block_tokens = block_npages * page_size
    if block_m < 16 or kv_block_tokens < 16 or head_dim < 16:
        raise ValueError(
            "ragged_page_attention_v3 (triton) requires block_m>=16, "
            f"kv_block_tokens>=16, head_dim>=16; got block_m={block_m}, "
            f"kv_block_tokens={kv_block_tokens}, head_dim={head_dim}."
        )

    q_sm, q_sh, q_sd = _contig_strides_3(queries.shape)
    kv_sp, kv_ss, kv_sc, kv_sd = _contig_strides_4(kv_pages.shape)
    bt_ss, bt_sp = pages_per_seq, 1
    o_sm, o_sh, o_sd = _contig_strides_3(queries.shape)

    qblocks_max = math.ceil(total_tokens / block_m)

    has_sink = softmax_aux is not None
    if softmax_aux is None:
        softmax_aux = jnp.zeros((1,), dtype=jnp.float32)
    else:
        if softmax_aux.shape != (num_q_heads,):
            raise ValueError("softmax_aux must have shape (num_q_heads,)")
        softmax_aux = softmax_aux.astype(jnp.float32)

    has_sliding = sliding_window is not None
    sliding_window_val = int(sliding_window or 0)

    has_softcap = logits_soft_cap is not None
    logits_soft_cap_val = float(logits_soft_cap or 0.0)

    has_q_scale = q_scale is not None
    q_scale_val = float(q_scale or 1.0)

    has_k_scale = k_scale is not None
    k_scale_val = float(k_scale or 1.0)

    has_v_scale = v_scale is not None
    v_scale_val = float(v_scale or 1.0)

    out_shape = [jax.ShapeDtypeStruct(queries.shape, queries.dtype)]
    metaparams = dict(
        NUM_REPEATS=num_repeats,
        MAX_NUM_SEQS=max_num_seqs,
        PAGES_PER_SEQ=pages_per_seq,
        PAGE_SIZE=page_size,
        BLOCK_M=block_m,
        BLOCK_NPAGES=block_npages,
        BLOCK_DMODEL=block_dmodel,
        MASK_VALUE=float(DEFAULT_MASK_VALUE),
        HAS_SINK=bool(has_sink),
        HAS_SLIDING=bool(has_sliding),
        HAS_SOFTCAP=bool(has_softcap),
        HAS_Q_SCALE=bool(has_q_scale),
        HAS_K_SCALE=bool(has_k_scale),
        HAS_V_SCALE=bool(has_v_scale),
        num_warps=4,
        num_stages=1,
    )

    (out,) = triton_call(
        queries,
        kv_pages,
        block_tables,
        kv_lens,
        query_start_loc,
        distribution,
        softmax_aux,
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
        grid=lambda META: (qblocks_max, num_q_heads, max_num_seqs),
        name="ejkernel::triton::ragged_page_attention_v3_fwd",
        **metaparams,
    )
    return out, kv_cache
