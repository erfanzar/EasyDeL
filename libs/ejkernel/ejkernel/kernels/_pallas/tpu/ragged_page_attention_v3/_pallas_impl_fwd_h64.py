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

"""Ragged Paged Attention V3 TPU kernel for head_dim=64 using Pallas.

This module provides a specialized implementation of the ragged paged attention
V3 kernel optimized for models with head_dim=64. This is a common configuration
in transformer models like GPT-2, and this kernel takes advantage of the smaller
head dimension for improved memory efficiency and performance.

Algorithm Overview:
    1. Merged K/V representation: Keys and values are concatenated along the
       head dimension (K,V -> 128-dim) for efficient memory layout
    2. Mixed operation batching: Handles decode-only, prefill-only, and mixed
       sequences within a single kernel invocation
    3. Double-buffered async DMA: Overlaps memory transfers with computation
    4. In-place KV cache updates: Updates cache during prefill operations
    5. Optimized tiling: Special block sizes tuned for 64-dim heads

Key Features:
    - head_dim=64 optimization: Memory layout concatenates K and V to form
      128-dim vectors, matching TPU tile sizes perfectly
    - Mixed prefill/decode: Single kernel handles both operation types
    - Chunked prefill: Processes prefill tokens in configurable chunk sizes
    - Sliding window attention: Optional window-based masking for long sequences
    - Quantization support: Optional q_scale, k_scale, v_scale parameters
    - Attention sinks: Optional softmax auxiliary logits for streaming

Memory Layout (head_dim=64 specific):
    - KV cache: [total_pages, page_size, num_kv_heads // packing, packing, 128]
      where the last dimension is K(64) + V(64) concatenated
    - Queries: [num_kv_heads, max_tokens, num_q_per_kv // packing, packing, 64]

Performance Considerations:
    - The 64->128 dimension packing enables efficient use of 128-wide TPU MXU
    - Reduced VMEM footprint compared to head_dim=128 variant
    - Tuned block sizes in _utils.py via get_tuned_block_sizes_h64()

Example:
    >>> # Attention for GPT-2 style model (head_dim=64)
    >>> output, updated_cache = ragged_paged_attention_hd64(
    ...     queries=q,                        # [tokens, 12, 64]
    ...     keys=k,                           # [tokens, 12, 64]
    ...     values=v,                         # [tokens, 12, 64]
    ...     kv_cache=cache,                   # Paged, K/V concatenated to 128
    ...     kv_lens=seq_lens,
    ...     block_tables=page_indices,
    ...     query_start_loc=cu_q_lens,
    ...     distribution=jnp.array([1, 2, 3]),
    ...     softmax_scale=1.0 / sqrt(64),
    ... )

Note:
    This kernel requires head_dim=64 and will assert if called with other
    dimensions. For head_dim=128 or larger, use the standard _pallas_impl_fwd.py
    implementation. The K/V concatenation to 128 dimensions is handled
    internally by the merge_kv function.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ejkernel.callib import ejit

from ._utils import align_to, cdiv, get_dtype_packing, get_tuned_block_sizes_h64

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)

DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024


def ref_ragged_paged_attention_hd64(
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
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
):
    """Reference implementation of V3 ragged paged attention for head_dim=64.

    Processes each sequence sequentially, updating the KV cache with new tokens
    and computing attention using the head_dim=64 K/V concatenated layout. Serves
    as ground truth for validating the Pallas kernel. Queries are zero-padded
    to 128 dimensions to match the K/V concatenated format.

    Args:
        queries: Query tokens [total_tokens, num_q_heads, 64].
        keys: Key tokens [total_tokens, num_kv_heads, 64].
        values: Value tokens [total_tokens, num_kv_heads, 64].
        kv_cache: Paged KV cache [pages, page_size, kv_heads/pack, pack, 128].
        kv_lens: KV context lengths per sequence [max_num_seqs].
        block_tables: Flattened page table [max_num_seqs * pages_per_seq].
        query_start_loc: Cumulative query positions [max_num_seqs+1].
        distribution: Batch composition [3] (decode_end, prefill_end, total).
        softmax_aux: Optional attention sink logits [num_q_heads].
        softmax_scale: Scaling factor for QK^T.
        sliding_window: Optional window size for local attention.
        logits_soft_cap: Optional soft cap for logits.
        mask_value: Value for masked positions.
        q_scale: Optional quantization scale for queries.
        k_scale: Optional quantization scale for keys.
        v_scale: Optional quantization scale for values.

    Returns:
        Tuple of (attention_output [total_tokens, num_q_heads, 64],
        updated_kv_cache).
    """
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE
    dynamic_validate_inputs(
        queries,
        keys,
        values,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        softmax_aux,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    actual_head_dim = queries.shape[2]
    actual_num_q_heads = queries.shape[1]
    actual_num_kv_heads = keys.shape[1]
    assert actual_head_dim == 64
    (
        _,
        page_size,
        _,
        kv_packing,
        actual_head_dim_x2,
    ) = kv_cache.shape

    assert actual_num_q_heads % actual_num_kv_heads == 0
    assert actual_head_dim_x2 == 128
    assert get_dtype_packing(kv_cache.dtype) == kv_packing
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    padded_actual_num_kv_heads = align_to(actual_num_kv_heads, kv_packing)
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = block_tables.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs

    merged_kv = merge_kv(keys, values)
    bkv_sz = None
    if sliding_window is not None:
        bkv_p, _ = get_tuned_block_sizes_h64(
            queries.dtype,
            kv_cache.dtype,
            actual_num_q_heads,
            actual_num_kv_heads,
            actual_head_dim,
            page_size,
            queries.shape[0],
            pages_per_seq,
        )
        bkv_sz = bkv_p * page_size
        assert bkv_sz > 0, "bkv_sz must be positive"
    queries = jnp.pad(queries, ((0, 0), (0, 0), (0, 64)), constant_values=0.0)
    outputs = []

    for i in range(distribution[-1]):
        q_start = query_start_loc[i]
        q_end = query_start_loc[i + 1]
        q_len = q_end - q_start

        kv_len = kv_lens[i]
        kv_start = 0
        if sliding_window is not None:
            kv_start = jnp.maximum(kv_len - sliding_window, 0)
            kv_start = (kv_start // bkv_sz) * bkv_sz
        kv_len_eff = kv_len - kv_start
        if kv_len_eff < q_len:
            raise ValueError(f"Sliding window {sliding_window} too small for q_len {q_len} (kv_len {kv_len}).")
        indices_start = i * pages_per_seq
        start_page = kv_start // page_size
        end_page = start_page + cdiv(kv_len_eff, page_size)
        indices = block_tables[indices_start + start_page : indices_start + end_page]
        q = queries[q_start:q_end, :, :]

        assert kv_len - q_len >= 0
        gathered_kv = kv_cache[indices]
        gathered_shape = gathered_kv.shape
        gathered_kv = gathered_kv.reshape(-1, *gathered_shape[-3:])
        gathered_kv = gathered_kv.at[kv_len_eff - q_len : kv_len_eff].set(merged_kv[q_start:q_end])
        kv_cache = kv_cache.at[indices].set(gathered_kv.reshape(gathered_shape))

        kv = gathered_kv.reshape(-1, padded_actual_num_kv_heads, actual_head_dim_x2)[:, :actual_num_kv_heads, :]
        kv = kv[:kv_len_eff, :, :]
        kv = jnp.repeat(kv, actual_num_q_heads_per_kv_head, axis=1)
        if q_scale is not None:
            q = q / q_scale
            if jnp.issubdtype(kv.dtype, jnp.floating):
                dtype_info = jnp.finfo(kv.dtype)
                minval = float(dtype_info.min)
                maxval = float(dtype_info.max)
                q = jnp.clip(q, min=minval, max=maxval)
            q = q.astype(kv.dtype)
        attn = jnp.einsum("qhd,khd->hqk", q, kv, preferred_element_type=jnp.float32)
        attn *= softmax_scale
        if k_scale is not None:
            attn *= k_scale
        if q_scale is not None:
            attn *= q_scale

        q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(jnp.int32, attn.shape, 1)
        kv_span = kv_start + jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
        mask = q_span < kv_span
        if sliding_window is not None:
            mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
        if logits_soft_cap is not None:
            attn = logits_soft_cap * jnp.tanh(attn / logits_soft_cap)
        attn = jnp.where(mask, mask_value, attn)

        if softmax_aux is not None:
            reshaped_attention_sink = softmax_aux.reshape(actual_num_q_heads, 1, 1)
            reshaped_attention_sink = jnp.repeat(reshaped_attention_sink, q_len, axis=1)
            attn = jnp.concat([reshaped_attention_sink, attn], axis=2)
            attn = jax.nn.softmax(attn, axis=-1).astype(kv.dtype)
            attn = attn[..., 1:]
        else:
            attn = jax.nn.softmax(attn, axis=-1).astype(kv.dtype)

        out = jnp.einsum("hqk,khd->qhd", attn, kv).astype(queries.dtype)
        if v_scale is not None:
            out *= v_scale

        outputs.append(out)

    result = jnp.concatenate(outputs, axis=0)
    result = result[:, :, actual_head_dim:]
    return result, kv_cache


def get_smem_estimate_bytes(max_num_seqs, pages_per_seq):
    """Estimate scalar memory (SMEM) usage for the head_dim=64 kernel.

    Computes the total SMEM needed for scalar prefetch arrays including
    kv_lens, block_tables, query_start_loc, distribution, and scratch
    index buffers. Each array is aligned to 128 elements with 32-bit words.

    Args:
        max_num_seqs: Maximum number of sequences in the batch.
        pages_per_seq: Maximum number of pages per sequence.

    Returns:
        Estimated SMEM usage in bytes.
    """
    total_bits = (
        align_to(max_num_seqs, 128) * 32
        + align_to(max_num_seqs * pages_per_seq, 128) * 32
        + align_to(max_num_seqs + 1, 128) * 32
        + 128 * 32
        + 128 * 32
        + 128 * 32
        + 128 * 32
    )
    return cdiv(total_bits, 8)


def get_vmem_estimate_bytes(
    actual_num_kv_heads,
    actual_num_q_heads_per_kv_head,
    actual_head_dim,
    bq_sz,
    bkv_sz,
    q_dtype,
    kv_dtype,
):
    """Estimate VMEM usage for the head_dim=64 kernel scratch buffers.

    Computes total VMEM needed for double-buffered KV blocks,
    double-buffered query blocks, softmax accumulators, and the
    output accumulator. Uses the 64->128 concatenated layout where
    head_dim is effectively doubled.

    Args:
        actual_num_kv_heads: Number of KV attention heads.
        actual_num_q_heads_per_kv_head: Query heads per KV head (GQA ratio).
        actual_head_dim: Unpadded head dimension (must be 64).
        bq_sz: Number of query tokens per block.
        bkv_sz: Number of KV tokens per block.
        q_dtype: Query dtype for packing factor calculation.
        kv_dtype: KV cache dtype for packing factor calculation.

    Returns:
        Estimated VMEM usage in bytes.
    """
    assert actual_head_dim == 64
    q_packing = get_dtype_packing(q_dtype)
    kv_packing = get_dtype_packing(kv_dtype)
    num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head, q_packing)
    num_kv_heads = align_to(actual_num_kv_heads, kv_packing)
    head_dim = actual_head_dim * 2

    total_bits = (
        (2 * bkv_sz * num_kv_heads * head_dim) * (32 // kv_packing)
        + 2 * (2 * actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * head_dim) * (32 // q_packing)
        + 2 * (actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * 128) * 32
        + (actual_num_kv_heads * bq_sz * num_q_heads_per_kv_head * head_dim) * 32
    )
    return cdiv(total_bits, 8)


DEFAULT_SCOPED_VMEM_LIMIT_BYTES = 16 << 20


def _clamp_block_sizes_to_vmem(
    *,
    actual_num_kv_heads,
    actual_num_q_heads_per_kv_head,
    actual_head_dim,
    page_size,
    q_dtype,
    kv_dtype,
    bkv_p,
    bq_sz,
    vmem_limit_bytes,
):
    """Clamp TPU block sizes until scratch buffers fit within scoped VMEM."""
    limit = DEFAULT_SCOPED_VMEM_LIMIT_BYTES
    if vmem_limit_bytes is not None:
        limit = min(int(vmem_limit_bytes), limit)

    bkv_p = max(1, int(bkv_p))
    bq_sz = max(1, int(bq_sz))

    estimate = get_vmem_estimate_bytes(
        actual_num_kv_heads=actual_num_kv_heads,
        actual_num_q_heads_per_kv_head=actual_num_q_heads_per_kv_head,
        actual_head_dim=actual_head_dim,
        bq_sz=bq_sz,
        bkv_sz=bkv_p * page_size,
        q_dtype=q_dtype,
        kv_dtype=kv_dtype,
    )
    while estimate > limit and bkv_p > 1:
        bkv_p = max(1, bkv_p // 2)
        estimate = get_vmem_estimate_bytes(
            actual_num_kv_heads=actual_num_kv_heads,
            actual_num_q_heads_per_kv_head=actual_num_q_heads_per_kv_head,
            actual_head_dim=actual_head_dim,
            bq_sz=bq_sz,
            bkv_sz=bkv_p * page_size,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
        )
    while estimate > limit and bq_sz > 1:
        bq_sz = max(1, bq_sz // 2)
        estimate = get_vmem_estimate_bytes(
            actual_num_kv_heads=actual_num_kv_heads,
            actual_num_q_heads_per_kv_head=actual_num_q_heads_per_kv_head,
            actual_head_dim=actual_head_dim,
            bq_sz=bq_sz,
            bkv_sz=bkv_p * page_size,
            q_dtype=q_dtype,
            kv_dtype=kv_dtype,
        )
    return bkv_p, bq_sz


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    actual_num_kv_heads,
    actual_head_dim,
    kv_dtype,
):
    """Compute the KV cache shape for head_dim=64 with K/V concatenated layout.

    Returns the 5D shape where keys and values are concatenated along the
    last dimension (64+64=128), rather than the x2 interleaved layout used
    for head_dim>=128.

    Args:
        total_num_pages: Total number of pages in the cache pool.
        page_size: Number of tokens per page.
        actual_num_kv_heads: Number of KV attention heads (before padding).
        actual_head_dim: Head dimension (must be 64).
        kv_dtype: KV cache dtype for packing factor calculation.

    Returns:
        Tuple of (total_num_pages, page_size, num_kv_heads // packing,
        packing, 128).
    """
    assert actual_head_dim == 64
    kv_packing = get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        page_size,
        align_to(actual_num_kv_heads, kv_packing) // kv_packing,
        kv_packing,
        128,
    )


def _ragged_paged_attention_kernel(
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    distribution_ref,
    sem_ids_ref,
    bo_ids_ref,
    bkv_update_ids_ref,
    q_hbm_ref,
    kv_hbm_ref,
    kv_cache_hbm_ref,
    attention_sink_ref,
    o_hbm_ref,
    updated_kv_cache_hbm_ref,
    bkv_x2_ref,
    bq_x2_ref,
    bo_x2_ref,
    sems,
    l_ref,
    m_ref,
    acc_ref,
    *,
    softmax_scale: float,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    bkv_p,
    bq_sz,
):
    """Pallas TPU kernel for ragged paged attention V3 with head_dim=64.

    Specialized kernel for head_dim=64 models where keys and values are
    concatenated into 128-dim vectors. Each program instance processes
    one sequence with double-buffered async DMA, flash attention, and
    in-place KV cache updates.

    The kernel handles three sequence types based on distribution:
    - Decode (seq_idx < decode_end): Single token, static q_len=1
    - Prefill (decode_end <= seq_idx < prefill_end): Chunked prefill
    - Mixed (prefill_end <= seq_idx < mixed_end): Variable length

    Args:
        kv_lens_ref: SMEM ref to KV context lengths [max_num_seqs].
        page_indices_ref: SMEM ref to flattened page table.
        cu_q_lens_ref: SMEM ref to cumulative query lengths [max_num_seqs+1].
        distribution_ref: SMEM ref to batch composition [3].
        sem_ids_ref: SMEM ref to current semaphore indices.
        bo_ids_ref: SMEM ref to output block tracking indices.
        bkv_update_ids_ref: SMEM ref to KV cache update tracking.
        q_hbm_ref: HBM ref to query tensor (padded to 128-dim).
        kv_hbm_ref: HBM ref to new KV tokens (K/V concatenated).
        kv_cache_hbm_ref: HBM ref to existing KV cache (input).
        attention_sink_ref: VMEM ref to attention sink logits or None.
        o_hbm_ref: HBM ref to output tensor.
        updated_kv_cache_hbm_ref: HBM ref to updated KV cache (output).
        bkv_x2_ref: VMEM double-buffer for KV cache blocks.
        bq_x2_ref: VMEM double-buffer for query blocks.
        bo_x2_ref: VMEM double-buffer for output blocks.
        sems: DMA semaphores [4, 2].
        l_ref: VMEM scratch for softmax normalizer.
        m_ref: VMEM scratch for softmax running maximum.
        acc_ref: VMEM scratch for attention output accumulator.
        softmax_scale: Scaling factor for QK^T dot products.
        sliding_window: Optional sliding window size.
        logits_soft_cap: Optional soft cap for attention logits.
        mask_value: Value for masked positions.
        q_scale: Optional quantization scale for queries.
        k_scale: Optional quantization scale for keys.
        v_scale: Optional quantization scale for values.
        chunk_prefill_size: Chunk size for prefill sequences.
        bkv_p: Number of KV pages per attention block.
        bq_sz: Number of query tokens per attention block.
    """
    assert q_hbm_ref.shape == o_hbm_ref.shape
    assert q_hbm_ref.shape[-1] == kv_cache_hbm_ref.shape[-1]
    (
        actual_num_kv_heads,
        _max_num_tokens,
        num_q_heads_per_kv_head_per_packing,
        q_packing,
        actual_head_dim_x2,
    ) = q_hbm_ref.shape
    (
        _total_num_pages,
        page_size,
        num_kv_heads_per_kv_packing,
        kv_packing,
        _,
    ) = kv_cache_hbm_ref.shape
    max_num_seqs = kv_lens_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs
    num_kv_heads = num_kv_heads_per_kv_packing * kv_packing
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_packing * q_packing
    q_dtype = q_hbm_ref.dtype
    kv_dtype = kv_cache_hbm_ref.dtype
    assert o_hbm_ref.dtype == q_dtype
    assert get_dtype_packing(q_dtype) == q_packing
    assert get_dtype_packing(kv_dtype) == kv_packing
    assert actual_head_dim_x2 == 128
    bkv_sz = bkv_p * page_size
    seq_idx = pl.program_id(0)
    num_seqs = pl.num_programs(0)
    decode_end = distribution_ref[0]
    prefill_end = distribution_ref[1]
    mixed_end = distribution_ref[2]

    q_start = cu_q_lens_ref[seq_idx]
    q_end = cu_q_lens_ref[seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[seq_idx]

    bkv_idx_start = 0 if sliding_window is None else jnp.maximum(kv_len - sliding_window, 0) // bkv_sz

    if sliding_window is None:
        next_bkv_idx_start = 0
    else:

        def get_next_bkv_idx_start():
            next_kv_len = kv_lens_ref[seq_idx + 1]
            return jnp.maximum(next_kv_len - sliding_window, 0) // bkv_sz

        next_bkv_idx_start = lax.cond(seq_idx + 1 < num_seqs, get_next_bkv_idx_start, lambda: 0)

    def flash_attention(
        q,
        kv,
        *,
        bq_idx,
        bkv_idx,
        kv_head_idx,
    ):
        assert len(q.shape) == 2
        assert q.shape[0] % num_q_heads_per_kv_head == 0
        assert q.shape[1] == actual_head_dim_x2
        assert kv.shape == (bkv_sz, actual_head_dim_x2)
        head_l_ref = l_ref.at[kv_head_idx, : q.shape[0]]
        head_m_ref = m_ref.at[kv_head_idx, : q.shape[0]]
        head_acc_ref = acc_ref.at[kv_head_idx, : q.shape[0]]

        def load_with_init(ref, init_val):
            return jnp.where(bkv_idx == bkv_idx_start, jnp.full_like(ref, init_val), ref[...])

        if q_scale is not None:
            q = q / q_scale
            if jnp.issubdtype(kv.dtype, jnp.floating):
                dtype_info = jnp.finfo(kv.dtype)
                minval = float(dtype_info.min)
                maxval = float(dtype_info.max)
                q = jnp.clip(q, min=minval, max=maxval)
            q = q.astype(kv.dtype)

        s = jnp.einsum("nd,md->nm", q, kv, preferred_element_type=jnp.float32)
        s *= softmax_scale
        if k_scale is not None:
            s *= k_scale
        if q_scale is not None:
            s *= q_scale

        q_span = kv_len - q_len + bq_idx * bq_sz + lax.broadcasted_iota(jnp.int32, s.shape, 0) // num_q_heads_per_kv_head
        k_span = bkv_idx * bkv_sz + lax.broadcasted_iota(jnp.int32, s.shape, 1)
        mask = q_span < k_span
        if sliding_window is not None:
            mask = jnp.logical_or(mask, q_span - sliding_window >= k_span)

        if logits_soft_cap is not None:
            s = logits_soft_cap * jnp.tanh(s / logits_soft_cap)
        s += jnp.where(mask, mask_value, 0.0)
        s_rowmax = jnp.max(s, axis=1, keepdims=True)

        if attention_sink_ref is not None:
            sinks = attention_sink_ref[kv_head_idx]
            actual_bq_sz = q.shape[0] // num_q_heads_per_kv_head
            m_prev_init = jnp.concat([sinks] * actual_bq_sz, axis=0)
            m_prev = jnp.where(bkv_idx == bkv_idx_start, m_prev_init, head_m_ref[...])
        else:
            m_prev = load_with_init(head_m_ref, -jnp.inf)

        m_curr = jnp.maximum(m_prev, s_rowmax)
        head_m_ref[...] = m_curr
        p = jnp.exp(s - broadcast_minor(m_curr, s.shape))

        pv = jnp.einsum("nm,md->nd", p, kv, preferred_element_type=jnp.float32)
        if v_scale is not None:
            pv *= v_scale

        p_rowsum = jnp.sum(p, axis=1, keepdims=True)
        exp_m_diff = jnp.exp(m_prev - m_curr)
        l_prev = load_with_init(head_l_ref, 1.0)
        l_curr = exp_m_diff * l_prev + p_rowsum
        head_l_ref[...] = l_curr
        o_prev = load_with_init(head_acc_ref, 0.0)
        o_curr = broadcast_minor(exp_m_diff, o_prev.shape) * o_prev + pv
        head_acc_ref[...] = o_curr

    def _async_copy(src, dst, sem, wait):
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
        sem = sems.at[0, bkv_sem_idx]
        vmem_ref = bkv_x2_ref.at[bkv_sem_idx]

        cache_hbm_shape = kv_cache_hbm_ref.shape
        cache_hbm_ref = kv_cache_hbm_ref.reshape(cache_hbm_shape[0] * cache_hbm_shape[1], *cache_hbm_shape[2:])
        kv_len = kv_lens_ref[seq_idx]
        kv_len_start = bkv_idx * bkv_sz
        kv_p_start = bkv_idx * bkv_p
        q_start = cu_q_lens_ref[seq_idx]
        q_end = cu_q_lens_ref[seq_idx + 1]
        q_len = q_end - q_start

        kv_left = kv_len - kv_len_start
        kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
        kv_left_frm_new = kv_left - kv_left_frm_cache
        bkv_p_frm_cache = jnp.minimum(cdiv(kv_left_frm_cache, page_size), bkv_p)
        bkv_sz_frm_new = jnp.minimum(jnp.maximum(bkv_sz - kv_left_frm_cache, 0), kv_left_frm_new)
        page_indices_offset = seq_idx * pages_per_seq + kv_p_start

        wait_update_kv_cache(bkv_sem_idx)

        def loop_body(i, offset):
            sz = jnp.minimum(page_size, kv_left_frm_cache - i * page_size)
            _async_copy(
                cache_hbm_ref.at[pl.ds(page_indices_ref[page_indices_offset + i] * page_size, sz)],
                vmem_ref.at[pl.ds(i * page_size, sz)],
                sem,
                wait,
            )
            return offset + sz

        offset = lax.fori_loop(
            0,
            bkv_p_frm_cache,
            loop_body,
            0,
            unroll=False,
        )

        @pl.when(bkv_sz_frm_new > 0)
        def _fetch_bkv_from_new_kv():
            new_kv_len_start = q_end - kv_left_frm_new
            _async_copy(
                kv_hbm_ref.at[pl.ds(new_kv_len_start, bkv_sz_frm_new)],
                vmem_ref.at[pl.ds(offset, bkv_sz_frm_new)],
                sem,
                wait,
            )

        return kv_len_start + offset, bkv_sz_frm_new

    def _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, *, wait=False):
        sem = sems.at[3, bkv_sem_idx]
        vmem_ref = bkv_x2_ref.at[bkv_sem_idx]
        bkv_id = offset // bkv_sz
        kv_p_start = offset // page_size
        kv_p_end = cdiv(offset + update_sz, page_size)
        ignore = offset % page_size
        p_ignore = kv_p_start - bkv_id * bkv_p
        page_indices_offset = seq_idx * pages_per_seq + kv_p_start

        cache_hbm_shape = updated_kv_cache_hbm_ref.shape
        cache_hbm_ref = updated_kv_cache_hbm_ref.reshape(cache_hbm_shape[0] * cache_hbm_shape[1], *cache_hbm_shape[2:])

        def loop_body(i, states):
            update_sz, ignore = states
            sz = jnp.minimum(page_size - ignore, update_sz)

            _async_copy(
                vmem_ref.at[pl.ds((p_ignore + i) * page_size + ignore, sz)],
                cache_hbm_ref.at[
                    pl.ds(
                        page_indices_ref[page_indices_offset + i] * page_size + ignore,
                        sz,
                    )
                ],
                sem,
                wait,
            )
            return update_sz - sz, 0

        lax.fori_loop(
            0,
            kv_p_end - kv_p_start,
            loop_body,
            (update_sz, ignore),
            unroll=False,
        )

    def _fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
        sem = sems.at[1, bq_sem_idx]
        vmem_ref = bq_x2_ref.at[bq_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(
            q_hbm_ref.at[:, pl.ds(q_len_start, sz)],
            vmem_ref.at[:, pl.ds(0, sz)],
            sem,
            wait,
        )

    def _send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False):
        sem = sems.at[2, bo_sem_idx]
        vmem_ref = bo_x2_ref.at[bo_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(
            vmem_ref.at[:, pl.ds(0, sz)],
            o_hbm_ref.at[:, pl.ds(q_len_start, sz)],
            sem,
            wait,
        )

    def start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

    def wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, wait=True)

    def start_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx)

    def wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx, wait=True)

    def start_send_bo(seq_idx, bo_idx, bo_sem_idx):
        bo_ids_ref[bo_sem_idx] = seq_idx
        bo_ids_ref[bo_sem_idx + 2] = bo_idx
        _send_bo(seq_idx, bo_idx, bo_sem_idx)

    def wait_send_bo(bo_sem_idx):
        old_seq_idx = bo_ids_ref[bo_sem_idx]
        old_bo_idx = bo_ids_ref[bo_sem_idx + 2]

        @pl.when(jnp.logical_and(0 <= old_seq_idx, old_seq_idx <= seq_idx))
        def _():
            _send_bo(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)

    def start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz):
        bkv_update_ids_ref[bkv_sem_idx] = seq_idx
        bkv_update_ids_ref[bkv_sem_idx + 2] = offset
        bkv_update_ids_ref[bkv_sem_idx + 4] = update_sz
        _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)

    def wait_update_kv_cache(bkv_sem_idx):
        update_sz = bkv_update_ids_ref[bkv_sem_idx + 4]

        @pl.when(update_sz > 0)
        def _():
            seq_idx = bkv_update_ids_ref[bkv_sem_idx]
            offset = bkv_update_ids_ref[bkv_sem_idx + 2]
            bkv_update_ids_ref[bkv_sem_idx + 4] = 0
            _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz, wait=True)

    def load_bq(bq_sem_idx, kv_head_idx, *, actual_bq_sz=bq_sz):
        q_ref = (
            bq_x2_ref.bitcast(jnp.uint32)
            .at[bq_sem_idx, kv_head_idx]
            .reshape(bq_sz * num_q_heads_per_kv_head_per_packing, actual_head_dim_x2)
        )
        return pltpu.bitcast(q_ref[: actual_bq_sz * num_q_heads_per_kv_head_per_packing], q_dtype)

    def strided_load(ref, start, step):
        assert get_dtype_packing(ref.dtype) == 1
        assert len(ref.shape) == 2
        _, l = ref.shape
        assert l == 128
        vec = ref[start::step]
        return vec

    def strided_load_bkv(bkv_sem_idx, start, step, *, bkv_mask):
        assert start % kv_packing == 0
        assert step % kv_packing == 0
        start //= kv_packing
        step //= kv_packing
        kv_ref = bkv_x2_ref.bitcast(jnp.uint32).at[bkv_sem_idx].reshape(bkv_sz * step, actual_head_dim_x2)

        kv = strided_load(kv_ref, start, step)
        kv = lax.select(bkv_mask, kv, jnp.zeros_like(kv))
        bitwidth = 32 // kv_packing
        repack_ty = jnp.dtype(f"uint{bitwidth}")
        lst = []
        for i in range(0, kv_packing):
            cur_kv = pltpu.bitcast((kv >> (i * bitwidth)).astype(repack_ty), kv_dtype)
            lst.append(cur_kv)
        return lst

    def broadcast_minor(src, shape):
        if src.shape == shape:
            return src
        assert src.shape[:-1] == shape[:-1]
        assert src.shape[-1] % 128 == 0
        target_minor = align_to(shape[-1], src.shape[-1])

        return jnp.concatenate([src for _ in range(target_minor // src.shape[-1])], axis=-1)[..., : shape[-1]]

    def process(static_q_len=None):
        num_bkv = cdiv(kv_len, bkv_sz)
        if static_q_len is None:
            actual_bq_sz = bq_sz
            num_bq = cdiv(q_len, actual_bq_sz)
        else:
            actual_bq_sz = min(bq_sz, static_q_len)
            num_bq = cdiv(static_q_len, actual_bq_sz)

        def get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx):
            next_bq_idx = bq_idx + 1
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bq_sem_idx = lax.select(bq_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bq_sem_idx

        def get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx):
            next_bkv_idx = bkv_idx + 1
            is_last_bkv = next_bkv_idx == num_bkv
            next_bq_idx = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bkv_sem_idx = lax.select(bkv_sem_idx == 0, 1, 0)

            next_bkv_idx = lax.select(
                is_last_bkv,
                lax.select(is_last_bq, next_bkv_idx_start, bkv_idx_start),
                next_bkv_idx,
            )
            return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx

        def compute_with_bq(bq_idx, _):
            bq_sem_idx = sem_ids_ref[0]
            next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx)

            @pl.when(next_seq_idx < num_seqs)
            def prefetch_next_bq():
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            def compute_with_bkv(bkv_idx, _):
                assert bkv_sz % kv_packing == 0
                actual_bkv_sz = jnp.minimum(bkv_sz, kv_len - bkv_idx * bkv_sz)
                bkv_shape = (bkv_sz, actual_head_dim_x2)
                bkv_mask = lax.broadcasted_iota(jnp.int32, bkv_shape, 0) < actual_bkv_sz

                bkv_sem_idx = sem_ids_ref[1]
                next_seq_idx, _, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx)

                @pl.when(next_seq_idx < num_seqs)
                def prefetch_next_bkv():
                    sem_ids_ref[1] = next_bkv_sem_idx
                    start_fetch_bkv(next_seq_idx, next_bkv_idx, next_bkv_sem_idx)

                @pl.when(bkv_idx == bkv_idx_start)
                def wait_cur_bq():
                    wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)

                offset, update_sz = wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

                @pl.when(jnp.logical_and(update_sz > 0, bq_idx == 0))
                def update_cur_bkv_to_cache():
                    start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)

                for kv_head_start in range(0, actual_num_kv_heads, kv_packing):
                    bkv_lst = strided_load_bkv(
                        bkv_sem_idx,
                        kv_head_start,
                        num_kv_heads,
                        bkv_mask=bkv_mask,
                    )
                    assert len(bkv_lst) == kv_packing
                    for i in range(kv_packing):
                        kv_head_idx = kv_head_start + i
                        if kv_head_idx >= actual_num_kv_heads:
                            break
                        bq = load_bq(bq_sem_idx, kv_head_idx, actual_bq_sz=actual_bq_sz)
                        bkv = bkv_lst[i]
                        flash_attention(
                            bq,
                            bkv,
                            bq_idx=bq_idx,
                            bkv_idx=bkv_idx,
                            kv_head_idx=kv_head_idx,
                        )

            lax.fori_loop(bkv_idx_start, num_bkv, compute_with_bkv, None, unroll=False)

            acc = acc_ref[...]
            l = broadcast_minor(l_ref[...], acc.shape)
            out = lax.div(acc, l) if q_dtype == jnp.float32 else (acc * pl.reciprocal(l, approx=True)).astype(q_dtype)

            bo_sem_idx = sem_ids_ref[2]
            sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
            wait_send_bo(bo_sem_idx)

            bo_x2_ref.at[bo_sem_idx].bitcast(jnp.int32).reshape(
                actual_num_kv_heads,
                bq_sz * num_q_heads_per_kv_head_per_packing,
                actual_head_dim_x2,
            )[...] = pltpu.bitcast(out, jnp.int32)

            start_send_bo(seq_idx, bq_idx, bo_sem_idx)

        lax.fori_loop(0, num_bq, compute_with_bq, None, unroll=False)

    @pl.when(seq_idx == 0)
    def prologue():
        start_fetch_bq(0, 0, 0)
        start_fetch_bkv(0, bkv_idx_start, 0)

    @pl.when(seq_idx < decode_end)
    def process_decode():
        process(static_q_len=1)

    @pl.when(jnp.logical_and(decode_end <= seq_idx, seq_idx < prefill_end))
    def process_prefill():
        process(static_q_len=chunk_prefill_size)

    @pl.when(jnp.logical_and(prefill_end <= seq_idx, seq_idx < mixed_end))
    def process_mixed():
        process()

    @pl.when(seq_idx == num_seqs - 1)
    def epilogue():
        for i in range(2):
            wait_send_bo(i)
            wait_update_kv_cache(i)


def merge_kv(
    k: jax.Array,
    v: jax.Array,
):
    """Concatenate key and value tensors along head dimension for head_dim=64.

    For head_dim=64 models, keys and values are concatenated to form 128-dim
    vectors (K(64)+V(64)=128), which perfectly matches TPU tile sizes. The
    result is padded to align KV heads to the packing factor.

    Args:
        k: Key tensor [max_num_tokens, actual_num_kv_heads, 64].
        v: Value tensor [max_num_tokens, actual_num_kv_heads, 64].
            Must have same shape and dtype as k.

    Returns:
        Merged KV tensor [max_num_tokens, num_kv_heads // packing,
        packing, 128] ready for cache storage.
    """
    assert k.shape == v.shape
    assert k.dtype == v.dtype
    max_num_tokens, actual_num_kv_heads, actual_head_dim = k.shape
    kv_packing = get_dtype_packing(k.dtype)
    num_kv_heads = align_to(actual_num_kv_heads, kv_packing)
    kv = jnp.pad(
        jnp.concat([k, v], axis=-1),
        (
            (0, 0),
            (0, num_kv_heads - actual_num_kv_heads),
            (0, 0),
        ),
        constant_values=0,
    ).reshape(
        max_num_tokens,
        num_kv_heads // kv_packing,
        kv_packing,
        actual_head_dim * 2,
    )
    return kv


def prepare_inputs(
    q: jax.Array,
    k: jax.Array,
    v: jax.Array,
    softmax_aux: jax.Array | None = None,
):
    """Transform inputs into TPU-optimized layout for head_dim=64.

    Reshapes queries into the kernel's expected layout with padding
    to 128 dimensions (matching K/V concatenation). Merges keys and
    values via concatenation. Optionally reshapes attention sink logits.

    Args:
        q: Query tensor [max_num_tokens, actual_num_q_heads, 64].
        k: Key tensor [max_num_tokens, actual_num_kv_heads, 64].
        v: Value tensor [max_num_tokens, actual_num_kv_heads, 64].
        softmax_aux: Optional attention sink logits [num_q_heads].

    Returns:
        Tuple of (q_packed, kv_merged, softmax_aux_reshaped) in
        TPU-optimized layout.
    """
    max_num_tokens, actual_num_q_heads, actual_head_dim = q.shape
    actual_num_kv_heads = k.shape[1]
    assert actual_num_q_heads % actual_num_kv_heads == 0
    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    q_packing = get_dtype_packing(q.dtype)
    num_q_heads_per_kv_head = align_to(actual_num_q_heads_per_kv_head, q_packing)
    head_dim = align_to(actual_head_dim, 128)
    q = (
        jnp.pad(
            q.reshape(
                max_num_tokens,
                actual_num_kv_heads,
                actual_num_q_heads_per_kv_head,
                actual_head_dim,
            ),
            (
                (0, 0),
                (0, 0),
                (0, num_q_heads_per_kv_head - actual_num_q_heads_per_kv_head),
                (0, head_dim - actual_head_dim),
            ),
            constant_values=0,
        )
        .reshape(
            max_num_tokens,
            actual_num_kv_heads,
            num_q_heads_per_kv_head // q_packing,
            q_packing,
            head_dim,
        )
        .swapaxes(0, 1)
    )

    kv = merge_kv(k, v)

    if softmax_aux is not None:
        softmax_aux = softmax_aux.reshape((-1, num_q_heads_per_kv_head, 1))
        softmax_aux = jnp.repeat(softmax_aux, 128, -1)

    return q, kv, softmax_aux


def prepare_outputs(
    out,
    actual_num_q_heads_per_kv_head: int,
    actual_head_dim: int,
):
    """Transform kernel output back to standard shape for head_dim=64.

    Reverses the layout transformation from prepare_inputs, extracting
    the value portion (latter 64 dims) from the 128-dim concatenated
    output and reshaping to [max_tokens, num_q_heads, 64].

    Args:
        out: Kernel output in TPU-packed 128-dim layout.
        actual_num_q_heads_per_kv_head: Unpadded query heads per KV head.
        actual_head_dim: Unpadded head dimension (64).

    Returns:
        Output tensor [max_num_tokens, actual_num_q_heads, 64].
    """
    (
        actual_num_kv_heads,
        max_num_tokens,
        num_q_heads_per_kv_head_per_q_packing,
        q_packing,
        actual_head_dim_x2,
    ) = out.shape
    actual_num_q_heads = actual_num_q_heads_per_kv_head * actual_num_kv_heads
    return (
        out.swapaxes(0, 1)
        .reshape(
            max_num_tokens,
            actual_num_kv_heads,
            num_q_heads_per_kv_head_per_q_packing * q_packing,
            actual_head_dim_x2,
        )[:, :, :actual_num_q_heads_per_kv_head, actual_head_dim:]
        .reshape(max_num_tokens, actual_num_q_heads, actual_head_dim)
    )


def dynamic_validate_inputs(
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
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
    """Validate inputs with both static and dynamic checks for head_dim=64.

    Performs all static shape/dtype validation, then checks data-dependent
    constraints such as distribution ordering, sequence count limits, token
    count validity, and per-sequence q_len <= kv_len invariants.

    Args:
        queries: Query tensor [total_tokens, num_q_heads, 64].
        keys: Key tensor [total_tokens, num_kv_heads, 64].
        values: Value tensor [total_tokens, num_kv_heads, 64].
        kv_cache: Paged KV cache tensor (K/V concatenated to 128-dim).
        kv_lens: KV context lengths [max_num_seqs].
        block_tables: Flattened page table.
        query_start_loc: Cumulative query positions [max_num_seqs+1].
        distribution: Batch composition [3].
        softmax_aux: Optional attention sink logits.
        softmax_scale: QK^T scaling factor.
        sliding_window: Optional sliding window size.
        logits_soft_cap: Optional logit soft cap.
        mask_value: Causal mask fill value.
        q_scale: Optional query quantization scale.
        k_scale: Optional key quantization scale.
        v_scale: Optional value quantization scale.
        chunk_prefill_size: Prefill chunk size.
        num_kv_pages_per_block: KV pages per kernel block.
        num_queries_per_block: Queries per kernel block.
        vmem_limit_bytes: VMEM budget limit.

    Raises:
        ValueError: If any input constraint is violated.
    """
    q, k, v = queries, keys, values
    static_validate_inputs(
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        softmax_aux,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        chunk_prefill_size=chunk_prefill_size,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    max_num_tokens = q.shape[0]
    total_num_pages = kv_cache.shape[0]
    page_size = kv_cache.shape[1]
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = block_tables.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs

    i, j, k = distribution
    if not (i <= j <= k):
        raise ValueError(f"Invalid distribution: {distribution=}")

    if k > max_num_seqs:
        raise ValueError(f"num_seqs={k} must be <= {max_num_seqs=}")

    if query_start_loc[k] > max_num_tokens:
        raise ValueError(f"Total q tokens {query_start_loc[k]} must be <= {max_num_tokens=}.")
    for i in range(k):
        q_len = query_start_loc[i + 1] - query_start_loc[i]
        kv_len = kv_lens[i]
        if not (0 < q_len <= kv_len):
            raise ValueError(f"Require 0 < {q_len=} <= {kv_len=} at sequence {i}.")
        page_cnt = cdiv(kv_len, page_size)
        if page_cnt > pages_per_seq:
            raise ValueError(
                f"Require {page_cnt=} <= {pages_per_seq=} at sequence {i} where {kv_len=} and {page_size=}."
            )
        for p in range(page_cnt):
            page_idx = block_tables[i * pages_per_seq + p]
            if not (0 <= page_idx < total_num_pages):
                raise ValueError(
                    f"Require 0 <= {page_idx=} < {total_num_pages=} at sequence {i} where {kv_len=} and {page_size=}."
                )


def static_validate_inputs(
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
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
    """Validate static shape and dtype constraints for the head_dim=64 kernel.

    Checks tensor shapes, dtypes, alignment requirements, and parameter
    validity without inspecting data values. Enforces head_dim=64 and
    the K/V concatenated 128-dim cache layout.

    Args:
        queries: Query tensor [total_tokens, num_q_heads, 64].
        keys: Key tensor [total_tokens, num_kv_heads, 64].
        values: Value tensor [total_tokens, num_kv_heads, 64].
        kv_cache: Paged KV cache [pages, page_size, kv_heads/pack, pack, 128].
        kv_lens: KV context lengths [max_num_seqs].
        block_tables: Flattened page table [max_num_seqs * pages_per_seq].
        query_start_loc: Cumulative query positions [max_num_seqs+1].
        distribution: Batch composition [3].
        softmax_aux: Optional attention sink logits [num_q_heads].
        softmax_scale: QK^T scaling factor.
        sliding_window: Optional sliding window size.
        logits_soft_cap: Optional logit soft cap.
        mask_value: Causal mask fill value.
        q_scale: Optional query quantization scale.
        k_scale: Optional key quantization scale.
        v_scale: Optional value quantization scale.
        chunk_prefill_size: Prefill chunk size.
        num_kv_pages_per_block: KV pages per kernel block.
        num_queries_per_block: Queries per kernel block.
        vmem_limit_bytes: VMEM budget limit.

    Raises:
        ValueError: If any shape, dtype, or parameter constraint is violated.
    """
    q, k, v = queries, keys, values
    if not (len(q.shape) == len(k.shape) == len(v.shape) == 3):
        raise ValueError(f"Expected 3D array for {q.shape=}, {k.shape=}, {v.shape=}")
    if k.shape != v.shape:
        raise ValueError(f"Expected {k.shape=} to be equal to {v.shape=}")
    if not (q.shape[0] == k.shape[0] == v.shape[0]):
        raise ValueError(f"Expected {q.shape[0]=} to be equal to {k.shape[0]=} and {v.shape[0]=}")
    if not (q.shape[2] == k.shape[2] == v.shape[2]):
        raise ValueError(f"Expected {q.shape[2]=} to be equal to {k.shape[2]=} and {v.shape[2]=}")
    if softmax_aux is not None:
        if softmax_aux.shape[0] != q.shape[1]:
            raise ValueError(f"Expected {softmax_aux.shape[0]=} to be equal to {q.shape[1]=} (num_q_heads).")
        if softmax_aux.dtype != jnp.float32:
            raise ValueError(f"Expected {softmax_aux.dtype=} to be equal to {jnp.float32=}.")

    actual_head_dim = q.shape[2]
    if actual_head_dim != 64:
        raise ValueError(f"Expected {actual_head_dim=} to be 64.")
    actual_num_q_heads = q.shape[1]
    actual_num_kv_heads = k.shape[1]

    if actual_num_q_heads % actual_num_kv_heads != 0:
        raise ValueError(f"Expected {actual_num_q_heads=} to be divisible by {actual_num_kv_heads=}.")

    (
        _,
        page_size,
        num_kv_heads_per_kv_packing,
        kv_packing,
        actual_head_dim_x2,
    ) = kv_cache.shape

    if actual_head_dim_x2 != 128:
        raise ValueError(f"Expected {actual_head_dim_x2=} is equal to 128")

    if not (kv_cache.dtype == k.dtype == v.dtype):
        raise ValueError(f"Expected {kv_cache.dtype=} to be equal to {k.dtype=} and {v.dtype=}.")

    if not jnp.issubdtype(kv_cache.dtype, jnp.floating):
        raise ValueError(f"Expected {kv_cache.dtype=} to be a floating point.")
    if kv_packing != get_dtype_packing(kv_cache.dtype):
        raise ValueError(f"{kv_packing=} does not match with {kv_cache.dtype=}")

    num_kv_heads = num_kv_heads_per_kv_packing * kv_packing
    if align_to(actual_num_kv_heads, kv_packing) != num_kv_heads:
        raise ValueError(f"Invalid {num_kv_heads=}, {actual_num_kv_heads=}, {kv_packing=}")

    if not (jnp.int32 == kv_lens.dtype == block_tables.dtype == query_start_loc.dtype == distribution.dtype):
        raise ValueError(
            f"Expected int32 dtype for {kv_lens.dtype=}, {block_tables.dtype=},"
            f" {query_start_loc.dtype=}, {distribution.dtype=}"
        )

    if not (len(kv_lens.shape) == len(block_tables.shape) == len(query_start_loc.shape) == 1):
        raise ValueError(f"Expected 1D array for {kv_lens.shape=}, {block_tables.shape=}, {query_start_loc.shape=}")

    max_num_seqs = kv_lens.shape[0]
    num_page_indices = block_tables.shape[0]
    if num_page_indices % max_num_seqs != 0:
        raise ValueError(f"Expected {num_page_indices=} to be divisible by {max_num_seqs=}.")
    if query_start_loc.shape != (max_num_seqs + 1,):
        raise ValueError(f"Expected {query_start_loc.shape=} to be ({max_num_seqs + 1},).")
    if distribution.shape != (3,):
        raise ValueError(f"Expected {distribution.shape=} to be (3,).")

    if page_size % kv_packing != 0:
        raise ValueError(f"{page_size=} must be divisible by {kv_packing=}.")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"{sliding_window=} must be positive.")
    if logits_soft_cap is not None and logits_soft_cap == 0.0:
        raise ValueError(f"{logits_soft_cap=} must not be 0.0.")
    if chunk_prefill_size is not None and chunk_prefill_size <= 0:
        raise ValueError(f"{chunk_prefill_size=} must be positive.")
    if num_kv_pages_per_block is not None:
        if num_kv_pages_per_block <= 0:
            raise ValueError(f"{num_kv_pages_per_block=} must be positive.")
    if num_queries_per_block is not None:
        if num_queries_per_block <= 0:
            raise ValueError(f"{num_queries_per_block=} must be positive.")
    if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
        raise ValueError(f"{vmem_limit_bytes=} must be positive.")

    del softmax_scale
    del mask_value
    del q_scale
    del k_scale
    del v_scale


@ejit(
    static_argnames=(
        "softmax_scale",
        "sliding_window",
        "logits_soft_cap",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "chunk_prefill_size",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
    ),
    donate_argnames=("kv_cache",),
)
def ragged_paged_attention(
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
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
    """Ragged paged attention V3 for head_dim=64 with mixed prefill and decode.

    Specialized entry point for models with head_dim=64 (e.g., GPT-2). Keys and
    values are concatenated into 128-dim vectors for optimal TPU tile utilization.
    Supports simultaneous KV cache updates, sliding window attention, quantized
    inference, and attention sinks.

    Args:
        queries: Concatenated query tokens [total_tokens, num_q_heads, 64].
        keys: Concatenated key tokens [total_tokens, num_kv_heads, 64] to be
            written to cache and used for attention.
        values: Concatenated value tokens [total_tokens, num_kv_heads, 64] to
            be written to cache and used for attention.
        kv_cache: Existing KV cache [pages, page_size, kv_heads/pack, pack, 128]
            with K and V concatenated along last dimension.
        kv_lens: KV context lengths per sequence [max_num_seqs].
        block_tables: Flattened page table [max_num_seqs * pages_per_seq].
        query_start_loc: Cumulative query positions [max_num_seqs+1].
        distribution: Batch composition [3] containing:
            - [0]: End index for decode-only sequences
            - [1]: End index for chunked-prefill sequences
            - [2]: Total number of sequences
        softmax_aux: Optional attention sink logits [num_q_heads].
        softmax_scale: Scaling factor for QK^T (typically 1/sqrt(head_dim)).
        sliding_window: Optional sliding window size for local attention.
        logits_soft_cap: Optional tanh-based soft cap for attention logits.
        mask_value: Fill value for causally masked positions.
        q_scale: Optional quantization scale for queries.
        k_scale: Optional quantization scale for keys.
        v_scale: Optional quantization scale for values.
        chunk_prefill_size: Chunk size for prefill processing.
        num_kv_pages_per_block: KV pages per flash attention block.
        num_queries_per_block: Query tokens per flash attention block.
        vmem_limit_bytes: VMEM budget limit for the Pallas kernel.

    Returns:
        Tuple of (attention_output [total_tokens, num_q_heads, 64],
        updated_kv_cache with new tokens written).

    Note:
        Uses get_tuned_block_sizes_h64() for optimal block size selection
        when num_kv_pages_per_block or num_queries_per_block is not specified.
    """
    q, k, v = queries, keys, values
    static_validate_inputs(
        q,
        k,
        v,
        kv_cache,
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        softmax_aux,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        chunk_prefill_size=chunk_prefill_size,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
    )

    actual_num_q_heads = q.shape[1]
    actual_head_dim = q.shape[2]
    actual_num_kv_heads = k.shape[1]

    actual_num_q_heads_per_kv_head = actual_num_q_heads // actual_num_kv_heads
    q, kv, softmax_aux = prepare_inputs(q, k, v, softmax_aux)
    (
        _,
        max_num_tokens,
        num_q_heads_per_kv_head_per_q_packing,
        q_packing,
        head_dim,
    ) = q.shape
    page_size = kv_cache.shape[1]
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = block_tables.shape[0]
    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs
    num_q_heads_per_kv_head = num_q_heads_per_kv_head_per_q_packing * q_packing

    bkv_p = num_kv_pages_per_block
    bq_sz = num_queries_per_block
    if bq_sz is None or bkv_p is None:
        bkv_p, bq_sz = get_tuned_block_sizes_h64(
            q.dtype,
            kv_cache.dtype,
            actual_num_q_heads,
            actual_num_kv_heads,
            actual_head_dim,
            page_size,
            max_num_tokens,
            pages_per_seq,
        )
    bkv_p, bq_sz = _clamp_block_sizes_to_vmem(
        actual_num_kv_heads=actual_num_kv_heads,
        actual_num_q_heads_per_kv_head=actual_num_q_heads_per_kv_head,
        actual_head_dim=actual_head_dim,
        page_size=page_size,
        q_dtype=q.dtype,
        kv_dtype=kv_cache.dtype,
        bkv_p=bkv_p,
        bq_sz=bq_sz,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    bkv_sz = bkv_p * page_size

    grid = (distribution[2],)

    in_specs = [
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.HBM),
        None if softmax_aux is None else pl.BlockSpec(memory_space=pltpu.VMEM),
    ]

    out_specs = [
        pl.BlockSpec(memory_space=pltpu.HBM),
        pl.BlockSpec(memory_space=pltpu.HBM),
    ]

    bkv_double_buf = pltpu.VMEM((2, bkv_sz, *kv_cache.shape[2:]), kv_cache.dtype)

    bq_double_buf = pltpu.VMEM((2, actual_num_kv_heads, bq_sz, *q.shape[2:]), q.dtype)

    bo_double_buf = bq_double_buf

    l_scratch = pltpu.VMEM(
        (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, 128),
        jnp.float32,
    )
    m_scratch = l_scratch

    acc_scratch = pltpu.VMEM(
        (actual_num_kv_heads, bq_sz * num_q_heads_per_kv_head, head_dim),
        jnp.float32,
    )

    scratch_shapes = [
        bkv_double_buf,
        bq_double_buf,
        bo_double_buf,
        pltpu.SemaphoreType.DMA((4, 2)),
        l_scratch,
        m_scratch,
        acc_scratch,
    ]

    scalar_prefetches = (
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        jnp.zeros((3,), jnp.int32),
        jnp.full((4,), -1, jnp.int32),
        jnp.full((6,), -1, jnp.int32),
    )

    scope_name = f"RPA-HD_64-bq_{bq_sz}-bkvp_{bkv_p}-p_{page_size}"
    kernel = jax.named_scope(scope_name)(
        pl.pallas_call(
            functools.partial(
                _ragged_paged_attention_kernel,
                softmax_scale=softmax_scale,
                sliding_window=sliding_window,
                logits_soft_cap=logits_soft_cap,
                mask_value=mask_value,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                chunk_prefill_size=chunk_prefill_size,
                bq_sz=bq_sz,
                bkv_p=bkv_p,
            ),
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=len(scalar_prefetches),
                in_specs=in_specs,
                out_specs=out_specs,
                grid=grid,
                scratch_shapes=scratch_shapes,
            ),
            compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",), vmem_limit_bytes=vmem_limit_bytes),
            out_shape=[
                jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),
                jax.ShapeDtypeStruct(shape=kv_cache.shape, dtype=kv_cache.dtype),
            ],
            input_output_aliases={7: 0, 9: 1},
            name=scope_name,
        )
    )

    output, updated_kv_cache = kernel(*scalar_prefetches, q, kv, kv_cache, softmax_aux)
    return (
        prepare_outputs(output, actual_num_q_heads_per_kv_head, actual_head_dim),
        updated_kv_cache,
    )
