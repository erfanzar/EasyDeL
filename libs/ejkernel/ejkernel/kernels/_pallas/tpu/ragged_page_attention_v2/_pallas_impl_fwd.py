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

"""Ragged Paged Attention V2 TPU kernel implementation using Pallas.

This module provides an optimized ragged paged attention implementation for
Google TPUs, designed to handle variable-length (ragged) sequences within a
batch while using a paged key-value cache. This is version 2 of the kernel
with improved async DMA handling and double-buffering.

Algorithm Overview:
    1. Multi-page async copy: Prefetch multiple KV pages from HBM to VMEM
       using asynchronous DMA transfers for latency hiding
    2. Interleaved K/V storage: Keys and values are interleaved in the cache
       (even indices = K, odd indices = V) for efficient memory access
    3. Online softmax with Flash Attention: Process KV blocks incrementally
       using the online softmax algorithm to maintain numerical stability
    4. Double-buffering: Overlap computation with memory transfers using
       two VMEM buffers for KV data
    5. Causal masking with sliding window support: Optionally apply sliding
       window attention for efficiency on long sequences

Key Features:
    - Ragged batch support: Efficiently handles variable query lengths within
      a batch using cumulative query lengths (cu_q_lens)
    - Paged KV cache: Non-contiguous memory layout using page tables for
      efficient memory management during inference
    - Async DMA prefetching: Overlaps memory transfers with computation for
      better TPU utilization
    - Softmax auxiliary logits: Optional attention sink support for streaming
      inference patterns
    - Logit soft capping: Optional tanh-based soft capping for numerical stability
    - Mixed precision: Supports both bfloat16 and float32 with efficient
      bitcast operations for packed data

Memory Layout:
    - KV pages: [num_pages, page_size, num_combined_kv_heads, head_dim]
      where num_combined_kv_heads includes interleaved K and V
    - Queries: [total_tokens, num_q_heads, head_dim]
    - Block tables: [max_num_seqs, pages_per_seq] for page index lookup

Example:
    >>> # Ragged attention with multiple sequences of different lengths
    >>> outputs = ragged_page_attention_kernel(
    ...     context_lens_ref=context_lens,    # [4, 8, 12]
    ...     page_indices_ref=page_indices,    # Page table for KV lookup
    ...     cu_q_lens_ref=cu_q_lens,          # [0, 2, 5, 9] cumulative
    ...     q_ref=queries,
    ...     kv_pages_hbm_ref=kv_pages,
    ...     softmax_scale=1.0 / sqrt(head_dim),
    ... )

Note:
    This kernel is optimized for TPU v4 and later generations with enhanced
    MXU and VMEM capabilities. The async copy descriptors enable efficient
    page prefetching while the kernel processes previous blocks.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


class MultiPageAsyncCopyDescriptor:
    """Descriptor for asynchronous DMA copy of multiple KV cache pages from HBM to VMEM.

    Manages the async transfer of multiple pages from the paged KV cache stored
    in HBM into a VMEM buffer on TPU. Handles boundary conditions where requested
    pages may exceed the valid range by mapping out-of-bounds indices to page 0.

    Attributes:
        _vmem_buf: VMEM buffer reference for storing the loaded pages.
        _async_copies: List of individual async copy operations, one per page slot.
    """

    def __init__(
        self,
        pages_hbm_ref,
        vmem_buf,
        sem,
        page_indices_ref,
        metadata,
    ):
        self._vmem_buf = vmem_buf
        seq_id, start_page_idx, end_page_idx = metadata
        self._async_copies = []
        for i in range(vmem_buf.shape[0]):
            page_idx = start_page_idx + i
            page_idx = jax.lax.select(page_idx < end_page_idx, page_idx, 0)
            self._async_copies.append(
                pltpu.make_async_copy(
                    pages_hbm_ref.at[page_indices_ref[seq_id, page_idx]],
                    vmem_buf.at[i],
                    sem,
                )
            )

    def start(self):
        """Starts the async copies."""
        for async_copy in self._async_copies:
            async_copy.start()

    def wait(self):
        """Wait for all async copies to complete and return the VMEM buffer.

        Returns:
            The VMEM buffer reference containing the loaded page data.
        """
        for async_copy in self._async_copies:
            async_copy.wait()
        return self._vmem_buf


def ref_ragged_page_attention(
    queries: jax.Array,
    kv_pages: jax.Array,
    context_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    num_seqs: jax.Array,
    *,
    softmax_scale: float = 1.0,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    softmax_aux: jax.Array | None = None,
):
    """Reference implementation of ragged paged attention for correctness testing.

    Processes each sequence sequentially using standard einsum-based attention.
    This is not optimized for performance but serves as a ground truth for
    validating the Pallas kernel implementation.

    The function iterates over sequences, gathers KV from pages, applies
    causal masking (with optional sliding window and soft capping), computes
    softmax attention, and concatenates outputs.

    Args:
        queries: Concatenated query tokens [total_tokens, num_q_heads, head_dim].
        kv_pages: Paged KV cache [num_pages, page_size, num_combined_kv_heads, head_dim].
            Keys at even indices, values at odd indices in dim 2.
        context_lens: KV context lengths per sequence [max_num_seqs].
        block_tables: Page table mapping [max_num_seqs, pages_per_seq].
        query_start_loc: Cumulative query positions [max_num_seqs+1].
        num_seqs: Array of shape [1] with the dynamic number of sequences.
        softmax_scale: Scaling factor for QK^T. Defaults to 1.0.
        sliding_window: Optional window size for local attention.
        logits_soft_cap: Optional soft cap for attention logits.
        mask_value: Value for masked positions. Defaults to DEFAULT_MASK_VALUE.
        softmax_aux: Optional attention sink logits [num_q_heads].

    Returns:
        Attention output [total_tokens, num_q_heads, head_dim] matching queries layout.
    """
    static_validate_inputs(
        queries,
        kv_pages,
        context_lens,
        block_tables,
        query_start_loc,
        num_seqs,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        mask_value=mask_value,
    )
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE
    _, _, num_combined_kv_heads, head_dim = kv_pages.shape
    assert num_combined_kv_heads % 2 == 0
    num_kv_heads = num_combined_kv_heads // 2
    num_q_heads = queries.shape[1]
    assert num_q_heads % num_kv_heads == 0
    num_query_per_kv = num_q_heads // num_kv_heads
    outputs = []
    for i in range(num_seqs[0]):
        q_start = query_start_loc[i]
        q_end = query_start_loc[i + 1]
        q_len = q_end - q_start
        kv_len = context_lens[i]
        indices = block_tables[i]
        q = queries[q_start:q_end]
        k = kv_pages[indices, :, 0::2, :].reshape(-1, num_kv_heads, head_dim)[:kv_len]
        v = kv_pages[indices, :, 1::2, :].reshape(-1, num_kv_heads, head_dim)[:kv_len]
        k = jnp.repeat(k, num_query_per_kv, axis=1)
        v = jnp.repeat(v, num_query_per_kv, axis=1)
        attn = jnp.einsum("qhd,khd->hqk", q, k, preferred_element_type=jnp.float32)
        attn *= softmax_scale
        q_span = (kv_len - q_len) + jax.lax.broadcasted_iota(jnp.int32, attn.shape, 1)
        kv_span = jax.lax.broadcasted_iota(jnp.int32, attn.shape, 2)
        mask = q_span < kv_span
        if sliding_window is not None:
            mask = jnp.logical_or(mask, q_span - sliding_window >= kv_span)
        if logits_soft_cap is not None:
            attn = logits_soft_cap * jnp.tanh(attn / logits_soft_cap)
        attn += jnp.where(mask, mask_value, 0.0)

        if softmax_aux is not None:
            reshaped_sink = softmax_aux.reshape(num_q_heads, 1, 1)
            reshaped_sink = jnp.repeat(reshaped_sink, q_len, axis=1)
            attn = jnp.concatenate([reshaped_sink, attn], axis=2)
            attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)
            attn = attn[..., 1:]
        else:
            attn = jax.nn.softmax(attn, axis=-1).astype(v.dtype)

        out = jnp.einsum("hqk,khd->qhd", attn, v).astype(queries.dtype)
        outputs.append(out)

    return jnp.concatenate(outputs, axis=0)


def dynamic_validate_inputs(
    q: jax.Array,
    kv_pages: jax.Array,
    context_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    num_seqs: jax.Array,
    *,
    softmax_scale: float | None = None,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    mask_value: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
    """Validate inputs with both static shape checks and dynamic value checks.

    Extends static_validate_inputs with runtime checks on actual tensor values.
    This includes verifying that num_seqs does not exceed the maximum, that
    pages_per_seq is sufficient for the maximum KV length, that total query
    tokens fit within the buffer, and that query lengths do not exceed KV lengths.

    Args:
        q: Query tensor [max_num_batched_tokens, num_q_heads, head_dim].
        kv_pages: Paged KV cache tensor.
        context_lens: KV context lengths per sequence.
        block_tables: Page table mapping.
        query_start_loc: Cumulative query positions.
        num_seqs: Dynamic number of sequences [1].
        softmax_scale: Attention scaling factor.
        sliding_window: Optional sliding window size.
        logits_soft_cap: Optional logit soft cap value.
        mask_value: Mask value for causal masking.
        num_kv_pages_per_block: KV pages per compute block.
        num_queries_per_block: Queries per compute block.
        vmem_limit_bytes: VMEM budget limit.

    Raises:
        ValueError: If any dynamic constraint is violated.
    """
    static_validate_inputs(
        q,
        kv_pages,
        context_lens,
        block_tables,
        query_start_loc,
        num_seqs,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        logits_soft_cap=logits_soft_cap,
        mask_value=mask_value,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
    )
    max_num_batched_tokens = q.shape[0]
    page_size = kv_pages.shape[1]
    max_num_seqs, pages_per_seq = block_tables.shape
    if num_seqs[0] > max_num_seqs:
        raise ValueError(f"{num_seqs[0]=} must be less or equal to {max_num_seqs=}")
    max_kv_len = jnp.max(context_lens)
    min_pages_per_seq = cdiv(max_kv_len, page_size)
    if pages_per_seq < min_pages_per_seq:
        raise ValueError(
            f"{pages_per_seq=} must be greater or equal to {min_pages_per_seq=} given {max_kv_len=} and {page_size=}."
        )
    if query_start_loc[num_seqs[0]] > max_num_batched_tokens:
        raise ValueError(
            f"Total q tokens {query_start_loc[num_seqs[0]]} must be less or equal to {max_num_batched_tokens=}."
        )
    for i in range(num_seqs[0]):
        q_len = query_start_loc[i + 1] - query_start_loc[i]
        kv_len = context_lens[i]
        if q_len > kv_len:
            raise ValueError(f"{q_len=} must be less or equal to {kv_len=} at sequence {i}.")


def static_validate_inputs(
    q: jax.Array,
    kv_pages: jax.Array,
    context_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    num_seqs: jax.Array,
    *,
    softmax_scale: float | None = None,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    mask_value: float | None = None,
    num_kv_pages_per_block: int | None = None,
    num_queries_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
):
    """Validate input tensor shapes, dtypes, and static parameter constraints.

    Performs compile-time checks on tensor shapes and dtypes to catch
    configuration errors early. Validates head dimension consistency,
    GQA head ratio divisibility, and parameter value ranges.

    Args:
        q: Query tensor [max_num_batched_tokens, num_q_heads, head_dim].
        kv_pages: Paged KV cache [num_pages, page_size, num_combined_kv_heads, head_dim].
        context_lens: KV lengths [max_num_seqs].
        block_tables: Page indices [max_num_seqs, pages_per_seq].
        query_start_loc: Cumulative query positions [max_num_seqs+1].
        num_seqs: Number of sequences [1].
        softmax_scale: Attention scaling factor (not validated).
        sliding_window: Must be positive if provided.
        logits_soft_cap: Must be non-zero if provided.
        mask_value: Mask value (not validated).
        num_kv_pages_per_block: Must be in (0, pages_per_seq] if provided.
        num_queries_per_block: Must be positive if provided.
        vmem_limit_bytes: Must be positive if provided.

    Raises:
        ValueError: If any static constraint is violated.
    """
    _, num_q_heads, head_dim = q.shape
    _, _, num_combined_kv_heads, head_dim_k = kv_pages.shape
    assert num_combined_kv_heads % 2 == 0
    num_kv_heads = num_combined_kv_heads // 2
    max_num_seqs, pages_per_seq = block_tables.shape
    if num_seqs.shape != (1,):
        raise ValueError(f"{num_seqs.shape=} must be (1,)")
    if head_dim_k != head_dim:
        raise ValueError(f"Q head_dim {head_dim} must be the same as that of K/V {head_dim_k}.")
    if context_lens.shape != (max_num_seqs,):
        raise ValueError(
            f"Expected {context_lens.shape=} to be ({max_num_seqs},) where `max_num_seqs` is `block_tables.shape[0]`."
        )
    if query_start_loc.shape != (max_num_seqs + 1,):
        raise ValueError(
            f"Expected {query_start_loc.shape=} to be ({max_num_seqs + 1},)  where "
            f"`max_num_seqs` is `block_tables.shape[0]`."
        )
    if context_lens.dtype != jnp.int32 or block_tables.dtype != jnp.int32 or query_start_loc.dtype != jnp.int32:
        raise ValueError(
            "The dtype of `context_lens`, `block_tables`, and `query_start_loc` must be"
            f" int32. Got {context_lens.dtype=}, {block_tables.dtype=},"
            f" {query_start_loc.dtype=}."
        )
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(f"{num_q_heads=} must be divisible by {num_kv_heads=}")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"{sliding_window=} must be positive.")
    if logits_soft_cap is not None and logits_soft_cap == 0.0:
        raise ValueError(f"{logits_soft_cap=} must not be 0.0.")
    if num_kv_pages_per_block is not None and not 0 < num_kv_pages_per_block <= pages_per_seq:
        raise ValueError(f"{num_kv_pages_per_block=} must be in range (0, {pages_per_seq}].")
    if num_queries_per_block is not None and num_queries_per_block <= 0:
        raise ValueError(f"{num_queries_per_block=} must be positive.")
    if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
        raise ValueError(f"{vmem_limit_bytes=} must be positive.")
    del softmax_scale
    del mask_value


def ragged_page_attention_kernel(
    context_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    seq_buf_idx_ref,
    num_seqs_ref,
    q_ref,
    kv_pages_hbm_ref,
    softmax_aux_ref,
    o_ref,
    kv_bufs,
    sems,
    l_ref,
    m_ref,
    acc_ref,
    *,
    softmax_scale: float,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
):
    """Pallas kernel for ragged paged attention with async DMA and online softmax.

    This is the main TPU kernel that processes ragged batches of queries against
    paged KV caches. It uses a two-level loop structure: the outer loop iterates
    over sequences that overlap with the current query block, and the inner loop
    iterates over KV blocks within each sequence.

    The kernel uses double-buffered async DMA to prefetch KV pages from HBM
    while computing attention on the current block. For each KV block, it
    performs flash attention with online softmax updates.

    Args:
        context_lens_ref: Scalar prefetch ref for KV context lengths [max_num_seqs].
        page_indices_ref: Scalar prefetch ref for page tables [max_num_seqs, pages_per_seq].
        cu_q_lens_ref: Scalar prefetch ref for cumulative query lengths [max_num_seqs+1].
        seq_buf_idx_ref: Scalar prefetch ref for (current_seq_idx, current_buffer_idx).
        num_seqs_ref: Scalar prefetch ref for dynamic number of sequences [1].
        q_ref: Query block ref [num_q_per_blk, num_q_heads_per_blk, head_dim].
        kv_pages_hbm_ref: KV cache HBM ref [num_pages, page_size, num_combined_kv_heads, head_dim].
        softmax_aux_ref: Attention sink logits ref in VMEM.
        o_ref: Output ref [num_q_per_blk, num_q_heads_per_blk, head_dim].
        kv_bufs: Double-buffered VMEM scratch for KV page loading.
        sems: DMA semaphores for async copy synchronization [2].
        l_ref: Softmax denominator accumulator scratch in VMEM.
        m_ref: Softmax max accumulator scratch in VMEM.
        acc_ref: Output accumulator scratch in VMEM.
        softmax_scale: Scaling factor for attention logits.
        sliding_window: Optional sliding window size for local attention.
        logits_soft_cap: Optional soft cap value for attention logits.
        mask_value: Value for masked positions in the attention matrix.
    """
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE
    num_q_per_blk, num_q_heads_per_blk, head_dim = q_ref.shape
    pages_per_seq = page_indices_ref.shape[-1]
    num_seqs = num_seqs_ref[0]
    _, num_kv_pages_per_blk, page_size, num_combined_kv_heads_per_blk, _ = kv_bufs.shape
    num_kv_heads_per_blk = num_combined_kv_heads_per_blk // 2
    num_kv_per_blk = num_kv_pages_per_blk * page_size
    num_q_heads_per_kv_head = num_q_heads_per_blk // num_kv_heads_per_blk
    heads_blk_idx, q_blk_idx = (
        pl.program_id(0),
        pl.program_id(1),
    )
    num_heads_blks = pl.num_programs(0)
    init_seq_idx = seq_buf_idx_ref[0]
    init_buf_idx = seq_buf_idx_ref[1]
    q_len_start = q_blk_idx * num_q_per_blk
    q_len_end = q_len_start + num_q_per_blk

    def create_kv_async_copy_descriptors(heads_blk_idx, seq_idx, kv_blk_idx, buf_idx):
        start_kv_page_idx = kv_blk_idx * num_kv_pages_per_blk
        end_kv_page_idx = jnp.minimum(pages_per_seq, cdiv(context_lens_ref[seq_idx], page_size))
        metadata = (seq_idx, start_kv_page_idx, end_kv_page_idx)
        heads_start = heads_blk_idx * num_combined_kv_heads_per_blk
        async_copy_kv = MultiPageAsyncCopyDescriptor(
            kv_pages_hbm_ref.at[:, :, pl.ds(heads_start, num_combined_kv_heads_per_blk), :],
            kv_bufs.at[buf_idx],
            sems.at[buf_idx],
            page_indices_ref,
            metadata,
        )
        return async_copy_kv

    def strided_load_kv(ref, start, step):
        packing = get_dtype_packing(ref.dtype)
        if packing == 1:
            return [ref[start::step, :]], [ref[start + 1 :: step, :]]
        assert packing in (2, 4, 8)
        assert step % packing == 0
        k_list, v_list = [], []
        b_start = start // packing
        b_step = step // packing
        b_ref = ref.bitcast(jnp.uint32)
        b = b_ref[b_start::b_step, :]

        if ref.dtype == jnp.bfloat16:
            bk = b << 16
            bv = b & jnp.uint32(0xFFFF0000)
            k = pltpu.bitcast(bk, jnp.float32).astype(jnp.bfloat16)
            v = pltpu.bitcast(bv, jnp.float32).astype(jnp.bfloat16)
            k_list.append(k)
            v_list.append(v)
        else:
            bitwidth = 32 // packing
            bitcast_dst_dtype = jnp.dtype(f"uint{bitwidth}")
            for i in range(0, packing, 2):
                bk = b >> (i * bitwidth)
                k = pltpu.bitcast(bk.astype(bitcast_dst_dtype), ref.dtype)
                k_list.append(k)
                bv = b >> ((i + 1) * bitwidth)
                v = pltpu.bitcast(bv.astype(bitcast_dst_dtype), ref.dtype)
                v_list.append(v)

        return k_list, v_list

    def fold_on_2nd_minor(vec):
        assert vec.dtype == jnp.bfloat16 or vec.dtype == jnp.float32
        assert len(vec.shape) >= 2
        last_dim = vec.shape[-1]
        packing = get_dtype_packing(vec.dtype)
        if vec.shape[-2] % packing != 0:
            vec = vec.astype(jnp.float32)
        return vec.reshape(-1, last_dim)

    @pl.when(heads_blk_idx + q_blk_idx == 0)
    def prefetch_first_kv_blk():
        async_copy_kv = create_kv_async_copy_descriptors(heads_blk_idx, init_seq_idx, 0, init_buf_idx)
        async_copy_kv.start()

    def is_cur_q_blk_needed(q_states):
        done, cur_seq_idx, _ = q_states
        should_run = jnp.logical_and(q_len_start < cu_q_lens_ref[num_seqs], cur_seq_idx < num_seqs)
        return jnp.logical_and(done == 0, should_run)

    def compute_with_cur_q_blk(q_states):
        done, cur_seq_idx, cur_buf_idx = q_states
        q_start = cu_q_lens_ref[cur_seq_idx]
        q_end = cu_q_lens_ref[cur_seq_idx + 1]
        q_len = q_end - q_start
        kv_len = context_lens_ref[cur_seq_idx]

        def get_next_prefetch_ids(heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx):
            next_kv_blk_idx = kv_blk_idx + 1
            is_last_kv_blk = next_kv_blk_idx * num_kv_per_blk >= kv_len
            next_kv_blk_idx = lax.select(
                is_last_kv_blk,
                0,
                next_kv_blk_idx,
            )
            is_cur_seq_end_in_cur_q_blk = q_end <= q_len_end
            next_seq_idx = lax.select(
                is_last_kv_blk,
                lax.select(is_cur_seq_end_in_cur_q_blk, cur_seq_idx + 1, cur_seq_idx),
                cur_seq_idx,
            )
            is_last_seq = next_seq_idx == num_seqs
            next_seq_idx = lax.select(
                is_last_seq,
                0,
                next_seq_idx,
            )
            next_heads_blk_idx = lax.select(
                is_last_seq,
                heads_blk_idx + 1,
                heads_blk_idx,
            )
            next_buf_idx = lax.select(cur_buf_idx == 0, 1, 0)
            return next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx

        def flash_attention(
            q,
            k,
            v,
            head_l_ref,
            head_m_ref,
            head_acc_ref,
            softmax_aux_ref,
            *,
            kv_blk_idx,
        ):
            assert q.shape == (num_q_per_blk * num_q_heads_per_kv_head, head_dim)
            assert k.shape == v.shape == (num_kv_per_blk, head_dim)
            assert k.dtype == v.dtype
            assert head_m_ref.shape == head_l_ref.shape == (num_q_per_blk * num_q_heads_per_kv_head, head_dim)
            assert head_acc_ref.shape == (num_q_per_blk, num_q_heads_per_kv_head, head_dim)
            kv_len_start = kv_blk_idx * num_kv_per_blk

            def masked_store(ref, val, start, end, group=1):
                iota = lax.broadcasted_iota(jnp.int32, ref.shape, 0) // group
                mask = jnp.logical_and(iota >= start, iota < end)
                ref[...] = jnp.where(mask, val, ref[...])

            def load_with_init(ref, init_val):
                return jnp.where(kv_blk_idx == 0, init_val, ref[...])

            kv_mask = lax.broadcasted_iota(jnp.int32, k.shape, 0) < kv_len - kv_len_start
            k = jnp.where(kv_mask, k, 0)
            v = jnp.where(kv_mask, v, 0)

            sinks = softmax_aux_ref[...]
            m_aux = jnp.concatenate([sinks] * num_q_per_blk, axis=0)
            l_aux = jnp.ones_like(m_aux)

            qk = jnp.einsum("nd,md->nm", q, k, preferred_element_type=jnp.float32) * softmax_scale
            store_start = jnp.maximum(q_start - q_len_start, 0)
            store_end = jnp.minimum(q_end - q_len_start, num_q_per_blk)

            row_ids = (
                (kv_len - q_len)
                + q_len_start
                - q_start
                + jax.lax.broadcasted_iota(
                    jnp.int32,
                    (num_q_per_blk * num_q_heads_per_kv_head, num_kv_per_blk),
                    0,
                )
                // num_q_heads_per_kv_head
            )
            col_ids = kv_len_start + jax.lax.broadcasted_iota(
                jnp.int32,
                (num_q_per_blk * num_q_heads_per_kv_head, num_kv_per_blk),
                1,
            )
            causal_mask = row_ids < col_ids
            if sliding_window is not None:
                causal_mask = jnp.logical_or(causal_mask, row_ids - sliding_window >= col_ids)
            if logits_soft_cap is not None:
                qk = logits_soft_cap * jnp.tanh(qk / logits_soft_cap)
            qk += jnp.where(causal_mask, mask_value, 0.0)
            m_curr = jnp.max(qk, axis=1, keepdims=True)
            s_curr = jnp.exp(qk - m_curr)
            qkv = jnp.dot(s_curr, v, preferred_element_type=jnp.float32)
            lm_store_shape = head_m_ref.shape
            m_curr = jnp.broadcast_to(m_curr, lm_store_shape)
            l_curr = jnp.broadcast_to(s_curr.sum(axis=1, keepdims=True), lm_store_shape)
            m_prev = load_with_init(head_m_ref, m_aux)
            l_prev = load_with_init(head_l_ref, l_aux)
            m_next = jnp.maximum(m_prev, m_curr)
            masked_store(head_m_ref, m_next, store_start, store_end, num_q_heads_per_kv_head)
            alpha = jnp.exp(m_prev - m_next)
            beta = jnp.exp(m_curr - m_next)
            l_alpha = alpha * l_prev
            l_next = l_alpha + beta * l_curr
            l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)
            masked_store(
                head_l_ref,
                l_next_safe,
                store_start,
                store_end,
                num_q_heads_per_kv_head,
            )

            def broadcast_to_shape(arr, shape):
                if arr.shape == shape:
                    return arr
                try:
                    return jnp.broadcast_to(arr, shape)
                except Exception:
                    pass

                if len(arr.shape) == len(shape) and arr.shape[0] == shape[0]:
                    if arr.shape[1] > shape[1]:
                        return arr[:, : shape[1]]
                    if shape[1] % arr.shape[1] == 0:
                        return jnp.concatenate([arr for _ in range(shape[1] // arr.shape[1])], axis=1)

                return jnp.broadcast_to(arr, shape)

            o_curr = load_with_init(head_acc_ref, 0.0).reshape(-1, head_dim)

            l_alpha = broadcast_to_shape(l_alpha, qkv.shape)
            beta = broadcast_to_shape(beta, qkv.shape)
            l_next_safe = broadcast_to_shape(l_next_safe, qkv.shape)
            out = lax.div(
                l_alpha * o_curr + beta * qkv,
                l_next_safe,
            )
            masked_store(
                head_acc_ref,
                out.reshape(head_acc_ref.shape),
                store_start,
                store_end,
            )

        def is_valid_kv_blk_in_cur_seq(kv_states):
            kv_blk_idx, _ = kv_states
            return kv_blk_idx * num_kv_per_blk < kv_len

        def compute_with_kv_blk_in_cur_seq(kv_states):
            kv_blk_idx, cur_buf_idx = kv_states
            next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx = get_next_prefetch_ids(
                heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx
            )

            @pl.when(next_heads_blk_idx < num_heads_blks)
            def prefetch_next_kv_blk():
                next_async_copy_kv = create_kv_async_copy_descriptors(
                    next_heads_blk_idx, next_seq_idx, next_kv_blk_idx, next_buf_idx
                )
                next_async_copy_kv.start()

            cur_async_copy_kv = create_kv_async_copy_descriptors(heads_blk_idx, cur_seq_idx, kv_blk_idx, cur_buf_idx)
            kv_ref = cur_async_copy_kv.wait().reshape(
                num_kv_pages_per_blk * page_size * num_combined_kv_heads_per_blk,
                head_dim,
            )
            kv_packing = get_dtype_packing(kv_ref.dtype)

            kv_load_step = max(1, kv_packing // 2)
            for kv_head_chunk_idx in range(0, num_kv_heads_per_blk, kv_load_step):
                k_list, v_list = strided_load_kv(kv_ref, kv_head_chunk_idx * 2, num_combined_kv_heads_per_blk)
                for step_idx in range(kv_load_step):
                    k = k_list[step_idx]
                    v = v_list[step_idx]
                    kv_head_idx = kv_head_chunk_idx + step_idx
                    global_kv_head_idx = heads_blk_idx * num_kv_heads_per_blk + kv_head_idx
                    q_head_idx = kv_head_idx * num_q_heads_per_kv_head
                    q = fold_on_2nd_minor(q_ref[:, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :])
                    flash_attention(
                        q,
                        k,
                        v,
                        l_ref.at[kv_head_idx],
                        m_ref.at[kv_head_idx],
                        acc_ref.at[:, q_head_idx : q_head_idx + num_q_heads_per_kv_head, :],
                        softmax_aux_ref[global_kv_head_idx],
                        kv_blk_idx=kv_blk_idx,
                    )
            return kv_blk_idx + 1, next_buf_idx

        _, next_buf_idx = lax.while_loop(
            is_valid_kv_blk_in_cur_seq,
            compute_with_kv_blk_in_cur_seq,
            (0, cur_buf_idx),
        )
        next_seq_idx = lax.select(q_end <= q_len_end, cur_seq_idx + 1, cur_seq_idx)
        done = lax.select(q_end < q_len_end, done, 1)
        return done, next_seq_idx, next_buf_idx

    _, seq_idx, buf_idx = lax.while_loop(
        is_cur_q_blk_needed,
        compute_with_cur_q_blk,
        (0, init_seq_idx, init_buf_idx),
    )

    seq_buf_idx_ref[0] = lax.select(seq_idx < num_seqs, seq_idx, 0)
    seq_buf_idx_ref[1] = buf_idx
    o_ref[...] = acc_ref[...].astype(q_ref.dtype)


def cdiv(a, b):
    """Compute ceiling division of a by b.

    Args:
        a: Dividend (numerator).
        b: Divisor (denominator). Must be non-zero.

    Returns:
        The smallest integer >= a/b.
    """
    assert b != 0
    return (a + b - 1) // b


def get_dtype_packing(dtype):
    """Compute how many elements of the given dtype fit in a 32-bit word.

    Args:
        dtype: JAX/NumPy dtype to compute packing for.

    Returns:
        Number of elements that fit in 32 bits (e.g., 2 for bfloat16, 1 for float32).
    """
    bits = jnp.dtype(dtype).itemsize * 8
    return 32 // bits


def get_min_heads_per_blk(num_q_heads, num_combined_kv_heads, q_dtype, kv_dtype):
    """Determine the minimum number of heads per processing block for XLA tiling.

    Computes the smallest number of query and combined KV heads per block that
    satisfies XLA's tiling constraints. XLA requires certain dimensions to be
    fully tileable (i.e., divisible by the dtype packing factor and resulting
    in a valid tile size like 1, 2, 4, 8, or a multiple of 8).

    Args:
        num_q_heads: Total number of query heads.
        num_combined_kv_heads: Total number of combined KV heads (K and V interleaved).
        q_dtype: Query tensor dtype for packing calculation.
        kv_dtype: KV cache dtype for packing calculation.

    Returns:
        Tuple of (num_q_heads_per_blk, num_combined_kv_heads_per_blk) that satisfies
        XLA tiling constraints.

    Raises:
        ValueError: If num_combined_kv_heads cannot be XLA fully tiled.
    """
    q_packing = get_dtype_packing(q_dtype)
    kv_packing = get_dtype_packing(kv_dtype)

    def can_be_xla_fully_tiled(x, packing):
        if x % packing != 0:
            return False
        x //= packing
        return x in (1, 2, 4, 8) or x % 8 == 0

    if not can_be_xla_fully_tiled(num_combined_kv_heads, kv_packing):
        raise ValueError(f"Not implemented: {num_combined_kv_heads=} can not be XLA fully tiled.")
    assert num_combined_kv_heads % 2 == 0
    num_kv_heads = num_combined_kv_heads // 2
    assert num_q_heads % num_kv_heads == 0
    ratio = num_q_heads // num_kv_heads

    max_combined_kv_tiling = 8 * kv_packing
    min_combined_kv_heads = (
        max_combined_kv_tiling if num_combined_kv_heads % max_combined_kv_tiling == 0 else num_combined_kv_heads
    )
    min_q_heads = min_combined_kv_heads // 2 * ratio
    if can_be_xla_fully_tiled(min_q_heads, q_packing):
        return min_q_heads, min_combined_kv_heads
    return num_q_heads, num_combined_kv_heads
