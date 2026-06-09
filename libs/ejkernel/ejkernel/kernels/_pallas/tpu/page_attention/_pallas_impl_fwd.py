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

# Reference: JAX Pallas PagedAttention TPU kernel
# https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/paged_attention/paged_attention_kernel.py

"""Paged Attention TPU kernel implementation using Pallas.

This module provides an optimized paged attention implementation for Google TPUs
designed for efficient inference with large key-value caches. Paged attention
enables non-contiguous memory allocation for KV cache, reducing memory
fragmentation and enabling efficient batched inference.

Algorithm Overview:
    The paged attention algorithm processes queries against a paged KV cache:
    1. Load query for current position
    2. Iterate through pages in the KV cache using block tables
    3. Asynchronously prefetch next page while computing current attention
    4. Apply online softmax for memory-efficient attention computation
    5. Accumulate weighted values and normalize

Key Features:
    - Non-contiguous KV cache with page-based memory management
    - Async page prefetching to hide memory latency
    - Online softmax for O(1) memory per attention head
    - Support for grouped-query attention (GQA)
    - Sliding window attention support
    - Logits soft capping for numerical stability
    - Megacore parallelization across batch or KV heads

TPU Optimizations:
    - Double-buffered page prefetching using async DMA
    - VMEM buffers for K/V pages to minimize HBM access
    - Semaphore-based synchronization for async copies
    - Optimized block sizes for TPU's systolic arrays

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._pallas.tpu.page_attention import _pallas_impl_fwd
    >>> # Query: [batch_size, num_q_heads, head_dim]
    >>> query = jnp.ones((4, 32, 128))
    >>> # KV cache: [num_kv_heads, total_pages, page_size, head_dim]
    >>> key_cache = jnp.ones((8, 1024, 16, 128))
    >>> value_cache = jnp.ones((8, 1024, 16, 128))
    >>> context_lens = jnp.array([100, 200, 150, 180])
    >>> block_tables = jnp.zeros((4, 64), dtype=jnp.int32)
    >>> output = _pallas_impl_fwd.ref_paged_attention(
    ...     query, key_cache, value_cache, context_lens, block_tables
    ... )
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


def ref_paged_attention(
    query: jax.Array,
    key_cache: jax.Array,
    value_cache: jax.Array,
    context_lens: jax.Array,
    block_tables: jax.Array,
    *,
    mask_value: float = DEFAULT_MASK_VALUE,
    attn_logits_soft_cap: float | None = None,
    sliding_window: int | None = None,
) -> jax.Array:
    """Reference implementation of paged attention for testing.

    Args:
        query: A [batch_size, num_q_heads, head_dim] jax.Array.
        key_cache: A [num_kv_heads, total_num_pages, page_size, head_dim] jax.Array.
        value_cache: A [num_kv_heads, total_num_pages, page_size, head_dim] jax.Array.
        context_lens: A i32[batch_size] jax.Array the length of each example.
        block_tables: A i32[batch_size, pages_per_sequence] jax.Array.
        mask_value: The value used for padding in attention.
        attn_logits_soft_cap: The value used for soft capping the attention logits.
        sliding_window: If set, only attend to the last `sliding_window` tokens.

    Returns:
        The output of attention([batch_size, num_q_heads, head_dim]).
    """
    batch_size, num_q_heads, head_dim = query.shape
    num_kv_heads, _total_num_pages, page_size, _ = key_cache.shape
    block_tables.shape[1]
    num_groups = num_q_heads // num_kv_heads

    outputs = []
    for b in range(batch_size):
        length = int(context_lens[b])
        num_pages = (length + page_size - 1) // page_size

        page_indices = block_tables[b, :num_pages]
        k_pages = key_cache[:, page_indices]
        v_pages = value_cache[:, page_indices]

        k = k_pages.reshape(num_kv_heads, -1, head_dim)[:, :length, :]
        v = v_pages.reshape(num_kv_heads, -1, head_dim)[:, :length, :]

        k = jnp.repeat(k, num_groups, axis=0)
        v = jnp.repeat(v, num_groups, axis=0)

        q = query[b]
        qk = jnp.einsum("hd,htd->ht", q, k)

        if attn_logits_soft_cap is not None:
            qk = attn_logits_soft_cap * jnp.tanh(qk / attn_logits_soft_cap)

        if sliding_window is not None:
            kv_positions = jnp.arange(length)
            sliding_mask = kv_positions < (length - sliding_window)
            qk = qk + jnp.where(sliding_mask, mask_value, 0.0)

        attn = jax.nn.softmax(qk, axis=-1)

        out = jnp.einsum("ht,htd->hd", attn, v)
        outputs.append(out)

    return jnp.stack(outputs, axis=0)


class MultiPageAsyncCopyDescriptor:
    """Descriptor for async DMA copies of one compute block of K/V pages from HBM to VMEM.

    Manages ``num_pages_to_load`` individual ``pltpu.AsyncCopy`` objects, one
    per KV page in the compute block.  Pages are looked up via
    ``block_tables[page_indices_start_offset + i]`` and written into
    consecutive slots of ``vmem_buffer``.

    Attributes:
        _vmem_buffer: VMEM destination buffer for the loaded pages.
        _num_pages_to_load: Number of pages in this compute block.
        _pages_hbm_ref: Sliced HBM reference (indexed by ``head_index`` if
            provided, or the full reference otherwise).
        _sem: Pallas semaphore used to synchronise this set of copies.
        _page_indices: The full block-tables array passed as a Pallas ref.
        _page_indices_start_offset: Offset into ``_page_indices`` where this
            compute block's page indices begin.
        _async_copies: Pre-built list of ``AsyncCopy`` descriptors.
    """

    def __init__(
        self,
        pages_hbm_ref,
        vmem_buffer,
        sem,
        block_tables,
        page_indices_start_offset,
        num_pages_to_load,
        head_index,
    ):
        self._vmem_buffer = vmem_buffer
        self._num_pages_to_load = num_pages_to_load
        if head_index is not None:
            self._pages_hbm_ref = pages_hbm_ref.at[head_index]

        else:
            self._pages_hbm_ref = pages_hbm_ref
        self._sem = sem
        self._page_indices = block_tables
        self._page_indices_start_offset = page_indices_start_offset
        self._async_copies = [self._make_async_copy(i) for i in range(self._num_pages_to_load)]

    def _make_async_copy(self, i):
        """Build an ``AsyncCopy`` descriptor for the i-th page in this block."""
        page_index = self._page_indices[self._page_indices_start_offset + i]
        return pltpu.make_async_copy(self._pages_hbm_ref.at[page_index], self._vmem_buffer.at[i], self._sem)

    def start(self):
        """Starts the async copies."""
        for async_copy in self._async_copies:
            async_copy.start()

    def wait_and_get_loaded(self) -> jax.Array:
        """Wait async copies and gets the loaded buffer as a jax.Array."""
        for async_copy in self._async_copies:
            async_copy.wait()
        head_dim = self._vmem_buffer.shape[-1]
        jax_array = self._vmem_buffer[...].astype(jnp.float32)
        return jax_array.reshape(-1, head_dim)


def paged_flash_attention_kernel(
    lengths_ref,
    page_indices_ref,
    buffer_index_ref,
    init_flag_ref,
    q_ref,
    k_pages_hbm_ref,
    v_pages_hbm_ref,
    o_ref,
    m_ref,
    l_ref,
    k_vmem_buffer,
    v_vmem_buffer,
    k_sems,
    v_sems,
    *,
    batch_size: int,
    pages_per_compute_block: int,
    pages_per_sequence: int,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    megacore_mode: str | None,
    sliding_window: int | None = None,
    program_ids=(),
):
    """Pallas kernel body for one tile of paged flash attention.

    Processes a single ``(batch, kv_head, compute_block)`` tile of the paged
    KV cache.  Async DMA prefetching hides HBM latency by double-buffering:
    while the current block's attention scores are computed the next block's
    pages are already being fetched.

    Online softmax is maintained via running ``m`` (row-max) and ``l``
    (sum-of-exponentials) statistics stored in ``m_ref`` and ``l_ref`` VMEM
    scratch buffers, updating them in-place across calls.  The output
    accumulator ``o_ref`` is also updated in-place.

    Args:
        lengths_ref: SMEM ref ``[batch_size]`` int32 context lengths.
        page_indices_ref: SMEM ref ``[batch_size * pages_per_sequence]`` int32
            page table.
        buffer_index_ref: SMEM ref ``[1]`` holding the current double-buffer
            index (0 or 1).
        init_flag_ref: SMEM ref ``[1]``; non-zero on the very first tile to
            trigger the initial prefetch.
        q_ref: VMEM ref ``[num_groups, head_dim]`` query block.
        k_pages_hbm_ref: HBM ref
            ``[num_kv_heads, total_pages, page_size, head_dim]`` key cache.
        v_pages_hbm_ref: HBM ref
            ``[num_kv_heads, total_pages, page_size, head_dim]`` value cache.
        o_ref: VMEM output ref ``[num_groups, head_dim]`` (updated in-place).
        m_ref: VMEM running row-max ref ``[num_groups, 1]`` (updated in-place).
        l_ref: VMEM running sum-of-exp ref ``[num_groups, 1]``
            (updated in-place).
        k_vmem_buffer: Double-buffered VMEM scratch
            ``[2, pages_per_compute_block, page_size, head_dim]`` for K pages.
        v_vmem_buffer: Double-buffered VMEM scratch
            ``[2, pages_per_compute_block, page_size, head_dim]`` for V pages.
        k_sems: Pallas semaphores ``[2]`` for K DMA synchronisation.
        v_sems: Pallas semaphores ``[2]`` for V DMA synchronisation.
        batch_size: Total number of sequences.
        pages_per_compute_block: Number of KV pages processed per kernel tile.
        pages_per_sequence: Maximum number of pages allocated per sequence.
        mask_value: Fill value for out-of-range attention logit positions.
        attn_logits_soft_cap: If not ``None``, apply ``soft_cap * tanh(x /
            soft_cap)`` to logits before masking.
        megacore_mode: One of ``"batch"``, ``"kv_head"``, or ``None``.  When
            set, multiple TPU cores share the work: ``"batch"`` partitions
            across the batch dimension and ``"kv_head"`` across KV heads.
        sliding_window: If not ``None``, only attend within the last
            ``sliding_window`` tokens.
        program_ids: Optional 4-tuple ``(core, b, h, i)`` used when called
            from ``paged_flash_attention_kernel_inline_seq_dim`` to supply
            custom program IDs instead of reading them from ``pl.program_id``.
    """
    if program_ids:
        core_index, b, h, i = program_ids
    else:
        core_index, b, h, i = (
            pl.program_id(0),
            pl.program_id(1),
            pl.program_id(2),
            pl.program_id(3),
        )
    num_kv_heads, _, page_size, _ = k_pages_hbm_ref.shape
    bk = page_size * pages_per_compute_block
    num_cores = pl.num_programs(0)

    b_step = num_cores if megacore_mode == "batch" else 1
    b_start = core_index if megacore_mode == "batch" else 0
    h_step = num_cores if megacore_mode == "kv_head" else 1
    h_start = core_index if megacore_mode == "kv_head" else 0

    h = h * h_step + h_start
    b = b * b_step + b_start
    length = lengths_ref[b]

    def compute_block_indices(b, h, i):
        def advance_b():
            next_b = b + b_step

            def advance_to_next_non_zero_length():
                next_next_b = next_b + b_step
                return lax.fori_loop(
                    lax.div(next_next_b, b_step),
                    lax.div(batch_size, b_step),
                    lambda _, b: jnp.where(lengths_ref[b] == 0, b + b_step, b),
                    next_next_b,
                )

            return (
                lax.cond(
                    jnp.logical_and(next_b < batch_size, lengths_ref[lax.clamp(0, next_b, batch_size - 1)] == 0),
                    advance_to_next_non_zero_length,
                    lambda: next_b,
                ),
                h_start,
                0,
            )

        def advance_h():
            next_h = h + h_step
            return lax.cond(next_h < num_kv_heads, lambda: (b, next_h, 0), advance_b)

        return lax.cond(i * bk < lengths_ref[b], lambda: (b, h, i), advance_h)

    def create_kv_async_copy_descriptors(b, h, i, buffer_index):
        page_offset = b * pages_per_sequence + i * pages_per_compute_block
        pages_to_load = pages_per_compute_block
        async_copy_k = MultiPageAsyncCopyDescriptor(
            k_pages_hbm_ref,
            k_vmem_buffer.at[buffer_index],
            k_sems.at[buffer_index],
            page_indices_ref,
            page_offset,
            pages_to_load,
            h,
        )
        async_copy_v = MultiPageAsyncCopyDescriptor(
            v_pages_hbm_ref,
            v_vmem_buffer.at[buffer_index],
            v_sems.at[buffer_index],
            page_indices_ref,
            page_offset,
            pages_to_load,
            h,
        )
        return async_copy_k, async_copy_v

    @pl.when(i * bk < length)
    def flash_attention():
        init_flag = init_flag_ref[0]
        init_flag_ref[0] = 0
        buffer_index = buffer_index_ref[0]
        next_b, next_h, next_i = compute_block_indices(b, h, i + 1)

        @pl.when(init_flag)
        def prefetch_first_block():
            async_copy_k, async_copy_v = create_kv_async_copy_descriptors(b, h, i, buffer_index)
            async_copy_k.start()
            async_copy_v.start()

        @pl.when(i == 0)
        def init():
            m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
            l_ref[...] = jnp.zeros_like(l_ref)
            o_ref[...] = jnp.zeros_like(o_ref)

        @pl.when(next_b < batch_size)
        def prefetch_next_block():
            next_buffer_index = jnp.where(buffer_index == 0, 1, 0)
            async_copy_next_k, async_copy_next_v = create_kv_async_copy_descriptors(
                next_b, next_h, next_i, next_buffer_index
            )
            async_copy_next_k.start()
            async_copy_next_v.start()
            buffer_index_ref[0] = next_buffer_index

        async_copy_k, async_copy_v = create_kv_async_copy_descriptors(b, h, i, buffer_index)
        q = q_ref[...].astype(jnp.float32)
        k = async_copy_k.wait_and_get_loaded()
        qk = jnp.einsum("gd,td->gt", q, k, preferred_element_type=jnp.float32)
        if attn_logits_soft_cap is not None:
            capped_qk = jnp.tanh(qk / attn_logits_soft_cap)
            qk = capped_qk * attn_logits_soft_cap

        kv_positions = i * bk + jax.lax.broadcasted_iota(jnp.int32, qk.shape, 1)
        mask = kv_positions < length

        if sliding_window is not None:
            sliding_mask = kv_positions >= (length - sliding_window)
            mask = jnp.logical_and(mask, sliding_mask)

        qk = qk + jnp.where(mask, 0.0, mask_value)
        m_curr = qk.max(axis=-1)

        s_curr = jnp.exp(qk - m_curr[..., None])
        m_prev, l_prev = m_ref[...], l_ref[...]
        l_curr = jax.lax.broadcast_in_dim(s_curr.sum(axis=-1), l_prev.shape, (0,))
        m_curr = jax.lax.broadcast_in_dim(m_curr, m_prev.shape, (0,))
        m_next = jnp.maximum(m_prev, m_curr)
        alpha = jnp.exp(m_prev - m_next)
        beta = jnp.exp(m_curr - m_next)
        l_next = alpha * l_prev + beta * l_curr
        m_ref[...], l_ref[...] = m_next, l_next

        v = async_copy_v.wait_and_get_loaded()
        o_curr = jnp.einsum("gt,td->gd", s_curr, v)

        o_ref[...] = ((l_prev * alpha * o_ref[...] + beta * o_curr) / l_next).astype(o_ref.dtype)


def paged_flash_attention_kernel_inline_seq_dim(
    lengths_ref,
    page_indices_ref,
    buffer_index_ref,
    init_flag_ref,
    q_ref,
    k_pages_hbm_ref,
    v_pages_hbm_ref,
    o_ref,
    m_ref,
    l_ref,
    k_vmem_buffer,
    v_vmem_buffer,
    k_sems,
    v_sems,
    *,
    batch_size: int,
    pages_per_compute_block: int,
    pages_per_sequence: int,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    megacore_mode: str | None,
    sliding_window: int | None = None,
):
    """Variant of ``paged_flash_attention_kernel`` that loops over the sequence dimension inline.

    Rather than using the KV sequence index as a Pallas grid dimension, this
    kernel uses a 3-D grid ``(core, batch, kv_head)`` and runs a
    ``lax.fori_loop`` over the sequence tiles.  This avoids the grid overhead
    of having a fourth grid dimension and can be more efficient when the
    number of sequence tiles is small.

    The kernel resets ``m_ref``, ``l_ref``, and ``o_ref`` to initial values
    before the loop begins.  All other semantics (online softmax, double-
    buffered prefetch, megacore modes, sliding window) are inherited from
    ``paged_flash_attention_kernel``, which is called on each loop iteration
    with explicit ``program_ids``.

    Args:
        lengths_ref: SMEM ref ``[batch_size]`` int32 context lengths.
        page_indices_ref: SMEM ref ``[batch_size * pages_per_sequence]`` int32
            page table.
        buffer_index_ref: SMEM ref ``[1]`` double-buffer index.
        init_flag_ref: SMEM ref ``[1]`` initial-prefetch trigger.
        q_ref: VMEM ref ``[num_groups, head_dim]`` query block.
        k_pages_hbm_ref: HBM key cache
            ``[num_kv_heads, total_pages, page_size, head_dim]``.
        v_pages_hbm_ref: HBM value cache
            ``[num_kv_heads, total_pages, page_size, head_dim]``.
        o_ref: VMEM output ref ``[num_groups, head_dim]``.
        m_ref: VMEM running row-max scratch ``[num_groups, 1]``.
        l_ref: VMEM running sum-of-exp scratch ``[num_groups, 1]``.
        k_vmem_buffer: Double-buffered VMEM K-page scratch.
        v_vmem_buffer: Double-buffered VMEM V-page scratch.
        k_sems: DMA semaphores ``[2]`` for K pages.
        v_sems: DMA semaphores ``[2]`` for V pages.
        batch_size: Total number of sequences.
        pages_per_compute_block: Number of KV pages per kernel tile.
        pages_per_sequence: Maximum pages per sequence.
        mask_value: Fill value for out-of-range logit positions.
        attn_logits_soft_cap: Optional soft cap for attention logits.
        megacore_mode: ``"batch"``, ``"kv_head"``, or ``None``.
        sliding_window: Optional sliding-window attention size.
    """
    core_index, b, h = pl.program_id(0), pl.program_id(1), pl.program_id(2)

    m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
    l_ref[...] = jnp.zeros_like(l_ref)
    o_ref[...] = jnp.zeros_like(o_ref)

    def body(i, _):
        paged_flash_attention_kernel(
            lengths_ref,
            page_indices_ref,
            buffer_index_ref,
            init_flag_ref,
            q_ref,
            k_pages_hbm_ref,
            v_pages_hbm_ref,
            o_ref,
            m_ref,
            l_ref,
            k_vmem_buffer,
            v_vmem_buffer,
            k_sems,
            v_sems,
            batch_size=batch_size,
            pages_per_compute_block=pages_per_compute_block,
            pages_per_sequence=pages_per_sequence,
            mask_value=mask_value,
            attn_logits_soft_cap=attn_logits_soft_cap,
            megacore_mode=megacore_mode,
            sliding_window=sliding_window,
            program_ids=(core_index, b, h, i),
        )
        return ()

    bk = pages_per_compute_block * k_pages_hbm_ref.shape[-2]

    if megacore_mode == "batch":
        num_cores = pl.num_programs(0)
        length = lengths_ref[b * num_cores + core_index]
    else:
        length = lengths_ref[b]

    lax.fori_loop(0, lax.div(length + bk - 1, bk), body, ())
