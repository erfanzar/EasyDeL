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

# Reference: JetStream chunked prefill attention
# https://github.com/AI-Hypercomputer/JetStream/blob/main/experimental/jax/inference/kernel/attention/tpu/chunked_prefill_attention.py

"""Chunked Prefill Paged Attention TPU kernel implementation.

This module provides an optimized chunked prefill attention implementation
for Google TPUs, designed for processing the last chunk of tokens during
the prefill phase while using a paged KV cache.

Unlike standard paged attention (which handles single-token decode), chunked
prefill processes multiple query tokens simultaneously, enabling efficient
batch processing during the initial prompt encoding phase.

Algorithm Overview:
    The chunked prefill algorithm processes a chunk of query tokens:
    1. Query positions span [context_len - chunk_size, context_len - 1]
    2. Each query can attend to all KV positions up to its own position (causal)
    3. KV cache is loaded page-by-page with async prefetching
    4. Online softmax accumulates attention across KV chunks
    5. Final output is rescaled by accumulated softmax denominators

Key Features:
    - Processes multiple query tokens in a single kernel call
    - Causal masking where each query attends to current and past positions
    - Paged KV cache with efficient page-based memory access
    - Sliding window attention support for local attention patterns
    - Async page prefetching using double-buffering
    - Grouped-query attention (GQA) support

TPU Optimizations:
    - Double-buffered async DMA for K/V page loading
    - VMEM scratch for softmax accumulators (m, l values)
    - Semaphore-based synchronization for async operations
    - Efficient handling of partial KV chunks at sequence boundaries

Example:
    >>> import jax.numpy as jnp
    >>> from ejkernel.kernels._pallas.tpu.prefill_page_attention import _pallas_impl_fwd
    >>> # Query chunk: [chunk_size, num_q_heads, head_dim]
    >>> query = jnp.ones((128, 32, 128))
    >>> # KV cache: [num_kv_heads, total_pages, page_size, head_dim]
    >>> key_cache = jnp.ones((8, 1024, 16, 128))
    >>> value_cache = jnp.ones((8, 1024, 16, 128))
    >>> context_len = jnp.array([256])
    >>> page_indices = jnp.zeros((16,), dtype=jnp.int32)
    >>> output = _pallas_impl_fwd.ref_prefill_page_attention(
    ...     query, key_cache, value_cache, context_len, page_indices
    ... )
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


def ref_prefill_page_attention(
    query: jax.Array,
    key_cache: jax.Array,
    value_cache: jax.Array,
    context_len: jax.Array,
    page_indices: jax.Array,
    *,
    softmax_scale: float | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    attn_logits_soft_cap: float | None = None,
    sliding_window: int | None = None,
) -> jax.Array:
    """Reference implementation of chunked prefill paged attention for testing.

    This processes a single sequence's prefill with paged KV cache.

    Args:
        query: A [chunk_size, num_q_heads, head_dim] jax.Array.
        key_cache: A [num_kv_heads, total_num_pages, page_size, head_dim] jax.Array.
        value_cache: A [num_kv_heads, total_num_pages, page_size, head_dim] jax.Array.
        context_len: Scalar or [1] jax.Array - the total context length (including this chunk).
        page_indices: A [num_pages_needed] jax.Array of page indices for this sequence.
        softmax_scale: Attention scaling factor (default: 1/sqrt(head_dim)).
        mask_value: The value used for padding/causal masking in attention.
        attn_logits_soft_cap: The value used for soft capping the attention logits.
        sliding_window: If set, only attend to the last `sliding_window` tokens.

    Returns:
        The output of attention [chunk_size, num_q_heads, head_dim].
    """
    chunk_size, num_q_heads, head_dim = query.shape
    num_kv_heads, _total_num_pages, page_size, _ = key_cache.shape
    num_groups = num_q_heads // num_kv_heads

    if softmax_scale is None:
        softmax_scale = 1.0 / jnp.sqrt(head_dim).astype(query.dtype)

    length = int(context_len.reshape(-1)[0]) if hasattr(context_len, "reshape") else int(context_len)

    num_pages = (length + page_size - 1) // page_size

    k_pages = key_cache[:, page_indices[:num_pages]]
    v_pages = value_cache[:, page_indices[:num_pages]]

    k = k_pages.reshape(num_kv_heads, -1, head_dim)[:, :length, :]
    v = v_pages.reshape(num_kv_heads, -1, head_dim)[:, :length, :]

    k = jnp.repeat(k, num_groups, axis=0)
    v = jnp.repeat(v, num_groups, axis=0)

    qk = jnp.einsum("qhd,hsd->hqs", query, k, preferred_element_type=jnp.float32)
    qk = qk * softmax_scale

    if attn_logits_soft_cap is not None:
        qk = attn_logits_soft_cap * jnp.tanh(qk / attn_logits_soft_cap)

    q_positions = (length - chunk_size) + jnp.arange(chunk_size)
    kv_positions = jnp.arange(length)

    causal_mask = q_positions[:, None] >= kv_positions[None, :]

    if sliding_window is not None:
        sliding_mask = kv_positions[None, :] >= (q_positions[:, None] - sliding_window + 1)
        causal_mask = jnp.logical_and(causal_mask, sliding_mask)

    causal_mask = jnp.broadcast_to(causal_mask, (num_q_heads, chunk_size, length))

    qk = qk + jnp.where(causal_mask, 0.0, mask_value)

    attn = jax.nn.softmax(qk, axis=-1)

    out = jnp.einsum("hqs,hsd->qhd", attn, v)

    return out.astype(query.dtype)


class MultiPageAsyncCopyDescriptor:
    """Descriptor for asynchronous DMA copy of multiple KV cache pages from HBM to VMEM.

    Manages the async transfer of multiple contiguous pages from the paged KV cache
    stored in HBM into a VMEM buffer on TPU. Uses DMA semaphores for synchronization
    between the copy engine and compute engine.

    This enables double-buffered prefetching where the next chunk's pages are loaded
    while the current chunk is being processed, hiding memory latency.

    Attributes:
        _vmem_buffer: VMEM buffer reference to copy pages into.
        _num_pages_to_load: Number of pages to transfer in this operation.
        _pages_hbm_ref: HBM reference to the source pages (possibly indexed by head).
        _sem: DMA semaphore for synchronization.
        _page_indices: Array mapping logical page positions to physical page indices.
        _page_offset: Starting offset into page_indices for this chunk.
        _async_copies: List of individual async copy operations, one per page.
    """

    def __init__(
        self,
        pages_hbm_ref,
        vmem_buffer,
        sem,
        page_indices,
        page_offset,
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
        self._page_indices = page_indices
        self._page_offset = page_offset
        self._async_copies = [self._make_async_copy(i) for i in range(self._num_pages_to_load)]

    def _make_async_copy(self, i):
        """Create an async DMA copy operation for a single page.

        Args:
            i: Local page index within this chunk (0 to num_pages_to_load - 1).

        Returns:
            A Pallas async copy descriptor for the i-th page transfer.
        """
        page_index = self._page_indices[self._page_offset + i]
        return pltpu.make_async_copy(self._pages_hbm_ref.at[page_index], self._vmem_buffer.at[i], self._sem)

    def start(self):
        """Initiate all asynchronous DMA copy operations.

        Starts the transfer of all pages from HBM to VMEM. The transfers
        proceed asynchronously and must be completed by calling
        ``wait_and_get_loaded`` before accessing the data.
        """
        for async_copy in self._async_copies:
            async_copy.start()

    def wait_and_get_loaded(self) -> jax.Array:
        """Wait for all async copies to complete and return the loaded data.

        Blocks until all page transfers are finished, then reshapes the
        loaded pages into a 2D matrix suitable for attention computation.

        Returns:
            A float32 array of shape [num_pages * page_size, head_dim]
            containing the loaded and flattened page data.
        """
        for async_copy in self._async_copies:
            async_copy.wait()
        head_dim = self._vmem_buffer.shape[-1]
        jax_array = self._vmem_buffer[...].astype(jnp.float32)
        return jax_array.reshape(-1, head_dim)


def chunked_prefill_attention_kernel(
    length_ref,
    page_indices_ref,
    buffer_index_ref,
    q_ref,
    k_pages_hbm_ref,
    v_pages_hbm_ref,
    out_ref,
    l_ref,
    m_ref,
    k_vmem_buffer,
    v_vmem_buffer,
    sem,
    *,
    chunk_size: int,
    page_size: int,
    num_kv_chunks: int,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    sliding_window: int | None,
):
    """Pallas kernel for chunked prefill attention with paged KV cache.

    This kernel processes the last ``chunk_size`` tokens of a sequence during the
    prefill phase. It implements online softmax with flash-attention-style
    accumulation across KV chunks loaded from a paged cache.

    The query positions span [length - chunk_size, ..., length - 1]. Each query
    position can attend to all KV positions from 0 up to its own position (causal).
    KV pages are loaded asynchronously using double-buffered DMA to hide memory
    latency.

    The kernel iterates over KV chunks, and within each chunk iterates over
    GQA groups. For each group, it computes attention scores, applies causal
    and optional sliding window masks, and updates the running softmax
    statistics (m for max, l for sum of exponentials).

    Args:
        length_ref: Scalar prefetch ref containing the total context length.
        page_indices_ref: Scalar prefetch ref containing page index mapping.
        buffer_index_ref: Scalar prefetch ref for double-buffer index tracking.
        q_ref: Query tensor ref of shape [1, group_size, chunk_size, head_dim].
        k_pages_hbm_ref: Key cache HBM ref [num_kv_heads, total_pages, page_size, head_dim].
        v_pages_hbm_ref: Value cache HBM ref [num_kv_heads, total_pages, page_size, head_dim].
        out_ref: Output ref of shape [1, group_size, chunk_size, head_dim].
        l_ref: Softmax denominator accumulator ref [1, group_size, chunk_size, 1].
        m_ref: Softmax max accumulator ref [1, group_size, chunk_size, 1].
        k_vmem_buffer: VMEM scratch for double-buffered key page loading.
        v_vmem_buffer: VMEM scratch for double-buffered value page loading.
        sem: DMA semaphore for async copy synchronization.
        chunk_size: Number of query tokens in this prefill chunk.
        page_size: Number of tokens stored per cache page.
        num_kv_chunks: Total number of KV chunks to iterate over.
        mask_value: Large negative value for masked attention positions.
        attn_logits_soft_cap: Optional soft cap for attention logits stability.
        sliding_window: Optional window size for local attention.
    """
    h = pl.program_id(0)
    head_dim = k_pages_hbm_ref.shape[3]
    group_size = q_ref.shape[1]
    length = length_ref[0]

    m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
    l_ref[...] = jnp.zeros_like(l_ref)
    out_ref[...] = jnp.zeros_like(out_ref)

    pages_per_chunk = chunk_size // page_size

    q_start_pos = length - chunk_size

    def create_kv_async_copy_descriptors(i, buffer_index):
        """Create async copy descriptors for KV chunk i."""
        page_offset = i * pages_per_chunk
        async_copy_k = MultiPageAsyncCopyDescriptor(
            k_pages_hbm_ref,
            k_vmem_buffer.at[buffer_index],
            sem,
            page_indices_ref,
            page_offset,
            pages_per_chunk,
            head_index=h,
        )
        async_copy_v = MultiPageAsyncCopyDescriptor(
            v_pages_hbm_ref,
            v_vmem_buffer.at[buffer_index],
            sem,
            page_indices_ref,
            page_offset,
            pages_per_chunk,
            head_index=h,
        )
        return async_copy_k, async_copy_v

    def per_kv_chunk_body(i, _):
        """Process KV chunk i."""

        @pl.when((i * chunk_size) < length)
        def body():
            buffer_index = buffer_index_ref[0]

            @pl.when((i == 0) & (h == 0))
            def prefetch_first_kv():
                async_copy_k, async_copy_v = create_kv_async_copy_descriptors(0, buffer_index)
                async_copy_k.start()
                async_copy_v.start()

            next_i = i + 1

            @pl.when((next_i < num_kv_chunks) & (next_i * chunk_size < length))
            def prefetch_next_block():
                next_buffer_index = jnp.where(buffer_index == 0, 1, 0)
                async_copy_next_k, async_copy_next_v = create_kv_async_copy_descriptors(next_i, next_buffer_index)
                async_copy_next_k.start()
                async_copy_next_v.start()
                buffer_index_ref[0] = next_buffer_index

            async_copy_k, async_copy_v = create_kv_async_copy_descriptors(i, buffer_index)
            k = async_copy_k.wait_and_get_loaded()
            v = async_copy_v.wait_and_get_loaded()

            mask_shape = (chunk_size, chunk_size)
            q_positions = q_start_pos + lax.broadcasted_iota(jnp.int32, mask_shape, 0)
            kv_positions = i * chunk_size + lax.broadcasted_iota(jnp.int32, mask_shape, 1)

            causal_mask = q_positions >= kv_positions

            valid_kv_mask = kv_positions < length
            causal_mask = jnp.logical_and(causal_mask, valid_kv_mask)

            if sliding_window is not None:
                sliding_mask = kv_positions >= (q_positions - sliding_window + 1)
                causal_mask = jnp.logical_and(causal_mask, sliding_mask)

            causal_mask_value = jnp.where(causal_mask, 0.0, mask_value)

            def per_group_body(group_idx, _):
                q = q_ref[0, group_idx]
                s = jnp.einsum("td,sd->ts", q, k, preferred_element_type=jnp.float32) + causal_mask_value

                if attn_logits_soft_cap is not None:
                    s = attn_logits_soft_cap * jnp.tanh(s / attn_logits_soft_cap)

                s_max = jnp.max(s, axis=1, keepdims=True)

                prev_m = m_ref[0, group_idx]
                prev_l = l_ref[0, group_idx]

                cur_m = jnp.maximum(prev_m, s_max)
                cur_m_to_attn_size = lax.broadcast_in_dim(cur_m, (chunk_size, chunk_size), (0, 1))

                p = jnp.exp(s - cur_m_to_attn_size)

                cur_l = jnp.exp(prev_m - cur_m) * prev_l + jnp.sum(p, axis=1, keepdims=True)

                out = out_ref[0, group_idx]

                out_ref[0, group_idx, :, :] = (
                    out * lax.broadcast_in_dim(jnp.exp(prev_m - cur_m), (chunk_size, head_dim), (0, 1)) + p @ v
                ).astype(out_ref.dtype)

                m_ref[0, group_idx, :, :] = cur_m
                l_ref[0, group_idx, :, :] = cur_l
                return ()

            lax.fori_loop(0, group_size, per_group_body, ())

        return ()

    lax.fori_loop(0, num_kv_chunks, per_kv_chunk_body, ())

    l = lax.broadcast_in_dim(l_ref[...], (1, group_size, chunk_size, head_dim), (0, 1, 2, 3))
    out_ref[...] = (out_ref[...] / l).astype(out_ref.dtype)
