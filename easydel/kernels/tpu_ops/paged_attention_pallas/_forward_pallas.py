# Copyright 2024 The JAX Authors.
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

# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

# This is a copied version of
# https://github.com/jax-ml/jax/blob/main/jax/experimental/pallas/ops/tpu/paged_attention


import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


class MultiPageAsyncCopyDescriptor:
    """Manages asynchronous copies of multiple K/V pages from HBM to VMEM.

    This class simplifies the process of initiating and waiting for multiple
    asynchronous DMA transfers (copies) for pages belonging to the Key or Value
    cache. It takes a list of page indices and orchestrates the copies into a
    specified VMEM buffer.

    Attributes:
      _vmem_buffer: The destination VMEM buffer slice for the copies.
      _num_pages_to_load: The number of pages to copy.
      _pages_hbm_ref: A Pallas reference to the K or V page cache in HBM.
      _sem: The semaphore used to coordinate the asynchronous copies.
      _page_indices: A Pallas reference to the array containing page indices.
      _page_indices_start_offset: The starting offset within `_page_indices`
        for the current set of pages.
      _async_copies: A list of `AsyncCopy` objects, one for each page.
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
        """Initializes the MultiPageAsyncCopyDescriptor.

        Args:
          pages_hbm_ref: Pallas Ref to the source K/V pages in HBM.
          vmem_buffer: Pallas Ref to the destination buffer in VMEM.
          sem: Pallas Ref for the semaphore to use for synchronization.
          block_tables: Pallas Ref to the array holding the indices of the pages
            to be loaded from HBM.
          page_indices_start_offset: Starting offset in `block_tables` array.
          num_pages_to_load: The number of pages to copy.
          head_index: The specific head index to load pages for, if the
            `pages_hbm_ref` has a head dimension. If None, assumes no head dim.
        """
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
        """Creates a single asynchronous copy operation for the i-th page."""
        page_index = self._page_indices[self._page_indices_start_offset + i]
        return pltpu.make_async_copy(
            self._pages_hbm_ref.at[page_index],
            self._vmem_buffer.at[i],
            self._sem,
        )

    def start(self):
        """Starts all the configured asynchronous copy operations."""
        for async_copy in self._async_copies:
            async_copy.start()

    def wait_and_get_loaded(self) -> jax.Array:
        """Waits for all copies to complete and returns the loaded data.

        Returns:
          A jax.Array containing the data loaded into the VMEM buffer, reshaped
          to combine the pages along the sequence dimension. The shape will be
          (num_pages_to_load * page_size, head_dim).
        """

        for async_copy in self._async_copies:
            async_copy.wait()
        head_dim = self._vmem_buffer.shape[-1]
        jax_array = self._vmem_buffer[...].astype(jnp.float32)
        return jax_array.reshape(-1, head_dim)


def paged_flash_attention_kernel(
    lengths_ref,
    page_indices_ref,
    buffer_index_ref,
    step_ref,
    q_ref,
    k_pages_hbm_ref,
    v_pages_hbm_ref,
    o_ref,
    m_ref,
    l_ref,
    k_vmem_buffer,
    v_vmem_buffer,
    sem,
    *,
    batch_size: int,
    pages_per_compute_block: int,
    pages_per_sequence: int,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    megacore_mode: str | None,
    program_ids=(),
):
    """Pallas kernel for paged attention, likely for the decode phase.

    This kernel computes attention for a single query token against paged
    Key-Value caches stored in HBM. It processes the KV cache in blocks of pages,
    using double buffering for asynchronous data loading and FlashAttention-style
    online softmax calculation.

    The kernel grid is expected to be (num_cores, batch_size // b_step,
    num_kv_heads // h_step, num_blocks_per_sequence). `megacore_mode` determines
    how work is distributed across cores (by batch or by KV head).

    Args:
      lengths_ref: SMEM Ref to sequence lengths for each batch item.
      page_indices_ref: Ref to page indices mapping sequence positions to HBM pages.
      buffer_index_ref: SMEM Ref storing the current VMEM buffer index (0 or 1)
        for double buffering.
      step_ref: SMEM Ref storing the current step/block index being processed.
      q_ref: VMEM Ref to the query vector(s) for the current token.
      k_pages_hbm_ref: HBM Ref to the Key cache pages.
      v_pages_hbm_ref: HBM Ref to the Value cache pages.
      o_ref: VMEM Ref to store the computed output attention vector(s).
      m_ref: VMEM Ref to store the running maximum logit (part of online softmax).
      l_ref: VMEM Ref to store the running sum of exp(logit - max_logit)
        (part of online softmax).
      k_vmem_buffer: VMEM Ref for the double buffer used to load Key pages.
      v_vmem_buffer: VMEM Ref for the double buffer used to load Value pages.
      sem: Pallas Ref for the semaphore used for async copy synchronization.
      batch_size: Total batch size.
      pages_per_compute_block: Number of KV cache pages processed per iteration.
      pages_per_sequence: Maximum number of pages allocated per sequence.
      mask_value: Value to use for masking attention logits (e.g., -inf).
      attn_logits_soft_cap: If not None, apply tanh capping to logits.
      megacore_mode: How to distribute work across TPU cores ('batch' or 'kv_head').
      program_ids: Optional tuple to directly provide program IDs, used when
        this kernel is called from another kernel (like the inline version).
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
                    jnp.logical_and(next_b < batch_size, lengths_ref[next_b] == 0),
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
            sem,
            page_indices_ref,
            page_offset,
            pages_to_load,
            h,
        )
        async_copy_v = MultiPageAsyncCopyDescriptor(
            v_pages_hbm_ref,
            v_vmem_buffer.at[buffer_index],
            sem,
            page_indices_ref,
            page_offset,
            pages_to_load,
            h,
        )
        return async_copy_k, async_copy_v

    @pl.when(i * bk < length)
    def flash_attention():  # pylint: disable=unused-variable
        step = step_ref[0]
        buffer_index = buffer_index_ref[0]

        @pl.when(i == 0)
        def init():  # pylint: disable=unused-variable
            m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
            l_ref[...] = jnp.zeros_like(l_ref)
            o_ref[...] = jnp.zeros_like(o_ref)

        @pl.when(step == 0)
        def prefetch_first_block():  # pylint: disable=unused-variable
            async_copy_k, async_copy_v = create_kv_async_copy_descriptors(b, h, i, buffer_index)
            async_copy_k.start()
            async_copy_v.start()

        next_b, next_h, next_i = compute_block_indices(b, h, i + 1)

        @pl.when(next_b < batch_size)
        def prefetch_next_block():  # pylint: disable=unused-variable
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
        qk = jnp.einsum("hd,td->ht", q, k, preferred_element_type=jnp.float32)
        if attn_logits_soft_cap is not None:
            capped_qk = jnp.tanh(qk / attn_logits_soft_cap)
            qk = capped_qk * attn_logits_soft_cap

        mask = i * bk + jax.lax.broadcasted_iota(jnp.int32, qk.shape, 1) < length
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
        l_next_safe = jnp.where(l_next == 0.0, 1.0, l_next)

        v = async_copy_v.wait_and_get_loaded()
        o_curr_times_l_curr = jnp.dot(s_curr, v)

        m_ref[...], l_ref[...] = m_next, l_next_safe
        o_ref[...] = ((l_prev * alpha * o_ref[...] + beta * o_curr_times_l_curr) / l_next_safe).astype(o_ref.dtype)

        step_ref[0] = step + 1


def paged_flash_attention_kernel_inline_seq_dim(
    lengths_ref,
    page_indices_ref,
    buffer_index_ref,
    step_ref,
    q_ref,
    k_pages_hbm_ref,
    v_pages_hbm_ref,
    o_ref,
    m_ref,
    l_ref,
    k_vmem_buffer,
    v_vmem_buffer,
    sem,
    *,
    batch_size: int,
    pages_per_compute_block: int,
    pages_per_sequence: int,
    mask_value: float,
    attn_logits_soft_cap: float | None,
    megacore_mode: str | None,
):
    """Pallas kernel for paged attention that loops over sequence blocks internally.

    This kernel performs the same computation as `paged_flash_attention_kernel`
    but iterates over the sequence blocks (`i`) using an internal `lax.fori_loop`
    instead of having `i` as a `program_id`. The grid for this kernel is
    typically (num_cores, batch_size // b_step, num_kv_heads // h_step).

    Args:
      lengths_ref: SMEM Ref to sequence lengths for each batch item.
      page_indices_ref: Ref to page indices mapping sequence positions to HBM pages.
      buffer_index_ref: SMEM Ref storing the current VMEM buffer index (0 or 1).
      step_ref: SMEM Ref storing the current step/block index being processed.
      q_ref: VMEM Ref to the query vector(s) for the current token.
      k_pages_hbm_ref: HBM Ref to the Key cache pages.
      v_pages_hbm_ref: HBM Ref to the Value cache pages.
      o_ref: VMEM Ref to store the computed output attention vector(s).
      m_ref: VMEM Ref to store the running maximum logit.
      l_ref: VMEM Ref to store the running sum of exp(logit - max_logit).
      k_vmem_buffer: VMEM Ref for the double buffer used to load Key pages.
      v_vmem_buffer: VMEM Ref for the double buffer used to load Value pages.
      sem: Pallas Ref for the semaphore used for async copy synchronization.
      batch_size: Total batch size.
      pages_per_compute_block: Number of KV cache pages processed per iteration.
      pages_per_sequence: Maximum number of pages allocated per sequence.
      mask_value: Value to use for masking attention logits.
      attn_logits_soft_cap: If not None, apply tanh capping to logits.
      megacore_mode: How to distribute work across TPU cores ('batch' or 'kv_head').
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
            step_ref,
            q_ref,
            k_pages_hbm_ref,
            v_pages_hbm_ref,
            o_ref,
            m_ref,
            l_ref,
            k_vmem_buffer,
            v_vmem_buffer,
            sem,
            batch_size=batch_size,
            pages_per_compute_block=pages_per_compute_block,
            pages_per_sequence=pages_per_sequence,
            mask_value=mask_value,
            attn_logits_soft_cap=attn_logits_soft_cap,
            megacore_mode=megacore_mode,
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


def prefill_attention_impl(
    length_ref,  # shape: (1,), smem,
    page_indices_ref,  # shape: (max_seq_len // page_size), smem,
    buffer_index_ref,  # shape: (1,), smem,
    q_ref,  # shape: (group_size, chunk, head_dim), vmem,
    k_pages_hbm_ref,  # shape: (num_kv_heads, num_pages, page_size, head_dim), hbm
    v_pages_hbm_ref,  # shape: (num_kv_heads, num_pages, page_size, head_dim), hbm
    out_ref,  # shape: (group_size, chunk, head_dim), vmem,
    l_ref,  # shape: (group_size, chunk, 1), vmem,
    m_ref,  # shape: (group_size, chunk, 1), vmem,
    k_vmem_buffer,  # shape: (2, page_per_chunk, page_size, head_dim), vmem,
    v_vmem_buffer,  # shape: (2, page_per_chunk, page_size, head_dim), vmem,
    sem,
):
    """Pallas kernel implementation for paged attention prefill phase.

    This kernel computes attention for a chunk of query tokens (part of the
    prompt) against the paged Key-Value cache built so far. It iterates through
    chunks of the KV cache, applying causal masking and using online softmax.
    Double buffering is used for loading KV cache chunks.

    The grid for this kernel is typically (num_kv_heads,). It processes one
    query chunk for all associated attention heads within a KV head group.

    Args:
      length_ref: SMEM Ref containing the total sequence length of the prompt.
      page_indices_ref: SMEM Ref containing the page indices for this sequence.
      buffer_index_ref: SMEM Ref storing the current VMEM buffer index (0 or 1).
      q_ref: VMEM Ref to the current chunk of query vectors.
      k_pages_hbm_ref: HBM Ref to the Key cache pages.
      v_pages_hbm_ref: HBM Ref to the Value cache pages.
      out_ref: VMEM Ref to store the computed output attention vectors for the chunk.
      l_ref: VMEM Ref to store the running sum part of online softmax.
      m_ref: VMEM Ref to store the running max logit part of online softmax.
      k_vmem_buffer: VMEM Ref for the double buffer used to load Key chunks.
      v_vmem_buffer: VMEM Ref for the double buffer used to load Value chunks.
      sem: Pallas Ref for the semaphore used for async copy synchronization.
    """
    h = pl.program_id(0)
    page_size = k_pages_hbm_ref.shape[2]
    head_dim = k_pages_hbm_ref.shape[3]
    group_size = q_ref.shape[0]
    num_kv_heads = k_pages_hbm_ref.shape[0]
    chunk_size = q_ref.shape[1]
    length = length_ref[0]
    q_chunk_idx = jax.lax.div(length, chunk_size)
    reminder = jax.lax.rem(length, chunk_size)
    q_chunk_idx -= jnp.where(reminder > 0, 0, 1)
    out_ref[...] = jnp.zeros_like(out_ref)

    def create_kv_async_copy_descriptors(h, i, buffer_index):
        pages_to_load = chunk_size // page_size
        page_offset = i * pages_to_load
        async_copy_k = MultiPageAsyncCopyDescriptor(
            k_pages_hbm_ref,
            k_vmem_buffer.at[buffer_index],
            sem,
            page_indices_ref,
            page_offset,
            pages_to_load,
            head_index=h,
        )
        async_copy_v = MultiPageAsyncCopyDescriptor(
            v_pages_hbm_ref,
            v_vmem_buffer.at[buffer_index],
            sem,
            page_indices_ref,
            page_offset,
            pages_to_load,
            head_index=h,
        )
        return async_copy_k, async_copy_v

    def next_block_indice(h, i):
        return jax.lax.cond((i + 1) * chunk_size < length, lambda: (h, i + 1), lambda: (h + 1, 0))

    def per_kv_chunk_body(i, _):
        @pl.when((i * chunk_size) < length)
        def body():
            buffer_index = buffer_index_ref[0]

            @pl.when(i == 0)
            def init():
                m_ref[...] = jnp.full_like(m_ref, -jnp.inf)
                l_ref[...] = jnp.zeros_like(l_ref)

                @pl.when(h == 0)
                def prefetch_first_kv():
                    # prefetch the first kv chunk.
                    async_copy_k, async_copy_v = create_kv_async_copy_descriptors(h, i, buffer_index)
                    async_copy_k.start()
                    async_copy_v.start()

            next_h, next_i = next_block_indice(h, i)

            @pl.when((next_h < num_kv_heads) & (next_i <= q_chunk_idx))
            def prefetch_next_block():
                # prefetch the kv chunk for next iteration.
                next_buffer_index = jnp.where(buffer_index == 0, 1, 0)
                async_copy_next_k, async_copy_next_v = create_kv_async_copy_descriptors(
                    next_h, next_i, next_buffer_index
                )

                async_copy_next_k.start()
                async_copy_next_v.start()
                buffer_index_ref[0] = next_buffer_index

            async_copy_k, async_copy_v = create_kv_async_copy_descriptors(h, i, buffer_index)

            k = async_copy_k.wait_and_get_loaded()
            v = async_copy_v.wait_and_get_loaded()

            mask_shape = (chunk_size, chunk_size)
            row_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 0)
            row_ids += q_chunk_idx * chunk_size
            col_ids = jax.lax.broadcasted_iota(jnp.int32, mask_shape, 1)
            col_ids += i * chunk_size
            causal_mask = col_ids <= row_ids
            causal_mask_value = jnp.where(causal_mask, 0.0, DEFAULT_MASK_VALUE)

            def per_group_body(group_idx, _):
                q = q_ref[group_idx]
                s = jnp.einsum("td,sd->ts", q, k, preferred_element_type=jnp.float32) + causal_mask_value
                # mask.
                s_max = jnp.max(s, axis=1, keepdims=True)

                prev_m = m_ref[group_idx]
                prev_l = l_ref[group_idx]

                cur_m = jnp.maximum(prev_m, s_max)
                cur_m_to_attn_size = jax.lax.broadcast_in_dim(cur_m, (chunk_size, chunk_size), (0, 1))

                p = jnp.exp(s - cur_m_to_attn_size)

                cur_l = jnp.exp(prev_m - cur_m) * prev_l + jnp.sum(p, axis=1, keepdims=True)

                out = out_ref[group_idx]

                out_ref[group_idx, :, :] = (
                    out * jax.lax.broadcast_in_dim(jnp.exp(prev_m - cur_m), (chunk_size, head_dim), (0, 1)) + p @ v
                ).astype(out_ref.dtype)  # p @ v  "ts,sd->td"

                m_ref[group_idx, :, :] = cur_m
                l_ref[group_idx, :, :] = cur_l
                return ()

            jax.lax.fori_loop(0, group_size, per_group_body, ())

        @pl.when(((i + 1) * chunk_size) >= length)
        def rescale():
            out_ref[...] = (
                out_ref[...] / jax.lax.broadcast_in_dim(l_ref[...], (group_size, chunk_size, head_dim), (0, 1, 2))
            ).astype(out_ref.dtype)

        return ()

    # loop over k, v cache chunk.
    jax.lax.fori_loop(0, lax.div(length + chunk_size - 1, chunk_size), per_kv_chunk_body, ())
