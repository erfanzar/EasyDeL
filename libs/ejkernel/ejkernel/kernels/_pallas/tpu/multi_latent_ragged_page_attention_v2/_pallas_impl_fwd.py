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

"""TPU-friendly multi-latent ragged paged attention v2 kernel.

This is adapted from the upstream TPU inference MLA v2 kernel and wrapped in
ejkernel's naming conventions so it can live alongside the existing v1 op.
"""

from __future__ import annotations

import functools
from enum import Enum

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)
DEFAULT_VMEM_LIMIT_BYTES = 100 * 1024 * 1024


def cdiv(a, b):
    """Ceiling division of a by b."""
    assert b != 0
    return (a + b - 1) // b


def align_to(x, a):
    """Round x up to the nearest multiple of a."""
    return cdiv(x, a) * a


def get_dtype_bitwidth(dtype):
    """Return the number of bits per element for the given dtype."""
    return jax.dtypes.itemsize_bits(dtype)


def get_dtype_packing(dtype):
    """Return how many elements of dtype fit in a 32-bit word."""
    bits = get_dtype_bitwidth(dtype)
    return 32 // bits


def get_kv_cache_shape(
    total_num_pages,
    page_size,
    kv_dim,
    kv_dtype,
):
    """Compute the 4-D shape of the paged KV cache tensor.

    The cache is laid out as (total_num_pages, page_size // packing,
    packing, aligned_kv_dim) so that sub-word dtypes (e.g. bf16) are
    packed into 32-bit words for efficient TPU DMA transfers.

    Args:
        total_num_pages: Total number of pages in the cache.
        page_size: Number of tokens per page.
        kv_dim: Combined latent KV + rope dimension.
        kv_dtype: Data type of the KV cache entries.

    Returns:
        A 4-tuple describing the cache shape.
    """
    kv_packing = get_dtype_packing(kv_dtype)
    return (
        total_num_pages,
        align_to(page_size, kv_packing) // kv_packing,
        kv_packing,
        align_to(kv_dim, 128),
    )


class MlaCase(Enum):
    """Represents the different cases for MLA."""

    DECODE = 0
    PREFILL = 1
    MIXED = 2

    @property
    def symbol(self):
        """Return a short single-character symbol for the case (d/p/m)."""
        return {
            MlaCase.DECODE: "d",
            MlaCase.PREFILL: "p",
            MlaCase.MIXED: "m",
        }[self]


def static_validate_inputs(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    cache_kv: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    num_kv_pages_per_blocks: tuple[int, int, int] | None = None,
    num_queries_per_blocks: tuple[int, int, int] | None = None,
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
):
    """Validate inputs to the MLA RPA v2 kernel statically at trace time.

    Checks tensor ranks, shape compatibility, dtype constraints, and
    scalar-parameter values. Raises ``ValueError`` for any violation so that
    shape errors are caught before entering JAX tracing.

    Args:
        ql_nope: Query nope tensor, shape ``(max_tokens, num_q_heads, lkv_dim)``.
        q_pe: Query rope tensor, shape ``(max_tokens, num_q_heads, r_dim)``.
        new_kv_c: New KV latent tensor, shape ``(max_tokens, lkv_dim)``.
        new_k_pe: New key rope tensor, shape ``(max_tokens, r_dim)``.
        cache_kv: Paged KV cache, shape
            ``(total_pages, page_size // packing, packing, lkv_dim + r_dim)``.
        kv_lens: Per-sequence KV lengths, int32 shape ``(max_num_seqs,)``.
        page_indices: Flattened page table, int32 shape
            ``(max_num_seqs * pages_per_seq,)``.
        cu_q_lens: Cumulative query lengths, int32 shape
            ``(max_num_seqs + 1,)``.
        distribution: Three-element int32 array ``[decode_end, prefill_end,
            mixed_end]`` partitioning the batch into scheduling regions.
        sm_scale: Softmax scaling factor (not validated, only consumed).
        sliding_window: If not ``None``, must be positive.
        soft_cap: If not ``None``, must not be 0.0.
        mask_value: Padding value for attention logits (not validated).
        q_scale: Optional per-sequence query scale (not validated).
        k_scale: Optional per-sequence key scale (not validated).
        v_scale: Optional per-sequence value scale (not validated).
        chunk_prefill_size: If not ``None``, must be positive.
        num_kv_pages_per_blocks: Three-tuple of block tile page counts; each
            entry must be positive.
        num_queries_per_blocks: Three-tuple of query-block sizes; each entry
            must be positive.
        vmem_limit_bytes: If not ``None``, must be positive.
        debug_mode: Flag passed through without validation.

    Raises:
        ValueError: If any shape, dtype, or scalar-parameter constraint
            is violated.
    """
    if len(ql_nope.shape) != 3:
        raise ValueError(f"Expected 3D array for {ql_nope.shape=}")
    if len(q_pe.shape) != 3:
        raise ValueError(f"Expected 3D array for {q_pe.shape=}")
    if len(new_kv_c.shape) != 2:
        raise ValueError(f"Expected 2D array for {new_kv_c.shape=}")
    if len(new_k_pe.shape) != 2:
        raise ValueError(f"Expected 2D array for {new_k_pe.shape=}")

    if ql_nope.shape[:2] != q_pe.shape[:2]:
        raise ValueError(f"Expected {ql_nope.shape[:2]=} to be equal to {q_pe.shape[:2]=}")
    if ql_nope.shape[0] != new_kv_c.shape[0]:
        raise ValueError(f"Expected {ql_nope.shape[0]=} to be equal to {new_kv_c.shape[0]=}")
    if new_kv_c.shape[0] != new_k_pe.shape[0]:
        raise ValueError(f"Expected {new_kv_c.shape[0]=} to be equal to {new_k_pe.shape[0]=}")
    if ql_nope.shape[2] != new_kv_c.shape[1]:
        raise ValueError(f"Expected {ql_nope.shape[2]=} to be equal to {new_kv_c.shape[1]=}")
    if q_pe.shape[2] != new_k_pe.shape[1]:
        raise ValueError(f"Expected {q_pe.shape[2]=} to be equal to {new_k_pe.shape[1]=}")

    actual_lkv_dim = ql_nope.shape[2]
    actual_r_dim = q_pe.shape[2]
    lkv_dim = align_to(actual_lkv_dim, 128)
    r_dim = align_to(actual_r_dim, 128)

    _, page_size_per_kv_packing, kv_packing, kv_dim = cache_kv.shape

    if lkv_dim + r_dim != kv_dim:
        raise ValueError(f"Expected {lkv_dim=} + {r_dim=} to be equal to {kv_dim=}")

    if cache_kv.dtype != new_kv_c.dtype:
        raise ValueError(f"Expected {cache_kv.dtype=} to be equal to {new_kv_c.dtype=}.")
    if cache_kv.dtype != new_k_pe.dtype:
        raise ValueError(f"Expected {cache_kv.dtype=} to be equal to {new_k_pe.dtype=}.")

    if not jnp.issubdtype(cache_kv.dtype, jnp.floating):
        raise ValueError(f"Expected {cache_kv.dtype=} to be a floating point.")

    if kv_packing != get_dtype_packing(cache_kv.dtype):
        raise ValueError(f"{kv_packing=} does not match with {cache_kv.dtype=}")

    if not (jnp.int32 == kv_lens.dtype == page_indices.dtype == cu_q_lens.dtype == distribution.dtype):
        raise ValueError(
            f"Expected int32 dtype for {kv_lens.dtype=}, {page_indices.dtype=},"
            f" {cu_q_lens.dtype=}, {distribution.dtype=}"
        )

    if not (len(kv_lens.shape) == len(page_indices.shape) == len(cu_q_lens.shape) == 1):
        raise ValueError(f"Expected 1D array for {kv_lens.shape=}, {page_indices.shape=}, {cu_q_lens.shape=}")

    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    if num_page_indices % max_num_seqs != 0:
        raise ValueError(f"Expected {num_page_indices=} to be divisible by {max_num_seqs=}.")
    if cu_q_lens.shape != (max_num_seqs + 1,):
        raise ValueError(f"Expected {cu_q_lens.shape=} to be ({max_num_seqs + 1},).")
    if distribution.shape != (3,):
        raise ValueError(f"Expected {distribution.shape=} to be (3,).")

    page_size = page_size_per_kv_packing * kv_packing
    if page_size % kv_packing != 0:
        raise ValueError(f"{page_size=} must be divisible by {kv_packing=}.")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError(f"{sliding_window=} must be positive.")
    if soft_cap is not None and soft_cap == 0.0:
        raise ValueError(f"{soft_cap=} must not be 0.0.")
    if chunk_prefill_size is not None and chunk_prefill_size <= 0:
        raise ValueError(f"{chunk_prefill_size=} must be positive.")
    if num_kv_pages_per_blocks is not None:
        for num_kv_pages_per_block in num_kv_pages_per_blocks:
            if num_kv_pages_per_block <= 0:
                raise ValueError(f"{num_kv_pages_per_block=} must be positive.")
    if num_queries_per_blocks is not None:
        for num_queries_per_block in num_queries_per_blocks:
            if num_queries_per_block <= 0:
                raise ValueError(f"{num_queries_per_block=} must be positive.")
    if vmem_limit_bytes is not None and vmem_limit_bytes <= 0:
        raise ValueError(f"{vmem_limit_bytes=} must be positive.")

    del sm_scale
    del mask_value
    del q_scale
    del k_scale
    del v_scale
    del debug_mode


def _mla_ragged_paged_attention_kernel(
    kv_lens_ref,
    page_indices_ref,
    cu_q_lens_ref,
    start_end_seq_idx_ref,
    sem_ids_ref,
    bo_ids_ref,
    bkv_update_ids_ref,
    ql_nope_hbm_ref,
    q_pe_hbm_ref,
    new_kv_c_hbm_ref,
    new_k_pe_hbm_ref,
    cache_kv_hbm_ref,
    o_hbm_ref,
    updated_cache_kv_hbm_ref,
    bkvc_x2_ref,
    bkpe_x2_ref,
    bq_nope_x2_ref,
    bq_rope_x2_ref,
    bo_x2_ref,
    sems,
    l_ref,
    m_ref,
    acc_ref,
    *,
    static_q_len: int,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    bkv_p,
    bq_sz,
    debug_mode: bool = False,
):
    """Pallas kernel body for MLA ragged paged attention v2.

    This function is the inner kernel executed by ``pl.pallas_call``. It
    orchestrates double-buffered asynchronous DMA transfers between HBM and
    VMEM for queries, KV blocks, and outputs, while computing tiled flash
    attention (split into nope and rope components) with online softmax
    rescaling. New KV tokens are merged into the paged cache in-place.

    The kernel iterates over query blocks (bq) and KV blocks (bkv) for
    each sequence in the assigned range [start_seq_idx, end_seq_idx),
    overlapping DMA prefetches with compute via semaphore-guarded
    double buffering.
    """
    assert ql_nope_hbm_ref.shape == o_hbm_ref.shape
    nope_dim = ql_nope_hbm_ref.shape[-1]
    pe_dim = q_pe_hbm_ref.shape[-1]
    assert nope_dim + pe_dim == cache_kv_hbm_ref.shape[-1]

    _, num_q_heads_per_q_packing, q_packing, lkv_dim = ql_nope_hbm_ref.shape
    r_dim = q_pe_hbm_ref.shape[-1]
    num_q_heads = num_q_heads_per_q_packing * q_packing
    total_num_pages, page_size_per_kv_packing, kv_packing, _ = cache_kv_hbm_ref.shape
    max_num_seqs = kv_lens_ref.shape[0]
    num_page_indices = page_indices_ref.shape[0]

    assert num_page_indices % max_num_seqs == 0
    pages_per_seq = num_page_indices // max_num_seqs
    q_dtype = ql_nope_hbm_ref.dtype
    kv_dtype = cache_kv_hbm_ref.dtype
    assert q_pe_hbm_ref.dtype == q_dtype
    assert o_hbm_ref.dtype == q_dtype
    assert get_dtype_packing(q_dtype) == q_packing
    assert get_dtype_packing(kv_dtype) == kv_packing
    assert lkv_dim % 128 == 0
    assert r_dim % 128 == 0
    bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
    bkv_sz = bkv_sz_per_kv_packing * kv_packing
    page_size = page_size_per_kv_packing * kv_packing

    start_seq_idx = start_end_seq_idx_ref[0]
    end_seq_idx = start_end_seq_idx_ref[1]
    seq_idx = pl.program_id(0) + start_seq_idx
    q_start = cu_q_lens_ref[seq_idx]
    q_end = cu_q_lens_ref[seq_idx + 1]
    q_len = q_end - q_start
    kv_len = kv_lens_ref[seq_idx]

    def debug_print(msg, *args):
        """Conditionally emit a Pallas debug print when debug_mode is enabled."""
        if debug_mode:
            pl.debug_print(msg, *args)

    def flash_attention(
        ql_nope,
        q_pe,
        kv_c,
        k_pe,
        *,
        bq_idx,
        bkv_idx,
    ):
        """Compute one tile of flash attention with online softmax rescaling.

        Scores are computed as the sum of nope and rope dot products, then
        masked for causal / sliding-window constraints. The running max (m),
        sum-of-exponentials (l), and weighted-value accumulator (acc) in
        scratch VMEM are updated in-place using the standard online softmax
        algorithm, avoiding full materialisation of the attention matrix.

        Args:
            ql_nope: Query nope block of shape (bq_sz * num_q_heads, lkv_dim).
            q_pe: Query rope block of shape (bq_sz * num_q_heads, r_dim).
            kv_c: KV latent block of shape (bkv_sz, lkv_dim).
            k_pe: Key rope block of shape (bkv_sz, r_dim).
            bq_idx: Index of the current query block within the sequence.
            bkv_idx: Index of the current KV block within the sequence.
        """
        assert len(ql_nope.shape) == 2
        assert len(q_pe.shape) == 2
        assert len(kv_c.shape) == 2
        assert len(k_pe.shape) == 2
        assert ql_nope.shape[0] % num_q_heads == 0
        assert ql_nope.shape[0] == q_pe.shape[0]
        assert q_pe.shape[0] % bq_sz == 0
        assert ql_nope.shape[1] == lkv_dim
        assert q_pe.shape[1] == r_dim
        assert kv_c.shape == (bkv_sz, lkv_dim)
        assert k_pe.shape == (bkv_sz, r_dim)
        head_l_ref = l_ref.at[: ql_nope.shape[0]]
        head_m_ref = m_ref.at[: ql_nope.shape[0]]
        head_acc_ref = acc_ref.at[: ql_nope.shape[0]]

        def load_with_init(ref, init_val):
            """Load a scratch ref, or initialise it with a constant on the first KV block."""
            return jnp.where(bkv_idx == 0, jnp.full_like(ref, init_val), ref[...])

        s_nope = jnp.einsum("nd,md->nm", ql_nope, kv_c, preferred_element_type=jnp.float32)
        s_pe = jnp.einsum("nd,md->nm", q_pe, k_pe, preferred_element_type=jnp.float32)
        s = s_nope + s_pe
        s *= sm_scale
        if k_scale is not None:
            s *= k_scale
        if q_scale is not None:
            s *= q_scale

        q_span = kv_len - q_len + bq_idx * bq_sz + lax.broadcasted_iota(jnp.int32, s.shape, 0) // num_q_heads
        k_span = bkv_idx * bkv_sz + lax.broadcasted_iota(jnp.int32, s.shape, 1)
        mask = q_span < k_span
        if sliding_window is not None:
            mask = jnp.logical_or(mask, q_span - sliding_window >= k_span)

        if soft_cap is not None:
            s = soft_cap * jnp.tanh(s / soft_cap)
        s = jnp.where(mask, mask_value, s)
        s_rowmax = jnp.max(s, axis=1, keepdims=True)
        m_prev = load_with_init(head_m_ref, -jnp.inf)
        m_curr = jnp.maximum(m_prev, s_rowmax)
        head_m_ref[...] = m_curr
        p = jnp.exp(s - broadcast_minor(m_curr, s.shape))

        pv = jnp.einsum("nm,md->nd", p, kv_c, preferred_element_type=jnp.float32)
        if v_scale is not None:
            pv *= v_scale

        p_rowsum = jnp.sum(p, axis=1, keepdims=True)
        exp_m_diff = jnp.exp(m_prev - m_curr)
        l_prev = load_with_init(head_l_ref, 0.0)
        l_curr = exp_m_diff * l_prev + p_rowsum
        head_l_ref[...] = l_curr
        o_prev = load_with_init(head_acc_ref, 0.0)
        o_curr = broadcast_minor(exp_m_diff, o_prev.shape) * o_prev + pv
        head_acc_ref[...] = o_curr

    def _async_copy(src, dst, sem, wait):
        """Issue or wait on an asynchronous DMA copy between HBM and VMEM."""
        if debug_mode:
            return
        cp = pltpu.make_async_copy(src, dst, sem)
        if wait:
            cp.wait()
        else:
            cp.start()

    def _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, *, wait=False):
        """Fetch a KV block from HBM into a double-buffered VMEM slot.

        The block is assembled from two sources: pages already committed to
        the paged cache and newly arriving KV tokens that have not yet been
        written back. When ``wait=False`` the DMA transfers are started
        asynchronously; when ``wait=True`` the function blocks until the
        previously started transfers complete.

        Returns:
            A (cache_offset, new_kv_size) tuple indicating where the cache
            portion ends and how many tokens came from the new-KV buffer.
        """
        sem = sems.at[0, bkv_sem_idx]
        bkvc_vmem_ref = bkvc_x2_ref.at[bkv_sem_idx]
        bkvpe_vmem_ref = bkpe_x2_ref.at[bkv_sem_idx]

        reshaped_cache_hbm_ref = cache_kv_hbm_ref.reshape(
            total_num_pages * page_size_per_kv_packing,
            *cache_kv_hbm_ref.shape[2:],
        )

        kv_len = kv_lens_ref[seq_idx]
        kv_len_start = bkv_idx * bkv_sz
        kv_p_start = bkv_idx * bkv_p

        q_start = cu_q_lens_ref[seq_idx]
        q_end = cu_q_lens_ref[seq_idx + 1]
        q_len = q_end - q_start

        kv_left = kv_len - kv_len_start
        kv_left_frm_cache = jnp.maximum(kv_left - q_len, 0)
        kv_left_frm_cache_per_kv_packing = cdiv(kv_left_frm_cache, kv_packing)
        kv_left_frm_new = kv_left - kv_left_frm_cache

        bkv_sz_frm_cache = jnp.minimum(kv_left_frm_cache, bkv_sz)
        bkv_sz_frm_new = jnp.minimum(bkv_sz - bkv_sz_frm_cache, kv_left_frm_new)
        bkv_sz_frm_cache_per_kv_packing = cdiv(bkv_sz_frm_cache, kv_packing)
        cdiv(bkv_sz_frm_new, kv_packing)
        page_indices_offset = seq_idx * pages_per_seq + kv_p_start

        new_kv_len_start = q_end - kv_left_frm_new
        new_kv_len_start_per_kv_packing = new_kv_len_start // kv_packing
        bkv_sz_frm_new_kv_packing_to_fetch = jnp.where(
            bkv_sz_frm_new > 0,
            cdiv(new_kv_len_start + bkv_sz_frm_new, kv_packing) - new_kv_len_start_per_kv_packing,
            0,
        )
        dma_bkv_sz = bkv_sz_frm_cache_per_kv_packing + bkv_sz_frm_new_kv_packing_to_fetch

        if not wait:
            wait_update_kv_cache(bkv_sem_idx)

            for i in range(bkv_p):
                sz_per_kv_packing = jnp.clip(
                    kv_left_frm_cache_per_kv_packing - i * page_size_per_kv_packing,
                    0,
                    page_size_per_kv_packing,
                )
                page_idx = jnp.minimum(page_indices_offset + i, num_page_indices - 1)
                _async_copy(
                    reshaped_cache_hbm_ref.at[
                        pl.ds(page_indices_ref[page_idx] * page_size_per_kv_packing, sz_per_kv_packing),
                        ...,
                        :nope_dim,
                    ],
                    bkvc_vmem_ref.at[pl.ds(i * page_size_per_kv_packing, sz_per_kv_packing)],
                    sem,
                    wait,
                )
                _async_copy(
                    reshaped_cache_hbm_ref.at[
                        pl.ds(page_indices_ref[page_idx] * page_size_per_kv_packing, sz_per_kv_packing),
                        ...,
                        nope_dim:,
                    ],
                    bkvpe_vmem_ref.at[pl.ds(i * page_size_per_kv_packing, sz_per_kv_packing)],
                    sem,
                    wait,
                )

            _async_copy(
                new_kv_c_hbm_ref.at[pl.ds(new_kv_len_start_per_kv_packing, bkv_sz_frm_new_kv_packing_to_fetch)],
                bkvc_vmem_ref.at[pl.ds(bkv_sz_frm_cache_per_kv_packing, bkv_sz_frm_new_kv_packing_to_fetch)],
                sem,
                wait,
            )
            _async_copy(
                new_k_pe_hbm_ref.at[pl.ds(new_kv_len_start_per_kv_packing, bkv_sz_frm_new_kv_packing_to_fetch)],
                bkvpe_vmem_ref.at[pl.ds(bkv_sz_frm_cache_per_kv_packing, bkv_sz_frm_new_kv_packing_to_fetch)],
                sem,
                wait,
            )
        else:
            dst_kvc = bkvc_vmem_ref.at[pl.ds(0, dma_bkv_sz)]
            _async_copy(src=dst_kvc, dst=dst_kvc, sem=sem, wait=True)
            dst_kvpe = bkvpe_vmem_ref.at[pl.ds(0, dma_bkv_sz)]
            _async_copy(src=dst_kvpe, dst=dst_kvpe, sem=sem, wait=True)

        return kv_len_start + bkv_sz_frm_cache, bkv_sz_frm_new

    def _pack_new_kv(bkv_sem_idx, offset, update_sz):
        """Re-pack newly fetched KV tokens into the sub-word packing layout.

        New KV tokens arrive with a potentially different packing alignment
        than their target position in the KV block buffer. This function
        performs bitwise shift-and-merge operations to rotate elements into
        the correct 32-bit word positions so the block can be used directly
        for attention and written back to the paged cache.

        Args:
            bkv_sem_idx: Index of the double-buffer slot holding the KV block.
            offset: Token offset within the full KV sequence where the update
                begins.
            update_sz: Number of new tokens to pack.
        """
        bkvc_vmem_ref = bkvc_x2_ref.at[bkv_sem_idx]
        bkvpe_vmem_ref = bkpe_x2_ref.at[bkv_sem_idx]

        update_kv_packing_iters = cdiv((offset % kv_packing) + update_sz, kv_packing)
        kv_packing_offset = offset % kv_packing
        new_kv_len_start = q_end - kv_len + offset
        new_kv_packing_offset = new_kv_len_start % kv_packing

        token_offset_in_bkv = offset % bkv_sz
        kv_packing_idx = token_offset_in_bkv // kv_packing

        shift_amount = kv_packing_offset - new_kv_packing_offset
        bits_per_element = get_dtype_bitwidth(bkvc_vmem_ref.dtype)
        shift_bits = bits_per_element * (shift_amount % kv_packing)
        shift_bits = shift_bits.astype(jnp.uint32)

        kv_packing_idx_new = cdiv(token_offset_in_bkv, kv_packing) + (-shift_amount) // kv_packing
        curr_kvc_reg = bkvc_vmem_ref[kv_packing_idx_new, :, :]
        curr_kpe_reg = bkvpe_vmem_ref[kv_packing_idx_new, :, :]
        next_kvc_reg = bkvc_vmem_ref[kv_packing_idx_new + 1, :, :]
        next_kpe_reg = bkvpe_vmem_ref[kv_packing_idx_new + 1, :, :]

        def merge_loop_body(i, vals):
            """Shift and merge one 32-bit word of new KV data into the block buffer."""
            (
                kv_packing_idx,
                kv_packing_idx_new,
                curr_kvc_reg,
                curr_kpe_reg,
                next_kvc_reg,
                next_kpe_reg,
            ) = vals
            curr_kvc_reg_u32 = pltpu.bitcast(curr_kvc_reg, jnp.uint32)
            curr_kpe_reg_u32 = pltpu.bitcast(curr_kpe_reg, jnp.uint32)
            next_kvc_reg_u32 = pltpu.bitcast(next_kvc_reg, jnp.uint32)
            next_kpe_reg_u32 = pltpu.bitcast(next_kpe_reg, jnp.uint32)

            shifted_kvc_u32 = lax.bitwise_or(
                lax.shift_right_logical(curr_kvc_reg_u32, 32 - shift_bits),
                lax.shift_left(next_kvc_reg_u32, shift_bits),
            )
            shifted_kpe_u32 = lax.bitwise_or(
                lax.shift_right_logical(curr_kpe_reg_u32, 32 - shift_bits),
                lax.shift_left(next_kpe_reg_u32, shift_bits),
            )

            rotated_kvc_u32 = lax.select(shift_bits == 0, curr_kvc_reg_u32, shifted_kvc_u32)
            rotated_kpe_u32 = lax.select(shift_bits == 0, curr_kpe_reg_u32, shifted_kpe_u32)

            next_kvc_reg_shifted = pltpu.bitcast(rotated_kvc_u32, next_kvc_reg.dtype)
            next_kpe_reg_shifted = pltpu.bitcast(rotated_kpe_u32, next_kpe_reg.dtype)

            offset_in_word = i * kv_packing + lax.broadcasted_iota(
                dtype=jnp.int32, shape=[kv_packing, lkv_dim], dimension=0
            )
            kvc_mask = jnp.logical_and(
                offset_in_word >= kv_packing_offset, offset_in_word < kv_packing_offset + update_sz
            )
            updated_kvc_reg = lax.select(kvc_mask, next_kvc_reg_shifted, bkvc_vmem_ref[kv_packing_idx, :, :])

            offset_in_word_pe = i * kv_packing + lax.broadcasted_iota(
                dtype=jnp.int32, shape=[kv_packing, r_dim], dimension=0
            )
            kpe_mask = jnp.logical_and(
                offset_in_word_pe >= kv_packing_offset, offset_in_word_pe < kv_packing_offset + update_sz
            )
            updated_kpe_reg = lax.select(kpe_mask, next_kpe_reg_shifted, bkvpe_vmem_ref[kv_packing_idx, :, :])

            bkvc_vmem_ref[kv_packing_idx, :, :] = updated_kvc_reg
            bkvpe_vmem_ref[kv_packing_idx, :, :] = updated_kpe_reg

            kv_packing_idx += 1
            kv_packing_idx_new += 1
            curr_kvc_reg = next_kvc_reg
            curr_kpe_reg = next_kpe_reg
            next_kvc_reg = bkvc_vmem_ref[kv_packing_idx_new + 1, :, :]
            next_kpe_reg = bkvpe_vmem_ref[kv_packing_idx_new + 1, :, :]
            return (
                kv_packing_idx,
                kv_packing_idx_new,
                curr_kvc_reg,
                curr_kpe_reg,
                next_kvc_reg,
                next_kpe_reg,
            )

        lax.fori_loop(
            0,
            update_kv_packing_iters,
            merge_loop_body,
            (
                kv_packing_idx,
                kv_packing_idx_new,
                curr_kvc_reg,
                curr_kpe_reg,
                next_kvc_reg,
                next_kpe_reg,
            ),
        )

    def _update_kv_cache(
        seq_idx,
        bkv_sem_idx,
        offset,
        update_sz,
        *,
        wait=False,
    ):
        """Write back updated KV tokens from VMEM to the paged cache in HBM.

        Iterates over the affected pages and issues asynchronous DMA copies
        for both the nope and rope components. When ``wait=True``, blocks
        until the previously started write-back DMAs complete.

        Args:
            seq_idx: Sequence index whose cache is being updated.
            bkv_sem_idx: Double-buffer slot index holding the data.
            offset: Token offset within the KV sequence to write.
            update_sz: Number of tokens to write back.
            wait: If True, wait for completion instead of starting new DMAs.
        """
        sem = sems.at[3, bkv_sem_idx]
        bkvc_vmem_ref = bkvc_x2_ref.at[bkv_sem_idx]
        bkvpe_vmem_ref = bkpe_x2_ref.at[bkv_sem_idx]

        update_kv_packing_iters = cdiv((offset % kv_packing) + update_sz, kv_packing)

        cache_kv_hbm_shape = updated_cache_kv_hbm_ref.shape
        reshaped_cache_kv_hbm_ref = updated_cache_kv_hbm_ref.reshape(
            cache_kv_hbm_shape[0] * cache_kv_hbm_shape[1],
            *cache_kv_hbm_shape[2:],
        )

        if not wait:
            kv_p_start = offset // page_size
            kv_p_end = cdiv(offset + update_sz, page_size)
            start_word_in_page = (offset % page_size) // kv_packing
            start_word_in_vmem = (offset % bkv_sz) // kv_packing
            words_to_transfer = update_kv_packing_iters
            page_indices_offset = seq_idx * pages_per_seq + kv_p_start

            def loop_body(i, states):
                """Transfer one page's worth of KV words from VMEM to HBM."""
                curr_word_in_page, words_to_transfer, curr_word_in_vmem = states
                sz_words = jnp.minimum(page_size_per_kv_packing - curr_word_in_page, words_to_transfer)
                page_idx = page_indices_ref[page_indices_offset + i]

                _async_copy(
                    bkvc_vmem_ref.at[pl.ds(curr_word_in_vmem, sz_words)],
                    reshaped_cache_kv_hbm_ref.at[
                        pl.ds(page_idx * page_size_per_kv_packing + curr_word_in_page, sz_words),
                        ...,
                        :nope_dim,
                    ],
                    sem,
                    wait=False,
                )
                _async_copy(
                    bkvpe_vmem_ref.at[pl.ds(curr_word_in_vmem, sz_words)],
                    reshaped_cache_kv_hbm_ref.at[
                        pl.ds(page_idx * page_size_per_kv_packing + curr_word_in_page, sz_words),
                        ...,
                        nope_dim:,
                    ],
                    sem,
                    wait=False,
                )
                return 0, words_to_transfer - sz_words, curr_word_in_vmem + sz_words

            lax.fori_loop(
                0,
                kv_p_end - kv_p_start,
                loop_body,
                (
                    start_word_in_page,
                    words_to_transfer,
                    start_word_in_vmem,
                ),
                unroll=False,
            )
        else:
            dma_sz_words = update_kv_packing_iters
            dst_kv = bkvc_vmem_ref.at[pl.ds(0, dma_sz_words)]
            _async_copy(src=dst_kv, dst=dst_kv, sem=sem, wait=True)
            dst_kv = bkvpe_vmem_ref.at[pl.ds(0, dma_sz_words)]
            _async_copy(src=dst_kv, dst=dst_kv, sem=sem, wait=True)

    def _fetch_bq(seq_idx, bq_idx, bq_sem_idx, *, wait=False):
        """Fetch a query block (nope + rope) from HBM into a VMEM double-buffer slot."""
        sem = sems.at[1, bq_sem_idx]
        bq_nope_vmem_ref = bq_nope_x2_ref.at[bq_sem_idx]
        bq_rope_vmem_ref = bq_rope_x2_ref.at[bq_sem_idx]

        q_len_start = cu_q_lens_ref[seq_idx] + bq_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(
            ql_nope_hbm_ref.at[pl.ds(q_len_start, sz)],
            bq_nope_vmem_ref.at[pl.ds(0, sz)],
            sem,
            wait,
        )
        _async_copy(
            q_pe_hbm_ref.at[pl.ds(q_len_start, sz)],
            bq_rope_vmem_ref.at[pl.ds(0, sz)],
            sem,
            wait,
        )

    def _send_bo(seq_idx, bo_idx, bo_sem_idx, *, wait=False):
        """Send the output block from VMEM back to HBM."""
        sem = sems.at[2, bo_sem_idx]
        vmem_ref = bo_x2_ref.at[bo_sem_idx]
        q_len_start = cu_q_lens_ref[seq_idx] + bo_idx * bq_sz
        q_end = cu_q_lens_ref[seq_idx + 1]
        sz = jnp.minimum(bq_sz, q_end - q_len_start)

        _async_copy(
            vmem_ref.at[pl.ds(0, sz)],
            o_hbm_ref.at[pl.ds(q_len_start, sz)],
            sem,
            wait,
        )

    def start_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        """Start an asynchronous KV block fetch from HBM to VMEM."""
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

    def wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx):
        """Block until the in-flight KV block fetch completes."""
        return _fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx, wait=True)

    def start_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        """Start an asynchronous query block fetch from HBM to VMEM."""
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx)

    def wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx):
        """Block until the in-flight query block fetch completes."""
        return _fetch_bq(seq_idx, bq_idx, bq_sem_idx, wait=True)

    def start_send_bo(seq_idx, bo_idx, bo_sem_idx):
        """Start an asynchronous output block send from VMEM to HBM."""
        bo_ids_ref[bo_sem_idx] = seq_idx
        bo_ids_ref[bo_sem_idx + 2] = bo_idx
        _send_bo(seq_idx, bo_idx, bo_sem_idx)

    def wait_send_bo(bo_sem_idx):
        """Wait for the previous output block DMA to complete, if any."""
        old_seq_idx = bo_ids_ref[bo_sem_idx]
        old_bo_idx = bo_ids_ref[bo_sem_idx + 2]

        @pl.when(jnp.logical_and(0 <= old_seq_idx, old_seq_idx <= seq_idx))
        def _():
            """Wait for the previous output DMA if its sequence is still in scope."""
            _send_bo(old_seq_idx, old_bo_idx, bo_sem_idx, wait=True)

    def start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz):
        """Start an asynchronous write-back of updated KV tokens to the paged cache."""
        bkv_update_ids_ref[bkv_sem_idx] = seq_idx
        bkv_update_ids_ref[bkv_sem_idx + 2] = offset
        bkv_update_ids_ref[bkv_sem_idx + 4] = update_sz
        _update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)

    def wait_update_kv_cache(bkv_sem_idx):
        """Wait for the previous KV cache write-back DMA to complete, if any."""
        update_sz = bkv_update_ids_ref[bkv_sem_idx + 4]

        @pl.when(update_sz > 0)
        def _():
            """Complete the pending KV cache write-back and reset the update size."""
            seq_idx_local = bkv_update_ids_ref[bkv_sem_idx]
            offset_local = bkv_update_ids_ref[bkv_sem_idx + 2]
            bkv_update_ids_ref[bkv_sem_idx + 4] = 0
            _update_kv_cache(seq_idx_local, bkv_sem_idx, offset_local, update_sz, wait=True)

    def load_bq(bq_sem_idx, *, actual_bq_sz=bq_sz):
        """Load a query block from VMEM, unpack from 32-bit words, and reshape for attention."""
        q_nope_ref = (
            bq_nope_x2_ref.bitcast(jnp.uint32)
            .at[bq_sem_idx]
            .reshape(
                bq_sz * num_q_heads_per_q_packing,
                lkv_dim,
            )
        )
        q_nope_vec = pltpu.bitcast(
            q_nope_ref[: actual_bq_sz * num_q_heads_per_q_packing],
            q_dtype,
        ).reshape(actual_bq_sz * num_q_heads, lkv_dim)
        q_rope_ref = (
            bq_rope_x2_ref.bitcast(jnp.uint32)
            .at[bq_sem_idx]
            .reshape(
                bq_sz * num_q_heads_per_q_packing,
                r_dim,
            )
        )
        q_rope_vec = pltpu.bitcast(
            q_rope_ref[: actual_bq_sz * num_q_heads_per_q_packing],
            q_dtype,
        ).reshape(actual_bq_sz * num_q_heads, r_dim)
        return q_nope_vec, q_rope_vec

    def load_bkv(bkv_sem_idx):
        """Load a KV block from VMEM, unpack from 32-bit words, and split into nope and rope."""
        bkvc_ref = (
            bkvc_x2_ref.bitcast(jnp.uint32)
            .at[bkv_sem_idx, :bkv_sz_per_kv_packing]
            .reshape(
                bkv_sz_per_kv_packing,
                lkv_dim,
            )
        )
        bkvc_vec = pltpu.bitcast(bkvc_ref[...], kv_dtype).reshape(bkv_sz, lkv_dim)

        bkpe_ref = (
            bkpe_x2_ref.bitcast(jnp.uint32)
            .at[bkv_sem_idx, :bkv_sz_per_kv_packing]
            .reshape(
                bkv_sz_per_kv_packing,
                r_dim,
            )
        )
        bkpe_vec = pltpu.bitcast(bkpe_ref[...], kv_dtype).reshape(bkv_sz, r_dim)
        return bkvc_vec, bkpe_vec

    def broadcast_minor(src, shape):
        """Tile the minor (last) axis of src to match the target shape via concatenation."""
        if src.shape == shape:
            return src
        assert src.shape[:-1] == shape[:-1]
        assert src.shape[-1] % 128 == 0
        target_minor = align_to(shape[-1], src.shape[-1])
        return jnp.concatenate([src for _ in range(target_minor // src.shape[-1])], axis=-1)[..., : shape[-1]]

    def process():
        """Main processing loop: iterate over query and KV blocks with double-buffered prefetch."""
        num_bkv = cdiv(kv_len, bkv_sz)
        if static_q_len is None:
            actual_bq_sz = bq_sz
            num_bq = cdiv(q_len, actual_bq_sz)
        else:
            actual_bq_sz = min(bq_sz, static_q_len)
            num_bq = cdiv(static_q_len, actual_bq_sz)

        def get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx):
            """Advance to the next query block, wrapping to the next sequence if needed."""
            next_bq_idx = bq_idx + 1
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bq_sem_idx = lax.select(bq_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bq_sem_idx

        def get_next_bkv_ids(seq_idx, bq_idx, bkv_idx, bkv_sem_idx):
            """Advance to the next KV block, wrapping through query blocks and sequences."""
            next_bkv_idx = bkv_idx + 1
            is_last_bkv = next_bkv_idx == num_bkv
            next_bkv_idx = lax.select(is_last_bkv, 0, next_bkv_idx)
            next_bq_idx = lax.select(is_last_bkv, bq_idx + 1, bq_idx)
            is_last_bq = next_bq_idx == num_bq
            next_bq_idx = lax.select(is_last_bq, 0, next_bq_idx)
            next_seq_idx = lax.select(is_last_bq, seq_idx + 1, seq_idx)
            next_bkv_sem_idx = lax.select(bkv_sem_idx == 0, 1, 0)
            return next_seq_idx, next_bq_idx, next_bkv_idx, next_bkv_sem_idx

        def compute_with_bq(bq_idx, _):
            """Process one query block: loop over all KV blocks and write the output."""
            bq_sem_idx = sem_ids_ref[0]
            next_seq_idx, next_bq_idx, next_bq_sem_idx = get_next_bq_ids(seq_idx, bq_idx, bq_sem_idx)

            @pl.when(next_seq_idx < end_seq_idx)
            def prefetch_next_bq():
                """Prefetch the next query block into the alternate VMEM buffer."""
                sem_ids_ref[0] = next_bq_sem_idx
                start_fetch_bq(next_seq_idx, next_bq_idx, next_bq_sem_idx)

            def compute_with_bkv(bkv_idx, _):
                """Process one KV block: prefetch next, pack new KV, and run flash attention."""
                bkv_sem_idx = sem_ids_ref[1]
                next_seq_idx, _, next_bkv_idx, next_bkv_sem_idx = get_next_bkv_ids(
                    seq_idx,
                    bq_idx,
                    bkv_idx,
                    bkv_sem_idx,
                )

                @pl.when(next_seq_idx < end_seq_idx)
                def prefetch_next_bkv():
                    """Prefetch the next KV block into the alternate VMEM buffer."""
                    sem_ids_ref[1] = next_bkv_sem_idx
                    start_fetch_bkv(next_seq_idx, next_bkv_idx, next_bkv_sem_idx)

                @pl.when(bkv_idx == 0)
                def wait_cur_bq():
                    """Ensure the current query block DMA has completed before use."""
                    wait_fetch_bq(seq_idx, bq_idx, bq_sem_idx)

                offset, update_sz = wait_fetch_bkv(seq_idx, bkv_idx, bkv_sem_idx)

                @pl.when(update_sz > 0)
                def pack_new_kv():
                    """Re-pack newly fetched KV tokens into the correct word alignment."""
                    _pack_new_kv(bkv_sem_idx, offset, update_sz)

                @pl.when(jnp.logical_and(update_sz > 0, bq_idx == 0))
                def update_cur_bkv_to_cache():
                    """Write back the newly packed KV tokens to the paged cache on the first query block."""
                    start_update_kv_cache(seq_idx, bkv_sem_idx, offset, update_sz)

                bkvc, bkpe = load_bkv(bkv_sem_idx)
                bq_nope_vec, bq_pe_vec = load_bq(bq_sem_idx, actual_bq_sz=actual_bq_sz)

                if debug_mode:
                    return

                flash_attention(
                    bq_nope_vec,
                    bq_pe_vec,
                    bkvc,
                    bkpe,
                    bq_idx=bq_idx,
                    bkv_idx=bkv_idx,
                )

            lax.fori_loop(0, num_bkv, compute_with_bkv, None, unroll=False)

            acc = acc_ref[...]
            l = broadcast_minor(l_ref[...], acc.shape)
            out = lax.div(acc, l) if q_dtype == jnp.float32 else (acc * pl.reciprocal(l, approx=True)).astype(q_dtype)

            bo_sem_idx = sem_ids_ref[2]
            sem_ids_ref[2] = lax.select(bo_sem_idx == 0, 1, 0)
            wait_send_bo(bo_sem_idx)

            bo_x2_ref.at[bo_sem_idx].bitcast(jnp.int32).reshape(
                bq_sz * num_q_heads_per_q_packing,
                lkv_dim,
            )[...] = pltpu.bitcast(out, jnp.int32)

            start_send_bo(seq_idx, bq_idx, bo_sem_idx)

        lax.fori_loop(0, num_bq, compute_with_bq, None, unroll=False)

    @pl.when(seq_idx == start_seq_idx)
    def prologue():
        """Initialise double buffers to zero and kick off the first query and KV block fetches."""
        start_fetch_bq(start_seq_idx, 0, 0)

        bkvc_x2_int32_ref = bkvc_x2_ref.bitcast(jnp.int32).reshape((2, -1, lkv_dim))
        bkvc_zeros = jnp.zeros(bkvc_x2_int32_ref.shape[1:], jnp.int32)
        bkpe_x2_int32_ref = bkpe_x2_ref.bitcast(jnp.int32).reshape((2, -1, r_dim))
        bkpe_zeros = jnp.zeros(bkpe_x2_int32_ref.shape[1:], jnp.int32)

        bkvc_x2_int32_ref[0] = bkvc_zeros
        bkpe_x2_int32_ref[0] = bkpe_zeros
        start_fetch_bkv(start_seq_idx, 0, 0)
        bkvc_x2_int32_ref[1] = bkvc_zeros
        bkpe_x2_int32_ref[1] = bkpe_zeros

    process()

    @pl.when(seq_idx == end_seq_idx - 1)
    def epilogue():
        """Drain all outstanding output and KV cache write-back DMAs."""
        for i in range(2):
            wait_send_bo(i)
            wait_update_kv_cache(i)


def prepare_q_inputs(q: jax.Array):
    """Pad and reshape query tensor into the packed layout expected by the kernel.

    Pads the head and dimension axes to TPU-friendly alignments and reshapes
    into (tokens, heads_per_packing, packing, head_dim).

    Args:
        q: Query tensor of shape (max_num_tokens, num_q_heads, head_dim).

    Returns:
        Packed query tensor of shape (max_num_tokens, num_q_heads // packing,
        packing, aligned_head_dim).
    """
    max_num_tokens, actual_num_q_heads, actual_head_dim = q.shape
    q_packing = get_dtype_packing(q.dtype)
    num_q_heads = align_to(actual_num_q_heads, q_packing)
    head_dim = align_to(actual_head_dim, 128)
    q = jnp.pad(
        q.reshape(max_num_tokens, actual_num_q_heads, actual_head_dim),
        (
            (0, 0),
            (0, num_q_heads - actual_num_q_heads),
            (0, head_dim - actual_head_dim),
        ),
        constant_values=0,
    ).reshape(
        max_num_tokens,
        num_q_heads // q_packing,
        q_packing,
        head_dim,
    )
    return q


def prepare_kv_inputs(kv: jax.Array):
    """Pad and reshape a 2-D KV tensor into the packed layout expected by the kernel.

    Pads the token axis to a multiple of the dtype packing factor and the
    dimension axis to a 128-aligned size, then reshapes into
    (tokens // packing, packing, aligned_head_dim).

    Args:
        kv: KV tensor of shape (max_num_tokens, head_dim).

    Returns:
        Packed KV tensor of shape (tokens // packing, packing, aligned_head_dim).
    """
    max_num_tokens, actual_head_dim = kv.shape
    kv_packing = get_dtype_packing(kv.dtype)
    if max_num_tokens % kv_packing != 0:
        pad = kv_packing - (max_num_tokens % kv_packing)
        kv = jnp.pad(kv, ((0, pad), (0, 0)), constant_values=0)

    head_dim = align_to(actual_head_dim, 128)
    kv = kv.reshape(-1, kv_packing, actual_head_dim)
    kv = jnp.pad(kv, ((0, 0), (0, 0), (0, head_dim - actual_head_dim)), constant_values=0)
    return kv


def prepare_outputs(out, actual_num_q_heads: int, actual_head_dim: int):
    """Unpack and slice the kernel output back to the original query dimensions.

    Args:
        out: Packed output from the kernel of shape
            (tokens, heads_per_packing, packing, aligned_head_dim).
        actual_num_q_heads: Original (unpadded) number of query heads.
        actual_head_dim: Original (unpadded) head dimension.

    Returns:
        Output tensor of shape (max_num_tokens, actual_num_q_heads,
        actual_head_dim).
    """
    max_num_tokens, num_q_heads_per_q_packing, q_packing, head_dim = out.shape
    return out.reshape(
        max_num_tokens,
        num_q_heads_per_q_packing * q_packing,
        head_dim,
    )[:, :actual_num_q_heads, :actual_head_dim]


@functools.partial(
    jax.jit,
    static_argnames=(
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "chunk_prefill_size",
        "num_kv_pages_per_block",
        "num_queries_per_block",
        "vmem_limit_bytes",
        "debug_mode",
    ),
    donate_argnames=("cache_kv",),
)
def _mla_ragged_paged_attention_v2_impl(
    ql_nope: jax.Array,
    q_pe: jax.Array,
    new_kv_c: jax.Array,
    new_k_pe: jax.Array,
    cache_kv: jax.Array,
    kv_lens: jax.Array,
    page_indices: jax.Array,
    cu_q_lens: jax.Array,
    distribution: jax.Array,
    *,
    sm_scale: float = 1.0,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """JIT-compiled orchestrator for MLA ragged paged attention v2 on TPU.

    Validates inputs, packs tensors into the sub-word layout, then dispatches
    up to three separate ``pl.pallas_call`` invocations — one each for the
    decode region, the prefill region, and the mixed region — as determined by
    ``distribution``.  Only the decode and mixed passes are actually invoked
    in the current implementation; the prefill region (indices
    ``[distribution[0], distribution[1])``) is not separately dispatched.

    The function is JIT-compiled via ``@functools.partial(jax.jit, ...)`` with
    a set of static argnames so that each unique combination of block-tiling
    parameters, scaling constants, and mode flags produces a distinct XLA
    computation.  The ``cache_kv`` argument is donated to avoid a copy.

    Args:
        ql_nope: Query nope tensor ``[max_tokens, num_q_heads, lkv_dim]``.
        q_pe: Query rope tensor ``[max_tokens, num_q_heads, r_dim]``.
        new_kv_c: New KV latent ``[max_tokens, lkv_dim]``; appended to cache.
        new_k_pe: New key rope ``[max_tokens, r_dim]``; appended to cache.
        cache_kv: Donated paged KV cache
            ``[total_pages, page_size // packing, packing, lkv_dim + r_dim]``.
        kv_lens: int32 ``[max_num_seqs]`` per-sequence KV lengths.
        page_indices: int32 ``[max_num_seqs * pages_per_seq]`` page table.
        cu_q_lens: int32 ``[max_num_seqs + 1]`` cumulative query lengths.
        distribution: int32 ``[3]`` array ``[decode_end, prefill_end,
            mixed_end]``.
        sm_scale: Softmax scaling factor (static).
        sliding_window: Optional sliding-window size (static).
        soft_cap: Optional logits soft-cap (static).
        mask_value: Padding value for masked attention logits (static).
        q_scale: Optional query scale multiplier (static).
        k_scale: Optional key scale multiplier (static).
        v_scale: Optional value scale multiplier (static).
        chunk_prefill_size: Reserved; not used in the current implementation
            (static).
        num_kv_pages_per_block: Number of KV pages per block tile.  May be a
            single int (broadcast to all three regions) or a three-tuple
            ``(decode, prefill, mixed)`` (static).
        num_queries_per_block: Number of query tokens per block tile.  Same
            broadcast semantics as ``num_kv_pages_per_block`` (static).
        vmem_limit_bytes: Optional VMEM budget passed to the Pallas compiler
            (static).
        debug_mode: Skip DMA calls and run in pure Python for debugging
            (static).

    Returns:
        A ``(output, updated_cache_kv)`` tuple where ``output`` has shape
        ``[max_tokens, actual_num_q_heads, actual_lkv_dim]`` and
        ``updated_cache_kv`` is the modified paged cache.

    Raises:
        ValueError: If ``num_kv_pages_per_block`` or ``num_queries_per_block``
            are ``None``, or if any shape/dtype constraint is violated.
    """
    if num_kv_pages_per_block is None or num_queries_per_block is None:
        raise ValueError("num_kv_pages_per_block and num_queries_per_block must be specified.")

    if isinstance(num_kv_pages_per_block, int):
        num_kv_pages_per_blocks = [num_kv_pages_per_block for _ in range(3)]
    else:
        num_kv_pages_per_blocks = num_kv_pages_per_block

    if isinstance(num_queries_per_block, int):
        num_queries_per_blocks = [num_queries_per_block for _ in range(3)]
    else:
        num_queries_per_blocks = num_queries_per_block

    static_validate_inputs(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        distribution,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        chunk_prefill_size=chunk_prefill_size,
        num_kv_pages_per_blocks=num_kv_pages_per_blocks,
        num_queries_per_blocks=num_queries_per_blocks,
        vmem_limit_bytes=vmem_limit_bytes,
        debug_mode=debug_mode,
    )

    _, actual_num_q_heads, actual_lkv_dim = ql_nope.shape

    ql_nope = prepare_q_inputs(ql_nope)
    q_pe = prepare_q_inputs(q_pe)
    new_kv_c = prepare_kv_inputs(new_kv_c)
    new_k_pe = prepare_kv_inputs(new_k_pe)
    lkv_dim = new_kv_c.shape[-1]
    r_dim = new_k_pe.shape[-1]

    _, page_size_per_kv_packing, kv_packing, _ = cache_kv.shape
    page_size = page_size_per_kv_packing * kv_packing
    _, num_q_heads_per_q_packing, q_packing, _ = ql_nope.shape
    max_num_seqs = kv_lens.shape[0]
    num_page_indices = page_indices.shape[0]
    assert num_page_indices % max_num_seqs == 0
    num_q_heads = num_q_heads_per_q_packing * q_packing

    def run_mla_kernel(
        ql_nope: jax.Array,
        q_pe: jax.Array,
        new_kv_c: jax.Array,
        new_k_pe: jax.Array,
        cache_kv: jax.Array,
        kv_lens: jax.Array,
        page_indices: jax.Array,
        cu_q_lens: jax.Array,
        start_seq_idx: jax.Array,
        end_seq_idx: jax.Array,
        static_q_len: int | None,
        num_kv_pages_per_block: int,
        num_queries_per_block: int,
        case: MlaCase = MlaCase.MIXED,
    ):
        """Build and invoke the Pallas kernel for a subset of sequences.

        Allocates VMEM scratch buffers and double-buffer semaphores, then
        calls ``pl.pallas_call`` with the appropriate grid, block specs, and
        compiler parameters for the given MLA case (decode, prefill, or
        mixed).

        Args:
            ql_nope: Packed query nope tensor.
            q_pe: Packed query rope tensor.
            new_kv_c: Packed new KV latent tensor.
            new_k_pe: Packed new key rope tensor.
            cache_kv: Paged KV cache tensor.
            kv_lens: Per-sequence KV lengths.
            page_indices: Flattened page table indices.
            cu_q_lens: Cumulative query lengths (size max_num_seqs + 1).
            start_seq_idx: First sequence index to process.
            end_seq_idx: One-past-last sequence index to process.
            static_q_len: If known at trace time, the fixed query length
                per sequence (e.g. 1 for decode); None for variable length.
            num_kv_pages_per_block: Number of KV pages per block tile.
            num_queries_per_block: Number of query tokens per block tile.
            case: Which MLA scheduling case to use.

        Returns:
            A (output, updated_cache_kv) tuple.
        """
        bkv_p = num_kv_pages_per_block
        bq_sz = num_queries_per_block
        bkv_sz_per_kv_packing = bkv_p * page_size_per_kv_packing
        bkv_buf_sz_per_kv_packing = bkv_sz_per_kv_packing + 2
        grid = (end_seq_idx - start_seq_idx,)

        in_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ]

        out_specs = [
            pl.BlockSpec(memory_space=pltpu.HBM),
            pl.BlockSpec(memory_space=pltpu.HBM),
        ]

        bkvc_double_buf = pltpu.VMEM(
            (2, bkv_buf_sz_per_kv_packing, kv_packing, lkv_dim),
            cache_kv.dtype,
        )
        bkpe_double_buf = pltpu.VMEM(
            (2, bkv_buf_sz_per_kv_packing, kv_packing, r_dim),
            cache_kv.dtype,
        )
        bq_nope_double_buf = pltpu.VMEM(
            (2, bq_sz, num_q_heads_per_q_packing, q_packing, lkv_dim),
            ql_nope.dtype,
        )
        bq_rope_double_buf = pltpu.VMEM(
            (2, bq_sz, num_q_heads_per_q_packing, q_packing, r_dim),
            q_pe.dtype,
        )
        bo_double_buf = bq_nope_double_buf

        l_scratch = pltpu.VMEM((bq_sz * num_q_heads, 128), jnp.float32)
        m_scratch = l_scratch
        acc_scratch = pltpu.VMEM((bq_sz * num_q_heads, lkv_dim), jnp.float32)

        scratch_shapes = [
            bkvc_double_buf,
            bkpe_double_buf,
            bq_nope_double_buf,
            bq_rope_double_buf,
            bo_double_buf,
            pltpu.SemaphoreType.DMA((4, 2)),
            l_scratch,
            m_scratch,
            acc_scratch,
        ]

        scalar_prefetches = (
            kv_lens,
            page_indices,
            cu_q_lens,
            jnp.array([start_seq_idx, end_seq_idx], jnp.int32),
            jnp.zeros((3,), jnp.int32),
            jnp.full((4,), -1, jnp.int32),
            jnp.full((6,), -1, jnp.int32),
        )

        scope_name = f"MLA-{case.symbol}-bq_{bq_sz}-bkvp_{bkv_p}-p_{page_size}"
        kernel = jax.named_scope(scope_name)(
            pl.pallas_call(
                functools.partial(
                    _mla_ragged_paged_attention_kernel,
                    sm_scale=sm_scale,
                    sliding_window=sliding_window,
                    soft_cap=soft_cap,
                    mask_value=mask_value,
                    q_scale=q_scale,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    static_q_len=static_q_len,
                    bq_sz=bq_sz,
                    bkv_p=bkv_p,
                    debug_mode=debug_mode,
                ),
                grid_spec=pltpu.PrefetchScalarGridSpec(
                    num_scalar_prefetch=len(scalar_prefetches),
                    in_specs=in_specs,
                    out_specs=out_specs,
                    grid=grid,
                    scratch_shapes=scratch_shapes,
                ),
                compiler_params=pltpu.CompilerParams(
                    dimension_semantics=("arbitrary",),
                    vmem_limit_bytes=vmem_limit_bytes,
                ),
                out_shape=[
                    jax.ShapeDtypeStruct(shape=ql_nope.shape, dtype=ql_nope.dtype),
                    jax.ShapeDtypeStruct(shape=cache_kv.shape, dtype=cache_kv.dtype),
                ],
                input_output_aliases={
                    7: 0,
                    11: 1,
                },
                name=scope_name,
            )
        )
        return kernel(
            *scalar_prefetches,
            ql_nope,
            q_pe,
            new_kv_c,
            new_k_pe,
            cache_kv,
        )

    ql_nope, updated_kv = run_mla_kernel(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        cache_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_kv_pages_per_block=num_kv_pages_per_blocks[0],
        num_queries_per_block=num_queries_per_blocks[0],
        start_seq_idx=jnp.array(0, jnp.int32),
        end_seq_idx=distribution[0],
        static_q_len=1,
        case=MlaCase.DECODE,
    )

    ql_nope, updated_kv = run_mla_kernel(
        ql_nope,
        q_pe,
        new_kv_c,
        new_k_pe,
        updated_kv,
        kv_lens,
        page_indices,
        cu_q_lens,
        num_kv_pages_per_block=num_kv_pages_per_blocks[2],
        num_queries_per_block=num_queries_per_blocks[2],
        start_seq_idx=distribution[1],
        end_seq_idx=distribution[2],
        static_q_len=None,
        case=MlaCase.MIXED,
    )
    output = prepare_outputs(ql_nope, actual_num_q_heads, actual_lkv_dim)
    return output, updated_kv


def mla_ragged_paged_attention_v2(
    queries_nope: jax.Array,
    queries_pe: jax.Array,
    keys_values: jax.Array,
    keys_pe: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    *,
    softmax_scale: float = 1.0,
    sliding_window: int | None = None,
    logits_soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    chunk_prefill_size: int | None = None,
    num_kv_pages_per_block: tuple[int, int, int] | int | None = None,
    num_queries_per_block: tuple[int, int, int] | int | None = None,
    vmem_limit_bytes: int | None = None,
    debug_mode: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Public entry point for MLA ragged paged attention v2 on TPU.

    Thin adapter that renames arguments from ejkernel's naming conventions
    (``queries_nope``, ``softmax_scale``, ``logits_soft_cap``,
    ``block_tables``, ``query_start_loc``) to the upstream internal names
    used by ``_mla_ragged_paged_attention_v2_impl``.  All behaviour is
    delegated without modification.

    Args:
        queries_nope: Query nope tensor ``[max_tokens, num_q_heads, lkv_dim]``.
        queries_pe: Query rope tensor ``[max_tokens, num_q_heads, r_dim]``.
        keys_values: New KV latent ``[max_tokens, lkv_dim]``.
        keys_pe: New key rope ``[max_tokens, r_dim]``.
        kv_cache: Donated paged KV cache
            ``[total_pages, page_size // packing, packing, lkv_dim + r_dim]``.
        kv_lens: int32 ``[max_num_seqs]`` per-sequence KV lengths.
        block_tables: int32 ``[max_num_seqs * pages_per_seq]`` page table.
        query_start_loc: int32 ``[max_num_seqs + 1]`` cumulative query lengths.
        distribution: int32 ``[3]`` array ``[decode_end, prefill_end,
            mixed_end]``.
        softmax_scale: Softmax scaling factor.
        sliding_window: Optional sliding-window attention size.
        logits_soft_cap: Optional soft cap for attention logits.
        mask_value: Padding value for masked attention logits.
        q_scale: Optional query scale multiplier.
        k_scale: Optional key scale multiplier.
        v_scale: Optional value scale multiplier.
        chunk_prefill_size: Reserved; not used in the current implementation.
        num_kv_pages_per_block: Number of KV pages per block tile.
        num_queries_per_block: Number of query tokens per block tile.
        vmem_limit_bytes: Optional VMEM budget for the Pallas compiler.
        debug_mode: Skip DMA calls for debugging.

    Returns:
        A ``(output, updated_cache_kv)`` tuple.
    """
    return _mla_ragged_paged_attention_v2_impl(
        ql_nope=queries_nope,
        q_pe=queries_pe,
        new_kv_c=keys_values,
        new_k_pe=keys_pe,
        cache_kv=kv_cache,
        kv_lens=kv_lens,
        page_indices=block_tables,
        cu_q_lens=query_start_loc,
        distribution=distribution,
        sm_scale=softmax_scale,
        sliding_window=sliding_window,
        soft_cap=logits_soft_cap,
        mask_value=mask_value,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        chunk_prefill_size=chunk_prefill_size,
        num_kv_pages_per_block=num_kv_pages_per_block,
        num_queries_per_block=num_queries_per_block,
        vmem_limit_bytes=vmem_limit_bytes,
        debug_mode=debug_mode,
    )


__all__ = (
    "DEFAULT_MASK_VALUE",
    "DEFAULT_VMEM_LIMIT_BYTES",
    "get_kv_cache_shape",
    "mla_ragged_paged_attention_v2",
)
