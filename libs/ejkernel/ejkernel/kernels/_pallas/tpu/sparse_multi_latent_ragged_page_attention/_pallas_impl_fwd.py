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

"""TPU Pallas sparse MLA ragged paged attention over per-token selected chunks.

Two stages:

1. XLA: write the step's latent tokens into the donated cache (the same scatter as the XLA
   reference, in the native packed layout) and turn each token's selection into compact
   metadata: the number of attendable chunks, the token position, the first token of its last
   attendable chunk and, for every attendable chunk (compacted to the front, ascending), its first
   packed word. Chunks that are entirely causal-, length- or window-masked are dropped here, so the
   kernel never fetches them.
2. Pallas: one program per query token walks its chunks in groups of ``G`` (``G * chunk_size``
   keys per step) through a ring of ``_NUM_BUFFERS`` VMEM tiles (the next groups are in flight
   while one is computed) and runs the MLA online-softmax (nope + rope logits, value = latent) on
   each tile. Chunk words are translated to cache rows inside the kernel through the token's
   block-table row, held in SMEM, so no per-chunk page lookup runs in XLA (above an SMEM budget the
   translation falls back to XLA). All copies of a group signal one semaphore, waited on once per
   group. A token whose attendable chunks are exactly ``0 .. n - 1`` (its whole visible context is
   selected, as for every context up to the indexer's top-k) fetches each group with one DMA when
   groups tile pages; other tokens issue one DMA per chunk.

Masking: valid chunks are unique (the selection contract of every backend) and sorted ascending
here, so only a token's last chunk can be partially causal-masked. With word-aligned chunks and no
sliding window the kernel therefore masks with one scalar limit per token; otherwise the metadata
also carries every chunk's first token and the kernel masks per key. The scalar limit and the
whole-group copies both rely on unique selections: a repeated chunk would not only be counted twice
(as in the XLA reference) but could also expose keys past the query position. Masked key slots are
still read and weighted by zero, so cache slots must hold finite values.

The cost per token is ``O(num_selected * max(chunk_size, kv_packing))`` HBM reads, independent of
context length. A chunk must lie inside one page (``page_size % chunk_size == 0``) and must not
straddle a packed word: either ``chunk_size % kv_packing == 0``, or the chunk is narrower than a
word (``kv_packing % chunk_size == 0``). In the latter case the whole containing word is fetched
and the lanes of neighbouring tokens are masked (see :func:`supports`).
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ...._xla.sparse_multi_latent_ragged_page_attention._xla_impl_fwd import (
    DEFAULT_MASK_VALUE,
    align_to,
    get_dtype_packing,
    token_layout,
    write_tokens_to_cache,
)

_FAR = 1 << 30  # token index that is never attendable (masks padding slots)
_NUM_BUFFERS = 3  # VMEM ring depth: up to two groups in flight while one is computed
_WORD_COL = 3  # metadata row: [n_valid, pos, last_start, words[0:Wp], (starts[0:Wp])]
# In-kernel page translation keeps one block-table row per token in SMEM and scalar-prefetches the
# token -> sequence map; beyond these sizes the translation runs in XLA instead (TPU SMEM is ~1 MiB).
_MAX_BLOCK_TABLE_ROW_BYTES = 64 * 1024
_MAX_PREFETCH_BYTES = 256 * 1024


def supports(kv_cache_shape: tuple[int, ...], chunk_size: int) -> bool:
    """Whether the Pallas kernel handles this cache layout and chunk size.

    Args:
        kv_cache_shape: ``[P, page_size/pack, pack, D]``.
        chunk_size: Tokens per selectable chunk.

    Returns:
        True when each chunk is word-aligned and contained in a single page.
    """
    _, page_size_per_pack, pack, _ = kv_cache_shape
    page_size = page_size_per_pack * pack
    # A chunk narrower than a packed word is fetched as its whole containing word and the
    # other lanes are masked, so it only has to stay inside one word.
    word_aligned = chunk_size % pack == 0 if chunk_size >= pack else pack % chunk_size == 0
    return word_aligned and page_size % chunk_size == 0 and chunk_size <= 128


def _fetch_span(chunk_size: int, kv_packing: int) -> int:
    """Tokens fetched per chunk: the chunk itself, or its containing packed word if narrower."""
    return max(int(chunk_size), int(kv_packing))


def _group_size(span: int, width: int, num_chunks_per_block: int | None) -> int:
    """Chunks fetched per kernel step (default: 128 fetched keys per step)."""
    g = int(num_chunks_per_block) if num_chunks_per_block else max(1, 128 // span)
    return max(1, min(g, width))


def _smem_len(n: int) -> int:
    """Length of a 1-D SMEM block: Mosaic needs a power of two >= 128."""
    return max(128, 1 << (int(n) - 1).bit_length())


def _needs_starts(chunk_size: int, kv_packing: int, sliding_window: int | None) -> bool:
    """Whether the kernel masks per key from each chunk's first token instead of one limit per token.

    Needed for a sliding window and for chunks narrower than a packed word (whose fetched word holds
    neighbouring, unselected tokens).
    """
    return sliding_window is not None or _fetch_span(chunk_size, kv_packing) != int(chunk_size)


def _translation_fits_smem(total_tokens: int, pages_per_seq: int) -> bool:
    """Whether the per-token block-table row and the token -> sequence map fit the SMEM budget."""
    return 4 * _smem_len(pages_per_seq) <= _MAX_BLOCK_TABLE_ROW_BYTES and 4 * int(total_tokens) <= _MAX_PREFETCH_BYTES


def build_metadata(
    kv_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    selected: jax.Array,
    *,
    total_tokens: int,
    chunk_size: int,
    page_size: int,
    kv_packing: int,
    group: int,
    sliding_window: int | None,
    translate_pages: bool,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, int]:
    """Per-token kernel metadata.

    Args:
        kv_lens: ``[S]`` context lengths including the current step.
        block_tables: ``[S * pages_per_seq]`` flat page table.
        query_start_loc: ``[S + 1]`` cumulative query counts.
        distribution: ``[3]``; ``distribution[2]`` is the number of live sequences.
        selected: ``[T, W]`` chunk ids, ``-1``-padded, unique per row.
        total_tokens: Packed query token count ``T``.
        chunk_size: Tokens per chunk.
        page_size: Tokens per cache page.
        kv_packing: Tokens per packed cache word.
        group: Chunks per kernel step; the chunk columns are padded to a multiple of it.
        sliding_window: Optional local window.
        translate_pages: Emit chunk words relative to the sequence (the kernel translates them
            through the block table) instead of physical cache word-rows.

    Returns:
        ``(meta [T, M] int32, seq [T], valid [T], pos [T], padded_width)``. Row ``t`` of ``meta`` is
        ``[n_valid, pos, last_start, words[0:Wp], (starts[0:Wp]), 0...]``: the number of attendable
        chunks, the token position, the first token of its last attendable chunk, then for every
        attendable chunk (compacted to the front, ascending) its first packed word, as the word index
        within the sequence (``translate_pages``) or as the physical cache word-row. Padding slots
        repeat the first chunk. ``starts`` (each chunk's first token, padding slots unattendable) is
        present only when :func:`_needs_starts`.
    """
    max_num_seqs = int(kv_lens.shape[0])
    pages_per_seq = int(block_tables.shape[0]) // max_num_seqs
    width = int(selected.shape[1])
    seq, pos, valid = token_layout(kv_lens, query_start_loc, distribution, total_tokens)
    kv_len = kv_lens[seq]

    start = selected * chunk_size
    ok = (selected >= 0) & valid[:, None] & (start <= pos[:, None]) & (start < kv_len[:, None])
    if sliding_window is not None:
        ok = ok & (start + chunk_size - 1 > pos[:, None] - int(sliding_window))
    key = jnp.sort(jnp.where(ok, start, _FAR), axis=1)
    n_valid = jnp.sum(ok, axis=1).astype(jnp.int32)

    padded_width = -(-width // group) * group
    key = jnp.pad(key, ((0, 0), (0, padded_width - width)), constant_values=_FAR)
    live = key < _FAR
    first = jnp.where(live[:, :1], key[:, :1], 0)
    safe = jnp.where(live, key, first)
    last_start = jnp.max(safe, axis=1)
    words_per_page = page_size // kv_packing
    if translate_pages:
        words = jnp.minimum(safe // kv_packing, pages_per_seq * words_per_page - 1)
    else:
        logical_page = jnp.clip(safe // page_size, 0, pages_per_seq - 1)
        physical_page = block_tables[seq[:, None] * pages_per_seq + logical_page]
        words = physical_page * words_per_page + (safe % page_size) // kv_packing

    cols = [n_valid[:, None], pos[:, None], last_start[:, None], words]
    if _needs_starts(chunk_size, kv_packing, sliding_window):
        cols.append(key)
    meta = jnp.concatenate(cols, axis=1).astype(jnp.int32)
    # Mosaic 1-D SMEM blocks must be a power of two >= 128 long.
    m = int(meta.shape[1])
    meta = jnp.pad(meta, ((0, 0), (0, _smem_len(m) - m)))
    return meta, seq, valid, pos, padded_width


def _broadcast_lanes(src: jax.Array, shape: tuple[int, int]) -> jax.Array:
    """Tile a ``[rows, 128]`` value along lanes to ``shape`` (as in the MLA v2 kernel)."""
    if src.shape == shape:
        return src
    reps = align_to(shape[-1], src.shape[-1]) // src.shape[-1]
    return jnp.concatenate([src] * reps, axis=-1)[..., : shape[-1]]


def _sparse_mla_kernel(
    *refs,
    translate_pages: bool,
    has_pe: bool,
    with_starts: bool,
    chunk_size: int,
    kv_packing: int,
    group: int,
    padded_width: int,
    nope_dim: int,
    sm_scale: float,
    sliding_window: int | None,
    soft_cap: float | None,
    mask_value: float,
    q_scale: float | None,
    k_scale: float | None,
    v_scale: float | None,
):
    """One query token: online softmax over its attendable chunks, ``group`` chunks per step.

    Refs: ``[seq], meta, [block-table row], q_nope, [q_pe], cache (HBM), out, kv ring, DMA semaphores, m, l, acc``;
    the bracketed refs exist only with in-kernel page translation (``seq`` is the scalar-prefetched token ->
    sequence map, consumed by the block-table row's index map) or a rope part.
    """
    refs = list(refs)
    if translate_pages:
        refs.pop(0)
    meta_ref = refs.pop(0)
    bt_ref = refs.pop(0) if translate_pages else None
    q_nope_ref = refs.pop(0)
    q_pe_ref = refs.pop(0) if has_pe else None
    cache_hbm_ref, o_ref, kv_buf, sems, m_ref, l_ref, acc_ref = refs

    num_pages, words_per_page, _, cache_dim = cache_hbm_ref.shape
    flat_cache = cache_hbm_ref.reshape(num_pages * words_per_page, kv_packing, cache_dim)
    span = _fetch_span(chunk_size, kv_packing)
    chunk_words = span // kv_packing
    group_words = group * chunk_words
    n_keys = group * span
    num_buffers = int(kv_buf.shape[0])
    kv_dtype = cache_hbm_ref.dtype
    start_col = _WORD_COL + padded_width

    n_valid = meta_ref[0]
    pos = meta_ref[1]
    last_start = meta_ref[2]
    n_groups = (n_valid + group - 1) // group

    m_ref[...] = jnp.full(m_ref.shape, -jnp.inf, jnp.float32)
    l_ref[...] = jnp.zeros(l_ref.shape, jnp.float32)
    acc_ref[...] = jnp.zeros(acc_ref.shape, jnp.float32)

    page_shift = int(words_per_page).bit_length() - 1
    pow2_page = words_per_page == 1 << page_shift

    def cache_row(word):
        """Physical cache word-row of a metadata word (translated here when the kernel owns paging)."""
        if bt_ref is None:
            return word
        if pow2_page:
            return (bt_ref[word >> page_shift] << page_shift) | (word & (words_per_page - 1))
        return bt_ref[word // words_per_page] * words_per_page + word % words_per_page

    def start_chunks(grp, slot):
        """One copy per chunk of group ``grp`` into ring slot ``slot``, issued as straight-line code.

        A rolled ``fori_loop`` over the copies made their issue dominate the kernel.
        """
        for g in range(group):
            row = cache_row(meta_ref[_WORD_COL + grp * group + g])
            pltpu.make_async_copy(
                flat_cache.at[pl.ds(row, chunk_words)],
                kv_buf.at[slot, pl.ds(g * chunk_words, chunk_words)],
                sems.at[slot],
            ).start()

    def start_dense(grp, slot):
        """Group ``grp`` of a token attending chunks ``0 .. n - 1``: consecutive words of one page, one copy."""
        row = cache_row(meta_ref[_WORD_COL + grp * group])
        pltpu.make_async_copy(flat_cache.at[pl.ds(row, group_words)], kv_buf.at[slot], sems.at[slot]).start()

    def wait_group(slot):
        # Every copy of a group signals the slot's semaphore and together they fill exactly one ring slot.
        pltpu.make_async_copy(kv_buf.at[slot], kv_buf.at[slot], sems.at[slot]).wait()

    q_nope = q_nope_ref[...]
    q_pe = q_pe_ref[...] if has_pe else None
    # f32 inputs must not silently drop to single-pass bf16 MXU (TPU DEFAULT); bf16 stays native.
    precision = lax.Precision.HIGHEST if q_nope.dtype == jnp.float32 else None
    # Without per-chunk starts: unique ascending chunks are all fully visible but the last, so key slot ``k``
    # (chunk ``k // span`` of the token's list) is visible iff ``k <= limit``; padding slots lie beyond it.
    limit = (n_valid - 1) * span + jnp.minimum(pos - last_start, span - 1)

    def compute(grp, slot):
        words = kv_buf.bitcast(jnp.uint32).at[slot].reshape(group_words, cache_dim)
        kv = pltpu.bitcast(words[...], kv_dtype).reshape(n_keys, cache_dim)
        if kv.dtype != q_nope.dtype:
            kv = kv.astype(q_nope.dtype)
        kv_c = kv[:, :nope_dim]

        s = jnp.einsum("hd,nd->hn", q_nope, kv_c, preferred_element_type=jnp.float32, precision=precision)
        if has_pe:
            s = s + jnp.einsum(
                "hd,nd->hn", q_pe, kv[:, nope_dim:], preferred_element_type=jnp.float32, precision=precision
            )
        s = s * sm_scale
        if k_scale is not None:
            s = s * k_scale
        if q_scale is not None:
            s = s * q_scale
        if soft_cap is not None:
            s = soft_cap * jnp.tanh(s / soft_cap)

        col = lax.broadcasted_iota(jnp.int32, s.shape, 1)
        if with_starts:
            slot_of_col = col // span
            chunk_start = jnp.full(s.shape, _FAR, jnp.int32)
            base = start_col + grp * group
            for g in range(group):
                chunk_start = jnp.where(slot_of_col == g, meta_ref[base + g], chunk_start)
            if span == chunk_size:
                k_pos = chunk_start + col % span
                ok = k_pos <= pos
            else:
                # The fetched word starts at the chunk's word boundary; lanes outside the chunk
                # belong to neighbouring (unselected) tokens and are masked.
                k_pos = (chunk_start // kv_packing) * kv_packing + col % span
                ok = (k_pos <= pos) & (k_pos >= chunk_start) & (k_pos < chunk_start + chunk_size)
            if sliding_window is not None:
                ok = jnp.logical_and(ok, k_pos > pos - sliding_window)
        else:
            ok = col <= limit - grp * n_keys
        s = jnp.where(ok, s, mask_value)

        m_prev = m_ref[...]
        m_curr = jnp.maximum(m_prev, jnp.max(s, axis=1, keepdims=True))
        m_ref[...] = m_curr
        p = jnp.where(ok, jnp.exp(s - _broadcast_lanes(m_curr, s.shape)), 0.0)
        pv = jnp.einsum(
            "hn,nd->hd", p, kv_c.astype(jnp.float32), preferred_element_type=jnp.float32, precision=precision
        )
        alpha = jnp.exp(m_prev - m_curr)
        l_ref[...] = alpha * l_ref[...] + jnp.sum(p, axis=1, keepdims=True)
        acc_ref[...] = _broadcast_lanes(alpha, acc_ref.shape) * acc_ref[...] + pv

    def attend(start_group):
        for k in range(num_buffers - 1):

            @pl.when(k < n_groups)
            def _(k=k):
                start_group(k, k)

        def body(grp, carry):
            slot = grp % num_buffers
            ahead = grp + num_buffers - 1

            @pl.when(ahead < n_groups)
            def _():
                start_group(ahead, ahead % num_buffers)

            wait_group(slot)
            compute(grp, slot)
            return carry

        lax.fori_loop(0, n_groups, body, 0)

    if with_starts or words_per_page % group_words:
        attend(start_chunks)
    else:
        # ``n_valid`` unique non-negative chunk ids whose largest is ``n_valid - 1`` are exactly ``0 .. n_valid - 1``;
        # groups then start on page-tiling word boundaries, so each lies in one page. Decided once per token,
        # so neither loop branches per group.
        dense = last_start == (n_valid - 1) * chunk_size

        @pl.when(dense)
        def _():
            attend(start_dense)

        @pl.when(jnp.logical_not(dense))
        def _():
            attend(start_chunks)

    l_val = _broadcast_lanes(l_ref[...], acc_ref.shape)
    out = jnp.where(l_val > 0, acc_ref[...] / jnp.maximum(l_val, 1e-30), 0.0)
    if v_scale is not None:
        out = out * v_scale
    o_ref[...] = out.astype(o_ref.dtype)


@functools.partial(
    jax.jit,
    static_argnames=(
        "chunk_size",
        "sm_scale",
        "sliding_window",
        "soft_cap",
        "mask_value",
        "q_scale",
        "k_scale",
        "v_scale",
        "num_chunks_per_block",
        "vmem_limit_bytes",
        "translate_pages",
    ),
    donate_argnames=("kv_cache",),
)
def sparse_mla_ragged_paged_attention_pallas(
    queries_nope: jax.Array,
    queries_pe: jax.Array,
    keys_values: jax.Array,
    keys_pe: jax.Array,
    kv_cache: jax.Array,
    kv_lens: jax.Array,
    block_tables: jax.Array,
    query_start_loc: jax.Array,
    distribution: jax.Array,
    selected: jax.Array,
    *,
    chunk_size: int,
    sm_scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
    mask_value: float | None = DEFAULT_MASK_VALUE,
    q_scale: float | None = None,
    k_scale: float | None = None,
    v_scale: float | None = None,
    num_chunks_per_block: int | None = None,
    vmem_limit_bytes: int | None = None,
    translate_pages: bool | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Write new tokens into the cache, then run the sparse Pallas attention kernel.

    Arguments and semantics match the XLA reference
    (``_xla.sparse_multi_latent_ragged_page_attention``). Requires :func:`supports`.

    Args:
        translate_pages: Translate chunk pages inside the kernel (``True``) or in XLA (``False``);
            ``None`` picks the kernel whenever its SMEM budget allows. Both give identical results.

    Returns:
        ``(outputs [T, H, L], updated_kv_cache)``.
    """
    if mask_value is None:
        mask_value = DEFAULT_MASK_VALUE
    total_tokens, num_heads, nope_actual = queries_nope.shape
    pe_actual = int(queries_pe.shape[-1])
    nope_dim = align_to(nope_actual, 128)
    pe_dim = align_to(pe_actual, 128)
    _, words_per_page, kv_packing, cache_dim = kv_cache.shape
    assert kv_packing == get_dtype_packing(kv_cache.dtype)
    assert cache_dim == nope_dim + pe_dim, (cache_dim, nope_dim, pe_dim)
    page_size = words_per_page * kv_packing
    max_num_seqs = int(kv_lens.shape[0])
    pages_per_seq = int(block_tables.shape[0]) // max_num_seqs
    width = int(selected.shape[1])
    span = _fetch_span(chunk_size, kv_packing)
    group = _group_size(span, width, num_chunks_per_block)
    has_pe = pe_dim > 0
    if translate_pages is None:
        translate_pages = _translation_fits_smem(int(total_tokens), pages_per_seq)
    translate_pages = bool(translate_pages)
    with_starts = _needs_starts(chunk_size, kv_packing, sliding_window)

    meta, seq, valid, pos, padded_width = build_metadata(
        kv_lens,
        block_tables,
        query_start_loc,
        distribution,
        selected,
        total_tokens=int(total_tokens),
        chunk_size=chunk_size,
        page_size=page_size,
        kv_packing=kv_packing,
        group=group,
        sliding_window=sliding_window,
        translate_pages=translate_pages,
    )
    kv_cache = write_tokens_to_cache(
        kv_cache, keys_values, keys_pe, block_tables, seq, pos, valid, pages_per_seq=pages_per_seq
    )

    q_nope = jnp.pad(queries_nope, ((0, 0), (0, 0), (0, nope_dim - nope_actual)))
    q_pe = jnp.pad(queries_pe, ((0, 0), (0, 0), (0, pe_dim - pe_actual))).astype(q_nope.dtype)
    meta_cols = int(meta.shape[1])

    # 1-D SMEM block per token: Mosaic needs the minor block dims to be (8, 128)-aligned or full,
    # which a squeezed leading dim of a 2-D SMEM array is not. Index maps take ``*_`` so they also
    # accept the scalar-prefetched sequence map.
    in_specs = [pl.BlockSpec((meta_cols,), lambda t, *_: (t,), memory_space=pltpu.SMEM)]
    operands = [meta.reshape(-1)]
    if translate_pages:
        # The token's block-table row; consecutive tokens of one sequence reuse the fetched block.
        bt_block = _smem_len(pages_per_seq)
        bt = jnp.asarray(block_tables, jnp.int32).reshape(max_num_seqs, pages_per_seq)
        if bt_block != pages_per_seq:
            bt = jnp.pad(bt, ((0, 0), (0, bt_block - pages_per_seq)))
        in_specs.append(pl.BlockSpec((bt_block,), lambda t, seq_ref: (seq_ref[t],), memory_space=pltpu.SMEM))
        operands.append(bt.reshape(-1))
    in_specs.append(pl.BlockSpec((None, num_heads, nope_dim), lambda t, *_: (t, 0, 0)))
    operands.append(q_nope)
    if has_pe:
        in_specs.append(pl.BlockSpec((None, num_heads, pe_dim), lambda t, *_: (t, 0, 0)))
        operands.append(q_pe)
    in_specs.append(pl.BlockSpec(memory_space=pltpu.HBM))
    operands.append(kv_cache)

    group_words = group * (span // kv_packing)
    scratch_shapes = [
        pltpu.VMEM((_NUM_BUFFERS, group_words, kv_packing, cache_dim), kv_cache.dtype),
        pltpu.SemaphoreType.DMA((_NUM_BUFFERS,)),
        pltpu.VMEM((num_heads, 128), jnp.float32),
        pltpu.VMEM((num_heads, 128), jnp.float32),
        pltpu.VMEM((num_heads, nope_dim), jnp.float32),
    ]
    kernel = functools.partial(
        _sparse_mla_kernel,
        translate_pages=translate_pages,
        has_pe=has_pe,
        with_starts=with_starts,
        chunk_size=int(chunk_size),
        kv_packing=int(kv_packing),
        group=int(group),
        padded_width=int(padded_width),
        nope_dim=int(nope_dim),
        sm_scale=float(sm_scale),
        sliding_window=sliding_window,
        soft_cap=soft_cap,
        mask_value=float(mask_value),
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )
    out_spec = pl.BlockSpec((None, num_heads, nope_dim), lambda t, *_: (t, 0, 0))
    compiler_params = pltpu.CompilerParams(dimension_semantics=("parallel",), vmem_limit_bytes=vmem_limit_bytes)
    out_shape = jax.ShapeDtypeStruct((int(total_tokens), num_heads, nope_dim), queries_nope.dtype)
    name = f"SPARSE-MLA-c{chunk_size}-g{group}-p{page_size}"
    if translate_pages:
        out = pl.pallas_call(
            kernel,
            grid_spec=pltpu.PrefetchScalarGridSpec(
                num_scalar_prefetch=1,
                grid=(int(total_tokens),),
                in_specs=in_specs,
                out_specs=out_spec,
                scratch_shapes=scratch_shapes,
            ),
            out_shape=out_shape,
            compiler_params=compiler_params,
            name=name,
        )(seq, *operands)
    else:
        out = pl.pallas_call(
            kernel,
            grid=(int(total_tokens),),
            in_specs=in_specs,
            out_specs=out_spec,
            out_shape=out_shape,
            scratch_shapes=scratch_shapes,
            compiler_params=compiler_params,
            name=name,
        )(*operands)
    return out[:, :, :nope_actual], kv_cache


__all__ = ("build_metadata", "sparse_mla_ragged_paged_attention_pallas", "supports")
