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

"""Fused paged DSA pool scoring + exact top-k on TPU (one pass, no serial merge loop).

For every query row ``t`` (sequence ``seq[t]``, ``n_pool[t]`` scorable pools):

    score[t, p] = sum_h w[t, h] * act(scale * <q[t, h], pool_key[seq, p]>)   for p < n_pool[t]

and the row keeps the ``k`` best pools (ties -> lowest pool id, as ``jax.lax.top_k``).

Pool summaries live in pages ``[P, pools_per_page, D]`` addressed by the KV page table, so
each row streams only its own pages up to ``n_pool[t]`` (cost proportional to the real
context, never ``max_model_len``).

Blocks. A program takes a block of consecutive rows, a multiple of 8 so every key tile is
``(8, 128)``: 8 rows for small calls (decode), doubled up to ``_MAX_BLOCK_ROWS`` while the grid
keeps at least ``_MIN_BLOCKS`` programs and the block's VMEM stays within
``_BLOCK_VMEM_BUDGET_BYTES`` (long page tables keep 8 rows). Programs alternate between the two halves of the grid
(even blocks first, then odd ones), which a megacore chip runs on its two TensorCores, so rows
whose context grows along the batch split evenly between the cores. A call of 2 to 8 rows (decode
with a few rows per device) also runs two programs: its rows are dealt alternately to two blocks,
so both TensorCores score real rows (two decode rows then take one block each).

Scoring. When every scorable row of a block reads one sequence (prefill: consecutive tokens of
one prompt, or a block holding a single decode row), each 128-pool chunk of that sequence is
fetched once and scored for all rows in one ``[rows * H, D] x [D, 128]`` matmul. Otherwise
(decode rows, a prompt boundary inside the block) every row fetches its own pages and runs its own
matmul. Both paths apply the same per-row operations to each row's ``[H, 128]`` products. f32
queries score in f32 at the given precision; bf16 queries over bf16 pages score in one bf16 MXU
pass with f32 accumulation, whose products are exact, as an f32 HIGHEST matmul of the upcast
values gives them. Chunks stream through ``_PAGE_SLOTS`` VMEM slots: while one chunk is scored,
the fetches of the next ``_PAGE_SLOTS - 1`` are in flight.

Selection. Scores are written as order-preserving ``int32`` keys into VMEM. The exact k-th key
``tau`` is found by MSB-first bisection over the scored chunks; the chunks after them, up to a
multiple of ``_FILL_CHUNKS``, hold ``INT32_MIN``, which no bisection count includes. Survivors
are compacted in ascending pool order, :func:`_compact_chunks` chunks per loop step. When no row
has to cut its keys equal to ``tau`` (the usual case) compaction keeps every key ``>= tau``;
otherwise it ranks the ties and keeps the lowest ids, as ``_pallas/tpu/topk`` does. Rows with
no scorable pool (padding) are settled before the bisection: their output is all ``-1``. A block
whose scored chunks hold fewer than ``k`` keys (short rows, such as decode at short context) skips
the bisection: it would end at ``INT32_MIN`` and keep every scored key.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..topk._pallas_impl_fwd import _monotone_key, _prefix_sum_lanes

_LANES = 128
_INT_MIN = -(2**31)
_TILE_ROWS = 8
_MAX_BLOCK_ROWS = 64
_MIN_BLOCKS = 8
_FILL_CHUNKS = 8
_COUNT_VREGS = 16
# Page-chunk slots in VMEM: while one chunk is scored, the fetches of the next _PAGE_SLOTS - 1 are in flight. Measured on
# one TPU v5p chip at GLM-5.3-Flash per-device shapes (32 heads x 128, k = 512, bf16 queries), us per call with 2 / 4 / 8
# slots: 2 decode rows at ~61K context 197.15 / 174.60 / 173.17; 2,048 prefill rows of one prompt at 32K (64-row
# blocks) 2,496.97 / 2,445.59 with 2 / 4 slots. All give the same ids. Every slot holds one DMA semaphore per row, and
# the TPU's semaphore memory does not fit 8 slots of 64-row blocks.
_PAGE_SLOTS = 4
# Row-chunks (rows x chunks) one compaction step covers at most (see _compact_chunks).
_COMPACT_ROW_CHUNKS = 256
_VMEM_LIMIT_BYTES = 96 * 1024 * 1024
# VMEM a block may use for its rows (keys, page buffers, pipelined query / weight / output rows); larger blocks are
# halved until they fit, down to 8 rows.
_BLOCK_VMEM_BUDGET_BYTES = 48 * 1024 * 1024


def _key_width(pages_per_seq: int, pools_per_page: int) -> int:
    """Key scratch columns: the longest row's pools, rounded up to ``_FILL_CHUNKS`` chunks of 128."""
    span = _FILL_CHUNKS * _LANES
    return -(-(pages_per_seq * pools_per_page) // span) * span


def _row_vmem_bytes(heads: int, dim: int, q_itemsize: int, page_itemsize: int, key_width: int, k: int) -> int:
    """VMEM one row of a block needs: its keys, its ``_PAGE_SLOTS`` page slots and its double-buffered query, weights
    and ids."""
    k_pad = -(-k // _LANES) * _LANES
    pages = _PAGE_SLOTS * _LANES * dim * page_itemsize
    return 4 * key_width + pages + 2 * heads * (dim * q_itemsize + _LANES * 4) + 8 * k_pad


def _block_rows(total: int, row_bytes: int = 0) -> int:
    """Rows per program: 8, doubled up to ``_MAX_BLOCK_ROWS`` while the grid keeps ``_MIN_BLOCKS`` programs and the
    block's VMEM (``row_bytes`` per row) stays within ``_BLOCK_VMEM_BUDGET_BYTES``."""
    rows = _TILE_ROWS
    while (
        rows < _MAX_BLOCK_ROWS
        and -(-int(total) // (2 * rows)) >= _MIN_BLOCKS
        and 2 * rows * int(row_bytes) <= _BLOCK_VMEM_BUDGET_BYTES
    ):
        rows *= 2
    return rows


def _compact_chunks(rows: int) -> int:
    """Chunks per compaction step for blocks of ``rows`` rows: ``_FILL_CHUNKS``, halved while one step would cover more
    than ``_COMPACT_ROW_CHUNKS`` row-chunks (8 chunks for 8- to 32-row blocks, 4 for 64-row blocks).

    Measured on one TPU v5p chip (4 page slots, 32 heads x 128, k = 512, bf16 queries, ~61K context unless noted), us
    per call with 2 / 4 / 8 chunks per step: 2 decode rows (8-row blocks) 174.44 / 142.26 / 127.83; 128 rows of one
    prompt (16-row blocks) 642.96 / 508.64 / 441.59; 256 rows of one prompt (32-row blocks) 792.58 / 656.82 / 614.29;
    2,048 rows of one prompt at 32K (64-row blocks) 2,445.98 / 2,258.40 / 2,363.88. All give the same ids.
    """
    chunks = _FILL_CHUNKS
    while chunks > 1 and chunks * int(rows) > _COMPACT_ROW_CHUNKS:
        chunks //= 2
    return chunks


def _num_blocks(total: int, rows: int) -> int:
    """Programs of ``rows`` rows covering ``total`` rows; an even count when there are two or more.

    Two to ``rows`` rows (decode on few rows per device) also take two programs, one per TensorCore of a megacore chip:
    :func:`_split_rows` deals them over both blocks.
    """
    blocks = -(-int(total) // rows)
    if blocks == 1 and int(total) > 1:
        return 2
    return blocks + blocks % 2 if blocks > 1 else blocks


def _split_rows(total: int, rows: int) -> np.ndarray | None:
    """Padded position of each query row when ``total`` rows fit one block of ``rows``: row ``i`` goes to block
    ``i % 2`` at slot ``i // 2``, so both programs score real rows. ``None`` when rows stay in order."""
    total = int(total)
    if total < 2 or total > rows:
        return None
    ids = np.arange(total)
    return (ids % 2) * rows + ids // 2


def padded_rows(total: int) -> int:
    """Upper bound on the rows the kernel runs (and scalar-prefetches) for ``total`` query rows.

    Exact unless the VMEM budget shrinks the blocks, which only lowers the padded row count.
    """
    rows = _block_rows(total)
    return _num_blocks(total, rows) * rows


def _block_of(step, num_blocks: int):
    """Row block of grid step ``step``: the first half of the grid runs the even blocks, the second the odd ones."""
    if num_blocks < 2:
        return step
    half = num_blocks // 2
    return jax.lax.select(step < half, 2 * step, 2 * step - (2 * half - 1))


def _kernel(
    seq_ref,
    npool_ref,
    bt_ref,
    q_ref,
    w_ref,
    pool_hbm,
    idx_ref,
    kbuf,
    sems,
    keys_ref,
    *,
    k: int,
    k_pad: int,
    group: int,
    pools_per_page: int,
    pages_per_seq: int,
    num_pages: int,
    num_blocks: int,
    sm_scale: float,
    relu: bool,
    precision,
    can_share: bool,
    compact_chunks: int,
    low: bool,
):
    rows, heads, dim = q_ref.shape
    base = _block_of(pl.program_id(0), num_blocks) * rows
    seqs = [seq_ref[base + r] for r in range(rows)]
    npools = [npool_ref[base + r] for r in range(rows)]
    row_steps = [(n + _LANES - 1) // _LANES for n in npools]
    n_steps = row_steps[0]
    for s in row_steps[1:]:
        n_steps = jnp.maximum(n_steps, s)
    # The program fetches each page once for all rows when every scorable row reads one sequence.
    lead = seqs[0]
    for r in range(1, rows):
        lead = jnp.where(npools[r] > 0, seqs[r], lead)
    shared = (npools[0] == 0) | (seqs[0] == lead)
    for r in range(1, rows):
        shared = shared & ((npools[r] == 0) | (seqs[r] == lead))

    def dma(seq, c, slot, r, wait: bool):
        for g in range(group):
            logical = jnp.minimum(c * group + g, pages_per_seq - 1)
            page = jnp.clip(bt_ref[seq * pages_per_seq + logical], 0, num_pages - 1)
            cp = pltpu.make_async_copy(pool_hbm.at[page], kbuf.at[slot, r, g], sems.at[slot, r])
            if wait:
                cp.wait()
            else:
                cp.start()

    def issue_shared(c, slot, wait: bool):
        dma(lead, c, slot, 0, wait)

    def issue_rows(c, slot, wait: bool):
        for r in range(rows):

            @pl.when(c < row_steps[r])
            def _(r=r):
                dma(seqs[r], c, slot, r, wait)

    rlane = jax.lax.broadcasted_iota(jnp.int32, (rows, _LANES), 1)
    npool_lane = jnp.concatenate([jnp.full((1, _LANES), n, jnp.int32) for n in npools], axis=0)  # [rows, 128]
    neg_inf_key = _monotone_key(jnp.full((rows, _LANES), -jnp.inf, jnp.float32))

    def chunk_at(c):
        return pl.ds(pl.multiple_of(c * _LANES, _LANES), _LANES)

    def products(qq, slot, r):
        kk = kbuf[slot, r].reshape(group * pools_per_page, dim)  # [128, D]
        if low:
            # bf16 queries over bf16 pages: one MXU pass with f32 accumulation; every product is exact. The precision
            # is explicit: an ambient higher default (jax.default_matmul_precision) would ask Mosaic for a multi-pass
            # f32 contraction, which it rejects for bf16 operands.
            return jax.lax.dot_general(
                qq,
                kk,
                (((1,), (1,)), ((), ())),
                precision=jax.lax.Precision.DEFAULT,
                preferred_element_type=jnp.float32,
            )
        return jax.lax.dot_general(
            qq.astype(jnp.float32),
            kk.astype(jnp.float32),
            (((1,), (1,)), ((), ())),
            precision=precision,
            preferred_element_type=jnp.float32,
        )

    def head_sum(r, s):
        """Row ``r``'s weighted head sum ``[1, 128]`` from its head products ``s`` ``[H, 128]``."""
        s = s * sm_scale
        if relu:
            s = jnp.maximum(s, 0.0)
        return jnp.sum(s * w_ref[r].astype(jnp.float32), axis=0, keepdims=True)

    def sums_shared(slot):
        s = products(q_ref[...].reshape(rows * heads, dim), slot, 0)  # [rows * H, 128]
        return [head_sum(r, s[r * heads : (r + 1) * heads]) for r in range(rows)]

    def sums_rows(slot):
        return [head_sum(r, products(q_ref[r], slot, r)) for r in range(rows)]

    def score(issue, sums):
        # The first _PAGE_SLOTS - 1 chunks start here; step c starts chunk c + _PAGE_SLOTS - 1 in the slot chunk c - 1
        # used, so that many fetches are in flight while chunk c is scored.
        for j in range(_PAGE_SLOTS - 1):

            @pl.when(j < n_steps)
            def _(j=j):
                issue(j, j, wait=False)

        def step(c, carry):
            slot = c % _PAGE_SLOTS
            ahead = c + (_PAGE_SLOTS - 1)

            @pl.when(ahead < n_steps)
            def _():
                issue(ahead, ahead % _PAGE_SLOTS, wait=False)

            issue(c, slot, wait=True)
            sc = jnp.concatenate(sums(slot), axis=0)  # [rows, 128]
            keys_ref[:, chunk_at(c)] = jnp.where(c * _LANES + rlane < npool_lane, _monotone_key(sc), neg_inf_key)
            return carry

        jax.lax.fori_loop(0, n_steps, step, 0)

    if can_share:

        @pl.when(shared)
        def _():
            score(issue_shared, sums_shared)

        @pl.when(jnp.logical_not(shared))
        def _():
            score(issue_rows, sums_rows)

    else:
        score(issue_rows, sums_rows)

    # Chunks up to the next multiple of _FILL_CHUNKS hold INT32_MIN: no bisection count includes them.
    def fill(c, carry):
        keys_ref[:, chunk_at(c)] = jnp.full((rows, _LANES), _INT_MIN, jnp.int32)
        return carry

    jax.lax.fori_loop(n_steps, (n_steps + _FILL_CHUNKS - 1) // _FILL_CHUNKS * _FILL_CHUNKS, fill, 0)

    # Exact k-th key by MSB-first bisection; counts read ``count_chunks`` chunks per loop step (a power of two that
    # divides _FILL_CHUNKS, so every counted chunk is scored or filled).
    count_chunks = _FILL_CHUNKS
    while count_chunks > 1 and count_chunks * rows > _COUNT_VREGS * _TILE_ROWS:
        count_chunks //= 2
    span = count_chunks * _LANES
    n_iter = (n_steps + count_chunks - 1) // count_chunks

    def count(predicate):
        def body(i, acc):
            return acc + predicate(keys_ref[:, pl.ds(pl.multiple_of(i * span, span), span)]).astype(jnp.int32)

        acc = jax.lax.fori_loop(0, n_iter, body, jnp.zeros((rows, span), jnp.int32))
        return jnp.sum(acc, axis=1, keepdims=True)

    def bisect(state):
        bit, biased, done = state
        candidate = biased | jnp.left_shift(jnp.int32(1), 31 - bit)
        n = count(lambda kc: kc >= (candidate ^ jnp.int32(_INT_MIN)))
        accept = (n >= k) & (done == 0)
        return bit + 1, jnp.where(accept, candidate, biased), jnp.where(accept & (n == k), 1, done)

    def unfinished(state):
        bit, _, done = state
        return (bit < 32) & (jnp.min(done) == 0)

    empty = npool_lane[:, :1] == 0  # [rows, 1]: output all -1 whatever tau is
    # A block whose scored chunks hold fewer than k keys (short rows, such as decode at short context) has fewer than
    # k keys above INT32_MIN, so no candidate is ever accepted and the bisection would end at tau = INT32_MIN: skip it.
    settled = empty | (n_steps * _LANES < k)
    init = (jnp.int32(0), jnp.zeros((rows, 1), jnp.int32), settled.astype(jnp.int32))
    _, biased, _ = jax.lax.while_loop(unfinished, bisect, init)
    tau = biased ^ jnp.int32(_INT_MIN)
    need = k - count(lambda kc: kc > tau)
    # Keys equal to tau in the scored chunks (the filler is INT32_MIN, counted only when tau is INT32_MIN).
    filler = (n_iter * count_chunks - n_steps) * _LANES
    n_eq = count(lambda kc: kc == tau) - jnp.where(tau == _INT_MIN, filler, 0)
    no_cut = jnp.min((empty | (n_eq <= need)).astype(jnp.int32)) == 1

    # Compact survivors in ascending pool order; with a cut, ties at tau keep the lowest ids.
    idx_ref[...] = jnp.full((rows, k_pad), -1, jnp.int32)
    out_lane = jax.lax.broadcasted_iota(jnp.int32, (rows, k_pad), 1)

    def compact(g, carry, exact: bool):
        offset, ties = carry
        chunks = [g * compact_chunks + j for j in range(compact_chunks)]
        sels = []
        for j, c in enumerate(chunks):
            kc = keys_ref[:, chunk_at(c)]
            # The first chunk of a group is always scored; later ones may lie past the row block's last chunk.
            live = None if j == 0 else (c * _LANES + rlane) < n_steps * _LANES
            if exact:
                eq = kc == tau
                if live is not None:
                    eq = eq & live
                eq = eq.astype(jnp.int32)
                tie_rank = _prefix_sum_lanes(eq, rlane) - eq + ties
                sel = (kc > tau) | ((eq == 1) & (tie_rank < need))
                ties = ties + jnp.sum(eq, axis=1, keepdims=True)
            else:
                sel = kc >= tau
            if live is not None:
                sel = sel & live
            sels.append(sel.astype(jnp.int32))
        incls = [_prefix_sum_lanes(sel, rlane) for sel in sels]
        # Source lane of the t-th selected key: the first lane with incl >= t + 1.
        poss = [jnp.zeros((rows, _LANES), jnp.int32) for _ in chunks]
        step = _LANES // 2
        while step >= 1:
            probes = [
                jnp.take_along_axis(incl, pos + step - 1, axis=1, mode="promise_in_bounds")
                for incl, pos in zip(incls, poss, strict=True)
            ]
            poss = [jnp.where(probe < rlane + 1, pos + step, pos) for probe, pos in zip(probes, poss, strict=True)]
            step //= 2
        ids = idx_ref[...]
        for c, incl, pos in zip(chunks, incls, poss, strict=True):
            cnt = incl[:, _LANES - 1 :]
            rotate = (rlane - offset) & (_LANES - 1)
            src = jnp.take_along_axis(c * _LANES + pos, rotate, axis=1, mode="promise_in_bounds")
            hit = (out_lane >= offset) & (out_lane < offset + cnt)
            ids = jnp.where(hit, jnp.tile(src, (1, k_pad // _LANES)), ids)
            offset = offset + cnt
        idx_ref[...] = ids
        return offset, ties

    n_groups = (n_steps + compact_chunks - 1) // compact_chunks
    start = (jnp.zeros((rows, 1), jnp.int32), jnp.zeros((rows, 1), jnp.int32))

    @pl.when(no_cut)
    def _():
        jax.lax.fori_loop(0, n_groups, functools.partial(compact, exact=False), start)

    @pl.when(jnp.logical_not(no_cut))
    def _():
        jax.lax.fori_loop(0, n_groups, functools.partial(compact, exact=True), start)

    # Masked (unscorable) pools can only be picked when a row has fewer than k pools.
    npool_col = jnp.concatenate([jnp.full((1, k_pad), n, jnp.int32) for n in npools], axis=0)
    ids = idx_ref[...]
    idx_ref[...] = jnp.where((ids >= 0) & (ids < npool_col), ids, -1)


def _paged_pool_topk_call(
    query: jax.Array,
    head_weights: jax.Array,
    pool_pages: jax.Array,
    block_tables: jax.Array,
    seq_ids: jax.Array,
    num_pools: jax.Array,
    *,
    k: int,
    sm_scale: float,
    relu: bool,
    precision,
    block_rows: int,
    compact_chunks: int,
) -> jax.Array:
    """:func:`paged_pool_topk_pallas` with explicit rows per program and chunks per compaction step."""
    total, heads, dim = query.shape
    num_pages, pools_per_page, _ = pool_pages.shape
    if _LANES % pools_per_page:
        raise ValueError(f"pools_per_page ({pools_per_page}) must divide {_LANES}.")
    if block_rows % _TILE_ROWS or _FILL_CHUNKS % compact_chunks:
        raise ValueError(f"block_rows ({block_rows}) and compact_chunks ({compact_chunks}) do not tile the keys.")
    group = _LANES // pools_per_page
    num_seqs, pages_per_seq = block_tables.shape
    k_pad = -(-k // _LANES) * _LANES
    max_pools = -(-(pages_per_seq * pools_per_page) // _LANES) * _LANES
    key_width = _key_width(pages_per_seq, pools_per_page)
    num_blocks = _num_blocks(total, block_rows)
    t_pad = num_blocks * block_rows
    pad = t_pad - total
    split = _split_rows(total, block_rows)

    def place(x):
        """Rows of ``x`` at their padded positions; padding rows are zero (``num_pools`` 0: output all -1)."""
        if split is None:
            return jnp.pad(x, ((0, pad),) + ((0, 0),) * (x.ndim - 1))
        return jnp.zeros((t_pad, *x.shape[1:]), x.dtype).at[split].set(x)

    q = place(query)
    w = place(head_weights.astype(jnp.float32))[..., None]  # [T, H, 1]
    seq = place(jnp.clip(seq_ids.astype(jnp.int32), 0, num_seqs - 1))
    npool = place(jnp.clip(num_pools.astype(jnp.int32), 0, max_pools))
    bt = block_tables.astype(jnp.int32).reshape(-1)

    kernel = functools.partial(
        _kernel,
        k=int(k),
        k_pad=int(k_pad),
        group=int(group),
        pools_per_page=int(pools_per_page),
        pages_per_seq=int(pages_per_seq),
        num_pages=int(num_pages),
        num_blocks=int(num_blocks),
        sm_scale=float(sm_scale),
        relu=bool(relu),
        precision=precision,
        # One [rows * H, D] query tile per program needs H on whole sublane tiles of the query dtype.
        can_share=heads % (8 * 4 // jnp.dtype(query.dtype).itemsize) == 0,
        compact_chunks=int(compact_chunks),
        low=jnp.dtype(query.dtype) == jnp.bfloat16 and jnp.dtype(pool_pages.dtype) == jnp.bfloat16,
    )

    def rows_map(b, *_):
        return (_block_of(b, num_blocks), 0, 0)

    ids = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(num_blocks,),
            in_specs=[
                pl.BlockSpec((block_rows, heads, dim), rows_map),
                pl.BlockSpec((block_rows, heads, 1), rows_map),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec((block_rows, k_pad), lambda b, *_: (_block_of(b, num_blocks), 0)),
            scratch_shapes=[
                pltpu.VMEM((_PAGE_SLOTS, block_rows, group, pools_per_page, dim), pool_pages.dtype),
                pltpu.SemaphoreType.DMA((_PAGE_SLOTS, block_rows)),
                pltpu.VMEM((block_rows, key_width), jnp.int32),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct((t_pad, k_pad), jnp.int32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",), vmem_limit_bytes=_VMEM_LIMIT_BYTES),
        name="paged_pool_topk",
    )(seq, npool, bt, q, w, pool_pages)
    return ids[:total, :k] if split is None else ids[split, :k]


@functools.partial(jax.jit, static_argnames=("k", "sm_scale", "relu", "precision"))
def paged_pool_topk_pallas(
    query: jax.Array,
    head_weights: jax.Array,
    pool_pages: jax.Array,
    block_tables: jax.Array,
    seq_ids: jax.Array,
    num_pools: jax.Array,
    *,
    k: int,
    sm_scale: float,
    relu: bool = True,
    precision=jax.lax.Precision.HIGHEST,
) -> jax.Array:
    """Select each row's ``k`` best pools from paged pool summaries.

    Args:
        query: ``[T, H, D]`` indexer queries.
        head_weights: ``[T, H]`` per-head weights (any head scaling already folded in).
        pool_pages: ``[P, pools_per_page, D]`` completed-pool summaries.
        block_tables: ``[S, pages_per_seq]`` page table shared with the KV cache.
        seq_ids: ``[T]`` sequence of each row.
        num_pools: ``[T]`` scorable pools per row (0 for padding rows).
        k: Pools kept per row (static).
        sm_scale: Score scale before the activation.
        relu: Apply ``relu`` per head before weighting.
        precision: Matmul precision for f32 scoring. A bf16 query over bf16 pages scores in one
            bf16 MXU pass with f32 accumulation instead (exact products, as HIGHEST gives for
            the upcast values).

    Returns:
        ``[T, k]`` int32 pool ids in ascending order, ``-1``-padded when a row has fewer
        than ``k`` pools. The set equals ``jax.lax.top_k`` over the row's scores.
    """
    _, heads, dim = query.shape
    row_bytes = _row_vmem_bytes(
        heads,
        dim,
        jnp.dtype(query.dtype).itemsize,
        jnp.dtype(pool_pages.dtype).itemsize,
        _key_width(int(block_tables.shape[1]), int(pool_pages.shape[1])),
        int(k),
    )
    block_rows = _block_rows(query.shape[0], row_bytes)
    return _paged_pool_topk_call(
        query,
        head_weights,
        pool_pages,
        block_tables,
        seq_ids,
        num_pools,
        k=k,
        sm_scale=sm_scale,
        relu=relu,
        precision=precision,
        block_rows=block_rows,
        compact_chunks=_compact_chunks(block_rows),
    )


__all__ = ("padded_rows", "paged_pool_topk_pallas")
