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
context, never ``max_model_len``). Eight rows share a program so every tile is ``(8, 128)``.
Scores are written as order-preserving ``int32`` keys into VMEM; the exact k-th key is found
by MSB-first bisection and the survivors are compacted in ascending pool order (same scheme
as ``_pallas/tpu/topk``), all over the scored chunks only.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ..topk._pallas_impl_fwd import _monotone_key, _prefix_sum_lanes

_LANES = 128
_INT_MIN = -(2**31)
_ROWS = 8
_VMEM_LIMIT_BYTES = 96 * 1024 * 1024


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
    sm_scale: float,
    relu: bool,
    precision,
):
    rows = _ROWS
    base = pl.program_id(0) * rows
    seqs = [seq_ref[base + r] for r in range(rows)]
    npools = [npool_ref[base + r] for r in range(rows)]
    row_steps = [(n + _LANES - 1) // _LANES for n in npools]
    n_steps = row_steps[0]
    for s in row_steps[1:]:
        n_steps = jnp.maximum(n_steps, s)

    def copy(r, c, g, slot):
        logical = jnp.minimum(c * group + g, pages_per_seq - 1)
        page = jnp.clip(bt_ref[seqs[r] * pages_per_seq + logical], 0, num_pages - 1)
        return pltpu.make_async_copy(pool_hbm.at[page], kbuf.at[slot, r, g], sems.at[slot, r])

    def issue(c, slot, wait: bool):
        for r in range(rows):

            @pl.when(c < row_steps[r])
            def _(r=r):
                for g in range(group):
                    cp = copy(r, c, g, slot)
                    if wait:
                        cp.wait()
                    else:
                        cp.start()

    @pl.when(n_steps > 0)
    def _():
        issue(0, 0, wait=False)

    lane = jax.lax.broadcasted_iota(jnp.int32, (1, _LANES), 1)
    neg_inf_key = _monotone_key(jnp.full((1, _LANES), -jnp.inf, jnp.float32))

    def score_step(c, carry):
        slot = c % 2

        @pl.when(c + 1 < n_steps)
        def _():
            issue(c + 1, 1 - slot, wait=False)

        issue(c, slot, wait=True)
        pid = c * _LANES + lane
        tiles = []
        for r in range(rows):
            kk = kbuf[slot, r].reshape(group * pools_per_page, -1).astype(jnp.float32)  # [128, D]
            qq = q_ref[r].astype(jnp.float32)  # [H, D]
            s = jax.lax.dot_general(
                qq, kk, (((1,), (1,)), ((), ())), precision=precision, preferred_element_type=jnp.float32
            )  # [H, 128]
            s = s * sm_scale
            if relu:
                s = jnp.maximum(s, 0.0)
            sc = jnp.sum(s * w_ref[r].astype(jnp.float32), axis=0, keepdims=True)  # [1, 128]
            key = jnp.where(pid < npools[r], _monotone_key(sc), neg_inf_key)
            tiles.append(key)
        keys_ref[:, pl.ds(pl.multiple_of(c * _LANES, _LANES), _LANES)] = jnp.concatenate(tiles, axis=0)
        return carry

    jax.lax.fori_loop(0, n_steps, score_step, 0)

    # Exact k-th key by MSB-first bisection over the scored chunks only.
    rlane = jax.lax.broadcasted_iota(jnp.int32, (rows, _LANES), 1)

    def chunk_keys(c):
        return keys_ref[:, pl.ds(pl.multiple_of(c * _LANES, _LANES), _LANES)]

    def count(predicate):
        def body(c, acc):
            return acc + predicate(chunk_keys(c)).astype(jnp.int32)

        acc = jax.lax.fori_loop(0, n_steps, body, jnp.zeros((rows, _LANES), jnp.int32))
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

    init = (jnp.int32(0), jnp.zeros((rows, 1), jnp.int32), jnp.zeros((rows, 1), jnp.int32))
    _, biased, _ = jax.lax.while_loop(unfinished, bisect, init)
    tau = biased ^ jnp.int32(_INT_MIN)
    need = k - count(lambda kc: kc > tau)

    # Compact survivors in ascending pool order; ties at tau keep the lowest ids.
    idx_ref[...] = jnp.full((rows, k_pad), -1, jnp.int32)
    out_lane = jax.lax.broadcasted_iota(jnp.int32, (rows, k_pad), 1)

    def compact(c, carry):
        offset, ties = carry
        kc = chunk_keys(c)
        eq = (kc == tau).astype(jnp.int32)
        tie_rank = _prefix_sum_lanes(eq, rlane) - eq + ties
        sel = ((kc > tau) | ((eq == 1) & (tie_rank < need))).astype(jnp.int32)
        incl = _prefix_sum_lanes(sel, rlane)
        pos = jnp.zeros((rows, _LANES), jnp.int32)
        step = _LANES // 2
        while step >= 1:
            probe = jnp.take_along_axis(incl, pos + step - 1, axis=1, mode="promise_in_bounds")
            pos = jnp.where(probe < rlane + 1, pos + step, pos)
            step //= 2
        cnt = incl[:, _LANES - 1 :]
        rotate = (rlane - offset) & (_LANES - 1)
        src = jnp.take_along_axis(c * _LANES + pos, rotate, axis=1, mode="promise_in_bounds")
        hit = (out_lane >= offset) & (out_lane < offset + cnt)
        idx_ref[...] = jnp.where(hit, jnp.tile(src, (1, k_pad // _LANES)), idx_ref[...])
        return offset + cnt, ties + jnp.sum(eq, axis=1, keepdims=True)

    jax.lax.fori_loop(0, n_steps, compact, (jnp.zeros((rows, 1), jnp.int32), jnp.zeros((rows, 1), jnp.int32)))

    # Masked (unscorable) pools can only be picked when a row has fewer than k pools.
    npool_col = jnp.concatenate([jnp.full((1, k_pad), n, jnp.int32) for n in npools], axis=0)
    ids = idx_ref[...]
    idx_ref[...] = jnp.where((ids >= 0) & (ids < npool_col), ids, -1)


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
        precision: Matmul precision for scoring.

    Returns:
        ``[T, k]`` int32 pool ids in ascending order, ``-1``-padded when a row has fewer
        than ``k`` pools. The set equals ``jax.lax.top_k`` over the row's scores.
    """
    total, heads, dim = query.shape
    num_pages, pools_per_page, _ = pool_pages.shape
    if _LANES % pools_per_page:
        raise ValueError(f"pools_per_page ({pools_per_page}) must divide {_LANES}.")
    group = _LANES // pools_per_page
    num_seqs, pages_per_seq = block_tables.shape
    k_pad = -(-k // _LANES) * _LANES
    max_pools = -(-(pages_per_seq * pools_per_page) // _LANES) * _LANES
    t_pad = -(-total // _ROWS) * _ROWS
    pad = t_pad - total
    q = jnp.pad(query, ((0, pad), (0, 0), (0, 0)))
    w = jnp.pad(head_weights.astype(jnp.float32), ((0, pad), (0, 0)))[..., None]  # [T, H, 1]
    seq = jnp.pad(jnp.clip(seq_ids.astype(jnp.int32), 0, num_seqs - 1), (0, pad))
    npool = jnp.pad(jnp.clip(num_pools.astype(jnp.int32), 0, max_pools), (0, pad))
    bt = block_tables.astype(jnp.int32).reshape(-1)

    kernel = functools.partial(
        _kernel,
        k=int(k),
        k_pad=int(k_pad),
        group=int(group),
        pools_per_page=int(pools_per_page),
        pages_per_seq=int(pages_per_seq),
        num_pages=int(num_pages),
        sm_scale=float(sm_scale),
        relu=bool(relu),
        precision=precision,
    )
    ids = pl.pallas_call(
        kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(t_pad // _ROWS,),
            in_specs=[
                pl.BlockSpec((_ROWS, heads, dim), lambda b, *_: (b, 0, 0)),
                pl.BlockSpec((_ROWS, heads, 1), lambda b, *_: (b, 0, 0)),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec((_ROWS, k_pad), lambda b, *_: (b, 0)),
            scratch_shapes=[
                pltpu.VMEM((2, _ROWS, group, pools_per_page, dim), pool_pages.dtype),
                pltpu.SemaphoreType.DMA((2, _ROWS)),
                pltpu.VMEM((_ROWS, max_pools), jnp.int32),
            ],
        ),
        out_shape=jax.ShapeDtypeStruct((t_pad, k_pad), jnp.int32),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",), vmem_limit_bytes=_VMEM_LIMIT_BYTES),
        name="paged_pool_topk",
    )(seq, npool, bt, q, w, pool_pages)
    return ids[:total, :k]


__all__ = ("paged_pool_topk_pallas",)
