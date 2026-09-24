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

"""Pallas TPU top-k: exact k-th-key bisection, stream compaction, small sort.

XLA lowers ``jax.lax.top_k`` on TPU to a sort of the whole reduction axis. Only
``k`` elements survive, so this kernel instead finds the exact ``k``-th largest
element and compacts the survivors, leaving a sort over ``k`` elements:

1. Every float maps to an ``int32`` key whose signed order is the IEEE
   totalOrder ``top_k`` uses on TPU (+NaN > +inf > ... > +0.0 > -0.0 > ...
   > -inf > -NaN, NaNs ordered by payload). The
   largest threshold ``t`` with ``count(key >= t) >= k`` is built MSB-first on
   VMEM-resident rows. A row stops early once some ``t`` selects exactly ``k``
   keys, since later bits can no longer change its selection.
2. Keys above the threshold, plus the lowest-index ties at it, are compacted in
   index order: a per-128-lane prefix sum and a lane-gather binary search give
   each output slot its source lane, and a rotation places the chunk at the
   row's running offset. Chunks are processed in independent groups, because
   the cross-lane unit is pipelined but has ~60-cycle latency.
3. A stable sort of the ``k`` survivors by descending key restores ``top_k``'s
   value order; equal keys keep their index order, i.e. lower index first.

Values are reconstructed from their keys, which is exact for float32 and for
narrower floats (the widening is exact), so the result equals ``top_k``'s bit
for bit.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array

from ejkernel.callib import ejit

_LANES = 128
_INT_MIN = -(2**31)
_CHUNK_GROUP = 4
_MAX_BLOCK_ROWS = 64
#: Up to this many 128-lane output blocks a masked sweep of every block is
#: cheapest; beyond it (k > 1024) merging only the touched blocks is faster.
_FULL_MERGE_MAX_BLOCKS = 8
#: Per grid step: double-buffered input rows, the key scratch, and the two
#: double-buffered outputs. Leaves headroom under ``_VMEM_LIMIT_BYTES``.
_VMEM_BUDGET_BYTES = 32 * 1024 * 1024
_VMEM_LIMIT_BYTES = 64 * 1024 * 1024


def _monotone_key(x: Array) -> Array:
    """``int32`` key whose signed order is IEEE totalOrder, as TPU ``top_k`` uses.

    Measured on TPU: positive NaNs rank above +inf ordered by payload, and
    negative NaNs below -inf, so no NaN special-casing is wanted.
    """
    bits = jax.lax.bitcast_convert_type(x.astype(jnp.float32), jnp.int32)
    return jnp.where(bits < 0, bits ^ jnp.int32(0x7FFFFFFF), bits)


def _key_to_value(key: Array) -> Array:
    """Invert :func:`_monotone_key` bit for bit, NaN payloads included."""
    bits = jnp.where(key < 0, key ^ jnp.int32(0x7FFFFFFF), key)
    return jax.lax.bitcast_convert_type(bits, jnp.float32)


def _prefix_sum_lanes(v: Array, lane: Array) -> Array:
    """Inclusive prefix sum along the 128-lane axis, in log steps."""
    shift = 1
    while shift < _LANES:
        v = v + jnp.where(lane >= shift, pltpu.roll(v, shift, 1), 0)
        shift *= 2
    return v


def _pick_block_rows(width: int, k_pad: int, itemsize: int) -> int:
    """Largest multiple of 8 rows (up to 64) whose working set fits VMEM."""
    per_row = 2 * width * itemsize + 4 * width + 2 * 2 * 4 * k_pad
    rows = min(_MAX_BLOCK_ROWS, _VMEM_BUDGET_BYTES // max(per_row, 1))
    return max(8, rows // 8 * 8)


def _threshold_topk_kernel(x_ref, idx_ref, key_ref, keys_ref, *, k: int, k_pad: int, rows: int, width: int):
    """Compact the top-``k`` keys of each row in ascending index order.

    Args:
        x_ref: Input rows ``[rows, width]``.
        idx_ref: Output ``[rows, k_pad]`` int32 source indices.
        key_ref: Output ``[rows, k_pad]`` int32 keys.
        keys_ref: Scratch ``[rows, width]`` int32 keys.
        k: Survivors per row.
        k_pad: ``k`` rounded up to a lane multiple.
        rows: Rows per grid step.
        width: Reduction width (a lane multiple).
    """
    lane = jax.lax.broadcasted_iota(jnp.int32, (rows, _LANES), 1)
    n_chunks = width // _LANES
    group = min(_CHUNK_GROUP, n_chunks)
    while n_chunks % group:
        group -= 1

    def fill(c, carry):
        start = pl.multiple_of(c * _LANES, _LANES)
        keys_ref[:, pl.ds(start, _LANES)] = _monotone_key(x_ref[:, pl.ds(start, _LANES)])
        return carry

    jax.lax.fori_loop(0, n_chunks, fill, 0, unroll=group)

    def chunk_keys(c):
        return keys_ref[:, pl.ds(pl.multiple_of(c * _LANES, _LANES), _LANES)]

    def count(predicate):
        def body(c, acc):
            return acc + predicate(chunk_keys(c)).astype(jnp.int32)

        acc = jax.lax.fori_loop(0, n_chunks, body, jnp.zeros((rows, _LANES), jnp.int32), unroll=group)
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

    start = (jnp.int32(0), jnp.zeros((rows, 1), jnp.int32), jnp.zeros((rows, 1), jnp.int32))
    _, biased, _ = jax.lax.while_loop(unfinished, bisect, start)
    tau = biased ^ jnp.int32(_INT_MIN)
    need = k - count(lambda kc: kc > tau)
    take_all_ties = jnp.all(count(lambda kc: kc == tau) == need)

    idx_ref[...] = jnp.zeros((rows, k_pad), jnp.int32)
    key_ref[...] = jnp.full((rows, k_pad), _INT_MIN, jnp.int32)
    out_lane = jax.lax.broadcasted_iota(jnp.int32, (rows, k_pad), 1)

    def compact_group(g, carry, exact_ties: bool):
        offset, ties = carry
        chunks = [g * group + j for j in range(group)]
        kcs = [chunk_keys(c) for c in chunks]
        sels = []
        for kc in kcs:
            if exact_ties:
                eq = (kc == tau).astype(jnp.int32)
                tie_rank = _prefix_sum_lanes(eq, lane) - eq + ties
                sels.append(((kc > tau) | ((eq == 1) & (tie_rank < need))).astype(jnp.int32))
                ties = ties + jnp.sum(eq, axis=1, keepdims=True)
            else:
                sels.append((kc >= tau).astype(jnp.int32))
        incls = [_prefix_sum_lanes(sel, lane) for sel in sels]
        # Source lane of the t-th selected key: the first lane with incl >= t + 1.
        poss = [jnp.zeros((rows, _LANES), jnp.int32) for _ in kcs]
        step = _LANES // 2
        while step >= 1:
            probes = [
                jnp.take_along_axis(incl, pos + step - 1, axis=1, mode="promise_in_bounds")
                for incl, pos in zip(incls, poss, strict=True)
            ]
            poss = [jnp.where(probe < lane + 1, pos + step, pos) for probe, pos in zip(probes, poss, strict=True)]
            step //= 2
        sources = []
        start = offset
        for c, kc, pos, incl in zip(chunks, kcs, poss, incls, strict=True):
            cnt = incl[:, _LANES - 1 :]
            rotate = (lane - offset) & (_LANES - 1)
            src_idx = jnp.take_along_axis(c * _LANES + pos, rotate, axis=1, mode="promise_in_bounds")
            src_key = jnp.take_along_axis(
                jnp.take_along_axis(kc, pos, axis=1, mode="promise_in_bounds"),
                rotate,
                axis=1,
                mode="promise_in_bounds",
            )
            sources.append((offset, cnt, src_idx, src_key))
            offset = offset + cnt

        if k_pad // _LANES > _FULL_MERGE_MAX_BLOCKS:
            # Rows advance at similar rates, so a group lands in a few output
            # blocks; merging only those beats sweeping every block for large k.
            first_block = jnp.min(start) // _LANES
            last_block = jnp.minimum((jnp.max(offset) - 1) // _LANES, k_pad // _LANES - 1)

            def merge(block, carry):
                base = pl.multiple_of(block * _LANES, _LANES)
                out_pos = base + lane
                blk_idx = idx_ref[:, pl.ds(base, _LANES)]
                blk_key = key_ref[:, pl.ds(base, _LANES)]
                for off, cnt, src_idx, src_key in sources:
                    hit = (out_pos >= off) & (out_pos < off + cnt)
                    blk_idx = jnp.where(hit, src_idx, blk_idx)
                    blk_key = jnp.where(hit, src_key, blk_key)
                idx_ref[:, pl.ds(base, _LANES)] = blk_idx
                key_ref[:, pl.ds(base, _LANES)] = blk_key
                return carry

            jax.lax.fori_loop(first_block, last_block + 1, merge, 0)
        else:
            new_idx, new_key = idx_ref[...], key_ref[...]
            for off, cnt, src_idx, src_key in sources:
                hit = (out_lane >= off) & (out_lane < off + cnt)
                new_idx = jnp.where(hit, jnp.tile(src_idx, (1, k_pad // _LANES)), new_idx)
                new_key = jnp.where(hit, jnp.tile(src_key, (1, k_pad // _LANES)), new_key)
            idx_ref[...] = new_idx
            key_ref[...] = new_key
        return offset, ties

    init = (jnp.zeros((rows, 1), jnp.int32), jnp.zeros((rows, 1), jnp.int32))

    @pl.when(take_all_ties)
    def _():
        jax.lax.fori_loop(0, n_chunks // group, functools.partial(compact_group, exact_ties=False), init)

    @pl.when(jnp.logical_not(take_all_ties))
    def _():
        jax.lax.fori_loop(0, n_chunks // group, functools.partial(compact_group, exact_ties=True), init)


def _threshold_topk_fwd_impl(operand: Array, k: int) -> tuple[Array, Array]:
    """Forward body over the last axis of a 2-D ``operand``."""
    num_rows, width = operand.shape
    k_pad = -(-k // _LANES) * _LANES
    padded_width = -(-width // _LANES) * _LANES
    block_rows = _pick_block_rows(padded_width, k_pad, jnp.dtype(operand.dtype).itemsize)
    padded_rows = -(-num_rows // block_rows) * block_rows
    x = operand
    if (padded_rows, padded_width) != (num_rows, width):
        # -inf padding sorts after every real key, and among equal -inf keys
        # the padded columns have the highest indices, so none can displace a
        # real element that top_k would return.
        x = jnp.pad(x, ((0, padded_rows - num_rows), (0, padded_width - width)), constant_values=-jnp.inf)

    idx, key = pl.pallas_call(
        functools.partial(_threshold_topk_kernel, k=k, k_pad=k_pad, rows=block_rows, width=padded_width),
        grid=(padded_rows // block_rows,),
        in_specs=[pl.BlockSpec((block_rows, padded_width), lambda i: (i, 0))],
        out_specs=(
            pl.BlockSpec((block_rows, k_pad), lambda i: (i, 0)),
            pl.BlockSpec((block_rows, k_pad), lambda i: (i, 0)),
        ),
        out_shape=(
            jax.ShapeDtypeStruct((padded_rows, k_pad), jnp.int32),
            jax.ShapeDtypeStruct((padded_rows, k_pad), jnp.int32),
        ),
        scratch_shapes=[pltpu.VMEM((block_rows, padded_width), jnp.int32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=_VMEM_LIMIT_BYTES,
        ),
        name="threshold_topk",
    )(x)
    idx, key = idx[:num_rows, :k], key[:num_rows, :k]
    # Survivors are in ascending index order; a stable sort on the bitwise
    # complement (descending key, no overflow) yields top_k's order.
    inverted, idx = jax.lax.sort((~key, idx), num_keys=1, is_stable=True)
    return _key_to_value(~inverted).astype(operand.dtype), idx


@functools.partial(jax.custom_vjp, nondiff_argnums=(1,))
def _threshold_topk(operand: Array, k: int) -> tuple[Array, Array]:
    """Differentiable wrapper: a ``pallas_call`` has no transpose rule."""
    return _threshold_topk_fwd_impl(operand, k)


def _threshold_topk_fwd(operand, k):
    values, indices = _threshold_topk_fwd_impl(operand, k)
    return (values, indices), (indices, operand.shape)


def _threshold_topk_bwd(k, res, cotangents):
    """Send value cotangents to the selected positions, as ``top_k`` does.

    Indices within a row are distinct, so this scatter-add equals a scatter.
    """
    del k
    indices, shape = res
    dvalues = cotangents[0]
    rows = jnp.arange(shape[0], dtype=indices.dtype)[:, None]
    return (jnp.zeros(shape, dvalues.dtype).at[rows, indices].add(dvalues),)


_threshold_topk.defvjp(_threshold_topk_fwd, _threshold_topk_bwd)


@ejit(static_argnames=["k"])
def topk_threshold_tpu(operand: Array, k: int) -> tuple[Array, Array]:
    """Exact ``jax.lax.top_k`` over the last axis of a 2-D float array.

    Args:
        operand: ``[rows, width]`` floating-point input.
        k: Static number of results, ``1 <= k <= width``.

    Returns:
        ``(values, indices)``, each ``[rows, k]``, equal to ``jax.lax.top_k``.

    Raises:
        ValueError: If ``k`` is outside ``[1, width]`` or ``operand`` is not a
            2-D floating-point array.
    """
    if operand.ndim != 2 or not jnp.issubdtype(operand.dtype, jnp.floating):
        raise ValueError(f"topk_threshold_tpu expects a 2-D float array, got {operand.dtype}{operand.shape}.")
    if not 1 <= k <= operand.shape[1]:
        raise ValueError(f"k must be in [1, {operand.shape[1]}], got {k}.")
    return _threshold_topk(operand, int(k))
