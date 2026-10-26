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

"""Pallas TPU sort-free top-k selection: the exact ``k``-th key of every row and its tie cut.

Consumers that need *which* items ``jax.lax.top_k`` picks, or the selection
threshold, but not the picks in score order (an attention mask or additive
bias, a gather mask, a sparse kernel that only reads the threshold) do not need
a sort. Per row this kernel finds:

1. ``tau``, the ``k``-th largest ``int32`` key, where the key's signed order is
   the IEEE totalOrder ``top_k`` uses on TPU (+NaN > +inf > ... > +0.0 > -0.0 >
   ... > -inf > -NaN). It is built MSB-first by 32 compare-and-count passes over
   the row block resident in VMEM: the scores are read from HBM once. With
   ``exact_key=False`` a row stops as soon as some threshold selects exactly
   ``k`` keys (its selection can no longer change), so ``tau`` is then only a
   lower bound that selects the same set.
2. the tie cut: the largest index whose key equals ``tau`` that ``top_k``
   still takes (it takes ties lowest index first). When every row takes all of
   its ties this is skipped; otherwise ``ceil(log2(width))`` counting passes
   bisect the index.

``top_k``'s pick set is then ``key > tau | (key == tau & index <= cut)``: one
elementwise pass the caller's XLA graph fuses into whatever consumes the mask.
Rows with fewer than ``k`` finite scores fill with their lowest-index ``-inf``
items, exactly as ``top_k`` does, because ``-inf`` is an ordinary key here.
Padded columns get the minimum key and the highest indices, so they can never
displace a real item, even a negative NaN.

The scores are detached: a selection is not differentiable and a
``pallas_call`` has no JVP rule.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jaxtyping import Array

from ejkernel.callib import ejit

from ._pallas_impl_fwd import _key_to_value, _monotone_key

_LANES = 128
_INT_MIN = -(2**31)
_CHUNK_GROUP = 4
_MAX_BLOCK_ROWS = 64
_VMEM_BUDGET_BYTES = 32 * 1024 * 1024
_VMEM_LIMIT_BYTES = 48 * 1024 * 1024

SELECT_MODES = ("exact_mask", "kth")


def select_block_rows(width: int, dtype: jnp.dtype) -> int | None:
    """Rows per grid step whose working set fits VMEM, or ``None`` when no block fits.

    Args:
        width: Reduction width before lane padding.
        dtype: Operand dtype; sub-32-bit dtypes pack two rows per sublane, so
            their blocks are multiples of 16 rows.

    Returns:
        A row count that is a multiple of the dtype's sublane tile, or
        ``None`` when even the minimum block exceeds the VMEM budget (the
        caller then uses the XLA reference).
    """
    itemsize = jnp.dtype(dtype).itemsize
    if itemsize > 4:
        return None
    sublanes = 8 * (4 // itemsize)
    padded = -(-width // _LANES) * _LANES
    per_row = 2 * padded * itemsize + 4 * padded + 2 * 2 * 4 * _LANES * 2
    rows = min(_MAX_BLOCK_ROWS, _VMEM_BUDGET_BYTES // per_row) // sublanes * sublanes
    return rows if rows >= sublanes else None


def _select_kernel(
    k_ref,
    x_ref,
    tau_ref,
    cut_ref,
    keys_ref,
    *,
    rows: int,
    width: int,
    valid_width: int,
    exact_key: bool,
    with_cut: bool,
):
    """Find each row's ``k``-th key and (optionally) the index cut among its ties.

    Args:
        k_ref: ``[rows, 128]`` int32 per-row ``k`` (lane-broadcast).
        x_ref: ``[rows, width]`` floating scores.
        tau_ref: Output ``[rows, 128]`` int32 ``k``-th key (lane-broadcast).
        cut_ref: Output ``[rows, 128]`` int32 largest tie index taken.
        keys_ref: Scratch ``[rows, width]`` int32 keys.
        rows: Rows per grid step.
        width: Lane-padded reduction width.
        valid_width: Real reduction width; later columns get the minimum key.
        exact_key: Run all 32 steps so ``tau`` is the exact ``k``-th key.
        with_cut: Compute the tie cut (otherwise ``cut`` is ``width - 1``).
    """
    lane = jax.lax.broadcasted_iota(jnp.int32, (rows, _LANES), 1)
    n_chunks = width // _LANES
    group = min(_CHUNK_GROUP, n_chunks)
    while n_chunks % group:
        group -= 1
    k = jnp.max(k_ref[...], axis=1, keepdims=True)  # lane-broadcast input -> [rows, 1]

    def fill(c, carry):
        start = pl.multiple_of(c * _LANES, _LANES)
        key = _monotone_key(x_ref[:, pl.ds(start, _LANES)])
        if valid_width < width:
            key = jnp.where(lane < valid_width - start, key, jnp.int32(_INT_MIN))
        keys_ref[:, pl.ds(start, _LANES)] = key
        return carry

    jax.lax.fori_loop(0, n_chunks, fill, 0, unroll=group)

    def count(predicate):
        def body(c, acc):
            start = pl.multiple_of(c * _LANES, _LANES)
            return acc + predicate(keys_ref[:, pl.ds(start, _LANES)], start).astype(jnp.int32)

        acc = jax.lax.fori_loop(0, n_chunks, body, jnp.zeros((rows, _LANES), jnp.int32), unroll=group)
        return jnp.sum(acc, axis=1, keepdims=True)

    # MSB-first over the biased (unsigned-order) key: ``biased ^ INT_MIN`` is the signed threshold.
    def bisect(state):
        bit, biased, done = state
        candidate = biased | jnp.left_shift(jnp.int32(1), 31 - bit)
        threshold = candidate ^ jnp.int32(_INT_MIN)
        n = count(lambda kc, _: kc >= threshold)
        accept = (n >= k) & (done == 0)
        if not exact_key:
            done = jnp.where(accept & (n == k), 1, done)
        return bit + 1, jnp.where(accept, candidate, biased), done

    def unfinished(state):
        bit, _, done = state
        return (bit < 32) & (jnp.min(done) == 0)

    start = (jnp.int32(0), jnp.zeros((rows, 1), jnp.int32), jnp.zeros((rows, 1), jnp.int32))
    _, biased, _ = jax.lax.while_loop(unfinished, bisect, start)
    tau = biased ^ jnp.int32(_INT_MIN)
    tau_ref[...] = jnp.broadcast_to(tau, (rows, _LANES))
    cut_ref[...] = jnp.full((rows, _LANES), width - 1, jnp.int32)

    if not with_cut:
        return

    # A row whose tau was found early selects exactly k keys >= tau, so it takes all of its ties.
    need = k - count(lambda kc, _: kc > tau)
    partial = need < count(lambda kc, _: kc == tau)

    @pl.when(jnp.max(partial.astype(jnp.int32)) > 0)
    def _():
        # Smallest index c with count(key == tau and index <= c) >= need.
        def locate(_, bounds):
            lo, hi = bounds
            mid = jnp.right_shift(lo + hi, 1)
            n = count(lambda kc, start: (kc == tau) & (lane <= mid - start))
            enough = n >= need
            return jnp.where(enough, lo, mid + 1), jnp.where(enough, mid, hi)

        steps = max(1, (width - 1).bit_length())
        init = (jnp.zeros((rows, 1), jnp.int32), jnp.full((rows, 1), width - 1, jnp.int32))
        cut, _ = jax.lax.fori_loop(0, steps, locate, init)
        cut_ref[...] = jnp.broadcast_to(jnp.where(partial, cut, width - 1), (rows, _LANES))


def _select_call(x: Array, ks: Array, *, exact_key: bool, with_cut: bool) -> tuple[Array, Array]:
    """Run :func:`_select_kernel` over a 2-D operand; returns per-row ``(tau, cut)``."""
    num_rows, width = x.shape
    block_rows = select_block_rows(width, x.dtype)
    if block_rows is None:
        raise ValueError(f"topk_select_tpu: width {width} of {x.dtype} does not fit a VMEM row block.")
    padded_width = -(-width // _LANES) * _LANES
    padded_rows = -(-num_rows // block_rows) * block_rows
    if (padded_rows, padded_width) != (num_rows, width):
        # Padded columns are forced to the minimum key in-kernel; padded rows are sliced off.
        x = jnp.pad(x, ((0, padded_rows - num_rows), (0, padded_width - width)))
    ks = jnp.pad(ks.astype(jnp.int32), (0, padded_rows - num_rows), constant_values=1)
    k_lanes = jnp.broadcast_to(ks[:, None], (padded_rows, _LANES))

    tau, cut = pl.pallas_call(
        functools.partial(
            _select_kernel,
            rows=block_rows,
            width=padded_width,
            valid_width=width,
            exact_key=exact_key,
            with_cut=with_cut,
        ),
        grid=(padded_rows // block_rows,),
        in_specs=[
            pl.BlockSpec((block_rows, _LANES), lambda i: (i, 0)),
            pl.BlockSpec((block_rows, padded_width), lambda i: (i, 0)),
        ],
        out_specs=(
            pl.BlockSpec((block_rows, _LANES), lambda i: (i, 0)),
            pl.BlockSpec((block_rows, _LANES), lambda i: (i, 0)),
        ),
        out_shape=(
            jax.ShapeDtypeStruct((padded_rows, _LANES), jnp.int32),
            jax.ShapeDtypeStruct((padded_rows, _LANES), jnp.int32),
        ),
        scratch_shapes=[pltpu.VMEM((block_rows, padded_width), jnp.int32)],
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel",),
            vmem_limit_bytes=_VMEM_LIMIT_BYTES,
        ),
        name="topk_select_threshold",
    )(k_lanes, x)
    return tau[:num_rows, 0], cut[:num_rows, 0]


@ejit(static_argnames=["mode"])
def topk_select_tpu(operand: Array, k: Array, mode: str) -> Array:
    """Sort-free ``jax.lax.top_k`` selection over the last axis of a 2-D float array.

    Args:
        operand: ``[rows, width]`` floating-point scores (at most 32-bit).
        k: ``[rows]`` int32 per-row selection size, already clipped to
            ``[0, width]`` (``"exact_mask"``) or ``[1, width]`` (``"kth"``).
        mode: ``"exact_mask"`` for ``top_k``'s pick set as a boolean mask, or
            ``"kth"`` for ``top_k(operand, k)[0][..., -1]``.

    Returns:
        Bool ``[rows, width]`` with exactly ``k`` true entries per row
        (``"exact_mask"``), or ``[rows]`` of ``operand.dtype`` (``"kth"``).

    Raises:
        ValueError: If ``mode`` is unknown, ``operand`` is not a 2-D float
            array of at most 32 bits, or a row block does not fit VMEM.
    """
    if mode not in SELECT_MODES:
        raise ValueError(f"topk_select_tpu mode must be one of {SELECT_MODES}, got {mode!r}.")
    if operand.ndim != 2 or not jnp.issubdtype(operand.dtype, jnp.floating) or operand.dtype.itemsize > 4:
        raise ValueError(f"topk_select_tpu expects a 2-D float array of <= 32 bits, got {operand.dtype}{operand.shape}.")
    x = jax.lax.stop_gradient(operand)
    k = jnp.asarray(k, jnp.int32).reshape(x.shape[0])
    tau, cut = _select_call(x, k, exact_key=mode == "kth", with_cut=mode == "exact_mask")
    if mode == "kth":
        # Narrower floats were widened exactly, so narrowing the decoded key back is exact.
        return _key_to_value(tau).astype(operand.dtype)
    keys = _monotone_key(x)
    index = jax.lax.broadcasted_iota(jnp.int32, x.shape, 1)
    tau, cut = tau[:, None], cut[:, None]
    return ((keys > tau) | ((keys == tau) & (index <= cut))) & (k[:, None] > 0)
