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

"""XLA reference for the fused top-k operation.

Two modes, because the callers that matter want different things from the same
reduction:

* ``"values"`` -- sorted top-k values and their indices, for a **static** ``k``.
  This is ``jax.lax.top_k`` semantics exactly, including its tie-break (lower
  index wins), and is what MoE routers and the DeepSeek-V4 DSA indexer consume.
* ``"mask"`` -- a boolean keep-mask for a **per-row dynamic** ``k``, which is
  what sampling top-k filtering needs. Sorting is the wrong tool there: the
  caller only needs "is this logit in the top k of its row", ``k`` differs per
  row so it cannot be static, and materialising sorted values/indices for a
  129k-wide vocab to then throw them away is pure waste.

The mask mode is exact: it takes the ``k``-th largest value per row and
thresholds against it, so ties at the threshold are all kept and a row may hold
more than ``k``. That matches the "ties may return more than k" contract the
sampler's existing threshold search already has, and it is deliberate -- an
exact-``k`` cut would have to impose an arbitrary order on equal logits.

Two sort-free modes serve consumers that need ``top_k``'s selection but not its
order:

* ``"exact_mask"`` -- the pick set of ``jax.lax.top_k`` as a boolean mask:
  exactly ``min(k, n)`` entries per row, every item above the ``k``-th key plus
  the lowest-index items equal to it (``top_k``'s tie-break). Rows with fewer
  than ``k`` finite scores fill with their lowest-index ``-inf`` items, as
  ``top_k`` does. A sparse-attention indexer that turns its selection into a
  mask or an additive bias wants this.
* ``"kth"`` -- ``jax.lax.top_k(x, k)[0][..., -1]``, the selection threshold,
  with the reduced axis removed.

Both bisect the same order-preserving integer key as ``"mask"``; ``k`` may be
static or per-row.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._registry import Backend, Platform, kernel_registry

_MODES = ("values", "mask", "exact_mask", "kth", "filter")


def _sortable_key(x: Array) -> Array:
    """Map floats and integers to unsigned keys in their original value order.

    Args:
        x: Float, integer, or boolean array.

    Returns:
        Unsigned integer keys, same shape as ``x``.
    """
    if jnp.issubdtype(x.dtype, jnp.bool_):
        return x.astype(jnp.uint32)
    if jnp.issubdtype(x.dtype, jnp.integer):
        wide = x.dtype.itemsize > 4
        unsigned = jnp.uint64 if wide else jnp.uint32
        if jnp.issubdtype(x.dtype, jnp.signedinteger):
            signed = jnp.int64 if wide else jnp.int32
            sign_bit = jnp.uint64(1 << 63) if wide else jnp.uint32(1 << 31)
            return jax.lax.bitcast_convert_type(x.astype(signed), unsigned) ^ sign_bit
        return x.astype(unsigned)
    # Flipping negative IEEE-754 bits and setting the sign bit of non-negatives
    # preserves float order in an unsigned comparison. float64 keeps a 64-bit key:
    # narrowing it to float32 would merge distinct values into false ties.
    if x.dtype.itemsize > 4:
        bits = jax.lax.bitcast_convert_type(x.astype(jnp.float64), jnp.uint64)
        sign = (bits >> jnp.uint64(63)).astype(jnp.bool_)
        return jnp.where(sign, ~bits, bits | jnp.uint64(1 << 63))
    bits = jax.lax.bitcast_convert_type(x.astype(jnp.float32), jnp.uint32)
    sign = (bits >> jnp.uint32(31)).astype(jnp.bool_)
    return jnp.where(sign, ~bits, bits | jnp.uint32(0x80000000))


def _key_to_value(key: Array, dtype: jnp.dtype) -> Array:
    """Invert :func:`_sortable_key` bit for bit (NaN payloads and signed zeros included).

    Args:
        key: Unsigned keys produced by :func:`_sortable_key` for ``dtype``.
        dtype: The original operand dtype.

    Returns:
        Values of ``dtype`` whose keys are ``key``.
    """
    dtype = jnp.dtype(dtype)
    if jnp.issubdtype(dtype, jnp.bool_):
        return key.astype(jnp.bool_)
    if jnp.issubdtype(dtype, jnp.integer):
        wide = dtype.itemsize > 4
        if jnp.issubdtype(dtype, jnp.signedinteger):
            signed = jnp.int64 if wide else jnp.int32
            sign_bit = jnp.uint64(1 << 63) if wide else jnp.uint32(1 << 31)
            return jax.lax.bitcast_convert_type(key ^ sign_bit, signed).astype(dtype)
        return key.astype(dtype)
    if dtype.itemsize > 4:
        top = jnp.uint64(1 << 63)
        bits = jnp.where((key & top) != 0, key ^ top, ~key)
        return jax.lax.bitcast_convert_type(bits, jnp.float64).astype(dtype)
    top = jnp.uint32(0x80000000)
    bits = jnp.where((key & top) != 0, key ^ top, ~key)
    # Narrower floats were widened exactly, so narrowing the k-th value back is exact.
    return jax.lax.bitcast_convert_type(bits, jnp.float32).astype(dtype)


def _moveaxis_to_last(x: Array, axis: int) -> tuple[Array, int]:
    """Bring ``axis`` to the end so the reduction is always over the last dim."""
    axis = axis if axis >= 0 else x.ndim + axis
    if axis == x.ndim - 1:
        return x, axis
    return jnp.moveaxis(x, axis, -1), axis


@kernel_registry.register("topk", Platform.XLA, Backend.ANY)
def topk(
    operand: Array,
    k: int | Array,
    axis: int = -1,
    mode: str = "values",
    mask_fill: float | None = None,
) -> tuple[Array, Array] | Array:
    """Exact top-k along ``axis``.

    Args:
        operand: Input array; the reduction runs over ``axis``.
        k: Number kept. Must be a Python ``int`` for ``mode="values"``; may be a
            traced per-row array (broadcastable to ``operand`` minus ``axis``)
            for the other modes.
        axis: Reduction axis. Defaults to the last.
        mode: ``"values"`` for (values, indices); ``"mask"`` for a keep-mask
            that keeps every tie at the threshold; ``"exact_mask"`` for the
            pick set of ``jax.lax.top_k`` (exactly ``min(k, n)`` per row, lower
            index wins ties); ``"kth"`` for the ``k``-th largest value;
            ``"filter"`` for ``operand`` with non-kept entries set to
            ``mask_fill``.
        mask_fill: Replacement for filtered-out entries in ``mode="filter"``.
            Defaults to the dtype's most negative finite value.

    Returns:
        ``(values, indices)`` for ``"values"``; a boolean array shaped like
        ``operand`` for ``"mask"`` and ``"exact_mask"``; ``operand``'s shape
        without ``axis`` for ``"kth"``; an array shaped like ``operand`` for
        ``"filter"``.

    Raises:
        ValueError: If ``mode`` is unknown, ``k`` is traced under
            ``mode="values"`` (a sorted top-k needs a static output width), or
            a static ``k`` is outside ``[1, n]`` under ``mode="kth"``.

    Note:
        ``"mask"`` and ``"filter"`` bisect for the k-th largest value instead of
        sorting. Sorting a row to read one element off it is the wrong
        complexity here: a full sort of a 129k-wide vocab measured 24x SLOWER
        than the threshold search it replaces (31.9 ms against 1.33 ms at
        ``[8, 16384]`` on CPU). The bisection runs on the order-preserving
        integer key of the float, so 32 counting passes land on the exact k-th
        value -- no sort, and no epsilon slop that a float-range bisection would
        leave around ties.

    Note:
        Ties at the threshold are all kept ("may return more than k"), matching
        the contract the sampler's existing search already has.
    """
    if mode not in _MODES:
        raise ValueError(f"unknown topk mode {mode!r}; expected one of {', '.join(map(repr, _MODES))}.")

    moved, _ = _moveaxis_to_last(operand, axis)

    if mode == "values":
        if isinstance(k, Array) and jnp.ndim(k) != 0:
            raise ValueError("mode='values' needs a static int k; use mode='mask' for per-row dynamic k.")
        values, indices = jax.lax.top_k(moved, int(k))
        if axis not in (-1, operand.ndim - 1):
            values = jnp.moveaxis(values, -1, axis)
            indices = jnp.moveaxis(indices, -1, axis)
        return values, indices

    width = moved.shape[-1]
    ks = jnp.asarray(k)
    # Normalise k to one entry per row with a trailing singleton, i.e. shape
    # ``moved.shape[:-1] + (1,)``, so it broadcasts against the reduction axis
    # and the bisection carry below keeps a fixed shape (a scalar k on a rank>=2
    # operand would otherwise grow the fori_loop carry from () to [rows, 1]).
    # Keying off ndim rather than the trailing extent matters when there is
    # exactly one row, where shape (1,) is otherwise indistinguishable from an
    # already-expanded k.
    if ks.ndim == moved.ndim:
        ks = jnp.broadcast_to(ks, (*moved.shape[:-1], 1))
    else:
        ks = jnp.broadcast_to(ks, moved.shape[:-1])[..., None]
    if mode == "kth":
        if isinstance(k, int) and not 1 <= k <= width:
            raise ValueError(f"mode='kth' needs 1 <= k <= {width}, got {k}.")
        # A traced per-row k is clamped: k = 0 has no k-th value.
        ks = jnp.clip(ks, 1, width)
    else:
        ks = jnp.clip(ks, 0, width)

    keys = _sortable_key(moved)
    # Largest key t with count(keys >= t) >= k. c(.) is non-increasing in t, so
    # a bisection over the full unsigned key range converges in 32 or 64 steps.
    lo = jnp.zeros_like(ks, dtype=keys.dtype)
    hi = jnp.full_like(lo, jnp.iinfo(keys.dtype).max)

    def _step(_, bounds):
        lo_i, hi_i = bounds
        # Round upward so the maximum representable key can be selected.
        mid = lo_i + ((hi_i - lo_i) >> 1) + ((hi_i - lo_i) & 1)
        count = (keys >= mid).sum(-1, keepdims=True)
        feasible = count >= ks
        return jnp.where(feasible, mid, lo_i), jnp.where(feasible, hi_i, mid - 1)

    lo, hi = jax.lax.fori_loop(0, keys.dtype.itemsize * 8, _step, (lo, hi))

    if mode == "kth":
        # ``lo`` is the k-th largest key; ``moved.shape[:-1]`` is ``operand``'s shape without ``axis``.
        return _key_to_value(lo[..., 0], operand.dtype)

    if mode == "exact_mask":
        # Every key above the k-th, then the lowest-index keys equal to it until k are taken.
        above = keys > lo
        ties = keys == lo
        need = ks - above.sum(-1, keepdims=True)
        tie_rank = jnp.cumsum(ties.astype(jnp.int32), axis=-1)  # 1-based rank of each tie in index order
        picked = (above | (ties & (tie_rank <= need))) & (ks > 0)
        return jnp.moveaxis(picked, -1, axis) if axis not in (-1, operand.ndim - 1) else picked

    keep = (keys >= lo) & (ks > 0)

    if mode == "mask":
        return jnp.moveaxis(keep, -1, axis) if axis not in (-1, operand.ndim - 1) else keep

    if mask_fill is None:
        if jnp.issubdtype(operand.dtype, jnp.bool_):
            fill = False
        elif jnp.issubdtype(operand.dtype, jnp.integer):
            fill = jnp.iinfo(operand.dtype).min
        else:
            fill = jnp.finfo(operand.dtype).min
    else:
        fill = mask_fill
    out = jnp.where(keep, moved, jnp.asarray(fill, operand.dtype))
    return jnp.moveaxis(out, -1, axis) if axis not in (-1, operand.ndim - 1) else out
