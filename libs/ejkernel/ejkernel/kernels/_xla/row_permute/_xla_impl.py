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

"""Row permutation by contiguous runs: run-table helpers and the XLA gather reference.

A run table ``runs [R, 3] int32`` holds ``(dst_start, src_start, length)`` triples and describes
``out[dst + j] = x[src + j]`` for ``0 <= j < length``. Zero-length runs are padding and are ignored; runs must not
overlap in the destination. Rows no run covers take ``init`` (zeros when no ``init`` is given).

The reference expands the table into a per-row gather index (one ``searchsorted`` over the destination starts) and
gathers, so its autodiff gradient is the gather's (a scatter-add, correct even when source ranges overlap).
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

_PAD_KEY = jnp.iinfo(jnp.int32).max


def sort_runs_by_destination(runs: Array) -> Array:
    """Return ``runs`` ordered by destination start with zero-length (padding) runs last.

    Args:
        runs: ``[R, 3]`` int32 ``(dst_start, src_start, length)``.

    Returns:
        The reordered ``[R, 3]`` int32 table.
    """
    runs = jnp.asarray(runs, jnp.int32)
    key = jnp.where(runs[:, 2] > 0, runs[:, 0], _PAD_KEY)
    return jnp.take(runs, jnp.argsort(key, stable=True), axis=0)


def runs_to_index(runs: Array, rows: int, fill: int = 0) -> tuple[Array, Array]:
    """Expand a run table into the equivalent gather index.

    Args:
        runs: ``[R, 3]`` int32 ``(dst_start, src_start, length)``; destination ranges must not overlap.
        rows: Number of destination rows.
        fill: Index used for rows no run covers.

    Returns:
        ``(index [rows] int32, covered [rows] bool)`` with ``out[i] = x[index[i]]`` wherever ``covered[i]``.
    """
    runs = jnp.asarray(runs, jnp.int32)
    i = jnp.arange(rows, dtype=jnp.int32)
    if runs.shape[0] == 0:
        return jnp.full((rows,), fill, jnp.int32), jnp.zeros((rows,), bool)
    runs = sort_runs_by_destination(runs)
    key = jnp.where(runs[:, 2] > 0, runs[:, 0], _PAD_KEY)
    k = jnp.searchsorted(key, i, side="right").astype(jnp.int32) - 1
    kc = jnp.clip(k, 0, runs.shape[0] - 1)
    dst, src, length = runs[kc, 0], runs[kc, 1], runs[kc, 2]
    covered = (k >= 0) & (length > 0) & (i < dst + length)
    index = jnp.where(covered, i - dst + src, jnp.int32(fill))
    return index.astype(jnp.int32), covered


def invert_runs(runs: Array) -> Array:
    """Return the inverse permutation's run table (``dst`` and ``src`` swapped).

    Args:
        runs: ``[R, 3]`` int32 ``(dst_start, src_start, length)``.

    Returns:
        ``[R, 3]`` int32 ``(src_start, dst_start, length)``.
    """
    runs = jnp.asarray(runs, jnp.int32)
    return jnp.stack([runs[:, 1], runs[:, 0], runs[:, 2]], axis=1)


def resolve_out_rows(x: Array, init: Array | None, out_rows: int | None) -> int:
    """Return the destination row count implied by ``init`` / ``out_rows`` (default ``x.shape[0]``).

    Raises:
        ValueError: If ``init`` disagrees with ``x`` or ``out_rows``.
    """
    if init is not None:
        if init.shape[1:] != x.shape[1:] or init.dtype != x.dtype:
            raise ValueError(f"init must be [out_rows, *x.shape[1:]] in x's dtype, got {init.shape} {init.dtype}")
        if out_rows is not None and int(out_rows) != init.shape[0]:
            raise ValueError(f"out_rows ({out_rows}) disagrees with init rows ({init.shape[0]})")
        return int(init.shape[0])
    return int(x.shape[0] if out_rows is None else out_rows)


def validate_runs(runs: Array) -> None:
    """Raise ``ValueError`` unless ``runs`` is an int32 ``[R, 3]`` table."""
    if runs.dtype != jnp.int32 or runs.ndim != 2 or runs.shape[1] != 3:
        raise ValueError(f"runs must be int32 [R, 3], got {runs.dtype} {runs.shape}")


def row_permute_xla(x: Array, runs: Array, init: Array | None = None, out_rows: int | None = None) -> Array:
    """``out[dst + j] = x[src + j]`` for every run, as one XLA gather.

    Args:
        x: ``[rows, ...]`` source rows (any dtype).
        runs: ``[R, 3]`` int32 ``(dst_start, src_start, length)``.
        init: Optional ``[out_rows, ...]`` values for rows no run covers (zeros when ``None``).
        out_rows: Destination rows (default ``x.shape[0]``, or ``init.shape[0]``).

    Returns:
        ``[out_rows, ...]`` in ``x.dtype``.
    """
    runs = jnp.asarray(runs)
    validate_runs(runs)
    out_rows = resolve_out_rows(x, init, out_rows)
    index, covered = runs_to_index(runs, out_rows)
    gathered = jnp.take(x, index, axis=0, mode="clip")
    base = jnp.zeros((out_rows, *x.shape[1:]), x.dtype) if init is None else init
    mask = covered.reshape((out_rows,) + (1,) * (x.ndim - 1))
    return jnp.where(mask, gathered, base)


__all__ = (
    "invert_runs",
    "resolve_out_rows",
    "row_permute_xla",
    "runs_to_index",
    "sort_runs_by_destination",
    "validate_runs",
)
