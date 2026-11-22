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

"""Grouping utilities for quantization.

This module provides helper functions for group-wise quantization operations,
including codebook quantization and tensor reshaping for per-group statistics.
"""

from __future__ import annotations

import jax
from jax import numpy as jnp


def _quantize_to_codebook_argmin(values: jax.Array, codebook: jax.Array) -> jax.Array:
    """Reference argmin-based codebook quantization.

    For each element of *values*, linearly scans the *codebook* via
    :func:`jax.lax.fori_loop` and returns the index of the closest entry
    by L1 distance.  Memory cost is O(values) rather than O(values * codebook).

    Args:
        values: Input array of arbitrary shape and float dtype.
        codebook: 1-D array of codebook entries.

    Returns:
        uint32 index array with the same shape as *values*.
    """
    min_dist = jnp.abs(values - codebook[0])
    min_idx = jnp.zeros(values.shape, dtype=jnp.int32)

    def _body(i: int, state: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        cur_min_dist, cur_min_idx = state
        dist = jnp.abs(values - codebook[i])
        better = dist < cur_min_dist
        cur_min_dist = jnp.where(better, dist, cur_min_dist)
        cur_min_idx = jnp.where(better, i, cur_min_idx)
        return cur_min_dist, cur_min_idx

    _, min_idx = jax.lax.fori_loop(1, codebook.shape[0], _body, (min_dist, min_idx))
    return min_idx.astype(jnp.uint32)


def _quantize_to_codebook_threshold(values: jax.Array, codebook: jax.Array) -> jax.Array:
    """Threshold bucketization codebook quantization.

    Sorts the codebook once, builds midpoint decision boundaries, then uses
    binary search (``jnp.searchsorted``) to assign each value to the nearest
    codebook bin.  Ties resolve toward the lower bucket (``side="left"``).

    On MPS backends, :func:`jnp.searchsorted` is emulated with a manual binary
    search loop because the native lowering is unreliable.

    Args:
        values: Input float array of arbitrary shape.
        codebook: 1-D float array of codebook entries.

    Returns:
        uint32 index array with the same shape as *values*, mapping each
        element to the index of its nearest codebook entry.
    """
    codebook = codebook.astype(jnp.float32)
    sorted_idx = jnp.argsort(codebook).astype(jnp.int32)
    sorted_vals = codebook[sorted_idx]
    if sorted_vals.shape[0] == 1:
        return jnp.zeros(values.shape, dtype=jnp.uint32)
    boundaries = ((sorted_vals[:-1] + sorted_vals[1:]) * 0.5).astype(jnp.float32)
    return _quantize_to_codebook_threshold_from_map(values, sorted_idx=sorted_idx, boundaries=boundaries)


def _quantize_to_codebook_threshold_from_map(
    values: jax.Array,
    *,
    sorted_idx: jax.Array,
    boundaries: jax.Array,
) -> jax.Array:
    """Threshold quantization using precomputed sorted-index and boundary tensors.

    Fast variant of :func:`_quantize_to_codebook_threshold` that avoids
    re-computing the sort on every call by accepting precomputed tensors from
    :func:`_build_threshold_map` (or the ``_get_*_threshold_map`` helpers).

    Args:
        values: Input float array of arbitrary shape.
        sorted_idx: int32 1-D array mapping sorted-bucket index → original
            codebook index (from :func:`_build_threshold_map`).
        boundaries: float32 1-D array of decision boundaries of length
            ``len(sorted_idx) - 1``.

    Returns:
        uint32 original-codebook-index array with the same shape as *values*.
    """
    values = values.astype(jnp.float32)

    if sorted_idx.shape[0] == 1:
        return jnp.zeros(values.shape, dtype=jnp.uint32)

    if jax.default_backend() == "mps":
        lo = jnp.zeros(values.shape, dtype=jnp.int32)
        hi = jnp.full(values.shape, boundaries.shape[0], dtype=jnp.int32)
        steps = int((boundaries.shape[0] + 1).bit_length())
        for _ in range(steps):
            mid = (lo + hi) >> 1
            mid_vals = jnp.take(boundaries, mid, mode="clip")
            go_left = values <= mid_vals
            hi = jnp.where(go_left, mid, hi)
            lo = jnp.where(go_left, lo, mid + 1)
        bins = lo
    else:
        bins = jnp.searchsorted(boundaries, values, side="left")

    bins = jnp.clip(bins, 0, sorted_idx.shape[0] - 1)
    return sorted_idx[bins].astype(jnp.uint32)


def _quantize_to_codebook(
    values: jax.Array,
    codebook: jax.Array,
    *,
    use_argmin_fallback: bool = False,
    sorted_idx: jax.Array | None = None,
    boundaries: jax.Array | None = None,
) -> jax.Array:
    """Quantize values to nearest codebook entries.

    For each value, finds the index of the closest codebook entry using
    absolute difference as the distance metric.

    Args:
        values: Input values to quantize, arbitrary shape.
        codebook: 1D array of codebook entries to quantize to.

    Returns:
        Array of codebook indices (uint32) with same shape as values.

    Example:
        >>> codebook = jnp.array([-1.0, 0.0, 1.0])
        >>> values = jnp.array([0.3, -0.8, 0.9])
        >>> indices = _quantize_to_codebook(values, codebook)  # [1, 0, 2]
    """
    codebook = jnp.asarray(codebook)
    if codebook.ndim != 1:
        raise ValueError("codebook must be 1D.")
    if codebook.shape[0] == 0:
        raise ValueError("codebook must be non-empty.")

    if use_argmin_fallback:
        return _quantize_to_codebook_argmin(values, codebook)
    if sorted_idx is not None and boundaries is not None:
        return _quantize_to_codebook_threshold_from_map(values, sorted_idx=sorted_idx, boundaries=boundaries)
    return _quantize_to_codebook_threshold(values, codebook)


def _reshape_groups(w: jax.Array, group_size: int) -> tuple[jax.Array, int]:
    """Reshape tensor for group-wise quantization.

    Splits the last dimension into groups of `group_size` elements,
    allowing per-group statistics (min, max, scale) to be computed.

    Args:
        w: Input tensor with at least 2 dimensions. The last dimension
            must be divisible by `group_size`.
        group_size: Number of elements per group.

    Returns:
        Tuple of (reshaped_tensor, n_groups) where:
            - reshaped_tensor has shape (*w.shape[:-1], n_groups, group_size)
            - n_groups is w.shape[-1] // group_size

    Raises:
        ValueError: If w has fewer than 2 dimensions.
        ValueError: If the last dimension is not divisible by group_size.

    Example:
        >>> w = jnp.ones((4, 128))
        >>> w_groups, n_groups = _reshape_groups(w, group_size=32)
        >>> w_groups.shape  # (4, 4, 32)
        >>> n_groups  # 4
    """
    if w.ndim < 2:
        raise ValueError("quantize expects inputs with two or more dimensions.")
    if w.shape[-1] % group_size != 0:
        raise ValueError("The last dimension of `w` must be divisible by `group_size`.")
    n_groups = w.shape[-1] // group_size
    return w.reshape(*w.shape[:-1], n_groups, group_size), n_groups


def _require_bits(bits: int, allowed: set[int]) -> int:
    """Validate and return the bit-width parameter.

    Args:
        bits: Requested bit-width for quantization.
        allowed: Set of allowed bit-width values.

    Returns:
        The validated bit-width as an integer.

    Raises:
        ValueError: If bits is not in the allowed set.

    Example:
        >>> _require_bits(4, {2, 4, 8})  # Returns 4
        >>> _require_bits(5, {2, 4, 8})  # Raises ValueError
    """
    bits = int(bits)
    if bits not in allowed:
        raise ValueError(f"Unsupported bit-width {bits}; allowed: {sorted(allowed)}.")
    return bits
