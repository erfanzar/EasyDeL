# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pooling (reduce-window) primitives on channels-last tensors.

All public functions here operate on inputs with shape ``(N, *spatial, C)``
and apply a window only over the spatial axes (the batch and channel
axes get implicit length-1 windows). Strides default to the window
shape — i.e. non-overlapping pooling, the standard pre-ResNet behavior.

The general entry point :func:`pool` accepts an arbitrary
:func:`jax.lax.reduce_window` reducer; :func:`max_pool` and
:func:`avg_pool` are thin wrappers tuned for the two most common cases.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import jax.lax as lax
import jax.numpy as jnp

from ..core._typing import Array, ArrayLike

__all__ = ["avg_pool", "max_pool", "pool"]

_PadSpec = str | Sequence[tuple[int, int]]


def _pool_window(
    window_shape: Sequence[int],
    strides: Sequence[int] | None,
    x_ndim: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Pad window/stride specs with leading/trailing 1s for batch and channel axes.

    :func:`jax.lax.reduce_window` expects per-axis window/stride sequences
    that include the batch and channel axes. Pooling is conceptually
    spatial-only, so this helper accepts the spatial-only form, pads
    each side with ``1``, and validates that the resulting length matches
    the input rank.

    Args:
        window_shape: Per-spatial-axis window sizes.
        strides: Per-spatial-axis strides; defaults to ``window_shape``.
        x_ndim: Rank of the input array (batch + spatial + channel).

    Returns:
        Tuple ``(window_with_NC, stride_with_NC)`` of length ``x_ndim``.

    Raises:
        ValueError: If ``window_shape`` or ``strides`` does not contain
            exactly one entry per spatial axis.
    """
    ws = (1, *tuple(window_shape), 1)
    if strides is None:
        strides = window_shape
    st = (1, *tuple(strides), 1)
    if len(ws) != x_ndim or len(st) != x_ndim:
        raise ValueError(
            f"window_shape (length {len(window_shape)}) and strides (length {len(st) - 2}) "
            f"must each have one entry per spatial axis ({x_ndim - 2})"
        )
    return ws, st


def pool(
    x: ArrayLike,
    init_value: ArrayLike,
    reduce_fn: Callable[[Array, Array], Array],
    window_shape: Sequence[int],
    *,
    strides: Sequence[int] | None = None,
    padding: _PadSpec = "VALID",
) -> Array:
    """General reduce-window pooling on ``(N, *spatial, C)`` inputs.

    Forwards to :func:`jax.lax.reduce_window` after padding the
    per-spatial-axis ``window_shape`` / ``strides`` with length-1 entries
    for the leading batch axis and trailing channel axis.

    Args:
        x: Input tensor with shape ``(N, *spatial, C)``.
        init_value: Identity element for ``reduce_fn`` (e.g. ``-inf``
            for max-pool, ``0`` for sum/avg-pool). Cast to ``x.dtype``.
        reduce_fn: A commutative, associative binary combiner
            (e.g. :func:`jax.lax.max`, :func:`jax.lax.add`).
        window_shape: Per-spatial-axis window sizes.
        strides: Per-spatial-axis strides. ``None`` -> equal to
            ``window_shape`` (non-overlapping pooling).
        padding: ``"VALID"`` / ``"SAME"`` (string) or per-spatial-axis
            ``(lo, hi)`` integer pairs.

    Returns:
        Pooled tensor of shape ``(N, *spatial_out, C)``.
    """
    xa = jnp.asarray(x)
    ws, st = _pool_window(window_shape, strides, xa.ndim)
    if isinstance(padding, str):
        pad = padding
    else:
        pad = [(0, 0), *[tuple(p) for p in padding], (0, 0)]
    return lax.reduce_window(xa, jnp.asarray(init_value), reduce_fn, ws, st, pad)


def max_pool(
    x: ArrayLike,
    window_shape: Sequence[int],
    *,
    strides: Sequence[int] | None = None,
    padding: _PadSpec = "VALID",
) -> Array:
    """Max-pool ``x`` over ``window_shape``.

    Uses ``-inf`` (in ``x.dtype``) as the reduce-window identity, so any
    real input value beats the padding sentinel.

    Args:
        x: Input tensor of shape ``(N, *spatial, C)``.
        window_shape: Per-spatial-axis window sizes.
        strides: Per-spatial-axis strides; defaults to ``window_shape``.
        padding: See :func:`pool`.

    Returns:
        The max-pooled tensor of shape ``(N, *spatial_out, C)``.
    """
    xa = jnp.asarray(x)
    init = jnp.array(-jnp.inf, dtype=xa.dtype)
    return pool(xa, init, lax.max, window_shape, strides=strides, padding=padding)


def avg_pool(
    x: ArrayLike,
    window_shape: Sequence[int],
    *,
    strides: Sequence[int] | None = None,
    padding: _PadSpec = "VALID",
    count_include_pad: bool = True,
) -> Array:
    """Average-pool ``x`` over ``window_shape``.

    Implemented as a sum-pool divided by the per-window count. When
    ``count_include_pad=True`` (or ``padding == "VALID"``, which has no
    padding to count) the denominator is the constant window volume,
    saving a second reduce-window pass. Otherwise the function reduces a
    second window of ones with the same padding to obtain the per-output
    "true" denominator.

    Args:
        x: Input tensor of shape ``(N, *spatial, C)``.
        window_shape: Per-spatial-axis window sizes.
        strides: Per-spatial-axis strides; defaults to ``window_shape``.
        padding: See :func:`pool`.
        count_include_pad: When ``False`` the average uses only
            non-padded positions in each window. Defaults to ``True``
            (the simpler, divide-by-volume convention).

    Returns:
        The average-pooled tensor of shape ``(N, *spatial_out, C)``.
    """
    xa = jnp.asarray(x)
    summed = pool(xa, jnp.array(0.0, dtype=xa.dtype), lax.add, window_shape, strides=strides, padding=padding)
    window_size = 1
    for w in window_shape:
        window_size *= w
    if count_include_pad or padding == "VALID":
        return summed / window_size
    ones = jnp.ones_like(xa)
    counts = pool(ones, jnp.array(0.0, dtype=xa.dtype), lax.add, window_shape, strides=strides, padding=padding)
    return summed / counts
