# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Layer-level normalization primitives.

This module provides functional :func:`layer_norm` (Ba et al., 2016) and
:func:`rms_norm` (Zhang & Sennrich, 2019). Both reduce over a single
*feature* axis (``axis=-1`` by default) and return an array of the same
shape and dtype as the input. Affine parameters are passed in by the
caller — the SpectraX modules in :mod:`spectrax.nn.norm` wrap these
functions and supply learnable parameters.
"""

from __future__ import annotations

import jax.numpy as jnp

from ..core._typing import Array, ArrayLike


def layer_norm(
    x: ArrayLike,
    scale: ArrayLike | None = None,
    bias: ArrayLike | None = None,
    *,
    axis: int = -1,
    eps: float = 1e-5,
) -> Array:
    """Layer normalization (Ba et al., 2016).

    Computes ``(x - mean) / sqrt(var + eps)`` along ``axis``, then
    optionally applies a per-feature affine transform
    ``y = scale * y + bias``. Mean and variance are computed with
    ``keepdims=True`` so they broadcast correctly back over ``x``.

    Args:
        x: Input tensor of any shape.
        scale: Optional per-feature scale applied after normalization;
            broadcastable to ``x.shape``. ``None`` skips scaling.
        bias: Optional per-feature bias applied after scaling;
            broadcastable to ``x.shape``. ``None`` skips the bias.
        axis: Axis to normalize over. Defaults to the last axis (the
            common feature dim).
        eps: Variance floor for numerical stability. Defaults to
            ``1e-5``.

    Returns:
        Normalized (and optionally affine-transformed) tensor with the
        same shape as ``x``.
    """
    xa = jnp.asarray(x)
    mean = jnp.mean(xa, axis=axis, keepdims=True)
    var = jnp.var(xa, axis=axis, keepdims=True)
    y = (xa - mean) * _rsqrt(var + eps)
    if scale is not None:
        y = y * jnp.asarray(scale)
    if bias is not None:
        y = y + jnp.asarray(bias)
    return y


def rms_norm(
    x: ArrayLike,
    scale: ArrayLike | None = None,
    *,
    axis: int = -1,
    eps: float = 1e-6,
) -> Array:
    """Root-mean-square normalization (Zhang & Sennrich, 2019).

    Divides by ``sqrt(mean(x**2) + eps)`` along ``axis`` and optionally
    multiplies by a per-feature scale. Skips the mean-subtraction and
    bias of :func:`layer_norm`, saving one reduction and an addition;
    popular in modern transformer language models (LLaMA, T5,
    PaLM, …).

    Args:
        x: Input tensor of any shape.
        scale: Optional per-feature scale broadcastable to ``x.shape``.
            ``None`` skips scaling.
        axis: Axis to compute the mean-of-squares over. Defaults to the
            last axis.
        eps: Mean-of-squares floor for numerical stability. Defaults to
            ``1e-6``.

    Returns:
        Normalized (and optionally scaled) tensor with the same shape
        as ``x``.
    """
    xa = jnp.asarray(x)
    sq_mean = jnp.mean(xa * xa, axis=axis, keepdims=True)
    y = xa * _rsqrt(sq_mean + eps)
    if scale is not None:
        y = y * jnp.asarray(scale)
    return y


def _rsqrt(x: Array) -> Array:
    """Reciprocal square root used by :func:`layer_norm` / :func:`rms_norm`.

    Implemented as ``1.0 / jnp.sqrt(x)`` rather than via
    :func:`jax.lax.rsqrt` so the autodiff path matches NumPy semantics
    on every backend; XLA still fuses the divide and sqrt on accelerators.

    Args:
        x: Input value consumed by the operation.

    Returns:
        Result described by this helper.
    """
    return 1.0 / jnp.sqrt(x)
