# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pointwise activation functions.

Each function is an elementwise map with no parameters of its own and
preserves the input shape and (where it makes sense) dtype. Most are
thin wrappers over :mod:`jax.nn` so that automatic differentiation and
custom JVPs flow through unchanged; a handful (``elu``, ``mish``,
``prelu``) are implemented directly so they can express their default
parameters in pure :mod:`jax.numpy` without an extra dispatch.

These are the building blocks used by :mod:`spectrax.nn` activation
modules; calling them directly is appropriate inside any pure-functional
forward pass.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..core._typing import Array, ArrayLike


def relu(x: ArrayLike) -> Array:
    """Rectified linear unit: ``max(0, x)``.

    Args:
        x: Input array of any shape and floating-point dtype.

    Returns:
        ``jax.nn.relu(x)`` — same shape and dtype as ``x``.
    """
    return jax.nn.relu(x)


def gelu(x: ArrayLike, *, approximate: bool = False) -> Array:
    """Gaussian error linear unit.

    The exact form is ``0.5 * x * (1 + erf(x / sqrt(2)))``; the
    approximate form uses ``0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))``,
    which is faster on accelerators that lack a hardware ``erf`` and
    matches the GPT-2 reference implementation.

    Args:
        x: Input array.
        approximate: When ``True`` use the tanh-based approximation;
            otherwise compute the exact erf form. Defaults to ``False``.

    Returns:
        ``jax.nn.gelu(x, approximate=approximate)``.
    """
    return jax.nn.gelu(x, approximate=approximate)


def silu(x: ArrayLike) -> Array:
    """Sigmoid-weighted linear unit (a.k.a. swish): ``x * sigmoid(x)``.

    Args:
        x: Input array.

    Returns:
        Elementwise ``x * sigmoid(x)`` with the input shape preserved.
    """
    return jax.nn.silu(x)


def tanh(x: ArrayLike) -> Array:
    """Hyperbolic tangent ``(e^x - e^-x) / (e^x + e^-x)``.

    Args:
        x: Input array.

    Returns:
        :func:`jax.numpy.tanh` of ``x``; output saturates at ``[-1, 1]``.
    """
    return jnp.tanh(x)


def sigmoid(x: ArrayLike) -> Array:
    """Logistic sigmoid: ``1 / (1 + exp(-x))``.

    Args:
        x: Input array.

    Returns:
        Elementwise sigmoid in ``(0, 1)``.
    """
    return jax.nn.sigmoid(x)


def softmax(x: ArrayLike, axis: int = -1) -> Array:
    """Numerically-stable softmax along ``axis``.

    Subtracts the per-slice maximum before exponentiating so the
    intermediate ``exp`` is bounded by ``1`` and large magnitudes do
    not overflow.

    Args:
        x: Input array.
        axis: Axis to normalize over. Defaults to the last axis.

    Returns:
        ``jax.nn.softmax(x, axis=axis)``; values sum to ``1`` along
        ``axis``.
    """
    return jax.nn.softmax(x, axis=axis)


def leaky_relu(x: ArrayLike, negative_slope: float = 0.01) -> Array:
    """Leaky ReLU: ``max(0, x) + negative_slope * min(0, x)``.

    Args:
        x: Input array.
        negative_slope: Slope applied to negative inputs. Defaults to
            ``0.01`` (the original LReLU constant).

    Returns:
        Elementwise leaky ReLU, same shape as ``x``.
    """
    return jax.nn.leaky_relu(x, negative_slope=negative_slope)


def elu(x: ArrayLike, alpha: float = 1.0) -> Array:
    """Exponential linear unit: ``x`` if ``x > 0`` else ``alpha * (exp(x) - 1)``.

    Implemented directly with :func:`jax.numpy.where` and
    :func:`jax.numpy.exp` so that ``alpha`` may flow through tracing
    as a Python ``float``.

    Args:
        x: Input array.
        alpha: Negative-saturation level (output approaches
            ``-alpha`` as ``x -> -inf``). Defaults to ``1.0``.

    Returns:
        Elementwise ELU.
    """
    return jnp.where(x > 0, x, alpha * (jnp.exp(x) - 1))


def selu(x: ArrayLike) -> Array:
    """Scaled exponential linear unit (Klambauer et al., 2017).

    Uses fixed scale ``lambda ~ 1.0507`` and ``alpha ~ 1.6733`` so that
    activations preserve unit mean and variance under stacking when
    weights are :func:`spectrax.init.kaiming_normal`-initialized.

    Args:
        x: Input array.

    Returns:
        ``jax.nn.selu(x)``.
    """
    return jax.nn.selu(x)


def celu(x: ArrayLike, alpha: float = 1.0) -> Array:
    """Continuously differentiable ELU: ``max(0, x) + min(0, alpha * (exp(x/alpha) - 1))``.

    Unlike :func:`elu`, the derivative is continuous at ``x = 0`` for
    all ``alpha``.

    Args:
        x: Input array.
        alpha: Negative-saturation level. Defaults to ``1.0``.

    Returns:
        ``jax.nn.celu(x, alpha=alpha)``.
    """
    return jax.nn.celu(x, alpha=alpha)


def glu(x: ArrayLike, axis: int = -1) -> Array:
    """Gated linear unit: split ``x`` in half along ``axis`` and gate.

    The two halves ``a`` and ``b`` along ``axis`` produce ``a * sigmoid(b)``;
    the output has half the size of ``x`` along ``axis``.

    Args:
        x: Input array. Must have an even-sized ``axis``.
        axis: Axis along which to split. Defaults to the last axis.

    Returns:
        ``jax.nn.glu(x, axis=axis)``.
    """
    return jax.nn.glu(x, axis=axis)


def hard_sigmoid(x: ArrayLike) -> Array:
    """Piecewise-linear approximation of the sigmoid.

    Equals ``relu6(x + 3) / 6`` — clamps to ``[0, 1]`` and avoids the
    exponential. Useful on hardware where ``exp`` is expensive (e.g. some
    quantized inference paths).

    Args:
        x: Input array.

    Returns:
        ``jax.nn.hard_sigmoid(x)``.
    """
    return jax.nn.hard_sigmoid(x)


def hard_tanh(x: ArrayLike) -> Array:
    """Clipped linear in ``[-1, 1]`` — i.e. ``clip(x, -1, 1)``.

    Args:
        x: Input array.

    Returns:
        ``jax.nn.hard_tanh(x)``.
    """
    return jax.nn.hard_tanh(x)


def hard_silu(x: ArrayLike) -> Array:
    """Piecewise-linear approximation of silu (swish): ``x * hard_sigmoid(x)``.

    Args:
        x: Input array.

    Returns:
        ``jax.nn.hard_silu(x)``.
    """
    return jax.nn.hard_silu(x)


def hard_swish(x: ArrayLike) -> Array:
    """Piecewise-linear approximation of swish: ``x * hard_sigmoid(x)``.

    Alias for :func:`hard_silu`; the two names refer to the same
    operation. ``swish`` is the original Google name and ``silu`` is the
    convention in PyTorch / JAX.

    Args:
        x: Input array of any shape and floating-point dtype.

    Returns:
        ``jax.nn.hard_silu(x)`` — same shape and dtype as ``x``.
    """
    return jax.nn.hard_silu(x)


def mish(x: ArrayLike) -> Array:
    """Mish activation (Misra, 2019): ``x * tanh(softplus(x))``.

    Implemented directly to keep the differentiable path explicit.

    Args:
        x: Input array.

    Returns:
        Elementwise Mish, same shape as ``x``.
    """
    return jnp.asarray(x) * jnp.tanh(jax.nn.softplus(x))


def soft_sign(x: ArrayLike) -> Array:
    """``x / (1 + |x|)`` — a smooth, bounded alternative to ``tanh``.

    Args:
        x: Input array.

    Returns:
        ``jax.nn.soft_sign(x)``; output is in ``(-1, 1)``.
    """
    return jax.nn.soft_sign(x)


def log_softmax(x: ArrayLike, axis: int = -1) -> Array:
    """Numerically-stable log-softmax along ``axis``.

    Equivalent to ``log(softmax(x, axis))`` but avoids materializing the
    intermediate ``exp`` / ``sum`` separately, which is the standard
    cross-entropy fast path.

    Args:
        x: Input array.
        axis: Axis to normalize over. Defaults to the last axis.

    Returns:
        ``jax.nn.log_softmax(x, axis=axis)``.
    """
    return jax.nn.log_softmax(x, axis=axis)


def log_sigmoid(x: ArrayLike) -> Array:
    """Log of the logistic sigmoid: ``-softplus(-x)``.

    Args:
        x: Input array.

    Returns:
        ``jax.nn.log_sigmoid(x)``; output is in ``(-inf, 0]``.
    """
    return jax.nn.log_sigmoid(x)


def prelu(x: ArrayLike, alpha: ArrayLike) -> Array:
    """Parametric ReLU: ``max(0, x) + alpha * min(0, x)``.

    Unlike :func:`leaky_relu`, the negative slope is a *learned*
    array. ``alpha`` is broadcast against ``x`` using normal NumPy
    broadcasting rules, so it can be a scalar (single channel slope),
    a per-feature vector, or any broadcast-compatible shape.

    Args:
        x: Input array.
        alpha: Per-element (or per-broadcast-unit) negative slope.

    Returns:
        Elementwise PReLU output, same shape as ``x``.
    """
    xa = jnp.asarray(x)
    return jnp.where(xa > 0, xa, jnp.asarray(alpha) * xa)
