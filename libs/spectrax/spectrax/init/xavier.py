# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Xavier / Glorot initializers (Glorot & Bengio, 2010).

Both variants scale the random draw so the per-layer signal variance is
preserved on average through a stack of linear layers, using the
fan-in *and* fan-out of the weight tensor. Compare with
:mod:`spectrax.init.kaiming` which scales by only ``fan_in`` (or
``fan_out``).

The companion helper :func:`_fan_in_fan_out` infers fans from a weight
shape and is shared with the Kaiming initializers.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from ..core._typing import Array, DType, Initializer, PRNGKey, Shape


def _fan_in_fan_out(shape: Shape) -> tuple[int, int]:
    """Compute ``(fan_in, fan_out)`` for a weight of the given ``shape``.

    Follows the conventions used by mainstream frameworks
    (PyTorch / Flax / TF):

    * Scalar shape ``()`` — both fans collapse to ``1``.
    * 1-D shape ``(n,)`` — both fans equal ``n`` (typical of bias-like
      vectors that have no clear in/out role).
    * 2-D shape ``(in, out)`` — dense-layer fans.
    * 3-D-or-higher shape ``(*kernel_spatial, in, out)`` — convolution
      fans, with the receptive-field volume (product of leading kernel
      dims) folded into both ``fan_in`` and ``fan_out``.

    Args:
        shape: The weight tensor's shape.

    Returns:
        A ``(fan_in, fan_out)`` pair of Python ``int``\\s.
    """
    if len(shape) == 0:
        return 1, 1
    if len(shape) == 1:
        return shape[0], shape[0]
    if len(shape) > 2:
        receptive = 1
        for s in shape[:-2]:
            receptive *= s
        fan_in = shape[-2] * receptive
        fan_out = shape[-1] * receptive
    else:
        fan_in = shape[-2]
        fan_out = shape[-1]
    return fan_in, fan_out


def xavier_uniform(gain: float = 1.0) -> Initializer:
    """Glorot-uniform initializer (Glorot & Bengio, 2010).

    Draws from ``U(-a, +a)`` with
    ``a = gain * sqrt(6 / (fan_in + fan_out))`` so the unit-input variance
    is approximately preserved through linear / convolutional layers.

    Args:
        gain: Multiplicative gain applied to the bound; commonly set to
            ``1.0`` for linear/sigmoid layers and ``sqrt(2)`` for ReLU.
            Defaults to ``1.0``.

    Returns:
        An :class:`~spectrax.typing.Initializer` returning the Glorot
        uniform draw.
    """

    def init(key: PRNGKey, shape: Shape, dtype: DType = jnp.float32) -> Array:
        """Uniformly sample scaled by the Glorot gain.

        Args:
            key: Logical key, path segment, or PRNG key used by the operation.
            shape: Array shape requested by the initializer or helper.
            dtype: Array dtype requested for the produced value.

        Returns:
            Result described by this helper.
        """
        fan_in, fan_out = _fan_in_fan_out(shape)
        a = gain * math.sqrt(6.0 / (fan_in + fan_out))
        return jax.random.uniform(key, shape, dtype=dtype, minval=-a, maxval=a)

    return init


def xavier_normal(gain: float = 1.0) -> Initializer:
    """Glorot-normal initializer (Glorot & Bengio, 2010).

    Draws from ``N(0, std**2)`` with
    ``std = gain * sqrt(2 / (fan_in + fan_out))`` — the normal counterpart
    of :func:`xavier_uniform`.

    Args:
        gain: Multiplicative gain applied to the standard deviation.
            Defaults to ``1.0``.

    Returns:
        An :class:`~spectrax.typing.Initializer` returning the Glorot
        normal draw.
    """

    def init(key: PRNGKey, shape: Shape, dtype: DType = jnp.float32) -> Array:
        """Normal sample scaled by the Glorot gain.

        Args:
            key: Logical key, path segment, or PRNG key used by the operation.
            shape: Array shape requested by the initializer or helper.
            dtype: Array dtype requested for the produced value.

        Returns:
            Result described by this helper.
        """
        fan_in, fan_out = _fan_in_fan_out(shape)
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        return jax.random.normal(key, shape, dtype=dtype) * std

    return init
