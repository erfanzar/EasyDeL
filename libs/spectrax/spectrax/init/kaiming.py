# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Kaiming / He initializers (He et al., 2015).

These scale by ``1 / sqrt(fan)`` (where ``fan`` is fan-in or fan-out)
times a per-nonlinearity gain so that signal variance is preserved
through layers with the chosen activation. Compare
:mod:`spectrax.init.xavier`, which uses the harmonic mean of fan-in
and fan-out.

The companion helper :func:`_gain` returns the ``gain`` constant for a
named nonlinearity; both initializers share the same fan calculation
(:func:`spectrax.init.xavier._fan_in_fan_out`).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from ..core._typing import Array, DType, Initializer, PRNGKey, Shape
from .xavier import _fan_in_fan_out


def _gain(nonlinearity: str) -> float:
    """Return the standard Kaiming gain for a named nonlinearity.

    The gains follow PyTorch's convention:

    * ``"linear"`` / ``"sigmoid"`` -> ``1.0``
    * ``"tanh"`` -> ``5/3``
    * ``"relu"`` / ``"gelu"`` / ``"silu"`` -> ``sqrt(2)``

    Args:
        nonlinearity: Name of the activation following the layer.
            Unknown names silently fall back to ``1.0`` (linear gain),
            so callers can pass arbitrary tags during development without
            crashing.

    Returns:
        The gain as a Python ``float``.
    """
    if nonlinearity in ("linear", "sigmoid"):
        return 1.0
    if nonlinearity == "tanh":
        return 5.0 / 3.0
    if nonlinearity in ("relu", "gelu", "silu"):
        return math.sqrt(2.0)
    return 1.0


def kaiming_uniform(nonlinearity: str = "relu", mode: str = "fan_in") -> Initializer:
    """He / Kaiming uniform initializer.

    Draws from ``U(-bound, +bound)`` with
    ``bound = gain * sqrt(3 / fan)`` where ``gain = _gain(nonlinearity)``
    and ``fan`` is ``fan_in`` or ``fan_out`` depending on ``mode``. The
    factor of ``3`` matches the variance of a uniform on
    ``[-bound, +bound]`` to ``gain**2 / fan``.

    Args:
        nonlinearity: Name of the activation following this layer (see
            :func:`_gain`). Defaults to ``"relu"``.
        mode: Either ``"fan_in"`` (preserve forward-pass variance,
            common for general use) or ``"fan_out"`` (preserve
            backward-pass variance). Defaults to ``"fan_in"``.

    Returns:
        An :class:`~spectrax.typing.Initializer` returning the Kaiming
        uniform draw.
    """
    gain = _gain(nonlinearity)

    def init(key: PRNGKey, shape: Shape, dtype: DType = jnp.float32) -> Array:
        """Uniformly sample scaled by the Kaiming bound.

        Args:
            key: Logical key, path segment, or PRNG key used by the operation.
            shape: Array shape requested by the initializer or helper.
            dtype: Array dtype requested for the produced value.

        Returns:
            Result described by this helper.
        """
        fan_in, fan_out = _fan_in_fan_out(shape)
        fan = fan_in if mode == "fan_in" else fan_out
        bound = gain * math.sqrt(3.0 / max(fan, 1))
        return jax.random.uniform(key, shape, dtype=dtype, minval=-bound, maxval=bound)

    return init


def kaiming_normal(nonlinearity: str = "relu", mode: str = "fan_in") -> Initializer:
    """He / Kaiming normal initializer.

    Draws from ``N(0, std**2)`` with ``std = gain / sqrt(fan)`` where
    ``gain = _gain(nonlinearity)`` and ``fan`` is selected by ``mode``.
    Together with a ReLU nonlinearity this is the standard initialization
    for very deep convolutional networks.

    Args:
        nonlinearity: Name of the activation following this layer (see
            :func:`_gain`). Defaults to ``"relu"``.
        mode: ``"fan_in"`` or ``"fan_out"``. Defaults to ``"fan_in"``.

    Returns:
        An :class:`~spectrax.typing.Initializer` returning the Kaiming
        normal draw.
    """
    gain = _gain(nonlinearity)

    def init(key: PRNGKey, shape: Shape, dtype: DType = jnp.float32) -> Array:
        """Normal sample scaled by the Kaiming std.

        Args:
            key: Logical key, path segment, or PRNG key used by the operation.
            shape: Array shape requested by the initializer or helper.
            dtype: Array dtype requested for the produced value.

        Returns:
            Result described by this helper.
        """
        fan_in, fan_out = _fan_in_fan_out(shape)
        fan = fan_in if mode == "fan_in" else fan_out
        std = gain / math.sqrt(max(fan, 1))
        return jax.random.normal(key, shape, dtype=dtype) * std

    return init
