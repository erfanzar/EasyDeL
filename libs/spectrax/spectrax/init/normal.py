# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Gaussian (normal / truncated-normal) initializers.

Both :func:`normal` and :func:`truncated_normal` are factories: they
return an :class:`~spectrax.typing.Initializer` capturing the standard
deviation (and bounds, for the truncated variant). The variance scaling
is independent of ``shape`` — for fan-in / fan-out aware scaling use
:func:`spectrax.init.xavier_normal` or :func:`spectrax.init.kaiming_normal`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..core._typing import Array, DType, Initializer, PRNGKey, Shape


def normal(stddev: float = 1.0, mean: float = 0.0) -> Initializer:
    """Return an initializer drawing samples from ``N(mean, stddev**2)``.

    Args:
        stddev: Standard deviation of the Gaussian. Defaults to ``1.0``.
        mean: Mean of the Gaussian. Defaults to ``0.0``.

    Returns:
        An :class:`~spectrax.typing.Initializer` that returns
        ``jax.random.normal(key, shape, dtype) * stddev + mean``.
    """

    def init(key: PRNGKey, shape: Shape, dtype: DType = jnp.float32) -> Array:
        """Draw ``jax.random.normal(key, shape) * stddev + mean``.

        Args:
            key: Logical key, path segment, or PRNG key used by the operation.
            shape: Array shape requested by the initializer or helper.
            dtype: Array dtype requested for the produced value.

        Returns:
            Result described by this helper.
        """
        return jax.random.normal(key, shape, dtype=dtype) * stddev + mean

    return init


def truncated_normal(
    stddev: float = 1.0,
    lower: float = -2.0,
    upper: float = 2.0,
) -> Initializer:
    """Return a truncated-normal initializer.

    Samples are drawn from ``N(0, 1)`` truncated to the open interval
    ``(lower, upper)`` (using :func:`jax.random.truncated_normal`) and
    then scaled by ``stddev``. Note that the scaling is applied
    *outside* the truncation, so the effective bounds of the returned
    samples are ``(lower * stddev, upper * stddev)``.

    Args:
        stddev: Multiplicative scale applied after truncation.
            Defaults to ``1.0``.
        lower: Lower truncation bound on the unit-variance draw.
            Defaults to ``-2.0``.
        upper: Upper truncation bound on the unit-variance draw.
            Defaults to ``2.0``.

    Returns:
        An :class:`~spectrax.typing.Initializer` returning the scaled
        truncated draws.
    """

    def init(key: PRNGKey, shape: Shape, dtype: DType = jnp.float32) -> Array:
        """Draw a truncated-normal sample and scale by ``stddev``.

        Args:
            key: Logical key, path segment, or PRNG key used by the operation.
            shape: Array shape requested by the initializer or helper.
            dtype: Array dtype requested for the produced value.

        Returns:
            Result described by this helper.
        """
        return jax.random.truncated_normal(key, lower, upper, shape, dtype=dtype) * stddev

    return init
