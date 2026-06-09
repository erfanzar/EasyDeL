# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Symmetric uniform initializer.

The returned values are drawn from ``U(-scale, +scale)``. This is
*unscaled* by fan; for fan-aware uniform initializers see
:func:`spectrax.init.xavier_uniform` and
:func:`spectrax.init.kaiming_uniform`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..core._typing import Array, DType, Initializer, PRNGKey, Shape


def uniform(scale: float = 1.0) -> Initializer:
    """Return a symmetric uniform initializer over ``[-scale, +scale]``.

    Args:
        scale: Half-width of the uniform support. Defaults to ``1.0``,
            i.e. samples in ``[-1, 1]``.

    Returns:
        An :class:`~spectrax.typing.Initializer` returning
        ``jax.random.uniform(key, shape, minval=-scale, maxval=scale)``.
    """

    def init(key: PRNGKey, shape: Shape, dtype: DType = jnp.float32) -> Array:
        """Draw ``jax.random.uniform(key, shape, minval=-scale, maxval=scale)``.

        Args:
            key: Logical key, path segment, or PRNG key used by the operation.
            shape: Array shape requested by the initializer or helper.
            dtype: Array dtype requested for the produced value.

        Returns:
            Result described by this helper.
        """
        return jax.random.uniform(key, shape, dtype=dtype, minval=-scale, maxval=scale)

    return init
