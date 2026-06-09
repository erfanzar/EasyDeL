# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministic constant initializers.

The PRNG key is accepted (because the ``Initializer`` protocol requires
it) but immediately discarded. Useful for biases (:func:`zeros`),
gain/scale parameters (:func:`ones`), or any layer that wants to start
from a specific value (:func:`constant`).
"""

from __future__ import annotations

import jax.numpy as jnp

from ..core._typing import Array, DType, Initializer, PRNGKey, Shape


def constant(value: float | int) -> Initializer:
    """Return an initializer that fills the output with ``value``.

    Args:
        value: The fill value, captured by closure. Cast to the output
            dtype at call time.

    Returns:
        An :class:`~spectrax.typing.Initializer` that ignores its PRNG
        key and returns ``jnp.full(shape, value, dtype)``.
    """

    def init(key: PRNGKey, shape: Shape, dtype: DType = jnp.float32) -> Array:
        """Return ``jnp.full(shape, value, dtype)``; the PRNG key is ignored.

        Args:
            key: Logical key, path segment, or PRNG key used by the operation.
            shape: Array shape requested by the initializer or helper.
            dtype: Array dtype requested for the produced value.

        Returns:
            Return ``jnp.full(shape, value, dtype)``; the PRNG key is ignored.
        """
        del key
        return jnp.full(shape, value, dtype=dtype)

    return init


def zeros(key: PRNGKey, shape: Shape, dtype: DType = jnp.float32) -> Array:
    """All-zeros initializer.

    Implements the :class:`~spectrax.typing.Initializer` protocol
    directly (i.e. it is *not* a factory — pass it as-is, not
    ``zeros()``). The PRNG key is accepted and discarded.

    Args:
        key: PRNG key, ignored.
        shape: Output shape.
        dtype: Output dtype. Defaults to ``jnp.float32``.

    Returns:
        ``jnp.zeros(shape, dtype=dtype)``.
    """
    del key
    return jnp.zeros(shape, dtype=dtype)


def ones(key: PRNGKey, shape: Shape, dtype: DType = jnp.float32) -> Array:
    """All-ones initializer.

    Like :func:`zeros`, this directly implements the initializer
    protocol — pass it as-is rather than ``ones()``. The PRNG key is
    accepted and discarded.

    Args:
        key: PRNG key, ignored.
        shape: Output shape.
        dtype: Output dtype. Defaults to ``jnp.float32``.

    Returns:
        ``jnp.ones(shape, dtype=dtype)``.
    """
    del key
    return jnp.ones(shape, dtype=dtype)
