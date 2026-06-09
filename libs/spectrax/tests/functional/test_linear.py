# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :func:`spectrax.functional.linear`."""

from __future__ import annotations

import jax.numpy as jnp

from spectrax.functional.linear import linear


def test_linear_without_bias():
    """Without a bias, ``linear`` reduces to matmul."""
    x = jnp.asarray([[1.0, 2.0, 3.0]])
    w = jnp.eye(3)
    assert jnp.array_equal(linear(x, w), x)


def test_linear_with_bias():
    """Bias is added after matmul."""
    x = jnp.asarray([[1.0, 2.0]])
    w = jnp.asarray([[1.0, 0.0], [0.0, 1.0]])
    b = jnp.asarray([10.0, 20.0])
    assert jnp.array_equal(linear(x, w, b), jnp.asarray([[11.0, 22.0]]))


def test_linear_broadcasts_leading_axes():
    """Leading batch axes pass through unchanged."""
    x = jnp.zeros((3, 4, 2))
    w = jnp.zeros((2, 5))
    assert linear(x, w).shape == (3, 4, 5)


def test_linear_accepts_arraylike_inputs():
    """Python lists / tuples are coerced to arrays."""
    y = linear([[1.0, 2.0]], [[1.0], [1.0]])
    assert jnp.array_equal(y, jnp.asarray([[3.0]]))
