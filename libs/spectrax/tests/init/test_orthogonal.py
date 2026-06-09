# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.init.orthogonal`."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from spectrax.init.orthogonal import orthogonal


def test_orthogonal_square_matrix_orthogonal():
    """For a square shape, the weight is approximately orthogonal."""
    init = orthogonal()
    y = init(jax.random.PRNGKey(0), (6, 6))
    identity = y @ y.T
    assert jnp.allclose(identity, jnp.eye(6), atol=1e-4)


def test_orthogonal_tall_matrix_has_orthonormal_columns():
    """Tall matrices have orthonormal columns ``Q^T Q == I``."""
    init = orthogonal()
    y = init(jax.random.PRNGKey(0), (8, 4))
    assert jnp.allclose(y.T @ y, jnp.eye(4), atol=1e-4)


def test_orthogonal_wide_matrix_has_orthonormal_rows():
    """Wide matrices have orthonormal rows ``Q Q^T == I``."""
    init = orthogonal()
    y = init(jax.random.PRNGKey(0), (4, 8))
    assert jnp.allclose(y @ y.T, jnp.eye(4), atol=1e-4)


def test_orthogonal_gain_scales():
    """``gain`` multiplies the resulting matrix."""
    init = orthogonal(gain=2.0)
    y = init(jax.random.PRNGKey(0), (4, 4))
    assert jnp.allclose(y @ y.T, 4.0 * jnp.eye(4), atol=1e-4)


def test_orthogonal_rank_below_two_falls_back_to_gaussian():
    """Rank-0/1 shapes fall back to scaled Gaussian noise."""
    init = orthogonal(gain=3.0)
    y = init(jax.random.PRNGKey(0), (5,))
    assert y.shape == (5,)


def test_orthogonal_reproducibility():
    """Identical seeds yield identical orthogonal matrices."""
    init = orthogonal()
    a = init(jax.random.PRNGKey(1), (5, 5))
    b = init(jax.random.PRNGKey(1), (5, 5))
    assert jnp.array_equal(a, b)
