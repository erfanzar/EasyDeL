# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.functional.activation`."""

from __future__ import annotations

import jax.numpy as jnp

from spectrax.functional.activation import gelu, relu, sigmoid, silu, softmax, tanh


def test_relu_zeros_negatives():
    """ReLU zeros every strictly-negative entry."""
    x = jnp.asarray([-2.0, -0.01, 0.0, 3.0])
    assert jnp.array_equal(relu(x), jnp.asarray([0.0, 0.0, 0.0, 3.0]))


def test_gelu_approximate_matches_tanh_form():
    """Approximate GELU is close to the exact form within a small delta."""
    x = jnp.linspace(-3, 3, 7)
    approx = gelu(x, approximate=True)
    exact = gelu(x, approximate=False)
    assert jnp.allclose(approx, exact, atol=1e-2)


def test_silu_formula():
    """``silu(x) == x * sigmoid(x)``."""
    x = jnp.linspace(-2, 2, 5)
    assert jnp.allclose(silu(x), x * sigmoid(x), atol=1e-6)


def test_tanh_bounded_in_minus_one_one():
    """``tanh`` outputs are in ``[-1, 1]`` (saturates at the extremes)."""
    y = tanh(jnp.asarray([-10.0, 0.0, 10.0]))
    assert jnp.all((y >= -1) & (y <= 1))
    assert float(y[1]) == 0.0


def test_sigmoid_zero_yields_half():
    """``sigmoid(0) == 0.5``."""
    assert jnp.isclose(sigmoid(jnp.asarray(0.0)), 0.5)


def test_softmax_sums_to_one():
    """Softmax along the last axis sums to 1."""
    x = jnp.asarray([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    y = softmax(x)
    assert jnp.allclose(jnp.sum(y, axis=-1), 1.0)


def test_softmax_custom_axis():
    """``axis=0`` normalizes across rows."""
    x = jnp.asarray([[1.0, 2.0], [3.0, 4.0]])
    y = softmax(x, axis=0)
    assert jnp.allclose(jnp.sum(y, axis=0), 1.0)
