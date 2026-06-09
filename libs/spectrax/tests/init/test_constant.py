# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.init.constant`."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from spectrax.init.constant import constant, ones, zeros


def test_zeros_fills_zero():
    """:func:`zeros` produces an all-zero array of the requested shape."""
    y = zeros(jax.random.PRNGKey(0), (3, 4))
    assert jnp.array_equal(y, jnp.zeros((3, 4)))


def test_zeros_respects_dtype():
    """:func:`zeros` honors the ``dtype`` keyword."""
    y = zeros(jax.random.PRNGKey(0), (2,), dtype=jnp.float16)
    assert y.dtype == jnp.float16


def test_ones_fills_one():
    """:func:`ones` produces an all-one array of the requested shape."""
    y = ones(jax.random.PRNGKey(0), (3, 4))
    assert jnp.array_equal(y, jnp.ones((3, 4)))


def test_ones_respects_dtype():
    """:func:`ones` honors the ``dtype`` keyword."""
    y = ones(jax.random.PRNGKey(0), (2,), dtype=jnp.int32)
    assert y.dtype == jnp.int32


def test_constant_fills_with_value():
    """:func:`constant` returns an initializer for the specified value."""
    init = constant(7.5)
    y = init(jax.random.PRNGKey(0), (4,))
    assert jnp.all(y == 7.5)


def test_constant_key_is_ignored():
    """The PRNG key does not affect the output of :func:`constant`."""
    init = constant(1)
    a = init(jax.random.PRNGKey(0), (3,))
    b = init(jax.random.PRNGKey(999), (3,))
    assert jnp.array_equal(a, b)
