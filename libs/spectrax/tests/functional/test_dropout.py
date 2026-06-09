# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :func:`spectrax.functional.dropout`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from spectrax.functional.dropout import dropout


def test_dropout_deterministic_passthrough():
    """``deterministic=True`` returns the input unchanged."""
    x = jnp.arange(10.0)
    assert jnp.array_equal(dropout(x, 0.5, deterministic=True), x)


def test_dropout_rate_zero_passthrough():
    """``rate=0`` is also a passthrough (no key required)."""
    x = jnp.ones(5)
    assert jnp.array_equal(dropout(x, 0.0), x)


def test_dropout_requires_key_when_not_deterministic():
    """A non-zero rate without a key raises :class:`ValueError`."""
    with pytest.raises(ValueError):
        dropout(jnp.ones(4), 0.5)


def test_dropout_scales_by_inverse_keep_rate():
    """Kept values are scaled by ``1 / (1 - rate)``."""
    x = jnp.ones(1000)
    y = dropout(x, 0.5, key=jax.random.PRNGKey(0))
    nonzero = y[y != 0]
    assert jnp.all(jnp.isclose(nonzero, 2.0))


def test_dropout_zeros_about_rate_fraction():
    """Roughly ``rate`` fraction of the elements are zeroed."""
    x = jnp.ones(10_000)
    y = dropout(x, 0.3, key=jax.random.PRNGKey(0))
    frac_zero = float((y == 0).mean())
    assert 0.25 < frac_zero < 0.35


def test_dropout_reproducibility():
    """The same key produces the same mask."""
    x = jnp.ones(20)
    key = jax.random.PRNGKey(42)
    a = dropout(x, 0.5, key=key)
    b = dropout(x, 0.5, key=key)
    assert jnp.array_equal(a, b)
