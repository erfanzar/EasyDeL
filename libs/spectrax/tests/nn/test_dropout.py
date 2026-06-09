# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.nn.dropout`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import spectrax as spx
from spectrax.core.graph import export
from spectrax.nn.dropout import Dropout
from spectrax.rng.rngs import Rngs


def test_dropout_rate_out_of_range():
    """``rate`` outside ``[0, 1)`` raises :class:`ValueError`."""
    with pytest.raises(ValueError):
        Dropout(-0.1)
    with pytest.raises(ValueError):
        Dropout(1.0)


def test_dropout_rate_zero_is_identity():
    """``rate=0`` is a strict pass-through, even without rngs."""
    d = Dropout(0.0)
    x = jnp.ones((10,))
    assert jnp.array_equal(d(x, rngs=None), x)


def test_dropout_eval_mode_passthrough():
    """In eval mode the layer ignores the mask and returns the input."""
    d = Dropout(0.5)
    d.eval()
    x = jnp.ones((10,))
    assert jnp.array_equal(d(x), x)


def test_dropout_deterministic_override():
    """``deterministic=True`` wins over training mode."""
    d = Dropout(0.5)
    d.train()
    x = jnp.ones((10,))
    assert jnp.array_equal(d(x, deterministic=True), x)


def test_dropout_training_requires_rngs():
    """In training mode a dropout with non-zero rate demands ``rngs``."""
    d = Dropout(0.5)
    with pytest.raises(RuntimeError):
        d(jnp.ones((4,)))


def test_dropout_rejects_non_rngs_type():
    """Passing a non-:class:`Rngs` raises :class:`TypeError`."""
    d = Dropout(0.5)
    with pytest.raises(TypeError):
        d(jnp.ones((4,)), rngs="not an rngs")


def test_dropout_rejects_both_x_and_inputs():
    """The ``inputs`` alias must not silently lose to positional ``x``."""
    d = Dropout(0.0)
    with pytest.raises(TypeError, match="both 'x' and 'inputs'"):
        d(jnp.ones((4,)), inputs=jnp.zeros((4,)))


def test_dropout_training_zeros_some_elements():
    """With ``rate=1-epsilon`` the output has near-all-zero elements."""
    d = Dropout(0.99)
    x = jnp.ones((1000,))
    y = d(x, rngs=Rngs(0))
    zeros = int((y == 0).sum())
    assert zeros > 900


def test_dropout_scales_kept_values_by_one_over_keep():
    """Kept elements are scaled by ``1 / (1 - rate)``."""
    d = Dropout(0.5)
    x = jnp.ones((1000,))
    y = d(x, rngs=Rngs(0))
    nonzero_values = y[y != 0]
    assert jnp.all(jnp.isclose(nonzero_values, 2.0))


def test_dropout_same_rngs_same_mask():
    """Two calls with the same root seed produce the same mask."""
    x = jnp.ones((50,))
    d1 = Dropout(0.5)
    d2 = Dropout(0.5)
    y1 = d1(x, rngs=Rngs(123))
    y2 = d2(x, rngs=Rngs(123))
    assert jnp.array_equal(y1, y2)


def test_dropout_different_rngs_different_mask():
    """Different seeds usually produce different masks."""
    x = jnp.ones((50,))
    y1 = Dropout(0.5)(x, rngs=Rngs(1))
    y2 = Dropout(0.5)(x, rngs=Rngs(2))
    assert not jnp.array_equal(y1, y2)


def test_dropout_constructor_rngs_advance_under_jit():
    """Constructor-owned RNGs remain live mutable state under ``spx.jit``."""
    d = Dropout(0.5, rngs=Rngs(0))
    d.train()

    @spx.jit(mutable="rng")
    def apply(module, x):
        """Apply the module."""
        return module(x)

    x = jnp.ones((32,))
    y1 = apply(d, x)
    y2 = apply(d, x)

    assert not jnp.array_equal(y1, y2)
    state = export(d)[1]
    default_stream = state.get("rng", "rngs.default")
    assert int(default_stream[-1]) == 2


def test_dropout_rate_recorded_as_static():
    """The rate is stored as a static hyperparameter."""
    assert Dropout(0.25)._spx_static["rate"] == 0.25
