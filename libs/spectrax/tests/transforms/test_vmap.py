# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.transforms.vmap`."""

from __future__ import annotations

import jax.numpy as jnp

import spectrax as spx
from spectrax.nn.linear import Linear
from spectrax.rng.rngs import Rngs


def test_vmap_preserves_leading_axis():
    """vmapped call produces a batched output."""
    m = Linear(4, 4, rngs=Rngs(0))
    fn = spx.vmap(lambda m, x: m(x), in_axes=(None, 0))
    x = jnp.ones((3, 4))
    assert fn(m, x).shape == (3, 4)


def test_vmap_matches_eager_batched():
    """Output equals applying eager module to each row."""
    m = Linear(3, 2, rngs=Rngs(0))
    x = jnp.arange(6.0).reshape(2, 3)
    batched = spx.vmap(lambda m, x: m(x), in_axes=(None, 0))(m, x)
    manual = jnp.stack([m(x[i]) for i in range(x.shape[0])])
    assert jnp.allclose(batched, manual, atol=1e-2, rtol=1e-2)


def test_vmap_decorator_form():
    """The decorator factory form works."""

    @spx.vmap(in_axes=(None, 0))
    def fn(m, x):
        """Helper function."""
        return m(x)

    m = Linear(4, 4, rngs=Rngs(0))
    assert fn(m, jnp.ones((5, 4))).shape == (5, 4)


def test_vmap_mutable_empty_is_passthrough():
    """Without batch-mutating layers, an empty ``mutable`` works."""
    m = Linear(4, 4, rngs=Rngs(0))
    out = spx.vmap(lambda m, x: m(x), in_axes=(None, 0))(m, jnp.ones((3, 4)))
    assert out.shape == (3, 4)


def test_vmap_axis_name_reducer():
    """``axis_name`` enables collective reductions inside the mapped fn."""
    import jax

    def meaner(m, x):
        """Compute the mean."""
        y = m(x)
        return jax.lax.pmean(y, axis_name="batch")

    m = Linear(4, 4, rngs=Rngs(0))
    out = spx.vmap(meaner, in_axes=(None, 0), axis_name="batch")(m, jnp.ones((3, 4)))
    assert out.shape == (3, 4)
    assert jnp.allclose(out[0], out[1])
    assert jnp.allclose(out[0], out[2])
