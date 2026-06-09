# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.nn.linear`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from spectrax.core.variable import Parameter
from spectrax.init import normal, zeros
from spectrax.nn.linear import Bilinear, Linear
from spectrax.rng.rngs import Rngs, resolve_rngs
from spectrax.rng.seed import seed


def test_linear_forward_shape():
    """``Linear(in, out)`` maps ``(..., in)`` to ``(..., out)``."""
    m = Linear(4, 8, rngs=Rngs(0))
    x = jnp.zeros((2, 4))
    assert m(x).shape == (2, 8)


def test_linear_weight_and_bias_allocated():
    """A default :class:`Linear` has ``weight`` and ``bias`` parameters."""
    m = Linear(4, 8, rngs=Rngs(0))
    assert isinstance(m.weight, Parameter)
    assert m.weight.shape == (4, 8)
    assert isinstance(m.bias, Parameter)
    assert m.bias.shape == (8,)


def test_linear_use_bias_false_skips_bias():
    """``use_bias=False`` omits the bias parameter entirely."""
    m = Linear(4, 8, use_bias=False, rngs=Rngs(0))
    assert not hasattr(m, "bias")


def test_linear_explicit_initializers():
    """Custom initializers control the parameter values."""
    m = Linear(4, 8, rngs=Rngs(0), w_init=zeros, b_init=zeros)
    assert jnp.array_equal(m.weight.value, jnp.zeros((4, 8)))
    assert jnp.array_equal(m.bias.value, jnp.zeros((8,)))


def test_linear_explicit_dtype():
    """``dtype`` controls the parameter dtype."""
    m = Linear(4, 8, rngs=Rngs(0), dtype=jnp.float16)
    assert m.weight.dtype == jnp.float16
    assert m.bias.dtype == jnp.float16


def test_linear_axis_names_attached():
    """The weight carries ``("in", "out")`` axis names by default."""
    m = Linear(4, 8, rngs=Rngs(0))
    assert m.weight.axis_names == ("in", "out")


def test_linear_sharding_tuple_normalized():
    """Tuple sharding is normalized onto the weight."""
    m = Linear(4, 8, rngs=Rngs(0), sharding=("dp", "mp"))
    assert m.weight.sharding is not None
    assert m.weight.sharding.axis_names == ("dp", "mp")


def test_linear_forward_matches_matmul():
    """Forward reduces to ``x @ W + b``."""
    m = Linear(3, 2, rngs=Rngs(0))
    x = jnp.asarray([[1.0, 2.0, 3.0]])
    y = m(x)
    ref = x @ m.weight.value + m.bias.value
    assert jnp.allclose(y, ref)


def test_linear_forward_no_bias_is_just_matmul():
    """Without bias the forward is pure matmul."""
    m = Linear(3, 2, use_bias=False, rngs=Rngs(0))
    x = jnp.asarray([[1.0, 2.0, 3.0]])
    ref = x @ m.weight.value
    assert jnp.allclose(m(x), ref)


def test_linear_with_normal_init():
    """Using :func:`spectrax.init.normal` produces non-zero weights."""
    m = Linear(4, 4, rngs=Rngs(0), w_init=normal(stddev=0.1))
    assert not jnp.all(m.weight.value == 0)


def test_linear_same_seed_same_weights():
    """Identical seeds yield identical initializations."""
    a = Linear(4, 4, rngs=Rngs(42))
    b = Linear(4, 4, rngs=Rngs(42))
    assert jnp.array_equal(a.weight.value, b.weight.value)


def test_linear_resolve_rngs_raises_when_no_context():
    """Constructing without rngs outside a seed context raises."""
    with pytest.raises(RuntimeError):
        Linear(4, 4)


def test_linear_resolve_rngs_uses_seed_context():
    """Inside a :func:`spectrax.seed` context, no ``rngs=`` is required."""
    with seed(3):
        m = Linear(4, 4)
    assert m.weight.shape == (4, 8) or m.weight.shape == (4, 4)


def test_linear_resolve_rngs_accepts_int():
    """A bare integer seed is wrapped into :class:`Rngs`."""
    m = Linear(4, 4, rngs=7)
    assert m.weight.value is not None


def test_resolve_rngs_is_passthrough_for_existing():
    """:func:`resolve_rngs` returns an existing :class:`Rngs` unchanged."""
    r = Rngs(1)
    assert resolve_rngs(r) is r


def test_bilinear_forward_shape():
    """:class:`Bilinear` maps ``(..., i)`` and ``(..., j)`` to ``(..., o)``."""
    m = Bilinear(3, 4, 5, rngs=Rngs(0))
    x1 = jnp.zeros((2, 3))
    x2 = jnp.zeros((2, 4))
    assert m(x1, x2).shape == (2, 5)


def test_bilinear_weight_and_bias_shapes():
    """Bilinear weight is 3-D; bias is per-output."""
    m = Bilinear(3, 4, 5, rngs=Rngs(0))
    assert m.weight.shape == (3, 4, 5)
    assert m.bias.shape == (5,)


def test_bilinear_without_bias():
    """``use_bias=False`` drops the bias parameter."""
    m = Bilinear(3, 4, 5, use_bias=False, rngs=Rngs(0))
    assert not hasattr(m, "bias")


def test_bilinear_matches_einsum():
    """Forward reduces to an einsum ``"...i,ijo,...j->...o"`` plus bias."""
    m = Bilinear(2, 3, 4, rngs=Rngs(0))
    x1 = jnp.asarray([[1.0, 2.0]])
    x2 = jnp.asarray([[1.0, 2.0, 3.0]])
    ref = jnp.einsum("...i,ijo,...j->...o", x1, m.weight.value, x2) + m.bias.value
    assert jnp.allclose(m(x1, x2), ref)


def test_bilinear_dtype_controls_parameters():
    """``dtype`` flows to the weight and bias dtypes."""
    m = Bilinear(2, 3, 4, rngs=Rngs(0), dtype=jnp.float16)
    assert m.weight.dtype == jnp.float16
    assert m.bias.dtype == jnp.float16
