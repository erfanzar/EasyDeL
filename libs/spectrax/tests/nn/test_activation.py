# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.nn.activation`."""

from __future__ import annotations

import jax.numpy as jnp

from spectrax.functional import activation as F
from spectrax.nn.activation import GELU, ReLU, Sigmoid, SiLU, Tanh


def test_relu_matches_functional():
    """The :class:`ReLU` layer matches :func:`spectrax.functional.relu`."""
    x = jnp.asarray([-1.0, 0.0, 2.0])
    assert jnp.array_equal(ReLU()(x), F.relu(x))


def test_gelu_exact_matches_functional():
    """Default :class:`GELU` uses the exact form."""
    x = jnp.linspace(-2, 2, 5)
    assert jnp.allclose(GELU()(x), F.gelu(x, approximate=False))


def test_gelu_approximate_flag_respected():
    """``approximate=True`` switches to the tanh approximation."""
    x = jnp.linspace(-2, 2, 5)
    assert jnp.allclose(GELU(approximate=True)(x), F.gelu(x, approximate=True))


def test_gelu_approximate_stored_as_static():
    """The ``approximate`` flag is recorded as a static field."""
    assert GELU(approximate=True)._spx_static["approximate"] is True


def test_silu_matches_functional():
    """The :class:`SiLU` layer matches :func:`spectrax.functional.silu`."""
    x = jnp.linspace(-2, 2, 5)
    assert jnp.allclose(SiLU()(x), F.silu(x))


def test_tanh_matches_functional():
    """The :class:`Tanh` layer matches :func:`spectrax.functional.tanh`."""
    x = jnp.linspace(-2, 2, 5)
    assert jnp.allclose(Tanh()(x), F.tanh(x))


def test_sigmoid_matches_functional():
    """The :class:`Sigmoid` layer matches :func:`spectrax.functional.sigmoid`."""
    x = jnp.linspace(-3, 3, 6)
    assert jnp.allclose(Sigmoid()(x), F.sigmoid(x))


def test_activation_layers_have_no_parameters():
    """Activation layers carry no variables."""
    from spectrax.core.graph import export

    for layer in (ReLU(), GELU(), SiLU(), Tanh(), Sigmoid()):
        g, s = export(layer)
        assert len(s) == 0
        assert g.var_refs == ()
