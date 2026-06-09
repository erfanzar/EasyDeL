# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.nn.mlp`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from spectrax.nn.mlp import MLPBlock
from spectrax.rng.rngs import Rngs


def test_mlp_default_hidden_is_4x_features():
    """Default ``hidden_features`` is ``4 * features``."""
    m = MLPBlock(8, rngs=Rngs(0))
    assert m.hidden_features == 32


def test_mlp_default_out_features_equals_features():
    """Default ``out_features`` is ``features``."""
    m = MLPBlock(8, rngs=Rngs(0))
    assert m.out_features == 8


def test_mlp_forward_shape_default():
    """Default MLP preserves the feature dimension."""
    m = MLPBlock(4, rngs=Rngs(0))
    x = jnp.zeros((2, 4))
    assert m(x).shape == (2, 4)


def test_mlp_custom_hidden_and_out():
    """Explicit ``hidden_features`` and ``out_features`` are respected."""
    m = MLPBlock(4, hidden_features=16, out_features=6, rngs=Rngs(0))
    x = jnp.zeros((1, 4))
    assert m(x).shape == (1, 6)


def test_mlp_unknown_activation_raises():
    """A bogus activation name raises :class:`ValueError`."""
    m = MLPBlock(4, activation="bogus", rngs=Rngs(0))
    with pytest.raises(ValueError):
        m(jnp.zeros((1, 4)))


@pytest.mark.parametrize("act", ["gelu", "relu", "silu"])
def test_mlp_supported_activations(act):
    """Each supported activation produces a finite output."""
    m = MLPBlock(3, activation=act, rngs=Rngs(0))
    y = m(jnp.ones((1, 3)))
    assert jnp.all(jnp.isfinite(y))


def test_mlp_dropout_is_zero_by_default_and_passthrough_eval():
    """Eval mode passes the activations through without RNG."""
    m = MLPBlock(4, dropout=0.5, rngs=Rngs(0))
    m.eval()
    y = m(jnp.ones((1, 4)))
    assert y.shape == (1, 4)


def test_mlp_training_with_dropout_requires_rngs():
    """Training mode with dropout requires an ``rngs`` argument."""
    m = MLPBlock(4, dropout=0.3, rngs=Rngs(0))
    m.train()
    with pytest.raises(RuntimeError):
        m(jnp.ones((1, 4)))


def test_mlp_children_are_linears_and_dropout():
    """The MLP has two :class:`Linear` children and one :class:`Dropout`."""
    from spectrax.nn.dropout import Dropout
    from spectrax.nn.linear import Linear

    m = MLPBlock(4, rngs=Rngs(0))
    assert isinstance(m.fc1, Linear)
    assert isinstance(m.fc2, Linear)
    assert isinstance(m.drop, Dropout)


def test_mlp_exposes_fc_sharding_controls():
    """Constructor sharding kwargs are forwarded to both linear layers."""
    m = MLPBlock(
        4,
        rngs=Rngs(0),
        fc1_sharding=("embed", "tp"),
        fc2_sharding=("tp", "embed"),
        fc1_bias_sharding=("tp",),
        fc2_bias_sharding=("embed",),
    )
    assert m.fc1.weight.sharding is not None
    assert m.fc1.weight.sharding.axis_names == ("embed", "tp")
    assert m.fc1.bias.sharding is not None
    assert m.fc1.bias.sharding.axis_names == ("tp",)
    assert m.fc2.weight.sharding is not None
    assert m.fc2.weight.sharding.axis_names == ("tp", "embed")
    assert m.fc2.bias.sharding is not None
    assert m.fc2.bias.sharding.axis_names == ("embed",)
