# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for dynamic deferred (lazy) parameter initialization."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import spectrax as spx
from spectrax.nn import Conv1d, Conv2d, Conv3d, Embed, Linear
from spectrax.rng.rngs import Rngs


def test_linear_deferred_materializes_on_first_call():
    """``Linear(in_features=None)`` resolves shape and materializes on first call."""
    m = Linear(None, 8, rngs=Rngs(0))
    x = jnp.zeros((2, 16))
    y = m(x)
    assert y.shape == (2, 8)
    assert m.in_features == 16
    assert not isinstance(m.weight, spx.DeferredParameter) or m.weight.is_materialized


def test_linear_deferred_output_shape():
    """Materialized output has the expected shape."""
    m = Linear(None, 4, rngs=Rngs(0))
    y = m(jnp.zeros((2, 7)))
    assert y.shape == (2, 4)
    assert m.in_features == 7


def test_linear_deferred_under_transform_raises():
    """Running deferred materialization inside ``jit`` raises."""
    m = Linear(None, 4, rngs=Rngs(0))
    fn = spx.jit(lambda m, x: m(x))
    with pytest.raises(spx.LazyInitUnderTransformError):
        fn(m, jnp.zeros((2, 4)))


def test_linear_deferred_preserves_hyperparameters():
    """``use_bias`` and ``out_features`` carry through to the concrete layer."""
    m = Linear(None, 6, use_bias=False, rngs=Rngs(0))
    m(jnp.zeros((1, 3)))
    assert m.out_features == 6
    assert m.use_bias is False


def test_embed_deferred_infers_vocab():
    """``Embed(num_embeddings=None)`` infers ``num_embeddings`` from the first input."""
    e = Embed(None, 8, rngs=Rngs(0))
    ids = jnp.asarray([0, 3, 5])
    y = e(ids)
    assert y.shape == (3, 8)
    assert e.num_embeddings == 6


def test_embed_deferred_under_transform_raises():
    """``Embed(num_embeddings=None)`` under ``jit`` raises."""
    e = Embed(None, 4, rngs=Rngs(0))
    fn = spx.jit(lambda m, ids: m(ids))
    with pytest.raises(spx.LazyInitUnderTransformError):
        fn(e, jnp.asarray([0, 1, 2]))


def test_conv1d_deferred_materializes():
    """``Conv1d(in_channels=None)`` materializes on first call."""
    c = Conv1d(None, 4, kernel_size=3, rngs=Rngs(0))
    y = c(jnp.zeros((1, 8, 2)))
    assert c.in_channels == 2
    assert y.shape[-1] == 4


def test_conv2d_deferred_materializes():
    """``Conv2d(in_channels=None)`` materializes on first call."""
    c = Conv2d(None, 3, kernel_size=3, rngs=Rngs(0))
    y = c(jnp.zeros((1, 8, 8, 2)))
    assert c.in_channels == 2
    assert y.shape[-1] == 3


def test_conv3d_deferred_materializes():
    """``Conv3d(in_channels=None)`` materializes on first call."""
    c = Conv3d(None, 2, kernel_size=2, rngs=Rngs(0))
    y = c(jnp.zeros((1, 4, 4, 4, 3)))
    assert c.in_channels == 3
    assert y.shape[-1] == 2


def test_conv_deferred_under_transform_raises():
    """Deferred conv under ``jit`` raises."""
    c = Conv2d(None, 3, kernel_size=3, rngs=Rngs(0))
    fn = spx.jit(lambda m, x: m(x))
    with pytest.raises(spx.LazyInitUnderTransformError):
        fn(c, jnp.zeros((1, 8, 8, 2)))


def test_deferred_layers_preserve_sharding_on_materialization():
    """Deferred parameters keep sharding metadata when they materialize."""
    embed = Embed(None, 4, rngs=Rngs(0), sharding=("vocab", "tp"))
    conv = Conv1d(
        None,
        4,
        kernel_size=3,
        rngs=Rngs(0),
        sharding=("kernel", "in", "tp"),
        bias_sharding=("tp",),
    )
    embed(jnp.asarray([0, 1, 2]))
    conv(jnp.zeros((1, 8, 2)))
    assert embed.weight.sharding is not None
    assert embed.weight.sharding.axis_names == ("vocab", "tp")
    assert conv.weight.sharding is not None
    assert conv.weight.sharding.axis_names == ("kernel", "in", "tp")
    assert conv.bias.sharding is not None
    assert conv.bias.sharding.axis_names == ("tp",)


def test_module_materialize_explicit():
    """``Module.materialize()`` initializes all deferred descendants."""

    class _Stack(spx.Module):
        """Helper module for testing."""

        def __init__(self):
            """Initialize with fc1, fc2."""
            super().__init__()
            self.fc1 = Linear(None, 5, rngs=Rngs(0))
            self.fc2 = Linear(None, 2, rngs=Rngs(1))

        def forward(self, x):
            """Run the forward pass."""
            return self.fc2(self.fc1(x))

    model = _Stack()
    model(jnp.zeros((1, 3)))
    model.materialize()
    assert model.fc1.in_features == 3
    assert model.fc2.in_features == 5
    assert model(jnp.zeros((1, 3))).shape == (1, 2)


def test_sequential_init_materializes_deferred():
    """``sequential_init`` resolves deferred parameters and materializes them."""

    class _Net(spx.Module):
        """Helper model module for testing."""

        def __init__(self):
            """Initialize with fc1, fc2."""
            super().__init__()
            self.fc1 = Linear(None, 8, rngs=Rngs(0))
            self.fc2 = Linear(8, 1, rngs=Rngs(1))

        def forward(self, x):
            """Run the forward pass."""
            return self.fc2(self.fc1(x))

    model = _Net()
    model.sequential_init(jnp.zeros((2, 4)))
    assert model.fc1.in_features == 4
    assert not isinstance(model.fc1.weight, spx.DeferredParameter) or model.fc1.weight.is_materialized
