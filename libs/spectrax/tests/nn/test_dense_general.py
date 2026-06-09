# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :class:`DenseGeneral` and :class:`Einsum`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from spectrax.nn.dense import DenseGeneral, Einsum
from spectrax.rng.rngs import Rngs


def test_dense_general_single_axis_matches_linear_semantics():
    """Contracting the last axis with scalar features behaves like Linear."""
    layer = DenseGeneral(features=6, axis=-1, in_shape=(4,), rngs=Rngs(0))
    x = jnp.ones((2, 3, 4))
    y = layer(x)
    assert y.shape == (2, 3, 6)


def test_dense_general_multi_axis_output_shape():
    """Multi-axis contraction + multi-axis features gives the expected shape."""
    layer = DenseGeneral(features=(4, 8), axis=(-2, -1), in_shape=(2, 6), rngs=Rngs(0))
    x = jnp.zeros((3, 5, 2, 6))
    y = layer(x)
    assert y.shape == (3, 5, 4, 8)


def test_dense_general_forward_matches_tensordot():
    """Forward exactly reproduces the tensordot+bias recipe."""
    layer = DenseGeneral(features=5, axis=-1, in_shape=(3,), rngs=Rngs(0))
    x = jnp.asarray([[1.0, 2.0, 3.0]])
    ref = x @ layer.weight.value + layer.bias.value
    assert jnp.allclose(layer(x), ref, atol=1e-6)


def test_dense_general_no_bias_has_no_bias_attribute():
    """``use_bias=False`` skips the bias."""
    layer = DenseGeneral(features=4, axis=-1, in_shape=(3,), use_bias=False, rngs=Rngs(0))
    assert not hasattr(layer, "bias")


def test_dense_general_requires_in_shape():
    """Without ``in_shape`` the constructor raises."""
    with pytest.raises(ValueError):
        DenseGeneral(features=4, axis=-1, rngs=Rngs(0))


def test_dense_general_rejects_mismatched_in_shape_length():
    """``len(in_shape)`` must match ``len(axis)``."""
    with pytest.raises(ValueError):
        DenseGeneral(features=4, axis=(-2, -1), in_shape=(3,), rngs=Rngs(0))


def test_einsum_forward_shape():
    """Einsum weight is allocated with the supplied shape."""
    layer = Einsum("...ij,jk->...ik", shape=(4, 8), rngs=Rngs(0))
    y = layer(jnp.zeros((3, 2, 4)))
    assert y.shape == (3, 2, 8)


def test_einsum_matches_direct_einsum_call():
    """Forward is literally :func:`jnp.einsum` on the parameters."""
    layer = Einsum("...ij,jk->...ik", shape=(3, 5), rngs=Rngs(0))
    x = jnp.ones((2, 3))
    ref = jnp.einsum("...ij,jk->...ik", x, layer.weight.value)
    assert jnp.allclose(layer(x), ref)


def test_einsum_requires_arrow():
    """Missing ``->`` raises."""
    with pytest.raises(ValueError):
        Einsum("abc", shape=(3,), rngs=Rngs(0))


def test_einsum_bias_requires_bias_shape():
    """``use_bias=True`` without ``bias_shape`` raises."""
    with pytest.raises(ValueError):
        Einsum("...i,ij->...j", shape=(3, 4), use_bias=True, rngs=Rngs(0))


def test_einsum_with_bias_adds_bias():
    """Bias is broadcast-added."""
    layer = Einsum(
        "...i,ij->...j",
        shape=(3, 4),
        use_bias=True,
        bias_shape=(4,),
        rngs=Rngs(0),
    )
    y = layer(jnp.zeros((2, 3)))
    assert jnp.allclose(y, layer.bias.value)


def test_dense_general_gradient_flows():
    """Gradients flow through the weight and bias."""
    layer = DenseGeneral(features=3, axis=-1, in_shape=(2,), rngs=Rngs(0))
    x = jnp.ones((1, 2))

    def loss(W):
        """Compute the loss."""
        layer.weight.value = W
        return layer(x).sum()

    g = jax.grad(loss)(layer.weight.value)
    assert g.shape == layer.weight.value.shape


def test_dense_general_and_einsum_accept_explicit_sharding():
    """General dense layers expose direct sharding hooks."""
    dense = DenseGeneral(
        features=3,
        axis=-1,
        in_shape=(2,),
        rngs=Rngs(0),
        sharding=("contract", "tp"),
        bias_sharding=("tp",),
    )
    einsum = Einsum(
        "...i,ij->...j",
        shape=(2, 3),
        use_bias=True,
        bias_shape=(3,),
        rngs=Rngs(0),
        sharding=("contract", "tp"),
        bias_sharding=("tp",),
    )
    assert dense.weight.sharding is not None
    assert dense.weight.sharding.axis_names == ("contract", "tp")
    assert dense.bias.sharding is not None
    assert dense.bias.sharding.axis_names == ("tp",)
    assert einsum.weight.sharding is not None
    assert einsum.weight.sharding.axis_names == ("contract", "tp")
    assert einsum.bias.sharding is not None
    assert einsum.bias.sharding.axis_names == ("tp",)
