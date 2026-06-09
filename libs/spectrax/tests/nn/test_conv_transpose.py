# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for transposed convolutions (:class:`ConvTranspose{1,2,3}d`)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from spectrax.nn.conv import ConvTranspose1d, ConvTranspose2d, ConvTranspose3d
from spectrax.rng.rngs import Rngs


def test_conv_transpose_1d_shape():
    """Stride-2 upsampling doubles the spatial axis (plus kernel-1)."""
    layer = ConvTranspose1d(3, 5, kernel_size=3, stride=2, rngs=Rngs(0))
    x = jnp.ones((2, 4, 3))
    y = layer(x)
    assert y.shape[0] == 2 and y.shape[-1] == 5
    assert y.shape[1] >= x.shape[1] * 2


def test_conv_transpose_2d_shape():
    """ConvTranspose2d on channels-last ``(N,H,W,C)``."""
    layer = ConvTranspose2d(2, 4, kernel_size=3, stride=2, rngs=Rngs(0))
    y = layer(jnp.ones((1, 4, 4, 2)))
    assert y.shape[0] == 1 and y.shape[-1] == 4


def test_conv_transpose_3d_shape():
    """ConvTranspose3d on ``(N,D,H,W,C)``."""
    layer = ConvTranspose3d(2, 3, kernel_size=3, stride=2, rngs=Rngs(0))
    y = layer(jnp.ones((1, 3, 3, 3, 2)))
    assert y.shape[0] == 1 and y.shape[-1] == 3


def test_conv_transpose_no_bias():
    """``use_bias=False`` omits the bias."""
    layer = ConvTranspose1d(2, 3, kernel_size=3, use_bias=False, rngs=Rngs(0))
    assert not hasattr(layer, "bias")


def test_conv_transpose_gradient_flows():
    """Gradients flow through the transposed kernel."""
    layer = ConvTranspose2d(2, 3, kernel_size=3, rngs=Rngs(0))
    x = jnp.ones((1, 3, 3, 2))

    def loss(W):
        """Compute the loss."""
        layer.weight.value = W
        return layer(x).sum()

    g = jax.grad(loss)(layer.weight.value)
    assert g.shape == layer.weight.value.shape
