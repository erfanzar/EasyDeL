# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :func:`spectrax.functional.conv`."""

from __future__ import annotations

import jax.numpy as jnp

from spectrax.functional.conv import conv


def test_conv_1d_valid_shape():
    """``VALID`` padding on a 1-D conv shrinks by ``k-1``."""
    x = jnp.zeros((1, 10, 3))
    w = jnp.zeros((3, 3, 4))
    y = conv(x, w)
    assert y.shape == (1, 8, 4)


def test_conv_2d_valid_shape():
    """``VALID`` padding on a 2-D conv shrinks both spatial dims."""
    x = jnp.zeros((2, 6, 6, 2))
    w = jnp.zeros((3, 3, 2, 5))
    y = conv(x, w)
    assert y.shape == (2, 4, 4, 5)


def test_conv_with_bias():
    """Bias broadcasts across spatial axes."""
    x = jnp.zeros((1, 4, 4, 2))
    w = jnp.zeros((1, 1, 2, 3))
    b = jnp.asarray([1.0, 2.0, 3.0])
    y = conv(x, w, b)
    assert jnp.all(y[..., 0] == 1.0)
    assert jnp.all(y[..., 2] == 3.0)


def test_conv_stride_reduces_spatial():
    """Stride reduces the output spatial extent."""
    x = jnp.zeros((1, 8, 8, 1))
    w = jnp.zeros((3, 3, 1, 1))
    y = conv(x, w, stride=2)
    assert y.shape == (1, 3, 3, 1)


def test_conv_dilation_expands_receptive_field():
    """Dilation ``d`` with kernel ``k`` has effective kernel ``1 + (k-1)*d``."""
    x = jnp.zeros((1, 9, 9, 1))
    w = jnp.zeros((3, 3, 1, 1))
    y = conv(x, w, dilation=2)
    assert y.shape == (1, 5, 5, 1)


def test_conv_explicit_padding():
    """Explicit per-axis padding preserves the input shape (SAME-style)."""
    x = jnp.zeros((1, 4, 4, 2))
    w = jnp.zeros((3, 3, 2, 2))
    y = conv(x, w, padding=((1, 1), (1, 1)))
    assert y.shape == (1, 4, 4, 2)


def test_conv_groups():
    """``groups > 1`` implements grouped / depthwise convolutions."""
    x = jnp.zeros((1, 4, 4, 4))
    w = jnp.zeros((3, 3, 2, 4))
    y = conv(x, w, groups=2)
    assert y.shape == (1, 2, 2, 4)
