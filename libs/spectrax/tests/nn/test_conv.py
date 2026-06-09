# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.nn.conv`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from spectrax.nn.conv import Conv, Conv1d, Conv2d, Conv3d
from spectrax.rng.rngs import Rngs


def test_conv1d_forward_shape_valid():
    """Conv1d ``VALID`` padding shrinks the spatial dimension by ``k-1``."""
    c = Conv1d(3, 8, kernel_size=3, rngs=Rngs(0))
    x = jnp.zeros((2, 10, 3))
    assert c(x).shape == (2, 8, 8)


def test_conv2d_forward_shape_same_padding():
    """Conv2d with SAME-style padding preserves spatial dimensions."""
    c = Conv2d(3, 4, kernel_size=3, padding=((1, 1), (1, 1)), rngs=Rngs(0))
    x = jnp.zeros((1, 8, 8, 3))
    assert c(x).shape == (1, 8, 8, 4)


def test_conv3d_forward_shape():
    """Conv3d operates on a five-dimensional channels-last input."""
    c = Conv3d(2, 2, kernel_size=2, rngs=Rngs(0))
    x = jnp.zeros((1, 4, 4, 4, 2))
    assert c(x).shape[-1] == 2


def test_conv1d_weight_shape():
    """Conv1d weight shape is ``(k, in/groups, out)``."""
    c = Conv1d(6, 4, kernel_size=5, groups=2, rngs=Rngs(0))
    assert c.weight.shape == (5, 3, 4)


def test_conv2d_kernel_tuple_size():
    """Tuple ``kernel_size`` is accepted and recorded."""
    c = Conv2d(3, 3, kernel_size=(3, 5), rngs=Rngs(0))
    assert c.kernel_size == (3, 5)


def test_conv2d_stride_broadcast():
    """Integer stride broadcasts across all spatial axes."""
    c = Conv2d(3, 3, kernel_size=3, stride=2, rngs=Rngs(0))
    assert c.stride == (2, 2)


def test_conv_dtype():
    """``dtype`` propagates to the weight and bias."""
    c = Conv2d(3, 3, kernel_size=3, rngs=Rngs(0), dtype=jnp.float16)
    assert c.weight.dtype == jnp.float16
    assert c.bias.dtype == jnp.float16


def test_conv_param_dtype_propagates_to_bias_without_compute_dtype():
    """``param_dtype`` should keep conv kernel and bias storage consistent."""
    c = Conv2d(3, 3, kernel_size=3, rngs=Rngs(0), param_dtype=jnp.bfloat16)
    assert c.weight.dtype == jnp.bfloat16
    assert c.bias.dtype == jnp.bfloat16


def test_conv_without_bias():
    """``use_bias=False`` drops the bias parameter."""
    c = Conv1d(3, 3, kernel_size=3, use_bias=False, rngs=Rngs(0))
    assert not hasattr(c, "bias")


def test_conv2d_groups_must_divide_in_channels():
    """Invalid ``groups`` value raises during conv execution."""
    c = Conv2d(4, 4, kernel_size=3, groups=3, rngs=Rngs(0))
    x = jnp.zeros((1, 4, 4, 4))
    with pytest.raises((ValueError, TypeError)):
        c(x)


def test_conv2d_dilation_broadcast():
    """Integer dilation broadcasts across spatial axes."""
    c = Conv2d(3, 3, kernel_size=3, dilation=2, rngs=Rngs(0))
    assert c.dilation == (2, 2)


def test_conv2d_preserves_axis_names_on_weight():
    """Weight carries spatial ``k`` plus ``("in", "out")`` axis names."""
    c = Conv2d(3, 3, kernel_size=3, rngs=Rngs(0))
    assert c.weight.axis_names[-2:] == ("in", "out")


def test_conv_generic_1d_from_int():
    """Generic Conv with int kernel_size acts as Conv1d."""
    c = Conv(3, 8, kernel_size=3, rngs=Rngs(0))
    assert c._N == 1
    assert c.kernel_size == (3,)
    x = jnp.zeros((2, 10, 3))
    assert c(x).shape == (2, 8, 8)


def test_conv_generic_2d_from_tuple():
    """Generic Conv with 2-tuple kernel_size acts as Conv2d."""
    c = Conv(3, 4, kernel_size=(3, 5), rngs=Rngs(0))
    assert c._N == 2
    assert c.kernel_size == (3, 5)
    x = jnp.zeros((1, 8, 8, 3))
    assert c(x).shape == (1, 6, 4, 4)


def test_conv_generic_3d_from_tuple():
    """Generic Conv with 3-tuple kernel_size acts as Conv3d."""
    c = Conv(2, 2, kernel_size=(2, 3, 3), rngs=Rngs(0))
    assert c._N == 3
    assert c.kernel_size == (2, 3, 3)
    x = jnp.zeros((1, 4, 4, 4, 2))
    assert c(x).shape == (1, 3, 2, 2, 2)
