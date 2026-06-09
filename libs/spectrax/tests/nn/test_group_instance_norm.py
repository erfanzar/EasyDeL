# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :class:`GroupNorm` and :class:`InstanceNorm`."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from spectrax.nn.norm import GroupNorm, InstanceNorm, LayerNorm


def test_group_norm_divisibility():
    """``num_channels % num_groups != 0`` is rejected."""
    with pytest.raises(ValueError):
        GroupNorm(num_groups=3, num_channels=4)


def test_group_norm_output_shape_preserved():
    """GroupNorm preserves input shape."""
    ln = GroupNorm(num_groups=2, num_channels=4)
    x = jnp.ones((1, 3, 4))
    y = ln(x)
    assert y.shape == x.shape


def test_group_norm_one_group_matches_layer_norm_on_flat_channels():
    """``num_groups=1`` computes stats over all channels (and spatial)."""
    key = jnp.asarray([0, 0, 0, 0], dtype=jnp.uint32)
    del key
    x = jnp.asarray([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
    gn = GroupNorm(num_groups=1, num_channels=4, affine=False)
    y = gn(x)
    flat = x.reshape(x.shape[0], -1)
    mean = flat.mean(axis=-1, keepdims=True)
    var = flat.var(axis=-1, keepdims=True)
    ref = (flat - mean) / jnp.sqrt(var + 1e-5)
    ref = ref.reshape(x.shape)
    assert jnp.allclose(y, ref, atol=1e-5)


def test_group_norm_per_channel_affine_applied():
    """Affine scale/bias scale and shift the output."""
    gn = GroupNorm(num_groups=2, num_channels=4)
    gn.weight.value = jnp.asarray([2.0, 2.0, 2.0, 2.0])
    gn.bias.value = jnp.asarray([1.0, 1.0, 1.0, 1.0])
    x = jnp.ones((1, 4))
    y = gn(x)
    assert jnp.allclose(y, jnp.ones_like(y), atol=1e-4)


def test_instance_norm_zero_mean_per_sample_per_channel():
    """InstanceNorm produces zero-mean output over spatial axes."""
    inorm = InstanceNorm(num_channels=3, affine=False)
    x = jnp.arange(2 * 4 * 3.0).reshape((2, 4, 3))
    y = inorm(x)
    mean = y.mean(axis=1)
    assert jnp.allclose(mean, jnp.zeros_like(mean), atol=1e-5)


def test_instance_norm_rejects_wrong_channel_count():
    """InstanceNorm validates channel dim."""
    inorm = InstanceNorm(num_channels=3)
    with pytest.raises(ValueError):
        inorm(jnp.zeros((1, 5, 2)))


def test_group_norm_groups_equal_channels_zero_mean_per_channel():
    """With ``num_groups == num_channels`` each channel is normalized independently across spatial."""
    _ = LayerNorm
    x = jnp.arange(1 * 3 * 4.0).reshape((1, 3, 4))
    gn = GroupNorm(num_groups=4, num_channels=4, affine=False)
    y = gn(x)
    assert jnp.allclose(y.mean(axis=1), jnp.zeros((1, 4)), atol=1e-5)
