# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for pooling modules and functional forms."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from spectrax.functional.pool import avg_pool, max_pool, pool
from spectrax.nn.pool import (
    AdaptiveAvgPool1d,
    AdaptiveAvgPool2d,
    AvgPool1d,
    AvgPool2d,
    MaxPool1d,
    MaxPool2d,
)


def test_max_pool_1d_manual():
    """Max pool on a 1-D sequence matches per-window max."""
    x = jnp.asarray([[[1.0], [3.0], [2.0], [5.0]]])
    y = max_pool(x, (2,), strides=(2,))
    assert jnp.array_equal(y, jnp.asarray([[[3.0], [5.0]]]))


def test_avg_pool_2d_manual():
    """Avg pool on a 2-D input matches per-window mean."""
    x = jnp.ones((1, 4, 4, 1)) * 2.0
    y = avg_pool(x, (2, 2), strides=(2, 2))
    assert y.shape == (1, 2, 2, 1)
    assert jnp.allclose(y, 2.0)


def test_max_pool_module_1d_output_shape():
    """``MaxPool1d`` module shape."""
    layer = MaxPool1d(kernel_size=2)
    x = jnp.ones((1, 8, 3))
    assert layer(x).shape == (1, 4, 3)


def test_max_pool_module_2d_output_shape():
    """``MaxPool2d`` module shape."""
    layer = MaxPool2d(kernel_size=2)
    x = jnp.ones((1, 8, 8, 3))
    assert layer(x).shape == (1, 4, 4, 3)


def test_max_pool_rejects_count_include_pad_false():
    """``count_include_pad`` is an avg-pool-only option."""
    with pytest.raises(ValueError, match="count_include_pad"):
        MaxPool1d(kernel_size=2, count_include_pad=False)


def test_avg_pool_module_2d_same_padding():
    """``padding='SAME'`` keeps the spatial size when stride=1."""
    layer = AvgPool2d(kernel_size=3, stride=1, padding="SAME")
    x = jnp.ones((1, 5, 5, 2))
    assert layer(x).shape == (1, 5, 5, 2)


def test_avg_pool_count_include_pad_false_normalizes_by_counts():
    """With ``count_include_pad=False`` padded cells are excluded from the denom."""
    x = jnp.ones((1, 2, 2, 1))
    y_true = avg_pool(x, (2, 2), strides=(1, 1), padding="SAME", count_include_pad=False)
    assert jnp.allclose(y_true, jnp.ones_like(y_true))


def test_adaptive_avg_pool_1d_produces_target_size():
    """Adaptive pool hits the user-specified output shape."""
    layer = AdaptiveAvgPool1d(output_size=4)
    x = jnp.ones((1, 16, 2))
    assert layer(x).shape == (1, 4, 2)


def test_adaptive_avg_pool_2d_produces_target_size():
    """AdaptiveAvgPool2d honors 2-D target."""
    layer = AdaptiveAvgPool2d(output_size=(3, 2))
    x = jnp.ones((1, 9, 8, 4))
    assert layer(x).shape == (1, 3, 2, 4)


def test_adaptive_avg_pool_non_divisible_input_hits_exact_target_size():
    """Adaptive pooling uses ragged windows when input is not divisible by output."""
    layer = AdaptiveAvgPool1d(output_size=3)
    x = jnp.arange(5.0).reshape(1, 5, 1)

    y = layer(x)

    assert y.shape == (1, 3, 1)
    assert jnp.allclose(y[0, :, 0], jnp.asarray([0.5, 2.0, 3.5]))


def test_adaptive_avg_pool_rejects_too_large_output_size():
    """Output larger than input raises."""
    layer = AdaptiveAvgPool1d(output_size=8)
    with pytest.raises(ValueError):
        layer(jnp.ones((1, 4, 2)))


def test_pool_general_custom_reducer_matches_sum():
    """:func:`pool` with :func:`jax.lax.add` matches :func:`avg_pool * window_size`."""
    import jax.lax as lax

    x = jnp.ones((1, 4, 1))
    summed = pool(x, jnp.array(0.0), lax.add, (2,), strides=(2,))
    avg = avg_pool(x, (2,), strides=(2,))
    assert jnp.allclose(summed, avg * 2.0)


def test_avg_pool_1d_shape():
    """Module wrapper produces expected shape."""
    layer = AvgPool1d(kernel_size=4, stride=2)
    assert layer(jnp.ones((1, 8, 3))).shape == (1, 3, 3)
