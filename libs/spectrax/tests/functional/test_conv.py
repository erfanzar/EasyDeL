# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :func:`spectrax.functional.conv`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
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


@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("groups", [1, 2])
def test_conv_highest_matches_numpy(rank, groups):
    """Explicit precision wins over a low ambient setting for grouped N-D convs."""
    rng = np.random.default_rng(23)
    x = rng.normal(size=(1, *((5,) * rank), 8)).astype(np.float32)
    w = rng.normal(size=(*((2,) * rank), 8 // groups, 6)).astype(np.float32)
    b = rng.normal(size=(6,)).astype(np.float32)
    # Independent VALID convolution: stride=2, dilation=2 gives two outputs per axis.
    expected = np.empty((1, *((2,) * rank), 6), dtype=np.float64)
    for pos in np.ndindex(*((2,) * rank)):
        window = tuple(slice(2 * p, 2 * p + 3, 2) for p in pos)
        patch = x[(0, *window, slice(None))].astype(np.float64)
        for group in range(groups):
            inputs = slice(group * (8 // groups), (group + 1) * (8 // groups))
            outputs = slice(group * (6 // groups), (group + 1) * (6 // groups))
            expected[(0, *pos, outputs)] = (
                patch[..., inputs].reshape(-1) @ w[..., outputs].astype(np.float64).reshape(-1, 6 // groups) + b[outputs]
            )
    with jax.default_matmul_precision("bfloat16"):
        actual = jax.jit(
            lambda a, k: conv(a, k, b, stride=2, dilation=2, groups=groups, precision=jax.lax.Precision.HIGHEST)
        )(jnp.asarray(x), jnp.asarray(w))
    assert actual.dtype == jnp.float32
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("ambient", [None, "bfloat16", "float32"])
def test_conv_default_precision_follows_ambient(ambient):
    """Omitting precision and passing None preserve the original JAX contract."""
    rng = np.random.default_rng(41)
    x = jnp.asarray(rng.normal(size=(1, 5, 5, 16)).astype(np.float32))
    w = jnp.asarray(rng.normal(size=(2, 2, 16, 4)).astype(np.float32))
    with jax.default_matmul_precision(ambient):
        omitted = jax.jit(lambda a, k: conv(a, k))(x, w)
        explicit_none = jax.jit(lambda a, k: conv(a, k, precision=None))(x, w)
        # Reference: raw lax.conv_general_dilated under the same ambient precision.
        expected = jax.jit(
            lambda a, k: jax.lax.conv_general_dilated(a, k, (1, 1), "VALID", dimension_numbers=("NHWC", "HWIO", "NHWC"))
        )(x, w)
    np.testing.assert_array_equal(omitted, expected)
    np.testing.assert_array_equal(explicit_none, expected)
