# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.functional.norm`."""

from __future__ import annotations

import jax.numpy as jnp

from spectrax.functional.norm import layer_norm, rms_norm


def test_layer_norm_zero_mean_unit_var():
    """LayerNorm yields zero-mean unit-variance output along ``axis``."""
    x = jnp.arange(20.0).reshape(2, 10)
    y = layer_norm(x)
    assert jnp.allclose(jnp.mean(y, axis=-1), 0.0, atol=1e-5)
    assert jnp.allclose(jnp.var(y, axis=-1), 1.0, atol=1e-3)


def test_layer_norm_with_scale_and_bias():
    """Scale and bias are applied after normalization."""
    x = jnp.asarray([[1.0, 2.0, 3.0]])
    y = layer_norm(x, scale=jnp.asarray([2.0, 2.0, 2.0]), bias=jnp.asarray([1.0, 1.0, 1.0]))
    assert y.shape == x.shape


def test_layer_norm_custom_axis():
    """Normalize along an explicit axis."""
    x = jnp.asarray([[1.0, 10.0], [2.0, 20.0]])
    y = layer_norm(x, axis=0)
    assert jnp.allclose(jnp.mean(y, axis=0), 0.0, atol=1e-5)


def test_rms_norm_divides_by_rms():
    """``rms_norm(x)`` equals ``x / sqrt(mean(x**2) + eps)``."""
    x = jnp.asarray([[3.0, 4.0, 0.0, 0.0]])
    y = rms_norm(x)
    manual = x / jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6)
    assert jnp.allclose(y, manual, atol=1e-5)


def test_rms_norm_with_scale():
    """Scale is applied after RMS division."""
    x = jnp.asarray([[3.0, 4.0]])
    y = rms_norm(x, scale=jnp.asarray([2.0, 2.0]))
    manual = x / jnp.sqrt(jnp.mean(x * x, axis=-1, keepdims=True) + 1e-6) * 2.0
    assert jnp.allclose(y, manual, atol=1e-5)


def test_rms_norm_custom_axis():
    """Normalize along an arbitrary axis."""
    x = jnp.asarray([[3.0, 0.0], [4.0, 0.0]])
    y = rms_norm(x, axis=0)
    assert y.shape == x.shape
