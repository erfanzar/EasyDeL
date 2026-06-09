# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for the long-tail activation functions added in plan 12."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from spectrax.functional import activation as F


@pytest.mark.parametrize(
    "fn",
    [
        F.leaky_relu,
        F.elu,
        F.selu,
        F.celu,
        F.hard_sigmoid,
        F.hard_tanh,
        F.hard_silu,
        F.hard_swish,
        F.mish,
        F.soft_sign,
        F.log_sigmoid,
    ],
)
def test_scalar_activation_preserves_shape_and_is_finite(fn):
    """Every activation is shape-preserving and produces finite output."""
    x = jnp.linspace(-3.0, 3.0, 7)
    y = fn(x)
    assert y.shape == x.shape
    assert jnp.all(jnp.isfinite(y))


def test_glu_halves_feature_axis():
    """GLU halves the size of the gated axis."""
    x = jnp.ones((2, 4))
    y = F.glu(x, axis=-1)
    assert y.shape == (2, 2)


def test_log_softmax_sums_to_zero_after_exp_along_axis():
    """Exponentiating log-softmax recovers a softmax that sums to 1."""
    x = jnp.asarray([[1.0, 2.0, 3.0]])
    y = F.log_softmax(x, axis=-1)
    probs = jnp.exp(y)
    assert jnp.allclose(probs.sum(axis=-1), 1.0, atol=1e-6)


def test_prelu_with_scalar_alpha_matches_leaky_relu():
    """PReLU with a scalar alpha equals :func:`leaky_relu`."""
    x = jnp.asarray([-1.0, -0.5, 0.0, 0.5, 1.0])
    y = F.prelu(x, jnp.asarray(0.2))
    assert jnp.allclose(y, F.leaky_relu(x, negative_slope=0.2))
