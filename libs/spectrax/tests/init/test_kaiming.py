# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.init.kaiming`."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from spectrax.init.kaiming import _gain, kaiming_normal, kaiming_uniform


def test_gain_linear_and_sigmoid():
    """Gain is 1 for ``'linear'`` and ``'sigmoid'``."""
    assert _gain("linear") == 1.0
    assert _gain("sigmoid") == 1.0


def test_gain_tanh():
    """Gain is ``5/3`` for ``'tanh'``."""
    assert _gain("tanh") == 5.0 / 3.0


def test_gain_relu_like():
    """ReLU / GELU / SiLU all get the ``sqrt(2)`` gain."""
    for n in ("relu", "gelu", "silu"):
        assert _gain(n) == math.sqrt(2.0)


def test_gain_unknown_defaults_to_one():
    """Unknown nonlinearity names default to gain 1."""
    assert _gain("mystery") == 1.0


def test_kaiming_uniform_bounds_fan_in():
    """Kaiming-uniform default mode is ``fan_in``."""
    init = kaiming_uniform("relu")
    y = init(jax.random.PRNGKey(0), (4, 6))
    bound = math.sqrt(2.0) * math.sqrt(3.0 / 4)
    assert jnp.all(jnp.abs(y) <= bound + 1e-6)


def test_kaiming_uniform_bounds_fan_out():
    """Mode ``fan_out`` uses the output fan for the bound."""
    init = kaiming_uniform("relu", mode="fan_out")
    y = init(jax.random.PRNGKey(0), (4, 6))
    bound = math.sqrt(2.0) * math.sqrt(3.0 / 6)
    assert jnp.all(jnp.abs(y) <= bound + 1e-6)


def test_kaiming_normal_std():
    """Kaiming-normal std is ``gain / sqrt(fan)``."""
    init = kaiming_normal("relu")
    y = init(jax.random.PRNGKey(0), (200, 200))
    expected = math.sqrt(2.0) / math.sqrt(200)
    assert abs(float(y.std()) - expected) < 0.01


def test_kaiming_reproducibility():
    """Identical seeds reproduce identical draws."""
    init = kaiming_normal()
    a = init(jax.random.PRNGKey(1), (4, 8))
    b = init(jax.random.PRNGKey(1), (4, 8))
    assert jnp.array_equal(a, b)
