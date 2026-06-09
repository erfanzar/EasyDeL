# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.init.xavier`."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from spectrax.init.xavier import _fan_in_fan_out, xavier_normal, xavier_uniform


def test_fan_in_fan_out_1d():
    """1-D shape -> both fans equal the single dim."""
    assert _fan_in_fan_out((5,)) == (5, 5)


def test_fan_in_fan_out_2d():
    """2-D weight shape gives direct fan-in / fan-out."""
    assert _fan_in_fan_out((3, 7)) == (3, 7)


def test_fan_in_fan_out_conv():
    """Conv weight shape multiplies by receptive-field size."""
    in_fan, out_fan = _fan_in_fan_out((3, 3, 4, 8))
    assert in_fan == 4 * 9
    assert out_fan == 8 * 9


def test_xavier_uniform_bounds():
    """Samples lie within ``±gain * sqrt(6 / (fan_in + fan_out))``."""
    init = xavier_uniform()
    y = init(jax.random.PRNGKey(0), (4, 6))
    expected_bound = math.sqrt(6.0 / (4 + 6))
    assert jnp.all(jnp.abs(y) <= expected_bound + 1e-6)


def test_xavier_uniform_gain_scales():
    """``gain`` multiplies the effective bound."""
    init = xavier_uniform(gain=2.0)
    y = init(jax.random.PRNGKey(0), (4, 6))
    expected_bound = 2.0 * math.sqrt(6.0 / (4 + 6))
    assert jnp.all(jnp.abs(y) <= expected_bound + 1e-6)


def test_xavier_normal_std_is_gain_sqrt_2_over_fan_sum():
    """Xavier-normal std equals ``gain * sqrt(2 / (fan_in + fan_out))``."""
    init = xavier_normal()
    y = init(jax.random.PRNGKey(0), (200, 200))
    expected_std = math.sqrt(2.0 / 400)
    assert abs(float(y.std()) - expected_std) < 0.005


def test_xavier_reproducibility():
    """Identical seeds yield identical draws."""
    init = xavier_uniform()
    a = init(jax.random.PRNGKey(1), (8, 8))
    b = init(jax.random.PRNGKey(1), (8, 8))
    assert jnp.array_equal(a, b)
