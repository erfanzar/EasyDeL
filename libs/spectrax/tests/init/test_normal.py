# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.init.normal`."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from spectrax.init.normal import normal, truncated_normal


def test_normal_zero_mean_unit_variance():
    """Large samples from default normal have ~0 mean and ~1 variance."""
    init = normal()
    y = init(jax.random.PRNGKey(0), (10_000,))
    assert abs(float(y.mean())) < 0.05
    assert abs(float(y.var()) - 1.0) < 0.1


def test_normal_stddev_and_mean():
    """Scale and shift work correctly."""
    init = normal(stddev=2.0, mean=5.0)
    y = init(jax.random.PRNGKey(0), (10_000,))
    assert abs(float(y.mean()) - 5.0) < 0.1
    assert abs(float(y.std()) - 2.0) < 0.1


def test_normal_shape_and_dtype():
    """Output shape and dtype match the arguments."""
    init = normal()
    y = init(jax.random.PRNGKey(0), (3, 4), dtype=jnp.float16)
    assert y.shape == (3, 4)
    assert y.dtype == jnp.float16


def test_normal_reproducibility():
    """Identical keys yield identical samples."""
    init = normal()
    a = init(jax.random.PRNGKey(42), (100,))
    b = init(jax.random.PRNGKey(42), (100,))
    assert jnp.array_equal(a, b)


def test_truncated_normal_bounded():
    """Truncated-normal samples stay within ``[lower*std, upper*std]``."""
    init = truncated_normal(stddev=1.0, lower=-2.0, upper=2.0)
    y = init(jax.random.PRNGKey(0), (10_000,))
    assert jnp.all(y >= -2.0)
    assert jnp.all(y <= 2.0)


def test_truncated_normal_stddev_scales_output():
    """``stddev`` multiplies the truncated sample."""
    init = truncated_normal(stddev=3.0)
    y = init(jax.random.PRNGKey(0), (1_000,))
    assert float(y.max()) <= 6.0
    assert float(y.min()) >= -6.0
