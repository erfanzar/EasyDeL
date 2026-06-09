# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.init.uniform`."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from spectrax.init.uniform import uniform


def test_uniform_bounds():
    """Samples lie in ``[-scale, scale]``."""
    init = uniform(scale=0.5)
    y = init(jax.random.PRNGKey(0), (10_000,))
    assert jnp.all(y >= -0.5)
    assert jnp.all(y <= 0.5)


def test_uniform_default_scale():
    """Default ``scale=1`` yields samples in ``[-1, 1]``."""
    init = uniform()
    y = init(jax.random.PRNGKey(0), (1_000,))
    assert jnp.all(y >= -1.0) and jnp.all(y <= 1.0)


def test_uniform_reproducible():
    """Identical keys yield identical draws."""
    init = uniform()
    a = init(jax.random.PRNGKey(7), (50,))
    b = init(jax.random.PRNGKey(7), (50,))
    assert jnp.array_equal(a, b)
