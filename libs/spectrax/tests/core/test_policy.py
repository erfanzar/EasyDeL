# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.core.policy`."""

from __future__ import annotations

import jax.numpy as jnp

from spectrax.core.policy import Policy


def test_policy_defaults_none():
    """Default :class:`Policy` leaves every dtype unset."""
    p = Policy()
    assert p.param_dtype is None
    assert p.compute_dtype is None
    assert p.output_dtype is None


def test_cast_param_identity_when_unset():
    """``cast_param`` returns the value unchanged when ``compute_dtype`` is ``None``."""
    p = Policy()
    x = jnp.ones(3, dtype=jnp.float32)
    assert p.cast_param(x).dtype == jnp.float32


def test_cast_param_casts_to_compute_dtype():
    """``cast_param`` casts to ``compute_dtype`` when set."""
    p = Policy(compute_dtype=jnp.float16)
    x = jnp.ones(3, dtype=jnp.float32)
    assert p.cast_param(x).dtype == jnp.float16


def test_cast_output_identity_when_unset():
    """``cast_output`` returns unchanged when ``output_dtype`` is ``None``."""
    p = Policy()
    x = jnp.ones(2, dtype=jnp.float32)
    assert p.cast_output(x).dtype == jnp.float32


def test_cast_output_casts_to_output_dtype():
    """``cast_output`` casts to the configured output dtype."""
    p = Policy(output_dtype=jnp.bfloat16)
    x = jnp.ones(2, dtype=jnp.float32)
    assert p.cast_output(x).dtype == jnp.bfloat16


def test_storage_dtype_prefers_param_dtype():
    """``storage_dtype`` returns ``param_dtype`` when set."""
    p = Policy(param_dtype=jnp.bfloat16)
    assert p.storage_dtype(jnp.float32) == jnp.bfloat16


def test_storage_dtype_falls_back_to_fallback():
    """``storage_dtype`` returns ``fallback`` when ``param_dtype`` is ``None``."""
    p = Policy()
    assert p.storage_dtype(jnp.float32) == jnp.float32


def test_storage_dtype_falls_back_to_none():
    """``storage_dtype(None)`` returns ``None`` when nothing is set."""
    p = Policy()
    assert p.storage_dtype(None) is None


def test_policy_is_frozen():
    """:class:`Policy` is a frozen dataclass: equal values compare equal."""
    p1 = Policy(param_dtype=jnp.float16, compute_dtype=jnp.float32)
    p2 = Policy(param_dtype=jnp.float16, compute_dtype=jnp.float32)
    assert p1 == p2
