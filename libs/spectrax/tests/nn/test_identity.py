# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.nn.identity`."""

from __future__ import annotations

import jax.numpy as jnp

from spectrax.nn.identity import Identity


def test_identity_passes_through():
    """The identity layer returns its input unchanged."""
    m = Identity()
    x = jnp.asarray([1.0, 2.0, 3.0])
    assert jnp.array_equal(m(x), x)


def test_identity_accepts_any_type():
    """Identity works on arbitrary Python values."""
    m = Identity()
    assert m(42) == 42
    assert m("x") == "x"


def test_identity_has_no_parameters():
    """Identity has no trainable parameters."""
    from spectrax.core.graph import export

    g, s = export(Identity())
    assert g.var_refs == ()
    assert len(s) == 0


def test_identity_ignores_kwargs():
    """Identity tolerates stray keyword arguments."""
    m = Identity()
    x = jnp.asarray(5.0)
    assert jnp.array_equal(m(x, rngs=None, deterministic=True), x)
