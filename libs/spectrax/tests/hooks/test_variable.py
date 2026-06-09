# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.hooks.variable`."""

from __future__ import annotations

import jax.numpy as jnp

from spectrax.core.variable import Parameter
from spectrax.hooks.variable import register_variable_hook


def test_register_variable_hook_fires_on_write():
    """Observer receives ``(var, old, new)`` on eager write."""
    p = Parameter(jnp.zeros(3))
    seen = []

    def obs(var, old, new):
        """Record a tuple summarising the write."""
        seen.append((var.ref_id, tuple(old.tolist()), tuple(new.tolist())))

    register_variable_hook(p, obs)
    p.value = jnp.ones(3)
    assert seen[0][1] == (0.0, 0.0, 0.0)
    assert seen[0][2] == (1.0, 1.0, 1.0)


def test_variable_hook_handle_remove_detaches():
    """``Handle.remove`` removes the observer."""
    p = Parameter(jnp.zeros(3))
    calls = []
    handle = register_variable_hook(p, lambda *_: calls.append(1))
    p.value = jnp.ones(3)
    handle.remove()
    p.value = jnp.zeros(3)
    assert calls == [1]


def test_multiple_hooks_fire_in_order():
    """Multiple hooks fire in registration order."""
    p = Parameter(jnp.zeros(1))
    order = []
    register_variable_hook(p, lambda *_: order.append("a"))
    register_variable_hook(p, lambda *_: order.append("b"))
    p.value = jnp.ones(1)
    assert order == ["a", "b"]
