# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.hooks.forward`."""

from __future__ import annotations

import jax.numpy as jnp

from spectrax.hooks.forward import (
    Handle,
    register_forward_hook,
    register_forward_pre_hook,
)
from spectrax.nn.linear import Linear
from spectrax.rng.rngs import Rngs


def test_register_forward_hook_returns_handle():
    """:func:`register_forward_hook` returns a :class:`Handle`."""
    m = Linear(4, 4, rngs=Rngs(0))
    h = register_forward_hook(m, lambda mod, a, kw, out: None)
    assert isinstance(h, Handle)


def test_pre_hook_can_swap_args():
    """A pre-hook that returns ``(args, kwargs)`` overrides the input."""
    m = Linear(2, 2, rngs=Rngs(0))
    captured = {}

    def pre(mod, args, kwargs):
        """Record the args and return a substitute."""
        captured["args"] = args
        return (jnp.zeros((1, 2)),), kwargs

    register_forward_pre_hook(m, pre)
    out = m(jnp.ones((1, 2)))
    assert captured["args"][0].shape == (1, 2)
    assert out.shape == (1, 2)


def test_post_hook_can_replace_output():
    """A post-hook can substitute the final output."""
    m = Linear(2, 2, rngs=Rngs(0))
    register_forward_hook(m, lambda mod, a, kw, out: out * 0.0)
    out = m(jnp.ones((1, 2)))
    assert jnp.array_equal(out, jnp.zeros((1, 2)))


def test_hook_handle_remove_detaches_callback():
    """Calling ``Handle.remove`` stops subsequent firings."""
    m = Linear(2, 2, rngs=Rngs(0))
    calls = []
    h = register_forward_hook(m, lambda *_: calls.append(1) or None)
    m(jnp.ones((1, 2)))
    h.remove()
    m(jnp.ones((1, 2)))
    assert calls == [1]


def test_pre_and_post_hooks_coexist():
    """Both pre- and post-hooks fire in a single forward call."""
    m = Linear(2, 2, rngs=Rngs(0))
    events = []
    register_forward_pre_hook(m, lambda mod, a, kw: events.append("pre"))
    register_forward_hook(m, lambda mod, a, kw, out: events.append("post"))
    m(jnp.ones((1, 2)))
    assert events == ["pre", "post"]
