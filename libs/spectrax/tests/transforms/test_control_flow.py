# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :mod:`spectrax.transforms.control_flow`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from spectrax.core.module import Module
from spectrax.core.variable import Buffer
from spectrax.nn.linear import Linear
from spectrax.rng.rngs import Rngs
from spectrax.transforms.control_flow import (
    cond,
    fori_loop,
    remat_scan,
    switch,
    while_loop,
)
from spectrax.transforms.scan import scan


class _Accum(Module):
    """Tiny module with a single buffer used to accumulate."""

    acc: Buffer

    def __init__(self) -> None:
        """Initialize with acc."""
        super().__init__()
        self.acc = Buffer(jnp.zeros(()), kind="batch_stats")

    def forward(self, x):
        """Run the forward pass."""
        return x


def test_cond_selects_true_branch_with_no_mutable():
    """Predicate True runs the ``on_true`` branch."""
    m = Linear(2, 3, rngs=Rngs(0))
    x = jnp.ones((1, 2))
    y = cond(
        jnp.bool_(True),
        lambda mod, x: mod(x) + 1.0,
        lambda mod, x: mod(x) - 1.0,
        m,
        x,
    )
    ref = m(x) + 1.0
    assert jnp.allclose(y, ref)


def test_cond_selects_false_branch():
    """Predicate False runs the ``on_false`` branch."""
    m = Linear(2, 3, rngs=Rngs(0))
    x = jnp.ones((1, 2))
    y = cond(
        jnp.bool_(False),
        lambda mod, x: mod(x) + 1.0,
        lambda mod, x: mod(x) - 1.0,
        m,
        x,
    )
    assert jnp.allclose(y, m(x) - 1.0)


def test_switch_picks_branch_by_index():
    """``switch`` selects the ``i``-th branch."""
    m = Linear(2, 3, rngs=Rngs(0))
    x = jnp.ones((1, 2))
    branches = [
        lambda mod, x: mod(x) * 0.0,
        lambda mod, x: mod(x) * 1.0,
        lambda mod, x: mod(x) * 2.0,
    ]
    y0 = switch(0, branches, m, x)
    y2 = switch(2, branches, m, x)
    assert jnp.allclose(y0, jnp.zeros_like(m(x)))
    assert jnp.allclose(y2, 2.0 * m(x))


def test_switch_requires_at_least_one_branch():
    """Empty ``branches`` raises."""
    m = Linear(2, 3, rngs=Rngs(0))
    with pytest.raises(ValueError):
        switch(0, [], m)


def test_while_loop_counter_no_mutation():
    """`while_loop` works with an integer carry and no state mutation."""
    m = Linear(2, 3, rngs=Rngs(0))

    def cond_fn(_mod, uc):
        """Condition function."""
        return uc < 5

    def body_fn(_mod, uc):
        """Body function for loop/cond."""
        return uc + 1

    final = while_loop(cond_fn, body_fn, m, 0)
    assert int(final) == 5


def test_fori_loop_accumulates_counter():
    """Simple fori_loop returns the expected accumulation."""
    m = Linear(2, 3, rngs=Rngs(0))

    def body_fn(i, _mod, uc):
        """Body function for loop/cond."""
        return uc + i

    total = fori_loop(0, 5, body_fn, m, jnp.int32(0))
    assert int(total) == 0 + 1 + 2 + 3 + 4


def test_fori_loop_with_batch_stats_mutation():
    """Fori loop carries a `batch_stats` mutation through iterations."""
    m = _Accum()

    def body(i, mod, uc):
        """Loop body function."""
        mod.acc.value = mod.acc.value + jnp.float32(i)
        return uc + 1

    fori_loop(0, 5, body, m, jnp.int32(0), mutable="batch_stats")
    assert float(m.acc.value) == 10.0


def test_remat_scan_matches_scan_numerics():
    """:func:`remat_scan` is numerically equivalent to plain :func:`scan`."""
    m = Linear(3, 3, rngs=Rngs(0))
    xs = jnp.arange(4 * 3.0).reshape((4, 3))

    def fn(mod, x):
        """Helper function."""
        return mod(x)

    y_scan = scan(fn, m, xs)
    y_remat = remat_scan(fn, m, xs)
    assert jnp.allclose(y_scan, y_remat, atol=1e-6)


def test_remat_scan_gradient_matches_scan():
    """Gradients through ``remat_scan`` match ``scan`` (up to epsilon)."""
    m = Linear(2, 2, rngs=Rngs(0))
    xs = jnp.ones((3, 2))

    def loss_plain(W):
        """loss_plain helper."""
        m.weight.value = W

        def fn(mod, x):
            """Helper function."""
            return mod(x).sum()

        return scan(fn, m, xs).sum()

    def loss_remat(W):
        """loss_remat helper."""
        m.weight.value = W

        def fn(mod, x):
            """Helper function."""
            return mod(x).sum()

        return remat_scan(fn, m, xs).sum()

    w = m.weight.value
    g_plain = jax.grad(loss_plain)(w)
    g_remat = jax.grad(loss_remat)(w)
    assert jnp.allclose(g_plain, g_remat, atol=1e-5)


def test_cond_carries_batch_stats_mutation_when_mutable():
    """`cond` with `mutable="batch_stats"` carries mutations to the live module."""
    m = _Accum()

    def on_true(mod, x):
        """True branch handler."""
        mod.acc.value = mod.acc.value + x
        return x

    def on_false(mod, _x):
        """False branch handler."""
        return jnp.float32(0.0)

    cond(jnp.bool_(True), on_true, on_false, m, jnp.float32(3.0), mutable="batch_stats")
    assert float(m.acc.value) == 3.0


def test_while_loop_mutates_batch_stats_when_allowed():
    """`while_loop` with `mutable="batch_stats"` accumulates across iterations."""
    m = _Accum()

    def cond_fn(_mod, uc):
        """Condition function."""
        return uc < 3

    def body_fn(mod, uc):
        """Body function for loop/cond."""
        mod.acc.value = mod.acc.value + 1.0
        return uc + 1

    while_loop(cond_fn, body_fn, m, jnp.int32(0), mutable="batch_stats")
    assert float(m.acc.value) == 3.0
