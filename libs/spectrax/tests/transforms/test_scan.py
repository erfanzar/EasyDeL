# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.transforms.scan`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import spectrax as spx
from spectrax.core.module import Module
from spectrax.core.variable import Buffer
from spectrax.nn.linear import Linear
from spectrax.rng.rngs import Rngs


class _Scale(Module):
    """Tiny read-only module used by associative-scan tests."""

    scale: Buffer

    def __init__(self, value: float = 1.0) -> None:
        """Initialize with scale."""
        super().__init__()
        self.scale = Buffer(jnp.array(value, dtype=jnp.float32), kind="batch_stats")

    def forward(self, x):
        """Run the forward pass."""
        return self.scale.value * x


class _Accumulator(Module):
    """Tiny mutable module used to ensure associative-scan rejects writes."""

    acc: Buffer

    def __init__(self) -> None:
        """Initialize with acc."""
        super().__init__()
        self.acc = Buffer(jnp.zeros((), dtype=jnp.float32), kind="batch_stats")

    def forward(self, x):
        """Run the forward pass."""
        return x


def test_scan_rejects_non_module_init():
    """:func:`scan` requires a :class:`Module` as ``init_module``."""
    with pytest.raises(TypeError):
        spx.scan(lambda m, x: x, "not a module", jnp.arange(3))


def test_scan_outputs_stacked_ys():
    """Scan stacks per-step outputs along a leading axis."""
    m = Linear(2, 2, rngs=Rngs(0))
    xs = jnp.ones((4, 2))

    def step(m, x):
        """Execute one training step and return the result."""
        return m(x)

    ys = spx.scan(step, m, xs)
    assert ys.shape == (4, 2)


def test_scan_length_inferred_from_xs():
    """Without ``length`` the scan uses ``xs.shape[0]``."""
    m = Linear(2, 2, rngs=Rngs(0))
    xs = jnp.zeros((5, 2))
    ys = spx.scan(lambda m, x: m(x), m, xs)
    assert ys.shape[0] == 5


def test_scan_invariant_preserved_when_no_mutation():
    """Without a ``mutable`` selector the invariant is preserved."""
    m = Linear(2, 2, rngs=Rngs(0))
    xs = jnp.zeros((3, 2))
    before_params = [v.value.copy() for _, v in spx.live_variables(m)]
    spx.scan(lambda m, x: m(x), m, xs)
    after_params = [v.value for _, v in spx.live_variables(m)]
    for b, a in zip(before_params, after_params, strict=False):
        assert jnp.array_equal(b, a)


def test_scan_unroll_accepted():
    """The ``unroll`` kwarg is accepted."""
    m = Linear(2, 2, rngs=Rngs(0))
    xs = jnp.zeros((4, 2))
    ys = spx.scan(lambda m, x: m(x), m, xs, unroll=2)
    assert ys.shape == (4, 2)


def test_associative_scan_rejects_non_module():
    """:func:`associative_scan` requires a :class:`Module` argument."""
    with pytest.raises(TypeError):
        spx.associative_scan(lambda m, a, b: a + b, "not a module", jnp.arange(3))


def test_associative_scan_matches_jax_prefix_sum():
    """Associative scan matches upstream JAX for a pure combine."""
    m = _Scale(1.0)
    xs = jnp.arange(6.0, dtype=jnp.float32).reshape(3, 2)

    ys = spx.associative_scan(lambda mod, a, b: mod.scale.value * (a + b), m, xs)
    ref = jax.lax.associative_scan(lambda a, b: m.scale.value * (a + b), xs)
    assert jnp.allclose(ys, ref)


def test_associative_scan_reverse_and_axis_forwarded():
    """``reverse`` and ``axis`` are passed through to JAX."""
    m = _Scale(1.0)
    xs = jnp.arange(8.0, dtype=jnp.float32).reshape(2, 4)

    ys = spx.associative_scan(lambda mod, a, b: mod.scale.value * (a + b), m, xs, reverse=True, axis=1)
    ref = jax.lax.associative_scan(lambda a, b: m.scale.value * (a + b), xs, reverse=True, axis=1)
    assert jnp.allclose(ys, ref)


def test_associative_scan_rejects_mutable_kwarg():
    """Associative scan has no state carry, so ``mutable=`` is unsupported."""
    m = _Scale(1.0)
    with pytest.raises(ValueError, match="does not support mutable"):
        spx.associative_scan(lambda mod, a, b: a + b, m, jnp.arange(3.0), mutable="batch_stats")


def test_associative_scan_rejects_module_writes():
    """Pairwise combines must be pure; writes raise ``IllegalMutationError``."""
    m = _Accumulator()

    def combine(mod, a, b):
        """Combine helper."""
        mod.acc.value = mod.acc.value + 1.0
        return a + b

    with pytest.raises(spx.IllegalMutationError, match="does not support module mutations"):
        spx.associative_scan(combine, m, jnp.arange(4.0, dtype=jnp.float32))
