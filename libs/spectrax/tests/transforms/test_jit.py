# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.transforms.jit`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import spectrax as spx
from spectrax.nn.linear import Linear
from spectrax.nn.norm import BatchNorm1d
from spectrax.rng.rngs import Rngs


class Net(spx.Module):
    """Linear + BatchNorm1d fixture for jit tests."""

    def __init__(self, *, rngs):
        """Create ``fc`` and ``bn``."""
        super().__init__()
        self.fc = Linear(4, 4, rngs=rngs)
        self.bn = BatchNorm1d(4)

    def forward(self, x, **_):
        """Apply ``fc`` then ``bn``."""
        return self.bn(self.fc(x))


def test_jit_compiles_and_runs_same_output():
    """jit wrapper matches eager output."""
    m = Net(rngs=Rngs(0))
    m.eval()
    x = jnp.ones((2, 4))
    eager = m(x)
    compiled = spx.jit(lambda m, x: m(x))(m, x)
    assert jnp.allclose(eager, compiled)


def test_jit_decorator_form():
    """The decorator factory form works."""

    @spx.jit
    def step(m, x):
        """Execute one training step and return the result."""
        return m(x)

    m = Net(rngs=Rngs(0))
    m.eval()
    assert step(m, jnp.ones((1, 4))).shape == (1, 4)


def test_jit_caches_compile_per_graphdef():
    """Compile cache contains one entry per distinct :class:`GraphDef`."""
    fn = spx.jit(lambda m, x: m(x))
    m = Linear(4, 4, rngs=Rngs(0))
    fn(m, jnp.ones((2, 4)))
    cache = fn._spx_compile_cache
    assert len(cache) == 1
    fn(m, jnp.ones((2, 4)))
    assert len(cache) == 1


def test_jit_mutable_required_for_batch_stats():
    """BatchNorm under ``jit`` without ``mutable=`` raises."""

    @spx.jit
    def step(m, x):
        """Execute one training step and return the result."""
        return m(x)

    m = Net(rngs=Rngs(0))
    m.train()
    with pytest.raises(spx.IllegalMutationError):
        step(m, jnp.ones((4, 4)))


def test_jit_mutable_declared_propagates_back():
    """Declaring ``mutable='batch_stats'`` propagates writes back to live module."""

    @spx.jit(mutable="batch_stats")
    def step(m, x):
        """Execute one training step and return the result."""
        return m(x)

    m = Net(rngs=Rngs(0))
    m.train()
    before = m.bn.running_mean.value.copy()
    step(m, jnp.ones((4, 4)))
    assert not jnp.array_equal(before, m.bn.running_mean.value)


def test_jit_mutable_empty_is_equivalent_to_no_mutable():
    """``mutable=()`` is treated like no mutable selector."""

    @spx.jit(mutable=())
    def step(m, x):
        """Execute one training step and return the result."""
        return m(x)

    m = Net(rngs=Rngs(0))
    m.eval()
    out = step(m, jnp.ones((1, 4)))
    assert out.shape == (1, 4)


def test_jit_module_in_kwargs():
    """Modules passed as keyword arguments are also handled."""

    @spx.jit
    def fn(x, *, model):
        """Helper function."""
        return model(x)

    m = Linear(4, 4, rngs=Rngs(0))
    assert fn(jnp.ones((1, 4)), model=m).shape == (1, 4)


def test_jit_lower_pure_function_matches_jax_aot_shape():
    """``spx.jit`` exposes ``lower`` for pure AOT-style call sites."""

    fn = spx.jit(lambda x: x * 2)
    compiled = fn.lower(jnp.ones((2, 3))).compile()
    out = compiled(jnp.ones((2, 3)))
    assert out.shape == (2, 3)
    assert jnp.allclose(out, 2)


def test_jit_lower_accepts_module_argument():
    """``lower`` preserves the module-aware direct-readonly path."""

    m = Linear(4, 4, rngs=Rngs(0))
    x = jnp.ones((2, 4))
    fn = spx.jit(lambda model, xb: model(xb))
    lowered = fn.lower(m, x)
    assert lowered.compile() is not None


def test_jit_mutable_cache_distinguishes_single_positional_call_shape():
    """Single-module fast path and packed kwargs path need separate cache entries."""

    fn = spx.jit(lambda model, xb, scale=1.0: model(xb) * scale, mutable="parameters")
    m = Linear(4, 4, rngs=Rngs(0))
    x = jnp.ones((2, 4))

    y1 = fn(m, x)
    y2 = fn(m, x, scale=jnp.asarray(2.0))

    assert y1.shape == y2.shape == (2, 4)
    assert len(fn._spx_compile_cache) == 2


def test_jit_lower_mutable_cache_distinguishes_single_positional_call_shape():
    """``lower`` must mirror the runtime cache key for mutable single-module calls."""

    fn = spx.jit(lambda model, xb, scale=1.0: model(xb) * scale, mutable="parameters")
    m = Linear(4, 4, rngs=Rngs(0))
    x = jnp.ones((2, 4))

    assert fn.lower(m, x).compile() is not None
    assert fn.lower(m, x, scale=jnp.asarray(2.0)).compile() is not None
    assert len(fn._spx_compile_cache) == 2


def test_jit_cache_distinguishes_positional_and_keyword_module_layouts():
    """Compile cache keys should include module layout, not just GraphDef."""

    fn = spx.jit(lambda x, model: model(x))
    m = Linear(4, 4, rngs=Rngs(0))
    x = jnp.ones((1, 4))

    y_pos = fn(x, m)
    y_kw = fn(x, model=m)

    assert y_pos.shape == y_kw.shape == (1, 4)
    assert len(fn._spx_compile_cache) == 2


def test_jit_with_mpmd_mesh_routes_to_sxjit():
    """Passing an MPMD mesh to ``spx.jit`` delegates to ``sxjit``."""
    n = len(jax.devices())
    mesh = spx.create_mesh(axis_dims=(n,), axis_names=("pp",), mpmd_axis="pp")

    @spx.jit(mesh=mesh)
    def step(x):
        """Execute one training step and return the result."""
        for stage in range(n - 1):
            x = x + 1
            x = spx.sxstage_iter(x, stage=stage)
        return x + 1

    out = step(jnp.asarray(0.0))

    assert hasattr(step, "_mpmd_state")
    assert float(out) == float(n)


def test_jit_with_mpmd_mesh_preserves_output_pytree():
    """The sxjit-backed spx.jit path should not leak flat runtime tuples."""
    n = len(jax.devices())
    mesh = spx.create_mesh(axis_dims=(n,), axis_names=("pp",), mpmd_axis="pp")

    @spx.jit(mesh=mesh)
    def step(x):
        """Execute one training step and return the result."""
        for stage in range(n - 1):
            x = x + 1
            x = spx.sxstage_iter(x, stage=stage)
        return {"aux": x + 2, "loss": x + 1}

    out = step(jnp.asarray(0.0))

    assert set(out) == {"aux", "loss"}
    assert float(out["loss"]) == float(n)
    assert float(out["aux"]) == float(n + 1)


def test_jit_with_non_mpmd_mesh_keeps_module_aware_jit_path():
    """A regular mesh should not opt into the MPMD runtime."""
    n = len(jax.devices())
    mesh = spx.create_mesh(axis_dims=(n,), axis_names=("dp",))

    @spx.jit(mesh=mesh)
    def step(x):
        """Execute one training step and return the result."""
        return x + 2

    out = step(jnp.asarray(1.0))

    assert not hasattr(step, "_mpmd_state")
    assert float(out) == 3.0


def test_jit_with_mpmd_mesh_rejects_unsupported_module_mutability():
    """The MPMD shortcut must not silently ignore module mutation settings."""
    n = len(jax.devices())
    mesh = spx.create_mesh(axis_dims=(n,), axis_names=("pp",), mpmd_axis="pp")

    with pytest.raises(ValueError, match="mutable"):
        spx.jit(lambda x: x, mesh=mesh, mutable="batch_stats")
