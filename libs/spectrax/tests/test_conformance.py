# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Conformance tests covering the numbered invariants in the build plan.

Every test here exercises a single structural invariant or user-visible
contract of spectrax. The fixtures at module scope are defined here
(rather than inside test functions) so that :func:`spectrax.bind` can
import them by qualified name when rebuilding modules from a
:class:`~spectrax.GraphDef`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import spectrax as spx
from spectrax import nn


class Net(spx.Module):
    """Simple ``Linear -> BatchNorm -> Linear`` fixture."""

    def __init__(self, *, rngs):
        """Initialize the three child layers."""
        super().__init__()
        self.fc1 = nn.Linear(4, 8, rngs=rngs)
        self.bn = nn.BatchNorm1d(8)
        self.fc2 = nn.Linear(8, 2, rngs=rngs)

    def forward(self, x, **_):
        """Thread ``x`` through ``fc1 -> bn -> fc2``."""
        x = self.fc1(x)
        x = self.bn(x)
        return self.fc2(x)


class Bigger(spx.Module):
    """Single-Linear fixture with a different structure than :class:`Net`."""

    def __init__(self, *, rngs):
        """Create one large ``Linear`` layer."""
        super().__init__()
        self.fc1 = nn.Linear(4, 16, rngs=rngs)

    def forward(self, x, **_):
        """Apply :attr:`fc1`."""
        return self.fc1(x)


class Cyclic(spx.Module):
    """Fixture used to build a self-referential cycle at test time."""

    def __init__(self, *, rngs):
        """Create a single child ``Linear``."""
        super().__init__()
        self.fc = nn.Linear(4, 4, rngs=rngs)


class Twice(spx.Module):
    """Fixture that calls its one child ``Linear`` twice."""

    def __init__(self, *, rngs):
        """Create the single child ``Linear``."""
        super().__init__()
        self.fc = nn.Linear(4, 4, rngs=rngs)

    def forward(self, x):
        """Sum two applications of the shared layer."""
        return self.fc(x) + self.fc(x)


class Sower(spx.Module):
    """Fixture that captures an intermediate via :meth:`sow`."""

    def __init__(self, *, rngs):
        """Create the single child ``Linear``."""
        super().__init__()
        self.fc = nn.Linear(4, 4, rngs=rngs)

    def forward(self, x, **_):
        """Apply :attr:`fc` and sow the result into ``intermediates``."""
        h = self.fc(x)
        self.sow("intermediates", "h", h)
        return h


def test_module_pytree_leaves_are_variable_arrays():
    """Invariant I1 (post-pytree-registration): modules are pytrees whose
    leaves are the raw arrays held in descendant
    :class:`~spectrax.Variable` cells.

    Replaces the older ``test_module_is_leaf`` assertion. With
    :class:`~spectrax.Module` registered as a JAX pytree,
    :func:`jax.tree_util.tree_leaves` decomposes the module into its
    parameter tensors (in canonical-path order), not a single
    module-shaped leaf.
    """
    rngs = spx.Rngs(0)
    m = Net(rngs=rngs)
    leaves = jax.tree_util.tree_leaves(m)
    _gdef, state = spx.export(m)
    expected_leaves = jax.tree_util.tree_leaves(state)
    assert leaves == expected_leaves
    assert all(hasattr(leaf, "shape") for leaf in leaves)


def test_graphdef_structural_equality_across_seeds():
    """Invariant I3: structurally-identical modules yield equal GraphDefs."""
    a = Net(rngs=spx.Rngs(0))
    b = Net(rngs=spx.Rngs(99))
    ga, _ = spx.export(a)
    gb, _ = spx.export(b)
    assert ga == gb
    assert hash(ga) == hash(gb)


def test_graphdef_differs_on_structure():
    """Invariant I3: differing structure yields different GraphDefs."""
    a = Net(rngs=spx.Rngs(0))
    b = Bigger(rngs=spx.Rngs(0))
    ga, _ = spx.export(a)
    gb, _ = spx.export(b)
    assert ga != gb


def test_paths_stable_across_exports():
    """Invariant I4: repeated exports of the same module yield equal outputs."""
    rngs = spx.Rngs(0)
    m = Net(rngs=rngs)
    g1, s1 = spx.export(m)
    g2, s2 = spx.export(m)
    assert g1 == g2
    assert sorted(s1.raw()["parameters"]) == sorted(s2.raw()["parameters"])


def test_cycle_raises():
    """A self-referential module raises :class:`spx.CyclicGraphError`."""
    rngs = spx.Rngs(0)
    m = Cyclic(rngs=rngs)
    m.fc2 = m
    with pytest.raises(spx.CyclicGraphError):
        spx.export(m)


def test_shared_submodule_gradient_accumulates():
    """Gradients through a shared submodule produce a non-empty parameters state."""
    rngs = spx.Rngs(0)
    m = Twice(rngs=rngs)
    x = jnp.ones((2, 4))

    def loss(m):
        """Squared-output loss."""
        return (m(x) ** 2).mean()

    grads = spx.grad(loss)(m)
    assert "parameters" in grads.collections()


def test_clone_severs_sharing():
    """``spx.clone`` produces a module sharing no Variable identity with the source."""
    rngs = spx.Rngs(0)
    m = Net(rngs=rngs)
    c = spx.clone(m)
    m_vars = {id(v) for _, v in spx.live_variables(m)}
    c_vars = {id(v) for _, v in spx.live_variables(c)}
    assert m_vars.isdisjoint(c_vars)


def test_variable_arithmetic_matches_value():
    """``Variable`` delegates arithmetic to ``.value``."""
    p = spx.Parameter(jnp.arange(6.0).reshape(2, 3))
    x = jnp.ones((3,))
    assert jnp.allclose(p @ x, p.value @ x)
    assert jnp.allclose(p + 1.0, p.value + 1.0)


def test_variable_write_propagates_through_alias():
    """Writes via ``.value`` are visible through every alias."""
    p = spx.Parameter(jnp.zeros((2,)))
    alias = p
    p.value = jnp.ones((2,))
    assert jnp.array_equal(alias.value, jnp.ones((2,)))


def test_jit_undeclared_mutation_raises():
    """Invariant I7: undeclared mutation under ``jit`` raises."""
    rngs = spx.Rngs(0)
    m = Net(rngs=rngs)
    x = jnp.ones((4, 4))

    @spx.jit
    def step(m, x):
        """Single forward pass; BN mutates batch_stats in training mode."""
        return m(x)

    m.train()
    with pytest.raises(spx.IllegalMutationError):
        step(m, x)


def test_jit_declared_mutation_succeeds():
    """Invariant I7 converse: declared mutation is allowed and propagates."""
    rngs = spx.Rngs(0)
    m = Net(rngs=rngs)
    x = jnp.ones((4, 4))

    @spx.jit(mutable="batch_stats")
    def step(m, x):
        """Forward with explicit ``mutable=`` selector."""
        return m(x)

    m.train()
    y = step(m, x)
    assert y.shape == (4, 2)


def test_grad_parameters_only():
    """``spx.grad`` with default ``wrt`` returns only the parameters collection."""
    rngs = spx.Rngs(0)
    m = Net(rngs=rngs)
    x = jnp.ones((4, 4))

    def loss(m):
        """Squared-output loss."""
        return (m(x) ** 2).mean()

    grads = spx.grad(loss)(m)
    assert "parameters" in grads.collections()
    assert "batch_stats" not in grads.collections()


def test_same_seed_same_inits():
    """Identical seeds yield identical parameter initializations."""
    a = nn.Linear(4, 4, rngs=spx.Rngs(123))
    b = nn.Linear(4, 4, rngs=spx.Rngs(123))
    assert jnp.array_equal(a.weight.value, b.weight.value)


def test_different_streams_produce_different_keys():
    """Named streams derive independent keys from the root seed."""
    rngs = spx.Rngs(0)
    k1 = rngs.parameters
    k2 = rngs.key("dropout")
    assert not jnp.array_equal(k1, k2)


def test_rngs_fork():
    """``Rngs.fork`` produces N pairwise-distinct Rngs."""
    rngs = spx.Rngs(0)
    forked = rngs.fork(4)
    assert len(forked) == 4
    keys = [forked[i].parameters for i in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            assert not jnp.array_equal(keys[i], keys[j])


def test_dropout_eval_passthrough():
    """Eval-mode dropout returns its input unchanged."""
    d = nn.Dropout(0.5)
    d.eval()
    x = jnp.ones((10,))
    assert jnp.array_equal(d(x), x)


def test_forward_hook_runs_eagerly():
    """Forward hooks fire with expected ``(module, args, kwargs, output)`` tuple."""
    rngs = spx.Rngs(0)
    m = nn.Linear(4, 4, rngs=rngs)
    calls = []
    m.register_forward_hook(lambda mod, args, kwargs, out: calls.append(out.shape))
    _ = m(jnp.zeros((2, 4)))
    assert calls == [(2, 4)]


def test_sow_intermediates_visible():
    """``sow('intermediates', ...)`` surfaces the captured value in State."""
    rngs = spx.Rngs(0)
    m = Sower(rngs=rngs)
    _ = m(jnp.ones((2, 4)))
    _, state = spx.export(m)
    assert "intermediates" in state.collections()


def test_selector_at_instances_of():
    """``at_instances_of`` picks out exactly the target layers' parameters."""
    rngs = spx.Rngs(0)
    m = Net(rngs=rngs)
    matches = spx.select().at_instances_of(nn.Linear).variables("parameters").apply(m)
    assert len(matches) == 4


def test_selector_where():
    """``where_variable`` narrows by a per-variable predicate."""
    rngs = spx.Rngs(0)
    m = Net(rngs=rngs)
    only_weights = spx.select().where_variable(lambda v, p: p.endswith("weight")).apply(m)
    assert only_weights, "selector returned no matches"
    assert all(p.endswith("weight") for p, _ in only_weights)


def test_export_bind_bitwise():
    """``export -> bind`` produces bitwise-equivalent outputs."""
    rngs = spx.Rngs(7)
    m = Net(rngs=rngs)
    x = jnp.linspace(-1, 1, 16).reshape(4, 4)
    y1 = m(x)
    g, s = spx.export(m)
    m2 = spx.bind(g, s)
    y2 = m2(x)
    assert jnp.allclose(y1, y2)
