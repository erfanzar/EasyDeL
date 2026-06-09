# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Positive tests for :class:`spectrax.Module`'s pytree registration.

Modules flatten via :func:`~spectrax.export` and unflatten via
:func:`~spectrax.bind`; the pytree leaves are the raw arrays from
descendant :class:`~spectrax.Variable` cells. These tests pin the
public contract: tree ops work on modules, roundtrips preserve state,
and the documented mutation trap (mutating inside plain
:func:`jax.jit`) is honoured.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import spectrax as spx
import spectrax.nn as spx_nn


class _Twice(spx.Module):
    """Module that reuses the same child at two paths (shared submodule)."""

    def __init__(self, rngs: spx.Rngs):
        """Build a shared :class:`~spectrax.nn.Linear` referenced twice."""
        super().__init__()
        self.shared = spx_nn.Linear(4, 4, rngs=rngs)
        self.also_shared = self.shared

    def forward(self, x):
        """Run the shared layer once; path-sharing is visible in ``export``."""
        return self.shared(x)


def test_tree_leaves_are_arrays():
    """``jax.tree.leaves(model)`` returns the Variable value arrays.

    The number and ordering of leaves matches flattening the
    module's :class:`~spectrax.State` directly — spectrax's canonical
    collection-sorted, path-sorted order.
    """
    m = spx_nn.Linear(4, 4, rngs=spx.Rngs(0))
    leaves = jax.tree_util.tree_leaves(m)
    _gdef, state = spx.export(m)
    assert len(leaves) == 2
    assert leaves == jax.tree_util.tree_leaves(state)
    for leaf in leaves:
        assert hasattr(leaf, "shape")
        assert hasattr(leaf, "dtype")


def test_tree_map_with_path_emits_flat_key_paths():
    """``tree_map_with_path`` reports ordinary flat JAX key paths."""
    m = spx_nn.Linear(4, 4, rngs=spx.Rngs(0))
    paths = []

    mapped = jax.tree_util.tree_map_with_path(lambda path, leaf: paths.append(path) or leaf, m)
    _path_leaves, path_treedef = jax.tree_util.tree_flatten_with_path(m)
    _plain_leaves, plain_treedef = jax.tree_util.tree_flatten(m)

    assert isinstance(mapped, spx_nn.Linear)
    assert path_treedef == plain_treedef
    assert all(not isinstance(path[0], tuple) for path in paths)
    assert (jax.tree_util.DictKey("parameters"), jax.tree_util.DictKey("weight")) in paths
    assert (jax.tree_util.DictKey("parameters"), jax.tree_util.DictKey("bias")) in paths


def test_tree_map_produces_new_module_with_mapped_leaves():
    """``jax.tree.map`` threads a function through every Variable leaf.

    The original module is untouched; the returned object is a fresh
    :class:`~spectrax.nn.Linear` whose weights and bias have been
    shifted by the mapped delta.
    """
    m = spx_nn.Linear(4, 4, rngs=spx.Rngs(0))
    w_before = jnp.asarray(m.weight.value)
    m2 = jax.tree.map(lambda a: a + 1.0, m)
    assert isinstance(m2, spx_nn.Linear)
    assert jnp.allclose(m.weight.value, w_before)
    assert jnp.allclose(m2.weight.value, w_before + 1.0)


def test_pytree_roundtrip_preserves_training_flag():
    """Flatten + unflatten preserves the ``eval()`` / ``train()`` toggle."""
    m = spx_nn.Linear(4, 4, rngs=spx.Rngs(0))
    m.eval()
    assert m._spx_training is False
    leaves, treedef = jax.tree_util.tree_flatten(m)
    m2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert m2._spx_training is False

    m.train()
    leaves, treedef = jax.tree_util.tree_flatten(m)
    m3 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert m3._spx_training is True


def test_pytree_roundtrip_preserves_hooks_and_policy():
    """Flatten + unflatten preserves registered hooks and dtype policy.

    Hooks and policy are carried through ``aux_data``; they are
    shallow-copied on unflatten so later mutation of the original
    doesn't leak back — but at flatten time, the current content is
    captured.
    """
    m = spx_nn.Linear(4, 4, rngs=spx.Rngs(0))
    calls: list[int] = []

    def my_hook(module, args, kwargs, out):
        """Test hook that records one call per forward."""
        calls.append(1)
        return None

    m.register_forward_hook(my_hook)
    assert len(m._spx_fwd_hooks) == 1

    leaves, treedef = jax.tree_util.tree_flatten(m)
    m2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert len(m2._spx_fwd_hooks) == 1
    assert m2._spx_fwd_hooks[0] is my_hook


def test_jit_pure_forward_flows_through_module():
    """Pure :func:`jax.jit` on a forward-only function accepts a module directly.

    No :func:`spectrax.jit` wrapper needed when the function doesn't
    mutate any :class:`~spectrax.Variable`.
    """
    m = spx_nn.Linear(4, 4, rngs=spx.Rngs(0))
    x = jnp.ones((2, 4))

    @jax.jit
    def fwd(m, x):
        """Forward-only; module flows through as a pytree."""
        return m(x)

    y_eager = m(x)
    y_jit = fwd(m, x)
    assert jnp.allclose(y_jit, y_eager)


def test_jit_mutating_forward_silently_drops_mutations():
    """Plain :func:`jax.jit` does *not* propagate ``.value =`` mutations.

    This is the documented trap: plain :func:`jax.jit` flattens the
    module, traces the body against a reconstituted copy whose
    Variables are fresh tracers, and the outer live module is
    untouched when the traced call returns. Use
    :func:`spectrax.jit` with ``mutable=...`` to propagate mutations.
    """
    m = spx_nn.Linear(4, 4, rngs=spx.Rngs(0))
    w_before = jnp.asarray(m.weight.value)

    @jax.jit
    def mutate(m, new):
        """Write into ``m.weight`` inside jit — silently lost on return."""
        m.weight.value = new
        return m(jnp.ones((2, 4)))

    _ = mutate(m, jnp.zeros_like(m.weight.value))
    assert jnp.allclose(m.weight.value, w_before)


def test_shared_submodules_survive_flatten_unflatten():
    """A shared-child module unflattens with the sharing topology intact.

    The existing :func:`~spectrax.export` records shared paths in
    :class:`~spectrax.GraphDef.shared_paths`; unflatten through
    :func:`~spectrax.bind` rebuilds the module such that both paths
    point at the same live child.
    """
    t = _Twice(rngs=spx.Rngs(0))
    assert t.shared is t.also_shared

    leaves, treedef = jax.tree_util.tree_flatten(t)
    t2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(t2, _Twice)
    assert t2.shared is t2.also_shared


def test_cycle_still_detected_via_pytree_flatten():
    """Self-referential modules still raise :class:`CyclicGraphError`.

    Pytree registration delegates flatten to :func:`~spectrax.export`,
    which already detects cycles — pytree semantics don't circumvent
    that guard.
    """

    class Cyclic(spx.Module):
        """Will be assigned a self-reference, for cycle testing."""

        def __init__(self, rngs: spx.Rngs):
            """Start without a cycle; caller inserts one after ``__init__``."""
            super().__init__()
            self.fc = spx_nn.Linear(4, 4, rngs=rngs)

        def forward(self, x):
            """Trivial forward — never actually called in this test."""
            return self.fc(x)

    m = Cyclic(rngs=spx.Rngs(0))
    m.fc2 = m

    with pytest.raises(spx.CyclicGraphError):
        jax.tree_util.tree_flatten(m)
