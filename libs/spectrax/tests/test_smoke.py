# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""End-to-end smoke tests covering the top-level spectrax workflow."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import spectrax as spx
from spectrax import functional as F
from spectrax import nn


class MLP(spx.Module):
    """Toy two-layer MLP used as a fixture by the tests in this file."""

    def __init__(self, d, h, o, *, rngs):
        """Construct fc1/fc2 with the given feature sizes and RNG source."""
        super().__init__()
        self.fc1 = nn.Linear(d, h, rngs=rngs)
        self.fc2 = nn.Linear(h, o, rngs=rngs)

    def forward(self, x, **_):
        """Run ``fc1 -> gelu -> fc2``."""
        x = F.gelu(self.fc1(x))
        return self.fc2(x)


def test_forward_shapes():
    """Output shape matches the declared ``o`` dimension."""
    rngs = spx.Rngs(0)
    m = MLP(16, 32, 4, rngs=rngs)
    x = jnp.zeros((8, 16))
    y = m(x)
    assert y.shape == (8, 4)


def test_export_bind_identity():
    """``bind(export(m))`` produces a module with identical outputs."""
    rngs = spx.Rngs(0)
    m = MLP(16, 32, 4, rngs=rngs)
    x = jnp.ones((2, 16))
    y1 = m(x)
    gdef, state = spx.export(m)
    m2 = spx.bind(gdef, state)
    y2 = m2(x)
    assert jnp.array_equal(y1, y2)


def test_graphdef_structural_hash():
    """Two independently-seeded equivalent modules have equal GraphDefs."""
    rngs1 = spx.Rngs(0)
    rngs2 = spx.Rngs(1)
    a = MLP(16, 32, 4, rngs=rngs1)
    b = MLP(16, 32, 4, rngs=rngs2)
    ga, _ = spx.export(a)
    gb, _ = spx.export(b)
    assert ga == gb
    assert hash(ga) == hash(gb)


def test_tied_weights_one_leaf():
    """A shared submodule shows up once in State and once in shared_paths."""
    rngs = spx.Rngs(0)

    class Tied(spx.Module):
        """Module with a deliberate alias to the same ``Linear`` sub-layer."""

        def __init__(self):
            """Create ``self.fc`` and alias it as ``self.tied``."""
            super().__init__()
            self.fc = nn.Linear(4, 4, rngs=rngs)
            self.tied = self.fc

        def forward(self, x):
            """Apply the shared layer twice."""
            return self.fc(x) + self.tied(x)

    m = Tied()
    gdef, state = spx.export(m)
    n_weight_leaves = sum(1 for _c, p, _v in state.items() if _c == "parameters" and p.endswith("weight"))
    assert n_weight_leaves == 1
    assert len(gdef.shared_paths) >= 1


def test_train_eval_propagates():
    """``train()`` / ``eval()`` propagate recursively to children."""
    rngs = spx.Rngs(0)
    m = MLP(8, 16, 4, rngs=rngs)
    assert m.training is True
    m.eval()
    assert m.training is False
    assert m.fc1.training is False


def test_grad_wrt_parameters():
    """``spx.grad`` returns a :class:`State` shaped like the ``parameters`` subset."""
    rngs = spx.Rngs(0)
    m = MLP(8, 16, 4, rngs=rngs)
    x = jnp.ones((2, 8))
    y = jnp.zeros((2, 4))

    def loss(m):
        """Mean-squared-error loss of the fixture MLP against ``y``."""
        return ((m(x) - y) ** 2).mean()

    grads = spx.grad(loss)(m)
    assert isinstance(grads, spx.State)
    assert "parameters" in grads.collections()
    _, state = spx.export(m)
    for c, p, v in state.items():
        if c == "parameters":
            assert grads.get(c, p).shape == v.shape


def test_jit_runs():
    """``spx.jit`` compiles and runs a simple Module-aware function."""
    rngs = spx.Rngs(0)
    m = MLP(8, 16, 4, rngs=rngs)
    x = jnp.ones((2, 8))
    fn = spx.jit(lambda m, x: m(x))
    y = fn(m, x)
    assert y.shape == (2, 4)


def test_module_pytree_leaves_are_variable_arrays():
    """Modules are registered pytrees; leaves are the Variable arrays.

    Replaces the older ``test_module_not_pytree_leaf`` smoke check.
    :class:`~spectrax.Module` is now a JAX pytree whose flatten /
    unflatten are the existing :func:`~spectrax.export` /
    :func:`~spectrax.bind` primitives, so
    :func:`jax.tree_util.tree_leaves` returns the raw array values of
    every descendant :class:`~spectrax.Variable`.
    """
    rngs = spx.Rngs(0)
    m = MLP(8, 16, 4, rngs=rngs)
    leaves = jax.tree_util.tree_leaves(m)
    _gdef, state = spx.export(m)
    expected = jax.tree_util.tree_leaves(state)
    assert leaves == expected
    assert all(hasattr(leaf, "shape") for leaf in leaves)


def test_export_bind_roundtrip():
    """``export`` then ``bind`` round-trips a module preserving outputs."""
    rngs = spx.Rngs(0)
    m = MLP(8, 16, 4, rngs=rngs)
    x = jnp.ones((2, 8))
    y1 = m(x)
    gdef, state = spx.export(m)
    m2 = spx.bind(gdef, state)
    y2 = m2(x)
    assert jnp.allclose(y1, y2)


def test_dropout_requires_rngs_in_training():
    """Dropout demands an ``Rngs`` in training mode; eval mode is a passthrough."""
    d = nn.Dropout(0.5)
    x = jnp.ones((4,))
    with pytest.raises(RuntimeError):
        d(x)
    d.eval()
    y = d(x)
    assert jnp.array_equal(y, x)


def test_selector_partition():
    """Collection-name selector sugar correctly splits a State."""
    rngs = spx.Rngs(0)
    m = MLP(8, 16, 4, rngs=rngs)
    _, state = spx.export(m)
    sel = spx.select().variables("parameters")
    matched, rest = sel.partition_state(m, state)
    assert "parameters" in matched.collections()
    assert "parameters" not in rest.collections()
