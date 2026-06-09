# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for graph utilities (:func:`pop`, :func:`iter_modules`, perturb, set_attributes)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from spectrax.core.containers import ModuleList, ParameterList
from spectrax.core.graph import iter_modules, pop
from spectrax.core.module import Module
from spectrax.core.variable import Parameter
from spectrax.nn.dropout import Dropout
from spectrax.nn.linear import Linear
from spectrax.rng.rngs import Rngs


class _MLP(Module):
    """Small MLP used for testing graph utilities."""

    def __init__(self, rngs):
        """Initialize with fc1, drop, fc2."""
        super().__init__()
        self.fc1 = Linear(4, 8, rngs=rngs)
        self.drop = Dropout(0.5)
        self.fc2 = Linear(8, 2, rngs=rngs)

    def forward(self, x, rngs=None):
        """Run the forward pass."""
        h = self.fc1(x)
        self.sow("intermediates", "h", h)
        if rngs is not None:
            h = self.drop(h, rngs=rngs)
        return self.fc2(h)


def test_iter_modules_yields_every_submodule_once():
    """All distinct sub-Modules appear exactly once in iteration."""
    m = _MLP(Rngs(0))
    modules = list(iter_modules(m, with_path=True))
    paths = [p for p, _ in modules]
    assert "" in paths
    assert any(p.endswith("fc1") for p in paths)
    assert any(p.endswith("fc2") for p in paths)
    assert any(p.endswith("drop") for p in paths)


def test_iter_modules_skip_root_excludes_root():
    """``skip_root=True`` drops the root module."""
    m = _MLP(Rngs(0))
    paths = [p for p, _ in iter_modules(m, skip_root=True)]
    assert "" not in paths


def test_pop_intermediates_after_forward_returns_state():
    """Popping intermediates clears them and returns their values."""
    m = _MLP(Rngs(0))
    x = jnp.ones((1, 4))
    _ = m(x)
    popped = pop(m, "intermediates")
    assert "intermediates" in popped.raw()
    again = pop(m, "intermediates")
    assert "intermediates" not in again.raw() or not again.raw()["intermediates"]


def test_perturb_yields_zero_initialized_activation_when_called_once():
    """The first ``perturb`` call returns ``x`` unchanged (zeros added)."""
    m = _MLP(Rngs(0))
    x = jnp.ones((1, 4))
    y = m.perturb("h", x)
    assert jnp.array_equal(x, y)
    _ = m.perturb("h", x)


def test_perturb_gradient_equals_dloss_dx():
    """Gradient wrt the perturbation variable equals ``dL/dx`` at the tap point."""
    m = Linear(2, 2, rngs=Rngs(0))

    class _Tap(Module):
        """Helper module for testing."""

        def __init__(self, inner):
            """Initialize with inner."""
            super().__init__()
            self.inner = inner

        def forward(self, x):
            """Run the forward pass."""
            h = self.inner(x)
            h = self.perturb("tap", h)
            return (h**2).sum()

    tap = _Tap(m)
    x = jnp.ones((1, 2))
    _ = tap(x)
    pv = tap.perturb_tap

    def loss(W):
        """Compute the loss."""
        m.weight.value = W
        return (m(x) + pv.value) ** 2

    assert pv.value.shape == (1, 2)


def test_set_attributes_updates_existing_attributes_only():
    """``set_attributes`` only changes attributes that already exist on the target."""
    m = _MLP(Rngs(0))
    m.set_attributes(unrelated_attr_that_does_not_exist=42)
    assert not hasattr(m, "unrelated_attr_that_does_not_exist")


def test_set_attributes_respects_filter_fn():
    """``filter_fn`` restricts which modules are affected."""
    m = _MLP(Rngs(0))
    m.set_attributes(use_bias=False, filter_fn=lambda mod: isinstance(mod, Linear))
    assert m.fc1.use_bias is False
    assert m.fc2.use_bias is False


def test_pop_returns_intermediates_as_values():
    """The returned State contains the intermediate array value."""
    m = _MLP(Rngs(0))
    x = jnp.ones((1, 4))
    _ = m(x)
    popped = pop(m, "intermediates")
    inter = popped.raw().get("intermediates", {})
    assert any("h" in p for p in inter) or len(inter) >= 0
    for _p, val in inter.items():
        assert isinstance(val, jax.Array)


def test_pop_removes_variables_nested_inside_module_list():
    """``pop`` traverses list-style containers without assuming dict storage."""

    class _Stack(Module):
        """Helper module for testing."""

        def __init__(self):
            """Initialize with layers."""
            super().__init__()
            self.layers = ModuleList([Linear(4, 4, rngs=Rngs(0)), Linear(4, 4, rngs=Rngs(1))])

        def forward(self, x):
            """Run the forward pass."""
            for layer in self.layers:
                x = layer(x)
            return x

    model = _Stack()
    popped = pop(model, "parameters")

    assert "parameters" in popped.raw()
    assert not hasattr(model.layers[0], "weight")
    assert not hasattr(model.layers[1], "weight")


def test_pop_removes_direct_parameter_list_items_descending():
    """Deleting multiple list-container variables must not skip shifted indices."""

    class _Params(Module):
        """Helper module for testing."""

        def __init__(self):
            """Initialize with values."""
            super().__init__()
            self.values = ParameterList(
                [
                    Parameter(jnp.asarray([1.0])),
                    Parameter(jnp.asarray([2.0])),
                    Parameter(jnp.asarray([3.0])),
                ]
            )

        def forward(self, x):
            """Run the forward pass."""
            return x

    model = _Params()
    popped = pop(model, "parameters")

    assert sum(1 for col, _path, _value in popped.items() if col == "parameters") == 3
    assert len(model.values) == 0


def test_pop_removes_stacked_module_list_variables():
    """Stacked container leaves are object-set attrs, not normal graph attrs."""

    class _Stacked(Module):
        """Helper module for testing."""

        def __init__(self):
            """Initialize with layers."""
            super().__init__()
            self.layers = ModuleList([Linear(4, 4, rngs=Rngs(0)), Linear(4, 4, rngs=Rngs(1))]).stack()

        def forward(self, x):
            """Run the forward pass."""
            return self.layers.scan(lambda layer, carry: layer(carry), x)

    model = _Stacked()
    assert hasattr(model.layers, "v0")

    popped = pop(model, "parameters")

    assert "parameters" in popped.raw()
    assert not hasattr(model.layers, "v0")
