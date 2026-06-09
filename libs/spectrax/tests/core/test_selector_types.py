# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for selecting variables by Python class.

Covers the new :meth:`Selector.of_type` / :meth:`Selector.not_of_type`
builders, the :func:`spectrax.of_type` module helper, and the extended
:func:`spectrax.as_selector` coercion that accepts Variable subclasses
and mixed-iterable inputs.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

import spectrax as spx
from spectrax.core.errors import SelectorError


class MyAdapterParam(spx.Variable):
    """Custom Variable subclass for testing class-based selection."""

    default_kind = "adapter"


class _GradModel(spx.Module):
    """Top-level Module with a stock :class:`~spectrax.Parameter` and a
    :class:`MyAdapterParam` — defined at module scope so spectrax's
    registry can resolve it during ``bind``.
    """

    def __init__(self):
        """Build a tiny scalar-loss model with two disjoint collections."""
        super().__init__()
        self.w = spx.Parameter(jnp.ones((4,), dtype=jnp.float32))
        self.a = MyAdapterParam(jnp.ones((4,), dtype=jnp.float32))

    def forward(self, x):
        """Scalar loss: sum of the elementwise triple product."""
        return (x * self.w.value * self.a.value).sum()


def _model():
    """Construct a bare :class:`~spectrax.Module` with two variables.

    Used by most tests below. Re-initializing the private slots after
    ``Module()`` is necessary because the base class's ``__init__`` is
    what wires up ``_spx_attr_order``, ``_spx_static`` etc. — calling
    it explicitly keeps the test model construction to three lines.
    """
    m = spx.Module()
    m.__init__()
    m.w = spx.Parameter(jnp.zeros((2, 3), dtype=jnp.float32))
    m.a = MyAdapterParam(jnp.ones((3,), dtype=jnp.float32))
    return m


def test_of_type_selects_by_variable_subclass():
    """``of_type(Cls)`` matches variables that are instances of ``Cls``."""
    m = _model()
    hits = spx.of_type(MyAdapterParam).apply(m)
    assert [p for p, _ in hits] == ["a"]
    assert type(hits[0][1]) is MyAdapterParam


def test_of_type_method_on_select_chain():
    """``select().of_type(Cls)`` composes with other chain methods."""
    m = _model()
    hits = spx.select().of_type(MyAdapterParam).apply(m)
    assert len(hits) == 1
    assert hits[0][0] == "a"


def test_of_type_rejects_non_variable_class():
    """Passing a non-Variable class raises :class:`SelectorError`."""
    with pytest.raises(SelectorError):
        spx.of_type(int)
    with pytest.raises(SelectorError):
        spx.select().of_type(str)


def test_not_of_type_excludes_subclass_instances():
    """``not_of_type(Cls)`` matches everything except instances of ``Cls``."""
    m = _model()
    hits = spx.select().not_of_type(MyAdapterParam).apply(m)
    paths = [p for p, _ in hits]
    assert "w" in paths
    assert "a" not in paths


def test_as_selector_accepts_variable_subclass_directly():
    """``as_selector(Cls)`` is sugar for ``of_type(Cls)``."""
    m = _model()
    a = spx.as_selector(MyAdapterParam).apply(m)
    b = spx.of_type(MyAdapterParam).apply(m)
    assert [p for p, _ in a] == [p for p, _ in b] == ["a"]


def test_as_selector_accepts_mixed_iterable():
    """``as_selector([name, Cls])`` unions name-kind and class filters."""
    m = _model()
    hits = spx.as_selector(["parameters", MyAdapterParam]).apply(m)
    paths = {p for p, _ in hits}
    assert paths == {"w", "a"}


def test_as_selector_rejects_mixed_garbage():
    """An iterable containing bad elements is rejected."""
    with pytest.raises(SelectorError):
        spx.as_selector(["parameters", 42])


def test_grad_wrt_class_returns_only_that_type():
    """``spx.grad(wrt=Cls)`` differentiates only the matching variables."""
    m = _GradModel()
    x = jnp.ones((4,), dtype=jnp.float32)

    @spx.jit
    def step(m, x):
        """Execute one training step and return the result."""

        def loss(m, x):
            """Compute the loss."""
            return m(x)

        return spx.grad(loss, wrt=MyAdapterParam)(m, x)

    grads = step(m, x)
    raw = grads.raw()
    assert raw.get("adapter")
    assert "parameters" not in raw or not raw["parameters"]


def test_partition_state_by_class_for_optax_style_split():
    """Class-based partition: ``as_selector(Cls).partition_state(...)``."""
    m = _model()
    _gdef, state = spx.export(m)
    trainable, frozen = spx.as_selector(MyAdapterParam).partition_state(m, state)
    assert list(trainable.raw().keys()) == ["adapter"]
    assert list(frozen.raw().keys()) == ["parameters"]
    assert sum(len(d) for d in trainable.raw().values()) == 1
    assert sum(len(d) for d in frozen.raw().values()) == 1


def test_of_type_subclass_relation_respected():
    """``of_type(ParentCls)`` matches subclass instances too (isinstance semantics)."""
    m = spx.Module()
    m.__init__()
    m.p = spx.Parameter(jnp.zeros((2,)))
    hits = spx.of_type(spx.Variable).apply(m)
    assert [p for p, _ in hits] == ["p"]
