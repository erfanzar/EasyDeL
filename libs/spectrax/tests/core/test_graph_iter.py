# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :func:`spectrax.iter_modules` ``select=``, :func:`spectrax.iter_variables`,
and :func:`spectrax.find`.

Exercises all three filtering styles on a small but non-trivial model
(two Blocks + a LoRA adapter), asserting both counts and exact paths.
"""

from __future__ import annotations

import pytest

import spectrax as spx
import spectrax.nn as spx_nn


class _Block(spx.Module):
    """Linear -> LayerNorm -> Linear — used in multiple tests below."""

    def __init__(self, seed: int = 0):
        """Build a three-submodule block seeded deterministically."""
        super().__init__()
        self.fc1 = spx_nn.Linear(4, 4, rngs=spx.Rngs(seed))
        self.ln = spx_nn.LayerNorm(4)
        self.fc2 = spx_nn.Linear(4, 4, rngs=spx.Rngs(seed + 100))

    def forward(self, x):
        """Run ``fc2 ∘ ln ∘ fc1`` for shape-compatibility smoke checks."""
        return self.fc2(self.ln(self.fc1(x)))


class _Net(spx.Module):
    """Two :class:`_Block` s wrapped by a :class:`~spectrax.nn.LoRA` adapter.

    The model has exactly 10 unique modules (root Net + 2 Blocks + 3
    submodules per Block + LoRA = 1+2+6+1) and 14 unique variables
    (2 LoRA factors + 12 Parameters across Blocks), giving several
    anchor counts for the filter tests below.
    """

    def __init__(self):
        """Wire the two seeded blocks and the LoRA adapter."""
        super().__init__()
        self.b0 = _Block(seed=0)
        self.b1 = _Block(seed=1)
        self.lora = spx_nn.LoRA(4, 2, 4, rngs=spx.Rngs(2))

    def forward(self, x):
        """Run the composed pipeline."""
        return self.lora(self.b1(self.b0(x)))


def test_iter_modules_default_yields_every_module_once():
    """With no selector, every unique Module is yielded in canonical order."""
    m = _Net()
    paths = [p for p, _ in spx.iter_modules(m)]
    assert len(paths) == 10
    assert "" in paths
    assert "b0" in paths
    assert "b0.fc1" in paths
    assert "lora" in paths


def test_iter_modules_select_class_filters_by_isinstance():
    """``select=Linear`` keeps only instances of :class:`~spectrax.nn.Linear`."""
    m = _Net()
    linears = list(spx.iter_modules(m, select=spx_nn.Linear))
    assert len(linears) == 4
    assert all(isinstance(mod, spx_nn.Linear) for _, mod in linears)
    assert sorted(p for p, _ in linears) == ["b0.fc1", "b0.fc2", "b1.fc1", "b1.fc2"]


def test_iter_modules_select_tuple_of_classes():
    """A tuple of classes acts as a union — keep any listed type."""
    m = _Net()
    hits = list(spx.iter_modules(m, select=(spx_nn.Linear, spx_nn.LayerNorm)))
    assert len(hits) == 6
    kinds = {type(mod).__name__ for _, mod in hits}
    assert kinds == {"Linear", "LayerNorm"}


def test_iter_modules_select_callable_predicate():
    """A ``(module, path) -> bool`` callable filters by arbitrary expression."""
    m = _Net()
    hits = list(spx.iter_modules(m, select=lambda mod, p: p.startswith("b0")))
    assert all(p.startswith("b0") for p, _ in hits)
    assert len(hits) == 4


def test_iter_modules_select_rejects_bad_input():
    """Non-class, non-callable ``select`` values raise :class:`TypeError`."""
    m = _Net()
    with pytest.raises(TypeError):
        list(spx.iter_modules(m, select="not a class or callable"))
    with pytest.raises(TypeError):
        list(spx.iter_modules(m, select=42))


def test_iter_modules_select_with_skip_root():
    """``skip_root`` and ``select`` compose: the root entry is always dropped."""
    m = _Net()
    hits = list(spx.iter_modules(m, skip_root=True, select=spx_nn.Linear))
    assert len(hits) == 4
    all_hits = list(spx.iter_modules(m, skip_root=True))
    assert all(p != "" for p, _ in all_hits)


def test_iter_modules_select_with_with_path_false():
    """``with_path=False`` yields bare modules; the filter still applies."""
    m = _Net()
    hits = list(spx.iter_modules(m, with_path=False, select=spx_nn.Linear))
    assert all(isinstance(mod, spx_nn.Linear) for mod in hits)
    assert len(hits) == 4


def test_iter_variables_no_filter_matches_live_variables():
    """``iter_variables(m)`` without a filter equals :func:`~spectrax.live_variables`."""
    m = _Net()
    a = list(spx.iter_variables(m))
    b = spx.live_variables(m)
    assert a == b


def test_iter_variables_filter_by_variable_subclass():
    """A :class:`~spectrax.Variable` subclass picks instances of that class."""
    m = _Net()
    hits = list(spx.iter_variables(m, select=spx_nn.LoraParameter))
    assert len(hits) == 2
    paths = {p for p, _ in hits}
    assert paths == {"lora.lora_a", "lora.lora_b"}


def test_iter_variables_filter_by_collection_string():
    """A collection-name string picks variables whose :attr:`kind` matches."""
    m = _Net()
    hits = list(spx.iter_variables(m, select="lora"))
    assert len(hits) == 2
    assert all(v.kind == "lora" for _, v in hits)


def test_iter_variables_filter_by_selector_object():
    """A :class:`~spectrax.Selector` passed through unchanged.

    :func:`spectrax.of_type` with :class:`~spectrax.Parameter` picks
    every Parameter but excludes :class:`~spectrax.nn.LoraParameter`
    because it does not subclass :class:`~spectrax.Parameter`.
    """
    m = _Net()
    sel = spx.of_type(spx.Parameter)
    hits = list(spx.iter_variables(m, select=sel))
    assert len(hits) == 12
    assert all(isinstance(v, spx.Parameter) for _, v in hits)
    assert not any(isinstance(v, spx_nn.LoraParameter) for _, v in hits)


def test_iter_variables_filter_by_callable_predicate():
    """A ``(variable, path) -> bool`` callable is wrapped into a Selector."""
    m = _Net()
    hits = list(spx.iter_variables(m, select=lambda v, p: "lora" in p))
    assert {p for p, _ in hits} == {"lora.lora_a", "lora.lora_b"}


def test_find_first_module_by_class():
    """``find`` auto-routes to module search when given a Module subclass."""
    m = _Net()
    hit = spx.find(m, spx_nn.Linear)
    assert hit is not None
    path, mod = hit
    assert path == "b0.fc1"
    assert isinstance(mod, spx_nn.Linear)


def test_find_first_variable_by_class():
    """Given a Variable subclass, ``find`` returns the first matching variable."""
    m = _Net()
    hit = spx.find(m, spx_nn.LoraParameter)
    assert hit is not None
    path, var = hit
    assert isinstance(var, spx_nn.LoraParameter)
    assert path == "lora.lora_a"


def test_find_first_variable_by_collection_string():
    """Collection-name strings route to variable search."""
    m = _Net()
    hit = spx.find(m, "lora")
    assert hit is not None
    _path, var = hit
    assert var.kind == "lora"


def test_find_returns_none_when_no_match():
    """``find`` returns ``None`` when nothing in the graph matches."""
    m = _Net()
    assert spx.find(m, "nonexistent_collection") is None

    class Unused(spx_nn.Linear):
        """Stand-in subclass that is never instantiated in the graph."""

    assert spx.find(m, Unused) is None


def test_find_requires_select():
    """Passing ``select=None`` to :func:`spectrax.find` is an error."""
    m = _Net()
    with pytest.raises(ValueError):
        spx.find(m, None)
