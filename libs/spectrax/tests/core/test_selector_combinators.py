# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for selector combinators: ``&``, ``-``, ``Everything``, ``Nothing``, path_*."""

from __future__ import annotations

from spectrax.core.module import Module
from spectrax.core.selector import (
    Everything,
    Nothing,
    all_of,
    any_of,
    not_,
    path_contains,
    path_endswith,
    path_startswith,
    select,
)
from spectrax.nn.linear import Linear
from spectrax.nn.norm import BatchNorm1d
from spectrax.rng.rngs import Rngs


class _Tower(Module):
    """Module with a Linear + BatchNorm submodule for selector testing."""

    def __init__(self, rngs):
        """Initialize with fc, bn."""
        super().__init__()
        self.fc = Linear(4, 4, rngs=rngs)
        self.bn = BatchNorm1d(4)


def test_everything_matches_all_variables():
    """``Everything`` returns every variable."""
    m = _Tower(Rngs(0))
    ev = Everything.apply(m)
    all_vars = select().apply(m)
    assert len(ev) == len(all_vars)


def test_nothing_matches_no_variable():
    """``Nothing`` returns an empty list."""
    m = _Tower(Rngs(0))
    assert Nothing.apply(m) == []


def test_intersection_combinator_returns_both_matches():
    """``a & b`` matches variables matched by both."""
    m = _Tower(Rngs(0))
    a = select().variables("parameters")
    b = select().at_instances_of(Linear)
    only_linear_params = (a & b).apply(m)
    paths = [p for p, _ in only_linear_params]
    assert all("fc" in p for p in paths)


def test_union_combinator_returns_either():
    """``a | b`` matches variables matched by either."""
    m = _Tower(Rngs(0))
    a = select().variables("parameters")
    b = select().variables("batch_stats")
    combined = (a | b).apply(m)
    kinds = {v.kind for _, v in combined}
    assert "parameters" in kinds
    assert "batch_stats" in kinds


def test_difference_combinator_subtracts():
    """``a - b`` matches a minus b."""
    m = _Tower(Rngs(0))
    a = Everything
    b = select().variables("batch_stats")
    diff = (a - b).apply(m)
    kinds = {v.kind for _, v in diff}
    assert "batch_stats" not in kinds


def test_all_of_empty_returns_everything():
    """``all_of()`` with no args returns :data:`Everything`."""
    assert all_of() is Everything


def test_any_of_empty_returns_nothing():
    """``any_of()`` with no args returns :data:`Nothing`."""
    assert any_of() is Nothing


def test_not_is_invert():
    """``not_(s)`` inverts the selector."""
    m = _Tower(Rngs(0))
    a = select().variables("parameters")
    inverted = not_(a).apply(m)
    kinds = {v.kind for _, v in inverted}
    assert "parameters" not in kinds


def test_path_contains_finds_substring():
    """``path_contains('fc')`` matches variables with 'fc' in path."""
    m = _Tower(Rngs(0))
    r = path_contains("fc").apply(m)
    assert all("fc" in p for p, _ in r)


def test_path_endswith_matches_suffix():
    """``path_endswith('weight')`` matches trailing path segments."""
    m = _Tower(Rngs(0))
    r = path_endswith("weight").apply(m)
    assert all(p.endswith("weight") for p, _ in r)


def test_path_startswith_matches_prefix():
    """``path_startswith('fc')`` matches prefix."""
    m = _Tower(Rngs(0))
    r = path_startswith("fc").apply(m)
    assert all(p.startswith("fc") for p, _ in r)


def test_composed_combinators():
    """``(a | b) & c`` composes cleanly."""
    m = _Tower(Rngs(0))
    sel = (select().variables("parameters") | select().variables("batch_stats")) & path_contains("bn")
    result = sel.apply(m)
    assert all("bn" in p for p, _ in result)
