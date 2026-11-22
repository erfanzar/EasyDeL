# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.inspect.tree`."""

from __future__ import annotations

from spectrax.core.state import State
from spectrax.inspect.tree import paths_and_shapes, tree_state
from spectrax.nn.linear import Linear
from spectrax.rng.rngs import Rngs


def test_tree_state_returns_state_instance():
    """:func:`tree_state` returns a :class:`State` object."""
    m = Linear(4, 4, rngs=Rngs(0))
    s = tree_state(m)
    assert isinstance(s, State)


def test_paths_and_shapes_covers_every_variable():
    """Every variable appears as one ``(collection, path, shape, dtype)`` row."""
    m = Linear(4, 8, rngs=Rngs(0))
    rows = paths_and_shapes(m)
    paths = {p for _, p, _, _ in rows}
    assert "weight" in paths and "bias" in paths


def test_paths_and_shapes_sorted_stable():
    """Rows are sorted by ``(collection, path)``."""
    m = Linear(4, 4, rngs=Rngs(0))
    rows = paths_and_shapes(m)
    keys = [(c, p) for c, p, _, _ in rows]
    assert keys == sorted(keys)


def test_paths_and_shapes_shape_matches_variable():
    """The ``shape`` column equals the variable's declared shape."""
    m = Linear(3, 5, rngs=Rngs(0))
    rows = {p: (s, d) for _, p, s, d in paths_and_shapes(m)}
    assert rows["weight"][0] == (3, 5)
    assert rows["bias"][0] == (5,)
