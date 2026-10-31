# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.inspect.repr`."""

from __future__ import annotations

from spectrax.inspect.repr import _ascii_tree, repr_module
from spectrax.nn.linear import Linear
from spectrax.rng.rngs import Rngs


def test_repr_module_returns_string():
    """``repr_module`` produces a non-empty string."""
    m = Linear(4, 4, rngs=Rngs(0))
    s = repr_module(m)
    assert isinstance(s, str)
    assert s


def test_repr_module_mentions_class():
    """Output includes the module class name."""
    m = Linear(4, 4, rngs=Rngs(0))
    assert "Linear" in repr_module(m)


def test_ascii_tree_shows_hyperparameters():
    """The PyTorch-style renderer shows the layer's static hyperparameters."""
    m = Linear(4, 4, rngs=Rngs(0))
    text = _ascii_tree(m)
    assert "in_features=4" in text
    assert "out_features=4" in text


def test_module_repr_delegates_to_repr_module():
    """``repr(module)`` ultimately calls :func:`repr_module`."""
    m = Linear(4, 4, rngs=Rngs(0))
    assert "Linear" in repr(m)
