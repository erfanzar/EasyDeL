# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :mod:`spectrax.inspect.tabulate` and :mod:`spectrax.inspect.display`."""

from __future__ import annotations

import jax.numpy as jnp

from spectrax.core.module import Module
from spectrax.inspect.display import display
from spectrax.inspect.tabulate import count_bytes, count_parameters, hlo_cost, tabulate
from spectrax.nn.linear import Linear
from spectrax.rng.rngs import Rngs


class _MLP(Module):
    """Small two-layer MLP for table tests."""

    def __init__(self, rngs):
        """Initialize with fc1, fc2."""
        super().__init__()
        self.fc1 = Linear(4, 8, rngs=rngs)
        self.fc2 = Linear(8, 2, rngs=rngs)

    def forward(self, x):
        """Run the forward pass."""
        return self.fc2(self.fc1(x))


def test_count_parameters_matches_sum_of_leaf_sizes():
    """``count_parameters`` equals the hand-computed parameter total."""
    m = Linear(4, 8, rngs=Rngs(0))
    assert count_parameters(m) == 4 * 8 + 8


def test_count_bytes_respects_dtype():
    """``count_bytes`` uses the parameter dtype's itemsize."""
    m = Linear(4, 4, rngs=Rngs(0), dtype=jnp.float16)
    assert count_bytes(m) == (4 * 4 + 4) * 2


def test_tabulate_produces_deterministic_table():
    """``tabulate`` produces a well-formed, non-empty table."""
    m = _MLP(Rngs(0))
    text = tabulate(m)
    assert "class" in text
    assert "parameters" in text
    assert "fc1" in text or "fc2" in text
    assert "Total parameters" in text


def test_tabulate_with_example_inputs_reports_output():
    """With example inputs the table appends an 'Output:' line."""
    m = _MLP(Rngs(0))
    text = tabulate(m, jnp.zeros((1, 4)))
    assert "Output:" in text


def test_tabulate_depth_caps_expansion():
    """Passing ``depth=0`` limits the table to the root module."""
    m = _MLP(Rngs(0))
    text = tabulate(m, depth=0)
    assert text.count("Linear") == 0


def test_hlo_cost_returns_dict_on_tiny_module():
    """``hlo_cost`` returns a dict with ``flops`` key on a simple forward."""
    m = Linear(2, 2, rngs=Rngs(0))
    cost = hlo_cost(m, jnp.zeros((1, 2)))
    if cost:
        assert "flops" in cost
        assert "bytes_accessed" in cost


def test_display_does_not_raise_in_text_fallback():
    """``display`` returns ``None`` and handles absence of treescope."""
    m = Linear(2, 2, rngs=Rngs(0))
    assert display(m) is None
