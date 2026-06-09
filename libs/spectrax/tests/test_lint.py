# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.lint`."""

from __future__ import annotations

import jax.numpy as jnp

from spectrax.core.module import Module
from spectrax.core.variable import Parameter
from spectrax.lint import check_unintentional_sharing
from spectrax.nn.linear import Linear
from spectrax.rng.rngs import Rngs


def test_lint_empty_for_clean_module():
    """A module without sharing reports no aliases."""
    m = Linear(4, 4, rngs=Rngs(0))
    assert check_unintentional_sharing(m) == []


def test_lint_flags_unintentional_shared_variable():
    """Tied parameters without a ``tie_group`` are flagged as alias pairs."""

    class Tied(Module):
        """Module with an accidentally-aliased parameter."""

        def __init__(self):
            """Create ``a`` and alias it as ``b``."""
            super().__init__()
            self.a = Parameter(jnp.zeros(4))
            self.b = self.a

        def forward(self, x):
            """Pass-through stub."""
            return x

    flagged = check_unintentional_sharing(Tied())
    assert flagged


def test_lint_respects_tie_group_tag():
    """Aliased variables with a ``tie_group`` metadata tag are NOT flagged."""

    class Tied(Module):
        """Module with an *intentionally* tied parameter."""

        def __init__(self):
            """Create ``a`` with a ``tie_group`` tag and alias as ``b``."""
            super().__init__()
            self.a = Parameter(jnp.zeros(4), metadata={"tie_group": "emb"})
            self.b = self.a

        def forward(self, x):
            """Pass-through stub."""
            return x

    assert check_unintentional_sharing(Tied()) == []


def test_lint_shared_module_not_variable_counts():
    """Sharing at the Module level (not Variable level) also produces pairs."""

    class TiedMod(Module):
        """Module that aliases an entire sub-module."""

        def __init__(self):
            """Create ``fc`` and alias as ``fc2``."""
            super().__init__()
            self.fc = Linear(4, 4, rngs=Rngs(0))
            self.fc2 = self.fc

        def forward(self, x):
            """Apply the shared layer twice."""
            return self.fc2(self.fc(x))

    flagged = check_unintentional_sharing(TiedMod())
    assert flagged
