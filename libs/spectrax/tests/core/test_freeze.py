# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :meth:`Module.freeze` and :meth:`Module.unfreeze`."""

from __future__ import annotations

from spectrax.core.graph import live_variables
from spectrax.core.module import Module
from spectrax.nn.linear import Linear
from spectrax.rng.rngs import Rngs


class _Tower(Module):
    """Two-layer model used for freeze tests."""

    def __init__(self, rngs):
        """Initialize with a, b."""
        super().__init__()
        self.a = Linear(4, 4, rngs=rngs)
        self.b = Linear(4, 4, rngs=rngs)


def test_freeze_moves_parameters_to_buffers():
    """After ``freeze('parameters')`` no variable has kind 'parameters'."""
    m = _Tower(Rngs(0))
    m.freeze("parameters")
    kinds = {v.kind for _, v in live_variables(m)}
    assert "parameters" not in kinds


def test_unfreeze_restores_original_kind():
    """``unfreeze`` round-trips kinds back to 'parameters'."""
    m = _Tower(Rngs(0))
    m.freeze("parameters")
    m.unfreeze("buffers")
    kinds = {v.kind for _, v in live_variables(m)}
    assert "parameters" in kinds


def test_freeze_selective_by_submodule():
    """Freezing only one submodule's parameters leaves the other's untouched."""
    from spectrax.core.selector import select

    m = _Tower(Rngs(0))
    m.freeze(select().variables("parameters").at_path("a.*"))
    kinds_a = {v.kind for p, v in live_variables(m) if p.startswith("a.")}
    kinds_b = {v.kind for p, v in live_variables(m) if p.startswith("b.")}
    assert "parameters" not in kinds_a
    assert "parameters" in kinds_b
