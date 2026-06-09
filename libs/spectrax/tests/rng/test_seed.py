# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.rng.seed`."""

from __future__ import annotations

import pytest

from spectrax.rng.rngs import Rngs
from spectrax.rng.seed import default_rngs, has_default_rngs, seed


def test_no_default_rngs_outside_seed():
    """:func:`default_rngs` raises when no seed context is active."""
    assert not has_default_rngs()
    with pytest.raises(RuntimeError):
        default_rngs()


def test_seed_context_pushes_rngs():
    """Inside ``seed(n)`` the default is the associated :class:`Rngs`."""
    with seed(3) as r:
        assert has_default_rngs()
        assert default_rngs() is r


def test_seed_context_pops_on_exit():
    """Leaving a seed block clears the default rngs."""
    with seed(0):
        pass
    assert not has_default_rngs()


def test_nested_seed_stacks_correctly():
    """Nested seeds form a stack — inner wins while active."""
    with seed(1) as outer, seed(2) as inner:
        assert default_rngs() is inner
    with seed(1) as outer:
        assert default_rngs() is outer


def test_seed_accepts_rngs_instance():
    """Passing an existing :class:`Rngs` is a passthrough."""
    r = Rngs(7)
    with seed(r) as active:
        assert active is r


def test_has_default_rngs_reflects_state():
    """``has_default_rngs`` transitions from False -> True -> False."""
    assert not has_default_rngs()
    with seed(0):
        assert has_default_rngs()
    assert not has_default_rngs()
