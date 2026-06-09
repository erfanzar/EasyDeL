# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.core.errors`."""

from __future__ import annotations

import pytest

from spectrax.core.errors import (
    CyclicGraphError,
    GraphStructureError,
    IllegalMutationError,
    LazyInitUnderTransformError,
    PolicyError,
    SelectorError,
    SpecTraxError,
)


def test_error_hierarchy_inheritance():
    """Every spectrax error descends from :class:`SpecTraxError`."""
    for cls in (
        CyclicGraphError,
        IllegalMutationError,
        LazyInitUnderTransformError,
        SelectorError,
        PolicyError,
        GraphStructureError,
    ):
        assert issubclass(cls, SpecTraxError)
        assert issubclass(cls, Exception)


def test_error_can_be_raised_and_caught_as_spectrax_error():
    """Any spectrax-specific error is catchable as :class:`SpecTraxError`."""
    with pytest.raises(SpecTraxError):
        raise CyclicGraphError("cycle")
    with pytest.raises(SpecTraxError):
        raise IllegalMutationError("bad")


def test_error_message_preserved():
    """The error message is stored and retrievable via ``str``."""
    e = SelectorError("boom")
    assert str(e) == "boom"


def test_error_distinct_types():
    """Each exception subclass is distinct from the others."""
    seen = {
        CyclicGraphError,
        IllegalMutationError,
        LazyInitUnderTransformError,
        SelectorError,
        PolicyError,
        GraphStructureError,
    }
    assert len(seen) == 6
