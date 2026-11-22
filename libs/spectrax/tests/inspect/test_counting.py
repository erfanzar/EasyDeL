# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for :func:`format_parameters` and counting helpers."""

from __future__ import annotations

from spectrax.inspect.counting import count_bytes, count_parameters, format_parameters
from spectrax.nn.linear import Linear
from spectrax.rng.rngs import Rngs


def test_count_parameters_matches_expected():
    """``count_parameters`` equals weight + bias element counts."""
    m = Linear(4, 8, rngs=Rngs(0))
    assert count_parameters(m) == 4 * 8 + 8


def test_count_bytes_fp32_matches_count_parameters_times_4():
    """fp32 defaults use 4 bytes per element."""
    m = Linear(4, 8, rngs=Rngs(0))
    assert count_bytes(m) == count_parameters(m) * 4


def test_format_parameters_sub_thousand():
    """Under 1k stays integer-rendered."""
    assert format_parameters(42) == "42"


def test_format_parameters_thousands():
    """Thousands render with K suffix."""
    assert format_parameters(12_345) == "12.3K"


def test_format_parameters_millions():
    """Millions render with M suffix."""
    assert format_parameters(1_200_000) == "1.2M"


def test_format_parameters_billions():
    """Billions render with B suffix."""
    assert format_parameters(1_500_000_000) == "1.5B"
