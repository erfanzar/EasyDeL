# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exhaustive tests for :mod:`spectrax.inspect.summary`."""

from __future__ import annotations

import jax.numpy as jnp

from spectrax.inspect.summary import summary
from spectrax.nn.linear import Linear
from spectrax.rng.rngs import Rngs


def test_summary_returns_string():
    """:func:`summary` returns a multi-line string."""
    m = Linear(4, 4, rngs=Rngs(0))
    s = summary(m)
    assert isinstance(s, str)
    assert "Linear" in s


def test_summary_shows_param_count():
    """The summary includes a ``parameters=`` tally."""
    m = Linear(4, 4, rngs=Rngs(0))
    s = summary(m)
    assert "parameters=" in s


def test_summary_with_example_inputs_reports_output_spec():
    """Passing example inputs triggers an output-spec line."""
    m = Linear(4, 4, rngs=Rngs(0))
    s = summary(m, jnp.zeros((2, 4)))
    assert "output" in s


def test_summary_is_deterministic():
    """Repeated calls produce identical text."""
    m = Linear(4, 4, rngs=Rngs(0))
    a = summary(m)
    b = summary(m)
    assert a == b


def test_summary_includes_path_rows():
    """Per-variable rows include ``weight`` and ``bias``."""
    m = Linear(4, 4, rngs=Rngs(0))
    s = summary(m)
    assert "weight" in s
    assert "bias" in s
