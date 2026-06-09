# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for schedule-driven treduce helpers."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from spectrax.runtime.mpmd.treduce import Add, Concat, _HashableOps, treduce_i
from spectrax.runtime.schedules import GPipe


def test_hashable_ops_uses_value_equality_for_cache_keys():
    """Fresh but equivalent operation tuples should hash and compare equal."""
    a = _HashableOps((Concat(4), Add()))
    b = _HashableOps((Concat(4), Add()))

    assert a == b
    assert hash(a) == hash(b)


def test_treduce_rejects_short_operation_sequence():
    """Missing per-output operations should fail instead of silently defaulting to Add."""

    def body(i):
        """Loop body function."""
        value = i.astype(jnp.float32)
        return value, value + 1.0

    with pytest.raises(ValueError, match="shorter than the body output"):
        treduce_i(body, 2, GPipe(microbatches=2), operation=(Concat(2),))
