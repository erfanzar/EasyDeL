# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Exact money arithmetic.

``BudgetRecord.spent`` accumulates across every settlement and is then compared to
``limit`` by a hard admission gate, so drift here is not cosmetic: it decides whether
work is admitted. These tests assert exact equality, which is the whole point of
routing the arithmetic through ``Decimal`` -- a tolerance would hide the defect.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from eyvan.domain import costing


def test_repeated_addition_does_not_drift() -> None:
    """The accumulation pattern budgets actually use: read, add one charge, store.

    Three naive ``+= 0.1`` steps land on 0.30000000000000004, which is *greater* than
    0.3 -- so a budget of exactly 0.3 would refuse a third charge it should admit.
    Settlement repeats this hundreds of times over a budget period.
    """

    naive = 0.0
    for _ in range(3):
        naive += 0.1
    assert naive != 0.3

    running = 0.0
    for _ in range(3):
        running = costing.total(running, 0.1)
    assert running == 0.3


def test_a_hundred_settlements_of_a_third_of_a_cent_stay_exact() -> None:
    running = 0.0
    for _ in range(100):
        running = costing.total(running, 0.003)
    assert running == 0.3


def test_money_reads_a_float_as_written_not_as_stored() -> None:
    """Records carry JSON floats, so parsing has to go through str."""

    assert costing.money(0.1) == Decimal("0.1")
    # The float literal is the assertion: Decimal(0.1) is the binary expansion
    # (0.1000000000000000055511151231257827...), and money() must not produce it.
    assert costing.money(0.1) != Decimal(0.1)  # noqa: RUF032
    assert costing.money("2.50") == Decimal("2.50")
    assert costing.money(Decimal("7")) == Decimal("7")


@pytest.mark.parametrize(
    ("hourly", "hours", "expected"),
    [
        (10.0, 0.1, 1.0),
        (0.07, 3.0, 0.21),
        (2.5, 4.0, 10.0),
        (0.0, 5.0, 0.0),
        (3.0, 0.0, 0.0),
    ],
)
def test_cost_for_hours_is_exact(hourly: float, hours: float, expected: float) -> None:
    assert costing.cost_for_hours(hourly, hours) == expected


def test_cost_for_hours_matches_the_schedulers_own_reservation_math() -> None:
    """Settlement and reservation must agree, or a full-length run never nets to zero.

    ``core.scheduler._cost`` reserves ``hourly * duration_hours`` in Decimal. A lease
    held for exactly its estimate must settle to exactly what it reserved, otherwise
    every completed job leaves a residue in the budget.
    """

    hourly, duration_seconds = 0.07, 3600 * 3
    reserved = Decimal(str(hourly)) * (Decimal(duration_seconds) / Decimal(3600))

    assert costing.cost_for_hours(hourly, duration_seconds / 3600) == float(reserved)


def test_scale_drops_resources_that_scale_to_nothing() -> None:
    """A persisted {"gpu": 0.0} reads as "used no GPU", not "had no GPU"."""

    assert costing.scale({"tpu": 8.0, "gpu": 0.0}, 0.5) == {"tpu": 4.0}
    assert costing.scale({"tpu": 8.0}, 0.0) == {}


def test_scale_is_exact_for_fractional_factors() -> None:
    assert costing.scale({"tpu": 3.0}, 0.1) == {"tpu": 0.3}


def test_an_empty_charge_collection_has_json_safe_zero_total() -> None:
    result = costing.total()

    assert result == 0.0
    assert isinstance(result, float)
