# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Finite-value contracts the money and belief helpers were missing.

``costing`` sits under every dollar in the ledger, but accepted NaN quietly and
crashed on magnitudes past the decimal context; ``CapacityBelief.observe``
admitted NaN latencies that poison the rolling p95 window permanently; a
zero-quantity resource forced a spurious ``cost_unknown``; and a predictive
budget of NaN/inf passed validation only to die later inside an ``int()``.
"""

from __future__ import annotations

import math
import sys
from datetime import UTC, datetime
from decimal import ROUND_HALF_UP, Decimal, localcontext

import pytest

from eyvan.domain.availability import CapacityBelief, CapacityPoolKey, ProvisionOutcome
from eyvan.domain.costing import cost_for_hours, scale, total
from eyvan.domain.models import CapacityCell, ResourceVector
from eyvan.domain.prediction import DemandObservation, PredictiveConfig, predict_demand


def test_total_survives_large_magnitudes() -> None:
    assert total(1e30) == pytest.approx(1e30)
    assert total(1e25, 1e25, 1e25) == pytest.approx(3e25)


@pytest.mark.parametrize("amount", [1e30, -1e30, 1e308, -1e308, sys.float_info.max, -sys.float_info.max])
def test_money_helpers_return_huge_finite_floats(amount: float) -> None:
    assert total(amount) == amount
    assert cost_for_hours(amount, 1) == amount
    assert scale({"gpu": amount}, 1) == {"gpu": amount}


def test_huge_money_cancellation_keeps_the_final_small_charge() -> None:
    assert total(1e308, -1e308, Decimal("0.12345678905")) == 0.123456789


@pytest.mark.parametrize(
    ("amount", "expected"),
    [("0.00000000005", 0.0), ("0.00000000015", 0.0000000002), ("9.99999999995", 10.0)],
)
def test_money_quantum_keeps_half_even_and_rounding_carry(amount: str, expected: float) -> None:
    assert total(amount) == expected


def test_money_conversion_preserves_the_callers_context_and_rounding() -> None:
    with localcontext() as context:
        context.prec = 6
        context.rounding = ROUND_HALF_UP
        before = context.copy()
        assert cost_for_hours("1.23456e30", 1) == 1.23456e30
        assert total("0.00000000005") == 0.0000000001
        assert context.prec == before.prec
        assert context.rounding == before.rounding
        assert context.flags == before.flags


@pytest.mark.parametrize("amount", ["1e309", "-1e309", "1e1000000000", "-1e1000000000"])
def test_unrepresentable_money_raises_domain_error_without_growing_precision(amount: str) -> None:
    with pytest.raises(ValueError, match="finite"):
        total(Decimal(amount))
    with pytest.raises(ValueError, match="finite"):
        cost_for_hours(Decimal(amount), 1)
    with pytest.raises(ValueError, match="finite"):
        scale({"gpu": 1.0}, Decimal(amount))


def test_money_arithmetic_overflow_raises_domain_error() -> None:
    with pytest.raises(ValueError, match="finite"):
        total(sys.float_info.max, sys.float_info.max)
    with pytest.raises(ValueError, match="finite"):
        cost_for_hours(sys.float_info.max, 2)
    with pytest.raises(ValueError, match="finite"):
        scale({"gpu": sys.float_info.max}, 2)


@pytest.mark.parametrize("negative", [False, True])
def test_money_quantization_cannot_cross_the_finite_float_boundary(negative: bool) -> None:
    with localcontext() as context:
        context.prec = 400
        boundary = Decimal(2**1024 - 2**970)
        amount = boundary - Decimal("0.00000000001")
        if negative:
            amount = -amount
        assert math.isfinite(float(amount))
        with pytest.raises(ValueError, match="finite"):
            total(amount)
        with pytest.raises(ValueError, match="finite"):
            cost_for_hours(amount, 1)
        with pytest.raises(ValueError, match="finite"):
            scale({"gpu": 1.0}, amount)


def test_total_and_scale_reject_nonfinite_money() -> None:
    with pytest.raises(ValueError, match="finite"):
        total(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        total(float("inf"))
    with pytest.raises(ValueError, match="finite"):
        scale({"gpu": 1.0}, float("nan"))


def _belief() -> CapacityBelief:
    return CapacityBelief(CapacityPoolKey("pool", "x", "y", "t", "c"))


def test_observe_rejects_nonfinite_latency() -> None:
    with pytest.raises(ValueError, match="latency"):
        _belief().observe(
            ProvisionOutcome.SUCCESS,
            now=datetime(2026, 9, 26, tzinfo=UTC),
            latency_s=float("nan"),
        )
    with pytest.raises(ValueError, match="latency"):
        _belief().observe(
            ProvisionOutcome.SUCCESS,
            now=datetime(2026, 9, 26, tzinfo=UTC),
            latency_s=float("inf"),
        )


class _PricedCell:
    """Just the attribute ``prices_cover`` reads; the full cell is irrelevant."""

    unit_prices = ResourceVector({"tpu": 1})


def test_zero_quantity_request_does_not_force_cost_unknown() -> None:
    request = ResourceVector({"tpu": 4, "gpu": 0})
    assert CapacityCell.prices_cover(_PricedCell(), request) is True


def test_predictive_config_rejects_nonfinite_budget() -> None:
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            PredictiveConfig(enabled=True, hourly_budget_usd=bad)


def test_infinite_cost_prediction_stays_unaffordable_without_crashing() -> None:
    prediction = predict_demand(
        [DemandObservation(weekday=1, hour=9, wanted=True) for _ in range(3)],
        weekday=1,
        hour=9,
        config=PredictiveConfig(enabled=True, hourly_budget_usd=0.3),
        cell_hourly_cost=float("inf"),
        max_cells=10,
    )
    assert prediction.cells == 0
