# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Predictive affordability respects decimal dollars without context rounding."""

from decimal import localcontext

import pytest

from eyvan.domain.prediction import DemandObservation, PredictiveConfig, predict_demand


def _prediction(*, budget: float, cost: float, max_cells: int = 10):
    return predict_demand(
        [DemandObservation(weekday=1, hour=9, wanted=True) for _ in range(3)],
        weekday=1,
        hour=9,
        config=PredictiveConfig(enabled=True, hourly_budget_usd=budget),
        cell_hourly_cost=cost,
        max_cells=max_cells,
    )


@pytest.mark.parametrize(
    ("budget", "cost", "cells"),
    [(0.3, 0.1, 3), (0.7, 0.1, 7), (1.0, 0.1, 10), (0.6, 0.2, 3)],
)
def test_exact_decimal_budget_affords_every_cell(budget, cost, cells):
    assert _prediction(budget=budget, cost=cost).cells == cells


def test_budget_just_under_threshold_does_not_round_up():
    assert _prediction(budget=0.29999999999999993, cost=0.1).cells == 2


def test_affordability_remains_bounded_by_max_cells():
    assert _prediction(budget=1.0, cost=0.1, max_cells=3).cells == 3


def test_tiny_positive_cost_remains_bounded_without_decimal_overflow():
    assert _prediction(budget=1.0, cost=1e-30, max_cells=3).cells == 3


def test_affordability_does_not_depend_on_decimal_context_precision():
    with localcontext() as context:
        context.prec = 2
        assert _prediction(budget=100.0, cost=0.1, max_cells=3).cells == 3


def test_positive_prediction_preserves_pattern_reason():
    prediction = _prediction(budget=0.3, cost=0.1)
    assert prediction.acts
    assert prediction.reason == "demand appeared in 100% of 3 observations of this hour"


def test_infinite_cost_remains_unaffordable():
    prediction = _prediction(budget=0.3, cost=float("inf"))
    assert prediction.cells == 0
    assert "no finite hourly cost" in prediction.reason


@pytest.mark.parametrize(("budget", "cap"), [(0.1, 10), (1.0, 1)])
def test_one_affordable_cell_is_actionable(budget, cap):
    prediction = _prediction(budget=budget, cost=0.1, max_cells=cap)
    assert prediction.cells == 1
    assert prediction.acts


@pytest.mark.parametrize("budget", [float("nan"), float("inf")])
def test_nonfinite_budget_preserves_existing_arithmetic_refusal(budget):
    with pytest.raises(ValueError):
        _prediction(budget=budget, cost=0.1)
