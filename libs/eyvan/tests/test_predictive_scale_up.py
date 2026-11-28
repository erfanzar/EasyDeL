# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A predictor that cannot spend until an operator funds it.

The design's own analysis is that predictive scale-up is economically strictly
worse than ``should_probe``: a probe spends to learn something the fleet can act
on, while a prediction spends on a guess about demand that has not arrived. Right,
it saves one provisioning latency. Wrong, it buys an accelerator nobody asked for
-- and it is wrong exactly when demand changes, which is when an idle accelerator
costs the most.

So the tests that matter most here are the ones asserting it does *nothing*. The
default state, the unfunded state, the thin-history state and the ambiguous-history
state are four separate refusals, and each is checked separately because a score
can be tuned until it says yes and four booleans cannot.
"""

from __future__ import annotations

import pytest

from eyvan.domain.prediction import (
    MINIMUM_AGREEMENT,
    MINIMUM_OBSERVATIONS,
    DemandObservation,
    PredictiveConfig,
    predict_demand,
)


def _history(*, wanted: int, total: int, weekday: int = 1, hour: int = 9) -> list[DemandObservation]:
    return [DemandObservation(weekday=weekday, hour=hour, wanted=index < wanted) for index in range(total)]


def _predict(observations, config, *, cost: float = 10.0, max_cells: int = 4):
    return predict_demand(observations, weekday=1, hour=9, config=config, cell_hourly_cost=cost, max_cells=max_cells)


FUNDED = PredictiveConfig(enabled=True, hourly_budget_usd=100.0)


# --------------------------------------------------------------------------- refusals


def test_it_is_off_by_default() -> None:
    """A feature nobody chose is not a feature.

    The default config spends nothing regardless of how convincing the history
    is, which is the only state most fleets will ever be in.
    """

    prediction = _predict(_history(wanted=10, total=10), PredictiveConfig())

    assert prediction.acts is False
    assert "disabled" in prediction.reason


def test_enabling_it_is_not_enough_to_spend() -> None:
    """Two independent decisions, so neither a stray flag nor a stray number acts.

    This is also the state an operator should sit in first: the prediction is
    computed and reported, nothing is bought, and what it *would* have done is
    observable before it costs anything.
    """

    prediction = _predict(_history(wanted=10, total=10), PredictiveConfig(enabled=True))

    assert prediction.acts is False
    assert prediction.cells == 0
    assert "unfunded" in prediction.reason
    # The history gates ran: the report names the pattern, not the missing budget.
    assert "100%" in prediction.reason


def test_an_unfunded_watcher_still_reports_a_thin_history() -> None:
    """Watch mode computes the prediction, so its refusals are history's, not the budget's.

    An operator watching before funding needs to see what the predictor thinks of
    the history it has -- not the same "no budget" line every hour, which would
    tell them nothing about what funding would buy.
    """

    prediction = _predict(
        _history(wanted=MINIMUM_OBSERVATIONS - 1, total=MINIMUM_OBSERVATIONS - 1), PredictiveConfig(enabled=True)
    )

    assert prediction.acts is False
    assert f"only {MINIMUM_OBSERVATIONS - 1} of {MINIMUM_OBSERVATIONS}" in prediction.reason
    assert "budget" not in prediction.reason


def test_a_thin_history_predicts_nothing() -> None:
    """Two weeks of coincidence must not warm capacity every week."""

    prediction = _predict(_history(wanted=MINIMUM_OBSERVATIONS - 1, total=MINIMUM_OBSERVATIONS - 1), FUNDED)

    assert prediction.acts is False
    assert str(MINIMUM_OBSERVATIONS) in prediction.reason


def test_an_ambiguous_history_predicts_nothing() -> None:
    """A pattern holding barely more than half the time buys an idle accelerator
    nearly half the time, and the purchase is money while the saving is latency."""

    prediction = _predict(_history(wanted=3, total=6), FUNDED)

    assert prediction.acts is False
    assert "below" in prediction.reason


def test_an_unpriced_pool_predicts_nothing() -> None:
    """A free cell is evidence of an unpriced pool, not of a working price.

    Dividing a budget by zero cost would warm the entire fleet, which is the
    single worst outcome this feature could produce.
    """

    prediction = _predict(_history(wanted=10, total=10), FUNDED, cost=0.0)

    assert prediction.acts is False
    assert "no hourly cost" in prediction.reason


def test_a_budget_too_small_for_one_cell_predicts_nothing() -> None:
    """Rounding up to one would make every budget a one-cell budget."""

    prediction = _predict(_history(wanted=10, total=10), PredictiveConfig(enabled=True, hourly_budget_usd=1.0))

    assert prediction.acts is False
    assert "affords no cell" in prediction.reason


def test_a_negative_budget_is_refused_at_construction() -> None:
    """A negative budget is a typo, and one that would otherwise read as "off"."""

    with pytest.raises(ValueError, match="cannot be negative"):
        PredictiveConfig(enabled=True, hourly_budget_usd=-1.0)


# --------------------------------------------------------------------------- acting


def test_a_funded_consistent_pattern_predicts_capacity() -> None:
    """The gate must not block the configuration it exists to permit."""

    # ``max_cells`` raised above what the budget affords, so this asserts the
    # budget-driven number rather than the pool ceiling -- two separate bounds
    # that a single fixture would conflate.
    prediction = _predict(_history(wanted=10, total=10), FUNDED, max_cells=50)

    assert prediction.acts is True
    assert prediction.cells == 10


def test_the_budget_bounds_the_prediction() -> None:
    """The operator's dollar figure is the ceiling, not a suggestion."""

    prediction = _predict(_history(wanted=10, total=10), PredictiveConfig(enabled=True, hourly_budget_usd=25.0))

    assert prediction.cells == 2


def test_the_pool_ceiling_also_bounds_it() -> None:
    """A budget larger than the pool must not predict cells the pool cannot hold."""

    prediction = _predict(_history(wanted=10, total=10), FUNDED, max_cells=3)

    assert prediction.cells == 3


def test_exactly_the_threshold_agreement_is_enough() -> None:
    """A boundary stated in the constant should behave as the constant says.

    Otherwise the documented threshold and the effective one differ by one
    observation, which nobody discovers until they are counting.
    """

    total = 6
    prediction = _predict(_history(wanted=round(total * MINIMUM_AGREEMENT), total=total), FUNDED)

    assert prediction.acts is True


def test_only_the_matching_hour_counts() -> None:
    """Demand at 09:00 on Monday says nothing about 09:00 on Tuesday.

    Pooling the week would let one busy day warm every day, which is the failure
    mode that makes a predictor expensive rather than merely useless.
    """

    other_hours = [DemandObservation(weekday=2, hour=9, wanted=True) for _ in range(10)]

    prediction = _predict(other_hours, FUNDED)

    assert prediction.acts is False
    assert "0 of" in prediction.reason


# --------------------------------------------------------------------------- honesty


def test_every_refusal_says_which_gate_stopped_it() -> None:
    """An operator reading a fleet that is not pre-warming should learn why.

    "Off", "unfunded", "uncertain" and "not expecting demand" call for four
    different responses, and a single "no" calls for guessing.
    """

    reasons = {
        _predict(_history(wanted=10, total=10), PredictiveConfig()).reason,
        _predict(_history(wanted=10, total=10), PredictiveConfig(enabled=True)).reason,
        _predict(_history(wanted=1, total=1), FUNDED).reason,
        _predict(_history(wanted=3, total=6), FUNDED).reason,
    }

    assert len(reasons) == 4
    assert all(reason for reason in reasons)


def test_the_predictor_has_no_clock_and_no_store() -> None:
    """Purity is what makes the economic argument checkable.

    A predictor that read the time or a database could only be argued about by
    watching a fleet for a month; this one can be argued about here.
    """

    import inspect

    source = inspect.getsource(predict_demand)

    for forbidden in ("utcnow", "datetime.now", "repository", "await "):
        assert forbidden not in source
