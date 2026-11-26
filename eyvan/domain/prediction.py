# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Warming capacity before it is asked for, and the four gates that stop it.

**This ships disabled, at a zero budget, and that is the design rather than a
staging decision.** The analysis that kept it off is that predictive scale-up is
economically strictly worse than ``CapacityBelief.should_probe``: a probe spends
to *learn* something the fleet does not know and can act on, while a prediction
spends on a guess about demand that has not arrived. When the guess is right it
saves one provisioning latency; when it is wrong it buys an accelerator nobody
asked for, and it is wrong exactly when demand changes, which is when an idle
accelerator costs the most.

So the honest version is not a better predictor. It is a predictor an operator
has to *fund* before it can spend, kept small enough to audit:

* **enabled** must be turned on -- a feature nobody chose is not a feature; and
* **budget** must be non-zero -- an operator states in dollars what a saved
  provisioning latency is worth to them; and
* **history** must be long enough that a pattern is a pattern; and
* **agreement** across observations, so one busy Tuesday does not warm every
  Tuesday forever.

Any one of the four failing means no prediction. They are separate rather than
combined into a score because a score can be tuned until it says yes, and four
booleans cannot.

Pure, and deliberately so. The decision is a function of history and
configuration with no clock of its own, no store and no provider -- which is
what makes an economic argument about it checkable rather than a matter of
watching a fleet for a month.
"""

from __future__ import annotations

import math
import typing as tp
from dataclasses import dataclass

from eyvan.domain.costing import money

#: Observations of the same hour needed before a pattern counts as one.
#:
#: Three weeks of one weekday. Two would let a fortnight's coincidence warm
#: capacity every week; more would mean a genuinely new schedule takes a month
#: to be noticed, and the cost of noticing late is the latency this exists to
#: remove -- exactly what the fleet already has.
MINIMUM_OBSERVATIONS = 3

#: How consistently demand must appear before it is predicted.
#:
#: Two thirds, not a majority. A pattern that holds barely more than half the
#: time is one that buys an idle accelerator nearly half the time, and the
#: purchase is real money while the saving is a latency.
MINIMUM_AGREEMENT = 2 / 3


@dataclass(frozen=True, slots=True)
class PredictiveConfig:
    """What an operator has to state before anything can be predicted.

    Both defaults are off. ``enabled`` alone still cannot spend, because the
    budget is zero -- two independent decisions, so neither a stray flag nor a
    stray number is sufficient on its own.
    """

    enabled: bool = False
    #: What one saved provisioning latency is worth, in dollars, per hour of
    #: warm capacity bought ahead of demand. Zero means the prediction is
    #: computed and never acted on, which is a useful state: an operator can
    #: watch what it *would* have done before paying for it.
    hourly_budget_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.hourly_budget_usd < 0:
            raise ValueError("a predictive budget cannot be negative")
        if not math.isfinite(self.hourly_budget_usd):
            raise ValueError("a predictive budget must be finite")


@dataclass(frozen=True, slots=True)
class DemandObservation:
    """One past hour, and whether this pool was wanted during it."""

    #: ISO weekday (Monday is 1) and hour, which is the coarsest grain a fleet's
    #: demand actually repeats on. Finer would need more history than any fleet
    #: has to say anything with confidence.
    weekday: int
    hour: int
    wanted: bool


@dataclass(frozen=True, slots=True)
class Prediction:
    """What the predictor concluded, and why.

    ``cells`` is zero whenever any gate failed, and ``reason`` says which -- so
    an operator reading a fleet that is not pre-warming learns whether the
    feature is off, unfunded, uncertain, or simply not expecting demand.
    """

    cells: int
    reason: str

    @property
    def acts(self) -> bool:
        return self.cells > 0


def predict_demand(
    observations: tp.Sequence[DemandObservation],
    *,
    weekday: int,
    hour: int,
    config: PredictiveConfig,
    cell_hourly_cost: float,
    max_cells: int,
) -> Prediction:
    """Decide how many cells to warm for one upcoming hour.

    Every gate is checked and named, in the order an operator would ask about
    them: is it on, is there enough history, does the history agree, and is it
    funded. An unfunded predictor still runs the history gates and reports the
    pattern it would have acted on, marked unfunded, so what it *would* do is
    observable before it costs anything.
    """

    if not config.enabled:
        return Prediction(0, "predictive scale-up is disabled")
    unfunded = config.hourly_budget_usd <= 0
    matching = [
        observation for observation in observations if observation.weekday == weekday and observation.hour == hour
    ]
    if len(matching) < MINIMUM_OBSERVATIONS:
        return Prediction(0, f"only {len(matching)} of {MINIMUM_OBSERVATIONS} observations of this hour")
    agreement = sum(1 for observation in matching if observation.wanted) / len(matching)
    if agreement < MINIMUM_AGREEMENT:
        return Prediction(0, f"demand appeared in {agreement:.0%} of observations, below {MINIMUM_AGREEMENT:.0%}")
    pattern = f"demand appeared in {agreement:.0%} of {len(matching)} observations of this hour"
    if unfunded:
        # The state an operator should sit in first. The prediction is computed
        # and reported, and nothing is bought, so what it *would* have done is
        # observable before it costs anything.
        return Prediction(0, f"{pattern}, unfunded")
    if cell_hourly_cost <= 0:
        # A free cell is not evidence of a working price; it is evidence of an
        # unpriced pool, and dividing a budget by it would warm the whole fleet.
        return Prediction(0, "this pool has no hourly cost, so a budget cannot bound the prediction")
    if not math.isfinite(cell_hourly_cost):
        # An infinite or NaN pool price cannot bound anything: refuse the pool
        # with a reason instead of crashing in int() on the ratio below.
        return Prediction(0, "this pool has no finite hourly cost, so a budget cannot bound the prediction")
    budget, cost = money(config.hourly_budget_usd), money(cell_hourly_cost)
    # The budget is validated finite at construction, and the cost was checked
    # just above, so both integer ratios are well defined. Integer ratios
    # retain decimal dollars without rounding the quotient or depending on the
    # process-wide Decimal precision for very small prices.
    budget_numerator, budget_denominator = budget.as_integer_ratio()
    cost_numerator, cost_denominator = cost.as_integer_ratio()
    affordable = (budget_numerator * cost_denominator) // (budget_denominator * cost_numerator)
    cells = min(affordable, max_cells)
    if cells <= 0:
        return Prediction(0, f"the budget affords no cell at {cell_hourly_cost}/h")
    return Prediction(cells, pattern)


__all__ = (
    "MINIMUM_AGREEMENT",
    "MINIMUM_OBSERVATIONS",
    "DemandObservation",
    "Prediction",
    "PredictiveConfig",
    "predict_demand",
)
