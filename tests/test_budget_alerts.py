# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Budget alerts, period rolling, and the forecast that was never computed.

``BudgetRecord.period`` was a string nothing acted on. That was survivable only
while ``spent`` held a duration *estimate* that ran roughly as often high as low;
once settlement became actual elapsed cost, ``spent`` became a true monotone
total and "monthly" became a lie -- every threshold would fire once in the
fleet's lifetime and never again.

Three properties follow, and each is asserted here:

**A threshold fires once per period.** An alert that repeats on every settlement
past 80% is an alert nobody reads.

**Rolling clears the fired set.** Carrying it across the boundary would silence
every warning for the rest of the fleet's life after the first month that
reached them -- the same lifetime-budget bug, one level up.

**The forecast is derived and pessimistic.** It assumes every active lease runs
to the end of the window, because the number exists to warn *before* a hard limit
starts refusing admission, and a forecast that assumed early completion would
under-warn exactly when it mattered.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from eyvan.records import (
    BudgetRecord,
    LeaseRecord,
    LeaseState,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    utcnow,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import NotFoundError


def _admin() -> Principal:
    return Principal(
        subject="root",
        grants=(RoleGrant(role=Role.FLEET_ADMIN), RoleGrant(role=Role.ORG_ADMIN, org_id="acme")),
    )


async def _service() -> ControlPlaneService:
    repository = InMemoryRepository()
    await repository.open()
    return ControlPlaneService(repository=repository)


def _budget(**overrides: object) -> BudgetRecord:
    defaults: dict[str, object] = {
        "org_id": "acme",
        "project_id": "research",
        "limit": 1000.0,
        "updated_by": "root",
    }
    return BudgetRecord(**{**defaults, **overrides})


async def _put(service: ControlPlaneService, budget: BudgetRecord) -> None:
    async with service.repository.transaction() as tx:
        await tx.put_budget(budget)


def test_thresholds_must_be_fractions_and_must_not_repeat() -> None:
    with pytest.raises(ValueError, match="fractions"):
        _budget(alert_thresholds=(1.5,))
    with pytest.raises(ValueError, match="fractions"):
        _budget(alert_thresholds=(0.0,))
    with pytest.raises(ValueError, match="repeat"):
        _budget(alert_thresholds=(0.8, 0.8))


def test_a_recorded_alert_must_name_a_declared_threshold() -> None:
    """Otherwise a stale alerted set silently suppresses a threshold nobody set."""

    with pytest.raises(ValueError, match="undeclared"):
        _budget(alert_thresholds=(0.8,), alerted_thresholds=(0.5,))


def test_utilization_is_zero_rather_than_undefined_against_a_zero_limit() -> None:
    """A zero-limit budget refuses everything; dividing by it would break every list."""

    assert _budget(limit=0.0, spent=0.0).utilization == 0.0
    assert _budget(limit=200.0, spent=50.0).utilization == 0.25


def test_only_newly_reached_thresholds_are_reported_as_crossed() -> None:
    budget = _budget(limit=1000.0, spent=400.0, alert_thresholds=(0.5, 0.8, 0.95))

    assert budget.thresholds_crossed_by(500.0) == (0.5,)
    assert budget.thresholds_crossed_by(960.0) == (0.5, 0.8, 0.95)

    already = budget.model_copy(update={"alerted_thresholds": (0.5,)})
    assert already.thresholds_crossed_by(820.0) == (0.8,), "0.5 has already been announced"


def test_a_zero_limit_budget_announces_nothing() -> None:
    """Every threshold would be trivially crossed, which is noise, not information."""

    assert _budget(limit=0.0, alert_thresholds=(0.5,)).thresholds_crossed_by(1.0) == ()


@pytest.mark.asyncio
async def test_the_status_route_reports_the_period_window() -> None:
    service = await _service()
    start = utcnow() - timedelta(days=3)
    await _put(service, _budget(period="monthly", period_started_at=start, spent=250.0))

    status = await service.budget_status(_admin(), org_id="acme", project_id="research")

    assert status.spent == 250.0
    assert status.period_started_at == start
    assert status.period_ends_at is not None
    assert status.period_ends_at > start
    assert status.utilization == 0.25


@pytest.mark.asyncio
async def test_the_forecast_projects_active_leases_to_the_end_of_the_window() -> None:
    service = await _service()
    now = utcnow()
    # A daily period that started 12 hours ago leaves 12 hours to project over.
    await _put(service, _budget(period="daily", period_started_at=now - timedelta(hours=12), spent=100.0, limit=500.0))
    async with service.repository.transaction() as tx:
        await tx.put_lease(
            LeaseRecord(
                id="lease-1",
                workload_id="wl-1",
                cell_id="cell-a",
                org_id="acme",
                project_id="research",
                state=LeaseState.ACTIVE,
                resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
                generation=1,
                fencing_token="token-1",
                expires_at=now + timedelta(hours=24),
                hourly_cost=10.0,
            )
        )

    status = await service.budget_status(_admin(), org_id="acme", project_id="research", now=now)

    assert status.forecast == pytest.approx(220.0), "100 spent + 10/hour for the remaining 12h"
    assert status.forecast_exceeds_limit is False


@pytest.mark.asyncio
async def test_a_forecast_above_the_limit_is_flagged_before_admission_is_refused() -> None:
    """The whole point: warn while raising the limit is still one of the options."""

    service = await _service()
    now = utcnow()
    await _put(service, _budget(period="daily", period_started_at=now - timedelta(hours=1), spent=100.0, limit=500.0))
    async with service.repository.transaction() as tx:
        await tx.put_lease(
            LeaseRecord(
                id="lease-1",
                workload_id="wl-1",
                cell_id="cell-a",
                org_id="acme",
                project_id="research",
                state=LeaseState.ACTIVE,
                resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1.0)),
                generation=1,
                fencing_token="token-1",
                expires_at=now + timedelta(hours=24),
                hourly_cost=100.0,
            )
        )

    status = await service.budget_status(_admin(), org_id="acme", project_id="research", now=now)

    assert status.forecast == pytest.approx(2400.0), "100 + 100/hour for the remaining 23h"
    assert status.forecast_exceeds_limit is True


@pytest.mark.asyncio
async def test_a_never_rolling_period_has_no_window_and_therefore_no_forecast() -> None:
    """None is the honest answer; zero would read as "projected to spend nothing"."""

    service = await _service()
    await _put(service, _budget(period="lifetime", period_started_at=utcnow(), spent=50.0))

    status = await service.budget_status(_admin(), org_id="acme", project_id="research")

    assert status.period_ends_at is None
    assert status.forecast is None
    assert status.forecast_exceeds_limit is False


@pytest.mark.asyncio
async def test_rolling_an_elapsed_period_resets_spend_and_clears_fired_alerts() -> None:
    service = await _service()
    start = utcnow() - timedelta(days=45)
    await _put(
        service,
        _budget(
            period="monthly",
            period_started_at=start,
            spent=900.0,
            alert_thresholds=(0.8,),
            alerted_thresholds=(0.8,),
        ),
    )

    rolled = await service.roll_budget_period(_admin(), org_id="acme", project_id="research")

    assert rolled.spent == 0.0
    assert rolled.alerted_thresholds == (), "a new period must be able to warn again"
    assert rolled.alert_thresholds == (0.8,), "the declaration itself survives"
    assert rolled.period_started_at is not None
    assert rolled.period_started_at > start


@pytest.mark.asyncio
async def test_rolling_a_live_period_is_a_no_op_rather_than_a_reset() -> None:
    """A force-reset would be a way to silently disable a budget, not administer one.

    Asserted on the shared roll itself, not only through ``roll_budget_period``.
    That method has a second guard -- it returns the stored record when the period
    start did not move -- so it would keep spend intact even if the roll wrongly
    zeroed it. Settlement calls the roll *directly* and uses what it returns, so
    the elapsed check has to hold on its own.
    """

    service = await _service()
    start = utcnow() - timedelta(days=2)
    live = _budget(period="monthly", period_started_at=start, spent=900.0)
    await _put(service, live)

    assert service._rolled_budget(live, utcnow()).spent == 900.0

    rolled = await service.roll_budget_period(_admin(), org_id="acme", project_id="research")

    assert rolled.spent == 900.0
    assert rolled.period_started_at == start


@pytest.mark.asyncio
async def test_rolling_twice_in_one_period_returns_the_same_record() -> None:
    service = await _service()
    await _put(service, _budget(period="monthly", period_started_at=utcnow() - timedelta(days=45), spent=900.0))

    first = await service.roll_budget_period(_admin(), org_id="acme", project_id="research")
    second = await service.roll_budget_period(_admin(), org_id="acme", project_id="research")

    assert second.period_started_at == first.period_started_at
    assert second.revision == first.revision, "an idempotent roll does not churn the revision"


@pytest.mark.asyncio
async def test_rolling_a_missing_budget_is_a_not_found() -> None:
    service = await _service()

    with pytest.raises(NotFoundError):
        await service.roll_budget_period(_admin(), org_id="acme", project_id="research")
