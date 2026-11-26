# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Regression contracts for period-aware fleet budget reconciliation."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from eyvan.records import (
    BudgetRecord,
    LeaseRecord,
    LeaseState,
    OutboxRecord,
    OutboxState,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
)
from eyvan.server.invariants import Invariant, check_fleet
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service import ControlPlaneService

_START = datetime(2026, 9, 28, tzinfo=UTC)
_ADMIN = Principal(subject="controller", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


def _lease(lease_id: str, *, created_at: datetime, hourly_cost: float) -> LeaseRecord:
    return LeaseRecord(
        id=lease_id,
        workload_id=f"missing-{lease_id}",
        cell_id="missing-cell",
        org_id="acme",
        project_id="research",
        state=LeaseState.ACTIVE,
        resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1)),
        hourly_cost=hourly_cost,
        fencing_token=f"fence-{lease_id}",
        created_at=created_at,
        run_started_at=created_at,
        expires_at=created_at + timedelta(days=3),
    )


async def _budget_repository(*, period: str = "daily", started_at: datetime | None = _START):
    repository = InMemoryRepository()
    async with repository.transaction() as tx:
        await tx.put_budget(
            BudgetRecord(
                org_id="acme",
                project_id="research",
                limit=100,
                period=period,
                period_started_at=started_at,
                updated_by="controller",
            )
        )
    return repository, ControlPlaneService(repository)


async def _settle(repository, service, monkeypatch, lease, *, at: datetime) -> None:
    # Patch only the charging clock: the public orphan settlement path still
    # performs authorization, generation checks, charging and durable writes.
    monkeypatch.setattr("eyvan.server.service._control_plane.lease_settlement.utcnow", lambda: at)
    async with repository.transaction() as tx:
        await tx.put_lease(lease)
    assert await service.settle_orphaned_lease(_ADMIN, lease.id, reason="orphaned", expected_generation=lease.generation)
    async with repository.transaction() as tx:
        stored = await tx.get_lease(lease.id)
    assert stored is not None
    assert stored.settled_at == at
    assert stored.state is LeaseState.RELEASED
    assert LeaseRecord.model_validate(stored.model_dump(mode="json")) == stored


async def test_i5_public_roll_retains_history_without_false_corruption(monkeypatch) -> None:
    repository, service = await _budget_repository()
    old_at = _START + timedelta(hours=1)
    await _settle(repository, service, monkeypatch, _lease("old", created_at=_START, hourly_cost=10), at=old_at)
    assert await check_fleet(repository, now=old_at) == ()

    boundary = _START + timedelta(days=1)
    rolled = await service.roll_budget_period(_ADMIN, org_id="acme", project_id="research", now=boundary)
    assert rolled.period_started_at == boundary
    assert rolled.spent == 0
    async with repository.transaction() as tx:
        assert len(await tx.list_leases()) == 1
    assert await check_fleet(repository, now=boundary) == ()


@pytest.mark.parametrize("spend_error", [0, -1, 1])
async def test_i5_settlement_roll_charges_spanning_lease_to_new_period(monkeypatch, spend_error) -> None:
    repository, service = await _budget_repository()
    await _settle(
        repository,
        service,
        monkeypatch,
        _lease("old", created_at=_START, hourly_cost=10),
        at=_START + timedelta(hours=1),
    )
    boundary = _START + timedelta(days=1)
    # The new lease started in the old window but settles exactly on the new
    # boundary. Charging is by settlement time, not by creation/run time.
    await _settle(
        repository,
        service,
        monkeypatch,
        _lease("spanning", created_at=boundary - timedelta(hours=1), hourly_cost=5),
        at=boundary,
    )
    async with repository.transaction() as tx:
        budget = await tx.get_budget("acme", "research")
        assert budget is not None and budget.spent == 5
        assert budget.period_started_at == boundary
        leases = await tx.list_leases()
        assert {lease.id: lease.settled_cost for lease in leases} == {"old": 10, "spanning": 5}
        if spend_error:
            await tx.put_budget(budget.model_copy(update={"spent": budget.spent + spend_error}))
    violations = await check_fleet(repository, now=boundary)
    if spend_error:
        assert len(violations) == 1
        assert violations[0].invariant is Invariant.BUDGET_MATCHES_SETTLED_COST
        assert "settled leases total 5.0" in violations[0].detail
    else:
        assert violations == ()


@pytest.mark.parametrize("period", ["daily", "lifetime", "unknown"])
async def test_i5_same_period_and_lifetime_keep_full_charge_accounting(monkeypatch, period) -> None:
    repository, service = await _budget_repository(period=period)
    for index, cost in enumerate((10, 5)):
        at = _START + timedelta(hours=index + 1)
        await _settle(
            repository,
            service,
            monkeypatch,
            _lease(f"same-{index}", created_at=at - timedelta(hours=1), hourly_cost=cost),
            at=at,
        )
    async with repository.transaction() as tx:
        budget = await tx.get_budget("acme", "research")
    assert budget is not None and budget.spent == 15
    assert await check_fleet(repository, now=at) == ()


@pytest.mark.parametrize("period", ["lifetime", "unknown"])
async def test_i5_nonrolling_budget_does_not_exclude_history(monkeypatch, period) -> None:
    repository, service = await _budget_repository(period=period, started_at=_START + timedelta(days=1))
    await _settle(
        repository,
        service,
        monkeypatch,
        _lease("lifetime-old", created_at=_START, hourly_cost=10),
        at=_START + timedelta(hours=1),
    )
    assert await check_fleet(repository, now=_START + timedelta(days=1)) == ()


async def test_i5_initialization_preserves_preexisting_spend_and_history(monkeypatch) -> None:
    """A start timestamp alone cannot distinguish initialization from reset."""

    repository, service = await _budget_repository(started_at=None)
    legacy = _lease("baseline", created_at=_START, hourly_cost=10).model_copy(
        update={
            "state": LeaseState.RELEASED,
            "settled_at": _START + timedelta(hours=1),
            "settlement_reason": "completed",
            "settled_cost": 10,
        }
    )
    legacy = LeaseRecord.model_validate(legacy.model_dump(mode="json"))
    async with repository.transaction() as tx:
        await tx.put_lease(legacy)
        budget = await tx.get_budget("acme", "research")
        assert budget is not None
        await tx.put_budget(budget.model_copy(update={"spent": 10}))
    boundary = _START + timedelta(days=1)
    initialized = await service.roll_budget_period(_ADMIN, org_id="acme", project_id="research", now=boundary)
    assert initialized.period_started_at == boundary
    assert initialized.spent == 10  # Initialization is not an elapsed-period reset.
    await _settle(
        repository,
        service,
        monkeypatch,
        _lease("after-initialization", created_at=boundary, hourly_cost=5),
        at=boundary + timedelta(hours=1),
    )
    async with repository.transaction() as tx:
        budget = await tx.get_budget("acme", "research")
        leases = await tx.list_leases()
    assert budget is not None and budget.spent == 15
    assert sum(lease.settled_cost for lease in leases if lease.settled_at >= boundary) == 5
    assert await check_fleet(repository, now=boundary + timedelta(hours=1)) == ()


async def test_i5_uninitialized_budget_keeps_historical_settlements() -> None:
    repository, _service = await _budget_repository(started_at=None)
    settled = _lease("legacy", created_at=_START, hourly_cost=10).model_copy(
        update={
            "state": LeaseState.RELEASED,
            "settled_at": _START + timedelta(hours=1),
            "settlement_reason": "completed",
            "settled_cost": 10,
        }
    )
    settled = LeaseRecord.model_validate(settled.model_dump(mode="json"))
    async with repository.transaction() as tx:
        await tx.put_lease(settled)
        budget = await tx.get_budget("acme", "research")
        assert budget is not None
        await tx.put_budget(budget.model_copy(update={"spent": 10}))
    assert await check_fleet(repository, now=_START + timedelta(days=1)) == ()


async def test_i3_old_retry_is_detected_independently_of_fresh_backlog() -> None:
    repository = InMemoryRepository()
    now = _START + timedelta(days=3)

    def command(command_id: str, *, created_at: datetime, available_at: datetime) -> OutboxRecord:
        return OutboxRecord(
            id=command_id,
            topic="capacity.apply",
            aggregate_type="cell",
            aggregate_id=f"cell-{command_id}",
            generation=1,
            idempotency_key=f"apply-{command_id}",
            state=OutboxState.FAILED,
            attempts=1,
            created_at=created_at,
            available_at=available_at,
            updated_at=now,
        )

    async with repository.transaction() as tx:
        for index in range(1000):
            await tx.put_outbox(command(f"fresh-{index:04d}", created_at=now, available_at=now))
        await tx.put_outbox(
            command("stuck", created_at=now - timedelta(hours=2), available_at=now + timedelta(seconds=60))
        )
    with_full_backlog = await check_fleet(repository, now=now)
    async with repository.transaction() as tx:
        completed = await tx.get_outbox("fresh-0000")
        assert completed is not None
        await tx.put_outbox(completed.model_copy(update={"state": OutboxState.SUCCEEDED}))
    with_reduced_backlog = await check_fleet(repository, now=now)
    assert [item.subject for item in with_reduced_backlog] == ["stuck"]
    assert with_reduced_backlog[0].invariant is Invariant.NO_STUCK_OUTBOX
    assert with_full_backlog == with_reduced_backlog
