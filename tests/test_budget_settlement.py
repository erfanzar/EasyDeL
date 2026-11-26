# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from eyvan.domain.models import InfeasibilityCode
from eyvan.records import (
    BudgetRecord,
    CapacityClass,
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    Principal,
    QuotaRecord,
    ResourceGang,
    ResourceHours,
    ResourceLimits,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import ConflictError
from eyvan.server.service.scheduling import CoreSchedulingService, rolled_budget, to_core_budget


def _principal(role: Role) -> Principal:
    if role is Role.FLEET_ADMIN:
        return Principal(subject="controller", grants=(RoleGrant(role=role),))
    return Principal(
        subject="operator",
        grants=(RoleGrant(role=role, org_id="acme", project_id="research"),),
    )


async def _running_workload(
    repository: InMemoryRepository,
    service: ControlPlaneService,
    *,
    key: str,
    reserved_cost: float = 5,
    resource_hours: ResourceHours | None = None,
    held_hours: float = 1.0,
) -> tuple[Principal, str, str]:
    """Create a RUNNING workload whose lease has been held for ``held_hours``.

    Settlement charges elapsed time, so a lease created and released inside one test
    would cost microseconds. Backdating the clock by a known amount is what lets these
    tests assert an exact figure instead of echoing the submitted estimate.
    """
    fleet_admin = _principal(Role.FLEET_ADMIN)
    operator = _principal(Role.PROJECT_OPERATOR)
    gang = ResourceGang(
        replicas=1,
        per_replica=ResourceVector(cpu=0, memory_gib=0, tpu=8),
    )
    workload, _ = await service.submit_workload(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name=f"training-{key}",
            execution=CommandExecution(command="python train.py"),
            gang=gang,
        ),
        idempotency_key=f"submit-{key}",
    )
    async with repository.transaction() as tx:
        await tx.put_cell(
            CellRecord(
                id="cell-a",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=gang.per_replica,
                allocatable=gang.per_replica,
            )
        )
    lease = LeaseRecord(
        id=f"lease-{key}",
        workload_id=workload.id,
        cell_id="cell-a",
        org_id=workload.org_id,
        project_id=workload.project_id,
        resources=gang,
        hourly_cost=reserved_cost,
        reserved_cost=reserved_cost,
        resource_hours_reserved=resource_hours or ResourceHours(tpu=3),
        fencing_token=f"fence-{key}",
        expires_at=utcnow() + timedelta(hours=1),
    )
    await service.create_lease(fleet_admin, lease, idempotency_key=f"lease-{key}")
    await service.transition_lease(fleet_admin, lease.id, LeaseState.ACTIVE)
    started = utcnow() - timedelta(hours=held_hours)
    async with repository.transaction() as tx:
        active = await tx.get_lease(lease.id)
        assert active is not None and active.run_started_at is not None
        await tx.put_lease(active.model_copy(update={"created_at": started, "run_started_at": started}))
    await service.transition_workload(fleet_admin, workload.id, WorkloadState.DISPATCHING)
    await service.transition_workload(fleet_admin, workload.id, WorkloadState.RUNNING)
    return fleet_admin, workload.id, lease.id


@pytest.mark.asyncio
async def test_terminal_settlement_is_atomic_idempotent_and_records_hard_budget_overrun() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    await service.put_budget(
        _principal(Role.PROJECT_ADMIN),
        BudgetRecord(
            org_id="acme",
            project_id="research",
            limit=10,
            spent=9,
            accelerator_hour_limits={"tpu": 4},
            accelerator_hours_spent={"tpu": 2},
            hard_limit=True,
            updated_by="operator",
        ),
        idempotency_key="budget",
    )
    fleet_admin, workload_id, lease_id = await _running_workload(repository, service, key="terminal")

    first, second = await asyncio.gather(
        service.settle_terminal_workload(
            fleet_admin,
            workload_id,
            WorkloadState.SUCCEEDED,
            action="workload.succeeded",
        ),
        service.settle_terminal_workload(
            fleet_admin,
            workload_id,
            WorkloadState.SUCCEEDED,
            action="workload.succeeded",
        ),
    )

    assert first[0].state is second[0].state is WorkloadState.SUCCEEDED
    async with repository.transaction() as tx:
        lease = await tx.get_lease(lease_id)
        budget = await tx.get_budget("acme", "research")
        events = await tx.list_events(org_id="acme", project_id="research", limit=100)
    assert lease is not None
    assert lease.state is LeaseState.RELEASED
    assert lease.settled_at is not None
    # Elapsed-cost settlement: one held hour at 5/hour. Approximate because the
    # clock advances between backdating the lease and releasing it.
    assert lease.settled_cost == pytest.approx(5, abs=0.01)
    # The reserved shape (tpu=3 over the estimated hour) re-scaled by hours held.
    assert lease.accelerator_hours_settled["tpu"] == pytest.approx(3, abs=0.01)
    assert lease.settlement_reason == "succeeded"
    assert LeaseRecord.model_validate(lease.model_dump(mode="json")) == lease
    assert budget is not None
    assert budget.spent == pytest.approx(14, abs=0.01)
    assert budget.accelerator_hours_spent["tpu"] == pytest.approx(5, abs=0.01)
    assert budget.revision == 2
    assert [event.action for event in events].count("workload.succeeded") == 1
    assert [event.action for event in events].count("lease.settled") == 1
    assert [event.action for event in events].count("budget.lease_settled") == 1


@pytest.mark.asyncio
async def test_preemption_settlement_requeues_once_and_charges_once() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    await service.put_budget(
        _principal(Role.PROJECT_ADMIN),
        BudgetRecord(org_id="acme", project_id="research", limit=100, updated_by="operator"),
        idempotency_key="budget",
    )
    fleet_admin, workload_id, lease_id = await _running_workload(repository, service, key="preempt")
    await service.transition_workload(fleet_admin, workload_id, WorkloadState.CHECKPOINTING)

    resumed, settled, budget = await service.settle_preempted_workload(
        fleet_admin,
        workload_id,
        checkpoint_uri="gs://checkpoints/preempted",
    )
    replayed, replayed_lease, replayed_budget = await service.settle_preempted_workload(
        fleet_admin,
        workload_id,
        checkpoint_uri="gs://checkpoints/preempted",
    )

    assert (resumed.state, resumed.attempt, resumed.generation) == (WorkloadState.PENDING, 2, 2)
    assert resumed.lease_id is None
    assert resumed.remote_id is None
    assert resumed.checkpoint_uri == "gs://checkpoints/preempted"
    assert settled is not None and settled.id == lease_id
    assert settled.state is LeaseState.RELEASED
    assert settled.settlement_reason == "preempted"
    assert budget is not None and budget.spent == pytest.approx(5, abs=0.01)
    assert replayed == resumed
    assert replayed_lease is None
    assert replayed_budget is not None and replayed_budget.revision == budget.revision
    async with repository.transaction() as tx:
        events = await tx.list_events(org_id="acme", project_id="research", limit=100)
    assert [event.action for event in events].count("workload.preempted") == 1
    assert [event.action for event in events].count("budget.lease_settled") == 1


@pytest.mark.asyncio
async def test_budget_configuration_update_preserves_server_owned_usage() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    admin = _principal(Role.PROJECT_ADMIN)
    await service.put_budget(
        admin,
        BudgetRecord(
            org_id="acme",
            project_id="research",
            limit=100,
            spent=25,
            accelerator_hours_spent={"tpu": 4},
            updated_by="operator",
        ),
        idempotency_key="budget-v1",
    )

    updated, _ = await service.put_budget(
        admin,
        BudgetRecord(
            org_id="acme",
            project_id="research",
            limit=200,
            spent=0,
            accelerator_hours_spent={},
            updated_by="operator",
        ),
        idempotency_key="budget-v2",
    )

    assert updated.spent == 25
    assert updated.accelerator_hours_spent == {"tpu": 4}
    with pytest.raises(ConflictError, match="currency and period"):
        await service.put_budget(
            admin,
            updated.model_copy(update={"currency": "EUR"}),
            idempotency_key="budget-v3",
        )


@pytest.mark.asyncio
async def test_fractional_settlement_uses_decimal_safe_accumulation() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    await service.put_budget(
        _principal(Role.PROJECT_ADMIN),
        BudgetRecord(org_id="acme", project_id="research", limit=0.3, updated_by="operator"),
        idempotency_key="budget",
    )
    fleet_admin, first_workload, _ = await _running_workload(
        repository,
        service,
        key="fraction-a",
        reserved_cost=0.1,
        resource_hours=ResourceHours(tpu=0.1),
    )
    await service.settle_terminal_workload(
        fleet_admin,
        first_workload,
        WorkloadState.SUCCEEDED,
        action="workload.succeeded",
    )
    fleet_admin, second_workload, _ = await _running_workload(
        repository,
        service,
        key="fraction-b",
        reserved_cost=0.2,
        resource_hours=ResourceHours(tpu=0.2),
    )
    await service.settle_terminal_workload(
        fleet_admin,
        second_workload,
        WorkloadState.SUCCEEDED,
        action="workload.succeeded",
    )

    async with repository.transaction() as transaction:
        budget = await transaction.get_budget("acme", "research")
    assert budget is not None
    assert budget.spent == pytest.approx(0.3, abs=0.01)
    assert budget.accelerator_hours_spent["tpu"] == pytest.approx(0.3, abs=0.01)
    assert to_core_budget(budget).remaining == 0


@pytest.mark.asyncio
async def test_terminal_settlement_rolls_back_state_when_current_lease_is_inconsistent() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    fleet_admin, workload_id, _ = await _running_workload(repository, service, key="rollback")
    async with repository.transaction() as tx:
        workload = await tx.get_workload(workload_id)
        assert workload is not None
        await tx.put_workload(workload.model_copy(update={"lease_id": "missing-lease"}))

    with pytest.raises(ConflictError, match="missing lease"):
        await service.settle_terminal_workload(
            fleet_admin,
            workload_id,
            WorkloadState.SUCCEEDED,
            action="workload.succeeded",
        )

    async with repository.transaction() as tx:
        workload = await tx.get_workload(workload_id)
        events = await tx.list_events(org_id="acme", project_id="research", limit=100)
    assert workload is not None and workload.state is WorkloadState.RUNNING
    assert "workload.succeeded" not in [event.action for event in events]


def test_hard_budget_records_monetary_and_accelerator_hour_overrun() -> None:
    budget = BudgetRecord(
        org_id="acme",
        project_id="research",
        limit=10,
        spent=11,
        accelerator_hour_limits={"tpu": 4},
        accelerator_hours_spent={"tpu": 5},
        hard_limit=True,
        updated_by="controller",
    )

    assert budget.spent == 11
    assert budget.accelerator_hours_spent == {"tpu": 5}


def test_resource_limits_are_zero_default_and_all_models_reject_non_finite_values() -> None:
    workload_resources = ResourceVector()
    quota = QuotaRecord(
        org_id="acme",
        project_id="research",
        limits={"tpu": 8},
        updated_by="operator",
    )
    legacy_quota = QuotaRecord.model_validate(
        {
            "org_id": "acme",
            "project_id": "research",
            "limits": ResourceVector(tpu=4),
            "updated_by": "operator",
        }
    )

    assert (workload_resources.cpu, workload_resources.memory_gib) == (1, 1)
    assert (quota.limits.cpu, quota.limits.memory_gib, quota.limits.tpu) == (0, 0, 8)
    assert (legacy_quota.limits.cpu, legacy_quota.limits.memory_gib, legacy_quota.limits.tpu) == (0, 0, 4)
    with pytest.raises(ValueError, match="finite number"):
        ResourceVector(cpu=float("inf"))


def test_cell_persists_topology_and_per_replica_capacity() -> None:
    cell = CellRecord(
        id="cell-topology",
        provider="local",
        region="local",
        capacity=ResourceVector(cpu=16, memory_gib=64, tpu=8),
        allocatable=ResourceVector(cpu=16, memory_gib=64, tpu=8),
        topology={"slice": "slice-a", "rack": "rack-2"},
        max_replica_resources=ResourceVector(cpu=8, memory_gib=32, tpu=4),
    )

    restored = CellRecord.model_validate(cell.model_dump(mode="json"))
    assert restored.topology == {"slice": "slice-a", "rack": "rack-2"}
    assert restored.max_replica_resources == ResourceLimits(cpu=8, memory_gib=32, tpu=4)


@pytest.mark.asyncio
async def test_a_job_stopped_early_is_charged_for_what_it_held_not_what_it_estimated() -> None:
    """`settled_cost = reserved_cost` charged the submitter's guess.

    A 1-hour-priced lease released after 6 minutes used to cost a full hour, because
    settlement copied the reservation estimate derived from
    ``estimated_duration_seconds``. Over-charging is the benign direction; the same
    bug under-charges a job that outruns its estimate, and that is what silently
    overruns a hard budget.
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    fleet_admin, workload_id, lease_id = await _running_workload(
        repository, service, key="short", reserved_cost=60, held_hours=0.1
    )

    await service.settle_terminal_workload(
        fleet_admin, workload_id, WorkloadState.SUCCEEDED, action="workload.succeeded"
    )

    async with repository.transaction() as tx:
        lease = await tx.get_lease(lease_id)
        budget = await tx.get_budget("acme", "research")

    assert lease is not None
    # 0.1 h at 60/hour == 6, not the 60 the estimate reserved.
    assert lease.settled_cost == pytest.approx(6, abs=0.05)
    assert lease.reserved_cost == 60
    assert lease.running_seconds == pytest.approx(360, abs=5)
    assert budget is None or budget.spent == pytest.approx(6, abs=0.05)


@pytest.mark.asyncio
async def test_a_job_that_outruns_its_estimate_is_charged_for_the_overrun() -> None:
    """The dangerous direction: under-charging lets a hard budget be exceeded."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    fleet_admin, workload_id, lease_id = await _running_workload(
        repository, service, key="long", reserved_cost=10, held_hours=4.0
    )

    await service.settle_terminal_workload(
        fleet_admin, workload_id, WorkloadState.SUCCEEDED, action="workload.succeeded"
    )

    async with repository.transaction() as tx:
        lease = await tx.get_lease(lease_id)

    assert lease is not None
    # 4 h at 10/hour == 40, four times what the one-hour estimate reserved.
    assert lease.settled_cost == pytest.approx(40, abs=0.05)
    assert lease.settled_cost > lease.reserved_cost


@pytest.mark.asyncio
async def test_queued_and_running_time_are_settled_separately() -> None:
    """Queued and running cost must be attributed separately; one number cannot carry both."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    fleet_admin, workload_id, lease_id = await _running_workload(
        repository, service, key="split", reserved_cost=10, held_hours=1.0
    )
    # Widen the held window without moving the run start: two hours queued.
    async with repository.transaction() as tx:
        lease = await tx.get_lease(lease_id)
        assert lease is not None
        await tx.put_lease(lease.model_copy(update={"created_at": lease.created_at - timedelta(hours=2)}))

    await service.settle_terminal_workload(
        fleet_admin, workload_id, WorkloadState.SUCCEEDED, action="workload.succeeded"
    )

    async with repository.transaction() as tx:
        settled = await tx.get_lease(lease_id)

    assert settled is not None
    assert settled.queued_seconds == pytest.approx(7200, abs=10)
    assert settled.running_seconds == pytest.approx(3600, abs=10)
    # Conservation: the two halves account for the whole held window, and cost
    # follows the total rather than either half.
    held_hours = (settled.queued_seconds + settled.running_seconds) / 3600
    assert settled.settled_cost == pytest.approx(10 * held_hours, abs=0.05)


def test_a_budget_period_rolls_instead_of_accumulating_forever() -> None:
    """`period` was a string nothing acted on.

    Survivable while ``spent`` was an estimate; once it became a true monotone total,
    a "monthly" budget was a lifetime budget and every alert threshold would fire
    exactly once in the fleet's lifetime.
    """

    now = utcnow()
    budget = BudgetRecord(
        org_id="acme",
        project_id="research",
        limit=100,
        spent=90,
        accelerator_hours_spent={"tpu": 12},
        period="monthly",
        period_started_at=now - timedelta(days=31),
        updated_by="operator",
    )

    rolled = ControlPlaneService._rolled_budget(budget, now)

    assert rolled.spent == 0
    assert rolled.accelerator_hours_spent == {}
    assert rolled.period_started_at is not None and rolled.period_started_at > budget.period_started_at


def test_a_budget_period_that_has_not_elapsed_keeps_accumulating() -> None:
    now = utcnow()
    budget = BudgetRecord(
        org_id="acme",
        project_id="research",
        limit=100,
        spent=90,
        period="monthly",
        period_started_at=now - timedelta(days=3),
        updated_by="operator",
    )

    assert ControlPlaneService._rolled_budget(budget, now) == budget


def test_an_unrecognized_budget_period_never_silently_resets_spend() -> None:
    """Refusing to roll is the safe direction: it over-counts rather than forgiving spend."""

    now = utcnow()
    budget = BudgetRecord(
        org_id="acme",
        project_id="research",
        limit=100,
        spent=90,
        period="fortnightly",
        period_started_at=now - timedelta(days=400),
        updated_by="operator",
    )

    assert ControlPlaneService._rolled_budget(budget, now) == budget


@pytest.mark.asyncio
async def test_settlement_rolls_an_elapsed_budget_period_before_charging() -> None:
    """The roll must happen on the settlement path, not merely be available.

    Testing ``_rolled_budget`` alone proves the arithmetic and nothing about whether
    anything calls it: a settlement landing after the window closed has to open the
    new one rather than be absorbed by the old.
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    fleet_admin, workload_id, _lease_id = await _running_workload(
        repository, service, key="rolled", reserved_cost=10, held_hours=1.0
    )
    async with repository.transaction() as tx:
        await tx.put_budget(
            BudgetRecord(
                org_id="acme",
                project_id="research",
                limit=1_000,
                spent=900,
                accelerator_hours_spent={"tpu": 400},
                period="monthly",
                period_started_at=utcnow() - timedelta(days=31),
                updated_by="operator",
            )
        )

    await service.settle_terminal_workload(
        fleet_admin, workload_id, WorkloadState.SUCCEEDED, action="workload.succeeded"
    )

    async with repository.transaction() as tx:
        budget = await tx.get_budget("acme", "research")

    assert budget is not None
    # The stale 900 is discarded and only this settlement's ~10 remains.
    assert budget.spent == pytest.approx(10, abs=0.05)
    assert budget.accelerator_hours_spent["tpu"] == pytest.approx(3, abs=0.05)


# --- admission-time period rolls ---------------------------------------------
#
# The roll used to run only where money moves (settlement), where an operator
# acts (the roll route) or where a status is read. Admission read the raw
# counter, so a hard budget whose period crossed with no in-flight lease to
# settle refused every placement forever: BUDGET_EXCEEDED against a fresh
# limit, priced with spend from a window that had already closed.


def _budget_record(**overrides: object) -> BudgetRecord:
    """An exhausted monthly budget whose window started 40 days ago."""

    defaults: dict[str, object] = {
        "org_id": "acme",
        "project_id": "research",
        "limit": 10.0,
        "spent": 10.0,
        "accelerator_hours_spent": {"tpu": 5.0},
        "period": "monthly",
        "period_started_at": utcnow() - timedelta(days=40),
        "updated_by": "operator",
    }
    return BudgetRecord(**{**defaults, **overrides})


def test_a_crossed_period_resets_spend_at_admission_time() -> None:
    record = _budget_record()

    core = to_core_budget(record, now=record.period_started_at + timedelta(days=40))

    assert core.spent == 0
    assert core.accelerator_hours_spent.as_dict() == {}
    assert core.remaining == core.limit


def test_an_open_period_keeps_its_spend() -> None:
    record = _budget_record(period_started_at=utcnow() - timedelta(days=1))

    core = to_core_budget(record, now=utcnow())

    assert core.spent == 10.0


def test_a_never_rolled_period_is_stamped_without_a_reset() -> None:
    """Control-plane parity: the first stamp begins the window, it does not forgive it."""

    record = _budget_record(period_started_at=None)

    core = to_core_budget(record, now=utcnow())

    assert core.spent == 10.0


def test_an_unknown_period_never_rolls() -> None:
    record = _budget_record(period="lifetime")

    core = to_core_budget(record, now=utcnow())

    assert core.spent == 10.0


def test_rolls_advance_by_whole_periods_aligned_to_the_window_origin() -> None:
    started = utcnow() - timedelta(days=70)

    rolled = rolled_budget(_budget_record(period_started_at=started), now=utcnow())

    assert rolled.spent == 0.0
    assert rolled.period_started_at == started + timedelta(seconds=2_592_000 * 2)


def test_a_crossed_hard_budget_stops_refusing_admission() -> None:
    """The failure path: every placement failed BUDGET_EXCEEDED against a closed window."""

    now = utcnow()
    workload = WorkloadRecord(
        id="wl-1",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=8.0)),
        ),
        state=WorkloadState.PENDING,
        created_by="alice",
    )
    cell = CellRecord(
        id="cell-a",
        provider="manual",
        region="local",
        state=CellState.READY,
        capacity=ResourceLimits(cpu=8.0, memory_gib=64.0),
        allocatable=ResourceLimits(cpu=8.0, memory_gib=64.0),
        unit_prices={"cpu": 1.0},
        capacity_class=CapacityClass.ON_DEMAND,
    )

    crossed = _budget_record(period_started_at=now - timedelta(days=40))
    admitted = CoreSchedulingService().schedule([workload], [cell], budgets=[crossed], now=now)
    assert [decision.workload_id for decision in admitted.placements] == ["wl-1"]
    assert not admitted.infeasibilities

    open_window = _budget_record(period_started_at=now - timedelta(days=1))
    blocked = CoreSchedulingService().schedule([workload], [cell], budgets=[open_window], now=now)
    assert not blocked.placements
    assert any(item.code is InfeasibilityCode.BUDGET_EXCEEDED for item in blocked.infeasibilities)
