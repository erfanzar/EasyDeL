# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generated service-boundary running retry histories, not dispatch simulations."""

from datetime import timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from test_group_lease_lifecycle import repository as _repository_fixture

from eyvan.records import (
    BudgetRecord,
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    Principal,
    ResourceGang,
    ResourceHours,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadState,
    utcnow,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import ConflictError

repository = _repository_fixture
ADMIN = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
OPERATOR = Principal(
    subject="operator", grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),)
)


async def _snapshot(repository, workload_id: str):
    async with repository.transaction() as tx:
        return {
            "workload": await tx.get_workload(workload_id),
            "leases": {lease.id: lease for lease in await tx.list_leases()},
            "budget": await tx.get_budget("acme", "research"),
            "events": await tx.list_events(org_id="acme", project_id="research", limit=500),
            "cells": {cell.id: cell for cell in await tx.list_cells()},
            "outbox": {command.id: command for command in await tx.list_outbox()},
        }


async def _seed(repository, key: str, max_failure_retries: int) -> tuple[str, CellRecord]:
    spec = WorkloadDefinition(
        name=f"running-retry-{key}",
        execution=CommandExecution(command="true"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=0, memory_gib=0, tpu=8)),
        accelerator_type="tpu-v5p",
        estimated_duration_seconds=3_600,
        max_failure_retries=max_failure_retries,
    )
    workload, _ = await ControlPlaneService(repository).submit_workload(
        OPERATOR, org_id="acme", project_id="research", spec=spec, idempotency_key=f"submit-{key}"
    )
    cell = CellRecord(
        id=f"cell-{key}",
        provider="local",
        region="local",
        state=CellState.READY,
        capacity=spec.gang.per_replica,
        allocatable=spec.gang.per_replica,
    )
    async with repository.transaction() as tx:
        await tx.put_cell(cell)
        await tx.put_budget(
            BudgetRecord(
                org_id="acme",
                project_id="research",
                limit=10_000,
                spent=17,
                revision=7,
                accelerator_hours_spent={"tpu": 11},
                updated_by="root",
            )
        )
    return workload.id, cell


async def _run(repository, workload_id: str, cell: CellRecord, key: str, rate: int, held_hours: int) -> LeaseRecord:
    async with repository.transaction() as tx:
        workload = await tx.get_workload(workload_id)
    assert workload is not None
    lease = LeaseRecord(
        id=f"lease-{key}-{workload.generation}",
        workload_id=workload.id,
        cell_id=cell.id,
        org_id=workload.org_id,
        project_id=workload.project_id,
        resources=workload.spec.gang,
        generation=workload.generation,
        cell_generation=cell.generation,
        fencing_token=f"fence-{key}-{workload.generation}",
        hourly_cost=rate,
        reserved_cost=rate,
        resource_hours_reserved=ResourceHours(tpu=2),
        expires_at=utcnow() + timedelta(hours=1),
    )
    admitted, duplicate = await ControlPlaneService(repository).create_lease(
        ADMIN, lease, idempotency_key=f"admit-{lease.id}"
    )
    assert admitted == lease and not duplicate
    await ControlPlaneService(repository).transition_lease(ADMIN, lease.id, LeaseState.ACTIVE)
    started = utcnow() - timedelta(hours=held_hours)
    async with repository.transaction() as tx:
        active = await tx.get_lease(lease.id)
        assert active is not None and active.run_started_at is not None
        # The service clock is deliberately backdated; no provider or dispatch is involved.
        await tx.put_lease(active.model_copy(update={"created_at": started, "run_started_at": started}))
    await ControlPlaneService(repository).transition_workload(ADMIN, workload.id, WorkloadState.DISPATCHING)
    await ControlPlaneService(repository).transition_workload(ADMIN, workload.id, WorkloadState.RUNNING)
    return lease


async def _retry_history(repository, failures: int, rate: int, held_hours: int) -> None:
    key = f"generated-{failures}-{rate}-{held_hours}"
    workload_id, cell = await _seed(repository, key, failures)
    initial = await _snapshot(repository, workload_id)
    initial_budget = initial["budget"]
    assert initial_budget is not None and initial_budget.spent == 17 and initial_budget.revision == 7
    released: dict[str, LeaseRecord] = {}

    for failure_index in range(failures):
        lease = await _run(repository, workload_id, cell, key, rate, held_hours)
        before = await _snapshot(repository, workload_id)
        workload = before["workload"]
        assert workload is not None and workload.state is WorkloadState.RUNNING
        service = ControlPlaneService(repository)
        resumed, settled, budget = await service.settle_failed_workload(
            ADMIN,
            workload_id,
            scheduling_reason=f"generated failure {failure_index}",
            expected_generation=workload.generation,
            expected_lease_id=lease.id,
            expected_state=WorkloadState.RUNNING,
        )
        assert settled is not None and budget is not None
        assert resumed.state is WorkloadState.PENDING
        assert resumed.generation == workload.generation + 1
        assert resumed.failure_count == failure_index + 1
        assert resumed.attempt == workload.attempt + 1
        assert resumed.lease_id is None and resumed.lease_ids == ()
        assert settled.settled_at is not None
        assert settled.id == lease.id and settled.state is LeaseState.RELEASED
        assert settled.settlement_reason == "failed"
        assert settled.settled_cost >= rate * held_hours
        assert settled.settled_cost <= rate * (held_hours + 0.01)
        assert settled.accelerator_hours_settled["tpu"] >= 2 * held_hours
        assert settled.accelerator_hours_settled["tpu"] <= 2 * (held_hours + 0.01)
        released[settled.id] = settled
        after = await _snapshot(repository, workload_id)
        assert after["leases"] == released  # Earlier releases are durable and unchanged.
        assert after["budget"].revision == initial_budget.revision + failure_index + 1
        assert after["budget"].spent == pytest.approx(
            initial_budget.spent + sum(item.settled_cost for item in released.values()), abs=0.01
        )
        assert after["budget"].accelerator_hours_spent["tpu"] == pytest.approx(
            initial_budget.accelerator_hours_spent["tpu"]
            + sum(item.accelerator_hours_settled["tpu"] for item in released.values()),
            abs=0.01,
        )
        stale = await _snapshot(repository, workload_id)
        with pytest.raises(ConflictError):
            await ControlPlaneService(repository).settle_failed_workload(
                ADMIN,
                workload_id,
                expected_generation=workload.generation,
                expected_lease_id=lease.id,
                expected_state=WorkloadState.RUNNING,
            )
        assert await _snapshot(repository, workload_id) == stale

    final_lease = await _run(repository, workload_id, cell, key, rate, held_hours)
    before = await _snapshot(repository, workload_id)
    workload = before["workload"]
    assert workload is not None and workload.state is WorkloadState.RUNNING
    service = ControlPlaneService(repository)
    completed, settled, budget = await service.settle_terminal_workload(
        ADMIN,
        workload_id,
        WorkloadState.SUCCEEDED,
        action="workload.succeeded",
        expected_generation=workload.generation,
        expected_lease_id=final_lease.id,
        expected_state=WorkloadState.RUNNING,
    )
    assert completed.state is WorkloadState.SUCCEEDED
    assert completed.failure_count == failures
    assert completed.generation == failures + 1
    # Terminal settlement keeps the historical current-lease pointer; retry
    # settlement above is the boundary that clears ownership before re-admission.
    assert settled is not None and settled.id == final_lease.id
    assert settled.settlement_reason == "succeeded"
    assert settled.settled_cost >= rate * held_hours
    assert settled.settled_cost <= rate * (held_hours + 0.01)
    assert settled.state is LeaseState.RELEASED and settled.settled_at is not None
    assert 2 * held_hours <= settled.accelerator_hours_settled["tpu"] <= 2 * (held_hours + 0.01)
    released[settled.id] = settled
    assert budget is not None and budget.revision == initial_budget.revision + failures + 1
    assert budget.spent == pytest.approx(initial_budget.spent + sum(item.settled_cost for item in released.values()))
    assert budget.accelerator_hours_spent["tpu"] == pytest.approx(
        initial_budget.accelerator_hours_spent["tpu"]
        + sum(item.accelerator_hours_settled["tpu"] for item in released.values())
    )
    settled_snapshot = await _snapshot(repository, workload_id)
    assert settled_snapshot["budget"] == budget and settled_snapshot["workload"] == completed
    replayed, replayed_lease, replayed_budget = await ControlPlaneService(repository).settle_terminal_workload(
        ADMIN,
        workload_id,
        WorkloadState.SUCCEEDED,
        action="workload.succeeded",
    )
    assert replayed == completed and replayed_lease == settled and replayed_budget == budget
    final = await _snapshot(repository, workload_id)
    assert final == settled_snapshot
    assert final["leases"] == released
    with pytest.raises(ConflictError):
        await ControlPlaneService(repository).settle_terminal_workload(
            ADMIN,
            workload_id,
            WorkloadState.SUCCEEDED,
            expected_generation=completed.generation - 1,
            expected_lease_id=final_lease.id,
            expected_state=WorkloadState.RUNNING,
        )
    assert await _snapshot(repository, workload_id) == final
    assert [event.action for event in final["events"]].count("budget.lease_settled") == failures + 1
    assert [event.action for event in final["events"]].count("lease.settled") == failures + 1


@settings(max_examples=20, deadline=None)
@given(
    failures=st.integers(min_value=1, max_value=3),
    rate=st.integers(min_value=0, max_value=30),
    held_hours=st.integers(min_value=1, max_value=3),
)
async def test_generated_running_retries_conserve_budget(failures: int, rate: int, held_hours: int) -> None:
    await _retry_history(InMemoryRepository(), failures, rate, held_hours)


@pytest.mark.parametrize("failures,rate,held_hours", [(1, 0, 1), (1, 7, 1), (2, 13, 2), (3, 17, 3)])
async def test_running_retry_conservation_across_repositories(
    repository, failures: int, rate: int, held_hours: int
) -> None:
    await _retry_history(repository, failures, rate, held_hours)
