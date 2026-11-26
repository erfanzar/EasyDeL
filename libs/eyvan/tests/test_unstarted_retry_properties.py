# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generated service-boundary histories, not physical dispatch/failover simulations."""

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
    OutboxRecord,
    OutboxState,
    Principal,
    ResourceGang,
    ResourceHours,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadDispatchReservation,
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
            "events": await tx.list_events(org_id="acme", project_id="research", limit=200),
            "cells": {cell.id: cell for cell in await tx.list_cells()},
            "outbox": {command.id: command for command in await tx.list_outbox()},
        }


async def _seed(repository, key: str) -> tuple[str, CellRecord]:
    service = ControlPlaneService(repository)
    spec = WorkloadDefinition(
        name=f"retry-{key}",
        execution=CommandExecution(command="true"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1)),
    )
    workload, _ = await service.submit_workload(
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
                limit=1000,
                spent=17,
                revision=7,
                accelerator_hours_spent={"tpu-v5p": 11},
                updated_by="root",
            )
        )
    return workload.id, cell


async def _admit(repository, workload_id: str, cell: CellRecord, key: str, rate: int = 13, hours: int = 2):
    service = ControlPlaneService(repository)
    async with repository.transaction() as tx:
        workload = await tx.get_workload(workload_id)
    assert workload is not None
    lease = LeaseRecord(
        id=f"lease-{key}-{workload.generation}",
        workload_id=workload.id,
        cell_id=cell.id,
        org_id=workload.org_id,
        project_id=workload.project_id,
        state=LeaseState.PENDING,
        resources=workload.spec.gang,
        generation=workload.generation,
        cell_generation=cell.generation,
        fencing_token=f"fence-{key}-{workload.generation}",
        hourly_cost=rate,
        reserved_cost=rate * hours,
        resource_hours_reserved=ResourceHours(cpu=hours),
        expires_at=utcnow() + timedelta(hours=hours),
    )
    admitted, duplicate = await service.create_lease(ADMIN, lease, idempotency_key=f"admit-{lease.id}")
    assert admitted == lease and not duplicate
    return lease


async def _retry_history(repository, repeats: int, rate: int = 13, hours: int = 2) -> None:
    key = f"generated-{repeats}-{rate}-{hours}"
    workload_id, cell = await _seed(repository, key)
    initial_budget = (await _snapshot(repository, workload_id))["budget"]
    released_history = {}
    for generation_index in range(repeats):
        lease = await _admit(repository, workload_id, cell, key, rate, hours)
        assert lease.generation == generation_index + 1
        before = await _snapshot(repository, workload_id)
        workload = before["workload"]
        assert workload is not None and before["budget"] == initial_budget
        service = ControlPlaneService(repository)  # Never rely on a facade's process-local cache.
        resumed, released = await service.requeue_unstarted_workload(
            ADMIN,
            workload_id,
            scheduling_reason=f"cell unavailable {generation_index}",
            expected_generation=workload.generation,
            expected_lease_id=lease.id,
            expected_cell_generation=cell.generation,
            expected_cell_state=cell.state,
            expected_unavailable_reason=cell.unavailable_reason,
            expected_cell_id=cell.id,
        )
        assert resumed.state is WorkloadState.PENDING
        assert resumed.generation == lease.generation + 1
        assert resumed.lease_id is None and resumed.lease_ids == ()
        assert released.id == lease.id and released.state is LeaseState.RELEASED
        assert released.settlement_reason == "unstarted" and released.settled_cost == 0
        assert released.settled_at is not None and released.resources == lease.resources
        assert released.accelerator_hours_settled == {}
        assert released.reserved_cost == lease.reserved_cost
        assert released.resource_hours_reserved == lease.resource_hours_reserved
        released_history[lease.id] = released
        after = await _snapshot(repository, workload_id)
        assert after["workload"] == resumed
        assert after["leases"] == released_history  # Every earlier release must remain durable and unchanged.
        assert after["budget"] == initial_budget  # Includes nonzero spend, revision, hours and timestamps.
        assert after["cells"] == before["cells"]
        assert after["events"][-2].action == "lease.released_unstarted"
        assert not any(event.action == "budget.lease_settled" for event in after["events"])
        with pytest.raises(ConflictError):
            await ControlPlaneService(repository).requeue_unstarted_workload(
                ADMIN,
                workload_id,
                scheduling_reason="stale replay",
                expected_generation=lease.generation,
                expected_lease_id=lease.id,
                expected_cell_generation=cell.generation,
                expected_cell_state=cell.state,
                expected_unavailable_reason=cell.unavailable_reason,
                expected_cell_id=cell.id,
            )
        assert await _snapshot(repository, workload_id) == after


@settings(max_examples=30, deadline=None)
@given(
    repeats=st.integers(min_value=1, max_value=4),
    rate=st.integers(min_value=0, max_value=30),
    hours=st.integers(min_value=1, max_value=4),
)
async def test_generated_undispatched_retries_conserve_budget(repeats: int, rate: int, hours: int) -> None:
    await _retry_history(InMemoryRepository(), repeats, rate, hours)


@pytest.mark.parametrize("repeats", [1, 2, 4])
async def test_unstarted_retry_conservation_across_repositories(repository, repeats: int) -> None:
    await _retry_history(repository, repeats)


async def test_requeue_refuses_a_dispatch_that_may_have_started(repository) -> None:
    workload_id, cell = await _seed(repository, "started")
    lease = await _admit(repository, workload_id, cell, "started")
    async with repository.transaction() as tx:
        workload = await tx.get_workload(workload_id)
        assert workload is not None
        reservation = WorkloadDispatchReservation(
            fence="dispatch-fence",
            outbox_id="dispatch-started",
            outbox_attempt=1,
            generation=workload.generation,
            workload_attempt=workload.attempt,
            lease_ids=(lease.id,),
        )
        await tx.put_outbox(
            OutboxRecord(
                id=reservation.outbox_id,
                topic="workload.dispatch",
                aggregate_type="workload",
                aggregate_id=workload.id,
                idempotency_key="dispatch-started",
                generation=workload.generation,
                state=OutboxState.PENDING,
                payload={"dispatch_reservation": reservation.model_dump(mode="json")},
            )
        )
        await tx.put_workload(workload.model_copy(update={"dispatch_reservation": reservation}))
    before = await _snapshot(repository, workload_id)
    with pytest.raises(ConflictError, match="may have started"):
        await ControlPlaneService(repository).requeue_unstarted_workload(
            ADMIN,
            workload_id,
            scheduling_reason="unsafe",
            expected_generation=lease.generation,
            expected_lease_id=lease.id,
            expected_cell_generation=cell.generation,
            expected_cell_state=cell.state,
            expected_unavailable_reason=cell.unavailable_reason,
            expected_cell_id=cell.id,
        )
    assert await _snapshot(repository, workload_id) == before
