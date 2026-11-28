# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bounded orphan histories: fencing, held-time accounting and durable replay."""

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
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService

repository = _repository_fixture
ADMIN = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))


async def _history(repository, shape, width, rate, held_seconds, estimated_hours, generation, repeats, queued_only):
    now = utcnow()
    scope = {"org_id": "acme", "project_id": "research"}
    resources = ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, tpu=width))
    workload = None
    if shape != "missing":
        workload = WorkloadRecord(
            id="workload",
            **scope,
            spec=WorkloadDefinition(name="orphan-history", execution=CommandExecution(command="true"), gang=resources),
            state=WorkloadState.FAILED if shape == "terminal" else WorkloadState.RUNNING,
            lease_id="replacement" if shape == "moved" else "lease",
            lease_ids=("replacement",) if shape == "moved" else ("lease",),
            created_by="root",
        )
    cell = CellRecord(
        id="cell",
        provider="local",
        region="local",
        state=CellState.READY,
        capacity=resources.per_replica,
        allocatable=resources.per_replica,
    )
    budget = BudgetRecord(
        **scope,
        limit=1_000_000,
        spent=7,
        accelerator_hours_spent={"tpu": 3},
        period_started_at=now,
        updated_by="root",
    )
    created_at = now - timedelta(seconds=held_seconds)
    lease = LeaseRecord(
        id="lease",
        workload_id="workload",
        cell_id="cell",
        **scope,
        state=LeaseState.PENDING if queued_only else LeaseState.ACTIVE,
        resources=resources,
        hourly_cost=rate,
        reserved_cost=rate * estimated_hours,
        resource_hours_reserved=ResourceHours(tpu=width * estimated_hours),
        generation=generation,
        fencing_token=f"cell:{generation}",
        created_at=created_at,
        run_started_at=None if queued_only else created_at + timedelta(seconds=held_seconds // 2),
        expires_at=now + timedelta(hours=1),
    )
    async with repository.transaction() as tx:
        await tx.put_cell(cell)
        await tx.put_budget(budget)
        if workload is not None:
            await tx.put_workload(workload)
        await tx.put_lease(lease)

    async def snapshot():
        async with repository.transaction() as tx:
            return (
                await tx.get_lease(lease.id),
                await tx.get_budget("acme", "research"),
                await tx.get_workload("workload"),
                await tx.get_cell(cell.id),
                await tx.list_events(**scope, limit=100),
            )

    initial = await snapshot()
    # A stale scan must not release capacity or charge even a genuine orphan.
    assert not await ControlPlaneService(repository).settle_orphaned_lease(
        ADMIN, lease.id, reason="stale-scan", expected_generation=generation + 1
    )
    assert await snapshot() == initial

    changed = await ControlPlaneService(repository).settle_orphaned_lease(
        ADMIN, lease.id, reason="orphan-history", expected_generation=generation
    )
    if shape == "claimed":
        # A live workload may have reclaimed the lease since the scanner saw it.
        assert not changed
        assert await snapshot() == initial
        return
    assert changed
    settled, charged, unchanged_workload, unchanged_cell, events = await snapshot()
    assert settled is not None and settled.settled_at is not None and charged is not None
    elapsed = (settled.settled_at - created_at).total_seconds()
    assert settled.state is LeaseState.RELEASED
    assert settled.settlement_reason == "orphan-history"
    assert settled.settled_cost == pytest.approx(rate * elapsed / 3600)
    assert settled.accelerator_hours_settled["tpu"] == pytest.approx(width * elapsed / 3600)
    assert settled.queued_seconds + settled.running_seconds == pytest.approx(elapsed)
    if queued_only:
        assert settled.running_seconds == 0
    else:
        assert settled.queued_seconds == pytest.approx(held_seconds // 2)
    assert charged.spent == pytest.approx(budget.spent + settled.settled_cost)
    assert charged.accelerator_hours_spent["tpu"] == pytest.approx(3 + settled.accelerator_hours_settled["tpu"])
    assert charged.revision == budget.revision + 1
    assert unchanged_workload == workload
    assert unchanged_cell == cell
    assert [event.action for event in events].count("lease.settled") == 1
    assert [event.action for event in events].count("budget.lease_settled") == 1
    first = await snapshot()
    for attempt in range(repeats):
        # New services and different replay reasons cannot charge or rewrite it.
        assert not await ControlPlaneService(repository).settle_orphaned_lease(
            ADMIN, lease.id, reason=f"replayed-{attempt}", expected_generation=generation
        )
        assert await snapshot() == first


@settings(max_examples=40, deadline=None)
@given(
    shape=st.sampled_from(["missing", "terminal", "moved", "claimed"]),
    width=st.integers(1, 16),
    rate=st.integers(1, 100),
    held_seconds=st.integers(1, 7200),
    estimated_hours=st.integers(1, 8),
    generation=st.integers(1, 20),
    repeats=st.integers(2, 5),
    queued_only=st.booleans(),
)
async def test_generated_orphan_history_conserves_accounting(
    shape, width, rate, held_seconds, estimated_hours, generation, repeats, queued_only
):
    await _history(
        InMemoryRepository(), shape, width, rate, held_seconds, estimated_hours, generation, repeats, queued_only
    )


@pytest.mark.parametrize("shape", ["missing", "terminal", "moved", "claimed"])
@pytest.mark.parametrize("queued_only", [False, True])
async def test_orphan_history_across_repositories(repository, shape, queued_only):
    await _history(repository, shape, 8, 17, 90, 3, 4, 3, queued_only)
