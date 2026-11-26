# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generated terminal-settlement conservation, not every controller transition."""

from datetime import timedelta

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from test_group_lease_lifecycle import repository as _repository_fixture

from eyvan.domain import PlacementMode, WorkerResourceScope
from eyvan.records import (
    BudgetRecord,
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    Principal,
    ResourceGang,
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


async def _exercise(repository, entries, repeats):
    now = utcnow()
    scope = {"org_id": "acme", "project_id": "research"}
    spec = WorkloadDefinition(
        name="settlement-property",
        execution=CommandExecution(command="true"),
        provider="gcp",
        required_executor="eyvan-native",
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, tpu=8)),
        worker_count=sum(entry[0] for entry in entries),
        worker_resource_scope=WorkerResourceScope.PER_WORKER,
        placement=PlacementMode.ELASTIC,
    )
    workload = WorkloadRecord(
        id="generated",
        **scope,
        spec=spec,
        state=WorkloadState.LEASED,
        lease_id="lease-0",
        lease_ids=tuple(f"lease-{i}" for i in range(len(entries))),
        created_by="root",
    )
    async with repository.transaction() as tx:
        await tx.put_budget(BudgetRecord(**scope, limit=1_000_000, updated_by="root"))
        for i, (width, rate, seconds) in enumerate(entries):
            await tx.put_cell(
                CellRecord(
                    id=f"cell-{i}",
                    provider="gcp",
                    capacity_backend="gcp-tpu",
                    provider_project="test-project",
                    region="us-central1",
                    zone="us-central1-a",
                    state=CellState.READY,
                    accelerator_type="v5p-8",
                    capacity=ResourceVector(cpu=width, tpu=8 * width),
                    allocatable=ResourceVector(cpu=width, tpu=8 * width),
                    worker_count=width,
                    provider_options={"node_count": width, **({"node_prefix": f"test-{i}"} if width > 1 else {})},
                    labels={"eyvan_worker_count": str(width)},
                    endpoint=f"http://agent-{i}:8090",
                )
            )
            await tx.put_lease(
                LeaseRecord(
                    id=f"lease-{i}",
                    workload_id=workload.id,
                    cell_id=f"cell-{i}",
                    **scope,
                    state=LeaseState.ACTIVE,
                    resources=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=width, tpu=8 * width)),
                    hourly_cost=rate,
                    created_at=now - timedelta(seconds=seconds),
                    run_started_at=now - timedelta(seconds=seconds),
                    expires_at=now + timedelta(hours=1),
                    fencing_token=f"fence-{i}",
                    group_id="generated-gang",
                    group_index=i,
                )
            )
        await tx.put_workload(workload)
    first_accounting = None
    for _ in range(repeats):
        # Reconstruct the service to avoid mistaking a process-local cache for
        # durable replay protection.
        await ControlPlaneService(repository).settle_terminal_workload(ADMIN, workload.id, WorkloadState.FAILED)
        async with repository.transaction() as tx:
            budget = await tx.get_budget("acme", "research")
            leases = [await tx.get_lease(identifier) for identifier in workload.lease_ids]
            cells = [await tx.get_cell(f"cell-{i}") for i in range(len(entries))]
        assert budget is not None
        assert all(lease is not None and lease.settled_at is not None for lease in leases)
        assert budget.spent == pytest.approx(sum(lease.settled_cost for lease in leases))
        assert budget.spent > 0
        assert [cell.worker_count for cell in cells] == [entry[0] for entry in entries]
        assert [lease.resources.per_replica.cpu for lease in leases] == [entry[0] for entry in entries]
        assert all(lease.resources.replicas == 1 for lease in leases)
        assert all(lease.state is LeaseState.RELEASED for lease in leases)
        accounting = (budget.spent, budget.revision, [(lease.settled_cost, lease.settled_at) for lease in leases])
        if first_accounting is None:
            first_accounting = accounting
        else:
            assert accounting == first_accounting


@settings(max_examples=30, deadline=None)
@given(
    entries=st.lists(st.tuples(st.integers(1, 4), st.integers(1, 100), st.integers(1, 7200)), min_size=1, max_size=4),
    repeats=st.integers(2, 5),
)
async def test_generated_terminal_settlement_conserves_accounting(entries, repeats):
    await _exercise(InMemoryRepository(), entries, repeats)


async def test_terminal_settlement_conservation_across_repositories(repository):
    await _exercise(repository, [(2, 11, 30), (3, 17, 90)], 3)
