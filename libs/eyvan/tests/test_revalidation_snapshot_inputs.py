# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Commit-time admission revalidation must see the same federation inputs as the plan.

``schedule_once`` tightens a grantee plane's budgets and quotas through its grants
and ``plane_id``. The in-transaction revalidation inside ``create_lease`` re-derived
the decision without them, so the authoritative check enforced weaker limits than
the plan that produced it -- a spend that crossed the grant (but not the local
budget) between snapshot and commit would have been admitted.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from eyvan.records import (
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    Principal,
    ResourceGang,
    ResourceHours,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    utcnow,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService


class _SpyScheduler:
    def __init__(self, inner) -> None:
        self.inner = inner
        self.calls: list[dict] = []

    def schedule(self, workloads, cells, **kwargs):
        self.calls.append(kwargs)
        return self.inner.schedule(workloads, cells, **kwargs)


@pytest.mark.asyncio
async def test_create_lease_revalidation_receives_federation_inputs() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    spy = _SpyScheduler(service.scheduler)
    service.scheduler = spy
    service.plane_id = "child-plane"
    admin = Principal(subject="system", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    operator = Principal(
        subject="operator",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )
    gang = ResourceGang(per_replica=ResourceVector(cpu=4, memory_gib=0))
    stored_cell, _ = await service.upsert_cell(
        admin,
        CellRecord(
            id="cell-a",
            provider="local",
            region="local",
            capacity=gang.per_replica,
            allocatable=gang.per_replica,
        ),
        idempotency_key="cell",
    )
    async with repository.transaction() as tx:
        await tx.put_cell(stored_cell.model_copy(update={"state": CellState.READY}))
    workload, _ = await service.submit_workload(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="train", execution=CommandExecution(command="python train.py"), gang=gang),
        idempotency_key="workload",
    )
    # create_lease validates the caller's lease against the scheduler's proposed
    # placement, so derive the record from the plan the way a real caller does.
    plan = await service.schedule_once(admin)
    proposed = next(p for p in plan.placements if p.workload_id == workload.id).lease
    spy.calls.clear()
    await service.create_lease(
        admin,
        LeaseRecord(
            id=proposed.lease_id,
            workload_id=workload.id,
            cell_id=stored_cell.id,
            org_id=workload.org_id,
            project_id=workload.project_id,
            resources=gang,
            fencing_token="fence-a",
            hourly_cost=float(proposed.hourly_cost),
            reserved_cost=float(proposed.reserved_cost),
            resource_hours_reserved=ResourceHours(
                **{name: float(quantity) for name, quantity in proposed.resource_hours_reserved.items()}
            ),
            expires_at=utcnow() + timedelta(minutes=5),
        ),
        idempotency_key="lease-a",
    )

    assert spy.calls, "create_lease must revalidate through the scheduler"
    assert spy.calls[0].get("plane_id") == "child-plane"
    assert "grants" in spy.calls[0]
    assert "quota_grants" in spy.calls[0]
