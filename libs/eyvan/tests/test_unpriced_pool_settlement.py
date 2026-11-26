# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Settlement on a pool that declares no prices.

``_accelerator_hours`` recovered the per-hour vector by dividing
``reserved_cost`` by ``hourly_cost``. On an unpriced pool both are zero, the
division is unavailable, and the fallback handed back ``resource_hours_reserved``
unscaled -- the *submitted estimate*. A 10h-estimate run stopped after ten
minutes settled ten tpu-hours into ``accelerator_hours_spent``, so a hard hour
budget was spent by jobs that barely ran. Money settles correctly on the same
pool (``cost_for_hours`` with a zero hourly is zero); this pins the hours.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from eyvan.records import (
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
    held_hours: float,
) -> tuple[Principal, str, str]:
    """Create a RUNNING workload on a zero-priced lease, backdated by ``held_hours``.

    Settlement charges elapsed time, so backdating the clock by a known amount is
    what lets the test assert an exact figure instead of echoing the estimate.
    """

    fleet_admin = _principal(Role.FLEET_ADMIN)
    operator = _principal(Role.PROJECT_OPERATOR)
    gang = ResourceGang(replicas=1, per_replica=ResourceVector(cpu=0, memory_gib=0, tpu=8))
    workload, _ = await service.submit_workload(
        operator,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="training-unpriced",
            execution=CommandExecution(command="python train.py"),
            gang=gang,
        ),
        idempotency_key="submit-unpriced",
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
        id="lease-unpriced",
        workload_id=workload.id,
        cell_id="cell-a",
        org_id=workload.org_id,
        project_id=workload.project_id,
        resources=gang,
        hourly_cost=0,
        reserved_cost=0,
        resource_hours_reserved=ResourceHours(tpu=3),
        fencing_token="fence-unpriced",
        expires_at=utcnow() + timedelta(hours=1),
    )
    await service.create_lease(fleet_admin, lease, idempotency_key="lease-unpriced")
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
async def test_an_unpriced_pool_settles_accelerator_hours_by_time_held_not_estimate() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    fleet_admin, workload_id, lease_id = await _running_workload(repository, service, held_hours=0.5)

    await service.settle_terminal_workload(
        fleet_admin, workload_id, WorkloadState.SUCCEEDED, action="workload.succeeded"
    )

    async with repository.transaction() as tx:
        lease = await tx.get_lease(lease_id)
        budget = await tx.get_budget("acme", "research")

    assert lease is not None
    # The reserved shape is 3 tpu-hours over the 1h estimate, i.e. 3 tpu/hour;
    # half an hour held settles half of it, not the whole estimate.
    assert lease.accelerator_hours_settled["tpu"] == pytest.approx(1.5, abs=0.02)
    assert lease.settled_cost == 0.0
    assert budget is None or budget.accelerator_hours_spent["tpu"] == pytest.approx(1.5, abs=0.02)
