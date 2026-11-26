# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Elastic progress must reflect bounded, exact scheduler pack widths."""

from __future__ import annotations

import pytest

from eyvan.capacity import CapacityStatus
from eyvan.records import (
    CallableExecution,
    CapacityClass,
    CapacityPoolRecord,
    CellState,
    Principal,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadProgressPhase,
    WorkloadState,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.callable_trust import CallableProjectTrustPolicy
from eyvan.server.service.scheduling import CoreSchedulingService
from eyvan.server.service.workload_progress import WorkloadProgressResolver


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("requested", "widths", "expected_ready", "expected_cells", "placeable"),
    [
        (3, (2, 2), 2, 1, False),
        (4, (3, 2, 2), 4, 2, True),
        (6, (4, 2), 6, 2, True),
        (6, (4, 3, 3, 1, 1), 6, 2, True),
    ],
)
async def test_elastic_progress_agrees_with_real_scheduler_pack(
    requested: int,
    widths: tuple[int, ...],
    expected_ready: int,
    expected_cells: int,
    placeable: bool,
) -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(
        repository,
        callable_trust=CallableProjectTrustPolicy.from_projects({"acme/research"}),
    )
    principal = Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )
    resources = ResourceVector(cpu=32, tpu=16)
    pool = CapacityPoolRecord(
        id="research-us-central1-c-v6e-16",
        fleet_name="research-fleet",
        node_type="v6e-16",
        provider="gcp",
        capacity_backend="gcp-tpu",
        provider_project="example-project-1",
        region="us-central1",
        zone="us-central1-c",
        capacity=resources,
        allocatable=resources,
        accelerator_type="v6e-16",
        capacity_class=CapacityClass.SPOT,
        executor="eyvan-native",
        agent_endpoint="http://127.0.0.1:8090",
        labels={"eyvan_resource_kind": "tpu"},
        max_cells=16,
    )
    async with repository.transaction() as transaction:
        await transaction.put_capacity_pool(pool)
    workload, replayed = await service.submit_pool_callable(
        principal,
        fleet_name=pool.fleet_name,
        org_id="acme",
        project_id="research",
        accelerator_type=pool.accelerator_type,
        pods=requested,
        execution=CallableExecution(artifact_uri="gs://trusted/train.pkl", sha256="a" * 64, python="3.11"),
        atomic=False,
        idempotency_key="elastic-progress",
    )
    assert not replayed
    assert workload.state is WorkloadState.PENDING
    cells = tuple(
        pool.make_cell(f"warm-{index}", worker_count=width).model_copy(
            update={"state": CellState.READY, "provider_phase": CapacityStatus.HEALTHY}
        )
        for index, width in enumerate(widths)
    )
    async with repository.transaction() as transaction:
        for cell in cells:
            await transaction.put_cell(cell)

    scheduled = CoreSchedulingService().schedule((workload,), cells)
    progress = await WorkloadProgressResolver(service, repository).resolve(principal, workload.id)

    assert bool(scheduled.placements) is placeable
    assert progress.capacity.ready_workers == expected_ready
    assert progress.capacity.ready_workers <= requested
    assert progress.capacity.ready_cells == expected_cells
    assert (progress.phase is WorkloadProgressPhase.READY_FOR_DISPATCH) is placeable
    if placeable:
        placement = scheduled.placements[0]
        selected = {lease.cell_id for lease in (placement.lease, *placement.member_leases)}
        assert sum(cell.worker_count for cell in cells if cell.id in selected) == requested
