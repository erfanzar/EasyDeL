# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Idle-capacity documentation must not promise a leased-job deadline."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import NoReturn

import pytest

from eyvan._internal.compat import UTC
from eyvan.config import EyvanSettings
from eyvan.records import (
    CapacityPoolRecord,
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    Principal,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService


@pytest.mark.parametrize(
    "document",
    (
        "docs/security.md",
        "docs/tutorial/11-artifacts-and-secrets.md",
        "docs/tutorial/12-security.md",
    ),
)
def test_idle_capacity_docs_do_not_promise_leased_workload_reaping(document: str) -> None:
    root = Path(__file__).resolve().parents[1]
    text = " ".join((root / document).read_text(encoding="utf-8").split())
    assert "reaped on the fleet's idle timeout regardless of what the workload believes" not in text, (
        f"{document} promises an idle-capacity timer will reap an actively leased workload; "
        "the controller deliberately excludes active lease cells"
    )
    assert "eligible unleased surplus capacity" in text
    assert "demand, protection, and minimum-capacity policy" in text
    assert "not a workload runtime or cost deadline" in text
    assert "lease expiry/quarantine" in text
    assert "fenced retirement" in text
    assert "explicit workload cancellation" in text


def _refuse_agent(cell: CellRecord) -> NoReturn:
    raise AssertionError(f"idle-capacity reconciliation must not contact an agent: {cell.id}")


@pytest.mark.asyncio
@pytest.mark.parametrize("leased", (False, True), ids=("unleased-surplus", "active-lease"))
async def test_old_idle_clock_is_not_an_active_workload_lifetime_bound(leased: bool) -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    controller = EyvanController(repository, service, settings=EyvanSettings(), agent_factory=_refuse_agent)
    now = datetime.now(UTC)
    old = now - timedelta(hours=1)
    pool = CapacityPoolRecord(
        id="research/cpu",
        fleet_name="research",
        node_type="cpu",
        provider="gcp",
        capacity_backend="gcp-compute",
        provider_project="project-a",
        region="us-central1",
        zone="us-central1-a",
        capacity={"cpu": 4, "memory_gib": 16},
        allocatable={"cpu": 4, "memory_gib": 16},
        executor="native-local",
        agent_endpoint="http://agent.invalid:8090",
        min_cells=0,
        max_cells=1,
        idle_timeout_s=1,
        shrink_stabilization_s=0,
    )
    principal = Principal(subject="admin", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    await service.apply_fleet(principal, "research", (pool,), idempotency_key="idle-docs-fleet")
    cell = pool.make_cell("idle-docs-cell").model_copy(
        update={"state": CellState.READY, "created_at": old, "idle_since": old}
    )
    workload = WorkloadRecord(
        id="idle-docs-workload",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="long-running",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 4, "memory_gib": 16}},
        ),
        state=WorkloadState.RUNNING,
        lease_id="idle-docs-lease",
        remote_id="execution-a",
        created_by="alice",
        created_at=old,
    )
    lease = LeaseRecord(
        id="idle-docs-lease",
        workload_id=workload.id,
        cell_id=cell.id,
        cell_generation=cell.generation,
        org_id=workload.org_id,
        project_id=workload.project_id,
        state=LeaseState.ACTIVE,
        resources=workload.spec.gang,
        fencing_token="lease-token",
        expires_at=now + timedelta(hours=1),
    )
    async with repository.transaction() as transaction:
        await transaction.put_cell(cell)
        if leased:
            await transaction.put_workload(workload)
            await transaction.put_lease(lease)

    result = await controller.reconcile_capacity_pools_once()

    assert result == {"launched": 0, "drained": 0 if leased else 1, "deleted": 0}
    async with repository.transaction() as transaction:
        retained_cell = await transaction.get_cell(cell.id)
        retained_workload = await transaction.get_workload(workload.id)
        retained_lease = await transaction.get_lease(lease.id)
    assert retained_cell is not None
    assert retained_cell.state is (CellState.READY if leased else CellState.DRAINING)
    if leased:
        assert retained_workload == workload
        assert retained_lease == lease
        assert retained_cell.idle_since == old
    else:
        assert retained_workload is None
        assert retained_lease is None
