# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""On-demand scheduling explanations for pending workloads."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.domain.models import InfeasibilityCode
from eyvan.records import (
    CellRecord,
    CellState,
    CommandExecution,
    OutboxRecord,
    OutboxState,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadState,
)
from eyvan.server.api.app import create_app
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import AuthorizationError, NotFoundError
from eyvan.server.service.scheduling import CoreSchedulingService


def _principal(role: Role, *, subject: str = "alice", project: str = "research") -> Principal:
    return Principal(subject=subject, grants=(RoleGrant(role=role, org_id="acme", project_id=project),))


def _spec(*, cpu: int = 4) -> WorkloadDefinition:
    return WorkloadDefinition(
        name="train",
        execution=CommandExecution(command="python train.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=cpu, memory_gib=cpu)),
    )


def _cell(cell_id: str, *, cpu: int, state: CellState = CellState.READY) -> CellRecord:
    return CellRecord(
        id=cell_id,
        provider="local",
        region="local",
        state=state,
        capacity=ResourceVector(cpu=cpu, memory_gib=cpu),
        allocatable=ResourceVector(cpu=cpu, memory_gib=cpu),
    )


async def _submit(service: ControlPlaneService, spec: WorkloadDefinition, key: str = "workload"):
    workload, _ = await service.submit_workload(
        _principal(Role.PROJECT_OPERATOR),
        org_id="acme",
        project_id="research",
        spec=spec,
        idempotency_key=key,
    )
    return workload


async def test_explain_reports_the_cell_a_pending_workload_would_be_placed_on() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    async with repository.transaction() as transaction:
        await transaction.put_cell(_cell("cell-a", cpu=8))
    workload = await _submit(service, _spec())

    explanation = await service.explain_workload(_principal(Role.PROJECT_OPERATOR), workload.id)

    assert explanation.workload_id == workload.id
    assert explanation.state is WorkloadState.PENDING
    assert explanation.placeable is True
    assert explanation.placement_cell_id == "cell-a"
    assert explanation.code is None


async def test_explain_reports_no_cells_when_the_fleet_is_empty() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    workload = await _submit(service, _spec())

    explanation = await service.explain_workload(_principal(Role.PROJECT_OPERATOR), workload.id)

    assert explanation.placeable is False
    assert explanation.code is InfeasibilityCode.NO_CELLS
    assert explanation.cells == ()


async def test_explain_surfaces_the_per_cell_rejection_detail() -> None:
    """The scheduler already computes why each cell was rejected; explain must expose it."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    async with repository.transaction() as transaction:
        await transaction.put_cell(_cell("cell-small-a", cpu=1))
        await transaction.put_cell(_cell("cell-small-b", cpu=2))
    workload = await _submit(service, _spec(cpu=64))

    explanation = await service.explain_workload(_principal(Role.PROJECT_OPERATOR), workload.id)

    assert explanation.placeable is False
    assert explanation.considered_cell_count == 2
    assert explanation.truncated_cell_count == 0
    # Deterministic order so operator output is stable across scheduling passes.
    assert [cell.cell_id for cell in explanation.cells] == ["cell-small-a", "cell-small-b"]
    assert all(cell.code is InfeasibilityCode.INSUFFICIENT_CAPACITY for cell in explanation.cells)
    assert all(cell.message for cell in explanation.cells)


async def test_explain_does_not_schedule_a_workload_that_is_not_pending() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    workload = await _submit(service, _spec())
    async with repository.transaction() as transaction:
        stored = await transaction.get_workload(workload.id)
        assert stored is not None
        await transaction.put_workload(stored.model_copy(update={"state": WorkloadState.SUCCEEDED}))

    explanation = await service.explain_workload(_principal(Role.PROJECT_OPERATOR), workload.id)

    assert explanation.state is WorkloadState.SUCCEEDED
    assert explanation.placeable is False
    assert explanation.code is None
    assert "only pending workloads are scheduled" in (explanation.message or "")


async def test_explain_does_not_call_a_settled_command_live() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    workload = await _submit(service, _spec())
    async with repository.transaction() as transaction:
        stored = await transaction.get_workload(workload.id)
        assert stored is not None
        await transaction.put_workload(stored.model_copy(update={"state": WorkloadState.SUCCEEDED}))
        await transaction.put_outbox(
            OutboxRecord(
                id="dispatch-corpse",
                topic="workload.dispatch",
                aggregate_type="workload",
                aggregate_id=workload.id,
                idempotency_key="dispatch:corpse",
                generation=stored.generation,
                payload={"lease_id": stored.lease_id or "lease_x", "cell_id": "cell_x"},
                state=OutboxState.FAILED,
                attempts=9,
            )
        )

    explanation = await service.explain_workload(_principal(Role.PROJECT_OPERATOR), workload.id)

    assert explanation.placeable is False
    message = explanation.message or ""
    assert "only pending workloads are scheduled" in message
    assert "being retried" not in message


async def test_explain_carries_the_durable_scheduling_reason() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    workload = await _submit(service, _spec())
    async with repository.transaction() as transaction:
        stored = await transaction.get_workload(workload.id)
        assert stored is not None
        await transaction.put_workload(stored.model_copy(update={"scheduling_reason": "no_cells: nothing registered"}))

    explanation = await service.explain_workload(_principal(Role.PROJECT_OPERATOR), workload.id)

    assert explanation.scheduling_reason == "no_cells: nothing registered"


async def test_explain_requires_read_access_to_the_workload_project() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    workload = await _submit(service, _spec())

    outsider = _principal(Role.PROJECT_OPERATOR, subject="mallory", project="other")
    with pytest.raises(AuthorizationError):
        await service.explain_workload(outsider, workload.id)


async def test_explain_rejects_an_unknown_workload() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())

    with pytest.raises(NotFoundError):
        await service.explain_workload(_principal(Role.PROJECT_OPERATOR), "wl-missing")


async def test_explain_route_returns_the_per_cell_detail() -> None:
    """Prove the HTTP route is wired, not just the service method."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    async with repository.transaction() as transaction:
        await transaction.put_cell(_cell("cell-small", cpu=1))
    workload = await _submit(service, _spec(cpu=64))

    # The controller is disabled so this asserts route wiring, not the reconciler's
    # (correct) decision to mark an agent-less cell unavailable.
    app = create_app(
        settings=EyvanSettings(dev_auth_enabled=True, controller_enabled=False),
        repository=repository,
        service=service,
    )
    with TestClient(app) as client:
        response = client.get(
            f"/api/v1/workloads/{workload.id}/explain",
            headers={
                "X-Eyvan-User": "alice",
                "X-Eyvan-Org": "acme",
                "X-Eyvan-Project": "research",
                "X-Eyvan-Roles": "operator",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["workload_id"] == workload.id
    assert payload["placeable"] is False
    assert payload["considered_cell_count"] == 1
    assert payload["cells"][0]["cell_id"] == "cell-small"
    assert payload["cells"][0]["code"] == InfeasibilityCode.INSUFFICIENT_CAPACITY.value
