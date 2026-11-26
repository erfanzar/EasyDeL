# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Reporting regressions and explicitly blocked billing repair contracts."""

from __future__ import annotations

import socket
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from eyvan._internal.platform import EyvanPlatformConfig
from eyvan.capacity import CapacityStatus
from eyvan.config import EyvanSettings
from eyvan.records import (
    CellRecord,
    CellState,
    CommandExecution,
    LeaseRecord,
    LeaseState,
    Principal,
    ResourceGang,
    ResourceLimits,
    ResourceVector,
    Role,
    RoleGrant,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
)
from eyvan.server.api.app import create_app
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import ConflictError
from eyvan.worker.agent.client import CellAgentClient
from eyvan.worker.agent.models import AgentObservation, WorkloadMutationRequest

ADMIN = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
START = datetime(2026, 1, 1, tzinfo=UTC)


async def _billed_service() -> ControlPlaneService:
    repository = InMemoryRepository()
    await repository.open()
    service = ControlPlaneService(repository)
    cell = CellRecord(
        id="cell-reporting",
        provider="manual",
        capacity_backend="manual",
        region="us-east5",
        state=CellState.READY,
        capacity=ResourceLimits(tpu=4),
        allocatable=ResourceLimits(tpu=4),
        unit_prices={"tpu": 2.0},
    )
    async with repository.transaction() as tx:
        await tx.put_cell(cell)
    await service.accrue_cell_billing(now=START)
    await service.accrue_cell_billing(now=START + timedelta(hours=1))
    return service


@pytest.mark.asyncio
async def test_price_update_does_not_reprice_previous_idle_hours() -> None:
    service = await _billed_service()
    before = await service.cell_billing_report(ADMIN)
    async with service.repository.transaction() as tx:
        cell = await tx.get_cell("cell-reporting")
    assert cell is not None
    with pytest.raises(ConflictError, match="billing rate"):
        await service.upsert_cell(ADMIN, cell.model_copy(update={"unit_prices": {"tpu": 4.0}}), idempotency_key="rate-b")
    async with service.repository.transaction() as tx:
        assert await tx.get_cell(cell.id) == cell
        assert await tx.get_idempotency("fleet", "cell.upsert", "rate-b") is None
    after = await service.cell_billing_report(ADMIN)
    assert before.billed_seconds == after.billed_seconds == 3600.0
    assert after.accrued_cost == before.accrued_cost == 8.0
    assert after.idle_cost == before.idle_cost == 8.0
    await service.accrue_cell_billing(now=START + timedelta(hours=2))
    later = await service.cell_billing_report(ADMIN)
    assert later.accrued_cost == later.idle_cost == 16.0
    await service.accrue_cell_billing(now=START + timedelta(hours=2))
    assert (await service.cell_billing_report(ADMIN)).accrued_cost == 16.0


class _HealthyAgent(CellAgentClient):
    def __init__(self) -> None:
        pass

    def reconcile(self, cell_id: str, request: WorkloadMutationRequest) -> AgentObservation:
        return AgentObservation(cell_id=cell_id, state=CapacityStatus.HEALTHY, generation=request.mutation.generation)


@pytest.mark.asyncio
async def test_recreation_closes_billing_clock_even_when_no_tick_sees_provisioning(monkeypatch) -> None:
    service = await _billed_service()
    async with service.repository.transaction() as tx:
        cell = await tx.get_cell("cell-reporting")
        assert cell is not None
        await tx.put_cell(cell.model_copy(update={"state": CellState.UNAVAILABLE}))
    monkeypatch.setattr("eyvan.server.service._control_plane.cells.utcnow", lambda: START + timedelta(hours=2))
    recreated, replayed = await service.recreate_cell(
        ADMIN, "cell-reporting", expected_generation=1, idempotency_key="recreate-reporting"
    )
    assert not replayed
    assert recreated.state is CellState.REGISTERING
    assert recreated.generation == 2
    controller = EyvanController(
        service.repository,
        service,
        settings=EyvanSettings(controller_enabled=False),
        agent_factory=lambda _: _HealthyAgent(),
    )
    try:
        assert await controller._observe_cell_once_locked(recreated, START + timedelta(hours=9))
    finally:
        await controller.close()
    async with service.repository.transaction() as tx:
        ready = await tx.get_cell("cell-reporting")
    assert ready is not None and ready.state is CellState.READY
    accrual = await service.accrue_cell_billing(now=START + timedelta(hours=10))
    # Only the hour since the new READY observation can be billable; the old
    # incarnation's previously persisted hour must also remain in the report.
    assert accrual.accrued_seconds <= 3600.0
    assert 3600.0 <= (await service.cell_billing_report(ADMIN)).billed_seconds <= 7200.0


VIEWER = Principal(subject="reader", grants=(RoleGrant(role=Role.PROJECT_VIEWER, org_id="acme", project_id="research"),))


class _ViewerVerifier:
    async def verify(self, token: str) -> Principal:
        assert token == "reporting-test-token"
        return VIEWER


async def _usage_service() -> ControlPlaneService:
    repository = InMemoryRepository()
    await repository.open()
    service = ControlPlaneService(repository)
    resources = ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1))
    async with repository.transaction() as tx:
        for name, created, price in (("old", START - timedelta(seconds=1), 99.0), ("included", START, 7.0)):
            await tx.put_workload(
                WorkloadRecord(
                    id=name,
                    org_id="acme",
                    project_id="research",
                    created_by=name,
                    created_at=created,
                    state=WorkloadState.SUCCEEDED,
                    spec=WorkloadDefinition(name=name, execution=CommandExecution(command="true"), gang=resources),
                )
            )
            await tx.put_lease(
                LeaseRecord(
                    id=f"lease-{name}",
                    workload_id=name,
                    cell_id="cell-reporting",
                    org_id="acme",
                    project_id="research",
                    state=LeaseState.RELEASED,
                    resources=resources,
                    fencing_token="test-fence",
                    expires_at=START + timedelta(hours=1),
                    settled_cost=price,
                    settled_at=START + timedelta(hours=1),
                    settlement_reason="released",
                    hourly_cost=price,
                    running_seconds=3600,
                )
            )
    return service


@pytest.mark.asyncio
@pytest.mark.parametrize("since", ["2026-01-01T00:00:00", "2026-01-01T00:00:00Z", "2026-01-01T02:00:00+02:00"])
async def test_usage_http_interprets_timezone_less_cutoff_as_utc(since, monkeypatch, tmp_path) -> None:
    def refuse_outbound(*args, **kwargs):
        raise AssertionError("outbound sockets are forbidden in this reporting regression")

    monkeypatch.setattr(socket.socket, "connect", refuse_outbound)
    service = await _usage_service()
    app = create_app(
        settings=EyvanSettings(
            controller_enabled=False,
            artifact_token_key="reporting-test-key-not-a-secret-0000",
            artifact_store_uri=str(tmp_path / "artifacts"),
        ),
        repository=service.repository,
        service=service,
        verifier=_ViewerVerifier(),
        platform_config=EyvanPlatformConfig(),
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://test"
    ) as client:
        response = await client.get(
            "/api/v1/usage",
            headers={"Authorization": "Bearer reporting-test-token"},
            params={"org_id": "acme", "project_id": "research", "since": since},
        )
    assert response.status_code == 200, response.text
    report = response.json()
    assert len(report["subjects"]) == 1
    assert report["subjects"][0]["subject"] == "included"
    assert report["subjects"][0]["workloads"] == report["subjects"][0]["leases"] == 1
    assert report["subjects"][0]["settled_cost"] == 7.0


@pytest.mark.asyncio
async def test_usage_service_interprets_timezone_less_cutoff_as_utc() -> None:
    service = await _usage_service()
    report = await service.project_usage(VIEWER, org_id="acme", project_id="research", since=START.replace(tzinfo=None))
    assert [row.subject for row in report.subjects] == ["included"]
    assert report.total_cost == 7.0
