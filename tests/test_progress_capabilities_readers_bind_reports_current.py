# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Progress capabilities and readers must bind reports to the current attempt."""

from __future__ import annotations

import asyncio

import pytest
from fastapi.testclient import TestClient

from eyvan.artifacts.managed import callable_uri, result_uri
from eyvan.config import EyvanSettings
from eyvan.records import (
    CallableExecution,
    Principal,
    Role,
    RoleGrant,
    UserProgressReport,
    WorkloadAttemptTopology,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    WorkloadTopologyCell,
)
from eyvan.server.api.app import create_app
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.workload_progress import WorkloadProgressResolver


def _principal() -> Principal:
    return Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )


async def _seed(service: ControlPlaneService, *, frozen: bool = False) -> WorkloadRecord:
    workload = WorkloadRecord(
        id="wl-progress-current",
        org_id="acme",
        project_id="research",
        created_by="alice",
        state=WorkloadState.RUNNING,
        attempt=1,
        spec=WorkloadDefinition(
            name="train",
            execution=CallableExecution(
                artifact_uri=callable_uri("0" * 64),
                sha256="0" * 64,
                python="3.11",
                result_uri=result_uri("progress-current-result"),
            ),
        ),
    )
    async with service.repository.transaction() as tx:
        await tx.put_workload(workload)
        if frozen:
            await tx.put_workload_topology(
                WorkloadAttemptTopology.freeze(
                    workload_id=workload.id,
                    attempt=1,
                    generation=workload.generation,
                    cells=[
                        WorkloadTopologyCell(
                            cell_id="cell-progress",
                            generation=1,
                            runtime_process_count=1,
                            global_rank_offset=0,
                        )
                    ],
                )
            )
    return workload


@pytest.mark.parametrize("frozen", [False, True])
def test_current_result_capability_cannot_overwrite_progress_with_a_future_attempt(tmp_path, frozen: bool) -> None:
    repository = InMemoryRepository()
    app = create_app(
        settings=EyvanSettings(
            dev_auth_enabled=True,
            public_url="https://eyvan.test",
            artifact_store_uri=str(tmp_path / "managed"),
            artifact_token_key="k" * 32,
        ),
        repository=repository,
    )
    service = app.state.service
    workload = asyncio.run(_seed(service, frozen=frozen))
    execution = workload.spec.execution
    assert isinstance(execution, CallableExecution)
    transfer = app.state.artifact_access.result_upload_transfer(
        execution.result_uri,
        org_id=workload.org_id,
        project_id=workload.project_id,
        workload_id=workload.id,
    )
    client = TestClient(app)
    request_kwargs = {
        "params": {"org_id": workload.org_id, "project_id": workload.project_id},
        "headers": {"X-Eyvan-Artifact-Token": transfer.token},
    }
    route = f"/api/v1/workloads/{workload.id}/progress-reports"
    current = client.post(route, json={"rank": 0, "attempt": 1, "step": 7, "total": 100}, **request_kwargs)
    assert current.status_code == 200, current.text
    future = client.post(route, json={"rank": 0, "attempt": 999, "step": 99, "total": 100}, **request_kwargs)

    async def stored() -> list[UserProgressReport]:
        async with repository.transaction() as tx:
            return await tx.list_progress_reports(workload.id)

    reports = asyncio.run(stored())
    assert [(report.attempt, report.step) for report in reports] == [(1, 7)], future.text
    assert future.status_code == 400, future.text


@pytest.mark.asyncio
@pytest.mark.parametrize("reader", ["list", "resolver"])
@pytest.mark.parametrize("noncurrent_attempt", [1, 999])
async def test_after_retry_readers_withhold_every_noncurrent_report(reader: str, noncurrent_attempt: int) -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository=repository)
    workload = await _seed(service)
    # Old persisted rows must also be withheld, including rows accepted before
    # the future-attempt write guard was deployed. Keep the valid current rank.
    async with repository.transaction() as tx:
        await tx.put_workload(workload.model_copy(update={"attempt": 2}))
        await tx.put_progress_report(workload.id, UserProgressReport(rank=0, attempt=noncurrent_attempt, step=99))
        await tx.put_progress_report(workload.id, UserProgressReport(rank=1, attempt=2, step=5))

    if reader == "list":
        reports = await service.list_progress_reports(_principal(), workload.id)
    else:
        progress = await WorkloadProgressResolver(service, repository).resolve(_principal(), workload.id)
        reports = progress.user_progress

    assert [(report.rank, report.attempt, report.step) for report in reports] == [(1, 2, 5)]


@pytest.mark.asyncio
async def test_stale_report_remains_a_noop_with_the_current_report_returned() -> None:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository=repository)
    workload = await _seed(service)
    async with repository.transaction() as tx:
        await tx.put_workload(workload.model_copy(update={"attempt": 2}))
    current = await service.record_progress_report(workload_id=workload.id, rank=0, attempt=2, step=5)
    stale = await service.record_progress_report(workload_id=workload.id, rank=0, attempt=1, step=99)
    assert stale == current
    async with repository.transaction() as tx:
        assert await tx.list_progress_reports(workload.id) == [current]
