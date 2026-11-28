# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Telling "waiting for capacity" apart from "can never run".

``explain_workload`` re-runs the pure scheduler, and the scheduler only ever
sees cells that already exist. A workload the autoscaler is actively building a
TPU slice for therefore got the identical answer to one whose constraints no
pool can ever satisfy:

    no cell satisfies all hard workload constraints

That is the question ``explain`` exists to answer, and for the most common
reason a job sits pending -- capacity is still being provisioned -- it gave the
answer that reads as "give up".
"""

from __future__ import annotations

import pytest
from test_controller import principal, snapshot  # noqa: F401  - shared harness

from eyvan.domain import InfeasibilityCode
from eyvan.records import (
    CellRecord,
    CellState,
    CommandExecution,
    OutboxRecord,
    OutboxState,
    ResourceVector,
    Role,
    WorkloadDefinition,
    WorkloadState,
)
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.scheduling import CoreSchedulingService


def _cell(cell_id: str, *, state: CellState, autoscale_for: str | None = None) -> CellRecord:
    return CellRecord(
        id=cell_id,
        provider="gcp",
        provider_project="gcp-project-a",
        region="us-east5",
        zone="us-east5-a",
        state=state,
        capacity=ResourceVector(cpu=32, memory_gib=0, tpu=8),
        allocatable=ResourceVector(cpu=32, memory_gib=0, tpu=8),
        accelerator_type="v5p-16",
        endpoint="http://agent:8090",
        pool_id=None if autoscale_for is None else "pool-a",
        autoscale_request_workload_id=autoscale_for,
    )


async def _pending_tpu_workload(service: ControlPlaneService) -> str:
    workload, _ = await service.submit_workload(
        principal(Role.PROJECT_OPERATOR),
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="train",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 8, "tpu": 8}},
            accelerator_type="v5p-16",
        ),
        idempotency_key="submit",
    )
    return workload.id


@pytest.mark.asyncio
async def test_a_workload_waiting_on_a_provisioning_cell_is_told_so() -> None:
    """The common case: the slice is being built, and the answer must say that."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    workload_id = await _pending_tpu_workload(service)
    async with repository.transaction() as transaction:
        await transaction.put_cell(_cell("cell-a", state=CellState.REGISTERING, autoscale_for=workload_id))

    explanation = await service.explain_workload(principal(Role.PROJECT_OPERATOR), workload_id)

    assert explanation.placeable is False
    assert explanation.provisioning_cell_ids == ("cell-a",)
    assert "being provisioned" in (explanation.message or "")


@pytest.mark.asyncio
async def test_a_genuinely_infeasible_workload_still_says_so() -> None:
    """Softening the message must not hide a workload that can never be placed."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    workload_id = await _pending_tpu_workload(service)

    explanation = await service.explain_workload(principal(Role.PROJECT_OPERATOR), workload_id)

    assert explanation.placeable is False
    assert explanation.provisioning_cell_ids == ()
    assert explanation.code is not None
    assert "being provisioned" not in (explanation.message or "")


@pytest.mark.asyncio
async def test_a_cell_provisioning_for_a_different_workload_is_not_claimed() -> None:
    """Another job's slice is not this job's answer.

    ``autoscale_request_workload_id`` is what reserves a cell for one workload,
    so reporting someone else's would tell an operator to keep waiting for
    capacity that will never be theirs.
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    workload_id = await _pending_tpu_workload(service)
    async with repository.transaction() as transaction:
        await transaction.put_cell(_cell("cell-b", state=CellState.REGISTERING, autoscale_for="wl_someone_else"))

    explanation = await service.explain_workload(principal(Role.PROJECT_OPERATOR), workload_id)

    assert explanation.provisioning_cell_ids == ()
    assert explanation.code is InfeasibilityCode.HARD_CONSTRAINT


@pytest.mark.asyncio
async def test_a_deleted_cell_is_never_reported_as_provisioning() -> None:
    """A cell on its way out is not capacity anyone is waiting for."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    workload_id = await _pending_tpu_workload(service)
    async with repository.transaction() as transaction:
        await transaction.put_cell(_cell("cell-c", state=CellState.DELETING, autoscale_for=workload_id))

    explanation = await service.explain_workload(principal(Role.PROJECT_OPERATOR), workload_id)

    assert explanation.provisioning_cell_ids == ()


@pytest.mark.asyncio
async def test_a_leased_workload_reports_why_its_delivery_is_failing() -> None:
    """Found on a live head: ten failed dispatches, and this endpoint said nothing.

    Past PENDING the scheduler has no opinion, and the honest answer to "why is
    this not running" moves to delivery. Reporting only the state produced a
    workload that sat `waiting for dispatch` for twenty-one minutes across ten
    attempts while `workload explain` answered "workload is leased; only pending
    workloads are scheduled" -- true, and useless.
    """

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    workload_id = await _pending_tpu_workload(service)

    async with repository.transaction() as transaction:
        workload = await transaction.get_workload(workload_id)
        await transaction.put_workload(workload.model_copy(update={"state": WorkloadState.LEASED}))
        await transaction.put_outbox(
            OutboxRecord(
                id="cmd_stuck",
                topic="workload.dispatch",
                aggregate_type="workload",
                aggregate_id=workload_id,
                idempotency_key=f"workload.dispatch:{workload_id}:{workload.generation}",
                generation=workload.generation,
                state=OutboxState.FAILED,
                attempts=10,
                last_error="native worker refused the submission: connection refused",
            )
        )

    explanation = await service.explain_workload(principal(Role.PROJECT_OPERATOR), workload_id)

    assert explanation.placeable is False
    assert explanation.delivery_attempts == 10
    assert explanation.delivery_state == OutboxState.FAILED.value
    assert "connection refused" in (explanation.delivery_error or "")
    # And the headline says it, so an operator reading one line already knows.
    assert "failed 10 time(s)" in (explanation.message or "")


@pytest.mark.asyncio
async def test_a_leased_workload_delivering_cleanly_reports_no_failure() -> None:
    """The softened message must not imply trouble where there is none."""

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    workload_id = await _pending_tpu_workload(service)

    async with repository.transaction() as transaction:
        workload = await transaction.get_workload(workload_id)
        await transaction.put_workload(workload.model_copy(update={"state": WorkloadState.LEASED}))

    explanation = await service.explain_workload(principal(Role.PROJECT_OPERATOR), workload_id)

    assert explanation.delivery_attempts == 0
    assert explanation.delivery_error is None
    assert "only pending workloads are scheduled" in (explanation.message or "")
