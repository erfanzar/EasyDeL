# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Child response-loss recovery must not consume another concurrency slot."""

from __future__ import annotations

import pytest

from eyvan.records import CommandExecution, Principal, Role, RoleGrant, WorkloadDefinition, WorkloadRecord, WorkloadState
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import ConflictError
from eyvan.server.service.subjobs import MAX_LIVE_CHILDREN, JobTreeTooWideError


def _principal() -> Principal:
    return Principal(
        subject="alice",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
    )


def _spec(name: str, *, launcher: bool = False) -> WorkloadDefinition:
    return WorkloadDefinition(
        name=name,
        execution=CommandExecution(command="python train.py"),
        can_submit_children=launcher,
    )


async def _family(count: int) -> tuple[InMemoryRepository, ControlPlaneService, WorkloadRecord, WorkloadRecord]:
    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    parent, replayed = await service.submit_workload(
        _principal(),
        org_id="acme",
        project_id="research",
        spec=_spec("launcher", launcher=True),
        idempotency_key="launcher",
    )
    assert replayed is False
    for index in range(count):
        child, replayed = await service.submit_child_workload(
            parent=parent,
            spec=_spec(f"child-{index}"),
            idempotency_key=f"child-{index}",
            expected_parent_generation=parent.generation,
        )
        assert replayed is False
    return repository, service, parent, child


async def test_response_loss_replay_at_live_child_ceiling_preserves_durable_records() -> None:
    repository, service, parent, child = await _family(MAX_LIVE_CHILDREN)
    async with repository.transaction() as tx:
        before_workloads = await tx.list_workloads()
        before_events = await tx.list_events(limit=1000)
    assert len(before_workloads) == MAX_LIVE_CHILDREN + 1

    recovered, replayed = await service.submit_child_workload(
        parent=parent,
        spec=_spec(f"child-{MAX_LIVE_CHILDREN - 1}"),
        idempotency_key=f"child-{MAX_LIVE_CHILDREN - 1}",
        expected_parent_generation=parent.generation,
    )
    assert replayed is True
    assert recovered == child

    with pytest.raises(JobTreeTooWideError, match="live children"):
        await service.submit_child_workload(
            parent=parent,
            spec=_spec("new-child"),
            idempotency_key="new-child",
            expected_parent_generation=parent.generation,
        )
    async with repository.transaction() as tx:
        assert await tx.list_workloads() == before_workloads
        assert await tx.list_events(limit=1000) == before_events
        assert len(await tx.list_child_workloads(parent.id)) == MAX_LIVE_CHILDREN


async def test_changed_request_at_live_child_ceiling_keeps_idempotency_conflict() -> None:
    repository, service, parent, _ = await _family(MAX_LIVE_CHILDREN)
    with pytest.raises(ConflictError, match="Idempotency-Key was already used with a different request"):
        await service.submit_child_workload(
            parent=parent,
            spec=_spec("changed-request"),
            idempotency_key=f"child-{MAX_LIVE_CHILDREN - 1}",
            expected_parent_generation=parent.generation,
        )
    async with repository.transaction() as tx:
        assert len(await tx.list_child_workloads(parent.id)) == MAX_LIVE_CHILDREN
        assert len(await tx.list_events(limit=1000)) == MAX_LIVE_CHILDREN + 1


async def test_replay_still_requires_current_parent_generation() -> None:
    repository, service, parent, _ = await _family(1)
    async with repository.transaction() as tx:
        await tx.put_workload(parent.model_copy(update={"generation": parent.generation + 1}))
    with pytest.raises(ConflictError, match="generation"):
        await service.submit_child_workload(
            parent=parent,
            spec=_spec("child-0"),
            idempotency_key="child-0",
            expected_parent_generation=parent.generation,
        )


@pytest.mark.parametrize("state", [WorkloadState.CANCELLING, WorkloadState.CANCELLED])
async def test_replay_still_refuses_fresh_terminal_or_cancelling_parent(state: WorkloadState) -> None:
    repository, service, parent, _ = await _family(1)
    async with repository.transaction() as tx:
        await tx.put_workload(parent.model_copy(update={"state": state}))
    with pytest.raises(ConflictError, match="cannot gain new children"):
        await service.submit_child_workload(
            parent=parent,
            spec=_spec("child-0"),
            idempotency_key="child-0",
            expected_parent_generation=parent.generation,
        )
