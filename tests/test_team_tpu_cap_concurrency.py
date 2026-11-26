# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The hard per-user compute caps hold under concurrent admissions across projects.

Six operations race for a cap of 8 units, with each workload asking for 4.
PostgreSQL also races new admissions with retry and approval of existing work.
Whatever interleaving the backend produces, at most two jobs may be admitted.
Clients retry serialization failures; the persisted rows must still fit the cap.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from eyvan.records import (
    CommandExecution,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    UserRecord,
    WorkloadDefinition,
    WorkloadState,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.persistence.base import TransientRepositoryError
from eyvan.server.persistence.local import LocalJsonRepository
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import ConflictError
from eyvan.server.service.scheduling import CoreSchedulingService

ADMIN = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
CAP = 8
EACH = 4
SUBMISSIONS = 6


def _alice(project: str) -> Principal:
    return Principal(subject="alice", grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id=project),))


def _job(index: int, resource: str) -> WorkloadDefinition:
    return WorkloadDefinition(
        name=f"job-{index}",
        execution=CommandExecution(command="python train.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector.model_validate({resource: EACH})),
    )


async def _race(repository, resource: str = "tpu", *, mixed: bool = False) -> None:
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService(), governance_enabled=True)
    async with repository.transaction() as tx:
        await tx.put_user(UserRecord(subject="alice", email="alice@example.com", created_by="root"))
    await service.set_user_rules(ADMIN, "alice", **{f"max_{resource}_units": CAP})

    if mixed:
        # One terminal row could re-enter; one held row already counts. Approval
        # must not double-charge it, and retry competes with new submissions.
        research = _alice("research")
        retry, _ = await service.submit_workload(
            research,
            org_id="acme",
            project_id="research",
            spec=_job(0, resource).model_copy(update={"max_retries": 1}),
            idempotency_key="seed-retry",
        )
        await service.cancel_workload(research, retry.id, idempotency_key="cancel")
        await service.set_user_rules(ADMIN, "alice", approval_tpu_threshold=0)
        held_spec = _job(1, resource).model_copy(
            update={"gang": ResourceGang(per_replica=ResourceVector.model_validate({"tpu": 1, resource: EACH}))}
        )
        held, _ = await service.submit_workload(
            _alice("other"), org_id="acme", project_id="other", spec=held_spec, idempotency_key="seed-held"
        )
        assert not held.schedulable
        [request] = await service.list_team_requests(ADMIN, state="open")
        await service.set_user_rules(ADMIN, "alice", approval_tpu_threshold=None)

    async def submit(index: int) -> str:
        project = "research" if index % 2 == 0 else "other"
        for _ in range(50):
            try:
                if mixed and index == 0:
                    await service.retry_workload(_alice(project), retry.id, idempotency_key="retry")
                    return "admitted"
                if mixed and index == 1:
                    await service.resolve_team_request(ADMIN, request.id, resolution="approved")
                    return "released"
                await service.submit_workload(
                    _alice(project),
                    org_id="acme",
                    project_id=project,
                    spec=_job(index, resource),
                    idempotency_key=f"k{index}",
                )
                return "admitted"
            except ConflictError as exc:
                assert f"{resource.upper()} quota" in str(exc), exc
                return "refused"
            except TransientRepositoryError:
                await asyncio.sleep(0.01)
        raise AssertionError("submission never settled")

    outcomes = await asyncio.gather(*(submit(index) for index in range(SUBMISSIONS)))

    async with repository.transaction() as tx:
        stored = [
            item
            for item in await tx.list_workloads()
            if item.created_by == "alice" and item.state is WorkloadState.PENDING
        ]
        if mixed:
            assert (await tx.get_workload(held.id)).schedulable
            assert outcomes.count("released") == 1
    used = sum(getattr(item.spec.gang.per_replica, resource) for item in stored)
    assert used <= CAP, f"{used} {resource} units admitted against a cap of {CAP}"
    assert outcomes.count("admitted") + outcomes.count("released") == len(stored) == CAP // EACH
    assert outcomes.count("refused") == SUBMISSIONS - CAP // EACH
    assert {item.project_id for item in stored} <= {"research", "other"}


@pytest.mark.asyncio
async def test_concurrent_submissions_never_exceed_the_cap_in_memory() -> None:
    repository = InMemoryRepository()
    await repository.open()
    await _race(repository)


@pytest.mark.asyncio
async def test_concurrent_submissions_never_exceed_the_cap_in_local_json(tmp_path) -> None:
    repository = LocalJsonRepository(tmp_path / "control-plane.json")
    await repository.open()
    try:
        await _race(repository)
    finally:
        await repository.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("resource", ["cpu", "gpu", "tpu"])
@pytest.mark.parametrize("mixed", [False, True], ids=["submit", "submit-retry-approval"])
async def test_concurrent_submissions_never_exceed_the_cap_on_postgresql(
    postgres_url: str, resource: str, mixed: bool
) -> None:
    engine = create_async_engine(postgres_url, poolclass=None)
    async with engine.begin() as connection:
        await connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        await connection.execute(text("CREATE SCHEMA public"))
    try:
        await create_schema(engine)
        repository = SqlAlchemyRepository(engine)
        await repository.open()
        try:
            await _race(repository, resource, mixed=mixed)
        finally:
            await repository.close()
    finally:
        await engine.dispose()
