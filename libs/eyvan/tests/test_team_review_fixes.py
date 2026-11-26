# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Review fixes for team tiers and the hard TPU cap.

Each case reproduces a defect found in review: retry bypassing the cap and the
callable rule, losing the last fleet admin through revoke or disable, owner-wide
usage reads, a time-limited admin grant surviving a tier change, and an older
client's rules body silently clearing the cap.
"""

from __future__ import annotations

import typing as tp
from datetime import timedelta

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from test_team_tiers import ADMIN, ALICE, BOB, _callable, _cap, _grants, _http, _service
from test_team_tiers import ADMIN_HEADERS as HTTP_ADMIN

from eyvan.records import (
    CommandExecution,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    RoleGrantRecord,
    WorkloadDefinition,
    WorkloadRecord,
    WorkloadState,
    utcnow,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.persistence.local import LocalJsonRepository
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema
from eyvan.server.persistence.state import StateSession
from eyvan.server.service.errors import AuthorizationError, ConflictError


def _job(tpu: int, *, retries: int = 1) -> WorkloadDefinition:
    return WorkloadDefinition(
        name="job",
        execution=CommandExecution(command="python train.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1, tpu=tpu)),
        max_retries=retries,
    )


async def _state(repository, workload_id: str) -> WorkloadState:
    async with repository.transaction() as tx:
        return (await tx.get_workload(workload_id)).state


# ------------------------------------------------------------ 1. retry bypass


@pytest.mark.asyncio
async def test_retry_cannot_push_a_trusted_user_past_the_cap() -> None:
    service, repository = await _service()
    await service.set_user_tier(ADMIN, "alice", "trusted", org_id="acme", project_id="research")
    await _cap(service, "alice", 8)
    first, _ = await service.submit_workload(
        ALICE, org_id="acme", project_id="research", spec=_job(8), idempotency_key="a"
    )
    await service.cancel_workload(ALICE, first.id, idempotency_key="cancel-a")
    assert await _state(repository, first.id) is WorkloadState.CANCELLED
    await service.submit_workload(ALICE, org_id="acme", project_id="research", spec=_job(8), idempotency_key="b")
    with pytest.raises(ConflictError, match=rf"cannot retry {first.id}: TPU quota: you have 8 of 8"):
        await service.retry_workload(ALICE, first.id, idempotency_key="retry-a")
    assert await _state(repository, first.id) is WorkloadState.CANCELLED


@pytest.mark.asyncio
async def test_retry_within_the_cap_still_works() -> None:
    service, _ = await _service()
    await _cap(service, "alice", 8)
    first, _ = await service.submit_workload(
        ALICE, org_id="acme", project_id="research", spec=_job(8), idempotency_key="a"
    )
    await service.cancel_workload(ALICE, first.id, idempotency_key="cancel-a")
    retried, _ = await service.retry_workload(ALICE, first.id, idempotency_key="retry-a")
    assert retried.state is WorkloadState.PENDING and retried.generation == first.generation + 1


@pytest.mark.asyncio
async def test_an_operator_cannot_retry_a_callable_whose_owner_lost_callable_rights() -> None:
    service, repository = await _service()
    await _cap(service, "alice", 4)
    await service.set_user_rules(ADMIN, "alice", max_cpu_units=1)
    job, _ = await service.submit_workload(
        ALICE,
        org_id="acme",
        project_id="research",
        spec=_callable(1).model_copy(update={"max_retries": 1}),
        idempotency_key="fn",
    )
    await service.cancel_workload(ALICE, job.id, idempotency_key="cancel")
    await _cap(service, "alice", None)  # a plain user without a cap may not run callables
    with pytest.raises(AuthorizationError, match="TPU quota required"):
        await service.retry_workload(BOB, job.id, idempotency_key="bob-retry")
    assert await _state(repository, job.id) is WorkloadState.CANCELLED


# ------------------------------------------------------- 2. last fleet admin


@pytest.mark.asyncio
async def test_revoking_the_last_fleet_admin_grant_is_refused_in_team_mode() -> None:
    service, repository = await _service()
    with pytest.raises(ConflictError, match="last enabled fleet admin"):
        await service.revoke_role_grant(ADMIN, ("root", "fleet_admin", "", ""))
    assert ("fleet_admin", "", "") in await _grants(repository, "root")
    # With a second enabled admin it goes through.
    await service.set_user_tier(ADMIN, "alice", "admin")
    assert await service.revoke_role_grant(ADMIN, ("root", "fleet_admin", "", "")) is True
    assert ("fleet_admin", "", "") not in await _grants(repository, "root")


@pytest.mark.asyncio
async def test_revoking_the_last_fleet_admin_is_unchanged_off_team_mode() -> None:
    service, repository = await _service(enabled=False)
    assert await service.revoke_role_grant(ADMIN, ("root", "fleet_admin", "", "")) is True
    assert ("fleet_admin", "", "") not in await _grants(repository, "root")


@pytest.mark.asyncio
async def test_disabling_the_last_other_fleet_admin_is_refused() -> None:
    service, repository = await _service()
    await service.set_user_tier(ADMIN, "alice", "admin")
    await service.revoke_role_grant(ADMIN, ("root", "fleet_admin", "", ""))
    # alice is now the only enabled team admin; a claim-only fleet admin disables her.
    with pytest.raises(ConflictError, match="last enabled fleet admin"):
        await service.set_user_disabled(ADMIN, "alice", disabled=True)
    async with repository.transaction() as tx:
        assert not (await tx.get_user("alice")).disabled
    # Disabling a non-admin is unaffected.
    await service.set_user_tier(ADMIN, "root", "admin")
    await service.set_user_tier(ADMIN, "alice", "user", org_id="acme", project_id="research")
    disabled = await service.set_user_disabled(ADMIN, "alice", disabled=True)
    assert disabled.disabled


# ------------------------------------------------------------- 3. cap reads


async def _seed_owners(repository) -> None:
    async with repository.transaction() as tx:
        for index, (owner, state) in enumerate(
            [
                ("alice", WorkloadState.PENDING),
                ("alice", WorkloadState.SUCCEEDED),
                ("bob", WorkloadState.PENDING),
                ("alice", WorkloadState.RUNNING),
            ]
        ):
            await tx.put_workload(
                WorkloadRecord(
                    id=f"wl_{index}",
                    org_id="acme",
                    project_id="research" if index % 2 else "other",
                    spec=_job(1),
                    state=state,
                    created_by=owner,
                )
            )


async def _assert_owner_filter(repository) -> None:
    await _seed_owners(repository)
    unfinished = frozenset({WorkloadState.PENDING, WorkloadState.RUNNING})
    async with repository.transaction() as tx:
        everything = await tx.list_workloads_created_by("alice")
        live = await tx.list_workloads_created_by("alice", states=unfinished)
        nobody = await tx.list_workloads_created_by("carol")
    assert sorted(item.id for item in everything) == ["wl_0", "wl_1", "wl_3"]
    assert sorted(item.id for item in live) == ["wl_0", "wl_3"]
    assert nobody == []


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "local", "sqlite"])
async def test_owner_workloads_are_filtered_in_every_backend(backend: str, tmp_path) -> None:
    engine = None
    if backend == "memory":
        repository: tp.Any = InMemoryRepository()
    elif backend == "local":
        repository = LocalJsonRepository(tmp_path / "control-plane.json")
    else:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'plane.db'}")
        await create_schema(engine)
        repository = SqlAlchemyRepository(engine)
    await repository.open()
    try:
        await _assert_owner_filter(repository)
    finally:
        await repository.close()
        if engine is not None:
            await engine.dispose()


@pytest.mark.asyncio
async def test_owner_workloads_are_filtered_on_postgresql(postgres_url: str) -> None:
    from sqlalchemy import text

    engine = create_async_engine(postgres_url, poolclass=None)
    async with engine.begin() as connection:
        await connection.execute(text("DROP SCHEMA IF EXISTS public CASCADE"))
        await connection.execute(text("CREATE SCHEMA public"))
    try:
        await create_schema(engine)
        repository = SqlAlchemyRepository(engine)
        await repository.open()
        try:
            await _assert_owner_filter(repository)
        finally:
            await repository.close()
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_a_batch_reads_its_owners_usage_once(monkeypatch) -> None:
    service, _ = await _service()
    await _cap(service, "alice", 64)
    calls: list[str] = []
    original = StateSession.list_workloads_created_by

    async def counting(self, created_by, *, states=None):
        calls.append(created_by)
        return await original(self, created_by, states=states)

    monkeypatch.setattr(StateSession, "list_workloads_created_by", counting)
    _, children = await service.submit_batch(
        ALICE, org_id="acme", project_id="research", template=_job(4), count=5, idempotency_key="b"
    )
    assert len(children) == 5
    assert calls == ["alice"]
    # The running total still binds within the transaction: 5 x 4 + 9 x 4 > 64.
    calls.clear()
    with pytest.raises(ConflictError, match="you have 64 of 64"):
        await service.submit_batch(
            ALICE, org_id="acme", project_id="research", template=_job(4), count=12, idempotency_key="b2"
        )
    assert calls == ["alice"]


# ------------------------------------------------- 5. time-limited admin grant


@pytest.mark.asyncio
async def test_admin_tier_replaces_a_time_limited_fleet_admin_grant() -> None:
    service, repository = await _service()
    now = utcnow()
    async with repository.transaction() as tx:
        await tx.put_role_grant(
            RoleGrantRecord(
                subject="alice",
                grant=RoleGrant(role=Role.FLEET_ADMIN),
                granted_by="root",
                granted_at=now,
                expires_at=now + timedelta(hours=1),
            )
        )
    await service.set_user_tier(ADMIN, "alice", "admin")
    async with repository.transaction() as tx:
        [grant] = [item for item in await tx.list_role_grants(subject="alice") if item.grant.role is Role.FLEET_ADMIN]
    assert grant.expires_at is None


# --------------------------------------------------- 6. old-client rules body


def test_an_omitted_rule_keeps_its_stored_value_and_null_clears_it() -> None:
    http = _http()
    caps = {"max_tpu_units": 8, "max_gpu_units": 0, "max_cpu_units": 0.5}
    full = {"allowed_fleets": None, "approval_tpu_threshold": 4, "guaranteed_tpu_units": 2, **caps}
    assert http.put("/api/v1/team/users/alice/rules", json=full, headers=HTTP_ADMIN).status_code == 200
    # An older client knows three rules and sends only those.
    old_client = {"allowed_fleets": None, "approval_tpu_threshold": 6, "guaranteed_tpu_units": 2}
    response = http.put("/api/v1/team/users/alice/rules", json=old_client, headers=HTTP_ADMIN)
    assert response.status_code == 200, response.text
    assert response.json()["rules"] == {**old_client, **caps}
    clear = {"max_cpu_units": None, "max_tpu_units": None}
    cleared = http.put("/api/v1/team/users/alice/rules", json=clear, headers=HTTP_ADMIN)
    assert cleared.json()["rules"] == {**old_client, **caps, **clear}
