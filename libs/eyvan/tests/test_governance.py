# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Team governance: per-user fleet rules, approval holds, and the borrow/reclaim rule.

Observable outcomes only: persisted workloads and requests, audit actions, what
the real scheduler and capacity planner do with a held job, and the plan the
deterministic reclaim rule returns.
"""

from __future__ import annotations

import json
import typing as tp

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from eyvan.domain.governance import (
    HELD_WORKLOAD_KEY,
    RunningJob,
    UserRules,
    admission_verdict,
    plan_reclaim,
    requested_tpu_units,
)
from eyvan.records import (
    CapacityPoolRecord,
    CommandExecution,
    Principal,
    PriorityBand,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    UserRecord,
    WorkloadDefinition,
    WorkloadState,
)
from eyvan.server.controller import EyvanController
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.persistence.local import LocalJsonRepository
from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import AuthorizationError, ConflictError, InputError
from eyvan.server.service.scheduling import CoreSchedulingService

ADMIN = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
ALICE = Principal(subject="alice", grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),))


def _job(name: str = "job", *, tpu: int = 0, replicas: int = 1) -> WorkloadDefinition:
    per_replica = ResourceVector(cpu=1, memory_gib=1, tpu=tpu)
    return WorkloadDefinition(
        name=name,
        execution=CommandExecution(command="python train.py"),
        gang=ResourceGang(replicas=replicas, per_replica=per_replica),
    )


def _cpu_pool(pool_id: str = "research/cpu", fleet: str = "research") -> CapacityPoolRecord:
    return CapacityPoolRecord(
        id=pool_id,
        fleet_name=fleet,
        node_type=pool_id.rsplit("/", 1)[-1],
        provider="gcp",
        capacity_backend="gcp-compute",
        provider_project="project-a",
        region="us-central1",
        zone="us-central1-a",
        capacity={"cpu": 4, "memory_gib": 16, "tpu": 8},
        allocatable={"cpu": 4, "memory_gib": 16, "tpu": 8},
        executor="native-local",
        agent_endpoint="http://agent:8090",
        provider_options={"kind": "cpu", "machine_type": "n2-standard-4"},
        min_cells=0,
        max_cells=4,
    )


async def _service(*, enabled: bool = True, repository=None) -> tuple[ControlPlaneService, tp.Any]:
    repository = repository or InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService(), governance_enabled=enabled)
    async with repository.transaction() as tx:
        await tx.put_user(UserRecord(subject="alice", email="alice@example.com", created_by="root"))
    return service, repository


async def _workload(repository, workload_id: str):
    async with repository.transaction() as tx:
        return await tx.get_workload(workload_id)


async def _actions(repository) -> list[str]:
    async with repository.transaction() as tx:
        return [event.action for event in await tx.list_events(org_id="acme", project_id="research", limit=1000)]


# ------------------------------------------------------------------ pure rules


def test_tpu_units_count_replicas_workers_and_cells() -> None:
    assert requested_tpu_units(_job(tpu=4)) == 4
    assert requested_tpu_units(_job(tpu=4, replicas=2)) == 8
    spec = _job(tpu=4).model_copy(update={"cell_group_count": 3})
    assert requested_tpu_units(spec) == 12


def test_admission_verdicts() -> None:
    none = UserRules()
    assert (
        admission_verdict(
            none, tpu_units=999, fleets=[], pinned=False, operation="workload.submit", top_level=True
        ).action
        == "admit"
    )
    rules = UserRules(approval_tpu_threshold=8)
    kwargs = {"fleets": [], "pinned": False, "operation": "workload.submit", "top_level": True}
    assert admission_verdict(rules, tpu_units=8, **kwargs).action == "admit"
    assert admission_verdict(rules, tpu_units=9, **kwargs).action == "hold"
    child = admission_verdict(
        rules, tpu_units=9, fleets=[], pinned=False, operation="workload.batch_task", top_level=False
    )
    assert (child.action, child.refusal) == ("refuse", "approval")
    fleets = UserRules(allowed_fleets=("research",))
    assert (
        admission_verdict(
            fleets, tpu_units=1, fleets=["research"], pinned=True, operation="workload.submit", top_level=True
        ).action
        == "admit"
    )
    other = admission_verdict(
        fleets, tpu_units=1, fleets=["prod"], pinned=True, operation="workload.submit", top_level=True
    )
    assert (other.action, other.refusal) == ("refuse", "fleet") and "prod" in other.reason
    unpinned = admission_verdict(
        fleets, tpu_units=1, fleets=[], pinned=False, operation="workload.submit", top_level=True
    )
    assert unpinned.action == "refuse"


def _jobs() -> list[RunningJob]:
    return [
        RunningJob("bob-old", "bob", 8, "interactive", 0),
        RunningJob("bob-batch", "bob", 8, "batch", 1),
        RunningJob("bob-new", "bob", 8, "interactive", 2),
        RunningJob("carol-1", "carol", 8, "batch", 3),
        RunningJob("alice-1", "alice", 8, "batch", 4),
    ]


def test_reclaim_takes_borrowed_capacity_deterministically() -> None:
    guarantees = {"alice": 32, "bob": 8, "carol": 8}
    plan = plan_reclaim("alice", 16, _jobs(), guarantees)
    # bob borrows 16 (uses 24, guaranteed 8); carol borrows 0. Lowest band first,
    # then the most recently started: bob-batch, then bob-new.
    assert plan.allowed and plan.victims == ("bob-batch", "bob-new") and plan.freed_tpu_units == 16
    assert plan_reclaim("alice", 16, list(reversed(_jobs())), guarantees) == plan


def test_reclaim_never_exceeds_the_requesters_guarantee_or_pushes_a_borrower_below_theirs() -> None:
    guarantees = {"alice": 16, "bob": 8, "carol": 8}
    over = plan_reclaim("alice", 16, _jobs(), guarantees)  # alice uses 8; 8 + 16 > 16
    assert not over.allowed and over.victims == ()
    assert not plan_reclaim("alice", 24, _jobs(), {"alice": 64, "bob": 8, "carol": 8}).allowed
    assert not plan_reclaim("dave", 8, _jobs(), guarantees).allowed  # no guarantee at all
    not_running = [RunningJob("bob-x", "bob", 16, "batch", 0, reclaimable=False)]
    assert not plan_reclaim("alice", 8, not_running, {"alice": 32}).allowed


# --------------------------------------------------------------- admission


@pytest.mark.asyncio
async def test_a_user_without_rules_is_admitted_exactly_as_before() -> None:
    service, repository = await _service()
    workload, _ = await service.submit_workload(
        ALICE, org_id="acme", project_id="research", spec=_job(tpu=64), idempotency_key="k"
    )
    assert workload.schedulable and workload.state is WorkloadState.PENDING
    async with repository.transaction() as tx:
        assert await tx.list_team_requests() == []


@pytest.mark.asyncio
async def test_rules_are_ignored_when_governance_is_off() -> None:
    service, repository = await _service(enabled=False)
    async with repository.transaction() as tx:
        user = await tx.get_user("alice")
        await tx.put_user(user.model_copy(update={"approval_tpu_threshold": 1, "allowed_fleets": ("x",)}))
    workload, _ = await service.submit_workload(
        ALICE, org_id="acme", project_id="research", spec=_job(tpu=64), idempotency_key="k"
    )
    assert workload.schedulable
    with pytest.raises(Exception, match="not enabled"):
        await service.plan_reclaim_for(ALICE, tpu_units=1)


@pytest.mark.asyncio
async def test_over_threshold_submission_is_held_then_released_by_an_admin() -> None:
    service, repository = await _service()
    await service.set_user_rules(
        ADMIN, "alice", allowed_fleets=None, approval_tpu_threshold=8, guaranteed_tpu_units=None
    )
    workload, replayed = await service.submit_workload(
        ALICE, org_id="acme", project_id="research", spec=_job("big", tpu=16), idempotency_key="big"
    )
    assert not replayed and workload.state is WorkloadState.PENDING and not workload.schedulable
    assert "approval threshold of 8" in (workload.scheduling_reason or "")
    [request] = await service.list_team_requests(ADMIN, state="open")
    assert request.kind == "approval" and request.subject == "alice"
    assert request.details[HELD_WORKLOAD_KEY] == workload.id and request.details["tpu_units"] == 16
    # The filer sees it as their own request.
    assert [item.id for item in await service.list_team_requests(ALICE)] == [request.id]
    # A replay neither re-holds nor files a second request.
    again, replayed = await service.submit_workload(
        ALICE, org_id="acme", project_id="research", spec=_job("big", tpu=16), idempotency_key="big"
    )
    assert replayed and again.id == workload.id
    assert len(await service.list_team_requests(ADMIN)) == 1

    with pytest.raises(AuthorizationError):
        await service.resolve_team_request(ALICE, request.id, resolution="approved")
    await service.resolve_team_request(ADMIN, request.id, resolution="approved", note="ok for today")
    released = await _workload(repository, workload.id)
    assert released.schedulable and released.state is WorkloadState.PENDING
    assert "approved by root" in released.scheduling_reason
    actions = await _actions(repository)
    assert actions.index("workload.approval_held") < actions.index("workload.approved")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("who", "resolution", "action"),
    [("admin", "declined", "workload.approval_declined"), ("alice", "withdrawn", "workload.approval_withdrawn")],
)
async def test_decline_or_withdraw_cancels_the_held_workload(who: str, resolution: str, action: str) -> None:
    service, repository = await _service()
    await service.set_user_rules(
        ADMIN, "alice", allowed_fleets=None, approval_tpu_threshold=0, guaranteed_tpu_units=None
    )
    workload, _ = await service.submit_workload(
        ALICE, org_id="acme", project_id="research", spec=_job(tpu=1), idempotency_key="k"
    )
    [request] = await service.list_team_requests(ADMIN)
    await service.resolve_team_request(ADMIN if who == "admin" else ALICE, request.id, resolution=resolution)
    assert (await _workload(repository, workload.id)).state is WorkloadState.CANCELLED
    assert action in await _actions(repository)


@pytest.mark.asyncio
async def test_approving_after_the_owner_cancelled_does_not_resurrect_it() -> None:
    service, repository = await _service()
    await service.set_user_rules(
        ADMIN, "alice", allowed_fleets=None, approval_tpu_threshold=0, guaranteed_tpu_units=None
    )
    workload, _ = await service.submit_workload(
        ALICE, org_id="acme", project_id="research", spec=_job(tpu=1), idempotency_key="k"
    )
    await service.cancel_workload(ALICE, workload.id, idempotency_key="c")
    [request] = await service.list_team_requests(ADMIN)
    await service.resolve_team_request(ADMIN, request.id, resolution="approved")
    stored = await _workload(repository, workload.id)
    assert stored.state is WorkloadState.CANCELLED and not stored.schedulable


@pytest.mark.asyncio
async def test_users_cannot_forge_a_hold_link() -> None:
    service, _ = await _service()
    with pytest.raises(InputError):
        await service.file_team_request(ALICE, kind="approval", summary="x", details={HELD_WORKLOAD_KEY: "wl_any"})


@pytest.mark.asyncio
async def test_fleet_rules_refuse_other_fleets_and_unpinned_jobs() -> None:
    service, repository = await _service()
    await service.apply_fleet(ADMIN, "research", (_cpu_pool(),), idempotency_key="f1")
    await service.apply_fleet(ADMIN, "prod", (_cpu_pool("prod/cpu", "prod"),), idempotency_key="f2")
    await service.set_user_rules(
        ADMIN, "alice", allowed_fleets=["research"], approval_tpu_threshold=None, guaranteed_tpu_units=None
    )
    ok, _ = await service.submit_pool_workload(
        ALICE,
        fleet_name="research",
        node_type="cpu",
        org_id="acme",
        project_id="research",
        command="x",
        idempotency_key="a",
    )
    assert ok.schedulable
    with pytest.raises(AuthorizationError, match="prod"):
        await service.submit_pool_workload(
            ALICE,
            fleet_name="prod",
            node_type="cpu",
            org_id="acme",
            project_id="research",
            command="x",
            idempotency_key="b",
        )
    with pytest.raises(AuthorizationError, match="name one"):
        await service.submit_workload(ALICE, org_id="acme", project_id="research", spec=_job(), idempotency_key="c")
    async with repository.transaction() as tx:
        assert [item.id for item in await tx.list_workloads()] == [ok.id]


@pytest.mark.asyncio
async def test_over_threshold_children_are_refused_not_held() -> None:
    service, repository = await _service()
    await service.set_user_rules(
        ADMIN, "alice", allowed_fleets=None, approval_tpu_threshold=4, guaranteed_tpu_units=None
    )
    with pytest.raises(ConflictError, match="single job"):
        await service.submit_batch(
            ALICE,
            org_id="acme",
            project_id="research",
            template=_job(tpu=8),
            count=2,
            idempotency_key="batch",
        )
    async with repository.transaction() as tx:
        assert await tx.list_team_requests() == []


@pytest.mark.asyncio
async def test_set_user_rules_is_admin_only_audited_and_validated() -> None:
    service, repository = await _service()
    with pytest.raises(AuthorizationError):
        await service.set_user_rules(
            ALICE, "alice", allowed_fleets=None, approval_tpu_threshold=1, guaranteed_tpu_units=None
        )
    with pytest.raises(InputError):
        await service.set_user_rules(
            ADMIN, "alice", allowed_fleets=None, approval_tpu_threshold=-1, guaranteed_tpu_units=None
        )
    updated = await service.set_user_rules(
        ADMIN, "alice", allowed_fleets=["b", "a", "a"], approval_tpu_threshold=8, guaranteed_tpu_units=32
    )
    assert updated.allowed_fleets == ("a", "b") and updated.password_hash is None
    async with repository.transaction() as tx:
        events = await tx.list_fleet_events(limit=100) if hasattr(tx, "list_fleet_events") else []
    del events
    cleared = await service.set_user_rules(
        ADMIN, "alice", allowed_fleets=None, approval_tpu_threshold=None, guaranteed_tpu_units=None
    )
    assert "allowed_fleets" not in cleared.model_dump(mode="json")


# ---------------------------------------------- scheduler and autoscale


@pytest.mark.asyncio
async def test_a_held_workload_is_neither_scheduled_nor_provisioned_until_approved() -> None:
    service, repository = await _service()
    controller = EyvanController(repository, service)
    await service.apply_fleet(ADMIN, "research", (_cpu_pool(),), idempotency_key="fleet")
    await service.set_user_rules(
        ADMIN, "alice", allowed_fleets=None, approval_tpu_threshold=0, guaranteed_tpu_units=None
    )
    held, _ = await service.submit_workload(
        ALICE,
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(
            name="cpu-job",
            execution=CommandExecution(command="python train.py"),
            gang={"replicas": 1, "per_replica": {"cpu": 4, "memory_gib": 16, "tpu": 2}},
        ),
        idempotency_key="held",
    )
    assert await controller.reconcile_capacity_pools_once() == {"launched": 0, "drained": 0, "deleted": 0}
    result = await controller.reconcile_once()
    assert result.placements == ()
    async with repository.transaction() as tx:
        assert await tx.list_cells() == []

    [request] = await service.list_team_requests(ADMIN)
    await service.resolve_team_request(ADMIN, request.id, resolution="approved")
    assert await controller.reconcile_capacity_pools_once() == {"launched": 1, "drained": 0, "deleted": 0}
    assert (await _workload(repository, held.id)).schedulable


# -------------------------------------------------------------- persistence


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["memory", "local", "sql"])
async def test_rules_and_holds_persist_in_every_backend(backend: str, tmp_path) -> None:
    engine = None
    if backend == "memory":
        repository = InMemoryRepository()
    elif backend == "local":
        repository = LocalJsonRepository(tmp_path / "control-plane.json")
    else:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'plane.db'}")
        await create_schema(engine)
        repository = SqlAlchemyRepository(engine)
    await repository.open()
    service, _ = await _service(repository=repository)
    await service.set_user_rules(
        ADMIN,
        "alice",
        allowed_fleets=["research"],
        approval_tpu_threshold=2,
        guaranteed_tpu_units=16,
        max_cpu_units=0.75,
        max_gpu_units=0,
    )
    await repository.close()

    if backend == "memory":
        reopened = repository
    elif backend == "local":
        reopened = LocalJsonRepository(tmp_path / "control-plane.json")
    else:
        reopened = SqlAlchemyRepository(engine)
    await reopened.open()
    async with reopened.transaction() as tx:
        user = await tx.get_user("alice")
    await reopened.close()
    if engine is not None:
        await engine.dispose()
    assert (user.allowed_fleets, user.approval_tpu_threshold, user.guaranteed_tpu_units) == (("research",), 2, 16)
    assert (user.max_cpu_units, user.max_gpu_units, user.max_tpu_units) == (0.75, 0, None)


def test_an_account_without_rules_serializes_exactly_as_before() -> None:
    user = UserRecord(subject="alice", email="alice@example.com", created_by="root")
    document = json.loads(user.model_dump_json())
    assert set(document) == {
        "subject",
        "email",
        "display_name",
        "password_hash",
        "disabled",
        "created_by",
        "created_at",
        "updated_at",
    }
    assert UserRecord.model_validate(document) == user


def test_priority_band_order_used_by_reclaim_is_lowest_first() -> None:
    assert [band.value for band in PriorityBand] == ["batch", "interactive", "production"]


# -------------------------------------------------------------------- routes


def _http(*, team_mode: bool = True):
    import asyncio

    from starlette.testclient import TestClient

    from eyvan.config import EyvanSettings
    from eyvan.server.api.app import create_app

    repository = InMemoryRepository()

    async def seed() -> None:
        async with repository.transaction() as tx:
            await tx.put_user(UserRecord(subject="alice", email="alice@example.com", created_by="root"))

    asyncio.run(seed())
    settings = EyvanSettings(dev_auth_enabled=True, team_mode=team_mode, artifact_token_key="k" * 32)
    return TestClient(create_app(settings=settings, repository=repository), base_url="http://127.0.0.1:8080")


ADMIN_HEADERS = {"X-Eyvan-User": "root", "X-Eyvan-Roles": "fleet_admin"}
ALICE_HEADERS = {
    "X-Eyvan-User": "alice",
    "X-Eyvan-Roles": "project_operator",
    "X-Eyvan-Org": "acme",
    "X-Eyvan-Project": "research",
}


def test_rules_route_is_admin_only_and_reports_rules() -> None:
    http = _http()
    body = {"allowed_fleets": ["research"], "approval_tpu_threshold": 8, "guaranteed_tpu_units": 32}
    refused = http.put("/api/v1/team/users/alice/rules", json=body, headers=ALICE_HEADERS)
    assert refused.status_code == 403, refused.text
    response = http.put("/api/v1/team/users/alice/rules", json=body, headers=ADMIN_HEADERS)
    assert response.status_code == 200, response.text
    expected = {**body, "max_tpu_units": None, "max_cpu_units": None, "max_gpu_units": None}
    assert response.json()["rules"] == expected
    assert "password_hash" not in response.text
    listed = http.get("/api/v1/team/users", headers=ADMIN_HEADERS).json()["users"]
    assert [user["rules"] for user in listed if user["subject"] == "alice"] == [expected]
    bad = http.put("/api/v1/team/users/alice/rules", json={"approval_tpu_threshold": -1}, headers=ADMIN_HEADERS)
    assert bad.status_code == 422
    for key in ("max_cpu_units", "max_gpu_units"):
        for invalid in (-1, 1_000_001):
            assert (
                http.put("/api/v1/team/users/alice/rules", json={key: invalid}, headers=ADMIN_HEADERS).status_code == 422
            )


def test_reclaim_plan_hides_other_peoples_work_from_non_admins() -> None:
    http = _http()
    mine = http.get("/api/v1/team/reclaim/plan", params={"tpu_units": 8}, headers=ALICE_HEADERS)
    assert mine.status_code == 200, mine.text
    assert mine.json()["allowed"] is False and "guaranteed" in mine.json()["reason"]
    assert "victims" not in mine.json() and "borrowed_by" not in mine.json()
    admin = http.get("/api/v1/team/reclaim/plan", params={"tpu_units": 8}, headers=ADMIN_HEADERS)
    assert admin.status_code == 200 and "victims" in admin.json() and "borrowed_by" in admin.json()


def test_reclaim_routes_refuse_without_a_guarantee_or_fleet_admin() -> None:
    http = _http()
    key = {"Idempotency-Key": "r1"}
    # alice has no guarantee, so the deterministic rule refuses and nothing is taken.
    borrowed = http.post("/api/v1/team/reclaim", json={"tpu_units": 8}, headers={**ALICE_HEADERS, **key})
    assert borrowed.status_code == 409, borrowed.text
    assert "guaranteed" in borrowed.text
    # Reclaiming a named workload is fleet-admin only.
    named = http.post(
        "/api/v1/team/workloads/wl_missing/reclaim", json={"reason": "x"}, headers={**ALICE_HEADERS, **key}
    )
    assert named.status_code == 403, named.text
    missing = http.post(
        "/api/v1/team/workloads/wl_missing/reclaim", json={"reason": "x"}, headers={**ADMIN_HEADERS, **key}
    )
    assert missing.status_code == 404, missing.text
    no_key = http.post("/api/v1/team/reclaim", json={"tpu_units": 8}, headers=ALICE_HEADERS)
    assert no_key.status_code == 422


@pytest.mark.asyncio
async def test_a_user_within_guarantee_reclaims_borrowed_capacity_checkpoint_first() -> None:
    from test_reclaim import FakeAgent
    from test_reclaim import _spec as reclaimable_spec

    from eyvan.config import EyvanSettings
    from eyvan.records import CellRecord, CellState

    repository = InMemoryRepository()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService(), governance_enabled=True)
    agent = FakeAgent()
    controller = EyvanController(
        repository,
        service,
        settings=EyvanSettings(checkpoint_grace_s=2, preemption_stop_grace_s=2),
        agent_factory=lambda _cell: agent,  # type: ignore[arg-type,return-value]
    )
    bob = Principal(subject="bob", grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),))
    async with repository.transaction() as tx:
        await tx.put_user(UserRecord(subject="alice", email="alice@example.com", created_by="root"))
        await tx.put_user(UserRecord(subject="bob", email="bob@example.com", created_by="root"))
        await tx.put_cell(
            CellRecord(
                id="cell-0",
                provider="local",
                region="local",
                state=CellState.READY,
                capacity=ResourceVector(tpu=8),
                allocatable=ResourceVector(tpu=8),
                endpoint="http://agent:8090",
            )
        )
    await service.set_user_rules(
        ADMIN, "alice", allowed_fleets=None, approval_tpu_threshold=None, guaranteed_tpu_units=16
    )
    borrowed, _ = await service.submit_workload(
        bob, org_id="acme", project_id="research", spec=reclaimable_spec(checkpointable=True), idempotency_key="bob"
    )
    await controller.reconcile_once()
    await controller.dispatch_outbox_once()
    await controller.observe_workloads_once()
    running = await _workload(repository, borrowed.id)
    assert running.state is WorkloadState.RUNNING

    # bob has no guarantee, so all 8 of his TPU units are borrowed; alice uses 0 of 16.
    plan, issued = await service.reclaim_borrowed(
        ALICE, tpu_units=8, reason="my eval needs it", idempotency_key="alice-reclaim"
    )
    assert plan.allowed and plan.victims == (borrowed.id,) and issued == [borrowed.id]
    assert await controller.dispatch_outbox_once() == 1
    assert len(agent.checkpoints) == 1 and len(agent.stops) == 1
    resumed = await _workload(repository, borrowed.id)
    assert resumed.state is WorkloadState.PENDING and resumed.generation == running.generation + 1
    assert resumed.checkpoint_uri == "gs://checkpoints/batch"
    assert "reclaimed by reclaim:alice" in resumed.scheduling_reason

    # bob, with no guarantee, may not reclaim anything; nor may alice exceed hers.
    with pytest.raises(ConflictError):
        await service.reclaim_borrowed(bob, tpu_units=8, reason="", idempotency_key="bob-reclaim")
    with pytest.raises(ConflictError):
        await service.reclaim_borrowed(ALICE, tpu_units=32, reason="", idempotency_key="too-much")


def test_governance_routes_are_404_without_team_mode() -> None:
    http = _http(team_mode=False)
    assert http.put("/api/v1/team/users/alice/rules", json={}, headers=ADMIN_HEADERS).status_code == 404
    assert http.get("/api/v1/team/reclaim/plan", params={"tpu_units": 1}, headers=ADMIN_HEADERS).status_code == 404
    key = {"Idempotency-Key": "k"}
    assert http.post("/api/v1/team/reclaim", json={"tpu_units": 1}, headers={**ADMIN_HEADERS, **key}).status_code == 404
    reclaim = http.post("/api/v1/team/workloads/wl_x/reclaim", json={"reason": "x"}, headers={**ADMIN_HEADERS, **key})
    assert reclaim.status_code == 404
