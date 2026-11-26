# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Team tiers (admin / trusted / user), the hard per-user TPU cap, and the team callable rule.

Observable outcomes only: persisted users, grants, workloads and audit events,
HTTP status codes and bodies, and the requests the CLI sends.
"""

from __future__ import annotations

import hashlib
import json
import typing as tp

import pytest
from click.testing import CliRunner

import eyvan.sdk.cli as cli
from eyvan.artifacts.managed import callable_uri
from eyvan.records import (
    CallableExecution,
    CommandExecution,
    InviteRecord,
    Principal,
    ResourceGang,
    ResourceVector,
    Role,
    RoleGrant,
    RoleGrantRecord,
    TeamTier,
    UserRecord,
    WorkloadDefinition,
    WorkloadState,
    utcnow,
)
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.callable_trust import CallableProjectTrustPolicy
from eyvan.server.service.errors import AuthorizationError, ConflictError, InputError
from eyvan.server.service.scheduling import CoreSchedulingService

ADMIN = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
ALICE = Principal(subject="alice", grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),))
ALICE_OTHER = Principal(
    subject="alice", grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="other"),)
)
BOB = Principal(subject="bob", grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),))
_DIGEST = hashlib.sha256(b"callable-payload").hexdigest()


def _job(tpu: int = 0, name: str = "job") -> WorkloadDefinition:
    return WorkloadDefinition(
        name=name,
        execution=CommandExecution(command="python train.py"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, memory_gib=1, tpu=tpu)),
    )


def _callable(tpu: int = 0) -> WorkloadDefinition:
    return WorkloadDefinition(
        name="fn",
        execution=CallableExecution(kind="callable", artifact_uri=callable_uri(_DIGEST), sha256=_DIGEST, python="3.11"),
        gang=ResourceGang(replicas=1, per_replica=ResourceVector(cpu=1, tpu=tpu)),
    )


async def _service(
    *, enabled: bool = True, trust: CallableProjectTrustPolicy | None = None
) -> tuple[ControlPlaneService, InMemoryRepository]:
    repository = InMemoryRepository()
    service = ControlPlaneService(
        repository,
        scheduler=CoreSchedulingService(),
        governance_enabled=enabled,
        callable_trust=trust or CallableProjectTrustPolicy(),
    )
    async with repository.transaction() as tx:
        await tx.put_user(UserRecord(subject="root", email="root@example.com", created_by="eyvan-bootstrap"))
        await tx.put_user(UserRecord(subject="alice", email="alice@example.com", created_by="root"))
        await tx.put_role_grant(RoleGrantRecord(subject="root", grant=RoleGrant(role=Role.FLEET_ADMIN), granted_by="x"))
    return service, repository


async def _grants(repository, subject: str) -> set[tuple[str, str, str]]:
    async with repository.transaction() as tx:
        return {
            (item.grant.role.value, item.grant.org_id or "", item.grant.project_id or "")
            for item in await tx.list_role_grants(subject=subject)
        }


async def _events(repository, action: str) -> list[dict[str, tp.Any]]:
    async with repository.transaction() as tx:
        events = await tx.list_events(limit=10_000)
    return [dict(event.payload) for event in events if event.action == action]


async def _cap(service: ControlPlaneService, subject: str, cap: int | None, threshold: int | None = None) -> None:
    await service.set_user_rules(
        ADMIN,
        subject,
        allowed_fleets=None,
        approval_tpu_threshold=threshold,
        guaranteed_tpu_units=None,
        max_tpu_units=cap,
    )


# ----------------------------------------------------------------- records


def test_new_fields_are_dropped_from_the_wire_at_their_defaults() -> None:
    user = UserRecord(subject="alice", email="alice@example.com", created_by="root")
    document = json.loads(user.model_dump_json())
    assert "trusted" not in document and "max_tpu_units" not in document
    flagged = user.model_copy(update={"trusted": True, "max_tpu_units": 8})
    stored = json.loads(flagged.model_dump_json())
    assert stored["trusted"] is True and stored["max_tpu_units"] == 8
    assert UserRecord.model_validate(stored) == flagged
    with pytest.raises(ValueError):
        UserRecord(subject="a", email="a@b.co", created_by="r", max_tpu_units=-1)
    invite = InviteRecord(
        id="inv_0123456789abcdef",
        email="a@b.co",
        subject="a",
        token_hash="0" * 64,
        created_by="root",
        expires_at=utcnow(),
    )
    assert "trusted" not in json.loads(invite.model_dump_json())
    assert json.loads(invite.model_copy(update={"trusted": True}).model_dump_json())["trusted"] is True


# --------------------------------------------------------------------- tier


@pytest.mark.asyncio
async def test_set_user_tier_moves_grants_and_flag_together_and_audits() -> None:
    service, repository = await _service()
    async with repository.transaction() as tx:
        await tx.put_role_grant(
            RoleGrantRecord(
                subject="alice",
                grant=RoleGrant(role=Role.PROJECT_VIEWER, org_id="acme", project_id="lab"),
                granted_by="root",
            )
        )
    trusted = await service.set_user_tier(ADMIN, "alice", "trusted", org_id="acme", project_id="research")
    assert trusted.trusted
    assert await _grants(repository, "alice") == {
        ("project_operator", "acme", "research"),
        ("project_viewer", "acme", "lab"),  # untouched
    }
    assert (await service.team_tiers(("alice",)))["alice"] is TeamTier.TRUSTED

    user = await service.set_user_tier(ADMIN, "alice", TeamTier.USER, org_id="acme", project_id="research")
    assert not user.trusted
    assert await _grants(repository, "alice") == {
        ("project_submitter", "acme", "research"),
        ("project_viewer", "acme", "lab"),
    }

    admin = await service.set_user_tier(ADMIN, "alice", "admin")
    assert not admin.trusted and ("fleet_admin", "", "") in await _grants(repository, "alice")
    assert (await service.team_tiers())["alice"] is TeamTier.ADMIN

    events = await _events(repository, "user.tier_updated")
    assert [(item["old_tier"], item["new_tier"]) for item in events] == [
        ("user", "trusted"),
        ("trusted", "user"),
        ("user", "admin"),
    ]
    assert events[0]["org_id"] == "acme" and events[0]["project_id"] == "research"
    assert all("password" not in json.dumps(item) for item in events)


@pytest.mark.asyncio
async def test_set_user_tier_validates_and_is_admin_only() -> None:
    service, _ = await _service()
    with pytest.raises(AuthorizationError):
        await service.set_user_tier(ALICE, "alice", "admin")
    with pytest.raises(InputError, match="org_id and a project_id"):
        await service.set_user_tier(ADMIN, "alice", "trusted")
    with pytest.raises(InputError, match="tier must be"):
        await service.set_user_tier(ADMIN, "alice", "owner", org_id="acme", project_id="research")


@pytest.mark.asyncio
async def test_the_last_enabled_fleet_admin_cannot_be_demoted() -> None:
    service, repository = await _service()
    with pytest.raises(ConflictError, match="last enabled fleet admin"):
        await service.set_user_tier(ADMIN, "root", "user", org_id="acme", project_id="research")
    # Rolled back: root still holds fleet_admin.
    assert ("fleet_admin", "", "") in await _grants(repository, "root")
    # A disabled second admin does not count.
    await service.set_user_tier(ADMIN, "alice", "admin")
    await service.set_user_disabled(ADMIN, "alice", disabled=True)
    with pytest.raises(ConflictError, match="last enabled fleet admin"):
        await service.set_user_tier(ADMIN, "root", "trusted", org_id="acme", project_id="research")
    await service.set_user_disabled(ADMIN, "alice", disabled=False)
    demoted = await service.set_user_tier(ADMIN, "root", "trusted", org_id="acme", project_id="research")
    assert demoted.trusted and ("fleet_admin", "", "") not in await _grants(repository, "root")


@pytest.mark.asyncio
async def test_a_trusted_invite_produces_a_trusted_account() -> None:
    service, _ = await _service()
    invite, raw = await service.invite_user(
        ADMIN,
        email="carol@example.com",
        subject="carol",
        grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="research"),),
        trusted=True,
    )
    assert invite.trusted
    user, _, _ = await service.accept_invite(
        raw,
        password="correct horse battery",  # pragma: allowlist secret
        session_ttl_s=60,
    )
    assert user.trusted
    assert (await service.team_tiers(("carol",)))["carol"] is TeamTier.TRUSTED
    with pytest.raises(AuthorizationError):
        await service.invite_user(ALICE, email="d@example.com", subject="d", trusted=True)


# ---------------------------------------------------------------- hard cap


@pytest.mark.asyncio
async def test_cap_refuses_n_plus_one_across_projects_counting_pending_and_held() -> None:
    service, repository = await _service()
    await _cap(service, "alice", 8, threshold=4)
    held, _ = await service.submit_workload(
        ALICE, org_id="acme", project_id="research", spec=_job(6), idempotency_key="held"
    )
    assert not held.schedulable  # held for approval, and still counted
    small, _ = await service.submit_workload(
        ALICE_OTHER, org_id="acme", project_id="other", spec=_job(2), idempotency_key="small"
    )
    assert small.schedulable
    with pytest.raises(ConflictError) as refused:
        await service.submit_workload(ALICE, org_id="acme", project_id="research", spec=_job(1), idempotency_key="x")
    assert str(refused.value) == "TPU quota: you have 8 of 8 TPU units in use or queued; this job asks for 1"
    # A job asking for no TPU is never refused by the cap.
    cpu, _ = await service.submit_workload(
        ALICE, org_id="acme", project_id="research", spec=_job(0), idempotency_key="cpu"
    )
    assert cpu.state is WorkloadState.PENDING
    # Cancelling frees the units.
    await service.cancel_workload(ALICE_OTHER, small.id, idempotency_key="c")
    fits, _ = await service.submit_workload(
        ALICE, org_id="acme", project_id="research", spec=_job(2), idempotency_key="fits"
    )
    assert fits.schedulable
    async with repository.transaction() as tx:
        stored = [item for item in await tx.list_workloads() if item.created_by == "alice"]
    assert sorted(item.id for item in stored) == sorted([held.id, small.id, cpu.id, fits.id])


@pytest.mark.asyncio
async def test_a_batch_is_capped_on_its_tasks_and_rolls_back_whole() -> None:
    service, repository = await _service()
    await _cap(service, "alice", 8)
    with pytest.raises(ConflictError, match="TPU quota"):
        await service.submit_batch(
            ALICE, org_id="acme", project_id="research", template=_job(4), count=3, idempotency_key="b"
        )
    async with repository.transaction() as tx:
        assert await tx.list_workloads() == []
    parent, children = await service.submit_batch(
        ALICE, org_id="acme", project_id="research", template=_job(4), count=2, idempotency_key="b2"
    )
    assert len(children) == 2 and parent.coordination_kind is not None


@pytest.mark.asyncio
async def test_approval_release_rechecks_the_cap() -> None:
    service, repository = await _service()
    await _cap(service, "alice", 8, threshold=4)
    held, _ = await service.submit_workload(
        ALICE, org_id="acme", project_id="research", spec=_job(6), idempotency_key="held"
    )
    await service.submit_workload(ALICE, org_id="acme", project_id="research", spec=_job(2), idempotency_key="b")
    # The admin lowers the cap while the job waits; releasing it now would exceed it.
    await _cap(service, "alice", 6, threshold=4)
    [request] = await service.list_team_requests(ADMIN, state="open")
    with pytest.raises(ConflictError, match=rf"cannot release {held.id} for alice: TPU quota: you have 2 of 6"):
        await service.resolve_team_request(ADMIN, request.id, resolution="approved")
    async with repository.transaction() as tx:
        assert not (await tx.get_workload(held.id)).schedulable
    [still_open] = await service.list_team_requests(ADMIN, state="open")
    assert still_open.id == request.id
    await _cap(service, "alice", 8, threshold=4)
    await service.resolve_team_request(ADMIN, request.id, resolution="approved")
    async with repository.transaction() as tx:
        assert (await tx.get_workload(held.id)).schedulable


@pytest.mark.asyncio
async def test_trusted_and_admin_are_not_held_but_are_still_capped() -> None:
    service, _ = await _service()
    await _cap(service, "alice", 8, threshold=0)
    await service.set_user_tier(ADMIN, "alice", "trusted", org_id="acme", project_id="research")
    ran, _ = await service.submit_workload(
        ALICE, org_id="acme", project_id="research", spec=_job(6), idempotency_key="a"
    )
    assert ran.schedulable
    assert await service.list_team_requests(ADMIN) == []
    with pytest.raises(ConflictError, match="TPU quota"):
        await service.submit_workload(ALICE, org_id="acme", project_id="research", spec=_job(4), idempotency_key="b")
    await service.set_user_tier(ADMIN, "alice", "admin")
    ok, _ = await service.submit_workload(ALICE, org_id="acme", project_id="research", spec=_job(2), idempotency_key="c")
    assert ok.schedulable
    with pytest.raises(ConflictError, match="TPU quota"):
        await service.submit_workload(ALICE, org_id="acme", project_id="research", spec=_job(1), idempotency_key="d")
    # A plain user with the same rules is held.
    await service.set_user_tier(ADMIN, "alice", "user", org_id="acme", project_id="research")
    await _cap(service, "alice", None, threshold=0)
    held, _ = await service.submit_workload(
        ALICE, org_id="acme", project_id="research", spec=_job(1), idempotency_key="e"
    )
    assert not held.schedulable


@pytest.mark.asyncio
async def test_the_cap_is_ignored_without_team_governance() -> None:
    service, _ = await _service(enabled=False)
    await _cap(service, "alice", 1)
    ok, _ = await service.submit_workload(
        ALICE, org_id="acme", project_id="research", spec=_job(64), idempotency_key="k"
    )
    assert ok.schedulable
    assert await service.team_tpu_cap(ALICE) is None


@pytest.mark.asyncio
async def test_rules_audit_includes_the_cap() -> None:
    service, repository = await _service()
    await _cap(service, "alice", 16)
    [payload] = await _events(repository, "user.rules_updated")
    assert payload["max_tpu_units"] == 16
    report = await service.team_tpu_cap(ALICE)
    assert report == {"tier": "user", "max_tpu_units": 16, "tpu_units_in_use_or_queued": 0}


# --------------------------------------------------------------- callables


@pytest.mark.asyncio
async def test_team_callable_rule_by_tier() -> None:
    service, _ = await _service()
    with pytest.raises(AuthorizationError, match="CPU quota required"):
        await service.submit_workload(
            ALICE, org_id="acme", project_id="research", spec=_callable(), idempotency_key="u0"
        )
    await _cap(service, "alice", 4)
    with pytest.raises(AuthorizationError, match="CPU quota required"):
        await service.submit_workload(
            ALICE, org_id="acme", project_id="research", spec=_callable(4), idempotency_key="u1"
        )
    await service.set_user_rules(ADMIN, "alice", max_cpu_units=2)
    ok, _ = await service.submit_workload(
        ALICE, org_id="acme", project_id="research", spec=_callable(4), idempotency_key="u1"
    )
    assert ok.state is WorkloadState.PENDING
    with pytest.raises(ConflictError, match="TPU quota"):
        await service.submit_workload(
            ALICE, org_id="acme", project_id="research", spec=_callable(1), idempotency_key="u2"
        )
    await _cap(service, "alice", None)
    await service.set_user_rules(ADMIN, "alice", max_cpu_units=None)
    await service.set_user_tier(ADMIN, "alice", "trusted", org_id="acme", project_id="research")
    trusted, _ = await service.submit_workload(
        ALICE, org_id="acme", project_id="research", spec=_callable(), idempotency_key="t"
    )
    assert trusted.state is WorkloadState.PENDING
    # Batch is gated the same way (every route stores through one path).
    await service.set_user_tier(ADMIN, "alice", "user", org_id="acme", project_id="research")
    with pytest.raises(AuthorizationError, match="CPU quota required"):
        await service.submit_batch(
            ALICE, org_id="acme", project_id="research", template=_callable(), count=2, idempotency_key="b"
        )


@pytest.mark.asyncio
async def test_principals_without_a_team_account_keep_the_project_rule() -> None:
    service, _ = await _service()
    with pytest.raises(AuthorizationError, match="not trusted for project acme/research"):
        await service.submit_workload(BOB, org_id="acme", project_id="research", spec=_callable(), idempotency_key="b")
    trusted_service, _ = await _service(trust=CallableProjectTrustPolicy.from_projects(["acme/research"]))
    ok, _ = await trusted_service.submit_workload(
        BOB, org_id="acme", project_id="research", spec=_callable(), idempotency_key="b"
    )
    assert ok.state is WorkloadState.PENDING
    # Project trust is a fallback for non-team principals, not a team cap exemption.
    with pytest.raises(AuthorizationError, match="CPU quota required"):
        await trusted_service.submit_workload(
            ALICE, org_id="acme", project_id="research", spec=_callable(), idempotency_key="a"
        )
    with pytest.raises(AuthorizationError, match="not trusted"):
        await service.authorize_callable_upload(BOB, org_id="acme", project_id="research")
    with pytest.raises(AuthorizationError, match="positive quota"):
        await service.authorize_callable_upload(ALICE, org_id="acme", project_id="research")


@pytest.mark.asyncio
async def test_non_team_planes_keep_the_project_rule_unchanged() -> None:
    service, _ = await _service(enabled=False)
    await service.set_user_tier(ADMIN, "alice", "trusted", org_id="acme", project_id="research")
    with pytest.raises(AuthorizationError, match="not trusted"):
        await service.submit_workload(ALICE, org_id="acme", project_id="research", spec=_callable(), idempotency_key="a")
    with pytest.raises(AuthorizationError, match="not trusted"):
        service.authorize_callable_artifact(ALICE, org_id="acme", project_id="research")
    with pytest.raises(AuthorizationError, match="not trusted"):
        await service.authorize_callable_upload(ALICE, org_id="acme", project_id="research")


# -------------------------------------------------------------------- HTTP


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


def test_tier_route_reports_tier_in_users_and_me() -> None:
    http = _http()
    me = http.get("/api/v1/team/me", headers=ALICE_HEADERS)
    assert me.status_code == 200 and me.json()["tier"] == "user"
    assert me.json()["user"]["tier"] == "user" and me.json()["user"]["trusted"] is False

    body = {"tier": "trusted", "org_id": "acme", "project_id": "research"}
    assert http.put("/api/v1/team/users/alice/tier", json=body, headers=ALICE_HEADERS).status_code == 403
    response = http.put("/api/v1/team/users/alice/tier", json=body, headers=ADMIN_HEADERS)
    assert response.status_code == 200, response.text
    assert response.json()["tier"] == "trusted" and response.json()["trusted"] is True
    listed = http.get("/api/v1/team/users", headers=ADMIN_HEADERS).json()["users"]
    assert [(row["tier"], row["trusted"]) for row in listed if row["subject"] == "alice"] == [("trusted", True)]
    assert http.get("/api/v1/team/me", headers=ALICE_HEADERS).json()["tier"] == "trusted"

    admin = http.put("/api/v1/team/users/alice/tier", json={"tier": "admin"}, headers=ADMIN_HEADERS)
    assert admin.status_code == 200 and admin.json()["tier"] == "admin"
    missing_scope = http.put("/api/v1/team/users/alice/tier", json={"tier": "user"}, headers=ADMIN_HEADERS)
    assert missing_scope.status_code == 400, missing_scope.text
    bad = http.put("/api/v1/team/users/alice/tier", json={"tier": "owner"}, headers=ADMIN_HEADERS)
    assert bad.status_code == 422


def test_rules_route_sets_the_cap() -> None:
    http = _http()
    body = {"allowed_fleets": None, "approval_tpu_threshold": None, "guaranteed_tpu_units": None, "max_tpu_units": 8}
    response = http.put("/api/v1/team/users/alice/rules", json=body, headers=ADMIN_HEADERS)
    assert response.status_code == 200, response.text
    assert response.json()["rules"]["max_tpu_units"] == 8
    assert (
        http.put("/api/v1/team/users/alice/rules", json={"max_tpu_units": -1}, headers=ADMIN_HEADERS).status_code == 422
    )


def test_invite_with_tier() -> None:
    http = _http()
    both = http.post(
        "/api/v1/team/invites",
        json={"email": "c@example.com", "subject": "c", "tier": "user", "role": "project_viewer", "org_id": "acme"},
        headers=ADMIN_HEADERS,
    )
    assert both.status_code == 400 and "not both" in both.text
    unscoped = http.post(
        "/api/v1/team/invites", json={"email": "c@example.com", "subject": "c", "tier": "user"}, headers=ADMIN_HEADERS
    )
    assert unscoped.status_code == 400
    trusted = http.post(
        "/api/v1/team/invites",
        json={"email": "c@example.com", "subject": "c", "tier": "trusted", "org_id": "acme", "project_id": "r"},
        headers=ADMIN_HEADERS,
    )
    assert trusted.status_code == 201, trusted.text
    assert trusted.json()["invite"]["roles"] == [{"role": "project_operator", "org_id": "acme", "project_id": "r"}]
    accepted = http.post(
        "/api/v1/team/invites/accept", json={"token": trusted.json()["token"], "password": "correct horse battery"}
    )
    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["user"]["tier"] == "trusted"
    # Accepting signed the client in as carol (the cookie wins over dev headers); act as the admin again.
    http.cookies.clear()
    admin = http.post(
        "/api/v1/team/invites", json={"email": "d@example.com", "subject": "d", "tier": "admin"}, headers=ADMIN_HEADERS
    )
    assert admin.status_code == 201 and admin.json()["invite"]["roles"] == [
        {"role": "fleet_admin", "org_id": None, "project_id": None}
    ]


def test_tier_route_is_404_without_team_mode() -> None:
    http = _http(team_mode=False)
    assert http.put("/api/v1/team/users/alice/tier", json={"tier": "admin"}, headers=ADMIN_HEADERS).status_code == 404


# --------------------------------------------------------------------- CLI


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, tp.Any] | None]] = []

    def request(self, method: str, path: str, *, params=None, json=None, headers=None) -> tp.Any:
        del params, headers
        self.calls.append((method, path, json))
        if path.endswith("/tier"):
            return {"subject": "alice", "tier": (json or {}).get("tier")}
        if path.endswith("/rules"):
            return {"rules": json}
        if path == "/api/v1/team/invites":
            return {"url": "https://plane/#/accept-invite/x"}
        return {}

    def replace_identity_headers(self, headers: tp.Mapping[str, str]) -> None:
        del headers


def _cli(monkeypatch, argv: list[str]) -> tuple[tp.Any, _Recorder]:
    fake = _Recorder()
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: fake)
    result = CliRunner().invoke(cli.main, ["--user", "root", "--org", "acme", "--project", "research", *argv])
    return result, fake


@pytest.mark.parametrize(("allowed", "label"), [(None, "any"), ([], "none"), (["research"], "research")])
def test_cli_rules_fleet_label(monkeypatch, allowed, label) -> None:
    monkeypatch.setattr(_Recorder, "request", lambda *args, **kwargs: {"rules": {"allowed_fleets": allowed}})
    result, _ = _cli(monkeypatch, ["users", "rules", "alice"])
    assert result.exit_code == 0, result.output
    assert f"fleets {label};" in result.output


def test_cli_users_tier(monkeypatch) -> None:
    result, fake = _cli(monkeypatch, ["users", "tier", "alice", "trusted"])
    assert result.exit_code == 0, result.output
    assert fake.calls == [
        ("PUT", "/api/v1/team/users/alice/tier", {"tier": "trusted", "org_id": "acme", "project_id": "research"})
    ]
    assert "alice is now trusted in acme/research" in result.output
    result, fake = _cli(monkeypatch, ["users", "tier", "alice", "admin"])
    assert result.exit_code == 0 and fake.calls[0][2] == {"tier": "admin"}


def test_cli_users_invite_tier_and_rules_max_tpu(monkeypatch) -> None:
    result, fake = _cli(monkeypatch, ["users", "invite", "bob@example.com", "--tier", "user"])
    assert result.exit_code == 0, result.output
    assert fake.calls[0][2] == {
        "email": "bob@example.com",
        "subject": "bob",
        "tier": "user",
        "org_id": "acme",
        "project_id": "research",
    }
    both, _ = _cli(monkeypatch, ["users", "invite", "b@example.com", "--tier", "user", "--role", "project_viewer"])
    assert both.exit_code != 0 and "not both" in both.output
    rules, fake = _cli(monkeypatch, ["users", "rules", "alice", "--max-tpu", "8", "--max-gpu", "0", "--max-cpu", "0.5"])
    assert rules.exit_code == 0, rules.output
    assert fake.calls[0][2]["max_tpu_units"] == 8 and "max TPU 8" in rules.output
    assert fake.calls[0][2]["max_gpu_units"] == 0 and "max GPU 0" in rules.output
    assert fake.calls[0][2]["max_cpu_units"] == 0.5 and "max CPU 0.5" in rules.output
