# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Team accounts: invites, password sign-in and personal API tokens.

Authority is never stored on the account: accepting an invite issues ordinary
durable role grants. Only hashes of secrets are persisted, and a raw secret is
returned once. Every flow runs against the in-memory and the local JSON
repository, and the local one is reopened from disk to prove durability.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from eyvan.identity.team_secrets import (
    SignInThrottle,
    hash_password,
    is_team_token,
    token_hash,
    verify_password,
)
from eyvan.records import Permission, Principal, Role, RoleGrant
from eyvan.server.persistence.local import LOCAL_CONTROL_PLANE_TEAM_SCHEMA_VERSION, LocalJsonRepository
from eyvan.server.persistence.memory import InMemoryRepository
from eyvan.server.service.control_plane import ControlPlaneService
from eyvan.server.service.errors import AuthorizationError, ConflictError, InputError, NotFoundError
from eyvan.server.service.scheduling import CoreSchedulingService

ADMIN = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
SUBMITTER = RoleGrant(role=Role.PROJECT_SUBMITTER, org_id="acme", project_id="research")
PASSWORD = "correct horse battery"
TTL = 3600


@pytest.fixture(params=["memory", "local", "sql"])
async def service(request, tmp_path):
    engine = None
    if request.param == "memory":
        repository = InMemoryRepository()
    elif request.param == "local":
        repository = LocalJsonRepository(tmp_path / "cp.json")
    else:
        from sqlalchemy.ext.asyncio import create_async_engine

        from eyvan.server.persistence.sqlalchemy import SqlAlchemyRepository, create_schema

        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'eyvan.db'}")
        await create_schema(engine)
        repository = SqlAlchemyRepository(engine)
    await repository.open()
    yield ControlPlaneService(repository, scheduler=CoreSchedulingService())
    await repository.close()
    if engine is not None:
        await engine.dispose()


async def _onboard(service: ControlPlaneService, *, subject: str = "alice", email: str = "Alice@Example.com"):
    invite, raw = await service.invite_user(ADMIN, email=email, subject=subject, grants=(SUBMITTER,))
    user, _, session = await service.accept_invite(raw, password=PASSWORD, display_name="Alice", session_ttl_s=TTL)
    return invite, raw, user, session


# --------------------------------------------------------------------- secrets


def test_password_hash_round_trips_and_is_salted() -> None:
    first, second = hash_password(PASSWORD), hash_password(PASSWORD)
    assert first != second and first.startswith("scrypt$")
    assert verify_password(PASSWORD, first) and not verify_password("wrong password!", first)
    assert not verify_password(PASSWORD, None) and not verify_password(PASSWORD, "garbage")
    with pytest.raises(ValueError):
        hash_password("short")


def test_throttle_blocks_after_the_limit_and_resets_on_success() -> None:
    now = [0.0]
    throttle = SignInThrottle(limit=3, window_s=60, clock=lambda: now[0])
    for _ in range(3):
        throttle.failed("a@x.io")
    assert throttle.blocked("a@x.io") and not throttle.blocked("b@x.io")
    now[0] = 61
    assert not throttle.blocked("a@x.io")
    throttle.failed("a@x.io")
    throttle.succeeded("a@x.io")
    assert not throttle.blocked("a@x.io")


# ------------------------------------------------------------- invite / accept


async def test_accepting_an_invite_creates_the_user_issues_its_grants_and_signs_in(service) -> None:
    invite, raw, user, session = await _onboard(service)
    assert user.email == "alice@example.com" and user.password_hash and PASSWORD not in user.password_hash
    assert invite.token_hash == token_hash(raw) and raw not in invite.model_dump_json()
    principal = await service.resolve_team_token(session)
    assert principal is not None and principal.subject == "alice"
    # Authority comes from durable grants, exactly like any other subject.
    grants = await service.active_grants_for("alice")
    assert SUBMITTER in grants
    assert service.authorizer.allows(
        principal.model_copy(update={"grants": grants}), Permission.SUBMIT, org_id="acme", project_id="research"
    )


async def test_an_invite_is_single_use(service) -> None:
    _, raw, _, _ = await _onboard(service)
    assert await service.preview_invite(raw) is None
    with pytest.raises(NotFoundError):
        await service.accept_invite(raw, password=PASSWORD, session_ttl_s=TTL)


async def test_an_expired_or_revoked_invite_cannot_be_accepted(service) -> None:
    _, expired_raw = await service.invite_user(ADMIN, email="e@x.io", subject="e", ttl=timedelta(seconds=-1))
    with pytest.raises(NotFoundError):
        await service.accept_invite(expired_raw, password=PASSWORD, session_ttl_s=TTL)
    invite, raw = await service.invite_user(ADMIN, email="r@x.io", subject="r")
    await service.revoke_invite(ADMIN, invite.id)
    assert await service.preview_invite(raw) is None
    with pytest.raises(NotFoundError):
        await service.accept_invite(raw, password=PASSWORD, session_ttl_s=TTL)


async def test_duplicate_users_and_open_invites_are_refused(service) -> None:
    await _onboard(service)
    with pytest.raises(ConflictError):
        await service.invite_user(ADMIN, email="other@x.io", subject="alice")
    with pytest.raises(ConflictError):
        await service.invite_user(ADMIN, email="ALICE@example.com", subject="alice2")
    await service.invite_user(ADMIN, email="bob@x.io", subject="bob")
    with pytest.raises(ConflictError):
        await service.invite_user(ADMIN, email="bob@x.io", subject="bob")


async def test_an_invite_cannot_hand_out_more_than_its_issuer_could_grant(service) -> None:
    project_admin = Principal(
        subject="pa", grants=(RoleGrant(role=Role.PROJECT_ADMIN, org_id="acme", project_id="research"),)
    )
    await service.invite_user(project_admin, email="dev@x.io", subject="dev", grants=(SUBMITTER,))
    with pytest.raises(AuthorizationError):
        await service.invite_user(
            project_admin, email="boss@x.io", subject="boss", grants=(RoleGrant(role=Role.FLEET_ADMIN),)
        )
    viewer = Principal(subject="v", grants=(RoleGrant(role=Role.PROJECT_VIEWER, org_id="acme", project_id="research"),))
    with pytest.raises(AuthorizationError):
        await service.invite_user(viewer, email="x@x.io", subject="x")


async def test_a_short_password_is_refused_before_anything_is_created(service) -> None:
    _, raw = await service.invite_user(ADMIN, email="s@x.io", subject="s")
    with pytest.raises(ValueError):
        await service.accept_invite(raw, password="short", session_ttl_s=TTL)
    assert await service.get_team_user("s") is None
    assert await service.preview_invite(raw) is not None


# ------------------------------------------------------------------- sign-in


async def test_password_login_succeeds_only_with_the_right_password(service) -> None:
    await _onboard(service)
    assert await service.password_login("alice@example.com", "wrong password!", session_ttl_s=TTL) is None
    assert await service.password_login("nobody@example.com", PASSWORD, session_ttl_s=TTL) is None
    result = await service.password_login(" ALICE@example.com ", PASSWORD, session_ttl_s=TTL)
    assert result is not None
    user, record, raw = result
    assert user.subject == "alice" and record.kind == "session" and record.expires_at is not None
    assert is_team_token(raw) and record.token_hash == token_hash(raw)
    assert (await service.resolve_team_token(raw)).subject == "alice"


async def test_signing_out_revokes_only_that_session(service) -> None:
    *_, first = await _onboard(service)
    _, _, second = await service.password_login("alice@example.com", PASSWORD, session_ttl_s=TTL)
    await service.end_session(first)
    assert await service.resolve_team_token(first) is None
    assert await service.resolve_team_token(second) is not None


async def test_unknown_and_foreign_bearer_tokens_resolve_to_nothing(service) -> None:
    assert await service.resolve_team_token("eyv_" + "A" * 43) is None
    assert await service.resolve_team_token("an.oidc.jwt") is None


# ---------------------------------------------------------------- API tokens


async def test_personal_tokens_are_listed_without_sessions_and_revocable(service) -> None:
    await _onboard(service)
    alice = Principal(subject="alice")
    record, raw = await service.create_api_token(alice, name="laptop", expires_in_days=30)
    assert record.kind == "api" and raw.startswith("eyv_") and raw not in record.model_dump_json()
    listed = await service.list_api_tokens(alice)
    assert [item.id for item in listed] == [record.id]
    assert (await service.resolve_team_token(raw)).subject == "alice"
    await service.revoke_api_token(alice, record.id)
    assert await service.resolve_team_token(raw) is None


async def test_one_user_cannot_touch_another_users_tokens(service) -> None:
    await _onboard(service)
    await _onboard(service, subject="bob", email="bob@x.io")
    record, _ = await service.create_api_token(Principal(subject="alice"), name="laptop")
    with pytest.raises(AuthorizationError):
        await service.revoke_api_token(Principal(subject="bob"), record.id)
    with pytest.raises(AuthorizationError):
        await service.list_api_tokens(Principal(subject="bob"), subject="alice")
    assert (await service.revoke_api_token(ADMIN, record.id)).revoked_at is not None


async def test_a_principal_without_an_account_gets_no_personal_token(service) -> None:
    with pytest.raises(AuthorizationError):
        await service.create_api_token(Principal(subject="oidc-person"), name="x")
    await _onboard(service)
    with pytest.raises(InputError):
        await service.create_api_token(Principal(subject="alice"), name="x", expires_in_days=0)


# --------------------------------------------------------------------- users


async def test_disabling_a_user_revokes_every_token_and_blocks_sign_in(service) -> None:
    *_, session = await _onboard(service)
    _, api_raw = await service.create_api_token(Principal(subject="alice"), name="cli")
    await service.set_user_disabled(ADMIN, "alice", disabled=True)
    assert await service.resolve_team_token(session) is None
    assert await service.resolve_team_token(api_raw) is None
    assert await service.password_login("alice@example.com", PASSWORD, session_ttl_s=TTL) is None
    await service.set_user_disabled(ADMIN, "alice", disabled=False)
    assert await service.password_login("alice@example.com", PASSWORD, session_ttl_s=TTL) is not None
    # Re-enabling never resurrects a revoked token.
    assert await service.resolve_team_token(api_raw) is None


async def test_only_a_fleet_admin_administers_accounts(service) -> None:
    await _onboard(service)
    alice = Principal(subject="alice", grants=(SUBMITTER,))
    for call in (
        service.list_team_users(alice),
        service.list_invites(alice),
        service.set_user_disabled(alice, "alice", disabled=True),
    ):
        with pytest.raises(AuthorizationError):
            await call
    with pytest.raises(InputError):
        await service.set_user_disabled(Principal(subject="root", grants=ADMIN.grants), "root", disabled=True)
    assert [user.subject for user in await service.list_team_users(ADMIN)] == ["alice"]


async def test_account_changes_are_audited_without_secrets(service) -> None:
    _, raw, _, session = await _onboard(service)
    async with service.repository.transaction() as tx:
        events = await tx.list_events(limit=100)
    actions = {event.action for event in events}
    assert {"invite.created", "user.created"} <= actions
    dumped = json.dumps([event.payload for event in events])
    assert raw not in dumped and session not in dumped and PASSWORD not in dumped


# ---------------------------------------------------------------- durability


async def test_accounts_survive_a_local_restart_in_a_v7_document(tmp_path) -> None:
    path = tmp_path / "cp.json"
    repository = LocalJsonRepository(path)
    await repository.open()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    *_, session = await _onboard(service)
    _, api_raw = await service.create_api_token(Principal(subject="alice"), name="cli")
    await repository.close()

    document = json.loads(path.read_text())
    assert document["version"] == LOCAL_CONTROL_PLANE_TEAM_SCHEMA_VERSION
    text = path.read_text()
    assert session not in text and api_raw not in text and PASSWORD not in text

    reopened = LocalJsonRepository(path)
    await reopened.open()
    try:
        again = ControlPlaneService(reopened, scheduler=CoreSchedulingService())
        assert (await again.resolve_team_token(api_raw)).subject == "alice"
        assert await again.password_login("alice@example.com", PASSWORD, session_ttl_s=TTL) is not None
    finally:
        await reopened.close()


async def test_a_plane_without_accounts_keeps_its_older_document_version(tmp_path) -> None:
    path = tmp_path / "cp.json"
    repository = LocalJsonRepository(path)
    await repository.open()
    service = ControlPlaneService(repository, scheduler=CoreSchedulingService())
    # Any commit at all; team tables stay empty.
    async with service.repository.transaction() as tx:
        await tx.list_users()
    await repository.close()
    if path.exists():
        assert json.loads(path.read_text())["version"] < LOCAL_CONTROL_PLANE_TEAM_SCHEMA_VERSION
