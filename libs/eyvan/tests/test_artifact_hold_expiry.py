# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A hold whose expiry is already in the past protects nothing.

``hold_artifact`` used to accept any timestamp, so a typo'd past date produced
an audit event saying an artifact was held while the collector could claim it
immediately. The creation entry point refuses an already-expired hold.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone, tzinfo

import httpx
import pytest

from eyvan.artifacts.managed import LocalManagedArtifactStore, callable_uri
from eyvan.config import EyvanSettings
from eyvan.records import Principal, Role, RoleGrant, utcnow
from eyvan.server.api.app import create_app
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.server.service.errors import AuthorizationError, InputError


async def _service_with_artifact(tmp_path):
    repository = InMemoryRepository()
    await repository.open()
    store = LocalManagedArtifactStore(tmp_path / "artifacts")
    service = ControlPlaneService(repository=repository, artifact_store=store)
    admin = Principal(
        subject="alice", grants=(RoleGrant(role=Role.PROJECT_ADMIN, org_id="acme", project_id="research"),)
    )
    payload = b"payload"
    from hashlib import sha256

    digest = sha256(payload).hexdigest()
    uri = callable_uri(digest)
    await service.upload_managed_artifact(org_id="acme", project_id="research", uri=uri, payload=payload)
    return service, admin, uri


@pytest.mark.asyncio
async def test_a_hold_expiring_in_the_past_is_refused(tmp_path) -> None:
    service, admin, uri = await _service_with_artifact(tmp_path)

    with pytest.raises(InputError, match="expir"):
        await service.hold_artifact(
            admin,
            org_id="acme",
            project_id="research",
            uri=uri,
            reason="keep for forensics",
            expires_at=utcnow() - timedelta(seconds=1),
        )


@pytest.mark.asyncio
async def test_a_hold_expiring_in_the_future_is_accepted(tmp_path) -> None:
    service, admin, uri = await _service_with_artifact(tmp_path)

    record = await service.hold_artifact(
        admin,
        org_id="acme",
        project_id="research",
        uri=uri,
        reason="keep for forensics",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )

    assert record.hold is not None


class _NoUTCOffset(tzinfo):
    def utcoffset(self, dt: datetime | None) -> None:
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize("expiry_tz", [None, _NoUTCOffset()], ids=["no-tzinfo", "no-utcoffset"])
@pytest.mark.parametrize("hours", [-1, 1], ids=["past", "future"])
async def test_a_naive_hold_expiry_is_a_controlled_input_error(tmp_path, expiry_tz, hours: int) -> None:
    service, admin, uri = await _service_with_artifact(tmp_path)
    expires_at = (utcnow() + timedelta(hours=hours)).replace(tzinfo=expiry_tz)

    with pytest.raises(InputError, match="timezone"):
        await service.hold_artifact(
            admin,
            org_id="acme",
            project_id="research",
            uri=uri,
            reason="keep for forensics",
            expires_at=expires_at,
        )

    async with service.repository.transaction() as tx:
        stored = await tx.get_managed_artifact("acme", "research", uri)
    assert stored is not None
    assert stored.hold is None


@pytest.mark.asyncio
@pytest.mark.parametrize("expiry_tz", [UTC, timezone(timedelta(hours=5, minutes=30)), None])
async def test_aware_future_and_indefinite_holds_are_persisted(tmp_path, expiry_tz) -> None:
    service, admin, uri = await _service_with_artifact(tmp_path)
    expires_at = None if expiry_tz is None else (utcnow() + timedelta(hours=1)).astimezone(expiry_tz)

    record = await service.hold_artifact(
        admin,
        org_id="acme",
        project_id="research",
        uri=uri,
        reason="keep for forensics",
        expires_at=expires_at,
    )

    assert record.hold is not None
    assert record.hold.expires_at == expires_at
    assert record.hold.held_by == admin.subject
    assert record.hold.active_at(utcnow())
    async with service.repository.transaction() as tx:
        stored = await tx.get_managed_artifact("acme", "research", uri)
    assert stored is not None
    assert stored.hold == record.hold


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "org_id", "project_id"),
    [
        (Role.PROJECT_VIEWER, "acme", "research"),
        (Role.PROJECT_ADMIN, "other", "research"),
        (Role.PROJECT_ADMIN, "acme", "other"),
    ],
)
async def test_naive_expiry_does_not_bypass_service_authorization(tmp_path, role, org_id, project_id) -> None:
    service, _admin, uri = await _service_with_artifact(tmp_path)
    principal = Principal(subject="bob", grants=(RoleGrant(role=role, org_id=org_id, project_id=project_id),))

    with pytest.raises(AuthorizationError, match="manage_project"):
        await service.hold_artifact(
            principal,
            org_id="acme",
            project_id="research",
            uri=uri,
            reason="keep",
            expires_at=datetime(2050, 1, 1),
        )

    async with service.repository.transaction() as tx:
        stored = await tx.get_managed_artifact("acme", "research", uri)
    assert stored is not None
    assert stored.hold is None


def _headers(*, role: str = "project_admin", org_id: str = "acme", project_id: str = "research") -> dict[str, str]:
    return {
        "X-Eyvan-User": "alice",
        "X-Eyvan-Roles": role,
        "X-Eyvan-Org": org_id,
        "X-Eyvan-Project": project_id,
    }


def _api(service: ControlPlaneService, tmp_path):
    return create_app(
        settings=EyvanSettings(
            dev_auth_enabled=True,
            artifact_store_uri=str(tmp_path / "artifacts"),
            artifact_token_key="k" * 32,
        ),
        repository=service.repository,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("hours", [-1, 1], ids=["past", "future"])
async def test_api_naive_hold_expiry_returns_400_not_500(tmp_path, hours: int) -> None:
    service, _admin, uri = await _service_with_artifact(tmp_path)
    expires_at = (utcnow() + timedelta(hours=hours)).replace(tzinfo=None)
    # Turn unhandled service exceptions into HTTP 500 so this asserts the
    # caller-visible boundary rather than httpx re-raising the original error.
    transport = httpx.ASGITransport(app=_api(service, tmp_path), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/artifacts/holds",
            headers=_headers(),
            json={
                "org_id": "acme",
                "project_id": "research",
                "uri": uri,
                "reason": "keep",
                "expires_at": expires_at.isoformat(),
            },
        )

    assert response.status_code == 400, response.text
    assert "timezone" in response.json()["detail"]
    async with service.repository.transaction() as tx:
        stored = await tx.get_managed_artifact("acme", "research", uri)
    assert stored is not None
    assert stored.hold is None


@pytest.mark.asyncio
@pytest.mark.parametrize("expiry_kind", ["future-utc", "future-offset", "expired", "indefinite"])
async def test_api_preserves_aware_and_indefinite_hold_behavior(tmp_path, expiry_kind: str) -> None:
    service, _admin, uri = await _service_with_artifact(tmp_path)
    expires_at = None
    if expiry_kind != "indefinite":
        expires_at = utcnow() + timedelta(hours=-1 if expiry_kind == "expired" else 1)
        if expiry_kind == "future-offset":
            expires_at = expires_at.astimezone(timezone(timedelta(hours=5, minutes=30)))
    transport = httpx.ASGITransport(app=_api(service, tmp_path), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/artifacts/holds",
            headers=_headers(),
            json={
                "org_id": "acme",
                "project_id": "research",
                "uri": uri,
                "reason": "keep",
                "expires_at": None if expires_at is None else expires_at.isoformat(),
            },
        )

    async with service.repository.transaction() as tx:
        stored = await tx.get_managed_artifact("acme", "research", uri)
    assert stored is not None
    if expiry_kind == "expired":
        assert response.status_code == 400, response.text
        assert "future" in response.json()["detail"]
        assert stored.hold is None
    else:
        assert response.status_code == 200, response.text
        assert response.json()["hold"]["held_by"] == "alice"
        assert stored.hold is not None
        assert stored.hold.expires_at == expires_at
        assert stored.hold.active_at(utcnow())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("role", "org_id", "project_id"),
    [("project_viewer", "acme", "research"), ("project_admin", "other", "research"), ("project_admin", "acme", "other")],
)
async def test_api_authorizes_scope_before_validating_naive_expiry(tmp_path, role, org_id, project_id) -> None:
    service, _admin, uri = await _service_with_artifact(tmp_path)
    transport = httpx.ASGITransport(app=_api(service, tmp_path), raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/artifacts/holds",
            headers=_headers(role=role, org_id=org_id, project_id=project_id),
            json={
                "org_id": "acme",
                "project_id": "research",
                "uri": uri,
                "reason": "keep",
                "expires_at": "2050-01-01T00:00:00",
            },
        )

    assert response.status_code == 403, response.text
    assert "manage_project" in response.json()["detail"]
    async with service.repository.transaction() as tx:
        stored = await tx.get_managed_artifact("acme", "research", uri)
    assert stored is not None
    assert stored.hold is None
