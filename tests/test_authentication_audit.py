# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Opt-in authentication audit records must not retain presented credentials."""

from dataclasses import replace

import httpx
import pytest
from starlette.testclient import TestClient
from test_group_lease_lifecycle import repository as _repository_fixture

from eyvan._internal.platform import EyvanPlatformConfig
from eyvan.config import EyvanSettings, RepositoryBackend
from eyvan.server.api.app import create_app

repository = _repository_fixture


@pytest.fixture(autouse=True)
def _isolated_api_home(monkeypatch, tmp_path) -> None:
    """Keep API startup out of the operator's artifact and credential state."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("EYVAN_ARTIFACT_STORE_URI", str(tmp_path / "managed-artifacts"))


@pytest.mark.parametrize("enabled", [False, True])
async def test_failed_authentication_is_audited_without_claimed_tenant_or_credentials(
    monkeypatch,
    tmp_path,
    enabled,
    repository,
):
    monkeypatch.setenv("EYVAN_AUTH_AUDIT_ENABLED", "1" if enabled else "0")
    settings = replace(
        EyvanSettings.from_env(),
        repository_backend=RepositoryBackend.MEMORY,
        controller_enabled=False,
        dev_auth_enabled=False,
        oidc_issuer=None,
        bootstrap_token="fixture-admin",
        artifact_store_uri=str(tmp_path / "managed-artifacts"),
        artifact_token_key="k" * 32,
    )
    app = create_app(settings=settings, repository=repository, platform_config=EyvanPlatformConfig())
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        refused = await client.get(
            "/api/v1/fleets/secret-looking-name",
            headers={"Authorization": "Bearer DO-NOT-RECORD", "X-Eyvan-Org": "forged"},
            params={"token": "HIDDEN-QUERY"},
        )
        assert refused.status_code == 401
        response = await client.get("/api/v1/fleet/events", headers={"Authorization": "Bearer fixture-admin"})
        assert response.status_code == 200
        events = [event for event in response.json() if event["action"] == "authentication_failed"]
    assert len(events) == (1 if enabled else 0)
    if enabled:
        event = events[0]
        assert event["actor"] == "anonymous"
        assert event["org_id"] is None and event["project_id"] is None
        assert event["payload"]["status_code"] == 401
        assert event["payload"]["method"] == "GET"
        assert "{fleet_name}" in event["payload"]["route"]
        rendered = str(event)
        for sensitive in ("DO-NOT-RECORD", "HIDDEN-QUERY", "secret-looking-name", "forged"):
            assert sensitive not in rendered


def test_an_audit_store_failure_never_changes_an_authentication_denial(monkeypatch, tmp_path, caplog):
    app = create_app(
        settings=EyvanSettings(
            controller_enabled=False,
            auth_audit_enabled=True,
            bootstrap_token="fixture-admin",
            artifact_token_key="k" * 32,
            artifact_store_uri=str(tmp_path / "managed-artifacts"),
        ),
        platform_config=EyvanPlatformConfig(),
    )

    async def unavailable(**_kwargs):
        raise RuntimeError("DO-NOT-LOG-SENSITIVE-DATABASE-DETAIL")

    monkeypatch.setattr(app.state.service, "record_authentication_failure", unavailable)
    with TestClient(app) as client:
        response = client.get("/api/v1/session", headers={"Authorization": "Bearer invalid"})
    assert response.status_code == 401
    assert "authentication_audit_write_failed" in caplog.text
    assert "DO-NOT-LOG-SENSITIVE-DATABASE-DETAIL" not in caplog.text


async def test_role_changes_and_policy_edits_retain_actor_scope_and_order():
    from eyvan.records import PolicyRecord, Principal, Role, RoleGrant, RoleGrantRecord
    from eyvan.server.persistence import InMemoryRepository
    from eyvan.server.service import ControlPlaneService

    repository = InMemoryRepository()
    service = ControlPlaneService(repository)
    admin = Principal(subject="audited-admin", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    grant = RoleGrantRecord(
        subject="alice",
        grant=RoleGrant(
            role=Role.PROJECT_OPERATOR,
            org_id="acme",
            project_id="research",
        ),
        granted_by="ignored-input",
    )
    await service.issue_role_grant(admin, grant, idempotency_key="grant")
    await service.revoke_role_grant(admin, ("alice", "project_operator", "acme", "research"))
    policy = PolicyRecord(org_id="acme", project_id="research", updated_by="ignored-input")
    await service.put_policy(admin, policy, idempotency_key="policy-1")
    await service.put_policy(
        admin, policy.model_copy(update={"allowed_regions": ("local",)}), idempotency_key="policy-2"
    )
    events = await service.list_fleet_events(admin)
    assert [event.action for event in events] == ["grant.issued", "grant.revoked", "policy.updated", "policy.updated"]
    assert all(event.actor == "audited-admin" for event in events)
    assert all((event.org_id, event.project_id) == ("acme", "research") for event in events)
    assert len({event.sequence for event in events}) == len(events)
