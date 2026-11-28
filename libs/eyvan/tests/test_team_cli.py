# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""`eyvan login` / `logout` / `users` / `tokens` against a real in-process team plane.

Only the network hop is replaced: requests go to the real FastAPI app through
Starlette's TestClient. The password is read from stdin (never a flag), the
token is stored owner-only outside the context file, and later commands send it.
"""

from __future__ import annotations

import os
import stat
from types import SimpleNamespace

import pytest
from click.testing import CliRunner
from starlette.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.sdk import cli
from eyvan.sdk.commands import team as team_commands
from eyvan.sdk.commands.http import EyvanApiError
from eyvan.sdk.team_credentials import TeamCredentialError, TeamCredentialStore
from eyvan.server.api.app import create_app

PLANE = "http://127.0.0.1:8080"
ADMIN = {"X-Eyvan-User": "root", "X-Eyvan-Roles": "fleet_admin"}
PASSWORD = "correct horse battery"


@pytest.fixture
def plane(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(tmp_path / "contexts.json"))
    settings = EyvanSettings(
        dev_auth_enabled=True, team_mode=True, controller_enabled=False, artifact_token_key="k" * 32
    )
    app = create_app(settings=settings)
    admin = TestClient(app, base_url=PLANE)
    invite = admin.post(
        "/api/v1/team/invites",
        json={
            "email": "alice@example.com",
            "subject": "alice",
            "role": "project_submitter",
            "org_id": "acme",
            "project_id": "r",
        },
        headers=ADMIN,
    ).json()
    TestClient(app, base_url=PLANE).post(
        "/api/v1/team/invites/accept", json={"token": invite["token"], "password": PASSWORD}
    )

    class _NoLifespan:
        def __init__(self, url: str, trust: object = None) -> None:
            self.client = TestClient(app, base_url=url)

        def __enter__(self) -> TestClient:
            return self.client

        def __exit__(self, *exc: object) -> bool:
            return False

    sent: list[dict[str, str]] = []

    class ForwardingClient:
        """Stands in for the CLI's EyvanClient: records identity headers, forwards to the app."""

        def __init__(self, url: str, *, headers) -> None:
            self.headers = dict(headers)
            self.http = TestClient(app, base_url=url)

        def request(self, method, path, *, params=None, json=None, headers=None, content=None):
            sent.append(self.headers)
            response = self.http.request(
                method, path, params=params, json=json, headers={**self.headers, **(headers or {})}
            )
            if response.status_code >= 400:
                raise EyvanApiError(response.status_code, response.text)
            return response.json()

    monkeypatch.setattr(team_commands, "_http", _NoLifespan)
    monkeypatch.setattr(team_commands, "connect_plane", lambda *a, **k: SimpleNamespace(notes=[], saved_context=None))
    monkeypatch.setattr(cli, "EyvanClient", ForwardingClient)
    return SimpleNamespace(app=app, admin=admin, sent=sent, home=tmp_path)


def _login(**kwargs):
    return CliRunner().invoke(
        cli.main, ["login", PLANE, "--email", "alice@example.com", "--password-stdin"], input=f"{PASSWORD}\n", **kwargs
    )


def test_login_stores_an_owner_only_token_that_later_commands_send(plane) -> None:
    result = _login()
    assert result.exit_code == 0, result.output
    assert "signed in to http://127.0.0.1:8080 as alice" in result.output
    path = plane.home / ".eyvan" / "team-credentials.json"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    credential = TeamCredentialStore().get(PLANE)
    assert credential is not None and credential.token.startswith("eyv_")
    assert credential.token not in result.output and PASSWORD not in path.read_text()
    # Not in the (non-secret) context file.
    contexts = plane.home / "contexts.json"
    assert not contexts.exists() or credential.token not in contexts.read_text()

    me = CliRunner().invoke(cli.main, ["users", "me"])
    assert me.exit_code == 0, me.output
    assert me.output.startswith("alice") and "project_submitter" in me.output
    assert plane.sent[-1] == {"Authorization": f"Bearer {credential.token}"}


def test_a_wrong_password_saves_nothing(plane) -> None:
    result = CliRunner().invoke(
        cli.main, ["login", PLANE, "--email", "alice@example.com", "--password-stdin"], input="wrong password!\n"
    )
    assert result.exit_code != 0 and "wrong user name or password" in result.output
    assert TeamCredentialStore().get(PLANE) is None


def test_login_accepts_the_user_name(plane) -> None:
    result = CliRunner().invoke(
        cli.main, ["login", PLANE, "--username", "alice", "--password-stdin"], input=f"{PASSWORD}\n"
    )
    assert result.exit_code == 0, result.output
    assert "as alice" in result.output
    assert TeamCredentialStore().get(PLANE) is not None


def test_logout_revokes_the_token_on_the_plane_and_forgets_it(plane) -> None:
    assert _login().exit_code == 0
    credential = TeamCredentialStore().get(PLANE)
    result = CliRunner().invoke(cli.main, ["logout", PLANE])
    assert result.exit_code == 0, result.output
    assert "revoked" in result.output
    assert TeamCredentialStore().get(PLANE) is None
    refused = TestClient(plane.app, base_url=PLANE).get(
        "/api/v1/team/me", headers={"Authorization": f"Bearer {credential.token}"}
    )
    assert refused.status_code == 401


def test_tokens_create_list_revoke(plane) -> None:
    assert _login().exit_code == 0
    created = CliRunner().invoke(cli.main, ["tokens", "create", "ci"])
    assert created.exit_code == 0, created.output
    raw = created.output.split()[0]
    assert raw.startswith("eyv_")
    listed = CliRunner().invoke(cli.main, ["tokens", "list", "--json"])
    assert "ci" in listed.output and raw not in listed.output
    token_id = next(
        line for line in CliRunner().invoke(cli.main, ["tokens", "list"]).output.splitlines() if " ci " in line
    )
    revoked = CliRunner().invoke(cli.main, ["tokens", "revoke", token_id.split()[0]])
    assert revoked.exit_code == 0, revoked.output
    assert (
        TestClient(plane.app, base_url=PLANE)
        .get("/api/v1/team/me", headers={"Authorization": f"Bearer {raw}"})
        .status_code
        == 401
    )


def test_a_plane_without_team_mode_is_refused_clearly(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(tmp_path / "contexts.json"))
    app = create_app(
        settings=EyvanSettings(dev_auth_enabled=True, controller_enabled=False, artifact_token_key="k" * 32)
    )

    class _NoLifespan:
        def __init__(self, url: str, trust: object = None) -> None:
            self.client = TestClient(app, base_url=url)

        def __enter__(self):
            return self.client

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(team_commands, "_http", _NoLifespan)
    result = CliRunner().invoke(cli.main, ["login", PLANE, "--email", "a@b.co", "--password-stdin"], input="x\n")
    assert result.exit_code != 0 and "does not run team accounts" in result.output


def test_a_group_readable_credential_file_is_refused(tmp_path) -> None:
    path = tmp_path / "team-credentials.json"
    path.write_text('{"version": 1, "planes": {}}')
    os.chmod(path, 0o644)
    with pytest.raises(TeamCredentialError, match="readable by other users"):
        TeamCredentialStore(path).get(PLANE)


def test_loopback_team_token_suppresses_development_fleet_admin_banner(plane, monkeypatch) -> None:
    """A signed-in non-admin team user on localhost must never see 'acting as fleet_admin'."""

    monkeypatch.setattr(cli, "stream_is_terminal", lambda _stream: True)

    # First login on a loopback TTY must not print the dev fleet_admin banner either.
    login_result = _login()
    assert login_result.exit_code == 0, login_result.output
    assert "acting as" not in login_result.output
    assert "fleet_admin" not in login_result.output

    # Subsequent commands with the saved team token suppress the banner — both allowed and 403 paths.
    me = CliRunner().invoke(cli.main, ["users", "me"])
    assert me.exit_code == 0, me.output
    assert "acting as" not in me.output
    assert "fleet_admin" not in me.output
    assert "development plane" not in me.output

    denied = CliRunner().invoke(cli.main, ["users", "list"])
    assert denied.exit_code != 0
    assert "acting as" not in denied.output
    assert "fleet_admin" not in denied.output
    assert "development plane" not in denied.output

    # After logout (no team token left), loopback development auth banner returns normally.
    assert CliRunner().invoke(cli.main, ["logout", PLANE]).exit_code == 0
    back_to_dev = CliRunner().invoke(cli.main, ["users", "list"])
    assert back_to_dev.exit_code == 0, back_to_dev.output
    assert "development plane" in back_to_dev.output
    assert "fleet_admin" in back_to_dev.output


@pytest.mark.parametrize("tier", ["trusted", "user"])
def test_login_with_invite_discovers_ready_fleet_and_saves_context_for_non_admin(
    tmp_path, monkeypatch, tier: str
) -> None:
    """End-to-end login --invite using real connect/probe against a team plane with a ready fleet."""

    import asyncio
    import json

    import httpx

    from eyvan.records import CapacityPoolRecord, Principal, Role, RoleGrant

    monkeypatch.setenv("HOME", str(tmp_path))
    contexts_path = tmp_path / "contexts.json"
    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(contexts_path))
    monkeypatch.setattr(cli, "stream_is_terminal", lambda _stream: True)

    settings = EyvanSettings(
        dev_auth_enabled=True, team_mode=True, controller_enabled=False, artifact_token_key="k" * 32
    )
    app = create_app(settings=settings)
    admin_principal = Principal(subject="root", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    pool = CapacityPoolRecord(
        id="research/v5p-8",
        fleet_name="research",
        node_type="v5p-8",
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
        max_cells=2,
        enabled=True,
    )
    asyncio.run(app.state.service.upsert_capacity_pool(admin_principal, pool, idempotency_key="seed-fleet"))

    admin_http = TestClient(app, base_url=PLANE)
    invite = admin_http.post(
        "/api/v1/team/invites",
        json={
            "email": f"{tier}@example.com",
            "subject": tier,
            "tier": tier,
            "org_id": "acme",
            "project_id": "research",
        },
        headers=ADMIN,
    ).json()

    class _NoLifespan:
        def __init__(self, url: str, trust: object = None) -> None:
            self.client = TestClient(app, base_url=url)

        def __enter__(self) -> TestClient:
            return self.client

        def __exit__(self, *exc: object) -> bool:
            return False

    real_client = httpx.Client

    def handle(request: httpx.Request) -> httpx.Response:
        resp = admin_http.request(
            request.method, str(request.url), headers=dict(request.headers), content=request.content
        )
        return httpx.Response(resp.status_code, headers=resp.headers, content=resp.content)

    monkeypatch.setattr(team_commands, "_http", _NoLifespan)
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(transport=httpx.MockTransport(handle), **kwargs))

    result = CliRunner().invoke(
        cli.main,
        ["login", PLANE, "--invite", invite["token"], "--password-stdin"],
        input=f"{PASSWORD}\n",
    )
    assert result.exit_code == 0, result.output
    assert "acting as" not in result.output
    assert "ready fleets: research" in result.output
    assert "ready fleets: none yet" not in result.output
    assert "active context: research (acme/research)" in result.output

    saved = json.loads(contexts_path.read_text(encoding="utf-8"))
    assert saved["active_context"] == "research"
    assert saved["contexts"]["research"]["fleet"] == "research"
    assert saved["contexts"]["research"]["org_id"] == "acme"
    assert saved["contexts"]["research"]["project_id"] == "research"
