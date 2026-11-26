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
