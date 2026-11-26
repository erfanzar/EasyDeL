# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``eyvan mcp config`` / ``install``: redacted entries, owner-only merge, no token on screen.

The CLI talks to a real in-process plane through a thin adapter, so ``install``
mints a real personal token and the written entry is proven to authenticate.
"""

from __future__ import annotations

import json
import stat
import typing as tp

import click
import pytest
from click.testing import CliRunner
from starlette.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.sdk.client.identity import ClientIdentity
from eyvan.sdk.commands.mcp import mcp, needs_private_network
from eyvan.sdk.commands.state import CliSession, ContextOverrides
from eyvan.sdk.contexts import FileContextStore
from eyvan.server.api.app import create_app

ORIGIN = "http://127.0.0.1:8080"
ADMIN = {"X-Eyvan-User": "root", "X-Eyvan-Roles": "fleet_admin"}
PASSWORD = "correct horse battery"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))


class _AppClient:
    """The CLI's ApiClient protocol, answered by an in-process plane."""

    def __init__(self, http: TestClient, headers: dict[str, str]) -> None:
        self.http = http
        self.headers = headers
        self.calls: list[tuple[str, str]] = []

    def request(self, method: str, path: str, *, params=None, json=None, headers=None) -> tp.Any:
        self.calls.append((method, path))
        response = self.http.request(method, path, params=params, json=json, headers={**self.headers, **(headers or {})})
        if response.status_code >= 400:
            raise click.ClickException(f"{response.status_code}: {response.text[:200]}")
        return response.json()


def _plane(*, team_mode: bool) -> TestClient:
    settings = EyvanSettings(dev_auth_enabled=True, team_mode=team_mode, artifact_token_key="k" * 32)
    return TestClient(create_app(settings=settings), base_url=ORIGIN)


def _login_token(http: TestClient) -> str:
    created = http.post(
        "/api/v1/team/invites",
        json={
            "email": "alice@example.com",
            "subject": "alice",
            "role": "project_operator",
            "org_id": "acme",
            "project_id": "r",
        },
        headers=ADMIN,
    )
    assert created.status_code == 201, created.text
    http.post("/api/v1/team/invites/accept", json={"token": created.json()["token"], "password": PASSWORD})
    http.cookies.clear()
    login = http.post(
        "/api/v1/team/login", json={"email": "alice@example.com", "password": PASSWORD, "token_name": "cli-test"}
    )
    assert login.status_code == 200, login.text
    return str(login.json()["token"])


def _session(tmp_path, http: TestClient, identity: ClientIdentity) -> tuple[CliSession, _AppClient]:
    client = _AppClient(http, identity.headers())
    store = FileContextStore(tmp_path / "contexts.json")
    session = CliSession(
        client=client,  # type: ignore[arg-type]
        identity=identity,
        context_store=store,
        context_config=store.load(),
        base_url=ORIGIN,
        org_id="acme",
        project_id="r",
        overrides=ContextOverrides(),
    )
    return session, client


def _team(tmp_path) -> tuple[TestClient, str, CliSession, _AppClient]:
    http = _plane(team_mode=True)
    token = _login_token(http)
    session, client = _session(tmp_path, http, ClientIdentity(token=token))
    return http, token, session, client


def test_config_prints_the_entry_with_the_token_redacted(tmp_path) -> None:
    _http, token, session, _client = _team(tmp_path)
    result = CliRunner().invoke(mcp, ["config", "--json"], obj=session)
    assert result.exit_code == 0, result.output
    assert token not in result.output
    document = json.loads(result.output)
    assert document == {
        "eyvan": {
            "transport": "streamable_http",
            "url": f"{ORIGIN}/api/v1/mcp",
            "headers": {"Authorization": "Bearer <redacted>", "X-Eyvan-Org": "acme", "X-Eyvan-Project": "r"},
            "allowPrivateNetwork": True,
        }
    }
    named = CliRunner().invoke(mcp, ["config", "--name", "plane"], obj=session)
    assert named.exit_code == 0 and token not in named.output and '"plane"' in named.output


def test_install_writes_owner_only_merges_and_never_prints_the_token(tmp_path) -> None:
    http, login_token, session, client = _team(tmp_path)
    path = tmp_path / "xerxes" / "mcp.json"
    path.parent.mkdir()
    path.write_text(json.dumps({"other": {"command": "other-server", "args": ["--x"]}}), encoding="utf-8")

    result = CliRunner().invoke(mcp, ["install", "--xerxes", "--path", str(path)], obj=session)
    assert result.exit_code == 0, result.output
    assert login_token not in result.output and "eyv_" not in result.output
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["other"] == {"command": "other-server", "args": ["--x"]}
    entry = document["eyvan"]
    assert entry["transport"] == "streamable_http" and entry["url"] == f"{ORIGIN}/api/v1/mcp"
    assert entry["allowPrivateNetwork"] is True
    minted = entry["headers"]["Authorization"].removeprefix("Bearer ")
    assert minted.startswith("eyv_") and minted != login_token  # a fresh, separate token
    assert minted not in result.output
    assert ("POST", "/api/v1/team/tokens") in client.calls

    # The written entry authenticates as alice, and the token is listed as revocable.
    reply = http.post(
        "/api/v1/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        headers=entry["headers"],
    )
    assert reply.status_code == 200 and reply.json()["result"]["tools"]
    names = [
        item["name"]
        for item in http.get("/api/v1/team/tokens", headers={"Authorization": f"Bearer {minted}"}).json()["tokens"]
    ]
    assert any(name.startswith("agent-xerxes-") for name in names)

    # A second install refuses to overwrite and changes nothing.
    before = path.read_bytes()
    refused = CliRunner().invoke(mcp, ["install", "--xerxes", "--path", str(path)], obj=session)
    assert refused.exit_code != 0 and "--yes" in refused.output
    assert path.read_bytes() == before
    replaced = CliRunner().invoke(mcp, ["install", "--xerxes", "--path", str(path), "--yes", "--json"], obj=session)
    assert replaced.exit_code == 0, replaced.output
    assert "eyv_" not in replaced.output
    again = json.loads(path.read_text(encoding="utf-8"))
    assert (
        set(again) == {"other", "eyvan"}
        and again["eyvan"]["headers"]["Authorization"] != entry["headers"]["Authorization"]
    )
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_install_preserves_the_list_shape(tmp_path) -> None:
    _http, _token, session, _client = _team(tmp_path)
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps([{"name": "a", "command": "a"}]), encoding="utf-8")
    result = CliRunner().invoke(mcp, ["install", "--xerxes", "--path", str(path), "--name", "plane"], obj=session)
    assert result.exit_code == 0, result.output
    document = json.loads(path.read_text(encoding="utf-8"))
    assert [item["name"] for item in document] == ["a", "plane"]
    assert document[1]["transport"] == "streamable_http"


def test_install_refuses_invalid_json_without_touching_it(tmp_path) -> None:
    _http, _token, session, client = _team(tmp_path)
    path = tmp_path / "mcp.json"
    path.write_text('{"secret": "sk-do-not-echo", ', encoding="utf-8")
    result = CliRunner().invoke(mcp, ["install", "--xerxes", "--path", str(path)], obj=session)
    assert result.exit_code != 0 and "not valid JSON" in result.output
    assert "sk-do-not-echo" not in result.output
    assert path.read_text(encoding="utf-8") == '{"secret": "sk-do-not-echo", '
    assert ("POST", "/api/v1/team/tokens") not in client.calls  # no token minted for nothing


def test_dev_auth_plane_needs_no_token(tmp_path) -> None:
    http = _plane(team_mode=False)
    identity = ClientIdentity(dev_user="alice", dev_roles=("project_operator",), dev_org="acme", dev_project="r")
    session, client = _session(tmp_path, http, identity)
    config = CliRunner().invoke(mcp, ["config"], obj=session)
    assert config.exit_code == 0, config.output
    assert "no team token is needed" in config.output and "Authorization" not in config.output

    path = tmp_path / "mcp.json"
    result = CliRunner().invoke(mcp, ["install", "--xerxes", "--path", str(path)], obj=session)
    assert result.exit_code == 0, result.output
    assert "no team token is needed and none was minted" in result.output
    assert ("POST", "/api/v1/team/tokens") not in client.calls
    entry = json.loads(path.read_text(encoding="utf-8"))["eyvan"]
    assert "Authorization" not in entry["headers"] and entry["headers"]["X-Eyvan-User"] == "alice"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    reply = http.post("/api/v1/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "ping"}, headers=entry["headers"])
    assert reply.status_code == 200


def test_install_requires_the_xerxes_target(tmp_path) -> None:
    _http, _token, session, _client = _team(tmp_path)
    result = CliRunner().invoke(mcp, ["install", "--path", str(tmp_path / "x.json")], obj=session)
    assert result.exit_code == 2 and "--xerxes" in result.output


def test_private_network_classification_matches_xerxes() -> None:
    assert needs_private_network("http://127.0.0.1:8080/api/v1/mcp")
    assert needs_private_network("http://localhost:8080/api/v1/mcp")
    assert needs_private_network("http://10.1.2.3/api/v1/mcp")
    assert needs_private_network("http://[::1]:8080/api/v1/mcp")
    assert not needs_private_network("https://plane.example.com/api/v1/mcp")
    assert not needs_private_network("https://8.8.8.8/api/v1/mcp")


def test_the_group_is_registered_on_the_cli_root() -> None:
    from eyvan.sdk.cli import main

    assert "mcp" in main.commands and {"config", "install"} <= set(main.commands["mcp"].commands)  # type: ignore[attr-defined]
