# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""``eyvan mcp install/config --client``: each client's real file shape, merged owner-only.

Every test runs against a real in-process team plane and tmp paths only; the
user's real ``~/.claude.json``, ``~/.codex`` and ``~/.cursor`` are never read
or written (``HOME`` points into ``tmp_path``).
"""

from __future__ import annotations

import json
import re
import stat
import tomllib
import typing as tp
from datetime import UTC, datetime

import click
import pytest
from click.testing import CliRunner
from starlette.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.sdk.client.identity import ClientIdentity
from eyvan.sdk.commands import mcp as mcp_module
from eyvan.sdk.commands.mcp import mcp
from eyvan.sdk.commands.state import CliSession, ContextOverrides
from eyvan.sdk.contexts import FileContextStore
from eyvan.server.api.app import create_app

ORIGIN = "http://127.0.0.1:8080"
ADMIN = {"X-Eyvan-User": "root", "X-Eyvan-Roles": "fleet_admin"}
PASSWORD = "correct horse battery"  # pragma: allowlist secret


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("CODEX_HOME", raising=False)


class _AppClient:
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


def _team(tmp_path) -> tuple[TestClient, str, CliSession, _AppClient]:
    settings = EyvanSettings(dev_auth_enabled=True, team_mode=True, artifact_token_key="k" * 32)
    http = TestClient(create_app(settings=settings), base_url=ORIGIN)
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
    token = str(login.json()["token"])
    identity = ClientIdentity(token=token)
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
    return http, token, session, client


def _tokens(http: TestClient, token: str) -> list[dict[str, tp.Any]]:
    return http.get("/api/v1/team/tokens", headers={"Authorization": f"Bearer {token}"}).json()["tokens"]


def _authenticates(http: TestClient, headers: dict[str, str]) -> bool:
    reply = http.post("/api/v1/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, headers=headers)
    return reply.status_code == 200 and bool(reply.json()["result"]["tools"])


def test_claude_code_user_scope_entry_is_merged_into_claude_json(tmp_path) -> None:
    http, login, session, client = _team(tmp_path)
    path = tmp_path / "home" / ".claude.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {
        "numStartups": 7,
        "projects": {"/work": {"mcpServers": {"local": {"command": "x"}}, "history": [1, 2]}},
        "mcpServers": {"other": {"type": "stdio", "command": "other-server", "args": ["--x"]}},
    }
    path.write_text(json.dumps(existing), encoding="utf-8")

    result = CliRunner().invoke(mcp, ["install", "--client", "claude-code"], obj=session)  # default path
    assert result.exit_code == 0, result.output
    assert "eyv_" not in result.output and login not in result.output
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["numStartups"] == 7 and document["projects"] == existing["projects"]
    assert document["mcpServers"]["other"] == existing["mcpServers"]["other"]
    entry = document["mcpServers"]["eyvan"]
    assert set(entry) == {"type", "url", "headers"}
    assert entry["type"] == "http" and entry["url"] == f"{ORIGIN}/api/v1/mcp"
    assert entry["headers"]["X-Eyvan-Org"] == "acme" and entry["headers"]["X-Eyvan-Project"] == "r"
    assert entry["headers"]["Authorization"].startswith("Bearer eyv_")
    assert _authenticates(http, entry["headers"])
    minted = [item for item in _tokens(http, login) if item["name"].startswith("agent-claude-code-")]
    assert len(minted) == 1
    assert re.fullmatch(r"agent-claude-code-\d{8}-\d{4}-.+", minted[0]["name"]) and len(minted[0]["name"]) <= 64
    days = (datetime.fromisoformat(minted[0]["expires_at"]) - datetime.now(UTC)).total_seconds() / 86400
    assert 89 < days <= 90  # the dashboard's Connect-an-agent default, not "never"
    assert "expires in 90 days" in result.output

    before = path.read_bytes()
    refused = CliRunner().invoke(mcp, ["install", "--client", "claude-code"], obj=session)
    assert refused.exit_code != 0 and "--yes" in refused.output and path.read_bytes() == before
    assert sum(1 for call in client.calls if call == ("POST", "/api/v1/team/tokens")) == 1  # nothing minted for it
    replaced = CliRunner().invoke(mcp, ["install", "--client", "claude-code", "--yes"], obj=session)
    assert replaced.exit_code == 0, replaced.output
    again = json.loads(path.read_text(encoding="utf-8"))
    assert set(again["mcpServers"]) == {"other", "eyvan"} and again["projects"] == existing["projects"]
    assert again["mcpServers"]["eyvan"]["headers"]["Authorization"] != entry["headers"]["Authorization"]


@pytest.mark.parametrize(
    ("extra", "expected_days", "note"),
    [(["--expires-in-days", "30"], 30, "expires in 30 days"), (["--no-expiry"], None, "never expires")],
)
def test_install_token_lifetime_is_chosen_explicitly(
    tmp_path, extra: list[str], expected_days: int | None, note: str
) -> None:
    http, login, session, _client = _team(tmp_path)
    path = tmp_path / "cursor.json"
    result = CliRunner().invoke(mcp, ["install", "--client", "cursor", "--path", str(path), *extra], obj=session)
    assert result.exit_code == 0, result.output
    assert note in result.output
    (minted,) = [item for item in _tokens(http, login) if item["name"].startswith("agent-cursor-")]
    if expected_days is None:
        assert minted["expires_at"] is None
    else:
        days = (datetime.fromisoformat(minted["expires_at"]) - datetime.now(UTC)).total_seconds() / 86400
        assert expected_days - 1 < days <= expected_days


def test_install_refuses_an_out_of_range_lifetime(tmp_path) -> None:
    _http, _login, session, client = _team(tmp_path)
    result = CliRunner().invoke(
        mcp, ["install", "--client", "cursor", "--path", str(tmp_path / "c.json"), "--expires-in-days", "0"], obj=session
    )
    assert result.exit_code == 2
    assert ("POST", "/api/v1/team/tokens") not in client.calls


def test_agent_token_name_is_bounded_and_stamped() -> None:
    name = mcp_module.agent_token_name("claude-code", "h" * 200, datetime(2026, 11, 18, 9, 5))
    assert name.startswith("agent-claude-code-20261118-0905-") and len(name) == 64


def test_cursor_entry_and_a_fresh_file(tmp_path) -> None:
    http, _login, session, _client = _team(tmp_path)
    path = tmp_path / "home" / ".cursor" / "mcp.json"
    result = CliRunner().invoke(mcp, ["install", "--client", "cursor", "--name", "plane"], obj=session)
    assert result.exit_code == 0, result.output
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    document = json.loads(path.read_text(encoding="utf-8"))
    assert list(document) == ["mcpServers"] and list(document["mcpServers"]) == ["plane"]
    entry = document["mcpServers"]["plane"]
    assert set(entry) == {"url", "headers"} and entry["url"] == f"{ORIGIN}/api/v1/mcp"
    assert _authenticates(http, entry["headers"])

    path.write_text(json.dumps({"mcpServers": {"gh": {"command": "gh-mcp"}}, "theme": "dark"}), encoding="utf-8")
    merged = CliRunner().invoke(mcp, ["install", "--client", "cursor", "--name", "plane"], obj=session)
    assert merged.exit_code == 0, merged.output
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["theme"] == "dark" and set(document["mcpServers"]) == {"gh", "plane"}


CODEX_EXISTING = """# my codex settings
model = "o3"
approval_policy = "on-request"

[mcp_servers.docs]
command = "docs-mcp"
args = ["--port", "1"]

[mcp_servers.docs.env]
TOKEN_FILE = "/tmp/x"

[profiles.fast]
model = "o4-mini"
"""


def test_codex_table_is_written_without_disturbing_other_tables(tmp_path) -> None:
    http, _login, session, _client = _team(tmp_path)
    path = tmp_path / "home" / ".codex" / "config.toml"  # default: ~/.codex/config.toml
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CODEX_EXISTING, encoding="utf-8")
    before = tomllib.loads(CODEX_EXISTING)

    result = CliRunner().invoke(mcp, ["install", "--client", "codex"], obj=session)
    assert result.exit_code == 0, result.output
    assert "eyv_" not in result.output
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    text = path.read_text(encoding="utf-8")
    assert text.startswith(CODEX_EXISTING)  # every byte of the original kept, comments included
    after = tomllib.loads(text)
    entry = after["mcp_servers"].pop("eyvan")
    assert after == before
    assert set(entry) == {"url", "http_headers"} and entry["url"] == f"{ORIGIN}/api/v1/mcp"
    assert entry["http_headers"]["X-Eyvan-Org"] == "acme"
    assert _authenticates(http, entry["http_headers"])

    refused = CliRunner().invoke(mcp, ["install", "--client", "codex"], obj=session)
    assert refused.exit_code != 0 and "--yes" in refused.output and path.read_text(encoding="utf-8") == text
    replaced = CliRunner().invoke(mcp, ["install", "--client", "codex", "--yes"], obj=session)
    assert replaced.exit_code == 0, replaced.output
    final = tomllib.loads(path.read_text(encoding="utf-8"))
    new_entry = final["mcp_servers"].pop("eyvan")
    assert final == before and new_entry["http_headers"]["Authorization"] != entry["http_headers"]["Authorization"]
    assert path.read_text(encoding="utf-8").count("[mcp_servers.eyvan]") == 1


def test_codex_home_is_honoured(tmp_path, monkeypatch) -> None:
    _http, _login, session, _client = _team(tmp_path)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "codex-home"))
    result = CliRunner().invoke(mcp, ["install", "--client", "codex"], obj=session)
    assert result.exit_code == 0, result.output
    assert "eyvan" in tomllib.loads((tmp_path / "codex-home" / "config.toml").read_text())["mcp_servers"]


def test_codex_dotted_key_definition_is_refused_before_minting(tmp_path) -> None:
    _http, _login, session, client = _team(tmp_path)
    path = tmp_path / "config.toml"
    original = 'mcp_servers.eyvan.url = "http://elsewhere/mcp"\nmodel = "o3"\n'
    path.write_text(original, encoding="utf-8")
    result = CliRunner().invoke(mcp, ["install", "--client", "codex", "--path", str(path), "--yes"], obj=session)
    assert result.exit_code != 0 and "by hand" in result.output
    assert path.read_text(encoding="utf-8") == original
    assert ("POST", "/api/v1/team/tokens") not in client.calls


def test_invalid_toml_is_refused_without_echoing_it(tmp_path) -> None:
    _http, _login, session, client = _team(tmp_path)
    path = tmp_path / "config.toml"
    path.write_text('api_key = "sk-do-not-echo\n', encoding="utf-8")
    result = CliRunner().invoke(mcp, ["install", "--client", "codex", "--path", str(path)], obj=session)
    assert result.exit_code != 0 and "not valid TOML" in result.output and "sk-do-not-echo" not in result.output
    assert ("POST", "/api/v1/team/tokens") not in client.calls


@pytest.mark.parametrize("client_key", ["xerxes", "claude-code", "codex", "cursor"])
def test_the_minted_token_is_revoked_when_the_write_fails(tmp_path, monkeypatch, client_key: str) -> None:
    http, login, session, client = _team(tmp_path)
    path = tmp_path / f"{client_key}.cfg"

    def broken(target, text) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(mcp_module, "write_private_text", broken)
    result = CliRunner().invoke(mcp, ["install", "--client", client_key, "--path", str(path)], obj=session)
    assert result.exit_code != 0 and "disk full" in result.output and "revoked" in result.output
    assert "eyv_" not in result.output
    assert not path.exists()
    minted = [item for item in _tokens(http, login) if item["name"].startswith(f"agent-{client_key}-")]
    assert len(minted) == 1 and minted[0]["revoked_at"] is not None
    assert ("DELETE", f"/api/v1/team/tokens/{minted[0]['id']}") in client.calls


def test_an_interrupt_after_minting_still_revokes_the_token(tmp_path, monkeypatch) -> None:
    http, login, session, client = _team(tmp_path)
    path = tmp_path / "mcp.json"

    def interrupted(target, text) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(mcp_module, "write_private_text", interrupted)
    result = CliRunner().invoke(mcp, ["install", "--client", "cursor", "--path", str(path)], obj=session)
    assert result.exit_code != 0 and "eyv_" not in result.output
    assert not path.exists()
    minted = [item for item in _tokens(http, login) if item["name"].startswith("agent-cursor-")]
    assert len(minted) == 1 and minted[0]["revoked_at"] is not None
    assert ("DELETE", f"/api/v1/team/tokens/{minted[0]['id']}") in client.calls


def test_a_mint_answer_without_the_token_revokes_the_record(tmp_path) -> None:
    http, login, session, client = _team(tmp_path)
    path = tmp_path / "mcp.json"
    real = client.request

    def without_token(method: str, route: str, **kwargs: tp.Any) -> tp.Any:
        answer = real(method, route, **kwargs)
        if (method, route) == ("POST", "/api/v1/team/tokens"):
            return {key: value for key, value in answer.items() if key != "token"}
        return answer

    client.request = without_token  # type: ignore[method-assign]
    result = CliRunner().invoke(mcp, ["install", "--client", "xerxes", "--path", str(path)], obj=session)
    assert result.exit_code != 0 and "unexpected answer" in result.output
    assert not path.exists()
    minted = [item for item in _tokens(http, login) if item["name"].startswith("agent-xerxes-")]
    assert len(minted) == 1 and minted[0]["revoked_at"] is not None


def test_install_help_warns_about_claude_code_rewrites(tmp_path) -> None:
    result = CliRunner().invoke(mcp, ["install", "--help"])
    assert result.exit_code == 0 and "Claude Code rewrites" in result.output


def test_xerxes_alias_and_client_choice(tmp_path) -> None:
    _http, _login, session, _client = _team(tmp_path)
    path = tmp_path / "mcp.json"
    alias = CliRunner().invoke(mcp, ["install", "--xerxes", "--path", str(path)], obj=session)
    assert alias.exit_code == 0, alias.output
    assert json.loads(path.read_text())["eyvan"]["transport"] == "streamable_http"
    both = CliRunner().invoke(mcp, ["install", "--xerxes", "--client", "codex", "--path", str(path)], obj=session)
    assert both.exit_code == 2
    missing = CliRunner().invoke(mcp, ["install", "--path", str(path)], obj=session)
    assert missing.exit_code == 2 and "--client" in missing.output
    bad_name = CliRunner().invoke(
        mcp, ["install", "--client", "codex", "--name", "a.b", "--path", str(path)], obj=session
    )
    assert bad_name.exit_code == 2


@pytest.mark.parametrize(
    "client_key,expected",
    [
        ("claude-code", {"type": "http", "url": f"{ORIGIN}/api/v1/mcp"}),
        ("cursor", {"url": f"{ORIGIN}/api/v1/mcp"}),
        ("codex", {"url": f"{ORIGIN}/api/v1/mcp"}),
    ],
)
def test_config_prints_each_clients_shape_redacted(tmp_path, client_key: str, expected: dict[str, str]) -> None:
    _http, login, session, client = _team(tmp_path)
    result = CliRunner().invoke(mcp, ["config", "--client", client_key, "--json"], obj=session)
    assert result.exit_code == 0, result.output
    assert login not in result.output and "eyv_" not in result.output
    entry = json.loads(result.output)["eyvan"]
    assert expected.items() <= entry.items()
    headers = entry["http_headers" if client_key == "codex" else "headers"]
    assert headers["Authorization"] == "Bearer <redacted>"
    snippet = CliRunner().invoke(mcp, ["config", "--client", client_key], obj=session)
    assert snippet.exit_code == 0 and "eyv_" not in snippet.output
    if client_key == "codex":  # pragma: allowlist secret
        assert "[mcp_servers.eyvan]" in snippet.output
    else:
        assert '"mcpServers"' in snippet.output
    assert ("POST", "/api/v1/team/tokens") not in client.calls
