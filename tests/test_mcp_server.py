# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The plane's MCP tool server over real HTTP: lifecycle, auth, scope, errors and bounds.

Proves: the JSON-RPC lifecycle Xerxes's Streamable HTTP client drives;
401 without a credential; a team token's tools run as that user (submissions
are theirs, another user gets the REST API's own refusal); tool failures are
``isError`` results; bodies and results are bounded; cross-origin browsers are
refused.
"""

from __future__ import annotations

import json
import typing as tp

import httpx
import pytest
from starlette.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.records import Principal, Role, RoleGrant
from eyvan.server.api import mcp_routes
from eyvan.server.api.app import create_app

ORIGIN = "http://127.0.0.1:8080"
ADMIN = {"X-Eyvan-User": "root", "X-Eyvan-Roles": "fleet_admin"}
DEV_ALICE = {"X-Eyvan-User": "alice", "X-Eyvan-Roles": "project_operator", "X-Eyvan-Org": "acme", "X-Eyvan-Project": "r"}
PASSWORD = "correct horse battery"
ACCEPT = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
HELLO = {"name": "hello", "execution": {"kind": "command", "command": "echo hi"}}


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch) -> None:
    # The tracker lookup and credential stores read ~/.eyvan; never the user's.
    monkeypatch.setenv("HOME", str(tmp_path))


def _client(*, team_mode: bool = True, **settings: tp.Any) -> TestClient:
    configured = EyvanSettings(dev_auth_enabled=True, team_mode=team_mode, artifact_token_key="k" * 32, **settings)
    return TestClient(create_app(settings=configured), base_url=ORIGIN)


def _bearer(http: TestClient, subject: str, *, project: str = "r", role: str = "project_operator") -> dict[str, str]:
    """Onboard ``subject`` with one grant on acme/``project`` and return their personal-token header."""

    created = http.post(
        "/api/v1/team/invites",
        json={
            "email": f"{subject}@example.com",
            "subject": subject,
            "role": role,
            "org_id": "acme",
            "project_id": project,
        },
        headers=ADMIN,
    )
    assert created.status_code == 201, created.text
    accepted = http.post("/api/v1/team/invites/accept", json={"token": created.json()["token"], "password": PASSWORD})
    assert accepted.status_code == 200, accepted.text
    http.cookies.clear()
    login = http.post(
        "/api/v1/team/login", json={"email": f"{subject}@example.com", "password": PASSWORD, "token_name": "mcp"}
    )
    assert login.status_code == 200, login.text
    return {"Authorization": f"Bearer {login.json()['token']}"}


def _rpc(
    http: TestClient,
    method: str,
    params: dict[str, tp.Any] | None = None,
    *,
    headers: dict[str, str],
    message_id: int = 1,
) -> httpx.Response:
    frame: dict[str, tp.Any] = {"jsonrpc": "2.0", "id": message_id, "method": method}
    if params is not None:
        frame["params"] = params
    return http.post("/api/v1/mcp", json=frame, headers={**ACCEPT, **headers})


def _call(http: TestClient, name: str, arguments: dict[str, tp.Any], *, headers: dict[str, str]) -> tuple[bool, tp.Any]:
    """Call one tool; return ``(isError, decoded text)``."""

    response = _rpc(
        http,
        "tools/call",
        {"name": name, "arguments": arguments},
        headers={**headers, "MCP-Protocol-Version": "2025-06-18"},
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert [item["type"] for item in result["content"]] == ["text"]
    return result["isError"], json.loads(result["content"][0]["text"])


def test_the_full_lifecycle_xerxes_drives() -> None:
    http = _client()
    alice = _bearer(http, "alice")

    initialized = _rpc(
        http,
        "initialize",
        {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "xerxes", "version": "1"}},
        headers=alice,
    )
    assert initialized.status_code == 200
    assert initialized.headers["content-type"].startswith("application/json")
    assert "mcp-session-id" not in initialized.headers  # stateless
    result = initialized.json()["result"]
    assert result["protocolVersion"] == "2025-06-18"
    assert "tools" in result["capabilities"]
    assert result["serverInfo"]["name"] == "eyvan" and result["serverInfo"]["version"]

    older = _rpc(http, "initialize", {"protocolVersion": "2024-11-05"}, headers=alice).json()["result"]
    assert older["protocolVersion"] == "2024-11-05"
    unknown = _rpc(http, "initialize", {"protocolVersion": "1999-01-01"}, headers=alice).json()["result"]
    assert unknown["protocolVersion"] == mcp_routes.SUPPORTED_PROTOCOL_VERSIONS[0]

    note = http.post(
        "/api/v1/mcp",
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
        headers={**ACCEPT, **alice, "MCP-Protocol-Version": "2025-06-18"},
    )
    assert note.status_code == 202 and note.content == b""

    assert _rpc(http, "ping", headers=alice).json() == {"jsonrpc": "2.0", "id": 1, "result": {}}

    listed = _rpc(http, "tools/list", {}, headers=alice).json()["result"]["tools"]
    names = [tool["name"] for tool in listed]
    for expected in (
        "list_fleets",
        "size_job",
        "list_easydel_trainers",
        "draft_workload",
        "file_request",
        "submit_workload",
        "get_workload",
        "cancel_workload",
        "get_workload_logs",
        "list_runs",
        "get_run_metrics",
        "draft_easydel_launch",
    ):
        assert expected in names
    assert len(names) == len(set(names))
    by_name = {tool["name"]: tool for tool in listed}
    assert by_name["submit_workload"]["inputSchema"]["required"] == ["spec"]
    assert "org_id" in by_name["submit_workload"]["inputSchema"]["properties"]
    assert all(tool["inputSchema"]["type"] == "object" for tool in listed)

    failed, trainers = _call(http, "list_easydel_trainers", {}, headers=alice)
    assert not failed and trainers["trainers"]
    failed, sized = _call(http, "size_job", {"parameters": 8e9, "trainer": "sft"}, headers=alice)
    assert not failed and sized["recommended"]

    assert http.get("/api/v1/mcp", headers=alice).status_code == 405
    assert http.delete("/api/v1/mcp", headers=alice).status_code == 405


def test_unauthenticated_requests_get_401() -> None:
    http = _client()
    response = _rpc(http, "initialize", {"protocolVersion": "2025-06-18"}, headers={})
    assert response.status_code == 401
    wrong = _rpc(http, "tools/list", {}, headers={"Authorization": "Bearer eyv_" + "x" * 43})
    assert wrong.status_code == 401
    assert "eyv_" + "x" * 43 not in wrong.text


def test_a_team_token_acts_as_its_user_and_nobody_else() -> None:
    http = _client()
    alice = _bearer(http, "alice")
    bob = _bearer(http, "bob", project="other")

    # The single granted scope is the default; no org/project needed.
    failed, submitted = _call(http, "submit_workload", {"spec": HELLO, "idempotency_key": "k-1"}, headers=alice)
    assert not failed, submitted
    workload = submitted["workload"]
    assert workload["created_by"] == "alice" and (workload["org_id"], workload["project_id"]) == ("acme", "r")
    assert submitted["replayed"] is False
    stored = http.get(f"/api/v1/workloads/{workload['id']}", headers=alice).json()
    assert stored["created_by"] == "alice" and stored["state"] == "pending"

    # The same key replays the same submission rather than creating another.
    failed, again = _call(http, "submit_workload", {"spec": HELLO, "idempotency_key": "k-1"}, headers=alice)
    assert not failed and again["replayed"] is True and again["workload"]["id"] == workload["id"]
    assert len(http.get("/api/v1/workloads", params={"org_id": "acme", "project_id": "r"}, headers=alice).json()) == 1

    # Bob gets exactly what the REST API answers him: 403, and nothing changes.
    rest_read = http.get(f"/api/v1/workloads/{workload['id']}", headers=bob)
    rest_cancel = http.post(f"/api/v1/workloads/{workload['id']}/cancel", headers={**bob, "Idempotency-Key": "b"})
    assert rest_read.status_code == 403 and rest_cancel.status_code == 403
    failed, read = _call(http, "get_workload", {"workload_id": workload["id"]}, headers=bob)
    assert failed and read["status"] == rest_read.status_code and read["type"] == "AuthorizationError"
    failed, cancel = _call(http, "cancel_workload", {"workload_id": workload["id"]}, headers=bob)
    assert failed and cancel["status"] == rest_cancel.status_code
    failed, logs = _call(http, "get_workload_logs", {"workload_id": workload["id"]}, headers=bob)
    assert failed and logs["status"] == 403
    # Naming alice's scope explicitly does not help bob either.
    failed, listing = _call(http, "list_workloads", {"org_id": "acme", "project_id": "r"}, headers=bob)
    assert failed and listing["status"] == 403
    failed, sneaky = _call(http, "submit_workload", {"spec": HELLO, "org_id": "acme", "project_id": "r"}, headers=bob)
    assert failed and sneaky["status"] == 403
    assert http.get(f"/api/v1/workloads/{workload['id']}", headers=alice).json()["state"] == "pending"

    failed, own = _call(http, "get_workload", {"workload_id": workload["id"]}, headers=alice)
    assert not failed and own["id"] == workload["id"]
    failed, logs = _call(http, "get_workload_logs", {"workload_id": workload["id"], "tail": 5}, headers=alice)
    assert not failed and "not been dispatched" in logs["logs"] and logs["lines_shown"] <= 5
    failed, mine = _call(http, "list_workloads", {}, headers=alice)
    assert not failed and [item["id"] for item in mine["workloads"]] == [workload["id"]]

    failed, cancelled = _call(http, "cancel_workload", {"workload_id": workload["id"]}, headers=alice)
    assert not failed and cancelled["cancel_requested"] is True
    assert http.get(f"/api/v1/workloads/{workload['id']}", headers=alice).json()["state"] == "cancelled"


def test_a_draft_comes_back_whole_and_submits_as_is() -> None:
    http = _client()
    failed, drafted = _call(
        http, "draft_workload", {"name": "train", "command": "python train.py", "tpu_chips": 4}, headers=DEV_ALICE
    )
    assert not failed and drafted["drafted"] is True
    draft = drafted["draft"]
    assert (draft["org_id"], draft["project_id"]) == ("acme", "r")
    assert http.get("/api/v1/workloads", params={"org_id": "acme", "project_id": "r"}, headers=ADMIN).json() == []
    failed, submitted = _call(http, "submit_workload", {"spec": draft["spec"]}, headers=DEV_ALICE)
    assert not failed and submitted["workload"]["name"] == "train"
    assert submitted["workload"]["created_by"] == "alice"

    # draft_easydel_launch insists on the EasyDeL fields instead of drafting a bare command.
    failed, refused = _call(http, "draft_easydel_launch", {"name": "x", "easydel_trainer": "sft"}, headers=DEV_ALICE)
    assert failed and "easydel_launch" in refused["error"]


def test_tool_failures_are_is_error_results_and_protocol_errors_are_json_rpc() -> None:
    http = _client()
    headers = {**DEV_ALICE, **ACCEPT}

    unknown_method = _rpc(http, "resources/list", {}, headers=DEV_ALICE).json()
    assert unknown_method["error"]["code"] == -32601 and unknown_method["id"] == 1

    parse = http.post("/api/v1/mcp", content=b"{not json", headers=headers)
    assert parse.status_code == 400 and parse.json()["error"]["code"] == -32700

    batch = http.post(
        "/api/v1/mcp", content=json.dumps([{"jsonrpc": "2.0", "id": 1, "method": "ping"}]), headers=headers
    )
    assert (
        batch.status_code == 400
        and batch.json()["error"]["code"] == -32600
        and "batch" in batch.json()["error"]["message"]
    )

    not_rpc = http.post("/api/v1/mcp", json={"id": 1, "method": "ping"}, headers=headers)
    assert not_rpc.status_code == 400 and not_rpc.json()["error"]["code"] == -32600

    unknown_tool = _rpc(http, "tools/call", {"name": "rm_rf", "arguments": {}}, headers=DEV_ALICE).json()
    assert unknown_tool["error"]["code"] == -32602

    bad_version = http.post(
        "/api/v1/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        headers={**headers, "MCP-Protocol-Version": "1999-01-01"},
    )
    assert bad_version.status_code == 400

    failed, missing = _call(http, "get_workload", {"workload_id": "wl_does_not_exist"}, headers=DEV_ALICE)
    assert failed and missing["status"] == 404
    failed, invalid = _call(http, "submit_workload", {"spec": {"name": "x"}}, headers=DEV_ALICE)
    assert failed and invalid["status"] == 422 and "execution" in invalid["error"]
    failed, extra = _call(http, "get_workload", {"workload_id": "w", "surprise": 1}, headers=DEV_ALICE)
    assert failed and "surprise" in extra["error"]
    failed, required = _call(http, "get_workload", {}, headers=DEV_ALICE)
    assert failed and "workload_id" in required["error"]
    failed, sizing = _call(http, "size_job", {}, headers=DEV_ALICE)
    assert failed and sizing["status"] == 400

    # No tracker configured: a clear refusal, not a crash.
    failed, runs = _call(http, "list_runs", {}, headers=DEV_ALICE)
    assert failed and runs["status"] == 503 and "not configured" in runs["error"]


def test_run_metrics_are_read_through_the_tracker_in_the_callers_scope() -> None:
    seen: list[httpx.Request] = []

    def tracker(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.path == "/api/v1/tracking/runs":
            return httpx.Response(200, json={"runs": [{"run_id": "run-1", "name": "ppo", "status": "running"}]})
        if request.url.path == "/api/v1/tracking/runs/run-1":
            return httpx.Response(200, json={"run_id": "run-1", "keys": ["loss", "reward"], "summary": {}})
        if request.url.path == "/api/v1/tracking/runs/run-1/series":
            keys = request.url.params.get_list("key")
            return httpx.Response(
                200,
                json={"series": [{"key": key, "rank": 0, "points": [[0, 1.0, 1.0, 1.0]], "total": 1} for key in keys]},
            )
        return httpx.Response(404, json={"detail": "run not found"})

    http = _client(tracker_url="http://tracker.test", tracker_token="tracker-secret")
    http.app.state.tracking_transport = httpx.MockTransport(tracker)

    failed, runs = _call(http, "list_runs", {}, headers=DEV_ALICE)
    assert not failed and runs["runs"][0]["run_id"] == "run-1"
    failed, metrics = _call(http, "get_run_metrics", {"run_id": "run-1", "keys": ["loss"]}, headers=DEV_ALICE)
    assert not failed and [series["key"] for series in metrics["series"]] == ["loss"]
    assert metrics["available_keys"] == ["loss", "reward"]
    assert all(request.url.params["org_id"] == "acme" and request.url.params["project_id"] == "r" for request in seen)
    assert seen[0].headers["Authorization"] == "Bearer tracker-secret"
    assert "tracker-secret" not in json.dumps(metrics)

    failed, absent = _call(http, "get_run_metrics", {"run_id": "nope"}, headers=DEV_ALICE)
    assert failed and absent["status"] == 404
    failed, foreign = _call(http, "list_runs", {"org_id": "other", "project_id": "x"}, headers=DEV_ALICE)
    assert failed and foreign["status"] == 403


def test_request_bodies_and_tool_results_are_bounded() -> None:
    http = _client()
    oversized = (
        b'{"jsonrpc":"2.0","id":1,"method":"ping","params":{"pad":"' + b"x" * mcp_routes.MAX_REQUEST_BYTES + b'"}}'
    )
    response = http.post("/api/v1/mcp", content=oversized, headers={**DEV_ALICE, **ACCEPT})
    assert response.status_code == 413

    async def huge(service: tp.Any, principal: tp.Any, scope: tp.Any, args: tp.Any) -> tp.Any:
        return {"blob": "x" * 50_000}

    mcp_routes.register_tool(mcp_routes.McpTool("huge_test_tool", "test only", {"type": "object"}, huge))
    try:
        reply = _rpc(http, "tools/call", {"name": "huge_test_tool", "arguments": {}}, headers=DEV_ALICE).json()
    finally:
        mcp_routes.TOOLS.pop("huge_test_tool", None)
    text = reply["result"]["content"][0]["text"]
    assert reply["result"]["isError"] is False
    assert len(text) <= mcp_routes.MAX_RESULT_CHARS and "truncated" in text

    assert mcp_routes.bounded_text({"a": 1}) == '{"a":1}'


def test_cross_origin_browsers_are_refused() -> None:
    http = _client()
    evil = http.post(
        "/api/v1/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        headers={**DEV_ALICE, **ACCEPT, "Origin": "https://evil.example"},
    )
    assert evil.status_code == 403
    # DNS rebinding: an attacker's name in both Host and Origin.
    rebound = http.post(
        "/api/v1/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        headers={**DEV_ALICE, **ACCEPT, "Origin": "http://attacker.example:8080", "Host": "attacker.example:8080"},
    )
    assert rebound.status_code == 403
    own = http.post(
        "/api/v1/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        headers={**DEV_ALICE, **ACCEPT, "Origin": ORIGIN},
    )
    assert own.status_code == 200
    assert mcp_routes.origin_allowed(
        "https://plane.example", public_url="https://plane.example/", host_header="plane.example", scheme="https"
    )
    assert not mcp_routes.origin_allowed("null", public_url=ORIGIN, host_header="127.0.0.1:8080", scheme="http")


def test_scope_defaults_only_when_unambiguous() -> None:
    one = Principal(subject="a", grants=(RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="r"),))
    two = Principal(
        subject="b",
        grants=(
            RoleGrant(role=Role.PROJECT_OPERATOR, org_id="acme", project_id="r"),
            RoleGrant(role=Role.PROJECT_VIEWER, org_id="acme", project_id="s"),
        ),
    )
    admin = Principal(subject="c", grants=(RoleGrant(role=Role.FLEET_ADMIN),))
    assert mcp_routes._granted_scope(one)[:2] == ("acme", "r")
    assert mcp_routes._granted_scope(two)[:2] == (None, None)
    assert mcp_routes._granted_scope(admin)[:2] == ("default", "default")
    with pytest.raises(mcp_routes.McpToolError, match="acme/r, acme/s"):
        mcp_routes.McpScope(None, None, candidates=mcp_routes._granted_scope(two)[2]).require()
