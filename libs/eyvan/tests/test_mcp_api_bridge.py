# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Generic MCP API tools: the plane's own routes, dispatched in process, as the caller.

Proves: a viewer token gets the route's own 403 where a submitter succeeds;
another tenant is refused; every denylisted route, the MCP endpoint itself and
unknown/uncatalogued paths are refused before dispatch; the forwarded request
carries no cookies, Origin or fetch metadata, keeps the original client
address and cannot be mistaken for the plane's own dashboard; the listing is
filtered and bounded; annotations describe what each tool does.
"""

from __future__ import annotations

import asyncio
import json
import typing as tp

import pytest
from fastapi import Depends, Header, Request
from starlette.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.records import Principal
from eyvan.server.api import mcp_api_bridge, mcp_routes
from eyvan.server.api.app import create_app
from eyvan.server.api.auth import authenticated_principal

ORIGIN = "http://127.0.0.1:8080"
ADMIN = {"X-Eyvan-User": "root", "X-Eyvan-Roles": "fleet_admin"}
PASSWORD = "correct horse battery"  # pragma: allowlist secret
ACCEPT = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
HELLO = {"name": "hello", "execution": {"kind": "command", "command": "echo hi"}}


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))


def _client(**settings: tp.Any) -> TestClient:
    settings.setdefault("dev_auth_enabled", True)
    configured = EyvanSettings(team_mode=True, artifact_token_key="k" * 32, **settings)
    return TestClient(create_app(settings=configured), base_url=ORIGIN)


def _bearer(
    http: TestClient,
    subject: str,
    *,
    project: str = "r",
    role: str = "project_operator",
    admin: dict[str, str] = ADMIN,
) -> dict[str, str]:
    created = http.post(
        "/api/v1/team/invites",
        json={
            "email": f"{subject}@example.com",
            "subject": subject,
            "role": role,
            "org_id": "acme",
            "project_id": project,
        },
        headers=admin,
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


def _call(http: TestClient, name: str, arguments: dict[str, tp.Any], *, headers: dict[str, str]) -> tuple[bool, tp.Any]:
    response = http.post(
        "/api/v1/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}},
        headers={**ACCEPT, "MCP-Protocol-Version": "2025-06-18", **headers},
    )
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    return result["isError"], json.loads(result["content"][0]["text"])


def _submit(http: TestClient, headers: dict[str, str], **extra: tp.Any) -> tuple[bool, tp.Any]:
    return _call(
        http,
        "eyvan_api_call",
        {"method": "POST", "path": "/api/v1/workloads", "body": {"org_id": "acme", "project_id": "r", "spec": HELLO}}
        | extra,
        headers=headers,
    )


def test_a_viewer_gets_the_routes_own_403_where_a_submitter_succeeds() -> None:
    http = _client()
    viewer = _bearer(http, "vera", role="project_viewer")
    submitter = _bearer(http, "sam", role="project_submitter")

    failed, refused = _submit(http, viewer)
    assert failed and refused["status"] == 403, refused
    assert http.get("/api/v1/workloads", params={"org_id": "acme", "project_id": "r"}, headers=ADMIN).json() == []

    failed, created = _submit(http, submitter, idempotency_key="sub-1")
    assert not failed, created
    assert created["status"] == 201 and created["idempotency_key"] == "sub-1"
    workload = created["body"]
    assert workload["created_by"] == "sam"
    stored = http.get(f"/api/v1/workloads/{workload['id']}", headers=ADMIN).json()
    assert stored["created_by"] == "sam" and (stored["org_id"], stored["project_id"]) == ("acme", "r")

    # The route's own idempotency: the same key replays, it does not duplicate.
    failed, again = _submit(http, submitter, idempotency_key="sub-1")
    assert not failed and again["replayed"] is True and again["body"]["id"] == workload["id"]
    # Without a key the bridge supplies a fresh one and reports it.
    failed, fresh = _submit(http, submitter)
    assert not failed and fresh["idempotency_key"].startswith("mcp-") and fresh["body"]["id"] != workload["id"]

    # The viewer may read what the REST API lets them read; org/project default to their scope.
    failed, listed = _call(http, "eyvan_api_get", {"path": "/api/v1/workloads"}, headers=viewer)
    assert not failed and listed["status"] == 200
    assert {item["id"] for item in listed["body"]} == {workload["id"], fresh["body"]["id"]}
    failed, cancel = _call(
        http, "eyvan_api_call", {"method": "POST", "path": f"/api/v1/workloads/{workload['id']}/cancel"}, headers=viewer
    )
    assert failed and cancel["status"] == 403
    assert http.get(f"/api/v1/workloads/{workload['id']}", headers=ADMIN).json()["state"] == "pending"

    # Validation failures come back with the route's status and detail.
    failed, invalid = _call(
        http,
        "eyvan_api_call",
        {"method": "POST", "path": "/api/v1/workloads", "body": {"org_id": "acme", "project_id": "r", "spec": {}}},
        headers=submitter,
    )
    assert failed and invalid["status"] == 422 and "spec" in invalid["error"]


def test_another_tenant_is_refused_by_the_route() -> None:
    http = _client()
    alice = _bearer(http, "alice")
    bob = _bearer(http, "bob", project="other")
    failed, created = _submit(http, alice)
    assert not failed
    workload_id = created["body"]["id"]

    failed, read = _call(http, "eyvan_api_get", {"path": f"/api/v1/workloads/{workload_id}"}, headers=bob)
    assert failed and read["status"] == 403
    failed, listing = _call(
        http, "eyvan_api_get", {"path": "/api/v1/workloads", "query": {"org_id": "acme", "project_id": "r"}}, headers=bob
    )
    assert failed and listing["status"] == 403
    failed, cancel = _call(
        http, "eyvan_api_call", {"method": "POST", "path": f"/api/v1/workloads/{workload_id}/cancel"}, headers=bob
    )
    assert failed and cancel["status"] == 403
    failed, own = _call(http, "eyvan_api_get", {"path": "/api/v1/workloads"}, headers=bob)
    assert not failed and own["body"] == []  # bob's default scope is acme/other
    assert http.get(f"/api/v1/workloads/{workload_id}", headers=ADMIN).json()["state"] == "pending"


DENIED: tp.Final = (
    ("POST", "/api/v1/mcp"),
    ("POST", "/api/v1/team/login"),
    ("POST", "/api/v1/team/logout"),
    ("POST", "/api/v1/team/invites/accept"),
    ("POST", "/api/v1/team/invites/preview"),
    ("GET", "/api/v1/team/status"),
    ("GET", "/api/v1/team/tokens"),
    ("POST", "/api/v1/team/tokens"),
    ("DELETE", "/api/v1/team/tokens/tok_0123456789abcdef"),  # pragma: allowlist secret
    ("GET", "/api/v1/dashboard/auth/login"),
    ("POST", "/api/v1/dashboard/auth/logout"),
    ("GET", "/api/v1/dashboard/config"),
    ("POST", "/api/v1/assistant/oauth/openrouter/start"),
    ("GET", "/api/v1/assistant/oauth/openrouter/callback/state"),
    ("POST", "/api/v1/assistant/claude-code/login"),
    ("GET", "/api/v1/events/stream"),
    ("GET", "/api/v1/workloads/wl_x/logs/stream"),
    ("POST", "/api/v1/assistant/chat/stream"),
    ("GET", "/api/v1/workloads/wl_x/logs"),
    ("GET", "/api/v1/artifacts/results/res_x"),
    ("PUT", "/api/v1/artifacts/callables/abc"),
    ("PUT", "/api/v1/artifacts/workspaces/abc"),
    ("GET", "/api/v1/tracking/runs/run-1/blobs/abc"),
    ("PUT", "/api/v1/artifacts/results/res_x/content"),
    ("GET", "/api/v1/artifacts/callables/abc/content"),
    ("GET", "/api/v1/artifacts/workspaces/abc/content"),
    ("GET", "/api/v1/secrets/content"),
    ("POST", "/api/v1/workloads/wl_x/progress-reports"),
    ("PUT", "/api/v1/workloads/wl_x/endpoint"),
    ("POST", "/api/v1/workloads/wl_x/children"),
    ("GET", "/api/v1/attachments"),
    ("POST", "/api/v1/attachments"),
    ("POST", "/api/v1/attachments/plan"),
    ("GET", "/api/v1/attachments/discovered"),
    ("GET", "/api/v1/attachments/att_x"),
    ("POST", "/api/v1/query"),
    ("POST", "/api/v1/secrets/rotate"),
    ("PUT", "/api/v1/team/users/alice/rules"),
    ("PUT", "/api/v1/federation/peers/peer-1"),
    ("POST", "/api/v1/federation/budget-grants"),
    ("POST", "/api/v1/federation/quota-grants"),
    ("GET", "/metrics"),
    ("GET", "/healthz"),
    ("GET", "/readyz"),
)


def _bridge(http: TestClient, method: str, path: str, headers: dict[str, str], **extra: tp.Any) -> tuple[bool, tp.Any]:
    if method == "GET":
        return _call(http, "eyvan_api_get", {"path": path, **extra}, headers=headers)
    return _call(http, "eyvan_api_call", {"method": method, "path": path, **extra}, headers=headers)


@pytest.mark.parametrize("method,path", DENIED)
def test_every_denylisted_route_is_refused_before_dispatch(method: str, path: str) -> None:
    http = _client()
    alice = _bearer(http, "alice")
    tokens_before = http.get("/api/v1/team/tokens", headers=alice).json()["tokens"]
    extra = {} if method == "GET" else {"body": {"name": "x"}}
    failed, refused = _bridge(http, method, path, alice, **extra)
    assert failed and refused["status"] == 403, refused
    assert "refused" in refused["error"]
    assert http.get("/api/v1/team/tokens", headers=alice).json()["tokens"] == tokens_before


def test_a_fleet_admin_token_cannot_create_accounts_or_grant_authority_through_the_bridge() -> None:
    http = _client()
    # A real team fleet admin with a personal bearer token.
    root = _bearer(http, "boss", role="fleet_admin")
    users_before = http.get("/api/v1/team/users", headers=ADMIN).json()
    invites_before = http.get("/api/v1/team/invites", headers=ADMIN).json()

    failed, invite = _call(
        http,
        "eyvan_api_call",
        {
            "method": "POST",
            "path": "/api/v1/team/invites",
            "body": {"email": "mallory@example.com", "subject": "mallory", "role": "fleet_admin"},
        },
        headers=root,
    )
    assert failed and invite["status"] == 403 and "account creation" in invite["error"], invite
    assert http.get("/api/v1/team/invites", headers=ADMIN).json() == invites_before
    assert http.get("/api/v1/team/users", headers=ADMIN).json() == users_before

    # Listing invites stays available (the denial is per method).
    failed, listed = _call(http, "eyvan_api_get", {"path": "/api/v1/team/invites"}, headers=root)
    assert not failed and listed["status"] == 200

    denied = (
        ("POST", "/api/v1/grants", {"subject": "mallory", "grant": {"role": "fleet_admin"}}),
        ("PUT", "/api/v1/team/users/boss/tier", {"tier": "admin"}),
        ("POST", "/api/v1/team/users/boss/enable", None),
    )
    for method, path, body in denied:
        arguments: dict[str, tp.Any] = {"method": method, "path": path}
        if body is not None:
            arguments["body"] = body
        failed, refused = _call(http, "eyvan_api_call", arguments, headers=root)
        assert failed and refused["status"] == 403 and "refused" in refused["error"], (path, refused)
    assert all(grant["subject"] != "mallory" for grant in http.get("/api/v1/grants", headers=ADMIN).json())

    failed, catalogue = _call(http, "eyvan_api_operations", {"search": "/api/v1/team/invites"}, headers=root)
    assert not failed
    listed_ops = {(item["method"], item["path"]) for item in catalogue["operations"]}
    assert ("GET", "/api/v1/team/invites") in listed_ops and ("POST", "/api/v1/team/invites") not in listed_ops


def test_no_result_echoes_the_callers_token() -> None:
    http = _client()
    alice = _bearer(http, "alice")
    raw = alice["Authorization"].removeprefix("Bearer ")
    _install_echo_route(http)
    # The echo route returns the Authorization header it received; the result must not carry it.
    response = http.post(
        "/api/v1/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": "eyvan_api_get", "arguments": {"path": "/api/v1/zz-test-echo"}},
        },
        headers={**ACCEPT, **alice},
    )
    assert response.status_code == 200
    assert raw not in response.text and raw[:20] not in response.text
    echoed = json.loads(response.json()["result"]["content"][0]["text"])
    assert echoed["body"]["subject"] == "alice"
    assert "redacted" in echoed["body"]["headers"]["authorization"]

    # Error paths too: arguments echoed back in an error never carry the token.
    for arguments in ({"path": f"/api/v1/{raw}"}, {"path": "/api/v1/workloads", "query": {raw[:60]: 1}}):
        reply = http.post(
            "/api/v1/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "eyvan_api_get", "arguments": arguments},
            },
            headers={**ACCEPT, **alice},
        )
        assert reply.status_code == 200 and raw not in reply.text and raw[:20] not in reply.text

    assert mcp_routes.redact_secrets("x Bearer abcdefghij y", ["Bearer abcdefghij"]) == "x <redacted> y"
    assert "eyv_" + "z" * 43 not in mcp_routes.redact_secrets("t=eyv_" + "z" * 43)


def test_listing_children_is_bridged_while_submitting_one_is_not() -> None:
    http = _client()
    alice = _bearer(http, "alice")
    failed, created = _submit(http, alice)
    assert not failed, created
    workload_id = created["body"]["id"]
    failed, children = _bridge(http, "GET", f"/api/v1/workloads/{workload_id}/children", alice)
    assert not failed and children["status"] == 200 and children["body"] == [], children
    failed, refused = _bridge(http, "POST", f"/api/v1/workloads/{workload_id}/children", alice, body={"spec": HELLO})
    assert failed and refused["status"] == 403 and "refused" in refused["error"]


def test_a_failed_write_returns_its_idempotency_key_and_says_retrying_is_safe() -> None:
    http = _client()
    alice = _bearer(http, "alice")
    failed, invalid = _call(
        http,
        "eyvan_api_call",
        {"method": "POST", "path": "/api/v1/workloads", "body": {"org_id": "acme", "project_id": "r", "spec": {}}},
        headers=alice,
    )
    assert failed and invalid["status"] == 422
    assert invalid["idempotency_key"].startswith("mcp-") and "safe" in invalid["error"]
    failed, chosen = _call(
        http,
        "eyvan_api_call",
        {
            "method": "POST",
            "path": "/api/v1/workloads/wl_missing/cancel",
            "idempotency_key": "agent-key-1",
        },
        headers=alice,
    )
    assert failed and chosen["status"] in {403, 404} and chosen["idempotency_key"] == "agent-key-1"


def test_a_write_that_times_out_keeps_running_and_reports_its_key(monkeypatch) -> None:
    http = _client()
    app = http.app
    finished: list[str] = []

    async def slow(
        principal: tp.Annotated[Principal, Depends(authenticated_principal)],
        idempotency_key: tp.Annotated[str, Header(alias="Idempotency-Key")],
    ) -> dict:
        await asyncio.sleep(0.3)
        finished.append(idempotency_key)
        return {"ok": True}

    app.add_api_route("/api/v1/zz-test-slow", slow, methods=["POST"], tags=["test"])  # type: ignore[attr-defined]
    app.router.routes.insert(0, app.router.routes.pop())  # type: ignore[attr-defined]
    monkeypatch.setattr(mcp_api_bridge, "DISPATCH_TIMEOUT_S", 0.05)
    scope = {
        "type": "http",
        "app": app,
        "method": "POST",
        "path": "/api/v1/mcp",
        "headers": [(b"x-eyvan-user", b"root"), (b"x-eyvan-roles", b"fleet_admin")],
        "client": ("203.0.113.9", 4242),
    }

    async def run() -> mcp_api_bridge.ApiBridgeError:
        with pytest.raises(mcp_api_bridge.ApiBridgeError) as caught:
            await mcp_api_bridge.dispatch(Request(scope), method="POST", path="/api/v1/zz-test-slow", dev_auth=True)
        await asyncio.sleep(0.6)  # the route was not cancelled: it finishes on its own
        return caught.value

    error = asyncio.run(run())
    assert error.status == 504
    assert error.idempotency_key is not None and error.idempotency_key.startswith("mcp-")
    assert "unknown" in error.message and "safe" in error.message and error.idempotency_key in error.message
    assert finished == [error.idempotency_key]


def test_the_denylist_and_catalogue_agree_with_the_app() -> None:
    app = _client().app
    catalogued = mcp_api_bridge.catalogue(app).operations
    assert len(catalogued) > 100  # the user-facing API, not a hand-picked handful
    assert all(mcp_api_bridge.denial_reason(item.path) is None for item in catalogued)
    assert all(item.path.startswith("/api/v1/") for item in catalogued)
    templates = {getattr(route, "path", None) for route in app.router.routes}
    # Every denylisted template is a real route (a renamed route must not silently drop out of the list).
    missing = [path for path in mcp_api_bridge.DENIED_PATHS if path not in templates]
    assert missing == [], missing
    routed = {
        (method, getattr(route, "path", None))
        for route in app.router.routes
        for method in (getattr(route, "methods", None) or ())
    }
    missing_ops = [key for key in mcp_api_bridge.DENIED_OPERATIONS if key not in routed]
    assert missing_ops == [], missing_ops
    for prefix, _reason in mcp_api_bridge.DENIED_PREFIXES:
        assert any(
            isinstance(path, str) and (path == prefix.rstrip("/") or path.startswith(prefix)) for path in templates
        ), prefix
    assert not any(item.path.startswith("/api/v1/attachments") for item in catalogued)
    assert all(mcp_api_bridge.denial_reason(item.path, item.method) is None for item in catalogued)
    assert ("POST", "/api/v1/workloads") in {(item.method, item.path) for item in catalogued}
    assert ("POST", "/api/v1/workloads/{workload_id}/cancel") in {(item.method, item.path) for item in catalogued}


def test_recursion_unknown_and_malformed_paths_are_refused() -> None:
    http = _client()
    alice = _bearer(http, "alice")
    frame = {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    failed, recursion = _bridge(http, "POST", "/api/v1/mcp", alice, body=frame)
    assert failed and recursion["status"] == 403 and "recursion" in recursion["error"]
    failed, recursion_get = _bridge(http, "GET", "/api/v1/mcp", alice)
    assert failed and recursion_get["status"] == 403

    failed, unknown = _bridge(http, "GET", "/api/v1/not-a-route", alice)
    assert failed and unknown["status"] == 404
    failed, wrong_method = _bridge(http, "GET", "/api/v1/workloads/wl_x/cancel", alice)
    assert failed and wrong_method["status"] == 405
    for path in (
        "/api/v1/workloads/../team/tokens",
        "/api/v1/workloads?org_id=acme",
        "/api/v1/team/%74okens",
        "/api/v1/workloads/",
        "/api/v1//workloads",
        "api/v1/workloads",
    ):
        failed, malformed = _bridge(http, "GET", path, alice)
        assert failed and malformed["status"] in {400, 403}, (path, malformed)
    failed, method = _call(http, "eyvan_api_call", {"method": "GET", "path": "/api/v1/workloads"}, headers=alice)
    assert failed and method["status"] == 400
    failed, extra = _call(http, "eyvan_api_get", {"path": "/api/v1/workloads", "body": {}}, headers=alice)
    assert failed and "body" in extra["error"]


def _install_echo_route(http: TestClient) -> None:
    async def echo(request: Request, principal: tp.Annotated[Principal, Depends(authenticated_principal)]) -> dict:
        client = request.client
        return {
            "subject": principal.subject,
            "headers": {name.lower(): value for name, value in request.headers.items()},
            "cookies": dict(request.cookies),
            "client": None if client is None else [client.host, client.port],
        }

    app = http.app
    app.add_api_route("/api/v1/zz-test-echo", echo, methods=["GET"], tags=["test"])  # type: ignore[attr-defined]
    # Ahead of a mounted dashboard bundle, which would otherwise answer first.
    app.router.routes.insert(0, app.router.routes.pop())  # type: ignore[attr-defined]


def test_the_forwarded_request_carries_only_the_callers_credentials() -> None:
    http = _client()
    alice = _bearer(http, "alice")
    _install_echo_route(http)
    noisy = {
        **alice,
        "Cookie": "eyvan_team_session=eyv_" + "c" * 43 + "; other=1",
        "Origin": ORIGIN,
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-Mode": "cors",
        "X-Forwarded-For": "127.0.0.1",
        "X-Eyvan-Artifact-Token": "capability",
    }
    failed, echoed = _bridge(http, "GET", "/api/v1/zz-test-echo", noisy)
    assert not failed, echoed
    body = echoed["body"]
    assert body["subject"] == "alice"
    assert body["cookies"] == {} and "cookie" not in body["headers"]
    for name in ("origin", "sec-fetch-site", "sec-fetch-mode", "x-forwarded-for", "x-eyvan-artifact-token"):
        assert name not in body["headers"], name
    # Forwarded (the route authenticated alice) but never echoed back to the agent.
    assert body["headers"]["authorization"] == "<redacted>"
    assert body["headers"]["host"] == "eyvan-mcp-bridge.invalid"  # never a loopback literal
    assert body["client"] == ["testclient", 50000]  # the original peer, not the transport's 127.0.0.1


def test_forwarded_headers_unit() -> None:
    scope = {
        "type": "http",
        "headers": [
            (b"authorization", b"Bearer t"),
            (b"cookie", b"a=b"),
            (b"origin", b"http://127.0.0.1:8080"),
            (b"x-eyvan-org", b"acme"),
            (b"x-eyvan-user", b"root"),
            (b"x-eyvan-roles", b"fleet_admin"),
            (b"x-eyvan-local-credential", b"secret"),
            (b"host", b"127.0.0.1:8080"),
        ],
    }
    request = Request(scope)
    assert mcp_api_bridge.forwarded_headers(request, dev_auth=False) == {
        "Authorization": "Bearer t",
        "X-Eyvan-Org": "acme",
    }
    assert mcp_api_bridge.forwarded_headers(request, dev_auth=True) == {
        "Authorization": "Bearer t",
        "X-Eyvan-Org": "acme",
        "X-Eyvan-User": "root",
        "X-Eyvan-Roles": "fleet_admin",
        "X-Eyvan-Local-Credential": "secret",
    }


def test_a_cookie_session_is_not_turned_into_a_forwarded_identity() -> None:
    origin = "https://plane.example.test"
    settings = EyvanSettings(team_mode=True, dev_auth_enabled=False, public_url=origin, bootstrap_token="b" * 40)
    http = TestClient(create_app(settings=settings), base_url=origin)
    bootstrap = {"Authorization": "Bearer " + "b" * 40}
    created = http.post(
        "/api/v1/team/invites",
        json={
            "email": "alice@example.com",
            "subject": "alice",
            "role": "project_operator",
            "org_id": "acme",
            "project_id": "r",
        },
        headers=bootstrap,
    )
    assert created.status_code == 201, created.text
    accepted = http.post("/api/v1/team/invites/accept", json={"token": created.json()["token"], "password": PASSWORD})
    assert accepted.status_code == 200 and http.cookies  # signed in by cookie only
    failed, refused = _call(http, "eyvan_api_get", {"path": "/api/v1/team/me"}, headers={"Origin": origin})
    assert failed and refused["status"] == 401 and "Bearer" in refused["error"]


def test_tools_are_listed_with_truthful_annotations() -> None:
    http = _client()
    alice = _bearer(http, "alice")
    listed = http.post(
        "/api/v1/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, headers={**ACCEPT, **alice}
    ).json()["result"]["tools"]
    by_name = {tool["name"]: tool for tool in listed}
    assert by_name["eyvan_api_operations"]["annotations"]["readOnlyHint"] is True
    assert by_name["eyvan_api_get"]["annotations"]["readOnlyHint"] is True
    assert by_name["eyvan_api_get"]["annotations"]["destructiveHint"] is False
    assert by_name["eyvan_api_call"]["annotations"]["readOnlyHint"] is False
    assert by_name["eyvan_api_call"]["annotations"]["destructiveHint"] is True
    assert by_name["eyvan_api_call"]["inputSchema"]["properties"]["method"]["enum"] == ["POST", "PUT", "PATCH", "DELETE"]
    assert by_name["eyvan_api_call"]["inputSchema"]["required"] == ["method", "path"]
    # The curated tools stay.
    for curated in ("submit_workload", "cancel_workload", "list_fleets", "get_run_metrics", "draft_easydel_launch"):
        assert curated in by_name


def test_the_operation_listing_is_filtered_and_bounded() -> None:
    http = _client()
    alice = _bearer(http, "alice")
    failed, everything = _call(http, "eyvan_api_operations", {}, headers=alice)
    assert not failed and everything["total"] > 100
    assert everything["shown"] == everything["total"]  # compact form fits whole
    assert all(isinstance(item, str) for item in everything["operations"])
    assert not any("/team/tokens" in item or item.endswith(" /api/v1/mcp") for item in everything["operations"])
    assert "workloads" in everything["areas"]

    failed, area = _call(http, "eyvan_api_operations", {"area": "governance"}, headers=alice)
    assert not failed and area["total"] > 0
    assert all(item["area"] == "governance" for item in area["operations"])

    failed, found = _call(http, "eyvan_api_operations", {"search": "cancel"}, headers=alice)
    assert not failed
    cancel = next(item for item in found["operations"] if item["path"] == "/api/v1/workloads/{workload_id}/cancel")
    assert cancel["method"] == "POST" and cancel["tool"] == "eyvan_api_call" and cancel["destructive"] is True
    assert {"name": "workload_id", "in": "path"}.items() <= cancel["parameters"][0].items()

    failed, submit = _call(
        http, "eyvan_api_operations", {"search": "/api/v1/workloads", "area": "workloads"}, headers=alice
    )
    assert not failed and all(item["area"] == "workloads" for item in submit["operations"])
    post = next(item for item in submit["operations"] if (item["method"], item["path"]) == ("POST", "/api/v1/workloads"))
    assert post["idempotency_key"] == "required" and "spec" in json.dumps(post["body_schema"])
    failed, tokens = _call(http, "eyvan_api_operations", {"search": "tokens"}, headers=alice)
    assert not failed and not any("/team/tokens" in item["path"] for item in tokens["operations"])

    # Every area listing fits a tool result without being cut.
    for name in everything["areas"]:
        text = mcp_routes.bounded_text(mcp_api_bridge.list_operations(http.app, area=name))
        assert "truncated" not in text and len(text) <= mcp_routes.MAX_RESULT_CHARS, name
    described = mcp_api_bridge.list_operations(http.app, search="/api/v1/workloads/{workload_id}/cancel")
    assert described["total"] == 1
    assert described["operations"][0]["path"] == "/api/v1/workloads/{workload_id}/cancel"
    assert isinstance(described["operations"][0], dict)  # a narrow match is described in full
