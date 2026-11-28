# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Team-mode HTTP surface: off by default, cookie and bearer sign-in, no secrets leaked."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.server.api.app import create_app

ORIGIN = "http://127.0.0.1:8080"
ADMIN = {"X-Eyvan-User": "root", "X-Eyvan-Roles": "fleet_admin"}
PASSWORD = "correct horse battery"


def _client(*, team_mode: bool = True) -> TestClient:
    settings = EyvanSettings(dev_auth_enabled=True, team_mode=team_mode, artifact_token_key="k" * 32)
    return TestClient(create_app(settings=settings), base_url=ORIGIN)


def _onboard(http: TestClient, email: str = "alice@example.com", subject: str = "alice") -> dict:
    created = http.post(
        "/api/v1/team/invites",
        json={"email": email, "subject": subject, "role": "project_submitter", "org_id": "acme", "project_id": "r"},
        headers=ADMIN,
    )
    assert created.status_code == 201, created.text
    return created.json()


def test_everything_but_status_is_404_when_team_mode_is_off() -> None:
    http = _client(team_mode=False)
    assert http.get("/api/v1/team/status").json() == {"team_mode": False}
    assert http.post("/api/v1/team/login", json={"email": "a@b.co", "password": "x" * 12}).status_code == 404
    assert http.get("/api/v1/team/users", headers=ADMIN).status_code == 404
    assert http.post("/api/v1/team/invites", json={"email": "a@b.co", "subject": "a"}, headers=ADMIN).status_code == 404


def test_invite_accept_sets_a_session_cookie_that_authenticates() -> None:
    http = _client()
    created = _onboard(http)
    token = created["token"]
    assert created["url"].endswith(f"/#/accept-invite/{token}")
    assert "token_hash" not in str(created["invite"])

    preview = http.post("/api/v1/team/invites/preview", json={"token": token})
    assert preview.json()["email"] == "alice@example.com"
    accepted = http.post(
        "/api/v1/team/invites/accept", json={"token": token, "password": PASSWORD, "display_name": "Alice"}
    )
    assert accepted.status_code == 200, accepted.text
    cookie = accepted.headers["set-cookie"]
    assert "eyvan_team_session=" in cookie and "HttpOnly" in cookie and "samesite=strict" in cookie.lower()

    me = http.get("/api/v1/team/me")
    assert me.status_code == 200 and me.json()["subject"] == "alice"
    assert {"role": "project_submitter", "org_id": "acme", "project_id": "r"} in me.json()["grants"]
    assert "password_hash" not in me.text


def test_https_team_cookie_and_logout_contract() -> None:
    origin = "https://plane.example.test"
    settings = EyvanSettings(
        dev_auth_enabled=True, team_mode=True, public_url=origin + ":443", artifact_token_key="k" * 32
    )
    http = TestClient(create_app(settings=settings), base_url=origin)
    invite = _onboard(http)["token"]
    accepted = http.post("/api/v1/team/invites/accept", json={"token": invite, "password": PASSWORD})
    assert accepted.status_code == 200
    name = "__Host-eyvan_team_session"
    cookie = next(iter(http.cookies.jar))
    assert cookie.name == name
    assert cookie.secure and cookie.path == "/" and not cookie.domain_specified
    assert cookie.has_nonstandard_attr("HttpOnly") and cookie.get_nonstandard_attr("SameSite") == "strict"

    # Only the host-prefixed name supplies a team identity over HTTPS.
    http.cookies.clear()
    http.cookies.set("eyvan_team_session", cookie.value)
    assert http.get("/api/v1/team/me").json().get("subject") != "alice"
    http.cookies.jar.set_cookie(cookie)
    for headers in (
        {},
        {"Origin": "https://other.example.test"},
        {"Origin": origin + ":444"},
        {"Origin": origin + "/path"},
        {"Origin": origin + "?query"},
        {"Origin": origin + "#fragment"},
        {"Origin": "https://user@plane.example.test"},
        [("Origin", origin), ("Origin", origin)],
    ):
        refused = http.post("/api/v1/team/logout", headers=headers)
        assert refused.status_code == 403 and "set-cookie" not in refused.headers
        assert http.get("/api/v1/team/me").json()["subject"] == "alice"

    ended = http.post("/api/v1/team/logout", headers={"Origin": origin})
    assert ended.status_code == 200
    assert name not in http.cookies
    http.cookies.jar.set_cookie(cookie)
    assert http.get("/api/v1/team/me").status_code == 401


def test_cookie_writes_must_come_from_the_dashboard_origin() -> None:
    http = _client()
    token = _onboard(http)["token"]
    http.post("/api/v1/team/invites/accept", json={"token": token, "password": PASSWORD})
    assert (
        http.post("/api/v1/team/tokens", json={"name": "x"}, headers={"Origin": "https://evil.example"}).status_code
        == 403
    )
    assert http.post("/api/v1/team/tokens", json={"name": "x"}, headers={"Origin": ORIGIN}).status_code == 201


def test_cli_login_returns_a_bearer_token_without_a_cookie() -> None:
    http = _client()
    token = _onboard(http)["token"]
    http.post("/api/v1/team/invites/accept", json={"token": token, "password": PASSWORD})
    http.cookies.clear()

    response = http.post(
        "/api/v1/team/login", json={"email": "alice@example.com", "password": PASSWORD, "token_name": "cli-laptop"}
    )
    assert response.status_code == 200
    assert "set-cookie" not in response.headers
    raw = response.json()["token"]
    bearer = {"Authorization": f"Bearer {raw}"}
    assert http.get("/api/v1/team/me", headers=bearer).json()["subject"] == "alice"
    listed = http.get("/api/v1/team/tokens", headers=bearer).json()["tokens"]
    assert [item["name"] for item in listed] == ["cli-laptop"] and raw not in str(listed)

    assert http.delete(f"/api/v1/team/tokens/{listed[0]['id']}", headers=bearer).status_code == 200
    refused = http.get("/api/v1/team/me", headers=bearer)
    assert refused.status_code == 401 and "eyvan login" in refused.json()["detail"]


def test_wrong_password_is_generic_and_throttled() -> None:
    http = _client()
    token = _onboard(http)["token"]
    http.post("/api/v1/team/invites/accept", json={"token": token, "password": PASSWORD})
    statuses = [
        http.post("/api/v1/team/login", json={"email": "alice@example.com", "password": "wrong password!"}).status_code
        for _ in range(6)
    ]
    assert statuses[:5] == [401] * 5 and statuses[5] == 429
    unknown = http.post("/api/v1/team/login", json={"email": "nobody@example.com", "password": "wrong password!"})
    assert unknown.json()["detail"] == "wrong user name or password"
    unknown_name = http.post("/api/v1/team/login", json={"email": "nobody", "password": "wrong password!"})
    assert unknown_name.status_code == 401 and unknown_name.json()["detail"] == "wrong user name or password"


def test_sign_in_accepts_the_user_name_as_well_as_the_email() -> None:
    http = _client()
    token = _onboard(http)["token"]
    http.post("/api/v1/team/invites/accept", json={"token": token, "password": PASSWORD})
    http.cookies.clear()
    by_name = http.post("/api/v1/team/login", json={"email": "Alice", "password": PASSWORD})
    assert by_name.status_code == 200, by_name.text
    assert by_name.json()["user"]["subject"] == "alice"
    http.cookies.clear()
    by_email = http.post("/api/v1/team/login", json={"email": "alice@example.com", "password": PASSWORD})
    assert by_email.status_code == 200
    wrong = http.post("/api/v1/team/login", json={"email": "alice", "password": "wrong password!"})
    assert wrong.status_code == 401


@pytest.mark.parametrize(
    "method,path",
    [("get", "/api/v1/team/users"), ("get", "/api/v1/team/invites"), ("post", "/api/v1/team/users/alice/disable")],
)
def test_account_administration_needs_fleet_admin(method: str, path: str) -> None:
    http = _client()
    token = _onboard(http)["token"]
    http.post("/api/v1/team/invites/accept", json={"token": token, "password": PASSWORD})
    assert http.request(method, path, headers={"Origin": ORIGIN}).status_code == 403


def test_disable_signs_the_user_out_everywhere() -> None:
    http = _client()
    token = _onboard(http)["token"]
    http.post("/api/v1/team/invites/accept", json={"token": token, "password": PASSWORD})
    assert http.get("/api/v1/team/me").status_code == 200
    # The admin acts from another client: this one carries alice's cookie, and a
    # team cookie is resolved before development headers (by design).
    admin = TestClient(http.app, base_url=ORIGIN)
    assert admin.post("/api/v1/team/users/alice/disable", headers=ADMIN).status_code == 200
    assert http.get("/api/v1/team/me").status_code == 401
