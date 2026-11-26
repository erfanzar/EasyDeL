# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Dashboard sessions are backend-for-frontend: tokens never reach the browser.

The dashboard used to keep OIDC tokens in ``sessionStorage`` — readable by any
cross-site script — and send them as bearer headers. The replacement keeps one
``__Host-eyvan_session`` cookie (HttpOnly, Secure, SameSite=Lax) in the browser and
the tokens in this process. These tests hold that contract on the wire:

* login redirects to the provider with a single-use PKCE transaction and sets
  a cookie the JavaScript of the dashboard cannot read;
* the callback validates state, exchanges the code against a fake provider,
  verifies the ID-token nonce, and opens a session with hardened cookie flags;
* an API request carrying the session cookie authenticates as the stored
  token; a dropped or expired session refuses with a sign-in-again message;
* logout is idempotent and removes the session;
* the dashboard's HTML answers with a strict Content-Security-Policy.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import typing as tp

import httpx
import pytest
from starlette.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.records import Principal, Role, RoleGrant
from eyvan.server.api.app import create_app
from eyvan.server.api.dashboard_sessions import (
    DASHBOARD_SESSION_EXPIRED,
    SESSION_COOKIE,
    TRANSACTION_COOKIE,
    DashboardSessionStore,
)

_ISSUER = "https://issuer.example"
_AUTHORIZE = f"{_ISSUER}/authorize"
_TOKEN_ENDPOINT = f"{_ISSUER}/token"
_GOOD_CODE = "good-code"


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _id_token(claims: dict) -> str:
    header = _b64url(json.dumps({"alg": "none"}).encode())
    payload = _b64url(json.dumps(claims).encode())
    return f"{header}.{payload}."


class _StaticVerifier:
    """Accepts exactly the token the fake provider mints, like the real JWKS path would."""

    async def verify(self, token: str) -> Principal:
        if token != "provider-access-token":
            raise ValueError("unknown token")
        return Principal(
            subject="alice",
            grants=(RoleGrant(role=Role.FLEET_ADMIN, org_id="acme", project_id="research"),),
        )


def _fake_provider(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/.well-known/openid-configuration":
        return httpx.Response(
            200,
            json={"issuer": _ISSUER, "authorization_endpoint": _AUTHORIZE, "token_endpoint": _TOKEN_ENDPOINT},
        )
    if request.url.path == "/token":
        form = dict(httpx.QueryParams(request.read().decode()))
        if form.get("grant_type") != "authorization_code":
            return httpx.Response(400, json={"error": "unsupported_grant_type"})
        if form.get("code") != _GOOD_CODE:
            return httpx.Response(400, json={"error": "invalid_grant"})
        if not form.get("code_verifier"):
            return httpx.Response(400, json={"error": "invalid_grant"})
        return httpx.Response(
            200,
            json={
                "access_token": "provider-access-token",
                "token_type": "Bearer",
                "expires_in": 3600,
                "id_token": _id_token({"sub": "alice"}),
            },
        )
    return httpx.Response(404, json={"error": "not found"})


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep app-created artifacts out of the operator's persistent HOME."""

    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture()
def client() -> tp.Iterator[TestClient]:
    app = create_app(
        settings=EyvanSettings(
            oidc_issuer=_ISSUER,
            oidc_dashboard_client_id="dashboard-client",
            public_url="https://plane.example.com",
        ),
        verifier=_StaticVerifier(),
    )

    def provider(request: httpx.Request) -> httpx.Response:
        response = _fake_provider(request)
        if request.url.path == "/token" and response.status_code == 200:
            document = response.json()
            document["id_token"] = _id_token(
                {
                    "iss": _ISSUER,
                    "sub": "alice",
                    "aud": "dashboard-client",
                    "nonce": app.state.provider_nonce,
                    "exp": time.time() + 3600,
                }
            )
            return httpx.Response(200, json=document)
        return response

    transport = httpx.MockTransport(provider)
    provider_client = httpx.AsyncClient(transport=transport)
    app.state.dashboard_http_client = provider_client
    with TestClient(app, base_url="https://plane.example.com") as test_client:
        yield test_client
    asyncio.run(provider_client.aclose())


def _begin(client: TestClient) -> tuple[str, str]:
    """Start a sign-in and return (transaction cookie value, provider state)."""

    response = client.get("/api/v1/dashboard/auth/login", follow_redirects=False)
    assert response.status_code == 307
    location = httpx.URL(response.headers["Location"])
    assert str(location.copy_with(query=None)) == _AUTHORIZE
    state = location.params["state"]
    # The fake provider echoes what it received in the authorization request.
    client.app.state.provider_nonce = location.params["nonce"]
    assert location.params["client_id"] == "dashboard-client"
    assert location.params["audience"] == "eyvan"
    assert location.params["response_type"] == "code"
    assert location.params["code_challenge_method"] == "S256"
    assert location.params["redirect_uri"] == "https://plane.example.com/api/v1/dashboard/auth/callback"
    set_cookie = response.headers["set-cookie"]
    assert "HttpOnly" in set_cookie
    assert "Secure" in set_cookie
    assert "SameSite=lax" in set_cookie
    assert f"{TRANSACTION_COOKIE}=" in set_cookie
    assert "Path=/;" in set_cookie
    assert "Domain=" not in set_cookie
    tx_value = next(
        part.split("=", 1)[1] for part in set_cookie.split(";") if part.strip().startswith(f"{TRANSACTION_COOKIE}=")
    )
    return tx_value, state


def test_login_redirects_to_the_provider_with_a_fresh_challenge(client: TestClient) -> None:
    _begin(client)


def test_login_without_oidc_is_a_404(tmp_path) -> None:
    app = create_app(
        settings=EyvanSettings(public_url="https://plane.example.com"),
        verifier=_StaticVerifier(),
    )
    with TestClient(app) as anonymous:
        response = anonymous.get("/api/v1/dashboard/auth/login", follow_redirects=False)

    assert response.status_code == 404


def test_the_callback_opens_a_session_with_a_hardened_cookie(client: TestClient) -> None:
    tx_value, state = _begin(client)

    response = client.get(
        f"/api/v1/dashboard/auth/callback?code={_GOOD_CODE}&state={state}",
        cookies={TRANSACTION_COOKIE: tx_value},
        follow_redirects=False,
    )

    assert response.status_code == 303
    assert response.headers["Location"] == "/"
    set_cookie = response.headers["set-cookie"]
    assert f"{SESSION_COOKIE}=" in set_cookie
    assert "HttpOnly" in set_cookie
    assert "Secure" in set_cookie
    assert "SameSite=lax" in set_cookie
    assert "Path=/" in set_cookie


def test_the_callback_refuses_a_replayed_or_unknown_transaction(client: TestClient) -> None:
    tx_value, state = _begin(client)
    client.get(
        f"/api/v1/dashboard/auth/callback?code={_GOOD_CODE}&state={state}",
        cookies={TRANSACTION_COOKIE: tx_value},
        follow_redirects=False,
    )

    replay = client.get(
        f"/api/v1/dashboard/auth/callback?code={_GOOD_CODE}&state={state}",
        cookies={TRANSACTION_COOKIE: tx_value},
        follow_redirects=False,
    )
    assert replay.status_code == 400
    assert "expired or is unknown" in replay.text


def test_the_callback_refuses_a_mismatched_state(client: TestClient) -> None:
    tx_value, _state = _begin(client)

    response = client.get(
        f"/api/v1/dashboard/auth/callback?code={_GOOD_CODE}&state=forged-state",
        cookies={TRANSACTION_COOKIE: tx_value},
        follow_redirects=False,
    )

    assert response.status_code == 400
    assert "mismatched" in response.text


def test_the_callback_surfaces_provider_errors(client: TestClient) -> None:
    tx_value, state = _begin(client)

    response = client.get(
        f"/api/v1/dashboard/auth/callback?code=bad-code&state={state}",
        cookies={TRANSACTION_COOKIE: tx_value},
        follow_redirects=False,
    )

    assert response.status_code == 502


def test_the_callback_verifies_the_id_token_nonce() -> None:
    app = create_app(
        settings=EyvanSettings(
            oidc_issuer=_ISSUER,
            oidc_dashboard_client_id="dashboard-client",
            public_url="https://plane.example.com",
        ),
        verifier=_StaticVerifier(),
    )
    app.state.dashboard_http_client = httpx.AsyncClient(transport=httpx.MockTransport(_fake_provider))

    def provider_with_nonce(nonce: str) -> httpx.AsyncClient:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/.well-known/openid-configuration":
                return httpx.Response(
                    200,
                    json={"issuer": _ISSUER, "authorization_endpoint": _AUTHORIZE, "token_endpoint": _TOKEN_ENDPOINT},
                )
            if request.url.path == "/token":
                return httpx.Response(
                    200,
                    json={
                        "access_token": "provider-access-token",
                        "token_type": "bearer",
                        "id_token": _id_token(
                            {
                                "iss": _ISSUER,
                                "sub": "alice",
                                "aud": "dashboard-client",
                                "nonce": nonce,
                                "exp": time.time() + 3600,
                            }
                        ),
                    },
                )
            return httpx.Response(404)

        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    with TestClient(app, base_url="https://plane.example.com") as test_client:
        store: DashboardSessionStore = app.state.dashboard_sessions
        tx_token, state, nonce, _verifier = store.begin_transaction()

        app.state.dashboard_http_client = provider_with_nonce("wrong-nonce")
        refused = test_client.get(
            f"/api/v1/dashboard/auth/callback?code={_GOOD_CODE}&state={state}",
            cookies={TRANSACTION_COOKIE: tx_token},
            follow_redirects=False,
        )
        assert refused.status_code == 502
        assert "nonce" in refused.text

        tx_token, state, nonce, _verifier = store.begin_transaction()
        app.state.dashboard_http_client = provider_with_nonce(nonce)
        accepted = test_client.get(
            f"/api/v1/dashboard/auth/callback?code={_GOOD_CODE}&state={state}",
            cookies={TRANSACTION_COOKIE: tx_token},
            follow_redirects=False,
        )
        assert accepted.status_code == 303


def test_a_session_cookie_authenticates_api_requests(client: TestClient) -> None:
    tx_value, state = _begin(client)
    callback = client.get(
        f"/api/v1/dashboard/auth/callback?code={_GOOD_CODE}&state={state}",
        cookies={TRANSACTION_COOKIE: tx_value},
        follow_redirects=False,
    )
    session_cookie = next(
        part.split("=", 1)[1]
        for part in callback.headers["set-cookie"].split(";")
        if part.strip().startswith(f"{SESSION_COOKIE}=")
    )

    session_response = client.get(
        "/api/v1/session",
        cookies={SESSION_COOKIE: session_cookie},
    )

    assert session_response.status_code == 200
    assert session_response.json()["subject"] == "alice"

    # Bearer authentication is unchanged and takes precedence over the cookie.
    bearer = client.get(
        "/api/v1/session",
        headers={"Authorization": "Bearer provider-access-token"},
        cookies={SESSION_COOKIE: "not-a-real-session"},
    )
    assert bearer.status_code == 200


def test_an_expired_session_refuses_with_sign_in_again(client: TestClient) -> None:
    tx_value, state = _begin(client)
    callback = client.get(
        f"/api/v1/dashboard/auth/callback?code={_GOOD_CODE}&state={state}",
        cookies={TRANSACTION_COOKIE: tx_value},
        follow_redirects=False,
    )
    session_cookie = next(
        part.split("=", 1)[1]
        for part in callback.headers["set-cookie"].split(";")
        if part.strip().startswith(f"{SESSION_COOKIE}=")
    )

    store: DashboardSessionStore = client.app.state.dashboard_sessions
    store.resolve(session_cookie).expires_at = time.monotonic() - 1

    response = client.get("/api/v1/session", cookies={SESSION_COOKIE: session_cookie})

    assert response.status_code == 401
    assert DASHBOARD_SESSION_EXPIRED in response.text


def test_logout_drops_the_session_and_is_idempotent(client: TestClient) -> None:
    tx_value, state = _begin(client)
    callback = client.get(
        f"/api/v1/dashboard/auth/callback?code={_GOOD_CODE}&state={state}",
        cookies={TRANSACTION_COOKIE: tx_value},
        follow_redirects=False,
    )
    session_cookie = next(
        part.split("=", 1)[1]
        for part in callback.headers["set-cookie"].split(";")
        if part.strip().startswith(f"{SESSION_COOKIE}=")
    )

    first = client.post(
        "/api/v1/dashboard/auth/logout",
        cookies={SESSION_COOKIE: session_cookie},
        headers={"Origin": "https://plane.example.com"},
    )
    assert first.status_code == 204
    after = client.get("/api/v1/session", cookies={SESSION_COOKIE: session_cookie})
    assert after.status_code == 401

    second = client.post(
        "/api/v1/dashboard/auth/logout",
        cookies={SESSION_COOKIE: session_cookie},
        headers={"Origin": "https://plane.example.com"},
    )
    assert second.status_code == 204
    anonymous = client.post("/api/v1/dashboard/auth/logout")
    assert anonymous.status_code == 204


def test_dashboard_html_carries_a_strict_content_security_policy(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    policy = response.headers["Content-Security-Policy"]
    assert "script-src 'self'" in policy
    assert "object-src 'none'" in policy
    assert "frame-ancestors 'none'" in policy
    assert "connect-src 'self'" in policy
    # Tracked media is shown from authenticated object URLs; blob: is allowed for
    # images and media only, never for scripts.
    directives = {part.strip().split(" ", 1)[0]: part.strip() for part in policy.split(";") if part.strip()}
    assert "blob:" in directives["img-src"]
    assert directives["media-src"] == "media-src 'self' blob:"
    assert "blob:" not in directives["script-src"]
    assert "blob:" not in directives["default-src"]
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"


def test_non_html_responses_do_not_get_the_html_only_headers(client: TestClient) -> None:
    response = client.get("/api/v1/dashboard/config")

    assert response.status_code == 200
    assert "Content-Security-Policy" not in response.headers
