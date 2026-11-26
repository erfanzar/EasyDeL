# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Same-site sibling origins must not exercise cookie authority."""

import pytest
from fastapi import HTTPException
from starlette.requests import Request

from eyvan.config import EyvanSettings
from eyvan.records import Principal
from eyvan.server.api.auth import RequestAuthenticator
from eyvan.server.api.dashboard_sessions import SESSION_COOKIE, DashboardSessionStore


class Verifier:
    async def verify(self, token: str) -> Principal:
        return Principal(subject="alice")


@pytest.mark.parametrize("origin", [None, "null", "https://evil.example.com", "http://plane.example.com"])
async def test_cookie_writes_require_the_configured_origin(origin) -> None:
    store = DashboardSessionStore(session_ttl_s=60, transaction_ttl_s=10)
    token = store.create_session(access_token="test", id_token=None, expires_in=60)
    auth = RequestAuthenticator(
        EyvanSettings(public_url="https://plane.example.com"), verifier=Verifier(), dashboard_sessions=store
    )
    headers = [(b"cookie", f"{SESSION_COOKIE}={token}".encode())]
    if origin is not None:
        headers.append((b"origin", origin.encode()))
    request = Request({"type": "http", "method": "POST", "path": "/", "headers": headers})
    with pytest.raises(HTTPException) as refused:
        await auth.authenticate(request)
    assert refused.value.status_code == 403


@pytest.mark.parametrize("method", ["GET", "POST"])
async def test_same_origin_cookie_requests_authenticate(method) -> None:
    store = DashboardSessionStore(session_ttl_s=60, transaction_ttl_s=10)
    token = store.create_session(access_token="test", id_token=None, expires_in=60)
    auth = RequestAuthenticator(
        EyvanSettings(public_url="https://plane.example.com"), verifier=Verifier(), dashboard_sessions=store
    )
    request = Request(
        {
            "type": "http",
            "method": method,
            "path": "/",
            "headers": [(b"cookie", f"{SESSION_COOKIE}={token}".encode()), (b"origin", b"https://plane.example.com")],
        }
    )
    assert (await auth.authenticate(request)).subject == "alice"


@pytest.mark.parametrize("origin", [None, "null", "https://evil.example.com", "https://plane.example.com"])
def test_logout_only_invalidates_sessions_for_same_origin_requests(origin) -> None:
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from eyvan.server.api.dashboard_sessions import register_dashboard_auth_routes

    store = DashboardSessionStore(session_ttl_s=60, transaction_ttl_s=10)
    token = store.create_session(access_token="test", id_token=None, expires_in=60)
    app = FastAPI()
    register_dashboard_auth_routes(app, settings=EyvanSettings(public_url="https://plane.example.com"), store=store)
    with TestClient(app, base_url="https://plane.example.com") as client:
        client.cookies.set(SESSION_COOKIE, token)
        response = client.post("/api/v1/dashboard/auth/logout", headers={"Origin": origin} if origin else {})
    if origin == "https://plane.example.com":
        assert response.status_code == 204
        assert store.resolve(token) is None
    else:
        assert response.status_code == 403
        assert store.resolve(token) is not None


@pytest.mark.parametrize("origin", ["null", "https://evil.example.com", "http://plane.example.com"])
def test_cross_origin_logout_without_cookie_cannot_emit_cookie_deletion(origin) -> None:
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from eyvan.server.api.dashboard_sessions import register_dashboard_auth_routes

    app = FastAPI()
    register_dashboard_auth_routes(
        app,
        settings=EyvanSettings(public_url="https://plane.example.com"),
        store=DashboardSessionStore(session_ttl_s=60, transaction_ttl_s=10),
    )
    with TestClient(app, base_url="https://plane.example.com") as client:
        # A cross-site top-level POST can omit the Lax cookie but still apply a
        # deletion Set-Cookie from the response. Validate Origin before emitting it.
        response = client.post("/api/v1/dashboard/auth/logout", headers={"Origin": origin})
    assert response.status_code == 403
    assert "set-cookie" not in response.headers
