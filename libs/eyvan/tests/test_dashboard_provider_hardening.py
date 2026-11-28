# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Provider URL and lifetime inputs are validated at the BFF boundary."""

import asyncio
import base64
import json
import time

import httpx
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.server.api.dashboard_sessions import (
    SESSION_COOKIE,
    TRANSACTION_COOKIE,
    DashboardSessionStore,
    register_dashboard_auth_routes,
)


def _app(*, issuer: str, store: DashboardSessionStore, provider) -> tuple[FastAPI, httpx.AsyncClient]:
    app = FastAPI()
    register_dashboard_auth_routes(
        app,
        settings=EyvanSettings(
            public_url="https://plane.example.com",
            oidc_issuer=issuer,
            oidc_dashboard_client_id="dashboard",
        ),
        store=store,
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(provider))
    app.state.dashboard_http_client = client
    return app, client


@pytest.mark.parametrize(
    "endpoint",
    [
        "javascript:alert(1)",
        "//attacker.example/authorize",
        "https://user:pass@provider.example/authorize",
        "https://provider.example/authorize#fragment",
        "https://provider.example:bad/authorize",
        "https://provider.example:65536/authorize",
        "https://provider.example:0/authorize",
        " https://provider.example/authorize",
        "https://provider.example/authorize\n",
    ],
)
def test_discovery_rejects_unsafe_authorization_endpoint(endpoint: str) -> None:
    issuer = f"https://unsafe-{abs(hash(endpoint))}.example.com"
    store = DashboardSessionStore(session_ttl_s=60, transaction_ttl_s=60)

    def provider(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"issuer": issuer, "authorization_endpoint": endpoint, "token_endpoint": f"{issuer}/token"},
        )

    app, provider_client = _app(issuer=issuer, store=store, provider=provider)
    try:
        with TestClient(app, base_url="https://plane.example.com", raise_server_exceptions=False) as client:
            response = client.get("/api/v1/dashboard/auth/login", follow_redirects=False)
        assert response.status_code == 502
    finally:
        asyncio.run(provider_client.aclose())


def test_discovery_rejects_unsafe_token_endpoint_before_callback_exchange() -> None:
    issuer = "https://unsafe-token-endpoint.example.com"
    store = DashboardSessionStore(session_ttl_s=60, transaction_ttl_s=60)
    transaction, state, _, _ = store.begin_transaction()

    def provider(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/.well-known/openid-configuration":
            return httpx.Response(
                200,
                json={
                    "issuer": issuer,
                    "authorization_endpoint": f"{issuer}/authorize",
                    "token_endpoint": "file:///etc/passwd",
                },
            )
        raise AssertionError("unsafe token endpoint was requested")

    app, provider_client = _app(issuer=issuer, store=store, provider=provider)
    try:
        with TestClient(app, base_url="https://plane.example.com", raise_server_exceptions=False) as client:
            client.cookies.set(TRANSACTION_COOKIE, transaction)
            response = client.get(
                "/api/v1/dashboard/auth/callback",
                params={"code": "code", "state": state},
                follow_redirects=False,
            )
        assert response.status_code == 502
        assert SESSION_COOKIE not in response.cookies
    finally:
        asyncio.run(provider_client.aclose())


@pytest.mark.parametrize("expires_in", ["not-a-number", True, -1, 0, 2**10000])
def test_invalid_provider_lifetime_fails_closed(expires_in) -> None:
    issuer = f"https://lifetime-invalid-{str(expires_in).replace('-', 'negative-')}.example.com"
    store = DashboardSessionStore(session_ttl_s=60, transaction_ttl_s=60)
    transaction, state, nonce, _ = store.begin_transaction()
    payload = (
        base64.urlsafe_b64encode(
            json.dumps(
                {"iss": issuer, "sub": "alice", "aud": "dashboard", "nonce": nonce, "exp": time.time() + 3600}
            ).encode()
        )
        .decode()
        .rstrip("=")
    )

    def provider(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/.well-known/openid-configuration":
            return httpx.Response(
                200,
                json={
                    "issuer": issuer,
                    "authorization_endpoint": f"{issuer}/authorize",
                    "token_endpoint": f"{issuer}/token",
                },
            )
        return httpx.Response(
            200,
            json={
                "access_token": "access",
                "token_type": "Bearer",
                "id_token": f"header.{payload}.signature",
                "expires_in": expires_in,
            },
        )

    app, provider_client = _app(issuer=issuer, store=store, provider=provider)
    try:
        with TestClient(app, base_url="https://plane.example.com", raise_server_exceptions=False) as client:
            client.cookies.set(TRANSACTION_COOKIE, transaction)
            response = client.get(
                "/api/v1/dashboard/auth/callback",
                params={"code": "code", "state": state},
                follow_redirects=False,
            )
        assert response.status_code == 502, response.text
        assert SESSION_COOKIE not in response.cookies
    finally:
        asyncio.run(provider_client.aclose())


def test_fractional_provider_lifetime_is_preserved() -> None:
    issuer = "https://lifetime-fractional.example.com"
    store = DashboardSessionStore(session_ttl_s=60, transaction_ttl_s=60)
    transaction, state, nonce, _ = store.begin_transaction()
    payload = (
        base64.urlsafe_b64encode(
            json.dumps(
                {"iss": issuer, "sub": "alice", "aud": "dashboard", "nonce": nonce, "exp": time.time() + 3600}
            ).encode()
        )
        .decode()
        .rstrip("=")
    )

    def provider(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/.well-known/openid-configuration":
            return httpx.Response(
                200,
                json={
                    "issuer": issuer,
                    "authorization_endpoint": f"{issuer}/authorize",
                    "token_endpoint": f"{issuer}/token",
                },
            )
        return httpx.Response(
            200,
            json={
                "access_token": "access",
                "token_type": "Bearer",
                "id_token": f"header.{payload}.signature",
                "expires_in": 1.5,
            },
        )

    app, provider_client = _app(issuer=issuer, store=store, provider=provider)
    try:
        with TestClient(app, base_url="https://plane.example.com", raise_server_exceptions=False) as client:
            client.cookies.set(TRANSACTION_COOKIE, transaction)
            response = client.get(
                "/api/v1/dashboard/auth/callback",
                params={"code": "code", "state": state},
                follow_redirects=False,
            )
        assert response.status_code == 303, response.text
        session_token = response.cookies.get(SESSION_COOKIE)
        assert session_token
        session = store.resolve(session_token)
        assert session is not None
        assert 1.0 < session.expires_at - time.monotonic() <= 1.5
    finally:
        asyncio.run(provider_client.aclose())


def test_absent_provider_lifetime_uses_server_cap() -> None:
    issuer = "https://lifetime-absent.example.com"
    store = DashboardSessionStore(session_ttl_s=60, transaction_ttl_s=60)
    transaction, state, nonce, _ = store.begin_transaction()
    payload = (
        base64.urlsafe_b64encode(
            json.dumps(
                {"iss": issuer, "sub": "alice", "aud": "dashboard", "nonce": nonce, "exp": time.time() + 3600}
            ).encode()
        )
        .decode()
        .rstrip("=")
    )

    def provider(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/.well-known/openid-configuration":
            return httpx.Response(
                200,
                json={
                    "issuer": issuer,
                    "authorization_endpoint": f"{issuer}/authorize",
                    "token_endpoint": f"{issuer}/token",
                },
            )
        return httpx.Response(
            200,
            json={"access_token": "access", "token_type": "Bearer", "id_token": f"header.{payload}.signature"},
        )

    app, provider_client = _app(issuer=issuer, store=store, provider=provider)
    try:
        with TestClient(app, base_url="https://plane.example.com", raise_server_exceptions=False) as client:
            client.cookies.set(TRANSACTION_COOKIE, transaction)
            response = client.get(
                "/api/v1/dashboard/auth/callback",
                params={"code": "code", "state": state},
                follow_redirects=False,
            )
        assert response.status_code == 303, response.text
        session_token = response.cookies.get(SESSION_COOKIE)
        assert session_token
        session = store.resolve(session_token)
        assert session is not None
        assert 59.0 < session.expires_at - time.monotonic() <= 60.0
    finally:
        asyncio.run(provider_client.aclose())
