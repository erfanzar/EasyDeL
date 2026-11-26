# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Malformed identity-provider responses fail as upstream errors, not crashes."""

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


@pytest.mark.parametrize("phase", ["discovery", "token"])
@pytest.mark.parametrize("body", ["[]", "null", "not-json", "unreachable"])
def test_invalid_provider_documents_return_bad_gateway(phase, body) -> None:
    app = FastAPI()
    store = DashboardSessionStore(session_ttl_s=60, transaction_ttl_s=60)
    # Unique issuer prevents the discovery cache from hiding the response under test.
    issuer = f"https://{phase}-{len(body)}.example.com"
    register_dashboard_auth_routes(
        app, settings=EyvanSettings(oidc_issuer=issuer, oidc_dashboard_client_id="dashboard"), store=store
    )

    def provider(request):
        if phase == "discovery" or request.url.path == "/token":
            if body == "unreachable":
                raise httpx.ConnectError("provider unreachable", request=request)
            return httpx.Response(200, text=body)
        return httpx.Response(
            200,
            json={
                "issuer": issuer,
                "authorization_endpoint": f"{issuer}/authorize",
                "token_endpoint": f"{issuer}/token",
            },
        )

    provider_client = httpx.AsyncClient(transport=httpx.MockTransport(provider))
    app.state.dashboard_http_client = provider_client
    try:
        with TestClient(app, base_url="https://plane.example.com", raise_server_exceptions=False) as client:
            if phase == "discovery":
                response = client.get("/api/v1/dashboard/auth/login", follow_redirects=False)
            else:
                transaction, state, _, _ = store.begin_transaction()
                client.cookies.set(TRANSACTION_COOKIE, transaction)
                response = client.get("/api/v1/dashboard/auth/callback", params={"code": "test", "state": state})
            assert response.status_code == 502
            assert "identity provider" in response.json()["detail"].lower()
    finally:
        asyncio.run(provider_client.aclose())


@pytest.mark.parametrize(
    "case",
    [
        "missing-token",
        "missing-nonce",
        "array",
        "unicode",
        "wrong-issuer",
        "wrong-audience",
        "missing-sub",
        "bad-exp",
        "missing-exp",
        "expired-exp",
        "null-azp",
        "surrogate-nonce",
        "multi-no-azp",
        "wrong-azp",
        "multi-audience",
        "valid",
    ],
)
def test_callback_requires_bound_valid_id_token_claims(case) -> None:
    store = DashboardSessionStore(session_ttl_s=60, transaction_ttl_s=60)
    transaction, state, nonce, _ = store.begin_transaction()
    claims = {
        "iss": "https://nonce.example.com",
        "sub": "alice",
        "aud": "dashboard",
        "nonce": nonce,
        "exp": time.time() + 3600,
    }
    if case == "missing-nonce":
        claims.pop("nonce")
    if case == "unicode":
        claims["nonce"] = "نامعتبر"
    if case == "wrong-issuer":
        claims["iss"] = "https://other.example.com"
    if case == "wrong-audience":
        claims["aud"] = "other-client"
    if case == "missing-sub":
        claims.pop("sub")
    if case == "bad-exp":
        claims["exp"] = True
    if case == "missing-exp":
        claims.pop("exp")
    if case == "expired-exp":
        claims["exp"] = time.time() - 1
    if case == "null-azp":
        claims["azp"] = None
    if case == "surrogate-nonce":
        claims["nonce"] = "\ud800"
    if case in {"multi-no-azp", "wrong-azp", "multi-audience"}:
        claims["aud"] = ["dashboard", "api"]
    if case == "wrong-azp":
        claims["azp"] = "other-client"
    if case == "multi-audience":
        claims["azp"] = "dashboard"
    payload = [] if case == "array" else claims
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    token = {"access_token": "test-access", "token_type": "Bearer"}
    if case != "missing-token":
        token["id_token"] = f"header.{encoded}.signature"

    def provider(request):
        if request.url.path == "/token":
            return httpx.Response(200, json=token)
        return httpx.Response(
            200,
            json={
                "issuer": "https://nonce.example.com",
                "authorization_endpoint": "https://nonce.example.com/authorize",
                "token_endpoint": "https://nonce.example.com/token",
            },
        )

    app = FastAPI()
    register_dashboard_auth_routes(
        app,
        settings=EyvanSettings(
            oidc_issuer="https://nonce.example.com",
            oidc_dashboard_client_id="dashboard",
        ),
        store=store,
    )
    provider_client = httpx.AsyncClient(transport=httpx.MockTransport(provider))
    app.state.dashboard_http_client = provider_client
    try:
        with TestClient(app, base_url="https://plane.example.com", raise_server_exceptions=False) as client:
            client.cookies.set(TRANSACTION_COOKIE, transaction)
            response = client.get(
                "/api/v1/dashboard/auth/callback", params={"code": "test", "state": state}, follow_redirects=False
            )
        accepted_cases = {"valid", "multi-audience"}
        assert response.status_code == (303 if case in accepted_cases else 502)
        if case not in accepted_cases:
            assert SESSION_COOKIE not in response.cookies
    finally:
        asyncio.run(provider_client.aclose())


async def test_cached_discovery_still_binds_the_exact_configured_issuer() -> None:
    provider_issuer = "https://cache-issuer-binding.example.com"

    def provider(request):
        return httpx.Response(
            200,
            json={
                "issuer": provider_issuer,
                "authorization_endpoint": f"{provider_issuer}/authorize",
                "token_endpoint": f"{provider_issuer}/token",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as provider_client:
        for issuer, expected in ((provider_issuer, 307), (provider_issuer + "/", 502)):
            app = FastAPI()
            register_dashboard_auth_routes(
                app,
                settings=EyvanSettings(oidc_issuer=issuer, oidc_dashboard_client_id="dashboard"),
                store=DashboardSessionStore(session_ttl_s=60, transaction_ttl_s=60),
            )
            app.state.dashboard_http_client = provider_client
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="https://plane.example.com"
            ) as client:
                response = await client.get("/api/v1/dashboard/auth/login", follow_redirects=False)
            assert response.status_code == expected
