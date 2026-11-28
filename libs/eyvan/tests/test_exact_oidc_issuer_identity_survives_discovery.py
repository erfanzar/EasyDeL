# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Exact OIDC issuer identity survives discovery and dashboard authentication."""

from __future__ import annotations

import io
import json
import time
import urllib.request

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import FastAPI, Request

from eyvan.config import EyvanSettings
from eyvan.server.api.auth import OIDCTokenVerifier, RequestAuthenticator
from eyvan.server.api.dashboard_sessions import (
    SESSION_COOKIE,
    TRANSACTION_COOKIE,
    DashboardSessionStore,
    register_dashboard_auth_routes,
)

AUDIENCE = "eyvan"
BASE_ISSUER = "https://writer-069-issuer.example"


@pytest.fixture(scope="module")
def signing_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def _token(key: rsa.RSAPrivateKey, issuer: str, *, audience: str = AUDIENCE, nonce: str | None = None) -> str:
    claims = {"iss": issuer, "sub": "alice", "aud": audience, "exp": int(time.time()) + 300}
    if nonce is not None:
        claims["nonce"] = nonce
    return jwt.encode(claims, key, algorithm="RS256", headers={"kid": "writer-069-key"})


def _local_verifier(monkeypatch, signing_key, *, issuer: str, discovered_issuer: str | None = None):
    """Replace only network transports; leave discovery and JWT validation real."""

    public_key = jwt.algorithms.RSAAlgorithm.to_jwk(signing_key.public_key(), as_dict=True)
    public_key.update({"kid": "writer-069-key", "use": "sig", "alg": "RS256"})
    jwks_url = f"{BASE_ISSUER}/jwks"

    def open_url(_opener, request, *_args, **_kwargs):
        assert request.full_url == jwks_url
        return io.BytesIO(json.dumps({"keys": [public_key]}).encode())

    # OpenerDirector.open is where every PyJWT version's fetch ends: older
    # releases go through urlopen, 2.15+ builds its own no-redirect opener.
    monkeypatch.setattr(urllib.request.OpenerDirector, "open", open_url)

    def discover(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == f"{BASE_ISSUER}/.well-known/openid-configuration"
        return httpx.Response(
            200, json={"issuer": discovered_issuer if discovered_issuer is not None else issuer, "jwks_uri": jwks_url}
        )

    original_client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda *args, **kwargs: original_client(*args, **{**kwargs, "transport": httpx.MockTransport(discover)}),
    )
    return OIDCTokenVerifier(issuer=issuer, audience=AUDIENCE), original_client


@pytest.mark.parametrize("suffix", ["", "/"])
async def test_exact_issuer_accepts_signed_access_token(signing_key, monkeypatch, suffix) -> None:
    issuer = BASE_ISSUER + suffix
    verifier, _ = _local_verifier(monkeypatch, signing_key, issuer=issuer)

    principal = await verifier.verify(_token(signing_key, issuer))

    assert principal.subject == "alice"


@pytest.mark.parametrize("suffix", ["", "/"])
async def test_discovery_slash_mismatch_is_rejected(signing_key, monkeypatch, suffix) -> None:
    issuer = BASE_ISSUER + suffix
    other_issuer = BASE_ISSUER + ("/" if not suffix else "")
    verifier, _ = _local_verifier(monkeypatch, signing_key, issuer=issuer, discovered_issuer=other_issuer)

    with pytest.raises(ValueError, match="discovery issuer does not match"):
        await verifier.verify(_token(signing_key, issuer))


@pytest.mark.parametrize("suffix", ["", "/"])
async def test_token_slash_mismatch_is_rejected(signing_key, monkeypatch, suffix) -> None:
    issuer = BASE_ISSUER + suffix
    other_issuer = BASE_ISSUER + ("/" if not suffix else "")
    verifier, _ = _local_verifier(monkeypatch, signing_key, issuer=issuer)

    with pytest.raises(jwt.InvalidIssuerError):
        await verifier.verify(_token(signing_key, other_issuer))


@pytest.mark.parametrize("suffix", ["", "/"])
async def test_bff_callback_cookie_authenticates_with_exact_issuer(signing_key, monkeypatch, suffix) -> None:
    issuer = BASE_ISSUER + suffix
    verifier, client_type = _local_verifier(monkeypatch, signing_key, issuer=issuer)
    settings = EyvanSettings(
        oidc_issuer=issuer,
        oidc_audience=AUDIENCE,
        oidc_dashboard_client_id="dashboard",
        public_url="https://plane.example.com",
        dev_auth_enabled=False,
    )
    store = DashboardSessionStore(session_ttl_s=60, transaction_ttl_s=60)
    transaction, state, nonce, _ = store.begin_transaction()
    access_token = _token(signing_key, issuer)
    id_token = _token(signing_key, issuer, audience="dashboard", nonce=nonce)

    def provider(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/token":
            return httpx.Response(200, json={"access_token": access_token, "id_token": id_token, "token_type": "Bearer"})
        assert request.url.path == "/.well-known/openid-configuration"
        return httpx.Response(
            200,
            json={
                "issuer": issuer,
                "authorization_endpoint": f"{BASE_ISSUER}/authorize",
                "token_endpoint": f"{BASE_ISSUER}/token",
            },
        )

    app = FastAPI()
    register_dashboard_auth_routes(app, settings=settings, store=store)
    authenticator = RequestAuthenticator(settings, verifier=verifier, dashboard_sessions=store)

    @app.get("/identity")
    async def identity(request: Request):
        principal = await authenticator.authenticate(request)
        return {"subject": principal.subject}

    async with client_type(transport=httpx.MockTransport(provider)) as provider_client:
        app.state.dashboard_http_client = provider_client
        async with client_type(transport=httpx.ASGITransport(app=app), base_url=settings.public_url) as client:
            client.cookies.set(TRANSACTION_COOKIE, transaction)
            callback = await client.get(
                "/api/v1/dashboard/auth/callback", params={"code": "local-code", "state": state}, follow_redirects=False
            )
            assert callback.status_code == 303
            assert SESSION_COOKIE in client.cookies
            response = await client.get("/identity")

    assert response.status_code == 200, response.text
    assert response.json() == {"subject": "alice"}
