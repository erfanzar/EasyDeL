# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""OIDC redirects preserve provider queries and browser cookies are host-bound."""

import base64
import hashlib
import json
import shutil
import subprocess
import time
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from fastapi import FastAPI, Request

from eyvan.config import EyvanSettings
from eyvan.records import Principal
from eyvan.server.api.auth import RequestAuthenticator
from eyvan.server.api.dashboard_sessions import (
    SESSION_COOKIE,
    TRANSACTION_COOKIE,
    DashboardSessionStore,
    register_dashboard_auth_routes,
)


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://provider.example/authorize?tenant=research&tag=one&tag=two&empty=",
        "https://provider.example?tenant=research&tag=one&tag=two&empty=",
        "https://provider.example/authorize?tenant=research&tag=one&tag=two&empty=&response_type=token&state=old",
        "https://provider.example/authorize",
    ],
)
async def test_authorization_query_preserves_provider_parameters_and_replaces_oidc_parameters(endpoint: str) -> None:
    issuer = f"https://writer070-query-{abs(hash(endpoint))}.example"
    store = DashboardSessionStore(session_ttl_s=60, transaction_ttl_s=60)
    app = FastAPI()
    register_dashboard_auth_routes(
        app,
        settings=EyvanSettings(
            public_url="https://plane.example.com", oidc_issuer=issuer, oidc_dashboard_client_id="dashboard"
        ),
        store=store,
    )

    def provider(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/.well-known/openid-configuration"
        return httpx.Response(
            200,
            json={"issuer": issuer, "authorization_endpoint": endpoint, "token_endpoint": f"{issuer}/token"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as provider_client:
        app.state.dashboard_http_client = provider_client
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://plane.example.com"
        ) as client:
            response = await client.get("/api/v1/dashboard/auth/login", follow_redirects=False)

    assert response.status_code == 307
    query = parse_qs(urlsplit(response.headers["location"]).query, keep_blank_values=True)
    if "tenant=" in endpoint:
        assert query["tenant"] == ["research"]
        assert query["tag"] == ["one", "two"]
        assert query["empty"] == [""]
    assert query["response_type"] == ["code"]
    assert query["client_id"] == ["dashboard"]
    assert query["redirect_uri"] == ["https://plane.example.com/api/v1/dashboard/auth/callback"]
    assert query["code_challenge_method"] == ["S256"]
    assert len(query["code_challenge"]) == 1
    assert len(query["nonce"]) == 1
    assert len(query["state"]) == 1
    assert query["state"] != ["old"]


class _Verifier:
    async def verify(self, token: str) -> Principal:
        assert token in {"attacker", "victim"}
        return Principal(subject=token)


def _identity_route(app: FastAPI, settings: EyvanSettings, store: DashboardSessionStore) -> None:
    auth = RequestAuthenticator(settings, verifier=_Verifier(), dashboard_sessions=store)

    @app.post("/identity")
    async def identity(request: Request) -> dict[str, str]:
        return {"subject": (await auth.authenticate(request)).subject}


async def test_legacy_parent_domain_session_cannot_swap_same_origin_identity() -> None:
    settings = EyvanSettings(public_url="https://plane.example.com")
    store = DashboardSessionStore(session_ttl_s=60, transaction_ttl_s=60)
    attacker = store.create_session(access_token="attacker", id_token=None, expires_in=60)
    app = FastAPI()
    _identity_route(app, settings, store)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=settings.public_url) as client:
        # HTTPX does not implement browser prefix rules. Its jar does reproduce
        # the vulnerable legacy parent-domain cookie selection for a fresh jar.
        client.cookies.set("eyvan_session", attacker, domain=".example.com", path="/")
        swapped = await client.post("/identity", headers={"Origin": settings.public_url})
        assert "eyvan_session=" in swapped.request.headers["cookie"]
        assert swapped.status_code == 401
        assert store.resolve(attacker) is not None
        bearer = await client.post(
            "/identity", headers={"Origin": "https://evil.example.com", "Authorization": "Bearer victim"}
        )
        assert bearer.status_code == 200
        assert bearer.json() == {"subject": "victim"}

        client.cookies.clear()
        client.cookies.set(SESSION_COOKIE, attacker, domain="plane.example.com", path="/")
        accepted = await client.post("/identity", headers={"Origin": settings.public_url})
        assert accepted.status_code == 200
        assert accepted.json() == {"subject": "attacker"}
        foreign = await client.post("/identity", headers={"Origin": "https://evil.example.com"})
        assert foreign.status_code == 403


def _assert_host_cookie(response: httpx.Response, name: str, *, deleted: bool = False) -> str:
    cookie = SimpleCookie()
    for header in response.headers.get_list("set-cookie"):
        cookie.load(header)
    assert name.startswith("__Host-")
    morsel = cookie[name]
    assert morsel["secure"]
    assert morsel["httponly"]
    assert morsel["samesite"] == "lax"
    assert morsel["path"] == "/"
    assert morsel["domain"] == ""
    if deleted:
        assert morsel["max-age"] == "0"
    return morsel.value


@pytest.mark.parametrize("denied", [False, True])
async def test_host_bound_cookie_login_pkce_and_deletion_contract(denied: bool) -> None:
    issuer = f"https://writer070-cookies-{denied}.example"
    settings = EyvanSettings(
        public_url="https://plane.example.com", oidc_issuer=issuer, oidc_dashboard_client_id="dashboard"
    )
    store = DashboardSessionStore(session_ttl_s=60, transaction_ttl_s=60)
    app = FastAPI()
    register_dashboard_auth_routes(app, settings=settings, store=store)
    _identity_route(app, settings, store)
    authorization: dict[str, list[str]] = {}

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
        assert request.url.path == "/token"
        form = parse_qs(request.content.decode())
        assert form["code"] == ["fresh-code"]
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(form["code_verifier"][0].encode()).digest()).decode().rstrip("=")
        )
        assert authorization["code_challenge"] == [challenge]
        payload = (
            base64.urlsafe_b64encode(
                json.dumps(
                    {
                        "iss": issuer,
                        "sub": "victim",
                        "aud": "dashboard",
                        "nonce": authorization["nonce"][0],
                        "exp": time.time() + 60,
                    }
                ).encode()
            )
            .decode()
            .rstrip("=")
        )
        return httpx.Response(
            200,
            json={
                "access_token": "victim",
                "id_token": f"header.{payload}.signature",
                "token_type": "Bearer",
                "expires_in": 60,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as provider_client:
        app.state.dashboard_http_client = provider_client
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=settings.public_url) as client:
            login = await client.get("/api/v1/dashboard/auth/login", follow_redirects=False)
            assert login.status_code == 307
            authorization = parse_qs(urlsplit(login.headers["location"]).query)
            transaction = _assert_host_cookie(login, TRANSACTION_COOKIE)
            parameters = {"state": authorization["state"][0]}
            parameters.update({"error": "access_denied"} if denied else {"code": "fresh-code"})
            callback = await client.get("/api/v1/dashboard/auth/callback", params=parameters, follow_redirects=False)
            assert callback.status_code == (307 if denied else 303)
            _assert_host_cookie(callback, TRANSACTION_COOKIE, deleted=True)
            assert store.pop_transaction(transaction) is None
            assert client.cookies.get(TRANSACTION_COOKIE) is None
            if denied:
                assert client.cookies.get(SESSION_COOKIE) is None
            else:
                session = _assert_host_cookie(callback, SESSION_COOKIE)
                assert store.resolve(session) is not None
                identity = await client.post("/identity", headers={"Origin": settings.public_url})
                assert identity.json() == {"subject": "victim"}
                logout = await client.post("/api/v1/dashboard/auth/logout", headers={"Origin": settings.public_url})
                assert logout.status_code == 204
                _assert_host_cookie(logout, SESSION_COOKIE, deleted=True)
                assert client.cookies.get(SESSION_COOKIE) is None
                assert store.resolve(session) is None
            replay = await client.get("/api/v1/dashboard/auth/callback", params=parameters, follow_redirects=False)
            assert replay.status_code == 400


def test_browser_rejects_sibling_domain_session_and_transaction_cookies() -> None:
    """Exercise actual browser cookie acceptance, entirely through fulfilled routes."""
    node = shutil.which("node")
    root = Path(__file__).resolve().parents[1]
    if node is None or not (root / "dashboard/node_modules/playwright/index.mjs").is_file():
        pytest.skip("optional browser regression requires installed dashboard Playwright and Node")
    program = r"""
import assert from 'node:assert/strict';
import {chromium} from './dashboard/node_modules/playwright/index.mjs';
const names = JSON.parse(process.argv[1]);
let browser;
try { browser = await chromium.launch({headless: true}); }
catch {
  try { browser = await chromium.launch({headless: true, channel: 'chrome'}); }
  catch { process.exit(77); }
}
try {
  const context = await browser.newContext({serviceWorkers: 'block'});
  // Fulfill every request: no DNS, socket, provider or external network contact.
  await context.route('**/*', async route => {
    const url = new URL(route.request().url());
    if (url.hostname === 'evil.example.com') {
      const planted = [...names, 'legacy_control'].map(name =>
        `${name}=attacker; Domain=example.com; Path=/; Secure; SameSite=Lax`).join('\n');
      await route.fulfill({status: 200, headers: {'set-cookie': planted}, body: 'sibling'});
    } else {
      await route.fulfill({status: 200, body: 'plane'});
    }
  });
  const page = await context.newPage();
  await page.goto('https://evil.example.com/plant');
  await page.goto('https://plane.example.com/');
  const cookies = await context.cookies('https://plane.example.com/');
  assert(cookies.some(cookie => cookie.name === 'legacy_control'), 'unprefixed Domain control must be accepted');
  for (const name of names) {
    assert(!cookies.some(cookie => cookie.name === name), `sibling planted ${name}`);
  }
  console.log(`browser ${browser.version()}: sibling Domain cookies rejected for ${names.join(', ')}`);
} finally { await browser.close(); }
"""
    completed = subprocess.run(
        [node, "--input-type=module", "-e", program, json.dumps([SESSION_COOKIE, TRANSACTION_COOKIE])],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode == 77:
        pytest.skip("optional browser regression requires a preinstalled Chromium or Chrome; no installation attempted")
    assert completed.returncode == 0, completed.stdout + completed.stderr
