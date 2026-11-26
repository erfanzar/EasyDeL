# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Keep the dashboard tutorial aligned with its server-side OIDC boundary."""

from pathlib import Path

import pytest

from eyvan.server.api.dashboard_sessions import SESSION_COOKIE, TRANSACTION_COOKIE, _session_cookie, _transaction_cookie

_ROOT = Path(__file__).resolve().parents[1]


def _auth_section() -> str:
    tutorial = (_ROOT / "docs/tutorial/13-dashboard.md").read_text(encoding="utf-8")
    section = tutorial.split("## Authentication and authorization\n", 1)[1].split("\n## Live updates", 1)[0]
    return " ".join(section.split())


def test_tutorial_describes_server_side_code_and_pkce_completion() -> None:
    section = _auth_section()
    assert "server-side authorization-code/PKCE" in section
    assert "`/api/v1/dashboard/auth/login`" in section
    assert "`/api/v1/dashboard/auth/callback`" in section
    assert "server exchanges the authorization code" in section
    assert "OIDC PKCE completion before rendering" not in section

    # These are the actual client handoff and server exchange, not a second
    # browser implementation of the flow described in the tutorial.
    client = (_ROOT / "dashboard/src/auth/oidc.ts").read_text(encoding="utf-8")
    server = (_ROOT / "eyvan/server/api/dashboard_sessions.py").read_text(encoding="utf-8")
    assert "globalThis.location.assign(LOGIN_PATH)" in client
    assert '_CALLBACK_PATH = "/api/v1/dashboard/auth/callback"' in server
    assert '"code_verifier": transaction.verifier' in server


def test_tutorial_names_cookie_and_process_affinity_deployment_requirements() -> None:
    section = _auth_section()
    assert f"`{SESSION_COOKIE}` cookie (HttpOnly, Secure, SameSite=Lax)" in section
    assert "HTTPS" in section
    assert "in-process" in section
    assert "sticky sessions" in section
    assert "login, callback, and subsequent API requests" in section


def test_tutorial_distinguishes_loopback_storage_from_oidc_tokens() -> None:
    section = _auth_section()
    assert "OIDC access/ID tokens, PKCE verifier/nonce, and expiry stay server-side" in section
    assert "Only the loopback development credential lives in `sessionStorage`" in section
    assert "not OIDC tokens" in section
    assert "expiry live in session storage" not in section


@pytest.mark.parametrize("document", ["docs/security.md", "docs/tutorial/13-dashboard.md"])
def test_cookie_docs_match_canonical_host_prefix_and_public_origin_contract(document: str) -> None:
    text = " ".join((_ROOT / document).read_text(encoding="utf-8").split())
    assert SESSION_COOKIE == "__Host-eyvan_session"
    assert TRANSACTION_COOKIE == "__Host-eyvan_oidc_tx"
    assert f"`{SESSION_COOKIE}`" in text
    assert f"`{TRANSACTION_COOKIE}`" in text
    assert "`Path=/`" in text
    assert "no `Domain`" in text
    assert "`eyvan_session`" not in text
    assert "`eyvan_oidc_tx`" not in text
    assert "legacy unprefixed cookies are not accepted" in text
    assert "`EYVAN_PUBLIC_URL`" in text
    assert "Host or forwarded headers" in text
    assert "Origin" in text and "HTTPS" in text
    for cookie in (_session_cookie("fixture-session", 60), _transaction_cookie("fixture-transaction", 60)):
        assert cookie["secure"] is True
        assert cookie["httponly"] is True
        assert cookie["samesite"] == "lax"
        assert cookie["path"] == "/"
        assert "domain" not in cookie
