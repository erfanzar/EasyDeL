# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Ground the dashboard release authentication reference in its BFF boundary."""

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _auth_reference() -> str:
    document = (_ROOT / "docs/reference/dashboard.md").read_text(encoding="utf-8")
    return document.split("## Authentication and authorization\n", 1)[1].split("\n## Live updates:", 1)[0]


def test_dashboard_reference_does_not_describe_the_removed_browser_oidc_flow() -> None:
    reference = " ".join(_auth_reference().split())
    client = (_ROOT / "dashboard/src/auth/oidc.ts").read_text(encoding="utf-8")
    gate = (_ROOT / "dashboard/src/auth/AuthGate.tsx").read_text(encoding="utf-8")

    # Source comparison, not a claim about an executed browser or security exploit.
    assert "globalThis.location.assign(LOGIN_PATH)" in client
    assert ".then(() => serverIdentity())" in gate
    assert "attempts OIDC PKCE completion" not in reference
    assert "Tokens, PKCE state/verifier/nonce, and expiry are held in session storage" not in reference
    assert "OIDC discovery is performed from the configured issuer" not in reference


def test_dashboard_reference_documents_server_endpoints_and_cookie_boundary() -> None:
    reference = " ".join(_auth_reference().split())
    client = (_ROOT / "dashboard/src/auth/oidc.ts").read_text(encoding="utf-8")
    backend = (_ROOT / "eyvan/server/api/dashboard_sessions.py").read_text(encoding="utf-8")

    endpoints = re.findall(r'export const (?:LOGIN|CALLBACK|LOGOUT)_PATH = "([^"]+)";', client)
    assert len(endpoints) == 3
    for endpoint in endpoints:
        assert endpoint in reference
    assert '"code_verifier": transaction.verifier' in backend
    assert "access_token=access_token" in backend
    assert "id_token=id_token" in backend
    for flag, value, documented in (
        ("httponly", "True", "HttpOnly"),
        ("secure", "True", "Secure"),
        ("samesite", '"lax"', "SameSite=Lax"),
    ):
        assert f'"{flag}": {value}' in backend
        assert documented in reference
    assert "server-side" in reference
    assert "PKCE" in reference
    assert "access and ID tokens" in reference
    assert "`serverIdentity()`" in reference
    assert "/api/v1/session" in reference


def test_dashboard_reference_distinguishes_the_loopback_storage_exception() -> None:
    reference = " ".join(_auth_reference().split())
    client = (_ROOT / "dashboard/src/auth/oidc.ts").read_text(encoding="utf-8")

    storage_keys = re.findall(r"store\??\.(?:getItem|setItem|removeItem)\((\w+)", client)
    storage_keys += re.findall(r"storage\(\)\?\.(?:getItem|setItem|removeItem)\((\w+)", client)
    assert storage_keys
    assert set(storage_keys) == {"LOCAL_CREDENTIAL_KEY"}
    assert 'export const LOCAL_CREDENTIAL_KEY = "eyvan.local.credential"' in client
    assert "loopback-only development credential" in reference
    assert "`sessionStorage`" in reference
    assert "`eyvan.local.credential`" in reference
    assert "not OIDC token storage" in reference
