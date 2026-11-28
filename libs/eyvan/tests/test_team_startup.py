# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Team mode on a network bind: TLS, a usable public URL, and a Secure session cookie."""

from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.server.api.app import create_app
from eyvan.server.api.server_bind import ServerBindPolicy
from eyvan.server.api.server_startup import (
    ServerStartupSettings,
    StartupConfigurationSource,
    unsatisfied_startup_requirements,
)

_NETWORK = ServerBindPolicy.from_host("0.0.0.0")
_LOOPBACK = ServerBindPolicy.from_host("127.0.0.1")
_TLS = {"tls_cert_file": Path("/nonexistent/server.pem"), "tls_key_file": Path("/nonexistent/server-key.pem")}


def _names(settings: EyvanSettings, bind: ServerBindPolicy = _NETWORK) -> tuple[str, ...]:
    return tuple(
        requirement.environment_variable
        for requirement in unsatisfied_startup_requirements(bind, settings, source=StartupConfigurationSource.EXPLICIT)
    )


def _team(**overrides: object) -> EyvanSettings:
    return EyvanSettings(team_mode=True, metrics_token="scrape-token", artifact_token_key="k" * 32, **overrides)


def test_a_network_team_plane_without_tls_is_refused() -> None:
    names = _names(_team(public_url="https://plane.example.com"))
    assert "EYVAN_TLS_CERT_FILE" in names


def test_a_network_team_plane_with_tls_and_a_real_public_url_starts() -> None:
    assert _names(_team(public_url="https://plane.example.com", **_TLS)) == ()


@pytest.mark.parametrize(
    "public_url",
    [
        "http://127.0.0.1:8080",  # the default: invite links nobody else can open
        "https://localhost:8080",
        "https://[::1]:8080",
        "http://plane.example.com",  # a password typed into an http page, with no explicit opt-in
        "not a url",
    ],
)
def test_a_network_team_plane_needs_a_usable_public_url(public_url: str) -> None:
    assert _names(_team(public_url=public_url, **_TLS)) == ("EYVAN_PUBLIC_URL",)


def test_the_explicit_plaintext_opt_in_also_accepts_an_http_public_url() -> None:
    settings = _team(public_url="http://10.0.0.5:8080", allow_insecure_plaintext_bind=True)
    assert _names(settings) == ()


def test_the_public_url_requirement_is_team_mode_only() -> None:
    settings = EyvanSettings(metrics_token="scrape-token", artifact_token_key="k" * 32, **_TLS)
    assert "EYVAN_PUBLIC_URL" not in _names(settings)


def test_a_loopback_team_plane_needs_nothing() -> None:
    assert _names(EyvanSettings(team_mode=True), _LOOPBACK) == ()


def test_create_app_refuses_a_misconfigured_network_team_plane() -> None:
    with pytest.raises(ValueError, match="EYVAN_PUBLIC_URL"):
        ServerStartupSettings.from_settings(_NETWORK, _team(**_TLS))
    with pytest.raises(ValueError, match="EYVAN_TLS_CERT_FILE"):
        create_app(settings=_team(public_url="https://plane.example.com"), server_bind=_NETWORK)


def _session_cookie(settings: EyvanSettings) -> str:
    http = TestClient(create_app(settings=settings), base_url="http://127.0.0.1:8080")
    invite = http.post(
        "/api/v1/team/invites",
        json={"email": "alice@example.com", "subject": "alice"},
        headers={"X-Eyvan-User": "root", "X-Eyvan-Roles": "fleet_admin"},
    ).json()
    accepted = http.post(
        "/api/v1/team/invites/accept", json={"token": invite["token"], "password": "correct horse battery"}
    )
    assert accepted.status_code == 200, accepted.text
    return accepted.headers["set-cookie"]


def test_the_session_cookie_is_secure_when_this_process_terminates_tls() -> None:
    cookie = _session_cookie(EyvanSettings(dev_auth_enabled=True, team_mode=True, artifact_token_key="k" * 32, **_TLS))
    assert "secure" in [part.strip().lower() for part in cookie.split(";")]


def test_the_session_cookie_is_not_secure_on_plain_loopback_http() -> None:
    cookie = _session_cookie(EyvanSettings(dev_auth_enabled=True, team_mode=True, artifact_token_key="k" * 32))
    assert "secure" not in [part.strip().lower() for part in cookie.split(";")]
