# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Security coverage for Eyvan's automatic loopback development credential."""

from __future__ import annotations

import stat
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.sdk.client.identity import resolve_control_plane_identity
from eyvan.server.api.auth import RequestAuthenticator
from eyvan.server.local.auth import (
    LOCAL_DEVELOPMENT_CREDENTIAL_HEADER,
    LocalDevelopmentCredential,
    LocalDevelopmentCredentialError,
    LocalDevelopmentCredentialStore,
)


def _identity_app(authenticator: RequestAuthenticator) -> FastAPI:
    """Expose one endpoint protected by an injected request authenticator."""

    app = FastAPI()

    @app.get("/identity")
    async def identity(request: Request) -> dict[str, str | list[str]]:
        principal = await authenticator.authenticate(request)
        return {
            "subject": principal.subject,
            "roles": [grant.role.value for grant in principal.grants],
        }

    return app


def _development_headers(*, credential: str | None = None) -> dict[str, str]:
    """Build development-role headers, optionally with the local credential."""

    headers = {
        "X-Eyvan-User": "alice",
        "X-Eyvan-Org": "acme",
        "X-Eyvan-Project": "research",
        "X-Eyvan-Roles": "fleet_admin",
    }
    if credential is not None:
        headers[LOCAL_DEVELOPMENT_CREDENTIAL_HEADER] = credential
    return headers


def test_local_development_credential_store_is_stable_and_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "profile" / "local-development-credential"
    store = LocalDevelopmentCredentialStore(path)

    created = store.load_or_create()
    loaded = store.load()

    assert loaded == created
    assert created.matches(created.value)
    assert not created.matches("forged")
    assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0
    assert stat.S_IMODE(path.parent.stat().st_mode) & 0o077 == 0


def test_local_development_credential_store_rejects_a_readable_secret(tmp_path: Path) -> None:
    path = tmp_path / "profile" / "local-development-credential"
    store = LocalDevelopmentCredentialStore(path)
    store.load_or_create()
    path.chmod(0o644)

    with pytest.raises(LocalDevelopmentCredentialError, match="must not be group- or world-readable"):
        store.load()


def test_local_development_credential_store_rejects_an_exposed_profile_directory(tmp_path: Path) -> None:
    path = tmp_path / "profile" / "local-development-credential"
    store = LocalDevelopmentCredentialStore(path)
    store.load_or_create()
    path.parent.chmod(0o755)

    with pytest.raises(LocalDevelopmentCredentialError, match="must not be group- or world-accessible"):
        store.load()


def test_loopback_client_identity_automatically_attaches_the_owner_credential(tmp_path: Path) -> None:
    credential_path = tmp_path / "profile" / "local-development-credential"
    credential = LocalDevelopmentCredentialStore(credential_path).load_or_create()

    loopback = resolve_control_plane_identity(
        ControlPlaneEndpoint("http://127.0.0.1:8080"),
        dev_user="alice",
        dev_org="acme",
        dev_project="research",
        dev_roles="fleet_admin",
        local_development_credential_path=credential_path,
    )
    remote = resolve_control_plane_identity(
        ControlPlaneEndpoint("https://eyvan.example.com"),
        dev_user="alice",
        dev_org="acme",
        dev_project="research",
        dev_roles="fleet_admin",
        local_development_credential_path=credential_path,
    )
    bearer = resolve_control_plane_identity(
        ControlPlaneEndpoint("http://127.0.0.1:8080"),
        token="oidc-token",
        local_development_credential_path=credential_path,
    )

    assert loopback.headers()[LOCAL_DEVELOPMENT_CREDENTIAL_HEADER] == credential.value
    assert LOCAL_DEVELOPMENT_CREDENTIAL_HEADER not in remote.headers()
    assert bearer.headers() == {"Authorization": "Bearer oidc-token"}


def test_loopback_client_identity_never_creates_a_server_credential(tmp_path: Path) -> None:
    credential_path = tmp_path / "profile" / "local-development-credential"

    identity = resolve_control_plane_identity(
        ControlPlaneEndpoint("http://127.0.0.1:8080"),
        dev_user="alice",
        local_development_credential_path=credential_path,
    )

    assert LOCAL_DEVELOPMENT_CREDENTIAL_HEADER not in identity.headers()
    assert not credential_path.parent.exists()


def test_automatic_loopback_auth_rejects_forged_role_headers_without_credential(tmp_path: Path) -> None:
    credential = LocalDevelopmentCredentialStore(tmp_path / "profile" / "local-development-credential").load_or_create()
    app = _identity_app(
        RequestAuthenticator(
            EyvanSettings(dev_auth_enabled=True),
            local_development_credential=credential,
        )
    )

    with TestClient(app) as client:
        missing = client.get("/identity", headers=_development_headers())
        forged = client.get("/identity", headers=_development_headers(credential="forged"))
        accepted = client.get("/identity", headers=_development_headers(credential=credential.value))

    assert missing.status_code == 401
    assert missing.json() == {"detail": "local Eyvan development credential is required"}
    assert forged.status_code == 401
    assert forged.json() == {"detail": "local Eyvan development credential is required"}
    assert accepted.status_code == 200
    assert accepted.json() == {"subject": "alice", "roles": ["fleet_admin"]}


def test_explicit_development_auth_keeps_header_only_behavior() -> None:
    app = _identity_app(RequestAuthenticator(EyvanSettings(dev_auth_enabled=True)))

    with TestClient(app) as client:
        response = client.get("/identity", headers=_development_headers())

    assert response.status_code == 200
    assert response.json() == {"subject": "alice", "roles": ["fleet_admin"]}


def test_local_development_credential_rejects_a_noncanonical_value() -> None:
    with pytest.raises(LocalDevelopmentCredentialError, match="malformed"):
        LocalDevelopmentCredential("not-a-secret")


# --------------------------------------------------------- the plane's own dashboard


def _dashboard_headers(**overrides: str) -> dict[str, str]:
    """What a browser sends when the page it loaded asks its own origin."""

    headers = {"Sec-Fetch-Site": "same-origin", "Sec-Fetch-Mode": "cors", "Host": "127.0.0.1:8080"}
    headers.update(overrides)
    return headers


def _credential_app(tmp_path: Path) -> tuple[FastAPI, LocalDevelopmentCredential]:
    """An app whose loopback auth demands the owner-only credential."""

    store = LocalDevelopmentCredentialStore(tmp_path / "credential")
    credential = store.load_or_create()
    app = _identity_app(
        RequestAuthenticator(
            EyvanSettings(dev_auth_enabled=True),
            local_development_credential=credential,
        )
    )
    return app, credential


def test_the_dashboard_is_refused_once_a_credential_exists(tmp_path: Path) -> None:
    """Sec-Fetch-Site and Host are request headers any local process can send.

    curl to 127.0.0.1 already sends Host, and one more forged header made the
    plane answer as the owner. A check that accepted them *instead of* the 0600
    credential was a bypass, so once a credential has been issued it is
    required of every caller -- the plane's own dashboard included. The
    dashboard presents the credential like any other client (the development
    launcher hands it the value the 0600 file holds).
    """

    app, _credential = _credential_app(tmp_path)
    with TestClient(app) as client:
        response = client.get("/identity", headers=_dashboard_headers())

    assert response.status_code == 401
    assert response.json() == {"detail": "local Eyvan development credential is required"}


def test_the_dashboard_presenting_the_credential_gets_the_owners_defaults(tmp_path: Path) -> None:
    """The page this plane served acts as this machine's owner, like before."""

    app, credential = _credential_app(tmp_path)
    headers = {**_dashboard_headers(), LOCAL_DEVELOPMENT_CREDENTIAL_HEADER: credential.value}
    with TestClient(app) as client:
        response = client.get("/identity", headers=headers)

    assert response.status_code == 200
    assert "fleet_admin" in response.json()["roles"]


def test_a_non_ascii_credential_header_answers_401_and_never_500(tmp_path: Path) -> None:
    """``compare_digest`` raises ``TypeError`` for non-ASCII ``str`` arguments.

    The header is caller-controlled, so unauthenticated garbage must come back
    as a refused credential, not as a server error.
    """

    app, _credential = _credential_app(tmp_path)
    # A caller puts raw bytes on the wire and the ASGI server decodes them as
    # latin-1, so the app sees the str that httpx refuses to send itself.
    headers = {**_dashboard_headers(), LOCAL_DEVELOPMENT_CREDENTIAL_HEADER: "tök".encode("latin-1")}
    with TestClient(app) as client:
        response = client.get("/identity", headers=headers)

    assert response.status_code == 401


def test_another_sites_page_reaching_this_plane_is_refused(tmp_path: Path) -> None:
    """The attack the credential exists for: any page a browser visits can send
    requests to 127.0.0.1. The browser labels those cross-site, and page script
    cannot forge that header -- Sec-Fetch-Site is on the forbidden list."""

    with TestClient(_credential_app(tmp_path)[0]) as client:
        response = client.get("/identity", headers=_dashboard_headers(**{"Sec-Fetch-Site": "cross-site"}))

    assert response.status_code == 401


def test_a_non_browser_client_on_loopback_still_needs_the_credential(tmp_path: Path) -> None:
    """The property that keeps this from meaning "loopback is trusted".

    Another account's script on this host reaches the same port and sends no
    fetch metadata at all. It is refused exactly as before.
    """

    with TestClient(_credential_app(tmp_path)[0]) as client:
        response = client.get("/identity")

    assert response.status_code == 401


def test_a_rebound_dns_name_is_refused_though_the_browser_calls_it_same_origin(tmp_path: Path) -> None:
    """Same-origin is computed against the origin the page came from, so an
    attacker resolving their own name to 127.0.0.1 produces a genuinely
    same-origin request. It carries their name in Host, which is why Host must
    be a loopback literal and never a name."""

    with TestClient(_credential_app(tmp_path)[0]) as client:
        response = client.get("/identity", headers=_dashboard_headers(Host="evil.example.com"))

    assert response.status_code == 401


def test_a_top_level_navigation_to_an_api_url_is_not_the_dashboard(tmp_path: Path) -> None:
    """Typing an API URL into the address bar is same-origin too, and there is
    no reason for a navigation to authenticate."""

    with TestClient(_credential_app(tmp_path)[0]) as client:
        response = client.get("/identity", headers=_dashboard_headers(**{"Sec-Fetch-Mode": "navigate"}))

    assert response.status_code == 401


def test_the_credential_still_authenticates_a_caller_that_has_it(tmp_path: Path) -> None:
    """The CLI's path is unchanged; the waiver is additional, not a replacement."""

    store = LocalDevelopmentCredentialStore(tmp_path / "credential")
    credential = store.load_or_create()
    app = _identity_app(
        RequestAuthenticator(EyvanSettings(dev_auth_enabled=True), local_development_credential=credential)
    )

    with TestClient(app) as client:
        response = client.get("/identity", headers=_development_headers(credential=credential.value))

    assert response.status_code == 200


def test_a_header_the_dashboard_sends_still_wins_over_the_default(tmp_path: Path) -> None:
    """Otherwise the scope picker would be inert: every request would act in the
    default project however the operator set it."""

    app, credential = _credential_app(tmp_path)
    headers = {
        **_dashboard_headers(),
        LOCAL_DEVELOPMENT_CREDENTIAL_HEADER: credential.value,
        "X-Eyvan-User": "someone",
        "X-Eyvan-Roles": "viewer",
    }
    with TestClient(app) as client:
        response = client.get("/identity", headers=headers)

    assert response.status_code == 200
    assert response.json() == {"subject": "someone", "roles": ["project_viewer"]}
