# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Connect must respect the target's local credential and URL boundaries."""

from __future__ import annotations

import os
from pathlib import Path

import httpx
import pytest
from click.testing import CliRunner

import eyvan.sdk.cli as cli
from eyvan._internal import plane_tls
from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.sdk.client.identity import ClientIdentity
from eyvan.sdk.commands.connect import ConnectTarget, connect, resolve_url
from eyvan.sdk.commands.state import CliSession, ContextOverrides
from eyvan.sdk.contexts import FileContextStore
from eyvan.server.local.auth import LOCAL_DEVELOPMENT_CREDENTIAL_HEADER, LocalDevelopmentCredentialStore


@pytest.mark.parametrize(
    ("host", "root_options", "connect_options", "expect_local", "bearer"),
    [
        ("https://remote.invalid:443", [], [], False, None),
        ("https://remote.invalid:443", ["--user", "explicit-user", "--role", "viewer"], [], False, None),
        ("https://remote.invalid:443", ["--token", "root-bearer"], [], False, "root-bearer"),
        ("https://remote.invalid:443", [], ["--token", "connect-bearer"], False, "connect-bearer"),
        ("http://127.0.0.1:8099", [], [], True, None),
        ("http://localhost:8099", [], [], True, None),
        ("http://[::1]:8099", [], [], True, None),
        ("https://[2001:db8::1]:9443", [], [], False, None),
    ],
)
def test_connect_restricts_owner_credential_to_loopback_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    host: str,
    root_options: list[str],
    connect_options: list[str],
    expect_local: bool,
    bearer: str | None,
) -> None:
    """Exercise the real old-loopback root identity and rejected remote probe."""

    for name in tuple(os.environ):
        if name.startswith("EYVAN_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("EYVAN_CONFIG_FILE", str(tmp_path / "contexts.json"))
    credential = LocalDevelopmentCredentialStore().load_or_create().value
    monkeypatch.setattr(cli, "discover_node_context", lambda: None)
    monkeypatch.setattr(cli, "discover_node_token_or_warn", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(cli, "EyvanClient", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(plane_tls, "control_plane_httpx_options", lambda: {})
    requests: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/health"):
            return httpx.Response(200, json={"status": "ok"})
        assert request.url.path.endswith("/session")
        return httpx.Response(403, json={"detail": "refused"})

    real_client = httpx.Client
    monkeypatch.setattr(
        httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(respond), **kwargs),
    )
    result = CliRunner().invoke(cli.main, [*root_options, "connect", "--host", host, *connect_options])

    assert result.exit_code == 1, result.output
    assert "refused this identity" in result.output
    assert [request.url.path for request in requests] == ["/api/v1/health", "/api/v1/session"]
    for request in requests:
        assert request.url == httpx.URL(f"{host}{request.url.path}")
        if expect_local:
            assert request.headers[LOCAL_DEVELOPMENT_CREDENTIAL_HEADER] == credential
        else:
            assert LOCAL_DEVELOPMENT_CREDENTIAL_HEADER not in request.headers
        if bearer:
            assert request.headers["Authorization"] == f"Bearer {bearer}"
        if "--user" in root_options:
            assert request.headers["X-Eyvan-User"] == "explicit-user"
            assert request.headers["X-Eyvan-Roles"] == "viewer"
    assert not (tmp_path / "contexts.json").exists()


@pytest.mark.parametrize(
    ("host", "port", "expected"),
    [
        ("http://[::1]:8080", None, "http://[::1]:8080"),
        ("http://[::1]:8080", 8099, "http://[::1]:8099"),
        ("https://[2001:db8::1]:9443", None, "https://[2001:db8::1]:9443"),
        ("https://[2001:db8::1]:9443", 8443, "https://[2001:db8::1]:8443"),
        ("[::1]", None, "http://[::1]:8080"),
        ("https://[2001:db8::1]/plane/", None, "https://[2001:db8::1]:8080/plane"),
    ],
)
def test_connect_preserves_ipv6_authorities(tmp_path: Path, host: str, port: int | None, expected: str) -> None:
    """URL resolution, endpoint validation, and the selected probe agree."""

    url = resolve_url(host, port)
    assert url == expected
    assert ControlPlaneEndpoint.parse(url).url == expected
    request = httpx.Request("GET", f"{url}/api/v1/health")
    assert request.url == httpx.URL(f"{expected}/api/v1/health")
    store = FileContextStore(tmp_path / "contexts.json")
    session = CliSession(
        client=None,  # type: ignore[arg-type]
        identity=ClientIdentity(dev_user="test-user"),
        context_store=store,
        context_config=store.load(),
        base_url="http://127.0.0.1:8080",
        org_id=None,
        project_id=None,
        overrides=ContextOverrides(),
    )

    def record_probe(selected_url: str, **_kwargs: object) -> ConnectTarget:
        return ConnectTarget(selected_url, False, "test refusal")

    outcome = connect(
        session,
        host=host,
        port=port,
        fleet=None,
        token=None,
        org_id=None,
        project_id=None,
        probe_plane=record_probe,
    )
    assert outcome.target.url == expected
    assert outcome.saved_context is None
    assert not store.resolved_path.exists()
