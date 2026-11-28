# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Observable regressions for SDK identity overrides and loopback diagnosis."""

from __future__ import annotations

from types import MappingProxyType

import httpx
import pytest
from starlette.requests import Request

from eyvan.config import EyvanSettings
from eyvan.records import Principal, Role, RoleGrant
from eyvan.sdk.client.identity import ClientIdentity
from eyvan.sdk.client.transport import AsyncApiTransport, EyvanHTTPError, SyncApiTransport, raise_for_status
from eyvan.server.api.auth import RequestAuthenticator
from eyvan.server.local.auth import LOCAL_DEVELOPMENT_CREDENTIAL_REQUIRED_DETAIL


class _LocalTokenVerifier:
    async def verify(self, token: str) -> Principal:
        return Principal(
            subject=token, grants=(RoleGrant(role=Role.PROJECT_VIEWER, org_id="org", project_id="project"),)
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("casing", ["matching", "lower"])
@pytest.mark.parametrize("identity_kind", ["bearer", "development", "custom"])
async def test_request_overrides_injected_headers_on_wire_and_in_authenticator(mode, casing, identity_kind):
    if identity_kind == "bearer":
        identity = ClientIdentity(token="default-principal")
        overrides = {"Authorization": "Bearer requested-principal"}
        defaults = identity.headers()
        transport_defaults = {}
    elif identity_kind == "development":
        identity = ClientIdentity(dev_user="default-user", dev_org="default-org", dev_project="default-project")
        overrides = {
            "X-Eyvan-User": "requested-user",
            "X-Eyvan-Org": "requested-org",
            "X-Eyvan-Project": "requested-project",
            "X-Eyvan-Roles": "viewer",
        }
        defaults = identity.headers()
        transport_defaults = {}
    else:
        identity = ClientIdentity()
        overrides = {"X-Custom": "requested-value"}
        defaults = {"X-Custom": "default-value"}
        transport_defaults = defaults
    caller = MappingProxyType({name.lower() if casing == "lower" else name: value for name, value in overrides.items()})
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={})

    client_type = httpx.Client if mode == "sync" else httpx.AsyncClient
    transport_type = SyncApiTransport if mode == "sync" else AsyncApiTransport
    raw = client_type(base_url="http://test", headers=defaults, transport=httpx.MockTransport(handler))
    original_client_headers = tuple(raw.headers.raw)
    original_caller_headers = tuple(caller.items())
    transport = transport_type("http://test", client=raw, identity=identity, headers=transport_defaults)
    try:
        if mode == "sync":
            response = transport.request_checked("GET", "/api/v1/example", headers=caller)
        else:
            response = await transport.request_checked("GET", "/api/v1/example", headers=caller)
        assert response.status_code == 200
        assert tuple(raw.headers.raw) == original_client_headers
    finally:
        if mode == "sync":
            raw.close()
        else:
            await raw.aclose()
    assert tuple(caller.items()) == original_caller_headers
    assert len(seen) == 1
    request = seen[0]
    for name, value in overrides.items():
        assert request.headers.get_list(name) == [value]

    if identity_kind != "custom":
        # An ASGI server lowercases wire names; preserve every emitted value and its order.
        inbound = Request(
            {
                "type": "http",
                "method": "GET",
                "path": "/api/v1/example",
                "headers": [(name.lower(), value) for name, value in request.headers.raw],
            }
        )
        authenticator = RequestAuthenticator(EyvanSettings(dev_auth_enabled=True), verifier=_LocalTokenVerifier())
        principal = await authenticator.authenticate(inbound)
        if identity_kind == "bearer":
            assert principal.subject == "requested-principal"
        else:
            assert principal.subject == "requested-user"
            assert principal.grants == (
                RoleGrant(role=Role.PROJECT_VIEWER, org_id="requested-org", project_id="requested-project"),
            )


@pytest.mark.parametrize(
    ("url", "endpoint"),
    [
        ("http://[::1]:8080/api/v1/fleets?redact=this#fragment", "http://[::1]:8080"),
        ("https://[::1]:8443/api/v1/fleets", "https://[::1]:8443"),
        ("http://[::1]/api/v1/fleets", "http://[::1]"),
        ("https://127.0.0.1:8443/api/v1/fleets", "https://127.0.0.1:8443"),
        ("https://localhost/api/v1/fleets", "https://localhost"),
    ],
)
def test_loopback_refusal_preserves_typed_error_and_url_authority(url, endpoint, tmp_path, monkeypatch):
    # Diagnosis must never consult this operator's actual credential or daemon records.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("eyvan.server.local.auth.DEFAULT_LOCAL_DEVELOPMENT_CREDENTIAL_PATH", tmp_path / "credential")
    request = httpx.Request("GET", url)
    response = httpx.Response(401, request=request, json={"detail": LOCAL_DEVELOPMENT_CREDENTIAL_REQUIRED_DETAIL})

    with pytest.raises(EyvanHTTPError) as caught:
        raise_for_status(response)

    error = caught.value
    assert error.status_code == 401
    assert error.response is response
    assert error.request is request
    assert error.detail == LOCAL_DEVELOPMENT_CREDENTIAL_REQUIRED_DETAIL
    assert f"A control plane is listening at {endpoint} but refused" in str(error)
    assert "no PID record for that endpoint" in str(error)
    assert "redact=this" not in str(error)
    assert "#fragment" not in str(error)
