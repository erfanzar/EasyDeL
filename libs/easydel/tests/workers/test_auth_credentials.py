"""API keys are read only from ``Authorization: Bearer`` and ``X-API-Key``; queries and bodies are ignored."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from easydel.inference.esurge.server.api_server import eSurgeApiServer
from easydel.workers.esurge.auth import EnhancedApiKeyManager
from easydel.workers.esurge.auth.credentials import api_key_from_headers
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from starlette.datastructures import Headers

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"Authorization": "Bearer sk-AbC"}, "sk-AbC"),
        ({"authorization": "bearer sk-AbC"}, "sk-AbC"),
        ({"AUTHORIZATION": "BEARER   sk-AbC  "}, "sk-AbC"),
        ({"X-API-Key": "sk-AbC"}, "sk-AbC"),
        ({"x-api-key": " sk-AbC "}, "sk-AbC"),
        ({"Authorization": "Bearer sk-bearer", "X-API-Key": "sk-header"}, "sk-bearer"),
        ({"Authorization": "Basic dXNlcjpwYXNz", "X-API-Key": "sk-header"}, "sk-header"),
        ({"Authorization": "Basic dXNlcjpwYXNz"}, None),
        ({"Authorization": "Bearer "}, None),
        ({"X-API-Key": "   "}, None),
        ({}, None),
    ],
)
def test_header_extraction(headers, expected):
    assert api_key_from_headers(headers) == expected
    assert api_key_from_headers(Headers(headers=headers)) == expected


def _app(manager: EnhancedApiKeyManager) -> TestClient:
    server = object.__new__(eSurgeApiServer)
    server._require_api_key = True
    server.auth_manager = manager
    app = FastAPI()

    async def probe(raw_request: Request):
        try:
            return {"key_ok": server._authorize_request(raw_request, endpoint="/v1/models") is not None}
        except HTTPException as exc:
            from fastapi.responses import JSONResponse

            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    app.add_api_route("/probe", probe, methods=["GET", "POST"])
    return TestClient(app)


def test_server_ignores_query_string_and_body_keys():
    manager = EnhancedApiKeyManager(require_api_key=True, enable_persistence=False)
    key, _ = manager.generate_api_key(name="k")
    with _app(manager) as client:
        assert client.get("/probe", headers={"Authorization": f"Bearer {key}"}).status_code == 200
        assert client.get("/probe", headers={"X-API-Key": key}).status_code == 200
        assert client.get("/probe", params={"api_key": key}).status_code == 401
        assert client.get("/probe", params={"user": key}).status_code == 401
        for field in ("api_key", "user", "token", "authorization"):
            assert client.post("/probe", json={field: key}).status_code == 401
        assert client.get("/probe", headers={"Authorization": f"Bearer {key.swapcase()}"}).status_code == 401


def test_admin_routes_read_keys_from_headers_only():
    from easydel.inference.esurge.server.auth_endpoints import AuthEndpointsMixin
    from easydel.workers.esurge.auth import ApiKeyRole

    manager = EnhancedApiKeyManager(require_api_key=True, enable_persistence=False)
    admin_key, _ = manager.generate_api_key(name="admin", role=ApiKeyRole.ADMIN)
    holder = SimpleNamespace(auth_manager=manager)

    def check(headers, query=None):
        request = SimpleNamespace(
            headers=Headers(headers=headers),
            query_params=query or {},
            client=SimpleNamespace(host="10.0.0.1"),
            state=SimpleNamespace(),
            url=SimpleNamespace(path="/v1/admin/keys"),
        )
        AuthEndpointsMixin._require_admin_role(holder, request)

    check({"X-API-Key": admin_key})
    with pytest.raises(HTTPException) as raised:
        check({}, query={"api_key": admin_key})
    assert raised.value.status_code == 401
