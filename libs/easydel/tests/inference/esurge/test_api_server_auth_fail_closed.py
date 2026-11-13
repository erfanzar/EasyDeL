"""``eSurgeApiServer`` refuses requests whenever ``require_api_key=True``, whatever state its auth manager is in."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from easydel.inference.esurge.server.api_server import eSurgeApiServer
from easydel.workers.esurge.auth import EnhancedApiKeyManager
from fastapi import HTTPException

pytestmark = pytest.mark.cpu_ok


def _server(*, require: bool, manager) -> eSurgeApiServer:
    server = object.__new__(eSurgeApiServer)
    server._require_api_key = require
    server.auth_manager = manager
    return server


def _request(headers: dict[str, str] | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        headers=headers or {},
        query_params={},
        client=SimpleNamespace(host="10.0.0.1"),
        state=SimpleNamespace(),
    )


def test_required_auth_without_a_manager_is_refused_not_opened():
    server = _server(require=True, manager=None)
    with pytest.raises(HTTPException) as raised:
        server._authorize_request(_request({"Authorization": "Bearer anything-long-enough"}), endpoint="/v1/models")
    assert raised.value.status_code == 401


def test_required_auth_is_enforced_even_when_the_manager_reports_disabled():
    manager = EnhancedApiKeyManager(require_api_key=False, enable_persistence=False)
    assert manager.enabled is False
    server = _server(require=True, manager=manager)
    with pytest.raises(HTTPException) as raised:
        server._authorize_request(_request(), endpoint="/v1/models")
    assert raised.value.status_code == 401


def test_required_auth_accepts_only_a_registered_key():
    manager = EnhancedApiKeyManager(require_api_key=True, enable_persistence=False)
    key, _ = manager.generate_api_key(name="k")
    server = _server(require=True, manager=manager)
    assert server._authorize_request(_request({"Authorization": f"Bearer {key}"}), endpoint="/v1/models") == key
    for wrong in (key.upper(), key[:-1], "sk-not-a-registered-key-at-all"):
        with pytest.raises(HTTPException) as raised:
            server._authorize_request(_request({"Authorization": f"Bearer {wrong}"}), endpoint="/v1/models")
        assert raised.value.status_code == 401


def test_optional_auth_with_no_keys_stays_open():
    server = _server(require=False, manager=EnhancedApiKeyManager(require_api_key=False, enable_persistence=False))
    assert server._authorize_request(_request(), endpoint="/v1/models") is None
