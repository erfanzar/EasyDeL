from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from fastapi import HTTPException, Request

from easydel.inference.esurge.server.auth_endpoints import AuthEndpointsMixin, UpdateApiKeyRequest
from easydel.workers.esurge.auth.auth_models import (
    ApiKeyMetadata,
    ApiKeyPermissions,
    ApiKeyRole,
    QuotaConfig,
    RateLimitConfig,
)


def _make_request() -> Request:
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": "/v1/admin/keys",
        "raw_path": b"/v1/admin/keys",
        "query_string": b"",
        "headers": [(b"authorization", b"Bearer sk-admin")],
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    }
    return Request(scope)


def _make_key_metadata() -> ApiKeyMetadata:
    return ApiKeyMetadata(
        key_id="key-1",
        key_prefix="sk-key-1...",
        hashed_key="hash-1",
        name="test-key",
        role=ApiKeyRole.USER,
        rate_limits=RateLimitConfig(
            requests_per_minute=120,
            requests_per_hour=1000,
            requests_per_day=3000,
            tokens_per_minute=10_000,
            tokens_per_hour=100_000,
            tokens_per_day=300_000,
        ),
        quota=QuotaConfig(
            max_total_tokens=1_000_000,
            max_total_requests=10_000,
            monthly_token_limit=100_000,
            monthly_request_limit=1000,
        ),
        permissions=ApiKeyPermissions(
            allowed_models=["model-a"],
            allowed_endpoints=["/v1/chat/completions"],
            allowed_ip_addresses=["127.0.0.1"],
            blocked_ip_addresses=["10.0.0.1"],
            enable_streaming=True,
            enable_function_calling=True,
            max_tokens_per_request=4096,
        ),
    )


class _DummyAuthManager:
    def __init__(self, metadata: ApiKeyMetadata):
        self._metadata = metadata
        self.last_update_kwargs: dict | None = None

    def validate_key(self, key: str):
        if key == "sk-admin":
            return SimpleNamespace(role=ApiKeyRole.ADMIN, name="admin")
        return None

    def list_keys(self, role=None, status=None):
        del role, status
        return []

    def get_key_by_id(self, key_id: str):
        if key_id == self._metadata.key_id:
            return self._metadata
        return None

    def update_key(self, **kwargs):
        self.last_update_kwargs = kwargs
        return kwargs.get("key_id") == self._metadata.key_id


class _DummyServer(AuthEndpointsMixin):
    def __init__(self, metadata: ApiKeyMetadata):
        self.auth_manager = _DummyAuthManager(metadata)


def test_list_api_keys_invalid_role_returns_400():
    server = _DummyServer(_make_key_metadata())

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(server.list_api_keys_endpoint(_make_request(), role="invalid-role"))

    assert exc_info.value.status_code == 400
    assert "Invalid role filter" in str(exc_info.value.detail)


def test_list_api_keys_invalid_status_returns_400():
    server = _DummyServer(_make_key_metadata())

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(server.list_api_keys_endpoint(_make_request(), status="invalid-status"))

    assert exc_info.value.status_code == 400
    assert "Invalid status filter" in str(exc_info.value.detail)


def test_update_api_key_accepts_zero_rate_limits_and_preserves_unset_fields():
    metadata = _make_key_metadata()
    server = _DummyServer(metadata)
    request = UpdateApiKeyRequest(requests_per_minute=0)

    asyncio.run(server.update_api_key_endpoint(metadata.key_id, request, _make_request()))

    update_kwargs = server.auth_manager.last_update_kwargs
    assert update_kwargs is not None

    rate_limits = update_kwargs["rate_limits"]
    assert isinstance(rate_limits, RateLimitConfig)
    assert rate_limits.requests_per_minute == 0
    assert rate_limits.requests_per_hour == metadata.rate_limits.requests_per_hour
    assert rate_limits.requests_per_day == metadata.rate_limits.requests_per_day


def test_update_api_key_allows_explicit_null_permissions():
    metadata = _make_key_metadata()
    server = _DummyServer(metadata)
    request = UpdateApiKeyRequest(allowed_models=None)

    asyncio.run(server.update_api_key_endpoint(metadata.key_id, request, _make_request()))

    update_kwargs = server.auth_manager.last_update_kwargs
    assert update_kwargs is not None

    permissions = update_kwargs["permissions"]
    assert isinstance(permissions, ApiKeyPermissions)
    assert permissions.allowed_models is None
    assert permissions.allowed_endpoints == metadata.permissions.allowed_endpoints
    assert permissions.enable_streaming == metadata.permissions.enable_streaming
