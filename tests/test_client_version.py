# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Client/server version negotiation: parsing, settings validation, and the API gate."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from eyvan._internal.client_version import (
    CLIENT_VERSION_HEADER,
    client_version,
    is_client_supported,
    parse_version,
    stale_client_message,
    validate_minimum_client_version,
)
from eyvan.config import EyvanSettings
from eyvan.sdk.client.transport import AsyncApiTransport, SyncApiTransport
from eyvan.server.api.app import create_app


@pytest.mark.parametrize(
    ("raw", "expected"),
    (
        ("0.1.0", (0, 1, 0)),
        ("1.2", (1, 2)),
        ("10.0.3.4", (10, 0, 3, 4)),
        ("  0.1.0  ", (0, 1, 0)),
        ("0", (0,)),
    ),
)
def test_parse_version_accepts_dotted_integers(raw: str, expected: tuple[int, ...]) -> None:
    assert parse_version(raw) == expected


@pytest.mark.parametrize(
    "raw",
    ("", "   ", "0.1.0rc1", "1.2.x", "v1.2.0", "1..2", "-1.0", None, "1.2.3.4.5.6.7.8.9"),
)
def test_parse_version_rejects_anything_not_purely_numeric(raw: str | None) -> None:
    assert parse_version(raw) is None


def test_unset_floor_accepts_every_client() -> None:
    assert is_client_supported("0.0.1", None) is True
    assert is_client_supported(None, None) is True
    assert is_client_supported("garbage", "") is True


def test_missing_or_unparseable_header_is_accepted() -> None:
    """Browsers, curl, and other non-SDK callers never send the header."""

    assert is_client_supported(None, "9.9.9") is True
    assert is_client_supported("not-a-version", "9.9.9") is True


@pytest.mark.parametrize(
    ("reported", "minimum", "supported"),
    (
        ("0.2.0", "0.2.0", True),
        ("0.2.1", "0.2.0", True),
        ("1.0.0", "0.2.0", True),
        ("0.1.9", "0.2.0", False),
        ("0.1.0", "0.2.0", False),
        # Differing segment counts compare by zero-padding, not lexically.
        ("0.2", "0.2.0", True),
        ("0.2.0", "0.2", True),
        ("0.2", "0.2.1", False),
        ("10.0.0", "9.0.0", True),
    ),
)
def test_version_floor_comparison(reported: str, minimum: str, supported: bool) -> None:
    assert is_client_supported(reported, minimum) is supported


def test_settings_reject_a_malformed_client_version_floor() -> None:
    with pytest.raises(ValueError, match="min_client_version must be dotted integers"):
        EyvanSettings(min_client_version="1.0.0-beta")

    validate_minimum_client_version(None)
    validate_minimum_client_version("0.1.0")


def test_settings_read_the_floor_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EYVAN_MIN_CLIENT_VERSION", raising=False)
    assert EyvanSettings.from_env().min_client_version is None

    monkeypatch.setenv("EYVAN_MIN_CLIENT_VERSION", "0.3.0")
    assert EyvanSettings.from_env().min_client_version == "0.3.0"

    monkeypatch.setenv("EYVAN_MIN_CLIENT_VERSION", "")
    assert EyvanSettings.from_env().min_client_version is None


def test_stale_client_message_names_both_versions_and_the_remedy() -> None:
    message = stale_client_message("0.1.0", "0.3.0")

    assert "0.1.0" in message
    assert "0.3.0" in message
    assert "Upgrade the client" in message
    assert "unknown" in stale_client_message(None, "0.3.0")


def test_sync_transport_sends_the_client_version_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    raw = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")
    transport = SyncApiTransport(client=raw)
    transport.request("GET", "/api/v1/workloads")

    assert seen[0].headers[CLIENT_VERSION_HEADER] == client_version()
    raw.close()


async def test_async_transport_sends_the_client_version_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    raw = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://test")
    transport = AsyncApiTransport(client=raw)
    await transport.request("GET", "/api/v1/workloads")

    assert seen[0].headers[CLIENT_VERSION_HEADER] == client_version()
    await raw.aclose()


def test_api_rejects_a_stale_client_before_authentication() -> None:
    """The gate runs as middleware, so skew is reported even on an unauthenticated call."""

    app = create_app(settings=EyvanSettings(dev_auth_enabled=True, min_client_version="0.3.0"))
    with TestClient(app) as client:
        response = client.get("/api/v1/workloads", headers={CLIENT_VERSION_HEADER: "0.1.0"})

    assert response.status_code == 426
    payload = response.json()
    assert payload["code"] == "client_too_old"
    assert payload["context"] == {"client_version": "0.1.0", "minimum_client_version": "0.3.0"}
    assert "0.3.0 or newer" in payload["detail"]


def test_api_rejects_a_stale_client_through_the_v1_compatibility_alias() -> None:
    app = create_app(settings=EyvanSettings(dev_auth_enabled=True, min_client_version="0.3.0"))
    with TestClient(app) as client:
        response = client.get("/v1/workloads", headers={CLIENT_VERSION_HEADER: "0.1.0"})

    assert response.status_code == 426
    assert response.json()["code"] == "client_too_old"


def test_api_admits_a_current_client_and_a_header_less_caller() -> None:
    app = create_app(settings=EyvanSettings(dev_auth_enabled=True, min_client_version="0.3.0"))
    with TestClient(app) as client:
        current = client.get("/api/v1/workloads", headers={CLIENT_VERSION_HEADER: "0.3.0"})
        headerless = client.get("/api/v1/workloads")

    assert current.status_code != 426
    assert headerless.status_code != 426


def test_api_without_a_configured_floor_admits_every_client() -> None:
    app = create_app(settings=EyvanSettings(dev_auth_enabled=True))
    with TestClient(app) as client:
        response = client.get("/api/v1/workloads", headers={CLIENT_VERSION_HEADER: "0.0.1"})

    assert response.status_code != 426
