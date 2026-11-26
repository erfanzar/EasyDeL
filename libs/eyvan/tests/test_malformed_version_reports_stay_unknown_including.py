# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Malformed version reports stay unknown, including on the pre-auth API gate."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

from eyvan._internal.client_version import CLIENT_VERSION_HEADER, is_client_supported, parse_version
from eyvan._internal.platform import EyvanPlatformConfig
from eyvan.config import EyvanSettings
from eyvan.server.api.app import create_app


@pytest.mark.parametrize("raw", ("².0", "9" * 5000), ids=("nondecimal-digit", "oversized-decimal"))
def test_malformed_numeric_reports_are_unknown(raw: str) -> None:
    assert parse_version(raw) is None
    assert is_client_supported(raw, "0.3.0") is True


def test_version_parser_bounds_total_input_before_stripping() -> None:
    assert parse_version("9" * 512) == (10**512 - 1,)
    assert parse_version("9" * 513) is None
    assert parse_version("1" * 256 + "." + "2" * 256) is None
    assert parse_version(" " * 513 + "0.3.0") is None
    assert parse_version("\u0660.\u0663.\u0660") == (0, 3, 0)
    assert parse_version(" 0.3.0 ") == (0, 3, 0)


def test_decimal_report_beyond_active_integer_limit_is_unknown() -> None:
    limit = sys.get_int_max_str_digits()
    if not limit:
        pytest.skip("This interpreter has disabled its integer-string conversion limit")
    assert parse_version("9" * (limit + 1)) is None


@pytest.mark.parametrize("minimum", ("0.3.0", None), ids=("enabled-floor", "disabled-floor"))
@pytest.mark.parametrize("path", ("/api/v1/workloads", "/v1/workloads"))
@pytest.mark.parametrize("malformed", (b"\xb2.0", b"9" * 5000), ids=("nondecimal-digit", "oversized-decimal"))
async def test_raw_malformed_numeric_headers_reach_authentication(
    minimum: str | None, path: str, malformed: bytes, tmp_path: Path
) -> None:
    app = create_app(
        settings=EyvanSettings(
            controller_enabled=False,
            dev_auth_enabled=False,
            min_client_version=minimum,
            artifact_token_key="writer-026-test-key-not-a-real-secret-0000",
            artifact_store_uri=str(tmp_path / "artifacts"),
        ),
        platform_config=EyvanPlatformConfig(),
    )
    # Exact bytes are important: TestClient re-encodes non-ASCII text headers.
    # ASGITransport invokes the real middleware/authenticator without a listener
    # or a lifespan/controller loop, and no request can reach provider I/O.
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        headerless = await client.get(path)
        assert headerless.status_code == 401
        assert headerless.json() == {"detail": "authentication required"}
        for reported in (b"not-a-version", b"0.3.0", malformed):
            response = await client.get(path, headers=[(CLIENT_VERSION_HEADER.encode("ascii"), reported)])
            assert response.status_code == headerless.status_code, reported[:30]
            assert response.json() == headerless.json()
        stale = await client.get(path, headers={CLIENT_VERSION_HEADER: "0.1.0"})
        if minimum is None:
            assert stale.status_code == 401
            assert stale.json() == headerless.json()
        else:
            assert stale.status_code == 426
            assert stale.json()["code"] == "client_too_old"
            assert stale.json()["context"] == {"client_version": "0.1.0", "minimum_client_version": minimum}
