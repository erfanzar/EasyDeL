# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A read answered with a retryable 503 is asked again, not reported as failure.

Read retries must honor retryable statuses and ``Retry-After`` so a transient
PostgreSQL conflict does not terminate workload polling.
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.sdk.client.retry import RetryPolicy
from eyvan.sdk.client.transport import AsyncApiTransport, EyvanHTTPError, SyncApiTransport

URL = "https://10.0.0.7:8080"


def _conflict() -> httpx.Response:
    return httpx.Response(
        503, headers={"Retry-After": "0"}, json={"detail": "retryable PostgreSQL transaction conflict (40001)"}
    )


def _answers(*responses: httpx.Response) -> tuple[httpx.MockTransport, list[str]]:
    queue = list(responses)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.method)
        return queue.pop(0)

    return httpx.MockTransport(handler), seen


def test_a_read_is_retried_through_a_transient_conflict() -> None:
    mock, seen = _answers(_conflict(), httpx.Response(200, json={"state": "running"}))
    transport = SyncApiTransport(ControlPlaneEndpoint.parse(URL), client=httpx.Client(base_url=URL, transport=mock))

    response = transport.request_checked("GET", "/api/v1/workloads/wl-1")

    assert response.json() == {"state": "running"}
    assert seen == ["GET", "GET"]


def test_a_read_that_keeps_conflicting_still_fails_after_the_policy_is_spent() -> None:
    mock, seen = _answers(_conflict(), _conflict(), _conflict())
    transport = SyncApiTransport(
        ControlPlaneEndpoint.parse(URL),
        client=httpx.Client(base_url=URL, transport=mock),
        retry_policy=RetryPolicy(max_attempts=3),
    )

    with pytest.raises(EyvanHTTPError):
        transport.request_checked("GET", "/api/v1/workloads/wl-1")
    assert seen == ["GET", "GET", "GET"]


def test_a_non_idempotent_request_is_not_retried_by_the_read_path() -> None:
    """Only `mutate` may retry a write, because only it carries an idempotency key."""

    mock, seen = _answers(_conflict())
    transport = SyncApiTransport(ControlPlaneEndpoint.parse(URL), client=httpx.Client(base_url=URL, transport=mock))

    with pytest.raises(EyvanHTTPError):
        transport.request_checked("POST", "/api/v1/workloads")
    assert seen == ["POST"]


def test_the_async_read_path_retries_too() -> None:
    mock, seen = _answers(_conflict(), httpx.Response(200, json={"state": "running"}))

    async def read() -> httpx.Response:
        transport = AsyncApiTransport(
            ControlPlaneEndpoint.parse(URL), client=httpx.AsyncClient(base_url=URL, transport=mock)
        )
        return await transport.request_checked("GET", "/api/v1/workloads/wl-1")

    assert asyncio.run(read()).json() == {"state": "running"}
    assert seen == ["GET", "GET"]
