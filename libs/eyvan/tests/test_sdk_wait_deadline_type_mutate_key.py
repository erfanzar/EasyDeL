# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""SDK contracts: wait deadline type, mutate key visibility and the follower cursor."""

from __future__ import annotations

import httpx
import pytest

from eyvan.sdk.client.async_client import AsyncEyvanClient
from eyvan.sdk.client.sync import EyvanClient
from eyvan.sdk.client.transport import EyvanUnavailableError, SyncApiTransport
from eyvan.sdk.follow import WorkloadFollower


def _down_client() -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("plane down", request=request)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_wait_deadline_mid_outage_raises_timeout_error() -> None:
    """A deadline that expires during an outage is a timeout, not an outage."""
    client = EyvanClient("http://127.0.0.1:8080", client=_down_client())

    with pytest.raises(TimeoutError):
        client.wait_workload("wl-1", timeout=0.2, poll_interval=0.05)


def test_zero_deadline_outage_does_not_claim_the_configured_grace_elapsed() -> None:
    with _down_client() as http_client:
        client = EyvanClient("http://127.0.0.1:8080", client=http_client)
        with pytest.raises(TimeoutError) as excinfo:
            client.wait_workload("wl-1", timeout=0)

    assert isinstance(excinfo.value.__cause__, EyvanUnavailableError)
    assert "unreachable" in str(excinfo.value)
    assert "unreachable for" not in str(excinfo.value)


@pytest.mark.asyncio
async def test_async_zero_deadline_outage_does_not_claim_the_configured_grace_elapsed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("plane down", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http_client:
        client = AsyncEyvanClient("http://127.0.0.1:8080", client=http_client)
        with pytest.raises(TimeoutError) as excinfo:
            await client.wait_workload("wl-1", timeout=0)

    assert isinstance(excinfo.value.__cause__, EyvanUnavailableError)
    assert "unreachable" in str(excinfo.value)
    assert "unreachable for" not in str(excinfo.value)


def test_mutate_attaches_the_idempotency_key_to_the_final_error() -> None:
    """After an ambiguous admission the caller must be able to see the key."""
    transport = SyncApiTransport("http://127.0.0.1:8080", client=_down_client())
    try:
        with pytest.raises(EyvanUnavailableError) as excinfo:
            transport.mutate("POST", "/api/v1/example", idempotency_key="key-123")
        assert excinfo.value.idempotency_key == "key-123"
    finally:
        transport.close()


def test_mutate_honors_a_caller_supplied_idempotency_header() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers["Idempotency-Key"])
        return httpx.Response(200, json={})

    transport = SyncApiTransport(
        "http://127.0.0.1:8080",
        client=httpx.Client(base_url="http://127.0.0.1:8080", transport=httpx.MockTransport(handler)),
    )
    try:
        transport.mutate("POST", "/api/v1/example", headers={"Idempotency-Key": "caller-key"})
    finally:
        transport.close()
    assert seen == ["caller-key"]


def test_follower_resets_the_cursor_on_attempt_change() -> None:
    """A new attempt with an empty next cursor must restart the log read."""

    class _FakeClient:
        def __init__(self) -> None:
            self.seen_cursors: list[str] = []

        def request(self, method: str, path: str, params: dict | None = None):
            self.seen_cursors.append((params or {}).get("after", ""))
            return {"attempt_changed": True, "lines": [], "next_cursor": ""}

    fake = _FakeClient()
    follower = WorkloadFollower(client=fake, poll_interval=0.0)  # type: ignore[arg-type]
    follower._cursor = "old-attempt-cursor"

    follower._drain("wl-1")

    assert fake.seen_cursors == ["old-attempt-cursor"]
    assert follower._cursor == "", "the previous attempt's cursor must not address the new attempt"
