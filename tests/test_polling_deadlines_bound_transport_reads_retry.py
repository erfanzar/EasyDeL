# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Polling deadlines bound transport reads, retry attempts and observers."""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from eyvan.records import CommandExecution, WorkloadDefinition, WorkloadRecord, WorkloadState
from eyvan.sdk.client.async_client import AsyncEyvanClient
from eyvan.sdk.client.sync import EyvanClient


def _terminal_response() -> httpx.Response:
    record = WorkloadRecord(
        id="wl-1",
        org_id="acme",
        project_id="research",
        spec=WorkloadDefinition(name="deadline", execution=CommandExecution(command="true")),
        state=WorkloadState.SUCCEEDED,
        created_by="alice",
    )
    return httpx.Response(200, json=record.model_dump(mode="json"))


@pytest.mark.parametrize("strict_read", [None, 0.001])
def test_sync_poll_bounds_each_httpx_timeout_without_relaxing_client(strict_read: float | None) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return _terminal_response()

    with httpx.Client(
        base_url="http://eyvan.test",
        timeout=httpx.Timeout(None, read=strict_read),
        transport=httpx.MockTransport(handler),
    ) as http_client:
        client = EyvanClient("http://eyvan.test", client=http_client)
        client.wait_workload("wl-1", timeout=1)
        client.get_workload("wl-1")
    assert all(value is not None and 0 < value <= 1 for value in seen[0].values())
    if strict_read is not None:
        assert seen[0]["read"] == strict_read
    assert seen[1] == {"connect": None, "read": strict_read, "write": None, "pool": None}


@pytest.mark.asyncio
@pytest.mark.parametrize("strict_read", [None, 0.001])
async def test_async_poll_bounds_each_httpx_timeout_without_relaxing_client(strict_read: float | None) -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return _terminal_response()

    async with httpx.AsyncClient(
        base_url="http://eyvan.test",
        timeout=httpx.Timeout(None, read=strict_read),
        transport=httpx.MockTransport(handler),
    ) as http_client:
        client = AsyncEyvanClient("http://eyvan.test", client=http_client)
        await client.wait_workload("wl-1", timeout=1)
        await client.get_workload("wl-1")
    assert all(value is not None and 0 < value <= 1 for value in seen[0].values())
    if strict_read is not None:
        assert seen[0]["read"] == strict_read
    assert seen[1] == {"connect": None, "read": strict_read, "write": None, "pool": None}


def test_sync_expired_retry_does_not_issue_another_rpc() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(503, headers={"Retry-After": "5"}) if len(requests) == 1 else _terminal_response()

    with httpx.Client(base_url="http://eyvan.test", transport=httpx.MockTransport(handler)) as http_client:
        client = EyvanClient("http://eyvan.test", client=http_client)
        with pytest.raises(TimeoutError, match="wl-1"):
            client.wait_workload("wl-1", timeout=0.01)
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_async_expired_retry_does_not_issue_another_rpc() -> None:
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(503, headers={"Retry-After": "5"}) if len(requests) == 1 else _terminal_response()

    async with httpx.AsyncClient(base_url="http://eyvan.test", transport=httpx.MockTransport(handler)) as http_client:
        client = AsyncEyvanClient("http://eyvan.test", client=http_client)
        with pytest.raises(TimeoutError, match="wl-1"):
            await client.wait_workload("wl-1", timeout=0.01)
    assert len(requests) == 1


def test_sync_observer_cannot_turn_a_late_observation_into_success() -> None:
    with httpx.Client(
        base_url="http://eyvan.test", transport=httpx.MockTransport(lambda request: _terminal_response())
    ) as http_client:
        client = EyvanClient("http://eyvan.test", client=http_client)
        with pytest.raises(TimeoutError, match="wl-1"):
            client.wait_workload("wl-1", timeout=0.01, on_update=lambda record: time.sleep(0.03))
        assert client.get_workload("wl-1").state is WorkloadState.SUCCEEDED


@pytest.mark.asyncio
async def test_async_observer_is_cancelled_and_deadline_context_is_restored() -> None:
    cancelled = []

    async def observer(record: WorkloadRecord) -> None:
        try:
            await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            cancelled.append(record.id)
            raise

    async with httpx.AsyncClient(
        base_url="http://eyvan.test", transport=httpx.MockTransport(lambda request: _terminal_response())
    ) as http_client:
        client = AsyncEyvanClient("http://eyvan.test", client=http_client)
        with pytest.raises(TimeoutError, match="wl-1"):
            await client.wait_workload("wl-1", timeout=0.01, on_update=observer)
        assert (await client.get_workload("wl-1")).state is WorkloadState.SUCCEEDED
    assert cancelled == ["wl-1"]


def test_sync_observer_reads_share_the_polling_budget() -> None:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.extensions["timeout"])
        return _terminal_response()

    with httpx.Client(base_url="http://eyvan.test", transport=httpx.MockTransport(handler)) as http_client:
        client = EyvanClient("http://eyvan.test", client=http_client)
        client.wait_workload("wl-1", timeout=1, on_update=lambda record: client.get_workload("wl-1"))
    assert len(seen) == 2
    assert all(0 < value <= 1 for timeouts in seen for value in timeouts.values())


@pytest.mark.asyncio
async def test_async_concurrent_wait_does_not_leak_deadline_to_another_task() -> None:
    entered = asyncio.Event()
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen[request.url.path] = request.extensions["timeout"]
        if request.url.path.endswith("wl-1"):
            entered.set()
            await asyncio.sleep(0.05)
        return _terminal_response()

    async with httpx.AsyncClient(base_url="http://eyvan.test", transport=httpx.MockTransport(handler)) as http_client:
        client = AsyncEyvanClient("http://eyvan.test", client=http_client)
        waiter = asyncio.create_task(client.wait_workload("wl-1", timeout=0.01))
        await entered.wait()
        assert (await client.get_workload("other")).state is WorkloadState.SUCCEEDED
        with pytest.raises(TimeoutError):
            await waiter
    assert all(0 < value <= 0.01 for value in seen["/api/v1/workloads/wl-1"].values())
    assert all(value == 5 for value in seen["/api/v1/workloads/other"].values())


@pytest.mark.asyncio
async def test_unbounded_async_wait_preserves_an_observers_own_timeout_error() -> None:
    error = TimeoutError("observer-owned timeout")

    async def observer(record: WorkloadRecord) -> None:
        raise error

    async with httpx.AsyncClient(
        base_url="http://eyvan.test", transport=httpx.MockTransport(lambda request: _terminal_response())
    ) as http_client:
        client = AsyncEyvanClient("http://eyvan.test", client=http_client)
        with pytest.raises(TimeoutError) as caught:
            await client.wait_workload("wl-1", on_update=observer)
    assert caught.value is error
