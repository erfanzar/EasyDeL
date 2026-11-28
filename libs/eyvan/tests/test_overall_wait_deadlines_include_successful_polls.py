# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Overall wait deadlines include successful polls and status-retry delays."""

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


@pytest.mark.parametrize("retry_after", [False, True], ids=["slow-success", "retry-after"])
def test_sync_wait_does_not_accept_a_terminal_response_after_deadline(retry_after: bool) -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        assert request.url.path == "/api/v1/workloads/wl-1"
        requests += 1
        if retry_after and requests == 1:
            return httpx.Response(503, headers={"Retry-After": "0.03"})
        if not retry_after:
            time.sleep(0.03)
        return _terminal_response()

    with httpx.Client(base_url="http://eyvan.test", transport=httpx.MockTransport(handler)) as http_client:
        client = EyvanClient("http://eyvan.test", client=http_client)
        with pytest.raises(TimeoutError, match="wl-1"):
            client.wait_workload("wl-1", timeout=0.001)


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_after", [False, True], ids=["slow-success", "retry-after"])
async def test_async_wait_does_not_accept_a_terminal_response_after_deadline(retry_after: bool) -> None:
    requests = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        assert request.url.path == "/api/v1/workloads/wl-1"
        requests += 1
        if retry_after and requests == 1:
            return httpx.Response(503, headers={"Retry-After": "0.03"})
        if not retry_after:
            await asyncio.sleep(0.03)
        return _terminal_response()

    async with httpx.AsyncClient(base_url="http://eyvan.test", transport=httpx.MockTransport(handler)) as http_client:
        client = AsyncEyvanClient("http://eyvan.test", client=http_client)
        with pytest.raises(TimeoutError, match="wl-1"):
            await client.wait_workload("wl-1", timeout=0.001)


@pytest.mark.parametrize("timeout", [None, 0, 1])
def test_sync_immediate_terminal_observation_retains_existing_timeout_semantics(timeout: float | None) -> None:
    with httpx.Client(
        base_url="http://eyvan.test", transport=httpx.MockTransport(lambda request: _terminal_response())
    ) as http_client:
        result = EyvanClient("http://eyvan.test", client=http_client).wait_workload("wl-1", timeout=timeout)
    assert result.id == "wl-1"
    assert result.state is WorkloadState.SUCCEEDED


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout", [None, 0, 1])
async def test_async_immediate_terminal_observation_retains_existing_timeout_semantics(timeout: float | None) -> None:
    async with httpx.AsyncClient(
        base_url="http://eyvan.test", transport=httpx.MockTransport(lambda request: _terminal_response())
    ) as http_client:
        result = await AsyncEyvanClient("http://eyvan.test", client=http_client).wait_workload("wl-1", timeout=timeout)
    assert result.id == "wl-1"
    assert result.state is WorkloadState.SUCCEEDED
