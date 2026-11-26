# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Owned controller HTTP pools remain bounded without interrupting their users."""

from __future__ import annotations

import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from eyvan.config import EyvanSettings
from eyvan.records import CellRecord
from eyvan.server.controller import lifecycle as lifecycle_module
from eyvan.server.controller.lifecycle import _ControllerLifecycle
from eyvan.server.persistence import InMemoryRepository
from eyvan.server.service import ControlPlaneService
from eyvan.worker.agent.client import CellAgentClient


def _controller(monkeypatch: pytest.MonkeyPatch, *, agent_factory=None) -> _ControllerLifecycle:
    # Exercise the same policy at a small bound, using real HTTP clients without
    # outbound requests. The unrepaired implementation ignores this bound.
    monkeypatch.setattr(lifecycle_module, "_AGENT_CLIENT_CACHE_SIZE", 4, raising=False)
    repository = InMemoryRepository()
    settings = EyvanSettings()
    return _ControllerLifecycle(
        repository, ControlPlaneService(repository), settings=settings, agent_factory=agent_factory
    )


def _cell(index: int) -> CellRecord:
    return CellRecord.model_construct(id=f"cell-{index}", endpoint=f"http://127.0.0.1:{18000 + index}")


@pytest.mark.asyncio
async def test_historical_endpoint_churn_releases_real_owned_transports(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = _controller(monkeypatch)
    transports = []
    agents = []
    try:
        for index in range(12):
            agent = controller._agent_factory(_cell(index))
            assert controller._agent_factory(_cell(index)) is agent
            transports.append(agent.client)
            agents.append(weakref.ref(agent))
        del agent
        gc.collect()

        assert sum(not transport.is_closed for transport in transports) <= 4
        assert all(transport.is_closed for transport in transports[:-4])
        assert all(reference() is None for reference in agents[:-4])
        assert len(controller._agent_clients) <= 4
    finally:
        await controller.close()
    assert all(transport.is_closed for transport in transports)


@pytest.mark.asyncio
async def test_evicted_endpoint_is_reused_and_not_closed_while_referenced(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = _controller(monkeypatch)
    first = controller._agent_factory(_cell(0))
    transport = first.client
    try:
        for index in range(1, 8):
            controller._agent_factory(_cell(index))
        assert not transport.is_closed
        assert controller._agent_factory(_cell(0)) is first
        # Evict again; only a caller still owns the original agent now.
        for index in range(8, 16):
            controller._agent_factory(_cell(index))
        del first
        gc.collect()
        assert transport.is_closed
    finally:
        await controller.close()


@pytest.mark.asyncio
async def test_cache_churn_does_not_close_an_inflight_http_request(monkeypatch: pytest.MonkeyPatch) -> None:
    entered = threading.Event()
    finish = threading.Event()
    transports = []

    def respond(request: httpx.Request) -> httpx.Response:
        entered.set()
        assert finish.wait(10), "test did not release the in-flight request"
        assert not transports[0].is_closed
        return httpx.Response(404, request=request)

    def make_agent(endpoint: str, *, token=None, mtls=None) -> CellAgentClient:
        transport = httpx.Client(base_url=endpoint, transport=httpx.MockTransport(respond))
        transports.append(transport)
        return CellAgentClient(endpoint, token=token, client=transport, mtls=mtls)

    monkeypatch.setattr(lifecycle_module, "CellAgentClient", make_agent)
    controller = _controller(monkeypatch)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(controller._agent_factory(_cell(0)).build)
            try:
                assert entered.wait(10), "HTTP request did not begin"
                for index in range(1, 12):
                    controller._agent_factory(_cell(index))
                gc.collect()
                assert not transports[0].is_closed
            finally:
                finish.set()
            assert future.result(timeout=10) is None
        gc.collect()
        assert transports[0].is_closed
    finally:
        await controller.close()


@pytest.mark.asyncio
async def test_recent_endpoint_reuse_refreshes_idle_retention(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = _controller(monkeypatch)
    try:
        transports = [controller._agent_factory(_cell(index)).client for index in range(4)]
        assert controller._agent_factory(_cell(0)).client is transports[0]
        controller._agent_factory(_cell(4))
        gc.collect()
        assert not transports[0].is_closed
        assert transports[1].is_closed
        assert all(not transport.is_closed for transport in transports[2:])
    finally:
        await controller.close()


@pytest.mark.asyncio
async def test_shutdown_closes_caller_held_evicted_owned_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    controller = _controller(monkeypatch)
    first = controller._agent_factory(_cell(0))
    try:
        for index in range(1, 12):
            controller._agent_factory(_cell(index))
        assert not first.client.is_closed
        await controller.close()
        assert first.client.is_closed
        assert not controller._agent_clients
    finally:
        await controller.close()


@pytest.mark.asyncio
async def test_injected_clients_are_not_owned_or_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = CellAgentClient("http://127.0.0.1:18000")
    controller = _controller(monkeypatch, agent_factory=lambda cell: agent)
    try:
        for index in range(12):
            assert controller._agent_factory(_cell(index)) is agent
        await controller.close()
        assert not agent.client.is_closed
    finally:
        agent.close()
