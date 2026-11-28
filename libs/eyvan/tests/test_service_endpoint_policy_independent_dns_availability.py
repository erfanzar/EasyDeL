# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Service endpoint policy regressions independent of DNS availability."""

from __future__ import annotations

import asyncio
import socket

import httpx
import pytest
from pydantic import ValidationError

from eyvan.records import ServiceProbeOutcome
from eyvan.server.api.schemas import ServiceEndpointRequest
from eyvan.server.controller.services import HttpServiceHealthProbe


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://127.0.0.1",
        "http://169.254.169.254",
        "http://[::1]",
        "http://[fe80::1]",
        "http://[fd00::1]",
        "http://[::ffff:7f00:1]",
        "http://[::ffff:a9fe:a9fe]",
    ],
)
def test_dns_failure_cannot_hide_a_forbidden_literal(monkeypatch: pytest.MonkeyPatch, endpoint: str) -> None:
    def unavailable(*args: object, **kwargs: object) -> None:
        raise socket.gaierror("test resolver unavailable")

    monkeypatch.setattr(socket, "getaddrinfo", unavailable)
    with pytest.raises(ValidationError, match="loopback, link-local, or unique-local"):
        ServiceEndpointRequest(endpoint=endpoint)


@pytest.mark.parametrize(
    ("address", "allowed"),
    [
        ("::ffff:127.0.0.1", False),
        ("::ffff:169.254.169.254", False),
        ("::ffff:10.0.0.1", True),
        ("::ffff:192.168.1.10", True),
    ],
)
def test_registration_applies_ipv4_policy_to_mapped_dns_answers(
    monkeypatch: pytest.MonkeyPatch, address: str, allowed: bool
) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, 0, 0, 0))],
    )
    if allowed:
        assert ServiceEndpointRequest(endpoint="http://replica.test").endpoint == "http://replica.test"
    else:
        with pytest.raises(ValidationError, match="loopback, link-local, or unique-local"):
            ServiceEndpointRequest(endpoint="http://replica.test")


@pytest.mark.parametrize("host", ["::ffff:127.0.0.1", "::ffff:169.254.169.254", "::1", "fe80::1"])
async def test_actual_httpx_literal_is_refused_even_with_an_injected_transport(
    monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    requests: list[str] = []
    original = httpx.AsyncClient

    def answer(request: httpx.Request) -> httpx.Response:
        requests.append(str(request.url))
        return httpx.Response(200)

    def transported(*args: object, **kwargs: object) -> httpx.AsyncClient:
        kwargs["transport"] = httpx.MockTransport(answer)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(httpx, "AsyncClient", transported)
    result = await HttpServiceHealthProbe()(endpoint=f"http://[{host}]", path="/healthz", timeout_s=1)

    assert requests == []
    assert result.outcome is ServiceProbeOutcome.UNKNOWN
    assert "loopback, link-local, or unique-local" in result.detail


async def test_dns_is_inside_the_probe_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    async def unavailable(*args: object, **kwargs: object) -> None:
        await asyncio.sleep(60)

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", unavailable)
    result = await asyncio.wait_for(
        HttpServiceHealthProbe()(endpoint="http://replica.test", path="/healthz", timeout_s=0.01), timeout=1
    )

    assert result.outcome is ServiceProbeOutcome.UNKNOWN
    assert "Timeout" in result.detail


async def test_cancelling_a_probe_does_not_turn_cancellation_into_a_verdict(monkeypatch: pytest.MonkeyPatch) -> None:
    started = asyncio.Event()

    async def unavailable(*args: object, **kwargs: object) -> None:
        started.set()
        await asyncio.sleep(60)

    monkeypatch.setattr(asyncio.get_running_loop(), "getaddrinfo", unavailable)
    task = asyncio.create_task(HttpServiceHealthProbe()(endpoint="http://replica.test", path="/healthz", timeout_s=1))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
