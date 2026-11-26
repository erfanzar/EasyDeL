# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Observe the real HTTP transport without permitting any network contact."""

from __future__ import annotations

import socket
import ssl

import httpcore
import pytest

from eyvan.records import ServiceProbeOutcome
from eyvan.server.api.schemas import ServiceEndpointRequest
from eyvan.server.controller.services import HttpServiceHealthProbe


@pytest.fixture(autouse=True)
def no_connections(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, ...]]:
    """Intercept every socket connect, including accidental proxy connections."""
    connections: list[tuple[object, ...]] = []

    def blocked(_socket: socket.socket, address: tuple[object, ...]) -> None:
        connections.append(address)
        raise OSError("test intercepted socket connection")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    return connections


def dns_answer(address: str, port: int | None) -> list[tuple[object, ...]]:
    family = socket.AF_INET6 if ":" in address else socket.AF_INET
    sockaddr = (address, port or 0, 0, 0) if family == socket.AF_INET6 else (address, port or 0)
    return [(family, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", sockaddr)]


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "169.254.169.254",
        "::1",
        "fe80::1",
        "fd00::1",
        "::ffff:127.0.0.1",
        "::ffff:169.254.169.254",
        "0.0.0.0",
        "::",
        "::ffff:0.0.0.0",
    ],
)
async def test_registration_dns_cannot_authorize_a_later_forbidden_connection(
    monkeypatch: pytest.MonkeyPatch, no_connections: list[tuple[object, ...]], address: str
) -> None:
    lookups: list[str] = []

    def resolve(host: str, port: int | None, *args: object, **kwargs: object) -> list[tuple[object, ...]]:
        lookups.append(host)
        return dns_answer("10.0.0.1" if len(lookups) == 1 else address, port)

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    endpoint = ServiceEndpointRequest(endpoint="http://replica.test:8080").endpoint
    result = await HttpServiceHealthProbe()(endpoint=endpoint, path="/healthz", timeout_s=1)

    assert no_connections == [], f"forbidden destination reached socket.connect: {no_connections}"
    assert result.outcome is ServiceProbeOutcome.UNKNOWN
    expected = (
        "unspecified" if address in {"0.0.0.0", "::", "::ffff:0.0.0.0"} else "loopback, link-local, or unique-local"
    )
    assert expected in result.detail
    assert lookups == ["replica.test", "replica.test"]


@pytest.mark.parametrize("address", ["0.0.0.0", "::", "::ffff:0.0.0.0"])
def test_registration_refuses_literal_and_dns_unspecified_addresses(monkeypatch, address: str) -> None:
    host = f"[{address}]" if ":" in address else address
    with pytest.raises(ValueError, match="service endpoint must not resolve"):
        ServiceEndpointRequest(endpoint=f"http://{host}:8080")
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, *args, **kwargs: dns_answer(address, port))
    with pytest.raises(ValueError, match="service endpoint must not resolve"):
        ServiceEndpointRequest(endpoint="http://replica.test:8080")


@pytest.mark.parametrize("address", ["10.0.0.1", "192.168.1.10", "::ffff:10.0.0.1", "::ffff:192.168.1.10"])
async def test_probe_pins_the_checked_address_without_another_dns_lookup(
    monkeypatch: pytest.MonkeyPatch, no_connections: list[tuple[object, ...]], address: str
) -> None:
    lookups: list[str] = []

    def resolve(host: str, port: int | None, *args: object, **kwargs: object) -> list[tuple[object, ...]]:
        lookups.append(host)
        return dns_answer(address if len(lookups) == 1 else "::ffff:127.0.0.1", port)

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    result = await HttpServiceHealthProbe()(endpoint="http://replica.test:8080", path="/healthz", timeout_s=1)

    assert no_connections
    expected = address.removeprefix("::ffff:")
    assert all(connection[0] == expected for connection in no_connections)
    assert lookups == ["replica.test"]
    assert result.outcome is ServiceProbeOutcome.UNKNOWN


class ReplyStream(httpcore.AsyncNetworkStream):
    """A controlled peer below HTTPX/httpcore, with observable HTTP and TLS bytes."""

    def __init__(self, *, redirect: bool = False, certificate_error: bool = False) -> None:
        self.writes = bytearray()
        self.tls: list[tuple[ssl.SSLContext, str | None]] = []
        self.redirect = redirect
        self.certificate_error = certificate_error

    async def read(self, max_bytes: int, timeout: float | None = None) -> bytes:
        if self.redirect:
            return b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1/secret\r\nContent-Length: 0\r\n\r\n"
        return b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}"

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        self.writes.extend(buffer)

    async def aclose(self) -> None:
        pass

    async def start_tls(
        self, ssl_context: ssl.SSLContext, server_hostname: str | None = None, timeout: float | None = None
    ) -> httpcore.AsyncNetworkStream:
        self.tls.append((ssl_context, server_hostname))
        if self.certificate_error:
            raise ssl.SSLCertVerificationError("test certificate mismatch")
        return self


@pytest.mark.parametrize("scheme", ["http", "https"])
@pytest.mark.parametrize("certificate_error", [False, True])
async def test_pinning_keeps_original_host_sni_and_certificate_checks(
    monkeypatch: pytest.MonkeyPatch,
    no_connections: list[tuple[object, ...]],
    scheme: str,
    certificate_error: bool,
) -> None:
    from httpcore._backends.anyio import AnyIOBackend

    stream = ReplyStream(certificate_error=certificate_error)
    targets: list[tuple[str, int]] = []
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, *args, **kwargs: dns_answer("10.0.0.1", port))

    async def connect(_self: object, host: str, port: int, **kwargs: object) -> ReplyStream:
        targets.append((host, port))
        return stream

    monkeypatch.setattr(AnyIOBackend, "connect_tcp", connect)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9000")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9000")
    monkeypatch.setenv("ALL_PROXY", "http://127.0.0.1:9000")
    monkeypatch.setenv("NO_PROXY", "")
    result = await HttpServiceHealthProbe()(endpoint=f"{scheme}://replica.test:8443/base", path="/healthz", timeout_s=1)

    assert targets == [("10.0.0.1", 8443)]
    assert no_connections == []
    if scheme == "https":
        [(context, hostname)] = stream.tls
        assert hostname == "replica.test"
        assert context.check_hostname
        assert context.verify_mode == ssl.CERT_REQUIRED
    else:
        assert stream.tls == []
    if scheme == "https" and certificate_error:
        assert result.outcome is ServiceProbeOutcome.UNKNOWN
        assert not stream.writes
    else:
        assert result.outcome is ServiceProbeOutcome.SUCCESS
        assert b"GET /base/healthz HTTP/1.1\r\n" in stream.writes
        assert b"Host: replica.test:8443\r\n" in stream.writes


async def test_a_redirect_cannot_escape_the_pinned_transport(
    monkeypatch: pytest.MonkeyPatch, no_connections: list[tuple[object, ...]]
) -> None:
    from httpcore._backends.anyio import AnyIOBackend

    stream = ReplyStream(redirect=True)
    targets: list[str] = []
    monkeypatch.setattr(socket, "getaddrinfo", lambda host, port, *args, **kwargs: dns_answer("10.0.0.1", port))

    async def connect(_self: object, host: str, port: int, **kwargs: object) -> ReplyStream:
        targets.append(host)
        return stream

    monkeypatch.setattr(AnyIOBackend, "connect_tcp", connect)
    result = await HttpServiceHealthProbe()(endpoint="http://replica.test", path="/healthz", timeout_s=1)

    assert targets == ["10.0.0.1"]
    assert no_connections == []
    assert result.outcome is ServiceProbeOutcome.FAILURE
    assert "302" in result.detail


async def test_all_dns_answers_are_checked_before_any_connection(
    monkeypatch: pytest.MonkeyPatch, no_connections: list[tuple[object, ...]]
) -> None:
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda host, port, *args, **kwargs: dns_answer("10.0.0.1", port) + dns_answer("::ffff:127.0.0.1", port),
    )
    result = await HttpServiceHealthProbe()(endpoint="http://replica.test", path="/healthz", timeout_s=1)

    assert no_connections == []
    assert result.outcome is ServiceProbeOutcome.UNKNOWN
    assert "loopback, link-local, or unique-local" in result.detail


async def test_connection_fallback_uses_only_the_original_checked_answers(
    monkeypatch: pytest.MonkeyPatch, no_connections: list[tuple[object, ...]]
) -> None:
    from httpcore._backends.anyio import AnyIOBackend

    lookups: list[str] = []
    targets: list[str] = []
    stream = ReplyStream()

    def resolve(host: str, port: int | None, *args: object, **kwargs: object) -> list[tuple[object, ...]]:
        lookups.append(host)
        return dns_answer("10.0.0.1", port) + dns_answer("192.168.1.10", port)

    async def connect(_self: object, host: str, port: int, **kwargs: object) -> ReplyStream:
        targets.append(host)
        if host == "10.0.0.1":
            raise httpcore.ConnectError("first checked address unreachable")
        return stream

    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(AnyIOBackend, "connect_tcp", connect)
    result = await HttpServiceHealthProbe()(endpoint="http://replica.test", path="/healthz", timeout_s=1)

    assert targets == ["10.0.0.1", "192.168.1.10"]
    assert lookups == ["replica.test"]
    assert no_connections == []
    assert result.outcome is ServiceProbeOutcome.SUCCESS
