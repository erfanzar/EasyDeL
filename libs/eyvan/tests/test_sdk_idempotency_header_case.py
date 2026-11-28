# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Case-insensitive SDK mutation keys as observed by HTTPX and the local API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import MappingProxyType

import httpx
import pytest
from fastapi.testclient import TestClient

from eyvan.config import EyvanSettings
from eyvan.sdk.client.identity import ClientIdentity
from eyvan.sdk.client.retry import RetryPolicy
from eyvan.sdk.client.transport import AsyncApiTransport, EyvanUnavailableError, SyncApiTransport
from eyvan.server.api.app import create_app

_RETRY = RetryPolicy(max_attempts=2, fallback_backoff_s=0)


@asynccontextmanager
async def _mock_transport(mode, handler, *, client_headers=None, transport_headers=None):
    client_type = httpx.Client if mode == "sync" else httpx.AsyncClient
    transport_type = SyncApiTransport if mode == "sync" else AsyncApiTransport
    raw = client_type(base_url="http://test", headers=client_headers, transport=httpx.MockTransport(handler))
    transport = transport_type("http://test", client=raw, headers=transport_headers, retry_policy=_RETRY)
    try:
        yield transport, raw
    finally:
        if mode == "sync":
            raw.close()
        else:
            await raw.aclose()


async def _mutate(transport, path="/api/v1/example", **kwargs):
    if isinstance(transport, SyncApiTransport):
        return transport.mutate("POST", path, **kwargs)
    return await transport.mutate("POST", path, **kwargs)


async def _checked(transport, path, **kwargs):
    if isinstance(transport, SyncApiTransport):
        return transport.request_checked("GET", path, **kwargs)
    return await transport.request_checked("GET", path, **kwargs)


def _wire_keys(request):
    return [(name.decode(), value.decode()) for name, value in request.headers.raw if name.lower() == b"idempotency-key"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "supplied",
    [
        {"Idempotency-Key": "caller-key"},
        {"idempotency-key": "caller-key"},
        {"iDeMpOtEnCy-KeY": "caller-key"},
        {"Idempotency-Key": "caller-key", "idempotency-key": "caller-key"},
        httpx.Headers({"Idempotency-Key": "caller-key", "idempotency-key": "caller-key"}),
    ],
)
async def test_mutation_collapses_case_variants_and_overrides_defaults(mode, supplied):
    seen = []

    def handler(request):
        seen.append(_wire_keys(request))
        assert request.headers["X-Caller"] == "preserved"
        return httpx.Response(503 if len(seen) == 1 else 200, headers={"Retry-After": "0"}, json={})

    if isinstance(supplied, httpx.Headers):
        caller = supplied.copy()
        caller["X-Caller"] = "preserved"
    else:
        caller = MappingProxyType({**supplied, "X-Caller": "preserved"})
    original = tuple(caller.items())
    async with _mock_transport(
        mode,
        handler,
        client_headers={"idempotency-key": "client-default"},
        transport_headers={"IDEMPOTENCY-KEY": "transport-default"},
    ) as (transport, raw):
        response = await _mutate(transport, headers=caller, idempotency_key="argument-key")
        assert response.status_code == 200
        assert raw.headers.get_list("Idempotency-Key") == ["client-default"]
    assert seen == [[("Idempotency-Key", "caller-key")]] * 2
    assert tuple(caller.items()) == original


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("header_name", ["idempotency-key", "iDeMpOtEnCy-KeY", None])
async def test_lost_response_recovery_key_matches_every_outgoing_attempt(mode, header_name):
    seen = []

    def handler(request):
        seen.append(_wire_keys(request))
        raise httpx.ReadError("response lost after admission", request=request)

    supplied = {header_name: "admitted-key"} if header_name else {}
    async with _mock_transport(
        mode,
        handler,
        client_headers={"idempotency-key": "client-default"},
        transport_headers={"IDEMPOTENCY-KEY": "transport-default"},
    ) as (transport, _):
        with pytest.raises(EyvanUnavailableError) as caught:
            await _mutate(transport, headers=MappingProxyType(supplied))
    key = caught.value.idempotency_key
    assert isinstance(key, str)
    assert key == "admitted-key" if header_name else key.startswith("sdk-")
    assert seen == [[("Idempotency-Key", key)]] * 2
    assert supplied == ({header_name: "admitted-key"} if header_name else {})


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize(
    "supplied",
    [
        {"Idempotency-Key": "one", "idempotency-key": "two"},
        {"idempotency-key": "two", "Idempotency-Key": "one"},
        {"iDeMpOtEnCy-KeY": "one", "idempotency-key": "two"},
        httpx.Headers({"Idempotency-Key": "one", "idempotency-key": "two"}),
    ],
)
async def test_conflicting_caller_keys_are_rejected_before_io(mode, supplied):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={})

    async with _mock_transport(mode, handler) as (transport, _):
        with pytest.raises(ValueError, match=r"conflicting.*[Ii]dempotency"):
            caller = supplied if isinstance(supplied, httpx.Headers) else MappingProxyType(supplied)
            await _mutate(transport, headers=caller, idempotency_key="argument-key")
    assert seen == []


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("header_name", ["Idempotency-Key", "idempotency-key", "iDeMpOtEnCy-KeY", None])
async def test_empty_supplied_keys_use_existing_factory_validation_before_io(mode, header_name):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={})

    async with _mock_transport(mode, handler) as (transport, _):
        with pytest.raises(ValueError, match="idempotency_key must be a non-empty string"):
            await _mutate(
                transport,
                headers={header_name: ""} if header_name else {},
                idempotency_key="fallback-key" if header_name else "",
            )
    assert seen == []


@asynccontextmanager
async def _local_api(mode, *, lose_responses=False):
    app = create_app(settings=EyvanSettings(dev_auth_enabled=True))
    seen = []

    @app.middleware("http")
    async def capture_headers(request, call_next):
        if request.method == "POST" and request.url.path == "/api/v1/workloads":
            seen.append(
                [value.decode() for name, value in request.scope["headers"] if name.lower() == b"idempotency-key"]
            )
        return await call_next(request)

    identity = ClientIdentity(dev_user="alice", dev_org="acme", dev_project="research", dev_roles=("org_admin",))
    if mode == "sync":
        api = TestClient(app)

        def handler(request):
            response = api.send(request)
            if lose_responses and request.method == "POST" and len(seen) <= _RETRY.max_attempts:
                raise httpx.ReadError("admitted response lost", request=request)
            return response
    else:
        api = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")

        async def handler(request):
            response = await api.send(request)
            if lose_responses and request.method == "POST" and len(seen) <= _RETRY.max_attempts:
                raise httpx.ReadError("admitted response lost", request=request)
            return response

    try:
        async with _mock_transport(mode, handler, transport_headers=identity.headers()) as (transport, _):
            yield transport, seen
    finally:
        if mode == "sync":
            api.close()
        else:
            await api.aclose()


_LOCAL_PAYLOAD = {
    "org_id": "acme",
    "project_id": "research",
    "spec": {"name": "case-replay", "execution": {"kind": "command", "command": "true"}},
}
_LOCAL_SCOPE = {"org_id": "acme", "project_id": "research"}


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_case_variant_keys_replay_one_workload_in_real_local_api(mode):
    async with _local_api(mode) as (transport, seen):
        first = await _mutate(
            transport, "/api/v1/workloads", json=_LOCAL_PAYLOAD, headers={"idempotency-key": "local-key"}
        )
        replay = await _mutate(
            transport, "/api/v1/workloads", json=_LOCAL_PAYLOAD, headers={"iDeMpOtEnCy-KeY": "local-key"}
        )
        listed = await _checked(transport, "/api/v1/workloads", params=_LOCAL_SCOPE)
        events = await _checked(transport, "/api/v1/events", params=_LOCAL_SCOPE)
    assert first.status_code == 201
    assert replay.status_code == 200
    assert replay.headers["Idempotent-Replay"] == "true"
    assert replay.json()["id"] == first.json()["id"]
    assert [workload["id"] for workload in listed.json()] == [first.json()["id"]]
    assert [event["action"] for event in events.json()] == ["workload.submitted"]
    assert seen == [["local-key"], ["local-key"]]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_recovery_metadata_replays_actual_admission_after_lost_responses(mode):
    async with _local_api(mode, lose_responses=True) as (transport, seen):
        with pytest.raises(EyvanUnavailableError) as caught:
            await _mutate(
                transport, "/api/v1/workloads", json=_LOCAL_PAYLOAD, headers={"iDeMpOtEnCy-KeY": "admitted-key"}
            )
        key = caught.value.idempotency_key
        assert key == "admitted-key"
        replay = await _mutate(transport, "/api/v1/workloads", json=_LOCAL_PAYLOAD, idempotency_key=key)
        listed = await _checked(transport, "/api/v1/workloads", params=_LOCAL_SCOPE)
        events = await _checked(transport, "/api/v1/events", params=_LOCAL_SCOPE)
    assert replay.status_code == 200
    assert replay.headers["Idempotent-Replay"] == "true"
    assert [workload["id"] for workload in listed.json()] == [replay.json()["id"]]
    assert [event["action"] for event in events.json()] == ["workload.submitted"]
    assert seen == [["admitted-key"]] * 3
