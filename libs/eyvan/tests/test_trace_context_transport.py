# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Opt-in trace-parent transport without baggage or authentication bypass."""

import httpx
import pytest
from fastapi.testclient import TestClient
from opentelemetry import baggage, context, trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from eyvan._internal.correlation import traced_operation
from eyvan.capacity import CapacityStatus
from eyvan.config import EyvanSettings
from eyvan.worker.agent.app import create_agent_app
from eyvan.worker.agent.client import CellAgentClient
from eyvan.worker.agent.models import AgentObservation


@pytest.fixture
def telemetry(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(trace, "get_tracer", provider.get_tracer)
    yield exporter
    provider.shutdown()


class ProbeService:
    def __init__(self):
        self.calls = 0
        self.baggage = None

    def reconcile(self, cell_id, request):
        self.calls += 1
        self.baggage = dict(baggage.get_all())
        with traced_operation("probe.provider", request.mutation.correlation_id):
            return AgentObservation(cell_id=cell_id, state=CapacityStatus.HEALTHY, generation=1)

    def close(self):
        pass


def _headers_from_client():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={})

    with httpx.Client(transport=httpx.MockTransport(respond), base_url="https://agent.invalid") as client:
        CellAgentClient("https://agent.invalid", client=client)._send("GET", "/probe")
    return dict(requests[0].headers)


def test_correlated_client_injects_only_traceparent(telemetry):
    token = context.attach(baggage.set_baggage("private", "must-not-forward"))
    try:
        with traced_operation("controller.delivery", "workload-1"):
            parent = trace.get_current_span().get_span_context()
            headers = _headers_from_client()
    finally:
        context.detach(token)
    assert headers["traceparent"] == f"00-{parent.trace_id:032x}-{parent.span_id:016x}-{int(parent.trace_flags):02x}"
    assert "baggage" not in headers
    assert "tracestate" not in headers


def test_uncorrelated_client_keeps_headers_absent_even_under_a_span(telemetry):
    with trace.get_tracer("test").start_as_current_span("unrelated"):
        assert "traceparent" not in _headers_from_client()
        with traced_operation("disabled", None):
            assert "traceparent" not in _headers_from_client()


@pytest.mark.parametrize("enabled", [False, True])
def test_authenticated_agent_joins_only_opted_in_parent(telemetry, enabled):
    with trace.get_tracer("test").start_as_current_span("controller") as parent:
        parent_context = parent.get_span_context()
        traceparent = f"00-{parent_context.trace_id:032x}-{parent_context.span_id:016x}-01"
    service = ProbeService()
    app = create_agent_app(service, EyvanSettings(agent_shared_secret="test-secret", correlation_enabled=enabled))
    payload = {
        "mutation": {"idempotency_key": "k", "generation": 1, "fencing_token": "f", "correlation_id": "workload-1"}
    }
    with TestClient(app) as client:
        headers = {
            "traceparent": traceparent,
            "baggage": "private=do-not-forward",
            "Authorization": "Bearer test-secret",
        }
        response = client.post("/api/v1/cells/cell/reconcile", json=payload, headers=headers)
        assert response.status_code == 200, response.text
    child = next(span for span in telemetry.get_finished_spans() if span.name == "probe.provider")
    if enabled:
        assert child.context.trace_id == parent_context.trace_id
        assert child.parent.span_id == parent_context.span_id
        assert child.parent.is_remote
    else:
        assert child.context.trace_id != parent_context.trace_id
    assert service.baggage == {}
    assert not trace.get_current_span().get_span_context().is_valid


def test_trace_header_never_bypasses_agent_authentication(telemetry):
    service = ProbeService()
    app = create_agent_app(service, EyvanSettings(agent_shared_secret="test-secret", correlation_enabled=True))
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/cells/cell/reconcile",
            json={"mutation": {"idempotency_key": "k", "generation": 1, "fencing_token": "f"}},
            headers={"traceparent": "00-" + "1" * 32 + "-" + "2" * 16 + "-01"},
        )
    assert response.status_code == 401
    assert service.calls == 0
    assert not telemetry.get_finished_spans()


@pytest.mark.parametrize("value", [None, "", "garbage", "x" * 513, "\ud800", "00-" + "0" * 32 + "-" + "2" * 16 + "-01"])
def test_invalid_parents_clear_ambient_context_and_restore_after_error(telemetry, value):
    from eyvan._internal.trace_context import remote_trace_parent

    with trace.get_tracer("test").start_as_current_span("outer") as outer:
        with pytest.raises(RuntimeError, match="failure"):
            with remote_trace_parent(value, enabled=True):
                assert not trace.get_current_span().get_span_context().is_valid
                raise RuntimeError("failure")
        assert trace.get_current_span().get_span_context() == outer.get_span_context()


def test_vendor_state_is_not_transmitted(telemetry):
    parent = trace.SpanContext(
        trace_id=1,
        span_id=2,
        is_remote=False,
        trace_flags=trace.TraceFlags(1),
        trace_state=trace.TraceState([("vendor", "private-value")]),
    )
    token = context.attach(trace.set_span_in_context(trace.NonRecordingSpan(parent)))
    try:
        with traced_operation("controller", "workload-1"):
            headers = _headers_from_client()
            assert "traceparent" in headers
            assert "tracestate" not in headers
    finally:
        context.detach(token)


async def test_concurrent_agent_requests_keep_distinct_parents(telemetry):
    import asyncio

    service = ProbeService()
    app = create_agent_app(service, EyvanSettings(agent_shared_secret="test-secret", correlation_enabled=True))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://agent") as client:

        async def send(identifier, digit):
            response = await client.post(
                "/api/v1/cells/cell/reconcile",
                json={
                    "mutation": {
                        "idempotency_key": identifier,
                        "generation": 1,
                        "fencing_token": "f",
                        "correlation_id": identifier,
                    }
                },
                headers={"Authorization": "Bearer test-secret", "traceparent": f"00-{digit * 32}-{digit * 16}-01"},
            )
            assert response.status_code == 200, response.text

        await asyncio.gather(send("first", "1"), send("second", "2"))
    spans = {span.attributes["eyvan.correlation_id"]: span for span in telemetry.get_finished_spans()}
    assert spans["first"].context.trace_id == int("1" * 32, 16)
    assert spans["second"].context.trace_id == int("2" * 32, 16)
    assert not trace.get_current_span().get_span_context().is_valid


def test_separate_process_joins_serialized_parent_through_authenticated_asgi(telemetry):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    with traced_operation("controller.delivery", "workload-process"):
        parent = trace.get_current_span().get_span_context()
        headers = _headers_from_client()
    code = """
import json, sys
from fastapi.testclient import TestClient
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from eyvan._internal.correlation import traced_operation
from eyvan.capacity import CapacityStatus
from eyvan.config import EyvanSettings
from eyvan.worker.agent.app import create_agent_app
from eyvan.worker.agent.models import AgentObservation
exporter = InMemorySpanExporter()
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(exporter))
trace.set_tracer_provider(provider)
class Probe:
    def close(self): pass
    def reconcile(self, cell_id, request):
        with traced_operation("child.provider", request.mutation.correlation_id):
            return AgentObservation(cell_id=cell_id, state=CapacityStatus.HEALTHY, generation=1)
app = create_agent_app(Probe(), EyvanSettings(agent_shared_secret="test-secret", correlation_enabled=True))
headers = json.loads(sys.argv[1])
headers["Authorization"] = "Bearer test-secret"
with TestClient(app) as client:
    response = client.post("/api/v1/cells/cell/reconcile", headers=headers, json={"mutation": {
        "idempotency_key": "k", "generation": 1, "fencing_token": "f", "correlation_id": "workload-process"}})
    assert response.status_code == 200, response.text
span, = exporter.get_finished_spans()
print(json.dumps({"trace": span.context.trace_id, "parent": span.parent.span_id, "remote": span.parent.is_remote}))
provider.shutdown()
"""
    result = subprocess.run(
        [sys.executable, "-c", code, json.dumps({"traceparent": headers["traceparent"]})],
        cwd=Path(__file__).resolve().parents[1],
        env={key: value for key, value in os.environ.items() if not key.startswith("EYVAN_") and key != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == {"trace": parent.trace_id, "parent": parent.span_id, "remote": True}


@pytest.mark.parametrize("enabled", [False, True])
async def test_real_outbox_delivery_obeys_admission_opt_in_under_ambient_trace(telemetry, enabled):
    from eyvan.server.controller import EyvanController
    from eyvan.server.persistence import InMemoryRepository
    from eyvan.server.service import ControlPlaneService

    sent = []

    class Delivery(EyvanController):
        async def _dispatch(self, command):
            sent.append(_headers_from_client())

    repository = InMemoryRepository()
    await repository.open()
    service = ControlPlaneService(repository, correlation_enabled=enabled)
    controller = Delivery(repository, service)
    try:
        async with repository.transaction() as transaction:
            await service._outbox(
                transaction,
                topic="workload.dispatch",
                aggregate_type="workload",
                aggregate_id="workload-1",
                idempotency_key="dispatch",
                generation=1,
                payload={},
            )
        with traced_operation("ambient", "unrelated-workload"):
            await controller.dispatch_outbox_once()
        assert len(sent) == 1
        assert ("traceparent" in sent[0]) is enabled
    finally:
        await controller.close()
        await repository.close()


async def test_endpoint_failure_does_not_leak_parent_into_next_request(telemetry):
    class FailingOnce(ProbeService):
        def reconcile(self, cell_id, request):
            observation = super().reconcile(cell_id, request)
            if self.calls == 1:
                raise RuntimeError("service failure")
            return observation

    app = create_agent_app(FailingOnce(), EyvanSettings(agent_shared_secret="test-secret", correlation_enabled=True))
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    payload = {
        "mutation": {"idempotency_key": "k", "generation": 1, "fencing_token": "f", "correlation_id": "workload-1"}
    }
    async with httpx.AsyncClient(transport=transport, base_url="http://agent") as client:
        response = await client.post(
            "/api/v1/cells/cell/reconcile",
            json=payload,
            headers={"Authorization": "Bearer test-secret", "traceparent": "00-" + "1" * 32 + "-" + "2" * 16 + "-01"},
        )
        assert response.status_code == 500
        assert not trace.get_current_span().get_span_context().is_valid
        response = await client.post(
            "/api/v1/cells/cell/reconcile",
            json=payload,
            headers={"Authorization": "Bearer test-secret"},
        )
        assert response.status_code == 200
    first, second = telemetry.get_finished_spans()
    assert first.context.trace_id == int("1" * 32, 16)
    assert second.parent is None
    assert second.context.trace_id != first.context.trace_id
