# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Opt-in native-worker tracing preserves authentication and retry semantics."""

import httpx
import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace
from test_trace_context_transport import telemetry as _telemetry_fixture

from eyvan._internal.correlation import traced_operation
from eyvan.worker.runtime.native_remote import NativeRemoteExecutor
from eyvan.worker.runtime.native_worker import create_native_worker_app

telemetry = _telemetry_fixture


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("method", ["GET", "POST"])
def test_native_client_retries_only_reads_without_losing_parent_or_bearer(telemetry, enabled, method):
    requests = []

    def respond(request):
        requests.append(request)
        if len(requests) == 1:
            raise httpx.RemoteProtocolError("closed keepalive")
        return httpx.Response(200, json={})

    with httpx.Client(transport=httpx.MockTransport(respond), base_url="http://worker") as client:
        executor = NativeRemoteExecutor("http://worker", secret="test-secret", client=client)
        with traced_operation("agent.delivery", "workload-1" if enabled else None):
            parent = trace.get_current_span().get_span_context()
            if method == "GET":
                assert executor._send(method, "/api/v1/health").status_code == 200
            else:
                with pytest.raises(httpx.RemoteProtocolError):
                    executor._send(method, "/api/v1/workloads")
        assert executor._headers == {"Authorization": "Bearer test-secret"}
    assert len(requests) == (2 if method == "GET" else 1)
    for request in requests:
        assert request.headers["Authorization"] == "Bearer test-secret"
        assert ("traceparent" in request.headers) is enabled
        if enabled:
            assert request.headers["traceparent"].split("-")[1:3] == [
                f"{parent.trace_id:032x}",
                f"{parent.span_id:016x}",
            ]
        assert "baggage" not in request.headers
        assert "tracestate" not in request.headers


class ExecutorProbe:
    def submit(self, spec, mutation):
        return "remote"


@pytest.mark.parametrize("enabled", [False, True])
def test_native_request_span_is_opt_in_and_uses_only_the_route_template(telemetry, enabled):
    app = create_native_worker_app(ExecutorProbe(), secret="test-secret", correlation_enabled=enabled)
    parent = "00-" + "1" * 32 + "-" + "2" * 16 + "-01"
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/health?private=must-not-export",
            headers={"Authorization": "Bearer test-secret", "traceparent": parent, "baggage": "private=hidden"},
        )
        assert response.status_code == 200
    spans = telemetry.get_finished_spans()
    assert len(spans) == int(enabled)
    if enabled:
        span = spans[0]
        assert span.name == "eyvan.native_worker_request"
        assert span.context.trace_id == int("1" * 32, 16)
        assert span.parent.span_id == int("2" * 16, 16)
        assert span.parent.is_remote
        assert span.attributes["http.route"] == "/api/v1/health"
        assert "must-not-export" not in str(span.attributes)
        assert "hidden" not in str(span.attributes)
    assert not trace.get_current_span().get_span_context().is_valid


def test_native_rejected_request_does_not_create_span(telemetry):
    app = create_native_worker_app(ExecutorProbe(), secret="test-secret", correlation_enabled=True)
    with TestClient(app) as client:
        response = client.get("/api/v1/health", headers={"traceparent": "x" * 513})
    assert response.status_code == 401
    assert not telemetry.get_finished_spans()


def test_native_correlation_setting_is_validated_before_consuming_secret(monkeypatch, tmp_path):
    from eyvan.worker.runtime.native_worker import main

    secret_file = tmp_path / "one-use-secret"
    secret_file.write_text("test-secret")
    monkeypatch.setenv("EYVAN_CORRELATION_ENABLED", "invalid")
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    with pytest.raises(SystemExit) as error:
        main(["--host", "127.0.0.1", "--state-dir", str(tmp_path / "state"), "--secret-file", str(secret_file)])
    assert error.value.code == 2
    assert secret_file.read_text() == "test-secret"


@pytest.mark.parametrize("enabled", [False, True])
def test_native_cli_forwards_only_enabled_correlation(monkeypatch, tmp_path, enabled):
    from eyvan.worker.runtime import native_worker

    captured = []
    monkeypatch.setenv("EYVAN_CORRELATION_ENABLED", "1" if enabled else "0")
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    monkeypatch.setattr(native_worker, "create_native_worker_app", lambda **kwargs: captured.append(kwargs))
    assert native_worker.main(["--host", "127.0.0.1", "--secret", "test-secret", "--state-dir", str(tmp_path)]) == 0
    assert ("correlation_enabled" in captured[0]) is enabled
    if enabled:
        assert captured[0]["correlation_enabled"] is True


async def test_native_failure_span_is_private_and_context_is_restored(telemetry):
    class FailingExecutor(ExecutorProbe):
        def submit(self, spec, mutation):
            raise RuntimeError("private-provider-error-do-not-export")

    app = create_native_worker_app(FailingExecutor(), secret="test-secret", correlation_enabled=True)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False), base_url="http://worker"
    ) as client:
        response = await client.post(
            "/api/v1/workloads",
            json={
                "spec": {"metadata": {"eyvan_workload_id": "workload-1"}},
                "mutation": {"idempotency_key": "k", "generation": 1, "fencing_token": "f"},
            },
            headers={"Authorization": "Bearer test-secret", "traceparent": "00-" + "1" * 32 + "-" + "2" * 16 + "-01"},
        )
        assert response.status_code == 500
        assert not trace.get_current_span().get_span_context().is_valid
        assert (await client.get("/api/v1/health", headers={"Authorization": "Bearer test-secret"})).status_code == 200
    failed, subsequent = telemetry.get_finished_spans()
    assert failed.status.status_code is trace.StatusCode.ERROR
    assert failed.status.description is None
    assert failed.attributes["error.type"] == "RuntimeError"
    assert not failed.events
    assert "private-provider-error" not in str(failed.attributes)
    assert subsequent.parent is None
    assert subsequent.context.trace_id != failed.context.trace_id


def test_parallel_slice_staging_preserves_parent_on_every_host(telemetry):
    from contextlib import ExitStack
    from threading import Barrier

    from eyvan.worker.runtime.native_slice import NativeSliceStageDriver
    from eyvan.worker.runtime.stage_executor import AssetStageVerdict

    requests = []
    barrier = Barrier(2)

    def respond(request):
        requests.append(request)
        barrier.wait(timeout=5)
        return httpx.Response(200, json={"asset": "asset", "action": "measure", "verdict": "ok", "size_bytes": 1})

    with ExitStack() as stack:
        endpoints = ["http://first", "http://second"]
        workers = [
            NativeRemoteExecutor(
                endpoint,
                secret="test-secret",
                client=stack.enter_context(httpx.Client(base_url=endpoint, transport=httpx.MockTransport(respond))),
            )
            for endpoint in endpoints
        ]
        staging = NativeSliceStageDriver(workers, endpoints, generation=1)
        with traced_operation("agent.stage", "workload-1"):
            parent = trace.get_current_span().get_span_context()
            result = staging.measure("cell", "asset", max_bytes=10)
        assert result.verdict is AssetStageVerdict.OK, result.detail
    assert len(requests) == 2
    for request in requests:
        assert request.headers["traceparent"].split("-")[1:3] == [f"{parent.trace_id:032x}", f"{parent.span_id:016x}"]


def test_controller_agent_native_worker_chain_joins_both_authenticated_hops(telemetry):
    from eyvan.capacity import CapacityStatus
    from eyvan.config import EyvanSettings
    from eyvan.worker.agent.app import create_agent_app
    from eyvan.worker.agent.client import CellAgentClient
    from eyvan.worker.agent.models import AgentObservation

    worker_app = create_native_worker_app(ExecutorProbe(), secret="worker-secret", correlation_enabled=True)
    with TestClient(worker_app) as worker_client:
        remote = NativeRemoteExecutor("http://worker", secret="worker-secret", client=worker_client)

        class AgentProbe:
            def close(self):
                pass

            def reconcile(self, cell_id, request):
                with traced_operation("agent.provider", request.mutation.correlation_id):
                    assert remote._send("GET", "/api/v1/health").status_code == 200
                    return AgentObservation(cell_id=cell_id, state=CapacityStatus.HEALTHY, generation=1)

        agent_app = create_agent_app(
            AgentProbe(), EyvanSettings(agent_shared_secret="agent-secret", correlation_enabled=True)
        )
        with TestClient(agent_app) as agent_client:
            client = CellAgentClient("http://agent", client=agent_client)
            with traced_operation("controller.delivery", "workload-1"):
                response = client._send(
                    "POST",
                    "/api/v1/cells/cell/reconcile",
                    headers={"Authorization": "Bearer agent-secret"},
                    json={
                        "mutation": {
                            "idempotency_key": "k",
                            "generation": 1,
                            "fencing_token": "f",
                            "correlation_id": "workload-1",
                        }
                    },
                )
                assert response.status_code == 200
    spans = {span.name: span for span in telemetry.get_finished_spans()}
    controller = spans["controller.delivery"]
    agent = spans["agent.provider"]
    worker = spans["eyvan.native_worker_request"]
    assert controller.context.trace_id == agent.context.trace_id == worker.context.trace_id
    assert agent.parent.span_id == controller.context.span_id
    assert worker.parent.span_id == agent.context.span_id
    assert agent.parent.is_remote and worker.parent.is_remote
