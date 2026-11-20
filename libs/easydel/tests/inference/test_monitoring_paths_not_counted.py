"""Monitoring polls (``/metrics``, ``/health``) must not inflate the server's request counters."""

import asyncio
from types import SimpleNamespace

import pytest
from easydel.inference.inference_engine_interface import BaseInferenceApiServer, ServerMetrics
from fastapi import FastAPI
from fastapi.testclient import TestClient

pytestmark = pytest.mark.cpu_ok


def _server() -> SimpleNamespace:
    app = FastAPI()
    server = SimpleNamespace(app=app, metrics=ServerMetrics(), _request_lock=asyncio.Lock(), _active_requests=set())
    BaseInferenceApiServer._setup_middleware(server)
    app.add_api_route("/metrics", lambda: {"ok": True}, methods=["GET"])
    app.add_api_route("/health", lambda: {"ok": True}, methods=["GET"])
    app.add_api_route("/v1/models", lambda: {"data": []}, methods=["GET"])
    return server


def test_monitoring_polls_are_not_counted_but_traffic_is():
    server = _server()
    client = TestClient(server.app)
    for _ in range(5):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["X-Request-ID"].startswith("req_")
    client.get("/health")
    assert (server.metrics.total_requests, server.metrics.successful_requests) == (0, 0)

    client.get("/v1/models")
    client.get("/v1/missing")
    assert server.metrics.total_requests == 2
    assert server.metrics.successful_requests == 1
    assert server.metrics.failed_requests == 1
    assert server._active_requests == set()
