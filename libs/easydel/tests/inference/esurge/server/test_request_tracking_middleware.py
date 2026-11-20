# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure-Python contract of the request-tracking ASGI middleware (no array computation)."""

from types import SimpleNamespace

from easydel.inference.inference_engine_interface import RequestTrackingMiddleware, ServerMetrics
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.testclient import TestClient


def _app():
    server = SimpleNamespace(metrics=ServerMetrics(), _active_requests=set())
    app = FastAPI()
    seen: dict[str, object] = {}

    @app.get("/ok")
    async def ok(request: Request):
        seen["state_id"] = request.state.request_id
        seen["active_during"] = set(server._active_requests)
        return {"ok": True}

    @app.get("/bad")
    async def bad():
        raise HTTPException(status_code=422, detail="nope")

    @app.get("/boom")
    async def boom():
        raise RuntimeError("boom")

    @app.get("/health")
    async def health():
        return {"status": "ready"}

    @app.get("/stream")
    async def stream():
        async def body():
            for i in range(3):
                seen.setdefault("active_while_streaming", []).append(len(server._active_requests))
                yield f"data: {i}\n\n"

        return StreamingResponse(body(), media_type="text/event-stream")

    app.add_middleware(RequestTrackingMiddleware, server=server)
    return app, server, seen


def test_request_id_reaches_state_and_header_and_counts_success():
    app, server, seen = _app()
    with TestClient(app) as client:
        response = client.get("/ok")
    assert response.status_code == 200
    rid = response.headers["X-Request-ID"]
    assert rid.startswith("req_") and seen["state_id"] == rid
    assert seen["active_during"] == {rid}
    assert server._active_requests == set()
    assert (server.metrics.total_requests, server.metrics.successful_requests, server.metrics.failed_requests) == (
        1,
        1,
        0,
    )


def test_error_statuses_and_exceptions_count_as_failures():
    app, server, _ = _app()
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/bad").status_code == 422
        assert client.get("/boom").status_code == 500
    assert server.metrics.total_requests == 2
    assert server.metrics.successful_requests == 0
    assert server.metrics.failed_requests == 2
    assert server._active_requests == set()


def test_monitoring_paths_get_an_id_but_are_not_counted():
    app, server, _ = _app()
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.headers["X-Request-ID"].startswith("req_")
    assert server.metrics.total_requests == 0 and server.metrics.successful_requests == 0
    assert server._active_requests == set()


def test_streamed_request_stays_active_until_the_body_is_sent():
    app, server, seen = _app()
    with TestClient(app) as client:
        response = client.get("/stream")
    assert response.text == "data: 0\n\ndata: 1\n\ndata: 2\n\n"
    assert response.headers["X-Request-ID"].startswith("req_")
    assert seen["active_while_streaming"] == [1, 1, 1]
    assert server._active_requests == set()
    assert server.metrics.successful_requests == 1


def test_server_installs_no_base_http_middleware():
    # BaseHTTPMiddleware moves every streamed chunk through an anyio stream + task group; the server must not use it.
    from easydel.inference.esurge.server.api_server import eSurgeApiServer

    server = object.__new__(eSurgeApiServer)
    server.app = FastAPI()
    server._setup_middleware()
    classes = [m.cls for m in server.app.user_middleware]
    assert RequestTrackingMiddleware in classes
    assert not any(isinstance(c, type) and issubclass(c, BaseHTTPMiddleware) for c in classes)
