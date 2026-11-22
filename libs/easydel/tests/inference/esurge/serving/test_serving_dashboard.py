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

"""The dashboard starts with the endpoint by default, and dp fronts answer /metrics for the whole deployment.

Pure Python: subprocesses, loopback and unix sockets only, no array computation.
"""

import asyncio
import copy
import http.client
import json
import os
import socket
import threading
import time

import pytest
from easydel.inference.esurge.metrics import MetricsCollector
from easydel.inference.esurge.serving import dashboard as dash
from easydel.inference.esurge.serving.config import (
    CoordinationSpec,
    ModelSpec,
    ServerSpec,
    ServingConfig,
    ServingConfigError,
)

pytestmark = pytest.mark.cpu_ok


# --- configuration and supervisor -------------------------------------------------------------------------------


def test_dashboard_is_on_by_default_and_can_be_turned_off():
    spec = ServerSpec()
    assert spec.dashboard is True and spec.dashboard_port == 9090
    assert ServerSpec(dashboard=False).dashboard is False
    with pytest.raises(ServingConfigError):
        ServerSpec(dashboard_port=8000)
    with pytest.raises(ServingConfigError):
        ServerSpec(dashboard="no")
    with pytest.raises(ServingConfigError):
        ServingConfig(model=ModelSpec(checkpoint="c"), coordination=CoordinationSpec(control_port=9089))
    # Off: the port is not reserved.
    ServingConfig(
        model=ModelSpec(checkpoint="c"), server=ServerSpec(dashboard=False), coordination=CoordinationSpec(9090)
    )


def test_start_dashboard_respects_the_switch(monkeypatch):
    from easydel.inference.esurge.serving import app

    started = []
    monkeypatch.setattr(dash.DashboardSupervisor, "start", lambda self: started.append(self) or self)
    off = ServingConfig(model=ModelSpec(checkpoint="c"), server=ServerSpec(dashboard=False))
    on = ServingConfig(model=ModelSpec(checkpoint="c"))
    assert app.start_dashboard(off, "10.0.0.1", 0) is None and started == []
    supervisor = app.start_dashboard(on, "10.0.0.1", 0)
    assert started == [supervisor]
    assert supervisor.listen == "10.0.0.1:9090" and supervisor.upstream == "http://10.0.0.1:8000"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _wait(predicate, seconds=10.0):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        time.sleep(0.05)
    return predicate()


def test_supervisor_restarts_the_dashboard_and_builds_at_most_once(tmp_path):
    seen = tmp_path / "argv"
    binary = tmp_path / "esurge-dashboard"
    binary.write_text(f'#!/bin/sh\necho "$@" >> {seen}\nexit 3\n')
    binary.chmod(0o755)
    builds = []

    def find(*, build, log, stop):
        builds.append(build)
        return str(binary)

    port = _free_port()
    supervisor = dash.DashboardSupervisor(
        host="127.0.0.1", port=port, endpoint_port=8000, log=lambda m: None, find_binary=find, max_backoff_s=0.05
    ).start()
    try:
        assert _wait(lambda: supervisor.starts >= 3)
    finally:
        supervisor.stop()
    assert builds[0] is True and not any(builds[1:])
    assert seen.read_text().splitlines()[0] == f"--upstream http://127.0.0.1:8000 --listen 127.0.0.1:{port}"


def test_supervisor_waits_while_the_port_is_taken_and_stop_ends_the_dashboard(tmp_path):
    binary = tmp_path / "esurge-dashboard"
    binary.write_text("#!/bin/sh\nexec sleep 300\n")
    binary.chmod(0o755)
    holder = socket.socket()
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    port = holder.getsockname()[1]
    supervisor = dash.DashboardSupervisor(
        host="127.0.0.1",
        port=port,
        endpoint_port=8000,
        log=lambda m: None,
        find_binary=lambda **kw: str(binary),
        max_backoff_s=0.1,
    ).start()
    try:
        time.sleep(0.6)
        assert supervisor.starts == 0  # nothing started while the port is taken
        holder.close()
        assert _wait(lambda: supervisor.starts == 1)
        pid = supervisor.pid
        os.kill(pid, 0)  # the dashboard is running
    finally:
        holder.close()
        supervisor.stop()
    # stop() ended (and reaped) the running dashboard.
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


# --- merging ------------------------------------------------------------------------------------------------------


def _collector(ids, ttft):
    collector = MetricsCollector(enable_detailed_logging=False)
    for rid in ids:
        collector.start_request(rid, 100, model_name="m")
        collector.request_metrics[rid].start_time -= ttft
        collector.record_first_token(rid)
        collector.complete_request(rid, finish_reason="stop")
    return collector


def _payload(collector, total, rate=1.5):
    return {
        "status": "ready",
        "uptime_seconds": 5.0,
        "total_requests": total,
        "successful_requests": total,
        "total_tokens_generated": 10 * total,
        "active_requests": 1,
        "average_tokens_per_second": rate,
        "engine_metrics": {"m": {"running_requests": 7, "waiting_requests": 0, **collector.model_summary("m")}},
    }


def _with_rows(payload, collector):
    payload = copy.deepcopy(payload)
    payload["engine_metrics"]["m"]["request_rows"] = collector.request_rows("m")
    return payload


def _one_collector(*collectors):
    """Oracle: a single collector that saw every front's requests."""
    union = MetricsCollector(enable_detailed_logging=False)
    for collector in collectors:
        union.completed_requests.extend(copy.deepcopy(list(collector.completed_requests)))
    return union


def test_merged_metrics_equal_one_collector_over_every_front():
    fast, slow = _collector(["a1", "a2", "a3"], 0.1), _collector(["b1"], 5.0)
    merged = dash.merge_front_metrics(
        [_with_rows(_payload(fast, 3), fast), _with_rows(_payload(slow, 1), slow)], window_seconds=300.0
    )
    expected = _one_collector(fast, slow).model_summary("m")
    assert merged["total_requests"] == 4 and merged["total_tokens_generated"] == 40 and merged["fronts"] == 2
    engine = merged["engine_metrics"]["m"]
    assert engine["aggregate"] == expected["aggregate"] and engine["aggregate"]["completed"] == 4
    assert [r["request_id"] for r in engine["recent_requests"]] == [r["request_id"] for r in expected["recent_requests"]]
    assert engine["running_requests"] == 7 and "request_rows" not in engine


def test_per_request_rate_is_averaged_over_fronts_not_added():
    payloads = [{"average_tokens_per_second": 10.0}, {"average_tokens_per_second": 30.0}, {"status": "ready"}]
    assert dash.merge_front_metrics(payloads, window_seconds=300.0)["average_tokens_per_second"] == 20.0


def test_front_socket_directory_is_private_and_fresh():
    first, second = dash.create_front_socket_dir(), dash.create_front_socket_dir()
    try:
        assert first != second
        info = os.stat(first)
        assert info.st_uid == os.getuid() and info.st_mode & 0o777 == 0o700
        assert len(dash.front_socket_path(first, 7)) < 108
    finally:
        os.rmdir(first)
        os.rmdir(second)


# --- the dp fronts end to end -------------------------------------------------------------------------------------


class _Unix(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("localhost", timeout=10)
        self._path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self._path)


def _get(conn, headers=None):
    conn.request("GET", "/metrics", headers=headers or {})
    response = conn.getresponse()
    return response.status, json.loads(response.read() or b"null")


def _garbage_peer(path):
    """A peer that answers 200 with a body that is not JSON."""
    server = dash.bind_unix_socket(path)

    def serve():
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            conn.recv(4096)
            conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\nConnection: close\r\n\r\nnot json")
            conn.close()

    threading.Thread(target=serve, daemon=True).start()
    return server


@pytest.fixture
def fronts(tmp_path):
    """Two fronts sharing one TCP port, each with its private socket; plus a garbage peer and a missing one."""
    import uvicorn
    from easydel.inference.esurge.serving.app import reuse_port_socket
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    directory = str(tmp_path)
    sockets = [dash.front_socket_path(directory, i) for i in range(4)]  # 2: garbage, 3: never started
    collectors = [_collector(["p1", "p2"], 0.2), _collector(["q1"], 0.4)]
    port = _free_port()
    servers = []

    def make(collector, total):
        api = FastAPI()

        @api.get("/metrics")
        def metrics(request: Request):
            key = request.headers.get("authorization")
            if key == "Bearer limited":
                return JSONResponse({"detail": "rate limited"}, status_code=429)
            if key != "Bearer good":
                return JSONResponse({"detail": "bad key"}, status_code=401)
            return _payload(collector, total)

        return api

    for index, (collector, total) in enumerate(zip(collectors, (2, 1), strict=True)):
        app = dash.FrontMetricsMiddleware(
            make(collector, total),
            own_socket=sockets[index],
            sockets=sockets,
            local_metrics=lambda c=collector, t=total: _payload(c, t),
            request_rows=lambda name, c=collector: c.request_rows(name),
            window_seconds=300.0,
        )
        server = uvicorn.Server(uvicorn.Config(app, log_level="warning"))
        listen = [reuse_port_socket("127.0.0.1", port), dash.bind_unix_socket(sockets[index])]
        threading.Thread(target=lambda s=server, k=listen: asyncio.run(s.serve(sockets=k)), daemon=True).start()
        servers.append(server)
    garbage = _garbage_peer(sockets[2])
    assert _wait(lambda: all(s.started for s in servers))
    yield port, sockets, collectors
    garbage.close()
    for server in servers:
        server.should_exit = True


def test_every_front_answers_metrics_for_the_whole_deployment(fronts):
    port, _, collectors = fronts
    expected = _one_collector(*collectors).model_summary("m")["aggregate"]
    for _ in range(6):  # the kernel spreads these over both fronts; each must report both
        status, body = _get(http.client.HTTPConnection("127.0.0.1", port, timeout=10), {"Authorization": "Bearer good"})
        assert status == 200
        assert (body["fronts"], body["fronts_expected"], body["total_requests"]) == (2, 4, 3)
        engine = body["engine_metrics"]["m"]
        assert engine["aggregate"] == expected and "request_rows" not in engine


def test_refusals_on_the_endpoint_port_come_back_unchanged(fronts):
    port = fronts[0]
    for headers, status in (
        ({}, 401),
        ({"Authorization": "Bearer nope"}, 401),
        ({"Authorization": "Bearer limited"}, 429),
    ):
        assert _get(http.client.HTTPConnection("127.0.0.1", port, timeout=10), headers)[0] == status


def test_only_the_private_socket_serves_peer_rows_without_a_key(fronts):
    port, sockets, _ = fronts
    status, body = _get(_Unix(sockets[0]))
    assert status == 200 and len(body["engine_metrics"]["m"]["request_rows"]) == 2
    assert "fronts" not in body  # a peer gets only this front's metrics, not a merge
    # The same unauthenticated call on the shared TCP port is refused.
    assert _get(http.client.HTTPConnection("127.0.0.1", port, timeout=10))[0] == 401
