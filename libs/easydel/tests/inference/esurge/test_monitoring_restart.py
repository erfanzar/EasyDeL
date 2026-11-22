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

"""Prometheus monitoring must be restartable on the same port and export preemption/system-info metrics."""

from __future__ import annotations

import socket
import threading
import urllib.request

import pytest
from easydel.inference.esurge import monitoring
from easydel.inference.esurge.metrics import MetricsCollector, initialize_metrics

pytestmark = pytest.mark.cpu_ok

if not monitoring.PROMETHEUS_AVAILABLE:  # pragma: no cover
    pytest.skip("prometheus_client not installed", allow_module_level=True)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _scrape(port: int) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as resp:
        return resp.read().decode()


def test_monitoring_can_restart_on_same_port_after_stop():
    port = _free_port()
    server = monitoring.start_monitoring_server(prometheus_port=port, update_interval=0.05)
    assert "esurge_" in _scrape(port)
    monitoring.stop_monitoring()
    assert monitoring._monitoring_server is None

    server = monitoring.start_monitoring_server(prometheus_port=port, update_interval=0.05)
    try:
        assert "esurge_" in _scrape(port)
    finally:
        monitoring.stop_monitoring()
    assert server.running is False


def test_restart_with_long_update_interval_leaves_no_updater_threads():
    baseline = set(threading.enumerate())
    server = monitoring.eSurgeMonitoringServer(prometheus_port=_free_port(), update_interval=60.0)
    try:
        for _ in range(2):
            server.start()
            assert "esurge_" in _scrape(server.prometheus_port)
            server.stop()
        assert [thread for thread in threading.enumerate() if thread not in baseline and thread.is_alive()] == []
    finally:
        server.stop()


def test_failed_monitoring_bind_does_not_poison_later_start():
    monitoring.stop_monitoring()
    with socket.socket() as occupied:
        occupied.bind(("0.0.0.0", 0))
        occupied.listen()
        with pytest.raises(OSError):
            monitoring.start_monitoring_server(prometheus_port=occupied.getsockname()[1])
        # A failed start must not retain the failed configuration or registered
        # collectors; a fresh start on another port must work immediately.
        port = _free_port()
        try:
            server = monitoring.start_monitoring_server(prometheus_port=port)
            assert "esurge_" in _scrape(port)
            assert server.running
        finally:
            monitoring.stop_monitoring()


def test_failed_updater_start_releases_listener_and_metric_names(monkeypatch):
    port = _free_port()
    server = monitoring.eSurgeMonitoringServer(prometheus_port=port)
    original_start = threading.Thread.start

    def fail_updater_start(thread):
        # The HTTP listener starts successfully first; only the polling worker
        # fails. Exercise rollback after the listening socket is already live.
        if getattr(thread, "_target", None) == server._update_metrics_loop:
            raise RuntimeError("updater startup failed")
        return original_start(thread)

    with monkeypatch.context() as patch:
        patch.setattr(threading.Thread, "start", fail_updater_start)
        with pytest.raises(RuntimeError, match="updater startup failed"):
            server.start()
    try:
        server.start()
        assert "esurge_" in _scrape(port)
    finally:
        server.stop()


def test_preempted_counter_and_system_info_are_updated():
    metrics = monitoring.PrometheusMetrics(prefix="r1test_")
    try:
        collector = initialize_metrics()
        assert isinstance(collector, MetricsCollector)
        collector.record_scheduler_metrics(num_waiting=0, num_running=1, num_scheduled_tokens=4, num_preempted=2)
        collector.record_scheduler_metrics(num_waiting=0, num_running=1, num_scheduled_tokens=4, num_preempted=1)
        metrics.update_from_metrics_collector(collector)
        metrics.update_from_metrics_collector(collector)  # no double counting
        counter_samples = {
            sample.name: sample.value
            for family in metrics.preempted_requests_total.collect()
            for sample in family.samples
        }
        assert counter_samples["r1test_preempted_requests_total"] == 3
        info_samples = [sample for family in metrics.system_info.collect() for sample in family.samples]
        assert info_samples[0].labels["python_version"]
    finally:
        metrics.close()
