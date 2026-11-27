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

"""Tests for Eyvan tracking integration in eSurge monitoring and MonitoringStack."""

from __future__ import annotations

import gc
import os
import socket
import threading
import time
import warnings
import weakref
from pathlib import Path

import eyvan.track as eyvan_track
import pytest
from easydel.inference.esurge import (
    EYVAN_TRACK_AVAILABLE,
    EyvanMetricsExporter,
    eSurge,
    monitoring,
    start_eyvan_tracking,
)
from easydel.inference.esurge.engine.monitoring_stack import MonitoringStack
from easydel.inference.esurge.metrics import MetricsCollector, initialize_metrics
from easydel.utils.eyvan_helpers import is_eyvan_auto_track_enabled, set_eyvan_auto_track_enabled
from eyvan.track._format import (
    META_CONFIG,
    META_FINISH,
    META_START,
    META_SUMMARY,
    STREAM_HIST,
    STREAM_META,
    STREAM_SCALARS,
    decode_segment,
)
from eyvan.track._store import LocalTrackingStore

pytestmark = pytest.mark.cpu_ok


@pytest.fixture(autouse=True)
def _clean_monitoring_and_eyvan_env(monkeypatch):
    """Ensure clean global monitoring state and isolated Eyvan env vars per test."""
    monkeypatch.delenv("EYVAN_TRACKING_URI", raising=False)
    monkeypatch.delenv("EYVAN_WORKLOAD_ID", raising=False)
    monkeypatch.delenv("EYVAN_DISTRIBUTED_PROCESS_ID", raising=False)
    monkeypatch.delenv("JAX_PROCESS_ID", raising=False)
    monitoring.stop_monitoring()
    try:
        yield
    finally:
        monitoring.stop_monitoring()


def _read_stream(store_dir: Path, run_id: str, stream: str) -> list[dict[str, object]]:
    store = LocalTrackingStore(store_dir)
    records: list[dict[str, object]] = []
    for key in store.list(f"{run_id}/{stream}"):
        records.extend(decode_segment(store.get(key)))
    return records


def _complete_request(
    collector: MetricsCollector,
    request_id: str,
    *,
    prompt_tokens: int = 16,
    generated_tokens: int = 8,
    ttft_delay: float = 0.002,
    gen_delay: float = 0.004,
    error: str | None = None,
) -> None:
    collector.start_request(request_id, prompt_tokens=prompt_tokens)
    if error is None:
        time.sleep(ttft_delay)
        collector.record_first_token(request_id)
        collector.add_generated_tokens(request_id, generated_tokens)
        time.sleep(gen_delay)
        collector.complete_request(request_id, finish_reason="stop")
    else:
        time.sleep(ttft_delay)
        collector.complete_request(request_id, finish_reason="abort", error=error)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_until(predicate, *, timeout: float = 3.0, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def test_eyvan_metrics_exporter_scalars_histograms_no_double_counting_and_summary(tmp_path):
    assert EYVAN_TRACK_AVAILABLE is True
    store_dir = tmp_path / "eyvan-store"
    collector = MetricsCollector()

    exporter = EyvanMetricsExporter(
        name="esurge-unit-run",
        project="esurge-test",
        group="unit-sweep",
        tags=("EasyDeL", "eSurge", "TestTag"),
        uri=str(store_dir),
        flush_interval_s=0.1,
        config={"max_model_len": 4096, "tensor_parallel": 2},
    )

    _complete_request(collector, "req-1", prompt_tokens=32, generated_tokens=12)
    _complete_request(collector, "req-2", prompt_tokens=16, generated_tokens=0, error="oom")
    collector.start_request("req-active", prompt_tokens=8)
    collector.record_scheduler_metrics(
        num_waiting=2,
        num_running=1,
        num_scheduled_tokens=48,
        num_preempted=2,
        batch_size=3,
        schedule_time=0.0015,
    )
    collector.record_runner_metrics(execution_time=0.02, batch_size=3, num_tokens=48)
    collector.record_cache_metrics(total_pages=100, used_pages=40, cache_hit_rate=0.75)

    # Poll 0: observes all newly recorded entries.
    scalars_0 = exporter.update_from_metrics_collector(collector)
    assert exporter.step == 1
    assert scalars_0["esurge/system/total_requests_completed"] == 1.0
    assert scalars_0["esurge/system/total_requests_failed"] == 1.0
    assert scalars_0["esurge/system/total_tokens_generated"] == 12.0
    assert scalars_0["esurge/system/active_requests"] == 1.0
    assert scalars_0["esurge/system/queue_size"] == 2.0
    assert scalars_0["esurge/system/running_requests"] == 1.0
    assert scalars_0["esurge/system/cache_hit_rate"] == pytest.approx(0.75)
    assert scalars_0["esurge/system/cache_utilization"] == pytest.approx(0.40)
    assert scalars_0["esurge/system/tokens_per_second"] == pytest.approx(48.0 / 0.02)
    assert scalars_0["esurge/system/requests_per_second"] > 0.0
    assert scalars_0["esurge/system/average_latency_s"] > 0.0
    assert scalars_0["esurge/system/average_ttft_s"] > 0.0
    assert scalars_0["esurge/system/average_throughput_tok_s"] > 0.0

    assert scalars_0["esurge/scheduler/waiting_requests"] == 2.0
    assert scalars_0["esurge/scheduler/running_requests"] == 1.0
    assert scalars_0["esurge/scheduler/scheduled_tokens"] == 48.0
    assert scalars_0["esurge/scheduler/batch_size"] == 3.0
    assert scalars_0["esurge/scheduler/schedule_time_s"] == pytest.approx(0.0015)
    assert scalars_0["esurge/scheduler/preempted_requests_total"] == 2.0

    assert scalars_0["esurge/runner/execution_time_s"] == pytest.approx(0.02)
    assert scalars_0["esurge/runner/batch_size"] == 3.0
    assert scalars_0["esurge/runner/num_tokens"] == 48.0
    assert scalars_0["esurge/runner/tokens_per_second"] == pytest.approx(2400.0)

    assert scalars_0["esurge/cache/total_pages"] == 100.0
    assert scalars_0["esurge/cache/used_pages"] == 40.0
    assert scalars_0["esurge/cache/free_pages"] == 60.0
    assert scalars_0["esurge/cache/utilization"] == pytest.approx(0.40)
    assert scalars_0["esurge/cache/cache_hit_rate"] == pytest.approx(0.75)

    # Poll 1 with no new events: must NOT double-count preemptions or histograms.
    scalars_1 = exporter.update_from_metrics_collector(collector)
    assert exporter.step == 2
    assert scalars_1["esurge/scheduler/preempted_requests_total"] == 2.0

    # Add one more completed request, scheduler entry, and runner entry -> Poll 2.
    _complete_request(collector, "req-3", prompt_tokens=10, generated_tokens=6)
    collector.record_scheduler_metrics(
        num_waiting=0,
        num_running=1,
        num_scheduled_tokens=16,
        num_preempted=1,
        batch_size=1,
        schedule_time=0.0025,
    )
    collector.record_runner_metrics(execution_time=0.01, batch_size=1, num_tokens=16)

    scalars_2 = exporter.update_from_metrics_collector(collector)
    assert exporter.step == 3
    assert scalars_2["esurge/system/total_requests_completed"] == 2.0
    assert scalars_2["esurge/system/total_tokens_generated"] == 18.0
    assert scalars_2["esurge/scheduler/preempted_requests_total"] == 3.0

    run_id = exporter.run_id
    assert run_id is not None
    assert exporter.store_root == store_dir.resolve().as_uri()

    # Close is idempotent and writes summary + terminal finish record.
    exporter.close(status="finished")
    exporter.close(status="finished")
    assert exporter.closed is True
    assert exporter.run.finished is True
    assert exporter.update_from_metrics_collector(collector) == {}

    meta_records = _read_stream(store_dir, run_id, STREAM_META)
    scalar_records = _read_stream(store_dir, run_id, STREAM_SCALARS)
    hist_records = _read_stream(store_dir, run_id, STREAM_HIST)

    meta_by_op = {rec["op"]: rec for rec in meta_records}
    assert META_START in meta_by_op
    assert meta_by_op[META_START]["name"] == "esurge-unit-run"
    assert meta_by_op[META_START]["group"] == "unit-sweep"
    assert {"easydel", "jax", "esurge", "inference", "EasyDeL", "eSurge", "TestTag"} <= set(
        meta_by_op[META_START]["tags"]
    )
    assert len(set(meta_by_op[META_START]["tags"])) == len(meta_by_op[META_START]["tags"])

    assert META_CONFIG in meta_by_op
    cfg = meta_by_op[META_CONFIG]["config"]
    assert isinstance(cfg, dict)
    assert cfg["max_model_len"] == 4096
    assert cfg["tensor_parallel"] == 2
    assert "python_version" in cfg

    assert META_SUMMARY in meta_by_op
    summary = meta_by_op[META_SUMMARY]["summary"]
    assert isinstance(summary, dict)
    assert summary["total_requests_completed"] == 2
    assert summary["total_requests_failed"] == 1
    assert summary["total_tokens_generated"] == 18
    assert summary["preempted_requests_total"] == 3
    assert summary["polls"] == 3

    assert META_FINISH in meta_by_op
    assert meta_by_op[META_FINISH]["status"] == "finished"

    # Scalars stream: 3 monotonic steps with elapsed_s axis.
    assert [rec["step"] for rec in scalar_records] == [0, 1, 2]
    for rec in scalar_records:
        assert "elapsed_s" in rec["x"]
        assert rec["x"]["elapsed_s"] >= 0.0

    # Histogram stream: entries ONLY on step 0 and step 2 (zero on idle step 1).
    hist_steps = {rec["step"] for rec in hist_records}
    assert hist_steps == {0, 2}

    step0_hists = {rec["key"]: rec for rec in hist_records if rec["step"] == 0}
    assert set(step0_hists) == {
        "esurge/hist/request_duration_s",
        "esurge/hist/ttft_s",
        "esurge/hist/tokens_per_second",
        "esurge/hist/schedule_duration_s",
        "esurge/hist/model_execution_duration_s",
    }
    assert step0_hists["esurge/hist/request_duration_s"]["n"] == 2  # req-1 + req-2
    assert step0_hists["esurge/hist/ttft_s"]["n"] == 1  # only req-1 had first token
    assert step0_hists["esurge/hist/tokens_per_second"]["n"] == 1
    assert step0_hists["esurge/hist/schedule_duration_s"]["n"] == 1
    assert step0_hists["esurge/hist/model_execution_duration_s"]["n"] == 1

    step2_hists = {rec["key"]: rec for rec in hist_records if rec["step"] == 2}
    assert step2_hists["esurge/hist/request_duration_s"]["n"] == 1
    assert step2_hists["esurge/hist/ttft_s"]["n"] == 1
    assert step2_hists["esurge/hist/tokens_per_second"]["n"] == 1
    assert step2_hists["esurge/hist/schedule_duration_s"]["n"] == 1
    assert step2_hists["esurge/hist/model_execution_duration_s"]["n"] == 1


def test_eyvan_exporter_bounded_history_wraparound_external_run_and_sanitization(tmp_path):
    # 1. Bounded deque wraparound must never double-count preemptions.
    store_dir = tmp_path / "wrap-store"
    collector = MetricsCollector(history_size=1)
    exporter = EyvanMetricsExporter(
        name="wrap-check",
        uri=str(store_dir),
        prefix="custom prefix!!",
        tags=("tag-a", " ", "tag-a", "x" * 100),
    )
    try:
        for _ in range(50):
            for _ in range(2):
                collector.record_scheduler_metrics(
                    num_waiting=0,
                    num_running=1,
                    num_scheduled_tokens=1,
                    num_preempted=1,
                    schedule_time=0.001,
                )
            s1 = exporter.update_from_metrics_collector(collector)
            s2 = exporter.update_from_metrics_collector(collector)
            key = "custom_prefix/scheduler/preempted_requests_total"
            assert s1[key] == s2[key]
        assert s2["custom_prefix/scheduler/preempted_requests_total"] == 50.0
    finally:
        exporter.close()

    # 2. Pre-built external run is NOT finished by exporter.close().
    ext_store = tmp_path / "external-store"
    ext_run = eyvan_track.start(name="external-run", uri=str(ext_store), flush_interval_s=0.1)
    try:
        ext_collector = MetricsCollector()
        _complete_request(ext_collector, "r1", generated_tokens=5)
        ext_exporter = EyvanMetricsExporter(run=ext_run, config={"attached": True})
        assert ext_exporter.owns_run is False
        ext_exporter.update_from_metrics_collector(ext_collector)
        ext_exporter.close()
        assert ext_exporter.closed is True
        assert ext_run.finished is False
        # Caller can still log and finish the run cleanly.
        ext_run.log({"caller/metric": 1.0}, step=10)
    finally:
        ext_run.finish("finished")
        if ext_run._thread is not None:
            ext_run._thread.join(timeout=2.0)

    meta = _read_stream(ext_store, ext_run.id, STREAM_META)
    ops = [rec["op"] for rec in meta]
    assert META_SUMMARY in ops
    assert META_FINISH in ops


def test_eyvan_exporter_ranks_zero_inactive_on_nonzero_rank(tmp_path, monkeypatch):
    # 1. Explicit EYVAN_DISTRIBUTED_PROCESS_ID > 0 keeps run inactive and skips polling thread.
    store_env = tmp_path / "rank-env"
    monkeypatch.setenv("EYVAN_DISTRIBUTED_PROCESS_ID", "2")
    baseline = set(threading.enumerate())
    exporter = start_eyvan_tracking(name="rank2", uri=str(store_env), update_interval=0.05)
    try:
        assert exporter.active is False
        assert exporter.run.rank == 2
        new_threads = [t for t in threading.enumerate() if t not in baseline and t.is_alive()]
        assert new_threads == []
    finally:
        monitoring.stop_monitoring()
    assert not list(store_env.glob("**/*.ndjson.gz"))

    # 2. Without env vars (or with blank/whitespace env vars), jax.process_index() > 0 keeps run inactive
    # without mutating os.environ.
    monkeypatch.setenv("EYVAN_DISTRIBUTED_PROCESS_ID", "   ")
    monkeypatch.setenv("JAX_PROCESS_ID", "")
    import jax

    monkeypatch.setattr(jax, "process_index", lambda: 3)
    store_jax = tmp_path / "rank-jax"
    exporter2 = EyvanMetricsExporter(name="rank3", uri=str(store_jax))
    try:
        assert exporter2.active is False
        assert exporter2.run.rank == 3
        assert os.environ["JAX_PROCESS_ID"] == ""
    finally:
        exporter2.close()
    assert not list(store_jax.glob("**/*.ndjson.gz"))


def test_monitoring_server_loop_isolates_exporter_exceptions_without_busy_spin(tmp_path):
    collector = initialize_metrics()
    _complete_request(collector, "iso-req-1", generated_tokens=6)

    store_dir = tmp_path / "server-isolated"
    baseline = set(threading.enumerate())
    exporter = EyvanMetricsExporter(name="server-isolated", uri=str(store_dir), flush_interval_s=0.1)

    prom_calls = 0

    class _FailingPrometheus:
        def update_from_metrics_collector(self, _collector):
            nonlocal prom_calls
            prom_calls += 1
            raise RuntimeError("prometheus boom")

        def close(self):
            return None

    server = monitoring.eSurgeMonitoringServer(
        enable_prometheus=False,
        update_interval=0.08,
        eyvan_exporter=exporter,
        eyvan_update_interval=0.05,
    )
    server.enable_prometheus = True
    server.prometheus_metrics = _FailingPrometheus()
    # Start only the updater loop (skip HTTP bind) with a failing Prometheus exporter.
    server.running = True
    server._update_stop_event = threading.Event()
    server._update_thread = threading.Thread(
        target=server._update_metrics_loop,
        args=(server._update_stop_event,),
        daemon=True,
    )
    server._update_thread.start()
    try:
        assert _wait_until(lambda: exporter.step >= 3, timeout=2.0)
        time.sleep(0.25)
    finally:
        server.stop()

    # Eyvan kept progressing despite Prometheus raising, and Prometheus stayed paced (no busy-spin).
    assert exporter.closed is True
    assert prom_calls >= 2
    assert prom_calls < 25
    assert [t for t in threading.enumerate() if t not in baseline and t.is_alive()] == []


def test_monitoring_server_eyvan_only_without_prometheus_and_dual_mode(tmp_path, monkeypatch):
    collector = initialize_metrics()
    _complete_request(collector, "srv-req-1", generated_tokens=10)
    collector.record_scheduler_metrics(num_waiting=1, num_running=1, num_scheduled_tokens=10, schedule_time=0.001)
    collector.record_runner_metrics(execution_time=0.01, batch_size=1, num_tokens=10)

    # 1. Eyvan-only mode works even when prometheus_client is unavailable.
    store_only = tmp_path / "server-eyvan-only"
    baseline = set(threading.enumerate())
    with monkeypatch.context() as patch:
        patch.setattr(monitoring, "PROMETHEUS_AVAILABLE", False)
        exporter = EyvanMetricsExporter(name="server-only", uri=str(store_only), flush_interval_s=0.1)
        server = monitoring.eSurgeMonitoringServer(
            enable_prometheus=False,
            eyvan_exporter=exporter,
            eyvan_update_interval=0.05,
        )
        server.start()
        try:
            assert server.prometheus_metrics is None
            assert _wait_until(lambda: exporter.step >= 2)
        finally:
            server.stop()

    assert exporter.closed is True
    assert exporter.run.finished is True
    assert [t for t in threading.enumerate() if t not in baseline and t.is_alive()] == []
    scalars = _read_stream(store_only, exporter.run_id, STREAM_SCALARS)
    assert len(scalars) >= 2

    # 2. Dual mode (Prometheus + Eyvan with separate intervals).
    if monitoring.PROMETHEUS_AVAILABLE:
        store_dual = tmp_path / "server-dual"
        baseline_dual = set(threading.enumerate())
        exporter_dual = EyvanMetricsExporter(name="server-dual", uri=str(store_dual), flush_interval_s=0.1)
        server_dual = monitoring.eSurgeMonitoringServer(
            prometheus_port=_free_port(),
            update_interval=0.04,
            enable_prometheus=True,
            eyvan_exporter=exporter_dual,
            eyvan_update_interval=0.15,
        )
        server_dual.start()
        try:
            assert server_dual.prometheus_metrics is not None
            assert _wait_until(lambda: exporter_dual.step >= 2)
        finally:
            server_dual.stop()
        assert exporter_dual.closed is True
        assert [t for t in threading.enumerate() if t not in baseline_dual and t.is_alive()] == []


def test_monitoring_stack_eyvan_only_lifecycle_and_zero_leaked_threads(tmp_path):
    store_dir = tmp_path / "stack-store"
    baseline = set(threading.enumerate())

    stack = MonitoringStack(info=lambda *args, **kwargs: None, model_name="Qwen/Qwen3-8B")
    urls = stack.start_monitoring(
        enable_prometheus=False,
        start_grafana=False,
        enable_console=False,
        enable_eyvan_track=True,
        eyvan_track_uri=str(store_dir),
        eyvan_track_group="serving-eval",
        eyvan_track_tags=("EasyDeL", "eSurge", "v6e"),
        eyvan_track_interval=0.05,
        eyvan_track_flush_interval_s=0.1,
    )
    assert "prometheus" not in urls
    assert "grafana" not in urls
    assert urls["eyvan_track"] == store_dir.resolve().as_uri()
    run_id = urls["eyvan_run_id"]
    assert run_id
    assert stack.monitoring_active is True
    assert stack.has_active_resources is True

    collector = initialize_metrics()
    _complete_request(collector, "stack-req-1", generated_tokens=14)
    collector.record_scheduler_metrics(num_waiting=0, num_running=1, num_scheduled_tokens=14, schedule_time=0.002)
    collector.record_runner_metrics(execution_time=0.015, batch_size=1, num_tokens=14)
    collector.record_cache_metrics(total_pages=64, used_pages=16, cache_hit_rate=0.5)

    assert _wait_until(lambda: stack._eyvan_exporter is not None and stack._eyvan_exporter.step >= 2)

    stack.stop_monitoring()
    assert stack.monitoring_active is False
    assert stack.has_active_resources is False
    assert stack.monitoring_urls is None
    assert [t for t in threading.enumerate() if t not in baseline and t.is_alive()] == []

    meta = {rec["op"]: rec for rec in _read_stream(store_dir, run_id, STREAM_META)}
    assert meta[META_START]["name"].startswith("qwen3-8b-esurge-")
    assert len(meta[META_START]["name"]) <= 64
    assert {"easydel", "jax", "esurge", "inference", "qwen3-8b", "EasyDeL", "eSurge", "v6e"} <= set(
        meta[META_START]["tags"]
    )
    assert meta[META_START]["group"] == "serving-eval"
    assert meta[META_CONFIG]["config"]["model_name"] == "Qwen/Qwen3-8B"
    assert meta[META_SUMMARY]["summary"]["total_requests_completed"] == 1
    assert meta[META_SUMMARY]["summary"]["total_tokens_generated"] == 14
    assert meta[META_FINISH]["status"] == "finished"

    scalars = _read_stream(store_dir, run_id, STREAM_SCALARS)
    assert len(scalars) >= 2
    assert any(rec["v"]["esurge/system/total_requests_completed"] == 1.0 for rec in scalars)

    # Also verify stack.close() cleanly flushes and closes.
    store_close = tmp_path / "stack-close"
    stack2 = MonitoringStack(info=lambda *args, **kwargs: None, model_name="Llama-Tiny")
    urls2 = stack2.start_monitoring(
        enable_prometheus=False,
        start_grafana=False,
        enable_eyvan_track=True,
        eyvan_track_uri=str(store_close),
        eyvan_track_interval=0.05,
        eyvan_track_flush_interval_s=0.1,
    )
    run_id2 = urls2["eyvan_run_id"]
    stack2.close()
    assert stack2.has_active_resources is False
    assert [t for t in threading.enumerate() if t not in baseline and t.is_alive()] == []
    meta2 = {rec["op"]: rec for rec in _read_stream(store_close, run_id2, STREAM_META)}
    assert meta2[META_FINISH]["status"] == "finished"


def test_monitoring_stack_auto_activation_explicit_disable_and_unavailable_fallback(tmp_path, monkeypatch):
    # 1. Auto-activation via EYVAN_TRACKING_URI when enable_eyvan_track=None.
    env_store = tmp_path / "env-auto-store"
    monkeypatch.setenv("EYVAN_TRACKING_URI", str(env_store))
    stack_auto = MonitoringStack(info=lambda *args, **kwargs: None, model_name="AutoModel")
    try:
        urls = stack_auto.start_monitoring(
            enable_prometheus=False,
            start_grafana=False,
            eyvan_track_interval=0.05,
            eyvan_track_flush_interval_s=0.1,
        )
        assert urls["eyvan_track"] == env_store.resolve().as_uri()
        assert "eyvan_run_id" in urls
    finally:
        stack_auto.stop_monitoring()

    # 2. Explicit enable_eyvan_track=False overrides EYVAN_TRACKING_URI, EYVAN_WORKLOAD_ID, and eyvan_track_uri.
    disabled_store = tmp_path / "disabled-store"
    monkeypatch.setenv("EYVAN_WORKLOAD_ID", "wl-disabled-check")
    stack_off = MonitoringStack(info=lambda *args, **kwargs: None)
    try:
        urls_off = stack_off.start_monitoring(
            enable_prometheus=False,
            start_grafana=False,
            enable_eyvan_track=False,
            eyvan_track_uri=str(disabled_store),
        )
        assert urls_off == {}
        assert stack_off._eyvan_exporter is None
    finally:
        stack_off.stop_monitoring()
    assert not disabled_store.exists()

    # 3. Auto-activation via explicit eyvan_track_uri when env vars are unset.
    monkeypatch.delenv("EYVAN_TRACKING_URI", raising=False)
    monkeypatch.delenv("EYVAN_WORKLOAD_ID", raising=False)
    uri_store = tmp_path / "uri-auto-store"
    stack_uri = MonitoringStack(info=lambda *args, **kwargs: None)
    try:
        urls_uri = stack_uri.start_monitoring(
            enable_prometheus=False,
            start_grafana=False,
            eyvan_track_uri=str(uri_store),
            eyvan_track_interval=0.05,
            eyvan_track_flush_interval_s=0.1,
        )
        assert urls_uri["eyvan_track"] == uri_store.resolve().as_uri()
        assert "eyvan_run_id" in urls_uri
    finally:
        stack_uri.stop_monitoring()

    # 4. Under pytest isolation, default (no URI, no env vars, enable_eyvan_track=None) stays hermetically off;
    #    when auto-track is enabled (normal non-pytest runtime), enable_eyvan_track=None auto-enables cleanly!
    assert is_eyvan_auto_track_enabled() is False
    stack_default = MonitoringStack(info=lambda *args, **kwargs: None)
    try:
        urls_default = stack_default.start_monitoring(enable_prometheus=False, start_grafana=False)
        assert urls_default == {}
        assert stack_default._eyvan_exporter is None
    finally:
        stack_default.stop_monitoring()

    auto_home = tmp_path / "esurge-auto-on-home"
    monkeypatch.setenv("HOME", str(auto_home))
    from easydel.trainers import training_configurations as tc

    monkeypatch.setattr(tc, "_import_eyvan_storage", lambda: None)
    with set_eyvan_auto_track_enabled(True):
        stack_auto_on = MonitoringStack(info=lambda *args, **kwargs: None, model_name="Qwen/Qwen3-8B")
        try:
            urls_auto_on = stack_auto_on.start_monitoring(
                enable_prometheus=False,
                start_grafana=False,
                eyvan_track_interval=0.05,
                eyvan_track_flush_interval_s=0.1,
            )
            expected_store = auto_home / ".eyvan" / "tracking" / "default" / "esurge"
            assert urls_auto_on["eyvan_track"] == expected_store.resolve().as_uri()
            auto_run_id = urls_auto_on["eyvan_run_id"]
            assert auto_run_id
        finally:
            stack_auto_on.stop_monitoring()
        assert _read_stream(expected_store, auto_run_id, STREAM_META)[-1]["op"] == META_FINISH

        # Explicit enable_eyvan_track=False disables cleanly even when auto-track is enabled.
        stack_explicit_off = MonitoringStack(info=lambda *args, **kwargs: None)
        try:
            assert (
                stack_explicit_off.start_monitoring(
                    enable_prometheus=False,
                    start_grafana=False,
                    enable_eyvan_track=False,
                )
                == {}
            )
            assert stack_explicit_off._eyvan_exporter is None
        finally:
            stack_explicit_off.stop_monitoring()

    # 5. Graceful fallback when eyvan.track is unavailable.
    monkeypatch.setattr(monitoring, "EYVAN_TRACK_AVAILABLE", False)
    monkeypatch.setattr(monitoring, "_eyvan_track", None)
    with pytest.raises(ImportError, match=r"eyvan\.track"):
        EyvanMetricsExporter()
    stack_missing = MonitoringStack(info=lambda *args, **kwargs: None)
    try:
        urls_missing = stack_missing.start_monitoring(
            enable_prometheus=False,
            start_grafana=False,
            enable_eyvan_track=True,
        )
        assert "eyvan_track" not in urls_missing
        assert stack_missing._eyvan_exporter is None
    finally:
        stack_missing.stop_monitoring()


def test_owner_isolation_weakref_gc_and_esurge_terminate_cleanup(tmp_path):
    baseline = set(threading.enumerate())
    initialize_metrics()

    # 1. Two independent stacks do not interfere with each other's Eyvan runs.
    store_a = tmp_path / "stack-a"
    store_b = tmp_path / "stack-b"
    stack_a = MonitoringStack(info=lambda *a, **kw: None, model_name="ModelA")
    stack_b = MonitoringStack(info=lambda *a, **kw: None, model_name="ModelB")
    try:
        urls_a = stack_a.start_monitoring(
            enable_prometheus=False,
            start_grafana=False,
            enable_eyvan_track=True,
            eyvan_track_uri=str(store_a),
            eyvan_track_interval=0.05,
            eyvan_track_flush_interval_s=0.1,
        )
        urls_b = stack_b.start_monitoring(
            enable_prometheus=False,
            start_grafana=False,
            enable_eyvan_track=True,
            eyvan_track_uri=str(store_b),
            eyvan_track_interval=0.05,
            eyvan_track_flush_interval_s=0.1,
        )
        exp_a = stack_a._eyvan_exporter
        exp_b = stack_b._eyvan_exporter
        assert exp_a is not exp_b

        stack_a.stop_monitoring()
        assert exp_a.closed is True
        assert exp_b.closed is False

        step_b_before = exp_b.step
        assert _wait_until(lambda: exp_b.step > step_b_before)
        stack_b.stop_monitoring()
        assert exp_b.closed is True
    finally:
        stack_a.stop_monitoring()
        stack_b.stop_monitoring()

    assert _read_stream(store_a, urls_a["eyvan_run_id"], STREAM_META)[-1]["op"] == META_FINISH
    assert _read_stream(store_b, urls_b["eyvan_run_id"], STREAM_META)[-1]["op"] == META_FINISH

    # 2. Abandoned stack is not pinned in globals and auto-closes its Eyvan run on GC.
    store_gc = tmp_path / "stack-gc"
    gc_stack = MonitoringStack(info=lambda *a, **kw: None, model_name="GCModel")
    gc_urls = gc_stack.start_monitoring(
        enable_prometheus=False,
        start_grafana=False,
        enable_eyvan_track=True,
        eyvan_track_uri=str(store_gc),
        eyvan_track_interval=0.05,
        eyvan_track_flush_interval_s=0.1,
    )
    gc_run_id = gc_urls["eyvan_run_id"]
    gc_ref = weakref.ref(gc_stack)
    del gc_stack
    gc.collect()
    assert gc_ref() is None
    assert _read_stream(store_gc, gc_run_id, STREAM_META)[-1]["op"] == META_FINISH

    # 3. eSurge.terminate() stops & flushes active Eyvan tracking (while pause's stop_monitoring=False preserves it).
    class _TerminateStub(eSurge):
        def __init__(self):
            self._info = lambda *a, **kw: None
            self._monitoring = MonitoringStack(info=self._info, model_name="StubModel")
            self._scheduler_lock = threading.RLock()
            self._scheduler_running = False
            self._engine_loop = None

        def _stop_idle_monitor(self) -> None:
            return None

        def _stop_worker_replay_loop(self) -> bool:
            return False

        def _drop_heap_freeze(self) -> None:
            return None

        def _stop_engine_output_worker(self) -> None:
            return None

    store_term = tmp_path / "engine-terminate"
    stub = _TerminateStub()
    term_urls = stub.start_monitoring(
        enable_prometheus=False,
        start_grafana=False,
        enable_eyvan_track=True,
        eyvan_track_uri=str(store_term),
        eyvan_track_interval=0.05,
        eyvan_track_flush_interval_s=0.1,
    )
    term_run_id = term_urls["eyvan_run_id"]
    # Pause path preserves monitoring.
    stub.terminate(stop_monitoring=False)
    assert stub._monitoring.has_active_resources is True
    # Normal terminate stops & flushes monitoring cleanly without holding _scheduler_lock.
    lock_free_during_stop: list[bool] = []
    orig_stop = stub.stop_monitoring

    def _check_lock_then_stop() -> None:
        result: list[bool] = []

        def _probe() -> None:
            acquired = stub._scheduler_lock.acquire(blocking=False)
            if acquired:
                stub._scheduler_lock.release()
            result.append(acquired)

        probe_thread = threading.Thread(target=_probe)
        probe_thread.start()
        probe_thread.join(timeout=2.0)
        lock_free_during_stop.extend(result)
        orig_stop()

    stub.stop_monitoring = _check_lock_then_stop
    stub.terminate()
    assert lock_free_during_stop == [True]
    assert stub._monitoring.has_active_resources is False
    meta_term = {rec["op"]: rec for rec in _read_stream(store_term, term_run_id, STREAM_META)}
    assert meta_term[META_FINISH]["status"] == "finished"

    assert [t for t in threading.enumerate() if t not in baseline and t.is_alive()] == []


def test_monitoring_stack_guard4_unwritable_store_warns_once_and_noops(monkeypatch, tmp_path):
    from easydel.trainers import training_configurations as tc

    baseline = set(threading.enumerate())
    monkeypatch.setattr(tc, "_import_eyvan_storage", lambda: None)

    # 1. Failing eyvan.track.start emits ONE UserWarning, marks _eyvan_init_failed, and never retries
    #    on subsequent start_monitoring() calls until stop_monitoring() resets the stack.
    start_calls = 0

    def _failing_start(**_kwargs):
        nonlocal start_calls
        start_calls += 1
        raise PermissionError("unwritable esurge store")

    fake_track = type("_FakeTrack", (), {"start": staticmethod(_failing_start)})()
    with monkeypatch.context() as patch:
        patch.setattr(monitoring, "_eyvan_track", fake_track)
        stack = MonitoringStack(info=lambda *a, **kw: None, model_name="Qwen/Qwen3-8B")
        try:
            with pytest.warns(UserWarning, match="PermissionError"):
                urls_1 = stack.start_monitoring(
                    enable_prometheus=False,
                    start_grafana=False,
                    enable_eyvan_track=True,
                    eyvan_track_uri=str(tmp_path / "ignored-store"),
                )
            assert urls_1 == {}
            assert stack._eyvan_init_failed is True
            assert start_calls == 1

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                urls_2 = stack.start_monitoring(
                    enable_prometheus=False,
                    start_grafana=False,
                    enable_eyvan_track=True,
                    eyvan_track_uri=str(tmp_path / "ignored-store"),
                )
            assert urls_2 == {}
            assert start_calls == 1
            assert caught == []
        finally:
            stack.stop_monitoring()
        assert stack._eyvan_init_failed is False

    # 2. Real read-only HOME directory (non-root): fails fast before spawning writer/poller threads,
    #    warns once, and stays hermetic.
    if hasattr(os, "geteuid") and os.geteuid() != 0:
        ro_home = tmp_path / "esurge_ro_home"
        ro_home.mkdir()
        ro_home.chmod(0o555)
        try:
            monkeypatch.setenv("HOME", str(ro_home))
            with set_eyvan_auto_track_enabled(True):
                stack_ro = MonitoringStack(info=lambda *a, **kw: None, model_name="Qwen/Qwen3-8B")
                try:
                    with pytest.warns(UserWarning, match="PermissionError|OSError"):
                        urls_ro = stack_ro.start_monitoring(enable_prometheus=False, start_grafana=False)
                    assert urls_ro == {}
                    assert stack_ro._eyvan_init_failed is True
                    assert [t for t in threading.enumerate() if t not in baseline and t.is_alive()] == []
                    with warnings.catch_warnings(record=True) as caught_ro:
                        warnings.simplefilter("always")
                        assert stack_ro.start_monitoring(enable_prometheus=False, start_grafana=False) == {}
                    assert caught_ro == []
                finally:
                    stack_ro.stop_monitoring()
        finally:
            ro_home.chmod(0o755)
    assert [t for t in threading.enumerate() if t not in baseline and t.is_alive()] == []
