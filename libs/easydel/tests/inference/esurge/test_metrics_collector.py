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

"""Regression tests for ``MetricsCollector`` locking, averaging and export."""

from __future__ import annotations

import json
import threading

import pytest
from easydel.inference.esurge.metrics import MetricsCollector

pytestmark = pytest.mark.cpu_ok


def _run_with_timeout(fn, timeout: float = 5.0):
    """Run ``fn`` on a thread; fail the test if it does not return (deadlock)."""
    result: dict = {}

    def _target():
        try:
            result["value"] = fn()
        except BaseException as exc:  # pragma: no cover - surfaced below
            result["error"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), "call deadlocked on the collector lock"
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _complete(collector: MetricsCollector, rid: str, tokens: int) -> None:
    collector.start_request(rid, prompt_tokens=4)
    collector.record_first_token(rid)
    collector.add_generated_tokens(rid, tokens)
    collector.complete_request(rid, finish_reason="stop")


def test_log_summary_force_does_not_deadlock(caplog):
    collector = MetricsCollector(log_interval=0.0)
    _complete(collector, "r1", 8)
    _run_with_timeout(lambda: collector.log_summary(force=True))
    assert collector.last_log_time > 0


def test_export_metrics_does_not_deadlock_and_includes_cache_events(tmp_path):
    collector = MetricsCollector()
    _complete(collector, "r1", 8)
    collector.record_cache_event("kv_cache_reinitialized", {"reason": "unit"})
    out = tmp_path / "metrics.json"
    _run_with_timeout(lambda: collector.export_metrics(str(out)))

    data = json.loads(out.read_text())
    assert data["system_metrics"]["total_requests_completed"] == 1
    assert data["counters"]["cache_event_kv_cache_reinitialized"] == 1
    assert [e["event"] for e in data["cache_events"]] == ["kv_cache_reinitialized"]
    assert data["cache_events"][0]["details"] == {"reason": "unit"}


def test_export_metrics_rejects_unknown_format(tmp_path):
    collector = MetricsCollector()
    with pytest.raises(ValueError, match="Unsupported format"):
        collector.export_metrics(str(tmp_path / "m.txt"), format="csv")


def test_system_metrics_average_is_plain_mean():
    collector = MetricsCollector()
    for i in range(3):
        _complete(collector, f"r{i}", 5)
    metrics = collector.get_system_metrics(window_seconds=60.0)
    assert metrics.total_requests_completed == 3
    assert metrics.total_tokens_generated == 15
    expected = sum(r.total_latency for r in collector.completed_requests) / 3
    assert metrics.average_latency == pytest.approx(expected)
    assert metrics.requests_per_second == pytest.approx(3 / 60.0)


def test_reset_metrics_clears_cache_events():
    collector = MetricsCollector()
    collector.record_cache_event("e")
    collector.reset_metrics()
    assert len(collector.cache_events) == 0
    assert not collector.counters
