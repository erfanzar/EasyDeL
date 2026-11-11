"""Prefix-cache usage in API responses and the per-model metrics summary behind ``/metrics``."""

import time
from types import SimpleNamespace

import pytest
from easydel.inference.esurge.engine.output_pipeline import _cache_usage_metrics
from easydel.inference.esurge.metrics import MetricsCollector
from easydel.inference.esurge.server import api_server as api_server_module
from easydel.inference.esurge.server.api_server import eSurgeApiServer
from easydel.inference.openai_api_modules import UsageInfo

pytestmark = pytest.mark.cpu_ok

PAGE = 128


def test_cache_written_counts_only_whole_prompt_pages_not_already_cached():
    assert _cache_usage_metrics(256, 64, PAGE) == {
        "cache_hit": True,
        "cache_read": 64,
        "cache_written": 192,
        "uncached_input": 192,
    }
    # The live 332-token prompt: cold commits two full pages; warm reads them and commits nothing new.
    assert _cache_usage_metrics(332, 0, PAGE) == {
        "cache_hit": False,
        "cache_read": 0,
        "cache_written": 256,
        "uncached_input": 332,
    }
    assert _cache_usage_metrics(332, 256, PAGE) == {
        "cache_hit": True,
        "cache_read": 256,
        "cache_written": 0,
        "uncached_input": 76,
    }


def test_disabled_prefix_cache_writes_nothing_and_reads_are_clamped_to_the_prompt():
    assert _cache_usage_metrics(100, 0, 0) == {
        "cache_hit": False,
        "cache_read": 0,
        "cache_written": 0,
        "uncached_input": 100,
    }
    clamped = _cache_usage_metrics(100, 150, PAGE)
    assert clamped["cache_read"] == 100
    assert clamped["uncached_input"] == 0
    assert clamped["cache_written"] == 0


def test_usage_info_sums_cache_fields_and_keeps_unknown_writes_unknown():
    outputs = [
        SimpleNamespace(metrics=_cache_usage_metrics(100, 25, PAGE)),
        SimpleNamespace(metrics=_cache_usage_metrics(300, 50, PAGE)),
    ]
    usage = UsageInfo(**UsageInfo.cache_fields(outputs))
    assert usage.cache_hit is True
    assert usage.cache_read == 75
    assert usage.cache_written == 206
    assert usage.uncached_input == 325

    # A remote engine that cannot report its writes makes the aggregate unknown rather than undercounted.
    outputs.append(SimpleNamespace(metrics={"cache_read": 0, "cache_written": None, "uncached_input": 50}))
    usage = UsageInfo(**UsageInfo.cache_fields(outputs))
    assert usage.cache_written is None
    assert usage.uncached_input == 375


def _record(collector, rid, *, model, prompt, cached, tokens, start, ttft, latency, error=None):
    """Run one request through the collector's public lifecycle, then pin its timestamps."""
    collector.start_request(rid, prompt_tokens=prompt, model_name=model)
    collector.record_cache_read(rid, cached)
    if ttft is not None:
        collector.record_first_token(rid)
    collector.add_generated_tokens(rid, tokens)
    collector.complete_request(rid, finish_reason=None if error else "stop", error=error)
    row = collector.completed_requests[-1]
    row.start_time = start
    row.time_to_first_token = ttft
    row.total_latency = latency
    row.end_time = start + latency


def _collector(now: float) -> MetricsCollector:
    collector = MetricsCollector()
    _record(collector, "r0", model="m", prompt=128, cached=0, tokens=32, start=now - 10, ttft=0.5, latency=2.0)
    _record(collector, "r1", model="m", prompt=128, cached=64, tokens=32, start=now - 9, ttft=0.25, latency=1.25)
    _record(collector, "r2", model="other", prompt=64, cached=0, tokens=8, start=now - 9, ttft=0.1, latency=0.5)
    _record(
        collector, "r3", model="m", prompt=16, cached=0, tokens=0, start=now - 10, ttft=None, latency=0.5, error="boom"
    )
    _record(collector, "old", model="m", prompt=999, cached=0, tokens=9, start=now - 1000, ttft=0.1, latency=1.0)
    return collector


def test_model_summary_aggregates_one_model_within_the_window():
    now = time.time()
    summary = _collector(now).model_summary("m", window_seconds=300.0)
    agg = summary["aggregate"]
    assert agg["completed"] == 2
    assert agg["failed"] == 1
    assert agg["prompt_tokens"] == 256
    assert agg["completion_tokens"] == 64
    assert agg["cache_hits"] == 1
    assert agg["cache_hit_rate"] == pytest.approx(0.5)
    assert agg["cache_read_tokens"] == 64
    assert agg["uncached_input_tokens"] == 192
    assert agg["token_cache_hit_rate"] == pytest.approx(0.25)
    # Nearest-rank percentiles over [0.25, 0.5] and [1.25, 2.0].
    assert agg["ttft_p50_s"] == pytest.approx(0.25)
    assert agg["ttft_p99_s"] == pytest.approx(0.5)
    assert agg["latency_p50_s"] == pytest.approx(1.25)
    assert agg["latency_p99_s"] == pytest.approx(2.0)
    # Prefill: 192 uncached tokens over 0.75 s of TTFT; decode: 62 post-first tokens over 2.5 s.
    assert agg["prefill_tps"] == pytest.approx(256.0)
    assert agg["decode_tps"] == pytest.approx(24.8)
    # Output: 64 tokens over the span from the earliest start (now-10) to the latest finish (now-7.75).
    assert agg["output_tps"] == pytest.approx(64 / 2.25)

    recent = summary["recent_requests"]
    assert [row["request_id"] for row in recent] == ["r1", "r0", "r3"]
    assert recent[0]["cache_hit"] is True
    assert recent[0]["uncached_input_tokens"] == 64
    assert recent[0]["decode_tps"] == pytest.approx(31.0)
    assert recent[2]["error"] == "boom"
    assert recent[2]["decode_tps"] is None


def test_model_summary_of_an_idle_model_reports_no_rates():
    agg = MetricsCollector().model_summary("m")["aggregate"]
    assert agg["completed"] == 0
    for field in ("cache_hit_rate", "token_cache_hit_rate", "ttft_p50_s", "prefill_tps", "decode_tps", "output_tps"):
        assert agg[field] is None


def test_metrics_endpoint_snapshot_reads_the_engine_collector(monkeypatch):
    collector = _collector(time.time())
    monkeypatch.setattr(api_server_module, "get_metrics_collector", lambda: collector)
    engine = SimpleNamespace(esurge_name="m", num_running_requests=2, num_pending_requests=3)
    snapshot = eSurgeApiServer._engine_metrics_snapshot(SimpleNamespace(esurge_map={"served-name": engine}))
    model = snapshot["served-name"]
    assert model["running_requests"] == 2
    assert model["waiting_requests"] == 3
    assert model["aggregate"]["completed"] == 2
    assert model["recent_requests"][0]["request_id"] == "r1"

    monkeypatch.setattr(api_server_module, "get_metrics_collector", lambda: None)
    empty = eSurgeApiServer._engine_metrics_snapshot(SimpleNamespace(esurge_map={"served-name": engine}))
    assert empty["served-name"]["recent_requests"] == []
