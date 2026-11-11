"""Public usage payloads include prefix-cache and uncached-input accounting."""

from threading import RLock
from types import SimpleNamespace

import pytest
from easydel.inference.esurge.engine.output_pipeline import _cache_usage_metrics
from easydel.inference.esurge.server.api_server import eSurgeApiServer
from easydel.inference.openai_api_modules import UsageInfo

pytestmark = pytest.mark.cpu_ok


def test_cache_metrics_cover_hit_read_write_and_uncached_input():
    metrics = _cache_usage_metrics(num_prompt_tokens=256, cached_tokens=64, prefix_caching_enabled=True)
    assert metrics == {"cache_hit": True, "cache_read": 64, "cache_written": 192, "uncached_input": 192}


def test_cache_metrics_distinguish_disabled_and_unknown_prefix_caching():
    disabled = _cache_usage_metrics(100, 0, False)
    assert disabled == {"cache_hit": False, "cache_read": 0, "cache_written": 0, "uncached_input": 100}
    unknown = _cache_usage_metrics(100, 25, None)
    assert unknown["cache_written"] is None
    assert unknown["uncached_input"] == 75


def test_usage_info_aggregates_cache_fields_and_preserves_unknown_writes():
    outputs = [
        SimpleNamespace(metrics=_cache_usage_metrics(100, 25, True)),
        SimpleNamespace(metrics=_cache_usage_metrics(200, 50, True)),
    ]
    usage = UsageInfo(**UsageInfo.cache_fields(outputs))
    assert usage.cache_hit is True
    assert usage.cache_read == 75
    assert usage.cache_written == 225
    assert usage.uncached_input == 225

    outputs.append(SimpleNamespace(metrics=_cache_usage_metrics(50, 0, None)))
    usage = UsageInfo(**UsageInfo.cache_fields(outputs))
    assert usage.cache_written is None
    assert usage.uncached_input == 275


def test_engine_metrics_snapshot_summarizes_recent_cache_and_prefill_decode():
    outputs = []
    for index, cached in enumerate((0, 64)):
        metrics = _cache_usage_metrics(128, cached, True)
        metrics.update(
            prompt_tokens=128,
            processing_time=2.0,
            first_token_time=0.5,
            tokens_per_second=32.0,
        )
        outputs.append(
            SimpleNamespace(
                request_id=f"r{index}",
                metrics=metrics,
                finished=True,
                num_generated_tokens=32,
                first_token_time=0.5,
                tokens_per_second=32.0,
                processing_time=2.0,
                outputs=[SimpleNamespace(finish_reason="stop")],
            )
        )
    engine = SimpleNamespace(
        _registry=SimpleNamespace(output_lock=RLock(), outputs={"r0": outputs[0], "r1": outputs[1]}),
        num_running_requests=2,
        num_pending_requests=3,
    )
    server = SimpleNamespace(esurge_map={"test-model": engine})
    snapshot = eSurgeApiServer._engine_metrics_snapshot(server)["test-model"]
    assert snapshot["running_requests"] == 2
    assert snapshot["waiting_requests"] == 3
    aggregate = snapshot["aggregate"]
    assert aggregate["completed"] == 2
    assert aggregate["cache_hits"] == 1
    assert aggregate["cache_read"] == 64
    assert aggregate["cache_written"] == 192
    assert aggregate["uncached_input"] == 192
    assert aggregate["prefill_tps"] == 192.0
    assert aggregate["decode_tps"] == pytest.approx(64.0 / 3.0)
    assert snapshot["recent_requests"][0]["cache_hit"] is True
