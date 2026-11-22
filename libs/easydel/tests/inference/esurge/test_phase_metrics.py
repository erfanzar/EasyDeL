"""Engine-step prefill/decode token split behind ``/metrics`` ``engine_metrics.<model>.phase``."""

import pytest
from easydel.inference.esurge import metrics as metrics_module
from easydel.inference.esurge.metrics import MetricsCollector, split_phase_tokens

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize(
    ("scheduled", "computed", "prompt", "expected"),
    [
        (512, 0, 2048, (512, 0)),  # chunk inside the prompt
        (512, 1792, 2048, (256, 256)),  # last prompt chunk plus generated tokens
        (1, 2048, 2048, (0, 1)),  # plain decode
        (4, 3000, 2048, (0, 4)),  # speculative verify rows are decode
        (76, 256, 332, (76, 0)),  # prefix-cache hit: only the uncached tail is prefill
        (0, 0, 10, (0, 0)),
        (-3, 0, 10, (0, 0)),
    ],
)
def test_split_phase_tokens(scheduled, computed, prompt, expected):
    assert split_phase_tokens(scheduled, computed, prompt) == expected


def test_phase_summary_counts_steps_rates_and_lifetime_totals(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(metrics_module.time, "time", lambda: now[0])
    collector = MetricsCollector()

    collector.record_step_tokens(4096, 0)  # prefill-only
    now[0] += 1.0
    collector.record_step_tokens(1024, 32)  # mixed
    now[0] += 1.0
    collector.record_step_tokens(0, 32)  # decode-only
    collector.record_step_tokens(0, 0)  # empty steps are ignored
    now[0] = 1010.0

    summary = collector.phase_summary(window_seconds=300.0)
    assert summary["prefill_tokens"] == 5120
    assert summary["decode_tokens"] == 64
    # The collector is 10 s old, younger than the window, so rates divide by its age.
    assert summary["prefill_tps"] == pytest.approx(512.0)
    assert summary["decode_tps"] == pytest.approx(6.4)
    assert (summary["prefill_steps"], summary["decode_steps"], summary["mixed_steps"]) == (1, 1, 1)
    assert (summary["prefill_tokens_total"], summary["decode_tokens_total"], summary["steps_total"]) == (5120, 64, 3)

    now[0] = 1000.0 + 301.5  # the first two steps leave the window, the totals stay
    late = collector.phase_summary(window_seconds=300.0)
    assert (late["prefill_tokens"], late["decode_tokens"], late["steps"]) == (0, 32, 1)
    assert late["decode_tps"] == pytest.approx(32 / 300.0)
    assert late["prefill_tokens_total"] == 5120

    collector.reset_metrics()
    cleared = collector.phase_summary(window_seconds=300.0)
    assert (cleared["steps"], cleared["prefill_tokens_total"]) == (0, 0)
