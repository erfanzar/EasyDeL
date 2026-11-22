"""Host-only observable trace/configuration regressions."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.runners import model_runner
from easydel.inference.esurge.runners.spec.strategy import DrafterSpeculation

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("nested", [False, True], ids=["legacy-flat", "nested-runtime"])
def test_public_runtime_options_reach_elarge_engine_kwargs(nested):
    from easydel.inference.esurge.config import eSurgeRuntimeConfig
    from easydel.infra.elarge.builders import normalize_esurge_sections, to_esurge_kwargs

    options = {"spec_debug_max_traces": 3, "spec_adaptive_conf_threshold": 0.7}
    raw = {"runtime": options} if nested else options
    assert normalize_esurge_sections(raw)["runtime"] == options
    runtime = to_esurge_kwargs({"model": {"name_or_path": "host-only-test"}, "esurge": raw})["runtime"]
    assert runtime.spec_debug_max_traces == 3
    assert runtime.spec_adaptive_conf_threshold == 0.7
    defaults = eSurgeRuntimeConfig.coerce_config(None)
    assert defaults.spec_debug_max_traces == 0
    assert defaults.spec_adaptive_conf_threshold == 0.0
    zeroes = runtime.replace(spec_debug_max_traces=0, spec_adaptive_conf_threshold=0.0)
    assert zeroes.spec_debug_max_traces == 0
    assert zeroes.spec_adaptive_conf_threshold == 0.0


def test_explicit_debug_trace_budget_records_and_limits_diagnostics():
    spec = DrafterSpeculation(
        runner=SimpleNamespace(), drafter=SimpleNamespace(), num_draft_tokens=2, debug_max_traces=1
    )
    meta = SimpleNamespace(
        request_id="a",
        row_pos=0,
        req_idx=0,
        start_pos=1,
        real_count=1,
        target_local_indices=[0],
        bonus_local_index=1,
        draft_token_positions=[2],
        scheduled_draft_tokens=[1],
        buffer_draft_tokens=[1],
    )
    for _ in range(2):
        spec.record_verify_trace(
            meta=meta,
            logits_rows=np.asarray([[0.0, 2.0, 1.0]]),
            accepted=1,
            corrected_token=2,
            greedy=True,
            source="test",
        )
    assert len(spec.debug_traces) == 1
    trace = spec.debug_traces[0]
    assert trace["accepted"] == 1
    assert trace["corrected_token"] == 2
    assert trace["rows"][0]["scheduler_buffer_match"]
    assert trace["rows"][0]["target_argmax"] == 1
    assert trace["rows"][0]["draft_rank_under_target"] == 1


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_adaptive_threshold_is_rejected(threshold):
    with pytest.raises(ValueError, match="adaptive_conf_threshold"):
        DrafterSpeculation(
            runner=SimpleNamespace(),
            drafter=SimpleNamespace(),
            num_draft_tokens=2,
            adaptive_conf_threshold=threshold,
        )


def test_prefix_drafter_default_token_budget_initializes_full_hidden_window(monkeypatch):
    class HostRunner(model_runner.eSurgeRunner):
        def _build_kv_cache_groups(self):
            return []

        def _apply_window_aware_runtime_cap(self, max_num_batched_tokens):
            return None

        def _setup_variables(self):
            pass  # Device buffers are deliberately outside this host-only test.

        def _log_startup_summary(self):
            pass

    class HostManager:
        def __init__(self, **kwargs):
            self.full_hidden_state_max_tokens = kwargs["full_hidden_state_max_tokens"]

    text_config = SimpleNamespace(attn_mechanism="ragged_page_attention_v3", layer_types=())
    model = SimpleNamespace(mesh=None, config=SimpleNamespace(get_text_config=lambda: text_config))
    model.esurge_compatible_model = model
    model.create_ragged_page_cache_config = lambda **kwargs: SimpleNamespace(
        num_pages=64,
        page_size=2,
        max_num_pages_per_req=8,
        data_parallel_size=1,
        get_max_num_seqs=lambda: 2,
    )
    monkeypatch.setattr(model_runner, "ExecutionManager", HostManager)
    runner = HostRunner(
        model,
        max_model_len=16,
        max_num_seqs=2,
        min_input_pad=1,
        drafter=SimpleNamespace(num_draft_tokens=2, supports_prefix_draft=True),
        spec_debug_max_traces=3,
        spec_adaptive_conf_threshold=0.7,
    )
    assert runner.max_num_batched_tokens == 16
    assert runner.executor_manager.full_hidden_state_max_tokens == 16
    assert runner.spec.debug_max_traces == 3
    assert runner.spec.adaptive_conf_threshold == 0.7
