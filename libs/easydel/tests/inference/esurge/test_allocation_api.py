"""Host-only regressions for the runner's public cache-allocation boundary."""

from types import SimpleNamespace

import pytest
from easydel.inference.esurge.runners import model_runner
from easydel.inference.esurge.runners.pipeline_execution_manager import (
    PipelineExecutionManager,
    RunnerExecutionManager,
)

pytestmark = pytest.mark.cpu_ok


def make_manager(*, pipeline_enabled, fail_attempts=0, error="RESOURCE_EXHAUSTED"):
    """Use real retry code with a host-only allocator instead of device buffers."""
    manager_type = PipelineExecutionManager if pipeline_enabled else RunnerExecutionManager
    manager = manager_type.__new__(manager_type)
    manager.kv_pages = None
    manager.metadata = SimpleNamespace(num_pages=1000, page_size=128, hbm_utilization=0.9, kvdtype=None)
    manager.pipeline_plan = SimpleNamespace(is_enabled=pipeline_enabled)
    manager.max_num_reqs = 4
    manager.max_model_len = 1024
    manager.speculative_recurrent_state_tokens = 2
    manager.speculative_trace_tokens = 3
    attempts = []
    pages = object()

    def allocate(**kwargs):
        attempts.append((manager.metadata.num_pages, kwargs))
        if len(attempts) <= fail_attempts:
            raise RuntimeError(error)
        return pages

    manager.model = SimpleNamespace(init_operations_cache=allocate)
    return manager, attempts, pages


@pytest.mark.parametrize("pipeline_enabled", [False, True], ids=["spmd", "pp"])
def test_public_allocation_is_idempotent_and_preserves_cache_layout(pipeline_enabled):
    manager, attempts, pages = make_manager(pipeline_enabled=pipeline_enabled)
    quantizer, masking = object(), object()
    assert manager.initialize_kv_cache(quantizer=quantizer, masking_details=masking) is pages
    assert manager.initialize_kv_cache(quantizer=object()) is pages
    assert len(attempts) == 1
    _, kwargs = attempts[0]
    assert kwargs["quantizer"] is quantizer
    assert kwargs["masking_details"] is masking
    assert kwargs["batch_size"] == 12  # 4 live rows + 2 speculative candidates per row.
    assert kwargs["spec_trace_len"] == 3
    assert kwargs["pipeline_plan"] is manager.pipeline_plan
    assert kwargs["max_length"] == 1024
    assert kwargs["page_size"] == 128
    manager.kv_pages = None  # pause with destroyed pages, followed by resume
    assert manager.initialize_kv_cache(quantizer=quantizer, masking_details=masking) is pages
    assert len(attempts) == 2


def test_public_pp_allocation_preserves_bounded_oom_shrink_retry():
    manager, attempts, pages = make_manager(pipeline_enabled=True, fail_attempts=2)
    assert manager.initialize_kv_cache(quantizer=None) is pages
    assert [count for count, _ in attempts] == [1000, 780, 608]
    assert manager.metadata.num_pages == 608


@pytest.mark.parametrize(
    "pipeline_enabled,error,expected_attempts",
    [(False, "RESOURCE_EXHAUSTED", 1), (True, "RESOURCE_EXHAUSTED", 10), (True, "invalid layout", 1)],
    ids=["spmd-no-retry", "pp-bounded", "non-oom-no-retry"],
)
def test_public_allocation_propagates_failure_without_installing_pages(pipeline_enabled, error, expected_attempts):
    manager, attempts, _ = make_manager(pipeline_enabled=pipeline_enabled, fail_attempts=100, error=error)
    with pytest.raises(RuntimeError, match=error):
        manager.initialize_kv_cache(quantizer=None)
    assert len(attempts) == expected_attempts
    assert manager.kv_pages is None


@pytest.mark.parametrize("turboquant", [False, True], ids=["standard-quantizer", "turboquant"])
def test_runner_reinitialization_uses_public_manager_allocation(turboquant):
    runner = model_runner.eSurgeRunner.__new__(model_runner.eSurgeRunner)
    manager, attempts, pages = make_manager(pipeline_enabled=True, fail_attempts=1)
    runner.executor_manager = manager
    quant_config = model_runner.TurboQuantConfig() if turboquant else object()
    masking, quantizer = object(), object()
    text_config = SimpleNamespace(kv_cache_quantization_config=quant_config, get_mask_details=lambda: masking)

    def make_quantizer(*, quantization_config):
        assert quantization_config is (None if turboquant else quant_config)
        return quantizer

    runner.model = SimpleNamespace(
        config=SimpleNamespace(get_text_config=lambda: text_config), _quant_class=make_quantizer
    )
    runner.initialize_kv_cache()
    assert manager.kv_pages is pages
    assert [count for count, _ in attempts] == [1000, 780]
    assert attempts[-1][1]["quantizer"] is quantizer
    assert attempts[-1][1]["masking_details"] is masking
    runner.initialize_kv_cache()
    assert len(attempts) == 2
