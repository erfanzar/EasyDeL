"""Host-only regressions for cache-pressure progress, DP cleanup and compile padding."""

from types import SimpleNamespace

import pytest
from easydel.inference.esurge.core.interface import CacheGroupsConfig, CacheGroupSpec, FullAttentionSpec
from easydel.inference.esurge.outputs import ModelRunnerOutput
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.scheduler.dp_scheduler import DPScheduler
from easydel.inference.esurge.scheduler.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams
from jax import numpy as jnp

# Only host queues, Python page IDs and JAX dtype metadata are used here.
pytestmark = pytest.mark.cpu_ok


def _config(num_pages):
    return CacheGroupsConfig(
        num_pages=num_pages,
        kv_cache_groups=[
            CacheGroupSpec(
                kv_cache_spec=FullAttentionSpec(
                    page_size=4, num_kv_heads=1, head_size=4, dtype=jnp.float32, use_mla=False
                ),
                layer_names=None,
            )
        ],
    )


def _request(req_id, prompt_length=5, max_tokens=4):
    return EngineRequest(
        request_id=req_id,
        prompt_token_ids=[1] * prompt_length,
        sampling_params=SamplingParams(max_tokens=max_tokens),
        eos_token_id=0,
    )


def _output(req_ids, token=10):
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={req_id: i for i, req_id in enumerate(req_ids)},
        sampled_token_ids=[[token] for _ in req_ids],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )


@pytest.mark.parametrize("margin", [0, 999])
def test_running_decode_finishes_with_no_free_pages_even_under_safety_reserve(margin):
    scheduler = Scheduler(
        kv_cache_config=_config(3),
        max_num_seqs=2,
        max_num_batched_tokens=8,
        max_model_len=32,
        num_pages=3,
        page_size=4,
        enable_prefix_caching=False,
        async_scheduling=False,
        token_safety_margin=margin,
    )
    request = _request("running")
    scheduler.add_request(request)
    first = scheduler.schedule()
    assert first.num_scheduled_tokens == {"running": 5}
    scheduler.update_from_output(first, _output(["running"]))
    assert scheduler.kv_cache_manager.page_pool.get_num_free_pages() == 0
    scheduler.add_request(_request("waiting", prompt_length=1))
    for _ in range(3):
        step = scheduler.schedule()
        assert step.num_scheduled_tokens == {"running": 1}
        assert step.total_num_scheduled_tokens <= 8
        scheduler.update_from_output(step, _output(["running"]))
    assert request.is_finished()
    assert scheduler.kv_cache_manager.page_pool.get_num_free_pages() == 2
    assert scheduler.schedule().num_scheduled_tokens == {"waiting": 1}


def test_running_request_reaches_preemption_when_an_additional_page_is_needed():
    scheduler = Scheduler(
        kv_cache_config=_config(3),
        max_num_seqs=2,
        max_num_batched_tokens=8,
        max_model_len=32,
        num_pages=3,
        page_size=4,
        enable_prefix_caching=False,
        async_scheduling=False,
        token_safety_margin=0,
    )
    scheduler.add_request(_request("older", prompt_length=4))
    scheduler.add_request(_request("newer", prompt_length=4))
    scheduled = scheduler.schedule()
    assert scheduled.num_scheduled_tokens == {"older": 4, "newer": 4}
    scheduler.update_from_output(scheduled, _output(["older", "newer"]))
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"older": 1}
    assert step.preempted_req_ids == {"newer"}


@pytest.mark.parametrize("worker", [False, True])
@pytest.mark.parametrize("include_finished_set", [False, True])
def test_dp_natural_completion_releases_coordinator_request_and_rank(worker, include_finished_set):
    scheduler = DPScheduler(
        kv_cache_config=_config(32),
        dp_size=2,
        max_num_seqs=4,
        max_num_batched_tokens=8,
        max_model_len=32,
        num_pages=32,
        page_size=4,
        enable_prefix_caching=False,
        async_scheduling=False,
        use_worker_processes=worker,
        include_finished_set=include_finished_set,
    )
    try:
        request = _request("finished", prompt_length=2, max_tokens=1)
        scheduler.add_request(request)
        outputs = scheduler.update_from_output(scheduler.schedule(), _output(["finished"]))
        result = outputs[0].outputs[0]
        assert result.request_id == "finished"
        assert result.finish_reason is not None
        assert list(request.output_token_ids) == [10]
        assert request.is_finished()
        assert scheduler.get_num_unfinished_requests() == 0
        assert "finished" not in scheduler.requests
        assert "finished" not in scheduler.req_id_to_dp_rank
        if not include_finished_set:
            assert outputs[0].finished_requests is None
        cleanup = scheduler.schedule()
        assert cleanup.finished_req_ids == {"finished"}
        scheduler.update_from_output(cleanup, _output([]))
        assert "finished" not in scheduler.requests
    finally:
        scheduler.shutdown()


def test_from_runner_compile_padding_does_not_increase_logical_concurrency():
    runner = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace()),
        kv_cache_groups=_config(32).kv_cache_groups,
        metadata=SimpleNamespace(num_pages=32, page_size=4, data_parallel_size=1),
        max_num_seqs=2,
        max_model_len=32,
        max_num_seq_buckets=[4, 8],
    )
    scheduler = Scheduler.from_runner(
        runner, max_num_batched_tokens=8, enable_prefix_caching=False, async_scheduling=False
    )
    try:
        for req_id in ("a", "b", "c"):
            scheduler.add_request(_request(req_id, prompt_length=1))
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == {"a": 1, "b": 1}
        assert len(scheduler.running) == 2
        assert len(scheduler.waiting) == 1
        assert output.suggested_bucket == 4
    finally:
        scheduler.shutdown()
