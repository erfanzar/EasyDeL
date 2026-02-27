from __future__ import annotations

import jax.numpy as jnp

from easydel.inference.esurge.config import CacheConfig, Config, SchedulerConfig
from easydel.inference.esurge.core.interface import CacheGroupSpec, CacheGroupsConfig, FullAttentionSpec
from easydel.inference.esurge.outputs import ModelRunnerOutput
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.scheduler.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams


def _make_scheduler(*, num_pages: int) -> Scheduler:
    page_size = 64
    config = Config(
        scheduler_config=SchedulerConfig(
            max_num_seqs=16,
            max_num_batched_tokens=2048,
            max_model_len=8208,
            token_safety_margin=None,
        ),
        cache_config=CacheConfig(
            num_pages=num_pages,
            page_size=page_size,
            enable_prefix_caching=False,
        ),
    )
    kv_cache_config = CacheGroupsConfig(
        num_pages=num_pages,
        kv_cache_groups=[
            CacheGroupSpec(
                kv_cache_spec=FullAttentionSpec(
                    page_size=page_size,
                    num_kv_heads=1,
                    head_size=4,
                    dtype=jnp.float32,
                    use_mla=False,
                ),
                layer_names=None,
            )
        ],
    )
    scheduler = Scheduler(config=config, kv_cache_config=kv_cache_config)
    scheduler.data_parallel_size = 4
    return scheduler


def _attach_allocate_slots_spy(scheduler: Scheduler) -> list[tuple[int | None, int | None]]:
    captured: list[tuple[int | None, int | None]] = []
    original = scheduler.kv_cache_manager.allocate_slots

    def _spy(*args, **kwargs):
        captured.append((kwargs.get("dp_shard_hint"), kwargs.get("data_parallel_size")))
        return original(*args, **kwargs)

    scheduler.kv_cache_manager.allocate_slots = _spy  # pyright: ignore[reportAttributeAccessIssue]
    return captured


def _add_request(scheduler: Scheduler, *, request_id: str = "req-fsdp") -> EngineRequest:
    request = EngineRequest(
        request_id=request_id,
        prompt_token_ids=[101, 102, 103],
        sampling_params=SamplingParams(max_tokens=16),
        eos_token_id=1,
    )
    scheduler.add_request(request)
    return request


def test_scheduler_disables_dp_local_hints_when_usable_pages_are_not_divisible() -> None:
    scheduler = _make_scheduler(num_pages=8780)  # usable pages: 8779 (not divisible by 4)
    _add_request(scheduler)
    captured = _attach_allocate_slots_spy(scheduler)

    output = scheduler.schedule()

    assert output.total_num_scheduled_tokens > 0
    assert captured
    assert all(hint is None for hint, _ in captured)
    assert all(dp_size == 4 for _, dp_size in captured)


def test_scheduler_uses_dp_local_hints_when_usable_pages_are_divisible() -> None:
    scheduler = _make_scheduler(num_pages=8781)  # usable pages: 8780 (divisible by 4)
    _add_request(scheduler)
    captured = _attach_allocate_slots_spy(scheduler)

    output = scheduler.schedule()

    assert output.total_num_scheduled_tokens > 0
    assert captured
    assert any(hint is not None for hint, _ in captured)
    assert all(dp_size == 4 for _, dp_size in captured)


def test_fsdp_like_scheduler_cycle_completes_request_with_fallback() -> None:
    scheduler = _make_scheduler(num_pages=8780)
    request = _add_request(scheduler, request_id="req-cycle")
    scheduler_output = scheduler.schedule()

    model_output = ModelRunnerOutput(
        req_ids=[request.request_id],
        req_id_to_index={request.request_id: 0},
        req_id_to_row_index={request.request_id: 0},
        sampled_token_ids=[[1]],  # eos token
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={request.request_id: None},
    )
    engine_outputs = scheduler.update_from_output(scheduler_output, model_output)

    assert scheduler.get_num_unfinished_requests() == 0
    assert request.request_id not in scheduler.requests
    assert 0 in engine_outputs
    assert engine_outputs[0].outputs
    assert engine_outputs[0].outputs[0].request_id == request.request_id
    assert engine_outputs[0].outputs[0].finished
