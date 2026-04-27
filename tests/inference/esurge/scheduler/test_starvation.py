"""Reproduce: 1 decode running, 31 waiting, budget=8191/8192, none scheduled."""

from __future__ import annotations

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import jax.numpy as jnp

from easydel.inference.esurge.config import CacheConfig, Config, SchedulerConfig
from easydel.inference.esurge.core.interface import (
    CacheGroupsConfig,
    CacheGroupSpec,
    FullAttentionSpec,
)
from easydel.inference.esurge.outputs import ModelRunnerOutput
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.scheduler.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams


def _make_scheduler(
    max_num_seqs: int = 128,
    max_num_batched_tokens: int = 8192,
    max_model_len: int = 65536,
    num_pages: int = 6000,
    page_size: int = 128,
) -> Scheduler:
    config = Config(
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
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
    return Scheduler(config=config, kv_cache_config=kv_cache_config)


def _make_request(request_id: str, prompt_len: int) -> EngineRequest:
    return EngineRequest(
        request_id=request_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=8192),
        eos_token_id=1,
    )


def _fake_model_output(scheduler_output) -> ModelRunnerOutput:
    num_scheduled = scheduler_output.num_scheduled_tokens
    req_ids = list(num_scheduled.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=[[42] for _ in req_ids],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )


def _prefill_one(scheduler: Scheduler, request_id: str, prompt_len: int):
    scheduler.add_request(_make_request(request_id, prompt_len))
    out = scheduler.schedule()
    scheduler.update_from_output(out, _fake_model_output(out))
    return out


class TestStarvation:
    def test_mixed_prompt_sizes(self):
        scheduler = _make_scheduler()
        _prefill_one(scheduler, "decode-0", 100)

        prompt_sizes = [
            500,
            1000,
            2000,
            4000,
            8000,
            9000,
            10000,
            12000,
            15000,
            20000,
            25000,
            30000,
            500,
            1000,
            2000,
            4000,
            8000,
            9000,
            10000,
            12000,
            15000,
            20000,
            25000,
            30000,
            500,
            1000,
            2000,
            4000,
            8000,
            9000,
            10000,
        ]
        for i, plen in enumerate(prompt_sizes):
            scheduler.add_request(_make_request(f"wait-{i}", plen))

        out = scheduler.schedule()
        assert out.total_num_scheduled_tokens > 1, (
            f"starvation: only {out.total_num_scheduled_tokens} tokens scheduled with {out.num_waiting_reqs} waiting"
        )

    def test_small_prompts_only(self):
        scheduler = _make_scheduler()
        _prefill_one(scheduler, "decode-0", 100)

        for i in range(31):
            scheduler.add_request(_make_request(f"wait-{i}", 500))

        out = scheduler.schedule()
        assert out.total_num_scheduled_tokens > 1
        assert len(out.scheduled_new_reqs) > 0

    def test_large_prompts_only(self):
        scheduler = _make_scheduler()
        _prefill_one(scheduler, "decode-0", 100)

        for i in range(31):
            scheduler.add_request(_make_request(f"wait-{i}", 20000))

        out = scheduler.schedule()
        assert out.total_num_scheduled_tokens > 1, (
            f"starvation: only {out.total_num_scheduled_tokens} tokens scheduled with {out.num_waiting_reqs} waiting"
        )
        assert len(out.scheduled_new_reqs) >= 1

    def test_many_decodes_plus_large_waiting(self):
        scheduler = _make_scheduler()
        for i in range(50):
            _prefill_one(scheduler, f"d{i}", 100)

        for i in range(20):
            scheduler.add_request(_make_request(f"wait-{i}", 25000))

        out = scheduler.schedule()
        decode_count = sum(1 for rid in out.num_scheduled_tokens if rid.startswith("d"))
        assert decode_count == 50
        assert len(out.scheduled_new_reqs) >= 1
