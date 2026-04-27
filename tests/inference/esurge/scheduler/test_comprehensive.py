"""Comprehensive scheduler tests — covers every scheduling path."""

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
from easydel.inference.esurge.request import EngineRequest, EngineRequestStatus
from easydel.inference.esurge.scheduler.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams


def make_scheduler(
    max_num_seqs: int = 128,
    max_num_batched_tokens: int = 8192,
    max_model_len: int = 65536,
    num_pages: int = 6000,
    page_size: int = 128,
    enable_prefix_caching: bool = False,
    chunked_prefill: bool = False,
    long_prefill_threshold: int | None = None,
    token_safety_margin: int | None = None,
    policy: str = "fcfs",
) -> Scheduler:
    config = Config(
        scheduler_config=SchedulerConfig(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            token_safety_margin=token_safety_margin,
            chunked_prefill_enabled=chunked_prefill,
            long_prefill_token_threshold=long_prefill_threshold,
            policy=policy,
        ),
        cache_config=CacheConfig(
            num_pages=num_pages,
            page_size=page_size,
            enable_prefix_caching=enable_prefix_caching,
        ),
    )
    kv = CacheGroupsConfig(
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
    return Scheduler(config=config, kv_cache_config=kv)


def make_req(rid: str, prompt_len: int, max_tokens: int = 8192) -> EngineRequest:
    return EngineRequest(
        request_id=rid,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=max_tokens),
        eos_token_id=1,
    )


def fake_output(sched_out, token_id: int = 42) -> ModelRunnerOutput:
    req_ids = list(sched_out.num_scheduled_tokens.keys())
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=[[token_id] for _ in req_ids],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )


def step(sched: Scheduler) -> tuple:
    out = sched.schedule()
    if out.total_num_scheduled_tokens > 0:
        mo = fake_output(out)
        engine_outs = sched.update_from_output(out, mo)
    else:
        engine_outs = {}
    return out, engine_outs


def prefill_request(sched: Scheduler, rid: str, prompt_len: int, max_tokens: int = 8192):
    sched.add_request(make_req(rid, prompt_len, max_tokens))
    return step(sched)[0]


class TestBasicScheduling:
    def test_single_request_prefill(self):
        s = make_scheduler()
        s.add_request(make_req("r1", 100))
        out, _ = step(s)
        assert out.total_num_scheduled_tokens == 100
        assert out.num_running_reqs == 1
        assert out.num_waiting_reqs == 0

    def test_decode_generates_one_token(self):
        s = make_scheduler()
        s.add_request(make_req("r1", 100))
        step(s)
        out2, _ = step(s)
        assert out2.num_scheduled_tokens.get("r1") == 1


class TestBatching:
    def test_all_small_requests_prefilled(self):
        s = make_scheduler(max_num_batched_tokens=8192)
        for i in range(10):
            s.add_request(make_req(f"r{i}", 500))
        out, _ = step(s)
        assert len(out.scheduled_new_reqs) == 10
        assert out.total_num_scheduled_tokens == 5000

    def test_decode_batch(self):
        s = make_scheduler(max_num_batched_tokens=8192)
        for i in range(10):
            s.add_request(make_req(f"r{i}", 500))
        step(s)
        out2, _ = step(s)
        assert out2.total_num_scheduled_tokens == 10


class TestBudgetExhaustion:
    def test_budget_limits_admission(self):
        s = make_scheduler(max_num_batched_tokens=2048)
        for i in range(10):
            s.add_request(make_req(f"r{i}", 500))
        out, _ = step(s)
        assert out.total_num_scheduled_tokens <= 2048
        assert out.num_waiting_reqs > 0
        assert 1 <= len(out.scheduled_new_reqs) < 10


class TestLargePrefillWithDecode:
    def test_no_starvation(self):
        s = make_scheduler(max_num_batched_tokens=8192)
        prefill_request(s, "decode0", 100)
        for i in range(5):
            s.add_request(make_req(f"large{i}", 20000))

        out, _ = step(s)
        assert "decode0" in out.num_scheduled_tokens
        assert len(out.scheduled_new_reqs) >= 1
        assert out.total_num_scheduled_tokens > 1

    def test_decode_gets_token_alongside_large_prefill(self):
        s = make_scheduler(max_num_batched_tokens=8192)
        prefill_request(s, "decode0", 100)
        s.add_request(make_req("large0", 20000))
        out, _ = step(s)
        assert out.num_scheduled_tokens.get("decode0") == 1
        assert out.num_scheduled_tokens.get("large0", 0) > 0


class TestPrefillBudgetSharing:
    def test_decodes_and_prefill_share_budget(self):
        s = make_scheduler(max_num_batched_tokens=8192)
        for i in range(30):
            prefill_request(s, f"d{i}", 100)
        s.add_request(make_req("big", 20000))
        out, _ = step(s)

        decode_count = sum(1 for rid in out.num_scheduled_tokens if rid.startswith("d"))
        big_tokens = out.num_scheduled_tokens.get("big", 0)
        assert decode_count == 30
        assert "big" in out.num_scheduled_tokens
        assert out.total_num_scheduled_tokens <= 8192
        assert big_tokens == 8192 - 30

    def test_non_inherently_large_skipped_when_batch_not_empty(self):
        s = make_scheduler(max_num_batched_tokens=8192, max_model_len=65536)
        for i in range(200):
            prefill_request(s, f"d{i}", 50)
        s.add_request(make_req("mid", 8000))

        out, _ = step(s)
        assert "mid" not in out.num_scheduled_tokens


class TestCompletion:
    def test_request_finishes_at_max_tokens(self):
        s = make_scheduler()
        s.add_request(make_req("r1", 100, max_tokens=5))
        step(s)
        for _ in range(10):
            step(s)
            if not s.has_unfinished_requests():
                break
        assert not s.has_unfinished_requests()
        assert len(s.running) == 0
        assert len(s.requests) == 0

    def test_request_finishes_on_eos(self):
        s = make_scheduler()
        s.add_request(make_req("r1", 100, max_tokens=1000))
        out, _ = step(s)
        mo = ModelRunnerOutput(
            req_ids=["r1"],
            req_id_to_index={"r1": 0},
            sampled_token_ids=[[1]],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
        )
        s.update_from_output(out, mo)
        assert not s.has_unfinished_requests()


class TestAbort:
    def test_external_abort(self):
        s = make_scheduler()
        s.add_request(make_req("r1", 100))
        step(s)
        assert len(s.running) == 1
        s.finish_requests("r1", EngineRequestStatus.FINISHED_ABORTED)
        assert len(s.running) == 0
        assert "r1" not in s.requests
        assert "r1" in s.finished_req_ids

    def test_abort_nonexistent_is_safe(self):
        s = make_scheduler()
        s.finish_requests("nonexistent", EngineRequestStatus.FINISHED_ABORTED)

    def test_batch_abort(self):
        s = make_scheduler()
        s.add_request(make_req("r1", 100))
        s.add_request(make_req("r2", 100))
        step(s)
        s.finish_requests(["r1", "r2"], EngineRequestStatus.FINISHED_ABORTED)
        assert s.get_num_unfinished_requests() == 0


class TestPreemption:
    def test_preemption_under_memory_pressure(self):
        s = make_scheduler(num_pages=50, page_size=128, max_num_batched_tokens=65536)
        for i in range(5):
            s.add_request(make_req(f"r{i}", 500))
        out1, _ = step(s)
        assert len(out1.scheduled_new_reqs) > 0


class TestMaxNumSeqs:
    def test_respects_limit(self):
        s = make_scheduler(max_num_seqs=4, max_num_batched_tokens=65536)
        for i in range(10):
            s.add_request(make_req(f"r{i}", 50))
        out, _ = step(s)
        assert out.num_running_reqs <= 4
        assert out.num_waiting_reqs == 10 - out.num_running_reqs


class TestEmpty:
    def test_empty_scheduler(self):
        s = make_scheduler()
        out, _ = step(s)
        assert out.total_num_scheduled_tokens == 0
        assert out.num_running_reqs == 0
        assert out.num_waiting_reqs == 0
        assert not s.has_unfinished_requests()


class TestChunkedPrefill:
    def test_chunks_large_prompt(self):
        s = make_scheduler(max_num_batched_tokens=2048, chunked_prefill=True)
        s.add_request(make_req("big", 10000))
        out1, _ = step(s)
        assert out1.total_num_scheduled_tokens <= 2048
        assert out1.num_running_reqs == 1
        out2, _ = step(s)
        assert out2.total_num_scheduled_tokens > 0


class TestLongPrefillThreshold:
    def test_caps_prefill_tokens(self):
        s = make_scheduler(max_num_batched_tokens=8192, long_prefill_threshold=2048)
        s.add_request(make_req("big", 20000))
        out, _ = step(s)
        assert out.total_num_scheduled_tokens <= 2048

    def test_threshold_with_active_decodes(self):
        s = make_scheduler(max_num_batched_tokens=8192, long_prefill_threshold=2048)
        for i in range(10):
            prefill_request(s, f"d{i}", 50)
        s.add_request(make_req("big", 20000))
        out, _ = step(s)
        big_tokens = out.num_scheduled_tokens.get("big", 0)
        assert big_tokens <= 2048
        decode_count = sum(1 for rid in out.num_scheduled_tokens if rid.startswith("d"))
        assert decode_count == 10


class TestSafetyMargin:
    def test_margin_reserves_tokens(self):
        s = make_scheduler(max_num_batched_tokens=8192, token_safety_margin=64)
        for i in range(10):
            prefill_request(s, f"d{i}", 50)
        s.add_request(make_req("new", 500))
        out, _ = step(s)
        assert out.total_num_scheduled_tokens <= 8192


class TestEdgeCases:
    def test_single_token_prompt(self):
        s = make_scheduler()
        s.add_request(make_req("tiny", 1, max_tokens=1))
        out, _ = step(s)
        assert out.total_num_scheduled_tokens == 1

    def test_single_token_finishes_on_eos(self):
        s = make_scheduler()
        s.add_request(make_req("tiny", 1, max_tokens=1))
        out, _ = step(s)
        mo = ModelRunnerOutput(
            req_ids=["tiny"],
            req_id_to_index={"tiny": 0},
            sampled_token_ids=[[1]],
            spec_token_ids=None,
            logprobs=None,
            prompt_logprobs_dict={},
        )
        s.update_from_output(out, mo)
        assert not s.has_unfinished_requests()


class TestMultipleLargePrefills:
    def test_progress_over_steps(self):
        s = make_scheduler(max_num_batched_tokens=8192)
        for i in range(5):
            s.add_request(make_req(f"big{i}", 20000))
        tokens_per_step = []
        for _ in range(10):
            out, _ = step(s)
            tokens_per_step.append(out.total_num_scheduled_tokens)
            if not s.has_unfinished_requests():
                break
        assert tokens_per_step[0] > 0
        assert sum(tokens_per_step) > tokens_per_step[0]


class TestDecodeNotBlockedByPrefill:
    def test_all_decodes_get_tokens(self):
        s = make_scheduler(max_num_batched_tokens=8192)
        for i in range(20):
            prefill_request(s, f"d{i}", 100)
        s.add_request(make_req("big", 30000))
        out, _ = step(s)

        decode_scheduled = sum(1 for rid in out.num_scheduled_tokens if rid.startswith("d"))
        big_tokens = out.num_scheduled_tokens.get("big", 0)
        assert decode_scheduled == 20
        assert big_tokens > 0
        assert big_tokens <= 8192 - 20


class TestWaitingStatusTransitions:
    def test_waiting_promoted_after_finish(self):
        s = make_scheduler(max_num_seqs=1)
        s.add_request(make_req("r1", 100))
        s.add_request(make_req("r2", 100))
        out1, _ = step(s)
        assert out1.num_running_reqs == 1
        assert out1.num_waiting_reqs == 1
        s.finish_requests("r1", EngineRequestStatus.FINISHED_ABORTED)
        out2, _ = step(s)
        assert out2.num_running_reqs == 1
        assert len(out2.scheduled_new_reqs) == 1


class TestPrefixCacheReset:
    def test_reset_returns_bool(self):
        s = make_scheduler(enable_prefix_caching=True)
        assert isinstance(s.reset_prefix_cache(), bool)


class TestRequestCounts:
    def test_counts_through_lifecycle(self):
        s = make_scheduler()
        assert s.get_request_counts() == (0, 0)
        s.add_request(make_req("r1", 100))
        assert s.get_request_counts() == (0, 1)
        step(s)
        assert s.get_request_counts() == (1, 0)
        s.finish_requests("r1", EngineRequestStatus.FINISHED_ABORTED)
        assert s.get_request_counts() == (0, 0)


class TestSuggestedBucket:
    def test_bucket_set_and_valid(self):
        s = make_scheduler(max_num_seqs=128)
        for i in range(5):
            s.add_request(make_req(f"r{i}", 50))
        out, _ = step(s)
        assert out.suggested_bucket is not None
        assert out.suggested_bucket >= out.num_running_reqs


class TestInherentlyTooLargeBoundary:
    def test_exact_boundary_admitted(self):
        s = make_scheduler(max_num_batched_tokens=8192)
        prefill_request(s, "d0", 100)
        s.add_request(make_req("exact", 20000))
        out, _ = step(s)
        assert "exact" in out.num_scheduled_tokens


class TestPriorityPolicy:
    def test_schedules_requests(self):
        s = make_scheduler(policy="priority", max_num_seqs=128, max_num_batched_tokens=65536)
        for i in range(5):
            s.add_request(make_req(f"r{i}", 100))
        out, _ = step(s)
        assert out.num_running_reqs == 5


class TestContinuousDecode:
    def test_request_finishes(self):
        s = make_scheduler()
        s.add_request(make_req("r1", 100, max_tokens=50))
        step(s)
        for _ in range(55):
            step(s)
            if not s.has_unfinished_requests():
                break
        assert not s.has_unfinished_requests()


class TestMixedInterleaving:
    def test_early_decode_and_late_prefill(self):
        s = make_scheduler(max_num_batched_tokens=4096, max_num_seqs=32)
        for i in range(5):
            prefill_request(s, f"early{i}", 100)
        for i in range(5):
            s.add_request(make_req(f"late{i}", 300))
        out, _ = step(s)

        early_count = sum(1 for rid in out.num_scheduled_tokens if rid.startswith("early"))
        late_count = sum(1 for rid in out.num_scheduled_tokens if rid.startswith("late"))
        assert early_count == 5
        assert late_count > 0
        assert out.total_num_scheduled_tokens <= 4096
