"""Admission waits until the pages for a request's whole remaining prompt are free (no array computation).

A waiting request starts its prefill only when the free pages cover its whole remaining prompt plus the lookahead
slots (net of prefix hits, per cache group, a windowed group counting the pages it holds at once), after the pages the
running requests still need for their own prompts. Otherwise it waits and the queue stops there (FCFS). With nothing
running, the queue head is admitted chunk by chunk, so a prompt larger than the free pool cannot block the queue.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest
from easydel.inference.esurge.core.interface import (
    CacheGroupsConfig,
    CacheGroupSpec,
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    SlidingWindowSpec,
)
from easydel.inference.esurge.core.manager import CacheManager
from easydel.inference.esurge.outputs import ModelRunnerOutput
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.scheduler.async_scheduler import AsyncScheduler
from easydel.inference.esurge.scheduler.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_ok

SAMPLED_TOKEN = 5
EOS_TOKEN = 2

# A long-prompt mix in a fixed shuffled order: 19 prompts of about 65K tokens and 13 of about 131K tokens, 64 tokens
# out each. Their pages (about 23,000) are eight times a pool of 2,826 pages of 128 tokens.
MIX_PROMPT_TOKENS = (
    65442, 65441, 130628, 130636, 130620, 65447, 65443, 65450, 65451, 130625, 65441, 130618, 130624, 130634, 65446,
    130628, 65436, 130633, 65424, 65438, 65448, 130624, 65430, 65439, 65449, 65448, 65433, 65445, 65445, 130630,
    130621, 130621,
)  # fmt: skip
MIX_OUTPUT_TOKENS = 64
MIX_SMALL_POOL = 2826
MIX_LARGE_POOL = 23565


class _PerChunkAdmission(AsyncScheduler):
    """Control arm: the previous rule, which admits a waiting request as soon as its first chunk's pages fit."""

    def _whole_prompt_pages_fit(self, request, new_computed_pages, reserved) -> bool:
        return True


class _PerChunkAdmissionSync(Scheduler):
    """Synchronous control arm (see :class:`_PerChunkAdmission`)."""

    def _whole_prompt_pages_fit(self, request, new_computed_pages, reserved) -> bool:
        return True


def _full_group(page_size: int) -> CacheGroupSpec:
    spec = FullAttentionSpec(page_size=page_size, num_kv_heads=1, head_size=4, dtype=np.float32, use_mla=False)
    return CacheGroupSpec(kv_cache_spec=spec, layer_names=None)


def _sliding_group(page_size: int, window: int) -> CacheGroupSpec:
    spec = SlidingWindowSpec(
        page_size=page_size, num_kv_heads=1, head_size=4, dtype=np.float32, use_mla=False, sliding_window=window
    )
    return CacheGroupSpec(kv_cache_spec=spec, layer_names=None)


def _chunked_group(page_size: int, chunk: int) -> CacheGroupSpec:
    spec = ChunkedLocalAttentionSpec(
        page_size=page_size, num_kv_heads=1, head_size=4, dtype=np.float32, use_mla=False, attention_chunk_size=chunk
    )
    return CacheGroupSpec(kv_cache_spec=spec, layer_names=None)


def _scheduler(groups, *, num_pages: int, cls=Scheduler, num_draft_tokens: int = 0) -> Scheduler:
    """A synchronous scheduler: pages of 16 tokens, 64-token steps, prefill chunks of at most 32 tokens."""
    return cls(
        kv_cache_config=CacheGroupsConfig(num_pages=num_pages, kv_cache_groups=groups),
        max_num_seqs=8,
        max_num_batched_tokens=64,
        max_model_len=4096,
        num_pages=num_pages,
        page_size=16,
        enable_prefix_caching=False,
        async_scheduling=False,
        long_prefill_token_threshold=32,
        chunked_prefill_enabled=True,
        num_draft_tokens=num_draft_tokens,
        use_eagle=num_draft_tokens > 0,
    )


def _request(rid: str, prompt_len: int, max_tokens: int = 4, first_token: int = 3) -> EngineRequest:
    return EngineRequest(
        request_id=rid,
        prompt_token_ids=[first_token] + [7] * (prompt_len - 1),
        sampling_params=SamplingParams(max_tokens=max_tokens, ignore_eos=True),
        eos_token_id=EOS_TOKEN,
    )


def _runner_output(sched_out, emit: set[str]) -> ModelRunnerOutput:
    req_ids = list(sched_out.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=[[SAMPLED_TOKEN] if rid in emit else [] for rid in req_ids],
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )


class _Trace:
    """What the scheduler decided, step by step: chunks, preemptions, re-prefilled positions and emitted tokens."""

    def __init__(self) -> None:
        self.chunks: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
        self.preempted: list[tuple[int, str]] = []
        self.reprefilled_tokens = 0
        self.tokens: dict[str, int] = defaultdict(int)
        self.first_token: dict[str, int] = {}
        self._high_water: dict[str, int] = {}

    def first_scheduled(self, rid: str) -> int:
        return self.chunks[rid][0][0]

    def record(self, step: int, sched_out, requests) -> None:
        for rid, num in sched_out.num_scheduled_tokens.items():
            end = requests[rid].num_computed_tokens  # schedule() has already advanced it by ``num``
            start = end - num
            high_water = self._high_water.get(rid, 0)
            self.reprefilled_tokens += max(0, min(end, high_water) - start)
            self._high_water[rid] = max(high_water, end)
            self.chunks[rid].append((step, start, end))
        self.preempted += [(step, rid) for rid in sorted(sched_out.preempted_req_ids)]

    def apply(self, step: int, engine_outputs) -> None:
        for outputs in engine_outputs.values():
            for output in outputs.outputs:
                self.tokens[output.request_id] += len(output.new_token_ids)
                self.first_token.setdefault(output.request_id, step)


def _sync_step(sched: Scheduler, trace: _Trace, step: int):
    """One synchronous step: schedule(), then the fake runner's output for it.

    The runner samples a token for a request only when this step computes the last of its tokens.
    """
    out = sched.schedule()
    trace.record(step, out, sched.requests)
    if out.total_num_scheduled_tokens > 0:
        emit = {
            rid
            for rid in out.num_scheduled_tokens
            if sched.requests[rid].num_computed_tokens >= sched.requests[rid].num_tokens
        }
        trace.apply(step, sched.update_from_output(out, _runner_output(out, emit)))
    return out


def _sync_finish(sched: Scheduler, trace: _Trace, start: int, max_steps: int = 400) -> None:
    for step in range(start, start + max_steps):
        if not sched.requests:
            return
        _sync_step(sched, trace, step)
    raise AssertionError(f"requests left after {max_steps} steps: {sorted(sched.requests)}")


def _run_overlap(sched: AsyncScheduler, max_steps: int) -> _Trace:
    """The engine loop's overlap order with async scheduling (engine/loop.py ``_run_overlap``), and a fake runner.

    A step whose rows all have one token is dispatched asynchronously, so the next ``schedule()`` runs before its
    ``update_from_output()``; every other step runs synchronously. The runner samples one token for exactly the
    requests whose output placeholders that ``schedule()`` raised.
    """
    trace = _Trace()
    step = 0

    def schedule():
        nonlocal step
        before = {rid: request.num_output_placeholders for rid, request in sched.requests.items()}
        out = sched.schedule()
        trace.record(step, out, sched.requests)
        emit = {
            rid for rid in out.num_scheduled_tokens if sched.requests[rid].num_output_placeholders > before.get(rid, 0)
        }
        step += 1
        return out, emit

    def update(out, emit) -> None:
        trace.apply(step, sched.update_from_output(out, _runner_output(out, emit)))

    def deferred(out) -> bool:
        return out.total_num_scheduled_tokens > 0 and all(n == 1 for n in out.num_scheduled_tokens.values())

    def needs_update(out) -> bool:
        return bool(
            out.finished_req_ids or out.preempted_req_ids or out.scheduled_new_reqs or out.scheduled_cached_reqs.num_reqs
        )

    pending = None
    while sched.requests or pending is not None:
        assert step < max_steps, f"requests left after {max_steps} steps: {len(sched.requests)}"
        if pending is not None:
            nxt = schedule()
            update(*pending)
            pending = None
            if nxt[0].total_num_scheduled_tokens > 0:
                if deferred(nxt[0]):
                    pending = nxt
                else:
                    update(*nxt)
            elif needs_update(nxt[0]):
                update(nxt[0], set())
            continue
        out, emit = schedule()
        if out.total_num_scheduled_tokens == 0:
            if needs_update(out):
                update(out, set())
        elif deferred(out):
            pending = (out, emit)
        else:
            update(out, emit)
    return trace


def _mix_scheduler(cls, num_pages: int = MIX_SMALL_POOL) -> AsyncScheduler:
    """The serving scheduler shape: one full-attention group of 128-token pages, prefix caching with recurrent
    snapshot slots, 32 sequences, 32,768-token steps, chunked prefill, async scheduling."""
    sched = cls(
        kv_cache_config=CacheGroupsConfig(num_pages=num_pages, kv_cache_groups=[_full_group(128)]),
        max_num_seqs=32,
        max_num_batched_tokens=32768,
        max_model_len=131072,
        num_pages=num_pages,
        page_size=128,
        enable_prefix_caching=True,
        max_num_seq_buckets=[32],
        async_scheduling=True,
        chunked_prefill_enabled=True,
        recurrent_prefix_snapshot_slots=16,
    )
    for i, prompt_len in enumerate(MIX_PROMPT_TOKENS):
        # A distinct first token per prompt: no prompt shares a full page with another, so there are no prefix hits.
        sched.add_request(_request(f"mix-{i:02d}", prompt_len, MIX_OUTPUT_TOKENS, first_token=1000 + i))
    return sched


def test_long_prompt_mix_on_a_small_pool_runs_without_preemption():
    trace = _run_overlap(_mix_scheduler(AsyncScheduler), max_steps=5000)
    ids = [f"mix-{i:02d}" for i in range(len(MIX_PROMPT_TOKENS))]
    assert trace.preempted == []
    assert trace.reprefilled_tokens == 0
    assert dict(trace.tokens) == dict.fromkeys(ids, MIX_OUTPUT_TOKENS)
    first_tokens = [trace.first_token[rid] for rid in ids]
    assert first_tokens == sorted(first_tokens), "first tokens left submission order"


def test_per_chunk_admission_preempts_and_reprefills_the_same_mix():
    """Control: admitting on the first chunk's pages makes long prompts preempt each other on this pool."""
    trace = _run_overlap(_mix_scheduler(_PerChunkAdmission), max_steps=5000)
    assert trace.preempted
    assert trace.reprefilled_tokens > 0


def test_a_pool_that_holds_the_mix_gets_the_per_chunk_schedule_unchanged():
    """When every prompt fits, the whole-prompt rule changes no decision: the same chunks in the same steps."""
    trace = _run_overlap(_mix_scheduler(AsyncScheduler, MIX_LARGE_POOL), max_steps=2000)
    control = _run_overlap(_mix_scheduler(_PerChunkAdmission, MIX_LARGE_POOL), max_steps=2000)
    assert trace.preempted == control.preempted == []
    assert dict(trace.chunks) == dict(control.chunks)
    assert dict(trace.tokens) == dict(control.tokens)


def _queue_behind_a_running_prompt(cls):
    """13 usable pages of 16 tokens. ``a`` (160 tokens, 10 pages) prefills 32 tokens per step; ``b`` (64 tokens,
    4 pages) and then ``c`` (16 tokens) arrive at the second step."""
    sched = _scheduler([_full_group(16)], num_pages=14, cls=cls)
    trace = _Trace()
    sched.add_request(_request("a", 160))
    _sync_step(sched, trace, 0)
    b = _request("b", 64)
    sched.add_request(b)
    sched.add_request(_request("c", 16))
    out = _sync_step(sched, trace, 1)
    b_fits_alone = sched.kv_cache_manager.get_num_pages_needed(b, b.num_tokens) <= (
        sched.kv_cache_manager.page_pool.get_num_free_pages()
    )
    _sync_finish(sched, trace, start=2)
    return out, b_fits_alone, trace


def test_running_prompt_keeps_its_pages_and_the_queue_stays_fcfs():
    out, b_fits_alone, trace = _queue_behind_a_running_prompt(Scheduler)
    # b's whole prompt fits the free pages, but not next to the 6 pages a's prompt still needs: b waits, and c, which
    # would fit, does not overtake it.
    assert b_fits_alone
    assert set(out.num_scheduled_tokens) == {"a"}
    a_prompt_done = max(step for step, _, end in trace.chunks["a"] if end <= 160)
    assert trace.first_scheduled("b") > a_prompt_done
    assert trace.first_scheduled("c") >= trace.first_scheduled("b")
    assert trace.preempted == []
    assert trace.reprefilled_tokens == 0
    assert dict(trace.tokens) == {"a": 4, "b": 4, "c": 4}


def test_per_chunk_admission_preempts_in_the_same_queue():
    """Control: the previous rule admits ``b`` next to the running prompt, and a request is preempted later."""
    out, _, trace = _queue_behind_a_running_prompt(_PerChunkAdmissionSync)
    assert "b" in out.num_scheduled_tokens
    assert trace.preempted
    assert trace.reprefilled_tokens > 0


@pytest.mark.parametrize("num_draft_tokens, b_waits", [(0, False), (3, True)])
def test_lookahead_slots_count_for_the_waiting_and_the_running_prompt(num_draft_tokens, b_waits):
    """15 usable pages of 16 tokens and one output token per request, so only prompt pages are ever drawn. Without
    lookahead, ``b``'s 4 pages fit next to the 6 that ``a``'s remaining prompt still needs, with one page to spare.
    3 lookahead slots make ``a`` hold one more page and ``b`` need one more, so ``b`` is one page short and waits."""
    sched = _scheduler([_full_group(16)], num_pages=16, num_draft_tokens=num_draft_tokens)
    trace = _Trace()
    sched.add_request(_request("a", 160, max_tokens=1))
    _sync_step(sched, trace, 0)
    sched.add_request(_request("b", 64, max_tokens=1))
    out = _sync_step(sched, trace, 1)
    assert ("b" not in out.num_scheduled_tokens) is b_waits
    _sync_finish(sched, trace, start=2)
    assert trace.preempted == []
    assert dict(trace.tokens) == {"a": 1, "b": 1}


def test_prompt_larger_than_the_free_pages_waits_until_they_are_free():
    """13 usable pages. ``big``'s whole prompt (10 pages) is more than the pages left while ``a`` (4 pages, then 5)
    runs, so it waits; once ``a`` finishes it runs to completion, and nothing is preempted."""
    sched = _scheduler([_full_group(16)], num_pages=14)
    trace = _Trace()
    sched.add_request(_request("a", 64))
    sched.add_request(_request("big", 160, first_token=4))
    _sync_finish(sched, trace, start=0)
    assert trace.first_scheduled("big") > max(step for step, _, _ in trace.chunks["a"])
    assert trace.preempted == []
    assert trace.reprefilled_tokens == 0
    assert dict(trace.tokens) == {"a": 4, "big": 4}


def test_prompt_larger_than_the_pool_is_admitted_when_nothing_runs():
    """9 usable pages hold 144 tokens. A 400-token prompt is never whole in the pool, so the whole-prompt rule alone
    would hold the queue forever; with nothing running its first chunk is admitted under the per-chunk rule."""
    sched = _scheduler([_full_group(16)], num_pages=10)
    sched.add_request(_request("huge", 400))
    sched.add_request(_request("next", 16))
    out = sched.schedule()
    assert out.num_scheduled_tokens["huge"] == 32
    assert [r.req_id for r in out.scheduled_new_reqs] == ["huge"]


@pytest.mark.parametrize("caching", [False, True])
@pytest.mark.parametrize("max_chunk", [32, 37])
@pytest.mark.parametrize("kind, window", [("sliding", 64), ("sliding", 50), ("chunked", 64), ("chunked", 48)])
def test_windowed_group_counts_the_pages_it_holds_at_once(kind, window, max_chunk, caching):
    """Reference-free bound: a windowed group never holds more pages at once than the count made at admission, and
    that count is far below the prompt's full page count."""
    group = _sliding_group(16, window) if kind == "sliding" else _chunked_group(16, window)
    manager = CacheManager(num_pages=200, kv_cache_groups=[group], max_model_len=4096, enable_caching=caching)
    request = _request("r", 1000, first_token=11)
    if caching:
        manager.get_computed_pages(request)
    needed = manager.get_num_pages_needed(request, request.num_tokens, max_chunk_tokens=max_chunk)
    full = manager.get_num_pages_needed(request, request.num_tokens)
    assert full == -(-request.num_tokens // 16)
    assert needed * 4 < full
    free_before = manager.page_pool.get_num_free_pages()
    peak = 0
    while request.num_computed_tokens < request.num_tokens:
        chunk = min(max_chunk, request.num_tokens - request.num_computed_tokens)
        assert manager.allocate_slots(request, chunk) is not None
        peak = max(peak, free_before - manager.page_pool.get_num_free_pages())
        request.num_computed_tokens += chunk
        manager.cache_pages(request, request.num_computed_tokens)
        rest = manager.get_num_pages_needed(request, request.num_tokens, max_chunk_tokens=max_chunk)
        held = free_before - manager.page_pool.get_num_free_pages()
        assert held + rest <= needed, "the count for the rest must stay within the admission count"
    assert 0 < peak <= needed


def test_windowed_prompt_is_admitted_on_the_pages_it_holds_at_once():
    """20 usable pages; two 1,000-token prompts under a 64-token sliding window. ``b``'s full page count (63) exceeds
    the free pool, but its window holds a few pages at a time, so it starts next to ``a`` and nothing is preempted."""
    sched = _scheduler([_sliding_group(16, 64)], num_pages=21)
    trace = _Trace()
    sched.add_request(_request("a", 1000))
    _sync_step(sched, trace, 0)
    sched.add_request(_request("b", 1000, first_token=4))
    free = sched.kv_cache_manager.page_pool.get_num_free_pages()
    out = _sync_step(sched, trace, 1)
    assert -(-1000 // 16) > free
    assert "b" in out.num_scheduled_tokens
    _sync_finish(sched, trace, start=2)
    assert trace.preempted == []
    assert dict(trace.tokens) == {"a": 4, "b": 4}
