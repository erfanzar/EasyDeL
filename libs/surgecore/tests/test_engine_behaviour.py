# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scheduling behaviour of surgecore's EngineCore, through the Python binding.

These are the behaviour checks of eSurge's former Python scheduler (queues, budget, chunked prefill, preemption,
admission, prefix publication, data-parallel ranks, speculative drafts, stop checks, the finished set and the
structured-output flag), stated against ``EngineCore``. A fake runner samples one token for every entry the plan
marks as emitting, unless a test gives the sampled lists itself. ``check()`` runs after every step.
"""

from __future__ import annotations

import threading
from collections import defaultdict

import numpy as np
import pytest
from surgecore import PLAN_HAS_VERIFY, PLAN_STRUCTURED_PENDING, CoreClosed, EngineCore, KvCore

pytestmark = pytest.mark.cpu_ok

FLAG_GRAMMAR = 1
KIND_RUNNING, KIND_NEW, KIND_RESUMED = 0, 1, 2
FINISH_STOP, FINISH_LENGTH, FINISH_ABORT = 1, 2, 3
STATUS_WAITING, STATUS_RUNNING, STATUS_PREEMPTED = 0, 1, 2


class Core:
    """An ``EngineCore`` with request names, a fake runner and step helpers."""

    def __init__(self, *, page_size=128, window=0, kind=0, groups=None, **kw):
        args = dict(
            max_rows=128,
            max_model_len=65536,
            num_pages=6000,
            dp_size=1,
            token_budget=8192,
            prefix_caching=False,
            async_scheduling=False,
        )
        args.update(kw)
        if groups is None:
            groups = [[kind, page_size, window]]
        self.core = EngineCore(groups=np.asarray(groups, np.int32), **args)
        self.name: dict[int, str] = {}
        self.handle: dict[str, int] = {}

    def add(self, name, prompt, *, max_tokens=8192, eos=1, stops=(), ignore_eos=False, priority=0, arrival=0.0,
            flags=0, dp_rank=-1, spans=()):  # fmt: skip
        tokens = np.arange(prompt, dtype=np.int32) if isinstance(prompt, int) else np.asarray(prompt, np.int32)
        (h,) = self.core.add_requests(
            prompt_tokens=tokens,
            prompt_offsets=np.asarray([0, tokens.size], np.int32),
            max_tokens=np.asarray([max_tokens], np.int64),
            eos_token=np.asarray([eos], np.int32),
            stop_tokens=np.asarray(list(stops), np.int32),
            stop_offsets=np.asarray([0, len(stops)], np.int32),
            ignore_eos=np.asarray([ignore_eos]),
            priority=np.asarray([priority], np.int64),
            arrival=np.asarray([arrival], np.float64),
            flags=np.asarray([flags], np.uint32),
            dp_rank=np.asarray([dp_rank], np.int64),
            image_spans=np.asarray(list(spans), np.int64).reshape(len(spans), 2) if spans else None,
            span_offsets=np.asarray([0, len(spans)], np.int32),
        ).tolist()
        self.name[h], self.handle[name] = name, h
        return h

    def schedule(self):
        plan = self.core.schedule()
        self.core.check()
        return plan

    def commit(self, plan, rows=None, *, token=42, accepted=None, drafts=None, present=None):
        if rows is None:
            rows = [[token] if e else [] for e in plan.emits.tolist()]
        flat = [t for r in rows for t in r]
        kw = {}
        if accepted is not None:
            kw["accepted"] = np.asarray(accepted, np.int32)
        if drafts is not None:
            kw["new_drafts"] = np.asarray([t for d in drafts for t in d], np.int32)
            kw["new_draft_offsets"] = np.cumsum([0] + [len(d) for d in drafts]).astype(np.int32)
        if present is not None:
            kw["present"] = np.asarray(present, np.uint8)
        ev = self.core.commit(
            plan,
            sampled_tokens=np.asarray(flat, np.int32),
            sampled_offsets=np.cumsum([0] + [len(r) for r in rows]).astype(np.int32),
            **kw,
        )
        self.core.check()
        return ev

    def step(self, token=42):
        plan = self.schedule()
        ev = self.commit(plan, token=token) if plan.n else None
        return plan, ev

    def prefill(self, name, prompt, **kw):
        self.add(name, prompt, **kw)
        return self.step()[0]

    def sched(self, plan) -> dict[str, int]:
        return {self.name[h]: s for h, s in zip(plan.handles.tolist(), plan.scheduled.tolist(), strict=True)}

    def new(self, plan) -> list[str]:
        return [self.name[h] for h, k in zip(plan.handles.tolist(), plan.kinds.tolist(), strict=True) if k == KIND_NEW]

    def emitted(self, ev) -> dict[str, list[int]]:
        if ev is None:
            return {}
        off, tok = ev.token_offsets.tolist(), ev.tokens.tolist()
        return {self.name[h]: tok[off[i] : off[i + 1]] for i, h in enumerate(ev.handles.tolist())}

    def info(self, name):
        return self.core.request_info(self.handle[name])

    def running(self) -> int:
        return int(self.core.counts()[0])

    def waiting(self) -> int:
        return int(self.core.counts()[1])

    def unfinished(self) -> int:
        return self.running() + self.waiting()

    def finish(self, names, reason=FINISH_ABORT):
        hs = [self.handle.get(n, (1 << 62) + i) for i, n in enumerate(names)]
        live = self.core.finish(np.asarray(hs, np.int64), reason).tolist()
        self.core.check()
        return [self.name[h] for h in live]


def _total(plan) -> int:
    return int(plan.scheduled.sum())


# ---- basic scheduling, batching and budget (former scheduler/test_comprehensive.py) ----


def test_single_request_prefill():
    c = Core()
    c.add("r1", 100)
    plan, _ = c.step()
    assert _total(plan) == 100 and c.running() == 1 and c.waiting() == 0


def test_decode_generates_one_token():
    c = Core()
    c.prefill("r1", 100)
    assert c.sched(c.step()[0]) == {"r1": 1}


def test_async_counts_final_placeholder_against_max_tokens():
    c = Core(max_rows=4, token_budget=64, max_model_len=128, num_pages=64, page_size=16, async_scheduling=True)
    c.add("r1", 4, max_tokens=1)
    prefill = c.schedule()
    assert _total(prefill) == 4
    assert c.info("r1")[5] == 1 and c.info("r1")[3] == 4  # one placeholder, no output token yet
    stale = c.schedule()
    assert stale.n == 0
    ev = c.commit(prefill)
    assert ev.finish.tolist() == [FINISH_LENGTH] and c.info("r1") is None


def test_all_small_requests_prefilled_then_decoded():
    c = Core()
    for i in range(10):
        c.add(f"r{i}", 500)
    plan, _ = c.step()
    assert len(c.new(plan)) == 10 and _total(plan) == 5000
    assert _total(c.step()[0]) == 10


def test_budget_limits_admission():
    c = Core(token_budget=2048)
    for i in range(10):
        c.add(f"r{i}", 500)
    plan, _ = c.step()
    assert _total(plan) <= 2048 and c.waiting() > 0 and 1 <= len(c.new(plan)) < 10


def test_decode_runs_next_to_large_prefills():
    c = Core()
    c.prefill("decode0", 100)
    for i in range(5):
        c.add(f"large{i}", 20000)
    plan, _ = c.step()
    s = c.sched(plan)
    assert s["decode0"] == 1 and len(c.new(plan)) >= 1 and s.get("large0", 0) > 0


def test_decodes_and_prefill_share_budget():
    c = Core()
    for i in range(30):
        c.prefill(f"d{i}", 100)
    c.add("big", 20000)
    s = c.sched(c.step()[0])
    assert sum(1 for k in s if k.startswith("d")) == 30
    assert s["big"] == 8192 - 30 and sum(s.values()) <= 8192


def test_prompt_below_budget_is_skipped_when_it_does_not_fit_a_non_empty_batch():
    c = Core()
    for i in range(200):
        c.prefill(f"d{i}", 50)
    c.add("mid", 8000)
    assert "mid" not in c.sched(c.step()[0])


def test_prompt_at_budget_is_admitted_next_to_decodes():
    c = Core()
    c.prefill("d0", 100)
    c.add("exact", 20000)
    assert "exact" in c.sched(c.step()[0])


@pytest.mark.parametrize("max_tokens, steps", [(5, 10), (50, 55)])
def test_request_finishes_at_max_tokens(max_tokens, steps):
    c = Core()
    c.add("r1", 100, max_tokens=max_tokens)
    tokens = []
    for _ in range(steps):
        tokens += c.emitted(c.step()[1]).get("r1", [])
        if not c.unfinished():
            break
    assert c.unfinished() == 0 and len(tokens) == max_tokens and c.info("r1") is None


@pytest.mark.parametrize("prompt", [100, 1])
def test_request_finishes_on_eos(prompt):
    c = Core()
    c.add("r1", prompt, max_tokens=1000, eos=1)
    ev = c.commit(c.schedule(), token=1)
    assert ev.finish.tolist() == [FINISH_STOP] and ev.stop_token.tolist() == [-1] and c.unfinished() == 0


def test_external_abort_and_batch_abort():
    c = Core()
    c.add("r1", 100)
    c.add("r2", 100)
    c.step()
    assert c.finish(["r1"]) == ["r1"] and c.running() == 1
    assert c.finish(["nonexistent", "r1"]) == []  # stale and unknown handles are skipped
    assert c.finish(["r2"]) == ["r2"] and c.unfinished() == 0
    assert sorted(c.name[h] for h in c.schedule().finished.tolist()) == ["r1", "r2"]


def test_memory_pressure_still_admits():
    c = Core(num_pages=50, token_budget=65536)
    for i in range(5):
        c.add(f"r{i}", 500)
    assert c.new(c.step()[0])


def test_running_cap_is_max_rows():
    c = Core(max_rows=4, token_budget=65536)
    for i in range(10):
        c.add(f"r{i}", 50)
    c.step()
    assert c.running() == 4 and c.waiting() == 6


def test_empty_core_schedules_nothing():
    c = Core()
    plan, ev = c.step()
    assert plan.n == 0 and ev is None and c.unfinished() == 0


def test_chunked_prefill_spans_steps():
    c = Core(token_budget=2048, chunked_prefill=True)
    c.add("big", 10000)
    first = c.step()[0]
    assert _total(first) <= 2048 and c.running() == 1
    assert _total(c.step()[0]) > 0


def test_long_prefill_threshold_caps_the_chunk():
    c = Core(long_prefill_threshold=2048)
    c.add("big", 20000)
    assert _total(c.step()[0]) <= 2048
    c = Core(long_prefill_threshold=2048)
    for i in range(10):
        c.prefill(f"d{i}", 50)
    c.add("big", 20000)
    s = c.sched(c.step()[0])
    assert s.get("big", 0) <= 2048 and sum(1 for k in s if k.startswith("d")) == 10


def test_safety_margin_keeps_the_budget():
    c = Core(token_safety_margin=64)
    for i in range(10):
        c.prefill(f"d{i}", 50)
    c.add("new", 500)
    assert _total(c.step()[0]) <= 8192


def test_large_prefills_progress_over_steps():
    c = Core()
    for i in range(5):
        c.add(f"big{i}", 20000)
    per_step = [_total(c.step()[0]) for _ in range(10)]
    assert per_step[0] > 0 and sum(per_step) > per_step[0]


def test_every_decode_gets_a_token_next_to_a_large_prefill():
    c = Core()
    for i in range(20):
        c.prefill(f"d{i}", 100)
    c.add("big", 30000)
    s = c.sched(c.step()[0])
    assert sum(1 for k in s if k.startswith("d")) == 20 and 0 < s["big"] <= 8192 - 20


def test_waiting_request_is_admitted_after_a_finish():
    c = Core(max_rows=1)
    c.add("r1", 100)
    c.add("r2", 100)
    c.step()
    assert (c.running(), c.waiting()) == (1, 1)
    c.finish(["r1"])
    plan, _ = c.step()
    assert c.new(plan) == ["r2"] and c.running() == 1


def test_reset_prefix_cache_refuses_while_pages_are_in_use():
    c = Core(prefix_caching=True, page_size=4)
    assert c.core.reset_prefix_cache() is True
    c.add("r1", 8)
    plan = c.schedule()
    assert c.core.reset_prefix_cache() is False
    c.commit(plan)
    c.finish(["r1"])
    assert c.core.reset_prefix_cache() is True


def test_counts_through_the_lifecycle():
    c = Core()
    assert (c.running(), c.waiting()) == (0, 0)
    c.add("r1", 100)
    assert (c.running(), c.waiting()) == (0, 1)
    c.step()
    assert (c.running(), c.waiting()) == (1, 0)
    c.finish(["r1"])
    assert (c.running(), c.waiting()) == (0, 0)


def test_req_bucket_holds_the_running_requests():
    c = Core()
    for i in range(5):
        c.add(f"r{i}", 50)
    plan, _ = c.step()
    assert plan.req_bucket >= c.running() == 5


def test_priority_policy_schedules_everything_that_fits():
    c = Core(policy=1, token_budget=65536)
    for i in range(5):
        c.add(f"r{i}", 100)
    c.step()
    assert c.running() == 5


def test_early_decodes_and_late_prefills_interleave():
    c = Core(token_budget=4096, max_rows=32)
    for i in range(5):
        c.prefill(f"early{i}", 100)
    for i in range(5):
        c.add(f"late{i}", 300)
    s = c.sched(c.step()[0])
    assert sum(1 for k in s if k.startswith("early")) == 5
    assert sum(1 for k in s if k.startswith("late")) > 0 and sum(s.values()) <= 4096


# ---- prefix publication (former scheduler/test_comprehensive.py TestPrefixPublication) ----


def _probe_hit(c: Core, prompt) -> int:
    """The prefix hit a new request gets (its plan's computed tokens); the probe is aborted afterwards."""
    c.add("probe", prompt, max_tokens=2)
    plan = c.schedule()
    hit = dict(zip(plan.handles.tolist(), plan.computed.tolist(), strict=True)).get(c.handle["probe"])
    if plan.n:
        c.commit(plan)
    c.finish(["probe"])
    return hit


@pytest.mark.parametrize("committed, hit", [(False, 0), (True, 4)])
def test_identical_prompts_in_one_batch_never_reuse_uncomputed_pages(committed, hit):
    c = Core(prefix_caching=True, page_size=4)
    c.add("first", 8)
    c.add("second", 8)
    batch = c.schedule()
    assert batch.computed.tolist() == [0, 0]
    if committed:
        c.commit(batch)
        c.finish(["first", "second"])
    else:
        c.finish(["first", "second"])  # freed before the step's KV was committed: nothing was published
        c.commit(batch)
    assert _probe_hit(c, 8) == hit


@pytest.mark.parametrize("committed, hit", [(False, 0), (True, 4)])
def test_chunked_prefill_publishes_after_a_commit_without_sampled_token(committed, hit):
    c = Core(prefix_caching=True, page_size=4, token_budget=4, chunked_prefill=True)
    c.add("first", 8)
    batch = c.schedule()
    assert c.sched(batch) == {"first": 4}
    if committed:
        c.commit(batch, [[]])
        c.finish(["first"])
    else:
        c.finish(["first"])
        c.commit(batch, [[]])
    assert _probe_hit(c, 8) == hit


def test_sliding_window_overlap_never_publishes_the_null_page():
    c = Core(prefix_caching=True, page_size=4, kind=1, window=4, max_rows=4, token_budget=8, max_model_len=128,
             num_pages=64, chunked_prefill=True)  # fmt: skip
    c.add("a", 24)
    first, second = c.schedule(), c.schedule()
    c.commit(first, [[]])
    c.commit(second, [[]])
    c.finish(["a"])
    assert _probe_hit(c, 5) == 0


def test_sliding_window_decode_page_chains_to_its_true_parent():
    c = Core(prefix_caching=True, page_size=4, kind=1, window=4, max_rows=4, token_budget=64, max_model_len=128,
             num_pages=64, chunked_prefill=True)  # fmt: skip
    c.add("a", 8)
    for _ in range(6):
        c.step()  # page 2 holds the generated tokens [42] * 4
    assert _probe_hit(c, [42, 42, 42, 42, 7]) == 0


# ---- starvation (former scheduler/test_starvation.py) ----

_MIXED_SIZES = [500, 1000, 2000, 4000, 8000, 9000, 10000, 12000, 15000, 20000, 25000, 30000] * 2 + [
    500, 1000, 2000, 4000, 8000, 9000, 10000,
]  # fmt: skip


@pytest.mark.parametrize("sizes", [_MIXED_SIZES, [500] * 31, [20000] * 31], ids=["mixed", "small", "large"])
def test_waiting_prompts_are_not_starved_by_a_decode(sizes):
    c = Core()
    c.prefill("decode-0", 100)
    for i, n in enumerate(sizes):
        c.add(f"wait-{i}", n)
    plan = c.schedule()
    assert _total(plan) > 1 and len(c.new(plan)) >= 1


def test_many_decodes_and_large_waiting_prompts():
    c = Core()
    for i in range(50):
        c.prefill(f"d{i}", 100)
    for i in range(20):
        c.add(f"wait-{i}", 25000)
    plan = c.schedule()
    assert sum(1 for k in c.sched(plan) if k.startswith("d")) == 50 and len(c.new(plan)) >= 1


# ---- queue order (former scheduler/test_skipped_waiting_order.py and test_request_queue.py) ----


def test_skipped_waiting_requests_keep_fcfs_order():
    c = Core(max_rows=8, token_budget=100, max_model_len=256, num_pages=64, page_size=16)
    c.add("first", 50, max_tokens=8)
    for rid in ("skip-a", "skip-b", "skip-c"):
        c.add(rid, 60, max_tokens=8)
    first_step = {}
    for step in range(30):
        plan, _ = c.step()
        for rid in c.new(plan):
            first_step[rid] = step
    assert first_step["first"] == 0
    assert first_step["skip-a"] < first_step["skip-b"] < first_step["skip-c"]


def _admission_order(c: Core) -> list[str]:
    """Admits one request at a time (``max_rows=1``), aborting each once admitted: the queue's order."""
    order = []
    while c.unfinished():
        plan = c.schedule()
        new = c.new(plan)
        c.commit(plan)
        order += new
        c.finish(new)
    return order


def test_fcfs_queue_serves_in_arrival_order():
    c = Core(max_rows=1)
    for rid in ("a", "b", "c"):
        c.add(rid, 4)
    assert _admission_order(c) == ["a", "b", "c"]


def test_priority_queue_lowest_value_first_then_arrival_then_insertion():
    c = Core(max_rows=1, policy=1)
    c.add("p5", 4, priority=5, arrival=0.0)
    c.add("p1-late", 4, priority=1, arrival=2.0)
    c.add("p1-early", 4, priority=1, arrival=1.0)
    c.add("tie-a", 4, priority=3, arrival=1.0)
    c.add("tie-b", 4, priority=3, arrival=1.0)
    assert _admission_order(c) == ["p1-early", "p1-late", "tie-a", "tie-b", "p5"]


@pytest.mark.parametrize("policy", [0, 1])
def test_removing_a_waiting_request_keeps_the_others_in_order(policy):
    c = Core(max_rows=1, policy=policy)
    for i, rid in enumerate(("a", "b", "c", "d")):
        c.add(rid, 4, priority=1, arrival=float(i))
    assert c.finish(["b"]) == ["b"]
    assert _admission_order(c) == ["a", "c", "d"]


# ---- preemption (former test_priority_preemption_scheduled_victim.py, test_scheduler_lifecycle.py) ----


def test_priority_preemption_of_an_already_scheduled_victim_is_consistent():
    # 3 usable pages of 16 tokens. D0 (low urgency: priority 100) is admitted first, D1 (priority 1) second; both
    # then hold one page and need a second at the same step. D0 takes the last free page, D1 finds none and preempts
    # D0, which was already scheduled this step: D0 is un-scheduled and its tokens are refunded.
    c = Core(max_rows=8, token_budget=64, max_model_len=256, num_pages=4, page_size=16, token_safety_margin=0,
             policy=1)  # fmt: skip
    c.add("D0", 15, max_tokens=64, priority=100)
    c.step()
    c.add("D1", 16, max_tokens=64, priority=1)
    c.step()
    plan = c.schedule()
    assert c.sched(plan) == {"D1": 1}
    assert [c.name[h] for h in plan.preempted.tolist()] == ["D0"]
    assert c.info("D0")[1] == STATUS_PREEMPTED and c.info("D1")[1] == STATUS_RUNNING
    assert plan.stats[4] - plan.stats[5] == 1  # budget used: D1's token only
    c.commit(plan)


@pytest.mark.parametrize("margin", [0, 999])
def test_running_decode_finishes_with_no_free_pages_even_under_a_safety_reserve(margin):
    c = Core(max_rows=2, token_budget=8, max_model_len=32, num_pages=3, page_size=4, token_safety_margin=margin)
    c.add("running", [1] * 5, max_tokens=4, eos=0)
    first = c.schedule()
    assert c.sched(first) == {"running": 5}
    c.commit(first, token=10)
    assert c.core.counts()[2] == 0
    c.add("waiting", [1], max_tokens=4, eos=0)
    for _ in range(3):
        plan = c.schedule()
        assert c.sched(plan) == {"running": 1} and _total(plan) <= 8
        c.commit(plan, token=10)
    assert c.info("running") is None and c.core.counts()[2] == 2
    assert c.sched(c.schedule()) == {"waiting": 1}


def test_running_request_preempts_when_it_needs_another_page():
    c = Core(max_rows=2, token_budget=8, max_model_len=32, num_pages=3, page_size=4, token_safety_margin=0)
    c.add("older", [1] * 4, max_tokens=4, eos=0)
    c.add("newer", [1] * 4, max_tokens=4, eos=0)
    plan = c.schedule()
    assert c.sched(plan) == {"older": 4, "newer": 4}
    c.commit(plan, token=10)
    plan = c.schedule()
    assert c.sched(plan) == {"older": 1} and [c.name[h] for h in plan.preempted.tolist()] == ["newer"]


def test_request_buckets_cap_running_at_max_rows_and_report_the_bucket():
    c = Core(max_rows=2, token_budget=8, max_model_len=32, num_pages=32, page_size=4, req_buckets=np.asarray([4, 8]))
    for rid in ("a", "b", "c"):
        c.add(rid, [1], max_tokens=4, eos=0)
    plan = c.schedule()
    assert c.sched(plan) == {"a": 1, "b": 1}
    assert (c.running(), c.waiting()) == (2, 1) and plan.req_bucket == 4


# ---- whole-prompt admission (former scheduler/test_admission_full_prompt.py) ----


class _Trace:
    """Chunks, preemptions, re-prefilled positions and emitted tokens, step by step."""

    def __init__(self) -> None:
        self.chunks: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
        self.preempted: list[tuple[int, str]] = []
        self.reprefilled = 0
        self.tokens: dict[str, int] = defaultdict(int)
        self.first_token: dict[str, int] = {}
        self._high: dict[str, int] = {}

    def first_scheduled(self, rid: str) -> int:
        return self.chunks[rid][0][0]

    def record(self, step: int, c: Core, plan) -> None:
        for h, start, n in zip(plan.handles.tolist(), plan.computed.tolist(), plan.scheduled.tolist(), strict=True):
            rid, end = c.name[h], start + n
            high = self._high.get(rid, 0)
            self.reprefilled += max(0, min(end, high) - start)
            self._high[rid] = max(high, end)
            self.chunks[rid].append((step, start, end))
        self.preempted += [(step, rid) for rid in sorted(c.name[h] for h in plan.preempted.tolist())]

    def apply(self, step: int, c: Core, ev) -> None:
        for rid, toks in c.emitted(ev).items():
            self.tokens[rid] += len(toks)
            self.first_token.setdefault(rid, step)


def _adm_core(num_pages, *, k=0, kind=0, window=0) -> Core:
    return Core(max_rows=8, token_budget=64, max_model_len=4096, num_pages=num_pages, page_size=16, kind=kind,
                window=window, long_prefill_threshold=32, chunked_prefill=True, num_draft_tokens=k,
                use_eagle=k > 0)  # fmt: skip


def _adm_add(c: Core, rid, n, max_tokens=4, first=3):
    c.add(rid, [first] + [7] * (n - 1), max_tokens=max_tokens, ignore_eos=True, eos=2)


def _sync_step(c: Core, trace: _Trace, step: int):
    plan = c.schedule()
    trace.record(step, c, plan)
    if plan.n:
        trace.apply(step, c, c.commit(plan, token=5))
    return plan


def _sync_finish(c: Core, trace: _Trace, start: int, max_steps: int = 400) -> None:
    for step in range(start, start + max_steps):
        if not c.unfinished():
            return
        _sync_step(c, trace, step)
    raise AssertionError("requests left")


def test_running_prompt_keeps_its_pages_and_the_queue_stays_fcfs():
    c = _adm_core(14)
    trace = _Trace()
    _adm_add(c, "a", 160)
    _sync_step(c, trace, 0)
    _adm_add(c, "b", 64)
    _adm_add(c, "c", 16)
    plan = _sync_step(c, trace, 1)
    assert 4 <= c.core.counts()[2]  # b's 4 pages fit the free pool alone ...
    assert set(c.sched(plan)) == {"a"}  # ... but not next to the pages a's prompt still needs
    _sync_finish(c, trace, 2)
    a_done = max(step for step, _, end in trace.chunks["a"] if end <= 160)
    assert trace.first_scheduled("b") > a_done and trace.first_scheduled("c") >= trace.first_scheduled("b")
    assert trace.preempted == [] and trace.reprefilled == 0
    assert dict(trace.tokens) == {"a": 4, "b": 4, "c": 4}


@pytest.mark.parametrize("k, b_waits", [(0, False), (3, True)])
def test_lookahead_slots_count_for_the_waiting_and_the_running_prompt(k, b_waits):
    c = _adm_core(16, k=k)
    trace = _Trace()
    _adm_add(c, "a", 160, max_tokens=1)
    _sync_step(c, trace, 0)
    _adm_add(c, "b", 64, max_tokens=1)
    plan = _sync_step(c, trace, 1)
    assert ("b" not in c.sched(plan)) is b_waits
    _sync_finish(c, trace, 2)
    assert trace.preempted == [] and dict(trace.tokens) == {"a": 1, "b": 1}


def test_prompt_larger_than_the_free_pages_waits_until_they_are_free():
    c = _adm_core(14)
    trace = _Trace()
    _adm_add(c, "a", 64)
    _adm_add(c, "big", 160, first=4)
    _sync_finish(c, trace, 0)
    assert trace.first_scheduled("big") > max(step for step, _, _ in trace.chunks["a"])
    assert trace.preempted == [] and trace.reprefilled == 0 and dict(trace.tokens) == {"a": 4, "big": 4}


def test_prompt_larger_than_the_pool_is_admitted_when_nothing_runs():
    c = _adm_core(10)
    _adm_add(c, "huge", 400)
    _adm_add(c, "next", 16)
    plan = c.schedule()
    assert c.sched(plan) == {"huge": 32} and c.new(plan) == ["huge"]


def test_windowed_prompt_is_admitted_on_the_pages_it_holds_at_once():
    c = _adm_core(21, kind=1, window=64)
    trace = _Trace()
    _adm_add(c, "a", 1000)
    _sync_step(c, trace, 0)
    _adm_add(c, "b", 1000, first=4)
    free = int(c.core.counts()[2])
    plan = _sync_step(c, trace, 1)
    assert -(-1000 // 16) > free and "b" in c.sched(plan)
    _sync_finish(c, trace, 2)
    assert trace.preempted == [] and dict(trace.tokens) == {"a": 4, "b": 4}


@pytest.mark.parametrize("caching", [False, True])
@pytest.mark.parametrize("max_chunk", [32, 37])
@pytest.mark.parametrize("kind, window", [(1, 64), (1, 50), (2, 64), (2, 48)])
def test_windowed_group_counts_the_pages_it_holds_at_once(kind, window, max_chunk, caching):
    """A windowed group never holds more pages at once than the count made at admission, far below the full count."""
    kv = KvCore(num_pages=200, groups=[(kind, 16, window, 0)], max_model_len=4096, caching=caching, eagle=False)
    tokens = [11] + [7] * 999
    n = len(tokens)
    slot = None
    if caching:
        slot = kv.open(0)
        kv.lookup(slot, tokens, False, n - 1)
    needed = kv.needed(slot, n, None, max_chunk)
    assert kv.needed(slot, n, None, None) == -(-n // 16)
    assert needed * 4 < -(-n // 16)
    free_before, peak, computed = kv.num_free, 0, 0
    while computed < n:
        chunk = min(max_chunk, n - computed)
        if slot is None:
            slot = kv.open(0)
        assert kv.allocate(slot, computed, chunk, 0, None, 0) is not None
        peak = max(peak, free_before - kv.num_free)
        computed += chunk
        if caching:
            need = kv.publish(slot, computed)
            if need is not None:
                lo, hi = need
                kv.publish(slot, computed, tokens[lo:hi], lo)
        held = free_before - kv.num_free
        assert held + kv.needed(slot, n, None, max_chunk) <= needed
    assert 0 < peak <= needed
    kv.check()


_MIX_PROMPTS = (
    65442, 65441, 130628, 130636, 130620, 65447, 65443, 65450, 65451, 130625, 65441, 130618, 130624, 130634, 65446,
    130628, 65436, 130633, 65424, 65438, 65448, 130624, 65430, 65439, 65449, 65448, 65433, 65445, 65445, 130630,
    130621, 130621,
)  # fmt: skip


def _run_overlap(c: Core, max_steps: int) -> _Trace:
    """The engine loop's overlap order: a step whose rows are all one token is committed after the next schedule."""
    trace, pending = _Trace(), None
    for step in range(max_steps):
        if not c.unfinished() and pending is None:
            return trace
        plan = c.schedule()
        trace.record(step, c, plan)
        if pending is not None:
            trace.apply(step, c, c.commit(pending))
            pending = None
        if plan.n and all(n == 1 for n in plan.scheduled.tolist()):
            pending = plan
        elif plan.n:
            trace.apply(step, c, c.commit(plan, token=5))
    raise AssertionError("requests left")


@pytest.mark.parametrize("num_pages", [2826, 23565], ids=["small_pool", "large_pool"])
def test_long_prompt_mix_runs_without_preemption(num_pages):
    """32 prompts of 65K-131K tokens (about 23,000 pages) on a pool of 2,826 or 23,565 pages of 128 tokens."""
    c = Core(groups=[[0, 128, 0]], max_rows=32, token_budget=32768, max_model_len=131072, num_pages=num_pages,
             prefix_caching=True, req_buckets=np.asarray([32]), async_scheduling=True, chunked_prefill=True,
             snapshot_slots=16)  # fmt: skip
    ids = [f"mix-{i:02d}" for i in range(len(_MIX_PROMPTS))]
    for i, n in enumerate(_MIX_PROMPTS):
        prompt = np.full(n, 7, np.int32)
        prompt[0] = 1000 + i
        c.add(ids[i], prompt, max_tokens=64, ignore_eos=True, eos=2)
    trace = _run_overlap(c, 5000)
    assert trace.preempted == [] and trace.reprefilled == 0
    assert dict(trace.tokens) == dict.fromkeys(ids, 64)
    firsts = [trace.first_token[rid] for rid in ids]
    assert firsts == sorted(firsts)


# ---- data-parallel ranks (former test_dp_scheduler.py, test_dp_prefix_probe_no_leak.py, test_dp_scheduler_fixes.py,
# ---- test_scheduler_lifecycle.py) ----


def _dp_core(**kw) -> Core:
    args = dict(dp_size=2, max_rows=4, token_budget=16, max_model_len=64, num_pages=64, page_size=4)
    args.update(kw)
    return Core(**args)


def test_dp_routes_offsets_pages_and_commits_per_entry():
    c = _dp_core()
    c.add("r0", 8, max_tokens=8)
    c.add("r1", 8, max_tokens=8)
    assert (c.core.rank_of(c.handle["r0"]), c.core.rank_of(c.handle["r1"])) == (0, 1)
    plan = c.schedule()
    assert plan.rank_offsets.tolist() == [0, 1, 2] and _total(plan) == 16 and plan.stats[7] == 8
    assert plan.new_pages.tolist() == [1, 2, 33, 34]
    ev = c.commit(plan, [[100], [101]])
    assert c.emitted(ev) == {"r0": [100], "r1": [101]}
    nxt = c.schedule()
    assert [c.name[h] for h in nxt.handles.tolist()] == ["r0", "r1"] and nxt.kinds.tolist() == [0, 0]
    assert c.sched(nxt) == {"r0": 1, "r1": 1}  # each sampled token was appended once
    c.commit(nxt, [[100], [101]])
    assert c.info("r0")[3] == 10 and c.info("r1")[3] == 10


def test_dp_idle_schedule_does_not_desynchronize_the_next_commit():
    c = _dp_core()
    assert c.schedule().n == 0
    c.add("r0", 8)
    c.add("r1", 8)
    ev = c.commit(c.schedule(), [[100], [101]])
    assert c.emitted(ev) == {"r0": [100], "r1": [101]}


def test_dp_admission_probe_leaves_other_ranks_untouched():
    c = _dp_core(max_rows=8, token_budget=64, max_model_len=256, prefix_caching=True)
    for i in range(6):
        c.add(f"r{i}", 8, max_tokens=8)
    counts = c.core.counts().tolist()
    assert counts[4 + 2] == counts[8 + 2] == 31  # every page but the null page is still free on both ranks
    plan = c.schedule()
    ranks = {c.name[h]: c.core.rank_of(h) for h in plan.handles.tolist()}
    pages, offs = plan.new_pages.tolist(), plan.new_page_offsets.tolist()
    for i, h in enumerate(plan.handles.tolist()):
        lo = ranks[c.name[h]] * 32
        assert all(lo < p < lo + 32 for p in pages[offs[i] : offs[i + 1]])


def test_dp_structured_pending_flag_from_any_rank():
    c = _dp_core(async_scheduling=True)
    assert not c.schedule().flags & PLAN_STRUCTURED_PENDING
    c.add("plain", 4, dp_rank=0)
    c.add("grammar", 4, dp_rank=1, flags=FLAG_GRAMMAR)
    first = c.schedule()
    assert not first.flags & PLAN_STRUCTURED_PENDING
    second = c.schedule()  # the grammar request still has its first placeholder
    assert second.flags & PLAN_STRUCTURED_PENDING
    c.commit(first)
    c.commit(second)


@pytest.mark.parametrize("include_finished_set", [False, True])
def test_dp_natural_completion_releases_the_request(include_finished_set):
    c = _dp_core(max_rows=4, token_budget=8, max_model_len=32, num_pages=32, include_finished_set=include_finished_set)
    h = c.add("finished", [1, 1], max_tokens=1, eos=0)
    ev = c.commit(c.schedule(), token=10)
    assert c.emitted(ev) == {"finished": [10]} and ev.finish.tolist() == [FINISH_LENGTH]
    assert c.info("finished") is None and c.unfinished() == 0
    assert ev.finished_all.tolist() == ([h] if include_finished_set else [])
    assert c.schedule().finished.tolist() == [h]


@pytest.mark.parametrize("hold", [0, 8])
def test_dp_admission_waits_until_every_rank_with_waiting_requests_has_a_free_row(hold):
    c = _dp_core(dp_admission_hold=hold)
    c.add("a0", 4, max_tokens=2, dp_rank=0)
    c.add("a1", 4, max_tokens=32, dp_rank=0)
    c.add("b0", 4, max_tokens=32, dp_rank=1)
    c.step()
    c.add("a2", 4, max_tokens=32, dp_rank=0)
    c.add("b1", 4, max_tokens=32, dp_rank=1)
    plan, _ = c.step()  # rank 0 is full; a0 emits its second token and finishes at this commit
    if hold:
        assert c.new(plan) == [] and c.sched(plan) == {"a0": 1, "a1": 1, "b0": 1}
        plan, _ = c.step()
        assert sorted(c.new(plan)) == ["a2", "b1"]  # both ranks start their prefills in the same step
    else:
        assert c.new(plan) == ["b1"]
        assert c.new(c.step()[0]) == ["a2"]


def test_dp_admission_hold_lasts_at_most_the_configured_steps():
    c = _dp_core(dp_admission_hold=2)
    for name, rank in (("a0", 0), ("a1", 0), ("b0", 1)):
        c.add(name, 4, max_tokens=32, dp_rank=rank)
    c.step()
    c.add("a2", 4, max_tokens=32, dp_rank=0)
    c.add("b1", 4, max_tokens=32, dp_rank=1)
    assert [c.new(c.step()[0]) for _ in range(3)] == [[], [], ["b1"]]


def test_dp_admission_rank_with_nothing_running_admits_at_once():
    c = _dp_core(dp_admission_hold=64)
    c.add("a0", 4, max_tokens=32, dp_rank=0)
    c.add("a1", 4, max_tokens=32, dp_rank=0)
    c.step()
    c.add("a2", 4, max_tokens=32, dp_rank=0)
    c.add("b0", 4, max_tokens=32, dp_rank=1)
    assert c.new(c.step()[0]) == ["b0"]


def test_dp_admission_hold_must_not_be_negative():
    with pytest.raises(ValueError, match="dp_admission_hold"):
        _dp_core(dp_admission_hold=-1)


def test_image_span_longer_than_the_budget_is_rejected_and_nothing_is_admitted():
    c = _dp_core()
    with pytest.raises(ValueError):
        c.add("img", 30, spans=[(4, 24)])
    assert c.unfinished() == 0


def test_concurrent_callers_share_the_core():
    c = _dp_core(max_rows=8, token_budget=64)
    errors: list[BaseException] = []
    stop = threading.Event()

    def read() -> None:
        try:
            while not stop.is_set():
                c.core.counts()
        except BaseException as exc:
            errors.append(exc)

    reader = threading.Thread(target=read)
    reader.start()
    try:
        for i in range(8):
            c.add(f"r{i}", 8, max_tokens=4)
        for _ in range(20):
            c.step()
    finally:
        stop.set()
        reader.join(timeout=5)
    assert not errors and c.unfinished() == 0


def test_shutdown_closes_the_core():
    c = _dp_core()
    c.core.shutdown()
    with pytest.raises(CoreClosed):
        c.core.schedule()
    with pytest.raises(CoreClosed):
        c.core.counts()


# ---- speculation (former test_scheduler_spec_budget.py, test_scheduler_logprobs.py) ----


def test_drafts_are_scheduled_when_they_use_up_the_budget():
    c = Core(max_rows=4, token_budget=4, max_model_len=64, num_pages=64, page_size=4, num_draft_tokens=3)
    c.add("r0", [1, 2, 3], max_tokens=16, eos=0)
    first = c.schedule()
    assert c.sched(first) == {"r0": 3}
    c.commit(first, [[100]], drafts=[[7, 8, 9]])
    second = c.schedule()
    assert c.sched(second) == {"r0": 4} and second.flags & PLAN_HAS_VERIFY
    assert second.draft_tokens.tolist() == [7, 8, 9] and second.new_tokens.tolist() == [100]
    c.commit(second, [[101]], accepted=[0], drafts=[[]])
    assert c.info("r0")[4] == 4  # all three drafts rejected: one new computed token


def test_drafts_are_trimmed_to_the_budget():
    c = Core(max_rows=4, token_budget=3, max_model_len=64, num_pages=64, page_size=4, num_draft_tokens=3)
    c.add("r0", [1, 2, 3], max_tokens=16, eos=0)
    c.commit(c.schedule(), [[100]], drafts=[[7, 8, 9]])
    plan = c.schedule()
    assert c.sched(plan) == {"r0": 3} and plan.draft_tokens.tolist() == [7, 8]
    assert c.core.drafts_of(c.handle["r0"]).tolist() == [7, 8]


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("max_tokens", [3, 8])
def test_speculative_acceptance_rolls_back_and_stops_inside_drafts(async_scheduling, max_tokens):
    c = Core(max_rows=4, token_budget=16, max_model_len=64, num_pages=64, page_size=4,
             async_scheduling=async_scheduling, num_draft_tokens=3)  # fmt: skip
    c.add("spec", [1, 2], max_tokens=max_tokens, eos=0)
    c.commit(c.schedule(), [[10]], drafts=[[21, 22, 23]])
    plan = c.schedule()
    assert plan.draft_tokens.tolist() == [21, 22, 23] and c.sched(plan) == {"spec": 4}
    ev = c.commit(plan, [[21, 22, 24]], accepted=[2], drafts=[[]])
    assert ev.logprob_starts.tolist() == [0]
    if max_tokens == 3:
        assert c.emitted(ev) == {"spec": [21, 22]} and ev.finish.tolist() == [FINISH_LENGTH]
        assert c.info("spec") is None
    else:
        assert c.emitted(ev) == {"spec": [21, 22, 24]} and ev.finish.tolist() == [0]
        assert c.info("spec")[4] == 5  # 2 + 3 scheduled, one draft rejected


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize("stop_kind", ["length", "eos", "stop_token"])
def test_events_truncate_at_the_stop_and_keep_logprob_offsets(async_scheduling, stop_kind):
    c = Core(max_rows=4, token_budget=16, max_model_len=64, num_pages=64, page_size=4,
             async_scheduling=async_scheduling)  # fmt: skip
    retained = 1 if async_scheduling else 2
    stop = {"length": 22, "eos": 0, "stop_token": 55}[stop_kind]
    stops = [55] if stop_kind == "stop_token" else []
    for rid in ("stop", "keep", "prefill"):
        c.add(rid, [1, 2], max_tokens=retained if rid == "stop" and stop_kind == "length" else 8, eos=0, stops=stops)
    plan = c.schedule()
    rows = {"stop": [stop] if async_scheduling else [21, stop, 23], "keep": [31] if async_scheduling else [31, 32]}
    rows["prefill"] = []
    ev = c.commit(plan, [rows[c.name[h]] for h in plan.handles.tolist()])
    out = c.emitted(ev)
    assert set(out) == {"stop", "keep"}
    assert out["stop"] == rows["stop"][:retained] and out["keep"] == rows["keep"]
    i = ev.handles.tolist().index(c.handle["stop"])
    assert ev.finish[i] == (FINISH_LENGTH if stop_kind == "length" else FINISH_STOP)
    assert ev.stop_token[i] == (55 if stop_kind == "stop_token" else -1)
    offsets = np.cumsum([0] + [len(rows[c.name[h]]) for h in plan.handles.tolist()])
    for j, h in enumerate(ev.handles.tolist()):
        assert ev.logprob_starts[j] == offsets[plan.handles.tolist().index(h)]


def test_a_bad_commit_raises_and_changes_nothing():
    c = Core(max_rows=4, token_budget=16, max_model_len=64, num_pages=64, page_size=4)
    c.add("r", [1, 2], max_tokens=8, eos=0)
    plan = c.schedule()
    before = c.info("r")
    with pytest.raises(ValueError):
        c.core.commit(plan, sampled_tokens=np.asarray([11, 12], np.int32), sampled_offsets=np.asarray([0, 3], np.int32))
    assert c.info("r") == before and c.core.num_inflight() == 1
    assert c.emitted(c.commit(plan, [[11, 12]])) == {"r": [11, 12]}


def test_absent_entries_get_nothing_and_keep_their_drafts():
    c = Core(max_rows=4, token_budget=16, max_model_len=64, num_pages=64, page_size=4, num_draft_tokens=2)
    c.add("a", [1, 2], max_tokens=8, eos=0)
    c.add("b", [1, 2], max_tokens=8, eos=0)
    c.commit(c.schedule(), [[10], [10]], drafts=[[3, 4], [5, 6]])
    plan = c.schedule()
    ev = c.commit(plan, [[3, 7], []], accepted=[1, -1], drafts=[[8, 9], []], present=[1, 0])
    assert c.emitted(ev) == {"a": [3, 7]}
    assert c.core.drafts_of(c.handle["a"]).tolist() == [8, 9]
    assert c.core.drafts_of(c.handle["b"]).tolist() == [5, 6]
    assert c.info("b")[3] == 3  # no token appended to the absent entry


# ---- stop checks (former test_engine_stop_conditions.py check_stop tests) ----


def test_custom_stop_token_stops_and_names_itself():
    c = Core(max_rows=4, token_budget=16, max_model_len=4096, num_pages=64, page_size=4)
    c.add("r", [1, 2, 3], max_tokens=16, eos=0, stops=[42], ignore_eos=True)
    ev = c.commit(c.schedule(), [[42]])
    assert ev.finish.tolist() == [FINISH_STOP] and ev.stop_token.tolist() == [42]


def test_ignored_eos_among_the_stop_ids_does_not_stop():
    c = Core(max_rows=4, token_budget=16, max_model_len=4096, num_pages=64, page_size=4)
    c.add("r", [1, 2, 3], max_tokens=16, eos=2, stops=[2], ignore_eos=True)
    ev = c.commit(c.schedule(), [[2]])
    assert ev.finish.tolist() == [0] and c.info("r")[1] == STATUS_RUNNING


# ---- the KV core's scheduler traces (former scheduler/test_kv_core_goldens.py) ----


def _kv_scheduler_goldens():
    import json
    import pathlib

    with np.load(pathlib.Path(__file__).parent / "data" / "kv_goldens.npz") as npz:
        return json.loads(npz["cases"].tobytes())["schedulers"]


_KV_LAYOUTS = {"full": [(0, 0)], "full_sw8": [(0, 0), (1, 8)], "ch8_full": [(2, 8), (0, 0)]}
_KV_SCHED = _kv_scheduler_goldens()


@pytest.mark.parametrize("index", range(len(_KV_SCHED)), ids=[g["name"] for g in _KV_SCHED])
def test_kv_core_scheduler_trace_reproduces(index):
    golden = _KV_SCHED[index]
    _name, is_async, layout, ps, num_pages, snaps, budget, max_seqs = golden["cfg"]
    c = Core(groups=[[k, ps, w] for k, w in _KV_LAYOUTS[layout]], max_rows=max_seqs, token_budget=budget,
             max_model_len=256, num_pages=num_pages, prefix_caching=True, async_scheduling=is_async,
             chunked_prefill=True, snapshot_slots=snaps)  # fmt: skip
    rng = np.random.default_rng(golden["seed"])
    for _ in range(2):
        rng.integers(3, 60, size=6 * ps * 4)
    for _ in range(12):
        rng.integers(2)
        rng.integers(0, 5 * ps * 2)
        rng.integers(60, 120, size=int(rng.integers(1, 3 * ps)))
    rng.integers(0, 30, size=12)
    prompts, arrivals = golden["prompts"], golden["arrivals"]
    for step, want in enumerate(golden["trace"]):
        for i, a in enumerate(arrivals):
            if a == step:
                c.add(f"q{i}", list(prompts[i]), max_tokens=int(rng.integers(2, 9)), eos=1)
        plan = c.schedule()
        hs, comp, kinds = plan.handles.tolist(), plan.computed.tolist(), plan.kinds.tolist()
        ng = len(_KV_LAYOUTS[layout])
        pages, offs = plan.new_pages.tolist(), plan.new_page_offsets.tolist()
        per = [[pages[offs[i * ng + g] : offs[i * ng + g + 1]] for g in range(ng)] for i in range(len(hs))]
        sampled = []
        for h in hs:
            info = c.core.request_info(h)
            produced = info[5] > 0 if is_async else info[4] >= info[2]
            sampled.append([int(rng.integers(5, 120))] if produced else [])
        cached = [i for i, k in enumerate(kinds) if k == KIND_RUNNING] + [
            i for i, k in enumerate(kinds) if k == KIND_RESUMED
        ]
        got = [
            [[c.name[h], n] for h, n in zip(hs, plan.scheduled.tolist(), strict=True)],
            [[c.name[hs[i]], per[i], comp[i]] for i, k in enumerate(kinds) if k == KIND_NEW],
            [[c.name[hs[i]], per[i]] for i in cached],
            sorted(c.name[h] for h in plan.preempted.tolist()),
            sorted([c.name[hs[e]], s] for e, s in plan.snapshot_restores.tolist()),
            sorted([c.name[hs[e]], s] for e, s in plan.snapshot_saves.tolist()),
            int(plan.stats[3]),
            plan.common_prefix_pages.tolist(),
        ]
        assert got == want, (golden["name"], step)
        if plan.n:
            c.commit(plan, sampled)


# ---- bucket-aligned prefill chunks (set_step_cost) ----

_COST_BUCKETS = np.asarray([4, 8, 16, 32, 64], np.int64)
_COST_MS = np.asarray([10.0, 10.0, 10.0, 40.0, 80.0])


def _decoding_core(n=3, **kw) -> Core:
    """A core whose ``n`` requests have finished their 4-token prompts and decode one token per step."""
    args = dict(max_rows=8, token_budget=64, max_model_len=256, num_pages=256, page_size=4, chunked_prefill=True)
    args.update(kw)
    c = Core(**args)
    for i in range(n):
        c.add(f"d{i}", [10 + i, 11, 12, 13], max_tokens=64)
    plan, _ = c.step()
    assert plan.scheduled.tolist() == [4] * n
    return c


def _scheduled(c: Core, plan) -> dict[str, int]:
    return {c.name[h]: n for h, n in zip(plan.handles.tolist(), plan.scheduled.tolist(), strict=True)}


def test_step_cost_trims_a_prefill_chunk_to_the_bucket_below_and_defers_the_rest():
    # 3 decode rows + a 20-token prompt = 23 tokens, bucket 32 (40 ms). Bucket 16 costs 10 ms and the 7 deferred
    # tokens' share of bucket 8 costs 8.75 ms, so the prompt gives 7 tokens to the next step.
    c = _decoding_core()
    c.core.set_step_cost(_COST_BUCKETS, _COST_MS)
    c.add("p", list(range(100, 120)), max_tokens=64)
    plan = c.schedule()
    assert _scheduled(c, plan) == {"d0": 1, "d1": 1, "d2": 1, "p": 13}
    assert plan.emits.tolist() == [1, 1, 1, 0]
    c.commit(plan)
    # 3 + 7 = 10 tokens: cutting 2 to reach bucket 8 saves nothing (10 ms + 2 tokens of bucket 4 > 0.95 * 10 ms).
    plan = c.schedule()
    assert _scheduled(c, plan) == {"d0": 1, "d1": 1, "d2": 1, "p": 7}
    assert plan.emits.tolist() == [1, 1, 1, 1]
    c.commit(plan)


@pytest.mark.parametrize("case", ["no_table", "flat_table", "no_chunked_prefill"])
def test_step_cost_leaves_steps_it_cannot_speed_up(case):
    c = _decoding_core(chunked_prefill=case != "no_chunked_prefill")
    if case == "flat_table":
        c.core.set_step_cost(_COST_BUCKETS, np.full(5, 10.0))
    elif case == "no_chunked_prefill":
        c.core.set_step_cost(_COST_BUCKETS, _COST_MS)
    c.add("p", list(range(100, 120)), max_tokens=64)
    assert _scheduled(c, c.schedule()) == {"d0": 1, "d1": 1, "d2": 1, "p": 20}


def test_step_cost_never_trims_decode_rows():
    # 5 decode rows run bucket 8 at 100 ms against 10 ms for bucket 4, but no row has a token to give back.
    c = _decoding_core(n=5)
    c.core.set_step_cost(np.asarray([4, 8, 32]), np.asarray([10.0, 100.0, 200.0]))
    assert _scheduled(c, c.schedule()) == {f"d{i}": 1 for i in range(5)}


def test_a_trimmed_chunk_hands_its_snapshot_save_to_the_next_chunk():
    c = _decoding_core(prefix_caching=True, snapshot_slots=4)
    c.core.set_step_cost(_COST_BUCKETS, _COST_MS)
    # 21 tokens: the snapshot boundary is 20 (the last full page before the last token). The chunk that would end
    # there is trimmed to 13 and gives up its save; the next chunk ends on the boundary and saves.
    c.add("p", list(range(100, 121)), max_tokens=64)
    plan = c.schedule()
    assert _scheduled(c, plan) == {"d0": 1, "d1": 1, "d2": 1, "p": 13}
    assert plan.snapshot_saves.tolist() == []
    c.commit(plan)
    plan = c.schedule()
    assert _scheduled(c, plan) == {"d0": 1, "d1": 1, "d2": 1, "p": 7}
    handles = plan.handles.tolist()
    assert [c.name[handles[e]] for e, _ in plan.snapshot_saves.tolist()] == ["p"]
    c.commit(plan)
    plan = c.schedule()
    assert _scheduled(c, plan) == {"d0": 1, "d1": 1, "d2": 1, "p": 1}
    assert plan.emits.tolist() == [1, 1, 1, 1]
    c.commit(plan)


def test_step_cost_trims_the_busiest_rank_of_a_data_parallel_step():
    c = _dp_core(chunked_prefill=True)
    c.add("a0", 4, max_tokens=8, dp_rank=0)
    c.add("a1", 4, max_tokens=8, dp_rank=1)
    c.step()
    # A step's tokens are its busiest rank's times 2: rank 0 holds 1 + 10 = 11 tokens, a 32-token step. Bucket 16
    # allows 8 tokens per rank, so the prompt gives 3 tokens to the next step.
    c.core.set_step_cost(np.asarray([4, 8, 16, 32]), np.asarray([10.0, 10.0, 10.0, 40.0]))
    c.add("p0", list(range(100, 110)), max_tokens=8, dp_rank=0)
    assert _scheduled(c, c.schedule()) == {"a0": 1, "a1": 1, "p0": 7}


def test_set_step_cost_refuses_bad_tables():
    c = Core()
    for buckets, ms in (([], []), ([8, 4], [1.0, 1.0]), ([4], [0.0]), ([4, 8], [1.0])):
        with pytest.raises(ValueError, match="step"):
            c.core.set_step_cost(np.asarray(buckets, np.int64), np.asarray(ms, np.float64))
