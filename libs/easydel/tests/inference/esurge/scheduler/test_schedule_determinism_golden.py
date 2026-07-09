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

"""Characterization: ``schedule()`` is a deterministic function of the request stream.

A scripted arrival pattern (mixed prompt lengths, chunked prefill, budget
pressure, staggered admissions) must produce the exact same per-step schedule
shape on every run and across refactors. TPU bench comparisons key on
per-bucket step histograms, which are only comparable when the schedule is
deterministic — this test turns schedule-shape drift into a CPU failure with a
readable diff instead of a confusing bench delta.

The multi-host unified control plane additionally *depends* on determinism:
workers replay the leader's SchedulerOutput stream, so any nondeterminism here
is a lockstep-divergence bug, not just a bench nuisance.

The goldens encode current behavior at the time of writing. If a deliberate
scheduler change breaks them, regenerate by running the scenario and updating
the literals — with a commit message explaining the schedule-shape change.
"""

from __future__ import annotations

import jax.numpy as jnp

from easydel.inference.esurge.core.interface import (
    CacheGroupsConfig,
    CacheGroupSpec,
    FullAttentionSpec,
)
from easydel.inference.esurge.outputs import ModelRunnerOutput
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.scheduler.async_scheduler import AsyncScheduler
from easydel.inference.esurge.scheduler.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams

PROMPT_LENS = [7, 33, 1, 64, 16]
ARRIVALS = {0: [0, 1], 3: [2], 5: [3, 4]}
NUM_STEPS = 40
MAX_TOKENS = 8

IDLE_ROW = ([], [], [], [], 0)

# fmt: off
GOLDEN_SYNC = [
    ([("req-0", 7), ("req-1", 25)], ["req-0", "req-1"], [], [], 32),
    ([("req-0", 1), ("req-1", 8)], [], [], [], 9),
    ([("req-0", 1), ("req-1", 1)], [], [], [], 2),
    ([("req-0", 1), ("req-1", 1), ("req-2", 1)], ["req-2"], [], [], 3),
    ([("req-0", 1), ("req-1", 1), ("req-2", 1)], [], [], [], 3),
    ([("req-0", 1), ("req-1", 1), ("req-2", 1), ("req-3", 29)], ["req-3"], [], [], 32),
    ([("req-0", 1), ("req-1", 1), ("req-2", 1), ("req-3", 29)], [], [], [], 32),
    ([("req-0", 1), ("req-1", 1), ("req-2", 1), ("req-3", 6)], [], [], [], 9),
    ([("req-1", 1), ("req-2", 1), ("req-3", 1), ("req-4", 16)], ["req-4"], ["req-0"], [], 19),
    ([("req-2", 1), ("req-3", 1), ("req-4", 1)], [], ["req-1"], [], 3),
    ([("req-2", 1), ("req-3", 1), ("req-4", 1)], [], [], [], 3),
    ([("req-3", 1), ("req-4", 1)], [], ["req-2"], [], 2),
    ([("req-3", 1), ("req-4", 1)], [], [], [], 2),
    ([("req-3", 1), ("req-4", 1)], [], [], [], 2),
    ([("req-3", 1), ("req-4", 1)], [], [], [], 2),
    ([("req-4", 1)], [], ["req-3"], [], 1),
    ([], [], ["req-4"], [], 0),
]

GOLDEN_ASYNC = [
    ([("req-0", 7), ("req-1", 25)], ["req-0", "req-1"], [], [], 32),
    ([("req-0", 1), ("req-1", 8)], [], [], [], 9),
    ([("req-0", 1), ("req-1", 1)], [], [], [], 2),
    ([("req-0", 1), ("req-1", 1), ("req-2", 1)], ["req-2"], [], [], 3),
    ([("req-0", 1), ("req-1", 1), ("req-2", 1)], [], [], [], 3),
    ([("req-0", 1), ("req-1", 1), ("req-2", 1), ("req-3", 29)], ["req-3"], [], [], 32),
    ([("req-0", 1), ("req-1", 1), ("req-2", 1), ("req-3", 29)], [], [], [], 32),
    ([("req-0", 1), ("req-1", 1), ("req-2", 1), ("req-3", 6)], [], [], [], 9),
    ([("req-1", 1), ("req-2", 1), ("req-3", 1), ("req-4", 16)], ["req-4"], ["req-0"], [], 19),
    ([("req-2", 1), ("req-3", 1), ("req-4", 1)], [], ["req-1"], [], 3),
    ([("req-2", 1), ("req-3", 1), ("req-4", 1)], [], [], [], 3),
    ([("req-3", 1), ("req-4", 1)], [], ["req-2"], [], 2),
    ([("req-3", 1), ("req-4", 1)], [], [], [], 2),
    ([("req-3", 1), ("req-4", 1)], [], [], [], 2),
    ([("req-3", 1), ("req-4", 1)], [], [], [], 2),
    ([("req-4", 1)], [], ["req-3"], [], 1),
    ([], [], ["req-4"], [], 0),
]
# fmt: on


def _make_kv() -> CacheGroupsConfig:
    return CacheGroupsConfig(
        num_pages=64,
        kv_cache_groups=[
            CacheGroupSpec(
                kv_cache_spec=FullAttentionSpec(
                    page_size=16,
                    num_kv_heads=1,
                    head_size=4,
                    dtype=jnp.float32,
                    use_mla=False,
                ),
                layer_names=None,
            )
        ],
    )


def _make_scheduler(async_mode: bool) -> Scheduler:
    cls = AsyncScheduler if async_mode else Scheduler
    return cls(
        kv_cache_config=_make_kv(),
        max_num_seqs=4,
        max_num_batched_tokens=32,
        max_model_len=128,
        num_pages=64,
        page_size=16,
        enable_prefix_caching=False,
        async_scheduling=async_mode,
        chunked_prefill_enabled=True,
    )


def _make_req(i: int) -> EngineRequest:
    return EngineRequest(
        request_id=f"req-{i}",
        prompt_token_ids=list(range(PROMPT_LENS[i])),
        sampling_params=SamplingParams(max_tokens=MAX_TOKENS),
        eos_token_id=1,
    )


def _fake_output(sched: Scheduler, sched_out, async_mode: bool) -> ModelRunnerOutput:
    """Runner-faithful fake: only requests whose prompt is fully computed sample.

    ``schedule()`` advances ``num_computed_tokens`` optimistically, so a
    mid-prefill chunk shows ``num_computed_tokens < num_prompt_tokens`` here
    and must emit an empty token list (the real runner samples nothing for
    it). Async scheduling exposes the same fact via output placeholders.
    """
    req_ids = list(sched_out.num_scheduled_tokens.keys())
    sampled: list[list[int]] = []
    for rid in req_ids:
        req = sched.requests[rid]
        if async_mode:
            produced = req.num_output_placeholders > 0
        else:
            produced = req.num_computed_tokens >= req.num_prompt_tokens
        sampled.append([42] if produced else [])
    return ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=sampled,
        spec_token_ids=None,
        logprobs=None,
        prompt_logprobs_dict={},
    )


def _run_scenario(async_mode: bool) -> list[tuple]:
    sched = _make_scheduler(async_mode)
    trace = []
    for step_idx in range(NUM_STEPS):
        for req_idx in ARRIVALS.get(step_idx, []):
            sched.add_request(_make_req(req_idx))
        out = sched.schedule()
        if out.total_num_scheduled_tokens > 0:
            sched.update_from_output(out, _fake_output(sched, out, async_mode))
        trace.append(
            (
                sorted(out.num_scheduled_tokens.items()),
                sorted(r.req_id for r in out.scheduled_new_reqs),
                sorted(out.finished_req_ids),
                sorted(out.preempted_req_ids),
                int(out.total_num_scheduled_tokens),
            )
        )
    return trace


def _normalize(row: tuple) -> tuple:
    """Compare list-typed goldens against the recorded tuples uniformly."""
    sched_items, new_ids, finished, preempted, total = row
    return (list(map(tuple, sched_items)), list(new_ids), list(finished), list(preempted), int(total))


def test_sync_schedule_matches_golden_and_is_reproducible():
    first = _run_scenario(async_mode=False)
    second = _run_scenario(async_mode=False)
    assert first == second, "two fresh sync schedulers diverged on an identical request stream"

    golden = [_normalize(r) for r in GOLDEN_SYNC]
    recorded = [_normalize(r) for r in first[: len(golden)]]
    assert recorded == golden
    assert all(_normalize(r) == _normalize(IDLE_ROW) for r in first[len(golden) :]), (
        "scheduler produced work after all requests finished"
    )


def test_async_schedule_matches_golden_and_is_reproducible():
    first = _run_scenario(async_mode=True)
    second = _run_scenario(async_mode=True)
    assert first == second, "two fresh async schedulers diverged on an identical request stream"

    golden = [_normalize(r) for r in GOLDEN_ASYNC]
    recorded = [_normalize(r) for r in first[: len(golden)]]
    assert recorded == golden
    assert all(_normalize(r) == _normalize(IDLE_ROW) for r in first[len(golden) :])
