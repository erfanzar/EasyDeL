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

"""eSurge's ``CacheManager`` and ``Scheduler`` reproduce the former Python KV core's decisions exactly.

The goldens (``libs/surgecore/tests/data/kv_goldens.npz``) were captured from the former Python KV core. Every
``CacheManager`` case is replayed through the real binding with surgecore's ``replay_case``; every scheduler trace is
re-run (same prompts, arrivals, sampled tokens and runner order) and compared step by step: scheduled tokens, new
requests with their page ids and hit lengths, cached requests' new page ids, preemptions, snapshot restores and saves,
free pages and common-prefix pages.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest
from easydel.inference.esurge.core import interface as itf
from easydel.inference.esurge.core.manager import CacheManager
from easydel.inference.esurge.outputs import ModelRunnerOutput
from easydel.inference.esurge.request import EngineRequest, EngineRequestStatus
from easydel.inference.esurge.scheduler.async_scheduler import AsyncScheduler
from easydel.inference.esurge.scheduler.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_ok

_PARITY = pathlib.Path(__file__).resolve().parents[5] / "surgecore" / "tests" / "test_kv_parity.py"
_spec = importlib.util.spec_from_file_location("surgecore_kv_parity", _PARITY)
_parity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_parity)
_HEADER, _BODY, _TOKENS = _parity.load_goldens()

# The capture's layouts by name: (kind, window_or_chunk, head_size); head size separates cache types.
_LAYOUTS = {"full": [(0, 0, 4)], "full_sw8": [(0, 0, 4), (1, 8, 4)], "ch8_full": [(2, 8, 4), (0, 0, 4)]}


def _spec_of(kind: int, page_size: int, window: int, head_size: int):
    common = dict(page_size=page_size, num_kv_heads=1, head_size=head_size, dtype=np.float32, use_mla=False)
    if kind == 0:
        return itf.FullAttentionSpec(**common)
    if kind == 1:
        return itf.SlidingWindowSpec(**common, sliding_window=window)
    return itf.ChunkedLocalAttentionSpec(**common, attention_chunk_size=window)


def _manager(groups, num_pages, max_model_len, caching, eagle) -> CacheManager:
    specs = [itf.CacheGroupSpec(_spec_of(k, ps, w, 4 + 4 * t)) for k, ps, w, t in groups]
    return CacheManager(num_pages, specs, max_model_len, enable_caching=caching, use_eagle=eagle)


@pytest.mark.parametrize("index", range(len(_BODY["cases"])))
def test_cache_manager_matches_python_goldens(index):
    _parity.replay_case(_BODY["cases"][index], _TOKENS, _manager, EngineRequestStatus.RUNNING)


def _run_scheduler(golden: dict) -> list:
    _name, is_async, layout, ps, num_pages, snaps, budget, max_seqs = golden["cfg"]
    rng = np.random.default_rng(golden["seed"])
    groups = [itf.CacheGroupSpec(_spec_of(k, ps, w, h)) for k, w, h in _LAYOUTS[layout]]
    sched = (AsyncScheduler if is_async else Scheduler)(
        kv_cache_config=itf.CacheGroupsConfig(num_pages=num_pages, kv_cache_groups=groups),
        max_num_seqs=max_seqs,
        max_num_batched_tokens=budget,
        max_model_len=256,
        num_pages=num_pages,
        page_size=ps,
        enable_prefix_caching=True,
        async_scheduling=is_async,
        chunked_prefill_enabled=True,
        recurrent_prefix_snapshot_slots=snaps,
    )
    # Replay the capture's random stream: prompt generation draws come first, then max_tokens and sampled tokens.
    for _ in range(2):
        rng.integers(3, 60, size=6 * ps * 4)
    for _ in range(12):
        rng.integers(2)
        rng.integers(0, 5 * ps * 2)
        rng.integers(60, 120, size=int(rng.integers(1, 3 * ps)))
    rng.integers(0, 30, size=12)
    prompts, arrivals = golden["prompts"], golden["arrivals"]
    trace = []
    for step in range(70):
        for i, a in enumerate(arrivals):
            if a == step:
                sched.add_request(
                    EngineRequest(
                        request_id=f"q{i}",
                        prompt_token_ids=list(prompts[i]),
                        sampling_params=SamplingParams(max_tokens=int(rng.integers(2, 9))),
                        eos_token_id=1,
                    )
                )
        out = sched.schedule()
        req_ids = list(out.num_scheduled_tokens)
        sampled = []
        for rid in req_ids:
            req = sched.requests[rid]
            produced = req.num_output_placeholders > 0 if is_async else req.num_computed_tokens >= req.num_prompt_tokens
            sampled.append([int(rng.integers(5, 120))] if produced else [])
        cached = out.scheduled_cached_reqs
        trace.append(
            [
                [[k, int(v)] for k, v in out.num_scheduled_tokens.items()],
                [[r.req_id, [list(g) for g in r.page_ids], int(r.num_computed_tokens)] for r in out.scheduled_new_reqs],
                [
                    [rid, [list(g) for g in pg] if pg is not None else None]
                    for rid, pg in zip(cached.req_ids, cached.new_page_ids, strict=True)
                ],
                sorted(out.preempted_req_ids),
                [list(x) for x in sorted(out.recurrent_snapshot_restores.items())],
                [list(x) for x in sorted(out.recurrent_snapshot_saves.items())],
                int(out.free_pages),
                [int(x) for x in out.num_common_prefix_pages],
            ]
        )
        if out.total_num_scheduled_tokens > 0:
            sched.update_from_output(
                out,
                ModelRunnerOutput(
                    req_ids=req_ids,
                    req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
                    sampled_token_ids=sampled,
                    spec_token_ids=None,
                    logprobs=None,
                    prompt_logprobs_dict={},
                ),
            )
    return trace


@pytest.mark.parametrize("index", range(len(_BODY["schedulers"])))
def test_scheduler_matches_python_goldens(index):
    golden = _BODY["schedulers"][index]
    trace = _run_scheduler(golden)
    for step, (got, want) in enumerate(zip(trace, golden["trace"], strict=True)):
        assert got == want, (golden["name"], step)
