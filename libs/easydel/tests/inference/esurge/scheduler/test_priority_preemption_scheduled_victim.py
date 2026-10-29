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

"""Regression test for priority preemption of an already-scheduled request (S1).

Under ``policy="priority"`` the preemption victim is chosen as
``max(running, key=(priority, arrival_time))``. When running order is not
priority-sorted (a low-urgency request admitted in an earlier step sits ahead
of a high-urgency one), that victim can be a request that was *already
scheduled* earlier in the very same step. The old code freed the victim's
pages but never removed it from ``scheduled_running_reqs`` /
``num_scheduled_tokens`` / ``req_to_new_page_ids`` nor refunded its budget or
rewound ``req_index`` -> the "total scheduled <= running" invariant raised
``RuntimeError`` (or emitted corrupt slices).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
from easydel.inference.esurge.core.interface import CacheGroupsConfig, CacheGroupSpec, FullAttentionSpec
from easydel.inference.esurge.request import EngineRequest, EngineRequestStatus
from easydel.inference.esurge.scheduler.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams

PAGE_SIZE = 16
NUM_PAGES = 4  # null page 0 + pages 1,2 (the two decoders) + page 3 (single free page)


def _make_priority_scheduler() -> Scheduler:
    kv_cache_config = CacheGroupsConfig(
        num_pages=NUM_PAGES,
        kv_cache_groups=[
            CacheGroupSpec(
                kv_cache_spec=FullAttentionSpec(
                    page_size=PAGE_SIZE,
                    num_kv_heads=1,
                    head_size=4,
                    dtype=jnp.float32,
                    use_mla=False,
                ),
                layer_names=None,
            )
        ],
    )
    return Scheduler(
        kv_cache_config=kv_cache_config,
        max_num_seqs=8,
        max_num_batched_tokens=64,
        max_model_len=256,
        num_pages=NUM_PAGES,
        page_size=PAGE_SIZE,
        enable_prefix_caching=False,
        async_scheduling=False,
        token_safety_margin=0,  # exercises the TokenBudgetManager.release() path
        policy="priority",
    )


def _install_running_decoder(scheduler: Scheduler, rid: str, priority: int) -> EngineRequest:
    """Register a fully-prefilled decode request occupying exactly one page.

    Its prompt fills one full page (PAGE_SIZE tokens) and it holds one pending
    sampled token, so its next decode step needs exactly one new page.
    """
    req = EngineRequest(
        request_id=rid,
        prompt_token_ids=list(range(PAGE_SIZE)),
        sampling_params=SamplingParams(max_tokens=64),
        eos_token_id=1,
        priority=priority,
    )
    pages = scheduler.kv_cache_manager.allocate_slots(req, num_new_tokens=PAGE_SIZE)
    assert pages is not None
    req.num_computed_tokens = PAGE_SIZE
    req.append_output_token_ids(4242)  # one sampled token pending -> boundary crossing next step
    req.status = EngineRequestStatus.RUNNING
    scheduler.requests[rid] = req
    scheduler.running.append(req)
    return req


def test_priority_preemption_of_already_scheduled_victim_is_consistent():
    scheduler = _make_priority_scheduler()

    # running order = [D0, D1]. D0 is low urgency (high priority *value*), so it
    # is the preemption victim even though it is processed (and scheduled)
    # first in the running loop.
    d0 = _install_running_decoder(scheduler, "D0", priority=100)
    d1 = _install_running_decoder(scheduler, "D1", priority=1)

    # Exactly one free page remains: D0's decode consumes it, then D1's decode
    # finds none and must preempt -> victim = D0 (already scheduled this step).
    assert scheduler.kv_cache_manager.page_pool.get_num_free_pages() == 1

    out = scheduler.schedule()  # must not raise RuntimeError

    scheduled_ids = set(out.num_scheduled_tokens)
    running_ids = {r.request_id for r in scheduler.running}

    # Invariant: everything scheduled is still running; counts are consistent.
    assert scheduled_ids <= running_ids
    assert out.total_num_scheduled_tokens == sum(out.num_scheduled_tokens.values())

    # The high-urgency request kept its slot; the low-urgency victim was evicted.
    assert d1.request_id in scheduled_ids
    assert d0.request_id not in scheduled_ids
    assert running_ids == {d1.request_id}
    assert d0.status == EngineRequestStatus.PREEMPTED
    assert any(r.request_id == d0.request_id for r in scheduler.waiting)
