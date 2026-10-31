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

"""Regression test for FCFS ordering of skipped-then-requeued waiting requests
(bug S4).

Requests skipped in one scheduling step (e.g. too large for the remaining
token budget with chunked prefill disabled) are re-queued to the front of the
waiting queue. The accumulation used ``prepend_request`` (appendleft) while the
requeue used ``prepend_requests`` (extendleft(reversed(...))), so the net
effect reversed the relative FCFS order of the skipped requests. Their original
arrival order must be preserved.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
from easydel.inference.esurge.core.interface import CacheGroupsConfig, CacheGroupSpec, FullAttentionSpec
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.esurge.scheduler.scheduler import Scheduler
from easydel.inference.sampling_params import SamplingParams


def _make_scheduler() -> Scheduler:
    kv_cache_config = CacheGroupsConfig(
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
    return Scheduler(
        kv_cache_config=kv_cache_config,
        max_num_seqs=8,
        max_num_batched_tokens=100,  # small per-step token budget
        max_model_len=256,
        num_pages=64,
        page_size=16,
        enable_prefix_caching=False,
        async_scheduling=False,
        chunked_prefill_enabled=False,
        policy="fcfs",
    )


def _request(rid: str, prompt_len: int) -> EngineRequest:
    return EngineRequest(
        request_id=rid,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=8),
        eos_token_id=1,
    )


def test_skipped_waiting_requests_keep_fcfs_order():
    scheduler = _make_scheduler()

    # "first" fits the budget and is scheduled (batch becomes non-empty). The
    # remaining three each exceed the leftover budget (60 > 100 - 50) but are
    # not inherently too large (60 < 100), so they are skipped and requeued.
    scheduler.add_request(_request("first", prompt_len=50))
    for rid in ("skip-a", "skip-b", "skip-c"):
        scheduler.add_request(_request(rid, prompt_len=60))

    out = scheduler.schedule()

    # "first" was scheduled this step.
    assert "first" in out.num_scheduled_tokens

    # The three skipped requests are back in the waiting queue in their
    # original arrival order -- not reversed.
    waiting_ids = [r.request_id for r in scheduler.waiting]
    assert waiting_ids == ["skip-a", "skip-b", "skip-c"]
