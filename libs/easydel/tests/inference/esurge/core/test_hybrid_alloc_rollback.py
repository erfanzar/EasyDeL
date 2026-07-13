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

"""Regression test for hybrid multi-group allocation-failure rollback (bug C3).

With a hybrid cache (>1 cache group) under DP page sharding, ``allocate_slots``
passes the global free-page pre-check (there is global capacity) but a later
group's per-shard allocation can still fail (that shard is exhausted). The old
rollback only detached the prefix ``new_computed_pages``; the pages already
allocated for earlier groups this call were left attached (ref_cnt=1) and
leaked. A subsequent re-schedule of the same request then tripped
``save_new_computed_pages``'s ``assert len(req_pages) == 0``.

The fix makes the per-group allocation transactional: on failure, every page
allocated for the request during this call is freed and detached, restoring the
pre-call state.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
from easydel.inference.esurge.core.interface import CacheGroupSpec, FullAttentionSpec, SlidingWindowSpec
from easydel.inference.esurge.core.manager import CacheManager, CachePages
from easydel.inference.esurge.request import EngineRequest
from easydel.inference.sampling_params import SamplingParams

PAGE_SIZE = 16
NUM_PAGES = 8  # dp_size=2 -> 4 pages/shard; shard 0 usable = pages [1, 2, 3]
DP = 2


def _hybrid_groups() -> list[CacheGroupSpec]:
    return [
        CacheGroupSpec(
            kv_cache_spec=FullAttentionSpec(
                page_size=PAGE_SIZE,
                num_kv_heads=1,
                head_size=8,
                dtype=jnp.float16,
                use_mla=False,
            )
        ),
        CacheGroupSpec(
            kv_cache_spec=SlidingWindowSpec(
                page_size=PAGE_SIZE,
                num_kv_heads=1,
                head_size=8,
                dtype=jnp.float16,
                use_mla=False,
                sliding_window=256,  # >= max_model_len: never drops pages here
            )
        ),
    ]


def _manager(enable_caching: bool) -> CacheManager:
    return CacheManager(
        num_pages=NUM_PAGES,
        kv_cache_groups=_hybrid_groups(),
        max_model_len=256,
        enable_caching=enable_caching,
    )


def _req(rid: str, prompt_len: int) -> EngineRequest:
    return EngineRequest(
        request_id=rid,
        prompt_token_ids=list(range(prompt_len)),
        sampling_params=SamplingParams(max_tokens=8),
        eos_token_id=1,
    )


def test_hybrid_shard_alloc_failure_leaves_no_leaked_pages():
    mgr = _manager(enable_caching=False)
    pool = mgr.page_pool

    # Occupy 2 of shard 0's 3 usable pages with a dummy pinned to shard 0
    # (1 page per group -> pages 1 and 2), leaving exactly one free page there.
    dummy = _req("dummy", PAGE_SIZE)
    assert mgr.allocate_slots(dummy, num_new_tokens=PAGE_SIZE, dp_shard_hint=0, data_parallel_size=DP) is not None

    free_before = pool.get_num_free_pages()

    # The victim (also shard 0) needs one page per group. The global pre-check
    # passes (shard 1 is entirely free), group 0 grabs shard 0's last free page,
    # then group 1's per-shard allocation fails.
    victim = _req("victim", PAGE_SIZE)
    result = mgr.allocate_slots(victim, num_new_tokens=PAGE_SIZE, dp_shard_hint=0, data_parallel_size=DP)

    assert result is None
    # No leak: the group-0 page was returned; free count is fully restored and
    # the victim holds no pages in any group.
    assert pool.get_num_free_pages() == free_before
    assert all(len(group) == 0 for group in mgr.coordinator.get_pages("victim"))


def test_hybrid_shard_alloc_failure_retry_with_prefix_hit_does_not_assert():
    mgr = _manager(enable_caching=True)
    pool = mgr.page_pool

    # Pull one shard-0 page per group to act as the victim's cached prefix
    # (pages 1 and 2), leaving exactly one free page (page 3) in shard 0.
    cp_full = pool.get_new_pages(1, dp_shard_hint=0, data_parallel_size=DP)
    cp_slide = pool.get_new_pages(1, dp_shard_hint=0, data_parallel_size=DP)
    computed = CachePages((list(cp_full), list(cp_slide)))

    victim = _req("victim", 32)

    def attempt():
        # 1 cached page/group + 1 new page/group needed. Group 0 grabs shard 0's
        # last free page; group 1's per-shard allocation then fails.
        return mgr.allocate_slots(
            victim,
            num_new_tokens=PAGE_SIZE,
            num_new_computed_tokens=PAGE_SIZE,
            new_computed_pages=computed,
            dp_shard_hint=0,
            data_parallel_size=DP,
        )

    assert attempt() is None
    # With the bug, the first attempt popped num_cached_page but left the
    # group-0 page attached, so this retry raises AssertionError inside
    # save_new_computed_pages. With the fix it cleanly returns None again.
    assert attempt() is None
    assert all(len(group) == 0 for group in mgr.coordinator.get_pages("victim"))
