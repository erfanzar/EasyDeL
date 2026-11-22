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

"""A failed allocation rolls back the prefix hit it attached (CacheManager over the Rust KV core)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.core.interface import CacheGroupSpec, FullAttentionSpec
from easydel.inference.esurge.core.manager import CacheManager

pytestmark = pytest.mark.cpu_ok

PAGE = 4


def _request(rid: str, length: int) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=rid,
        num_tokens=length,
        num_computed_tokens=0,
        all_token_ids=list(range(length)),
        sampling_params=None,
    )


def _manager() -> CacheManager:
    spec = FullAttentionSpec(page_size=PAGE, num_kv_heads=1, head_size=4, dtype=np.float32, use_mla=False)
    # 8 pages, dp 2 -> 4 pages per shard; shard 0 can use pages 1-3, shard 1 pages 4-7.
    return CacheManager(num_pages=8, kv_cache_groups=[CacheGroupSpec(spec)], max_model_len=64, enable_caching=True)


def test_allocate_slots_rolls_back_saved_prefix_pages_on_shard_allocation_error():
    manager = _manager()
    seed = _request("seed", 2 * PAGE + 1)
    assert manager.allocate_slots(seed, num_new_tokens=2 * PAGE, dp_shard_hint=0, data_parallel_size=2) is not None
    manager.cache_pages(seed, 2 * PAGE)
    manager.free(seed)
    manager.free_page_hashes(seed)
    free_order = manager._core.free_order()

    request = _request("req-1", 6 * PAGE)
    cached, hit = manager.get_computed_pages(request, dp_shard_hint=0, data_parallel_size=2)
    assert hit == 2 * PAGE and cached.get_page_ids() == ([1, 2],)
    # Four more pages fit the pool (5 free besides the hit) but not shard 0 (page 3 only): the shard allocation
    # fails after the hit pages were attached, and the call must leave everything as it was.
    result = manager.allocate_slots(
        request,
        num_new_tokens=4 * PAGE,
        num_new_computed_tokens=hit,
        new_computed_pages=cached,
        dp_shard_hint=0,
        data_parallel_size=2,
    )
    assert result is None
    assert manager.get_page_ids("req-1") == ([],)
    assert manager._core.refs() == [0] * 7
    assert sorted(manager._core.free_order()) == sorted(free_order)
    manager._core.check()
    # The hit pages stay published and a retry within the shard succeeds.
    cached, hit = manager.get_computed_pages(request, dp_shard_hint=0, data_parallel_size=2)
    assert hit == 2 * PAGE
    pages = manager.allocate_slots(
        request,
        num_new_tokens=PAGE,
        num_new_computed_tokens=hit,
        new_computed_pages=cached,
        dp_shard_hint=0,
        data_parallel_size=2,
    )
    assert pages is not None and pages.get_page_ids() == ([3],)
    assert manager.get_page_ids("req-1") == ([1, 2, 3],)
