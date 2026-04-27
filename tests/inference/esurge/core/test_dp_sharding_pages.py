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

from __future__ import annotations

from easydel.inference.esurge.core.dp_sharding import (
    dp_shard_for_page_id,
    dp_shard_page_bounds,
    pages_per_dp_shard,
)
from easydel.inference.esurge.core.page_pool import PagePool


def test_pages_per_dp_shard_requires_num_pages_divisible() -> None:
    # num_pages itself must be divisible by dp_size (JAX sharding requirement).
    assert pages_per_dp_shard(8780, 4) == 2195  # 8780 % 4 == 0
    assert pages_per_dp_shard(8781, 4) is None  # 8781 % 4 != 0


def test_dp_shard_mapping_uses_floor_division() -> None:
    pages_per_shard = 3
    dp_size = 3
    assert dp_shard_for_page_id(0, pages_per_shard, dp_size) is None
    assert dp_shard_for_page_id(1, pages_per_shard, dp_size) == 0
    assert dp_shard_for_page_id(2, pages_per_shard, dp_size) == 0
    assert dp_shard_for_page_id(3, pages_per_shard, dp_size) == 1
    assert dp_shard_for_page_id(4, pages_per_shard, dp_size) == 1
    assert dp_shard_for_page_id(5, pages_per_shard, dp_size) == 1
    assert dp_shard_for_page_id(6, pages_per_shard, dp_size) == 2
    assert dp_shard_for_page_id(7, pages_per_shard, dp_size) == 2
    assert dp_shard_for_page_id(8, pages_per_shard, dp_size) == 2


def test_dp_shard_bounds_shard0_skips_null_page() -> None:
    # Shard 0 owns page 0 (null) so usable range starts at 1.
    assert dp_shard_page_bounds(0, 4) == (1, 4)
    assert dp_shard_page_bounds(1, 4) == (4, 8)


def test_page_pool_shard_allocation_uses_usable_ranges() -> None:
    pool = PagePool(num_pages=12, enable_caching=False)  # 12 % 4 == 0
    # pages_per_shard = 12 // 4 = 3
    # shard 0: pages [0,1,2] → usable [1,2]
    # shard 3: pages [9,10,11]
    shard0 = pool.get_new_pages(2, dp_shard_hint=0, data_parallel_size=4)
    shard3 = pool.get_new_pages(2, dp_shard_hint=3, data_parallel_size=4)

    assert all(1 <= p.page_id < 3 for p in shard0)
    assert all(9 <= p.page_id < 12 for p in shard3)


def test_page_pool_ignores_invalid_dp_shard_partition() -> None:
    pool = PagePool(num_pages=9, enable_caching=False)  # 9 % 4 != 0
    pages = pool.get_new_pages(1, dp_shard_hint=3, data_parallel_size=4)
    assert pages[0].page_id == 1
