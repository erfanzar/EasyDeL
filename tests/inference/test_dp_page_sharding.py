from __future__ import annotations

from easydel.inference.esurge.core.dp_sharding import (
    dp_shard_for_page_id,
    dp_shard_page_bounds,
    pages_per_dp_shard,
)
from easydel.inference.esurge.core.page_pool import PagePool


def test_pages_per_dp_shard_excludes_null_page_from_partition() -> None:
    # num_pages includes page 0 (reserved null page), so usable pages are num_pages - 1.
    assert pages_per_dp_shard(8780, 4) is None
    assert pages_per_dp_shard(8781, 4) == 2195


def test_dp_shard_mapping_uses_page_id_offset() -> None:
    pages_per_shard = 2
    dp_size = 3
    assert dp_shard_for_page_id(0, pages_per_shard, dp_size) is None
    assert dp_shard_for_page_id(1, pages_per_shard, dp_size) == 0
    assert dp_shard_for_page_id(2, pages_per_shard, dp_size) == 0
    assert dp_shard_for_page_id(3, pages_per_shard, dp_size) == 1
    assert dp_shard_for_page_id(4, pages_per_shard, dp_size) == 1
    assert dp_shard_for_page_id(5, pages_per_shard, dp_size) == 2
    assert dp_shard_for_page_id(6, pages_per_shard, dp_size) == 2


def test_dp_shard_bounds_start_after_null_page() -> None:
    assert dp_shard_page_bounds(0, 4) == (1, 5)
    assert dp_shard_page_bounds(1, 4) == (5, 9)


def test_page_pool_shard_allocation_uses_usable_ranges() -> None:
    pool = PagePool(num_pages=9, enable_caching=False)  # usable pages: 1..8
    shard0 = pool.get_new_pages(2, dp_shard_hint=0, data_parallel_size=4)
    shard3 = pool.get_new_pages(2, dp_shard_hint=3, data_parallel_size=4)

    assert all(1 <= p.page_id < 3 for p in shard0)
    assert all(7 <= p.page_id < 9 for p in shard3)


def test_page_pool_ignores_invalid_dp_shard_partition() -> None:
    pool = PagePool(num_pages=8, enable_caching=False)  # usable pages: 1..7 (not divisible by 4)
    pages = pool.get_new_pages(1, dp_shard_hint=3, data_parallel_size=4)
    assert pages[0].page_id == 1
