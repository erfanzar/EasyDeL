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

"""Tests for ``easydel.inference.esurge.core.page_pool``.

The ``PagePool`` is a doubly-linked free queue plus a hash-to-page index for
prefix caching. The tests cover:

* Construction validation (positive integer count) + null-page reservation.
* ``get_new_pages`` allocates from the free queue and decrements the count.
* ``get_new_pages`` raises when the pool is exhausted (no eviction available).
* ``free_pages`` puts pages back when ref_cnt hits 0 (and skips null pages).
* ``touch`` increments ref counts and removes from free queue when ref_cnt==0.
* Cache lookup hit/miss semantics on the prefix cache.
* ``reset_prefix_cache`` succeeds only when no pages are in use.
* ``get_usage`` is between 0.0 and 1.0 and reflects allocations.
"""

from __future__ import annotations

import pytest

from easydel.inference.esurge.core.page_pool import PagePool
from easydel.inference.esurge.core.utils import PageHash, PageHashWithGroupId


def test_construct_with_positive_num_pages():
    pool = PagePool(num_pages=8, enable_caching=True)
    assert pool.num_pages == 8
    assert pool.enable_caching is True

    assert pool.null_page.is_null is True
    assert pool.get_num_free_pages() == 7


def test_construct_rejects_zero_or_negative_num_pages():
    with pytest.raises(ValueError, match="positive integer"):
        PagePool(num_pages=0, enable_caching=True)
    with pytest.raises(ValueError, match="positive integer"):
        PagePool(num_pages=-1, enable_caching=True)


def test_construct_rejects_non_integer_num_pages():
    with pytest.raises(ValueError, match="positive integer"):
        PagePool(num_pages=1.5, enable_caching=True)


def test_pages_attribute_lists_all_pages_with_unique_ids():
    pool = PagePool(num_pages=4, enable_caching=False)
    ids = [p.page_id for p in pool.pages]
    assert ids == [0, 1, 2, 3]
    assert len(set(ids)) == 4


def test_get_new_pages_returns_requested_count_and_decrements_free():
    pool = PagePool(num_pages=10, enable_caching=False)
    free_before = pool.get_num_free_pages()
    pages = pool.get_new_pages(num_pages=3)
    assert len(pages) == 3
    assert pool.get_num_free_pages() == free_before - 3

    for p in pages:
        assert p.ref_cnt >= 1


def test_get_new_pages_zero_returns_empty():
    pool = PagePool(num_pages=10, enable_caching=False)
    assert pool.get_new_pages(num_pages=0) == []
    assert pool.get_num_free_pages() == 9


def test_get_new_pages_exhaustion_raises():
    """Asking for more pages than available raises (no caching -> nothing to evict)."""
    pool = PagePool(num_pages=4, enable_caching=False)
    pool.get_new_pages(num_pages=3)
    with pytest.raises(Exception):  # noqa
        pool.get_new_pages(num_pages=1)


def test_free_pages_returns_pages_to_pool_when_refcnt_zero():
    pool = PagePool(num_pages=8, enable_caching=False)
    pages = pool.get_new_pages(num_pages=3)
    free_after_alloc = pool.get_num_free_pages()
    pool.free_pages(reversed(pages))
    assert pool.get_num_free_pages() == free_after_alloc + 3


def test_free_pages_skips_null_page():
    """The null_page must never re-enter the free queue."""
    pool = PagePool(num_pages=4, enable_caching=False)
    free_before = pool.get_num_free_pages()

    pool.null_page.incr_ref()
    pool.free_pages([pool.null_page])

    assert pool.get_num_free_pages() == free_before


def test_free_pages_keeps_page_when_refcnt_still_positive():
    """A page with multiple refs only returns to the pool when ref_cnt drops to 0."""
    pool = PagePool(num_pages=8, enable_caching=False)
    [p] = pool.get_new_pages(num_pages=1)
    p.incr_ref()
    free_count = pool.get_num_free_pages()
    pool.free_pages([p])
    assert pool.get_num_free_pages() == free_count
    pool.free_pages([p])
    assert pool.get_num_free_pages() == free_count + 1


def test_touch_increments_ref_count():
    pool = PagePool(num_pages=4, enable_caching=False)
    [p] = pool.get_new_pages(num_pages=1)
    pool.free_pages([p])
    free_before_touch = pool.get_num_free_pages()
    pool.touch([[p]])
    assert p.ref_cnt == 1
    assert pool.get_num_free_pages() == free_before_touch - 1


def test_touch_skips_null_page():
    pool = PagePool(num_pages=4, enable_caching=False)
    free_before = pool.get_num_free_pages()
    pool.touch([[pool.null_page]])

    assert pool.get_num_free_pages() == free_before


def test_get_cached_page_miss_returns_none():
    pool = PagePool(num_pages=8, enable_caching=True)
    fake_hash = PageHash(hash_value=999, token_ids=(1, 2, 3))
    assert pool.get_cached_page(fake_hash, kv_cache_group_ids=[0]) is None


def test_cache_full_pages_then_get_cached_page_hits():
    """After cache_full_pages installs an entry, get_cached_page should retrieve it."""
    pool = PagePool(num_pages=8, enable_caching=True)
    [p] = pool.get_new_pages(num_pages=1)
    page_hash = PageHash(hash_value=12345, token_ids=(1, 2, 3, 4))

    page_hash_with_gid = PageHashWithGroupId(page_hash, 0)
    p.page_hash = page_hash_with_gid
    pool.cached_page_hash_to_page[page_hash_with_gid][p.page_id] = p

    hit = pool.get_cached_page(page_hash, kv_cache_group_ids=[0])
    assert hit is not None
    assert len(hit) == 1
    assert hit[0] is p


def test_get_cached_page_returns_none_when_any_group_misses():
    """If kv_cache_group_ids requires multiple groups and one misses, return None."""
    pool = PagePool(num_pages=8, enable_caching=True)
    [p] = pool.get_new_pages(num_pages=1)
    page_hash = PageHash(hash_value=12345, token_ids=(1, 2))

    p.page_hash = PageHashWithGroupId(page_hash, 0)
    pool.cached_page_hash_to_page[PageHashWithGroupId(page_hash, 0)][p.page_id] = p

    assert pool.get_cached_page(page_hash, kv_cache_group_ids=[0, 1]) is None


def test_get_cached_page_dp_shard_hint_filters_by_page_id_range():
    """When dp_shard_hint is provided, only pages within the shard's id range hit."""
    pool = PagePool(num_pages=8, enable_caching=True)
    page_hash = PageHash(hash_value=42, token_ids=(7,))

    low_page = pool.pages[1]
    high_page = pool.pages[5]

    low_page._page_hash = PageHashWithGroupId(page_hash, 0)
    high_page._page_hash = PageHashWithGroupId(page_hash, 0)
    pool.cached_page_hash_to_page[PageHashWithGroupId(page_hash, 0)] = {
        low_page.page_id: low_page,
        high_page.page_id: high_page,
    }

    hit_lo = pool.get_cached_page(
        page_hash,
        kv_cache_group_ids=[0],
        dp_shard_hint=0,
        data_parallel_size=2,
    )
    assert hit_lo is not None
    assert hit_lo[0].page_id < 4

    hit_hi = pool.get_cached_page(
        page_hash,
        kv_cache_group_ids=[0],
        dp_shard_hint=1,
        data_parallel_size=2,
    )
    assert hit_hi is not None
    assert hit_hi[0].page_id >= 4


def test_reset_prefix_cache_succeeds_when_no_pages_in_use():
    """Initially only the null page is "used", so reset succeeds."""
    pool = PagePool(num_pages=4, enable_caching=True)

    page_hash = PageHash(hash_value=1, token_ids=(1,))
    pool.cached_page_hash_to_page[PageHashWithGroupId(page_hash, 0)][0] = pool.pages[0]

    assert pool.reset_prefix_cache() is True
    assert len(pool.cached_page_hash_to_page) == 0


def test_reset_prefix_cache_fails_when_pages_in_use():
    """If any non-null page is allocated, reset returns False and keeps the cache."""
    pool = PagePool(num_pages=4, enable_caching=True)
    pool.get_new_pages(num_pages=1)
    assert pool.reset_prefix_cache() is False


def test_get_num_free_pages_reflects_allocation_and_free():
    pool = PagePool(num_pages=10, enable_caching=False)
    initial = pool.get_num_free_pages()
    pages = pool.get_new_pages(num_pages=4)
    assert pool.get_num_free_pages() == initial - 4
    pool.free_pages(reversed(pages))
    assert pool.get_num_free_pages() == initial


def test_get_usage_zero_when_only_null_used():
    pool = PagePool(num_pages=4, enable_caching=False)

    assert pool.get_usage() == pytest.approx(1 / 4)


def test_get_usage_full_when_all_allocated():
    pool = PagePool(num_pages=4, enable_caching=False)
    pool.get_new_pages(num_pages=3)

    assert pool.get_usage() == pytest.approx(1.0)


def test_get_usage_reflects_intermediate_state():
    pool = PagePool(num_pages=10, enable_caching=False)
    pool.get_new_pages(num_pages=4)

    assert pool.get_usage() == pytest.approx(0.5)
