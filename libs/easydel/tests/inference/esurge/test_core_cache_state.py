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

"""Regression tests for per-request cache state in the eSurge core page managers.

These tests exercise pure-Python bookkeeping (page lists, reference counts); no array computation runs.
"""

from __future__ import annotations

import numpy as np
import pytest
from easydel.inference.esurge.core.interface import FullAttentionSpec
from easydel.inference.esurge.core.page_pool import PagePool
from easydel.inference.esurge.core.single_type_cache_manager import FullAttentionManager


def _manager(num_pages: int = 16) -> tuple[FullAttentionManager, PagePool]:
    spec = FullAttentionSpec(page_size=16, num_kv_heads=2, head_size=8, dtype=np.float16, use_mla=False)
    pool = PagePool(num_pages=num_pages, enable_caching=True)
    return FullAttentionManager(spec, pool, kv_cache_group_id=0), pool


@pytest.mark.cpu_ok
def test_failed_allocation_without_prefix_hits_allows_later_prefix_hit():
    """A rolled-back attempt with no computed pages must not poison a retry that finds prefix hits."""
    mgr, pool = _manager()

    # First attempt: no prefix hits, allocation failed, coordinator rolls back with empty lists.
    mgr.save_new_computed_pages("req", [])
    mgr.rollback_new_computed_pages("req", [])

    # Retry: another request has meanwhile published a matching prefix page.
    [published] = pool.get_new_pages(num_pages=1)
    mgr.save_new_computed_pages("req", [published])

    assert mgr.req_to_pages["req"] == [published]
    assert mgr.num_cached_page["req"] == 1


@pytest.mark.cpu_ok
def test_empty_rollback_keeps_pages_already_held_by_request():
    """Rolling back an empty attempt must not drop pages the request already owns."""
    mgr, pool = _manager()
    [held] = pool.get_new_pages(num_pages=1)
    mgr.save_new_computed_pages("req", [held])

    mgr.save_new_computed_pages("req", [])
    mgr.rollback_new_computed_pages("req", [])

    assert mgr.req_to_pages["req"] == [held]
    assert mgr.num_cached_page["req"] == 1


@pytest.mark.cpu_ok
def test_free_pages_raises_on_ref_count_underflow():
    """Freeing a non-null page more often than it was referenced is a loud error, not silent corruption."""
    pool = PagePool(num_pages=8, enable_caching=False)
    [page] = pool.get_new_pages(num_pages=1)
    pool.free_pages([page])
    assert page.ref_cnt == 0

    with pytest.raises(RuntimeError, match="underflow"):
        pool.free_pages([page])


@pytest.mark.cpu_ok
def test_free_pages_tolerates_null_page_without_requeueing_it():
    """The shared null page may appear in request page lists and is never pooled or guarded."""
    pool = PagePool(num_pages=8, enable_caching=False)
    free_before = pool.get_num_free_pages()

    pool.free_pages([pool.null_page])

    assert pool.get_num_free_pages() == free_before


@pytest.mark.cpu_ok
def test_invalid_batch_free_leaves_valid_pages_owned():
    pool = PagePool(num_pages=8, enable_caching=False)
    first, second = pool.get_new_pages(num_pages=2)
    pool.free_pages([second])
    free_before = pool.get_num_free_pages()

    with pytest.raises(RuntimeError, match="underflow"):
        pool.free_pages([first, second])

    assert first.ref_cnt == 1
    assert pool.get_num_free_pages() == free_before
    pool.free_pages([first])
    assert pool.get_num_free_pages() == free_before + 1


@pytest.mark.cpu_ok
def test_repeated_reference_frees_page_only_once():
    pool = PagePool(num_pages=8, enable_caching=False)
    [page] = pool.get_new_pages(num_pages=1)
    pool.touch(([page],))
    free_before = pool.get_num_free_pages()

    pool.free_pages([page, page])

    assert page.ref_cnt == 0
    assert pool.get_num_free_pages() == free_before + 1
    allocated = pool.get_new_pages(num_pages=free_before + 1)
    assert len({p.page_id for p in allocated}) == len(allocated)


@pytest.mark.cpu_ok
def test_duplicate_over_release_is_atomic():
    pool = PagePool(num_pages=8, enable_caching=False)
    [page] = pool.get_new_pages(num_pages=1)
    free_before = pool.get_num_free_pages()

    with pytest.raises(RuntimeError, match="underflow"):
        pool.free_pages([page, page])

    assert page.ref_cnt == 1
    assert pool.get_num_free_pages() == free_before
