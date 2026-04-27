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

"""Tests for ``easydel.inference.esurge.core.single_type_cache_manager``.

The single-type cache managers track per-request page allocations on top of
a shared ``PagePool``. We test behavior end-to-end with a real (small) page
pool so allocation/free actually affect the pool's free-page count.

Coverage:
* Factory ``get_manager_for_kv_cache_spec`` dispatches by spec type
* ``allocate_new_pages`` honors ``cdiv(num_tokens, page_size)`` and reuses
  pages that are already allocated to the request
* ``free`` returns pages to the pool and clears tracking maps
* ``save_new_computed_pages`` / ``rollback_new_computed_pages`` round-trip
* ``find_longest_cache_hit`` returns empty pages on a cold pool
* ``MambaManager.allocate_new_pages`` enforces the 1-page-per-request invariant
* Cross-manager: ``SlidingWindowManager`` / ``MambaManager`` report 0 common
  prefix pages by design
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from easydel.inference.esurge.core.interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from easydel.inference.esurge.core.page_pool import PagePool
from easydel.inference.esurge.core.single_type_cache_manager import (
    ChunkedLocalAttentionManager,
    FullAttentionManager,
    MambaManager,
    SingleTypeCacheManager,
    SlidingWindowManager,
    get_manager_for_kv_cache_spec,
    spec_manager_map,
)


def _full_spec(*, page_size: int = 16) -> FullAttentionSpec:
    return FullAttentionSpec(
        page_size=page_size,
        num_kv_heads=2,
        head_size=8,
        dtype=jnp.float16,
        use_mla=False,
    )


def _sliding_spec(*, page_size: int = 16, window: int = 64) -> SlidingWindowSpec:
    return SlidingWindowSpec(
        page_size=page_size,
        num_kv_heads=2,
        head_size=8,
        dtype=jnp.float16,
        use_mla=False,
        sliding_window=window,
    )


def _chunked_spec(*, page_size: int = 16, chunk: int = 64) -> ChunkedLocalAttentionSpec:
    return ChunkedLocalAttentionSpec(
        page_size=page_size,
        num_kv_heads=2,
        head_size=8,
        dtype=jnp.float16,
        use_mla=False,
        attention_chunk_size=chunk,
    )


def _mamba_spec() -> MambaSpec:
    return MambaSpec(
        page_size=1,
        shapes=((16, 64),),
        dtype=jnp.float16,
    )


def _make_pool(num_pages: int = 32, *, enable_caching: bool = True) -> PagePool:
    return PagePool(num_pages=num_pages, enable_caching=enable_caching)


def test_single_type_cache_manager_is_abstract():
    with pytest.raises(TypeError):
        SingleTypeCacheManager(_full_spec(), _make_pool(), kv_cache_group_id=0)


@pytest.mark.parametrize(
    "spec_factory,manager_cls",
    [
        (_full_spec, FullAttentionManager),
        (_sliding_spec, SlidingWindowManager),
        (_chunked_spec, ChunkedLocalAttentionManager),
        (_mamba_spec, MambaManager),
    ],
    ids=["full", "sliding", "chunked-local", "mamba"],
)
def test_factory_dispatches_by_spec_type(spec_factory, manager_cls):
    pool = _make_pool()
    manager = get_manager_for_kv_cache_spec(spec_factory(), page_pool=pool, kv_cache_group_id=0)
    assert isinstance(manager, manager_cls)


def test_spec_manager_map_includes_all_known_specs():
    """The dispatch table must include every spec type the factory handles."""
    assert FullAttentionSpec in spec_manager_map
    assert SlidingWindowSpec in spec_manager_map
    assert ChunkedLocalAttentionSpec in spec_manager_map
    assert MambaSpec in spec_manager_map


def test_factory_unknown_spec_type_raises():
    """An unregistered spec class triggers a KeyError from the dispatch dict."""

    class UnknownSpec:
        page_size = 1

    with pytest.raises(KeyError):
        get_manager_for_kv_cache_spec(UnknownSpec(), page_pool=_make_pool(), kv_cache_group_id=0)


def test_full_manager_initial_state():
    pool = _make_pool()
    mgr = FullAttentionManager(_full_spec(page_size=16), pool, kv_cache_group_id=0)
    assert mgr.page_size == 16
    assert mgr.kv_cache_group_id == 0
    assert mgr._null_page is pool.null_page
    assert mgr.req_to_pages == {}
    assert mgr.num_cached_page == {}


def test_full_manager_allocate_new_pages_uses_cdiv():
    """100 tokens with page_size=16 -> ceil(100/16) = 7 pages."""
    pool = _make_pool(num_pages=64)
    mgr = FullAttentionManager(_full_spec(page_size=16), pool, kv_cache_group_id=0)
    free_before = pool.get_num_free_pages()
    pages = mgr.allocate_new_pages("r1", num_tokens=100)
    assert len(pages) == 7
    assert pool.get_num_free_pages() == free_before - 7
    assert len(mgr.req_to_pages["r1"]) == 7


def test_full_manager_allocate_no_op_when_already_sufficient():
    """Re-allocating when the request already has enough pages returns []."""
    pool = _make_pool(num_pages=64)
    mgr = FullAttentionManager(_full_spec(page_size=16), pool, kv_cache_group_id=0)
    mgr.allocate_new_pages("r1", num_tokens=100)
    free_after_first = pool.get_num_free_pages()

    new_pages = mgr.allocate_new_pages("r1", num_tokens=80)
    assert new_pages == []
    assert pool.get_num_free_pages() == free_after_first


def test_full_manager_allocate_extends_when_more_pages_needed():
    pool = _make_pool(num_pages=64)
    mgr = FullAttentionManager(_full_spec(page_size=16), pool, kv_cache_group_id=0)
    mgr.allocate_new_pages("r1", num_tokens=16)
    new_pages = mgr.allocate_new_pages("r1", num_tokens=64)
    assert len(new_pages) == 3
    assert len(mgr.req_to_pages["r1"]) == 4


def test_full_manager_free_returns_pages_to_pool():
    pool = _make_pool(num_pages=32)
    mgr = FullAttentionManager(_full_spec(page_size=16), pool, kv_cache_group_id=0)
    free_initial = pool.get_num_free_pages()
    mgr.allocate_new_pages("r1", num_tokens=64)
    mgr.free("r1")
    assert pool.get_num_free_pages() == free_initial
    assert "r1" not in mgr.req_to_pages
    assert "r1" not in mgr.num_cached_page


def test_full_manager_free_unknown_request_is_safe():
    """Freeing a request that was never allocated must not raise."""
    pool = _make_pool()
    mgr = FullAttentionManager(_full_spec(), pool, kv_cache_group_id=0)
    mgr.free("never-existed")


def test_get_num_pages_to_allocate_zero_for_zero_tokens():
    pool = _make_pool()
    mgr = FullAttentionManager(_full_spec(page_size=16), pool, kv_cache_group_id=0)
    assert mgr.get_num_pages_to_allocate("r1", num_tokens=0, new_computed_pages=[]) == 0


def test_get_num_pages_to_allocate_accounts_for_existing_pages():
    pool = _make_pool(num_pages=32)
    mgr = FullAttentionManager(_full_spec(page_size=16), pool, kv_cache_group_id=0)
    mgr.allocate_new_pages("r1", num_tokens=32)

    assert mgr.get_num_pages_to_allocate("r1", num_tokens=64, new_computed_pages=[]) == 2


def test_save_new_computed_pages_marks_request_as_having_cached_prefix():
    pool = _make_pool(num_pages=32)
    mgr = FullAttentionManager(_full_spec(page_size=16), pool, kv_cache_group_id=0)

    pages = mgr.allocate_new_pages("origin", num_tokens=32)
    mgr.save_new_computed_pages("r2", new_computed_pages=pages)
    assert mgr.req_to_pages["r2"] == pages
    assert mgr.num_cached_page["r2"] == 2


def test_save_new_computed_pages_idempotent_on_repeated_empty_save():
    """Calling save_new_computed_pages('r1', []) after the first save is a no-op."""
    pool = _make_pool(num_pages=32)
    mgr = FullAttentionManager(_full_spec(page_size=16), pool, kv_cache_group_id=0)
    pages = mgr.allocate_new_pages("origin", num_tokens=32)
    mgr.save_new_computed_pages("r2", new_computed_pages=pages)

    mgr.save_new_computed_pages("r2", new_computed_pages=[])
    assert len(mgr.req_to_pages["r2"]) == 2


def test_rollback_new_computed_pages_undoes_save():
    pool = _make_pool(num_pages=32)
    mgr = FullAttentionManager(_full_spec(page_size=16), pool, kv_cache_group_id=0)
    pages = mgr.allocate_new_pages("origin", num_tokens=32)
    mgr.save_new_computed_pages("r2", new_computed_pages=pages)
    mgr.rollback_new_computed_pages("r2", new_computed_pages=pages)

    assert "r2" not in mgr.req_to_pages
    assert "r2" not in mgr.num_cached_page


def test_rollback_new_computed_pages_empty_input_is_noop():
    pool = _make_pool(num_pages=32)
    mgr = FullAttentionManager(_full_spec(), pool, kv_cache_group_id=0)

    mgr.rollback_new_computed_pages("any", new_computed_pages=[])
    assert mgr.req_to_pages == {}


def test_rollback_new_computed_pages_for_unknown_request_is_noop():
    pool = _make_pool()
    mgr = FullAttentionManager(_full_spec(), pool, kv_cache_group_id=0)
    fake_pages = mgr.allocate_new_pages("origin", num_tokens=16)
    mgr.rollback_new_computed_pages("never-saved", new_computed_pages=fake_pages)

    assert len(mgr.req_to_pages["origin"]) == 1


def test_full_manager_find_longest_cache_hit_empty_on_cold_pool():
    """Without any cached pages, the cache hit is empty."""
    pool = _make_pool()
    spec = _full_spec(page_size=16)
    pages = FullAttentionManager.find_longest_cache_hit(
        page_hashes=[],
        max_length=128,
        kv_cache_group_ids=[0],
        page_pool=pool,
        kv_cache_spec=spec,
        use_eagle=False,
    )
    assert pages == ([],)


def test_mamba_manager_find_longest_cache_hit_always_empty():
    """Mamba layers do not participate in prefix caching."""
    pool = _make_pool()
    spec = _mamba_spec()
    pages = MambaManager.find_longest_cache_hit(
        page_hashes=[],
        max_length=128,
        kv_cache_group_ids=[0, 1],
        page_pool=pool,
        kv_cache_spec=spec,
        use_eagle=False,
    )

    assert pages == ([], [])


def test_full_manager_find_longest_cache_hit_rejects_wrong_spec_type():
    """Asserts the spec is full or chunked-local; passing a mamba spec raises."""
    pool = _make_pool()
    with pytest.raises(AssertionError):
        FullAttentionManager.find_longest_cache_hit(
            page_hashes=[],
            max_length=128,
            kv_cache_group_ids=[0],
            page_pool=pool,
            kv_cache_spec=_mamba_spec(),
            use_eagle=False,
        )


def test_mamba_manager_find_longest_cache_hit_rejects_wrong_spec_type():
    pool = _make_pool()
    with pytest.raises(AssertionError):
        MambaManager.find_longest_cache_hit(
            page_hashes=[],
            max_length=128,
            kv_cache_group_ids=[0],
            page_pool=pool,
            kv_cache_spec=_full_spec(),
            use_eagle=False,
        )


def test_full_manager_common_prefix_pages_zero_for_unallocated_request():
    pool = _make_pool()
    mgr = FullAttentionManager(_full_spec(), pool, kv_cache_group_id=0)

    assert mgr.get_num_common_prefix_pages("r1", num_scheduled_requests=4) == 0


def test_sliding_window_manager_common_prefix_always_zero():
    """Sliding-window doesn't participate in cascade attention -- always 0."""
    pool = _make_pool()
    mgr = SlidingWindowManager(_sliding_spec(), pool, kv_cache_group_id=0)
    mgr.allocate_new_pages("r1", num_tokens=64)
    assert mgr.get_num_common_prefix_pages("r1", num_scheduled_requests=2) == 0


def test_mamba_manager_common_prefix_always_zero():
    """Mamba layers don't share state across requests."""
    pool = _make_pool()
    mgr = MambaManager(_mamba_spec(), pool, kv_cache_group_id=0)
    assert mgr.get_num_common_prefix_pages("r1", num_scheduled_requests=4) == 0


def test_full_manager_remove_skipped_pages_is_noop():
    pool = _make_pool()
    mgr = FullAttentionManager(_full_spec(), pool, kv_cache_group_id=0)
    mgr.allocate_new_pages("r1", num_tokens=64)
    pages_before = list(mgr.req_to_pages["r1"])
    mgr.remove_skipped_pages("r1", num_computed_tokens=32)
    assert mgr.req_to_pages["r1"] == pages_before


def test_mamba_manager_remove_skipped_pages_is_noop():
    pool = _make_pool()
    mgr = MambaManager(_mamba_spec(), pool, kv_cache_group_id=0)
    mgr.remove_skipped_pages("r1", num_computed_tokens=100)


def test_mamba_manager_allocates_single_page_for_request():
    pool = _make_pool()
    mgr = MambaManager(_mamba_spec(), pool, kv_cache_group_id=0)
    pages = mgr.allocate_new_pages("r1", num_tokens=1)
    assert len(pages) == 1
    assert len(mgr.req_to_pages["r1"]) == 1


def test_mamba_manager_re_allocate_keeps_one_page():
    pool = _make_pool()
    mgr = MambaManager(_mamba_spec(), pool, kv_cache_group_id=0)
    mgr.allocate_new_pages("r1", num_tokens=1)
    new = mgr.allocate_new_pages("r1", num_tokens=1)

    assert new == []
    assert len(mgr.req_to_pages["r1"]) == 1


def test_sliding_window_manager_records_window():
    pool = _make_pool()
    spec = _sliding_spec(window=128)
    mgr = SlidingWindowManager(spec, pool, kv_cache_group_id=0)
    assert mgr.sliding_window == 128
    assert mgr._null_page is pool.null_page
