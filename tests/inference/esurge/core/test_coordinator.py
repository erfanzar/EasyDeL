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

"""Tests for ``easydel.inference.esurge.core.coordinator``.

The coordinator orchestrates per-attention-type cache managers behind a
unified API. Tests focus on:

* Factory dispatch (``get_kv_cache_coordinator``) by ``enable_caching`` and
  ``len(kv_cache_groups)``.
* Construction-time validation:
    - ``UnitaryCacheCoordinator`` accepts exactly one cache group
    - ``HybridCacheCoordinator`` requires exactly one full-attention type
      and one other type, non-interleaved
    - ``HybridCacheCoordinator`` requires page sizes to be divisible
* ``CacheCoordinatorNoPrefixCache`` always reports zero common prefix and
  empty cache hits (it's the no-cache path).
* Coordinator fans out simple operations (``free``, ``get_pages``) across
  managers.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from easydel.inference.esurge.core.coordinator import (
    CacheCoordinator,
    CacheCoordinatorNoPrefixCache,
    HybridCacheCoordinator,
    UnitaryCacheCoordinator,
    get_kv_cache_coordinator,
)
from easydel.inference.esurge.core.interface import (
    CacheGroupSpec,
    FullAttentionSpec,
    SlidingWindowSpec,
)


def _full_attention_spec(*, page_size: int = 16) -> FullAttentionSpec:
    return FullAttentionSpec(
        page_size=page_size,
        num_kv_heads=2,
        head_size=8,
        dtype=jnp.float16,
        use_mla=False,
    )


def _sliding_window_spec(*, page_size: int = 16, window: int = 64) -> SlidingWindowSpec:
    return SlidingWindowSpec(
        page_size=page_size,
        num_kv_heads=2,
        head_size=8,
        dtype=jnp.float16,
        use_mla=False,
        sliding_window=window,
    )


def _full_group(*, page_size: int = 16) -> CacheGroupSpec:
    return CacheGroupSpec(kv_cache_spec=_full_attention_spec(page_size=page_size))


def _sliding_group(*, page_size: int = 16, window: int = 64) -> CacheGroupSpec:
    return CacheGroupSpec(kv_cache_spec=_sliding_window_spec(page_size=page_size, window=window))


def test_factory_returns_no_prefix_cache_when_caching_disabled():
    coord = get_kv_cache_coordinator(
        num_pages=64,
        kv_cache_groups=[_full_group()],
        max_model_len=128,
        use_eagle=False,
        enable_caching=False,
    )
    assert isinstance(coord, CacheCoordinatorNoPrefixCache)


def test_factory_returns_unitary_when_one_group_with_caching():
    coord = get_kv_cache_coordinator(
        num_pages=64,
        kv_cache_groups=[_full_group()],
        max_model_len=128,
        use_eagle=False,
        enable_caching=True,
    )
    assert isinstance(coord, UnitaryCacheCoordinator)


def test_factory_returns_hybrid_when_multiple_groups_with_caching():
    coord = get_kv_cache_coordinator(
        num_pages=64,
        kv_cache_groups=[_full_group(page_size=16), _sliding_group(page_size=16)],
        max_model_len=128,
        use_eagle=False,
        enable_caching=True,
    )
    assert isinstance(coord, HybridCacheCoordinator)


def test_factory_returns_no_prefix_for_zero_groups_when_caching_disabled():
    """``CacheCoordinatorNoPrefixCache`` accepts zero groups per its docstring."""
    coord = get_kv_cache_coordinator(
        num_pages=64,
        kv_cache_groups=[],
        max_model_len=128,
        use_eagle=False,
        enable_caching=False,
    )
    assert isinstance(coord, CacheCoordinatorNoPrefixCache)
    assert coord.num_single_type_manager == 0


def test_cache_coordinator_is_abstract():
    """Cannot instantiate ``CacheCoordinator`` directly -- has abstract method."""
    with pytest.raises(TypeError):
        CacheCoordinator(
            num_pages=10,
            kv_cache_groups=[],
            max_model_len=64,
            use_eagle=False,
            enable_caching=False,
        )


def test_no_prefix_cache_returns_empty_hit():
    coord = CacheCoordinatorNoPrefixCache(
        num_pages=64,
        kv_cache_groups=[_full_group()],
        max_model_len=128,
        use_eagle=False,
    )
    pages, hit_len = coord.find_longest_cache_hit(
        page_hashes=[],
        max_cache_hit_length=512,
    )
    assert hit_len == 0
    assert len(pages) == 1
    assert pages[0] == []


def test_no_prefix_cache_zero_common_prefix_pages():
    """Without caching, common prefix pages are always 0 per group."""
    coord = CacheCoordinatorNoPrefixCache(
        num_pages=64,
        kv_cache_groups=[_full_group(), _sliding_group()],
        max_model_len=128,
        use_eagle=False,
    )
    common = coord.get_num_common_prefix_pages(request_id="r1", num_scheduled_requests=4)
    assert common == [0, 0]


def test_no_prefix_cache_get_pages_returns_empty_per_group():
    """No managed requests -> all groups return empty page lists."""
    coord = CacheCoordinatorNoPrefixCache(
        num_pages=64,
        kv_cache_groups=[_full_group()],
        max_model_len=128,
        use_eagle=False,
    )
    pages = coord.get_pages("never-allocated")
    assert pages == ([],)


def test_no_prefix_cache_free_unknown_request_does_not_raise():
    """Freeing a request that was never allocated must be a no-op (cleanup safety)."""
    coord = CacheCoordinatorNoPrefixCache(
        num_pages=64,
        kv_cache_groups=[_full_group()],
        max_model_len=128,
        use_eagle=False,
    )
    coord.free("never-existed")


def test_unitary_records_kv_cache_spec_and_page_size():
    spec = _full_attention_spec(page_size=32)
    coord = UnitaryCacheCoordinator(
        num_pages=64,
        kv_cache_groups=[CacheGroupSpec(kv_cache_spec=spec)],
        max_model_len=128,
        use_eagle=False,
        enable_caching=True,
    )
    assert coord.kv_cache_spec is spec
    assert coord.page_size == 32


def test_unitary_rejects_multi_group_via_super_first():
    """The current implementation hits the multi-group ValueError after super().__init__.

    The order is: super().__init__ runs (creating managers), then the assertion
    raises. We just verify the construction fails -- the exact error path is
    less important than the fact that multi-group + UnitaryCoordinator is rejected.
    """
    with pytest.raises((ValueError, AssertionError)):
        UnitaryCacheCoordinator(
            num_pages=64,
            kv_cache_groups=[_full_group(), _full_group()],
            max_model_len=128,
            use_eagle=False,
            enable_caching=True,
        )


def test_hybrid_full_then_sliding_ordering_recorded():
    """When full-attention groups precede other groups, ``full_attn_first=True``."""
    coord = HybridCacheCoordinator(
        num_pages=64,
        kv_cache_groups=[_full_group(page_size=16), _sliding_group(page_size=16)],
        max_model_len=128,
        use_eagle=False,
        enable_caching=True,
    )
    assert coord.full_attention_group_ids == [0]
    assert coord.other_group_ids == [1]
    assert coord.full_attn_first is True


def test_hybrid_sliding_then_full_ordering_recorded():
    coord = HybridCacheCoordinator(
        num_pages=64,
        kv_cache_groups=[_sliding_group(page_size=16), _full_group(page_size=16)],
        max_model_len=128,
        use_eagle=False,
        enable_caching=True,
    )
    assert coord.full_attention_group_ids == [1]
    assert coord.other_group_ids == [0]
    assert coord.full_attn_first is False


def test_hybrid_rejects_interleaved_groups():
    """Interleaved (full, sliding, full) is unsupported -- splitter raises."""
    with pytest.raises(ValueError, match="interleave"):
        HybridCacheCoordinator(
            num_pages=64,
            kv_cache_groups=[
                _full_group(page_size=16),
                _sliding_group(page_size=16),
                _full_group(page_size=16),
            ],
            max_model_len=128,
            use_eagle=False,
            enable_caching=True,
        )


def test_hybrid_rejects_no_full_attention_group():
    """A hybrid coord requires at least one full-attention group.

    With two same-window SlidingWindowSpec groups, the loop accepts both as
    ``other_type_id``, then the post-loop check raises because
    ``full_attention_type_id is None``.
    """
    with pytest.raises(ValueError, match="full attention"):
        HybridCacheCoordinator(
            num_pages=64,
            kv_cache_groups=[
                _sliding_group(page_size=16, window=64),
                _sliding_group(page_size=16, window=64),
            ],
            max_model_len=128,
            use_eagle=False,
            enable_caching=True,
        )


def test_hybrid_rejects_no_other_group():
    """A hybrid coord requires at least one non-full-attention group.

    With two same-page_size FullAttentionSpec groups, the loop accepts both
    as full-attention; the post-loop ``other_type_id is None`` check raises.
    """
    with pytest.raises(ValueError, match="other"):
        HybridCacheCoordinator(
            num_pages=64,
            kv_cache_groups=[
                _full_group(page_size=16),
                _full_group(page_size=16),
            ],
            max_model_len=128,
            use_eagle=False,
            enable_caching=True,
        )


def test_hybrid_rejects_two_different_full_attention_types():
    """Two full-attention groups with DIFFERENT page sizes trigger the inner type-mismatch error."""
    with pytest.raises(ValueError, match="exactly one type of full attention"):
        HybridCacheCoordinator(
            num_pages=64,
            kv_cache_groups=[
                _full_group(page_size=16),
                _full_group(page_size=32),
            ],
            max_model_len=128,
            use_eagle=False,
            enable_caching=True,
        )


def test_hybrid_rejects_two_different_other_attention_types():
    """Two sliding-window groups with DIFFERENT windows have different type_ids -> error."""
    with pytest.raises(ValueError, match="exactly one (other type|type of other groups)"):  # noqa: RUF043
        HybridCacheCoordinator(
            num_pages=64,
            kv_cache_groups=[
                _sliding_group(page_size=16, window=64),
                _sliding_group(page_size=16, window=32),
            ],
            max_model_len=128,
            use_eagle=False,
            enable_caching=True,
        )


def test_hybrid_rejects_indivisible_page_sizes_when_caching_enabled():
    """``other_page_size % full_attention_page_size`` must be 0 when caching is on."""
    with pytest.raises(ValueError, match="divisible"):
        HybridCacheCoordinator(
            num_pages=64,
            kv_cache_groups=[
                _full_group(page_size=16),
                _sliding_group(page_size=24),
            ],
            max_model_len=128,
            use_eagle=False,
            enable_caching=True,
        )


def test_hybrid_records_full_attention_and_other_specs():
    full_spec = _full_attention_spec(page_size=16)
    other_spec = _sliding_window_spec(page_size=16, window=64)
    coord = HybridCacheCoordinator(
        num_pages=64,
        kv_cache_groups=[
            CacheGroupSpec(kv_cache_spec=full_spec),
            CacheGroupSpec(kv_cache_spec=other_spec),
        ],
        max_model_len=128,
        use_eagle=False,
        enable_caching=True,
    )
    assert coord.full_attention_spec is full_spec
    assert coord.other_spec is other_spec
    assert coord.full_attention_page_size == 16
    assert coord.other_page_size == 16


def test_get_pages_unknown_request_returns_empty_per_group():
    coord = UnitaryCacheCoordinator(
        num_pages=64,
        kv_cache_groups=[_full_group()],
        max_model_len=128,
        use_eagle=False,
        enable_caching=True,
    )

    pages = coord.get_pages("ghost-request-id")
    assert pages == ([],)


def test_get_num_pages_to_allocate_zero_for_unallocated_zero_token_request():
    coord = UnitaryCacheCoordinator(
        num_pages=64,
        kv_cache_groups=[_full_group()],
        max_model_len=128,
        use_eagle=False,
        enable_caching=True,
    )
    n = coord.get_num_pages_to_allocate(
        request_id="r1",
        num_tokens=0,
        new_computed_pages=([],),
    )
    assert n == 0


def test_free_request_without_allocation_is_safe():
    """Freeing a never-allocated request must not raise across all coordinator types."""
    coords = [
        get_kv_cache_coordinator(64, [_full_group()], 128, use_eagle=False, enable_caching=True),
        get_kv_cache_coordinator(64, [_full_group()], 128, use_eagle=False, enable_caching=False),
        get_kv_cache_coordinator(
            64,
            [_full_group(page_size=16), _sliding_group(page_size=16)],
            128,
            use_eagle=False,
            enable_caching=True,
        ),
    ]
    for c in coords:
        c.free("never-allocated")
