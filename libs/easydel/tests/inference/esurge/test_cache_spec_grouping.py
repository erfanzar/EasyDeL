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

"""Regression tests for KV-cache spec grouping and CachePages helpers.

Pure-Python bookkeeping only (spec construction and validation); no array
computation runs, so every test here is CPU-safe.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.core import ChunkedLocalAttentionManager, ChunkedLocalAttentionSpec
from easydel.inference.esurge.core.interface import (
    CacheGroupSpec,
    FullAttentionSpec,
    SlidingWindowSpec,
    create_kv_cache_specs_from_config,
)
from easydel.inference.esurge.core.manager import CacheManager, CachePages
from easydel.inference.esurge.core.page_pool import PagePool
from easydel.inference.esurge.core.single_type_cache_manager import get_manager_for_kv_cache_spec
from easydel.infra.utils import AttnMaskDetail, AttnMaskType


class _ConfigStub:
    """Minimal config surface consumed by create_kv_cache_specs_from_config."""

    def __init__(self, mask_details):
        self._mask_details = mask_details

    def get_mask_details(self):
        return self._mask_details


def _full_spec(**overrides):
    kwargs = {"page_size": 16, "num_kv_heads": 2, "head_size": 8, "dtype": np.float16, "use_mla": False}
    kwargs.update(overrides)
    return FullAttentionSpec(**kwargs)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("include_full", [False, True])
def test_chunked_layers_keep_full_history_cache_group(include_full):
    """CHUNK and FULL layers with equal KV geometry share conservative page retention."""
    mask_details = {1: AttnMaskDetail(mask_type=AttnMaskType.CHUNK, size=0, chunks=64)}
    if include_full:
        mask_details = {0: AttnMaskDetail(mask_type=AttnMaskType.FULL, size=0), **mask_details}
    config = _ConfigStub(mask_details)

    specs = create_kv_cache_specs_from_config(config, page_size=16, num_kv_heads=2, head_size=8, dtype=np.float16)

    assert len(specs) == 1
    assert isinstance(specs[0].kv_cache_spec, FullAttentionSpec)
    assert specs[0].kv_cache_spec.attention_chunk_size is None
    assert specs[0].layer_names == (["layer.0", "layer.1"] if include_full else ["layer.1"])


@pytest.mark.cpu_ok
def test_chunk_grouping_retains_full_history_across_mask_sizes():
    """Chunk-mask geometry does not split conservative full-history allocation."""
    config = _ConfigStub(
        {
            0: AttnMaskDetail(mask_type=AttnMaskType.CHUNK, size=32, chunks=64),
            1: AttnMaskDetail(mask_type=AttnMaskType.CHUNK, size=64),
            2: AttnMaskDetail(mask_type=AttnMaskType.CHUNK, size=128),
        }
    )
    specs = create_kv_cache_specs_from_config(config, page_size=16, num_kv_heads=2, head_size=8, dtype=np.float16)

    assert len(specs) == 1
    assert isinstance(specs[0].kv_cache_spec, FullAttentionSpec)
    assert specs[0].kv_cache_spec.attention_chunk_size is None
    assert specs[0].layer_names == ["layer.0", "layer.1", "layer.2"]


@pytest.mark.cpu_ok
@pytest.mark.parametrize("size,chunks", [(0, None), (-1, None), (64, 0), (64, -1)])
def test_chunk_grouping_keeps_unused_mask_sizes_out_of_full_allocation(size, chunks):
    config = _ConfigStub({0: AttnMaskDetail(mask_type=AttnMaskType.CHUNK, size=size, chunks=chunks)})
    specs = create_kv_cache_specs_from_config(config, page_size=16, num_kv_heads=2, head_size=8, dtype=np.float16)
    assert len(specs) == 1
    assert isinstance(specs[0].kv_cache_spec, FullAttentionSpec)
    assert specs[0].kv_cache_spec.attention_chunk_size is None
    assert specs[0].layer_names == ["layer.0"]


@pytest.mark.cpu_ok
@pytest.mark.parametrize("enable_caching", [False, True])
def test_explicit_mixed_groups_evict_only_chunk_pages_at_boundary(enable_caching):
    """Explicit chunk groups preserve FULL ownership independently of config allocation."""
    specs = [
        CacheGroupSpec(_full_spec(), ["layer.0"]),
        CacheGroupSpec(
            ChunkedLocalAttentionSpec(
                page_size=16, num_kv_heads=2, head_size=8, dtype=np.float16, use_mla=False, attention_chunk_size=64
            ),
            ["layer.1"],
        ),
    ]
    manager = CacheManager(num_pages=32, kv_cache_groups=specs, max_model_len=256, enable_caching=enable_caching)
    request = SimpleNamespace(request_id="mixed", num_computed_tokens=0)
    first = manager.allocate_slots(request, num_new_tokens=64)
    assert first is not None
    full_pages, chunk_pages = first.pages
    assert len(full_pages) == len(chunk_pages) == 4
    assert not {page.page_id for page in full_pages} & {page.page_id for page in chunk_pages}

    request.num_computed_tokens = 64
    next_pages = manager.allocate_slots(request, num_new_tokens=1)
    assert next_pages is not None
    full_ids, chunk_ids = manager.get_page_ids(request.request_id)
    assert full_ids[:4] == [page.page_id for page in full_pages]
    assert chunk_ids[:4] == [manager.page_pool.null_page.page_id] * 4
    assert len(full_ids) == len(chunk_ids) == 5
    assert all(page.ref_cnt == 1 for page in full_pages)
    assert all(page.ref_cnt == 0 for page in chunk_pages)
    assert full_ids[-1] != chunk_ids[-1]


@pytest.mark.cpu_ok
def test_explicit_chunk_spec_keeps_registered_manager_and_chunk_eviction():
    """Public chunk cache APIs remain usable independently of the model-config fallback."""
    spec = ChunkedLocalAttentionSpec(
        page_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=np.float16,
        use_mla=False,
        attention_chunk_size=64,
    )
    pool = PagePool(num_pages=8, enable_caching=False)
    manager = get_manager_for_kv_cache_spec(spec, page_pool=pool, kv_cache_group_id=0)
    assert isinstance(manager, ChunkedLocalAttentionManager)
    assert spec.max_memory_usage_bytes(max_model_len=512, max_num_batched_tokens=16) == 5 * spec.page_size_bytes

    allocated = manager.allocate_new_pages("request", num_tokens=80)
    manager.remove_skipped_pages("request", num_computed_tokens=64)

    assert len(allocated) == 5
    assert [page.ref_cnt for page in allocated] == [0, 0, 0, 0, 1]
    assert manager.req_to_pages["request"] == [pool.null_page] * 4 + allocated[4:]


@pytest.mark.cpu_ok
def test_hybrid_full_and_sliding_layers_split_into_two_groups():
    config = _ConfigStub(
        {
            0: AttnMaskDetail(mask_type=AttnMaskType.SLIDING, size=128),
            1: AttnMaskDetail(mask_type=AttnMaskType.FULL, size=0),
            2: AttnMaskDetail(mask_type=AttnMaskType.SLIDING, size=128),
        }
    )

    specs = create_kv_cache_specs_from_config(config, page_size=16, num_kv_heads=2, head_size=8, dtype=np.float16)

    by_type = {type(group.kv_cache_spec): group for group in specs}
    assert set(by_type) == {FullAttentionSpec, SlidingWindowSpec}
    assert by_type[SlidingWindowSpec].kv_cache_spec.sliding_window == 128
    assert by_type[SlidingWindowSpec].layer_names == ["layer.0", "layer.2"]
    assert by_type[FullAttentionSpec].layer_names == ["layer.1"]


@pytest.mark.cpu_ok
def test_full_attention_merge_unifies_equal_window_specs():
    merged = FullAttentionSpec.merge([_full_spec(sliding_window=256), _full_spec(sliding_window=256)])

    assert merged.sliding_window == 256
    assert merged.attention_chunk_size is None


@pytest.mark.cpu_ok
def test_full_attention_merge_rejects_inconsistent_window_sizes():
    with pytest.raises(ValueError, match="same window size"):
        FullAttentionSpec.merge([_full_spec(sliding_window=128), _full_spec(sliding_window=256)])


@pytest.mark.cpu_ok
def test_full_attention_merge_rejects_mixed_window_and_chunk():
    with pytest.raises(AssertionError):
        FullAttentionSpec.merge([_full_spec(sliding_window=128), _full_spec(attention_chunk_size=64)])


@pytest.mark.cpu_ok
def test_cache_pages_new_empty_preserves_group_structure():
    probe = CachePages(([], []))

    assert probe.new_empty().pages == ([], [])

    manager = CacheManager(
        num_pages=8,
        kv_cache_groups=[CacheGroupSpec(kv_cache_spec=_full_spec())],
        max_model_len=64,
        enable_caching=True,
    )

    from_manager = manager.create_empty_page_list()
    assert from_manager.pages == ([],)
    assert from_manager.get_page_ids() == ([],)
