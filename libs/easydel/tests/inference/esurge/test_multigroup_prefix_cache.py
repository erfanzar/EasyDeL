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

"""CPU-only explicit CHUNK group coverage and conservative model-config allocation."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.core.interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    create_kv_cache_specs_from_config,
)
from easydel.inference.esurge.core.manager import CacheManager
from easydel.infra.utils import AttnMaskType
from easydel.modules.llama4.llama4_configuration import Llama4TextConfig


def _groups(include_full, reverse=False):
    # Exercise the public explicit-group manager API. Model-derived CHUNK
    # layers retain full history and do not opt into eviction automatically.
    from easydel.inference.esurge.core.interface import CacheGroupSpec

    geometry = dict(page_size=16, num_kv_heads=2, head_size=8, dtype=np.float16, use_mla=False)
    groups = [
        CacheGroupSpec(ChunkedLocalAttentionSpec(**geometry, attention_chunk_size=size), [f"layer.{index}"])
        for index, size in enumerate((64, 128))
    ]
    if include_full:
        groups.append(CacheGroupSpec(FullAttentionSpec(**geometry), ["layer.2"]))
    return groups[::-1] if reverse else groups


def _request(request_id, length):
    return SimpleNamespace(
        request_id=request_id,
        num_tokens=length,
        num_computed_tokens=0,
        all_token_ids=list(range(length)),
        sampling_params=None,
    )


@pytest.mark.cpu_ok
@pytest.mark.parametrize("include_full", [False, True])
@pytest.mark.parametrize("enable_caching", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_multiple_configured_chunks_allocate_and_evict_independently(include_full, enable_caching, reverse):
    """Both prefix modes support every emitted group and preserve each chunk's lifetime."""
    groups = _groups(include_full, reverse)
    manager = CacheManager(96, groups, 256, enable_caching=enable_caching)
    request = _request("owner", 129)
    first = manager.allocate_slots(request, num_new_tokens=64)
    assert first is not None
    all_ids = [page.page_id for pages in first.pages for page in pages]
    assert len(set(all_ids)) == 4 * len(groups)

    request.num_computed_tokens = 64
    assert manager.allocate_slots(request, num_new_tokens=64) is not None
    for group, pages, initial in zip(groups, manager.coordinator.get_pages("owner"), first.pages, strict=True):
        assert len(pages) == 8
        if getattr(group.kv_cache_spec, "attention_chunk_size", None) == 64:
            assert all(page.is_null for page in pages[:4])
            assert all(page.ref_cnt == 0 for page in initial)
        else:
            assert pages[:4] == initial
            assert all(page.ref_cnt == 1 for page in initial)

    request.num_computed_tokens = 128
    assert manager.allocate_slots(request, num_new_tokens=1) is not None
    for group, pages in zip(groups, manager.coordinator.get_pages("owner"), strict=True):
        assert len(pages) == 9
        assert all(page.is_null for page in pages[:8]) == isinstance(group.kv_cache_spec, ChunkedLocalAttentionSpec)
        assert not pages[8].is_null
    manager.free(request)
    assert manager.page_pool.get_num_free_pages() == 95


@pytest.mark.cpu_ok
@pytest.mark.parametrize("include_full", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("use_eagle", [False, True])
def test_multiple_chunks_prefix_hit_can_be_attached_and_extended(include_full, reverse, use_eagle):
    """Real cached KV is reused per group, while old chunks are represented by null pages."""
    groups = _groups(include_full, reverse)
    manager = CacheManager(96, groups, 256, enable_caching=True, use_eagle=use_eagle)
    seed = _request("seed", 97)
    assert manager.allocate_slots(seed, num_new_tokens=96) is not None
    manager.cache_pages(seed, num_computed_tokens=96)
    original_ids = manager.get_page_ids(seed.request_id)
    manager.free(seed)

    request = _request("reuse", 97)
    cached, length = manager.get_computed_pages(request)
    assert length == (80 if use_eagle else 96)
    assert all(len(pages) == length // 16 for pages in cached.pages)
    for group, pages, ids in zip(groups, cached.pages, original_ids, strict=True):
        skipped = 4 if getattr(group.kv_cache_spec, "attention_chunk_size", None) == 64 else 0
        assert all(page.is_null for page in pages[:skipped])
        assert [page.page_id for page in pages[skipped:]] == ids[skipped : length // 16]

    assert (
        manager.allocate_slots(
            request, num_new_tokens=97 - length, num_new_computed_tokens=length, new_computed_pages=cached
        )
        is not None
    )
    for pages, prefix in zip(manager.coordinator.get_pages(request.request_id), cached.pages, strict=True):
        assert pages[: len(prefix)] == prefix
        assert len(pages) == 7
        assert all(page.ref_cnt == 1 for page in pages if not page.is_null)
    manager.free(request)
    assert manager.page_pool.get_num_free_pages() == 95


@pytest.mark.cpu_ok
@pytest.mark.parametrize("include_full", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
def test_shorter_group_hit_rechecks_chunk_boundary_null_pages(include_full, reverse):
    """A 64 -> 48 -> 32 endpoint reduction must restore needed KV, not truncate nulls."""
    groups = _groups(include_full, reverse)
    manager = CacheManager(96, groups, 256, enable_caching=True)
    seed = _request("seed", 65)
    allocated = manager.allocate_slots(seed, num_new_tokens=64)
    assert allocated is not None
    hashes = []
    # Publish different amounts of computed KV per group through the page pool's
    # public publication API. The 64-token chunk has only 32 tokens cached;
    # the 128-token chunk has 48; optional FULL has all 64.
    for group_id, (group, pages) in enumerate(zip(groups, allocated.pages, strict=True)):
        chunk_size = getattr(group.kv_cache_spec, "attention_chunk_size", None)
        count = {64: 2, 128: 3, None: 4}[chunk_size]
        manager.page_pool.cache_full_pages(seed, pages, hashes, 0, count, 16, group_id)

    request = _request("lookup", 65)
    cached, length = manager.get_computed_pages(request)
    assert length == 32
    for actual, original in zip(cached.pages, allocated.pages, strict=True):
        assert actual == original[:2]
        assert all(not page.is_null for page in actual)
    assert (
        manager.allocate_slots(request, num_new_tokens=1, num_new_computed_tokens=length, new_computed_pages=cached)
        is not None
    )
    for pages in cached.pages:
        assert all(page.ref_cnt == 2 for page in pages)
    manager.free(request)
    manager.free(seed)
    assert manager.page_pool.get_num_free_pages() == 95


@pytest.mark.cpu_ok
@pytest.mark.parametrize("include_full", [False, True])
def test_multiple_chunks_prefix_hits_obey_dp_shard_locality(include_full):
    groups = _groups(include_full)
    manager = CacheManager(96, groups, 256, enable_caching=True)
    seed = _request("seed", 97)
    assert manager.allocate_slots(seed, num_new_tokens=96, dp_shard_hint=1, data_parallel_size=2) is not None
    manager.cache_pages(seed, num_computed_tokens=96)
    manager.free(seed)

    request = _request("lookup", 97)
    cached, length = manager.get_computed_pages(request, dp_shard_hint=1, data_parallel_size=2)
    assert length == 96
    assert all(48 <= page.page_id < 96 for pages in cached.pages for page in pages if not page.is_null)
    missed, length = manager.get_computed_pages(request, dp_shard_hint=0, data_parallel_size=2)
    assert length == 0
    assert missed.pages == tuple([] for _ in groups)
    assert manager.probe_num_computed_tokens(request) == 96


@pytest.mark.cpu_ok
@pytest.mark.parametrize("include_full", [False, True])
def test_multiple_chunks_and_sliding_windows_share_prefix_cache(include_full):
    from easydel.inference.esurge.core.interface import CacheGroupSpec, SlidingWindowSpec

    groups = _groups(include_full)
    geometry = dict(page_size=16, num_kv_heads=2, head_size=8, dtype=np.float16, use_mla=False)
    groups.extend(
        CacheGroupSpec(SlidingWindowSpec(**geometry, sliding_window=size), [f"sliding.{index}"])
        for index, size in enumerate((32, 64))
    )
    assert len(groups) == (5 if include_full else 4)
    manager = CacheManager(96, groups, 256, enable_caching=True)
    seed = _request("seed", 97)
    assert manager.allocate_slots(seed, num_new_tokens=96) is not None
    manager.cache_pages(seed, num_computed_tokens=96)
    manager.free(seed)
    request = _request("reuse", 97)
    cached, length = manager.get_computed_pages(request)
    assert length == 96
    assert all(len(pages) == 6 for pages in cached.pages)
    assert manager.allocate_slots(request, 1, num_new_computed_tokens=length, new_computed_pages=cached) is not None
    manager.free(request)
    assert manager.page_pool.get_num_free_pages() == 95


@pytest.mark.cpu_ok
def test_llama4_configured_full_and_chunk_layer_ownership():
    """Llama4 chunk masks retain all token pages alongside full-attention layers."""
    config = Llama4TextConfig(num_hidden_layers=6, attention_chunk_size=16, no_rope_layers=[0, 1, 0, 1, 0, 1])
    details = config.get_mask_details()
    assert {detail.mask_type for detail in details.values()} == {AttnMaskType.FULL, AttnMaskType.CHUNK}
    groups = create_kv_cache_specs_from_config(config, page_size=16, num_kv_heads=2, head_size=8, dtype=np.float16)
    assert len(groups) == 1
    assert isinstance(groups[0].kv_cache_spec, FullAttentionSpec)
    assert groups[0].kv_cache_spec.attention_chunk_size is None
    assert groups[0].layer_names == [f"layer.{index}" for index in details]
    manager = CacheManager(64, groups, 64, enable_caching=True)
    assert manager.allocate_slots(_request("llama4", 17), num_new_tokens=17) is not None
