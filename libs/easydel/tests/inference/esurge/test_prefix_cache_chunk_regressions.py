"""Host-only regressions for page-aligned CHUNK hits and multimodal publication."""

from types import SimpleNamespace

import numpy as np
import pytest
from easydel.inference.esurge.core.interface import create_kv_cache_specs_from_config
from easydel.inference.esurge.core.manager import CacheManager
from easydel.infra.utils import AttnMaskDetail, AttnMaskType


def _manager(*, chunk=None, include_full=False, eagle=False, caching=True):
    details = {}
    if chunk is not None:
        details[0] = AttnMaskDetail(mask_type=AttnMaskType.CHUNK, size=chunk)
    if include_full or chunk is None:
        details[len(details)] = AttnMaskDetail(mask_type=AttnMaskType.FULL, size=0)
    groups = create_kv_cache_specs_from_config(
        SimpleNamespace(get_mask_details=lambda: details),
        page_size=16,
        num_kv_heads=2,
        head_size=8,
        dtype=np.float16,
    )
    if chunk is not None:
        from easydel.inference.esurge.core.interface import CacheGroupSpec, ChunkedLocalAttentionSpec, FullAttentionSpec

        geometry = dict(page_size=16, num_kv_heads=2, head_size=8, dtype=np.float16, use_mla=False)
        groups = [CacheGroupSpec(ChunkedLocalAttentionSpec(**geometry, attention_chunk_size=chunk), ["layer.0"])]
        if include_full:
            groups.append(CacheGroupSpec(FullAttentionSpec(**geometry), ["layer.1"]))
    return CacheManager(96, groups, 256, enable_caching=caching, use_eagle=eagle)


def _request(rid, length, *, image=None, prompt_logprobs=None):
    return SimpleNamespace(
        request_id=rid,
        num_tokens=length,
        num_computed_tokens=0,
        all_token_ids=list(range(length)),
        sampling_params=SimpleNamespace(prompt_logprobs=prompt_logprobs),
        has_vision=image is not None,
        mm_features=[SimpleNamespace(modality="image", mm_hash=image)] if image is not None else None,
    )


@pytest.mark.cpu_ok
@pytest.mark.parametrize("include_full", [False, True])
@pytest.mark.parametrize("eagle", [False, True])
def test_cold_unaligned_chunk_does_not_claim_missing_kv(include_full, eagle):
    manager = _manager(chunk=24, include_full=include_full, eagle=eagle)
    request = _request("cold", 25)
    cached, length = manager.get_computed_pages(request)
    assert length == 0
    assert all(not pages for pages in cached.pages)
    assert manager.allocate_slots(request, num_new_tokens=25) is not None
    assert all(not page.is_null for pages in manager.coordinator.get_pages("cold") for page in pages)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("include_full", [False, True])
@pytest.mark.parametrize("chunk", [24, 64])
@pytest.mark.parametrize("eagle", [False, True])
def test_single_and_hybrid_chunks_reuse_prefix_with_eagle(include_full, chunk, eagle):
    manager = _manager(chunk=chunk, include_full=include_full, eagle=eagle)
    source = _request("source", 97)
    assert manager.allocate_slots(source, num_new_tokens=96) is not None
    manager.cache_pages(source, num_computed_tokens=96)
    manager.free(source)
    request = _request("lookup", 97)
    cached, length = manager.get_computed_pages(request)
    assert length == (80 if eagle else 96)
    assert (
        manager.allocate_slots(
            request, num_new_tokens=97 - length, num_new_computed_tokens=length, new_computed_pages=cached
        )
        is not None
    )
    # Every page containing needed tokens at the resumed endpoint must be real.
    for group, pages in zip(manager.coordinator.kv_cache_groups, cached.pages, strict=True):
        size = getattr(group.kv_cache_spec, "attention_chunk_size", None)
        start = (length // size * size) // 16 if size else 0
        assert all(not page.is_null for page in pages[start:])


@pytest.mark.cpu_ok
def test_eagle_shortening_rechecks_previously_null_chunk_pages():
    manager = _manager(chunk=24, eagle=True)
    source = _request("source", 49)
    allocated = manager.allocate_slots(source, num_new_tokens=16)
    assert allocated is not None
    manager.cache_pages(source, num_computed_tokens=16)
    cached, length = manager.get_computed_pages(_request("lookup", 49))
    # 48 is a chunk boundary, but EAGLE resumes before it. Page 1 misses,
    # moving the endpoint back to 16; the needed page 0 must be recovered.
    assert length == 16
    assert cached.pages[0] == allocated.pages[0]
    assert not cached.pages[0][0].is_null


@pytest.mark.cpu_ok
@pytest.mark.parametrize("lookup_first", [False, True])
def test_prompt_logprob_vision_publication_and_extension_never_hit_text(lookup_first):
    manager = _manager()
    image = _request("image", 33, image="image-A", prompt_logprobs=0)
    if lookup_first:
        assert manager.get_computed_pages(image)[1] == 0
    assert manager.allocate_slots(image, num_new_tokens=32) is not None
    manager.cache_pages(image, num_computed_tokens=32)
    image.num_computed_tokens = 32
    image.num_tokens = 49
    image.all_token_ids = list(range(49))
    assert manager.allocate_slots(image, num_new_tokens=16) is not None
    manager.cache_pages(image, num_computed_tokens=48)
    manager.free(image)
    assert manager.get_computed_pages(_request("text", 49))[1] == 0
    assert manager.get_computed_pages(_request("other_image", 49, image="image-B"))[1] == 0
    assert manager.get_computed_pages(_request("same_image", 49, image="image-A"))[1] == 48


@pytest.mark.cpu_ok
@pytest.mark.parametrize("caching", [False, True])
@pytest.mark.parametrize("seed", ["random", "123"])
def test_cache_manager_accepts_supported_pythonhashseed(monkeypatch, caching, seed):
    monkeypatch.setenv("PYTHONHASHSEED", seed)
    manager = _manager(caching=caching)
    source = _request("source", 33)
    assert manager.allocate_slots(source, num_new_tokens=32) is not None
    manager.cache_pages(source, num_computed_tokens=32)
    manager.free(source)
    assert manager.get_computed_pages(_request("lookup", 33))[1] == (32 if caching else 0)
