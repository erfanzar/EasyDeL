"""CPU-only runtime dispatch and page-pool sizing contracts (no array math)."""

from types import SimpleNamespace

import jax
import pytest
from easydel.caching import (
    PagedCacheBudget,
    RaggedPagesCacheConfig,
    RaggedPagesMetadata,
    UnifiedAttentionCacheConfig,
    cap_paged_cache_pages,
    compute_paged_cache_budget,
)
from easydel.infra.sharding import coerce_runtime_sharding_resolver
from jax import numpy as jnp
from spectrax import PartitionAxis

pytestmark = pytest.mark.cpu_ok


def _payload(name, page_size):
    # Identity sentinels prove every read/write field travels together without
    # creating or evaluating JAX arrays.
    return RaggedPagesMetadata(
        pages_tables=object(),
        context_lens=object(),
        query_start_loc=object(),
        num_seqs=object(),
        slot_mapping=object(),
        num_kv_update_slices=object(),
        position_ids=name,
        page_size=page_size,
        version="v2",
    )


def test_group_selection_keeps_reads_writes_lengths_and_layout_together():
    full = _payload("full", 128)
    window = _payload("window", 64)
    grouped = full.replace(cache_groups=(full, window), layer_to_cache_group=(1, 0, 1))
    for layer, expected in enumerate((window, full, window)):
        selected = grouped.for_layer(layer)
        assert selected is expected
        assert selected.pages_tables is expected.pages_tables
        assert selected.slot_mapping is expected.slot_mapping
        assert selected.context_lens is expected.context_lens
        assert selected.position_ids == expected.position_ids
        assert selected.for_layer(layer) is selected
    assert grouped.for_group(0) is full
    assert grouped.for_group(1) is window


def test_single_table_api_is_unchanged_and_invalid_group_mapping_fails_closed():
    legacy = _payload("legacy", 128)
    assert legacy.for_layer(None) is legacy
    assert legacy.for_layer(500) is legacy
    assert legacy.for_group(0) is legacy
    with pytest.raises(ValueError):
        legacy.for_group(1)
    grouped = legacy.replace(cache_groups=(legacy,), layer_to_cache_group=(2,))
    for layer in (None, -1, 1):
        with pytest.raises(ValueError, match="no mapping"):
            grouped.for_layer(layer)
    with pytest.raises(ValueError, match="Invalid cache group"):
        grouped.for_layer(0)
    with pytest.raises(ValueError, match="nested"):
        grouped.replace(cache_groups=(grouped,)).for_group(0)


def test_group_payloads_survive_pytree_roundtrip_and_static_layer_mapping():
    full, window = _payload("full", 128), _payload("window", 64)
    root = full.replace(cache_groups=(full, window), layer_to_cache_group=(0, 1))
    leaves, tree = jax.tree_util.tree_flatten(root)
    restored = jax.tree_util.tree_unflatten(tree, leaves)
    assert restored.layer_to_cache_group == (0, 1)
    assert restored.for_layer(1).pages_tables is window.pages_tables
    assert restored.for_layer(1).slot_mapping is window.slot_mapping
    assert restored.for_layer(1).page_size == 64


def test_mixed_budget_charges_busiest_physical_stage_and_consumes_generators_once():
    result = compute_paged_cache_budget(
        {0: 100, 1: 200, 2: 500},
        available_bytes=8000,
        page_size=16,
        data_parallel_size=4,
        stage_layer_groups=(iter(group) for group in ((0, 2), (1,))),
        max_cache_tokens=190,
    )
    assert result == PagedCacheBudget(num_pages=8, page_bytes=600, num_hidden_layers=2)
    assert cap_paged_cache_pages(100, page_size=16, data_parallel_size=4, max_cache_tokens=190) == 8


@pytest.mark.parametrize("ceiling", [1, 31, 63])
def test_override_never_exceeds_sub_dp_token_ceiling(ceiling):
    with pytest.raises(ValueError, match="No DP-aligned cache page"):
        compute_paged_cache_budget(
            {0: 16},
            available_bytes=None,
            page_size=16,
            data_parallel_size=4,
            num_pages_override=100,
            max_cache_tokens=ceiling,
        )


@pytest.mark.parametrize("cache_class", [RaggedPagesCacheConfig, UnifiedAttentionCacheConfig])
@pytest.mark.parametrize("ceiling, expected_pages", [(1, None), (63, None), (64, 4), (159, 8)])
def test_homogeneous_configs_obey_shared_dp_and_token_ceiling(monkeypatch, cache_class, ceiling, expected_pages):
    monkeypatch.setattr(cache_class, "_compute_free_hbm", staticmethod(lambda **_kw: 1 << 30))
    resolver = coerce_runtime_sharding_resolver(PartitionAxis(kv_head_axis="tp", data_parallel_axis="dp"))
    kwargs = dict(
        mesh=SimpleNamespace(shape={"tp": 1, "dp": 4}),
        runtime_sharding_resolver=resolver,
        kvdtype=jnp.bfloat16,
        num_hidden_layers=2,
        num_kv_heads=2,
        max_model_length=1024,
        page_size=16,
        max_cache_tokens=ceiling,
    )
    if cache_class is RaggedPagesCacheConfig:
        kwargs["kv_head_dim_size"] = 64
    else:
        kwargs["head_dim"] = 64
    if expected_pages is None:
        with pytest.raises(ValueError, match="No DP-aligned cache page"):
            cache_class.create(**kwargs)
    else:
        config = cache_class.create(**kwargs)
        assert config.num_pages == expected_pages
        assert config.num_pages * config.page_size <= ceiling
