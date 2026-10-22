"""Host metadata and abstract tracing regressions: no array computation."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.caching import AttentionMetadataBuilder, OperationsMetadata, RecurrentMetadata, resolve_ragged_cache_layout
from easydel.caching.hybrid.cache import HybridCacheView
from easydel.caching.kda.cache import KDAMetadata
from easydel.caching.mla_ragged_page import cache as mla_module
from easydel.caching.ragged_page.cache import RaggedPagesCacheView
from easydel.caching.recurrent.cache import RecurrentCacheView
from easydel.infra.sharding import coerce_runtime_sharding_resolver
from jax import numpy as jnp
from spectrax import PartitionAxis

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("heads,shards,expected", [(1, 2, 1), (3, 2, 1), (2, 2, 2), (4, 2, 2)])
def test_v2_shards_always_hold_whole_kv_pairs(heads, shards, expected):
    dtype, resolved = resolve_ragged_cache_layout(
        "bfloat16", version="v2", num_kv_heads=heads, k_headdim=128, kv_head_shards=shards
    )
    assert dtype == jnp.dtype("bfloat16")
    assert resolved == expected
    assert heads % resolved == 0


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_head64_public_kv_pages_have_equal_logical_shapes(version):
    metadata = SimpleNamespace(is_v2=version == "v2", is_v3=version == "v3", k_headdim=64, v_headdim=64)
    shape = (2, 4, 4, 64) if version == "v2" else (2, 4, 1, 2, 128)

    def pages(storage):
        view = RaggedPagesCacheView(metadata=metadata, layer_index=0, kv_pages=storage)
        return view.key_pages, view.value_pages

    key, value = jax.eval_shape(pages, jax.ShapeDtypeStruct(shape, jnp.float32))
    assert key.shape == value.shape == (2, 4, 2, 64)


@pytest.mark.parametrize("layer_type", ["full_attention", "parallel_hybrid"])
def test_hybrid_kv_update_accepts_traced_per_row_positions(layer_type):
    def append(old, tokens, positions):
        view = HybridCacheView(
            key=old,
            value=old,
            positions=positions,
            metadata=None,
            conv_state=None,
            recurrent_state=None,
            layer_type=layer_type,
        )
        key, value, updated = view.concatenate_to_cache(key_states=tokens, value_states=tokens)
        return key, value, updated.positions

    key, value, positions = jax.eval_shape(
        append,
        jax.ShapeDtypeStruct((2, 8, 1, 4), jnp.float32),
        jax.ShapeDtypeStruct((2, 1, 1, 4), jnp.float32),
        jax.ShapeDtypeStruct((2,), jnp.int32),
    )
    assert key.shape == value.shape == (2, 8, 1, 4)
    assert positions.shape == (2,)


@pytest.mark.parametrize("metadata", [RecurrentMetadata(), KDAMetadata(), OperationsMetadata.wrap(RecurrentMetadata())])
def test_recurrent_metadata_is_an_empty_compilable_pytree(metadata):
    leaves, tree = jax.tree_util.tree_flatten(metadata)
    assert leaves == []
    assert type(jax.tree_util.tree_unflatten(tree, leaves)) is type(metadata)
    assert jax.eval_shape(lambda x: None, metadata) is None


@pytest.mark.parametrize("position_shape", [(), (2,)])
def test_recurrent_conv_positions_trace_as_shared_or_paired_batch_indices(position_shape):
    def append(old, state, positions):
        view = RecurrentCacheView(
            conv_state=old, recurrent_state=None, positions=None, metadata=SimpleNamespace(conv_kernel_size=4)
        )
        return view.concatenate_to_cache(conv_state=state, cache_position=positions)[0]

    result = jax.eval_shape(
        append,
        jax.ShapeDtypeStruct((2, 3, 4), jnp.float32),
        jax.ShapeDtypeStruct((2, 3), jnp.float32),
        jax.ShapeDtypeStruct(position_shape, jnp.int32),
    )
    assert result.shape == (2, 3, 4)


@pytest.mark.parametrize(
    "scheduled,computed,active,expected",
    [
        ([3, 1, 0], [0, 5, 0], [True, True, False], [1, 2, 2]),
        ([1, 3, 1], [4, 0, 5], [True, True, True], [2, 3, 3]),
        ([1, 1, 0], [4, 5, 0], [True, True, False], [2, 2, 2]),
    ],
)
def test_raw_v3_factories_preserve_request_type_counts(scheduled, computed, active, expected):
    kwargs = dict(
        scheduled_full=np.array(scheduled),
        active_mask_full=np.array(active),
        num_computed_tokens=np.array(computed),
        page_table=np.zeros((3, 2), np.int32),
        version="v3",
    )
    ragged = AttentionMetadataBuilder.compute_ragged_batch_fields_cpu(**kwargs)
    paged = AttentionMetadataBuilder.compute_paged_attention_batch_fields_cpu(
        **kwargs,
        token_ids=np.zeros((3, 16), np.int32),
        num_tokens_static=8,
        padded_num_reqs_in=None,
        min_input_pad=4,
    )
    np.testing.assert_array_equal(ragged["request_distribution"], expected)
    np.testing.assert_array_equal(paged["request_distribution"], expected)
    np.testing.assert_array_equal(ragged["query_start_loc"], paged["query_start_loc"][: len(ragged["query_start_loc"])])


@pytest.mark.parametrize("explicit_page_size,expected", [(None, [[28, 32], [0, 4], [4, 1]]), (8, [[56], [0], [5]])])
def test_raw_v2_factory_uses_config_geometry_without_overriding_explicit_arguments(explicit_page_size, expected):
    config = SimpleNamespace(page_size=4, num_slices_per_kv_cache_update_page=4, max_num_pages_per_req=2)
    kwargs = dict(
        scheduled_full=[5, 0],
        active_mask_full=[True, False],
        num_computed_tokens=[0, 0],
        page_table=[[7, 8], [0, 0]],
        version="v2",
        ragged_config=config,
    )
    if explicit_page_size is not None:
        kwargs["page_size"] = explicit_page_size
    out = AttentionMetadataBuilder.compute_ragged_batch_fields_cpu(**kwargs)
    np.testing.assert_array_equal(out["slot_mapping"][:, : len(expected[0])], expected)
    paged = AttentionMetadataBuilder.compute_paged_attention_batch_fields_cpu(
        **kwargs,
        token_ids=np.zeros((2, 16), np.int32),
        num_tokens_static=8,
        padded_num_reqs_in=None,
        min_input_pad=2,
    )
    np.testing.assert_array_equal(paged["slot_mapping"][:, : len(expected[0])], expected)


@pytest.mark.parametrize("ceiling", [1, 127, 128, 511])
def test_mla_rejects_cap_smaller_than_one_dp_aligned_pool(monkeypatch, ceiling):
    monkeypatch.setattr(mla_module, "per_device_hbm_budget_bytes", lambda *_a, **_kw: 1 << 24)
    resolver = coerce_runtime_sharding_resolver(PartitionAxis(data_parallel_axis="dp", kv_head_axis=None))
    with pytest.raises(ValueError, match="No DP-aligned cache page"):
        mla_module.MLARaggedPagesCacheConfig.create(
            mesh=SimpleNamespace(shape={"dp": 4}),
            runtime_sharding_resolver=resolver,
            kvdtype=jnp.bfloat16,
            num_hidden_layers=1,
            num_kv_heads=1,
            max_model_length=128,
            kv_lora_rank=128,
            page_size=128,
            max_cache_tokens=ceiling,
        )
