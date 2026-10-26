"""Public paged adapter shape/layout contracts, abstract tracing only."""

import jax
import pytest
from easydel.caching import RaggedPagesCacheConfig, RaggedPagesCacheView, RaggedPagesMetadata
from easydel.caching.unified_attention.cache import UnifiedAttentionCacheConfig, UnifiedAttentionCacheView
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.layers.attention import FlexibleAttentionModule
from easydel.operations import OperationMetadata, OperationRegistry
from easydel.operations.requirements import CacheType, ExecutionMode, MetadataField
from ejkernel.types import MaskInfo
from jax import numpy as jnp

pytestmark = pytest.mark.cpu_ok


def _metadata():
    config = EasyDeLBaseConfig(sharding_axis_dims=(1, 1, -1, 1, 1, 1))
    return OperationMetadata(runtime_dtype=jnp.float32, runtime_softmax_dtype=jnp.float32, base_config=config)


def _cache(layout, head_dim, storage):
    if layout == "unified":
        config = UnifiedAttentionCacheConfig(1, 4, 1, head_dim, page_size=2, num_pages=4)
        return UnifiedAttentionCacheView(config, 0, storage[0], storage[1])
    config = RaggedPagesCacheConfig(1, 4, 1, head_dim, head_dim, page_size=2, num_pages=4, version=layout)
    return RaggedPagesCacheView(config, 0, storage[0])


def _storage_shapes(layout, head_dim):
    if layout == "unified":
        return [(4, 2, 1, head_dim)] * 2
    config = RaggedPagesCacheConfig(1, 4, 1, head_dim, head_dim, page_size=2, num_pages=4, version=layout)
    return [config.get_shape_and_axes()[0]]


@pytest.mark.parametrize("layout", ["v2", "v3", "unified"])
@pytest.mark.parametrize("head_dim", [4, 64])
@pytest.mark.parametrize("kind", ["dropout", "ring"])
def test_paged_dropout_traces_layout_preserving_cache_updates(layout, head_dim, kind):
    metadata = _metadata()
    name = (
        ("ring" if layout == "unified" else "ring_paged")
        if kind == "ring"
        else {
            "v2": "ragged_page_attention_v2",
            "v3": "ragged_page_attention_v3",
            "unified": "unified_attention",
        }[layout]
    )
    operation = OperationRegistry.create(name, metadata)
    shapes = [(4, 2, head_dim), (4, 1, head_dim), (4, 1, head_dim), (3, 2), (3,), (4,)]
    inputs = [jax.ShapeDtypeStruct(shape, jnp.float32 if i < 3 else jnp.int32) for i, shape in enumerate(shapes)]
    inputs += [jax.ShapeDtypeStruct(shape, jnp.float32) for shape in _storage_shapes(layout, head_dim)]
    inputs += [jax.ShapeDtypeStruct((2,), jnp.uint32)]

    def forward(q, k, v, table, contexts, starts, *storage_and_rng):
        cache = _cache(layout, head_dim, storage_and_rng[:-1])
        runtime = RaggedPagesMetadata(table, contexts, starts, contexts, page_size=2, version="v3")
        result = operation(
            query=q,
            key=k,
            value=v,
            cache_view=cache,
            cache_metadata=runtime,
            compute_dtype=jnp.float32,
            deterministic=kind == "ring",
            dropout_prob=0.25,
            dropout_rng=storage_and_rng[-1],
            normalize_output=kind == "ring",
        )
        updated = result.cache_view
        arrays = (updated.key_cache, updated.value_cache) if layout == "unified" else (updated.kv_pages,)
        return result.attention_outputs, *arrays

    program = jax.make_jaxpr(forward)(*inputs)
    assert [aval.shape for aval in program.out_avals] == [(4, 2, head_dim), *_storage_shapes(layout, head_dim)]
    assert all(aval.dtype == jnp.float32 for aval in program.out_avals)
    # Public output includes the new cache state as distinct scatter results,
    # rather than dropping writes or returning a dense replacement cache.
    assert len(program.out_avals) == (3 if layout == "unified" else 2)


@pytest.mark.parametrize("layout", ["v2", "v3", "unified"])
@pytest.mark.parametrize("rank_major", [False, True])
@pytest.mark.parametrize("kind", ["ring", "dropout", "cache_only"])
def test_rank_major_batched_masks_bias_segments_and_cache_only_contract(layout, rank_major, kind):
    operation = OperationRegistry.create(
        "unified_attention" if kind == "cache_only" and layout == "unified" else "paged_attention_reference",
        _metadata(),
    )
    shapes = [(2, 2, 2, 4), (2, 2, 1, 4), (2, 2, 1, 4), (2, 2), (2, 2)]
    inputs = [jax.ShapeDtypeStruct(shape, jnp.float32 if i < 3 else jnp.int32) for i, shape in enumerate(shapes)]
    inputs += [jax.ShapeDtypeStruct(shape, jnp.float32) for shape in _storage_shapes(layout, 4)]

    def forward(q, k, v, qs, ks, *storage):
        cache = _cache(layout, 4, storage)
        runtime = RaggedPagesMetadata(
            jnp.arange(8, dtype=jnp.int32).reshape(2, 2, 2) % 4,
            jnp.full((2, 2), 2, dtype=jnp.int32),
            jnp.broadcast_to(jnp.arange(3, dtype=jnp.int32), (2, 3)) if rank_major else jnp.arange(5),
            jnp.full((2, 2), 2, dtype=jnp.int32),
            page_size=2,
        )
        result = operation(
            query=q,
            **({} if kind == "cache_only" else {"key": k, "value": v}),
            cache_view=cache,
            cache_metadata=runtime,
            mask_info=MaskInfo.from_segments(qs, ks),
            bias=jnp.zeros((2, 2, 2, 4), dtype=jnp.float32),
            alibi_slopes=jnp.ones((2,), dtype=jnp.float32),
            deterministic=kind == "ring",
            dropout_prob=0.25,
            dropout_rng=jax.random.key(15),
            dense_impl="ring" if kind == "ring" else "vanilla",
            compute_dtype=jnp.float32,
        )
        return result.attention_outputs

    program = jax.make_jaxpr(forward)(*inputs)
    assert [(aval.shape, aval.dtype) for aval in program.out_avals] == [((2, 2, 2, 4), jnp.float32)]
    # Both independently supplied segment operands must affect the program
    # rather than being dropped at the adapter boundary.
    used = [var for eqn in program.jaxpr.eqns for var in eqn.invars]
    assert any(var is program.jaxpr.invars[3] for var in used)
    assert any(var is program.jaxpr.invars[4] for var in used)


@pytest.mark.parametrize("layout", ["v2", "v3", "unified"])
@pytest.mark.parametrize("feature", ["mask", "bias", "segments"])
@pytest.mark.parametrize("dropout", [False, True])
def test_direct_registered_bthd_paged_features_keep_batch_mapping(layout, feature, dropout):
    """Four requests in two batches must retain batch-local feature rows."""
    operation = OperationRegistry.create(
        {"v2": "ragged_page_attention_v2", "v3": "ragged_page_attention_v3", "unified": "unified_attention"}[layout],
        _metadata(),
    )
    shapes = [(2, 2, 2, 4), (2, 2, 1, 4), (2, 2, 1, 4), *_storage_shapes(layout, 4)]
    descriptors = [jax.ShapeDtypeStruct(shape, jnp.float32) for shape in shapes]

    def forward(q, k, v, *storage):
        runtime = RaggedPagesMetadata(
            jnp.arange(8, dtype=jnp.int32).reshape(4, 2) % 4,
            jnp.ones((4,), dtype=jnp.int32),
            jnp.arange(5, dtype=jnp.int32),
            jnp.ones((4,), dtype=jnp.int32),
            page_size=2,
        )
        kwargs = {}
        if feature == "mask":
            kwargs["mask_info"] = MaskInfo(_attention_mask=jnp.ones((2, 1, 2, 4), dtype=jnp.bool_))
        elif feature == "segments":
            kwargs["mask_info"] = MaskInfo.from_segments(
                jnp.ones((2, 2), dtype=jnp.int32), jnp.ones((2, 4), dtype=jnp.int32)
            )
        else:
            kwargs["bias"] = jnp.zeros((2, 2, 2, 4), dtype=jnp.float32)
        result = operation(
            query=q,
            key=k,
            value=v,
            cache_view=_cache(layout, 4, storage),
            cache_metadata=runtime,
            deterministic=not dropout,
            dropout_prob=0.25,
            dropout_rng=jax.random.key(21),
            normalize_output=False,
            compute_dtype=jnp.float32,
            **kwargs,
        )
        cache = result.cache_view
        pages = (cache.key_cache, cache.value_cache) if layout == "unified" else (cache.kv_pages,)
        return result.attention_outputs, *pages

    program = jax.make_jaxpr(forward)(*descriptors)
    assert [(aval.shape, aval.dtype) for aval in program.out_avals] == [
        ((2, 2, 2, 4), jnp.float32),
        *[(shape, jnp.float32) for shape in _storage_shapes(layout, 4)],
    ]


@pytest.mark.parametrize("mode", list(ExecutionMode))
def test_registered_ring_declares_paged_request_contract(mode):
    operation = OperationRegistry.create("ring_paged", _metadata())
    requirements = operation.get_requirements(mode)
    assert requirements.cache.is_compatible_with(CacheType.RAGGED_PAGES)
    for field in (MetadataField.PAGES_TABLES, MetadataField.CONTEXT_LENS, MetadataField.QUERY_START_LOC):
        assert field in requirements.metadata.required


@pytest.mark.parametrize("layout", ["dense", "v2", "v3", "unified"])
@pytest.mark.parametrize("normalized", [False, True])
def test_public_flexible_layer_routes_active_dropout_and_normalization(layout, normalized):
    config = EasyDeLBaseConfig(
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
    )
    layer = FlexibleAttentionModule(config, softmax_scale=0.5, dropout_prob=0.25)
    descriptors = [jax.ShapeDtypeStruct(shape, jnp.float32) for shape in [(1, 4, 2, 4), (1, 4, 1, 4), (1, 4, 1, 4)]]
    if layout != "dense":
        descriptors += [jax.ShapeDtypeStruct(shape, jnp.float32) for shape in _storage_shapes(layout, 4)]

    def forward(q, k, v, *storage):
        cache = None if layout == "dense" else _cache(layout, 4, storage)
        runtime = (
            None
            if cache is None
            else RaggedPagesMetadata(
                jnp.array([[0, 1], [2, 3]], dtype=jnp.int32),
                jnp.array([2, 2], dtype=jnp.int32),
                jnp.array([0, 2, 4], dtype=jnp.int32),
                jnp.array([2, 2], dtype=jnp.int32),
                page_size=2,
            )
        )
        result = layer(
            query_states=q,
            key_states=k,
            value_states=v,
            mode=None,
            cache_view=cache,
            cache_metadata=runtime,
            deterministic=False,
            dropout_rng=jax.random.key(17),
            normalize_output=normalized,
            compute_dtype=jnp.float32,
        )
        return result.attention_outputs

    program = jax.make_jaxpr(forward)(*descriptors)
    assert [(aval.shape, aval.dtype) for aval in program.out_avals] == [((1, 4, 2, 4), jnp.float32)]


@pytest.mark.parametrize("normalized", [True, False])
def test_vanilla_weight_dropout_traces_fp32_weights_and_value_layout(normalized):
    operation = OperationRegistry.create("vanilla", _metadata())
    descriptors = [
        jax.ShapeDtypeStruct((1, 3, 2, 4), jnp.bfloat16),
        jax.ShapeDtypeStruct((1, 5, 1, 4), jnp.bfloat16),
        jax.ShapeDtypeStruct((1, 5, 1, 6), jnp.bfloat16),
        jax.ShapeDtypeStruct((1, 1, 3, 5), jnp.bool_),
        jax.ShapeDtypeStruct((2,), jnp.uint32),
    ]

    def forward(q, k, v, mask, rng):
        result = operation(
            q,
            k,
            v,
            mask_info=MaskInfo(_attention_mask=mask),
            compute_dtype=jnp.bfloat16,
            dropout_prob=0.25,
            deterministic=False,
            dropout_rng=rng,
            normalize_output=normalized,
        )
        return result.attention_outputs, result.attention_weights

    program = jax.make_jaxpr(forward)(*descriptors)
    assert [(aval.shape, aval.dtype) for aval in program.out_avals] == [
        ((1, 3, 2, 6), jnp.bfloat16),
        ((1, 2, 3, 5), jnp.float32),
    ]
