"""CPU-only configuration and real registered-adapter integration; host kernel substitutes."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.caching import (
    ParallelHybridCacheView,
    RaggedPagesCacheConfig,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    UnifiedAttentionCacheView,
)
from easydel.caching.unified_attention.cache import UnifiedAttentionCacheConfig
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.infra.etils import EasyDeLBackends
from easydel.layers.attention import FlexibleAttentionModule
from easydel.operations import OperationRegistry
from easydel.operations.kernels import unified_attention as unified_kernel
from spectrax import common_types

from .test_cache_group_dispatch import _MaskAwarePageAttention, _metadata, _performer

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("layout", ["v2", "v3", "unified"])
@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
def test_real_registered_paged_dropout_traces_cache_preserving_layer_dispatch(layout, hybrid, normalized):
    """Abstract tracing runs registered operations without array allocation or execution."""
    config = EasyDeLBaseConfig(
        attn_mechanism={
            "v2": "ragged_page_attention_v2",
            "v3": "ragged_page_attention_v3",
            "unified": "unified_attention",
        }[layout],
        attn_dtype=np.float32,
        attn_softmax_dtype=np.float32,
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
        layer_types=["chunked_attention"],
        attention_chunk_size=2,
    )
    performer = FlexibleAttentionModule(config, softmax_scale=0.5)
    if layout == "unified":
        cache_config = UnifiedAttentionCacheConfig(1, 4, 1, 4, page_size=2, num_pages=4)
        storage_shapes = [(4, 2, 1, 4)] * 2
    else:
        cache_config = RaggedPagesCacheConfig(1, 4, 1, 4, 4, page_size=2, num_pages=4, version=layout)
        storage_shapes = [cache_config.get_shape_and_axes()[0]]

    def forward(q, k, v, tables, contexts, starts, rng, *storage):
        if layout == "unified":
            view = UnifiedAttentionCacheView(cache_config, 0, storage[0], storage[1])
        else:
            view = RaggedPagesCacheView(cache_config, 0, storage[0])
        if hybrid:
            view = ParallelHybridCacheView(transformer=view, recurrent=None)
        runtime = RaggedPagesMetadata(tables, contexts, starts, contexts, page_size=2, version="v3")
        result = performer(
            q,
            k,
            v,
            mode=common_types.MODE_DECODE,
            cache_view=view,
            cache_metadata=runtime,
            deterministic=False,
            dropout_prob=0.25,
            dropout_rng=rng,
            normalize_output=normalized,
        )
        updated = result.cache_view.transformer if hybrid else result.cache_view
        arrays = (updated.key_cache, updated.value_cache) if layout == "unified" else (updated.kv_pages,)
        return result.attention_outputs, *arrays

    shapes = [(4, 2, 4), (4, 1, 4), (4, 1, 4), (3, 2), (3,), (4,), (2,)]
    descriptors = [
        jax.ShapeDtypeStruct(shape, np.float32 if i < 3 else np.uint32 if i == 6 else np.int32)
        for i, shape in enumerate(shapes)
    ]
    descriptors += [jax.ShapeDtypeStruct(shape, np.float32) for shape in storage_shapes]
    program = jax.make_jaxpr(forward)(*descriptors)
    assert [(aval.shape, aval.dtype) for aval in program.out_avals] == [
        ((4, 2, 4), np.dtype(np.float32)),
        *((shape, np.dtype(np.float32)) for shape in storage_shapes),
    ]


@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("mode", [common_types.MODE_PREFILL, common_types.MODE_DECODE])
@pytest.mark.parametrize("view_class", [RaggedPagesCacheView, UnifiedAttentionCacheView])
def test_real_config_chunk_reaches_paged_dispatch_without_synthetic_cache_fields(monkeypatch, hybrid, mode, view_class):
    performer = _performer(monkeypatch)
    # Uses the production configuration-to-mask conversion. `size` is the sliding
    # window, while `chunks` carries attention_chunk_size in this mapping.
    config = EasyDeLBaseConfig(layer_types=["full_attention", "chunked_attention"], attention_chunk_size=4)
    object.__setattr__(performer, "config", config)
    object.__setattr__(performer, "impl", _MaskAwarePageAttention())
    if view_class is UnifiedAttentionCacheView:
        view = view_class(
            metadata=None, layer_index=1, key_cache=np.zeros((1, 1, 1, 1)), value_cache=np.zeros((1, 1, 1, 1))
        )
    else:
        view = view_class(metadata=None, layer_index=1, kv_pages=np.zeros((1, 1, 1, 1)))
    assert not hasattr(view, "masking_details")
    if hybrid:
        view = ParallelHybridCacheView(transformer=view, recurrent=None, layer_index=0)
    q = np.zeros((1, 1, 1, 1), dtype=np.float32)
    result = performer(q, q, q, mode=mode, cache_view=view)
    np.testing.assert_array_equal(result.attention_outputs, np.full_like(q, 4))
    assert isinstance(result.cache_view, type(view))


@pytest.mark.parametrize("chunk_size", [None, 0, -4])
@pytest.mark.parametrize("backend", ["ragged_page_attention_v3", "unified_attention"])
@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("mode", [common_types.MODE_PREFILL, common_types.MODE_DECODE])
@pytest.mark.parametrize("view_class", [RaggedPagesCacheView, UnifiedAttentionCacheView])
def test_configured_chunk_without_positive_size_fails_before_dispatch(
    monkeypatch, chunk_size, backend, hybrid, mode, view_class
):
    performer = _performer(monkeypatch, backend=backend, decode_backend=backend)
    config = EasyDeLBaseConfig(layer_types=["full_attention", "chunked_attention"], attention_chunk_size=chunk_size)
    object.__setattr__(performer, "config", config)
    details = config.get_mask_details()[1]
    assert details.chunks == chunk_size
    assert details.size is None
    if view_class is UnifiedAttentionCacheView:
        view = view_class(metadata=None, layer_index=1, key_cache=None, value_cache=None)
    else:
        view = view_class(metadata=None, layer_index=1, kv_pages=None)
    if hybrid:
        # The component index, not the wrapper index, owns the mask configuration.
        view = ParallelHybridCacheView(transformer=view, recurrent=None, layer_index=0)
    # No arrays are needed: an invalid CHUNK contract must not reach the backend.
    with pytest.raises(ValueError, match="positive configured chunk size"):
        performer(None, None, None, mode=mode, cache_view=view)


@pytest.mark.parametrize("mode", [common_types.MODE_PREFILL, common_types.MODE_DECODE])
def test_hybrid_uses_real_registered_unified_adapter_and_preserves_both_components(monkeypatch, mode):
    performer = _performer(monkeypatch)
    resolver = SimpleNamespace(resolve=lambda *, shape, **kwargs: (None,) * len(shape))
    resolver.with_mesh = lambda mesh: resolver
    metadata = SimpleNamespace(
        backend=EasyDeLBackends.CPU,
        requires_cache=None,
        runtime_dtype=np.float32,
        runtime_sharding_resolver=resolver,
        mesh=None,
        get_operation_config=lambda name: None,
    )
    # The actual registered class, requirements, dispatch and forward_native run.
    # Only the final device kernel is replaced with a host page-table read.
    operation = OperationRegistry.create("unified_attention", metadata)
    object.__setattr__(performer, "impl", operation)
    monkeypatch.setattr(unified_kernel.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(unified_kernel, "jnp", np)
    monkeypatch.setattr(unified_kernel, "dp_page_axis", lambda view: None)
    monkeypatch.setattr(unified_kernel, "mesh_axis_size", lambda mesh, axis: 1)

    def host_kernel(query, keys, values, contexts, tables, starts, *args, **kwargs):
        assert keys.shape == values.shape == (4, 1, 1, 1)
        return np.broadcast_to(values[int(tables[0, 0]), 0], query.shape)

    monkeypatch.setattr(unified_kernel, "unified_attention", host_kernel)
    component = UnifiedAttentionCacheView(
        metadata=None,
        layer_index=1,
        key_cache=np.zeros((4, 1, 1, 1), dtype=np.float32),
        value_cache=np.array([3, 7, 13, 29], dtype=np.float32).reshape(4, 1, 1, 1),
    )
    recurrent = SimpleNamespace(conv_state=None, recurrent_state=None, positions=None, seqlen_offset=None)
    wrapper = ParallelHybridCacheView(transformer=component, recurrent=recurrent, layer_index=1)
    runtime = _metadata(0)
    runtime.cache_groups = (_metadata(1), _metadata(3))
    runtime.layer_to_cache_group = (0, 1)
    q = np.zeros((1, 1, 1, 1), dtype=np.float32)
    result = performer(q, q, q, mode=mode, cache_view=wrapper, cache_metadata=runtime)
    np.testing.assert_array_equal(result.attention_outputs, np.full_like(q, 29))
    assert isinstance(result.cache_view, ParallelHybridCacheView)
    assert result.cache_view.transformer is component
    assert result.cache_view.recurrent is recurrent
    np.testing.assert_array_equal(result.cache_view.transformer.value_cache, component.value_cache)
