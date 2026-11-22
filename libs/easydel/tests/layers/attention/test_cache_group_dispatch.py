"""Host-only cache-group dispatch and shared vision embedding compatibility checks.

NumPy substitutes for the layer array namespace; no JAX arrays or kernels run.
"""

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
from easydel.caching import (
    OperationsMetadata,
    ParallelHybridCacheView,
    RaggedPagesCacheView,
    RaggedPagesMetadata,
    UnifiedAttentionCacheView,
)
from easydel.infra.utils import AttnMaskDetail, AttnMaskType
from easydel.layers.attention import AttentionModule, FlexibleAttentionModule, _flexible
from easydel.layers.embeddings import VisionPatchEmbed, _vision
from easydel.operations import AttentionOutput
from spectrax import common_types

pytestmark = pytest.mark.cpu_ok


def _metadata(page, *, version="v3"):
    return RaggedPagesMetadata(
        pages_tables=np.array([[page]], dtype=np.int32),
        context_lens=np.array([1], dtype=np.int32),
        query_start_loc=np.array([0, 1], dtype=np.int32),
        num_seqs=np.array([1], dtype=np.int32),
        slot_mapping=np.array([[page], [0], [1]], dtype=np.int32),
        position_ids=np.array([0], dtype=np.int32),
        request_distribution=np.array([1, 1, 1], dtype=np.int32),
        version=version,
    )


class _PageAttention:
    metadata = SimpleNamespace(runtime_dtype=np.float32)

    def __init__(self, name="ragged_page_attention_v3"):
        self.name = name

    def get_impl_name(self):
        return self.name

    def __call__(self, *, cache_metadata, cache_view, chunk_size=None, sliding_window=None, **kwargs):
        # Independently read the page selected by dispatch; a stale global table reads
        # a different value even when the write correctly selected the local group.
        page = int(cache_metadata.pages_tables[0, 0])
        output = cache_view.kv_pages[page].reshape(1, 1, 1, 1)
        return AttentionOutput(attention_outputs=output, cache_view=cache_view)


def _performer(monkeypatch, backend="ragged_page_attention_v3", decode_backend=None):
    monkeypatch.setattr(_flexible, "_attention_mesh_context", lambda config: nullcontext())
    monkeypatch.setattr(_flexible.jax, "default_backend", lambda: "cpu")
    module = object.__new__(FlexibleAttentionModule)
    for name, value in {
        "config": SimpleNamespace(output_attentions=False),
        "impl": _PageAttention(backend),
        "impl_decode": _PageAttention(decode_backend) if decode_backend else None,
        "deterministic": True,
        "dropout_prob": 0.0,
        "softmax_scale": 1.0,
    }.items():
        object.__setattr__(module, name, value)
    return module


@pytest.mark.parametrize("view_class", [RaggedPagesCacheView, UnifiedAttentionCacheView])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("version", ["v2", "v3"])
@pytest.mark.parametrize("hybrid", [False, True])
def test_group_metadata_selects_matching_writes_and_attention_reads(monkeypatch, view_class, wrapped, version, hybrid):
    first, second = _metadata(1, version=version), _metadata(3, version=version)
    grouped = _metadata(0, version=version)
    grouped.cache_groups = (first, second)
    grouped.layer_to_cache_group = (0, 1)
    runtime = OperationsMetadata(ragged=grouped) if wrapped else grouped

    def write(self, *, key, value, cache_metadata):
        # Unlike production cache views, this host fake does NOT select for_layer;
        # it exposes whether the attention layer prepared the write correctly.
        page = int(cache_metadata.slot_mapping[0, 0])
        self.kv_pages[page] = value.reshape(-1)[0]
        return self

    monkeypatch.setattr(view_class, "concatenate_to_cache", write)
    attention = object.__new__(AttentionModule)
    object.__setattr__(attention, "dtype", np.float32)
    performer = _performer(monkeypatch)
    values = np.array([11, 29], dtype=np.float32)
    for layer_index in (0, 1):
        view = object.__new__(view_class)
        view.layer_index = layer_index
        view.kv_pages = np.zeros(4, dtype=np.float32)
        view.masking_details = None
        if hybrid:
            view = ParallelHybridCacheView(transformer=view, recurrent=None, layer_index=layer_index)
        q = np.zeros((1, 1, 1, 1), dtype=np.float32)
        value = values[layer_index : layer_index + 1].reshape(1, 1, 1, 1)
        _, _, _, _, updated, local = attention.concatenate(
            query=q, key=q, value=value, mask_info=None, cache_view=view, cache_metadata=runtime
        )
        assert local is (first, second)[layer_index]
        # Direct callers may still supply the wrapper to forward. Both entry points
        # must independently resolve it, not rely on concatenate's return being used.
        result = performer(q, q, value, mode=common_types.MODE_DECODE, cache_view=updated, cache_metadata=runtime)
        np.testing.assert_array_equal(result.attention_outputs, value)
        np.testing.assert_array_equal(view.kv_pages[[1, 3]], [11, 0] if layer_index == 0 else [0, 29])


class _MaskAwarePageAttention(_PageAttention):
    def __call__(self, *, value, chunk_size=None, sliding_window=None, cache_view, **kwargs):
        # Encode the public kernel options into the output so missing forwarding is observable.
        result = np.full_like(value, chunk_size if chunk_size is not None else sliding_window or 0)
        return AttentionOutput(attention_outputs=result, cache_view=cache_view)


@pytest.mark.parametrize("kind,size", [(AttnMaskType.FULL, 0), (AttnMaskType.SLIDING, 5), (AttnMaskType.CHUNK, 4)])
@pytest.mark.parametrize("mode", [common_types.MODE_PREFILL, common_types.MODE_DECODE])
def test_layer_mask_details_reach_registered_dispatch(monkeypatch, kind, size, mode):
    performer = _performer(monkeypatch)
    object.__setattr__(performer, "impl", _MaskAwarePageAttention())
    view = object.__new__(RaggedPagesCacheView)
    view.layer_index = 0
    # Real paged schemas have no masking_details field. Configuration must
    # reach dispatch directly, without a synthetic attribute attached to the view.
    performer.config.get_mask_details = lambda: {0: AttnMaskDetail(mask_type=kind, size=size)}
    q = np.zeros((1, 1, 1, 1), dtype=np.float32)
    result = performer(q, q, q, mode=mode, cache_view=view)
    np.testing.assert_array_equal(result.attention_outputs, np.full_like(q, size))


@pytest.mark.parametrize("backend", ["ragged_page_attention_v2", "unified_attention", "vanilla"])
def test_unsupported_chunk_backend_fails_instead_of_silently_ignoring_mask(monkeypatch, backend):
    performer = _performer(monkeypatch, decode_backend=backend)
    view = object.__new__(RaggedPagesCacheView)
    view.layer_index = 0
    view.masking_details = AttnMaskDetail(mask_type=AttnMaskType.CHUNK, size=4)
    with pytest.raises(NotImplementedError, match="ragged_page_attention_v3"):
        performer(None, None, None, mode=common_types.MODE_DECODE, cache_view=view)


@pytest.mark.parametrize("view_class", [RaggedPagesCacheView, UnifiedAttentionCacheView])
@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
def test_paged_dropout_never_routes_partial_dense_qkv_into_vanilla(monkeypatch, view_class, hybrid, normalize):
    performer = _performer(monkeypatch)
    object.__setattr__(performer, "metadata", _PageAttention.metadata)
    object.__setattr__(performer, "_requires_cache", True)
    view = object.__new__(view_class)
    view.layer_index = 0
    view.kv_pages = np.array([17], dtype=np.float32)

    # The selected page contains prior context, unlike the current dense V=0.
    class Reference(_PageAttention):
        def __call__(self, *, value, cache_view, normalize_output, dropout_rng, **kwargs):
            assert isinstance(cache_view, view_class)
            weight = np.random.default_rng(dropout_rng).random() >= 0.5
            scale = 1 if normalize_output else 3
            output = np.full_like(value, cache_view.kv_pages[0] * weight * 2 * scale)
            return AttentionOutput(attention_outputs=output, cache_view=cache_view)

    def create(impl_name, **kwargs):
        assert impl_name == "paged_attention_reference"
        return Reference(impl_name)

    monkeypatch.setattr(_flexible.OperationRegistry, "create", create)
    if hybrid:
        view = ParallelHybridCacheView(transformer=view, recurrent=None)
    q = np.zeros((1, 1, 1, 1), dtype=np.float32)
    result = performer(
        q,
        q,
        q,
        mode=None,
        cache_view=view,
        deterministic=False,
        dropout_prob=0.5,
        dropout_rng=42,
        normalize_output=normalize,
    )
    np.testing.assert_array_equal(result.attention_outputs, np.full_like(q, 34 if normalize else 102))
    assert isinstance(result.cache_view, type(view))


def test_shared_patch_embedding_preserves_flattened_channel_first_pixels_and_dtype(monkeypatch):
    from easydel.modules.qwen2_vl.modeling_qwen2_vl import Qwen2VLPatchEmbed

    assert VisionPatchEmbed is Qwen2VLPatchEmbed

    # A host convolution with an independently known kernel avoids JAX allocation.
    # Its parameter storage remains under proj, as it does in the real module.
    class HostConv:
        def __init__(self, **kwargs):
            assert kwargs["kernel_size"] == kwargs["stride"] == (2, 2, 2)
            assert kwargs["use_bias"] is False
            assert kwargs["precision"] == "highest"
            self.kernel = np.arange(32, dtype=kwargs["param_dtype"]).reshape(2, 2, 2, 2, 2) / 32

        def __call__(self, pixels):
            assert pixels.dtype == np.float32
            return np.einsum("nthwc,thwcd->nd", pixels, self.kernel.astype(pixels.dtype))[:, None, None, None, :]

    manager = object()

    def shard(outputs, *, partition_manager, **kwargs):
        assert partition_manager is manager
        return outputs

    monkeypatch.setattr(_vision.nn, "Conv", HostConv)
    monkeypatch.setattr(_vision, "jnp", np)
    monkeypatch.setattr(_vision, "apply_logical_sharding", shard)
    layer = VisionPatchEmbed(
        patch_size=2,
        temporal_patch_size=2,
        in_channels=2,
        embed_dim=2,
        precision="highest",
        dtype=np.float32,
        param_dtype=np.float16,
        partition_manager=manager,
        rngs=object(),
    )
    pixels = np.arange(48, dtype=np.float16).reshape(3, 16) / 16
    actual = layer(pixels)
    # Explicit flattened-channel-first loop, not the production transpose/conv path.
    expected = np.zeros((3, 2), dtype=np.float32)
    for n in range(3):
        for c in range(2):
            for t in range(2):
                for h in range(2):
                    for w in range(2):
                        expected[n] += pixels[n, c * 8 + t * 4 + h * 2 + w] * layer.proj.kernel[t, h, w, c].astype(
                            np.float32
                        )
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    assert actual.dtype == np.float32
    assert layer.proj.kernel.dtype == np.float16
