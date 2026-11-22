"""Sliding-cache masks retain cache-owned coordinates through attention dispatch."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.caching import ParallelHybridCacheView, TransformerCacheConfig, TransformerCacheView, TransformerMetadata
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.infra.utils import AttnMaskDetail, AttnMaskType
from easydel.layers.attention import AttentionModule, FlexibleAttentionModule, _flexible
from easydel.operations import AttentionOutput
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from spectrax import common_types


class _PreparedCacheAttention(AttentionModule):
    def _handle_cache_concat(self, *, key, value, mask_info, cache_view, **kwargs):
        # Supply the cache's already-aligned result; exercise public concatenate.
        return key, value, mask_info, cache_view, cache_view.masking_details


def _view(key, value, indexes, starts):
    return TransformerCacheView(
        key=key,
        value=value,
        indexes=indexes,
        starts=starts,
        metadata=TransformerCacheConfig.create(4, 3, 1, num_heads=1, head_dim=2),
        masking_details=AttnMaskDetail(mask_type=AttnMaskType.SLIDING, size=3),
        maximum_sequence_length=32,
        layer_index=0,
    )


def _attention(cls=AttentionModule):
    attention = cls.__new__(cls)
    attention.dtype = np.float32
    attention.config = SimpleNamespace(runtime_sharding_resolver=None, kv_cache_quantization_config=None)
    return attention


def _window_bounds(window):
    return (max(window - 1, 0),) * 2 if isinstance(window, int) else window


@pytest.mark.cpu_ok
@pytest.mark.parametrize("length", [1, 4])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", [None, 3, 2, (1, 0), (0, 2), 0, 5, (4, 4)])
def test_baked_cache_window_and_distinct_overrides_preserve_aligned_public_outputs(monkeypatch, length, causal, window):
    monkeypatch.setattr(_flexible, "jnp", np)
    query_positions = np.broadcast_to(np.arange(8, 8 + length), (4, length))
    key_positions = np.broadcast_to(np.arange(6, 8 + length), (4, 2 + length))
    relative = key_positions[:, None, :] - query_positions[:, :, None]
    configured = (relative >= -2) & (relative <= (0 if causal else 2))
    mask = MaskInfo(
        _attention_mask=configured[:, None],
        _q_segment_ids=np.ones_like(query_positions),
        _kv_segment_ids=np.ones_like(key_positions),
        q_positions=query_positions,
        kv_positions=key_positions,
        sliding_window_baked_in=True,
        causal_mask_baked_in=causal,
    )
    key = np.arange(4 * (2 + length) * 2, dtype=np.float32).reshape(4, 2 + length, 1, 2)
    value = key + 10
    query = np.zeros((4, length, 1, 2), np.float32)
    view = _view(key[:, -3:], value[:, -3:], np.full((4,), 8 + length), np.zeros((4,), np.int32))
    metadata = TransformerMetadata(indexes=view.indexes, starts=view.starts)
    out_key, out_value, out_mask, _, updated, out_metadata = _attention(_PreparedCacheAttention).concatenate(
        query=query,
        key=key,
        value=value,
        mask_info=mask,
        cache_view=view,
        cache_metadata=metadata,
        mode=common_types.MODE_DECODE if length == 1 else common_types.MODE_PREFILL,
        sliding_window=window,
        causal=causal,
    )
    expected = configured
    if window is not None:
        left, right = _window_bounds(window)
        expected = expected & (relative >= -left) & (relative <= right)
    np.testing.assert_array_equal(out_mask.attention_mask, expected[:, None])
    assert out_key is key and out_value is value and updated is view and out_metadata is metadata
    assert out_mask.q_positions is query_positions and out_mask.kv_positions is key_positions
    np.testing.assert_array_equal(out_mask.q_segment_ids, mask.q_segment_ids)
    np.testing.assert_array_equal(out_mask.kv_segment_ids, mask.kv_segment_ids)
    assert out_mask.sliding_window_baked_in


@pytest.mark.cpu_ok
@pytest.mark.parametrize("cache_kind", ["none", "unconfigured", "full", "chunk"])
@pytest.mark.parametrize("length", [1, 4])
@pytest.mark.parametrize("window", [0, 2, (0, 1), 5])
def test_explicit_window_intersects_caller_baked_mask_without_configured_sliding_cache(
    monkeypatch, cache_kind, length, window
):
    monkeypatch.setattr(_flexible, "jnp", np)
    query_positions = np.broadcast_to(np.arange(8, 8 + length), (4, length))
    key_positions = np.broadcast_to(np.arange(5, 8 + length), (4, 3 + length))
    relative = key_positions[:, None, :] - query_positions[:, :, None]
    caller_visible = (relative >= -3) & (relative <= 2)
    # A caller-owned exclusion must survive intersection, even for a wider override.
    caller_visible[:, :, 2] = False
    mask = MaskInfo(
        _attention_mask=caller_visible[:, None],
        _q_segment_ids=np.ones_like(query_positions),
        _kv_segment_ids=np.ones_like(key_positions),
        q_positions=query_positions,
        kv_positions=key_positions,
        sliding_window_baked_in=True,
    )
    key = np.arange(4 * (3 + length) * 2, dtype=np.float32).reshape(4, 3 + length, 1, 2)
    value = key + 10
    view = None
    metadata = None
    if cache_kind != "none":
        view = _view(key[:, -3:], value[:, -3:], np.full((4,), 8 + length), np.zeros((4,), np.int32))
        view.masking_details = (
            None
            if cache_kind == "unconfigured"
            else AttnMaskDetail(mask_type=AttnMaskType.FULL if cache_kind == "full" else AttnMaskType.CHUNK, size=4)
        )
        metadata = TransformerMetadata(indexes=view.indexes, starts=view.starts)
    cls = AttentionModule if view is None else _PreparedCacheAttention
    out_key, out_value, out_mask, _, updated, out_metadata = _attention(cls).concatenate(
        query=np.zeros((4, length, 1, 2), np.float32),
        key=key,
        value=value,
        mask_info=mask,
        cache_view=view,
        cache_metadata=metadata,
        mode=common_types.MODE_DECODE if length == 1 else common_types.MODE_PREFILL,
        sliding_window=window,
        causal=False,
    )
    left, right = _window_bounds(window)
    expected = caller_visible & (relative >= -left) & (relative <= right)
    np.testing.assert_array_equal(out_mask.attention_mask, expected[:, None])
    assert out_key is key and out_value is value and updated is view and out_metadata is metadata
    assert out_mask.q_positions is query_positions and out_mask.kv_positions is key_positions
    np.testing.assert_array_equal(out_mask.q_segment_ids, mask.q_segment_ids)
    np.testing.assert_array_equal(out_mask.kv_segment_ids, mask.kv_segment_ids)
    assert out_mask.sliding_window_baked_in


@pytest.mark.cpu_ok
@pytest.mark.parametrize("window", [None, 3])
def test_baked_configured_window_needs_no_position_reconstruction(monkeypatch, window):
    monkeypatch.setattr(_flexible, "jnp", np)
    key = np.arange(24, dtype=np.float32).reshape(4, 3, 1, 2)
    mask = MaskInfo(_attention_mask=np.ones((4, 1, 1, 3), dtype=np.bool_), sliding_window_baked_in=True)
    view = _view(key, key, np.full((4,), 20), np.zeros((4,), np.int32))
    k, v, actual, _, _, metadata = _attention(_PreparedCacheAttention).concatenate(
        query=key[:, :1],
        key=key,
        value=key,
        mask_info=mask,
        cache_view=view,
        sliding_window=window,
        mode=common_types.MODE_DECODE,
    )
    assert k is key and v is key and actual is mask
    np.testing.assert_array_equal(actual.attention_mask, np.ones((4, 1, 1, 3), dtype=np.bool_))
    np.testing.assert_array_equal(metadata.indexes, view.indexes)


def _performer(decode_backend=None):
    config = EasyDeLBaseConfig(
        attn_mechanism="vanilla",
        decode_attn_mechanism=decode_backend,
        attn_dtype=np.float32,
        attn_softmax_dtype=np.float32,
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
    )
    return FlexibleAttentionModule(config, softmax_scale=0.5)


class _WindowReportingAttention:
    metadata = SimpleNamespace(runtime_dtype=np.float32)

    def get_impl_name(self):
        return "vanilla"

    def __call__(self, *, value, sliding_window, cache_view, **kwargs):
        # Make the backend's effective window observable in the public output.
        output = np.full_like(value, -1 if sliding_window is None else sliding_window)
        return AttentionOutput(attention_outputs=output, cache_view=cache_view)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("baked", [False, True])
@pytest.mark.parametrize("hybrid", [False, True])
@pytest.mark.parametrize("mode", [common_types.MODE_PREFILL, common_types.MODE_DECODE])
def test_dense_baked_window_is_not_applied_again_by_prefill_or_decode_dispatch(monkeypatch, baked, hybrid, mode):
    from .test_cache_group_dispatch import _performer as host_performer

    performer = host_performer(monkeypatch, backend="vanilla", decode_backend="vanilla")
    object.__setattr__(performer, "impl", _WindowReportingAttention())
    object.__setattr__(performer, "impl_decode", _WindowReportingAttention())
    states = np.zeros((4, 1, 1, 2), np.float32)
    view = _view(states, states, np.full((4,), 9), np.zeros((4,), np.int32))
    if hybrid:
        view = ParallelHybridCacheView(transformer=view, recurrent=None)
    mask = MaskInfo(_attention_mask=np.ones((4, 1, 1, 1), dtype=bool), sliding_window_baked_in=baked)
    output = performer(states, states, states, mode=mode, cache_view=view, mask_info=mask, sliding_window=3)
    np.testing.assert_array_equal(output.attention_outputs, np.full_like(states, -1 if baked else 3))
    assert output.cache_view is view or (hybrid and output.cache_view.transformer is view.transformer)


def _step(query, key, value, view, performer, window, causal):
    k, v, mask, _, updated, metadata = _attention().concatenate(
        query=query,
        key=key,
        value=value,
        cache_view=view,
        sliding_window=window,
        causal=causal,
    )
    output = performer(
        query_states=query,
        key_states=k,
        value_states=v,
        mask_info=mask,
        # Like UnifiedAttention, retain the option selected before concatenate:
        # the dispatcher must honor the newly baked absolute-position mask.
        sliding_window=window,
        cache_view=updated,
        cache_metadata=metadata,
        causal=causal,
        mode=common_types.MODE_PREFILL if query.shape[1] > 1 else common_types.MODE_DECODE,
        precision=jax.lax.Precision.HIGHEST,
    ).attention_outputs
    return output, updated, mask, metadata


@pytest.mark.cpu_ok
@pytest.mark.parametrize("decode_backend", [None, "autoregressive_decodeattn"])
@pytest.mark.parametrize("length", [1, 4])
@pytest.mark.parametrize("window", [None, 3, 2, (1, 0), (0, 2), 0, 5, (4, 4)])
def test_real_cache_and_registered_attention_trace_aligned_coordinates_without_execution(length, window, decode_backend):
    performer = _performer(decode_backend)

    def forward(query, key, old_key, indexes):
        view = _view(old_key, old_key, indexes, jnp.zeros_like(indexes))
        output, updated, mask, metadata = _step(query, key, key, view, performer, window, False)
        return output, updated.indexes, mask.q_positions, mask.kv_positions, metadata.indexes

    traced = jax.make_jaxpr(forward)(
        jax.ShapeDtypeStruct((4, length, 1, 2), np.float32),
        jax.ShapeDtypeStruct((4, length, 1, 2), np.float32),
        jax.ShapeDtypeStruct((4, 3, 1, 2), np.float32),
        jax.ShapeDtypeStruct((4,), np.int32),
    )
    kv_length = 3 if length == 1 else 3 + length
    assert [aval.shape for aval in traced.out_avals] == [(4, length, 1, 2), (4,), (4, length), (4, kv_length), (4,)]


@pytest.mark.parametrize("decode_backend", [None, "autoregressive_decodeattn"])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("window", [None, 3, 2, (1, 0), (0, 2), 0, 5, (4, 4)])
def test_real_cached_attention_long_prefill_and_overflow_match_full_history(causal, window, decode_backend):
    """Accelerator-only independent parity across prefill, decode and block appends."""
    rng = np.random.default_rng(29)
    query = rng.normal(size=(4, 13, 1, 2)).astype(np.float32)
    key = rng.normal(size=query.shape).astype(np.float32)
    value = rng.normal(size=query.shape).astype(np.float32)
    performer = _performer(decode_backend)
    view = _view(
        jnp.zeros((4, 3, 1, 2), np.float32),
        jnp.zeros((4, 3, 1, 2), np.float32),
        jnp.zeros((4,), np.int32),
        jnp.zeros((4,), np.int32),
    )
    start = 0
    for length in (5, 1, 2, 1, 4):
        end = start + length
        actual, view, mask, metadata = _step(
            jnp.asarray(query[:, start:end]),
            jnp.asarray(key[:, start:end]),
            jnp.asarray(value[:, start:end]),
            view,
            performer,
            window,
            causal,
        )
        qpos = np.arange(start, end)
        kpos = np.arange(end)
        relative = kpos[None, :] - qpos[:, None]
        visible = (relative >= -2) & (relative <= (0 if causal else 2))
        if window is not None:
            left, right = _window_bounds(window)
            visible &= (relative >= -left) & (relative <= right)
        logits = 0.5 * np.einsum("bqhd,bkhd->bhqk", query[:, start:end], key[:, :end])
        logits = np.where(visible[None, None], logits, -np.inf)
        weights = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        weights /= np.sum(weights, axis=-1, keepdims=True)
        expected = np.einsum("bhqk,bkhd->bqhd", weights, value[:, :end])
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(view.indexes, np.full((4,), end))
        np.testing.assert_array_equal(metadata.indexes, view.indexes)
        np.testing.assert_array_equal(mask.q_positions, np.broadcast_to(qpos, (4, length)))
        np.testing.assert_array_equal(view.key, key[:, end - 3 : end])
        np.testing.assert_array_equal(view.value, value[:, end - 3 : end])
        start = end


@pytest.mark.parametrize("decode_backend", [None, "autoregressive_decodeattn"])
def test_real_rolled_cache_decode_keeps_absolute_start_bounds(decode_backend):
    """Absolute starts must not be compared to columns of an already-rolled cache."""
    performer = _performer(decode_backend)
    query = jnp.zeros((4, 1, 1, 2), np.float32)
    old_key = jnp.zeros((4, 3, 1, 2), np.float32)
    old_value = jnp.asarray(np.broadcast_to(np.arange(5, 8)[None, :, None, None], (4, 3, 1, 2)), np.float32)
    view = _view(old_key, old_value, jnp.full((4,), 8, np.int32), jnp.asarray([0, 1, 4, 6], np.int32))
    actual, updated, mask, metadata = _step(query, query, jnp.full_like(query, 8), view, performer, 3, True)
    # Every row can see absolute keys [6, 7, 8], regardless of the original
    # left-padding start. Uniform query logits give their independent mean.
    np.testing.assert_allclose(actual, np.full((4, 1, 1, 2), 7, np.float32), rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(mask.kv_positions, np.broadcast_to([6, 7, 8], (4, 3)))
    np.testing.assert_array_equal(mask.attention_mask, np.ones((4, 1, 1, 3), bool))
    np.testing.assert_array_equal(updated.starts, [0, 1, 4, 6])
    np.testing.assert_array_equal(metadata.indexes, np.full((4,), 9))
