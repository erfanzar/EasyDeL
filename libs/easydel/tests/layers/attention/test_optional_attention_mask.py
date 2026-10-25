"""Optional attention masks and bidirectional cache append contracts."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
from easydel.caching import TransformerCacheConfig, TransformerCacheView
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.layers.attention import AttentionModule, FlexibleAttentionModule, UnifiedAttention, _flexible
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from spectrax import common_types


def _attention():
    attention = AttentionModule.__new__(AttentionModule)
    attention.dtype = np.float32
    attention.config = SimpleNamespace(runtime_sharding_resolver=None, kv_cache_quantization_config=None)
    return attention


@pytest.mark.cpu_ok
@pytest.mark.parametrize("query_length,key_length", [(1, 1), (3, 3), (2, 5)])
@pytest.mark.parametrize("causal", [False, True])
def test_absent_dense_mask_keeps_kv_and_builds_neutral_lazy_bias(monkeypatch, query_length, key_length, causal):
    # NumPy substitutes allocation only: no JAX array computations run on CPU.
    monkeypatch.setattr(_flexible, "jnp", np)
    query = np.zeros((4, query_length, 2, 3), np.float32)
    key = np.arange(4 * key_length * 3, dtype=np.float32).reshape(4, key_length, 1, 3)
    value = key + 1
    key_out, value_out, mask, bias, view, metadata = _attention().concatenate(
        query=query, key=key, value=value, causal=causal
    )
    assert key_out is key and value_out is value
    assert mask is None and view is None and metadata is None
    np.testing.assert_array_equal(bias(), np.zeros((4, 1, query_length, key_length), np.float32))
    assert bias().dtype == np.float32


@pytest.mark.cpu_ok
def test_explicit_mask_and_lazy_bias_are_preserved(monkeypatch):
    monkeypatch.setattr(_flexible, "jnp", np)
    visible = np.array([[[[True, False], [True, True]]]])
    expected = np.where(visible, 0.0, np.finfo(np.float32).min).astype(np.float32)
    mask = SimpleNamespace(create_bias=lambda dtype: expected.astype(dtype))
    states = np.zeros((1, 2, 1, 3), np.float32)
    _, _, actual_mask, bias, _, _ = _attention().concatenate(query=states, key=states, value=states, mask_info=mask)
    assert actual_mask is mask
    np.testing.assert_array_equal(bias(), expected)


def _append(attention, query, key, value, old_key, old_value, indexes, causal):
    config = TransformerCacheConfig.create(4, 6, 1, num_heads=1, head_dim=2)
    view = TransformerCacheView(
        key=old_key,
        value=old_value,
        indexes=indexes,
        starts=jnp.zeros_like(indexes),
        metadata=config,
        maximum_sequence_length=6,
        layer_index=0,
    )
    k, v, mask, _, updated, _ = attention.concatenate(
        query=query,
        key=key,
        value=value,
        cache_view=view,
        mode=common_types.MODE_DECODE,
        causal=causal,
    )
    return k, v, mask.attention_mask, updated.indexes


@pytest.mark.cpu_ok
@pytest.mark.parametrize("causal", [False, True])
def test_absent_cached_mask_traces_real_cache_append_without_array_execution(causal):
    attention = _attention()
    current = jax.ShapeDtypeStruct((4, 2, 1, 2), np.float32)
    history = jax.ShapeDtypeStruct((4, 6, 1, 2), np.float32)
    indexes = jax.ShapeDtypeStruct((4,), np.int32)
    traced = jax.make_jaxpr(lambda *args: _append(attention, *args, causal))(
        current, current, current, history, history, indexes
    )
    assert [(aval.shape, aval.dtype) for aval in traced.out_avals] == [
        ((4, 6, 1, 2), np.dtype(np.float32)),
        ((4, 6, 1, 2), np.dtype(np.float32)),
        ((4, 1, 2, 6), np.dtype(np.bool_)),
        ((4,), np.dtype(np.int32)),
    ]


@pytest.mark.parametrize("causal", [False, True])
def test_absent_cached_mask_respects_bidirectional_or_causal_append(causal):
    """Accelerator-only: current-block future visibility must follow causal."""
    query = jnp.zeros((4, 2, 1, 2), np.float32)
    key = jnp.arange(16, dtype=np.float32).reshape(4, 2, 1, 2)
    value = key + 20
    old_key = jnp.zeros((4, 6, 1, 2), np.float32)
    old_value = jnp.zeros_like(old_key)
    k, v, mask, indexes = _append(_attention(), query, key, value, old_key, old_value, jnp.full((4,), 2), causal)
    np.testing.assert_array_equal(k[:, 2:4], key)
    np.testing.assert_array_equal(v[:, 2:4], value)
    expected = np.array([[True, True, True, not causal, False, False], [True, True, True, True, False, False]])
    np.testing.assert_array_equal(mask, np.broadcast_to(expected, (4, 1, 2, 6)))
    np.testing.assert_array_equal(indexes, [4, 4, 4, 4])


@pytest.mark.cpu_ok
def test_missing_mask_with_sliding_window_retains_each_querys_local_keys(monkeypatch):
    from .test_sliding_prefill import _numpy_vmap

    monkeypatch.setattr(_flexible, "jnp", np)
    monkeypatch.setattr(_flexible.jax, "vmap", _numpy_vmap)
    monkeypatch.setattr(
        _flexible.jax.lax,
        "broadcasted_iota",
        lambda dtype, shape, axis: np.broadcast_to(
            np.arange(shape[axis], dtype=dtype).reshape(tuple(shape[i] if i == axis else 1 for i in range(len(shape)))),
            shape,
        ),
    )
    query = np.zeros((4, 5, 1, 2), np.float32)
    _, _, mask, _, _, _ = _attention().concatenate(query=query, key=query, value=query, sliding_window=2)
    assert isinstance(mask, MaskInfo)
    expected = np.abs(np.arange(5)[:, None] - np.arange(5)[None, :]) <= 1
    np.testing.assert_array_equal(mask.attention_mask, np.broadcast_to(expected, (4, 1, 5, 5)))
    assert mask.sliding_window_baked_in


class _ProjectedAttention(UnifiedAttention):
    """Fixed projections; production concatenate and registered backend stay intact."""

    def _project_qkv(self, hidden_states):
        return hidden_states, hidden_states, hidden_states

    def _apply_rotary(self, query, key, position_ids, frequencies):
        return query, key

    def apply_qkv_shardings(self, query, key, value):
        return query, key, value

    def shard_attention_prod(self, inputs, **kwargs):
        return inputs


@pytest.mark.cpu_ok
@pytest.mark.parametrize("attention_type", ["standard", "mla"])
@pytest.mark.parametrize("causal", [False, True])
def test_unified_absent_mask_traces_registered_dense_backend_without_array_execution(attention_type, causal):
    config = EasyDeLBaseConfig(
        attn_mechanism="vanilla",
        attn_dtype=np.float32,
        attn_softmax_dtype=np.float32,
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
    )
    attention = _ProjectedAttention.__new__(_ProjectedAttention)
    values = {
        "config": config,
        "dtype": np.float32,
        "num_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 2 if attention_type == "mla" else 4,
        "q_head_dim": 4,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "kv_lora_rank": 2,
        "v_head_dim": 2,
        "use_mla_lora": False,
        "use_qk_norm": False,
        "causal": causal,
        "sliding_window": None,
        "attention_type": attention_type,
        "precision": jax.lax.Precision.HIGHEST,
        "q_proj": lambda inputs: inputs,
        "kv_a_proj_with_mqa": lambda inputs: inputs,
        "kv_a_layernorm": lambda inputs: inputs,
        "kv_b_proj": lambda inputs: jnp.tile(inputs, (1, 1, 2)),
        "o_proj": lambda inputs: inputs,
        "attention_performer": FlexibleAttentionModule(config, softmax_scale=0.5),
    }
    for name, value in values.items():
        object.__setattr__(attention, name, value)

    traced = jax.make_jaxpr(
        lambda hidden, positions, freqs: (
            attention(
                hidden,
                mask_info=None,
                position_ids=positions,
                frequencies=freqs,
                mode=common_types.MODE_TRAIN,
            ).attention_output
        )
    )(
        jax.ShapeDtypeStruct((4, 3, 4), np.float32),
        jax.ShapeDtypeStruct((4, 3), np.int32),
        jax.ShapeDtypeStruct((3, 2), np.float32),
    )
    width = 2 if attention_type == "mla" else 4
    assert [(aval.shape, aval.dtype) for aval in traced.out_avals] == [((4, 3, width), np.dtype(np.float32))]
