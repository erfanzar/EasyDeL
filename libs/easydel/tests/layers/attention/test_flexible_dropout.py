"""Attention dropout changes public outputs and respects deterministic / override controls."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.infra.base_config import EasyDeLBaseConfig
from easydel.layers.attention import FlexibleAttentionModule, UnifiedAttention
from jax import numpy as jnp
from spectrax import common_types


def _module(dropout_prob=0.5, backend="vanilla", rngs=None):
    config = EasyDeLBaseConfig(
        attn_mechanism=backend,
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
    )
    return FlexibleAttentionModule(config, softmax_scale=1.0, dropout_prob=dropout_prob, requires_cache=False, rngs=rngs)


def _call(module, **kwargs):
    # Uniform logits, distinct values: the exact no-dropout answer is their mean.
    batch = jax.device_count()
    q = jnp.zeros((batch, 4, 1, 2), jnp.float32)
    v = jnp.broadcast_to(jnp.arange(1, 9, dtype=jnp.float32).reshape(1, 4, 1, 2), q.shape)
    return module(q, q, v, mode=None, causal=False, output_attentions=True, **kwargs)


@pytest.mark.parametrize("backend", ["vanilla", "sdpa", "cudnn", "cuda_flash_attn2", "flash_attn2", "blocksparse"])
def test_dropout_is_reproducible_and_changes_attention_weights(backend):
    module = _module(backend=backend)
    key = jax.random.key(42)
    dropped = _call(module, deterministic=False, dropout_rng=key)
    repeated = _call(module, deterministic=False, dropout_rng=key)
    baseline = _call(module)
    np.testing.assert_allclose(baseline.attention_outputs, np.broadcast_to([4.0, 5.0], baseline.attention_outputs.shape))
    np.testing.assert_array_equal(dropped.attention_outputs, repeated.attention_outputs)
    weights = np.asarray(dropped.attention_weights)
    assert (weights == 0.0).any()
    assert (weights == 0.5).any()  # uniform .25 weights, scaled by 1 / keep_probability
    assert not np.array_equal(dropped.attention_outputs, baseline.attention_outputs)


def test_zero_override_and_deterministic_disable_dropout():
    module = _module()
    baseline = _call(module)
    zero = _call(module, deterministic=False, dropout_prob=0.0)
    deterministic = _call(module, deterministic=True, dropout_rng=jax.random.key(0))
    np.testing.assert_array_equal(zero.attention_outputs, baseline.attention_outputs)
    np.testing.assert_array_equal(deterministic.attention_weights, baseline.attention_weights)


def test_positive_override_enables_dropout_on_zero_default():
    module = _module(0.0)
    result = _call(module, deterministic=False, dropout_prob=0.5, dropout_rng=jax.random.key(42))
    assert (np.asarray(result.attention_weights) == 0.0).any()


def test_active_dropout_without_rng_raises():
    with pytest.raises(ValueError, match="dropout_rng"):
        _call(_module(), deterministic=False)


@pytest.mark.parametrize("backend", ["vanilla", "sdpa", "flash_attn2"])
def test_varlen_dropout_masks_padding_and_keeps_live_weights(backend):
    module = _module(backend=backend)
    batch = jax.device_count()
    q = jnp.zeros((batch, 4, 1, 2), jnp.float32)
    value = jnp.broadcast_to(jnp.arange(1, 9, dtype=jnp.float32).reshape(1, 4, 1, 2), q.shape)
    result = module(
        q,
        q,
        value,
        mode=common_types.MODE_TRAIN,
        deterministic=False,
        dropout_rng=jax.random.key(42),
        cum_seqlens_q=jnp.arange(batch + 1) * 2,
        cum_seqlens_k=jnp.arange(batch + 1) * 2,
        causal=False,
        output_attentions=True,
    )
    weights = np.asarray(result.attention_weights)
    np.testing.assert_array_equal(weights[..., 2:, :], 0)
    np.testing.assert_array_equal(weights[..., :2, 2:], 0)
    assert set(np.unique(weights[..., :2, :2])).issubset({0.0, 1.0})
    assert (weights[..., :2, :2] == 1.0).any()
    expected = np.einsum("bhqk,bkhd->bqhd", weights, np.asarray(value))
    np.testing.assert_allclose(result.attention_outputs, expected)


def test_varlen_dropout_retains_lazy_bias_with_real_registered_backend():
    module = _module(backend="sdpa")
    batch = jax.device_count()
    q = jnp.zeros((batch, 4, 1, 2), jnp.float32)
    value = jnp.broadcast_to(jnp.arange(1, 9, dtype=jnp.float32).reshape(1, 4, 1, 2), q.shape)
    bias = jnp.broadcast_to(jnp.arange(4, dtype=jnp.float32), (batch, 1, 4, 4))
    arguments = dict(
        mode=common_types.MODE_TRAIN,
        dropout_rng=jax.random.key(42),
        deterministic=False,
        causal=False,
        output_attentions=True,
        cum_seqlens_q=jnp.arange(batch + 1) * 2,
        cum_seqlens_k=jnp.arange(batch + 1) * 2,
    )
    lazy = module(q, q, value, init_bias=lambda: bias, **arguments)
    explicit = module(q, q, value, bias=bias, **arguments)
    np.testing.assert_array_equal(lazy.attention_weights, explicit.attention_weights)
    np.testing.assert_array_equal(lazy.attention_outputs, explicit.attention_outputs)
    weights = np.asarray(lazy.attention_weights)
    # Independent reference: live keys have probabilities softmax([0, 1]); inverted
    # dropout doubles surviving weights. This must not fall back to a uniform row.
    reference = np.exp(np.array([0, 1], dtype=np.float32))
    reference = 2 * reference / reference.sum()
    live = weights[..., :2, :2]
    for key_index in range(2):
        np.testing.assert_allclose(
            live[..., key_index], np.where(live[..., key_index] > 0, reference[key_index], 0), rtol=1e-5
        )
    np.testing.assert_array_equal(weights[..., 2:, :], 0)
    np.testing.assert_array_equal(weights[..., :2, 2:], 0)


@pytest.mark.parametrize("backend", ["vanilla", "sdpa", "flash_attn2"])
def test_unnormalized_dropout_retains_numerator_with_real_registered_backend(backend):
    module = _module(backend=backend)
    key = jax.random.key(42)
    result = _call(module, deterministic=False, dropout_rng=key, normalize_output=False)
    repeated = _call(module, deterministic=False, dropout_rng=key, normalize_output=False)
    # All logits are zero: exp(logits - max(logits)) is one, NOT 1 / kv_len.
    keep = np.asarray(jax.random.bernoulli(key, 0.5, result.attention_weights.shape))
    expected_weights = keep.astype(np.float32) * 2
    values = np.broadcast_to(np.arange(1, 9, dtype=np.float32).reshape(1, 4, 1, 2), (jax.device_count(), 4, 1, 2))
    expected = np.einsum("bhqk,bkhd->bqhd", expected_weights, values)
    np.testing.assert_array_equal(result.attention_weights, expected_weights)
    np.testing.assert_allclose(result.attention_outputs, expected)
    np.testing.assert_array_equal(result.attention_outputs, repeated.attention_outputs)


class _ProjectedAttention(UnifiedAttention):
    """Fixed projections isolate the production standard-forward dropout path."""

    def _project_qkv(self, hidden_states):
        return jnp.zeros_like(hidden_states), jnp.zeros_like(hidden_states), hidden_states

    def _apply_rotary(self, query_states, key_states, position_ids, frequencies):
        return query_states, key_states

    def apply_qkv_shardings(self, query_states, key_states, value_states):
        return query_states, key_states, value_states

    def concatenate(self, *, key, value, mask_info, cache_view, cache_metadata, **kwargs):
        return key, value, mask_info, None, cache_view, cache_metadata

    def shard_attention_prod(self, inputs, **kwargs):
        return inputs


def test_unified_training_uses_configured_dropout_and_constructor_rngs_without_extra_arguments():
    config = EasyDeLBaseConfig(
        attn_mechanism="sdpa",
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
        attention_dropout=0.5,
    )
    attention = _ProjectedAttention.__new__(_ProjectedAttention)
    for name, value in {
        "config": config,
        "num_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "attention_type": "standard",
        "use_qk_norm": False,
        "causal": False,
        "sliding_window": None,
        "precision": None,
        "o_proj": lambda inputs: inputs,
    }.items():
        object.__setattr__(attention, name, value)
    attention.attention_performer = attention._create_attention_performer(config, spx.Rngs(42))
    attention.train()
    hidden = jnp.broadcast_to(jnp.arange(1, 17, dtype=jnp.float32).reshape(1, 8, 2), (jax.device_count(), 8, 2))
    positions = jnp.broadcast_to(jnp.arange(8), hidden.shape[:2])
    trained = attention(hidden, mask_info=None, position_ids=positions, mode=common_types.MODE_TRAIN)
    repeated = attention(hidden, mask_info=None, position_ids=positions, mode=common_types.MODE_TRAIN)
    attention.eval()
    evaluated = attention(hidden, mask_info=None, position_ids=positions, mode=common_types.MODE_TRAIN)
    expected = np.broadcast_to([8.0, 9.0], hidden.shape)
    np.testing.assert_allclose(evaluated.attention_output, expected)
    assert not np.array_equal(trained.attention_output, expected)
    assert not np.array_equal(trained.attention_output, repeated.attention_output)
