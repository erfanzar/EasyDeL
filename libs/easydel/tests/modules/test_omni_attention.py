"""Observable cache, batched masking, and checkpoint-conversion regressions."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavAttention,
    Qwen3OmniMoeCode2WavTransformerLayer,
    Qwen3OmniMoeThinkerTextModel,
)
from easydel.modules.qwen3_omni_moe.qwen3_omni_moe_configuration import (
    Qwen3OmniMoeCode2WavConfig,
    Qwen3OmniMoeTextConfig,
)
from jax import numpy as jnp
from spectrax import common_types


@pytest.fixture(autouse=True)
def _highest_matmul_precision():
    # Reference parity must not depend on TPU's default bf16 dot lowering.
    with jax.default_matmul_precision("highest"):
        yield


def _code2wav_config():
    return Qwen3OmniMoeCode2WavConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        sliding_window=2,
        attn_mechanism="vanilla",
    )


def test_code2wav_padding_mask_matches_independent_batch_members():
    model = Qwen3OmniMoeCode2WavAttention(
        _code2wav_config(), layer_idx=0, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0)
    )
    states = jnp.asarray(np.random.default_rng(0).normal(size=(2, 5, 16)), dtype=jnp.float32)
    mask = jnp.array([[True, True, True, False, False], [True, True, True, True, True]])
    batched = model(states, attention_mask=mask)
    for row in range(2):
        separate = model(states[row : row + 1], attention_mask=mask[row : row + 1])
        np.testing.assert_allclose(np.asarray(batched[row : row + 1]), np.asarray(separate), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("sliding_window", [None, 2])
def test_code2wav_mask_excludes_earlier_keys_from_later_valid_queries(sliding_window):
    config = _code2wav_config()
    config.sliding_window = sliding_window
    model = Qwen3OmniMoeCode2WavAttention(
        config, layer_idx=0, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0)
    )
    states = jnp.asarray(np.random.default_rng(1).normal(size=(2, 5, 16)), dtype=jnp.float32)
    mask = jnp.array([[True, False, True, True, True], [True, True, True, True, True]])
    changed = states.at[0, 1].add(10)
    baseline = model(states, attention_mask=mask)
    masked = model(changed, attention_mask=mask)
    # Key 1 is causal and (when configured) inside query 2's sliding window.
    # Only padding exclusion can prevent it from influencing that valid query.
    np.testing.assert_allclose(np.asarray(baseline[0, 2]), np.asarray(masked[0, 2]), rtol=1e-5, atol=1e-5)
    visible = mask.at[0, 1].set(True)
    assert not np.allclose(
        np.asarray(model(states, attention_mask=visible)[0, 2]),
        np.asarray(model(changed, attention_mask=visible)[0, 2]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_code2wav_sliding_window_excludes_older_causal_keys():
    model = Qwen3OmniMoeCode2WavAttention(
        _code2wav_config(), layer_idx=0, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0)
    )
    states = jnp.asarray(np.random.default_rng(2).normal(size=(2, 5, 16)), dtype=jnp.float32)
    baseline = model(states)
    older_key_changed = model(states.at[:, 0].add(10))
    np.testing.assert_allclose(np.asarray(baseline[:, 4]), np.asarray(older_key_changed[:, 4]), rtol=1e-5, atol=1e-5)
    # A nearby, unmasked key must still affect the same query.
    nearby_key_changed = model(states.at[:, 3].add(10))
    assert not np.allclose(np.asarray(baseline[:, 4]), np.asarray(nearby_key_changed[:, 4]), rtol=1e-5, atol=1e-5)


def test_code2wav_split_qkv_checkpoint_rule_preserves_each_projection():
    torch = pytest.importorskip("torch")
    layer = Qwen3OmniMoeCode2WavTransformerLayer(
        _code2wav_config(), layer_idx=0, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0)
    )
    rule = layer.self_attn.reform_param["qkv_proj.weight$"]
    weights = tuple(torch.arange(256, dtype=torch.float32).reshape(16, 16) + offset for offset in (0, 1000, 2000))
    fused = rule["fuser"](torch, *weights)
    expected = torch.cat(weights, dim=0)
    torch.testing.assert_close(fused, expected)
    recovered = rule["inverse_fuser"](torch, fused)
    for actual, source in zip(recovered, weights, strict=True):
        torch.testing.assert_close(actual, source)


def test_thinker_cache_advances_and_decode_matches_full_sequence():
    config = Qwen3OmniMoeTextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        mlp_only_layers=[0, 1],
        max_position_embeddings=16,
        attn_dtype=jnp.float32,
        kvdtype=jnp.float32,
        attn_mechanism="vanilla",
        scan_layers=False,
    )
    model = Qwen3OmniMoeThinkerTextModel(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision="highest", rngs=spx.Rngs(0)
    )
    ids = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    full = model(input_ids=ids, output_attentions=True)
    assert len(full.attentions) == 2
    cache = model.init_cache(batch_size=1, max_length=4)
    prefill = model(input_ids=ids[:, :3], past_key_values=cache, mode=common_types.MODE_PREFILL)
    for view in prefill.past_key_values.views:
        np.testing.assert_array_equal(np.asarray(view.indexes), np.array([3]))
        assert np.any(np.asarray(view.key) != 0)
    decoded = model(
        input_ids=ids[:, 3:],
        position_ids=jnp.array([[3]]),
        past_key_values=prefill.past_key_values,
        mode=common_types.MODE_DECODE,
    )
    for view in decoded.past_key_values.views:
        np.testing.assert_array_equal(np.asarray(view.indexes), np.array([4]))
    np.testing.assert_allclose(
        np.asarray(decoded.last_hidden_state[:, -1]),
        np.asarray(full.last_hidden_state[:, -1]),
        rtol=2e-4,
        atol=2e-4,
    )
