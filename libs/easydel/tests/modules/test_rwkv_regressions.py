"""RWKV regressions (accelerator only)."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.rwkv.modeling_rwkv import RwkvForCausalLM, RwkvModel, init_state, rwkv_linear_attention
from easydel.modules.rwkv.rwkv_configuration import RwkvConfig
from jax import numpy as jnp


def _config(**kwargs):
    return RwkvConfig(
        vocab_size=64,
        context_length=32,
        hidden_size=16,
        num_hidden_layers=4,
        rescale_every=2,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        **kwargs,
    )


def test_reference_linear_attention_is_consistent_across_chunks():
    key = jnp.linspace(-1.0, 1.0, 2 * 6 * 4).reshape(2, 6, 4)
    value = jnp.cos(key)
    time_decay = jnp.linspace(-0.5, 0.5, 4)
    time_first = jnp.linspace(0.1, 0.4, 4)
    full, _ = rwkv_linear_attention(time_decay, time_first, key, value)
    keys, values = np.asarray(key), np.asarray(value)
    decay, first_weight = -np.exp(np.asarray(time_decay)), np.asarray(time_first)
    expected = []
    for position in range(keys.shape[1]):
        scores = np.concatenate(
            [
                keys[:, :position] + decay * (position - 1 - np.arange(position))[None, :, None],
                keys[:, position : position + 1] + first_weight,
            ],
            axis=1,
        )
        weights = np.exp(scores - scores.max(axis=1, keepdims=True))
        expected.append((weights * values[:, : position + 1]).sum(axis=1) / weights.sum(axis=1))
    np.testing.assert_allclose(np.asarray(full), np.stack(expected, axis=1), rtol=1e-5, atol=1e-5)
    first, state = rwkv_linear_attention(time_decay, time_first, key[:, :3], value[:, :3], return_state=True)
    second, _ = rwkv_linear_attention(time_decay, time_first, key[:, 3:], value[:, 3:], state=state)
    np.testing.assert_allclose(np.asarray(full), np.asarray(jnp.concatenate([first, second], 1)), rtol=1e-5, atol=1e-5)


def test_rescale_layers_preserves_outputs_and_is_reversible():
    model = RwkvModel(_config(), rngs=spx.Rngs(0))
    ids = jnp.array([[1, 2, 3, 4, 5]], jnp.int32)
    ref = model(input_ids=ids, use_cache=False, output_hidden_states=True)
    weight_before = np.asarray(model.blocks[3].attention.output.weight.value).copy()

    model.eval()
    assert model.layers_are_rescaled
    scaled = model(input_ids=ids, use_cache=False, output_hidden_states=True)
    np.testing.assert_array_equal(np.asarray(model.blocks[3].attention.output.weight.value), weight_before)
    np.testing.assert_allclose(np.asarray(scaled.hidden_states[1]), np.asarray(ref.hidden_states[1]) / 2, atol=1e-6)
    np.testing.assert_allclose(np.asarray(scaled.last_hidden_state), np.asarray(ref.last_hidden_state), atol=2e-3)

    model.train()
    assert not model.layers_are_rescaled
    restored = model(input_ids=ids, use_cache=False).last_hidden_state
    np.testing.assert_array_equal(np.asarray(restored), np.asarray(ref.last_hidden_state))


@pytest.mark.parametrize("scan_layers", [False, True])
@pytest.mark.parametrize("attention_hidden_size", [8, 24])
def test_recurrent_cache_continuation_supports_scans_and_unequal_widths(scan_layers, attention_hidden_size):
    model = RwkvModel(
        _config(scan_layers=scan_layers, attention_hidden_size=attention_hidden_size),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    model.eval()
    ids = jnp.array([[2, 3, 4, 5]], jnp.int32)
    full = model(input_ids=ids, use_cache=False)
    assert full.state is None
    with jax.checking_leaks():
        prompt = model(input_ids=ids[:, :2], use_cache=True)
    assert [value.shape for value in prompt.state] == [(1, 16, 4)] * 2 + [(1, attention_hidden_size, 4)] * 3
    original = [np.asarray(value).copy() for value in prompt.state]
    with jax.checking_leaks():
        continuation = model(input_ids=ids[:, 2:], state=prompt.state, use_cache=True)
    np.testing.assert_allclose(
        np.asarray(continuation.last_hidden_state), np.asarray(full.last_hidden_state[:, 2:]), rtol=1e-5, atol=1e-5
    )
    for old, value in zip(original, prompt.state, strict=True):
        np.testing.assert_array_equal(np.asarray(value), old)
    final = model(input_ids=ids, use_cache=True)
    for actual, expected in zip(continuation.state, final.state, strict=True):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-5, atol=1e-5)


def test_generation_and_layer_initial_states_use_attention_width():
    model = RwkvForCausalLM(
        _config(attention_hidden_size=24), dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0)
    )
    ids = jnp.array([[2, 3]], jnp.int32)
    prepared = model.prepare_inputs_for_generation(ids, max_length=8, pad_token_id=0)
    assert [value.shape for value in prepared["state"]] == [(1, 16, 4)] * 2 + [(1, 24, 4)] * 3
    time_state, channel_state = init_state(16, batch_size=1, attention_hidden_size=24)
    assert [value.shape for value in time_state] == [(1, 16)] + [(1, 24)] * 3
    assert channel_state.shape == (1, 16)
    model.eval()
    result = model(input_ids=ids, **prepared)
    assert result.logits.shape == (1, 2, 64)
    assert np.isfinite(np.asarray(result.logits)).all()
