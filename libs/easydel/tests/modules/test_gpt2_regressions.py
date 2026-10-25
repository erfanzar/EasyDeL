# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPT-2: Conv1D bias/sharding, cross-attention construction, encoder-decoder forward, attention scaling flags.

These execute JAX array computation and must run on an accelerator.
"""

import numpy as np
import pytest
import spectrax as spx
from easydel.modules.gpt2.gpt2_configuration import GPT2Config
from easydel.modules.gpt2.modeling_gpt2 import Conv1D, GPT2Attention, GPT2Block, GPT2Model
from jax import lax
from jax import numpy as jnp
from spectrax.common_types import MODE_TRAIN


def _config(**overrides):
    kwargs = {
        "vocab_size": 64,
        "n_positions": 32,
        "n_embd": 32,
        "n_layer": 2,
        "n_head": 4,
        "embd_pdrop": 0.0,
        "resid_pdrop": 0.0,
        "attn_pdrop": 0.0,
        "attn_mechanism": "vanilla",
        "attn_dtype": jnp.float32,
    }
    kwargs.update(overrides)
    config = GPT2Config(**kwargs)
    config.sharding_axis_dims = (1, 1, 1, 1, 1, 1)
    config.attach_custom_arguments()
    return config


def test_conv1d_without_bias_runs():
    layer = Conv1D(
        8,
        16,
        use_bias=False,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )
    inputs = jnp.ones((2, 3, 8), dtype=jnp.float32)
    out = layer(inputs)
    assert out.shape == (2, 3, 16)
    np.testing.assert_allclose(np.asarray(out), np.asarray(inputs) @ np.asarray(layer.weight.value).T, rtol=2e-5)


def test_conv1d_sharding_axis_is_applied_to_parameters():
    kwargs = {"dtype": jnp.float32, "param_dtype": jnp.float32, "rngs": spx.Rngs(0)}
    plain = Conv1D(8, 16, **kwargs)
    column = Conv1D(8, 16, sharding_axis="column", **kwargs)
    row = Conv1D(8, 16, sharding_axis="row", **kwargs)
    assert plain.weight.sharding is None
    assert column.weight.sharding is not None
    assert row.weight.sharding is not None
    assert column.weight.sharding != row.weight.sharding


def test_cross_attention_block_is_non_causal_and_keeps_param_dtype():
    config = _config()
    config.add_cross_attention = True
    with config.mesh:
        block = GPT2Block(
            config,
            layer_idx=0,
            dtype=jnp.bfloat16,
            param_dtype=jnp.float32,
            precision=lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
    assert block.crossattention.causal is False
    assert block.attn.causal is True
    assert block.crossattention.c_attn.weight.value.dtype == jnp.float32
    assert block.crossattention.c_proj.weight.value.dtype == jnp.float32
    assert block.crossattention.c_attn.precision == lax.Precision.HIGHEST
    assert block.crossattention.c_proj.precision == lax.Precision.HIGHEST


@pytest.mark.parametrize("output_attentions", [False, True])
def test_encoder_decoder_forward_without_output_attentions(output_attentions):
    config = _config()
    config.add_cross_attention = True
    with config.mesh:
        model = GPT2Model(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        input_ids = jnp.arange(8, dtype=jnp.int32).reshape(1, 8) % config.vocab_size
        encoder_states = jnp.ones((1, 6, config.hidden_size), dtype=jnp.float32)
        out = model(input_ids=input_ids, encoder_hidden_states=encoder_states, output_attentions=output_attentions)
    assert out.last_hidden_state.shape == (1, 8, config.hidden_size)
    if output_attentions:
        assert len(out.cross_attentions) == config.num_hidden_layers
        for weights in out.cross_attentions:
            # Identical encoder keys: every decoder query must see all six encoder tokens.
            np.testing.assert_allclose(np.asarray(weights), 1.0 / 6.0, rtol=1e-5)
    else:
        assert out.cross_attentions is None


@pytest.mark.parametrize("scale_weights", [False, True])
@pytest.mark.parametrize("inverse_layer", [False, True])
def test_attention_scaling_flags_match_reference(scale_weights, inverse_layer):
    config = _config(
        n_embd=8,
        n_head=2,
        scale_attn_weights=scale_weights,
        scale_attn_by_inverse_layer_idx=inverse_layer,
    )
    with config.mesh:
        layer = GPT2Attention(
            config,
            layer_idx=2,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=lax.Precision.HIGHEST,
            causal=False,
            rngs=spx.Rngs(0),
        )
        hidden = np.arange(24, dtype=np.float32).reshape(1, 3, 8) / 4.0
        out = layer(jnp.asarray(hidden), None, jnp.arange(3)[None, :], MODE_TRAIN, output_attentions=True)
    qkv = hidden @ np.asarray(layer.c_attn.weight.value).T + np.asarray(layer.c_attn.bias.value)
    query, key, _ = (x.reshape(1, 3, 2, 4) for x in np.split(qkv, 3, axis=-1))
    scale = (4**-0.5 if scale_weights else 1.0) / (3 if inverse_layer else 1)
    scores = np.einsum("bqhd,bkhd->bhqk", query, key) * scale
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    np.testing.assert_allclose(np.asarray(out.attention_weight), probabilities, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("mechanism", ["vanilla", "sdpa", "flash_attn2", "ring"])
def test_reorder_and_upcast_avoids_float16_score_overflow(mechanism):
    config = _config(
        n_embd=8,
        n_head=2,
        attn_mechanism=mechanism,
        attn_dtype=jnp.float16,
        reorder_and_upcast_attn=True,
    )
    with config.mesh:
        layer = GPT2Attention(
            config,
            layer_idx=0,
            dtype=jnp.float16,
            param_dtype=jnp.float32,
            precision=lax.Precision.HIGHEST,
            causal=False,
            rngs=spx.Rngs(0),
        )
        layer.c_attn.weight.value = jnp.tile(jnp.eye(8) * 16.0, (3, 1))
        hidden = jnp.asarray(np.arange(24).reshape(1, 3, 8) + 1, dtype=jnp.float16)
        out = layer(hidden, None, jnp.arange(3)[None, :], MODE_TRAIN, output_attentions=True)
    qk = (np.asarray(hidden, dtype=np.float32) * 16).reshape(1, 3, 2, 4)
    scores = np.einsum("bqhd,bkhd->bhqk", qk, qk) / 2.0
    assert np.max(scores) > np.finfo(np.float16).max
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    assert np.isfinite(np.asarray(out.attention_output)).all()
    # Non-vanilla implementations need not materialize weights. Compare the public
    # projected output against the independent fp32 attention reference for every backend.
    attended = np.einsum("bhqk,bkhd->bqhd", probabilities, qk).reshape(1, 3, 8).astype(np.float16)
    expected = attended @ np.asarray(layer.c_proj.weight.value, dtype=np.float16).T
    expected += np.asarray(layer.c_proj.bias.value, dtype=np.float16)
    np.testing.assert_allclose(np.asarray(out.attention_output), expected, rtol=2e-3, atol=2e-3)
    if mechanism == "vanilla":
        np.testing.assert_allclose(np.asarray(out.attention_weight), probabilities, atol=1e-6)
    assert out.attention_output.dtype == jnp.float16


@pytest.mark.parametrize("encoder_length", [4, 6])
def test_encoder_padding_is_masked_for_every_decoder_query(encoder_length):
    config = _config(n_embd=8, n_head=2, add_cross_attention=True)
    with config.mesh:
        model = GPT2Model(
            config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        input_ids = jnp.arange(4, dtype=jnp.int32)[None, :]
        encoder_states = jnp.ones((1, encoder_length, 8), dtype=jnp.float32)
        encoder_mask = jnp.asarray([[1, 1] + [0] * (encoder_length - 2)])
        first = model(
            input_ids=input_ids,
            encoder_hidden_states=encoder_states,
            encoder_attention_mask=encoder_mask,
            output_attentions=True,
        )
        # Change only masked keys/values. Neither weights on valid keys nor
        # the public final hidden state may depend on those padded values.
        encoder_states = encoder_states.at[:, 2:, :].set(1000.0)
        second = model(
            input_ids=input_ids,
            encoder_hidden_states=encoder_states,
            encoder_attention_mask=encoder_mask,
            output_attentions=True,
        )
    expected = np.zeros((1, 2, 4, encoder_length), dtype=np.float32)
    expected[..., :2] = 0.5
    for output in (first, second):
        for weights in output.cross_attentions:
            np.testing.assert_allclose(np.asarray(weights), expected, atol=1e-6, rtol=1e-5)
    np.testing.assert_allclose(np.asarray(first.last_hidden_state), np.asarray(second.last_hidden_state), atol=1e-6)


@pytest.mark.parametrize("output_hidden_states", [False, True])
def test_hidden_states_include_final_normalized_output(output_hidden_states):
    config = _config(n_embd=8, n_head=2)
    with config.mesh:
        model = GPT2Model(
            config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        output = model(input_ids=jnp.arange(4)[None, :], output_hidden_states=output_hidden_states)
    if output_hidden_states:
        assert len(output.hidden_states) == config.num_hidden_layers + 1
        np.testing.assert_array_equal(np.asarray(output.hidden_states[-1]), np.asarray(output.last_hidden_state))
    else:
        assert output.hidden_states is None
