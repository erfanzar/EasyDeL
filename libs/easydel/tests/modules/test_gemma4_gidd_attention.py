# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Attention numerical and cache regressions (accelerator only)."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.gemma4 import Gemma4TextConfig
from easydel.modules.gemma4.modeling_gemma4 import Gemma4TextModel
from easydel.modules.gidd import GiddConfig
from easydel.modules.gidd.modeling_gidd import GiddAttention
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from spectrax import common_types


def test_gidd_query_key_normalization_matches_independent_per_head_reference():
    config = GiddConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=16,
        use_qk_norm=True,
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    attention = GiddAttention(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(31)
    )
    hidden = jnp.asarray(np.random.default_rng(32).normal(size=(1, 3, 32)), jnp.float32)
    # Only the projection is shared with the system under test; normalization,
    # QK, softmax and PV are computed independently in NumPy.
    q, k, v = [x.reshape(1, 3, 2, 16) for x in np.split(np.asarray(attention.qkv_proj(hidden)), 3, axis=-1)]
    q = q / np.sqrt(np.sum(q**2, axis=-1, keepdims=True) + config.qk_norm_eps)
    k = k / np.sqrt(np.sum(k**2, axis=-1, keepdims=True) + config.qk_norm_eps)
    scores = np.einsum("bqhd,bkhd->bhqk", q, k) * (2 * np.log(config.max_position_embeddings))
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    expected = np.einsum("bhqk,bkhd->bqhd", probabilities, v).reshape(1, 3, 32)
    expected = np.asarray(attention.o_proj(jnp.asarray(expected)))
    output = attention(
        hidden,
        mask_info=MaskInfo.from_segments(jnp.ones((1, 3), jnp.int32)),
        noise_mask=None,
        position_ids=jnp.zeros((1, 3), jnp.int32),
        mode=common_types.MODE_TRAIN,
    )
    np.testing.assert_allclose(np.asarray(output.attention_output), expected, rtol=2e-5, atol=2e-6)


def _gemma4_config(**kwargs):
    return Gemma4TextConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        global_head_dim=8,
        layer_types=["sliding_attention", "full_attention"] * 2,
        sliding_window=4,
        max_position_embeddings=16,
        hidden_size_per_layer_input=0,
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
        output_attentions=False,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        **kwargs,
    )


@pytest.mark.parametrize("bidirectional", [None, "vision", "all"])
@pytest.mark.parametrize("with_token_types", [False, True])
def test_gemma4_public_attention_weights_obey_bidirectional_mode(bidirectional, with_token_types):
    config = _gemma4_config(use_bidirectional_attention=bidirectional)
    model = Gemma4TextModel(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(33)
    )
    token_types = np.asarray([[0, 1, 1, 0, 0]], np.int32)
    output = model(
        input_ids=jnp.asarray([[1, 2, 3, 4, 5]], jnp.int32),
        token_type_ids=jnp.asarray(token_types) if with_token_types else None,
        output_attentions=True,
    )
    row, col = np.arange(5)[:, None], np.arange(5)[None, :]
    full = np.ones((5, 5), bool) if bidirectional == "all" else col <= row
    local = full & (np.abs(col - row) < config.sliding_window)
    if bidirectional == "vision" and with_token_types:
        vision = (token_types[0, :, None] == 1) & (token_types[0, None, :] == 1)
        full |= vision
        local |= vision
    assert len(output.attentions) == 4
    for layer_type, weights in zip(config.layer_types, output.attentions, strict=True):
        expected = local if layer_type == "sliding_attention" else full
        actual = np.asarray(weights)
        assert actual.shape == (1, 2, 5, 5)
        np.testing.assert_array_equal(actual > 0, np.broadcast_to(expected, actual.shape))
        np.testing.assert_allclose(actual.sum(axis=-1), 1.0, rtol=1e-6, atol=1e-6)


def test_gemma4_shared_cached_prefill_decode_and_sliding_overflow_match_full_history():
    config = _gemma4_config(num_kv_shared_layers=2)
    model = Gemma4TextModel(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(34)
    )
    tokens = jnp.asarray([[1, 2, 3, 4, 5, 6, 7]], jnp.int32)
    positions = jnp.zeros_like(tokens)  # Identity RoPE, isolating cache history and masks.
    reference = model(input_ids=tokens, position_ids=positions, output_attentions=True)
    cache = model.init_cache(batch_size=1, max_length=8)
    assert cache.views[2] is None and cache.views[3] is None
    start = 0
    for width in (2, 3, 1, 1):
        stop = start + width
        output = model(
            input_ids=tokens[:, start:stop],
            position_ids=positions[:, start:stop],
            attention_mask=jnp.ones((1, width), jnp.bool_),
            past_key_values=cache,
            cache_metadata=None,
            output_attentions=True,
        )
        cache = output.past_key_values
        np.testing.assert_allclose(
            np.asarray(output.last_hidden_state),
            np.asarray(reference.last_hidden_state[:, start:stop]),
            rtol=3e-5,
            atol=3e-6,
        )
        # Donor and shared layers see exactly the same positions, including
        # multi-token prefill's tail that does not fit in final storage.
        for donor, shared in ((0, 2), (1, 3)):
            np.testing.assert_array_equal(
                np.asarray(output.attentions[donor]) > 0,
                np.asarray(output.attentions[shared]) > 0,
            )
            np.testing.assert_array_equal(np.asarray(cache.views[donor].indexes), [stop])
        start = stop


def test_gemma4_bidirectional_all_cached_prefill_preserves_future_visibility():
    config = _gemma4_config(use_bidirectional_attention="all")
    model = Gemma4TextModel(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(35)
    )
    output = model(
        input_ids=jnp.asarray([[1, 2, 3, 4, 5]], jnp.int32),
        past_key_values=model.init_cache(batch_size=1, max_length=8),
        cache_metadata=None,
        output_attentions=True,
    )
    for layer_type, weights in zip(config.layer_types, output.attentions, strict=True):
        actual = np.asarray(weights)
        assert (actual[..., 0, 1] > 0).all()
        if layer_type == "full_attention":
            assert (actual[..., 0, 4] > 0).all()
        else:
            np.testing.assert_array_equal(actual[..., 0, 3:5], 0)
