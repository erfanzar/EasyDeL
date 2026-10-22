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

"""Gemma2 and GIDD array-computation regressions (accelerator only)."""

import numpy as np
import pytest
import spectrax as spx
from easydel.modules.gemma2 import Gemma2Config
from easydel.modules.gemma2.modeling_gemma2 import Gemma2DecoderLayer
from easydel.modules.gidd import GiddConfig
from easydel.modules.gidd.modeling_gidd import GiddForDiffusionLM
from jax import numpy as jnp


def _gidd(tie: bool):
    cfg = GiddConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=32,
        head_init_scale=0.1,
        tie_word_embeddings=tie,
        attn_mechanism="vanilla",
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    return GiddForDiffusionLM(config=cfg, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))


def test_gidd_tied_embeddings_drive_logits():
    model = _gidd(tie=True)
    hidden = jnp.asarray(np.random.default_rng(0).standard_normal((1, 4, 32)), jnp.float32)
    got = model.compute_lm_logits(hidden)
    want = hidden @ model.get_embedding().weight.value.T
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-5, atol=1e-5)


def test_gidd_untied_head_uses_own_weight():
    model = _gidd(tie=False)
    hidden = jnp.asarray(np.random.default_rng(1).standard_normal((1, 4, 32)), jnp.float32)
    got = model.compute_lm_logits(hidden)
    want = model.lm_head(hidden)
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-6, atol=1e-6)


def test_gemma2_decoder_layer_is_sliding_matches_layer_types():
    cfg = Gemma2Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    for idx, layer_type in enumerate(cfg.layer_types):
        layer = Gemma2DecoderLayer(cfg, layer_idx=idx, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        assert layer.is_sliding == (layer_type == "sliding_attention")


def test_gidd_log_snr_conditions_embeddings_and_logits():
    model = _gidd(tie=False)
    inputs = jnp.asarray(np.random.default_rng(9).normal(size=(2, 3, 32)), jnp.float32)
    unconditioned = model(inputs_embeds=inputs, output_hidden_states=True)
    neutral = model(inputs_embeds=inputs, log_snr=jnp.asarray(0.0), output_hidden_states=True)
    np.testing.assert_array_equal(np.asarray(neutral.logits), np.asarray(unconditioned.logits))
    log_snr = np.asarray([-2.0, 2.0], np.float32)
    conditioned = model(inputs_embeds=inputs, log_snr=jnp.asarray(log_snr), output_hidden_states=True)
    level = (1.0 / (1.0 + np.exp(log_snr)) - 0.5)[:, None, None]
    inv_freq = np.exp(-np.log(10000.0) * np.arange(16, dtype=np.float32) / 15)
    phase = level * inv_freq
    encoding = np.concatenate((np.sin(phase), np.cos(phase) - 1.0), axis=-1)
    np.testing.assert_allclose(
        np.asarray(conditioned.hidden_states[0]), np.asarray(inputs) + encoding, rtol=1e-5, atol=1e-6
    )
    assert not np.allclose(np.asarray(conditioned.logits), np.asarray(unconditioned.logits))
    token_levels = jnp.broadcast_to(jnp.asarray(log_snr)[:, None], (2, 3))
    per_token = model(inputs_embeds=inputs, log_snr=token_levels)
    np.testing.assert_allclose(np.asarray(per_token.logits), np.asarray(conditioned.logits), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("batch_size", "values", "expanded"),
    [
        (2, [-2.0, 0.5, 2.0], [[-2.0, 0.5, 2.0]] * 2),
        (2, [[-2.0, 0.5, 2.0]], [[-2.0, 0.5, 2.0]] * 2),
        (2, [[-2.0], [2.0]], [[-2.0] * 3, [2.0] * 3]),
        (2, [-2.0, 2.0], [[-2.0] * 3, [2.0] * 3]),
        (2, [0.5], [[0.5] * 3] * 2),
        # Ambiguous vectors stay per-example; an explicit row selects per-token values.
        (3, [-2.0, 0.5, 2.0], [[-2.0] * 3, [0.5] * 3, [2.0] * 3]),
        (3, [[-2.0, 0.5, 2.0]], [[-2.0, 0.5, 2.0]] * 3),
    ],
    ids=["sequence-vector", "sequence-row", "batch-column", "batch-vector", "singleton", "ambiguous", "explicit-row"],
)
def test_gidd_log_snr_broadcasting_matches_per_token_conditioning(batch_size, values, expanded):
    model = _gidd(tie=False)
    inputs = np.random.default_rng(11).normal(size=(batch_size, 3, 32)).astype(np.float32)
    output = model(inputs_embeds=jnp.asarray(inputs), log_snr=jnp.asarray(values), output_hidden_states=True)
    expanded = np.asarray(expanded, dtype=np.float32)
    level = 1.0 / (1.0 + np.exp(expanded)) - 0.5
    inv_freq = np.exp(-np.log(10000.0) * np.arange(16, dtype=np.float32) / 15)
    phase = level[..., None] * inv_freq
    encoding = np.concatenate((np.sin(phase), np.cos(phase) - 1.0), axis=-1)
    np.testing.assert_allclose(np.asarray(output.hidden_states[0]), inputs + encoding, rtol=1e-5, atol=1e-6)
    expanded_output = model(inputs_embeds=jnp.asarray(inputs), log_snr=jnp.asarray(expanded))
    np.testing.assert_allclose(np.asarray(output.logits), np.asarray(expanded_output.logits), rtol=1e-5, atol=1e-6)


def test_gidd_log_snr_endpoints_are_finite_and_bad_shapes_raise():
    model = _gidd(tie=False)
    inputs = jnp.asarray(np.random.default_rng(10).normal(size=(2, 3, 32)), jnp.float32)
    output = model(inputs_embeds=inputs, log_snr=jnp.asarray([-jnp.inf, jnp.inf]))
    assert np.isfinite(np.asarray(output.logits)).all()
    with pytest.raises(ValueError, match="log_snr must be"):
        model(inputs_embeds=inputs, log_snr=jnp.zeros((4,)))
