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

"""Cohere, Cohere2, DBRX and CLIP numerical regressions (accelerator only)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.auto.auto_modeling import AutoEasyDeLModelForEmbedding
from easydel.modules.clip.clip_configuration import CLIPTextConfig, CLIPVisionConfig
from easydel.modules.clip.modeling_clip import CLIPEncoder, CLIPVisionEmbeddings
from easydel.modules.cohere.cohere_configuration import CohereConfig
from easydel.modules.cohere.modeling_cohere import CohereForSequenceClassification
from easydel.modules.cohere2.cohere2_configuration import Cohere2Config
from easydel.modules.cohere2.modeling_cohere2 import Cohere2ForCausalLM, Cohere2ForSequenceClassification
from easydel.modules.dbrx.dbrx_configuration import DbrxAttentionConfig, DbrxConfig, DbrxFFNConfig
from easydel.modules.dbrx.modeling_dbrx import (
    DbrxExperts,
    DbrxFFN,
    DbrxForSequenceClassification,
    DbrxModel,
    DbrxNormAttentionNorm,
    DbrxRouter,
)
from ejkernel.types import MaskInfo


def _tiny_dbrx() -> DbrxConfig:
    config = DbrxConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        vocab_size=128,
        max_seq_len=64,
        attn_mechanism="vanilla",
        attn_config=DbrxAttentionConfig(kv_n_heads=2),
        ffn_config=DbrxFFNConfig(ffn_hidden_size=64, moe_num_experts=4, moe_top_k=2, moe_jitter_eps=0.1),
    )
    config.moe_force_xla_gmm = True
    return config


def test_cohere2_lm_head_uses_tied_embeddings():
    config = Cohere2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        sliding_window=8,
        sliding_window_pattern=2,
        max_position_embeddings=64,
        attn_mechanism="vanilla",
    )
    assert config.tie_word_embeddings
    with config.mesh:
        model = Cohere2ForCausalLM(
            config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        hidden = jax.random.normal(jax.random.PRNGKey(1), (1, 4, 64), dtype=jnp.float32)
        logits = model.apply_lm_head(hidden)
        projected = model.make_lm_head_fn()(hidden)
        expected = (hidden @ model.get_embedding().weight.value.T) * config.logit_scale
    np.testing.assert_allclose(np.asarray(logits), np.asarray(expected), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(projected), np.asarray(expected), rtol=1e-4, atol=1e-4)


def test_dbrx_model_returns_updated_cache():
    config = _tiny_dbrx()
    with config.mesh:
        model = DbrxModel(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        model.eval()
        cache = model.init_cache(batch_size=1, max_length=16)
        out = model(input_ids=jnp.ones((1, 4), dtype=jnp.int32), past_key_values=cache)
        assert out.past_key_values is not None
        for view in out.past_key_values.views:
            np.testing.assert_array_equal(np.asarray(view.indexes), [4])
        next_out = model(
            input_ids=jnp.ones((1, 1), dtype=jnp.int32),
            position_ids=jnp.full((1, 1), 4, dtype=jnp.int32),
            past_key_values=out.past_key_values,
        )
    for view in next_out.past_key_values.views:
        np.testing.assert_array_equal(np.asarray(view.indexes), [5])


def test_dbrx_router_jitter_is_bounded_around_one():
    config = _tiny_dbrx()
    with config.mesh:
        router = DbrxRouter(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        noise = router.jitter(jnp.ones((4, 16, 64), dtype=jnp.float32))
    noise = np.asarray(noise)
    assert noise.min() >= 1.0 - 0.1 - 1e-6
    assert noise.max() <= 1.0 + 0.1 + 1e-6
    assert noise.std() > 0


def test_clip_patch_embedding_honours_param_dtype():
    config = CLIPVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        image_size=16,
        patch_size=8,
    )
    embeddings = CLIPVisionEmbeddings(config, dtype=jnp.float32, param_dtype=jnp.bfloat16, rngs=spx.Rngs(0))
    assert embeddings.patch_embedding.weight.value.dtype == jnp.bfloat16


@pytest.mark.parametrize("method", ["fused_moe", "standard_moe"])
def test_dbrx_ffn_jitters_router_only_during_training(method):
    config = _tiny_dbrx()
    config.moe_method = method
    # Fixed assignment and a single normalized expert make expert outputs independent
    # of router logits, so jitter leaking into expert inputs is observable.
    config.ffn_config.uniform_expert_assignment = True
    config.ffn_config.moe_top_k = 1
    with config.mesh:
        ffn = DbrxFFN(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        x = jnp.ones((2, 8, config.d_model), dtype=jnp.float32)
        ffn.train()
        train_output, train_logits = ffn(x)
        _, next_train_logits = ffn(x)
        ffn.eval()
        eval_output, eval_logits = ffn(x)
        _, next_eval_logits = ffn(x)
        expected, _, _ = ffn.router(x, deterministic=True)
    assert not np.array_equal(np.asarray(train_logits), np.asarray(next_train_logits))
    np.testing.assert_allclose(np.asarray(train_output), np.asarray(eval_output), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(np.asarray(eval_logits), np.asarray(expected), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(np.asarray(eval_logits), np.asarray(next_eval_logits))


@pytest.mark.parametrize("dropout", ["embedding", "attention"])
def test_dbrx_configured_dropout_changes_training_but_not_eval(dropout):
    config = _tiny_dbrx()
    config.ffn_config.moe_jitter_eps = None
    if dropout == "embedding":
        config.emb_pdrop = 0.5
    else:
        config.attn_config.attn_pdrop = 0.5
    with config.mesh:
        model = DbrxModel(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        input_ids = jnp.arange(1, 17, dtype=jnp.int32).reshape(1, 16)
        model.train()
        train_output = model(input_ids=input_ids).last_hidden_state
        next_train_output = model(input_ids=input_ids).last_hidden_state
        model.eval()
        eval_output = model(input_ids=input_ids).last_hidden_state
        next_eval_output = model(input_ids=input_ids).last_hidden_state
    assert not np.array_equal(np.asarray(train_output), np.asarray(next_train_output))
    np.testing.assert_array_equal(np.asarray(eval_output), np.asarray(next_eval_output))


@pytest.mark.parametrize("family", ["cohere", "cohere2", "dbrx"])
def test_sequence_classification_pools_last_attended_token(family):
    common = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=64,
        num_labels=3,
        pad_token_id=0,
        attn_mechanism="vanilla",
    )
    if family == "cohere":
        config, cls = CohereConfig(**common), CohereForSequenceClassification
    elif family == "cohere2":
        config, cls = Cohere2Config(**common), Cohere2ForSequenceClassification
    else:
        config, cls = _tiny_dbrx(), DbrxForSequenceClassification
        config.num_labels = 3
        config.pad_token_id = 0
    with config.mesh:
        model = cls(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        model.eval()
        input_ids = jnp.asarray([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.int32)
        attention_mask = jnp.asarray([[0, 1, 1, 0], [1, 1, 0, 0]], dtype=jnp.bool_)
        hidden = model.get_decoder()(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        token_logits = model.score(hidden)
        expected = jnp.stack([token_logits[0, 2], token_logits[1, 1]])
        actual = model(input_ids=input_ids, attention_mask=attention_mask).logits
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-4, atol=1e-4)


def test_clip_projection_auto_loader_and_default_inputs():
    import torch

    config = CLIPTextConfig(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        projection_dim=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=8,
        eos_token_id=2,
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
    )
    with jax.default_matmul_precision("highest"), config.mesh:
        model = AutoEasyDeLModelForEmbedding.from_config(
            config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        model.eval()
        ids = jnp.asarray([[3, 4, 2, 1], [5, 6, 7, 2]], dtype=jnp.int32)
        actual = model(input_ids=ids)
        explicit = model(
            input_ids=ids,
            mask_info=MaskInfo.from_attention_mask(jnp.ones_like(ids, dtype=jnp.bool_)),
            position_ids=jnp.broadcast_to(jnp.arange(4)[None, :], ids.shape),
        )
        exported = model.to_torch().eval()
    assert actual.text_embeds.shape == (2, 16)
    np.testing.assert_allclose(np.asarray(actual.text_embeds), np.asarray(explicit.text_embeds), rtol=1e-5, atol=1e-5)
    with torch.no_grad():
        reference = exported(input_ids=torch.tensor(np.asarray(ids), dtype=torch.long)).text_embeds
    np.testing.assert_allclose(np.asarray(actual.text_embeds), reference.numpy(), rtol=1e-4, atol=1e-4)


def test_dbrx_single_expert_public_api_matches_independent_reference():
    config = _tiny_dbrx()
    with config.mesh:
        experts = DbrxExperts(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        x = jnp.ones((1, 4, config.d_model), dtype=jnp.float32)
        actual = experts.mlp(x, expert_idx=1)
        shape = (config.ffn_config.moe_num_experts, config.ffn_config.ffn_hidden_size, config.d_model)
        gate = np.asarray(x) @ np.asarray(experts.mlp.w1.value).reshape(shape)[1].T
        up = np.asarray(x) @ np.asarray(experts.mlp.v1.value).reshape(shape)[1].T
        expected = ((gate / (1 + np.exp(-gate))) * up) @ np.asarray(experts.mlp.w2.value).reshape(shape)[1]
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-4, atol=1e-4)


def test_clip_encoder_keeps_public_causal_mask():
    text_config = CLIPTextConfig(
        num_hidden_layers=0, max_position_embeddings=8, mask_max_position_embeddings=8, precompute_masks=True
    )
    encoder = CLIPEncoder(text_config, rngs=spx.Rngs(0))
    cached_mask = encoder.causal_mask
    mask = np.asarray(getattr(cached_mask, "value", cached_mask))
    assert mask.shape[-2:] == (8, 8)
    np.testing.assert_array_equal(mask.reshape(-1, 8, 8)[0].astype(bool), np.tril(np.ones((8, 8), dtype=bool)))
    vision = CLIPEncoder(CLIPVisionConfig(num_hidden_layers=0), rngs=spx.Rngs(0))
    assert vision.causal_mask is None


def test_dbrx_residual_dropout_applies_once():
    config = _tiny_dbrx()
    config.resid_pdrop = 0.5
    with config.mesh:
        layer = DbrxNormAttentionNorm(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0), layer_idx=0)
        layer.train()
        x = jax.random.normal(jax.random.PRNGKey(1), (1, 8, config.d_model))
        mask_info = MaskInfo.from_attention_mask(jnp.ones((1, 8), dtype=jnp.bool_))
        positions = jnp.arange(8)[None, :]
        attn = layer.attn(
            hidden_states=layer.norm_1(x), mask_info=mask_info, position_ids=positions, mode="train"
        ).attention_output
        actual = layer(hidden_states=x, mask_info=mask_info, position_ids=positions, mode="train").residual_states
    delta = np.asarray(actual - x)
    scaled = 2 * np.asarray(attn)
    assert np.all(np.isclose(delta, 0, atol=1e-6) | np.isclose(delta, scaled, rtol=1e-4, atol=1e-6))
    assert np.any(np.abs(delta) > 1e-6)
    assert np.any(np.isclose(delta, 0, atol=1e-6) & (np.abs(scaled) > 1e-6))
