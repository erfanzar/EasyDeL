"""Arctic classifier, embedding pooling and MoE numerics (accelerator-only)."""

from types import SimpleNamespace

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules._base import BaseEmbeddingModule, BaseSequenceClassificationModule
from easydel.modules.arctic import ArcticForSequenceClassification
from easydel.modules.arctic.modeling_arctic import ArcticDecoderLayer, ArcticMoeBlock
from jax import numpy as jnp

from .test_wrapper_compute import FixtureBackbone, make_vlm, tiny_config


def test_arctic_classifier_preserves_embedding_only_last_token_fallback():
    model = ArcticForSequenceClassification(
        config=tiny_config(num_labels=2, pad_token_id=None),
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(10),
    )
    model.model = FixtureBackbone(model.config)
    hidden = jnp.arange(16, dtype=jnp.float32).reshape(1, 2, 8)
    out = model(inputs_embeds=hidden)
    np.testing.assert_allclose(out.logits, model.score(hidden[:, -1]), rtol=1e-5, atol=1e-6)
    assert out.past_key_values == "updated-cache"


def test_embedding_weighted_mean_is_position_weighted_with_or_without_mask():
    model = BaseEmbeddingModule(
        config=tiny_config(),
        base_model_class=FixtureBackbone,
        pooling_strategy="weighted_mean",
        normalize_embeddings=False,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(11),
    )
    hidden = jnp.array([[[1.0, 0.0] * 4, [3.0, 6.0] * 4]])
    expected = np.array([[(1.0 + 2.0 * 3.0) / 3.0, 4.0] * 4])
    np.testing.assert_allclose(model(inputs_embeds=hidden).embeddings, expected, rtol=1e-6)
    np.testing.assert_allclose(
        model(inputs_embeds=hidden, attention_mask=jnp.ones((1, 2))).embeddings, expected, rtol=1e-6
    )


def test_classification_max_pools_representations_before_cross_channel_head():
    model = BaseSequenceClassificationModule(
        config=tiny_config(num_labels=1, pad_token_id=0),
        base_model_class=FixtureBackbone,
        pooling_strategy="max",
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(12),
    )
    model.model = FixtureBackbone(model.config)
    model.score.weight.value = jnp.ones_like(model.score.weight.value)
    hidden = jnp.array([[[1.0, 0.0] * 4, [0.0, 1.0] * 4]])
    np.testing.assert_allclose(model(inputs_embeds=hidden).logits, [[8.0]], rtol=1e-6)


@pytest.mark.parametrize("method", ["fused_moe", "dense_moe", "standard_moe"])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("chunk_size", [None, 3])
def test_arctic_padding_neither_inflates_capacity_nor_consumes_real_assignments(method, side, chunk_size):
    config = tiny_config(
        moe_layer_frequency=1,
        moe_method=method,
        moe_train_capacity_factor=2.0,
        moe_min_capacity=0,
        moe_chunk_size=chunk_size,
    )
    config.moe_force_xla_gmm = True
    block = ArcticMoeBlock(
        config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(13),
        layer_idx=0,
    )
    block.train()
    block.gate.weight.value = jnp.zeros_like(block.gate.weight.value)
    compact, _ = block(jnp.ones((1, 2, 8)))
    # Four experts, two real top-1 tokens, factor two => capacity one.
    assert np.any(np.abs(np.asarray(compact[:, :1])) > 0)
    np.testing.assert_array_equal(compact[:, 1:], np.zeros((1, 1, 8)))
    validity = jnp.array([[0, 0, 1, 1]] if side == "left" else [[1, 1, 0, 0]], dtype=bool)
    padded, _ = block(jnp.ones((1, 4, 8)), token_validity=validity)
    real = padded[:, 2:] if side == "left" else padded[:, :2]
    pads = padded[:, :2] if side == "left" else padded[:, 2:]
    np.testing.assert_allclose(real, compact, rtol=1e-5, atol=1e-7)
    np.testing.assert_array_equal(pads, np.zeros((1, 2, 8)))


@pytest.mark.parametrize("parallel", [False, True])
def test_arctic_decoder_threads_padding_validity_in_both_residual_branches(parallel):
    class SilentAttention(spx.Module):
        def forward(self, hidden_states, *args, **kwargs):
            return SimpleNamespace(
                attention_output=jnp.zeros_like(hidden_states), attention_weight=None, cache_view=None
            )

    config = tiny_config(
        moe_layer_frequency=1,
        moe_method="dense_moe",
        parallel_attn_mlp_res=parallel,
        moe_train_capacity_factor=2.0,
    )
    layer = ArcticDecoderLayer(
        config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(14),
        layer_idx=0,
    )
    layer.self_attn = SilentAttention()
    layer.block_sparse_moe.gate.weight.value = jnp.zeros_like(layer.block_sparse_moe.gate.weight.value)
    # Random initialization makes the three expert projections O(1e-6), so
    # their residual contribution can disappear within allclose's tolerance.
    # Deterministic averaging projections keep the retained route observable
    # and give an independent SwiGLU reference without loosening tolerances.
    experts = layer.block_sparse_moe.experts
    experts.gate_up_proj.weight.value = jnp.full_like(experts.gate_up_proj.weight.value, 1.0 / config.hidden_size)
    experts.w2.weight.value = jnp.full_like(experts.w2.weight.value, 1.0 / config.intermediate_size)
    if parallel:
        layer.residual_mlp.w2.weight.value = jnp.zeros_like(layer.residual_mlp.w2.weight.value)
    layer.train()

    def run(validity):
        seq_len = validity.shape[1]
        return layer(
            jnp.ones((1, seq_len, 8)),
            mask_info=SimpleNamespace(_q_segment_ids=jnp.where(validity, 0, -1)),
            position_ids=jnp.arange(seq_len)[None],
            mode="train",
        ).hidden_states

    compact = run(jnp.ones((1, 2), dtype=bool))
    padded = run(jnp.array([[False, False, True, True]]))
    np.testing.assert_allclose(padded[:, 2:], compact, rtol=1e-5, atol=1e-6)
    # Uniform routing gives the retained top-1 expert probability 1/4;
    # capacity ceil(2 * 2 / 4) = 1 drops the second real token entirely.
    normalized = 1.0 / np.sqrt(1.0 + config.rms_norm_eps)
    expert_output = normalized**2 / (1.0 + np.exp(-normalized)) / config.num_local_experts
    np.testing.assert_allclose(compact[:, :1], 1.0 + expert_output, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(compact[:, 1:], np.ones((1, 1, 8)))
    np.testing.assert_array_equal(padded[:, :2], np.ones((1, 2, 8)))
    assert not np.allclose(compact[:, :1], compact[:, 1:])


def test_public_multimodal_merge_replaces_both_modalities_and_preserves_text():
    model = make_vlm(image_token_index=3, video_token_index=4)
    ids = jnp.array([[1, 3, 2, 4]])
    text = jnp.ones((1, 4, 8))
    visual = jnp.stack((jnp.full((8,), 3.0), jnp.full((8,), 4.0)))
    actual = model.merge_multimodal_embeddings(ids, text, visual, placeholder_token_id=[3, 4])
    np.testing.assert_array_equal(actual, np.array([[[1.0] * 8, [3.0] * 8, [1.0] * 8, [4.0] * 8]]))
