"""CLIP vision and DBRX numerical regressions (accelerator-only)."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.auto.auto_modeling import AutoEasyDeLModelForImageClassification
from easydel.modules.clip.clip_configuration import CLIPConfig, CLIPVisionConfig
from easydel.modules.clip.modeling_clip import CLIPVisionModel
from easydel.modules.dbrx.dbrx_configuration import DbrxAttentionConfig, DbrxConfig, DbrxFFNConfig
from easydel.modules.dbrx.modeling_dbrx import DbrxBlock, DbrxForCausalLM, DbrxModel
from ejkernel.types import MaskInfo
from jax import numpy as jnp


def _vision_config():
    return CLIPVisionConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_hidden_layers=1,
        image_size=16,
        patch_size=8,
        num_labels=3,
        attn_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )


def _dbrx_config():
    config = DbrxConfig(
        d_model=64,
        n_heads=4,
        n_layers=1,
        vocab_size=128,
        max_seq_len=32,
        attn_config=DbrxAttentionConfig(kv_n_heads=2),
        ffn_config=DbrxFFNConfig(ffn_hidden_size=64, moe_num_experts=4, moe_top_k=2),
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    config.moe_force_xla_gmm = True
    return config


def test_clip_public_vision_model_preserves_nhwc_and_nchw_parity():
    config = _vision_config()
    with jax.default_matmul_precision("highest"), config.mesh:
        model = CLIPVisionModel(
            config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(0)
        )
        model.eval()
        pixels = jnp.asarray(np.random.default_rng(0).normal(size=(4, 16, 16, 3)), dtype=jnp.float32)
        nhwc = model(pixel_values=pixels)
        nchw = model(pixel_values=jnp.transpose(pixels, (0, 3, 1, 2)))
    assert nhwc.last_hidden_state.shape == (4, 5, 32)
    assert nhwc.last_hidden_state.dtype == jnp.float32
    np.testing.assert_allclose(
        np.asarray(nhwc.last_hidden_state), np.asarray(nchw.last_hidden_state), atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(np.asarray(nhwc.pooler_output), np.asarray(nchw.pooler_output), atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("composite", [False, True])
def test_clip_auto_classifier_resolves_vision_config_and_legacy_composite(composite):
    vision = _vision_config()
    config = (
        CLIPConfig(vision_config=vision.to_dict(), num_labels=3, sharding_axis_dims=(1, 1, 1, 1, 1, 1))
        if composite
        else vision
    )
    with jax.default_matmul_precision("highest"), config.mesh:
        model = AutoEasyDeLModelForImageClassification.from_config(
            config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
        )
        model.eval()
        pixels = jnp.ones((4, 16, 16, 3), dtype=jnp.float32)
        nhwc = model(pixel_values=pixels).logits
        nchw = model(pixel_values=jnp.transpose(pixels, (0, 3, 1, 2))).logits
    assert nhwc.shape == (4, 3)
    np.testing.assert_allclose(np.asarray(nhwc), np.asarray(nchw), atol=1e-6, rtol=1e-6)


def test_dbrx_all_norms_match_independent_low_variance_reference():
    config = _dbrx_config()
    with config.mesh:
        model = DbrxModel(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        host = np.linspace(-0.001, 0.001, 64, dtype=np.float32).reshape(1, 1, 64)
        reference = (host - host.mean(axis=-1, keepdims=True)) / np.sqrt(host.var(axis=-1, keepdims=True) + 1e-5)
        norms = [model.blocks[0].norm_attn_norm.norm_1, model.blocks[0].norm_attn_norm.norm_2, model.norm_f]
        for norm in norms:
            actual = norm(jnp.asarray(host))
            np.testing.assert_allclose(np.asarray(actual), reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("probability", [0.25, 0.5])
def test_dbrx_full_block_drops_ffn_residual_branch_only_in_training(probability):
    config = _dbrx_config()
    # SpectraX inverted dropout accepts rates in [0, 1), so rate=1 is invalid.
    config.resid_pdrop = probability
    with jax.default_matmul_precision("highest"), config.mesh:
        block = DbrxBlock(
            config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(0),
            layer_idx=0,
        )
        # Isolate FFN residual dropout while keeping the real attention and MoE.
        block.norm_attn_norm.dropout.rate = 0.0
        hidden = jnp.asarray(np.random.default_rng(0).normal(size=(4, 4, 64)), dtype=jnp.float32)
        mask = MaskInfo.from_attention_mask(jnp.ones((4, 4), dtype=jnp.bool_))
        positions = jnp.broadcast_to(jnp.arange(4)[None, :], (4, 4))
        block.eval()
        residual = block.norm_attn_norm(
            hidden_states=hidden, mask_info=mask, position_ids=positions, mode="train"
        ).residual_states
        eval_first = block(hidden_states=hidden, mask_info=mask, position_ids=positions, mode="train").hidden_states
        eval_second = block(hidden_states=hidden, mask_info=mask, position_ids=positions, mode="train").hidden_states
        block.train()
        train_first = block(hidden_states=hidden, mask_info=mask, position_ids=positions, mode="train").hidden_states
        train_second = block(hidden_states=hidden, mask_info=mask, position_ids=positions, mode="train").hidden_states

    residual = np.asarray(residual)
    eval_first = np.asarray(eval_first)
    branch = eval_first - residual
    assert np.any(np.abs(branch) > 1e-5)
    # Independently check inverted-dropout math on public full-block outputs:
    # each FFN element is either dropped or kept and scaled by 1 / (1 - p).
    retained = residual + branch / (1.0 - probability)
    for training_output in (train_first, train_second):
        actual = np.asarray(training_output)
        dropped = actual == residual
        assert np.any(dropped & (np.abs(branch) > 1e-5))
        assert np.any(~dropped & (np.abs(branch) > 1e-5))
        np.testing.assert_allclose(actual, np.where(dropped, residual, retained), atol=1e-6, rtol=1e-5)
    assert not np.array_equal(np.asarray(train_first), np.asarray(train_second))
    np.testing.assert_array_equal(eval_first, np.asarray(eval_second))


def test_dbrx_public_embeddings_only_lm_matches_token_ids():
    config = _dbrx_config()
    with jax.default_matmul_precision("highest"), config.mesh:
        model = DbrxForCausalLM(
            config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(0)
        )
        model.eval()
        ids = jnp.arange(1, 17, dtype=jnp.int32).reshape(4, 4)
        mask = jnp.ones_like(ids, dtype=jnp.bool_)
        embeds = model.get_embedding()(ids)
        token_logits = model(input_ids=ids, attention_mask=mask).logits
        embedded_logits = model(inputs_embeds=embeds, attention_mask=mask).logits
    np.testing.assert_allclose(np.asarray(embedded_logits), np.asarray(token_logits), rtol=1e-5, atol=1e-5)
