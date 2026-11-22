"""MTP observable-loss and chain regressions (accelerator-only)."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5TextConfig,
    Qwen3_5VisionConfig,
)
from easydel.modules.qwen3_5_moe import (
    Qwen3_5MoeConfig,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
    Qwen3_5MoeTextConfig,
    Qwen3_5MoeVisionConfig,
)
from jax import numpy as jnp


def _model(moe, vision):
    text_cls = Qwen3_5MoeTextConfig if moe else Qwen3_5TextConfig
    text = text_cls(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=16,
        layer_types=["full_attention"],
        partial_rotary_factor=1.0,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_attention_separate_proj=False,
        mtp_num_hidden_layers=1,
        mtp_loss_coef=0.3,
        attn_mechanism="vanilla",
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    text.moe_force_xla_gmm = True
    config = text
    model_cls = Qwen3_5MoeForCausalLM if moe else Qwen3_5ForCausalLM
    if vision:
        vision_cls = Qwen3_5MoeVisionConfig if moe else Qwen3_5VisionConfig
        vision_config = vision_cls(
            depth=1,
            hidden_size=32,
            intermediate_size=64,
            num_heads=2,
            in_channels=3,
            patch_size=2,
            temporal_patch_size=2,
            spatial_merge_size=2,
            out_hidden_size=32,
            num_position_embeddings=16,
            deepstack_visual_indexes=[],
        )
        config_cls = Qwen3_5MoeConfig if moe else Qwen3_5Config
        config = config_cls(
            text_config=text,
            vision_config=vision_config,
            image_token_id=31,
            video_token_id=30,
            vision_start_token_id=29,
            vision_end_token_id=28,
            sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        )
        model_cls = Qwen3_5MoeForConditionalGeneration if moe else Qwen3_5ForConditionalGeneration
    return model_cls(
        config, dtype=jnp.float32, param_dtype=jnp.float32, precision=jax.lax.Precision.HIGHEST, rngs=spx.Rngs(0)
    )


@pytest.mark.parametrize("moe", [False, True])
@pytest.mark.parametrize("vision", [False, True])
def test_mtp_loss_is_exposed_and_matches_independent_skip_one_ce(moe, vision):
    model = _model(moe, vision)
    ids = jnp.array([[1, 2, 3, 4]], dtype=jnp.int32)
    output = model(input_ids=ids)
    assert output.mtp_loss is not None
    assert output.mtp_logits.shape == (1, 4, 32)
    logits = np.asarray(output.mtp_logits, dtype=np.float64)[0, :2]
    reference = (np.logaddexp.reduce(logits, axis=-1) - logits[np.arange(2), [3, 4]]).mean() * 0.3
    np.testing.assert_allclose(output.mtp_loss, reference, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(output.aux_loss, output.mtp_loss, rtol=2e-5, atol=2e-5)
    chain = model.compute_mtp_chain(output, ids, n_steps=1)
    np.testing.assert_allclose(chain[0], output.mtp_logits, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize(
    "model_cls",
    [Qwen3_5ForCausalLM, Qwen3_5ForConditionalGeneration, Qwen3_5MoeForCausalLM, Qwen3_5MoeForConditionalGeneration],
)
def test_mtp_loss_masks_targets_and_handles_single_token_sequences(model_cls):
    # Public loss API needs no initialized weights; use independent uniform-logit CE.
    logits = jnp.zeros((1, 4, 5), dtype=jnp.float32)
    ids = jnp.array([[1, 2, 3, 4]])
    mask = jnp.array([[1, 1, 1, 0]])
    actual = model_cls.compute_mtp_loss(None, logits, ids, attention_mask=mask)
    np.testing.assert_allclose(actual, np.log(5), rtol=1e-6)
    ignored = model_cls.compute_mtp_loss(None, logits[:, :1], ids[:, :1])
    np.testing.assert_allclose(ignored, 0.0, atol=0.0)
