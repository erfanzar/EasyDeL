"""Qwen2 dense/MoE routing, pooling and custom RoPE regressions (accelerator only)."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.qwen2.modeling_qwen import Qwen2ForSequenceClassification
from easydel.modules.qwen2.qwen_configuration import Qwen2Config
from easydel.modules.qwen2_moe.modeling_qwen2_moe import (
    Qwen2MoeAttention,
    Qwen2MoeForSequenceClassification,
    Qwen2MoeModel,
)
from easydel.modules.qwen2_moe.qwen2_moe_configuration import Qwen2MoeConfig
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from spectrax import common_types


def config_for(config_class, **kwargs):
    config = config_class(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
        attn_mechanism="vanilla",
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        **kwargs,
    )
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape((1,) * 6), ("pp", "dp", "fsdp", "ep", "tp", "sp"))
    config.set_model_mesh(mesh)
    return config


MOE_KWARGS = dict(num_experts=2, num_experts_per_tok=2, moe_intermediate_size=16, shared_expert_intermediate_size=16)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("schedule", [dict(mlp_only_layers=[0]), dict(decoder_sparse_step=2)])
def test_mixed_dense_and_sparse_layers_preserve_batch_and_router_outputs(batch_size, schedule):
    config = config_for(Qwen2MoeConfig, output_router_logits=True, **MOE_KWARGS, **schedule)
    model = Qwen2MoeModel(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    model.eval()
    ids = jnp.arange(batch_size * 3, dtype=jnp.int32).reshape(batch_size, 3)
    batched = model(input_ids=ids)
    assert batched.last_hidden_state.shape == (batch_size, 3, 16)
    assert len(batched.router_logits) == 1
    assert batched.router_logits[0].shape[-1] == 2
    for index in range(batch_size):
        single = model(input_ids=ids[index : index + 1]).last_hidden_state
        np.testing.assert_allclose(batched.last_hidden_state[index : index + 1], single, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "config_class,model_class",
    [(Qwen2Config, Qwen2ForSequenceClassification), (Qwen2MoeConfig, Qwen2MoeForSequenceClassification)],
)
def test_classification_pools_last_unmasked_token_without_pad_id(config_class, model_class):
    extra = (
        {**MOE_KWARGS, "output_router_logits": True, "router_aux_loss_coef": 0.125}
        if config_class is Qwen2MoeConfig
        else {}
    )
    config = config_for(config_class, pad_token_id=None, num_labels=3, **extra)
    model = model_class(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(2))
    model.eval()
    ids = jnp.array([[1, 2, 3], [4, 5, 6]], dtype=jnp.int32)
    mask = jnp.array([[1, 1, 0], [0, 1, 1]], dtype=jnp.bool_)
    outputs = model(input_ids=ids, attention_mask=mask)
    token_outputs = model.model(input_ids=ids, attention_mask=mask)
    scores = np.asarray(model.score(token_outputs.last_hidden_state))
    np.testing.assert_allclose(outputs.logits, scores[np.arange(2), [1, 2]], rtol=1e-5, atol=1e-5)
    assert outputs.past_key_values is not None
    without_head = model(input_ids=ids, attention_mask=mask, apply_lm_head=False, output_hidden_states=True)
    assert without_head.logits is None
    assert without_head.past_key_values is not None
    np.testing.assert_allclose(without_head.hidden_states[-1], token_outputs.last_hidden_state, rtol=1e-5, atol=1e-5)
    if config_class is Qwen2MoeConfig:
        np.testing.assert_allclose(outputs.aux_loss, 0.25, rtol=1e-5)


def test_custom_head_dim_rotary_works_without_precomputed_frequencies():
    config = config_for(Qwen2MoeConfig, head_dim=4, **MOE_KWARGS)
    attention = Qwen2MoeAttention(config, dtype=jnp.float32, param_dtype=jnp.float32, layer_idx=0, rngs=spx.Rngs(1))
    inputs = jnp.ones((1, 3, 16), dtype=jnp.float32)
    mask = MaskInfo.dynamic_init(inputs_embeds=inputs)
    outputs = attention(inputs, mask, mask.q_position_ids, common_types.MODE_TRAIN, frequencies=None)
    assert outputs.attention_output.shape == inputs.shape
    assert np.isfinite(np.asarray(outputs.attention_output)).all()
