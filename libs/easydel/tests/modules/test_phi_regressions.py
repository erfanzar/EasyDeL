"""Phi-family dropout, attention and router-output regression tests (accelerator only)."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.phi.modeling_phi import PhiAttention, PhiModel
from easydel.modules.phi.phi_configuration import PhiConfig
from easydel.modules.phi3.modeling_phi3 import Phi3Attention, Phi3Model
from easydel.modules.phi3.phi3_configuration import Phi3Config
from easydel.modules.phimoe.modeling_phimoe import PhiMoEAttention, PhiMoeForCausalLM, PhiMoeModel
from easydel.modules.phimoe.phimoe_configuration import PhiMoeConfig
from ejkernel.types import MaskInfo
from jax import numpy as jnp
from spectrax import common_types


def tiny_config(config_class, **kwargs):
    config = config_class(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        attn_mechanism="vanilla",
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        **kwargs,
    )
    config.set_model_mesh(
        jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape((1,) * 6), ("pp", "dp", "fsdp", "ep", "tp", "sp"))
    )
    return config


@pytest.mark.parametrize(
    "config_class,model_class", [(PhiConfig, PhiModel), (Phi3Config, Phi3Model), (PhiMoeConfig, PhiMoeModel)]
)
def test_embedding_dropout_is_applied_before_first_layer(config_class, model_class):
    config = tiny_config(
        config_class, embd_pdrop=0.5, **({"num_local_experts": 2} if config_class is PhiMoeConfig else {})
    )
    model = model_class(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(4))
    inputs = jnp.ones((1, 16, 16), jnp.float32)
    model.eval()
    evaluation = model(inputs_embeds=inputs, output_hidden_states=True).hidden_states[0]
    np.testing.assert_array_equal(evaluation, inputs)
    model.train()
    training = np.asarray(model(inputs_embeds=inputs, output_hidden_states=True).hidden_states[0])
    assert np.any(training == 0.0)
    assert np.any(training == 2.0)
    assert np.all((training == 0.0) | (training == 2.0))


@pytest.mark.parametrize("config_class,attention_class", [(PhiConfig, PhiAttention), (Phi3Config, Phi3Attention)])
def test_residual_dropout_is_owned_by_decoder_not_attention(config_class, attention_class):
    config = tiny_config(config_class, resid_pdrop=0.5)
    attention = attention_class(config, layer_idx=0, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(1))
    inputs = jnp.arange(48, dtype=jnp.float32).reshape(1, 3, 16) / 48
    mask = MaskInfo.dynamic_init(inputs_embeds=inputs)
    args = dict(hidden_states=inputs, mask_info=mask, position_ids=mask.q_position_ids, mode=common_types.MODE_TRAIN)
    attention.eval()
    expected = attention(**args).attention_output
    attention.train()
    actual = attention(**args).attention_output
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert config.resid_pdrop == 0.5


def test_phi3_call_time_attention_weights_override_config():
    config = tiny_config(Phi3Config, output_attentions=False)
    attention = Phi3Attention(config, layer_idx=0, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(3))
    inputs = jnp.ones((1, 3, 16), jnp.float32)
    mask = MaskInfo.dynamic_init(inputs_embeds=inputs)
    outputs = attention(inputs, mask, mask.q_position_ids, common_types.MODE_TRAIN, output_attentions=True)
    assert outputs.attention_weight is not None
    assert outputs.attention_weight.shape[-2:] == (3, 3)
    np.testing.assert_allclose(np.asarray(outputs.attention_weight).sum(-1), 1.0, rtol=1e-5)


@pytest.mark.parametrize("config_class,model_class", [(PhiConfig, PhiModel), (Phi3Config, Phi3Model)])
def test_forward_consumes_public_cached_rotary_frequencies(config_class, model_class):
    model = model_class(tiny_config(config_class), dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(5))
    model.eval()
    ids = jnp.array([[1, 7, 3, 11]], dtype=jnp.int32)
    cache = model.frequencies
    baseline = model(input_ids=ids).last_hidden_state
    # Replacing the public rotary cache must affect the forward, rather than
    # being ignored in favor of freshly generated config frequencies.
    cache.value = jnp.zeros_like(cache.value)
    changed = model(input_ids=ids).last_hidden_state
    assert model.frequencies is cache
    assert not np.array_equal(baseline, changed)


def test_phimoe_preserves_explicit_attention_precision():
    config = tiny_config(PhiMoeConfig, num_local_experts=2, attn_dtype=jnp.float32, attn_softmax_dtype=jnp.float32)
    PhiMoEAttention(config, layer_idx=0, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=spx.Rngs(0))
    assert config.attn_dtype == jnp.float32
    assert config.attn_softmax_dtype == jnp.float32


def test_phimoe_router_logits_and_coefficient_scaled_aux_loss():
    config = tiny_config(PhiMoeConfig, num_local_experts=2, output_router_logits=True, router_aux_loss_coef=0.125)
    model = PhiMoeForCausalLM(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(7))
    model.eval()
    inputs = jnp.array([[1, 2, 3]], dtype=jnp.int32)
    outputs = model(input_ids=inputs)
    assert len(outputs.router_logits) == 1
    assert outputs.router_logits[0].shape == (1, 3, 2)
    # Both of the two experts are selected for every token, so the independent
    # load-balancing reference is 2 * mean(p0 + p1) = 2.
    np.testing.assert_allclose(outputs.aux_loss, 0.25, rtol=1e-5)
    assert outputs.logits.shape == (1, 3, 32)
    disabled = model(input_ids=inputs, output_router_logits=False, apply_lm_head=False)
    assert disabled.router_logits is None
    assert disabled.aux_loss is None
    assert disabled.logits is None
