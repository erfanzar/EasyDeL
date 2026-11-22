"""Numerical regressions for Qwen3-MoE outputs (accelerator-only)."""

import numpy as np
import pytest
import spectrax as spx
from easydel.modules.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM, Qwen3MoeForSequenceClassification
from jax import numpy as jnp


def _config(**overrides):
    kwargs = dict(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_experts=2,
        num_experts_per_tok=1,
        mlp_only_layers=[0],
        max_position_embeddings=16,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        attn_mechanism="vanilla",
        pad_token_id=0,
        num_labels=3,
    )
    kwargs.update(overrides)
    config = Qwen3MoeConfig(**kwargs)
    config.moe_force_xla_gmm = True
    return config


@pytest.mark.parametrize("cached", [False, True])
def test_router_outputs_skip_dense_layers_and_never_collect_cache_views(cached):
    config = _config()
    model = Qwen3MoeForCausalLM(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    cache = model.init_cache(batch_size=1, max_length=8) if cached else None
    output = model(
        input_ids=jnp.array([[1, 2, 3]], dtype=jnp.int32),
        past_key_values=cache,
        output_router_logits=True,
        output_hidden_states=True,
    )
    assert len(output.router_logits) == 1
    router = output.router_logits[0]
    assert router.shape[-1] == config.num_experts
    assert router.size == 3 * config.num_experts
    assert np.isfinite(np.asarray(router)).all()
    assert np.isfinite(np.asarray(output.aux_loss)).all()
    assert len(output.hidden_states) == config.num_hidden_layers + 1
    np.testing.assert_allclose(output.hidden_states[-1], output.last_hidden_state, atol=1e-6)


def test_all_dense_router_outputs_are_empty_and_hidden_history_includes_final_norm():
    config = _config(mlp_only_layers=[0, 1])
    model = Qwen3MoeForCausalLM(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    output = model(input_ids=jnp.array([[1, 2, 3]]), output_router_logits=True, output_hidden_states=True)
    assert output.router_logits == ()
    assert output.aux_loss is None
    assert len(output.hidden_states) == 3
    np.testing.assert_allclose(output.hidden_states[-1], output.last_hidden_state, atol=1e-6)


def test_sequence_classification_uses_attention_mask_with_precomputed_embeddings():
    config = _config(num_hidden_layers=1, mlp_only_layers=[0])
    model = Qwen3MoeForSequenceClassification(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(1))
    embeds = jnp.arange(2 * 4 * config.hidden_size, dtype=jnp.float32).reshape(2, 4, config.hidden_size) / 100
    mask = jnp.array([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=jnp.int32)
    output = model(inputs_embeds=embeds, attention_mask=mask)
    token_output = model.model(inputs_embeds=embeds, attention_mask=mask)
    token_logits = model.score(token_output.last_hidden_state)
    expected = jnp.stack([token_logits[0, 1], token_logits[1, 2]])
    np.testing.assert_allclose(output.logits, expected, atol=1e-6)
    skipped = model(inputs_embeds=embeds, attention_mask=mask, apply_lm_head=False, output_hidden_states=True)
    assert skipped.logits is None
    assert len(skipped.hidden_states) == 2
