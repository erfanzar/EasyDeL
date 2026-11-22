"""Thinker hidden and embedding inputs use their separate checkpoint projections."""

import numpy as np
import spectrax as spx
from easydel.modules.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeTalkerForConditionalGeneration
from easydel.modules.qwen3_omni_moe.qwen3_omni_moe_configuration import Qwen3OmniMoeTalkerConfig
from jax import numpy as jnp


def test_talker_projects_thinker_hidden_states_and_text_embeddings_separately():
    text_config = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "mlp_only_layers": [0],
        "attn_mechanism": "vanilla",
        "scan_layers": False,
    }
    predictor_config = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 8,
        "num_code_groups": 2,
        "attn_mechanism": "vanilla",
        "scan_layers": False,
    }
    config = Qwen3OmniMoeTalkerConfig(
        text_config=text_config,
        code_predictor_config=predictor_config,
        thinker_hidden_size=16,
        num_code_groups=2,
    )
    model = Qwen3OmniMoeTalkerForConditionalGeneration(
        config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0)
    )
    hidden = jnp.asarray(np.random.default_rng(0).normal(size=(1, 3, 16)), dtype=jnp.float32)
    embeds = hidden + 2
    mask = jnp.array([[True, False, True]])
    expected_embeds = jnp.where(mask[..., None], model.hidden_projection(hidden), model.text_projection(embeds))
    expected = model(inputs_embeds=expected_embeds).logits
    actual = model(
        thinker_hidden_states=hidden,
        thinker_inputs_embeds=embeds,
        thinker_multimodal_mask=mask,
    ).logits
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=2e-5, atol=2e-5)
    hidden_only = model(thinker_hidden_states=hidden).logits
    projected_hidden = model(inputs_embeds=model.hidden_projection(hidden)).logits
    np.testing.assert_allclose(np.asarray(hidden_only), np.asarray(projected_hidden), rtol=2e-5, atol=2e-5)
