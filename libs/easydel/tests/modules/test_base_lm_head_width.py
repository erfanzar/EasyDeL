"""Accelerator regressions for explicit versus default base LM-head widths."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules._base import BaseCausalLMModule
from easydel.modules.arctic import ArcticConfig, ArcticModel
from jax import numpy as jnp


@pytest.mark.parametrize("input_dim", [None, 4])
@pytest.mark.parametrize("head_name", ["lm_head", "output"])
def test_base_lm_head_input_width_preserves_projection_and_parameter_path(input_dim, head_name):
    config = ArcticConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
        num_local_experts=2,
        num_experts_per_tok=1,
        moe_layer_frequency=2,
        tie_word_embeddings=False,
    )
    model = BaseCausalLMModule(
        config=config,
        base_model_class=ArcticModel,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
        lm_head_name=head_name,
        lm_head_input_dim=input_dim,
    )
    width = 8 if input_dim is None else input_dim
    kernel = np.arange(width * 16, dtype=np.float32).reshape(width, 16) / 128
    hidden = np.arange(4 * width, dtype=np.float32).reshape(4, width) / 16
    model.get_lm_head().weight.value = jnp.asarray(kernel)
    assert config.hidden_size == 8
    assert model.get_lm_head().weight.value.shape == (width, 16)
    _, state = spx.export(model)
    paths = {tuple(str(key.key) for key in path) for path, _ in jax.tree_util.tree_flatten_with_path(state)[0]}
    assert ("parameters", head_name, "weight") in paths
    expected = hidden @ kernel
    np.testing.assert_allclose(model.compute_lm_logits(jnp.asarray(hidden)), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(model.make_lm_head_fn()(jnp.asarray(hidden)), expected, rtol=1e-5, atol=1e-6)
