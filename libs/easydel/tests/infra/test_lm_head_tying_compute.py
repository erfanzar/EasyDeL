"""Constructor-tied trace-safe LM projections; accelerator-only numerical regressions."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules._base import BaseCausalLMModule, BaseConditionalGenerationModule
from easydel.modules.llama import LlamaConfig, LlamaModel
from jax import numpy as jnp


@pytest.mark.parametrize("wrapper", [BaseCausalLMModule, BaseConditionalGenerationModule])
@pytest.mark.parametrize("constructor_tie,config_tie", [(True, False), (False, True), (False, False)])
def test_make_lm_head_fn_matches_independent_projection_inside_traced_loop(wrapper, constructor_tie, config_tie):
    config = LlamaConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=16,
        tie_word_embeddings=config_tie,
    )
    config.add_basic_configurations(sharding_axis_dims=(1, 1, -1, 1, 1, 1), use_sharding_constraint=False)
    with config.mesh:
        model = wrapper(
            config=config,
            base_model_class=LlamaModel,
            tie_word_embeddings=constructor_tie,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=jax.lax.Precision.HIGHEST,
            rngs=spx.Rngs(1),
        )
        hidden = jnp.arange(16, dtype=jnp.float32).reshape(1, 2, 8) / 16
        project = model.make_lm_head_fn()
        actual = jax.jit(lambda values: jax.lax.map(project, values))(hidden)
        weight = (
            model.get_embedding().weight.value.T if constructor_tie or config_tie else model.get_lm_head().weight.value
        )
        expected = np.asarray(hidden) @ np.asarray(weight)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(model.apply_lm_head(hidden), expected, rtol=1e-5, atol=1e-6)
