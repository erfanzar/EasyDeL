"""StableLM regressions (accelerator only)."""

import jax
import numpy as np
import spectrax as spx
from easydel.modules.stablelm.modeling_stablelm import StableLmModel
from easydel.modules.stablelm.stablelm_configuration import StableLmConfig
from jax import numpy as jnp


def test_first_scanned_forward_does_not_cache_a_tracer():
    config = StableLmConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
        partial_rotary_factor=0.5,
        attn_mechanism="vanilla",
        scan_layers=True,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    model = StableLmModel(config, dtype=jnp.float32, param_dtype=jnp.float32, rngs=spx.Rngs(0))
    ids = jnp.array([[1, 2, 3]], jnp.int32)
    with jax.checking_leaks():
        first = model(input_ids=ids, trace=False).last_hidden_state
    second = model(input_ids=ids, trace=False).last_hidden_state
    assert first.shape == (1, 3, 16)
    np.testing.assert_allclose(np.asarray(first), np.asarray(second), rtol=1e-5, atol=1e-5)
