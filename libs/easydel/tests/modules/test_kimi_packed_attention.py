"""Hardware-only regression for MoonViT's shared packed-image attention mask."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.kimi_vl.kimi_vl_configuration import MoonViTConfig
from easydel.modules.kimi_vl.modeling_kimi_vl import MoonVitEncoderLayer
from jax import numpy as jnp


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_packed_images_match_separate_encoder_passes(dtype):
    config = MoonViTConfig(
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        intermediate_size=16,
        sharding_axis_dims=(1, 1, -1, 1, 1, 1),
    )
    layer = MoonVitEncoderLayer(
        base_config=config,
        num_heads=2,
        hidden_dim=8,
        mlp_dim=16,
        activation=jax.nn.gelu,
        dtype=dtype,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(0),
    )
    inputs = jnp.asarray(np.sin(np.arange(40).reshape(5, 8)), dtype=dtype)
    angles = jnp.asarray(np.arange(10).reshape(5, 2) / 7, dtype=jnp.float32)
    frequencies = jnp.exp(1j * angles)
    packed = layer(inputs, jnp.asarray([0, 2, 5], jnp.int32), frequencies)
    separate = jnp.concatenate(
        [
            layer(inputs[:2], jnp.asarray([0, 2], jnp.int32), frequencies[:2]),
            layer(inputs[2:], jnp.asarray([0, 3], jnp.int32), frequencies[2:]),
        ],
        axis=0,
    )
    assert packed.shape == inputs.shape
    assert packed.dtype == dtype
    tolerance = 2e-2 if dtype == jnp.bfloat16 else 1e-5
    np.testing.assert_allclose(
        np.asarray(packed, dtype=np.float32),
        np.asarray(separate, dtype=np.float32),
        rtol=tolerance,
        atol=tolerance,
    )
