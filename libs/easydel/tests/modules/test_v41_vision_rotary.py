"""Vision attention matches an independent full-width split-half RoPE reference (accelerator only)."""

import jax
import numpy as np
import spectrax as spx
from easydel.modules.deepseek_v41.deepseek_v41_configuration import DeepseekV41Config
from easydel.modules.deepseek_v41.modeling_deepseek_v41_vision import DeepseekV41VisionAttention, vision_cos_sin
from jax import numpy as jnp


def test_vision_attention_split_half_rotary_reference():
    config = DeepseekV41Config(
        hidden_size=16,
        vision_config={"hidden_size": 16, "num_attention_heads": 2, "num_hidden_layers": 1},
    )
    # Compare rotary/attention math, not TPU's default reduced-precision f32
    # contractions against NumPy's full-f32 contractions.
    attention = DeepseekV41VisionAttention(
        config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(7),
    )
    hidden = jnp.asarray(np.random.default_rng(0).normal(size=(1, 6, 16)), dtype=jnp.float32)
    cos, sin = vision_cos_sin(2, 3, 4, 10000.0)
    q, k, v = [array.reshape(1, 6, 2, 8) for array in np.split(np.asarray(attention.wqkv(hidden)), 3, axis=-1)]

    def rotate(array):
        first, second = np.split(array, 2, axis=-1)
        return np.concatenate([first * cos - second * sin, second * cos + first * sin], axis=-1)

    q, k = rotate(q), rotate(k)
    scores = np.einsum("bqhd,bkhd->bhqk", q, k) / np.sqrt(8)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    attended = np.einsum("bhqk,bkhd->bqhd", weights, v).reshape(1, 6, 16)
    expected = attention.wo(jnp.asarray(attended, dtype=jnp.float32))
    actual = attention(hidden, jnp.asarray(cos), jnp.asarray(sin))
    assert actual.shape == hidden.shape
    assert actual.dtype == jnp.float32
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
