"""Pixtral rotary dtype preservation and call-time attention outputs (accelerator only)."""

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.pixtral.modeling_pixtral import PixtralAttention, apply_rotary_pos_emb
from easydel.modules.pixtral.pixtral_configuration import PixtralVisionConfig
from ejkernel.types import MaskInfo
from jax import numpy as jnp


@pytest.mark.parametrize("batched_tables", [False, True])
def test_rotary_preserves_activation_dtype_and_broadcasting(batched_tables):
    query = jnp.asarray(np.arange(48).reshape(2, 3, 2, 4) / 48, dtype=jnp.bfloat16)
    key = -query
    angles = np.arange(12, dtype=np.float32).reshape(3, 4) / 12
    cos, sin = np.cos(angles), np.sin(angles)
    if batched_tables:
        cos, sin = cos[None], sin[None]
    rotated_q, rotated_k = apply_rotary_pos_emb(query, key, jnp.asarray(cos), jnp.asarray(sin))
    q = np.asarray(query, dtype=np.float32)
    rotated = np.concatenate((-q[..., 2:], q[..., :2]), axis=-1)
    reference = q * np.cos(angles)[None, :, None, :] + rotated * np.sin(angles)[None, :, None, :]
    assert rotated_q.dtype == query.dtype
    assert rotated_k.dtype == key.dtype
    assert rotated_q.shape == query.shape
    np.testing.assert_allclose(np.asarray(rotated_q, dtype=np.float32), reference, atol=0.008, rtol=0.008)
    np.testing.assert_array_equal(rotated_k, -rotated_q)


def test_call_time_attention_weights_override_config():
    config = PixtralVisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        head_dim=8,
        image_size=16,
        patch_size=8,
        attn_mechanism="vanilla",
        output_attentions=False,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
    )
    config.set_model_mesh(
        jax.sharding.Mesh(np.asarray(jax.devices()[:1]).reshape((1,) * 6), ("pp", "dp", "fsdp", "ep", "tp", "sp"))
    )
    attention = PixtralAttention(config, dtype=jnp.bfloat16, param_dtype=jnp.bfloat16, rngs=spx.Rngs(0))
    inputs = jnp.ones((1, 3, 16), dtype=jnp.bfloat16)
    mask = MaskInfo.dynamic_init(inputs_embeds=inputs)
    outputs = attention(
        inputs, mask, mask.q_position_ids, output_attentions=True, frequencies=jnp.zeros((3, 8), dtype=jnp.float32)
    )
    assert outputs.attention_output.dtype == jnp.bfloat16
    assert outputs.attention_weight is not None
    assert outputs.attention_weight.shape[-2:] == (3, 3)
    np.testing.assert_allclose(np.asarray(outputs.attention_weight, dtype=np.float32).sum(-1), 1.0, atol=0.01)
