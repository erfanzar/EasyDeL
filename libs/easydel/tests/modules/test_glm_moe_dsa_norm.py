"""GLM-MoE-DSA index-key normalization retains its checkpoint layout and fp32 math.

These tests compute arrays and are reserved for accelerator validation.
"""

import numpy as np
import spectrax as spx
from easydel.modules.glm_moe_dsa.glm_moe_dsa_configuration import GlmMoeDsaConfig
from easydel.modules.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaIndexer
from jax import numpy as jnp


def test_indexer_key_norm_preserves_parameters_and_values():
    config = GlmMoeDsaConfig(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=8,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        qk_rope_head_dim=2,
    )
    config.sharding_axis_dims = (1, 1, 1, 1, 1, 1)
    with config.mesh:
        indexer = GlmMoeDsaIndexer(config, dtype=jnp.bfloat16, param_dtype=jnp.float32, rngs=spx.Rngs(0))
        norm = indexer.k_norm
        norm.weight.value = jnp.asarray([1.0, 2.0, 0.5, -1.0])
        norm.bias.value = jnp.asarray([0.1, -0.2, 0.3, 0.4])
        inputs = jnp.asarray([[1.0, 2.0, 4.0, 8.0]], dtype=jnp.bfloat16)
        actual = norm(inputs)
    reference = np.asarray(inputs, dtype=np.float32)
    reference = (reference - reference.mean(axis=-1, keepdims=True)) / np.sqrt(
        reference.var(axis=-1, keepdims=True) + 1e-6
    )
    reference = reference * np.asarray(norm.weight.value) + np.asarray(norm.bias.value)
    assert norm.weight.value.shape == norm.bias.value.shape == (4,)
    assert norm.weight.value.dtype == norm.bias.value.dtype == jnp.float32
    assert actual.dtype == jnp.bfloat16
    np.testing.assert_allclose(np.asarray(actual, dtype=np.float32), reference, rtol=5e-3, atol=5e-3)
