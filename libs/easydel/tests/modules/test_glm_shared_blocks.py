# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent numerical checks for shared GLM blocks (accelerator-only)."""

import numpy as np
import pytest
import spectrax as spx
from easydel.modules.glm import GlmConfig
from easydel.modules.glm.modeling_glm import GlmMLP
from easydel.modules.glm4 import Glm4Config
from easydel.modules.glm4.modeling_glm4 import Glm4MLP
from easydel.modules.glm4_moe_lite import Glm4MoeLiteConfig
from easydel.modules.glm4_moe_lite.modeling_glm4_moe_lite import Glm4MoeLiteAttention
from easydel.modules.glm4v import Glm4vTextConfig
from easydel.modules.glm4v.modeling_glm4v import Glm4vTextMLP
from ejkernel.types import MaskInfo
from jax import numpy as jnp


@pytest.mark.parametrize(
    "config_class,mlp_class", [(GlmConfig, GlmMLP), (Glm4Config, Glm4MLP), (Glm4vTextConfig, Glm4vTextMLP)]
)
def test_fused_glm_mlp_matches_numpy(config_class, mlp_class):
    config = config_class(hidden_size=16, intermediate_size=32, num_attention_heads=2, hidden_act="silu")
    config.use_fused_mlp = False
    mlp = mlp_class(config, rngs=spx.Rngs(0), dtype=jnp.float32, param_dtype=jnp.float32)
    hidden = np.linspace(-1, 1, 48, dtype=np.float32).reshape(1, 3, 16)
    projected = hidden @ np.asarray(mlp.gate_up_proj.weight.value)
    gate, up = np.split(projected, 2, axis=-1)
    expected = (gate / (1 + np.exp(-gate)) * up) @ np.asarray(mlp.down_proj.weight.value)
    output = mlp(jnp.asarray(hidden))
    assert output.shape == (1, 3, 16)
    assert output.dtype == jnp.float32
    np.testing.assert_allclose(output, expected, rtol=2e-4, atol=2e-5)


def _rms_norm(x, weight, epsilon):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + epsilon) * np.asarray(weight)


@pytest.mark.parametrize("q_lora_rank", [None, 8])
def test_lite_shared_mla_matches_numpy_with_yarn_scale(q_lora_rank):
    config = Glm4MoeLiteConfig(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        attention_bias=False,
        attention_dropout=0.0,
        max_position_embeddings=128,
        rope_scaling={"rope_type": "yarn", "factor": 4.0, "mscale_all_dim": 1.0},
        attn_mechanism="vanilla",
    )
    attention = Glm4MoeLiteAttention(config, rngs=spx.Rngs(0), layer_idx=0, dtype=jnp.float32, param_dtype=jnp.float32)
    hidden = np.linspace(-1, 1, 48, dtype=np.float32).reshape(1, 3, 16)
    if q_lora_rank is None:
        query = hidden @ np.asarray(attention.q_proj.weight.value)
    else:
        query = hidden @ np.asarray(attention.q_a_proj.weight.value)
        query = _rms_norm(query, attention.q_a_layernorm.weight.value, config.rms_norm_eps)
        query = query @ np.asarray(attention.q_b_proj.weight.value)
    query = query.reshape(1, 3, 2, 8)
    compressed = hidden @ np.asarray(attention.kv_a_proj_with_mqa.weight.value)
    latent = _rms_norm(compressed[..., :8], attention.kv_a_layernorm.weight.value, config.rms_norm_eps)
    decompressed = (latent @ np.asarray(attention.kv_b_proj.weight.value)).reshape(1, 3, 2, 8)
    keys = np.concatenate([decompressed[..., :4], np.broadcast_to(compressed[..., None, 8:], (1, 3, 2, 4))], axis=-1)
    values = decompressed[..., 4:]
    scale = 8**-0.5 * (1 + 0.1 * np.log(4.0)) ** 2
    scores = np.einsum("bqhd,bkhd->bhqk", query, keys) * scale
    scores = np.where(np.tril(np.ones((3, 3), dtype=bool)), scores, -np.inf)
    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    expected = np.einsum("bhqk,bkhd->bqhd", probabilities, values).reshape(1, 3, 8)
    expected = expected @ np.asarray(attention.o_proj.weight.value)
    hidden_jax = jnp.asarray(hidden)
    result = attention(
        hidden_states=hidden_jax,
        position_ids=jnp.arange(3)[None, :],
        mask_info=MaskInfo.dynamic_init(inputs_embeds=hidden_jax),
        mode="train",
        frequencies=None,
    )
    assert result.attention_output.shape == (1, 3, 16)
    np.testing.assert_allclose(result.attention_output, expected, rtol=2e-4, atol=2e-5)
