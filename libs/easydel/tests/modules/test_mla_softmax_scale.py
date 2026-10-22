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

"""DeepSeek-V2/V3 MLA softmax scale comes from the shared UnifiedAttention YaRN handling (accelerator only)."""

import math

import jax
import numpy as np
import pytest
import spectrax as spx
from easydel.modules.deepseek_v2.deepseek_configuration import DeepseekV2Config
from easydel.modules.deepseek_v2.modeling_deepseek import DeepseekV2Attention
from easydel.modules.deepseek_v3.deepseek_configuration import DeepseekV3Config
from easydel.modules.deepseek_v3.modeling_deepseek import DeepseekV3Attention
from ejkernel.types import MaskInfo
from jax import numpy as jnp


def _v2_config():
    return DeepseekV2Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=4,
        num_experts_per_tok=2,
        kv_lora_rank=16,
        q_lora_rank=None,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
        first_k_dense_replace=0,
        max_position_embeddings=64,
        layer_types=["full_attention"],
    )


def _v3_config():
    return DeepseekV3Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        n_routed_experts=8,
        n_group=2,
        topk_group=1,
        num_experts_per_tok=2,
        kv_lora_rank=16,
        q_lora_rank=None,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        v_head_dim=8,
        attn_mechanism="vanilla",
        attn_dtype=jnp.float32,
        attn_softmax_dtype=jnp.float32,
        first_k_dense_replace=0,
        num_nextn_predict_layers=0,
        max_position_embeddings=64,
        layer_types=["full_attention"],
    )


def _attention(attention_cls, config):
    return attention_cls(
        config=config,
        layer_idx=0,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        precision=jax.lax.Precision.HIGHEST,
        rngs=spx.Rngs(0),
    )


def _assert_attention_output(attention, scale):
    """Independent NumPy causal attention using the layer's projected Q/K/V."""
    hidden = jnp.asarray(np.random.default_rng(2).normal(size=(4, 4, 64)), dtype=jnp.float32)
    q = np.asarray(attention.q_proj(hidden)).reshape(4, 4, 4, 16)
    compressed = attention.kv_a_proj_with_mqa(hidden)
    kv = np.asarray(attention.kv_b_proj(attention.kv_a_layernorm(compressed[..., :16]))).reshape(4, 4, 4, 16)
    keys = np.concatenate(
        [kv[..., :8], np.broadcast_to(np.asarray(compressed[..., 16:])[:, :, None], (4, 4, 4, 8))], axis=-1
    )
    values = kv[..., 8:]
    scores = np.einsum("bqhd,bkhd->bhqk", q, keys) * scale
    scores = np.where(np.tri(4, dtype=bool)[None, None], scores, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    attended = np.einsum("bhqk,bkhd->bqhd", weights, values).reshape(4, 4, 32)
    expected = attention.o_proj(jnp.asarray(attended))
    # Identity RoPE isolates scale consumption from positional-frequency math.
    rotary_width = 4 if attention.config.rope_interleave else 8
    frequencies = jnp.concatenate([jnp.ones((64, rotary_width)), jnp.zeros((64, rotary_width))], axis=-1)
    actual = attention(
        hidden,
        mask_info=MaskInfo.from_attention_mask(jnp.ones((4, 4), dtype=jnp.bool_)),
        position_ids=jnp.broadcast_to(jnp.arange(4), (4, 4)),
        mode="train",
        frequencies=frequencies,
    ).attention_output
    assert actual.shape == hidden.shape
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize(
    "config_factory,attention_cls",
    [(_v2_config, DeepseekV2Attention), (_v3_config, DeepseekV3Attention)],
)
@jax.default_matmul_precision("highest")
def test_mla_softmax_scale_without_yarn(config_factory, attention_cls):
    """The full query width (16), not the value width (8), scales the actual attention output."""
    attention = _attention(attention_cls, config_factory())
    expected = (8 + 8) ** -0.5
    assert attention.attention_performer.softmax_scale == pytest.approx(expected)
    _assert_attention_output(attention, expected)


@pytest.mark.parametrize(
    "config_factory,attention_cls",
    [(_v2_config, DeepseekV2Attention), (_v3_config, DeepseekV3Attention)],
)
@jax.default_matmul_precision("highest")
def test_mla_softmax_scale_applies_yarn_mscale(config_factory, attention_cls):
    """Use the independent YaRN formula and check its effect on attention outputs."""
    config = config_factory()
    config.rope_scaling = {
        "type": "yarn",
        "factor": 4.0,
        "mscale_all_dim": 1.0,
        "original_max_position_embeddings": 64,
    }
    attention = _attention(attention_cls, config)
    mscale = 1.0 + 0.1 * math.log(4.0)
    expected = (8 + 8) ** -0.5 * mscale * mscale
    assert attention.attention_performer.softmax_scale == pytest.approx(expected)
    _assert_attention_output(attention, expected)
