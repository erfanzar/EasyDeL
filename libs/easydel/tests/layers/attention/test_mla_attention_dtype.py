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

"""MLA attention compute dtype: ``mla_attn_dtype`` overrides ``attn_dtype``, and ``None`` follows it."""

from __future__ import annotations

import easydel as ed
import jax.numpy as jnp
import spectrax as spx
from easydel.modules.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaAttention


def _config(**overrides):
    return ed.GlmMoeDsaConfig(
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        kv_lora_rank=16,
        q_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        index_topk=2,
        index_head_dim=4,
        index_n_heads=2,
        sharding_axis_dims=(1, 1, 1, 1, 1, 1),
        **overrides,
    )


def _attention(config):
    with config.mesh:
        return GlmMoeDsaAttention(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=None,
            rngs=spx.Rngs(0),
            layer_idx=0,
        )


def test_unset_mla_dtype_follows_attn_dtype_changed_after_construction():
    config = _config()
    assert config.mla_attn_dtype is None
    assert config.mla_attn_softmax_dtype is None
    config.attn_dtype = jnp.float32
    attention = _attention(config)
    assert jnp.dtype(attention.attention_performer.metadata.runtime_dtype) == jnp.float32
    # The sparse-selected and pool paths compute in the dense performer's dtype.
    assert jnp.dtype(attention._attention_dtype) == jnp.float32


def test_explicit_mla_dtypes_override_attn_dtypes():
    config = _config(mla_attn_dtype=jnp.bfloat16, mla_attn_softmax_dtype=jnp.float32)
    config.attn_dtype = jnp.float32
    config.attn_softmax_dtype = jnp.bfloat16
    metadata = _attention(config).attention_performer.metadata
    assert jnp.dtype(metadata.runtime_dtype) == jnp.bfloat16
    assert jnp.dtype(metadata.runtime_softmax_dtype) == jnp.float32


def test_mla_dtype_override_leaves_the_model_config_unchanged():
    config = _config(mla_attn_dtype=jnp.bfloat16)
    config.attn_dtype = jnp.float32
    _attention(config)
    assert jnp.dtype(config.attn_dtype) == jnp.float32
    assert jnp.dtype(config.mla_attn_dtype) == jnp.bfloat16
