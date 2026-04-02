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

"""Regression tests for MLA ragged-cache routing and dimensions."""

import types

import jax.numpy as jnp
import pytest
from eformer import common_types
from flax import nnx as nn

import easydel as ed
from easydel.caching import MLARaggedPagesCacheConfig, MLARaggedPagesCacheView, ParallelHybridCacheView, RaggedPagesCacheConfig
from easydel.caching.mla_ragged_page import cache as mla_ragged_cache_mod
from easydel.modules.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaAttention


def _capture_create(monkeypatch: pytest.MonkeyPatch, cache_cls: type):
    calls: list[dict] = []

    def _capture(**kwargs):
        calls.append(kwargs)
        return types.SimpleNamespace(**kwargs)

    monkeypatch.setattr(cache_cls, "create", staticmethod(_capture))
    return calls


def _make_dummy_module(model_cls: type, config):
    dummy = types.SimpleNamespace(config=config, mesh=config.mesh)
    if hasattr(model_cls, "_create_mla_ragged_page_cache_config"):
        dummy._create_mla_ragged_page_cache_config = types.MethodType(
            model_cls._create_mla_ragged_page_cache_config,
            dummy,
        )
    return dummy


def test_kimi_linear_generic_mla_cache_uses_kv_lora_rank_for_v2(monkeypatch: pytest.MonkeyPatch):
    mla_calls = _capture_create(monkeypatch, MLARaggedPagesCacheConfig)
    standard_calls = _capture_create(monkeypatch, RaggedPagesCacheConfig)

    config = ed.KimiLinearConfig(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=128,
        kv_lora_rank=96,
        qk_nope_head_dim=48,
        qk_rope_head_dim=32,
        v_head_dim=64,
    )
    config.attn_mechanism = "multi_latent_ragged_page_attention_v2"

    dummy = _make_dummy_module(ed.KimiLinearForCausalLM, config)
    ed.KimiLinearForCausalLM.create_ragged_page_cache_config(dummy, max_length=128)

    assert not standard_calls
    assert mla_calls[0]["kv_lora_rank"] == config.kv_lora_rank
    assert mla_calls[0]["kv_lora_rank"] != config.qk_nope_head_dim


def test_xerxes2_generic_mla_cache_uses_kv_lora_dim_for_v2(monkeypatch: pytest.MonkeyPatch):
    mla_calls = _capture_create(monkeypatch, MLARaggedPagesCacheConfig)
    standard_calls = _capture_create(monkeypatch, RaggedPagesCacheConfig)

    config = ed.Xerxes2Config(
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=128,
        kv_lora_dim=80,
        qk_nope_head_dim=40,
        qk_rope_head_dim=24,
        vhead_dim=64,
    )
    config.attn_mechanism = "multi_latent_ragged_page_attention_v2"

    dummy = _make_dummy_module(ed.Xerxes2ForCausalLM, config)
    ed.Xerxes2ForCausalLM.create_ragged_page_cache_config(dummy, max_length=128)

    assert not standard_calls
    assert mla_calls[0]["kv_lora_rank"] == config.kv_lora_dim
    assert mla_calls[0]["kv_lora_rank"] != config.qk_nope_head_dim


def test_glm4_moe_lite_routes_v2_to_mla_cache(monkeypatch: pytest.MonkeyPatch):
    mla_calls = _capture_create(monkeypatch, MLARaggedPagesCacheConfig)
    standard_calls = _capture_create(monkeypatch, RaggedPagesCacheConfig)

    config = ed.Glm4MoeLiteConfig(
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=128,
        kv_lora_rank=96,
        q_lora_rank=64,
        qk_nope_head_dim=48,
        qk_rope_head_dim=32,
        v_head_dim=64,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
    )
    config.attn_mechanism = "multi_latent_ragged_page_attention_v2"

    dummy = _make_dummy_module(ed.Glm4MoeLiteForCausalLM, config)
    ed.Glm4MoeLiteForCausalLM.create_ragged_page_cache_config(dummy, max_length=128)

    assert not standard_calls
    assert mla_calls[0]["kv_lora_rank"] == config.kv_lora_rank


def test_deepseek_v2_routes_v2_to_mla_cache(monkeypatch: pytest.MonkeyPatch):
    mla_calls = _capture_create(monkeypatch, MLARaggedPagesCacheConfig)
    standard_calls = _capture_create(monkeypatch, RaggedPagesCacheConfig)

    config = ed.DeepseekV2Config(
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=128,
        kv_lora_rank=96,
        q_lora_rank=64,
        qk_nope_head_dim=48,
        qk_rope_head_dim=32,
        v_head_dim=64,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
    )
    config.attn_mechanism = "multi_latent_ragged_page_attention_v2"

    dummy = _make_dummy_module(ed.DeepseekV2ForCausalLM, config)
    ed.DeepseekV2ForCausalLM.create_ragged_page_cache_config(dummy, max_length=128)

    assert not standard_calls
    assert mla_calls[0]["kv_lora_rank"] == config.kv_lora_rank


def test_deepseek_v3_routes_v2_to_mla_cache(monkeypatch: pytest.MonkeyPatch):
    mla_calls = _capture_create(monkeypatch, MLARaggedPagesCacheConfig)
    standard_calls = _capture_create(monkeypatch, RaggedPagesCacheConfig)

    config = ed.DeepseekV3Config(
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=128,
        kv_lora_rank=96,
        q_lora_rank=64,
        qk_nope_head_dim=48,
        qk_rope_head_dim=32,
        v_head_dim=64,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        first_k_dense_replace=0,
    )
    config.attn_mechanism = "multi_latent_ragged_page_attention_v2"

    dummy = _make_dummy_module(ed.DeepseekV3ForCausalLM, config)
    ed.DeepseekV3ForCausalLM.create_ragged_page_cache_config(dummy, max_length=128)

    assert not standard_calls
    assert mla_calls[0]["kv_lora_rank"] == config.kv_lora_rank


def test_glm_moe_dsa_routes_v2_to_mla_cache(monkeypatch: pytest.MonkeyPatch):
    mla_calls = _capture_create(monkeypatch, MLARaggedPagesCacheConfig)
    standard_calls = _capture_create(monkeypatch, RaggedPagesCacheConfig)

    config = ed.GlmMoeDsaConfig(
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=128,
        kv_lora_rank=96,
        q_lora_rank=64,
        qk_nope_head_dim=48,
        qk_rope_head_dim=32,
        v_head_dim=64,
        n_shared_experts=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        index_topk=32,
        index_head_dim=16,
        index_n_heads=4,
    )
    config.attn_mechanism = "multi_latent_ragged_page_attention_v2"

    dummy = _make_dummy_module(ed.GlmMoeDsaForCausalLM, config)
    ed.GlmMoeDsaForCausalLM.create_ragged_page_cache_config(dummy, max_length=128)

    assert not standard_calls
    assert mla_calls[0]["kv_lora_rank"] == config.kv_lora_rank


def test_glm_moe_dsa_v2_uses_absorbed_mla_path_for_hybrid_cache(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(mla_ragged_cache_mod, "per_device_hbm_budget_bytes", lambda *_args, **_kwargs: 1 << 26)

    config = ed.GlmMoeDsaConfig(
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
    )
    config.attn_mechanism = "multi_latent_ragged_page_attention_v2"

    captured: dict[str, object] = {}
    dummy = _make_dummy_module(ed.GlmMoeDsaForCausalLM, config)

    with config.mesh:
        cache_cfg = ed.GlmMoeDsaForCausalLM.create_ragged_page_cache_config(dummy, max_length=16, dtype=jnp.float32)
        inner_cache_view = MLARaggedPagesCacheView.init(
            cache_cfg,
            layer_index=0,
            mesh=config.mesh,
            partition_manager=config.partition_manager,
        )
        cache_view = ParallelHybridCacheView(transformer=inner_cache_view, recurrent=None)

        attn = GlmMoeDsaAttention(
            config=config,
            dtype=jnp.float32,
            param_dtype=jnp.float32,
            precision=None,
            rngs=nn.Rngs(0),
            layer_idx=0,
        )

        class _DummyIndexer:
            def __call__(self, **_kwargs):
                return types.SimpleNamespace(topk_indices=None, cached_keys=None)

        def _fake_concatenate(*, query, key, value, mask_info, cache_view, cache_metadata, **_kwargs):
            def _init_attention_bias():
                return jnp.zeros((query.shape[0], 1, query.shape[1], key.shape[1]), dtype=query.dtype)

            return key, value, mask_info, _init_attention_bias, cache_view, cache_metadata

        def _fake_forward(**kwargs):
            captured.update(kwargs)
            batch_size, seq_len, num_heads, latent_dim = kwargs["queries_nope"].shape
            return types.SimpleNamespace(
                attention_outputs=jnp.zeros((batch_size * seq_len, num_heads, latent_dim), dtype=jnp.float32),
                attention_weights=None,
                cache_view=kwargs["cache_view"],
            )

        attn.indexer = _DummyIndexer()
        attn.concatenate = _fake_concatenate
        attn.attention_performer.forward = _fake_forward

        hidden_states = jnp.ones((1, 3, config.hidden_size), dtype=jnp.float32)
        position_ids = jnp.arange(hidden_states.shape[1], dtype=jnp.int32)[None, :]
        outputs = attn.forward_mla(
            hidden_states=hidden_states,
            mask_info=None,
            position_ids=position_ids,
            mode=common_types.MODE_PREFILL,
            cache_view=cache_view,
            cache_metadata=None,
            frequencies=None,
        )

    assert outputs.attention_output.shape == (1, 3, config.hidden_size)
    assert captured["queries_nope"].shape == (1, 3, config.num_attention_heads, config.kv_lora_rank)
    assert captured["queries_pe"].shape == (1, 3, config.num_attention_heads, config.qk_rope_head_dim)
    assert captured["keys_values"].shape == (1, 3, config.kv_lora_rank)
    assert captured["keys_pe"].shape == (1, 3, config.qk_rope_head_dim)
    assert "values" not in captured
    assert captured["softmax_scale"] == pytest.approx((config.qk_nope_head_dim + config.qk_rope_head_dim) ** -0.5)
