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

"""GLM family config and cache-config plumbing regressions (no array computation)."""

import types

import pytest
from easydel.caching import MLARaggedPagesCacheConfig, RaggedPagesCacheConfig
from easydel.infra.mixins.generation import EasyGenerationMixin
from easydel.layers.attention import AttentionMechanisms
from easydel.layers.rotary._scaling import rope_scaling_from_rope_parameters
from easydel.modules.glm4_moe_lite import Glm4MoeLiteConfig, Glm4MoeLiteForCausalLM
from easydel.modules.glm4_moe_lite.glm4_moe_lite_configuration import (
    rope_scaling_from_rope_parameters as lite_rope_scaling_from_rope_parameters,
)
from easydel.modules.glm4v.glm4v_configuration import GLM4V_DEFAULT_ROPE_SCALING, Glm4vTextConfig
from easydel.modules.glm4v.glm4v_configuration import (
    rope_scaling_from_rope_parameters as glm4v_rope_scaling_from_rope_parameters,
)
from easydel.modules.glm4v_moe import Glm4vMoeTextConfig
from easydel.modules.glm4v_moe.glm4v_moe_configuration import (
    rope_scaling_from_rope_parameters as moe_rope_scaling_from_rope_parameters,
)


@pytest.mark.cpu_ok
def test_rope_scaling_family_reexports_are_neutral_compatibility_helper():
    assert glm4v_rope_scaling_from_rope_parameters is rope_scaling_from_rope_parameters
    assert lite_rope_scaling_from_rope_parameters is rope_scaling_from_rope_parameters
    assert moe_rope_scaling_from_rope_parameters is rope_scaling_from_rope_parameters


@pytest.mark.cpu_ok
def test_rope_scaling_explicit_wins_and_aliases_type():
    scaling = {"type": "linear", "factor": 2.0}
    out = rope_scaling_from_rope_parameters({"rope_type": "yarn"}, scaling)
    assert out == {"type": "linear", "factor": 2.0, "rope_type": "linear"}
    assert out is not scaling
    assert scaling == {"type": "linear", "factor": 2.0}


@pytest.mark.cpu_ok
@pytest.mark.parametrize("scaling", [{}, {"rope_type": "yarn"}, {"type": "linear", "rope_type": "yarn"}])
def test_rope_scaling_explicit_mapping_preserves_identity_and_precedence(scaling):
    out = rope_scaling_from_rope_parameters(
        {"rope_type": "dynamic", "factor": 8.0}, scaling, default=GLM4V_DEFAULT_ROPE_SCALING
    )
    assert out is scaling
    assert "factor" not in out
    assert "mrope_section" not in out
    if "type" in scaling:
        assert out["rope_type"] == "yarn"


@pytest.mark.cpu_ok
def test_rope_scaling_preserves_all_supported_parameter_fields_and_value_identity():
    fields = {
        "factor": 4.0,
        "original_max_position_embeddings": 8192,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "short_factor": [1.0, 2.0],
        "long_factor": [4.0, 8.0],
        "beta_fast": 32.0,
        "beta_slow": 1.0,
        "extrapolation_factor": 1.0,
        "attn_factor": 1.1,
        "mscale": 0.5,
        "mscale_all_dim": 0.75,
        "mrope_section": [8, 12, 12],
        "mrope_interleaved": True,
    }
    params = {"rope_type": "yarn", **fields, "ignored": object(), "rope_theta": 10000.0}
    original = dict(params)
    out = rope_scaling_from_rope_parameters(params, None)
    assert out == {"rope_type": "yarn", **fields}
    assert out is not params
    assert params == original
    for key, value in fields.items():
        assert out[key] is value


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    "params,expected",
    [
        ({"type": "linear", "factor": 2.0}, {"rope_type": "default", "factor": 2.0}),
        ({"type": "linear", "rope_type": "yarn"}, {"rope_type": "yarn"}),
    ],
)
def test_rope_scaling_parameter_type_retains_legacy_behavior(params, expected):
    original = dict(params)
    assert rope_scaling_from_rope_parameters(params, None) == expected
    assert params == original


@pytest.mark.cpu_ok
def test_rope_scaling_absent_and_empty_inputs_are_distinct():
    default = {"rope_type": "yarn", "factor": 8.0}
    assert rope_scaling_from_rope_parameters(None, None) is None
    assert rope_scaling_from_rope_parameters(None, None, default=default) == default
    assert rope_scaling_from_rope_parameters({}, None, default=default) == {"rope_type": "default"}
    assert rope_scaling_from_rope_parameters(None, {}, default=default) == {}
    empty_default = {}
    out = rope_scaling_from_rope_parameters(None, None, default=empty_default)
    assert out == {}
    assert out is not empty_default


@pytest.mark.cpu_ok
def test_rope_scaling_default_retains_shallow_copy_and_list_copy_semantics():
    nested = {"preserved": [1]}
    list_item = {"also_preserved": True}
    default = {"rope_type": "default", "custom_mapping": nested, "custom_list": [list_item]}
    out = rope_scaling_from_rope_parameters(None, None, default=default)
    assert out == default
    assert out is not default
    assert out["custom_mapping"] is nested
    assert out["custom_list"] is not default["custom_list"]
    assert out["custom_list"][0] is list_item
    out["custom_list"].append(99)
    assert default["custom_list"] == [list_item]


@pytest.mark.cpu_ok
def test_rope_scaling_copies_known_keys_from_rope_parameters():
    params = {"rope_type": "yarn", "factor": 4.0, "mrope_section": [8, 12, 12], "ignored": 1}
    out = rope_scaling_from_rope_parameters(params, None)
    assert out == {"rope_type": "yarn", "factor": 4.0, "mrope_section": [8, 12, 12]}


@pytest.mark.cpu_ok
def test_rope_scaling_default_is_none_or_independent_copy():
    assert rope_scaling_from_rope_parameters(None, None) is None
    out = rope_scaling_from_rope_parameters(None, None, default=GLM4V_DEFAULT_ROPE_SCALING)
    assert out == GLM4V_DEFAULT_ROPE_SCALING
    out["mrope_section"].append(99)
    assert GLM4V_DEFAULT_ROPE_SCALING["mrope_section"] == [8, 12, 12]


@pytest.mark.cpu_ok
def test_glm4v_family_configs_keep_mrope_default_and_lite_keeps_none():
    for cfg in (Glm4vTextConfig(), Glm4vMoeTextConfig()):
        # The config may add extra keys (rope_theta, partial_rotary_factor); the mRoPE preset must be intact.
        for key, value in GLM4V_DEFAULT_ROPE_SCALING.items():
            assert cfg.rope_scaling[key] == value
    # The lite config has no mRoPE preset: it must not gain ``mrope_section``.
    lite = Glm4MoeLiteConfig(rope_parameters=None, rope_scaling=None)
    assert "mrope_section" not in (lite.rope_scaling or {})


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    "mechanism",
    [AttentionMechanisms.MULTI_LATENT_RAGGED_PAGE_ATTENTION_V1, "multi_latent_ragged_page_attention_v2"],
)
def test_lite_mla_cache_config_uses_shared_builder(monkeypatch, mechanism):
    """The public builder preserves the latent cache geometry for enum and string mechanisms."""

    def fake_create(**kwargs):
        return types.SimpleNamespace(**kwargs)

    monkeypatch.setattr(MLARaggedPagesCacheConfig, "create", staticmethod(fake_create))

    text_config = types.SimpleNamespace(
        attn_mechanism=mechanism,
        kvdtype="bfloat16",
        mesh="mesh",
        runtime_sharding_resolver="resolver",
        num_hidden_layers=3,
        num_attention_heads=4,
        kv_lora_rank=32,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=16,
    )
    fake_self = types.SimpleNamespace(
        config=types.SimpleNamespace(
            get_text_config=lambda: text_config,
            qk_nope_head_dim=16,
            qk_rope_head_dim=16,
            v_head_dim=16,
        ),
        mesh="mesh",
    )
    fake_self._create_mla_ragged_page_cache_config = types.MethodType(
        EasyGenerationMixin._create_mla_ragged_page_cache_config, fake_self
    )

    result = Glm4MoeLiteForCausalLM.create_ragged_page_cache_config(fake_self, 256, page_size=64)

    assert result.kv_lora_rank == 32
    assert result.qk_rope_head_dim == 16
    assert result.num_hidden_layers == 3
    assert result.num_kv_heads == 4
    assert result.max_model_length == 256
    assert result.page_size == 64


@pytest.mark.cpu_ok
@pytest.mark.parametrize("mechanism,version", [("ragged_page_attention_v2", "v2"), ("ragged_page_attention_v3", "v3")])
def test_lite_standard_cache_preserves_distinct_key_value_widths(monkeypatch, mechanism, version):
    """Decompressed caches keep MLA key/value geometry, page limits and stage sizing."""
    monkeypatch.setattr(RaggedPagesCacheConfig, "create", staticmethod(lambda **kwargs: types.SimpleNamespace(**kwargs)))
    config = types.SimpleNamespace(
        attn_mechanism=mechanism,
        kvdtype="bfloat16",
        mesh="mesh",
        runtime_sharding_resolver="resolver",
        num_hidden_layers=5,
        num_attention_heads=4,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=12,
    )
    config.get_text_config = lambda: config
    model = types.SimpleNamespace(
        config=config,
        _standard_ragged_layer_indices_for_cache_budget=lambda: list(range(5)),
        _pp_stage_cache_layer_count=lambda indices, total_layers: 3,
    )
    model._create_standard_ragged_page_cache_config = types.MethodType(
        Glm4MoeLiteForCausalLM._create_standard_ragged_page_cache_config, model
    )
    result = Glm4MoeLiteForCausalLM.create_ragged_page_cache_config(
        model, 256, page_size=64, max_cache_tokens=512, hbm_utilization=0.7
    )
    assert result.k_headdim == 24
    assert result.v_headdim == 12
    assert result.num_kv_heads == 4
    assert result.num_hidden_layers == 3
    assert result.max_cache_tokens == 512
    assert result.page_size == 64
    assert result.hbm_utilization == 0.7
    assert result.version == version


@pytest.mark.cpu_ok
def test_lite_standard_cache_accepts_config_only_caller(monkeypatch):
    """The historical public builder needs only config/mesh, not bound helpers."""
    monkeypatch.setattr(RaggedPagesCacheConfig, "create", staticmethod(lambda **kwargs: types.SimpleNamespace(**kwargs)))
    config = types.SimpleNamespace(
        attn_mechanism="ragged_page_attention_v2",
        kvdtype="bfloat16",
        mesh="mesh",
        runtime_sharding_resolver="resolver",
        num_hidden_layers=5,
        num_attention_heads=4,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=12,
    )
    config.get_text_config = lambda: config
    dummy = types.SimpleNamespace(config=config, mesh="mesh")
    result = Glm4MoeLiteForCausalLM.create_ragged_page_cache_config(dummy, 256, page_size=64)
    assert (result.k_headdim, result.v_headdim, result.num_kv_heads) == (24, 12, 4)
    assert (result.num_hidden_layers, result.max_model_length, result.page_size) == (5, 256, 64)
