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

"""RoPE fields of transformers-v5-saved configs reach EasyDeL's rotary builders.

transformers v5 serializes ``rope_theta`` (and for some families
``partial_rotary_factor`` / per-layer scaling) only inside ``rope_parameters``.
"""

import copy
import json

import easydel as ed
import pytest
import transformers

LLAMA3_SCALING = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


def _hf_v5_dict(hf_cls_name: str, **kwargs) -> dict:
    """What ``save_pretrained`` writes with transformers v5 (``rope_theta`` only nested)."""
    config = getattr(transformers, hf_cls_name)(**kwargs)
    payload = json.loads(config.to_json_string(use_diff=False))
    payload.pop("transformers_version", None)
    return payload


def test_llama3_v5_config_keeps_rope_theta():
    payload = _hf_v5_dict(
        "LlamaConfig", rope_theta=500000.0, rope_scaling=dict(LLAMA3_SCALING), max_position_embeddings=131072
    )
    assert "rope_theta" not in payload
    config = ed.LlamaConfig.from_dict(copy.deepcopy(payload))
    assert config.rope_theta == 500000.0
    assert config.rope_parameters["rope_theta"] == 500000.0
    rope = config._get_rope_config()
    assert rope.rope_type == "llama3"
    assert rope.factor == 8.0

    # Direct construction follows HF priority (``rope_parameters`` wins) too.
    payload.pop("model_type", None)
    assert ed.LlamaConfig(**copy.deepcopy(payload)).rope_theta == 500000.0


@pytest.mark.parametrize(
    ("hf_cls_name", "ed_cls_name", "kwargs"),
    [
        ("Olmo2Config", "Olmo2Config", {"rope_theta": 500000.0}),
        ("SeedOssConfig", "SeedOssConfig", {"rope_theta": 10000000.0}),
        ("SmolLM3Config", "SmolLM3Config", {"rope_theta": 5000000.0}),
        ("Qwen3NextConfig", "Qwen3NextConfig", {"rope_theta": 10000000.0, "partial_rotary_factor": 0.5}),
        ("FalconConfig", "FalconConfig", {"rope_theta": 500042.0}),
        ("CohereConfig", "CohereConfig", {"rope_theta": 8000000.0}),
    ],
)
def test_v5_rope_theta_reaches_easydel(hf_cls_name, ed_cls_name, kwargs):
    config = getattr(ed, ed_cls_name).from_dict(_hf_v5_dict(hf_cls_name, **kwargs))
    assert config.rope_theta == kwargs["rope_theta"]
    if "partial_rotary_factor" in kwargs:
        assert config.partial_rotary_factor == kwargs["partial_rotary_factor"]


def test_late_rope_theta_override_stays_consistent_after_reload():
    payload = _hf_v5_dict("LlamaConfig", rope_theta=500000.0)
    config = ed.LlamaConfig.from_dict(payload, rope_theta=1234.0)
    assert config.rope_theta == 1234.0
    assert config.rope_parameters["rope_theta"] == 1234.0
    saved = json.loads(config.to_json_string())
    assert "_rope_fields_synced" not in saved
    assert ed.LlamaConfig.from_dict(saved).rope_theta == 1234.0


def test_phi4_mini_partial_rotary_factor_only_in_rope_parameters():
    rope_parameters = {
        "rope_type": "longrope",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.75,
        "original_max_position_embeddings": 4096,
        "short_factor": [1.0] * 48,
        "long_factor": [2.0] * 48,
    }
    config = ed.Phi3Config.from_dict(
        {
            "hidden_size": 3072,
            "num_attention_heads": 24,
            "max_position_embeddings": 131072,
            "original_max_position_embeddings": 4096,
            "rope_parameters": rope_parameters,
        }
    )
    assert config.partial_rotary_factor == 0.75
    assert config._get_rope_config().rope_type == "longrope"


@pytest.mark.parametrize("legacy_type", ["su", "yarn"])
def test_phi3_legacy_scaling_names_dispatch_to_longrope(legacy_type):
    config = ed.Phi3Config(rope_scaling={"type": legacy_type, "short_factor": [1.0] * 48, "long_factor": [1.0] * 48})
    assert config._get_rope_config().rope_type == "longrope"


def test_gpt_neox_v5_rotary_fields():
    config = ed.GPTNeoXConfig.from_dict(_hf_v5_dict("GPTNeoXConfig", rotary_pct=0.5, rotary_emb_base=20000))
    assert config.rotary_pct == 0.5
    assert config.rotary_emb_base == 20000
    # GPT-NeoX sizes partial RoPE via ``rotary_pct``; it must not be applied twice.
    assert getattr(config, "partial_rotary_factor", 1.0) == 1.0


def test_gemma3_global_layers_keep_linear_scaling():
    config = ed.Gemma3TextConfig(rope_scaling={"rope_type": "linear", "factor": 8.0})
    rope = config._get_rope_config()
    assert rope.rope_type == "linear"
    assert rope.factor == 8.0
    assert config.rope_parameters["sliding_attention"]["rope_type"] == "default"
    assert "factor" not in config.rope_parameters["sliding_attention"]


def test_gemma3_v5_nested_rope_parameters_roundtrip():
    payload = _hf_v5_dict(
        "Gemma3TextConfig",
        rope_theta=1000000.0,
        rope_local_base_freq=12345.0,
        rope_scaling={"rope_type": "linear", "factor": 8.0},
    )
    config = ed.Gemma3TextConfig.from_dict(copy.deepcopy(payload))
    assert config.rope_theta == 1000000.0
    assert config.rope_local_base_freq == 12345.0
    assert config._get_rope_config().factor == 8.0
    restored = ed.Gemma3TextConfig.from_dict(json.loads(config.to_json_string()))
    assert restored.rope_local_base_freq == 12345.0
    assert restored._get_rope_config().rope_type == "linear"
    assert restored._get_rope_config().factor == 8.0


def test_exaone4_accepts_official_llama3_scaling():
    config = ed.Exaone4Config(rope_theta=1000000.0, rope_scaling=dict(LLAMA3_SCALING), max_position_embeddings=131072)
    rope = config._get_rope_config()
    assert rope.rope_type == "llama3"
    assert rope.low_freq_factor == 1.0
    assert rope.high_freq_factor == 4.0
    assert rope.original_max_position_embeddings == 8192


def test_smollm3_accepts_yarn_scaling():
    config = ed.SmolLM3Config(rope_scaling={"type": "yarn", "factor": 2.0, "original_max_position_embeddings": 65536})
    rope = config._get_rope_config()
    assert rope.rope_type == "yarn"
    assert rope.original_max_position_embeddings == 65536


def test_yarn_attention_factor_and_truncate_reach_rope_config():
    config = ed.LlamaConfig(
        max_position_embeddings=131072,
        rope_scaling={
            "rope_type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 4096,
            "attention_factor": 1.25,
            "truncate": False,
        },
    )
    scaling = config._get_rope_config().to_dict()
    assert scaling["attention_factor"] == 1.25
    assert scaling["truncate"] is False
