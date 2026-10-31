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

"""DeepSeek-V4.1 configuration contracts (pure Python, no array compute).

Expected values come from the released DeepSeek-V4.1-Flash ``config.json`` and the ownership rules of the
reference ``inference/model.py`` (``Attention.__init__``, ``SharedAttentionRuntime``, ``EngramLayout``).
"""

import json

import pytest
from easydel.modules.deepseek_v41 import DeepseekV41Config

pytestmark = pytest.mark.cpu_ok


def _small(**overrides):
    kwargs = dict(
        num_hidden_layers=6,
        num_nextn_predict_layers=1,
        compress_ratios=[0, 2, 2, 1, 1, 1, 0],
        kv_source_layer_ids=[1, 3],
        index_source_layer_ids=[1, 3, 5],
        candidate_source_layer_id=3,
        engram_layer_ids=[0, 2],
        engram_vocab_size=1000,
        engram_n_heads=2,
        dspark_target_layer_ids=[4, 5],
    )
    kwargs.update(overrides)
    return DeepseekV41Config(**kwargs)


def test_released_schedule_ownership():
    cfg = DeepseekV41Config()
    assert len(cfg.compress_ratios) == 43
    assert [cfg.layer_compress_ratio(i) for i in (0, 1, 2, 19, 20, 39, 40, 42)] == [0, 0, 2, 2, 1, 1, 0, 0]
    expected_kv = {0: None, 1: None, 2: 2, 7: 2, 8: 8, 13: 8, 14: 14, 19: 14, 20: 20, 39: 20, 40: None}
    assert {i: cfg.kv_source_layer(i) for i in expected_kv} == expected_kv
    assert [cfg.topk_source_layer(i) for i in (3, 23, 24, 27, 39)] == [2, 20, 24, 24, 36]
    # 24/28/32/36 run the indexer but read layer 20's keys; only KV sources own wk / k_norm.
    assert cfg.index_key_source_layer(24) == 20 and not cfg.owns_index_keys(24)
    assert cfg.index_key_source_layer(14) == 14 and cfg.owns_index_keys(14)
    assert [cfg.uses_candidates(i) for i in (2, 14, 20, 24, 36, 25)] == [False, False, False, True, True, False]
    assert cfg.rope_label(1) == "main" and cfg.rope_label(2) == "compress" and cfg.rope_label(41) == "main"


def test_released_engram_layout_reproduces_table_rows():
    cfg = DeepseekV41Config()
    assert cfg.engram_table_rows() == [384006168, 384016682]
    assert cfg.engram_primes()[0][0][:3] == (16000057, 16000079, 16000081)
    assert DeepseekV41Config(engram_num_embeddings=[384006168, 384016682]).engram_table_rows() == [
        384006168,
        384016682,
    ]
    with pytest.raises(ValueError, match="prime-bucket layout"):
        DeepseekV41Config(engram_num_embeddings=[384006168, 1]).engram_table_rows()


def test_rope_parameters_split_main_and_compress():
    cfg = DeepseekV41Config(
        rope_scaling={
            "rope_type": "yarn",
            "factor": 16,
            "beta_fast": 32,
            "beta_slow": 1,
            "original_max_position_embeddings": 65536,
        }
    )
    main, compress = cfg.v4_rope_params("main"), cfg.v4_rope_params("compress")
    assert main == {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.125}
    assert compress["rope_type"] == "yarn" and compress["factor"] == 16 and compress["attention_factor"] == 1.0
    assert compress["rope_theta"] == 160000.0 and compress["partial_rotary_factor"] == 0.125
    assert cfg.qk_nope_head_dim == 448


def test_composite_hf_config_is_absorbed():
    payload = {
        "model_type": "deepseek_v41",
        "architectures": ["DeepseekV41ForCausalLM"],
        "image_token_id": 129264,
        "quantization_config": {"quant_method": "fp8", "weight_block_size": [32, 32], "expert_dtype": "fp4"},
        "text_config": {
            "model_type": "deepseek_v41_text",
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "num_nextn_predict_layers": 1,
            "compress_ratios": [0, 2, 1, 1, 0],
            "kv_source_layer_ids": [1, 2],
            "index_source_layer_ids": [1, 2],
            "candidate_source_layer_id": 2,
            "engram_layer_ids": [1],
            "dspark_target_layer_ids": [3],
        },
        "vision_config": {"model_type": "deepseek_v41_vision", "hidden_size": 32, "num_hidden_layers": 2},
    }
    cfg = DeepseekV41Config(**payload)
    assert cfg.hidden_size == 64 and cfg.num_hidden_layers == 4
    assert cfg.kv_source_layer(3) == 2 and cfg.kv_source_layer(1) == 1
    assert cfg.vision_config["hidden_size"] == 32
    assert cfg.quantization_config["quant_method"] == "fp8"
    # Explicit flat kwargs win over the nested text_config.
    assert DeepseekV41Config(**{**payload, "hidden_size": 128}).hidden_size == 128


def test_small_config_roundtrip_preserves_schedule():
    cfg = _small()
    again = DeepseekV41Config.from_dict(json.loads(json.dumps(cfg.to_dict())))
    for name in ("compress_ratios", "kv_source_layer_ids", "index_source_layer_ids", "engram_layer_ids"):
        assert getattr(again, name) == getattr(cfg, name)
    assert again.v4_rope_params("compress") == cfg.v4_rope_params("compress")
    assert again.engram_table_rows() == cfg.engram_table_rows()


def test_moe_schedule_has_no_hash_layers():
    cfg = DeepseekV41Config()
    assert cfg.mlp_layer_types == ["moe"] * 43
    assert cfg.mlp_bias is False and cfg.hc_use_fused_coefficients is False
    with pytest.raises(ValueError, match="no hash MoE"):
        _small(mlp_layer_types=["hash_moe"] + ["moe"] * 6)
    with pytest.raises(ValueError, match="no hash MoE"):
        _small(mlp_layer_types=["moe"] * 6)


def test_backbone_only_ratios_are_padded_for_mtp():
    cfg = _small(compress_ratios=[0, 2, 2, 1, 1, 1])
    assert cfg.compress_ratios == [0, 2, 2, 1, 1, 1, 0]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"compress_ratios": [0, 2, 2, 1, 1]}, "compress_ratios must list"),
        ({"compress_ratios": [0, 2, 2, 1, 1, 1, 2]}, "window-only"),
        ({"kv_source_layer_ids": [1]}, "ratios must match"),
        ({"kv_source_layer_ids": [0, 3]}, "cannot produce compressed KV"),
        ({"compress_ratios": [2, 2, 2, 1, 1, 1, 0]}, "no KV source"),
        ({"index_source_layer_ids": [3, 5]}, "no index source"),
        ({"candidate_source_layer_id": 4}, "candidate_source_layer_id"),
        ({"engram_layer_ids": [6]}, "engram_layer_ids"),
        ({"engram_num_embeddings": [1]}, "one entry per engram layer"),
        ({"index_source_layer_ids": [1, 5]}, "different compressed caches"),
        (
            {
                "compress_ratios": [0, 2, 2, 2, 2, 2, 0],
                "index_source_layer_ids": [1, 4],
                "candidate_source_layer_id": 1,
            },
            "must itself be an index source",
        ),
        (
            {
                "compress_ratios": [0, 2, 2, 1, 1, 1, 0],
                "index_source_layer_ids": [1, 3, 5],
                "candidate_source_layer_id": 1,
            },
            "different compressed caches",
        ),
    ],
)
def test_invalid_schedules_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        _small(**overrides)
