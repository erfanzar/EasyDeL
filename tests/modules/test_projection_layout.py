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
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
import spectrax as spx

from easydel.layers import ColumnParallelLinear, eLoRA
from easydel.layers.layouts import (
    FusedColumnLayout,
    FusedExpertLayout,
    FusedSegment,
    dense_gate_up_layout,
    dense_qkv_layout,
)
from easydel.utils.parameters_transformation import StateDictConverter


def test_fused_column_layout_splits_contiguous_without_tp_mesh():
    layout = FusedColumnLayout(
        segments=(
            FusedSegment("a", 2, "a_proj"),
            FusedSegment("b", 3, "b_proj"),
            FusedSegment("c", 1, "c_proj"),
        )
    )
    x = np.arange(12).reshape(2, 6)

    a, b, c = layout.split(x)

    assert a.shape == (2, 2)
    assert b.shape == (2, 3)
    assert c.shape == (2, 1)
    assert np.array_equal(a, x[:, :2])
    assert np.array_equal(b, x[:, 2:5])
    assert np.array_equal(c, x[:, 5:])


def test_dense_layouts_generate_bidirectional_reform_rules():
    qkv = dense_qkv_layout(8, 4)
    qkv_rules = qkv.reform_param("qkv_proj", include_bias=True)
    gate_up = dense_gate_up_layout(16)
    gate_up_rules = gate_up.reform_param("gate_up_proj", include_bias=True)

    for rules in (qkv_rules, gate_up_rules):
        StateDictConverter.validate_reform_param_schema(rules)
        for rule in rules.values():
            assert "sources" in rule
            assert callable(rule["fuser"])
            assert callable(rule["inverse_fuser"])


def test_moe_expert_layouts_generate_bidirectional_reform_rules():
    split_rules = FusedExpertLayout(
        target_prefix="experts.gate_up_proj",
        gate_prefix="experts.gate_proj",
        up_prefix="experts.up_proj",
    ).reform_param(include_bias=True)
    fused_rules = FusedExpertLayout(
        target_prefix="experts.gate_up_proj",
        source_prefix="experts.gate_up_proj",
        source_is_fused=True,
    ).reform_param(include_bias=True)

    for rules in (split_rules, fused_rules):
        StateDictConverter.validate_reform_param_schema(rules)
        for rule in rules.values():
            assert rule["already_converted"] is True
            assert callable(rule["fuser"])
            assert callable(rule["inverse_fuser"])


def test_dense_qkv_reform_rule_round_trips_torch_tensors():
    torch = pytest.importorskip("torch")
    rules = dense_qkv_layout(8, 4).reform_param("qkv_proj")
    rule = rules["qkv_proj.weight$"]
    q = torch.arange(8 * 3, dtype=torch.float32).reshape(8, 3)
    k = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3) + 100
    v = torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3) + 200

    fused = rule["fuser"](torch, q, k, v)
    q_rt, k_rt, v_rt = rule["inverse_fuser"](torch, fused)

    assert torch.equal(q_rt, q)
    assert torch.equal(k_rt, k)
    assert torch.equal(v_rt, v)


def test_moe_expert_reform_rule_round_trips_torch_tensors():
    torch = pytest.importorskip("torch")
    rules = FusedExpertLayout().reform_param()
    rule = rules["gate_up_proj.weight$"]
    gate = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
    up = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3) + 100

    fused = rule["fuser"](torch, gate, up)
    gate_rt, up_rt = rule["inverse_fuser"](torch, fused)

    assert torch.equal(gate_rt, gate)
    assert torch.equal(up_rt, up)


def test_qwen3_next_linear_attention_layout_round_trips_torch_tensors():
    torch = pytest.importorskip("torch")
    from easydel.modules.qwen3_next.modeling_qwen3_next import Qwen3NextLinearAttentionLayout
    from easydel.modules.qwen3_next.qwen3_next_configuration import Qwen3NextConfig

    layout = Qwen3NextLinearAttentionLayout(
        key_dim=4,
        value_dim=6,
        config=Qwen3NextConfig(),
    )
    rule = layout.reform_param()["in_proj_qkvz.weight$"]
    qkv = torch.arange((4 + 4 + 6) * 3, dtype=torch.float32).reshape(14, 3)
    z = torch.arange(6 * 3, dtype=torch.float32).reshape(6, 3) + 100

    fused = rule["fuser"](torch, qkv, z)
    qkv_rt, z_rt = rule["inverse_fuser"](torch, fused)

    assert torch.equal(qkv_rt, qkv)
    assert torch.equal(z_rt, z)


@pytest.mark.parametrize(
    ("separate_proj", "merged_split_proj", "expects_fused_rules"),
    [(False, False, False), (True, False, False), (False, True, True)],
)
def test_qwen3_next_linear_attention_reform_rules_match_projection_layout(
    separate_proj,
    merged_split_proj,
    expects_fused_rules,
):
    from easydel.modules.qwen3_next.modeling_qwen3_next import Qwen3NextLinearAttention
    from easydel.modules.qwen3_next.qwen3_next_configuration import Qwen3NextConfig

    config = Qwen3NextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=3,
        linear_attention_separate_proj=separate_proj,
        linear_attention_merged_split_proj=merged_split_proj,
    )

    layer = Qwen3NextLinearAttention(config, rngs=spx.Rngs(0), layer_idx=0)

    assert hasattr(layer, "in_proj_qkv") is separate_proj
    assert hasattr(layer, "in_proj_qkvz") is not separate_proj
    assert ("in_proj_qkvz.weight$" in layer.reform_param) is expects_fused_rules
    assert ("in_proj_ba.weight$" in layer.reform_param) is expects_fused_rules
    StateDictConverter.validate_reform_param_schema(layer.reform_param)


def test_qwen3_next_linear_attention_rejects_conflicting_projection_modes():
    from easydel.modules.qwen3_next.qwen3_next_configuration import Qwen3NextConfig

    with pytest.raises(ValueError, match="mutually exclusive"):
        Qwen3NextConfig(
            vocab_size=32,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=16,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
            linear_num_key_heads=1,
            linear_num_value_heads=2,
            linear_conv_kernel_dim=3,
            linear_attention_separate_proj=True,
            linear_attention_merged_split_proj=True,
        )


def test_qwen3_5_text_config_passes_linear_attention_layout_flags():
    from easydel.modules.qwen3_5.qwen3_5_configuration import Qwen3_5TextConfig

    config = Qwen3_5TextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_num_key_heads=1,
        linear_num_value_heads=2,
        linear_conv_kernel_dim=3,
        linear_attention_separate_proj=False,
        linear_attention_merged_split_proj=True,
    )

    assert config.linear_attention_separate_proj is False
    assert config.linear_attention_merged_split_proj is True


def test_reform_param_schema_rejects_malformed_fusion_rule():
    with pytest.raises(ValueError, match=r"sources.*fuser.*inverse_fuser"):
        StateDictConverter.validate_reform_param_schema(
            {
                "qkv_proj.weight$": {
                    "sources": ("q_proj.weight", "k_proj.weight", "v_proj.weight"),
                }
            }
        )

    with pytest.raises(ValueError, match="inverse_fuser"):
        StateDictConverter.validate_reform_param_schema(
            {
                "qkv_proj.weight$": {
                    "sources": ("q_proj.weight", "k_proj.weight", "v_proj.weight"),
                    "fuser": lambda *args: args[-1],
                }
            }
        )


def test_reform_param_fuser_introspection_accepts_variadic_torch_argument():
    torch = pytest.importorskip("torch")
    rule = {
        "fuser": lambda torch, *tensors: torch.cat(tensors, dim=0),
        "inverse_fuser": lambda torch, tensor: torch.chunk(tensor, 2, dim=0),
    }
    a = torch.ones(1, 2)
    b = torch.zeros(1, 2)

    fused = StateDictConverter.fuse_reform_param_tensors(rule, [a, b])
    a_rt, b_rt = StateDictConverter.inverse_fuse_reform_param_tensor(rule, fused)

    assert torch.equal(fused, torch.cat((a, b), dim=0))
    assert torch.equal(a_rt, a)
    assert torch.equal(b_rt, b)


def test_reform_param_schema_rejects_one_way_split_rule():
    with pytest.raises(ValueError, match="inverse_spliter"):
        StateDictConverter.validate_reform_param_schema(
            {
                "packed.weight$": {
                    "splits": [
                        {
                            "name": "unpacked.weight",
                            "spliter": lambda x: x,
                        }
                    ]
                }
            }
        )


def test_fused_column_layout_rejects_indivisible_tp_segments(monkeypatch):
    from easydel.layers.layouts import _dense as dense_layout_module

    layout = FusedColumnLayout(
        segments=(
            FusedSegment("q", 3, "q_proj"),
            FusedSegment("k", 1, "k_proj"),
        )
    )
    x = np.arange(8).reshape(2, 4)
    monkeypatch.setattr(dense_layout_module, "tensor_parallel_size", lambda config, arr=None: 2)

    with pytest.raises(ValueError, match=r"segment_sizes=.*tp_size=2"):
        layout.split(x)


def test_column_parallel_layout_api_does_not_shadow_module_reform_param():
    layout = dense_gate_up_layout(8)
    layer = ColumnParallelLinear(
        4,
        layout.segment_sizes,
        use_bias=False,
        rngs=spx.Rngs(0),
        layout=layout,
    )

    assert not hasattr(layer, "reform_param")
    rules = layer.build_reform_param("gate_up_proj")
    StateDictConverter.validate_reform_param_schema(rules)


def test_lora_rejects_fused_projection_until_layout_aware_loading_exists():
    layout = dense_gate_up_layout(8)
    layer = ColumnParallelLinear(
        4,
        layout.segment_sizes,
        use_bias=False,
        rngs=spx.Rngs(0),
        layout=layout,
    )

    with pytest.raises(NotImplementedError, match="fused projection layouts"):
        eLoRA(4, rank=2, d_out=layout.out_features, base_module=layer, rngs=spx.Rngs(1))
