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

import jax
import jax.numpy as jnp
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
from easydel.modules.qwen3_next.modeling_qwen3_next import (
    Qwen3NextFullAttention,
    Qwen3NextMLP,
    _qwen3_next_tp_ring_column_linear,
)
from easydel.modules.qwen3_next.qwen3_next_configuration import Qwen3NextConfig
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
            assert callable(rule["native_fuser"])


def _prefix_reform_param(rules, prefix):
    prefixed = {}
    for key, rule in rules.items():
        full_rule = rule.copy()
        full_rule["sources"] = tuple(f"{prefix}.{source}" for source in rule["sources"])
        prefixed[f"{prefix}.{key}"] = full_rule
    return prefixed


def test_dense_fused_layout_fuses_native_tuple_state_with_tp_interleave(monkeypatch):
    from easydel.layers.layouts import _reform as reform_module

    monkeypatch.setattr(reform_module, "tensor_parallel_size", lambda config, arr=None: 2)
    mlp_prefix = "model.language_model.layers.0.mlp"
    attn_prefix = "model.language_model.layers.0.self_attn"
    reform_param = {
        **_prefix_reform_param(dense_gate_up_layout(4).reform_param("gate_up_proj"), mlp_prefix),
        **_prefix_reform_param(dense_qkv_layout(4, 2).reform_param("qkv_proj"), attn_prefix),
    }

    gate = np.arange(2 * 4, dtype=np.float32).reshape(2, 4)
    up = np.arange(2 * 4, dtype=np.float32).reshape(2, 4) + 100
    q = np.arange(2 * 4, dtype=np.float32).reshape(2, 4) + 200
    k = np.arange(2 * 2, dtype=np.float32).reshape(2, 2) + 300
    v = np.arange(2 * 2, dtype=np.float32).reshape(2, 2) + 400
    state = {
        ("parameters", "model", "language_model", "layers", 0, "mlp", "gate_proj", "weight"): gate,
        ("parameters", "model", "language_model", "layers", 0, "mlp", "up_proj", "weight"): up,
        ("parameters", "model", "language_model", "layers", 0, "self_attn", "q_proj", "weight"): q,
        ("parameters", "model", "language_model", "layers", 0, "self_attn", "k_proj", "weight"): k,
        ("parameters", "model", "language_model", "layers", 0, "self_attn", "v_proj", "weight"): v,
    }

    counts = StateDictConverter.apply_native_reform_param_fusions(state, reform_param)

    gate_up_key = ("parameters", "model", "language_model", "layers", 0, "mlp", "gate_up_proj", "weight")
    qkv_key = ("parameters", "model", "language_model", "layers", 0, "self_attn", "qkv_proj", "weight")
    expected_gate_up = np.concatenate((gate[:, :2], up[:, :2], gate[:, 2:], up[:, 2:]), axis=-1)
    expected_qkv = np.concatenate((q[:, :2], k[:, :1], v[:, :1], q[:, 2:], k[:, 1:], v[:, 1:]), axis=-1)

    assert counts["dense-MLP gate/up groups"] == 1
    assert counts["dense-attention Q/K/V groups"] == 1
    assert gate_up_key in state
    assert qkv_key in state
    assert ("parameters", "model", "language_model", "layers", 0, "mlp", "gate_proj", "weight") not in state
    assert ("parameters", "model", "language_model", "layers", 0, "self_attn", "q_proj", "weight") not in state
    assert np.array_equal(np.asarray(state[gate_up_key]), expected_gate_up)
    assert np.array_equal(np.asarray(state[qkv_key]), expected_qkv)


def test_qwen3_next_separate_mlp_gate_up_matches_fused_projection():
    base_kwargs = dict(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    fused_config = Qwen3NextConfig(**base_kwargs, separate_mlp_gate_up_proj=False)
    separate_config = Qwen3NextConfig(**base_kwargs, separate_mlp_gate_up_proj=True)
    fused_mlp = Qwen3NextMLP(
        config=fused_config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(0),
    )
    separate_mlp = Qwen3NextMLP(
        config=separate_config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(0),
    )
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5, base_kwargs["hidden_size"]), dtype=jnp.float32)

    fused = fused_mlp(x)
    separate = separate_mlp(x)

    np.testing.assert_allclose(np.asarray(separate), np.asarray(fused), rtol=1e-5, atol=1e-5)


def test_qwen3_next_tp_ring_mlp_gate_up_matches_separate_projection():
    if len(jax.devices()) < 2:
        pytest.skip("requires at least two local devices for TP ring sharding")

    base_kwargs = dict(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        sharding_axis_dims=(1, 1, -1, 1, 2, 1),
    )
    separate_config = Qwen3NextConfig(**base_kwargs, separate_mlp_gate_up_proj=True)
    ring_config = Qwen3NextConfig(
        **base_kwargs,
        separate_mlp_gate_up_proj=True,
        use_tp_ring_mlp_column_matmul=True,
    )
    separate_mlp = Qwen3NextMLP(
        config=separate_config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(0),
    )
    ring_mlp = Qwen3NextMLP(
        config=ring_config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(0),
    )
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5, base_kwargs["hidden_size"]), dtype=jnp.float32)

    with separate_config.mesh:
        separate = separate_mlp(x)
    with ring_config.mesh:
        ring = ring_mlp(x)

    np.testing.assert_allclose(np.asarray(ring), np.asarray(separate), rtol=1e-5, atol=1e-5)


def test_qwen3_next_tp_ring_full_attention_qkv_matches_projection():
    if len(jax.devices()) < 2:
        pytest.skip("requires at least two local devices for TP ring sharding")

    config = Qwen3NextConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        sharding_axis_dims=(1, 1, -1, 1, 2, 1),
        use_tp_ring_full_attention_qkv_matmul=True,
    )
    attn = Qwen3NextFullAttention(
        config=config,
        dtype=jnp.float32,
        param_dtype=jnp.float32,
        rngs=spx.Rngs(0),
        layer_idx=0,
    )
    x = jax.random.normal(jax.random.PRNGKey(1), (2, 5, config.hidden_size), dtype=jnp.float32)

    with config.mesh:
        reference = attn.qkv_proj(x)
        ring = _qwen3_next_tp_ring_column_linear(
            attn.qkv_proj,
            x,
            config=config,
            precision=None,
        )

    assert ring is not None
    np.testing.assert_allclose(np.asarray(ring), np.asarray(reference), rtol=1e-5, atol=1e-5)


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


def test_qwen3_next_linear_attention_layout_fuses_native_tuple_state():
    from easydel.modules.qwen3_next.modeling_qwen3_next import Qwen3NextLinearAttentionLayout
    from easydel.modules.qwen3_next.qwen3_next_configuration import Qwen3NextConfig

    layout = Qwen3NextLinearAttentionLayout(
        key_dim=4,
        value_dim=6,
        config=Qwen3NextConfig(),
    )
    local_rules = layout.reform_param()
    reform_param = {}
    for key, rule in local_rules.items():
        full_key = f"model.layers.0.linear_attn.{key}"
        full_rule = rule.copy()
        full_rule["sources"] = tuple(f"model.layers.0.linear_attn.{source}" for source in rule["sources"])
        reform_param[full_key] = full_rule

    qkv = np.arange(3 * 14, dtype=np.float32).reshape(3, 14)
    z = np.arange(3 * 6, dtype=np.float32).reshape(3, 6) + 100
    beta = np.arange(3 * 2, dtype=np.float32).reshape(3, 2) + 200
    alpha = np.arange(3 * 2, dtype=np.float32).reshape(3, 2) + 300
    state = {
        ("parameters", "model", "layers", 0, "linear_attn", "in_proj_qkv", "weight"): qkv,
        ("parameters", "model", "layers", 0, "linear_attn", "in_proj_z", "weight"): z,
        ("parameters", "model", "layers", 0, "linear_attn", "in_proj_b", "weight"): beta,
        ("parameters", "model", "layers", 0, "linear_attn", "in_proj_a", "weight"): alpha,
    }

    counts = StateDictConverter.apply_native_reform_param_fusions(state, reform_param)

    q, k, v = np.split(qkv, [4, 8], axis=-1)
    expected_qkvz = np.concatenate((q, k, v, z), axis=-1)
    expected_ba = np.concatenate((beta, alpha), axis=-1)
    qkvz_key = ("parameters", "model", "layers", 0, "linear_attn", "in_proj_qkvz", "weight")
    ba_key = ("parameters", "model", "layers", 0, "linear_attn", "in_proj_ba", "weight")

    assert counts["linear-attention qkv/z weight groups"] == 1
    assert counts["linear-attention beta/alpha weight groups"] == 1
    assert qkvz_key in state
    assert ba_key in state
    assert ("parameters", "model", "layers", 0, "linear_attn", "in_proj_qkv", "weight") not in state
    assert ("parameters", "model", "layers", 0, "linear_attn", "in_proj_z", "weight") not in state
    assert np.array_equal(np.asarray(state[qkvz_key]), expected_qkvz)
    assert np.array_equal(np.asarray(state[ba_key]), expected_ba)


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


def test_qwen3_next_linear_attention_merged_split_falls_back_to_split_under_tp(monkeypatch):
    import easydel.modules.qwen3_next.modeling_qwen3_next as qwen3_next_modeling

    monkeypatch.setattr(qwen3_next_modeling, "_qwen3_next_tp_size", lambda config: 4)
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
        linear_attention_separate_proj=False,
        linear_attention_merged_split_proj=True,
    )

    layer = qwen3_next_modeling.Qwen3NextLinearAttention(config, rngs=spx.Rngs(0), layer_idx=0)

    assert layer.uses_merged_split_proj is False
    assert layer.uses_split_proj is True
    assert hasattr(layer, "in_proj_qkv") is True
    assert hasattr(layer, "in_proj_z") is True
    assert hasattr(layer, "in_proj_qkvz") is False
    assert hasattr(layer, "in_proj_b") is True
    assert hasattr(layer, "in_proj_a") is True
    assert "in_proj_qkvz.weight$" not in layer.reform_param
    assert "in_proj_ba.weight$" not in layer.reform_param
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
