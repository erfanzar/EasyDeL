"""Kimi-Linear, HunYuan-VL, InternLM2 and InternVL config and registry contracts (no numerics)."""

import pytest
from easydel.infra.factory import TaskType, registry
from easydel.modules.hunyuan_vl.hunyuan_vl_configuration import (
    HunYuanVLConfig,
    HunYuanVLTextConfig,
    HunYuanVLVisionConfig,
)
from easydel.modules.hunyuan_vl.modeling_hunyuan_vl import (
    HunYuanVLModel,
    HunYuanVLTextModel,
    HunYuanVLVisionModel,
)
from easydel.modules.internlm2 import InternLM2Config
from easydel.modules.internvl import InternVLConfig
from easydel.modules.kimi_linear import KimiLinearConfig


@pytest.mark.cpu_ok
def test_kimi_router_output_defaults_and_serialization():
    assert KimiLinearConfig().output_router_logits is False
    config = KimiLinearConfig(output_router_logits=True)
    assert config.output_router_logits is True
    assert KimiLinearConfig.from_dict(config.to_dict()).output_router_logits is True


@pytest.mark.cpu_ok
@pytest.mark.parametrize("explicit_legacy", [False, True])
def test_kimi_explicit_attention_schedule_matches_all_public_views(explicit_legacy):
    schedule = ["kda_linear_attention", "full_attention", "kda_linear_attention"]
    kwargs = {"linear_attn_config": {"kda_layers": [1, 3], "full_attn_layers": [2]}} if explicit_legacy else {}
    config = KimiLinearConfig(num_hidden_layers=3, layer_types=schedule, **kwargs)
    for actual in (config, KimiLinearConfig.from_dict(config.to_dict())):
        assert actual.layer_types == schedule
        assert actual.get_layer_types() == tuple(schedule)
        assert actual.is_linear_attn
        assert [actual.is_kda_layer(i) for i in range(3)] == [True, False, True]
        assert [actual.is_mla_layer(i) for i in range(3)] == [False, True, False]
        assert actual.linear_attn_config["kda_layers"] == [1, 3]
        assert actual.linear_attn_config["full_attn_layers"] == [2]


@pytest.mark.cpu_ok
def test_kimi_conflicting_attention_schedules_are_rejected():
    with pytest.raises(ValueError, match="different attention schedules"):
        KimiLinearConfig(
            num_hidden_layers=2,
            layer_types=["kda_linear_attention", "full_attention"],
            linear_attn_config={"kda_layers": [], "full_attn_layers": [1, 2]},
        )
    config = KimiLinearConfig(num_hidden_layers=2)
    assert not config.is_linear_attn
    assert all(config.is_mla_layer(i) for i in range(2))
    for outside in (-1, 2):
        assert not config.is_kda_layer(outside)
        assert not config.is_mla_layer(outside)


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    ("config_class", "task", "module"),
    [
        (HunYuanVLTextConfig, TaskType.BASE_MODULE, HunYuanVLTextModel),
        (HunYuanVLVisionConfig, TaskType.BASE_VISION, HunYuanVLVisionModel),
        (HunYuanVLConfig, TaskType.BASE_MODULE, HunYuanVLModel),
    ],
)
def test_hunyuan_standalone_and_composite_registry_resolution(config_class, task, module):
    registration = registry.get_module_registration(task, config_class.model_type)
    assert registration.module is module
    assert registration.config is config_class
    assert module._model_type == config_class.model_type
    # The vision-tower key remains a public registry alias.
    assert registry.get_module_registration(TaskType.BASE_VISION, "hunyuan_vl").module is HunYuanVLVisionModel


@pytest.mark.cpu_ok
@pytest.mark.parametrize("rope_type", ["linear", "dynamic", "yarn"])
def test_hunyuan_rope_scaling_parameters_survive_round_trip(rope_type):
    scaling = {"rope_type": rope_type, "factor": 4.0}
    if rope_type == "yarn":
        scaling.update(beta_fast=16.0, beta_slow=2.0, original_max_position_embeddings=128)
    config = HunYuanVLTextConfig(hidden_size=32, num_attention_heads=2, rope_scaling=scaling)
    for actual in (config, HunYuanVLTextConfig.from_dict(config.to_dict())):
        for key, value in scaling.items():
            assert actual.rope_scaling[key] == value
        rope = actual._get_rope_config()
        assert rope.rope_type == rope_type
        assert rope.factor == 4.0
    assert scaling["factor"] == 4.0


@pytest.mark.cpu_ok
def test_hunyuan_legacy_dynamic_alpha_folds_only_its_scaling_factor():
    config = HunYuanVLTextConfig(
        hidden_size=32,
        num_attention_heads=2,
        rope_theta=10000.0,
        rope_scaling={"rope_type": "dynamic", "alpha": 2.0, "factor": 4.0},
    )
    assert config.rope_scaling["rope_type"] == "default"
    assert "factor" not in config.rope_scaling
    assert config.rope_theta == pytest.approx(10000.0 * 2.0 ** (16.0 / 14.0))


@pytest.mark.cpu_ok
def test_internlm2_attention_dropout_is_a_serialized_config_field():
    assert InternLM2Config().attention_dropout == 0.0
    config = InternLM2Config(attention_dropout=0.25)
    assert InternLM2Config.from_dict(config.to_dict()).attention_dropout == 0.25


@pytest.mark.cpu_ok
def test_internvl_multilayer_selection_cannot_be_empty():
    with pytest.raises(ValueError, match="at least one"):
        InternVLConfig(vision_feature_layer=[])
