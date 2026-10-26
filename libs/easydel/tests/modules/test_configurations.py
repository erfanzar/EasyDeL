"""Non-computational configuration regressions (MPT, Muse-Glimmer, OpenELM, OPT)."""

import pytest
from easydel.infra.factory import TaskType, registry
from easydel.modules.mosaic_mpt.mosaic_configuration import MptAttentionConfig, MptConfig
from easydel.modules.muse_glimmer.modeling_muse_glimmer import MuseGlimmerModel, MuseGlimmerTextModel
from easydel.modules.muse_glimmer.muse_glimmer_configuration import MuseGlimmerConfig, MuseGlimmerTextConfig
from easydel.modules.openelm.openelm_configuration import OpenELMConfig
from easydel.modules.opt.opt_configuration import OPTConfig


@pytest.mark.cpu_ok
@pytest.mark.parametrize(
    "config_class,model_class",
    [(MuseGlimmerTextConfig, MuseGlimmerTextModel), (MuseGlimmerConfig, MuseGlimmerModel)],
)
def test_muse_base_model_registry_matches_public_config(config_class, model_class):
    config = config_class()
    registration = registry.get_module_registration(TaskType.BASE_MODULE, config.model_type)
    assert registration.module is model_class
    assert registration.config is config_class


@pytest.mark.cpu_ok
@pytest.mark.parametrize("attn_type,kv_heads", [("multihead_attention", 4), ("multiquery_attention", 1)])
def test_mpt_attention_type_sets_cache_head_count(attn_type, kv_heads):
    config = MptConfig(d_model=32, n_heads=4, n_layers=1, attn_config=MptAttentionConfig(attn_type=attn_type))
    assert config.num_key_value_heads == kv_heads
    restored = MptConfig.from_dict(config.to_dict())
    assert restored.num_key_value_heads == kv_heads


@pytest.mark.cpu_ok
@pytest.mark.parametrize("groups", [None, 1, 2])
def test_openelm_optional_gqa_groups(groups):
    # Scalar multipliers keep this entirely in Python (no jnp.linspace).
    config = OpenELMConfig(
        num_transformer_layers=2,
        model_dim=32,
        head_dim=8,
        qkv_multipliers=1.0,
        ffn_multipliers=2.0,
        num_gqa_groups=groups,
    )
    assert config.num_gqa_groups == (groups or 1)
    assert config.num_query_heads == [4, 4]
    assert config.num_kv_heads == [4 // (groups or 1)] * 2


@pytest.mark.cpu_ok
@pytest.mark.parametrize("groups", [0, -1, 1.5])
def test_openelm_invalid_gqa_groups(groups):
    with pytest.raises(ValueError, match="num_gqa_groups"):
        OpenELMConfig(model_dim=32, head_dim=8, num_gqa_groups=groups)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("scan_layers", [False, True])
def test_opt_fractional_layerdrop_scan_configuration_roundtrip(scan_layers):
    config = OPTConfig(num_hidden_layers=8, layerdrop=0.5, scan_layers=scan_layers)
    restored = OPTConfig.from_dict(config.to_dict())
    assert restored.num_hidden_layers == 8
    assert restored.layerdrop == 0.5
    assert restored.scan_layers is scan_layers
