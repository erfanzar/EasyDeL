"""Non-numerical Qwen3.5 configuration regression tests."""

import importlib
import sys
from types import ModuleType

import pytest
from easydel.modules.qwen3_5.qwen3_5_configuration import Qwen3_5TextConfig
from easydel.modules.qwen3_5_moe import qwen3_5_moe_configuration
from easydel.modules.qwen3_5_moe.qwen3_5_moe_configuration import Qwen3_5MoeTextConfig
from transformers import modeling_rope_utils


@pytest.mark.cpu_ok
def test_config_import_preserves_third_party_load_balancing_loss(monkeypatch):
    """Loading a config must never replace a third-party numerical function."""
    name = "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"
    hf_module = ModuleType(name)

    def reference_loss(*args, **kwargs):
        raise RuntimeError("reference shape error")

    hf_module.load_balancing_loss_func = reference_loss
    monkeypatch.setitem(sys.modules, name, hf_module)
    # Also expose it on the lazy parent so a from-import resolves this sentinel.
    hf_package = importlib.import_module("transformers.models.qwen3_5_moe")
    monkeypatch.setattr(hf_package, "modeling_qwen3_5_moe", hf_module, raising=False)
    importlib.reload(qwen3_5_moe_configuration)
    assert hf_module.load_balancing_loss_func is reference_loss
    with pytest.raises(RuntimeError, match="reference shape error"):
        hf_module.load_balancing_loss_func()


@pytest.mark.cpu_ok
@pytest.mark.parametrize("config_class", [Qwen3_5TextConfig, Qwen3_5MoeTextConfig])
@pytest.mark.parametrize("supported", [False, True])
def test_mrope_configuration_preserves_sections_and_normalizes_hf_type(monkeypatch, config_class, supported):
    validators = {"default": object()}
    if supported:
        validators["mrope"] = object()
    monkeypatch.setattr(modeling_rope_utils, "ROPE_VALIDATION_FUNCTIONS", validators, raising=False)
    scaling = {"mrope_section": [2, 1, 1], "mrope_interleaved": True}
    config = config_class(rope_scaling=scaling, linear_attention_separate_proj=False)
    expected = "mrope" if supported else "default"
    assert config.rope_scaling["rope_type"] == expected
    assert config.rope_scaling["type"] == expected
    assert config.rope_scaling["mrope_section"] == [2, 1, 1]
    assert config.rope_scaling["mrope_interleaved"] is True
    assert scaling == {"mrope_section": [2, 1, 1], "mrope_interleaved": True}


@pytest.mark.cpu_ok
@pytest.mark.parametrize("config_class", [Qwen3_5TextConfig, Qwen3_5MoeTextConfig])
def test_explicit_non_mrope_scaling_is_preserved(config_class):
    config = config_class(rope_scaling={"rope_type": "linear", "factor": 2.0}, linear_attention_separate_proj=False)
    assert config.rope_scaling["rope_type"] == "linear"
    assert config.rope_scaling["factor"] == 2.0
