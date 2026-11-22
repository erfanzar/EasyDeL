"""Non-computational registry/configuration/generation-policy contracts for VisionLlama."""

from types import SimpleNamespace

import pytest
from easydel.infra.factory import TaskType, registry
from easydel.infra.loss_utils import CausalLMLossStrategy, LossConfig
from easydel.modules._base import BaseCausalLMModule
from easydel.modules.llama import (
    LlamaConfig,
    LlamaForCausalLM,
    VisionLlamaConfig,
    VisionLlamaForCausalLM,
    VisionLlamaModel,
)


@pytest.mark.cpu_ok
def test_vision_llama_has_distinct_config_and_task_registrations():
    assert registry.get_config("vision_llama") is VisionLlamaConfig
    assert registry.get_module_registration(TaskType.BASE_MODULE, "vision_llama").module is VisionLlamaModel
    assert registry.get_module_registration(TaskType.CAUSAL_LM, "vision_llama").module is VisionLlamaForCausalLM
    assert registry.get_module_registration(TaskType.CAUSAL_LM, "llama").module is LlamaForCausalLM
    assert LlamaConfig.model_type == "llama"


@pytest.mark.cpu_ok
def test_actual_vision_model_participates_in_joint_vocab_loss_planning():
    # Use the real model's inherited forward/projection protocol without
    # parameter initialization, a device mesh, or array allocation.
    model = object.__new__(VisionLlamaForCausalLM)
    config = VisionLlamaConfig(vocab_size=16, vision_vocab_size=8)
    object.__setattr__(model, "config", config)
    labels = SimpleNamespace(ndim=2, shape=(1024, 12289))
    strategy = CausalLMLossStrategy()
    joint_plan = strategy.plan_forward(module=model, labels=labels, loss_config=LossConfig(), batch={}, loss_kwargs={})
    assert joint_plan.forward_kwargs == {"apply_lm_head": False}
    # Text-only checkpoints retain their old vocabulary and smaller loss plan.
    text_model = object.__new__(LlamaForCausalLM)
    object.__setattr__(text_model, "config", LlamaConfig(vocab_size=16))
    text_plan = strategy.plan_forward(
        module=text_model, labels=labels, loss_config=LossConfig(), batch={}, loss_kwargs={}
    )
    assert text_plan.forward_kwargs == {}
    assert config.vocab_size == config.vision_token_offset == 16
    assert config.total_vocab_size == 24


@pytest.mark.cpu_ok
@pytest.mark.parametrize("sample_mode", ["all", "random", "text", "vision"])
def test_vision_token_namespace_and_modes_roundtrip(sample_mode):
    config = VisionLlamaConfig(vocab_size=12, vision_vocab_size=5, tie_vision_embeddings=True, sample_mode=sample_mode)
    restored = VisionLlamaConfig.from_dict(config.to_dict())
    assert restored.model_type == "vision_llama"
    assert restored.vocab_size == 12
    assert restored.vision_token_offset == 12
    assert restored.total_vocab_size == 17
    assert restored.tie_vision_embeddings is True
    assert restored.sample_mode == sample_mode


@pytest.mark.cpu_ok
def test_legacy_explicit_vision_config_reload_uses_vision_registration():
    legacy = VisionLlamaConfig(vocab_size=12, vision_vocab_size=5).to_dict()
    legacy["model_type"] = "llama"
    restored = VisionLlamaConfig.from_dict(legacy)
    assert restored.model_type == "vision_llama"
    assert restored.total_vocab_size == 17
    assert registry.get_module_registration(TaskType.CAUSAL_LM, restored.model_type).module is VisionLlamaForCausalLM


@pytest.mark.cpu_ok
@pytest.mark.parametrize("kwargs", [{"vision_vocab_size": 0}, {"sample_mode": "unknown"}])
def test_invalid_vision_settings_fail_at_configuration_boundary(kwargs):
    with pytest.raises(ValueError):
        VisionLlamaConfig(**kwargs)


@pytest.mark.cpu_ok
@pytest.mark.parametrize("sample_mode", ["all", "random", "text", "vision"])
def test_generate_applies_policy_without_mutating_config_or_caller_processors(monkeypatch, sample_mode):
    # Isolate the public generation-policy wrapper from numerical generation.
    # Real decoding, processors and cache reuse are covered in the hardware file.
    monkeypatch.setattr(BaseCausalLMModule, "generate", lambda self, **kwargs: kwargs)
    model = object.__new__(VisionLlamaForCausalLM)
    object.__setattr__(model, "config", VisionLlamaConfig(vocab_size=12, vision_vocab_size=5, sample_mode=sample_mode))
    caller_processors = []
    output = model.generate(input_ids="prompt", prng_key="key", trace=False, logits_processor=caller_processors)
    assert output["input_ids"] == "prompt"
    assert output["prng_key"] == "key"
    assert output["trace"] is False
    assert output.get("do_sample", False) is (sample_mode == "random")
    assert len(output["logits_processor"]) == 1
    assert caller_processors == []
    assert model.config.sample_mode == sample_mode


@pytest.mark.cpu_ok
def test_generate_mode_override_does_not_change_saved_config(monkeypatch):
    monkeypatch.setattr(BaseCausalLMModule, "generate", lambda self, **kwargs: kwargs)
    model = object.__new__(VisionLlamaForCausalLM)
    object.__setattr__(model, "config", VisionLlamaConfig(sample_mode="all"))
    output = model.generate(input_ids="prompt", sample_mode="random")
    assert output["do_sample"] is True
    assert model.config.sample_mode == "all"
