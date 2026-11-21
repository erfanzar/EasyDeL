"""Config validation of on-policy distillation trust-region mode: the sampler must draw from the student's
untruncated, unpenalised ``softmax(logits / T)`` so the acceptance ratio is the target/proposal ratio."""

import pytest
from easydel.trainers.on_policy_distillation_trainer import OnPolicyDistillationConfig

pytestmark = pytest.mark.cpu_ok


def test_trust_region_defaults_sample_from_the_untruncated_student():
    config = OnPolicyDistillationConfig(distillation_mode="trust_region")
    assert config.generation_top_k == 0
    assert config.generation_top_p == pytest.approx(1.0)
    assert config.generation_presence_penalty in (None, 0.0)
    assert config.generation_frequency_penalty in (None, 0.0)
    assert config.generation_repetition_penalty in (None, 1.0)
    assert config.generation_do_sample is True


def test_divergence_mode_keeps_the_on_policy_truncation_defaults():
    config = OnPolicyDistillationConfig()
    assert config.generation_top_k == 50
    assert config.generation_top_p == pytest.approx(0.95)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"top_k": 20},
        {"top_p": 0.9},
        {"generation_top_k": 50},
        {"generation_top_p": 0.95},
        {"presence_penalty": 0.5},
        {"frequency_penalty": 0.1},
        {"repetition_penalty": 1.1},
        {"generation_repetition_penalty": 1.2},
        {"generation_extra_kwargs": {"min_p": 0.05}},
        {"generation_config_overrides": {"top_k": 8}},
    ],
)
def test_trust_region_refuses_truncated_or_penalised_proposals(kwargs):
    with pytest.raises(ValueError, match="untruncated"):
        OnPolicyDistillationConfig(distillation_mode="trust_region", **kwargs)
    OnPolicyDistillationConfig(**kwargs)  # divergence mode accepts the same sampler settings


def test_trust_region_refuses_teacher_generated_completions():
    with pytest.raises(ValueError, match="generate_with_teacher"):
        OnPolicyDistillationConfig(distillation_mode="trust_region", generate_with_teacher=True)
    assert OnPolicyDistillationConfig(generate_with_teacher=True).generate_with_teacher is True


def test_trust_region_accepts_explicitly_neutral_sampler_settings():
    config = OnPolicyDistillationConfig(
        distillation_mode="trust_region",
        top_k=None,
        top_p=1.0,
        generation_top_k=0,
        generation_repetition_penalty=1.0,
        generation_presence_penalty=0.0,
    )
    assert config.generation_top_k == 0
    assert config.generation_top_p == pytest.approx(1.0)
