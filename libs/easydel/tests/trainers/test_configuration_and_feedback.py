"""Host-only regressions for trainer setup and feedback contracts."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from easydel.trainers.group_relative_policy_optimization.grpo_trainer import GRPOTrainer
from easydel.trainers.reward_trainer import RewardConfig
from easydel.trainers.rlvr_trainer import RLVRConfig, RLVRTrainer
from easydel.trainers.sdft_trainer import SDFTConfig, SDFTTrainer
from easydel.trainers.self_distillation_policy_optimization import SDPOConfig, SDPOTrainer
from easydel.trainers.self_distillation_policy_optimization import sdpo_trainer as sdpo_module
from easydel.trainers.supervised_fine_tuning_trainer import SFTConfig

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("config_cls", [RewardConfig, SFTConfig])
def test_unsupported_offloading_is_not_silently_accepted(config_cls):
    with pytest.raises(ValueError, match="activation_offloading"):
        config_cls(activation_offloading=True)


@pytest.mark.parametrize("config_cls", [RewardConfig, SFTConfig])
def test_dataset_process_count_must_be_positive(config_cls):
    with pytest.raises(ValueError, match="dataset_num_proc"):
        config_cls(dataset_num_proc=0)


@pytest.mark.parametrize(
    "configured_weights, expected", [(None, [1.0, 1.0]), ([0.25], [1.0, 0.25]), ([0.5, 0.25], [0.5, 0.25])]
)
def test_rlvr_config_reuse_preserves_reward_weights(monkeypatch, configured_weights, expected):
    def initialize(self, *, arguments, reward_funcs, **kwargs):
        self.arguments = arguments
        self.reward_funcs = reward_funcs

    monkeypatch.setattr(GRPOTrainer, "__init__", initialize)
    config = RLVRConfig(answer_key="answer", reward_weights=configured_weights)
    first = RLVRTrainer(config, model=None, external_reward_funcs=lambda **kwargs: [1.0])
    second = RLVRTrainer(config, model=None, external_reward_funcs=lambda **kwargs: [1.0])
    assert config.reward_weights == configured_weights
    assert first.arguments is not config
    assert first.arguments.reward_weights == second.arguments.reward_weights == expected
    assert len(first.reward_funcs) == len(expected)


def test_rlvr_no_rewarder_reports_configuration_error():
    config = RLVRConfig(answer_key=None, test_key=None)
    with pytest.raises(ValueError, match="at least one verifier"):
        RLVRTrainer(config, model=None)


def test_sdft_template_feedback_receives_real_rewards(monkeypatch):
    def initialize(self, *, arguments, feedback_func, **kwargs):
        self.arguments = arguments
        self.feedback_func = feedback_func

    monkeypatch.setattr(SDPOTrainer, "__init__", initialize)
    config = SDFTConfig(teacher_prompt_template="{prompt}{privileged_context}{feedback}")
    trainer = SDFTTrainer(
        config,
        model=None,
        feedback_func=lambda prompts, completions, rewards: [f"score={score}" for score in rewards],
    )
    assert trainer.feedback_func(["p", "p"], ["bad", "good"], [-0.5, 1.0]) == ["pscore=-0.5", "pscore=1.0"]


@pytest.mark.parametrize(
    "config_cls, extra", [(SDFTConfig, {"generate_from_teacher": True}), (SDPOConfig, {"teacher_regularization": "ema"})]
)
def test_teacher_reference_exists_with_zero_beta(monkeypatch, config_cls, extra):
    def initialize(self, **kwargs):
        self.arguments = kwargs["arguments"]
        self.model_state = SimpleNamespace(weights=[1.0])
        self.ref_state = None

    monkeypatch.setattr(GRPOTrainer, "__init__", initialize)
    monkeypatch.setattr(sdpo_module, "copy_frozen_policy", deepcopy)
    config = config_cls(beta=0.0, disable_dropout=False, **extra)
    trainer = SDPOTrainer(config, model=None, reward_funcs=lambda **kwargs: [0.0])
    assert trainer.ref_state is not None
    assert trainer.ref_state is not trainer.model_state
    trainer.model_state.weights[0] = 2.0
    assert trainer.ref_state.weights == [1.0]


def test_environment_feedback_is_replaced_only_when_group_has_solution(monkeypatch):
    # No arrays: device_get preserves these plain host reward scalars.
    trainer = object.__new__(SDPOTrainer)
    trainer.arguments = SDPOConfig(environment_feedback_only_without_solution=True)
    trainer.feedback_func = lambda **kwargs: ["first error", "unused", "third error", "fourth error"]
    separators = trainer._get_rich_feedback(
        completion_prompts=["p", "p", "q", "q"],
        completions=["bad", "solution", "bad again", "also bad"],
        rewards=[0.0, 1.0, 0.0, 0.0],
        generation_factor=2,
    )
    assert "solution" in separators[0]
    assert "first error" not in separators[0]
    assert "third error" in separators[2]
    assert "fourth error" in separators[3]
