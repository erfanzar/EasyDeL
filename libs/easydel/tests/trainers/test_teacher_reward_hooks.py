"""Host-only regressions for shared teacher-only distillation reward hooks."""

import pytest
from easydel.trainers.group_relative_policy_optimization import GRPOTrainer
from easydel.trainers.model_loading import zero_reward_func
from easydel.trainers.sdft_trainer import _zero_reward_func
from easydel.trainers.ssd_trainer import SSDConfig, SSDTrainer

pytestmark = pytest.mark.cpu_ok


@pytest.mark.parametrize("reward_func", [zero_reward_func, _zero_reward_func])
@pytest.mark.parametrize("completions", [[], ["first", "second"], ("one",)])
def test_zero_rewards_preserve_completion_count_and_legacy_api(reward_func, completions):
    assert reward_func(completions=completions, prompts=["prompt"], unused_metadata=True) == [0.0] * len(completions)


def test_ssd_installs_teacher_only_rewards_without_initializing_runtime(monkeypatch):
    def initialize(self, *, reward_funcs, **kwargs):
        self.reward_funcs = reward_funcs

    monkeypatch.setattr(GRPOTrainer, "__init__", initialize)
    trainer = SSDTrainer(SSDConfig(), model=None)
    assert trainer.reward_funcs(completions=["first", "second"], prompts=["prompt"]) == [0.0, 0.0]
    assert trainer.reward_funcs(completions=[]) == []
