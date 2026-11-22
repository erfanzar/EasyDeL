"""Compaction configuration failures must not silently disable the feature."""

import importlib

import pytest
from easydel.trainers._online_rl.trainer import OnlineActorCriticTrainer
from easydel.trainers.agentic_rollout import CompactionController
from easydel.trainers.compaction_reinforcement_learning_trainer.compaction_rl_config import CompactionRLConfig
from easydel.trainers.compaction_reinforcement_learning_trainer.compaction_rl_trainer import CompactionRLTrainer

pytestmark = pytest.mark.cpu_ok


def test_default_compaction_controller_is_constructed(monkeypatch):
    monkeypatch.setattr(OnlineActorCriticTrainer, "__init__", lambda self, **kwargs: None)
    arguments = object.__new__(CompactionRLConfig)
    trainer = CompactionRLTrainer(arguments=arguments, model=None, reward_funcs=None)
    assert isinstance(trainer.compaction_controller, CompactionController)
    assert trainer.compaction_controller.context_budget == arguments.context_budget


def test_compaction_import_error_is_not_swallowed(monkeypatch):
    module = importlib.import_module("easydel.trainers.compaction_reinforcement_learning_trainer.compaction_rl_trainer")
    monkeypatch.setattr(OnlineActorCriticTrainer, "__init__", lambda self, **kwargs: None)

    def broken_controller(**kwargs):
        raise ImportError("controller dependency unavailable")

    monkeypatch.setattr(module, "CompactionController", broken_controller)
    with pytest.raises(ImportError, match="controller dependency unavailable"):
        CompactionRLTrainer(arguments=object.__new__(CompactionRLConfig), model=None, reward_funcs=None)
