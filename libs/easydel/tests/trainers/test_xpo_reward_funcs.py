"""Regression test: ``XPOTrainer`` accepts a one-element tuple of reward functions."""

from __future__ import annotations

from collections import UserList
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_ok


def _reward(prompts, completions, **kwargs):
    return [0.0] * len(completions)


def _build(monkeypatch, reward_funcs):
    from easydel.trainers.group_relative_policy_optimization.grpo_trainer import GRPOTrainer
    from easydel.trainers.xpo_trainer.xpo_trainer import XPOTrainer

    def fake_init(self, **kwargs):
        # Avoid initializing a real model; retain the public reward-function state.
        self.reward_funcs = kwargs["reward_funcs"]

    monkeypatch.setattr(GRPOTrainer, "__init__", fake_init)
    arguments = SimpleNamespace(beta=0.1, alpha=1e-5, loss_type="sigmoid", trainable_selector="parameters")
    return XPOTrainer(arguments, model=object(), reward_funcs=reward_funcs, processing_class=object())


@pytest.mark.parametrize("container", [tuple, list, UserList])
def test_tuple_reward_funcs_is_unpacked(monkeypatch, container):
    trainer = _build(monkeypatch, container([_reward]))
    assert trainer.reward_funcs[0](prompts=["question"], completions=["answer"]) == [0.0]


def test_single_callable_is_wrapped(monkeypatch):
    trainer = _build(monkeypatch, _reward)
    assert trainer.reward_funcs[0](prompts=["question"], completions=["answer"]) == [0.0]


def test_multiple_reward_funcs_rejected(monkeypatch):
    with pytest.raises(ValueError, match="single reward"):
        _build(monkeypatch, (_reward, _reward))
