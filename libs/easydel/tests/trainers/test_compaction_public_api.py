"""Host-only compatibility contract for the public compaction PPO objective."""

import inspect

import pytest
from easydel.trainers.compaction_reinforcement_learning_trainer import _fn

pytestmark = pytest.mark.cpu_ok


def test_compaction_policy_loss_preserves_public_signature_and_export():
    """Shared implementation changes must not alter the scalar wrapper API."""
    assert "compaction_ppo_policy_loss" in _fn.__all__
    parameters = inspect.signature(_fn.compaction_ppo_policy_loss).parameters
    assert list(parameters) == [
        "current_logps",
        "behavior_logps",
        "advantages",
        "action_mask",
        "cliprange",
        "normalizer",
    ]
    for name in ("current_logps", "behavior_logps", "advantages", "action_mask"):
        assert parameters[name].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        assert parameters[name].default is inspect.Parameter.empty
    assert parameters["cliprange"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["cliprange"].default is inspect.Parameter.empty
    assert parameters["normalizer"].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters["normalizer"].default is None
