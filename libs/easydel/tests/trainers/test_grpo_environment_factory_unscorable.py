# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GRPO ``environment_factory`` feedback reports unscorable steps as NaN, never as a placeholder 0.0."""

import math

import pytest
from easydel.trainers._shared.environment import ResetResult, StepResult
from easydel.trainers.group_relative_policy_optimization.grpo_trainer import GRPOTrainer


class _StubEnv:
    """Scores ``"ok"`` actions 1.0; ``"judge_fail"``/``"discard"`` are unscorable; ``"reset_fail"`` fails reset."""

    def __init__(self):
        self.closed = False
        self.stepped = False
        self._action = None

    def reset(self, seed=None):
        return ResetResult(observation="task", info={"env_error": False})

    def step(self, action):
        self.stepped = True
        if action == "judge_fail":
            return StepResult(observation="", reward=0.0, terminated=True, info={"env_error": True, "unscorable": True})
        if action == "discard":
            return StepResult(observation="", reward=0.0, terminated=True, info={"discard": True})
        return StepResult(observation="", reward=1.0 if action == "ok" else 0.0, terminated=True, info={})

    def close(self):
        self.closed = True


class _ResetFailEnv(_StubEnv):
    def reset(self, seed=None):
        return ResetResult(observation="", info={"env_error": True})


def _trainer(factory):
    trainer = object.__new__(GRPOTrainer)
    trainer.environment_factory = factory
    trainer._environment_tools = []
    return trainer


@pytest.mark.cpu_ok
def test_unscorable_steps_report_nan_rewards():
    envs = []

    def factory():
        envs.append(_StubEnv())
        return envs[-1]

    out = _trainer(factory)._run_environment_feedback(
        action_texts=["ok", "judge_fail", "wrong", "discard"], tool_calls=[None] * 4
    )
    rewards = out["environment_rewards"]
    assert rewards[0] == 1.0
    assert math.isnan(rewards[1])
    assert rewards[2] == 0.0  # a genuine zero reward stays 0.0
    assert math.isnan(rewards[3])
    assert out["environment_infos"][1]["env_error"] is True
    assert all(env.closed for env in envs)


@pytest.mark.cpu_ok
def test_failed_reset_reports_nan_without_stepping():
    envs = []

    def factory():
        envs.append(_ResetFailEnv())
        return envs[-1]

    out = _trainer(factory)._run_environment_feedback(action_texts=["ok"], tool_calls=[None])
    assert math.isnan(out["environment_rewards"][0])
    assert out["environment_terminated"] == [True]
    assert not envs[0].stepped
    assert envs[0].closed
