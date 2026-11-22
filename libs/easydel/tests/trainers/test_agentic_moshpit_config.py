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

from __future__ import annotations

from dataclasses import dataclass

import easydel.trainers.agentic_moshpit as agentic_moshpit_pkg
import pytest
from easydel.trainers.agentic_moshpit import AgenticMoshPitConfig, compute_advantages_agentic

pytestmark = pytest.mark.cpu_ok


def test_agentic_reward_mode_selects_matching_default_estimator():
    assert AgenticMoshPitConfig(reward_mode="episode").advantage_estimator == "grpo"
    assert AgenticMoshPitConfig(reward_mode="step").advantage_estimator == "step_reinforce"
    assert AgenticMoshPitConfig(reward_mode="gigpo").advantage_estimator == "gigpo"


def test_agentic_explicit_advantage_estimator_takes_precedence():
    cfg = AgenticMoshPitConfig(reward_mode="step", advantage_estimator="agentic_reinforce")

    assert cfg.advantage_estimator == "agentic_reinforce"


def test_agentic_rejects_invalid_reward_mode():
    with pytest.raises(ValueError, match="reward_mode"):
        AgenticMoshPitConfig(reward_mode="bad")


def test_agentic_exports_compute_advantages_agentic():
    assert "compute_advantages_agentic" in agentic_moshpit_pkg.__all__
    assert agentic_moshpit_pkg.compute_advantages_agentic is compute_advantages_agentic


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"max_steps": 0}, "max_steps"),
        ({"max_steps": -3}, "max_steps"),
        ({"max_steps": False}, "max_steps"),
        ({"group_size": 0}, "group_size"),
        ({"group_size": -1}, "group_size"),
        ({"group_size": True}, "group_size"),
        ({"num_env_groups": 0}, "num_env_groups"),
        ({"num_env_groups": -2}, "num_env_groups"),
        ({"step_reward_gamma": -0.01}, "step_reward_gamma"),
        ({"step_reward_gamma": 1.01}, "step_reward_gamma"),
        ({"step_reward_gamma": float("nan")}, "step_reward_gamma"),
        ({"step_reward_gamma": float("inf")}, "step_reward_gamma"),
        ({"step_reward_gamma": True}, "step_reward_gamma"),
    ],
)
def test_agentic_config_validates_bounds(kwargs, match):
    with pytest.raises(ValueError, match=match):
        AgenticMoshPitConfig(**kwargs)


def test_agentic_config_accepts_boundary_gamma_values():
    assert AgenticMoshPitConfig(step_reward_gamma=0.0).step_reward_gamma == 0.0
    assert AgenticMoshPitConfig(step_reward_gamma=1.0).step_reward_gamma == 1.0


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"num_iterations": 2}, "num_iterations"),
        ({"steps_per_generation": 2}, "steps_per_generation"),
        ({"mask_truncated_completions": True}, "mask_truncated_completions"),
        ({"off_policy_mask_threshold": 0.5}, "off_policy_mask_threshold"),
    ],
)
def test_agentic_config_rejects_unsupported_grpo_knobs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        AgenticMoshPitConfig(**kwargs)


@pytest.mark.parametrize("divergence_type", ["topk_tv", "topk_kl"])
def test_agentic_config_rejects_dppo_topk_divergence(divergence_type):
    @dataclass
    class _DPPOAgenticConfig(AgenticMoshPitConfig):
        divergence_type: str = ""

    with pytest.raises(ValueError, match="top-k"):
        _DPPOAgenticConfig(loss_type="dppo", divergence_type=divergence_type)
