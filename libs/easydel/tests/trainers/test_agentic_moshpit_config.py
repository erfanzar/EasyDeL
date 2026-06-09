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

import pytest

from easydel.trainers.agentic_moshpit import AgenticMoshPitConfig


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
