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
"""eLarge wiring for YAML-only environment settings (AgentFlow, AsyncGRPO)."""

from __future__ import annotations

import pytest
from easydel.infra.elarge.model import eLargeModel

pytestmark = pytest.mark.cpu_ok


def _elm(trainer_cfg: dict) -> eLargeModel:
    elm = object.__new__(eLargeModel)
    elm._config = {"model": {"name_or_path": "dummy-model"}, "trainer": dict(trainer_cfg)}
    elm._model = object()
    elm._state = None
    elm._tokenizer = object()
    elm.build_reward_model = lambda: None
    return elm


def _build(elm: eLargeModel) -> dict:
    captured: dict = {}

    class _CaptureTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    elm.build_trainer(trainer_class=_CaptureTrainer)
    return captured


def test_agent_flow_yaml_environment_reaches_training_arguments():
    elm = eLargeModel(
        {
            "model": {"name_or_path": "dummy-model"},
            "trainer": {
                "trainer_type": "agent_flow",
                "environment": {"name": "code_judge", "max_steps": 1},
                "environment_backend": "local",
                "environment_endpoints": ["unix:///dev/shm/fake.sock"],
                "total_batch_size": 1,
                "use_wandb": False,
            },
        }
    )
    _, arguments = elm._build_training_arguments_for_type()
    assert arguments.__class__.__name__ == "AgentFlowConfig"
    assert arguments.environment == {"name": "code_judge", "max_steps": 1}
    assert arguments.environment_backend == "local"
    assert arguments.environment_endpoints == ["unix:///dev/shm/fake.sock"]


def test_agent_flow_build_trainer_with_environment_only_in_config():
    captured = _build(_elm({"trainer_type": "agent_flow", "environment": "code_judge", "environment_backend": "local"}))
    arguments = captured["arguments"]
    assert arguments.environment == "code_judge"
    assert arguments.environment_backend == "local"
    assert "reward_funcs" not in captured


def test_environment_keys_still_stripped_for_arguments_without_those_fields():
    elm = eLargeModel(
        {
            "model": {"name_or_path": "dummy-model"},
            "trainer": {"trainer_type": "sft", "environment": "code_judge", "total_batch_size": 1},
        }
    )
    _, arguments = elm._build_training_arguments_for_type()
    assert not hasattr(arguments, "environment")


def test_async_grpo_environment_only_uses_zero_weight_placeholder_reward():
    from easydel.trainers.async_grpo_trainer import environment_reward_placeholder

    captured = _build(_elm({"trainer_type": "async_grpo", "environment": "code_judge", "environment_backend": "local"}))
    assert captured["reward_funcs"] is environment_reward_placeholder
    assert environment_reward_placeholder.weight == 0.0
    assert environment_reward_placeholder(prompts=["p", "q"], completions=["a", "b"]) == [0.0, 0.0]
    assert callable(captured["environment_factory"])
    assert captured["arguments"].environment == "code_judge"
