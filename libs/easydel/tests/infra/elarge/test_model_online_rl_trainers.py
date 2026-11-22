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
from easydel.infra.elarge.model import eLargeModel


@pytest.mark.parametrize(
    ("trainer_type", "canonical", "arguments_name"),
    [
        ("single-rollout-asynchronous-optimization", "sao", "SAOConfig"),
        ("compaction-rl", "compaction_rl", "CompactionRLConfig"),
    ],
)
def test_build_online_actor_critic_arguments_from_elarge_config(trainer_type, canonical, arguments_name):
    elm = eLargeModel(
        {
            "model": {"name_or_path": "dummy-model"},
            "trainer": {"trainer_type": trainer_type, "total_batch_size": 1, "use_wandb": False},
        }
    )

    trainer_config, arguments = elm._build_training_arguments_for_type()

    assert trainer_config["trainer_type"] == canonical
    assert arguments.__class__.__name__ == arguments_name
    assert arguments.num_return_sequences == 1
    assert arguments.num_generations == 1


def _capture_build(trainer_type: str, **runtime_kwargs):
    captured = {}

    class _CaptureTrainer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    elm = object.__new__(eLargeModel)
    elm._config = {
        "model": {"name_or_path": "dummy-model"},
        "trainer": {"trainer_type": trainer_type},
    }
    elm._model = object()
    elm._state = None
    elm._tokenizer = object()
    elm.build_training_arguments = lambda *args, **kwargs: "arguments"

    reward_func = object()
    train_dataset = object()
    eval_dataset = object()
    trainer = elm.build_trainer(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=reward_func,
        trainer_class=_CaptureTrainer,
        **runtime_kwargs,
    )
    assert isinstance(trainer, _CaptureTrainer)
    return captured, elm, reward_func, train_dataset, eval_dataset


@pytest.mark.parametrize("trainer_type", ["sao", "single-rollout-asynchronous-optimization"])
def test_build_sao_trainer_routes_online_actor_critic_runtime_kwargs(trainer_type):
    critic_model = object()
    critic_selector = object()
    rollout_provider = object()
    environment_factory = object()
    tools = [object()]

    captured, elm, reward_func, train_dataset, eval_dataset = _capture_build(
        trainer_type,
        critic_model=critic_model,
        critic_trainable_selector=critic_selector,
        rollout_provider=rollout_provider,
        environment_factory=environment_factory,
        tools=tools,
        tool_guard=object(),
        compaction_controller=object(),
    )

    assert captured["arguments"] == "arguments"
    assert captured["model"] is elm._model
    assert captured["reward_funcs"] is reward_func
    assert captured["critic_model"] is critic_model
    assert "critic_state" not in captured
    assert captured["critic_trainable_selector"] is critic_selector
    assert captured["rollout_provider"] is rollout_provider
    assert captured["environment_factory"] is environment_factory
    assert captured["tools"] is tools
    assert captured["train_dataset"] is train_dataset
    assert captured["eval_dataset"] is eval_dataset
    assert "tool_guard" not in captured
    assert "compaction_controller" not in captured


@pytest.mark.parametrize("trainer_type", ["compaction_rl", "compaction-rl"])
def test_build_compaction_rl_trainer_routes_compaction_runtime_kwargs(trainer_type):
    critic_state = object()
    rollout_provider = object()
    environment_factory = object()
    tools = [object()]
    tool_guard = object()
    compaction_controller = object()

    captured, elm, reward_func, train_dataset, eval_dataset = _capture_build(
        trainer_type,
        critic_state=critic_state,
        rollout_provider=rollout_provider,
        env_factory=environment_factory,
        tools=tools,
        tool_guard=tool_guard,
        compaction_controller=compaction_controller,
    )

    assert captured["arguments"] == "arguments"
    assert captured["model"] is elm._model
    assert captured["reward_funcs"] is reward_func
    assert "critic_model" not in captured
    assert captured["critic_state"] is critic_state
    assert captured["rollout_provider"] is rollout_provider
    assert captured["environment_factory"] is environment_factory
    assert captured["tools"] is tools
    assert captured["tool_guard"] is tool_guard
    assert captured["compaction_controller"] is compaction_controller
    assert captured["train_dataset"] is train_dataset
    assert captured["eval_dataset"] is eval_dataset
