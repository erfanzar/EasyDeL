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
"""eLarge ``agent_flow`` with an OnlineDPO inner trainer, and AgentFlow ``base_trainer_type`` guards.

No array computation: model loading is stubbed and the inner trainer's constructor only records its kwargs.
"""

from __future__ import annotations

import functools

import pytest
from easydel.infra.elarge.model import eLargeModel
from easydel.trainers.agent_flow_trainer import AgentFlowConfig, AgentFlowTrainer
from easydel.trainers.agent_flow_trainer.agent_flow_trainer import agent_flow_supplied_reward
from easydel.trainers.online_dpo_trainer import OnlineDPOConfig, OnlineDPOTrainer

pytestmark = pytest.mark.cpu_ok

_REFERENCE = object()


class _Tok:
    pad_token_id = 0


def _elm(trainer_cfg: dict, *, with_reference: bool = False) -> eLargeModel:
    elm = object.__new__(eLargeModel)
    elm._config = {"model": {"name_or_path": "dummy-model"}, "trainer": dict(trainer_cfg)}
    elm._model = object()
    elm._state = None
    elm._tokenizer = _Tok()
    elm.build_reward_model = lambda: None
    elm.build_reference_model = (lambda: _REFERENCE) if with_reference else (lambda: None)
    return elm


@pytest.fixture
def captured_inner(monkeypatch):
    """Replace OnlineDPOTrainer.__init__ with a recorder that keeps the real constructor signature."""
    captured: dict = {}
    original = OnlineDPOTrainer.__init__

    @functools.wraps(original)
    def _record(self, **kwargs):
        captured.update(kwargs)
        self.arguments = kwargs["arguments"]
        self.processing_class = kwargs.get("processing_class")
        self._preprocess_batch_input = lambda state, batch, is_train: ({}, {})

    monkeypatch.setattr(OnlineDPOTrainer, "__init__", _record)
    return captured


_YAML_TRAINER = {
    "trainer_type": "agent_flow",
    "base_trainer_type": "online_dpo",
    "environment": "code_judge",
    "environment_backend": "local",
    "max_turns": 2,
    "group_size": 2,
    "beta": 0.25,
    "loss_type": "ipo",
    "label_smoothing": 0.05,
    "max_new_tokens": 24,
    "learning_rate": 3e-6,
    "weight_decay": 0.0,
    "total_batch_size": 2,
    "max_prompt_length": 64,
    "use_wandb": False,
}


def test_yaml_agent_flow_online_dpo_builds_online_dpo_inner_with_forwarded_hyperparameters(captured_inner):
    trainer = _elm(_YAML_TRAINER).build_trainer()

    assert isinstance(trainer, AgentFlowTrainer)
    assert isinstance(trainer.trainer, OnlineDPOTrainer)
    assert trainer.arguments.base_trainer_type == "online_dpo"

    inner_args = captured_inner["arguments"]
    assert type(inner_args) is OnlineDPOConfig
    assert inner_args.beta == 0.25
    # DPOConfig normalizes loss_type to a tuple of variants.
    assert tuple(inner_args.loss_type) == ("ipo",)
    assert inner_args.label_smoothing == 0.05
    assert inner_args.learning_rate == 3e-6
    assert inner_args.weight_decay == 0.0
    assert inner_args.max_prompt_length == 64
    assert inner_args.total_batch_size == 2
    # OnlineDPO maps max_new_tokens onto the generation/completion budget.
    assert inner_args.max_new_tokens == 24
    assert inner_args.max_completion_length == 24
    # Wrapper-only fields stay on the AgentFlow config.
    assert inner_args.trainer_prefix == "OnlineDPO"
    assert not hasattr(inner_args, "environment")

    # Environment-only setup: the wrapper supplies rewards and pairs, so the inner gets the placeholder.
    assert captured_inner["reward_funcs"] is agent_flow_supplied_reward
    assert "data_tokenize_fn" not in captured_inner
    assert trainer.arguments.environment == "code_judge"
    assert trainer.arguments.environment_backend == "local"
    assert trainer.arguments.max_turns == 2
    assert callable(trainer._env_factory)
    # Pairs need two episodes per group.
    assert trainer._resolve_group_size(True) == 2


def test_yaml_reference_model_reaches_online_dpo_inner(captured_inner):
    _elm(_YAML_TRAINER, with_reference=True).build_trainer()
    assert captured_inner["reference_model"] is _REFERENCE


def test_unset_dpo_fields_keep_online_dpo_defaults(captured_inner):
    cfg = {k: v for k, v in _YAML_TRAINER.items() if k not in {"beta", "loss_type", "label_smoothing"}}
    _elm(cfg).build_trainer()
    defaults = OnlineDPOConfig()
    inner_args = captured_inner["arguments"]
    assert inner_args.beta == defaults.beta
    assert inner_args.loss_type == defaults.loss_type


def test_explicit_base_config_is_used_as_given(captured_inner):
    cfg = dict(_YAML_TRAINER, base_config={"beta": 0.7})
    _elm(cfg).build_trainer()
    inner_args = captured_inner["arguments"]
    assert type(inner_args) is OnlineDPOConfig
    assert inner_args.beta == 0.7


@pytest.mark.parametrize(
    "name",
    ["grpo", "gmpo", "gdpo", "reinforce_plus_plus", "rloo", "ppo", "vapo", "online_dpo", "gkd", "seq_kd", "sdft"],
)
def test_supported_base_trainer_types_accepted(name):
    assert AgentFlowConfig(base_trainer_type=name.upper()).base_trainer_type == name


@pytest.mark.parametrize("name", ["sft", "dpo", "kto", "orpo", "reward", "not_a_trainer"])
def test_unsupported_base_trainer_types_rejected(name):
    with pytest.raises(ValueError, match="not supported by AgentFlowTrainer"):
        AgentFlowConfig(base_trainer_type=name)


def test_recursive_base_trainer_type_rejected():
    with pytest.raises(ValueError, match="recursive"):
        AgentFlowConfig(base_trainer_type="agent-flow")


def test_yaml_unsupported_base_trainer_type_fails_at_argument_build():
    elm = eLargeModel(
        {
            "model": {"name_or_path": "dummy-model"},
            "trainer": {"trainer_type": "agent_flow", "base_trainer_type": "sft", "environment": "code_judge"},
        }
    )
    with pytest.raises(ValueError, match="not supported by AgentFlowTrainer"):
        elm._build_training_arguments_for_type()
