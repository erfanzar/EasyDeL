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

from types import SimpleNamespace

import pytest

from easydel.trainers.binary_classifier_optimization_trainer.bco_config import BCOConfig
from easydel.trainers.binary_classifier_optimization_trainer.bco_trainer import BCOTrainer
from easydel.trainers.contrastive_preference_optimization_trainer.cpo_config import CPOConfig
from easydel.trainers.direct_preference_optimization_trainer.dpo_config import DPOConfig
from easydel.trainers.odds_ratio_preference_optimization_trainer.orpo_config import ORPOConfig
from easydel.trainers.odds_ratio_preference_optimization_trainer.orpo_trainer import ORPOTrainer


@pytest.mark.parametrize(
    ("config_cls", "trainer_name"),
    [
        (BCOConfig, "BCO"),
        (CPOConfig, "CPO"),
        (DPOConfig, "DPO"),
        (ORPOConfig, "ORPO"),
    ],
)
def test_preference_generate_during_eval_is_not_silently_ignored(config_cls, trainer_name):
    del trainer_name
    cfg = config_cls(generate_during_eval=True, evaluation_steps=7)

    assert cfg.generation_interval == 7
    assert cfg.use_esurge_generation is True


class _EvalModel:
    def __init__(self):
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1


def test_bco_disable_dropout_puts_policy_and_reference_in_eval_mode():
    policy = SimpleNamespace(model=_EvalModel())
    reference = SimpleNamespace(model=_EvalModel())

    BCOTrainer._disable_state_dropout(policy, reference)

    assert policy.model.eval_calls == 1
    assert reference.model.eval_calls == 1


def test_orpo_disable_dropout_puts_policy_in_eval_mode():
    policy = SimpleNamespace(model=_EvalModel())

    ORPOTrainer._disable_state_dropout(policy)

    assert policy.model.eval_calls == 1
