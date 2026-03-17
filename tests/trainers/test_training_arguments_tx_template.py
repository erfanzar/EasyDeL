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

import optax

from easydel.trainers.training_configurations import TrainingArguments


def _base_training_args() -> dict:
    return {
        "model_name": "dummy-model",
        "save_directory": "tmp-easydel-tests",
        "total_batch_size": 1,
        "learning_rate": 1e-5,
        "num_train_epochs": 1,
    }


def test_get_tx_template_applies_pruning_wrapper():
    class WrappedTx:
        def __init__(self, base_tx):
            self.base_tx = base_tx

        def init(self, params):
            return self.base_tx.init(params)

        def update(self, updates, state, params=None):
            return self.base_tx.update(updates, state, params)

    class DummyPruningModule:
        def wrap_optax(self, tx):
            return WrappedTx(tx)

    args = TrainingArguments(**_base_training_args(), pruning_module=DummyPruningModule())

    tx_template = args.get_tx_template()

    assert isinstance(tx_template, WrappedTx)


def test_get_tx_template_without_pruning_returns_plain_optimizer():
    args = TrainingArguments(**_base_training_args())

    tx_template = args.get_tx_template()

    assert isinstance(tx_template, optax.GradientTransformation)
