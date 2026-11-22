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

from easydel.trainers.trainer.trainer import Trainer


def _make_step_resolution_trainer(*, max_training_steps):
    trainer = object.__new__(Trainer)
    trainer.arguments = SimpleNamespace(
        max_training_steps=max_training_steps,
        max_evaluation_steps=None,
        total_batch_size=4,
        eval_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        per_epoch_training_steps=None,
        per_epoch_evaluation_steps=None,
    )
    return trainer


def test_resolve_step_count_auto_discovers_steps_from_dataset_length():
    trainer = _make_step_resolution_trainer(max_training_steps=None)

    resolution = Trainer._resolve_step_count(
        trainer,
        list(range(17)),
        source=None,
        is_train=True,
        drop_remainder=True,
    )

    assert resolution.steps == 4
    assert resolution.num_examples == 17
    assert resolution.num_examples_exact is True
    assert resolution.auto_discovered is True
    assert resolution.auto_clamped is False


def test_resolve_step_count_clamps_requested_steps_to_dataset_capacity():
    trainer = _make_step_resolution_trainer(max_training_steps=8)

    resolution = Trainer._resolve_step_count(
        trainer,
        list(range(17)),
        source=None,
        is_train=True,
        drop_remainder=True,
    )

    assert resolution.steps == 4
    assert resolution.num_examples == 17
    assert resolution.num_examples_exact is True
    assert resolution.auto_discovered is False
    assert resolution.auto_clamped is True


def test_resolve_step_count_matches_drop_remainder_training_capacity():
    trainer = _make_step_resolution_trainer(max_training_steps=None)
    trainer.arguments.total_batch_size = 8

    resolution = Trainer._resolve_step_count(
        trainer,
        list(range(17_300)),
        source=None,
        is_train=True,
        drop_remainder=True,
    )

    assert resolution.steps == 2162
