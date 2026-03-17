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

import pytest

from easydel.trainers.training_utils import resolve_total_steps


def test_forced_training_steps_are_optimizer_steps():
    assert (
        resolve_total_steps(
            forced_steps=100,
            total_data_len=None,
            batch_size=32,
            num_epochs=1,
            gradient_accumulation_steps=4,
            is_train=True,
        )
        == 100
    )


def test_computed_training_steps_apply_gradient_accumulation():
    assert (
        resolve_total_steps(
            forced_steps=None,
            total_data_len=3200,
            batch_size=32,
            num_epochs=1,
            gradient_accumulation_steps=4,
            is_train=True,
        )
        == 25
    )


def test_eval_steps_do_not_apply_gradient_accumulation():
    assert (
        resolve_total_steps(
            forced_steps=None,
            total_data_len=3200,
            batch_size=32,
            num_epochs=1,
            gradient_accumulation_steps=4,
            is_train=False,
        )
        == 100
    )


def test_missing_total_data_len_raises():
    with pytest.raises(ValueError):
        resolve_total_steps(
            forced_steps=None,
            total_data_len=None,
            batch_size=32,
            num_epochs=1,
            gradient_accumulation_steps=1,
            is_train=True,
        )
