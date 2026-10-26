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

"""``extra_optimizer_kwargs`` must not silently override ``TrainingArguments`` optimizer fields."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from easydel.trainers.training_configurations import TrainingArguments


def _args(tmp_path, **kwargs) -> TrainingArguments:
    return TrainingArguments(save_directory=str(tmp_path / "ckpt"), **kwargs)


@pytest.mark.parametrize(
    ("field", "own", "extra"),
    [
        ("weight_decay", 0.01, 0.1),
        ("learning_rate", 1e-4, 5e-4),
        ("clip_grad", 1.0, 2.0),
    ],
)
def test_conflicting_extra_kwarg_raises(tmp_path, field, own, extra):
    with pytest.raises(ValueError, match=field):
        _args(tmp_path, **{field: own}, extra_optimizer_kwargs={field: extra})


def test_conflict_message_names_the_training_arguments_field(tmp_path):
    with pytest.raises(ValueError, match=r"TrainingArguments\.max_training_steps"):
        _args(tmp_path, max_training_steps=100, extra_optimizer_kwargs={"steps": 200})


def test_identical_value_is_accepted_and_merged(tmp_path):
    args = _args(tmp_path, weight_decay=0.01, learning_rate=1e-4, extra_optimizer_kwargs={"weight_decay": 0.01})

    assert args.optimizer_kwargs["weight_decay"] == 0.01
    assert args.optimizer_kwargs["learning_rate"] == 1e-4


def test_numerically_equal_int_and_float_are_identical(tmp_path):
    args = _args(tmp_path, clip_grad=1.0, extra_optimizer_kwargs={"clip_grad": 1})

    assert args.optimizer_kwargs["clip_grad"] == 1


def test_non_colliding_keys_pass_through(tmp_path):
    args = _args(tmp_path, extra_optimizer_kwargs={"b1": 0.8, "b2": 0.95})

    assert args.optimizer_kwargs["b1"] == 0.8
    assert args.optimizer_kwargs["b2"] == 0.95


def test_mu_dtype_spellings_agree_and_unset_field_accepts_any_value(tmp_path):
    same = _args(tmp_path, tx_mu_dtype=jnp.bfloat16, extra_optimizer_kwargs={"mu_dtype": "bfloat16"})
    assert same.optimizer_kwargs["mu_dtype"] == "bfloat16"

    unset = _args(tmp_path, extra_optimizer_kwargs={"mu_dtype": "float32"})
    assert unset.optimizer_kwargs["mu_dtype"] == "float32"

    with pytest.raises(ValueError, match="tx_mu_dtype"):
        _args(tmp_path, tx_mu_dtype=jnp.bfloat16, extra_optimizer_kwargs={"mu_dtype": "float32"})


def test_json_roundtrip_with_identical_extra_kwarg(tmp_path):
    args = _args(tmp_path, weight_decay=0.02, extra_optimizer_kwargs={"weight_decay": 0.02, "b1": 0.9})
    path = tmp_path / "args.json"
    args.save_arguments(path)

    loaded = TrainingArguments.load_arguments(path)

    assert loaded.to_dict() == args.to_dict()
    assert loaded.optimizer_kwargs["weight_decay"] == 0.02
