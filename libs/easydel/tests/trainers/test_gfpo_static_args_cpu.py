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

"""GFPO must only override ``num_generations`` in the GRPO static-argument tuples.

Pure Python: the parent compile step is replaced by a stub that installs one sentinel per
``grpo_step`` static parameter, so no array computation runs.
"""

import inspect
from types import SimpleNamespace

import pytest
from easydel.trainers.group_filtered_policy_optimization.gfpo_trainer import GFPOTrainer
from easydel.trainers.group_relative_policy_optimization import GRPOTrainer
from easydel.trainers.group_relative_policy_optimization._fn import grpo_step

pytestmark = pytest.mark.cpu_ok

_STATIC_NAMES = tuple(inspect.signature(grpo_step).parameters)[2:]


def _sentinels(prefix: str, num_generations: int = 8) -> tuple:
    return (num_generations, *(f"{prefix}:{name}" for name in _STATIC_NAMES[1:]))


@pytest.mark.parametrize("num_remains_in_group", [2, None])
def test_gfpo_static_args_bind_to_grpo_step_with_only_generation_count_changed(monkeypatch, num_remains_in_group):
    train_parent = _sentinels("train")
    eval_parent = _sentinels("eval")
    parent_output = object()

    def fake_parent_configure(self):
        self._train_shared_fn_static_args = train_parent
        self._eval_shared_fn_static_args = eval_parent
        return parent_output

    monkeypatch.setattr(GRPOTrainer, "configure_functions", fake_parent_configure)
    trainer = object.__new__(GFPOTrainer)
    trainer.arguments = SimpleNamespace(num_remains_in_group=num_remains_in_group)

    assert trainer.configure_functions() is parent_output

    for prefix, statics in (
        ("train", trainer._train_shared_fn_static_args),
        ("eval", trainer._eval_shared_fn_static_args),
    ):
        bound = inspect.signature(grpo_step).bind("state", "batch", *statics).arguments
        expected_generations = 8 if num_remains_in_group is None else num_remains_in_group
        assert bound["num_generations"] == expected_generations
        for name in _STATIC_NAMES[1:]:
            assert bound[name] == f"{prefix}:{name}", name


@pytest.mark.parametrize(
    ("eval_generations", "expected"),
    [(2, 2), (4, 4), (6, 4), (8, 4)],
)
def test_gfpo_eval_group_size_is_not_raised_above_the_unfiltered_eval_group(monkeypatch, eval_generations, expected):
    """Eval groups that are already <= ``num_remains_in_group`` are never filtered, so keep their size."""
    train_parent = _sentinels("train")
    eval_parent = _sentinels("eval", num_generations=eval_generations)

    def fake_parent_configure(self):
        self._train_shared_fn_static_args = train_parent
        self._eval_shared_fn_static_args = eval_parent
        return object()

    monkeypatch.setattr(GRPOTrainer, "configure_functions", fake_parent_configure)
    trainer = object.__new__(GFPOTrainer)
    trainer.arguments = SimpleNamespace(num_remains_in_group=4)
    trainer.configure_functions()

    assert trainer._eval_shared_fn_static_args[0] == expected
    assert trainer._train_shared_fn_static_args[0] == 4
