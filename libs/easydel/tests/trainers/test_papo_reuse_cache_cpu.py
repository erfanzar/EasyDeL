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

"""PAPO batch-reuse and reward-column routing contracts (no array computation)."""

import pytest
from easydel.trainers.group_relative_policy_optimization import GRPOTrainer
from easydel.trainers.papo_trainer.papo_trainer import PAPOTrainer

pytestmark = pytest.mark.cpu_ok


def test_reuse_cache_hit_returns_grpo_cached_batch_without_reshaping(monkeypatch):
    """While GRPO's reuse buffer is live, PAPO must not re-shape or re-read reward columns.

    The cached batch is already PAPO-shaped and carries the original prompt's reward columns;
    re-extracting from the new dataloader row would shape it a second time.
    """
    trainer = object.__new__(PAPOTrainer)
    trainer._buffered_grpo_remaining = 2
    sentinel = ({"cached": True}, {"m": 1})
    seen = {}

    def fake_grpo(self, state, batch, is_train):
        seen["args"] = (state, batch, is_train)
        return sentinel

    def forbidden(*args, **kwargs):
        raise AssertionError("PAPO shaping must be skipped on a reuse-cache hit")

    monkeypatch.setattr(GRPOTrainer, "_preprocess_batch_input", fake_grpo)
    monkeypatch.setattr(PAPOTrainer, "_apply_papo_objective_shaping", forbidden)
    monkeypatch.setattr(PAPOTrainer, "_extract_papo_reward_columns", forbidden)

    out = trainer._preprocess_batch_input(state="S", batch={"input_ids": None}, is_train=True)
    assert out is sentinel
    assert seen["args"] == ("S", {"input_ids": None}, True)


@pytest.mark.parametrize(
    ("batch", "expected"),
    [
        (
            {"perception_reward": [1], "der_reward1": [2], "der_rewards2": [3]},
            {"perception": [1], "der1": [2], "der2": [3]},
        ),
        ({"papo_perception_reward": [9], "perception_rewards": [0]}, {"perception": [9]}),
        ({"other": [1]}, {}),
    ],
)
def test_reward_column_aliases_map_to_canonical_names(batch, expected):
    trainer = object.__new__(PAPOTrainer)
    assert trainer._extract_papo_reward_columns(batch) == expected
