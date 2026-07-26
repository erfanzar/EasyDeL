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

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from easydel.trainers._online_rl.trajectory import (
    OnlineRLTrajectory,
    OnlineRLTrajectoryCollator,
    TrajectorySegment,
    TrajectorySpan,
    collate_trajectories,
)


def _trajectory(*, policy_version=3, input_ids=None):
    return OnlineRLTrajectory(
        input_ids=np.arange(10, 18) if input_ids is None else input_ids,
        attention_mask=np.ones(8, dtype=bool),
        action_mask=np.asarray([0, 0, 1, 1, 0, 1, 0, 1], dtype=bool),
        rewards=np.asarray([0, 0, 0, 0.2, 0, -0.1, 0, 1.0], dtype=np.float32),
        behavior_logps=np.asarray([0, 0, -0.1, -0.2, 0, -0.3, 0, -0.4], dtype=np.float32),
        spans=(
            TrajectorySpan(0, 2, "prompt", segment_id=0),
            TrajectorySpan(2, 4, "action", segment_id=0),
            TrajectorySpan(4, 5, "observation", segment_id=0),
            TrajectorySpan(5, 6, "action", segment_id=1),
            TrajectorySpan(6, 7, "observation", segment_id=1),
            TrajectorySpan(7, 8, "compaction", segment_id=2),
        ),
        segments=(
            TrajectorySegment(0, 0, 5, "execution"),
            TrajectorySegment(1, 5, 7, "execution"),
            TrajectorySegment(2, 7, 8, "compaction", is_terminal=True),
        ),
        policy_version=policy_version,
        metadata={"source": "test"},
    )


def test_trajectory_records_are_frozen_and_own_read_only_arrays():
    source_ids = np.arange(10, 18)
    trajectory = _trajectory(input_ids=source_ids)
    source_ids[0] = -1

    assert trajectory.input_ids[0] == 10
    assert trajectory.optimized_token_count == 4
    assert trajectory.metadata["source"] == "test"
    with pytest.raises(ValueError):
        trajectory.input_ids[0] = 999
    with pytest.raises(ValueError):
        trajectory.behavior_logps[2] = 0
    with pytest.raises(TypeError):
        trajectory.metadata["source"] = "mutated"
    with pytest.raises(FrozenInstanceError):
        trajectory.policy_version = 9


def test_trajectory_rejects_role_and_region_invariant_violations():
    with pytest.raises(ValueError, match="outside attention_mask"):
        OnlineRLTrajectory(
            input_ids=np.arange(3),
            attention_mask=np.asarray([1, 1, 0]),
            action_mask=np.asarray([0, 0, 1]),
            rewards=np.zeros(3),
        )

    with pytest.raises(ValueError, match="only select action"):
        OnlineRLTrajectory(
            input_ids=np.arange(3),
            attention_mask=np.ones(3),
            action_mask=np.asarray([0, 1, 0]),
            rewards=np.zeros(3),
            transition_mask=np.asarray([1, 0, 0]),
        )

    with pytest.raises(ValueError, match="unknown segment"):
        OnlineRLTrajectory(
            input_ids=np.arange(3),
            attention_mask=np.ones(3),
            action_mask=np.asarray([0, 1, 0]),
            rewards=np.zeros(3),
            spans=(TrajectorySpan(1, 2, "action", segment_id=4),),
            segments=(TrajectorySegment(0, 0, 3),),
        )

    with pytest.raises(ValueError, match="behavior_logps"):
        OnlineRLTrajectory(
            input_ids=np.arange(3),
            attention_mask=np.ones(3),
            action_mask=np.asarray([0, 1, 0]),
            rewards=np.zeros(3),
            behavior_logps=np.asarray([0.0, np.nan, 0.0]),
        )


def test_collator_preserves_context_attention_separately_from_policy_actions():
    trajectory = _trajectory()
    batch = OnlineRLTrajectoryCollator(max_length=10, pad_token_id=0)([trajectory])

    np.testing.assert_array_equal(batch["attention_mask"][0], [1, 1, 1, 1, 1, 1, 1, 1, 0, 0])
    np.testing.assert_array_equal(batch["action_mask"][0], [0, 0, 1, 1, 0, 1, 0, 1, 0, 0])
    np.testing.assert_array_equal(batch["target_action_mask"][0], [0, 1, 1, 0, 1, 0, 1, 0, 0])
    np.testing.assert_array_equal(batch["target_ids"][0], [11, 12, 13, 14, 15, 16, 17, 0, 0])
    np.testing.assert_array_equal(batch["target_segment_ids"][0], [0, 0, 0, 0, 1, 1, 2, -1, -1])
    np.testing.assert_allclose(
        batch["target_behavior_logps"][0, [1, 2, 4, 6]],
        [-0.1, -0.2, -0.3, -0.4],
    )
    assert np.all(np.isnan(batch["behavior_logps"][0, [0, 1, 4, 6, 8, 9]]))
    assert batch["trajectory_ids"].tolist() == [0]
    assert batch["policy_versions"].tolist() == [3]

    # The observation tokens (14 and 16) remain attended but never optimize
    # the policy.
    assert batch["attention_mask"][0, 4]
    assert not batch["action_mask"][0, 4]
    assert batch["attention_mask"][0, 6]
    assert not batch["action_mask"][0, 6]


def test_default_transitions_stop_at_declared_segment_boundaries():
    trajectory = _trajectory()
    np.testing.assert_array_equal(trajectory.transition_mask, [0, 0, 1, 0, 0, 0, 0, 0])


def test_collator_preserves_explicit_trajectory_grouping_ids():
    first = _trajectory()
    second = OnlineRLTrajectory(
        input_ids=first.input_ids,
        attention_mask=first.attention_mask,
        action_mask=first.action_mask,
        rewards=first.rewards,
        behavior_logps=first.behavior_logps,
        spans=first.spans,
        segments=first.segments,
        trajectory_id=12,
        policy_version=first.policy_version,
    )

    batch = collate_trajectories([first, second], max_length=8, pad_token_id=0)

    np.testing.assert_array_equal(batch["trajectory_ids"], [0, 12])


def test_collator_automatic_trajectory_ids_do_not_collide_with_explicit_ids():
    implicit = _trajectory()
    explicit = OnlineRLTrajectory(
        input_ids=implicit.input_ids,
        attention_mask=implicit.attention_mask,
        action_mask=implicit.action_mask,
        rewards=implicit.rewards,
        behavior_logps=implicit.behavior_logps,
        spans=implicit.spans,
        segments=implicit.segments,
        trajectory_id=0,
        policy_version=implicit.policy_version,
    )

    batch = collate_trajectories([implicit, explicit], max_length=8, pad_token_id=0)

    np.testing.assert_array_equal(batch["trajectory_ids"], [1, 0])


def test_right_truncation_severs_transition_to_a_removed_action():
    trajectory = OnlineRLTrajectory(
        input_ids=np.arange(10, 18),
        attention_mask=np.ones(8),
        action_mask=np.asarray([0, 0, 1, 0, 0, 1, 0, 1]),
        rewards=np.zeros(8),
    )
    batch = collate_trajectories([trajectory], max_length=6, pad_token_id=0, truncation_side="right")

    np.testing.assert_array_equal(batch["input_ids"][0], [10, 11, 12, 13, 14, 15])
    # Token 15 originally continued to action token 17. Once 17 is truncated,
    # token 15 must be a terminal boundary in the in-buffer GAE chain.
    assert not batch["transition_mask"][0, 5]
    assert not batch["target_transition_mask"][0, 4]
    np.testing.assert_array_equal(batch["target_action_mask"][0], [0, 1, 0, 0, 1])


def test_left_padding_does_not_train_first_valid_token_from_a_pad_logit():
    trajectory = OnlineRLTrajectory(
        input_ids=np.asarray([50, 51, 52]),
        attention_mask=np.ones(3),
        action_mask=np.asarray([1, 1, 1]),
        rewards=np.ones(3),
    )
    batch = collate_trajectories(
        [trajectory],
        max_length=5,
        pad_token_id=0,
        padding_side="left",
    )

    np.testing.assert_array_equal(batch["input_ids"][0], [0, 0, 50, 51, 52])
    # Target 50 is preceded by padding and is excluded; 51 and 52 are valid.
    np.testing.assert_array_equal(batch["target_action_mask"][0], [0, 0, 1, 1])
    assert np.all(np.isnan(batch["behavior_logps"]))
    assert np.all(np.isnan(batch["target_behavior_logps"]))


def test_left_truncation_keeps_latest_tokens_and_target_alignment():
    trajectory = _trajectory(policy_version=7)
    batch = collate_trajectories(
        [trajectory],
        max_length=5,
        pad_token_id=0,
        truncation_side="left",
    )

    np.testing.assert_array_equal(batch["input_ids"][0], [13, 14, 15, 16, 17])
    np.testing.assert_array_equal(batch["target_ids"][0], [14, 15, 16, 17])
    np.testing.assert_array_equal(batch["target_action_mask"][0], [0, 1, 0, 1])
    assert batch["policy_versions"].tolist() == [7]
