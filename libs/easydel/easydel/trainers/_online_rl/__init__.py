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
"""Shared implementation primitives for EasyDeL online-RL trainers."""

from .async_rollout import (
    AsyncRolloutCancelled,
    AsyncRolloutError,
    AsyncRolloutWorkerError,
    BoundedAsyncRolloutCoordinator,
    CompletedRollout,
    PolicySnapshot,
    RolloutTicket,
)
from .config import OnlineActorCriticConfig
from .math import (
    cross_segment_advantage_correction,
    following_segment_token_counts,
    global_token_denominator,
    global_token_mean,
    length_adaptive_lambda,
    skip_observation_gae,
    target_aligned_action_mask,
    target_aligned_token_ids,
)
from .trainer import OnlineActorCriticTrainer, RolloutProvider
from .trajectory import (
    OnlineRLTrajectory,
    OnlineRLTrajectoryCollator,
    PaddingSide,
    SegmentKind,
    SpanKind,
    TrajectorySegment,
    TrajectorySpan,
    TruncationSide,
    collate_trajectories,
)

__all__ = (
    "AsyncRolloutCancelled",
    "AsyncRolloutError",
    "AsyncRolloutWorkerError",
    "BoundedAsyncRolloutCoordinator",
    "CompletedRollout",
    "OnlineActorCriticConfig",
    "OnlineActorCriticTrainer",
    "OnlineRLTrajectory",
    "OnlineRLTrajectoryCollator",
    "PaddingSide",
    "PolicySnapshot",
    "RolloutProvider",
    "RolloutTicket",
    "SegmentKind",
    "SpanKind",
    "TrajectorySegment",
    "TrajectorySpan",
    "TruncationSide",
    "collate_trajectories",
    "cross_segment_advantage_correction",
    "following_segment_token_counts",
    "global_token_denominator",
    "global_token_mean",
    "length_adaptive_lambda",
    "skip_observation_gae",
    "target_aligned_action_mask",
    "target_aligned_token_ids",
)
