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

"""Public exports for GRPO replay-buffer training.

This package provides the replay-buffer configuration, trainer, and internal
buffer implementation used to mix fresh eSurge rollouts with previously
sampled GRPO batches.
"""

from .grpo_replay_buffer_config import GRPOWithReplayBufferConfig
from .grpo_replay_buffer_trainer import GRPOWithReplayBufferTrainer
from .replay_buffer import _ReplayBuffer

GRPOReplayBufferConfig = GRPOWithReplayBufferConfig
GRPOReplayBufferTrainer = GRPOWithReplayBufferTrainer

__all__ = (
    "GRPOReplayBufferConfig",
    "GRPOReplayBufferTrainer",
    "GRPOWithReplayBufferConfig",
    "GRPOWithReplayBufferTrainer",
    "_ReplayBuffer",
)
