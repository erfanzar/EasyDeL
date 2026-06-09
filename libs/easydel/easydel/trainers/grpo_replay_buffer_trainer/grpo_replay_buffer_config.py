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
"""GRPO replay-buffer trainer extension."""

from __future__ import annotations

from dataclasses import dataclass, field

from easydel.utils import Registry

from ..group_relative_policy_optimization import GRPOConfig


@Registry.register("trainer-arguments", "grpo_with_replay_buffer")
@dataclass
class GRPOWithReplayBufferConfig(GRPOConfig):
    """GRPO configuration with bounded in-process replay-buffer reuse.

    ``replay_buffer_size`` controls how many previous rollout batches can be
    retained and mixed into later updates. A value of zero keeps the base GRPO
    behavior while preserving the same trainer surface.
    """

    trainer_prefix: str | None = field(default="GRPOWithReplayBuffer")
    replay_buffer_size: int = field(default=64)

    def __post_init__(self, max_sequence_length: int | None, quantization_block: int | None) -> None:
        """Validate replay-buffer capacity after GRPO config normalization.

        The base GRPO config resolves generation, batching, and loss invariants.
        This hook only validates the extension capacity: ``0`` disables replay
        while preserving the same trainer class, and positive values bound the
        number of score-prioritized rollout groups kept on the host.
        """
        super().__post_init__(max_sequence_length=max_sequence_length, quantization_block=quantization_block)
        if self.replay_buffer_size < 0:
            raise ValueError("`replay_buffer_size` must be non-negative.")
