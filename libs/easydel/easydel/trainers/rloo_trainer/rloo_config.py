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
"""RLOO trainer config aliases backed by GRPO."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field

from easydel.utils import Registry

from ..group_relative_policy_optimization import GRPOConfig


@Registry.register("trainer-arguments", "rloo")
@dataclass
class RLOOConfig(GRPOConfig):
    """Configuration for leave-one-out policy optimization over GRPO rollouts.

    RLOO is implemented through the GRPO rollout and loss stack with
    ``advantage_estimator='leave_one_out'``. The config requires more than one
    generated completion per prompt so each completion can be compared against
    the mean of its siblings.
    """

    trainer_prefix: str | None = field(default="RLOO")
    num_return_sequences: int = field(default=2)
    num_generations: int | None = field(default=None)
    beta: float = field(default=0.05)
    scale_rewards: str | bool = field(default="none")
    advantage_estimator: tp.Literal["leave_one_out"] = field(default="leave_one_out")

    def __post_init__(self, max_sequence_length: int | None, quantization_block: int | None) -> None:
        """Normalize GRPO fields and enforce the leave-one-out group size.

        The parent config resolves the effective generation count from GRPO
        aliases before validation. RLOO then requires at least two completions
        per prompt because each completion's baseline is computed from the
        other samples in the same group.
        """
        super().__post_init__(max_sequence_length=max_sequence_length, quantization_block=quantization_block)
        if self.num_generations is None or self.num_generations <= 1:
            raise ValueError("RLOO requires `num_generations > 1` for leave-one-out advantages.")
