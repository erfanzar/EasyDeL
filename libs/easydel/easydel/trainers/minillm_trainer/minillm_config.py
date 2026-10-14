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
"""MiniLLM reverse-KL trainer helpers."""

from __future__ import annotations

from dataclasses import dataclass, field

from easydel.utils import Registry

from ..group_relative_policy_optimization import GRPOConfig


@Registry.register("trainer-arguments", "minillm")
@dataclass
class MiniLLMConfig(GRPOConfig):
    """Configuration for MiniLLM-style reverse-KL advantage training.

    EasyDeL implements the reverse-KL advantage path on top of GRPO/eSurge
    rollouts. ``single_step_decomposition`` keeps the same batch path and
    changes how the per-token advantage is accumulated; ``kd_temperature``
    scales the sampled-token reverse-KL signal.
    """

    trainer_prefix: str | None = field(default="MiniLLM")
    rkl_advantage: bool = field(default=True)
    single_step_decomposition: bool = field(default=False)
    kd_temperature: float = field(default=1.0)
    gamma: float = field(default=0.0)
    length_normalization: bool = field(default=True)

    def __post_init__(self, max_sequence_length: int | None, quantization_block: int | None) -> None:
        """Validate MiniLLM-specific reverse-KL advantage knobs.

        The base GRPO config validates rollout, batching, and optimizer fields.
        This hook checks the teacher-advantage parameters that are consumed
        during preprocessing: ``kd_temperature`` must be positive because it
        divides log-probability deltas, and ``gamma`` must be non-negative
        because it is used as a suffix-return discount.
        """
        super().__post_init__(max_sequence_length=max_sequence_length, quantization_block=quantization_block)
        if self.kd_temperature <= 0.0:
            raise ValueError("`kd_temperature` must be positive.")
        if self.gamma < 0.0:
            raise ValueError("`gamma` must be non-negative.")
