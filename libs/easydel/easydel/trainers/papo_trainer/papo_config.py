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
"""PAPO objective shaping for GRPO rollouts."""

from __future__ import annotations

from dataclasses import dataclass, field

from easydel.utils import Registry

from ..group_relative_policy_optimization import GRPOConfig


@Registry.register("trainer-arguments", "papo")
@dataclass
class PAPOConfig(GRPOConfig):
    """Configuration for PAPO reward and objective shaping on GRPO batches.

    The trainer uses perception and DER reward columns when present, then masks
    completion-token objectives according to ``mask_type`` and ``mask_ratio``.
    The validation keeps shaping weights and mask parameters in a bounded range.
    """

    trainer_prefix: str | None = field(default="PAPO")
    perception_loss_weight: float = field(default=0.1)
    mask_ratio: float = field(default=0.3)
    mask_type: str = field(default="random")
    der_loss_weight1: float = field(default=0.03)
    der_loss_weight2: float = field(default=0.03)

    def __post_init__(self, max_sequence_length: int | None, quantization_block: int | None) -> None:
        """Validate PAPO shaping parameters after GRPO config normalization.

        The inherited GRPO config resolves rollout lengths, generation counts,
        and training defaults first. PAPO then enforces bounded masking and
        non-negative auxiliary reward weights so token masking and reward-column
        shaping cannot invert or amplify losses through invalid config values.
        """
        super().__post_init__(max_sequence_length=max_sequence_length, quantization_block=quantization_block)
        if not 0.0 <= self.mask_ratio <= 1.0:
            raise ValueError("`mask_ratio` must be in [0, 1].")
        if self.mask_type not in {"random", "prefix", "suffix", "alternate"}:
            raise ValueError("`mask_type` must be one of 'random', 'prefix', 'suffix', or 'alternate'.")
        if self.perception_loss_weight < 0.0 or self.der_loss_weight1 < 0.0 or self.der_loss_weight2 < 0.0:
            raise ValueError("PAPO loss weights must be non-negative.")
