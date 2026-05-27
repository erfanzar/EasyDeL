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
"""SSD eSurge self-distillation trainer."""

from __future__ import annotations

from dataclasses import dataclass, field

from easydel.utils import Registry

from ..self_distillation_policy_optimization import SDPOConfig


@Registry.register("trainer-arguments", "ssd")
@dataclass
class SSDConfig(SDPOConfig):
    """Configuration for single-sample self-distillation with eSurge rollouts.

    SSD fixes generation to one sampled completion per prompt and optionally
    filters empty completions before feeding the batch into the SDPO-style loss.
    """

    trainer_prefix: str | None = field(default="SSD")
    filter_empty: bool = field(default=True)

    def __post_init__(self, max_sequence_length: int | None, quantization_block: int | None) -> None:
        """Force SSD to single-completion rollouts before parent validation.

        SSD trains on one self-generated completion per prompt rather than a
        grouped reward objective. The hook pins both generation aliases to one
        and then lets the SDPO parent normalize lengths, sharding, and optimizer
        configuration.
        """
        self.num_generations = 1
        self.num_return_sequences = 1
        super().__post_init__(max_sequence_length=max_sequence_length, quantization_block=quantization_block)
