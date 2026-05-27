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
"""Online DPO built from eSurge completions."""

from __future__ import annotations

from dataclasses import dataclass, field

from easydel.utils import Registry

from ..direct_preference_optimization_trainer import DPOConfig


@Registry.register("trainer-arguments", "online_dpo")
@dataclass
class OnlineDPOConfig(DPOConfig):
    """Configuration for online DPO pairs generated through local eSurge.

    Each prompt produces two completions that are scored by reward functions and
    converted into chosen/rejected DPO examples inside the trainer. Reference
    log-prob precompute is disabled because the preference pairs do not exist
    until on-policy generation runs.
    """

    trainer_prefix: str | None = field(default="OnlineDPO")
    max_new_tokens: int | None = field(default=64)
    temperature: float = field(default=0.9)
    top_p: float = field(default=1.0)
    top_k: int = field(default=0)
    repetition_penalty: float = field(default=1.0)
    missing_eos_penalty: float | None = field(default=None)
    reward_weights: list[float] | tuple[float, ...] | None = field(default=None)

    def __post_init__(self, max_sequence_length: int | None, quantization_block: int | None) -> None:
        """Derive eSurge generation fields and validate online-pair settings.

        Online DPO creates chosen/rejected pairs after prompt generation, so it
        cannot precompute reference log-probabilities on the raw dataset. The
        method maps the public ``max_new_tokens`` and sampling fields onto the
        generation fields consumed by the shared eSurge rollout path and
        validates the optional missing-EOS penalty before parent normalization.
        """
        if self.precompute_ref_log_probs:
            raise ValueError(
                "OnlineDPO generates preference pairs on-policy; reference log-prob precompute is not valid."
            )
        if self.max_new_tokens is not None:
            self.generation_max_new_tokens = int(self.max_new_tokens)
            self.max_completion_length = int(self.max_new_tokens)
        self.generation_temperature = self.temperature
        self.generation_top_p = self.top_p
        self.generation_top_k = self.top_k
        self.generation_repetition_penalty = self.repetition_penalty
        self.generation_num_return_sequences = 2
        self.use_esurge_generation = True
        if self.missing_eos_penalty is not None and self.missing_eos_penalty <= 0.0:
            raise ValueError("`missing_eos_penalty` must be positive when set.")
        super().__post_init__(max_sequence_length=max_sequence_length, quantization_block=quantization_block)
