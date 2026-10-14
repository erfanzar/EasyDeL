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
"""Self-distillation fine-tuning trainer aliases."""

from __future__ import annotations

from dataclasses import dataclass, field

from easydel.utils import Registry

from ..self_distillation_policy_optimization import SDPOConfig


@Registry.register("trainer-arguments", "self_distillation")
@dataclass
class SelfDistillationConfig(SDPOConfig):
    """Self-distillation config alias for SDPO-compatible training.

    The alias exposes a broader self-distillation registry name while retaining
    SDPO's actual fields and validation. It does not introduce a separate loss
    path.
    """

    trainer_prefix: str | None = field(default="SelfDistillation")


@Registry.register("trainer-arguments", "sdft")
@dataclass
class SDFTConfig(SelfDistillationConfig):
    """Configuration for supervised self-distillation fine-tuning.

    SDFT can format teacher prompts, optionally generate teacher-side content,
    and skip a fixed number of completion loss tokens before applying the
    SDPO-compatible self-distillation objective.
    """

    trainer_prefix: str | None = field(default="SDFT")
    generate_from_teacher: bool = field(default=False)
    teacher_prompt_template: str | None = field(default=None)
    num_loss_tokens_to_skip: int = field(default=0)

    def __post_init__(self, max_sequence_length: int | None, quantization_block: int | None) -> None:
        """Validate SDFT-only options after SDPO config normalization.

        SDFT inherits SDPO generation, reward, and loss settings. This hook
        adds checks for the SDFT teacher-template side channel and the
        completion-token skip count so malformed templates fail at config
        construction instead of during trainer preprocessing.
        """
        super().__post_init__(max_sequence_length=max_sequence_length, quantization_block=quantization_block)
        if self.num_loss_tokens_to_skip < 0:
            raise ValueError("`num_loss_tokens_to_skip` must be non-negative.")
        if self.teacher_prompt_template is not None and (
            "{prompt}" not in self.teacher_prompt_template or "{privileged_context}" not in self.teacher_prompt_template
        ):
            raise ValueError("`teacher_prompt_template` must contain `{prompt}` and `{privileged_context}`.")
