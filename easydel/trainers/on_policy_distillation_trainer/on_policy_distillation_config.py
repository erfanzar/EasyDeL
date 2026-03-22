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
from __future__ import annotations

from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..distillation_trainer.distillation_config import DistillationConfig


@Registry.register("trainer-arguments", "on_policy_distillation")
@dataclass
class OnPolicyDistillationConfig(DistillationConfig):
    """Configuration for on-policy knowledge distillation training.

    Unlike standard (offline) distillation which operates on a fixed dataset
    with pre-computed teacher outputs, on-policy distillation generates
    completions from prompts during training and distills teacher knowledge
    on the student's own generated sequences.

    This configuration extends DistillationConfig with parameters for
    controlling the online generation process during training.

    Attributes:
        max_prompt_length: Maximum number of tokens for prompts.
        max_completion_length: Maximum number of new tokens to generate per prompt.
        num_generations_per_prompt: How many completions to sample per prompt.
        temperature_sampling: Sampling temperature for generation (separate
            from distillation temperature).
        top_k: Top-k sampling parameter for generation.
        top_p: Top-p (nucleus) sampling parameter for generation.
        presence_penalty: Presence penalty for generation.
        frequency_penalty: Frequency penalty for generation.
        repetition_penalty: Repetition penalty for generation.
        generate_with_teacher: If True, teacher generates completions instead
            of student.
        skip_apply_chat_template: Whether to skip chat template application.
    """

    trainer_prefix: str | None = field(
        default="OnPolicyDistillation",
        metadata={"help": "Prefix used for trainer logs, checkpoints, and wandb runs."},
    )
    remove_unused_columns: bool | None = field(
        default=False,
        metadata={"help": "Whether to remove unused columns from the dataset."},
    )
    max_prompt_length: int = field(
        default=512,
        metadata={"help": "Maximum number of tokens for prompts."},
    )
    max_completion_length: int = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to generate per prompt."},
    )
    num_generations_per_prompt: int = field(
        default=1,
        metadata={"help": "How many completions to sample per prompt."},
    )
    temperature_sampling: float = field(
        default=0.7,
        metadata={
            "help": "Sampling temperature for generation. Separate from the distillation "
            "temperature which controls logit softening for KL loss."
        },
    )
    top_k: int | None = field(
        default=50,
        metadata={"help": "Top-k sampling parameter for generation. None disables top-k."},
    )
    top_p: float = field(
        default=0.95,
        metadata={"help": "Top-p (nucleus) sampling parameter for generation."},
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={"help": "Presence penalty applied during generation."},
    )
    frequency_penalty: float = field(
        default=0.0,
        metadata={"help": "Frequency penalty applied during generation."},
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "Repetition penalty applied during generation."},
    )
    generate_with_teacher: bool = field(
        default=False,
        metadata={
            "help": "If True, the teacher model generates completions instead of the student. "
            "Useful for comparison or curriculum-style distillation."
        },
    )
    skip_apply_chat_template: bool = field(
        default=False,
        metadata={"help": "Whether to skip chat template application on prompts."},
    )

    def __post_init__(
        self,
        max_sequence_length: int | None,
        quantization_block: int | None,
    ):
        default_completion = type(self).__dataclass_fields__["max_completion_length"].default
        if self.max_length is not None:
            if self.max_length < self.max_prompt_length:
                raise ValueError(
                    f"`max_length` ({self.max_length}) must be >= `max_prompt_length` ({self.max_prompt_length})."
                )
            max_allowed_completion = self.max_length - self.max_prompt_length
            if self.max_completion_length == default_completion:
                self.max_completion_length = max_allowed_completion
            elif self.max_completion_length > max_allowed_completion:
                raise ValueError(
                    "`max_prompt_length + max_completion_length` "
                    f"({self.max_prompt_length} + {self.max_completion_length}) must be <= `max_length` "
                    f"({self.max_length})."
                )

        self.max_length = self.max_prompt_length + self.max_completion_length

        if hasattr(super(), "__post_init__"):
            super().__post_init__(
                max_sequence_length=None,
                quantization_block=quantization_block,
            )

    __hash__ = hash_fn
