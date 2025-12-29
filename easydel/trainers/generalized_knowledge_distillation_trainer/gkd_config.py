# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

from ..supervised_fine_tuning_trainer import SFTConfig


@Registry.register("trainer-arguments", "gkd")
@dataclass
class GKDConfig(SFTConfig):
    """Configuration for the :class:`~easydel.trainers.GKDTrainer`.

    Extends :class:`~easydel.trainers.SFTConfig` with the knobs required for
    Generalized Knowledge Distillation (GKD):

    Args:
        temperature: Sampling temperature used both for generation and for
            temperature scaling in the Jensen-Shannon loss.
        lmbda: Probability of replacing a training batch with on-policy
            student generations.
        beta: Interpolation factor of the generalized JSD loss. ``beta=0`` is
            KL(student‖teacher) and ``beta=1`` is KL(teacher‖student).
        max_new_tokens: Maximum number of tokens generated during on-policy
            rollouts.
        disable_dropout: Whether to disable dropout layers in both student
            and teacher models.
        seq_kd: When ``True`` always replaces the batch with teacher
            generations before applying the student on-policy sampling.
    """

    trainer_prefix: str | None = field(
        default="gkdtrainer",
        metadata={"help": "Default prefix used for checkpoints and logs."},
    )
    temperature: float = field(
        default=0.9,
        metadata={"help": "Generation temperature as well as the softmax temperature used when computing the GKD loss."},
    )
    lmbda: float = field(
        default=0.5,
        metadata={
            "help": "Probability of performing on-policy student generation for a batch. "
            "Set to 0 to disable on-policy sampling."
        },
    )
    beta: float = field(
        default=0.5,
        metadata={
            "help": "Interpolation factor in the generalized Jensen-Shannon divergence. "
            "0.0 reduces to KL(student‖teacher) while 1.0 becomes KL(teacher‖student)."
        },
    )
    max_new_tokens: int = field(
        default=128,
        metadata={"help": "Maximum number of tokens to generate during on-policy rollouts."},
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Disable dropout in both student and teacher models for more stable distillation."},
    )
    seq_kd: bool = field(
        default=False,
        metadata={"help": "Enable sequence-level KD by always regenerating the batch with the teacher before training."},
    )

    __hash__ = hash_fn

    def __post_init__(self, max_sequence_length: int | None):
        super().__post_init__(max_sequence_length=max_sequence_length)
        if not 0.0 <= self.lmbda <= 1.0:
            raise ValueError("`lmbda` must be within [0, 1].")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("`beta` must be within [0, 1].")
        if self.max_new_tokens <= 0:
            raise ValueError("`max_new_tokens` must be strictly positive.")
