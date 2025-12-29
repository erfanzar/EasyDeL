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

import typing as tp
from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..training_configurations import TrainingArguments


@Registry.register("trainer-arguments", "distillation")
@dataclass
class DistillationConfig(TrainingArguments):
    """Configuration class for knowledge distillation training.

    This configuration extends TrainingArguments with parameters specific to
    knowledge distillation, where a smaller student model learns to mimic
    a larger teacher model's behavior.

    Knowledge distillation uses temperature scaling to soften the probability
    distributions from both models, allowing the student to learn from the
    teacher's confidence across all classes rather than just hard labels.

    Attributes:
        trainer_prefix (str | None): Prefix for trainer logs and checkpoints.
            Default: "distillationtrainer"
        temperature (float): Temperature parameter for softening probability
            distributions. Higher values create softer distributions, revealing
            more information about the teacher's relative confidence across classes.
            Typical values range from 3.0 to 10.0. Default: 2.0
        alpha (float): Weight balancing distillation loss vs supervised loss.
            - alpha=1.0: Pure distillation (only learn from teacher)
            - alpha=0.0: Pure supervised learning (only learn from labels)
            - 0<alpha<1: Combination of both losses
            Default: 0.9 (90% distillation, 10% supervised)

    Example:
        >>> config = DistillationConfig(
        ...     temperature=5.0,
        ...     alpha=0.7,
        ...     learning_rate=1e-4,
        ...     num_train_epochs=10
        ... )

    Note:
        The distillation loss is computed as:
        Loss = alpha * KL(student/T, teacher/T) + (1-alpha) * CE(student, labels)
        where T is the temperature parameter.
    """

    trainer_prefix: str | None = field(
        default="distillationtrainer", metadata={"help": "Prefix used for trainer logs, checkpoints, and wandb runs."}
    )
    temperature: float = field(
        default=2.0,
        metadata={
            "help": "Temperature for softening probability distributions. Higher values "
            "create softer distributions, revealing more about teacher's confidence."
        },
    )
    alpha: float = field(
        default=0.9,
        metadata={
            "help": "Weight for distillation loss vs supervised loss. "
            "1.0 = pure distillation, 0.0 = pure supervised learning."
        },
    )
    hidden_state_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Optional coefficient for matching student and teacher hidden states. "
                "Set to 0 to disable hidden-state distillation."
            )
        },
    )
    hidden_state_layers: tuple[int, ...] | None = field(
        default=None,
        metadata={
            "help": (
                "Indices of transformer layers whose hidden states should be distilled. "
                "Negative indices follow Python semantics. Defaults to the final layer when omitted."
            )
        },
    )
    hidden_state_loss: tp.Literal["mse"] = field(
        default="mse",
        metadata={"help": "Distance function used for hidden-state distillation. Currently only 'mse' is supported."},
    )
    attention_loss_weight: float = field(
        default=0.0,
        metadata={
            "help": (
                "Optional coefficient for matching attention probability tensors. "
                "Set to 0 to disable attention-head distillation."
            )
        },
    )
    attention_layers: tuple[int, ...] | None = field(
        default=None,
        metadata={
            "help": (
                "Indices of attention layers whose probability matrices should be distilled. "
                "Negative indices follow Python semantics. Defaults to all available layeatrs when omitted."
            )
        },
    )
    attention_normalize: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to L1-normalize attention matrices before computing the distillation loss. "
                "Useful when working with models that emit un-normalized attention weights."
            )
        },
    )

    def __post_init__(self, max_sequence_length: int | None):
        if self.hidden_state_layers is not None:
            self.hidden_state_layers = tuple(int(i) for i in self.hidden_state_layers)
        if self.attention_layers is not None:
            self.attention_layers = tuple(int(i) for i in self.attention_layers)
        if hasattr(super(), "__post_init__"):
            super().__post_init__(max_sequence_length=max_sequence_length)

    __hash__ = hash_fn
