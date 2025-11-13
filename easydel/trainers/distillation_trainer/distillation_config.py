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
from dataclasses import field

from eformer.pytree import auto_pytree

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..training_configurations import TrainingArguments


@Registry.register("trainer-arguments", "distillation")
@auto_pytree
class DistillationConfig(TrainingArguments):
    """Configuration class for knowledge distillation training.

    This configuration extends TrainingArguments with parameters specific to
    knowledge distillation, where a smaller student model learns to mimic
    a larger teacher model's behavior.

    Supports multiple distillation strategies:
    - Logit distillation: Temperature-scaled KL divergence on output logits
    - Attention transfer: Matching attention patterns between teacher and student
    - Feature matching: Matching intermediate hidden representations

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
        use_attention_transfer (bool): Enable attention transfer distillation.
            Minimizes cosine distance between student and teacher attention maps.
            Default: False
        attention_loss_weight (float): Weight for attention transfer loss.
            Only used if use_attention_transfer=True. Default: 0.1
        attention_match_layers (tuple[int, ...] | None): Layer indices to match
            attention maps. If None, matches all layers. Example: (6, 12, 18)
            Default: None (all layers)
        use_feature_matching (bool): Enable feature matching distillation.
            Minimizes MSE between student and teacher hidden states.
            Default: False
        feature_loss_weight (float): Weight for feature matching loss.
            Only used if use_feature_matching=True. Default: 0.1
        feature_match_layers (tuple[int, ...] | None): Layer indices to match
            hidden states. If None, matches all layers. Example: (6, 12, 18)
            Default: None (all layers)

    Example:
        >>> # Standard logit distillation
        >>> config = DistillationConfig(
        ...     temperature=5.0,
        ...     alpha=0.7,
        ...     learning_rate=1e-4,
        ...     num_train_epochs=10
        ... )
        >>>
        >>> # With attention transfer
        >>> config = DistillationConfig(
        ...     temperature=2.0,
        ...     alpha=0.8,
        ...     use_attention_transfer=True,
        ...     attention_loss_weight=0.1,
        ...     attention_match_layers=(6, 12, 18),
        ...     learning_rate=1e-4
        ... )
        >>>
        >>> # With both attention and feature matching
        >>> config = DistillationConfig(
        ...     temperature=2.0,
        ...     alpha=0.7,
        ...     use_attention_transfer=True,
        ...     attention_loss_weight=0.1,
        ...     use_feature_matching=True,
        ...     feature_loss_weight=0.2,
        ...     learning_rate=1e-4
        ... )

    Note:
        The total distillation loss is computed as:
        Loss = alpha * KL(student/T, teacher/T) + (1-alpha) * CE(student, labels)
               + attention_weight * AttentionLoss + feature_weight * FeatureLoss
        where T is the temperature parameter.

        When teacher and student have different hidden dimensions or number of layers,
        automatic pooling is applied to match shapes.
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
    use_attention_transfer: bool = field(
        default=False,
        metadata={
            "help": "Enable attention transfer distillation. Minimizes cosine distance "
            "between student and teacher attention maps."
        },
    )
    attention_loss_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for attention transfer loss. Only used if use_attention_transfer=True."
        },
    )
    attention_match_layers: tuple[int, ...] | None = field(
        default=None,
        metadata={
            "help": "Layer indices to match attention maps. If None, matches all layers. "
            "Example: (6, 12, 18) to match layers 6, 12, and 18."
        },
    )
    use_feature_matching: bool = field(
        default=False,
        metadata={
            "help": "Enable feature matching distillation. Minimizes MSE between "
            "student and teacher hidden states."
        },
    )
    feature_loss_weight: float = field(
        default=0.1,
        metadata={
            "help": "Weight for feature matching loss. Only used if use_feature_matching=True."
        },
    )
    feature_match_layers: tuple[int, ...] | None = field(
        default=None,
        metadata={
            "help": "Layer indices to match hidden states. If None, matches all layers. "
            "Example: (6, 12, 18) to match layers 6, 12, and 18."
        },
    )
    __hash__ = hash_fn
