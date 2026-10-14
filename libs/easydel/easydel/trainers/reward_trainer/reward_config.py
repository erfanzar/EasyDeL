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
"""Configuration dataclass for the reward-model trainer.

Defines :class:`RewardConfig`, which extends :class:`TrainingArguments`
with reward-model-specific knobs: maximum sequence length, an optional
reward-centring penalty (so the predicted scalar score has zero mean),
dropout disabling, and the dataset preprocessing worker count.
"""

import warnings
from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..training_configurations import TrainingArguments


@Registry.register("trainer-arguments", "reward")
@dataclass
class RewardConfig(TrainingArguments):
    """Configuration class for Reward Model training.

    Reward models are crucial components in RLHF pipelines, learning to predict
    human preferences between different model outputs. The trained reward model
    serves as a proxy for human judgment, providing feedback signals for
    policy optimization.

    This configuration extends TrainingArguments with parameters specific to
    training reward models using pairwise preference data. The model learns
    to assign higher scores to preferred (chosen) responses compared to
    non-preferred (rejected) responses.

    Key concepts:
    - Bradley-Terry model: P(chosen > rejected) = sigmoid(r_chosen - r_rejected)
    - Margin-based losses: Optionally enforce minimum score differences
    - Reward centering: Regularization to maintain mean-zero rewards

    Attributes:
        trainer_prefix (str | None): Prefix for trainer logs and checkpoints.
            Default: "rewardtrainer"
        max_length (int | None): Maximum length of sequences (prompt + completion).
            Sequences exceeding this limit are filtered out. Default: 1024
        disable_dropout (bool): Whether to disable dropout during training for
            more deterministic behavior. Recommended for reward models. Default: True
        dataset_num_proc (int | None): Number of processes for parallel dataset
            preprocessing. None uses sequential processing. Default: None
        center_rewards_coefficient (float | None): Coefficient for reward centering
            regularization. Encourages the model to output mean-zero rewards,
            preventing reward drift. Default: 0.1
        remove_unused_columns (bool | None): Whether to remove columns not used
            by the model's forward pass. Only set True if dataset is pretokenized.
            Default: False

    Example:
        >>> config = RewardConfig(
        ...     max_length=2048,
        ...     center_rewards_coefficient=0.01,
        ...     learning_rate=2e-5,
        ...     num_train_epochs=1
        ... )

    Note:
        The reward model typically uses the same architecture as the base LLM
        but with a scalar reward head instead of the language modeling head.
        Training requires paired preference data with chosen and rejected examples.
    """

    trainer_prefix: str | None = field(
        default="Reward",
        metadata={"help": "default prefix name for trainer."},
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "The initial learning rate for reward-model training."},
    )
    chat_template_path: str | None = field(
        default=None,
        metadata={
            "help": (
                "Optional local Jinja template path or tokenizer source whose chat template "
                "should be copied onto the processing class."
            )
        },
    )
    max_length: int | None = field(
        default=1024,
        metadata={
            "help": "Maximum length of the sequences (prompt + completion) in the batch, "
            "filters out entries that exceed the limit."
        },
    )
    eos_token: str | None = field(
        default=None,
        metadata={"help": "Optional EOS token override applied to the processing class."},
    )
    pad_to_multiple_of: int | None = field(
        default=None,
        metadata={"help": "If set, reward sequences are padded to a multiple of this value."},
    )
    disable_dropout: bool = field(
        default=True,
        metadata={"help": "Whether to disable dropout in the model."},
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes to use for processing the dataset."},
    )
    center_rewards_coefficient: float | None = field(
        default=0.1,
        metadata={"help": "Coefficient to incentivize the reward model to output mean-zero rewards."},
    )
    activation_offloading: bool = field(
        default=False,
        metadata={"help": "TRL compatibility field. EasyDeL RewardTrainer does not support activation offloading."},
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={
            "help": "Whether to remove the columns that are not used by the model's forward pass. Can be `True` "
            "only if the dataset is pretokenized."
        },
    )
    pad_token: str | None = field(
        default=None,
        metadata={
            "help": (
                "Deprecated tokenizer pad-token override. Prefer setting the processing class "
                "before constructing the trainer."
            )
        },
    )

    def __post_init__(
        self,
        max_sequence_length: int | None,
        quantization_block: int | None,
    ):
        if self.pad_to_multiple_of is not None and self.pad_to_multiple_of <= 0:
            raise ValueError("`pad_to_multiple_of` must be a positive integer when set.")
        if self.pad_token is not None:
            warnings.warn(
                "`pad_token` is deprecated. Set `processing_class.pad_token` before constructing RewardTrainer.",
                FutureWarning,
                stacklevel=3,
            )
        if hasattr(super(), "__post_init__"):
            super().__post_init__(
                max_sequence_length=max_sequence_length,
                quantization_block=quantization_block,
            )

    __hash__ = hash_fn
