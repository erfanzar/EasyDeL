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
import typing as tp
from dataclasses import field

from eformer.pytree import auto_pytree

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..training_configurations import TrainingArguments


@Registry.register("trainer-arguments", "grpo")
@auto_pytree
class GRPOConfig(TrainingArguments):
    """Configuration class for Group Relative Policy Optimization training.

    GRPO is an efficient RLHF algorithm that optimizes policies using group-based
    relative comparisons of rewards. It provides better training stability compared
    to standard PPO by normalizing rewards within groups of samples.

    This configuration extends TrainingArguments with GRPO-specific parameters
    for controlling the policy optimization process, reward computation, and
    generation sampling strategies.

    Key concepts:
    - Group-based normalization: Rewards are normalized within groups to reduce variance
    - KL regularization: Prevents the policy from deviating too far from reference
    - Reference model syncing: Optionally updates reference model during training
    """

    trainer_prefix: str | None = field(
        default="grpotrainer",
        metadata={"help": "default prefix name for trainer."},
    )
    remove_unused_columns: bool | None = field(
        default=False,
        metadata={"help": "Whether to remove unused columns from the dataset."},
    )
    max_prompt_length: int = field(
        default=512,
        metadata={"help": "The maximum length of the prompt."},
    )
    max_completion_length: int = field(
        default=256,
        metadata={"help": "The maximum length of the completion."},
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "The number of processes to use for dataset processing."},
    )
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "The learning rate."},
    )
    beta: float = field(
        default=0.04,
        metadata={"help": "The beta parameter for GRPO."},
    )
    sync_ref_model: bool = field(
        default=False,
        metadata={"help": "Whether to periodically sync the reference model with the policy model."},
    )
    ref_model_mixup_alpha: float = field(
        default=0.9,
        metadata={"help": "The alpha parameter for mixing the reference model with the policy model."},
    )
    ref_model_sync_steps: int = field(
        default=64,
        metadata={"help": "The number of steps between syncing the reference model."},
    )
    tools: list[dict | tp.Callable] | None = field(
        default=None,
        metadata={"help": "Additional tools for training."},
    )
    skip_apply_chat_template: bool = field(
        default=False,
        metadata={"help": "whenever to skip extracting prompt from dataset."},
    )
    num_return_sequences: int = field(
        default=False,
        metadata={
            "help": "The number of sequences to return for each input prompt. Used during sampling to "
            "generate multiple completions per prompt."
        },
    )

    top_p: float = field(
        default=0.95,
        metadata={
            "help": "Top-p (nucleus) sampling threshold. Tokens are sampled from the smallest possible set whose "
            "cumulative probability exceeds this value."
        },
    )

    top_k: int = field(
        default=50,
        metadata={"help": "Top-k sampling threshold. Limits sampling to the top-k most probable tokens at each step."},
    )

    temperature: float = field(
        default=0.7,
        metadata={
            "help": "Sampling temperature. Higher values (e.g., >1.0) produce more random outputs, while "
            "lower values (e.g., <1.0) make the output more deterministic."
        },
    )

    def __post_init__(self):
        """Post initialization to set dependent parameters."""
        self.max_sequence_length = self.max_prompt_length + self.max_completion_length

        if hasattr(super(), "__post_init__"):
            super().__post_init__()

    __hash__ = hash_fn
