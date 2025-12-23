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
from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..training_configurations import TrainingArguments


@Registry.register("trainer-arguments", "grpo")
@dataclass
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
    epsilon: float = field(
        default=0.2,
        metadata={"help": "Lower clipping bound for importance sampling weights."},
    )
    epsilon_high: float | None = field(
        default=None,
        metadata={"help": "Upper clipping bound for importance sampling weights. If None, defaults to `epsilon`."},
    )
    delta: float | None = field(
        default=None,
        metadata={
            "help": "Optional two-sided clipping bound. If set, importance weights are additionally clipped to `delta`."
        },
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
    num_iterations: int = field(
        default=1,
        metadata={"help": "How many optimizer updates to perform per generated batch."},
    )
    loss_type: str = field(
        default="dapo",
        metadata={"help": "Loss variant to use. One of ['grpo', 'bnpo', 'dr_grpo', 'dapo', 'cispo']."},
    )
    importance_sampling_level: str = field(
        default="token",
        metadata={"help": "Importance sampling applied per 'token' or aggregated per 'sequence'."},
    )
    reward_weights: list[float] | None = field(
        default=None,
        metadata={
            "help": "Optional weights for each reward function. Must match the number of reward functions if set."
        },
    )
    scale_rewards: str | bool = field(
        default="group",
        metadata={
            "help": "Reward scaling strategy: 'group', 'batch', 'none', or the booleans True/False for group/none."
        },
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
        default=4,
        metadata={
            "help": "The number of sequences to return for each input prompt. Used during sampling to "
            "generate multiple completions per prompt."
        },
    )
    num_generations: int | None = field(
        default=None,
        metadata={"help": "Alias for num_return_sequences to keep parity with TRL's interface."},
    )
    temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature used during generation."},
    )
    top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p nucleus sampling parameter."},
    )
    top_k: int | None = field(
        default=None,
        metadata={"help": "Top-k sampling parameter. None disables top-k."},
    )
    min_p: float | None = field(
        default=None,
        metadata={"help": "Minimum token probability threshold (see HF top-p-min sampling)."},
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={"help": "Repetition penalty applied during generation."},
    )
    generation_kwargs: dict | None = field(
        default=None,
        metadata={"help": "Additional generation kwargs forwarded to the generation config."},
    )
    chat_template_kwargs: dict | None = field(
        default=None,
        metadata={"help": "Extra kwargs forwarded to chat template application during generation."},
    )
    mask_truncated_completions: bool = field(
        default=False,
        metadata={"help": "If True, drop completions that do not terminate with EOS from the loss calculation."},
    )
    top_entropy_quantile: float = field(
        default=1.0,
        metadata={"help": "Keep only the top quantile of tokens by entropy in the loss (1.0 disables filtering)."},
    )

    def __post_init__(self, max_sequence_length: int | None):
        """Post initialization to set dependent parameters."""
        self._handle_deprecated_max_sequence_length(max_sequence_length)

        if self.max_length is not None:
            if self.max_length < self.max_prompt_length:
                raise ValueError(
                    f"`max_length` ({self.max_length}) must be >= `max_prompt_length` ({self.max_prompt_length})."
                )
            self.max_completion_length = self.max_length - self.max_prompt_length

        self.max_length = self.max_prompt_length + self.max_completion_length

        if self.num_generations is None:
            self.num_generations = self.num_return_sequences
        else:
            self.num_return_sequences = self.num_generations

        if self.epsilon_high is None:
            self.epsilon_high = self.epsilon

        if self.scale_rewards is True:
            self.scale_rewards = "group"
        elif self.scale_rewards is False:
            self.scale_rewards = "none"

        if hasattr(super(), "__post_init__"):
            super().__post_init__(max_sequence_length=None)

    __hash__ = hash_fn
