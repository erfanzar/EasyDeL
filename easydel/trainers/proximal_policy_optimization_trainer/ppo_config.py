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

"""Configuration classes for PPO training.

This module defines PPOConfig, a dataclass that holds all hyperparameters
for Proximal Policy Optimization training including:
- Prompt/completion length limits
- PPO-specific parameters (clip range, KL coefficient, GAE lambda)
- Value function training parameters
- Generation sampling parameters
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..training_configurations import TrainingArguments


@Registry.register("trainer-arguments", "ppo")
@dataclass
class PPOConfig(TrainingArguments):
    """Configuration class for Proximal Policy Optimization (PPO) training.

    This config is intended for RLHF-style PPO on language models where:
    - Prompts are provided by the dataset (left-padded)
    - Completions are generated online
    - A reward function/model scores (prompt, completion)
    - A value head is trained as a baseline, and PPO clipping stabilizes updates
    """

    trainer_prefix: str | None = field(
        default="ppotrainer",
        metadata={"help": "Default prefix name for trainer outputs/checkpoints."},
    )
    remove_unused_columns: bool | None = field(
        default=False,
        metadata={"help": "Whether to remove unused columns from the dataset."},
    )
    max_prompt_length: int = field(
        default=512,
        metadata={"help": "The maximum length of the prompt (left-padded)."},
    )
    max_completion_length: int = field(
        default=256,
        metadata={"help": "The maximum length of the completion (right-padded)."},
    )
    dataset_num_proc: int | None = field(
        default=None,
        metadata={"help": "Number of processes for dataset preprocessing."},
    )
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "Learning rate for PPO fine-tuning."},
    )
    num_ppo_epochs: int = field(
        default=1,
        metadata={"help": "Number of PPO epochs per rollout batch. (Currently best-effort; keep 1 for parity.)"},
    )
    kl_coef: float = field(
        default=0.05,
        metadata={"help": "KL coefficient for the non-score reward: r_kl = -kl_coef * KL(pi||ref)."},
    )
    kl_estimator: tp.Literal["k1", "k3"] = field(
        default="k1",
        metadata={"help": "KL estimator to use ('k1' or 'k3')."},
    )
    cliprange: float = field(
        default=0.2,
        metadata={"help": "PPO clip range for the policy objective."},
    )
    vf_coef: float = field(
        default=0.1,
        metadata={"help": "Value-function loss coefficient."},
    )
    cliprange_value: float = field(
        default=0.2,
        metadata={"help": "Value-function clip range."},
    )
    gamma: float = field(
        default=1.0,
        metadata={"help": "Discount factor for GAE/returns."},
    )
    lam: float = field(
        default=0.95,
        metadata={"help": "Lambda for GAE."},
    )
    whiten_rewards: bool = field(
        default=False,
        metadata={"help": "Whether to whiten (scale) token rewards before GAE."},
    )
    whiten_advantages: bool = field(
        default=True,
        metadata={"help": "Whether to normalize advantages to zero mean / unit variance."},
    )
    entropy_coef: float = field(
        default=0.0,
        metadata={"help": "Optional entropy bonus coefficient (0 disables)."},
    )
    missing_eos_penalty: float | None = field(
        default=None,
        metadata={"help": "Optional penalty subtracted from score when no EOS is generated."},
    )
    tools: list[dict | tp.Callable] | None = field(
        default=None,
        metadata={"help": "Additional tools for chat-template function calling."},
    )
    reward_weights: list[float] | None = field(
        default=None,
        metadata={
            "help": "Optional weights for each reward function. Must match the number of reward functions if set."
        },
    )
    skip_apply_chat_template: bool = field(
        default=False,
        metadata={"help": "If True, skip extracting prompt from dataset via chat template."},
    )
    num_return_sequences: int = field(
        default=1,
        metadata={"help": "Number of completions per prompt during generation."},
    )
    num_generations: int | None = field(
        default=None,
        metadata={"help": "Alias for num_return_sequences to keep parity with other trainers."},
    )
    temperature: float = field(
        default=0.7,
        metadata={"help": "Sampling temperature for rollout generation."},
    )
    top_p: float = field(
        default=0.95,
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
        metadata={"help": "Additional generation kwargs forwarded to generation config."},
    )
    chat_template_kwargs: dict | None = field(
        default=None,
        metadata={"help": "Extra kwargs forwarded to chat template application during generation."},
    )
    mask_truncated_completions: bool = field(
        default=False,
        metadata={"help": "If True, drop completions that do not terminate with EOS from loss calculation."},
    )

    def __post_init__(self, max_sequence_length: int | None):
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

        if hasattr(super(), "__post_init__"):
            super().__post_init__(max_sequence_length=None)

    __hash__ = hash_fn
