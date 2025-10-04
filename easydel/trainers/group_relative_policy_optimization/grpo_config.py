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
    epsilon_low: float = field(
        default=0.2,
        metadata={
            "help": "Lower clipping bound (1-ε_low) for the PPO style ratio used in GRPO updates."
        },
    )
    epsilon_high: float = field(
        default=0.2,
        metadata={
            "help": "Upper clipping bound (1+ε_high) for the PPO style ratio used in GRPO updates."
        },
    )
    per_token_weighting: bool = field(
        default=True,
        metadata={
            "help": "If True, scale each token's loss contribution by the inverse completion length (DAPO style)."
        },
    )
    adv_estimator: tp.Literal["group", "gae", "truncated"] = field(
        default="group",
        metadata={
            "help": "Select the advantage estimator. 'group' performs z-score normalization within generations, "
            "'gae' uses Monte Carlo rewards with (γ, λ), and 'truncated' applies a k-step return."
        },
    )
    gae_gamma: float = field(
        default=0.99,
        metadata={"help": "Discount factor γ used when adv_estimator is 'gae' or 'truncated'."},
    )
    gae_lambda: float = field(
        default=0.95,
        metadata={"help": "Trace parameter λ for GAE style advantage estimation."},
    )
    truncated_return_k: int = field(
        default=1,
        metadata={
            "help": "Number of steps for truncated returns when adv_estimator='truncated'. 1 reduces to Monte Carlo."
        },
    )
    z_score_epsilon: float = field(
        default=1e-4,
        metadata={"help": "Numerical stabilizer added to the group standard deviation when normalising rewards."},
    )
    length_shaping: tp.Literal["none", "linear", "punitive"] = field(
        default="none",
        metadata={
            "help": "Reward shaping applied based on completion length. 'linear' matches DAPO, 'punitive' applies a "
            "strong penalty for overlong completions."
        },
    )
    length_cache_tokens: int | None = field(
        default=None,
        metadata={
            "help": "Cache span used by the length shaping schedule. Defaults to 10% of max_completion_length."
        },
    )
    length_reward_scale: float = field(
        default=1.0,
        metadata={"help": "Scaling factor applied to any additional length shaping reward."},
    )
    enforce_mixed_sampling: bool = field(
        default=True,
        metadata={
            "help": "Ensure each prompt group contains diversified rewards. When variance collapses, advantages are "
            "jittered to avoid zero gradients."
        },
    )
    dynamic_sampling_max_tries: int = field(
        default=0,
        metadata={
            "help": "Number of additional regeneration attempts when enforce_mixed_sampling fails. 0 disables retries."
        },
    )
    dynamic_sampling_jitter: float = field(
        default=1e-3,
        metadata={
            "help": "Magnitude of the fallback perturbation injected into collapsed reward groups."
        },
    )
    positive_reward_threshold: float = field(
        default=0.0,
        metadata={
            "help": "Threshold used to mark a completion as positive when computing sampling diagnostics."
        },
    )
    kl_target: float | None = field(
        default=None,
        metadata={
            "help": "Optional moving-average KL target. When exceeded, the reference policy reset logic is triggered."
        },
    )
    kl_horizon: int = field(
        default=100,
        metadata={"help": "Horizon (in steps) for the exponential moving average used when tracking KL."},
    )
    reference_reset_steps: int | None = field(
        default=None,
        metadata={
            "help": "Force a reference reset every N steps regardless of KL. None disables the periodic trigger."
        },
    )
    reference_reset_style: tp.Literal["none", "hard", "mix"] = field(
        default="hard",
        metadata={
            "help": "Strategy for refreshing the reference model. 'hard' copies the policy, 'mix' interpolates using "
            "ref_model_mixup_alpha, 'none' keeps the current reference."},
    )
    entropy_floor: float | None = field(
        default=None,
        metadata={
            "help": "Optional minimum token entropy. If the running entropy drops below this floor the trainer will "
            "report it and may trigger a reference reset when combined with kl_target."
        },
    )

    def __post_init__(self):
        """Post initialization to set dependent parameters."""
        self.max_sequence_length = self.max_prompt_length + self.max_completion_length

        if hasattr(super(), "__post_init__"):
            super().__post_init__()

        if self.length_cache_tokens is None:
            self.length_cache_tokens = max(1, int(0.1 * self.max_completion_length))
        if self.truncated_return_k < 1:
            self.truncated_return_k = 1

    __hash__ = hash_fn
