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

"""Configuration for Compaction Reinforcement Learning (CompactionRL)."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from .._online_rl.config import OnlineActorCriticConfig

DEFAULT_SUMMARY_INSTRUCTION = (
    "Summarize the completed work, important observations, tool results, "
    "remaining goals, and constraints so the task can continue from a shorter context."
)
DEFAULT_RESUME_TEMPLATE = (
    "The previous context was compacted. Continue the original task using this summary:\n\n{summary}"
)


@Registry.register(
    "trainer-arguments",
    ["compaction_rl", "compaction-rl", "compaction_reinforcement_learning"],
)
@dataclass
class CompactionRLConfig(OnlineActorCriticConfig):
    """Hyperparameters for critic-based CompactionRL.

    CompactionRL trains the same policy to both execute an agentic task and
    summarize its context when the remaining token budget falls below a
    threshold.  An assistant action and its environment observation are an
    atomic pair: compaction is checked only after the observation arrives.
    The resumed context contains the original system message, a summary
    template, and the newest complete action/observation pairs that fit.

    Execution and summary segments receive the same final task reward.  There
    is no built-in summary-specific reward.  Actor loss is PPO-style and
    normalized once over all generated assistant tokens.  Local segment GAE is
    corrected by ``(gamma * lambda) ** N_after``, where ``N_after`` counts only
    optimized tokens in later segments of the same trajectory.

    Public sources do not disclose the exact GLM-5.2 summary instruction or
    resume template.  The defaults below are explicit EasyDeL templates and
    can be replaced without changing the algorithm.

    Attributes:
        total_batch_size: Global rollout batch size.
        weight_decay: Decoupled AdamW decay. Zero reproduces Adam.
        context_budget: Maximum context tokens available to the agent.
        compaction_threshold: Compact when
            ``context_budget - current_tokens < compaction_threshold``.
        max_compactions: Maximum summaries generated in one trajectory.
        recent_steps: Preferred number of complete action/observation pairs
            retained after compaction.
        max_assistant_tokens: Per-turn assistant response cap.
        max_summary_tokens: Summary generation cap.
        max_turns: Maximum environment turns in a trajectory.
        summary_instruction: Instruction appended to request a summary.
        resume_template: Template containing a ``{summary}`` placeholder.
        train_summary_tokens: Include sampled summary tokens in actor loss.
        cross_trajectory_gae: Apply the published cross-segment correction.
        compaction_failure_policy: Behavior when no reconstructed context fits.
        guard_failure_policy: Default behavior if the intent judge fails.
    """

    trainer_prefix: str | None = field(
        default="CompactionRL",
        metadata={"help": "Default prefix for CompactionRL logs and checkpoints."},
    )
    learning_rate: float = field(
        default=2e-6,
        metadata={"help": "Actor learning rate used by the public CompactionRL recipe."},
    )
    total_batch_size: int = field(
        default=128,
        metadata={"help": "Global rollout batch size used by the public CompactionRL experiments."},
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Zero decay makes EasyDeL's AdamW implementation equivalent to the published Adam update."},
    )
    kl_coef: float = field(
        default=0.0,
        metadata={
            "help": "Optional reference-policy KL reward coefficient; zero matches the published CompactionRL objective."
        },
    )
    critic_learning_rate: float = field(
        default=3e-6,
        metadata={"help": "Critic learning rate used by the public CompactionRL recipe."},
    )
    critic_pretrain_steps: int = field(
        default=50,
        metadata={"help": "Critic-only warmup updates before the first actor update."},
    )
    critic_updates_per_actor_update: int = field(
        default=2,
        metadata={"help": "Number of critic updates before every actor update."},
    )
    critic_trainable_mode: tp.Literal["full_except_attention", "all"] = field(
        default="all",
        metadata={"help": "CompactionRL trains the complete critic by default."},
    )
    critic_gae_mode: tp.Literal["fixed", "match_policy"] = field(
        default="match_policy",
        metadata={"help": "Use the same length-adaptive lambda for critic targets and policy advantages."},
    )
    max_prompt_length: int = field(
        default=55_296,
        metadata={"help": "Maximum initial prompt length for the 64K public recipe."},
    )
    max_completion_length: int = field(
        default=10_240,
        metadata={"help": "Maximum assistant response length used by the public recipe."},
    )
    num_return_sequences: int = field(
        default=1,
        metadata={"help": "CompactionRL uses group size one."},
    )
    num_generations: int | None = field(
        default=1,
        metadata={"help": "Alias for group size; fixed to one."},
    )
    num_ppo_epochs: int = field(
        default=1,
        metadata={"help": "Number of PPO actor passes over each completed trajectory batch."},
    )
    context_budget: int = field(
        default=65_536,
        metadata={"help": "Total context-token budget C."},
    )
    compaction_threshold: int = field(
        default=10_240,
        metadata={"help": "Compact when the remaining context is strictly below this threshold."},
    )
    max_compactions: int = field(
        default=3,
        metadata={"help": "Maximum number of compactions in one trajectory."},
    )
    recent_steps: int = field(
        default=2,
        metadata={"help": "Preferred number of recent atomic action/observation pairs to retain."},
    )
    max_assistant_tokens: int = field(
        default=10_240,
        metadata={"help": "Maximum tokens sampled for one assistant action."},
    )
    max_summary_tokens: int = field(
        default=2_048,
        metadata={"help": "Maximum tokens sampled for one compaction summary."},
    )
    max_turns: int = field(
        default=250,
        metadata={"help": "Maximum environment turns in one trajectory."},
    )
    summary_instruction: str = field(
        default=DEFAULT_SUMMARY_INSTRUCTION,
        metadata={"help": "Instruction used to request a policy-generated summary."},
    )
    resume_template: str = field(
        default=DEFAULT_RESUME_TEMPLATE,
        metadata={"help": "Resume template containing a `{summary}` placeholder."},
    )
    train_summary_tokens: bool = field(
        default=True,
        metadata={"help": "Whether sampled summary tokens participate in the actor loss."},
    )
    cross_trajectory_gae: bool = field(
        default=True,
        metadata={"help": "Apply the published cross-segment GAE correction."},
    )
    token_level_loss: bool = field(
        default=True,
        metadata={"help": "Normalize once over all optimized assistant tokens."},
    )
    compaction_failure_policy: tp.Literal["keep_context", "raise"] = field(
        default="raise",
        metadata={"help": "Policy when no valid reconstructed context fits: keep_context or raise."},
    )
    guard_failure_policy: tp.Literal["block", "allow"] = field(
        default="block",
        metadata={"help": "Policy when the optional intent judge fails: block or allow."},
    )

    def __post_init__(
        self,
        max_sequence_length: int | None,
        quantization_block: int | None,
    ) -> None:
        """Normalize inherited fields and enforce CompactionRL invariants."""
        super().__post_init__(
            max_sequence_length=max_sequence_length,
            quantization_block=quantization_block,
        )
        if self.num_return_sequences != 1 or self.num_generations != 1:
            raise ValueError(
                "CompactionRL uses group size one: `num_return_sequences` and `num_generations` must both be 1."
            )
        if self.context_budget <= 0:
            raise ValueError("`context_budget` must be positive.")
        if not 0 < self.compaction_threshold < self.context_budget:
            raise ValueError("`compaction_threshold` must be positive and smaller than `context_budget`.")
        if self.max_compactions < 0:
            raise ValueError("`max_compactions` must be non-negative.")
        if self.recent_steps < 0:
            raise ValueError("`recent_steps` must be non-negative.")
        if self.max_assistant_tokens <= 0:
            raise ValueError("`max_assistant_tokens` must be positive.")
        if self.max_summary_tokens <= 0:
            raise ValueError("`max_summary_tokens` must be positive.")
        if self.max_turns <= 0:
            raise ValueError("`max_turns` must be positive.")
        if "{summary}" not in self.resume_template:
            raise ValueError("`resume_template` must contain a `{summary}` placeholder.")
        if self.compaction_failure_policy not in ("keep_context", "raise"):
            raise ValueError("`compaction_failure_policy` must be 'keep_context' or 'raise'.")
        if self.guard_failure_policy not in ("block", "allow"):
            raise ValueError("`guard_failure_policy` must be 'block' or 'allow'.")
        if not self.token_level_loss:
            raise ValueError("CompactionRL requires token-level global loss normalization.")
        if not self.global_token_normalization:
            raise ValueError("CompactionRL requires `global_token_normalization=True`.")

    __hash__ = hash_fn


__all__ = (
    "DEFAULT_RESUME_TEMPLATE",
    "DEFAULT_SUMMARY_INSTRUCTION",
    "CompactionRLConfig",
)
