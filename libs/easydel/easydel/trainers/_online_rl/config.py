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

"""Shared configuration contract for critic-based online-RL trainers."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field

from easydel.utils.compiling_utils import hash_fn

from ..proximal_policy_optimization_trainer import PPOConfig


@dataclass
class OnlineActorCriticConfig(PPOConfig):
    """Common actor/critic and rollout settings used by online-RL trainers.

    Unlike :class:`~easydel.trainers.PPOConfig`, the online actor/critic
    substrate owns a distinct critic state, optimizer, scheduler, and
    checkpoint.  The policy and critic can therefore use different learning
    rates and update cadences.  Target masks are aligned with
    ``input_ids[:, 1:]`` and distinguish generated actions from attended
    observations.

    This class is internal and intentionally is not registered as a trainer
    configuration.  Public algorithms subclass it and register their own
    stable names.

    Attributes:
        critic_learning_rate: Optimizer learning rate for the critic.
        critic_learning_rate_end: Optional final critic learning rate.
        critic_optimizer: Optional optimizer name overriding the actor
            optimizer for the critic.
        critic_scheduler: Optional scheduler name overriding the actor
            scheduler for the critic.
        critic_warmup_steps: Scheduler warmup steps for the critic.
        critic_updates_per_actor_update: Number of critic updates performed
            before each actor update.
        critic_pretrain_steps: Extra critic-only updates performed before the
            first actor update.
        critic_cliprange_value: Symmetric value-prediction clipping range.
        critic_gae_lambda: GAE trace decay used to construct critic targets.
        critic_gae_mode: Whether critic targets use the fixed
            ``critic_gae_lambda`` or the same per-trajectory lambda as policy
            advantages.
        critic_trainable_mode: Default structural critic selector.  The
            ``"full_except_attention"`` mode freezes parameters whose
            canonical paths contain ``attention`` or ``attn``; ``"all"``
            trains all critic parameters.
        policy_gae_mode: Whether policy GAE uses the fixed ``lam`` inherited
            from PPO or the length-adaptive GLM formula.
        length_adaptive_alpha: ``alpha`` in
            ``lambda = 1 - 1 / (alpha * action_token_count)``.
        global_token_normalization: Divide losses by the number of action
            tokens in the complete batch, including under gradient
            accumulation.
        rollout_logprob_source: Source of behavior-policy log probabilities.
            ``"recompute"`` scores an immutable rollout snapshot;
            ``"engine"`` requires token log probabilities supplied by the
            rollout provider.
        max_inflight_rollouts: Upper bound for asynchronous rollout workers.
        rollout_queue_size: Maximum completed-rollout queue size.  Zero uses
            ``max_inflight_rollouts``.
        weight_sync_steps: Actor updates between rollout-policy snapshots.
        max_policy_staleness: Maximum accepted behavior-policy lag. ``None``
            disables learner-side rejection.
        rollout_timeout: Optional timeout in seconds for a rollout.
    """

    critic_learning_rate: float = field(
        default=5e-6,
        metadata={"help": "Learning rate for the separately optimized critic."},
    )
    critic_learning_rate_end: float | None = field(
        default=None,
        metadata={"help": "Optional final learning rate for the critic scheduler."},
    )
    critic_optimizer: str | None = field(
        default=None,
        metadata={"help": "Optional critic optimizer override; defaults to the actor optimizer."},
    )
    critic_scheduler: str | None = field(
        default=None,
        metadata={"help": "Optional critic scheduler override; defaults to the actor scheduler."},
    )
    critic_warmup_steps: int = field(
        default=0,
        metadata={"help": "Number of warmup steps for the critic learning-rate schedule."},
    )
    critic_updates_per_actor_update: int = field(
        default=2,
        metadata={"help": "Number of critic optimizer steps before each actor optimizer step."},
    )
    critic_pretrain_steps: int = field(
        default=0,
        metadata={"help": "Extra critic-only updates performed before the first actor update."},
    )
    critic_cliprange_value: float = field(
        default=0.2,
        metadata={"help": "Symmetric clipping range for critic value updates."},
    )
    critic_gae_lambda: float = field(
        default=1.0,
        metadata={"help": "GAE lambda used to build fixed critic regression targets."},
    )
    critic_gae_mode: tp.Literal["fixed", "match_policy"] = field(
        default="fixed",
        metadata={"help": "Use a fixed critic lambda or match the policy's per-trajectory lambda."},
    )
    critic_trainable_mode: tp.Literal["full_except_attention", "all"] = field(
        default="full_except_attention",
        metadata={"help": "Structural trainable selector used when initializing the critic from the actor."},
    )
    policy_gae_mode: tp.Literal["fixed", "length_adaptive"] = field(
        default="length_adaptive",
        metadata={"help": "Use a fixed lambda or the length-adaptive GLM lambda for actor advantages."},
    )
    length_adaptive_alpha: float = field(
        default=1.5,
        metadata={"help": "Alpha in lambda = 1 - 1 / (alpha * number_of_action_tokens)."},
    )
    global_token_normalization: bool = field(
        default=True,
        metadata={"help": "Normalize actor and critic losses over all action tokens in the full batch."},
    )
    rollout_logprob_source: tp.Literal["recompute", "engine"] = field(
        default="recompute",
        metadata={"help": "Source of token-aligned behavior-policy log probabilities."},
    )
    max_inflight_rollouts: int = field(
        default=32,
        metadata={"help": "Maximum number of asynchronous rollouts in flight."},
    )
    rollout_queue_size: int = field(
        default=0,
        metadata={"help": "Bound on completed rollouts waiting for the learner; zero follows max_inflight_rollouts."},
    )
    weight_sync_steps: int = field(
        default=1,
        metadata={"help": "Actor steps between rollout-policy snapshot refreshes."},
    )
    max_policy_staleness: int | None = field(
        default=None,
        metadata={"help": "Maximum accepted behavior-policy lag in actor steps; None disables rejection."},
    )
    rollout_timeout: float | None = field(
        default=120.0,
        metadata={"help": "Optional per-rollout timeout in seconds."},
    )

    def __post_init__(
        self,
        max_sequence_length: int | None,
        quantization_block: int | None,
    ) -> None:
        """Normalize PPO fields and validate the shared actor/critic contract."""
        super().__post_init__(
            max_sequence_length=max_sequence_length,
            quantization_block=quantization_block,
        )
        if self.critic_learning_rate <= 0:
            raise ValueError("`critic_learning_rate` must be positive.")
        if self.critic_learning_rate_end is not None and self.critic_learning_rate_end < 0:
            raise ValueError("`critic_learning_rate_end` must be non-negative when set.")
        if self.critic_warmup_steps < 0:
            raise ValueError("`critic_warmup_steps` must be non-negative.")
        if self.critic_updates_per_actor_update < 1:
            raise ValueError("`critic_updates_per_actor_update` must be at least 1.")
        if self.critic_pretrain_steps < 0:
            raise ValueError("`critic_pretrain_steps` must be non-negative.")
        if self.critic_cliprange_value < 0:
            raise ValueError("`critic_cliprange_value` must be non-negative.")
        if not 0.0 <= self.critic_gae_lambda <= 1.0:
            raise ValueError("`critic_gae_lambda` must be in [0, 1].")
        if self.critic_gae_mode not in ("fixed", "match_policy"):
            raise ValueError("`critic_gae_mode` must be 'fixed' or 'match_policy'.")
        if self.critic_trainable_mode not in ("full_except_attention", "all"):
            raise ValueError("`critic_trainable_mode` must be 'full_except_attention' or 'all'.")
        if self.policy_gae_mode not in ("fixed", "length_adaptive"):
            raise ValueError("`policy_gae_mode` must be 'fixed' or 'length_adaptive'.")
        if self.length_adaptive_alpha <= 0:
            raise ValueError("`length_adaptive_alpha` must be positive.")
        if self.rollout_logprob_source not in ("recompute", "engine"):
            raise ValueError("`rollout_logprob_source` must be 'recompute' or 'engine'.")
        if self.max_inflight_rollouts < 1:
            raise ValueError("`max_inflight_rollouts` must be at least 1.")
        if self.rollout_queue_size < 0:
            raise ValueError("`rollout_queue_size` must be non-negative.")
        if self.weight_sync_steps < 1:
            raise ValueError("`weight_sync_steps` must be at least 1.")
        if self.max_policy_staleness is not None and self.max_policy_staleness < 0:
            raise ValueError("`max_policy_staleness` must be non-negative when set.")
        if self.rollout_timeout is not None and self.rollout_timeout <= 0:
            raise ValueError("`rollout_timeout` must be positive when set.")

    __hash__ = hash_fn


__all__ = ("OnlineActorCriticConfig",)
