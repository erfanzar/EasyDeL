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

"""Configuration for Single-Rollout Asynchronous Optimization (SAO)."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from .._online_rl.config import OnlineActorCriticConfig


@Registry.register(
    "trainer-arguments",
    ["sao", "single-rollout-asynchronous-optimization", "single_rollout_asynchronous_optimization"],
)
@dataclass
class SAOConfig(OnlineActorCriticConfig):
    """Hyperparameters for Single-Rollout Asynchronous Optimization.

    SAO consumes one rollout per prompt and uses direct double-sided
    importance sampling (DIS) to reject action tokens whose behavior-policy
    ratio lies outside a strict asymmetric interval.  A separately optimized
    critic supplies skip-observation GAE advantages.

    For an action token ``a_t`` sampled by ``pi_rollout``, SAO forms

    ``r_t = exp(log pi_theta(a_t) - log pi_rollout(a_t))``

    and keeps it only when

    ``1 - dis_epsilon_low < r_t < 1 + dis_epsilon_high``.

    The inequalities are strict.  The retained ratio is detached in the
    score-function surrogate, preventing autodiff from adding an unintended
    derivative through the importance weight.

    The defaults mirror the public GLM-5.2 reasoning recipe where the paper
    discloses a value.  Coding experiments used a wider lower threshold
    (``dis_epsilon_low=0.8`` and ``dis_epsilon_high=3.0``), which can be
    selected explicitly.

    Attributes:
        trainer_prefix: Prefix used for logs and checkpoints.
        dis_epsilon_low: Lower DIS rejection width. Must be in ``(0, 1]``.
        dis_epsilon_high: Upper DIS rejection width. Must be positive.
        learning_rate: Actor learning rate.
        total_batch_size: Global rollout batch size.
        critic_learning_rate: Critic learning rate.
        critic_pretrain_steps: Critic-only warmup updates before the first
            actor update.
        critic_updates_per_actor_update: Critic updates per actor update.
        num_return_sequences: Exactly one rollout is required per prompt.
        num_generations: Alias that is likewise fixed to one.
        asynchronous_rollouts: Whether a rollout provider may use the bounded
            completion-order coordinator.
    """

    trainer_prefix: str | None = field(
        default="SAO",
        metadata={"help": "Default prefix for SAO logs and checkpoints."},
    )
    learning_rate: float = field(
        default=1e-6,
        metadata={"help": "Actor learning rate used by the public SAO recipe."},
    )
    total_batch_size: int = field(
        default=128,
        metadata={"help": "Global rollout batch size used by the public SAO experiments."},
    )
    kl_coef: float = field(
        default=0.0,
        metadata={"help": "Optional reference-policy KL reward coefficient; zero matches the published SAO objective."},
    )
    critic_learning_rate: float = field(
        default=5e-6,
        metadata={"help": "Critic learning rate used by the public SAO recipe."},
    )
    critic_learning_rate_end: float | None = field(
        default=5e-6,
        metadata={"help": "Keep the critic learning rate constant after its warmup."},
    )
    critic_scheduler: str | None = field(
        default="linear",
        metadata={"help": "Linear warmup followed by a constant critic learning rate."},
    )
    critic_warmup_steps: int = field(
        default=10,
        metadata={"help": "Learning-rate warmup steps for the SAO critic optimizer."},
    )
    critic_pretrain_steps: int = field(
        default=0,
        metadata={
            "help": (
                "Optional EasyDeL value-pretraining phase. The SAO paper uses a separately "
                "pretrained critic corpus but does not prescribe a fixed online step count."
            )
        },
    )
    critic_updates_per_actor_update: int = field(
        default=2,
        metadata={"help": "Number of critic updates before every actor update."},
    )
    critic_gae_lambda: float = field(
        default=1.0,
        metadata={"help": "GAE lambda used for critic regression targets."},
    )
    policy_gae_mode: tp.Literal["fixed", "length_adaptive"] = field(
        default="length_adaptive",
        metadata={"help": "SAO actor GAE mode: fixed or length_adaptive."},
    )
    length_adaptive_alpha: float = field(
        default=1.5,
        metadata={"help": "Alpha used by SAO's length-adaptive policy GAE."},
    )
    dis_epsilon_low: float = field(
        default=0.3,
        metadata={"help": "Strict lower DIS rejection width."},
    )
    dis_epsilon_high: float = field(
        default=5.0,
        metadata={"help": "Strict upper DIS rejection width."},
    )
    num_return_sequences: int = field(
        default=1,
        metadata={"help": "SAO requires exactly one rollout per prompt."},
    )
    num_generations: int | None = field(
        default=1,
        metadata={"help": "Alias for num_return_sequences; fixed to one by SAO."},
    )
    num_ppo_epochs: int = field(
        default=1,
        metadata={"help": "SAO performs exactly one actor update for each consumed rollout batch."},
    )
    asynchronous_rollouts: bool = field(
        default=True,
        metadata={"help": "Enable completion-order asynchronous rollout providers."},
    )
    rollout_logprob_source: tp.Literal["recompute", "engine"] = field(
        default="engine",
        metadata={"help": "Use token log probabilities preserved by the asynchronous rollout engine."},
    )

    def __post_init__(
        self,
        max_sequence_length: int | None,
        quantization_block: int | None,
    ) -> None:
        """Normalize inherited fields and enforce SAO invariants."""
        super().__post_init__(
            max_sequence_length=max_sequence_length,
            quantization_block=quantization_block,
        )
        if self.num_return_sequences != 1 or self.num_generations != 1:
            raise ValueError("SAO is single-rollout: `num_return_sequences` and `num_generations` must both be 1.")
        if self.num_ppo_epochs != 1:
            raise ValueError("SAO performs one actor update per consumed rollout batch; `num_ppo_epochs` must be 1.")
        if not 0.0 < self.dis_epsilon_low <= 1.0:
            raise ValueError("`dis_epsilon_low` must be in (0, 1].")
        if self.dis_epsilon_high <= 0.0:
            raise ValueError("`dis_epsilon_high` must be positive.")
        if self.policy_gae_mode not in ("fixed", "length_adaptive"):
            raise ValueError("`policy_gae_mode` must be 'fixed' or 'length_adaptive'.")

    __hash__ = hash_fn


__all__ = ("SAOConfig",)
