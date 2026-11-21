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
"""VAPO configuration: PPO with VAPO (arXiv:2504.05118) defaults."""

from __future__ import annotations

from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..proximal_policy_optimization_trainer import PPOConfig


@Registry.register("trainer-arguments", "vapo")
@dataclass
class VAPOConfig(PPOConfig):
    """Value-model-based Augmented PPO (VAPO) hyperparameters.

    VAPO (Yue et al., 2025, arXiv:2504.05118) is PPO with a set of fixes for
    long chain-of-thought reasoning. Every technique is a :class:`PPOConfig`
    knob, so this class only changes defaults:

    * decoupled GAE: critic targets use ``gae_lambda_critic=1.0`` (Monte-Carlo
      returns), while the policy uses its own lambda;
    * length-adaptive GAE: the policy lambda is
      ``clip(1 - 1 / (length_adaptive_gae_alpha * l), 0, 1)`` per trajectory;
    * clip-higher: the ratio is clipped to ``[1 - cliprange, 1 + 0.28]``;
    * token-level policy-gradient loss normalised over all rollout tokens;
    * positive-example NLL with weight ``0.1`` on completions whose score is
      above ``positive_reward_threshold``.

    Value pretraining (``value_pretrain_steps``) and the KL penalty
    (``kl_coef``) keep their PPO defaults. Set them explicitly to reproduce a
    specific VAPO run, for example ``value_pretrain_steps=50, kl_coef=0.0``.

    Attributes:
        trainer_prefix: Prefix for logs and checkpoints. Default ``"VAPO"``.
        gae_lambda_critic: Critic GAE lambda. Default ``1.0``.
        length_adaptive_gae: Use the length-adaptive policy lambda.
            Default ``True``.
        cliprange_high: Upper policy-ratio clip. Default ``0.28``.
        token_level_loss: Token-level policy-gradient normalisation.
            Default ``True``.
        positive_nll_coef: Positive-example NLL weight. Default ``0.1``.
    """

    trainer_prefix: str | None = field(
        default="VAPO",
        metadata={"help": "Default prefix name for trainer outputs/checkpoints."},
    )
    gae_lambda_critic: float | None = field(
        default=1.0,
        metadata={"help": "Decoupled GAE lambda for value targets. VAPO uses 1.0."},
    )
    length_adaptive_gae: bool = field(
        default=True,
        metadata={"help": "Use the per-trajectory policy lambda clip(1 - 1 / (alpha * length), 0, 1)."},
    )
    cliprange_high: float | None = field(
        default=0.28,
        metadata={"help": "Upper policy-ratio clip (clip-higher)."},
    )
    token_level_loss: bool = field(
        default=True,
        metadata={"help": "Normalize the policy loss by the total completion-token count of the rollout batch."},
    )
    positive_nll_coef: float = field(
        default=0.1,
        metadata={"help": "Weight of the NLL auxiliary loss on positive (score > threshold) completions."},
    )

    __hash__ = hash_fn
