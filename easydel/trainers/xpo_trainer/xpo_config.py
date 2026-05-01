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

from __future__ import annotations

import collections.abc
import typing as tp
from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..group_relative_policy_optimization.grpo_config import GRPOConfig


@Registry.register("trainer-arguments", "xpo")
@dataclass
class XPOConfig(GRPOConfig):
    """Hyperparameters for Exploratory Preference Optimization (XPO).

    XPO is an online preference-learning algorithm that pairs a
    DPO-style preference loss with an explicit *exploration* bonus.
    For each prompt the trainer samples *two* completions from the
    current policy and one completion from a frozen reference policy
    (the latter acts as a stand-in for an oracle answer). A reward
    function then ranks the two policy completions, and the
    higher-ranked completion is used as ``chosen`` while the other is
    used as ``rejected``. The loss is

    ``L = L_DPO(beta; chosen, rejected, ref)
          - alpha * mean(log pi_theta(ref_completion | prompt))``

    where ``L_DPO`` is the standard ``"sigmoid"`` Bradley-Terry log-
    sigmoid surrogate (or the IPO squared-loss when
    ``loss_type="ipo"``), ``beta`` controls the KL penalty toward the
    reference policy, and the second term -- the XPO-specific
    exploration bonus -- pushes the policy to keep probability mass on
    the *reference* completion. This balances exploitation (preference
    signal) against exploration (don't collapse onto a narrow mode).
    Both ``beta`` and ``alpha`` accept either a scalar (constant
    schedule) or a sequence (epoch-wise schedule whose tail value is
    reused once the schedule is exhausted).

    Inherits all generation / sampling / dataset knobs from
    :class:`GRPOConfig`; XPO fixes ``num_return_sequences`` to 1
    because the additional reference-policy sample is generated
    separately inside the trainer rather than via group sampling.

    Attributes:
        trainer_prefix: Prefix used when naming logs / checkpoints /
            W&B runs. Default: ``"XPO"``.
        loss_type: Selects the DPO surrogate. ``"sigmoid"`` (default)
            uses the Bradley-Terry log-sigmoid loss; ``"ipo"`` uses the
            squared-margin Identity Preference Optimization variant.
        beta: KL-penalty weight against the reference policy. Pass a
            scalar for a constant schedule or a sequence for an
            epoch-wise schedule (final value is reused when the
            schedule is exhausted).
        alpha: Exploration-bonus weight on
            ``-mean(log pi_theta(ref_completion | prompt))``. Default
            is the single-element list ``[1e-5]``, which the
            ``__post_init__`` collapses to the scalar ``1e-5``.
            Sequence form enables epoch-wise scheduling.
        missing_eos_penalty: Optional penalty subtracted from the
            reward of completions that do not emit an EOS token before
            the length cap; encourages well-formed sequences. ``None``
            disables the penalty.
    """

    trainer_prefix: str | None = field(
        default="XPO",
        metadata={"help": "default prefix name for trainer."},
    )

    loss_type: tp.Literal["sigmoid", "ipo"] = field(
        default="sigmoid",
        metadata={"help": "Choice of DPO loss. Matches the TRL implementation options."},
    )

    beta: float | collections.abc.Sequence[float] = field(
        default=0.1,
        metadata={
            "help": "Prospect-theoretic scaling for the KL penalty. A list enables epoch-wise scheduling; the final "
            "value is reused when the schedule is exhausted."
        },
    )

    alpha: float | collections.abc.Sequence[float] = field(
        default_factory=lambda: [1e-5],
        metadata={
            "help": "Weight of the exploratory term that encourages the policy to assign probability mass to "
            "reference completions. Supports epoch-wise scheduling."
        },
    )

    missing_eos_penalty: float | None = field(
        default=None,
        metadata={
            "help": "Optional penalty subtracted from the reward when a completion does not emit an EOS token."
            " Encourages shorter generations when set."
        },
    )

    def __post_init__(
        self,
        max_sequence_length: int | None,
        quantization_block: int | None,
    ):
        """Normalize XPO scheduling parameters and forward to the base class.

        - Forwards a deprecated ``max_sequence_length`` to the base class.
        - Collapses single-element ``alpha`` / ``beta`` sequences to
          scalars to keep the JIT specialization stable.

        Args:
            max_sequence_length (int | None): Deprecated alias for
                ``max_length``; forwarded to the GRPO base class.
            quantization_block (int | None): Optional quantization
                block size forwarded to the base class.
        """
        self._handle_deprecated_max_sequence_length(max_sequence_length)
        if isinstance(self.alpha, collections.abc.Sequence) and len(self.alpha) == 1:
            self.alpha = self.alpha[0]
        if isinstance(self.beta, collections.abc.Sequence) and len(self.beta) == 1:
            self.beta = self.beta[0]
        if hasattr(super(), "__post_init__"):
            super().__post_init__(
                max_sequence_length=None,
                quantization_block=quantization_block,
            )

    __hash__ = hash_fn
