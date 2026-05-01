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
"""Configuration dataclass for the Nash-MD trainer.

Defines :class:`NashMDConfig`, which adds the mirror-descent step
size ``mixture_coef`` (mixture probability between the policy and
reference), KL clipping, and judge/oracle handling to the GRPO base
config.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..group_relative_policy_optimization.grpo_config import GRPOConfig


@Registry.register("trainer-arguments", ["nash-md", "nash_md"])
@dataclass
class NashMDConfig(GRPOConfig):
    """Configuration for the :class:`~easydel.trainers.NashMDTrainer`.

    Nash-MD (Munos et al. 2024) frames RLHF as computing a Nash
    equilibrium between the policy and a frozen reference under a
    learned preference oracle. The trainer produces auxiliary
    completions from the *geometric mixture* of the policy and
    reference distributions
    ``pi_mix = pi_policy^mixture_coef * pi_ref^(1 - mixture_coef)``,
    treats those mixture samples as the opposing player, and updates
    the policy via a mirror-descent style step regularised by the KL
    against the reference.

    Inherits the entire GRPO surface (rollout count, sampling knobs,
    reward functions, advantage normalisation). Construct with
    dict-literal kwargs:

    >>> cfg = NashMDConfig(beta=0.1, mixture_coef=0.5,
    ...                    missing_eos_penalty=0.0)

    Attributes:
        trainer_prefix: Default prefix used for checkpoints/logs
            (``"NashMD"``).
        beta: KL coefficient against the reference. Either a scalar
            (constant schedule) or a sequence consumed one entry per
            epoch (with the last entry reused thereafter). Default
            ``0.1``.
        mixture_coef: Geometric-mixture weight on the policy logits
            when sampling the opposing player. ``0`` recovers the
            reference, ``1`` recovers the policy. Either a scalar or
            an epoch-wise schedule. Default ``[0.5]``.
        missing_eos_penalty: Optional reward penalty subtracted from
            completions that do not emit an EOS token within
            ``max_completion_length``. Encourages early termination.
            ``None`` disables the penalty.
    """

    trainer_prefix: str | None = field(
        default="NashMD",
        metadata={"help": "default prefix name for trainer."},
    )

    beta: float | Sequence[float] = field(
        default=0.1,
        metadata={
            "help": "Scaling applied to the KL component. A single value keeps the coefficient fixed while a list "
            "allows epoch-wise scheduling following the TRL Nash-MD recipe."
        },
    )

    mixture_coef: float | Sequence[float] = field(
        default_factory=lambda: [0.5],
        metadata={
            "help": "Logit mixture coefficient between the policy and reference models. Can be provided as a list to "
            "schedule the value across epochs."
        },
    )

    missing_eos_penalty: float | None = field(
        default=None,
        metadata={
            "help": "Penalty applied to the reward when a completion does not end with an EOS token. Set to `None` to "
            "disable the penalty."
        },
    )

    def __post_init__(
        self,
        max_sequence_length: int | None,
        quantization_block: int | None,
    ):
        """Forward to the GRPO base then unwrap singleton scalar schedules.

        ``mixture_coef`` and ``beta`` may be supplied as either scalars
        or single-element sequences; the latter is collapsed for
        backward compatibility.

        Args:
            max_sequence_length: Legacy alias for ``max_length``.
            quantization_block: Legacy alias for the quantization group
                size.
        """
        super().__post_init__(
            max_sequence_length=max_sequence_length,
            quantization_block=quantization_block,
        )
        if isinstance(self.mixture_coef, Sequence) and len(self.mixture_coef) == 1:
            self.mixture_coef = self.mixture_coef[0]
        if isinstance(self.beta, Sequence) and len(self.beta) == 1:
            self.beta = self.beta[0]

    __hash__ = hash_fn
