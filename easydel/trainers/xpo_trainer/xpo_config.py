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

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..group_relative_policy_optimization.grpo_config import GRPOConfig


@Registry.register("trainer-arguments", "xpo")
@dataclass
class XPOConfig(GRPOConfig):
    """Configuration for the XPO (Exploratory Preference Optimization) trainer.

    Extends GRPOConfig with hyperparameters required by the XPO objective,
    which combines DPO-style preference learning with exploratory sampling.
    The configuration controls the loss variant, KL penalty scaling, and
    the exploratory weighting parameter alpha.

    Attributes:
        loss_type: Choice of DPO loss function ("sigmoid" or "ipo").
        beta: Scaling factor for the KL penalty. Can be a float or sequence for epoch-wise scheduling.
        alpha: Weight of exploratory term encouraging probability mass on reference completions.
            Supports epoch-wise scheduling via sequence.
        missing_eos_penalty: Optional penalty subtracted from reward when completion lacks EOS token.
        num_return_sequences: Number of completions to sample per prompt (XPO uses 1).
    """

    trainer_prefix: str | None = field(
        default="xpotrainer",
        metadata={"help": "default prefix name for trainer."},
    )

    loss_type: tp.Literal["sigmoid", "ipo"] = field(
        default="sigmoid",
        metadata={"help": "Choice of DPO loss. Matches the TRL implementation options."},
    )

    beta: float | tp.Sequence[float] = field(
        default=0.1,
        metadata={
            "help": "Prospect-theoretic scaling for the KL penalty. A list enables epoch-wise scheduling; the final "
            "value is reused when the schedule is exhausted."
        },
    )

    alpha: float | tp.Sequence[float] = field(
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

    def __post_init__(self, max_sequence_length: int | None):
        self._handle_deprecated_max_sequence_length(max_sequence_length)
        if isinstance(self.alpha, tp.Sequence) and len(self.alpha) == 1:
            self.alpha = self.alpha[0]
        if isinstance(self.beta, tp.Sequence) and len(self.beta) == 1:
            self.beta = self.beta[0]
        if hasattr(super(), "__post_init__"):
            super().__post_init__(max_sequence_length=None)

    __hash__ = hash_fn
