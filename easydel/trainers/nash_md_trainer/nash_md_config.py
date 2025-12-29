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

from collections.abc import Sequence
from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..group_relative_policy_optimization.grpo_config import GRPOConfig


@Registry.register("trainer-arguments", ["nash-md", "nash_md"])
@dataclass
class NashMDConfig(GRPOConfig):
    """Configuration for the :class:`~easydel.trainers.NashMDTrainer`.

    The configuration mirrors the Hugging Face TRL Nash-MD setup and extends
    :class:`~easydel.trainers.GRPOConfig` with the additional hyper-parameters
    required by the Nash mixture-of-decoders objective.

    Attributes
    ----------
    beta:
        Prospect theoretic scaling applied to the KL component. When a list
        is provided, values are consumed sequentially on every new epoch and
        the final value is reused afterwards.
    mixture_coef:
        Mixture coefficient used when generating the auxiliary completions
        from the geometric mixture of policy and reference models. Similar to
        ``beta`` this can be scheduled per epoch by providing a list.
    missing_eos_penalty:
        Optional penalty that is subtracted from the reward when a completion
        does not emit an EOS token. This encourages the policy to terminate
        generations earlier than ``max_new_tokens``.
    """

    trainer_prefix: str | None = field(
        default="nashmdtrainer",
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

    def __post_init__(self, max_sequence_length: int | None):
        super().__post_init__(max_sequence_length=max_sequence_length)
        if isinstance(self.mixture_coef, Sequence) and len(self.mixture_coef) == 1:
            self.mixture_coef = self.mixture_coef[0]
        if isinstance(self.beta, Sequence) and len(self.beta) == 1:
            self.beta = self.beta[0]

    __hash__ = hash_fn
