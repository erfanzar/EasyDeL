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
"""Configuration dataclass for the GFPO trainer.

Defines :class:`GFPOConfig`, which adds rollout filtering knobs
(``filtered_group_size``, scoring strategy, length bias) on top of the
GRPO base config.
"""

from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..group_relative_policy_optimization import GRPOConfig


@Registry.register("trainer-arguments", "gfpo")
@dataclass
class GFPOConfig(GRPOConfig):
    """Configuration class for Group Filtered Policy Optimization (GFPO) training.

    GFPO (arXiv:2508.09726) extends GRPO with a *post-rollout
    filtering* step that mitigates response-length inflation. After
    sampling ``num_generations`` completions per prompt, the trainer
    scores each completion using a weighted combination of:

    * **Length score** -- penalises long completions (when
      ``filter_by_length`` is ``True``).
    * **Reward efficiency** -- ``reward / num_tokens``, which favours
      completions that achieve high reward per generated token (when
      ``filter_by_efficiency`` is ``True``).

    Then only the top-``num_remains_in_group`` completions per prompt
    are kept and used to estimate the group-relative advantages that
    drive the policy update. ``num_remains_in_group is None``
    short-circuits filtering (behaviour identical to GRPO).

    Inherits the entire GRPO surface (rollout count, reward functions,
    advantage normalisation, KL coefficient, ...).

    Construct with dict-literal kwargs:

    >>> cfg = GFPOConfig(num_generations=8, num_remains_in_group=4,
    ...                  length_weight=0.5, efficiency_weight=0.5)

    Attributes:
        trainer_prefix: Default prefix used for checkpoints/logs
            (``"GFPO"``).
        num_remains_in_group: Number of completions retained per
            prompt after filtering. Must satisfy
            ``2 <= num_remains_in_group < num_generations``. ``None``
            disables filtering and recovers the GRPO behaviour.
        filter_by_length: When ``True``, the length score contributes
            to the per-completion ranking. Shorter completions are
            preferred.
        filter_by_efficiency: When ``True``, the reward-per-token
            efficiency contributes to the ranking. Higher efficiency
            is preferred.
        length_weight: Weight on the length component of the filter
            score (``0..1``).
        efficiency_weight: Weight on the efficiency component of the
            filter score (``0..1``).
    """

    trainer_prefix: str | None = field(
        default="GFPO",
        metadata={"help": "default prefix name for trainer."},
    )
    num_remains_in_group: int | None = field(
        default=None,
        metadata={
            "help": "Number of samples to retain after filtering per group. "
            "Must be >= 2 and < num_generations if specified. "
            "If None, no filtering is applied (behaves like GRPO)."
        },
    )
    filter_by_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to include response length in filter scoring. Shorter responses are preferred when enabled."
        },
    )
    filter_by_efficiency: bool = field(
        default=True,
        metadata={
            "help": "Whether to include reward-per-token efficiency in filter scoring. "
            "Higher reward-per-token responses are preferred when enabled."
        },
    )
    length_weight: float = field(
        default=0.5,
        metadata={
            "help": "Weight for length component in filter scoring (0 to 1). "
            "Higher values prioritize shorter responses more."
        },
    )
    efficiency_weight: float = field(
        default=0.5,
        metadata={
            "help": "Weight for efficiency component in filter scoring (0 to 1). "
            "Higher values prioritize reward-per-token more."
        },
    )

    def __post_init__(
        self,
        max_sequence_length: int | None,
        quantization_block: int | None,
    ):
        """Forward to the GRPO base class then validate GFPO-specific knobs.

        Args:
            max_sequence_length: Legacy alias for ``max_length`` forwarded
                to the base class.
            quantization_block: Legacy alias for the quantization group
                size.

        Raises:
            ValueError: If ``num_remains_in_group`` is set but smaller
                than 2, or not strictly smaller than ``num_generations``.
        """
        super().__post_init__(
            max_sequence_length=max_sequence_length,
            quantization_block=quantization_block,
        )

        if self.num_remains_in_group is not None:
            if self.num_remains_in_group < 2:
                raise ValueError(f"num_remains_in_group must be >= 2, got {self.num_remains_in_group}")
            if self.num_generations is not None and self.num_remains_in_group >= self.num_generations:
                raise ValueError(
                    f"num_remains_in_group ({self.num_remains_in_group}) must be < "
                    f"num_generations ({self.num_generations})"
                )

    __hash__ = hash_fn
