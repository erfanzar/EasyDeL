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

from dataclasses import dataclass, field

from easydel.utils import Registry
from easydel.utils.compiling_utils import hash_fn

from ..group_relative_policy_optimization import GRPOConfig


@Registry.register("trainer-arguments", "gfpo")
@dataclass
class GFPOConfig(GRPOConfig):
    """Configuration class for Group Filtered Policy Optimization training.

    GFPO (Group Filtered Policy Optimization) is a Microsoft algorithm that reduces
    response length inflation while maintaining accuracy. The key idea is to generate
    more samples per prompt during training, then filter to keep only the most
    efficient ones based on length and reward-per-token metrics.

    This approach follows the principle: "Sample more at training time, think less
    at inference time." By training on shorter, more efficient responses, models
    learn to generate concise reasoning at inference time.

    Key features:
    - Generates more samples per prompt (num_generations)
    - Filters to keep top K samples (num_remains_in_group)
    - Filters based on response length and reward-per-token efficiency
    - Reduces length inflation by 46-85% while maintaining accuracy

    Reference:
        "Sample More to Think Less: Group Filtered Policy Optimization for
        Concise Reasoning" (arXiv:2508.09726)

    Example:
        >>> config = GFPOConfig(
        ...     per_device_train_batch_size=4,
        ...     num_generations=8,           # Generate more samples
        ...     num_remains_in_group=4,      # Keep top 4 after filtering
        ...     learning_rate=1e-6,
        ... )
        >>> trainer = GFPOTrainer(
        ...     arguments=config,
        ...     model=model,
        ...     reward_funcs=reward_model,
        ...     train_dataset=dataset,
        ...     processing_class=tokenizer
        ... )
        >>> trainer.train()
    """

    trainer_prefix: str | None = field(
        default="gfpotrainer",
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

    def __post_init__(self, max_sequence_length: int | None):
        """Post initialization to validate GFPO-specific parameters."""
        super().__post_init__(max_sequence_length=max_sequence_length)

        if self.num_remains_in_group is not None:
            if self.num_remains_in_group < 2:
                raise ValueError(f"num_remains_in_group must be >= 2, got {self.num_remains_in_group}")
            if self.num_generations is not None and self.num_remains_in_group >= self.num_generations:
                raise ValueError(
                    f"num_remains_in_group ({self.num_remains_in_group}) must be < "
                    f"num_generations ({self.num_generations})"
                )

    __hash__ = hash_fn
