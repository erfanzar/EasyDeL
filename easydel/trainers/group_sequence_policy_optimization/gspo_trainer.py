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

from easydel.infra.base_module import EasyDeLBaseModule
from easydel.infra.base_state import EasyDeLState
from easydel.infra.utils import ProcessingClassType
from easydel.utils import Registry

from ..group_relative_policy_optimization import GRPOTrainer
from .gspo_config import GSPOConfig

if tp.TYPE_CHECKING:
    from datasets import Dataset, IterableDataset

    from easydel.data.core.protocols import ShardedDataSource


RewardFunc = tp.Union[EasyDeLBaseModule, EasyDeLState, tp.Callable[[list, list], list[float]]]  # noqa: UP007


@Registry.register("trainer", "gspo")
class GSPOTrainer(GRPOTrainer):
    """Group Sequence Policy Optimization trainer for RLHF.

    GSPO is a variant of GRPO that computes importance sampling at the sequence
    level rather than token level. This provides improved training stability and
    performance, particularly beneficial for Mixture-of-Experts (MoE) models.

    The key algorithmic difference from GRPO is that GSPO defines the importance
    ratio based on sequence likelihood and performs sequence-level clipping,
    rewarding, and optimization. This leads to:
    - Superior training efficiency compared to GRPO
    - Enhanced stability for MoE model training
    - Simplified RL infrastructure requirements

    GSPO was developed by Alibaba Qwen team and contributed to the improvements
    in Qwen3 models.

    Attributes:
        arguments: GSPOConfig instance with GSPO-specific hyperparameters
        ref_state: Reference model state for computing log probabilities
        processing_class: Tokenizer or processor for text encoding
        reward_processing_classes: Optional separate processors for reward models
        generation_config: Configuration for response generation
        data_tokenize_fn: Function to tokenize dataset samples

    Reference:
        Group Sequence Policy Optimization (arXiv:2507.18071)

    Example:
        >>> config = GSPOConfig(
        ...     per_device_train_batch_size=4,
        ...     num_generations=4,
        ...     max_prompt_length=512,
        ...     max_completion_length=256,
        ...     learning_rate=1e-6,
        ... )
        >>> trainer = GSPOTrainer(
        ...     arguments=config,
        ...     model=model,
        ...     reward_funcs=reward_model,
        ...     train_dataset=dataset,
        ...     processing_class=tokenizer
        ... )
        >>> trainer.train()
    """

    arguments: GSPOConfig

    def __init__(
        self,
        arguments: GSPOConfig,
        model: EasyDeLBaseModule | EasyDeLState | None,
        reward_funcs: RewardFunc | list[RewardFunc],
        train_dataset: Dataset | IterableDataset | ShardedDataSource | None = None,
        eval_dataset: Dataset | IterableDataset | ShardedDataSource | dict[str, Dataset] | None = None,
        processing_class: ProcessingClassType = None,
        reward_processing_classes: ProcessingClassType = None,
        data_tokenize_fn: tp.Callable | None = None,
    ):
        """Initialize the GSPO trainer.

        Args:
            arguments: GSPOConfig containing training hyperparameters with
                GSPO-specific defaults (sequence-level importance sampling,
                smaller epsilon bounds, no KL regularization).
            model: The policy model to train. Can be an EasyDeLBaseModule or
                EasyDeLState. Will be converted to state if needed.
            reward_funcs: Reward function(s) for scoring completions. Can be
                a single function/model or a list for multi-reward training.
            train_dataset: Training dataset containing prompts.
            eval_dataset: Optional evaluation dataset.
            processing_class: Tokenizer for encoding text.
            reward_processing_classes: Optional separate tokenizers for reward
                models if they differ from the policy model tokenizer.
            data_tokenize_fn: Optional custom tokenization function.

        Raises:
            AssertionError: If arguments is not a GSPOConfig instance.
        """
        assert isinstance(arguments, GSPOConfig), (
            f"arguments must be `GSPOConfig` but got {type(arguments)}. "
            "Use GSPOConfig for GSPO training or GRPOConfig for GRPO training."
        )
        super().__init__(
            arguments=arguments,
            model=model,
            reward_funcs=reward_funcs,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            data_tokenize_fn=data_tokenize_fn,
        )
