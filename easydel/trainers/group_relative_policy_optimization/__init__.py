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

"""Group Relative Policy Optimization (GRPO) trainer module for EasyDeL.

This module implements Group Relative Policy Optimization, an efficient variant
of reinforcement learning from human feedback (RLHF) that optimizes policies
using group-based relative comparisons. GRPO provides a more stable and efficient
alternative to traditional PPO-based RLHF approaches.

The module includes:
- GRPOConfig: Configuration class for GRPO training parameters
- GRPOTrainer: Main trainer class implementing the GRPO algorithm
- Support for group-based reward normalization and optimization
- Efficient batched generation and evaluation

Key Features:
- Group-based relative reward computation for stability
- Support for multiple reward models and ensemble averaging
- KL divergence regularization from reference policy
- Adaptive clipping and normalization strategies
- Integration with JAX/Flax for distributed training

Example:
    >>> from easydel.trainers import GRPOConfig, GRPOTrainer
    >>> config = GRPOConfig(
    ...     group_size=4,
    ...     kl_coef=0.1,
    ...     max_length=512
    ... )
    >>> trainer = GRPOTrainer(
    ...     arguments=config,
    ...     model=model,
    ...     reference_model=ref_model,
    ...     reward_model=reward_model,
    ...     processing_class=tokenizer,
    ...     train_dataset=dataset
    ... )
    >>> trainer.train()

References:
    - Group Relative Policy Optimization paper and related RLHF literature
"""

from .grpo_config import GRPOConfig
from .grpo_trainer import GRPOTrainer

__all__ = "GRPOConfig", "GRPOTrainer"
