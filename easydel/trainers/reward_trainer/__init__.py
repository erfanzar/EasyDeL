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

"""Reward Model trainer module for EasyDeL.

This module provides training capabilities for reward models used in
Reinforcement Learning from Human Feedback (RLHF). Reward models learn
to predict human preferences between different model outputs, serving as
a proxy for human judgment in the RLHF pipeline.

The module includes:
- RewardConfig: Configuration class for reward model training parameters
- RewardTrainer: Main trainer class for reward model training
- Support for pairwise ranking loss functions
- Efficient batch processing for preference pairs

Key Features:
- Bradley-Terry model implementation for preference learning
- Support for margin-based ranking losses
- Handling of chosen/rejected response pairs
- Gradient accumulation and mixed precision training
- Integration with JAX/Flax for efficient computation

Example:
    >>> from easydel.trainers import RewardConfig, RewardTrainer
    >>> config = RewardConfig(
    ...     max_length=512,
    ...     learning_rate=2e-5,
    ...     use_margin_loss=True
    ... )
    >>> trainer = RewardTrainer(
    ...     arguments=config,
    ...     model=reward_model,
    ...     processing_class=tokenizer,
    ...     train_dataset=preference_dataset
    ... )
    >>> trainer.train()

Use Cases:
- Training reward models for RLHF pipelines
- Learning human preference functions
- Ranking model outputs by quality
- Providing feedback signals for policy optimization

References:
    - Ouyang et al., "Training language models to follow instructions with human feedback"
      (https://arxiv.org/abs/2203.02155)
    - Stiennon et al., "Learning to summarize with human feedback"
      (https://arxiv.org/abs/2009.01325)
"""

from .reward_config import RewardConfig
from .reward_trainer import RewardTrainer

__all__ = ("RewardConfig", "RewardTrainer")
