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

"""Odds Ratio Preference Optimization (ORPO) trainer module for EasyDeL.

This module implements Odds Ratio Preference Optimization, a novel approach
to preference learning that uses odds ratios to model preferences between
chosen and rejected responses. ORPO provides a mathematically principled
alternative to traditional preference optimization methods.

The module includes:
- ORPOConfig: Configuration class for ORPO training parameters
- ORPOTrainer: Main trainer class implementing the ORPO algorithm
- Odds ratio-based loss functions for preference modeling
- Support for various regularization strategies

Key Features:
- Odds ratio formulation for robust preference learning
- Implicit reward modeling through log-odds differences
- Support for both classification and ranking objectives
- Efficient batch processing with gradient accumulation
- Integration with JAX/Flax for distributed training

Example:
    >>> from easydel.trainers import ORPOConfig, ORPOTrainer
    >>> config = ORPOConfig(
    ...     beta=0.1,
    ...     max_length=512,
    ...     learning_rate=5e-6
    ... )
    >>> trainer = ORPOTrainer(
    ...     arguments=config,
    ...     model=model,
    ...     processing_class=tokenizer,
    ...     train_dataset=dataset
    ... )
    >>> trainer.train()

References:
    - Hong et al., "ORPO: Monolithic Preference Optimization without Reference Model"
      (https://arxiv.org/abs/2403.07691)
"""

from .orpo_trainer import ORPOConfig, ORPOTrainer

__all__ = ("ORPOConfig", "ORPOTrainer")
