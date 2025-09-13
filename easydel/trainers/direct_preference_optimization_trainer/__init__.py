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

"""Direct Preference Optimization (DPO) trainer module for EasyDeL.

This module provides an implementation of the Direct Preference Optimization algorithm
for training language models from human preferences. DPO is a simpler alternative to
RLHF (Reinforcement Learning from Human Feedback) that directly optimizes for human
preferences without requiring a separate reward model.

The module includes:
- DPOConfig: Configuration class for DPO training parameters
- DPOTrainer: Main trainer class implementing the DPO algorithm
- Various loss functions: sigmoid, hinge, IPO, and other variants
- Support for reference-free training and model synchronization

Key Features:
- Multiple loss function variants for different optimization objectives
- Support for encoder-decoder and decoder-only architectures
- Gradient accumulation and mixed precision training
- Precomputed reference model log probabilities for efficiency
- Integration with JAX/Flax for distributed training

Example:
    >>> from easydel.trainers import DPOConfig, DPOTrainer
    >>> config = DPOConfig(
    ...     beta=0.1,
    ...     loss_type="sigmoid",
    ...     max_length=512
    ... )
    >>> trainer = DPOTrainer(
    ...     arguments=config,
    ...     model=model,
    ...     reference_model=ref_model,
    ...     processing_class=tokenizer,
    ...     train_dataset=dataset
    ... )
    >>> trainer.train()

References:
    - Rafailov et al., "Direct Preference Optimization: Your Language Model is
      Secretly a Reward Model" (https://arxiv.org/abs/2305.18290)
"""

from .dpo_trainer import DPOConfig, DPOTrainer

__all__ = "DPOConfig", "DPOTrainer"
