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

"""Knowledge Distillation trainer module for EasyDeL.

This module provides an implementation of knowledge distillation for training
smaller student models to mimic the behavior of larger teacher models. Knowledge
distillation enables model compression while maintaining much of the teacher
model's performance.

The module includes:
- DistillationConfig: Configuration class for distillation training parameters
- DistillationTrainer: Main trainer class implementing knowledge distillation
- Support for various distillation loss functions and strategies
- Temperature-based softmax scaling for probability matching

Key Features:
- Flexible temperature scaling for controlling distillation softness
- Support for combining distillation loss with supervised learning loss
- Efficient batch processing with JAX/Flax
- Support for both encoder-decoder and decoder-only architectures
- Gradient accumulation and mixed precision training

Example:
    >>> from easydel.trainers import DistillationConfig, DistillationTrainer
    >>> config = DistillationConfig(
    ...     temperature=3.0,
    ...     alpha=0.5,
    ...     max_length=512
    ... )
    >>> trainer = DistillationTrainer(
    ...     arguments=config,
    ...     student_model=student,
    ...     teacher_model=teacher,
    ...     processing_class=tokenizer,
    ...     train_dataset=dataset
    ... )
    >>> trainer.train()

References:
    - Hinton et al., "Distilling the Knowledge in a Neural Network"
      (https://arxiv.org/abs/1503.02531)
"""

from .distillation_config import DistillationConfig
from .distillation_trainer import DistillationTrainer

__all__ = ("DistillationConfig", "DistillationTrainer")
