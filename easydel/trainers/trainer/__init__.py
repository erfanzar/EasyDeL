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

"""Core training functionality for EasyDeL.

This module provides the main Trainer class for training neural network models
with JAX/Flax. The Trainer handles:

- Model initialization and management
- Training loop execution
- Gradient computation and optimization
- Checkpointing and model saving
- Evaluation and metrics tracking
- Distributed training across devices
- Support for various model architectures (language, vision, multimodal)

Key Components:
    - Trainer: Main training orchestrator class
    - Training functions: Core training step implementations
    - Model outputs: Training result data structures

Example:
    >>> from easydel.trainers import Trainer
    >>> trainer = Trainer(
    ...     model=model,
    ...     train_dataset=train_data,
    ...     eval_dataset=eval_data,
    ...     config=training_config
    ... )
    >>> trainer.train()
"""

from .trainer import Trainer

__all__ = ("Trainer",)
