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

"""Supervised Fine-Tuning (SFT) trainer module for EasyDeL.

This module provides comprehensive supervised fine-tuning capabilities for
language models, enabling adaptation of pre-trained models to specific tasks
and domains through supervised learning on labeled datasets.

The module includes:
- SFTConfig: Configuration class for supervised fine-tuning parameters
- SFTTrainer: Main trainer class implementing supervised fine-tuning
- Support for various training strategies and optimizations
- Integration with instruction tuning and conversational datasets

Key Features:
- Standard cross-entropy loss for next-token prediction
- Support for instruction-following and conversational formats
- Packing of multiple sequences for efficient training
- Response template masking for instruction tuning
- Gradient accumulation and mixed precision training
- LoRA and QLoRA support for parameter-efficient fine-tuning
- Integration with JAX/Flax for distributed training

Example:
    >>> from easydel.trainers import SFTConfig, SFTTrainer
    >>> config = SFTConfig(
    ...     max_length=2048,
    ...     learning_rate=2e-5,
    ...     packing=True,
    ...     num_train_epochs=3
    ... )
    >>> trainer = SFTTrainer(
    ...     arguments=config,
    ...     model=model,
    ...     processing_class=tokenizer,
    ...     train_dataset=dataset,
    ...     formatting_func=format_instruction
    ... )
    >>> trainer.train()

Use Cases:
- Instruction tuning for following user commands
- Domain adaptation for specialized tasks
- Conversational model fine-tuning
- Task-specific model specialization
- Continued pre-training on domain data

References:
    - Wei et al., "Finetuned Language Models are Zero-Shot Learners"
      (https://arxiv.org/abs/2109.01652)
    - Chung et al., "Scaling Instruction-Finetuned Language Models"
      (https://arxiv.org/abs/2210.11416)
"""

from .sft_config import SFTConfig
from .sft_trainer import SFTTrainer

__all__ = "SFTConfig", "SFTTrainer"
