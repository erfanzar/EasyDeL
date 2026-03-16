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

"""Reinforcement Learning with Verifiable Rewards (RLVR) trainer module.

RLVR is a single-turn RL pipeline where completions are scored by
deterministic verifiable reward functions (math answer checking,
code test execution, format compliance) instead of learned reward
models. Combined with GRPO for critic-free advantage estimation,
this provides a fully self-contained training pipeline.

Inspired by DeepSeek-R1 and Alibaba's ROLL framework.

Example:
    >>> from datasets import load_dataset
    >>> from easydel.trainers import RLVRConfig, RLVRTrainer
    >>> from easydel.trainers.rlvr import MathVerifier
    >>>
    >>> config = RLVRConfig(
    ...     max_prompt_length=1024,
    ...     max_completion_length=2048,
    ...     num_return_sequences=4,
    ...     answer_key="answer",
    ... )
    >>> gsm8k = load_dataset("openai/gsm8k", "main", split="train")
    >>> trainer = RLVRTrainer(
    ...     arguments=config,
    ...     model=model,
    ...     train_dataset=gsm8k,
    ...     processing_class=tokenizer,
    ... )
    >>> trainer.train()
"""

from .reward_verifiers import (
    CodeVerifier,
    FormatVerifier,
    LengthPenaltyVerifier,
    MathVerifier,
)
from .rlvr_config import RLVRConfig
from .rlvr_trainer import RLVRTrainer

__all__ = (
    "CodeVerifier",
    "FormatVerifier",
    "LengthPenaltyVerifier",
    "MathVerifier",
    "RLVRConfig",
    "RLVRTrainer",
)
