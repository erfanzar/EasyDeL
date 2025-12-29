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

"""Qwen2 - Alibaba Cloud's next-generation large language model.

This module implements the Qwen2 (Qianwen 2) architecture from Alibaba Cloud, featuring
improved training efficiency and model performance. Qwen2 models range from small to
large scales and are designed for multilingual understanding and generation.

**Key Features**:
- Grouped-query attention (GQA) for efficient inference
- Optional sliding window attention for longer contexts
- RMSNorm for training stability
- SwiGLU activation in feedforward layers
- Extended vocabulary (151K+ tokens) for better multilingual support

**Architecture Highlights**:
- Pre-normalization transformer architecture
- Configurable attention patterns (full or sliding window)
- Support for very long contexts (up to 32K+ tokens)
- Flexible rope_scaling for context extension
- Layer-specific attention type configuration

**Available Model Variants**:
- Qwen2Model: Base transformer model with hidden states
- Qwen2ForCausalLM: Model with LM head for text generation
- Qwen2ForSequenceClassification: Model with classification head

Example:
    >>> from easydel.modules.qwen2 import (
    ...     Qwen2Config,
    ...     Qwen2ForCausalLM,
    ... )
    >>> config = Qwen2Config(
    ...     hidden_size=4096,
    ...     num_hidden_layers=32,
    ...     num_attention_heads=32,
    ...     num_key_value_heads=32,  # Or fewer for GQA
    ...     max_position_embeddings=32768,
    ... )
    >>> model = Qwen2ForCausalLM(
    ...     config=config,
    ...     dtype=jnp.bfloat16,
    ...     param_dtype=jnp.bfloat16,
    ...     rngs=nn.Rngs(0),
    ... )
"""

from .modeling_qwen import Qwen2ForCausalLM, Qwen2ForSequenceClassification, Qwen2Model
from .qwen_configuration import Qwen2Config

__all__ = (
    "Qwen2Config",
    "Qwen2ForCausalLM",
    "Qwen2ForSequenceClassification",
    "Qwen2Model",
)
