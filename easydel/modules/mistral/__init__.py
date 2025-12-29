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

"""Mistral - Efficient transformer model with sliding window attention.

This module implements the Mistral AI architecture, featuring efficient sliding window
attention for handling longer sequences with reduced computational cost. Mistral models
are designed to balance performance and efficiency, making them suitable for both
research and production deployments.

**Key Features**:
- Sliding window attention for efficient long-context processing
- Grouped-query attention (GQA) for improved inference speed
- RMSNorm for stable training dynamics
- SwiGLU activation in MLP blocks
- RoPE (Rotary Position Embeddings) for position encoding

**Architecture Highlights**:
- Configurable sliding window size for local attention patterns
- Pre-normalization architecture for training stability
- Support for mixed sliding/full attention across layers
- Optional rope_scaling for extended context lengths
- Efficient memory usage through attention windowing

**Available Model Variants**:
- MistralModel: Base transformer model with hidden state outputs
- MistralForCausalLM: Model with language modeling head for generation
- MistralForSequenceClassification: Model with classification head

Example:
    >>> from easydel.modules.mistral import (
    ...     MistralConfig,
    ...     MistralForCausalLM,
    ... )
    >>> config = MistralConfig(
    ...     hidden_size=4096,
    ...     num_hidden_layers=32,
    ...     num_attention_heads=32,
    ...     num_key_value_heads=8,  # GQA
    ...     sliding_window=4096,  # Local attention window
    ... )
    >>> model = MistralForCausalLM(
    ...     config=config,
    ...     dtype=jnp.bfloat16,
    ...     param_dtype=jnp.bfloat16,
    ...     rngs=nn.Rngs(0),
    ... )
"""

from .mistral_configuration import MistralConfig
from .modeling_mistral import MistralForCausalLM, MistralForSequenceClassification, MistralModel

__all__ = (
    "MistralConfig",
    "MistralForCausalLM",
    "MistralForSequenceClassification",
    "MistralModel",
)
