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

"""Gemma - Google's family of lightweight, state-of-the-art open models.

This module implements the Gemma architecture, a decoder-only transformer model
developed by Google. Gemma models are designed to be efficient and performant,
building on the same research and technology used for the Gemini models.

**Key Features**:
- Decoder-only transformer architecture with RoPE (Rotary Position Embeddings)
- RMSNorm for layer normalization with learnable scale parameters
- Grouped-query attention (GQA) for improved efficiency
- Gated MLP blocks with approximate GeLU activation
- Input embedding scaling by sqrt(hidden_size) for stability

**Architecture Details**:
- Multi-head attention with configurable key-value heads
- Pre-normalization (LayerNorm before attention/FFN)
- Optional gradient checkpointing for memory efficiency
- Support for various quantization schemes via bits parameter

**Available Model Variants**:
- GemmaModel: Base model with transformer layers
- GemmaForCausalLM: Model with language modeling head for text generation
- GemmaForSequenceClassification: Model with classification head for sequence labeling

Example:
    >>> from easydel.modules.gemma import (
    ...     GemmaConfig,
    ...     GemmaForCausalLM,
    ... )
    >>> config = GemmaConfig(
    ...     hidden_size=3072,
    ...     num_hidden_layers=28,
    ...     num_attention_heads=16,
    ... )
    >>> model = GemmaForCausalLM(
    ...     config=config,
    ...     dtype=jnp.bfloat16,
    ...     param_dtype=jnp.bfloat16,
    ...     rngs=nn.Rngs(0),
    ... )
"""

from .gemma_configuration import GemmaConfig
from .modeling_gemma import GemmaForCausalLM, GemmaForSequenceClassification, GemmaModel

__all__ = (
    "GemmaConfig",
    "GemmaForCausalLM",
    "GemmaForSequenceClassification",
    "GemmaModel",
)
