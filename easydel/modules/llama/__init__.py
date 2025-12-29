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

"""LLaMA - Large Language Model Meta AI architecture.

This module implements the LLaMA (Large Language Model Meta AI) family of models,
a collection of foundation language models ranging from 7B to 65B+ parameters.
LLaMA is designed to be efficient, accessible, and performant for research and
commercial applications.

**Key Features**:
- Decoder-only transformer architecture with RoPE (Rotary Position Embeddings)
- RMSNorm for layer normalization instead of LayerNorm
- SwiGLU activation function in feedforward layers
- Grouped-query attention (GQA) for efficient inference
- Support for extended context lengths via rope_scaling

**Architecture Highlights**:
- Pre-normalization using RMSNorm for training stability
- Rotary embeddings applied to query and key projections
- Gated MLP blocks for improved expressiveness
- Optional gradient checkpointing and quantization support
- Flexible attention patterns via layer_types parameter

**Available Model Variants**:
- LlamaModel: Base transformer model outputting hidden states
- LlamaForCausalLM: Model with language modeling head for text generation
- LlamaForSequenceClassification: Model with classification head for sequence tasks
- VisionLlamaConfig: Extended configuration for vision-augmented models

Example:
    >>> from easydel.modules.llama import (
    ...     LlamaConfig,
    ...     LlamaForCausalLM,
    ... )
    >>> config = LlamaConfig(
    ...     hidden_size=4096,
    ...     num_hidden_layers=32,
    ...     num_attention_heads=32,
    ...     num_key_value_heads=8,  # GQA with 4x fewer KV heads
    ... )
    >>> model = LlamaForCausalLM(
    ...     config=config,
    ...     dtype=jnp.bfloat16,
    ...     param_dtype=jnp.bfloat16,
    ...     rngs=nn.Rngs(0),
    ... )
"""

from .llama_configuration import LlamaConfig, VisionLlamaConfig
from .modeling_llama import LlamaForCausalLM, LlamaForSequenceClassification, LlamaModel

__all__ = (
    "LlamaConfig",
    "LlamaForCausalLM",
    "LlamaForSequenceClassification",
    "LlamaModel",
    "VisionLlamaConfig",
)
