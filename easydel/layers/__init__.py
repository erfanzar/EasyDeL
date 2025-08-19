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

"""EasyDeL Layers Module.

Provides optimized neural network layers and components for building
deep learning models in JAX/Flax. Includes advanced attention mechanisms,
efficient caching systems, quantization support, and various layer types.

Key Components:
    - Linear and parallel linear layers with sharding support
    - Multiple attention mechanisms (Flash, Ring, Paged, etc.)
    - Advanced caching systems for transformers and state-space models
    - Quantization layers (8-bit, NF4)
    - Normalization layers (RMSNorm, LayerNorm)
    - Mixture of Experts (MoE) layers
    - Rotary position embeddings (RoPE)

Submodules:
    attention: Various attention implementations
    attention_operator: Low-level attention operators
    caching: KV-cache and state caching systems
    linear: Linear and parallel linear layers
    moe: Mixture of Experts layers
    norms: Normalization layers
    ops: Custom operations (GLA, Lightning attention)
    quantization: Quantized layer implementations
    rotary_embedding: RoPE and position embeddings

Example:
    >>> from easydel.layers import ParallelLinear, RMSNorm
    >>> from easydel.layers.attention import FlashAttention
    >>>
    >>> # Create a parallel linear layer
    >>> linear = ParallelLinear(
    ...     features=768,
    ...     use_bias=False,
    ...     dtype=jnp.bfloat16
    ... )
    >>>
    >>> # Use Flash Attention
    >>> attn = FlashAttention(
    ...     num_heads=12,
    ...     head_dim=64
    ... )

Note:
    This module provides the building blocks for constructing
    efficient neural networks with support for distributed training,
    mixed precision, and hardware acceleration.
"""
