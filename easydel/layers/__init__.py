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

This module provides optimized neural network layers and components for building
deep learning models in JAX/Flax. It includes advanced attention mechanisms,
efficient caching systems, quantization support, and various layer types designed
for high-performance distributed training and inference.

The layers module serves as the foundation for building transformer-based models
with support for modern optimization techniques including FlashAttention, Ring
Attention, paged attention, and various quantization strategies.

Key Components:
    Linear Layers:
        - ParallelLinear: Tensor-parallel linear layer with automatic sharding
        - ColumnParallelLinear: Column-wise partitioned linear layer
        - RowParallelLinear: Row-wise partitioned linear layer
        - Quantized variants for memory-efficient inference

    Attention Mechanisms:
        - FlexibleAttentionModule: Unified interface for multiple attention types
        - UnifiedAttention: Base class supporting RoPE, MLA, and ALiBi
        - Support for Flash, Ring, Splash, Paged, and SDPA attention

    Caching Systems:
        - TransformerCacheView: Standard KV cache for autoregressive generation
        - RaggedPagesCacheView: Paged attention cache for variable-length batches
        - UnifiedAttentionCacheView: vLLM-style unified paged attention cache

    Quantization:
        - QuantizationConfig: Configuration for various quantization schemes
        - EasyQuantizer: Unified quantization interface
        - Support for INT8, NF4, MXFP4, MXFP8 quantization

    Normalization:
        - RMSNorm: Root Mean Square Layer Normalization
        - LayerNorm: Standard Layer Normalization

    Position Embeddings:
        - RotaryEmbedding: Standard RoPE implementation
        - Llama3RotaryEmbedding: Llama 3 style RoPE with scaling
        - YaRNScalingRotaryEmbedding: YaRN extended context RoPE
        - DeepseekScalingRotaryEmbedding: DeepSeek-style RoPE

    Mixture of Experts:
        - BaseMoeModule: Base class for MoE implementations
        - MoE routing strategies and load balancing

Submodules:
    attention:
        Core attention implementations including FlexibleAttentionModule
        and the AttentionMechanisms enum.

    attention_unified:
        UnifiedAttention class supporting standard, MLA, and ALiBi attention.

    caching:
        KV-cache implementations for transformers and state-space models.

    components:
        Reusable neural network components including linear layers,
        embeddings, normalization, MoE layers, and rotary embeddings.

    decoder_base:
        Base utilities for transformer decoder layers.

    operations:
        Low-level attention operation implementations and registry.

    quantization:
        Quantization utilities and configurations.

Example:
    Basic usage with attention and linear layers::

        >>> import jax.numpy as jnp
        >>> from easydel.layers import FlexibleAttentionModule, AttentionMechanisms
        >>> from easydel.layers.components import ParallelLinear, RMSNorm
        >>>
        >>> # Create a parallel linear layer with automatic sharding
        >>> linear = ParallelLinear(
        ...     in_features=768,
        ...     out_features=768,
        ...     use_bias=False,
        ...     dtype=jnp.bfloat16
        ... )
        >>>
        >>> # Create attention with automatic hardware optimization
        >>> attn = FlexibleAttentionModule(
        ...     base_config=config,
        ...     softmax_scale=0.125,
        ...     attn_mechanism=AttentionMechanisms.AUTO
        ... )

    Using quantization for memory-efficient inference::

        >>> from easydel.layers.quantization import QuantizationConfig, QuantizationType
        >>>
        >>> # Configure NF4 quantization
        >>> quant_config = QuantizationConfig(
        ...     quantization_type=QuantizationType.NF4,
        ...     block_size=64
        ... )

    Using the UnifiedAttention base class::

        >>> from easydel.layers.attention_unified import UnifiedAttention
        >>>
        >>> class CustomAttention(UnifiedAttention):
        ...     def _postprocess_qkv(self, q, k, v):
        ...         # Apply custom Q/K normalization
        ...         q = self.q_norm(q)
        ...         k = self.k_norm(k)
        ...         return q, k, v

Note:
    This module is designed for high-performance distributed training and
    inference with support for:
    - Multi-device tensor parallelism via JAX sharding
    - Mixed precision training with bfloat16/float16
    - Hardware-specific optimizations for TPU and GPU
    - Memory-efficient attention and caching mechanisms

See Also:
    - easydel.infra: Infrastructure and configuration utilities
    - easydel.models: Pre-built model implementations using these layers
    - eformer: External library for efficient attention operations
"""
