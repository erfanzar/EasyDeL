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

"""Neural network components for building transformer models.

This module provides reusable neural network components including linear layers,
embeddings, normalization layers, Mixture of Experts (MoE), rotary position
embeddings, and quantization utilities.

All components are designed for distributed training with JAX/Flax and support
tensor parallelism through automatic sharding.

Submodules:
    embeddings:
        Embedding layers with sharding support.

    linears:
        Linear layers with various parallelism strategies:
        - ParallelLinear: Basic parallel linear layer
        - ColumnParallelLinear: Column-wise tensor parallelism
        - RowParallelLinear: Row-wise tensor parallelism
        - Quantized variants for memory efficiency
        - MoE variants for expert-parallel computation

    moe:
        Mixture of Experts components:
        - BaseMoeModule: Base class for MoE implementations
        - Routing strategies (Top-K, Expert Choice)
        - Load balancing strategies
        - MoE metrics tracking

    norms:
        Normalization layers:
        - RMSNorm: Root Mean Square Layer Normalization

    quants:
        Quantization utilities:
        - QuantizationConfig: Configuration for quantization
        - EasyQuantizer: Unified quantization interface
        - Various straight-through quantization functions

    rotary_embedding:
        Rotary Position Embedding (RoPE) implementations:
        - Standard RoPE
        - Llama 3 style with scaling
        - YaRN for extended context
        - DeepSeek scaling
        - Phi-3 long context

Classes:
    Linear Layers:
        ParallelLinear: Basic tensor-parallel linear layer.
        ColumnParallelLinear: Column-partitioned linear (for QKV projections).
        RowParallelLinear: Row-partitioned linear (for output projections).
        ParallelLinearQuantized: Quantized variant of ParallelLinear.
        ColumnParallelLinearQuantized: Quantized column-parallel linear.
        RowParallelLinearQuantized: Quantized row-parallel linear.
        ParallelMoELinear: MoE-aware parallel linear.
        ColumnParallelMoELinear: Column-parallel MoE linear.
        RowParallelMoELinear: Row-parallel MoE linear.

    Embeddings:
        Embed: Embedding layer with vocabulary sharding.

    Normalization:
        RMSNorm: Root Mean Square Layer Normalization.

    Mixture of Experts:
        BaseMoeModule: Base class for MoE layer implementations.
        MoeRoutingStrategy: Enum for routing strategies (TOP_K, EXPERT_CHOICE).
        MoeLoadBalancingStrategy: Enum for load balancing approaches.
        MoeMetrics: Container for MoE training metrics.
        MoeFusedHooks: Hooks for fused MoE operations.
        MoEMethods: Collection of MoE utility methods.

    Rotary Embeddings:
        RotaryEmbedding: Standard RoPE implementation.
        Llama3RotaryEmbedding: Llama 3 RoPE with frequency scaling.
        YaRNScalingRotaryEmbedding: YaRN extended context RoPE.
        DeepseekScalingRotaryEmbedding: DeepSeek-style RoPE.
        DynamicNTKScalingRotaryEmbedding: Dynamic NTK scaling RoPE.
        LinearScalingRotaryEmbedding: Linear scaling RoPE.
        Phi3LongRoPEScaledRotaryEmbedding: Phi-3 long context RoPE.
        MultiModalRotaryEmbedding: RoPE for multimodal models.
        RopeConfig: Configuration for RoPE parameters.

    Quantization:
        QuantizationConfig: Configuration for quantization schemes.
        QuantizationType: Enum of quantization types.
        EasyQuantizer: Unified quantization interface.

Functions:
    Rotary Embedding Computation:
        get_rope: Get RoPE instance for a specific configuration.
        get_frequencies: Compute RoPE frequencies.
        get_inv_frequencies: Compute inverse frequencies for RoPE.
        compute_basic_frequencies: Standard frequency computation.
        compute_basic_inv_frequencies: Standard inverse frequency computation.
        compute_llama3_frequencies: Llama 3 frequency computation.
        compute_llama3_inv_frequencies: Llama 3 inverse frequencies.
        compute_yarn_frequencies: YaRN frequency computation.
        compute_yarn_inv_frequencies: YaRN inverse frequencies.
        compute_deepseek_frequencies: DeepSeek frequency computation.
        compute_dynamic_frequencies: Dynamic NTK frequency computation.
        compute_linear_frequencies: Linear scaling frequency computation.
        compute_phi3_frequencies: Phi-3 frequency computation.

    MoE Utilities:
        get_moe_partition_spec: Get partition spec for MoE layers.

    Quantization:
        quantize: Main quantization function.
        straight_through: Generic straight-through estimator.
        straight_through_8bit: 8-bit quantization.
        straight_through_nf4: NF4 quantization.
        straight_through_mxfp4: MXFP4 quantization.
        straight_through_mxfp8: MXFP8 quantization.
        straight_through_nvfp8: NVIDIA FP8 quantization.
        straight_through_1bit: 1-bit quantization.

Example:
    Using linear layers::

        >>> from easydel.layers.components import (
        ...     ColumnParallelLinear,
        ...     RowParallelLinear,
        ...     RMSNorm
        ... )
        >>> import jax.numpy as jnp
        >>>
        >>> # Create column-parallel projection (e.g., for QKV)
        >>> qkv_proj = ColumnParallelLinear(
        ...     in_features=768,
        ...     out_features=768 * 3,
        ...     use_bias=False,
        ...     dtype=jnp.bfloat16,
        ...     rngs=rngs
        ... )
        >>>
        >>> # Create row-parallel output projection
        >>> o_proj = RowParallelLinear(
        ...     in_features=768,
        ...     out_features=768,
        ...     dtype=jnp.bfloat16,
        ...     rngs=rngs
        ... )

    Using rotary embeddings::

        >>> from easydel.layers.components import RotaryEmbedding, get_rope
        >>>
        >>> # Create standard RoPE
        >>> rope = RotaryEmbedding(
        ...     dim=64,
        ...     max_position_embeddings=4096,
        ...     base=10000.0,
        ...     dtype=jnp.bfloat16
        ... )
        >>>
        >>> # Apply to query and key
        >>> q_rotated, k_rotated = rope(query, key, position_ids)

See Also:
    - easydel.layers.attention: Attention modules using these components
    - easydel.layers.decoder_base: Decoder layer utilities
"""

from .embeddings import Embed
from .linears import (
    ColumnParallelLinear,
    ColumnParallelLinearQuantized,
    ColumnParallelMoELinear,
    ParallelLinear,
    ParallelLinearQuantized,
    ParallelMoELinear,
    RowParallelLinear,
    RowParallelLinearQuantized,
    RowParallelMoELinear,
)
from .moe import (
    BaseMoeModule,
    MoeFusedHooks,
    MoeLoadBalancingStrategy,
    MoEMethods,
    MoeMetrics,
    MoeRoutingStrategy,
    get_moe_partition_spec,
)
from .norms import RMSNorm
from .quants import (
    EasyQuantizer,
    QuantizationConfig,
    QuantizationType,
    quantize,
    straight_through,
    straight_through_1bit,
    straight_through_8bit,
    straight_through_mxfp4,
    straight_through_mxfp8,
    straight_through_nf4,
    straight_through_nvfp8,
)
from .rotary_embedding import (
    DeepseekScalingRotaryEmbedding,
    DynamicNTKScalingRotaryEmbedding,
    LinearScalingRotaryEmbedding,
    Llama3RotaryEmbedding,
    MultiModalRotaryEmbedding,
    Phi3LongRoPEScaledRotaryEmbedding,
    RopeConfig,
    RotaryEmbedding,
    YaRNScalingRotaryEmbedding,
    compute_basic_frequencies,
    compute_basic_inv_frequencies,
    compute_deepseek_frequencies,
    compute_dynamic_frequencies,
    compute_linear_frequencies,
    compute_llama3_frequencies,
    compute_llama3_inv_frequencies,
    compute_phi3_frequencies,
    compute_yarn_frequencies,
    compute_yarn_inv_frequencies,
    get_frequencies,
    get_inv_frequencies,
    get_rope,
)

__all__ = (
    "BaseMoeModule",
    "ColumnParallelLinear",
    "ColumnParallelLinearQuantized",
    "ColumnParallelMoELinear",
    "DeepseekScalingRotaryEmbedding",
    "DynamicNTKScalingRotaryEmbedding",
    "EasyQuantizer",
    "Embed",
    "LinearScalingRotaryEmbedding",
    "Llama3RotaryEmbedding",
    "MoEMethods",
    "MoeFusedHooks",
    "MoeLoadBalancingStrategy",
    "MoeMetrics",
    "MoeRoutingStrategy",
    "MultiModalRotaryEmbedding",
    "ParallelLinear",
    "ParallelLinearQuantized",
    "ParallelMoELinear",
    "Phi3LongRoPEScaledRotaryEmbedding",
    "QuantizationConfig",
    "QuantizationType",
    "RMSNorm",
    "RopeConfig",
    "RotaryEmbedding",
    "RowParallelLinear",
    "RowParallelLinearQuantized",
    "RowParallelMoELinear",
    "YaRNScalingRotaryEmbedding",
    "compute_basic_frequencies",
    "compute_basic_inv_frequencies",
    "compute_deepseek_frequencies",
    "compute_dynamic_frequencies",
    "compute_linear_frequencies",
    "compute_llama3_frequencies",
    "compute_llama3_inv_frequencies",
    "compute_phi3_frequencies",
    "compute_yarn_frequencies",
    "compute_yarn_inv_frequencies",
    "get_frequencies",
    "get_inv_frequencies",
    "get_moe_partition_spec",
    "get_rope",
    "quantize",
    "straight_through",
    "straight_through_1bit",
    "straight_through_8bit",
    "straight_through_mxfp4",
    "straight_through_mxfp8",
    "straight_through_nf4",
    "straight_through_nvfp8",
)
