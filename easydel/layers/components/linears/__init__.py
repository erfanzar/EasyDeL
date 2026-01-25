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

"""Linear layer components for EasyDeL.

This package provides various linear transformation layers optimized for
distributed training and inference. It includes standard parallel linear layers,
quantized variants for memory efficiency, and specialized layers for
Mixture-of-Experts (MoE) architectures.

Modules:
    _linear: Core parallel linear layers (ParallelLinear, Row/ColumnParallelLinear).
    _linear_moe: MoE-specific linear layers with grouped matmul support.
    _linear_quantized: Quantized linear layers (INT8, NF4, MXFP formats).
    _linear_moe_quantized: Placeholder for quantized MoE layers.
    _utils: Low-level utilities for quantized operations.

Classes:
    ParallelLinear: Base linear layer with optional parallelism support.
    RowParallelLinear: Row-parallel linear (input dimension partitioned).
    ColumnParallelLinear: Column-parallel linear (output dimension partitioned).
    ParallelLinearQuantized: Quantized linear with multiple format support.
    RowParallelLinearQuantized: Row-parallel quantized linear.
    ColumnParallelLinearQuantized: Column-parallel quantized linear.
    ParallelMoELinear: Base MoE linear layer for expert-grouped computation.
    RowParallelMoELinear: Row-parallel MoE linear.
    ColumnParallelMoELinear: Column-parallel MoE linear.

Parallelism Strategies:
    **Row Parallelism**: The input dimension is partitioned across devices.
    Each device holds a subset of input features and computes partial results
    that are then reduced (summed) across devices. Typically used for the
    second layer in MLP blocks.

    **Column Parallelism**: The output dimension is partitioned across devices.
    Each device computes a different slice of output features independently
    without requiring communication. Typically used for the first layer in
    MLP blocks.

Example:
    Basic usage with standard linear layers:

    >>> from easydel.layers.components.linears import (
    ...     ColumnParallelLinear,
    ...     RowParallelLinear,
    ... )
    >>> from flax import nnx as nn
    >>>
    >>> # Create a two-layer MLP with tensor parallelism
    >>> up_proj = ColumnParallelLinear(768, 3072, rngs=nn.Rngs(0))
    >>> down_proj = RowParallelLinear(3072, 768, rngs=nn.Rngs(1))
    >>>
    >>> # Forward pass
    >>> x = jnp.ones((32, 768))
    >>> hidden = jax.nn.gelu(up_proj(x))
    >>> output = down_proj(hidden)

    Using quantized layers:

    >>> from easydel.layers.components.linears import ColumnParallelLinearQuantized
    >>> from easydel.layers.components.quants import QuantizationConfig, QuantizationType
    >>>
    >>> config = QuantizationConfig(dtype=QuantizationType.INT8)
    >>> quant_layer = ColumnParallelLinearQuantized(
    ...     in_features=768,
    ...     out_features=3072,
    ...     config=config,
    ...     rngs=nn.Rngs(0)
    ... )

    MoE linear layers:

    >>> from easydel.layers.components.linears import (
    ...     ColumnParallelMoELinear,
    ...     RowParallelMoELinear,
    ... )
    >>>
    >>> # Create MoE FFN layers
    >>> wi = ColumnParallelMoELinear(8, 768, 3072, rngs=rngs)
    >>> wd = RowParallelMoELinear(8, 3072, 768, rngs=rngs)
    >>>
    >>> # Forward with grouped tokens
    >>> wi_out = wi(sorted_tokens, group_sizes, sorted_experts)
    >>> wd_out = wd(activated, group_sizes, sorted_experts)
"""

from ._linear import ColumnParallelLinear, ParallelLinear, RowParallelLinear
from ._linear_moe import ColumnParallelMoELinear, ParallelMoELinear, RowParallelMoELinear
from ._linear_quantized import ColumnParallelLinearQuantized, ParallelLinearQuantized, RowParallelLinearQuantized

__all__ = (
    "ColumnParallelLinear",
    "ColumnParallelLinearQuantized",
    "ColumnParallelMoELinear",
    "ParallelLinear",
    "ParallelLinearQuantized",
    "ParallelMoELinear",
    "RowParallelLinear",
    "RowParallelLinearQuantized",
    "RowParallelMoELinear",
)
