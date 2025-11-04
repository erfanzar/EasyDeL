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

"""EasyDeL Mixture-of-Experts (MoE) layers and utilities.

This package provides comprehensive building blocks for implementing Mixture of Experts
models in JAX with EasyDeL, supporting various routing strategies, load balancing techniques,
and distributed training optimizations.

Core Components
---------------

**Base Classes:**
    - `BaseMoeModule`: Abstract base class for MoE implementations with routing,
      permutation, metrics computation, and distributed execution utilities.

**Linear Layers:**
    - `ParallelMoELinear`: Batched per-expert linear transformation layer with support
      for ragged/grouped matrix multiplication
    - `RowParallelMoELinear`: Row-parallel variant (input dimension partitioned)
    - `ColumnParallelMoELinear`: Column-parallel variant (output dimension partitioned)

**Enumerations:**
    - `MoEMethods`: Execution methods (FUSED_MOE, STANDARD_MOE, DENSE_MOE)
    - `MoeRoutingStrategy`: Token routing strategies (TOP_K, SWITCH, EXPERT_CHOICE, HASH)
    - `MoeLoadBalancingStrategy`: Load balancing loss strategies (STANDARD, SWITCH_TRANSFORMER, EXPERT_CHOICE)

**Data Classes:**
    - `MoeFusedHooks`: Hook system for custom interventions during MoE execution
    - `MoeMetrics`: Container for MoE metrics (expert loads, routing entropy, auxiliary losses)

Features
--------

**Routing Strategies:**
    - Top-K routing with weight normalization
    - Switch Transformer (top-1) routing
    - Expert Choice routing (inverted selection)
    - Hash-based deterministic routing
    - Custom hooks for implementing novel routing methods

**Load Balancing:**
    - Standard load balancing loss
    - Switch Transformer auxiliary loss
    - Expert Choice variance-based loss
    - Router z-loss for stability

**Distributed Training:**
    - Expert Parallelism (EP): Partition experts across devices
    - Tensor Parallelism (TP): Partition weight matrices within experts
    - Data Parallelism (DP): Replicate across data batches
    - Fully Sharded Data Parallel (FSDP): Memory-efficient parameter sharding
    - Sequence Parallelism (SP): Partition sequence dimension
    - Expert Tensor Mode: Alternative sharding with experts on TP axis

**Execution Modes:**
    - Fused MoE: Optimized grouped matmul with shard_map (best for TPU/GPU)
    - Standard MoE: Traditional permute-compute-unpermute (most flexible)
    - Dense MoE: Per-token einsum operations (debugging/fallback)

Example Usage
-------------

    >>> from easydel.layers.moe import (
    ...     BaseMoeModule,
    ...     MoEMethods,
    ...     MoeRoutingStrategy,
    ...     MoeLoadBalancingStrategy,
    ... )
    >>>
    >>> # Configure MoE execution
    >>> config.moe_method = MoEMethods.FUSED_MOE
    >>> config.routing_strategy = MoeRoutingStrategy.TOP_K
    >>> config.load_balancing_strategy = MoeLoadBalancingStrategy.STANDARD
    >>>
    >>> # Create custom MoE layer by extending BaseMoeModule
    >>> class MyMoELayer(BaseMoeModule):
    ...     def __init__(self, config):
    ...         super().__init__(config)
    ...         # Initialize gate and expert layers
    ...
    ...     def __call__(self, hidden_states):
    ...         # Use helper methods for MoE computation
    ...         return self.moe_call(...)

See Also
--------

- Module documentation: easydel.layers.moe.moe.BaseMoeModule
- Utilities documentation: easydel.layers.moe.utils
- Linear layers documentation: easydel.layers.moe.linear

Notes
-----

The modules are designed to work seamlessly with JAX distributed meshes and
EFormer's PartitionManager for automatic sharding. All implementations support
gradient checkpointing and mixed precision training.
"""

from .linear import ColumnParallelMoELinear, ParallelMoELinear, RowParallelMoELinear
from .moe import BaseMoeModule
from .utils import (
    MoeFusedHooks,
    MoeLoadBalancingStrategy,
    MoEMethods,
    MoeMetrics,
    MoeRoutingStrategy,
)

__all__ = (
    "BaseMoeModule",
    "ColumnParallelMoELinear",
    "MoEMethods",
    "MoeFusedHooks",
    "MoeLoadBalancingStrategy",
    "MoeMetrics",
    "MoeRoutingStrategy",
    "ParallelMoELinear",
    "RowParallelMoELinear",
)
