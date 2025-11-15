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

**Partition Spec Utilities:**
    - `get_moe_partition_spec`: Generate partition specs for MoE weight tensors

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
    - **3D Expert Mesh**: Simplified mesh combining FSDP, EP, and SP into single expert dimension

**Execution Modes:**
    - Fused MoE: Optimized grouped matmul with shard_map (best for TPU/GPU)
    - Standard MoE: Traditional permute-compute-unpermute (most flexible)
    - Dense MoE: Per-token einsum operations (debugging/fallback)

**Communication Patterns:**
    - Ring-of-Experts: Efficient all-gather pattern for expert parallelism
    - All-to-All: Ragged all-to-all communication for token redistribution
    - Automatic selection based on mesh configuration

Recent Improvements
-------------------

**3D Expert Mesh (v0.0.81+):**
    The MoE implementation has been refactored to use a simplified 3D expert mesh
    that combines FSDP, EP, and SP axes into a single unified expert dimension.
    This provides:

    - Cleaner sharding specifications with (dp, expert, tp) layout
    - Simplified partition spec generation via `get_moe_partition_spec`
    - Better compatibility with grouped matmul kernels
    - Automatic resharding between 5D model mesh and 3D expert mesh

    The 3D expert mesh is created automatically in `BaseMoeModule._create_expert_mesh()`
    and used internally for all MoE operations, with automatic resharding at boundaries.

Example Usage
-------------

**Basic MoE Layer:**

    >>> from easydel.layers.moe import (
    ...     BaseMoeModule,
    ...     MoEMethods,
    ...     MoeRoutingStrategy,
    ...     MoeLoadBalancingStrategy,
    ... )
    >>> from flax import nnx as nn
    >>>
    >>> # Configure MoE execution
    >>> config.moe_method = MoEMethods.FUSED_MOE
    >>> config.routing_strategy = MoeRoutingStrategy.TOP_K
    >>> config.load_balancing_strategy = MoeLoadBalancingStrategy.STANDARD
    >>> config.n_routed_experts = 8
    >>> config.num_experts_per_tok = 2
    >>>
    >>> # Create custom MoE layer by extending BaseMoeModule
    >>> class MyMoELayer(BaseMoeModule):
    ...     def __init__(self, config, rngs):
    ...         super().__init__(config)
    ...         self.gate = nn.Linear(config.hidden_size, config.n_routed_experts, rngs=rngs)
    ...         # Initialize expert FFN weights...
    ...
    ...     def __call__(self, hidden_states):
    ...         output, router_logits = self.moe_call(
    ...             hidden_state=hidden_states,
    ...             gate_layer=self.gate,
    ...             wi_kernel=self.wi_kernel,
    ...             wu_kernel=self.wu_kernel,
    ...             wd_kernel=self.wd_kernel,
    ...             act_fn=nn.silu,
    ...         )
    ...         return output

**Custom Routing with Hooks:**

    >>> from easydel.layers.moe import MoeFusedHooks
    >>>
    >>> # Define custom weight refinement
    >>> def temperature_scaling(weights):
    ...     temperature = 0.5
    ...     return jax.nn.softmax(weights / temperature)
    >>>
    >>> # Create hooks with custom logic
    >>> hooks = MoeFusedHooks(refine_weights_hook=temperature_scaling)
    >>>
    >>> # Use hooks in MoE layer
    >>> moe_layer = MyMoELayer(config, moe_hooks=hooks, rngs=rngs)

**Distributed Training Setup:**

    >>> import jax
    >>> from jax.sharding import Mesh
    >>> from eformer.escale import PartitionManager
    >>>
    >>> # Create 5D mesh for model training
    >>> devices = jax.devices()
    >>> mesh = Mesh(
    ...     devices.reshape(1, 1, 4, 2, 1),  # (dp, fsdp, ep, tp, sp)
    ...     axis_names=("dp", "fsdp", "expert", "tensor", "sequence")
    ... )
    >>>
    >>> # Configure partition manager
    >>> config.mesh = mesh
    >>> config.partition_manager = PartitionManager(mesh, ...)
    >>>
    >>> # MoE layer automatically creates 3D expert mesh:
    >>> # Combines FSDP*EP*SP into single expert axis (4*1 = 4)
    >>> # Resulting expert_mesh shape: (dp=1, expert=4, tp=2)

See Also
--------

- Module documentation: :class:`easydel.layers.moe.moe.BaseMoeModule`
- Utilities documentation: :mod:`easydel.layers.moe.utils`
- Linear layers documentation: :mod:`easydel.layers.moe.linear`

Notes
-----

The modules are designed to work seamlessly with JAX distributed meshes and
EFormer's PartitionManager for automatic sharding. All implementations support
gradient checkpointing and mixed precision training.

For optimal performance on TPUs, use `MoEMethods.FUSED_MOE` with the grouped
matmul kernel. On GPUs, both FUSED_MOE and STANDARD_MOE work well, with FUSED_MOE
providing better performance for large expert counts.
"""

from .linear import ColumnParallelMoELinear, ParallelMoELinear, RowParallelMoELinear
from .moe import BaseMoeModule
from .utils import (
    MoeFusedHooks,
    MoeLoadBalancingStrategy,
    MoEMethods,
    MoeMetrics,
    MoeRoutingStrategy,
    get_moe_partition_spec,
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
    "get_moe_partition_spec",
)
