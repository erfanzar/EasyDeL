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

"""EasyDeL Mixture-of-Experts layers and utilities.

This package exposes the core building blocks for MoE models, including:

- `BaseMoeModule`: abstract base with routing, metrics, and distributed helpers
- `ParallelMoELinear` (+ row/column specializations): per-expert linear layers
- Utility types and enums for routing/load-balancing and fused execution policies

The modules are designed to work with JAX distributed meshes and support
expert/tensor/data parallelism.
"""

from .linear import ColumnParallelMoELinear, ParallelMoELinear, RowParallelMoELinear
from .moe import BaseMoeModule
from .utils import (
    MoeFusedHooks,
    MoeLoadBalancingStrategy,
    MoeMetrics,
    MoeRoutingStrategy,
)

__all__ = (
    "BaseMoeModule",
    "ColumnParallelMoELinear",
    "MoeFusedHooks",
    "MoeLoadBalancingStrategy",
    "MoeMetrics",
    "MoeRoutingStrategy",
    "ParallelMoELinear",
    "RowParallelMoELinear",
)
