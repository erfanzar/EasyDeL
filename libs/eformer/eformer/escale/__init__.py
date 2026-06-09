# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
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

"""eScale: JAX sharding and mesh management utilities for distributed computation.

This module provides comprehensive utilities for managing JAX device meshes and
sharding strategies for distributed machine learning workloads. It supports
various parallelism paradigms including data parallelism, tensor parallelism,
fully-sharded data parallelism (FSDP), sequence parallelism, and expert parallelism.

Key Components:
    - **Mesh Creation**: Functions to create and manage JAX device meshes
      (`create_mesh`, `create_cpu_mesh`, `parse_mesh_from_string`)
    - **Sharding Rules**: Classes for defining automatic sharding strategies
      (`AutoShardingRule`, `ShapeBasedShardingRule`, `MemoryConstrainedShardingRule`)
    - **Partition Management**: Tools for defining and applying partition specifications
      (`PartitionAxis`, `PartitionManager`, `auto_partition_spec`)
    - **Constraint Application**: Functions for applying sharding constraints to arrays
      (`with_sharding_constraint`, `apply_logical_sharding`)
    - **Analysis Tools**: Utilities for analyzing and validating sharding strategies
      (`ShardingAnalyzer`, `analyze_sharding_strategy`)

Typical Usage:
    >>> from eformer.escale import create_mesh, PartitionAxis, PartitionManager
    >>> # Create a mesh with data and model parallelism
    >>> mesh = create_mesh(axis_dims=(2, 4), axis_names=('dp', 'tp'))
    >>> # Define partition configuration
    >>> paxis = PartitionAxis(data_parallel_axis='dp', tensor_parallel_axis='tp')
    >>> # Use within a mesh context
    >>> with mesh:
    ...     # Apply sharding to arrays
    ...     pass

See Also:
    - `eformer.escale.mesh`: Low-level mesh creation utilities
    - `eformer.escale.partition`: Partition specification and management
    - `eformer.escale.helpers`: Sharding rule implementations
"""

from jax.sharding import NamedSharding, PartitionSpec

from .helpers import (
    AutoShardingRule,
    CompositeShardingRule,
    MemoryConstrainedShardingRule,
    ShapeBasedShardingRule,
    ShardingAnalyzer,
    ShardingRule,
    barrier_sync,
)
from .mesh import (
    MeshPartitionHelper,
    cpu_context,
    create_cpu_mesh,
    create_mesh,
    force_cpu,
    parse_mesh_from_string,
)
from .partition import (
    PartitionAxis,
    PartitionManager,
    analyze_sharding_strategy,
    apply_logical_sharding,
    auto_namedsharding,
    auto_partition_spec,
    auto_shard_array,
    convert_sharding_strategy,
    create_pattern_based_partition_spec,
    extract_sharding_structure,
    extract_shardings,
    get_axes_size_in_mesh,
    get_corrected_named_sharding,
    get_current_partition_manager,
    get_incontext_mesh,
    get_mesh_axis_names,
    get_mesh_axis_size,
    get_names_from_partition_spec,
    get_partition_manager,
    get_partition_spec,
    get_shardings_with_structure,
    get_submesh_device_index,
    make_shard_and_gather_fns,
    match_partition_rules,
    names_in_current_mesh,
    optimize_sharding_for_memory,
    validate_sharding_config,
    vrn_auto_partition_spec,
    with_sharding_constraint,
)

lax_reshard = with_sharding_constraint

__all__ = (
    "AutoShardingRule",
    "CompositeShardingRule",
    "MemoryConstrainedShardingRule",
    "MeshPartitionHelper",
    "NamedSharding",
    "PartitionAxis",
    "PartitionManager",
    "PartitionSpec",
    "ShapeBasedShardingRule",
    "ShardingAnalyzer",
    "ShardingRule",
    "analyze_sharding_strategy",
    "apply_logical_sharding",
    "auto_namedsharding",
    "auto_partition_spec",
    "auto_shard_array",
    "barrier_sync",
    "convert_sharding_strategy",
    "cpu_context",
    "create_cpu_mesh",
    "create_mesh",
    "create_pattern_based_partition_spec",
    "extract_sharding_structure",
    "extract_shardings",
    "force_cpu",
    "get_axes_size_in_mesh",
    "get_corrected_named_sharding",
    "get_current_partition_manager",
    "get_incontext_mesh",
    "get_mesh_axis_names",
    "get_mesh_axis_size",
    "get_names_from_partition_spec",
    "get_partition_manager",
    "get_partition_spec",
    "get_shardings_with_structure",
    "get_submesh_device_index",
    "lax_reshard",
    "make_shard_and_gather_fns",
    "match_partition_rules",
    "names_in_current_mesh",
    "optimize_sharding_for_memory",
    "parse_mesh_from_string",
    "validate_sharding_config",
    "vrn_auto_partition_spec",
    "with_sharding_constraint",
)
