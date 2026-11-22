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

"""Partition specification and sharding constraint utilities.

This submodule provides tools for creating and managing JAX partition
specifications (PartitionSpecs) and applying sharding constraints to arrays.
It bridges logical model dimensions (like batch, sequence, heads) to physical
mesh dimensions.

Key Components:
    - **PartitionAxis**: Configuration class mapping logical axes to mesh dimensions
    - **PartitionManager**: Context manager for applying sharding constraints
    - **auto_partition_spec**: Automatic partition spec generation based on shapes
    - **with_sharding_constraint**: Apply sharding constraints with auto-correction

Functions for Partition Specs:
    auto_partition_spec: Generate optimal PartitionSpec for an array.
    vrn_auto_partition_spec: Variant with different dimension ordering.
    auto_shard_array: Shard array using automatically determined spec.
    match_partition_rules: Match parameter names to partition rules via regex.
    create_pattern_based_partition_spec: Create specs using string patterns.

Functions for Mesh Introspection:
    get_incontext_mesh: Get the active mesh from current context.
    get_mesh_axis_names: Get axis names from a mesh.
    get_mesh_axis_size: Get size of mesh axes using collectives.
    get_axes_size_in_mesh: Get product of axis sizes from mesh shape.

Functions for Sharding Analysis:
    analyze_sharding_strategy: Analyze effectiveness of a sharding strategy.
    extract_shardings: Extract NamedShardings from a sharded PyTree.
    get_partition_spec: Get partition specs from a sharded PyTree.

Example:
    >>> from eformer.escale.partition import PartitionAxis, auto_partition_spec
    >>> # Create partition configuration
    >>> paxis = PartitionAxis(
    ...     batch_axis=('fsdp', 'dp'),
    ...     head_axis='tp',
    ... )
    >>> # Or auto-generate specs
    >>> spec = auto_partition_spec(array, mesh=mesh)
"""

from .auto_spec import (
    auto_namedsharding,
    auto_partition_spec,
    auto_shard_array,
    convert_sharding_strategy,
    optimize_sharding_for_memory,
    validate_sharding_config,
    vrn_auto_partition_spec,
)
from .constraints import (
    analyze_sharding_strategy,
    create_pattern_based_partition_spec,
    extract_sharding_structure,
    extract_shardings,
    get_axes_size_in_mesh,
    get_corrected_named_sharding,
    get_incontext_mesh,
    get_mesh_axis_names,
    get_mesh_axis_size,
    get_names_from_partition_spec,
    get_partition_spec,
    get_shardings_with_structure,
    get_submesh_device_index,
    make_shard_and_gather_fns,
    match_partition_rules,
    names_in_current_mesh,
    with_sharding_constraint,
)
from .manager import (
    PartitionAxis,
    PartitionManager,
    apply_logical_sharding,
    get_current_partition_manager,
    get_partition_manager,
)

__all__ = (
    "PartitionAxis",
    "PartitionManager",
    "analyze_sharding_strategy",
    "apply_logical_sharding",
    "auto_namedsharding",
    "auto_partition_spec",
    "auto_shard_array",
    "convert_sharding_strategy",
    "create_pattern_based_partition_spec",
    "extract_sharding_structure",
    "extract_shardings",
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
    "make_shard_and_gather_fns",
    "match_partition_rules",
    "names_in_current_mesh",
    "optimize_sharding_for_memory",
    "validate_sharding_config",
    "vrn_auto_partition_spec",
    "with_sharding_constraint",
)
