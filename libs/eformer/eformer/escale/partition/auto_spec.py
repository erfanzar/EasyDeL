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


import typing as tp

import chex
import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from .constraints import get_incontext_mesh, with_sharding_constraint


def auto_partition_spec(
    x: chex.Array,
    mesh: Mesh | None = None,
    names: list[str | tuple[str, ...]] | None = None,
    min_sharding_size: int | None = None,
    reverse: bool = False,
) -> PartitionSpec:
    """
    Create an optimized PartitionSpec to shard an array across a device mesh.

    Args:
            x: The input array to be sharded.
            mesh: The device mesh to shard across. If None, uses the current thread's mesh.
            names: List of mesh axis names to use for sharding. If None, derives from mesh shape.
            min_sharding_size: Minimum size of array to shard. If None, uses mesh device count.
            reverse: If True, reverses dimension sorting order for sharding assignment.

    Returns:
            PartitionSpec: Optimized sharding specification for the input array.

    Raises:
            ValueError: If mesh is unavailable or invalid names are provided.
            TypeError: If input types are incorrect.
    """
    if not isinstance(x, chex.Array | np.ndarray):
        raise TypeError(f"Expected array input, got {type(x)}")

    if mesh is None:
        mesh = get_incontext_mesh()

    min_sharding_size = min_sharding_size or np.prod(mesh.devices.shape)

    array_size = np.prod(x.shape)
    if array_size < min_sharding_size:
        return PartitionSpec()

    if not names:
        names = [mesh.axis_names[i] for i in np.argsort([-s for s in mesh.devices.shape])]

    mesh_sizes = {
        name: np.prod([mesh.shape[n] for n in name]) if isinstance(name, tuple) else mesh.shape[name] for name in names
    }

    dim_indices = np.argsort([-dim if not reverse else dim for dim in x.shape])

    partition_spec = [None] * len(x.shape)
    remaining_names = list(dict.fromkeys(names))

    for dim_idx in dim_indices:
        dim_size = x.shape[dim_idx]

        best_name = None
        for name in remaining_names:
            mesh_size = mesh_sizes[name]
            if dim_size % mesh_size == 0:
                best_name = name
                break

        if best_name:
            partition_spec[dim_idx] = best_name
            remaining_names.remove(best_name)

        if not remaining_names:
            break

    return PartitionSpec(*partition_spec)


def vrn_auto_partition_spec(
    x: chex.Array,
    mesh: Mesh | None = None,
    names: list[str | tuple[str, ...]] | None = None,
    min_sharding_size: int | None = None,
    reverse: bool = False,
) -> PartitionSpec:
    """
    Create an optimized PartitionSpec to shard an array across a device mesh.

    Args:
            x: The input array to be sharded.
            mesh: The device mesh to shard across. If None, uses the current thread's mesh.
            names: List of mesh axis names to use for sharding. If None, derives from mesh shape.
            min_sharding_size: Minimum size of array to shard. If None, uses the product of mesh device shape.
            reverse: If True, reverses the sorting order of array dimensions.

    Returns:
            A PartitionSpec describing optimal array sharding.

    Raises:
            ValueError: If mesh is unavailable or invalid names are provided.
            TypeError: If input types are incorrect.
    """
    if not isinstance(x, np.ndarray | chex.Array):
        raise TypeError(f"Expected array input, got {type(x)}")

    if mesh is None:
        mesh = get_incontext_mesh()
    min_sharding_size = min_sharding_size or int(np.prod(mesh.devices.shape))
    array_size = np.prod(x.shape)
    if array_size < min_sharding_size:
        return PartitionSpec()

    if not names:
        names = [mesh.axis_names[i] for i in np.argsort([-s for s in mesh.devices.shape])]

    mesh_sizes = {
        name: np.prod([mesh.shape[n] for n in name]) if isinstance(name, tuple) else mesh.shape[name] for name in names
    }

    partition_spec = [None] * len(x.shape)
    dim_order = np.argsort([-dim for dim in x.shape] if not reverse else x.shape)
    remaining_names = names.copy()
    for dim_idx in dim_order:
        dim_size = x.shape[dim_idx]
        for name in remaining_names:
            mesh_size = mesh_sizes[name]

            if dim_size % mesh_size == 0:
                partition_spec[dim_idx] = name
                remaining_names.remove(name)
                break

    return PartitionSpec(*partition_spec)


def auto_shard_array(
    x: chex.Array,
    mesh: Mesh | None = None,
    names: list[str | tuple[str, ...]] | None = None,
    min_sharding_size: int | None = None,
    reverse: bool = False,
):
    """
    Shards an array across a device mesh according to an automatically derived PartitionSpec.

    This function acts as a wrapper around `pjit(x, in_axis_resources=...)`.

    Args:
            x: The input array to be sharded.
            mesh: The device mesh to shard across. If None, uses the current thread's mesh.
            names: List of mesh axis names to use for sharding. If None, derives from mesh shape.
            min_sharding_size: Minimum size of array to shard. If None, uses the product of mesh device shape.
            reverse: If True, reverses the sorting order of array dimensions.

    Returns:
            The sharded array.
    """
    if mesh is None:
        mesh = get_incontext_mesh()
    partition_spec = auto_partition_spec(
        x=x,
        mesh=mesh,
        names=names,
        min_sharding_size=min_sharding_size,
        reverse=reverse,
    )
    with mesh:
        return with_sharding_constraint(arr=x, sharding=partition_spec)


def auto_namedsharding(
    mesh: Mesh | None = None,
    names: list[str | tuple[str, ...]] | None = None,
    min_sharding_size: int | None = None,
    reverse: bool = False,
):
    """
    Returns a function that creates a NamedSharding for an array based on the provided parameters.

    Args:
            mesh: The device mesh to shard across. If None, uses the current thread's mesh.
            names: List of mesh axis names to use for sharding. If None, derives from mesh shape.
            min_sharding_size: Minimum size of array to shard. If None, uses the product of mesh device shape.
            reverse: If True, reverses the sorting order of array dimensions.

    Returns:
            A function that takes an array as input and returns a NamedSharding object.
    """

    def _named_sharding_fn(x: chex.Array):
        return NamedSharding(
            mesh,
            auto_partition_spec(
                x=x,
                mesh=mesh,
                names=names,
                min_sharding_size=min_sharding_size,
                reverse=reverse,
            ),
        )

    return _named_sharding_fn


def optimize_sharding_for_memory(
    pytree: tp.Any,
    mesh: Mesh | None = None,
    max_memory_per_device: int | None = None,
    names: list[str] | None = None,
) -> dict[str, PartitionSpec]:
    """Optimize sharding strategy to fit within per-device memory constraints.

    Generates partition specifications that ensure each array's per-device
    memory footprint stays within the specified limit. Arrays smaller than
    the limit remain unsharded for efficiency.

    Args:
        pytree: A PyTree of arrays to generate partition specs for.
        mesh: The JAX mesh to shard across. If None, uses the current
            context's mesh.
        max_memory_per_device: Maximum bytes per device. Arrays larger than
            this will be sharded to fit. If None, no memory constraint is
            applied (defaults to auto_partition_spec behavior).
        names: List of mesh axis names to consider for sharding.
            If None, uses all axis names from the mesh.

    Returns:
        A dictionary mapping paths to PartitionSpecs optimized for the
        memory constraint.

    Example:
        >>> # Optimize for 8GB per device
        >>> specs = optimize_sharding_for_memory(
        ...     params,
        ...     mesh=mesh,
        ...     max_memory_per_device=8 * 1024**3
        ... )
    """
    if mesh is None:
        mesh = get_incontext_mesh()
    if names is None:
        names = list(mesh.axis_names)

    def get_optimal_spec(name: str, array: chex.Array) -> PartitionSpec:
        array_size = np.prod(array.shape) * array.dtype.itemsize
        if max_memory_per_device is None or array_size < max_memory_per_device:
            return PartitionSpec()

        return auto_partition_spec(array, mesh=mesh, names=names, min_sharding_size=None)

    return jax.tree_util.tree_map_with_path(get_optimal_spec, pytree)


def validate_sharding_config(
    pytree: tp.Any,
    partition_specs: dict[str, PartitionSpec],
    mesh: Mesh | None = None,
) -> list[str]:
    """Validate sharding configuration and return any issues found.

    Checks that partition specifications are compatible with array shapes
    and mesh configuration. Identifies potential problems like:
    - Array dimensions not divisible by mesh axis sizes
    - Small arrays that might not benefit from sharding

    Args:
        pytree: A PyTree of arrays to validate.
        partition_specs: Dictionary mapping paths to PartitionSpecs.
        mesh: The JAX mesh to validate against. If None, uses the
            current context's mesh.

    Returns:
        A list of issue descriptions. Empty list means no issues found.

    Example:
        >>> issues = validate_sharding_config(params, specs, mesh)
        >>> if issues:
        ...     for issue in issues:
        ...         print(f"Warning: {issue}")
    """
    if mesh is None:
        mesh = get_incontext_mesh()
    issues = []

    def validate_leaf(path: str, array: np.ndarray, spec: PartitionSpec):
        for dim, axis_name in enumerate(spec):
            if axis_name is not None:
                if array.shape[dim] % mesh.shape[axis_name] != 0:
                    issues.append(
                        f"Array at {path} with shape {array.shape} not divisible "
                        f"by mesh axis {axis_name} size {mesh.shape[axis_name]}"
                    )

        if np.prod(array.shape) < 1024 and spec != PartitionSpec():
            issues.append(f"Small array at {path} might not benefit from sharding")

    jax.tree_util.tree_map_with_path(validate_leaf, pytree, partition_specs)
    return issues


def convert_sharding_strategy(
    array: chex.Array,
    old_partition_specs: dict[str, PartitionSpec],
    old_mesh: Mesh,
    new_mesh: Mesh,
    strategy: str = "preserve_balance",
) -> dict[str, PartitionSpec]:
    """Convert sharding strategy between different mesh configurations.

    When migrating models between different mesh topologies (e.g., from
    8 to 16 devices), this function adapts partition specifications to
    maintain similar parallelism characteristics.

    Args:
        array: Reference array used to determine valid new partition specs.
        old_partition_specs: Dictionary of current partition specifications.
        old_mesh: The original mesh configuration.
        new_mesh: The target mesh configuration to convert to.
        strategy: Conversion strategy. Currently supports:
            - "preserve_balance": Maintains similar parallelization factor
              by using the old spec's total split factor as the minimum
              sharding size for the new spec.

    Returns:
        A dictionary of new partition specifications adapted to new_mesh.

    Example:
        >>> # Convert from 8-device to 16-device mesh
        >>> old_mesh = create_mesh((2, 4), ('dp', 'tp'))
        >>> new_mesh = create_mesh((4, 4), ('dp', 'tp'))
        >>> new_specs = convert_sharding_strategy(
        ...     array, old_specs, old_mesh, new_mesh
        ... )
    """
    new_specs = {}

    for name, old_spec in old_partition_specs.items():
        if strategy == "preserve_balance":
            old_parallel_factor = 1
            for axis in old_spec:
                if axis is not None:
                    old_parallel_factor *= old_mesh.shape[axis]

            new_spec = auto_partition_spec(
                x=array,
                mesh=new_mesh,
                min_sharding_size=old_parallel_factor,
            )
            new_specs[name] = new_spec

    return new_specs
