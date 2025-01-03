# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
from jax.interpreters import pxla
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from .constraints import with_sharding_constraint


def auto_partition_spec(
	x: chex.Array,
	mesh: tp.Optional[Mesh] = None,
	names: tp.Optional[tp.List[tp.Union[str, tp.Tuple[str, ...]]]] = None,
	min_sharding_size: tp.Optional[int] = None,
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
	# Validate input array
	if not isinstance(x, (chex.Array, np.ndarray)):
		raise TypeError(f"Expected array input, got {type(x)}")

	# Get or validate mesh
	if mesh is None:
		mesh = pxla.thread_resources.env.physical_mesh
		if mesh.empty:
			raise ValueError(
				"No mesh available. Provide a mesh or use within a mesh context."
			)

	# Calculate minimum sharding size
	min_sharding_size = min_sharding_size or np.prod(mesh.devices.shape)

	# Early return for small arrays
	array_size = np.prod(x.shape)
	if array_size < min_sharding_size:
		return PartitionSpec()

	# Prepare mesh names
	if not names:
		# Sort mesh axes by size in descending order
		names = [mesh.axis_names[i] for i in np.argsort([-s for s in mesh.devices.shape])]

	# Create mesh size lookup for efficient access
	mesh_sizes = {
		name: (
			np.prod([mesh.shape[n] for n in name])
			if isinstance(name, tp.Tuple)
			else mesh.shape[name]
		)
		for name in names
	}

	# Sort dimensions by size
	dim_indices = np.argsort([-dim if not reverse else dim for dim in x.shape])

	# Initialize partition spec
	partition_spec = [None] * len(x.shape)
	remaining_names = set(names)

	# Assign sharding
	for dim_idx in dim_indices:
		dim_size = x.shape[dim_idx]

		# Find best matching mesh axis
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
	mesh: tp.Optional[Mesh] = None,
	names: tp.Optional[tp.List[tp.Union[str, tp.Tuple[str, ...]]]] = None,
	min_sharding_size: tp.Optional[int] = None,
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
	# Input validation
	if not isinstance(x, (np.ndarray, chex.Array)):
		raise TypeError(f"Expected array input, got {type(x)}")

	# Get mesh
	if mesh is None:
		mesh = pxla.thread_resources.env.physical_mesh
		if mesh.empty:
			raise ValueError(
				"No mesh available. Provide a mesh or use within a mesh context manager."
			)

	# Calculate minimum sharding size
	min_sharding_size = min_sharding_size or int(np.prod(mesh.devices.shape))

	# Early return for small arrays
	array_size = np.prod(x.shape)
	if array_size < min_sharding_size:
		return PartitionSpec()

	# Prepare mesh names
	if not names:
		# Sort mesh axes by size in descending order
		names = [mesh.axis_names[i] for i in np.argsort([-s for s in mesh.devices.shape])]

	# Pre-calculate mesh sizes for performance
	mesh_sizes = {
		name: (
			np.prod([mesh.shape[n] for n in name])
			if isinstance(name, tp.Tuple)
			else mesh.shape[name]
		)
		for name in names
	}

	# Initialize partition spec
	partition_spec = [None] * len(x.shape)

	# Calculate dimension ordering
	dim_order = np.argsort([-dim for dim in x.shape] if not reverse else x.shape)

	# Assign sharding
	remaining_names = names.copy()
	for dim_idx in dim_order:
		dim_size = x.shape[dim_idx]

		# Find the best matching mesh axis
		for name in remaining_names:
			mesh_size = mesh_sizes[name]

			if dim_size % mesh_size == 0:
				partition_spec[dim_idx] = name
				remaining_names.remove(name)
				break

	return PartitionSpec(*partition_spec)


def auto_shard_array(
	x: chex.Array,
	mesh: tp.Optional[Mesh] = None,
	names: tp.Optional[tp.List[tp.Union[str, tp.Tuple[str, ...]]]] = None,
	min_sharding_size: tp.Optional[int] = None,
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
		mesh = pxla.thread_resources.env.physical_mesh
		if mesh.empty:
			raise ValueError(
				"`auto_shard_array` needs to be used with a mesh. Pass a mesh as an argument "
				"or use this function under a mesh context manager."
			)
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
	mesh: tp.Optional[Mesh] = None,
	names: tp.Optional[tp.List[tp.Union[str, tp.Tuple[str, ...]]]] = None,
	min_sharding_size: tp.Optional[int] = None,
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
	mesh: tp.Optional[Mesh] = None,
	max_memory_per_device: tp.Optional[int] = None,
	names: tp.Optional[tp.List[str]] = None,
) -> tp.Dict[str, PartitionSpec]:
	"""
	Optimizes sharding strategy to fit within memory constraints.
	"""
	if mesh is None:
		mesh = pxla.thread_resources.env.physical_mesh

	if names is None:
		names = list(mesh.axis_names)

	def get_optimal_spec(name: str, array: chex.Array) -> PartitionSpec:
		array_size = np.prod(array.shape) * array.dtype.itemsize
		if array_size < max_memory_per_device:
			return PartitionSpec()

		return auto_partition_spec(array, mesh=mesh, names=names, min_sharding_size=None)

	return jax.tree_util.tree_map_with_path(get_optimal_spec, pytree)


def validate_sharding_config(
	pytree: tp.Any,
	partition_specs: tp.Dict[str, PartitionSpec],
	mesh: tp.Optional[Mesh] = None,
) -> tp.List[str]:
	"""
	Validates sharding configuration and returns list of warnings/errors.
	"""
	if mesh is None:
		mesh = pxla.thread_resources.env.physical_mesh

	issues = []

	def validate_leaf(path: str, array: np.ndarray, spec: PartitionSpec):
		# Check dimension compatibility
		for dim, axis_name in enumerate(spec):
			if axis_name is not None:
				if array.shape[dim] % mesh.shape[axis_name] != 0:
					issues.append(
						f"Array at {path} with shape {array.shape} not divisible "
						f"by mesh axis {axis_name} size {mesh.shape[axis_name]}"
					)

		# Check for potential performance issues
		if np.prod(array.shape) < 1024 and spec != PartitionSpec():
			issues.append(f"Small array at {path} might not benefit from sharding")

	jax.tree_util.tree_map_with_path(validate_leaf, pytree, partition_specs)
	return issues


def convert_sharding_strategy(
	array: chex.Array,
	old_partition_specs: tp.Dict[str, PartitionSpec],
	old_mesh: Mesh,
	new_mesh: Mesh,
	strategy: str = "preserve_balance",
) -> tp.Dict[str, PartitionSpec]:
	"""
	Converts sharding strategy between different mesh configurations.
	"""
	new_specs = {}

	for name, old_spec in old_partition_specs.items():
		if strategy == "preserve_balance":
			# Try to maintain similar parallelization degree
			old_parallel_factor = 1
			for axis in old_spec:
				if axis is not None:
					old_parallel_factor *= old_mesh.shape[axis]

			new_spec = auto_partition_spec(
				x=array,  # You'd need to have access to the array here
				mesh=new_mesh,
				min_sharding_size=old_parallel_factor,
			)
			new_specs[name] = new_spec

	return new_specs
