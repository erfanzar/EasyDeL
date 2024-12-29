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
import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
import typing as tp

import jax
import numpy as np
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec

jax.config.update("jax_platform_name", "cpu")


class MeshPartitionHelper:
	def __init__(self, mesh: Mesh):
		self.mesh = mesh
		self.axis_sizes = dict(zip(self.mesh.axis_names, self.mesh.devices.shape))

	def analyze_pytree(
		self, pytree: tp.Any
	) -> tp.Dict[tp.Tuple[int, ...], PartitionSpec]:
		"""Analyze pytree and suggest partitioning for each unique array shape."""
		shapes_dict = {}

		def collect_shapes(x):
			if hasattr(x, "shape"):
				shapes_dict[x.shape] = None
			return x

		jax.tree_util.tree_map(collect_shapes, pytree)

		# Analyze each unique shape
		for shape in shapes_dict.keys():
			shapes_dict[shape] = self._suggest_methods(shape)

		return shapes_dict

	def _suggest_methods(self, shape: tp.Tuple[int, ...]) -> tp.List[tp.Tuple]:
		"""Suggest sharding methods based on array shape and mesh.
		Now returns tuples of methods for combined sharding.
		"""
		methods = []
		dims = len(shape)

		# Prioritize combined ('fsdp', 'sp') if both are available and suitable
		if (
			dims > 1
			and "fsdp" in self.axis_sizes
			and "sp" in self.axis_sizes
			and shape[0] * shape[1] >= self.axis_sizes["fsdp"] * self.axis_sizes["sp"]
		):
			methods.append(("fsdp", "sp"))

		# For batch dimension (usually first dim)
		if dims > 0 and "dp" in self.axis_sizes:
			methods.append(("dp",))

		# For sequence dimension (usually second dim)
		# Only suggest 'sp' alone if ('fsdp', 'sp') was not already added
		if dims > 1 and "sp" in self.axis_sizes and ("fsdp", "sp") not in methods:
			methods.append(("sp",))

		# For hidden/feature dimensions
		if "tp" in self.axis_sizes:
			methods.append(("tp",))

		# Add FSDP for any remaining large dimensions
		# Only suggest 'fsdp' alone if ('fsdp', 'sp') was not already added
		if "fsdp" in self.axis_sizes and all(
			"fsdp" not in m for m in methods
		):  # Avoid adding fsdp alone if combined is present
			methods.append(("fsdp",))

		return methods

	def create_partition_spec(
		self,
		array_shape: tp.Tuple[int, ...],
		methods: tp.List[tp.Tuple],
		min_shard_size: int = 1024,
	) -> PartitionSpec:
		if not array_shape:
			return PartitionSpec()

		dims = len(array_shape)
		spec = [None] * dims

		# Calculate total elements and minimum elements per device
		total_elements = np.prod(array_shape)
		total_devices = int(np.prod(self.mesh.devices.shape))
		min_elements_per_device = max(min_shard_size, total_elements // (total_devices * 2))

		# First pass: iterate through suggested method combinations
		for method_tuple in methods:
			# Calculate combined mesh size for the method tuple
			combined_mesh_size = np.prod(
				[self.axis_sizes[m] for m in method_tuple if m in self.axis_sizes]
			)

			# Try to shard based on the method tuple
			if len(method_tuple) == 1:
				method = method_tuple[0]
				for dim, dim_size in enumerate(array_shape):
					if (
						dim_size >= min_elements_per_device
						and dim_size % self.axis_sizes[method] == 0
						and spec[dim] is None
					):
						spec[dim] = method
						break
			elif len(method_tuple) == 2:
				# For combined methods like ('fsdp', 'sp')
				if (
					dims >= 2
					and (array_shape[0] * array_shape[1]) >= combined_mesh_size
					and (array_shape[0] * array_shape[1]) % combined_mesh_size == 0
				):
					# Case 1: Both dimensions can be sharded with ('fsdp', 'sp')
					if (
						array_shape[0] >= self.axis_sizes[method_tuple[0]]
						and array_shape[1] >= self.axis_sizes[method_tuple[1]]
						and spec[0] is None
						and spec[1] is None
					):
						spec[0], spec[1] = method_tuple
						break
				# Case 2: Only the first dimension is suitable for ('fsdp', 'sp')
				elif (
					dims >= 2
					and array_shape[0] >= combined_mesh_size
					and array_shape[0] % combined_mesh_size == 0
					and spec[0] is None
				):
					spec[0] = method_tuple
					break

		# Second pass: try sharding with individual methods
		#             or apply ('fsdp', 'sp') if suggested but not applied in the first pass
		print(spec)
		if all(s is None for s in spec):
			for method_tuple in methods:
				if len(method_tuple) == 1:
					method = method_tuple[0]
					for dim, dim_size in enumerate(array_shape):
						if (
							dim_size >= min_shard_size
							and dim_size % self.axis_sizes[method] == 0
							and spec[dim] is None
						):
							spec[dim] = method
							break
				elif len(method_tuple) == 2:
					# Directly apply ('fsdp', 'sp') if suggested but not used in the first pass
					if method_tuple == ("fsdp", "sp"):
						if spec[0] is None and spec[1] is None:
							spec[0], spec[1] = method_tuple
							break
						elif spec[0] is None:
							spec[0] = method_tuple
							break

		return PartitionSpec(*spec)

	def shard_array(self, array, partition_spec):
		return jax.device_put(array, jax.sharding.NamedSharding(self.mesh, partition_spec))

	def auto_shard_pytree(self, pytree: tp.Any, min_shard_size: int = 1024):
		"""Automatically shard entire pytree based on analysis."""
		shape_specs = self.analyze_pytree(pytree)

		def shard_leaf(x):
			if hasattr(x, "shape"):
				methods = shape_specs[x.shape]
				spec = self.create_partition_spec(x.shape, methods, min_shard_size)
				return self.shard_array(x, spec)
			return x

		return jax.tree_util.tree_map(shard_leaf, pytree)


if __name__ == "__main__":
	mesh = Mesh(mesh_utils.create_device_mesh((2, 2, 1, 2)), ("dp", "fsdp", "sp", "tp"))

	helper = MeshPartitionHelper(mesh)

	# Auto-shard entire pytree
	# sharded_pytree = helper.auto_shard_pytree(model_params)

	# Or get specific partition spec
	array_shape = (16, 512, 1024)
	methods = helper._suggest_methods(array_shape)
	spec = helper.create_partition_spec(array_shape, methods)
	print(spec, methods)
	# Output: PartitionSpec(('fsdp', 'sp'), None, None) [('fsdp', 'sp'), ('dp',), ('tp',)]

	array_shape = (2, 16, 512, 1024)
	methods = helper._suggest_methods(array_shape)
	spec = helper.create_partition_spec(array_shape, methods)
	print(spec, methods)
	# Output: PartitionSpec(('fsdp', 'sp'), None, None, None) [('fsdp', 'sp'), ('dp',), ('tp',)]
