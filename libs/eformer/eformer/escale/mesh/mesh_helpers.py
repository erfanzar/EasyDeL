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

import jax
import numpy as np
from jax.sharding import Mesh, PartitionSpec


class MeshPartitionHelper:
    """Helper class for analyzing and applying partition strategies to PyTrees.

    This class provides utilities for automatically determining optimal
    sharding strategies based on array shapes and mesh configuration. It
    supports various parallelism patterns including FSDP, data parallelism,
    tensor parallelism, and sequence parallelism.

    The helper analyzes array shapes and suggests appropriate sharding
    methods based on the available mesh axes and their sizes.

    Attributes:
        mesh: The JAX mesh to use for sharding.
        axis_sizes: Dictionary mapping axis names to their sizes.

    Example:
        >>> mesh = create_mesh(axis_dims=(2, 4, 1, 2, 1),
        ...                    axis_names=('dp', 'fsdp', 'ep', 'tp', 'sp'))
        >>> helper = MeshPartitionHelper(mesh)
        >>> # Analyze and auto-shard a model
        >>> sharded_params = helper.auto_shard_pytree(model_params)
    """

    def __init__(self, mesh: Mesh):
        """Initialize the MeshPartitionHelper.

        Args:
            mesh: The JAX mesh to use for partition analysis and sharding.
        """
        self.mesh = mesh
        self.axis_sizes = dict(zip(self.mesh.axis_names, self.mesh.devices.shape, strict=False))

    def analyze_pytree(self, pytree: tp.Any) -> dict[tuple[int, ...], PartitionSpec]:
        """Analyze a PyTree and suggest partitioning methods for each unique shape.

        Collects all unique array shapes in the PyTree and determines
        appropriate sharding methods for each based on the mesh configuration.

        Args:
            pytree: A PyTree of arrays to analyze.

        Returns:
            A dictionary mapping array shapes to lists of suggested sharding
            method tuples. Each method tuple contains axis names to use
            for sharding (e.g., ('fsdp', 'sp') for combined sharding).

        Example:
            >>> helper = MeshPartitionHelper(mesh)
            >>> shape_methods = helper.analyze_pytree(params)
            >>> # {(1024, 4096): [('fsdp', 'sp'), ('tp',)], ...}
        """
        shapes_dict = {}

        def collect_shapes(x):
            if hasattr(x, "shape"):
                shapes_dict[x.shape] = None
            return x

        jax.tree_util.tree_map(collect_shapes, pytree)

        for shape in shapes_dict.keys():
            shapes_dict[shape] = self._suggest_methods(shape)

        return shapes_dict

    def _suggest_methods(self, shape: tuple[int, ...]) -> list[tuple]:
        """Suggest sharding methods based on array shape and mesh configuration.

        Analyzes the array shape and available mesh axes to determine which
        sharding strategies could be applied. Returns tuples of axis names
        that can be used together for multi-dimensional sharding.

        The method considers:
        - Combined FSDP+SP sharding for 2D+ arrays with sufficient size
        - Data parallelism (dp) for batch dimensions
        - Sequence parallelism (sp) for sequence dimensions
        - Tensor parallelism (tp) for model dimensions
        - FSDP for weight sharding

        Args:
            shape: The shape of the array to find sharding methods for.

        Returns:
            A list of tuples, where each tuple contains axis names that can
            be used together for sharding. Tuples are ordered by priority,
            with preferred methods first.
        """
        methods = []
        dims = len(shape)

        if (
            dims > 1
            and "fsdp" in self.axis_sizes
            and "sp" in self.axis_sizes
            and shape[0] * shape[1] >= self.axis_sizes["fsdp"] * self.axis_sizes["sp"]
        ):
            methods.append(("fsdp", "sp"))

        if dims > 0 and "dp" in self.axis_sizes:
            methods.append(("dp",))

        if dims > 1 and "sp" in self.axis_sizes and ("fsdp", "sp") not in methods:
            methods.append(("sp",))

        if "tp" in self.axis_sizes:
            methods.append(("tp",))

        if "fsdp" in self.axis_sizes and all("fsdp" not in m for m in methods):
            methods.append(("fsdp",))

        return methods

    def create_partition_spec(
        self,
        array_shape: tuple[int, ...],
        methods: list[tuple],
        min_shard_size: int = 1024,
    ) -> PartitionSpec:
        """Create a PartitionSpec for an array using suggested sharding methods.

        Takes a list of sharding method tuples and determines how to apply
        them to the array dimensions. Handles both single-axis and
        multi-axis (combined) sharding methods.

        Args:
            array_shape: The shape of the array to create a spec for.
            methods: List of sharding method tuples from _suggest_methods.
            min_shard_size: Minimum number of elements per shard to consider
                sharding worthwhile. Prevents over-sharding small arrays.
                Defaults to 1024.

        Returns:
            A PartitionSpec assigning mesh axes to array dimensions.
            Returns an empty PartitionSpec for scalar arrays or if no
            suitable sharding is found.

        Example:
            >>> helper = MeshPartitionHelper(mesh)
            >>> methods = helper._suggest_methods((1024, 4096))
            >>> spec = helper.create_partition_spec((1024, 4096), methods)
            >>> # PartitionSpec('fsdp', 'tp')
        """
        if not array_shape:
            return PartitionSpec()

        dims = len(array_shape)
        spec = [None] * dims

        total_elements = np.prod(array_shape)
        total_devices = int(np.prod(self.mesh.devices.shape))
        min_elements_per_device = max(min_shard_size, total_elements // (total_devices * 2))

        for method_tuple in methods:
            combined_mesh_size = np.prod([self.axis_sizes[m] for m in method_tuple if m in self.axis_sizes])

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
                if (
                    dims >= 2
                    and (array_shape[0] * array_shape[1]) >= combined_mesh_size
                    and (array_shape[0] * array_shape[1]) % combined_mesh_size == 0
                ):
                    if (
                        array_shape[0] >= self.axis_sizes[method_tuple[0]]
                        and array_shape[1] >= self.axis_sizes[method_tuple[1]]
                        and spec[0] is None
                        and spec[1] is None
                    ):
                        spec[0], spec[1] = method_tuple
                        break
                elif (
                    dims >= 2
                    and array_shape[0] >= combined_mesh_size
                    and array_shape[0] % combined_mesh_size == 0
                    and spec[0] is None
                ):
                    spec[0] = method_tuple
                    break

        if all(s is None for s in spec):
            for method_tuple in methods:
                if len(method_tuple) == 1:
                    method = method_tuple[0]
                    for dim, dim_size in enumerate(array_shape):
                        if dim_size >= min_shard_size and dim_size % self.axis_sizes[method] == 0 and spec[dim] is None:
                            spec[dim] = method
                            break
                elif len(method_tuple) == 2:
                    if method_tuple == ("fsdp", "sp"):
                        if spec[0] is None and spec[1] is None:
                            spec[0], spec[1] = method_tuple
                            break
                        elif spec[0] is None:
                            spec[0] = method_tuple
                            break

        return PartitionSpec(*spec)

    def shard_array(self, array, partition_spec):
        """Shard an array according to a partition specification.

        Places the array on devices according to the given partition spec,
        creating a distributed array with NamedSharding.

        Args:
            array: The array to shard.
            partition_spec: The PartitionSpec defining how to distribute
                the array across devices.

        Returns:
            A JAX array distributed across devices according to the
            partition specification.
        """
        return jax.device_put(array, jax.sharding.NamedSharding(self.mesh, partition_spec))

    def auto_shard_pytree(self, pytree: tp.Any, min_shard_size: int = 1024):
        """Automatically shard an entire PyTree based on shape analysis.

        Analyzes all arrays in the PyTree, determines optimal sharding
        strategies, and applies them. This is a convenience method that
        combines analyze_pytree, create_partition_spec, and shard_array.

        Args:
            pytree: A PyTree of arrays to shard.
            min_shard_size: Minimum number of elements to consider sharding.
                Arrays smaller than this remain unsharded. Defaults to 1024.

        Returns:
            A PyTree with the same structure where each array has been
            sharded according to its optimal partition specification.

        Example:
            >>> helper = MeshPartitionHelper(mesh)
            >>> sharded_params = helper.auto_shard_pytree(model_params)
        """
        shape_specs = self.analyze_pytree(pytree)

        def shard_leaf(x):
            if hasattr(x, "shape"):
                methods = shape_specs[x.shape]
                spec = self.create_partition_spec(x.shape, methods, min_shard_size)
                return self.shard_array(x, spec)
            return x

        return jax.tree_util.tree_map(shard_leaf, pytree)
