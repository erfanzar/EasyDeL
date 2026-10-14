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


import abc
import threading
import time
import typing as tp
import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, PartitionSpec

from ..partition import get_incontext_mesh

_sync_counter = 0
_sync_counter_lock = threading.Lock()


class ShardingRule(abc.ABC):
    """Abstract base class for defining sharding rules.

    Sharding rules define how arrays in a PyTree should be partitioned
    across devices in a mesh. Subclasses must implement the `apply` method
    to provide specific sharding logic.

    This class follows the Strategy pattern, allowing different sharding
    algorithms to be swapped at runtime.

    Example:
        >>> class CustomRule(ShardingRule):
        ...     def apply(self, pytree):
        ...         return jax.tree_util.tree_map(
        ...             lambda x: PartitionSpec('data'),
        ...             pytree
        ...         )
    """

    @abc.abstractmethod
    def apply(self, pytree: tp.Any) -> tp.Any:
        """Apply the sharding rule to a PyTree of arrays.

        Args:
            pytree: A PyTree (nested structure) of arrays to generate
                partition specifications for.

        Returns:
            A PyTree with the same structure as the input, where each leaf
            is replaced with its corresponding PartitionSpec.
        """


class AutoShardingRule(ShardingRule):
    """Automatically determines sharding based on array shapes and mesh configuration.

    This rule analyzes array shapes and assigns mesh axes to array dimensions
    to achieve optimal parallelism. It prioritizes larger dimensions and ensures
    that array dimensions are divisible by the corresponding mesh axis sizes.

    The algorithm works as follows:
    1. Skip arrays smaller than `min_shard_size` (they remain unsharded)
    2. Sort array dimensions by size (largest first, unless `reverse=True`)
    3. For each dimension, find the first available mesh axis that divides evenly
    4. Assign that axis to the dimension and remove it from available axes

    Attributes:
        mesh: The JAX mesh to shard across.
        axis_names: List of mesh axis names to consider for sharding.
        min_shard_size: Minimum array size (in elements) to apply sharding.
        reverse: If True, processes smaller dimensions first.

    Example:
        >>> mesh = create_mesh(axis_dims=(2, 4), axis_names=('dp', 'tp'))
        >>> rule = AutoShardingRule(mesh=mesh, axis_names=['dp', 'tp'])
        >>> # Apply to model parameters
        >>> specs = rule.apply(model_params)
    """

    def __init__(
        self,
        mesh: Mesh | None = None,
        axis_names: list[str] | None = None,
        min_shard_size: int | None = None,
        reverse: bool = False,
    ):
        """Initialize the AutoShardingRule.

        Args:
            mesh: The JAX mesh to use for sharding. If None, uses the
                mesh from the current context.
            axis_names: List of mesh axis names to use for sharding.
                If None, uses all axis names from the mesh.
            min_shard_size: Minimum number of elements in an array to
                apply sharding. Arrays smaller than this remain unsharded.
                If None, defaults to the total number of devices in the mesh.
            reverse: If True, processes smaller array dimensions first
                instead of larger ones. Defaults to False.
        """
        self.mesh = mesh or get_incontext_mesh()
        self.axis_names = axis_names or list(self.mesh.axis_names)
        self.min_shard_size = min_shard_size or np.prod(self.mesh.shape)
        self.reverse = reverse

    def _get_optimal_partition(self, array_shape: tuple[int, ...]) -> PartitionSpec:
        """Determine the optimal partition specification for an array shape.

        Args:
            array_shape: The shape of the array to partition.

        Returns:
            A PartitionSpec assigning mesh axes to array dimensions for
            optimal sharding, or an empty PartitionSpec if the array is
            too small to benefit from sharding.
        """
        if np.prod(array_shape) < self.min_shard_size:
            return PartitionSpec()

        partition_spec = [None] * len(array_shape)
        remaining_axes = set(self.axis_names)

        dim_order = np.argsort([-d if not self.reverse else d for d in array_shape])

        for dim_idx in dim_order:
            dim_size = array_shape[dim_idx]

            best_axis = None
            for axis in remaining_axes:
                mesh_size = self.mesh.shape[axis]
                if dim_size % mesh_size == 0:
                    best_axis = axis
                    break

            if best_axis:
                partition_spec[dim_idx] = best_axis
                remaining_axes.remove(best_axis)

            if not remaining_axes:
                break

        return PartitionSpec(*partition_spec)

    def apply(self, pytree: tp.Any) -> tp.Any:
        """Apply auto-sharding to all arrays in a PyTree.

        Args:
            pytree: A PyTree of arrays to generate partition specs for.

        Returns:
            A PyTree with the same structure containing PartitionSpecs
            optimized for each array's shape.
        """
        return jax.tree_util.tree_map(
            lambda x: self._get_optimal_partition(x.shape),
            pytree,
        )


class CompositeShardingRule(ShardingRule):
    """Combines multiple sharding rules with priority ordering.

    This rule applies multiple sharding rules in sequence and selects the
    first non-empty PartitionSpec for each array. This is useful for
    implementing fallback strategies where a specific rule might not
    produce valid sharding for all arrays.

    The priority order is determined by the order of rules passed to
    the constructor. Earlier rules have higher priority.

    Attributes:
        rules: Tuple of ShardingRule instances to combine.

    Example:
        >>> # Try shape-based first, fall back to auto sharding
        >>> shape_rule = ShapeBasedShardingRule({(None, 1024): PartitionSpec('tp')})
        >>> auto_rule = AutoShardingRule(mesh=mesh)
        >>> combined = CompositeShardingRule(shape_rule, auto_rule)
        >>> specs = combined.apply(model_params)
    """

    def __init__(self, *rules: ShardingRule):
        """Initialize the CompositeShardingRule.

        Args:
            *rules: Variable number of ShardingRule instances to combine.
                Rules are applied in order, with earlier rules having
                higher priority.
        """
        self.rules = rules

    def apply(self, pytree: tp.Any) -> tp.Any:
        """Apply combined sharding rules to a PyTree.

        Applies all rules and for each leaf selects the first non-empty
        PartitionSpec. If all rules return empty specs, an empty spec
        is used.

        Args:
            pytree: A PyTree of arrays to generate partition specs for.

        Returns:
            A PyTree with the same structure containing the highest-priority
            non-empty PartitionSpec for each array.
        """

        def combine_specs(*specs):
            for spec in specs:
                if spec != PartitionSpec():
                    return spec
            return PartitionSpec()

        results = [rule.apply(pytree) for rule in self.rules]
        return jax.tree_util.tree_map(combine_specs, *results)


class MemoryConstrainedShardingRule(ShardingRule):
    """Creates sharding based on per-device memory constraints.

    This rule ensures that each device's memory usage stays within a
    specified limit by sharding large arrays across the mesh. It
    prioritizes sharding the largest dimensions first and uses the
    largest mesh axes for maximum memory reduction.

    The algorithm:
    1. If array fits in memory, return empty PartitionSpec (no sharding)
    2. Sort mesh axes by size (largest first for maximum memory reduction)
    3. Sort array dimensions by size (largest first)
    4. Iteratively assign mesh axes to dimensions until memory fits

    Attributes:
        max_memory_per_device: Maximum bytes allowed per device.
        mesh: The JAX mesh to shard across.
        axis_names: List of mesh axis names to consider for sharding.

    Example:
        >>> # Allow max 1GB per device
        >>> rule = MemoryConstrainedShardingRule(
        ...     max_memory_per_device=1024**3,
        ...     mesh=mesh
        ... )
        >>> specs = rule.apply(large_model_params)
    """

    def __init__(
        self,
        max_memory_per_device: int,
        mesh: Mesh | None = None,
        axis_names: list[str] | None = None,
    ):
        """Initialize the MemoryConstrainedShardingRule.

        Args:
            max_memory_per_device: Maximum memory in bytes that each device
                should hold after sharding.
            mesh: The JAX mesh to use for sharding. If None, uses the
                mesh from the current context.
            axis_names: List of mesh axis names to use for sharding.
                If None, uses all axis names from the mesh.
        """
        self.max_memory_per_device = max_memory_per_device
        self.mesh = mesh or get_incontext_mesh()
        self.axis_names = axis_names or list(self.mesh.axis_names)

    def _calculate_partition_spec(self, array: jnp.ndarray) -> PartitionSpec:
        """Calculate partition spec to fit array within memory constraints.

        Args:
            array: The array to calculate sharding for.

        Returns:
            A PartitionSpec that shards the array to fit within the
            memory constraint, or an empty PartitionSpec if the array
            already fits.
        """
        array_size = np.prod(array.shape) * array.dtype.itemsize
        if array_size <= self.max_memory_per_device:
            return PartitionSpec()

        partition_spec = [None] * len(array.shape)
        remaining_size = array_size

        sorted_axes = sorted(self.axis_names, key=lambda x: self.mesh.shape[x], reverse=True)

        dim_order = np.argsort([-d for d in array.shape])

        for dim_idx in dim_order:
            if remaining_size <= self.max_memory_per_device:
                break

            dim_size = array.shape[dim_idx]

            for axis in sorted_axes:
                mesh_size = self.mesh.shape[axis]
                if dim_size % mesh_size == 0:
                    partition_spec[dim_idx] = axis
                    remaining_size //= mesh_size
                    sorted_axes.remove(axis)
                    break

        return PartitionSpec(*partition_spec)

    def apply(self, pytree: tp.Any) -> tp.Any:
        """Apply memory-constrained sharding to all arrays in a PyTree.

        Args:
            pytree: A PyTree of arrays to generate partition specs for.

        Returns:
            A PyTree with the same structure containing PartitionSpecs
            that ensure each array fits within the memory constraint.
        """
        return jax.tree_util.tree_map(self._calculate_partition_spec, pytree)


class ShapeBasedShardingRule(ShardingRule):
    """Creates sharding based on array shape patterns.

    This rule allows defining specific sharding strategies for arrays
    matching certain shape patterns. Patterns can include wildcards (None)
    to match any dimension size.

    This is useful when you know certain shapes should always be sharded
    in a particular way, such as embedding tables or attention weights.

    Attributes:
        shape_patterns: Dictionary mapping shape patterns to PartitionSpecs.

    Example:
        >>> # Shard arrays with shape (vocab_size, embed_dim) along first axis
        >>> patterns = {
        ...     (None, 1024): PartitionSpec('tp', None),  # Embedding tables
        ...     (1024, 1024): PartitionSpec('tp', None),  # Square weight matrices
        ... }
        >>> rule = ShapeBasedShardingRule(patterns)
        >>> specs = rule.apply(model_params)
    """

    def __init__(self, shape_patterns: dict[tuple[int | None, ...], PartitionSpec]):
        """Initialize the ShapeBasedShardingRule.

        Args:
            shape_patterns: Dictionary mapping shape patterns to PartitionSpecs.
                Patterns are tuples where each element is either:
                - An integer: matches that exact dimension size
                - None: matches any dimension size (wildcard)
        """
        self.shape_patterns = shape_patterns

    def _match_shape_pattern(self, array_shape: tuple[int, ...], pattern: tuple[int | None, ...]) -> bool:
        """Check if an array shape matches a pattern.

        Args:
            array_shape: The shape of the array to check.
            pattern: The pattern to match against, with None as wildcards.

        Returns:
            True if the array shape matches the pattern, False otherwise.
        """
        if len(array_shape) != len(pattern):
            return False
        return all(p is None or p == s for p, s in zip(pattern, array_shape, strict=False))

    def _get_partition_spec(self, array: jnp.ndarray) -> PartitionSpec:
        """Get the partition spec for an array based on shape patterns.

        Searches through defined patterns in order and returns the spec
        for the first matching pattern.

        Args:
            array: The array to find a partition spec for.

        Returns:
            The PartitionSpec for the first matching pattern, or an empty
            PartitionSpec if no patterns match.
        """
        for pattern, spec in self.shape_patterns.items():
            if self._match_shape_pattern(array.shape, pattern):
                return spec
        return PartitionSpec()

    def apply(self, pytree: tp.Any) -> tp.Any:
        """Apply shape-based sharding to all arrays in a PyTree.

        Args:
            pytree: A PyTree of arrays to generate partition specs for.

        Returns:
            A PyTree with the same structure containing PartitionSpecs
            based on matching shape patterns.
        """
        return jax.tree_util.tree_map(self._get_partition_spec, pytree)


class ShardingAnalyzer:
    """
    Analyzes and validates sharding strategies.

    Attributes:
            mesh (Mesh): The mesh configuration for sharding. If not provided, it defaults to the physical mesh from the
                thread resources.

    Methods:
            validate_partition_specs(pytree: tp.Any, partition_specs: tp.Any) -> tp.List[str]:
                    Validates the compatibility of partition specifications with the shapes of arrays in the pytree.
                    Args:
                            pytree (tp.Any): A pytree of arrays to be validated.
                            partition_specs (tp.Any): A pytree of partition specifications corresponding to the arrays.
                    Returns:
                            tp.List[str]: A list of issues found during validation. If empty, no issues were found.

            estimate_memory_usage(pytree: tp.Any, partition_specs: tp.Any) -> tp.Dict[str, int]:
                    Estimates the memory usage per device after applying the sharding strategy.
                    Args:
                            pytree (tp.Any): A pytree of arrays for which memory usage is to be estimated.
                            partition_specs (tp.Any): A pytree of partition specifications corresponding to the arrays.
                    Returns:
                            tp.Dict[str, int]: A dictionary containing the total memory size and the size per device.
    """

    def __init__(self, mesh: Mesh | None = None):
        """Initialize the ShardingAnalyzer.

        Args:
            mesh: The JAX mesh to analyze sharding against. If None,
                uses the mesh from the current context.
        """
        self.mesh = mesh or get_incontext_mesh()

    def validate_partition_specs(self, pytree: tp.Any, partition_specs: tp.Any) -> list[str]:
        """Validate compatibility of partition specs with array shapes.

        Checks that each array dimension is divisible by the corresponding
        mesh axis size specified in the partition spec.

        Args:
            pytree: A PyTree of arrays to validate.
            partition_specs: A PyTree of PartitionSpecs with the same structure
                as pytree.

        Returns:
            A list of validation issue messages. Empty list means no issues.

        Example:
            >>> analyzer = ShardingAnalyzer(mesh)
            >>> issues = analyzer.validate_partition_specs(params, specs)
            >>> if issues:
            ...     print("Validation failed:", issues)
        """
        issues = []

        def validate_leaf(array: jnp.ndarray, spec: PartitionSpec):
            if spec == PartitionSpec():
                return

            for dim, axis_name in enumerate(spec):
                if axis_name is not None:
                    if array.shape[dim] % self.mesh.shape[axis_name] != 0:
                        issues.append(
                            f"Array shape {array.shape} not divisible by mesh "
                            f"axis {axis_name} size {self.mesh.shape[axis_name]}"
                        )

        jax.tree_util.tree_map(validate_leaf, pytree, partition_specs)
        return issues

    def estimate_memory_usage(self, pytree: tp.Any, partition_specs: tp.Any) -> dict[str, int]:
        """Estimate memory usage per device after applying sharding.

        Calculates the total memory footprint and estimates how much
        memory each device will need after the sharding strategy is applied.

        Args:
            pytree: A PyTree of arrays to estimate memory for.
            partition_specs: A PyTree of PartitionSpecs with the same structure
                as pytree.

        Returns:
            A dictionary containing:
                - "total_size": Total memory in bytes before sharding
                - "size_per_device": Estimated memory per device after sharding

        Example:
            >>> analyzer = ShardingAnalyzer(mesh)
            >>> usage = analyzer.estimate_memory_usage(params, specs)
            >>> print(f"Memory per device: {usage['size_per_device'] / 1e9:.2f} GB")
        """

        def calculate_size(array: jnp.ndarray, spec: PartitionSpec):
            size = np.prod(array.shape) * array.dtype.itemsize

            if spec != PartitionSpec():
                for axis_name in spec:
                    if axis_name is not None:
                        size //= self.mesh.shape[axis_name]

            return size

        total_size = jax.tree_util.tree_reduce(
            lambda x, y: x + y,
            jax.tree_util.tree_map(calculate_size, pytree, partition_specs),
        )

        return {
            "total_size": total_size,
            "size_per_device": total_size // np.prod(self.mesh.shape),
        }


def create_monitored_function(
    fn: tp.Callable,
    partition_specs: tp.Any,
    analyzer: ShardingAnalyzer,
) -> tp.Callable:
    """Create a monitored version of a function with sharding analysis.

    Wraps a function to automatically validate sharding, measure execution
    time, and track memory usage. Useful for debugging and optimizing
    distributed training loops.

    Args:
        fn: The function to wrap with monitoring.
        partition_specs: The partition specifications to validate inputs against.
        analyzer: A ShardingAnalyzer instance for validation and memory estimation.

    Returns:
        A wrapped function that returns a tuple of (result, metrics) where
        metrics contains execution_time, memory_usage, and validation_issues.

    Example:
        >>> analyzer = ShardingAnalyzer(mesh)
        >>> monitored_train_step = create_monitored_function(
        ...     train_step, partition_specs, analyzer
        ... )
        >>> result, metrics = monitored_train_step(params, batch)
        >>> print(f"Execution time: {metrics['execution_time']:.2f}s")

    Warning:
        If validation issues are detected, a warning is raised but execution
        continues. This allows debugging without interrupting training.
    """

    def monitored_fn(*args, **kwargs):
        start_time = time.time()

        validation_issues = analyzer.validate_partition_specs(args[0], partition_specs)
        if validation_issues:
            warnings.warn(
                f"Sharding validation issues: {validation_issues}",
                stacklevel=1,
            )

        result = fn(*args, **kwargs)
        execution_time = time.time() - start_time

        metrics = {
            "execution_time": execution_time,
            "memory_usage": analyzer.estimate_memory_usage(args[0], partition_specs),
            "validation_issues": validation_issues,
        }

        return result, metrics

    return monitored_fn


def barrier_sync(timeout: float = 200):
    """Synchronize all JAX processes at a barrier point.

    Blocks execution until all processes in the distributed JAX runtime reach
    this barrier. This is essential for ensuring consistency across distributed
    training, especially before/after collective operations or checkpointing.

    The function uses a global counter to create unique barrier names, allowing
    multiple barriers to be used sequentially without conflicts.

    Args:
        timeout: Maximum time to wait for all processes to reach the barrier,
            in seconds. Defaults to 200 seconds (3.33 minutes). If the timeout
            is exceeded, a RuntimeError will be raised by the underlying JAX
            distributed client.

    Returns:
        None

    Raises:
        RuntimeError: If the JAX distributed client is not initialized. This
            typically means JAX was not started in distributed mode or the
            distributed runtime failed to initialize.

    Note:
        - This function is a no-op when running with a single process
          (jax.process_count() == 1), allowing code to work seamlessly
          in both single and multi-process environments.
        - Each call increments a global counter to ensure unique barrier names,
          preventing conflicts when multiple barriers are used in sequence.
        - The timeout is converted to milliseconds for the underlying JAX API.

    Example:
        >>>
        >>> model = train_step(model, batch)
        >>> barrier_sync()
        >>> if jax.process_index() == 0:
        ...     save_checkpoint(model)
        >>> barrier_sync()

        >>>
        >>> barrier_sync(timeout=600)

    Warning:
        Ensure all processes call barrier_sync() the same number of times and
        in the same order, or deadlocks may occur. Conditional barriers based
        on process rank should be avoided.
    """
    global _sync_counter
    if jax.process_count() == 1:
        return
    import jax._src.distributed as distributed

    client = distributed.global_state.client

    if client is None:
        raise RuntimeError("barrier_sync requires jax distributed client to be initialized")

    with _sync_counter_lock:
        _sync_counter += 1
        counter = _sync_counter
    client.wait_at_barrier(f"easy_barrier_sync_{counter}", timeout_in_ms=int(timeout * 1000.0))
