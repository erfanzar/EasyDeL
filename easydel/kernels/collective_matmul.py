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


# Implementation by @erfanzar,
# with a few bug fixes and adjustments.

import enum
import typing as tp
from functools import partial

import jax
import numpy as np
from eformer import escale as es
from eformer.escale.partition.constraints import AxisType
from jax import Array, lax
from jax import numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as Ps


def calculate_mesh_dimension_size(sharding_axis_names: AxisType) -> int:
    """
    Calculates the total number of devices along the specified mesh dimension(s).

    This function computes the product of the number of devices along each specified
    mesh dimension, providing the total size of the submesh defined by these axes.

    Args:
        sharding_axis_names: A single mesh dimension name (str) or a sequence (list/tuple)
                   of mesh dimension names. For sequences, the order doesn't
                   affect the result since multiplication is commutative.

    Returns:
        int: The total number of devices in the submesh defined by the dimension(s).
             Returns 1 if sharding_axis_names is an empty sequence.

    Raises:
        TypeError: If sharding_axis_names is not a str or a sequence of str.

    Examples:
        >>> calculate_mesh_dimension_size("data")  # Single dimension
        8
        >>> calculate_mesh_dimension_size(["data", "model"])  # Multiple dimensions
        32
    """
    if isinstance(sharding_axis_names, str):
        # Size along a single axis dimension
        return lax.psum(1, axis_name=sharding_axis_names)
    elif isinstance(sharding_axis_names, list | tuple):
        if not sharding_axis_names:
            return 1  # The size of a submesh with zero dimensions is 1

        # Calculate the product of sizes along each specified axis
        dimension_product = 1
        for dimension in sharding_axis_names:
            dimension_product *= lax.psum(1, axis_name=dimension)
        return dimension_product
    else:
        raise TypeError(
            f"Input 'sharding_axis_names' must be a string or sequence (list/tuple), "
            f"but got type {type(sharding_axis_names)}"
        )


def compute_device_linear_index(sharding_axis_names: AxisType) -> int:
    """
    Computes the linear index of the current device within the specified mesh dimensions.

    This function flattens the multi-dimensional coordinates of the device within
    the submesh defined by sharding_axis_names into a single integer index using row-major ordering.

    Args:
        sharding_axis_names: A single mesh dimension name (str) or a sequence (list/tuple)
                   of mesh dimension names, ordered from major to minor dimensions.
                   The order is important as it affects the resulting linear index.

    Returns:
        int: The 0-based linear index of the current device within the submesh.
             Returns 0 if sharding_axis_names is an empty sequence.

    Raises:
        TypeError: If sharding_axis_names is not a str or a sequence of str.

    Examples:
        >>> compute_device_linear_index("data")  # Single dimension
        2
        >>> compute_device_linear_index(["data", "model"])  # Multi-dimensional
        9

    Note:
        The calculation assumes row-major ordering where the rightmost dimension
        varies fastest (similar to C-style arrays).
    """
    if isinstance(sharding_axis_names, str):
        # Index along a single axis dimension
        return lax.axis_index(axis_name=sharding_axis_names)
    elif isinstance(sharding_axis_names, list | tuple):
        if not sharding_axis_names:
            return 0  # Index within a zero-dimensional submesh is 0

        device_index = 0
        stride = 1
        # Iterate from the minor axis to the major axis (reverse of the input order)
        # This implements row-major flattening: idx = sum(coord[dim] * stride[dim])
        for dimension in reversed(sharding_axis_names):
            dimension_index = lax.axis_index(axis_name=dimension)
            device_index += dimension_index * stride
            # Update stride for the next (more major) dimension
            dimension_size = lax.psum(1, axis_name=dimension)
            stride *= dimension_size
        return device_index
    else:
        raise TypeError(
            f"Input 'sharding_axis_names' must be a string or sequence (list/tuple), "
            f"but got type {type(sharding_axis_names)}"
        )


def prepare_matrix_for_all_gather(
    matrix: Array,
    device_mesh: Mesh,
    partition_dims: AxisType,
) -> Array:
    """
    Prepares a matrix for all-gather collective matrix multiplication by reshuffling data.

    This function reorganizes the input matrix across devices to optimize the subsequent
    all-gather collective matrix multiplication operation. It performs data swapping
    between pairs of devices to ensure proper data alignment.

    Args:
        matrix: The input matrix to be prepared
        device_mesh: The device mesh used for distributed computation
        partition_dims: The dimension names along which the matrix is partitioned

    Returns:
        Array: The prepared matrix with data appropriately reshuffled

    Note:
        This preprocessing step is crucial for the efficiency of the subsequent
        all-gather collective matrix multiplication.
    """

    def reshuffle_data(matrix: Array) -> Array:
        device_idx = compute_device_linear_index(sharding_axis_names=partition_dims)
        total_devices = calculate_mesh_dimension_size(sharding_axis_names=partition_dims)
        chunk_size = matrix.shape[0] // total_devices
        half_chunk_size = chunk_size // 2

        def swap_chunks(iter_idx, current_matrix):
            # Swap data between pairs of devices
            idx_1 = ((device_idx + iter_idx) % total_devices) * chunk_size
            idx_2 = ((device_idx - iter_idx) % total_devices) * chunk_size

            # Extract chunks to swap
            chunk_1 = jax.lax.dynamic_slice_in_dim(current_matrix, idx_1, half_chunk_size, axis=0)
            chunk_2 = jax.lax.dynamic_slice_in_dim(current_matrix, idx_2, half_chunk_size, axis=0)

            # Update matrix with swapped chunks
            current_matrix = jax.lax.dynamic_update_slice_in_dim(current_matrix, chunk_1, idx_2, axis=0)
            current_matrix = jax.lax.dynamic_update_slice_in_dim(current_matrix, chunk_2, idx_1, axis=0)
            return current_matrix

        # Perform swapping for all relevant device pairs
        matrix = jax.lax.fori_loop(1, total_devices // 2 + 1, swap_chunks, matrix)
        return matrix

    return shard_map(
        f=reshuffle_data,
        mesh=device_mesh,
        in_specs=matrix.sharding.spec,
        out_specs=matrix.sharding.spec,
    )(matrix)


def prepare_matrix_for_reduce_scatter(
    matrix: Array,
    device_mesh: Mesh,
    partition_dims: AxisType,
) -> Array:
    """
    Prepares a matrix for reduce-scatter collective matrix multiplication by reshuffling data.

    This function reorganizes the input matrix across devices to optimize the subsequent
    reduce-scatter collective matrix multiplication operation. It performs data swapping
    between pairs of devices along the column dimension.

    Args:
        matrix: The input matrix to be prepared
        device_mesh: The device mesh used for distributed computation
        partition_dims: The dimension names along which the matrix is partitioned

    Returns:
        Array: The prepared matrix with data appropriately reshuffled

    Note:
        This preprocessing step ensures efficient communication patterns during
        the subsequent reduce-scatter collective matrix multiplication.
    """

    def reshuffle_data(matrix: Array) -> Array:
        device_idx = compute_device_linear_index(sharding_axis_names=partition_dims)
        total_devices = calculate_mesh_dimension_size(sharding_axis_names=partition_dims)
        column_chunk_size = matrix.shape[1] // total_devices
        half_column_chunk_size = column_chunk_size // 2

        def swap_column_chunks(iter_idx, current_matrix):
            # Swap column data between pairs of devices
            idx_1 = ((device_idx + iter_idx) % total_devices) * column_chunk_size
            idx_2 = ((device_idx - iter_idx) % total_devices) * column_chunk_size

            # Extract column chunks to swap
            column_chunk_1 = jax.lax.dynamic_slice_in_dim(current_matrix, idx_1, half_column_chunk_size, axis=1)
            column_chunk_2 = jax.lax.dynamic_slice_in_dim(current_matrix, idx_2, half_column_chunk_size, axis=1)

            # Update matrix with swapped column chunks
            current_matrix = jax.lax.dynamic_update_slice_in_dim(current_matrix, column_chunk_1, idx_2, axis=1)
            current_matrix = jax.lax.dynamic_update_slice_in_dim(current_matrix, column_chunk_2, idx_1, axis=1)
            return current_matrix

        # Perform swapping for all relevant device pairs
        matrix = jax.lax.fori_loop(1, total_devices // 2 + 1, swap_column_chunks, matrix)
        return matrix

    return shard_map(
        f=reshuffle_data,
        mesh=device_mesh,
        in_specs=matrix.sharding.spec,
        out_specs=matrix.sharding.spec,
    )(matrix)


def perform_reduce_scatter_matmul(lhs: Array, rhs: Array, partition_dims: AxisType) -> Array:
    """
    Performs matrix multiplication with a reduce-scatter communication pattern.

    This function implements an efficient distributed matrix multiplication algorithm
    that uses a reduce-scatter communication pattern to minimize data movement.
    The algorithm processes chunks of the right-hand side matrix and accumulates
    partial results while shuffling data between devices.

    Args:
        lhs: Left-hand side matrix
        rhs: Right-hand side matrix (should be pre-processed with prepare_matrix_for_reduce_scatter)
        partition_dims: Dimension names for collective operations

    Returns:
        Array: The result of the distributed matrix multiplication

    Note:
        This implementation achieves better performance compared to naive distributed
        matrix multiplication by optimizing communication patterns.
    """
    device_idx = compute_device_linear_index(sharding_axis_names=partition_dims)
    total_devices = calculate_mesh_dimension_size(sharding_axis_names=partition_dims)
    column_chunk_size = rhs.shape[1] // total_devices

    # Start with the chunk one ahead of the current device
    initial_chunk_idx = ((device_idx + 1) % total_devices) * column_chunk_size
    initial_chunk = jax.lax.dynamic_slice_in_dim(
        rhs,
        start_index=initial_chunk_idx,
        slice_size=column_chunk_size,
        axis=1,
    )
    initial_result = lhs @ initial_chunk

    # Initialize accumulators for forward and backward communication
    half_chunk_size = column_chunk_size // 2
    accumulator_shape = (lhs.shape[0], half_chunk_size)
    forward_accumulator = jnp.zeros(shape=accumulator_shape, dtype=lhs.dtype)
    backward_accumulator = jnp.zeros(shape=accumulator_shape, dtype=lhs.dtype)

    def process_iteration(iter_idx, carry):
        forward_accumulator, backward_accumulator, current_result = carry

        # Combine accumulators and add current result
        combined_accumulator = jnp.concatenate((forward_accumulator, backward_accumulator), axis=1)
        combined_accumulator += current_result

        # Split for next iteration
        forward_accumulator, backward_accumulator = jnp.split(combined_accumulator, 2, axis=1)

        # Process next chunk
        next_chunk_idx = ((device_idx + 1 + iter_idx) % total_devices) * column_chunk_size
        next_chunk = jax.lax.dynamic_slice_in_dim(
            rhs,
            start_index=next_chunk_idx,
            slice_size=column_chunk_size,
            axis=1,
        )
        next_result = lhs @ next_chunk

        # Communicate accumulators to adjacent devices
        forward_accumulator = jax.lax.ppermute(
            forward_accumulator,
            partition_dims,
            [(j, (j + 1) % total_devices) for j in range(total_devices)],
        )
        backward_accumulator = jax.lax.ppermute(
            backward_accumulator,
            partition_dims,
            [(j, (j - 1) % total_devices) for j in range(total_devices)],
        )

        return forward_accumulator, backward_accumulator, next_result

    # Process all remaining chunks
    forward_accumulator, backward_accumulator, final_result = jax.lax.fori_loop(
        1,
        total_devices,
        process_iteration,
        (forward_accumulator, backward_accumulator, initial_result),
    )

    # Final accumulation
    final_combined = jnp.concatenate((forward_accumulator, backward_accumulator), axis=1)
    final_combined += final_result

    return final_combined


def perform_all_gather_matmul(lhs: Array, rhs: Array, partition_dims: AxisType) -> Array:
    """
    Performs matrix multiplication with an all-gather communication pattern.

    This function implements an efficient distributed matrix multiplication algorithm
    that uses an all-gather communication pattern. It processes chunks of the right-hand
    side matrix row-wise and accumulates partial results while shuffling the left-hand
    side matrix between devices.

    Args:
        lhs: Left-hand side matrix
        rhs: Right-hand side matrix (should be pre-processed with prepare_matrix_for_all_gather)
        partition_dims: Dimension names for collective operations

    Returns:
        Array: The result of the distributed matrix multiplication

    Note:
        This implementation achieves better performance compared to naive distributed
        matrix multiplication by optimizing communication patterns.
    """
    device_idx = compute_device_linear_index(sharding_axis_names=partition_dims)
    total_devices = calculate_mesh_dimension_size(sharding_axis_names=partition_dims)
    row_chunk_size = rhs.shape[0] // total_devices

    # Initialize result accumulator
    result_shape = (lhs.shape[0], rhs.shape[1])
    result_accumulator = jnp.zeros(shape=result_shape, dtype=lhs.dtype)

    # Split left-hand side for communication
    forward_lhs, backward_lhs = jnp.split(lhs, 2, axis=1)

    def process_iteration(iter_idx, carry):
        accumulator, forward_lhs_part, backward_lhs_part = carry

        # Get the current chunk of the right-hand side
        current_row_idx = ((device_idx + iter_idx) % total_devices) * row_chunk_size
        current_rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, current_row_idx, row_chunk_size, axis=0)

        # Recombine left-hand side and compute partial result
        current_lhs = jnp.concatenate((forward_lhs_part, backward_lhs_part), axis=1)
        partial_result = current_lhs @ current_rhs_chunk

        # Accumulate result
        updated_accumulator = accumulator + partial_result

        # Communicate left-hand side parts to adjacent devices
        updated_forward_lhs = jax.lax.ppermute(
            forward_lhs_part,
            partition_dims,
            [(j, (j + 1) % total_devices) for j in range(total_devices)],
        )
        updated_backward_lhs = jax.lax.ppermute(
            backward_lhs_part,
            partition_dims,
            [(j, (j - 1) % total_devices) for j in range(total_devices)],
        )

        return updated_accumulator, updated_forward_lhs, updated_backward_lhs

    # Process all but the last chunk through iterations
    result_accumulator, forward_lhs, backward_lhs = jax.lax.fori_loop(
        0,
        total_devices - 1,
        process_iteration,
        (result_accumulator, forward_lhs, backward_lhs),
    )

    # Process final chunk (no need for communication after this)
    final_row_idx = ((device_idx + total_devices - 1) % total_devices) * row_chunk_size
    final_rhs_chunk = jax.lax.dynamic_slice_in_dim(rhs, final_row_idx, row_chunk_size, axis=0)
    final_lhs = jnp.concatenate((forward_lhs, backward_lhs), axis=1)
    final_result = final_lhs @ final_rhs_chunk
    result_accumulator += final_result

    return result_accumulator


@enum.unique
class MatrixMultiplyMethod(enum.Enum):
    """
    Enumeration of distributed matrix multiplication methods.

    Attributes:
        ALL_GATHER: Matrix multiplication using all-gather communication pattern.
            Suitable when the output needs to be fully replicated across devices.
        REDUCE_SCATTER: Matrix multiplication using reduce-scatter communication pattern.
            Efficient when the output can be partitioned across devices.
    """

    ALL_GATHER = enum.auto()
    REDUCE_SCATTER = enum.auto()


def create_distributed_matmul(
    method: MatrixMultiplyMethod,
    partition_dims: AxisType,
) -> tp.Callable[[Array, Array], Array]:
    """
    Creates a distributed matrix multiplication function using the specified method.

    This factory function returns a specialized matrix multiplication function that
    implements the requested distributed computation strategy.

    Args:
        method: The distributed matrix multiplication method to use
        partition_dims: Dimension names for collective operations

    Returns:
        A function that performs distributed matrix multiplication using the specified method

    Raises:
        ValueError: If an unsupported matrix multiplication method is provided

    Example:
        >>> matmul_fn = create_distributed_matmul(MatrixMultiplyMethod.ALL_GATHER, "data")
        >>> result = matmul_fn(left_matrix, right_matrix)
    """
    if method == MatrixMultiplyMethod.ALL_GATHER:
        return partial(perform_all_gather_matmul, partition_dims=partition_dims)
    elif method == MatrixMultiplyMethod.REDUCE_SCATTER:
        return partial(perform_reduce_scatter_matmul, partition_dims=partition_dims)
    else:
        raise ValueError(f"Unsupported distributed matrix multiplication method: {method}")


def test_reduce_scatter_matmul():
    """
    Tests the reduce-scatter distributed matrix multiplication implementation.

    This function creates a test case with random matrices, computes the expected result
    using standard matrix multiplication, and verifies that the distributed implementation
    produces the same result within numerical tolerance.

    Returns:
        tuple: A tuple containing (actual_result, expected_result)
    """
    # Create device mesh for distributed computation
    device_mesh = es.create_mesh((1, 1, 1, -1, 1))

    # Generate random test matrices
    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    left_matrix = jax.random.uniform(key1, shape=(8, 64), dtype=jnp.float32)
    right_matrix = jax.random.uniform(key2, shape=(64, 32), dtype=jnp.float32)

    # Compute expected result using standard matrix multiplication
    expected_result = left_matrix @ right_matrix

    # Prepare right matrix for distributed computation
    distributed_right = jax.device_put(right_matrix, NamedSharding(device_mesh, Ps("tp", None)))
    prepared_right = prepare_matrix_for_reduce_scatter(distributed_right, device_mesh, "tp")

    # Function to be executed on each device
    def distributed_matmul_wrapper(left, right, method, dims):
        return create_distributed_matmul(method, dims)(left, right)

    # Execute distributed matrix multiplication
    actual_result = shard_map(
        f=partial(
            distributed_matmul_wrapper,
            method=MatrixMultiplyMethod.REDUCE_SCATTER,
            dims="tp",
        ),
        mesh=device_mesh,
        in_specs=(Ps(("sp", "fsdp"), "tp"), Ps("tp", ("sp", "fsdp"))),
        out_specs=Ps(("sp", "fsdp"), "tp"),
    )(left_matrix, prepared_right)

    # Verify results
    error_magnitude = jnp.sum(jnp.abs(actual_result - expected_result))
    print(f"Reduce-Scatter Matrix Multiply Error: {error_magnitude}")
    np.testing.assert_allclose(actual_result, expected_result, rtol=1e-6)

    return actual_result, expected_result


def test_all_gather_matmul():
    """
    Tests the all-gather distributed matrix multiplication implementation.

    This function creates a test case with random matrices, computes the expected result
    using standard matrix multiplication, and verifies that the distributed implementation
    produces the same result within numerical tolerance.

    Returns:
        tuple: A tuple containing (actual_result, expected_result)
    """
    # Create device mesh for distributed computation
    device_mesh = es.create_mesh((1, 1, 1, -1, 1))

    # Generate random test matrices
    key1, key2 = jax.random.split(jax.random.PRNGKey(0))
    left_matrix = jax.random.uniform(key1, shape=(8, 64), dtype=jnp.float32)
    right_matrix = jax.random.uniform(key2, shape=(64, 32), dtype=jnp.float32)

    # Compute expected result using standard matrix multiplication
    expected_result = left_matrix @ right_matrix

    # Prepare right matrix for distributed computation
    distributed_right = jax.device_put(right_matrix, NamedSharding(device_mesh, Ps(("sp", "fsdp"), "tp")))
    prepared_right = prepare_matrix_for_all_gather(distributed_right, device_mesh, "tp")

    # Function to be executed on each device
    def distributed_matmul_wrapper(left, right, method, dims):
        return create_distributed_matmul(method, dims)(left, right)

    # Execute distributed matrix multiplication
    actual_result = shard_map(
        f=partial(
            distributed_matmul_wrapper,
            method=MatrixMultiplyMethod.ALL_GATHER,
            dims="tp",
        ),
        mesh=device_mesh,
        in_specs=(Ps(("sp", "fsdp"), "tp"), Ps(("sp", "fsdp"), "tp")),
        out_specs=Ps(("sp", "fsdp"), "tp"),
    )(left_matrix, prepared_right)

    # Verify results
    error_magnitude = jnp.sum(jnp.abs(actual_result - expected_result))
    print(f"All-Gather Matrix Multiply Error: {error_magnitude}")
    np.testing.assert_allclose(actual_result, expected_result, rtol=1e-6)

    return actual_result, expected_result


def run_all_tests():
    """
    Runs all distributed matrix multiplication tests and reports results.

    This function executes both the all-gather and reduce-scatter matrix
    multiplication tests and collects their results.

    Returns:
        dict: A dictionary containing test results for both methods
    """
    print("=== Testing Distributed Matrix Multiplication Implementations ===")

    print("\nRunning All-Gather Matrix Multiplication test...")
    ag_result, ag_expected = test_all_gather_matmul()

    print("\nRunning Reduce-Scatter Matrix Multiplication test...")
    rs_result, rs_expected = test_reduce_scatter_matmul()

    print("\n=== All tests completed successfully! ===")

    return {
        "all_gather": {
            "result": ag_result,
            "expected": ag_expected,
            "error": jnp.sum(jnp.abs(ag_result)) - jnp.sum(jnp.abs(ag_expected)),
        },
        "reduce_scatter": {
            "result": rs_result,
            "expected": rs_expected,
            "error": jnp.sum(jnp.abs(rs_result)) - jnp.sum(jnp.abs(rs_expected)),
        },
    }


if __name__ == "__main__":
    test_results = run_all_tests()

    # Report summary of test results
    print("\nError Summary:")
    print(f"All-Gather Method Error: {test_results['all_gather']['error']}")
    print(f"Reduce-Scatter Method Error: {test_results['reduce_scatter']['error']}")
