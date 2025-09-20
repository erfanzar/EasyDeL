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


import functools
from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ._utils import assert_is_supported_dtype, select_input_dtype


def _validate_args(
    *,
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    expected_rhs_dims: int = 3,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.dtype]:
    """Validate input arguments for grouped matrix multiplication operations.

    This function performs comprehensive validation of the input tensors to ensure
    they meet the requirements for grouped matrix multiplication. It checks tensor
    dimensions, data types, and determines the appropriate input dtype for computation.

    Args:
        lhs: Left-hand side matrix for multiplication. Must be a 2D tensor with shape [m, k]
            where m is the total number of rows across all groups and k is the inner dimension.
        rhs: Right-hand side matrix/tensor for multiplication. For grouped_matmul, expects a 3D tensor
            with shape [num_groups, k, n]. For transposed_grouped_matmul, expects a 2D tensor with shape [m, n].
        group_sizes: 1D tensor of group sizes with shape [num_groups] and dtype int32.
            Each element specifies the number of rows in the corresponding group.
            Must sum to m (the first dimension of lhs).
        expected_rhs_dims: Expected number of dimensions for rhs tensor. Defaults to 3 for grouped_matmul,
            but can be set to 2 for transposed_grouped_matmul operation.

    Returns:
        A tuple containing:
            - lhs: The validated left-hand side tensor (unchanged)
            - group_sizes: The validated group sizes tensor (unchanged)
            - input_dtype: The selected dtype for computation, determined by examining
              both lhs and rhs dtypes to ensure compatibility

    Raises:
        ValueError: If lhs is not a 2D tensor
        ValueError: If rhs does not have the expected number of dimensions
        ValueError: If group_sizes is not int32 dtype
        AssertionError: If lhs or rhs have unsupported dtypes (via assert_is_supported_dtype)

    Notes:
        - The function uses assert_is_supported_dtype to ensure tensors have TPU-compatible dtypes
        - The select_input_dtype function determines the optimal dtype for TPU computation
          based on both input tensors
    """

    if lhs.ndim != 2:
        raise ValueError(f"Expected 2-tensor for 'lhs' but got {lhs.ndim}-tensor.")
    assert_is_supported_dtype(lhs.dtype)

    if rhs.ndim != expected_rhs_dims:
        raise ValueError(f"Expected {expected_rhs_dims}-tensor for 'rhs' but got {rhs.ndim}-tensor.")
    assert_is_supported_dtype(rhs.dtype)

    if group_sizes.dtype != jnp.int32:
        raise ValueError(f"Expected 32-bit integer 'group_sizes' but got {group_sizes.dtype}.")

    return lhs, group_sizes, select_input_dtype(lhs, rhs)


def _calculate_num_tiles(x: int, tx: int) -> int:
    """Calculate the number of tiles needed for a dimension requiring exact divisibility.

    This function computes how many tiles of size tx are needed to cover dimension x,
    enforcing that x must be evenly divisible by tx. This is used for dimensions that
    require exact tiling without remainder for correct TPU kernel execution.

    Args:
        x: The dimension size to be tiled (e.g., matrix dimension m, k, or n)
        tx: The tile size for dimension x (must evenly divide x)

    Returns:
        The number of tiles needed (x // tx)

    Raises:
        ValueError: If x is not evenly divisible by tx, indicating incompatible
                   dimension and tile size combination

    Example:
        >>> _calculate_num_tiles(256, 64)
        >>> _calculate_num_tiles(250, 64)

    Notes:
        - This function is used for dimensions that must align perfectly with TPU tiles
        - For dimensions that can handle partial tiles, use _calculate_irregular_num_tiles instead
    """
    tiles, rem = divmod(x, tx)
    if rem:
        raise ValueError(f"{x} must be divisible by x-dimension tile size ({tx}).")
    return tiles


def _calculate_irregular_num_tiles(x: int, tx: int) -> tuple[int, int]:
    """Calculate the number of tiles needed for a dimension allowing partial tiles.

    This function computes how many tiles of size tx are needed to cover dimension x,
    including a potential partial tile if x is not evenly divisible by tx. This is
    used for dimensions that can handle irregular tiling with masking.

    Args:
        x: The dimension size to be tiled (e.g., matrix dimension k or n)
        tx: The tile size for dimension x

    Returns:
        A tuple containing:
            - tiles: Total number of tiles needed (including partial tile if necessary)
            - rem: The remainder (size of the partial tile), or 0 if x is evenly divisible

    Example:
        >>> _calculate_irregular_num_tiles(250, 64)
        >>> _calculate_irregular_num_tiles(256, 64)

    Notes:
        - The function rounds up the number of tiles to ensure full coverage
        - The remainder is used for masking operations in the kernel to handle partial tiles
        - This is particularly useful for k and n dimensions in matrix multiplication
          where padding or masking can be applied
    """
    tiles, rem = divmod(x, tx)
    if rem:
        tiles += 1
    return tiles, rem


GroupMetadata = Any


def make_group_metadata(
    *,
    group_sizes: jnp.ndarray,
    m: int,
    tm: int,
    start_group: jnp.ndarray,
    num_nonzero_groups: int,
    visit_empty_groups: bool = True,
) -> GroupMetadata:
    """Create metadata for efficient grouped matrix multiplication on TPU.

    This function generates the metadata structures needed to efficiently execute grouped
    matrix multiplication on TPU hardware. It handles the complex mapping between groups
    and tiles, accounting for groups that may not align with tile boundaries and managing
    partial tiles that span multiple groups.

    The algorithm works by:
    1. Computing group offsets (CSR-like row offsets for each group)
    2. Calculating tile assignments for each group, handling boundary cases
    3. Creating mappings from grid indices to group IDs and tile IDs
    4. Accounting for sharding when groups are distributed across devices

    Args:
        group_sizes: 1D array of shape [num_groups] with int32 dtype. Each element
            specifies the number of rows in the corresponding group. Must sum to m.
        m: Total number of rows across all groups in the left-hand side matrix.
        tm: Tile size for the m dimension. Must evenly divide m for correctness.
        start_group: Scalar indicating the first group to process (0-indexed).
            Used for sharding groups across multiple devices.
        num_nonzero_groups: Number of consecutive groups to process starting from
            start_group. Enables processing a subset of groups.
        visit_empty_groups: If True, allocate tiles for groups with size 0.
            Required for transposed_grouped_matmul to ensure output is properly zeroed for empty groups.
            If False, empty groups are skipped (used in grouped_matmul for efficiency).

    Returns:
        A tuple containing:
            - group_metadata: A tuple of three arrays:
                * group_offsets: Shape [num_groups+1], int32. CSR-style offsets where
                  group_offsets[i] is the starting row of group i, and
                  group_offsets[num_groups] = m.
                * group_ids: Shape [m_tiles + num_groups - 1], int32. Maps each grid
                  index to its corresponding group ID.
                * m_tile_ids: Shape [m_tiles + num_groups - 1], int32. Maps each grid
                  index to its corresponding m-dimension tile ID.
            - num_tiles: Total number of tiles to execute for the specified groups.

    Algorithm Details:
        The function handles several complex cases:
        - Groups that don't start or end on tile boundaries require partial tile processing
        - Tiles that span multiple groups need to be visited multiple times
        - Empty groups may need special handling depending on the operation (grouped_matmul vs transposed_grouped_matmul)
        - Sharding requires adjusting the metadata to process only local groups

    TPU Optimizations:
        - Tiles are sized to match TPU's native matrix multiply units (typically 128x128)
        - Metadata is structured to minimize memory access patterns
        - Grid layout ensures coalesced memory access and efficient tile reuse
        - Partial tiles are handled through masking rather than padding to save memory

    Example:
        >>> group_sizes = jnp.array([100, 150, 50], dtype=jnp.int32)
        >>> metadata, num_tiles = make_group_metadata(
        ...     group_sizes=group_sizes, m=300, tm=128,
        ...     start_group=jnp.array(0), num_nonzero_groups=3)

    """
    num_groups = group_sizes.shape[0]
    end_group = start_group + num_nonzero_groups - 1

    group_ends = jnp.cumsum(group_sizes)
    group_offsets = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends])

    rounded_group_ends = ((group_ends + tm - 1) // tm * tm).astype(jnp.int32)

    group_starts = jnp.concatenate([jnp.zeros(1, dtype=jnp.int32), group_ends[:-1]])
    rounded_group_starts = group_starts // tm * tm

    rounded_group_sizes = rounded_group_ends - rounded_group_starts
    rounded_group_sizes = jnp.where(group_sizes == 0, 0, rounded_group_sizes)

    group_tiles = rounded_group_sizes // tm

    if visit_empty_groups:
        group_tiles = jnp.where(group_sizes == 0, 1, group_tiles)

    tiles_m = _calculate_num_tiles(m, tm)
    group_ids = jnp.repeat(
        jnp.arange(num_groups, dtype=jnp.int32),
        group_tiles,
        total_repeat_length=tiles_m + num_groups - 1,
    )

    partial_tile_mask = jnp.logical_or((group_offsets[:-1] % tm) == 0, group_sizes == 0)

    if visit_empty_groups:
        partial_tile_mask = jnp.where(group_sizes == 0, 0, partial_tile_mask)

    partial_tile_ids = jnp.where(partial_tile_mask, tiles_m, group_offsets[:-1] // tm)

    tile_visits = jnp.histogram(partial_tile_ids, bins=tiles_m, range=(0, tiles_m - 1))[0] + 1

    m_tile_ids = jnp.repeat(
        jnp.arange(tiles_m, dtype=jnp.int32),
        tile_visits.astype(jnp.int32),
        total_repeat_length=tiles_m + num_groups - 1,
    )

    first_tile_in_shard = (group_ids < start_group).sum()
    group_ids = jnp.roll(group_ids, shift=-first_tile_in_shard, axis=0)
    m_tile_ids = jnp.roll(m_tile_ids, shift=-first_tile_in_shard, axis=0)

    iota = jnp.arange(num_groups, dtype=jnp.int32)
    active_group_mask = jnp.logical_and(iota <= end_group, iota >= start_group)
    group_tiles = jnp.where(active_group_mask, group_tiles, 0)
    num_tiles = group_tiles.sum()
    return (group_offsets, group_ids, m_tile_ids), num_tiles


def _get_group_size(*, grid_id: jnp.ndarray, group_metadata: GroupMetadata) -> jnp.ndarray:
    """Calculate the number of rows in the group being processed by a grid index.

    This helper function determines the size (number of rows) of the group that
    corresponds to a given grid index in the TPU kernel execution grid.

    Args:
        grid_id: Scalar or array representing the current grid index in the kernel.
            This is typically obtained from pl.program_id() in the Pallas kernel.
        group_metadata: Tuple containing group metadata arrays:
            - group_offsets: CSR-style offsets for each group
            - group_ids: Mapping from grid indices to group IDs
            - m_tile_ids: Mapping from grid indices to tile IDs (unused here)

    Returns:
        The number of rows in the group corresponding to grid_id. Returns 0 for
        empty groups.

    Notes:
        - This function is used within TPU kernels to determine group boundaries
        - The group size is calculated as group_offsets[i+1] - group_offsets[i]
        - Essential for handling variable-sized groups in grouped operations
    """
    group_offsets, group_ids = group_metadata[:2]
    group_id = group_ids[grid_id]
    group_start = group_offsets[group_id]
    group_end = group_offsets[group_id + 1]
    return group_end - group_start


def _get_store_mask(
    *,
    grid_id: jnp.ndarray,
    group_metadata: GroupMetadata,
    tm: int,
    tn: int,
) -> jnp.ndarray:
    """Generate a mask for valid elements within a tile for the current group.

    This function creates a boolean mask that identifies which elements in a tile
    belong to the current group being processed. This is crucial for handling tiles
    that span multiple groups or contain partial group data.

    Args:
        grid_id: Current grid index in the kernel execution, obtained from pl.program_id().
        group_metadata: Tuple containing:
            - group_offsets: CSR-style row offsets for each group
            - group_ids: Mapping from grid indices to group IDs
            - m_tile_ids: Mapping from grid indices to m-dimension tile IDs
        tm: Tile size for the m dimension (number of rows in a tile).
        tn: Tile size for the n dimension (number of columns in a tile).

    Returns:
        A boolean mask of shape [tm, tn] where True indicates the element belongs
        to the current group and should be included in computation/storage.

    Algorithm:
        1. Determine the current group and its row boundaries from metadata
        2. Calculate the global row indices for the current tile
        3. Create a mask where elements are True if their row index falls within
           the group's boundaries

    TPU Optimization Notes:
        - Uses broadcasted_iota for efficient index generation on TPU
        - Boolean masks enable predicated execution on TPU's vector units
        - Avoids explicit loops by using vectorized comparisons

    Example:
        If processing a tile starting at row 120 with tm=128, and the current group
        spans rows 100-180, the mask will be True for all elements since the entire
        tile falls within the group.
    """
    group_offsets, group_ids, m_tile_ids = group_metadata[:3]
    group_id = group_ids[grid_id]
    group_start = group_offsets[group_id]
    group_end = group_offsets[group_id + 1]
    m_id = m_tile_ids[grid_id] * tm
    iota = jax.lax.broadcasted_iota(jnp.int32, (tm, tn), 0) + m_id
    return jnp.logical_and(iota >= group_start, iota < group_end)


def _zero_uninitialized_memory(
    out: jnp.ndarray,
    *,
    start_group: jnp.ndarray,
    num_nonzero_groups: int,
    group_metadata: GroupMetadata,
) -> jnp.ndarray:
    """Zero out memory regions in output that weren't written by the kernel.

    When processing a subset of groups (e.g., in sharded execution), some regions
    of the output tensor may not be written to by the kernel. This function ensures
    those regions are properly zeroed to maintain correctness.

    Args:
        out: Output tensor of shape [m, n] containing the computation results.
            May have uninitialized regions if not all groups were processed.
        start_group: Index of the first group that was processed (0-based).
        num_nonzero_groups: Number of consecutive groups that were processed
            starting from start_group.
        group_metadata: Tuple containing group offset information:
            - group_offsets: Array mapping group indices to row offsets

    Returns:
        Output tensor with uninitialized regions zeroed. Elements corresponding
        to processed groups remain unchanged.

    Algorithm:
        1. Calculate the row range that was actually processed
        2. Create a mask for valid rows based on group boundaries
        3. Zero out rows outside the processed range

    Use Cases:
        - Sharded execution where each device processes a subset of groups
        - Partial group processing for memory efficiency
        - Ensuring deterministic output when not all groups are computed

    Notes:
        - Essential for correctness when num_current_groups < num_total_groups
        - Uses broadcasted operations for efficiency on TPU
        - Preserves computed values while zeroing unwritten memory
    """
    group_offsets = group_metadata[0]
    group_start = group_offsets[start_group]
    group_end = group_offsets[start_group + num_nonzero_groups]
    valid_mask = jax.lax.broadcasted_iota(jnp.int32, (out.shape[0],), 0)
    valid_mask = (valid_mask >= group_start) & (valid_mask < group_end)
    return jnp.where(valid_mask[:, None], out, 0)


LutFn = Callable[[int, int, int], tuple[int, int, int] | None]


@functools.partial(
    jax.jit,
    static_argnames=["preferred_element_type", "tiling", "transpose_rhs", "interpret"],
)
def grouped_matmul(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128),
    group_offset: jnp.ndarray | None = None,
    existing_out: jnp.ndarray | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
) -> jnp.ndarray:
    """Grouped Matrix Multiplication: Compute separate matrix products for each group.

    This function performs grouped matrix multiplication where different row slices of
    the left-hand side matrix are multiplied with different matrices from the right-hand
    side tensor. Mathematically, for each group i:
        out[start_i:end_i, :] = lhs[start_i:end_i, :] @ rhs[i, :, :]
    where start_i and end_i are determined by group_sizes.

    The implementation uses Pallas to generate efficient TPU kernels that:
    - Process multiple groups in a single kernel launch
    - Handle groups that don't align with tile boundaries
    - Support incremental accumulation for memory efficiency
    - Optimize memory access patterns for TPU's memory hierarchy

    Args:
        lhs: Left-hand side matrix of shape [m, k] where m is the total number
            of rows across all groups and k is the inner dimension.
        rhs: Right-hand side tensor of shape [num_groups, k, n] containing a
            separate matrix for each group. Can be transposed if transpose_rhs=True.
        group_sizes: 1D array of shape [num_groups] with int32 dtype. Each element
            specifies the number of rows in lhs belonging to that group.
            Must sum to m (first dimension of lhs).
        preferred_element_type: Output dtype. Defaults to float32. The kernel uses
            float32 for accumulation regardless, then casts to this type.
        tiling: Tile sizes as (tm, tk, tn) tuple, or a callable that returns tile
            sizes based on problem dimensions. Typical TPU tiles are 128x128.
            The callable signature is (m, k, n) -> (tm, tk, tn) | None.
        group_offset: Starting group index for sharded execution. Defaults to 0.
            Useful when distributing groups across multiple devices.
        existing_out: Optional pre-existing output tensor to accumulate into.
            Must have shape [m, n] and dtype matching preferred_element_type.
            Enables incremental computation and memory reuse.
        transpose_rhs: If True, expects rhs shape [num_groups, n, k] and transposes
            during multiplication. Useful for certain memory layouts.
        interpret: Run kernel in interpret mode for debugging. Slower but provides
            better error messages and doesn't require compilation.

    Returns:
        Output matrix of shape [m, n] containing the concatenated results of all
        group matrix multiplications.

    Algorithm Overview:
        1. Validate inputs and determine computation parameters
        2. Create group metadata for efficient tile-to-group mapping
        3. Define Pallas kernel that:
           - Loads tiles from lhs and group-specific slices from rhs
           - Accumulates partial products in on-chip memory
           - Masks and stores results respecting group boundaries
        4. Launch kernel with appropriate grid dimensions
        5. Zero unprocessed regions if doing partial computation

    TPU Optimizations:
        - Tiles sized to match TPU's Matrix Multiply Units (typically 128x128)
        - Prefetch scheduling for hiding memory latency
        - VMEM scratch space for accumulation to minimize HBM traffic
        - Efficient masking for partial tiles using TPU's predication
        - Dimension semantics hints for XLA compiler optimization

    Example:
        >>>
        >>> lhs = jnp.randn(300, 64)
        >>> rhs = jnp.randn(3, 64, 32)
        >>> group_sizes = jnp.array([100, 150, 50], dtype=jnp.int32)
        >>> result = grouped_matmul(lhs, rhs, group_sizes)

    Notes:
        - The k dimension can have partial tiles (handled via masking)
        - The m dimension must be divisible by tm for correctness
        - Empty groups (size 0) are skipped for efficiency
        - Cost estimation helps XLA make scheduling decisions
    """

    if existing_out is not None:
        assert isinstance(existing_out, jax.Array)
        expected_dtype = existing_out.dtype
        if expected_dtype != preferred_element_type:
            raise ValueError("Existing output dtype must match preferred_element_type.")
    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    else:
        if group_offset.shape:
            raise ValueError(f"group_offset must be a ()-shaped array. Got: {group_offset.shape}.")
        group_offset = group_offset[None]
    num_current_groups = rhs.shape[0]
    num_total_groups = group_sizes.shape[0]
    lhs, group_sizes, input_dtype = _validate_args(lhs=lhs, rhs=rhs, group_sizes=group_sizes)

    m, k, n = (lhs.shape[0], lhs.shape[1], rhs.shape[2])
    if transpose_rhs:
        n = rhs.shape[1]

    if callable(tiling):
        tiling = tiling(m, k, n)

    if tiling is None:
        raise ValueError(f"No tuned tiling found for (m, k, n) = ({m}, {k}, {n})")

    tm, tk, tn = tiling
    tiles_k, k_rem = _calculate_irregular_num_tiles(k, tk)
    tiles_n, n_rem = _calculate_irregular_num_tiles(n, tn)
    del n_rem

    group_metadata, num_active_tiles = make_group_metadata(  # pylint: disable=unbalanced-tuple-unpacking
        group_sizes=group_sizes,
        m=m,
        tm=tm,
        start_group=group_offset[0],
        num_nonzero_groups=rhs.shape[0],
        visit_empty_groups=False,
    )

    if transpose_rhs:
        dot_general_dims = (((1,), (1,)), ((), ()))
    else:
        dot_general_dims = (((1,), (0,)), ((), ()))

    def kernel(
        group_metadata,
        group_offset,
        lhs: jax.Array,
        rhs: jax.Array,
        existing_out,
        out,
        acc_scratch,
    ):
        group_offsets, group_ids, m_tile_ids = group_metadata
        del group_offsets, group_ids, group_offset

        grid_id = pl.program_id(1)
        k_i = pl.program_id(2)

        @pl.when(k_i == 0)
        def _zero_acc():
            acc_scratch[...] = jnp.zeros_like(acc_scratch)

            if existing_out is not None:
                prev_grid_id = jnp.where(grid_id > 0, grid_id - 1, 0)
                is_first_processed_group = grid_id == 0
                m_tile_changed = m_tile_ids[grid_id] != m_tile_ids[prev_grid_id]
                first_time_seeing_out = jnp.logical_or(is_first_processed_group, m_tile_changed)

                @pl.when(first_time_seeing_out)
                def _init_out():
                    out[...] = existing_out[...]

        def mask_k_rem(x: jax.Array, *, dim: int):
            if k_rem == 0:
                return x

            orig_dtype = x.dtype
            iota = lax.broadcasted_iota(jnp.int32, x.shape, dim)
            x = x.astype(jnp.float32)
            return jnp.where(iota < k_rem, x, 0).astype(orig_dtype)

        def _store_accum():
            mask = _get_store_mask(
                grid_id=grid_id,
                group_metadata=group_metadata,
                tm=tm,
                tn=tn,
            )
            to_store = acc_scratch[...]
            out[...] = jax.lax.select(mask[...], to_store, out[...].astype(jnp.float32)).astype(preferred_element_type)

        def _accum(is_last_k_tile):
            if is_last_k_tile:
                mask_k_rem_lhs = partial(mask_k_rem, dim=1)
                mask_k_rem_rhs = partial(mask_k_rem, dim=int(transpose_rhs))
            else:

                def mask_k_rem_lhs(x):
                    return x

                def mask_k_rem_rhs(x):
                    return x

            loaded_lhs = mask_k_rem_lhs(lhs[...]).astype(input_dtype)
            loaded_rhs = mask_k_rem_rhs(rhs[...]).astype(input_dtype)

            acc_scratch[...] += jax.lax.dot_general(
                loaded_lhs,
                loaded_rhs,
                preferred_element_type=jnp.float32,
                dimension_numbers=dot_general_dims,
            )
            if is_last_k_tile:
                _store_accum()

        lax.cond(
            k_i == tiles_k - 1,
            partial(_accum, True),
            partial(_accum, False),
        )

    def lhs_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset):
        group_offsets, group_ids, m_tile_ids = group_metadata
        del n_i, group_offsets, group_ids, group_offset
        return m_tile_ids[grid_id], k_i

    def rhs_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset):
        group_offsets, group_ids, m_tile_ids = group_metadata
        del group_offsets, m_tile_ids
        if transpose_rhs:
            k_i, n_i = n_i, k_i

        return group_ids[grid_id] - group_offset[0], k_i, n_i

    def out_transform_indices(n_i, grid_id, k_i, group_metadata, group_offset):
        group_offsets, group_ids, m_tile_ids = group_metadata
        del k_i, group_offsets, group_ids, group_offset
        return m_tile_ids[grid_id], n_i

    out_block_spec = pl.BlockSpec((tm, tn), out_transform_indices)
    if existing_out is None:
        in_out_block_spec: Any = None
        input_output_aliases = {}
    else:
        in_out_block_spec = out_block_spec
        existing_out_arg_index = 6

        input_output_aliases = {existing_out_arg_index: 0}

    lhs_block_spec = pl.BlockSpec((tm, tk), lhs_transform_indices)
    if transpose_rhs:
        rhs_block_spec = pl.BlockSpec((None, tn, tk), rhs_transform_indices)
    else:
        rhs_block_spec = pl.BlockSpec((None, tk, tn), rhs_transform_indices)

    lhs_bytes = lhs.size * lhs.itemsize

    rhs_bytes = (k * n) * rhs.itemsize
    out_bytes = (m * n) * jnp.dtype(preferred_element_type).itemsize
    max_active_tiles = group_metadata[1].size
    bytes_accessed = (lhs_bytes * tiles_n) + (rhs_bytes * max_active_tiles) + out_bytes
    flops = 2 * m * k * n
    cost_estimate = pl.CostEstimate(flops=flops, bytes_accessed=bytes_accessed, transcendentals=0)

    pallas_call_fn = pl.pallas_call
    call_gmm = pallas_call_fn(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), preferred_element_type),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                lhs_block_spec,
                rhs_block_spec,
                in_out_block_spec,
            ],
            out_specs=out_block_spec,
            grid=(tiles_n, num_active_tiles, tiles_k),
            scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)],
        ),
        input_output_aliases=input_output_aliases,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary", "arbitrary")),
        interpret=interpret,
        cost_estimate=cost_estimate,
    )

    _lhs_contracting_axis, rhs_contracting_axis = dot_general_dims[0]

    rhs_contracting_axis = map(lambda x: x + 1, rhs_contracting_axis)

    out = call_gmm(
        group_metadata,
        group_offset,
        lhs,
        rhs,
        existing_out,
    )
    if existing_out is None and num_current_groups < num_total_groups:
        out = _zero_uninitialized_memory(
            out,
            start_group=group_offset[0],
            num_nonzero_groups=rhs.shape[0],
            group_metadata=group_metadata,
        )
    return out


@functools.partial(
    jax.jit,
    static_argnames=[
        "preferred_element_type",
        "tiling",
        "num_actual_groups",
        "interpret",
    ],
)
def transposed_grouped_matmul(
    lhs: jnp.ndarray,
    rhs: jnp.ndarray,
    group_sizes: jnp.ndarray,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128),
    group_offset: jnp.ndarray | None = None,
    num_actual_groups: int | None = None,
    existing_out: jnp.ndarray | None = None,
    interpret: bool = False,
) -> jnp.ndarray:
    """Transposed Grouped Matrix Multiplication: Compute grouped products with transposed access pattern.

    This function performs grouped matrix multiplication where different column slices of
    the left-hand side matrix are multiplied with different row slices of the right-hand
    side matrix, producing a separate output matrix for each group. Mathematically, for
    each group i:
        out[i, :, :] = lhs[:, start_i:end_i] @ rhs[start_i:end_i, :]
    where start_i and end_i are determined by cumulative group_sizes.

    This operation is particularly useful for:
    - Attention mechanisms where different heads process different feature slices
    - Expert routing in Mixture-of-Experts models
    - Block-sparse operations where groups represent different blocks

    The implementation uses Pallas to generate efficient TPU kernels that:
    - Process multiple groups while maintaining separate outputs
    - Handle empty groups by zeroing their outputs
    - Support incremental accumulation across tiles
    - Optimize for TPU's memory hierarchy and compute units

    Args:
        lhs: Left-hand side matrix of shape [k, m] where k is the output dimension
            and m is the total size across all groups.
        rhs: Right-hand side matrix of shape [m, n] where m matches lhs and n is
            the final output dimension.
        group_sizes: 1D array of shape [num_groups] with int32 dtype. Each element
            specifies the size of that group in the m dimension. Must sum to m.
        preferred_element_type: Output dtype. Defaults to float32. Internal
            accumulation uses float32 regardless, with final cast to this type.
        tiling: Tile sizes as (tm, tk, tn) tuple, or a callable that returns tile
            sizes based on problem dimensions. Standard TPU tiles are 128x128.
            The callable signature is (m, k, n) -> (tm, tk, tn) | None.
        group_offset: Starting group index for sharded execution. Defaults to 0.
            Enables distributing groups across multiple devices.
        num_actual_groups: Number of groups to actually compute starting from
            group_offset. Defaults to all remaining groups. Useful for sharding.
        existing_out: Optional pre-existing output tensor to accumulate into.
            Must have shape [num_actual_groups, k, n] and matching dtype.
            Enables incremental computation and gradient accumulation.
        interpret: Run kernel in interpret mode for debugging. Slower but provides
            better error messages and avoids compilation.

    Returns:
        3D output tensor of shape [num_actual_groups, k, n] where each slice
        out[i] contains the matrix product for group i.

    Algorithm Overview:
        1. Validate inputs and configure computation parameters
        2. Create group metadata with visit_empty_groups=True to ensure all outputs
           are properly initialized (even for empty groups)
        3. Define Pallas kernel that:
           - Maintains separate accumulator for each group
           - Masks inputs based on group boundaries
           - Handles group transitions by storing/resetting accumulators
           - Zeros output for empty groups
        4. Launch kernel with grid covering all tiles and groups
        5. Handle output accumulation if existing_out provided

    TPU Optimizations:
        - Tile operations aligned with TPU's 128x128 systolic arrays
        - Accumulation in VMEM (on-chip memory) to minimize HBM bandwidth
        - Prefetch scheduling to overlap compute and memory operations
        - Efficient masking using TPU's predicated execution
        - Group transitions handled without kernel restarts

    Key Differences from grouped_matmul:
        - Output is 3D with separate matrix per group (vs 2D concatenated)
        - Groups index into both lhs columns and rhs rows (vs only lhs rows)
        - Empty groups must be visited to zero their outputs
        - Accumulator management includes group transition logic

    Example:
        >>>
        >>> lhs = jnp.randn(64, 300)
        >>> rhs = jnp.randn(300, 32)
        >>> group_sizes = jnp.array([100, 150, 50], dtype=jnp.int32)
        >>> result = transposed_grouped_matmul(lhs, rhs, group_sizes)
        >>>
        >>>
        >>>

    Notes:
        - The m dimension must be divisible by tm for correctness
        - Empty groups produce zero matrices in the output
        - Partial tiles are handled through masking
        - Cost estimation guides XLA's scheduling decisions
        - The lhs matrix is internally transposed for efficient access patterns
    """
    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    else:
        group_offset = group_offset[None]
    lhs, group_sizes, input_dtype = _validate_args(lhs=lhs, rhs=rhs, group_sizes=group_sizes, expected_rhs_dims=2)

    k, m, n = (lhs.shape[0], lhs.shape[1], rhs.shape[1])
    num_groups = group_sizes.shape[0]
    num_actual_groups = num_actual_groups if num_actual_groups is not None else num_groups

    if callable(tiling):
        tiling = tiling(m, k, n)

    if tiling is None:
        raise ValueError(f"No tuned tiling found for (m, k, n) = ({m}, {k}, {n})")

    tm, tk, tn = tiling
    tiles_k, k_rem = _calculate_irregular_num_tiles(k, tk)
    del k_rem
    tiles_n, n_rem = _calculate_irregular_num_tiles(n, tn)
    del n_rem

    group_metadata, num_active_tiles = make_group_metadata(
        group_sizes=group_sizes,
        m=m,
        tm=tm,
        start_group=group_offset[0],
        num_nonzero_groups=num_actual_groups,
        visit_empty_groups=True,
    )

    def kernel(
        group_metadata,
        group_offset,
        lhs,
        rhs,
        existing_out,
        out,
        acc_scratch,
    ):
        grid_id = pl.program_id(2)
        group_offsets, group_ids, m_tile_ids = group_metadata
        del group_offsets, group_offset, m_tile_ids

        group = group_ids[grid_id]
        prev_grid_id = jnp.where(grid_id > 0, grid_id - 1, 0)
        prev_group = group_ids[prev_grid_id]

        group_has_changed = jnp.logical_or(grid_id == 0, prev_group != group)

        @pl.when(group_has_changed)
        def _zero_acc():
            acc_scratch[...] = jnp.zeros_like(acc_scratch)

        dont_skip = _get_group_size(grid_id=grid_id, group_metadata=group_metadata) > 0

        @pl.when(dont_skip)
        def _do():
            rhs_mask = _get_store_mask(
                grid_id=grid_id,
                group_metadata=group_metadata,
                tm=tm,
                tn=tn,
            )
            lhs_mask = _get_store_mask(
                grid_id=grid_id,
                group_metadata=group_metadata,
                tm=tm,
                tn=tk,
            )

            loaded_lhs = lhs[...]
            loaded_lhs = (
                lax.select(
                    lhs_mask[...],
                    loaded_lhs.astype(jnp.float32),
                    jnp.zeros_like(lhs, jnp.float32),
                )
                .astype(input_dtype)
                .swapaxes(0, 1)
            )
            loaded_rhs = rhs[...]
            loaded_rhs = lax.select(
                rhs_mask[...],
                loaded_rhs.astype(jnp.float32),
                jnp.zeros_like(rhs, jnp.float32),
            ).astype(input_dtype)

            acc_scratch[...] += lax.dot(
                loaded_lhs,
                loaded_rhs,
                preferred_element_type=jnp.float32,
            )

        is_end_of_grid = grid_id == (pl.num_programs(2) - 1)
        next_grid_id = jnp.where(is_end_of_grid, grid_id, grid_id + 1)
        next_group = group_ids[next_grid_id]

        group_is_changing = jnp.logical_or(is_end_of_grid, group != next_group)

        @pl.when(group_is_changing)
        def _store_accum():
            to_store = acc_scratch[...]
            if existing_out is not None:
                to_store += existing_out[...].astype(jnp.float32)
            out[...] = to_store.astype(preferred_element_type)

    def lhs_transform_indices(n_i, k_i, grid_id, group_metadata, group_offset):
        group_offsets, group_ids, m_tile_ids = group_metadata
        del n_i, group_offsets, group_ids, group_offset
        return m_tile_ids[grid_id], k_i

    def rhs_transform_indices(n_i, k_i, grid_id, group_metadata, group_offset):
        group_offsets, group_ids, m_tile_ids = group_metadata
        del k_i, group_offsets, group_ids, group_offset
        return m_tile_ids[grid_id], n_i

    def out_transform_indices(n_i, k_i, grid_id, group_metadata, group_offset):
        group_offsets, group_ids, m_tile_ids = group_metadata
        del group_offsets, m_tile_ids

        return group_ids[grid_id] - group_offset[0], k_i, n_i

    out_block_spec = pl.BlockSpec((None, tk, tn), out_transform_indices)
    if existing_out is None:
        in_out_block_spec: Any = None
        input_output_aliases = {}
    else:
        in_out_block_spec = out_block_spec
        input_output_aliases = {6: 0}

    lhs_block_spec = pl.BlockSpec((tm, tk), lhs_transform_indices)
    rhs_block_spec = pl.BlockSpec((tm, tn), rhs_transform_indices)

    lhs_bytes = lhs.size * lhs.itemsize
    rhs_bytes = rhs.size * rhs.itemsize
    out_bytewidth = jnp.dtype(preferred_element_type).itemsize
    out_bytes = (num_actual_groups * k * n) * out_bytewidth
    bytes_accessed = (lhs_bytes * tiles_n) + (rhs_bytes * tiles_k) + out_bytes
    flops = 2 * m * k * n
    cost_estimate = pl.CostEstimate(flops=flops, bytes_accessed=bytes_accessed, transcendentals=0)
    lhs = lhs.swapaxes(0, 1)

    call_gmm = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((num_actual_groups, k, n), preferred_element_type),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                lhs_block_spec,
                rhs_block_spec,
                in_out_block_spec,
            ],
            out_specs=out_block_spec,
            grid=(tiles_n, tiles_k, num_active_tiles),
            scratch_shapes=[pltpu.VMEM((tk, tn), jnp.float32)],
        ),
        input_output_aliases=input_output_aliases,
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary", "arbitrary")),
        interpret=interpret,
        cost_estimate=cost_estimate,
    )

    out = call_gmm(
        group_metadata,
        group_offset,
        lhs,
        rhs,
        existing_out,
    )
    return out
