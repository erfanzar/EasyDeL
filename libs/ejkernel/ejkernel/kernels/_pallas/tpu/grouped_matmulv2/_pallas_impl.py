# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
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


"""Pallas TPU kernel implementations for grouped matrix multiplication V2.

This module contains the low-level Pallas kernel implementations for both
grouped_matmul and transposed_grouped_matmul operations on TPU. These are
the V2 implementations featuring input buffer count configuration for
improved memory management on TPU.

The module provides:
    - ``_validate_args``: Input validation for grouped matmul operations
    - ``make_group_metadata``: Group-to-tile mapping metadata generation
    - ``_get_store_mask``: Per-tile store mask for group boundary handling
    - ``grouped_matmul``: Forward grouped matrix multiplication kernel
    - ``transposed_grouped_matmul``: Transposed grouped matmul for gradient computation

Key differences from V1:
    - Configurable ``input_buffer_count`` for memory/compute tradeoff
    - Uses ``buffered_pallas_call`` for improved buffer management
    - Simplified API without ``existing_out`` parameter

TPU Kernel Design:
    Both kernels use Pallas ``PrefetchScalarGridSpec`` with scalar prefetch
    for group metadata. The grid is organized as (tiles_n, tiles_m, tiles_k)
    for the forward pass and (tiles_n, tiles_k, tiles_m) for the transposed
    variant. VMEM scratch space is used for float32 accumulation.
"""

import functools
import json
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from ejkernel.callib import buffered_pallas_call


def _validate_args(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    *,
    expected_rhs_dims: int = 3,
) -> jax.Array:
    """Validate input arguments for grouped matrix multiplication.

    Checks that lhs and rhs have the expected number of dimensions and that
    group_sizes has a 32-bit integer dtype. Converts group_sizes to int32
    if it is uint32.

    Args:
        lhs: Left-hand side matrix, expected to be 2D with shape [m, k].
        rhs: Right-hand side tensor, expected to be ``expected_rhs_dims``-dimensional.
            Typically 3D [num_groups, k, n] for forward or 2D [m, n] for transposed.
        group_sizes: 1D array of group sizes. Must have int32 or uint32 dtype.
        expected_rhs_dims: Expected number of dimensions for rhs. Defaults to 3
            for the forward pass, should be 2 for the transposed variant.

    Returns:
        The group_sizes array cast to int32 dtype.

    Raises:
        ValueError: If lhs is not 2D, rhs does not match expected_rhs_dims,
            or group_sizes is not a 32-bit integer type.
    """
    if lhs.ndim != 2:
        raise ValueError(f"Expected 2-tensor for 'lhs' but got {lhs.ndim}-tensor.")

    if rhs.ndim != expected_rhs_dims:
        raise ValueError(f"Expected {expected_rhs_dims}-tensor for 'rhs' but got {rhs.ndim}-tensor.")

    if group_sizes.dtype not in (jnp.int32, jnp.uint32):
        raise ValueError(f"Expected 32-bit integer 'group_sizes' but got {group_sizes.dtype}.")
    return group_sizes.astype(jnp.int32)


GroupMetadata = tuple[jax.Array, jax.Array, jax.Array]


def make_group_metadata(
    *,
    group_sizes: jax.Array,
    m: int,
    tm: int,
    start_group: jax.Array,
    num_nonzero_groups: int,
    visit_empty_groups: bool,
) -> tuple[GroupMetadata, jax.Array]:
    """Create the metadata needed for grouped matmul computation.

    Args:
      group_sizes: A 1d, jax.Array with shape `[num_groups]` and `jnp.int32`
        dtype.
      m: The number of rows in lhs.
      tm: The m-dimension tile size being used.
      start_group: The group in group sizes to start computing from. This is
        particularly useful for when rhs num_groups is sharded.
      num_nonzero_groups: Number of groups in group sizes to compute on. Useful in
        combination with group_offset.
      visit_empty_groups: If True, do not squeeze tiles for empty groups out of
        the metadata. This is necessary for transposed_grouped_matmul, where we at least need to zero
        the output for each group.

    Returns:
      tuple of:
        group_offsets: A 1d, jax.Array with shape [num_groups+1] and jnp.int32
          dtype. group_offsets[i] indicates the row at which group [i] starts in
          the lhs matrix and group_offsets[i-1] = m.
        group_ids: A 1d, jax.Array with shape [m_tiles + num_groups] and
          jnp.int32 dtype. group_ids[i] indicates which group grid index 'i' will
          work on.
        m_tile_ids: A 1d, jax.Array with shape [m_tiles + num_groups] and
          jnp.int32. m_tile_ids[i] indicates which m-dimension tile grid index 'i'
          will work on.
      num_tiles: The number of m-dimension tiles to execute.
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

    if m % tm != 0:
        raise NotImplementedError(f"{m=} must be divisible by tile size ({tm}).")

    tiles_m = m // tm
    group_ids = jnp.repeat(
        jnp.arange(num_groups, dtype=jnp.int32),
        group_tiles,
        total_repeat_length=tiles_m + num_groups - 1,
    )

    partial_tile_mask = ((group_offsets[:-1] % tm) == 0) | (group_sizes == 0)

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
    active_group_mask = (iota <= end_group) & (iota >= start_group)
    group_tiles = jnp.where(active_group_mask, group_tiles, 0)
    num_tiles = group_tiles.sum()
    return (group_offsets, group_ids, m_tile_ids), num_tiles


def _get_store_mask(
    *,
    grid_id: jax.Array,
    group_metadata: GroupMetadata,
    tm: int,
    tn: int,
) -> jax.Array:
    """Generate a boolean mask for rows belonging to the current group in the current tile.

    This function creates a mask to ensure that only the rows that belong to the
    current group are written to the output matrix. This is critical for handling
    partial tiles where a tile may span multiple groups or where a group doesn't
    perfectly align with tile boundaries.

    Args:
        grid_id: The current grid index being processed.
        group_metadata: Tuple containing (group_offsets, group_ids, m_tile_ids)
            which define the group-to-tile mapping.
        tm: The m-dimension tile size (number of rows per tile).
        tn: The n-dimension tile size (number of columns per tile).

    Returns:
        A boolean array of shape [tm, tn] where True indicates that the
        corresponding element belongs to the current group and should be stored.
    """
    group_offsets, group_ids, m_tile_ids = group_metadata
    group_id = group_ids[grid_id]
    group_start = group_offsets[group_id]
    group_end = group_offsets[group_id + 1]
    m_id = m_tile_ids[grid_id] * tm
    iota = jax.lax.broadcasted_iota(jnp.int32, (tm, tn), 0) + m_id
    return (iota >= group_start) & (iota < group_end)


_TilingFn = Callable[[int, int, int], tuple[int, int, int] | None]


@functools.partial(
    jax.jit,
    static_argnames=[
        "preferred_element_type",
        "tiling",
        "transpose_rhs",
        "interpret",
        "input_buffer_count",
    ],
)
def grouped_matmul(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    preferred_element_type: jnp.dtype,
    tiling: tuple[int, int, int] | _TilingFn | None = (128, 128, 128),
    input_buffer_count: int = 2,
    group_offset: jax.Array | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
) -> jax.Array:
    """Compute grouped matrix multiplication: lhs[start_i:end_i, :] @ rhs[i] for each group i.

    Performs N separate matrix multiplications where each group's rows from lhs
    are multiplied with the corresponding group-specific weight matrix from rhs.
    The results are concatenated into a single output matrix.

    This V2 implementation features configurable input buffer counts for
    improved memory management on TPU via ``buffered_pallas_call``.

    Args:
        lhs: Left-hand side matrix of shape [m, k] where m is the total number
            of rows across all groups and k is the inner dimension.
        rhs: Right-hand side tensor of shape [num_groups, k, n] containing a
            separate matrix for each group. If transpose_rhs is True, expected
            shape is [num_groups, n, k].
        group_sizes: 1D array of shape [num_groups] with int32 dtype. Each element
            specifies the number of rows in lhs belonging to that group.
            Must sum to m (first dimension of lhs).
        preferred_element_type: Output dtype for the result matrix.
        tiling: Tile sizes as (tm, tk, tn) tuple, or a callable returning tile
            sizes given (m, k, n). None raises ValueError.
        input_buffer_count: Number of input buffers for the Pallas call. Higher
            values use more memory but can improve throughput. Defaults to 2.
        group_offset: Starting group index for sharded execution. Useful when
            distributing groups across multiple devices. Defaults to 0.
        transpose_rhs: If True, transposes the last two dimensions of rhs
            during multiplication. Defaults to False.
        interpret: Run kernel in interpret mode for debugging. Slower but provides
            better error messages. Defaults to False.

    Returns:
        Output matrix of shape [m, n] containing the concatenated results of all
        group matrix multiplications, with dtype matching preferred_element_type.

    Raises:
        ValueError: If lhs is not 2D, rhs is not 3D, group_sizes is not int32,
            group_offset is not scalar, or no tiling is available.
    """
    group_sizes = _validate_args(lhs, rhs, group_sizes)

    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    else:
        if group_offset.shape:
            raise ValueError(f"group_offset must be a ()-shaped array. Got: {group_offset.shape}.")
        group_offset = group_offset[None]

    m, k = lhs.shape
    n = rhs.shape[1 if transpose_rhs else 2]

    if callable(tiling):
        tiling = tiling(m, k, n)

    if tiling is None:
        raise ValueError(f"No tuned tiling found for (m, k, n) = ({m}, {k}, {n})")

    tm, tk, tn = tiling
    tiles_k = pl.cdiv(k, tk)
    tiles_n = pl.cdiv(n, tn)

    group_metadata, num_active_tiles = make_group_metadata(
        group_sizes=group_sizes,
        m=m,
        tm=tm,
        start_group=group_offset[0],
        num_nonzero_groups=rhs.shape[0],
        visit_empty_groups=False,
    )
    group_offsets, group_ids, _ = group_metadata

    def kernel(group_metadata, _, lhs_ref, rhs_ref, out_ref, acc_scratch):
        if transpose_rhs:
            dimension_numbers = (((1,), (1,)), ((), ()))
        else:
            dimension_numbers = (((1,), (0,)), ((), ()))

        def dot_general(x, y, preferred_element_type):
            return jax.lax.dot_general(
                x,
                y,
                dimension_numbers=dimension_numbers,
                preferred_element_type=preferred_element_type,
            )

        grid_id = pl.program_id(1)
        k_i = pl.program_id(2)

        @pl.when(k_i == 0)
        def _zero_acc():
            acc_scratch[...] = jnp.zeros_like(acc_scratch)

        def accum(is_last_k_tile):
            with jax.named_scope(f"accum-last_k_tile={is_last_k_tile}"):
                lhs = jax.tree.map(lambda x: x[...], lhs_ref)
                rhs = jax.tree.map(lambda x: x[...], rhs_ref)
                scales = []

                if is_last_k_tile and (k_rem := k % tk) != 0:

                    def iota(x, d):
                        return lax.broadcasted_iota(jnp.int32, x.shape, d)

                    lhs = jnp.where(iota(lhs, 1) < k_rem, lhs, 0)
                    rhs = jnp.where(iota(rhs, 1 if transpose_rhs else 0) < k_rem, rhs, 0)

                def is_int(x):
                    return jnp.issubdtype(x.dtype, jnp.integer)

                acc_dtype = jnp.int32 if is_int(lhs) and is_int(rhs) else jnp.float32
                out = dot_general(lhs, rhs, acc_dtype)

                for scale, axis in scales:
                    out *= pltpu.repeat(scale, out.shape[axis] // scale.shape[axis], axis)

                acc_scratch[...] += out.astype(acc_scratch.dtype)

                if is_last_k_tile:
                    mask = _get_store_mask(grid_id=grid_id, group_metadata=group_metadata, tm=tm, tn=tn)
                    acc = acc_scratch[...]
                    acc = jax.lax.select(mask, acc, out_ref[...].astype(acc.dtype))
                    out_ref[...] = acc.astype(preferred_element_type)

        lax.cond(k_i == tiles_k - 1, lambda: accum(True), lambda: accum(False))

    def lhs_index_map(n_i, grid_id, k_i, group_metadata, group_offset):
        del n_i, group_offset

        _, _, m_tile_ids = group_metadata
        return m_tile_ids[grid_id], k_i

    def rhs_index_map(n_i, grid_id, k_i, group_metadata, group_offset):
        _, group_ids, _ = group_metadata
        if transpose_rhs:
            k_i, n_i = n_i, k_i

        return group_ids[grid_id] - group_offset[0], k_i, n_i

    def out_index_map(n_i, grid_id, k_i, group_metadata, group_offset):
        del k_i, group_offset

        _, _, m_tile_ids = group_metadata
        return m_tile_ids[grid_id], n_i

    lhs_block_spec = pl.BlockSpec((tm, tk), lhs_index_map)
    if transpose_rhs:
        rhs_block_spec = pl.BlockSpec((None, tn, tk), rhs_index_map)
    else:
        rhs_block_spec = pl.BlockSpec((None, tk, tn), rhs_index_map)
    out_block_spec = pl.BlockSpec((tm, tn), out_index_map)

    lhs_bytes = jax.tree.reduce(lambda acc, x: acc + x.size * x.itemsize, lhs, 0)

    rhs_bytes = k * n * rhs.itemsize

    out_bytes = (m * n) * jnp.dtype(preferred_element_type).itemsize
    bytes_accessed = lhs_bytes * tiles_n + rhs_bytes * group_ids.size + out_bytes
    cost_estimate = pl.CostEstimate(flops=2 * m * k * n, bytes_accessed=bytes_accessed, transcendentals=0)
    kernel_name = f"gmm_{tm}x{tk}x{tn}"
    if transpose_rhs:
        kernel_name += "_transpose_rhs"
    metadata = dict(
        prefer_element_type=jnp.dtype(preferred_element_type).name,
        tiling=dict(tile_m=tm, tile_k=tk, tile_n=tn),
        transpose_rhs=transpose_rhs,
    )
    call_gmm = buffered_pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), preferred_element_type),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[lhs_block_spec, rhs_block_spec],
            out_specs=out_block_spec,
            grid=(tiles_n, num_active_tiles, tiles_k),
            scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary", "arbitrary")),
        interpret=interpret,
        cost_estimate=cost_estimate,
        name=kernel_name,
        metadata=dict(xprof_metadata=json.dumps(metadata)),
        input_buffer_count=(input_buffer_count, 2),
    )

    with jax.named_scope(kernel_name):
        out = call_gmm(group_metadata, group_offset, lhs, rhs)

    if rhs.shape[0] < group_sizes.shape[0]:
        group_start = group_offsets[group_offset[0]]
        group_end = group_offsets[group_offset[0] + rhs.shape[0]]
        row_idxs = jnp.arange(out.shape[0], dtype=jnp.int32)
        valid_mask = (row_idxs >= group_start) & (row_idxs < group_end)
        out = jnp.where(valid_mask[:, None], out, 0)
    return out


@functools.partial(
    jax.jit,
    static_argnames=[
        "preferred_element_type",
        "tiling",
        "num_actual_groups",
        "interpret",
        "input_buffer_count",
    ],
)
def transposed_grouped_matmul(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    preferred_element_type: jnp.dtype,
    tiling: tuple[int, int, int] | _TilingFn | None = (128, 128, 128),
    input_buffer_count: int = 2,
    group_offset: jax.Array | None = None,
    num_actual_groups: int | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Compute transposed grouped matrix multiplication for gradient computation.

    For each group i, computes lhs[:, start_i:end_i] @ rhs[start_i:end_i, :]
    where start_i and end_i are determined by group_sizes. This is the transposed
    variant used primarily for computing gradients with respect to the right-hand
    side weight matrix in the backward pass of grouped_matmul.

    The kernel visits empty groups (setting their output to zero) and accumulates
    results across the m-dimension tiles for each group.

    Args:
        lhs: Left-hand side matrix of shape [k, m] (transposed from the forward
            pass lhs). Internally transposed to [m, k] for kernel execution.
        rhs: Right-hand side matrix of shape [m, n] (the gradient from the
            forward pass output).
        group_sizes: 1D array of shape [num_groups] with int32 dtype. Each element
            specifies the number of rows belonging to that group.
        preferred_element_type: Output dtype for the result tensor.
        tiling: Tile sizes as (tm, tk, tn) tuple, or a callable returning tile
            sizes given (m, k, n). None raises ValueError.
        input_buffer_count: Number of input buffers for the Pallas call. Defaults to 2.
        group_offset: Starting group index for sharded execution. Defaults to 0.
        num_actual_groups: Number of groups to actually compute, starting from
            group_offset. Defaults to num_groups if None. Used when groups are
            distributed across multiple devices.
        interpret: Run kernel in interpret mode for debugging. Defaults to False.

    Returns:
        3D tensor of shape [num_actual_groups, k, n] where each slice [i, :, :]
        contains the result of the transposed matmul for group i.

    Raises:
        ValueError: If lhs is not 2D, rhs is not 2D, group_sizes is not int32,
            or no tiling is available.
    """
    group_sizes = _validate_args(lhs, rhs, group_sizes, expected_rhs_dims=2)

    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    else:
        group_offset = group_offset[None]

    k, m = lhs.shape
    n = rhs.shape[1]

    lhs = jax.tree.map(lambda x: x.mT, lhs)

    num_groups = group_sizes.shape[0]
    num_actual_groups = num_actual_groups if num_actual_groups is not None else num_groups

    if callable(tiling):
        tiling = tiling(m, k, n)

    if tiling is None:
        raise ValueError(f"No tuned tiling found for (m, k, n) = ({m}, {k}, {n})")

    tm, tk, tn = tiling
    tiles_k = pl.cdiv(k, tk)
    tiles_n = pl.cdiv(n, tn)

    group_metadata, num_active_tiles = make_group_metadata(
        group_sizes=group_sizes,
        m=m,
        tm=tm,
        start_group=group_offset[0],
        num_nonzero_groups=num_actual_groups,
        visit_empty_groups=True,
    )

    def kernel(group_metadata, _, lhs_ref, rhs_ref, out_ref, acc_scratch):
        grid_id = pl.program_id(2)
        group_offsets, group_ids, _ = group_metadata
        group = group_ids[grid_id]
        prev_group = group_ids[jnp.where(grid_id > 0, grid_id - 1, 0)]

        @pl.when((grid_id == 0) | (group != prev_group))
        def _zero_acc():
            acc_scratch[...] = jnp.zeros_like(acc_scratch)

        group_size = group_offsets[group + 1] - group_offsets[group]

        @pl.when(group_size > 0)
        def _do():
            def dot(x, y, preferred_element_type):
                return lax.dot(x, y, preferred_element_type=preferred_element_type)

            lhs = jax.tree.map(lambda x: x[...], lhs_ref)
            rhs = jax.tree.map(lambda x: x[...], rhs_ref)
            scales = []

            kwargs = dict(grid_id=grid_id, group_metadata=group_metadata, tm=tm)
            lhs = jnp.where(_get_store_mask(**kwargs, tn=tk), lhs, 0)
            rhs = jnp.where(_get_store_mask(**kwargs, tn=tn), rhs, 0)

            def is_int(x):
                return jnp.issubdtype(x.dtype, jnp.integer)

            acc_dtype = jnp.int32 if is_int(lhs) and is_int(rhs) else jnp.float32
            out = dot(lhs.T, rhs, acc_dtype)

            for scale, axis in scales:
                out *= pltpu.repeat(scale, out.shape[axis] // scale.shape[axis], axis)

            acc_scratch[...] += out.astype(acc_scratch.dtype)

        is_end_of_grid = grid_id == (pl.num_programs(2) - 1)
        next_group = group_ids[jnp.where(is_end_of_grid, grid_id, grid_id + 1)]

        @pl.when(is_end_of_grid | (group != next_group))
        def _store_accum():
            out_ref[...] = acc_scratch[...].astype(preferred_element_type)

    def lhs_index_map(n_i, k_i, grid_id, group_metadata, group_offset):
        del n_i, group_offset

        _, _, m_tile_ids = group_metadata
        return m_tile_ids[grid_id], k_i

    def rhs_index_map(n_i, k_i, grid_id, group_metadata, group_offset):
        del k_i, group_offset

        _, _, m_tile_ids = group_metadata
        return m_tile_ids[grid_id], n_i

    def out_index_map(n_i, k_i, grid_id, group_metadata, group_offset):
        _, group_ids, _ = group_metadata

        return group_ids[grid_id] - group_offset[0], k_i, n_i

    lhs_block_spec = pl.BlockSpec((tm, tk), lhs_index_map)
    rhs_block_spec = pl.BlockSpec((tm, tn), rhs_index_map)
    out_block_spec = pl.BlockSpec((None, tk, tn), out_index_map)

    lhs_bytes = jax.tree.reduce(lambda acc, x: acc + x.size * x.itemsize, lhs, 0)
    rhs_bytes = jax.tree.reduce(lambda acc, x: acc + x.size * x.itemsize, rhs, 0)
    out_bytes = (num_actual_groups * k * n) * jnp.dtype(preferred_element_type).itemsize
    bytes_accessed = (lhs_bytes * tiles_n) + (rhs_bytes * tiles_k) + out_bytes
    cost_estimate = pl.CostEstimate(flops=2 * m * k * n, bytes_accessed=bytes_accessed, transcendentals=0)

    kernel_name = f"tgmm_{tm}x{tk}x{tn}"
    metadata = dict(
        tiling=dict(tile_m=tm, tile_k=tk, tile_n=tn),
        prefer_element_type=jnp.dtype(preferred_element_type).name,
        num_actual_groups=num_actual_groups,
    )
    call_gmm = buffered_pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((num_actual_groups, k, n), preferred_element_type),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[lhs_block_spec, rhs_block_spec],
            out_specs=out_block_spec,
            grid=(tiles_n, tiles_k, num_active_tiles),
            scratch_shapes=[pltpu.VMEM((tk, tn), jnp.float32)],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary", "arbitrary")),
        interpret=interpret,
        cost_estimate=cost_estimate,
        name=kernel_name,
        metadata=dict(xprof_metadata=json.dumps(metadata)),
        input_buffer_count=(input_buffer_count, input_buffer_count),
    )

    with jax.named_scope(kernel_name):
        return call_gmm(group_metadata, group_offset, lhs, rhs)
