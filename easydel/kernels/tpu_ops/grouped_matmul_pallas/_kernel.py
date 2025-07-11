# Copyright 2024 The JAX Authors.
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


import re
import typing
from functools import partial

import chex
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from easydel.utils.compiling_utils import ejit

GroupMetadata = typing.TypeVar("GroupMetadata", bound=tuple[chex.Array, chex.Array, chex.Array])
TilinFn = typing.Callable[[int, int, int], tuple[int, int, int] | None]


class ValidationError(Exception): ...


def select_input_dtype(lhs: jnp.ndarray, rhs: jnp.ndarray) -> jnp.dtype:
    """Selects the input dtype based on the device and input dtypes."""
    if version := re.compile(r"TPU v(\d+)").match(jax.local_devices()[0].device_kind):
        tpu_generation = int(version[1])
    else:
        raise NotImplementedError("only TPU devices are supported")
    supports_bfloat16_matmul = "TPU" not in jax.local_devices()[0].device_kind or tpu_generation >= 4

    if supports_bfloat16_matmul and lhs.dtype == jnp.bfloat16 and rhs.dtype == jnp.bfloat16:
        return jnp.bfloat16
    else:
        return jnp.float32


def _identity(x: chex.Array) -> chex.Array:
    """Identity function."""
    return x


def _validate_args(
    *,
    lhs: chex.Array,
    rhs: chex.Array,
    group_sizes: chex.Array,
    expected_rhs_dims: int = 3,
) -> tuple[chex.Array, chex.Array, jnp.dtype]:
    """Validates the arguments for the gmm function."""
    if lhs.ndim != 2:
        raise ValidationError(f"Expected 2-tensor for 'lhs' but got {lhs.ndim}-tensor.")
    assert lhs.dtype in (jnp.bfloat16, jnp.float32), f"Expected bfloat16 or float32 for 'lhs' but got {lhs.dtype}."

    if rhs.ndim != expected_rhs_dims:
        raise ValidationError(f"Expected {expected_rhs_dims}-tensor for 'rhs' but got {rhs.ndim}-tensor.")
    assert rhs.dtype in (jnp.bfloat16, jnp.float32), f"Expected bfloat16 or float32 for 'rhs' but got {rhs.dtype}."

    if group_sizes.dtype != jnp.int32:
        raise ValidationError(f"Expected 32-bit integer 'group_sizes' but got {group_sizes.dtype}.")

    return lhs, group_sizes, select_input_dtype(lhs, rhs)


def _calculate_num_tiles(x: int, tx: int) -> int:
    tiles, rem = divmod(x, tx)
    if rem:
        raise ValueError(f"{x} must be divisible by x-dimension tile size ({tx}).")
    return tiles


def _calculate_irregular_num_tiles(x: int, tx: int) -> tuple[int, int]:
    tiles, rem = divmod(x, tx)
    if rem:
        tiles += 1
    return tiles, rem


def make_group_metadata(
    *,
    group_sizes: chex.Array,
    m: int,
    tm: int,
    start_group: chex.Array,
    num_nonzero_groups: int,
    visit_empty_groups: bool = True,
) -> GroupMetadata:
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
    mr = tiles_m + num_groups - 1
    group_ids = jnp.repeat(jnp.arange(num_groups, dtype=jnp.int32), group_tiles, total_repeat_length=mr)

    partial_tile_mask = jnp.logical_or((group_offsets[:-1] % tm) == 0, group_sizes == 0)

    if visit_empty_groups:
        partial_tile_mask = jnp.where(group_sizes == 0, 0, partial_tile_mask)

    partial_tile_ids = jnp.where(partial_tile_mask, tiles_m, group_offsets[:-1] // tm)

    tile_visits = jnp.histogram(partial_tile_ids, bins=tiles_m, range=(0, tiles_m - 1))[0] + 1
    mr = tiles_m + num_groups - 1
    m_tile_ids = jnp.repeat(jnp.arange(tiles_m, dtype=jnp.int32), tile_visits.astype(jnp.int32), total_repeat_length=mr)

    first_tile_in_shard = (group_ids < start_group).sum()
    group_ids = jnp.roll(group_ids, shift=-first_tile_in_shard, axis=0)
    m_tile_ids = jnp.roll(m_tile_ids, shift=-first_tile_in_shard, axis=0)
    iota = jnp.arange(num_groups, dtype=jnp.int32)
    active_group_mask = jnp.logical_and(iota <= end_group, iota >= start_group)
    group_tiles = jnp.where(active_group_mask, group_tiles, 0)
    num_tiles = group_tiles.sum()
    return (group_offsets, group_ids, m_tile_ids), num_tiles


def _get_group_size(*, grid_id: chex.Array, group_metadata: GroupMetadata) -> chex.Array:
    """Calculate the number of rows in the current group."""
    group_offsets, group_ids = group_metadata[:2]
    group_id = group_ids[grid_id]
    group_start = group_offsets[group_id]
    group_end = group_offsets[group_id + 1]
    return group_end - group_start


def _get_store_mask(*, grid_id: chex.Array, group_metadata: GroupMetadata, tm: int, tn: int) -> chex.Array:
    """Mask for rows that belong to the current group in the current tile."""
    group_offsets, group_ids, m_tile_ids = group_metadata[:3]
    group_id = group_ids[grid_id]
    group_start = group_offsets[group_id]
    group_end = group_offsets[group_id + 1]
    m_id = m_tile_ids[grid_id] * tm
    iota = jax.lax.broadcasted_iota(jnp.int32, (tm, tn), 0) + m_id
    return jnp.logical_and(iota >= group_start, iota < group_end)


def _zero_uninitialized_memory(
    out: chex.Array,
    *,
    start_group: chex.Array,
    num_nonzero_groups: int,
    group_metadata: GroupMetadata,
) -> chex.Array:
    """Zero out uninitialized memory from output."""
    group_offsets = group_metadata[0]
    group_start = group_offsets[start_group]
    group_end = group_offsets[start_group + num_nonzero_groups]
    valid_mask = jax.lax.broadcasted_iota(jnp.int32, (out.shape[0],), 0)
    valid_mask = (valid_mask >= group_start) & (valid_mask < group_end)
    return jnp.where(valid_mask[:, None], out, 0)


@ejit(static_argnames=("preferred_element_type", "tiling", "transpose_rhs", "interpret"))
def gmm(
    lhs: chex.Array,
    rhs: chex.Array,
    group_sizes: chex.Array,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int] | TilinFn | None = (128, 128, 128),
    group_offset: chex.Array | None = None,
    existing_out: chex.Array | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
) -> chex.Array:
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

    group_metadata, num_active_tiles = make_group_metadata(
        group_sizes=group_sizes,
        m=m,
        tm=tm,
        start_group=group_offset[0],
        num_nonzero_groups=rhs.shape[0],
        visit_empty_groups=False,
    )

    def kernel(group_metadata, group_offset, lhs, rhs, existing_out, out, acc_scratch):
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

        def mask_k_rem(x, *, dim):
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
                mask_k_rem_lhs = _identity
                mask_k_rem_rhs = _identity

            if transpose_rhs:
                dot_general_dims = (((1,), (1,)), ((), ()))
            else:
                dot_general_dims = (((1,), (0,)), ((), ()))

            loaded_lhs = lhs[...]
            loaded_rhs = rhs[...]
            acc_scratch[...] += lax.dot_general(
                mask_k_rem_lhs(loaded_lhs).astype(input_dtype),
                mask_k_rem_rhs(loaded_rhs).astype(input_dtype),
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
        in_out_block_spec = None
        input_output_aliases = {}
    else:
        in_out_block_spec = out_block_spec
        input_output_aliases = {6: 0}

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
    call_gmm = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((m, n), preferred_element_type),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[lhs_block_spec, rhs_block_spec, in_out_block_spec],
            out_specs=out_block_spec,
            grid=(tiles_n, num_active_tiles, tiles_k),
            scratch_shapes=[pltpu.VMEM((tm, tn), jnp.float32)],
        ),
        input_output_aliases=input_output_aliases,
        compiler_params=pltpu.TPUCompilerParams(dimension_semantics=("parallel", "arbitrary", "arbitrary")),
        interpret=interpret,
        cost_estimate=cost_estimate,
    )

    out = call_gmm(group_metadata, group_offset, lhs, rhs, existing_out)
    if existing_out is None and num_current_groups < num_total_groups:
        out = _zero_uninitialized_memory(
            out,
            start_group=group_offset[0],
            num_nonzero_groups=rhs.shape[0],
            group_metadata=group_metadata,
        )
    return out


@ejit(static_argnames=("preferred_element_type", "tiling", "num_actual_groups", "interpret"))
def tgmm(
    lhs: chex.Array,
    rhs: chex.Array,
    group_sizes: chex.Array,
    preferred_element_type: jnp.dtype = jnp.float32,
    tiling: tuple[int, int, int] | TilinFn | None = (128, 128, 128),
    group_offset: chex.Array | None = None,
    num_actual_groups: int | None = None,
    existing_out: chex.Array | None = None,
    interpret: bool = False,
) -> chex.Array:
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
        raise ValidationError(f"No tuned tiling found for (m, k, n) = ({m}, {k}, {n})")

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

    def kernel(group_metadata, group_offset, lhs, rhs, existing_out, out, acc_scratch):
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
            rhs_mask = _get_store_mask(grid_id=grid_id, group_metadata=group_metadata, tm=tm, tn=tn)
            lhs_mask = _get_store_mask(grid_id=grid_id, group_metadata=group_metadata, tm=tm, tn=tk)

            loaded_lhs = lhs[...]
            loaded_rhs = rhs[...]
            loaded_lhs = lax.select(
                lhs_mask[...],
                loaded_lhs.astype(jnp.float32),
                jnp.zeros_like(lhs, jnp.float32),
            ).swapaxes(0, 1)
            loaded_rhs = lax.select(
                rhs_mask[...],
                loaded_rhs.astype(jnp.float32),
                jnp.zeros_like(rhs, jnp.float32),
            )

            acc_scratch[...] += lax.dot(
                loaded_lhs.astype(input_dtype),
                loaded_rhs.astype(input_dtype),
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
        in_out_block_spec = None
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
            in_specs=[lhs_block_spec, rhs_block_spec, in_out_block_spec],
            out_specs=out_block_spec,
            grid=(tiles_n, tiles_k, num_active_tiles),
            scratch_shapes=[pltpu.VMEM((tk, tn), jnp.float32)],
        ),
        input_output_aliases=input_output_aliases,
        compiler_params=pltpu.TPUCompilerParams(dimension_semantics=("parallel", "arbitrary", "arbitrary")),
        interpret=interpret,
        cost_estimate=cost_estimate,
    )

    out = call_gmm(group_metadata, group_offset, lhs, rhs, existing_out)
    return out
