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

"""TPU grouped matmul v3 adapted from upstream TPU inference gmm_v2.

This kernel keeps the upstream metadata-driven gm tiling and emit_pipeline
structure, but is vendored into ejkernel under the grouped_matmulv3 family.
The public interface wraps this file into the same grouped-matmul signature
we use for v1/v2.
"""

from __future__ import annotations

import dataclasses
import functools
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def swigluoai(
    gate: jax.Array,
    up: jax.Array,
    *,
    alpha: float = 1.702,
    limit: float = 7.0,
) -> jax.Array:
    """Activation used in some models such as GPT-OSS."""
    gate = jnp.clip(gate, a_max=limit)
    up = jnp.clip(up, min=-limit, max=limit)
    glu = gate * jax.nn.sigmoid(alpha * gate)
    return (up + 1.0) * glu


def apply_act_fn(acc: jax.Array, fuse_act: str | None):
    """Apply an optional fused activation to the accumulator."""
    if fuse_act is None:
        return acc

    acc_gate, acc_up = jnp.split(acc, 2, -1)
    match fuse_act:
        case "silu":
            return jax.nn.silu(acc_gate) * acc_up
        case "gelu":
            return jax.nn.gelu(acc_gate) * acc_up
        case "swigluoai":
            return swigluoai(acc_gate, acc_up)
        case _:
            raise NotImplementedError(f"Unsupported activation function: {fuse_act}")


def align_to(x, a):
    """Round ``x`` up to the nearest multiple of ``a``."""
    return pl.cdiv(x, a) * a


class RhsRef(ABC):
    """Abstract base for RHS tile references inside the Pallas kernel.

    Concrete subclasses (``WeightsRef``, ``FusedWeightsRef``) expose a
    uniform interface that the inner kernel uses regardless of whether the
    weight is plain, scaled, biased, or fused with a gated activation.
    """

    @abstractmethod
    def get_weight(self) -> jax.Array:
        """Return the raw weight tile (possibly bitcast from a packed dtype)."""
        ...

    @abstractmethod
    def get_scale(self) -> jax.Array:
        """Return the per-block quantisation scale tile."""
        ...

    @abstractmethod
    def get_bias(self) -> jax.Array:
        """Return the per-group bias tile."""
        ...


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class WeightsRef(RhsRef):
    """Concrete RHS tile reference for a single (non-fused) weight.

    Registered as a JAX pytree node so it can be passed through
    ``pallas_call`` ``in_specs`` / ``out_specs``.

    Attributes:
        weight: Weight tile reference (a Pallas ``BlockRef`` or plain array).
        scale: Optional quantisation scale tile reference.
        bias: Optional additive bias tile reference.
    """

    weight: Any
    scale: Any | None
    bias: Any | None

    def get_weight(self) -> jax.Array:
        """Load and return the weight tile from VMEM."""
        return self.weight[...]

    def get_scale(self) -> jax.Array:
        """Load and return the scale tile; asserts scale is not None."""
        assert self.scale is not None
        return self.scale[...]

    def get_bias(self) -> jax.Array:
        """Load and return the bias tile; asserts bias is not None."""
        assert self.bias is not None
        return self.bias[...]


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class FusedWeightsRef(RhsRef):
    """RHS tile reference for fused gate+up projections (e.g. SwiGLU / SiLU).

    Concatenates the ``gate`` and ``up`` tiles along the last axis so that
    ``inner_kernel`` can apply the activation across the combined slice.

    Attributes:
        gate: ``WeightsRef`` for the gate projection.
        up: ``WeightsRef`` for the up projection.
    """

    gate: WeightsRef
    up: WeightsRef

    def get_weight(self) -> jax.Array:
        """Concatenate gate and up weight tiles along axis -1."""
        w_gate = self.gate.get_weight()
        w_up = self.up.get_weight()
        return jnp.concatenate([w_gate, w_up], axis=-1)

    def get_scale(self) -> jax.Array:
        """Concatenate gate and up scale tiles along axis -1."""
        s_gate = self.gate.get_scale()
        s_up = self.up.get_scale()
        return jnp.concatenate([s_gate, s_up], axis=-1)

    def get_bias(self) -> jax.Array:
        """Concatenate gate and up bias tiles along axis -1."""
        b_gate = self.gate.get_bias()
        b_up = self.up.get_bias()
        return jnp.concatenate([b_gate, b_up], axis=-1)


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class MetadataRef:
    """SMEM scratch holding the per-grid-step group and row-offset mappings.

    Built at kernel entry by ``fill_metadata`` and read by ``inner_kernel``
    and ``IndexMaps`` to navigate the ragged group structure.

    Attributes:
        gm_id_to_group_id: ``SMEM[(max_num_gm,), int32]`` mapping each
            grid step (gm_id) to the owning group index within the current shard.
        gm_id_to_m_offset: ``SMEM[(max_num_gm + 1,), int32]`` cumulative
            row offsets; ``gm_id_to_m_offset[gm_id]`` is the first row of
            that step, ``gm_id_to_m_offset[gm_id + 1]`` the exclusive end.
    """

    gm_id_to_group_id: jax.Array
    gm_id_to_m_offset: jax.Array


@dataclasses.dataclass(frozen=True)
class TileSizes:
    """Static tile dimensions for the v3 grouped matmul kernel.

    Attributes:
        tile_m: Tile size along the M (token / row) dimension.
        tile_k: Tile size along the K (contraction) dimension.
        tile_n: Tile size along the N (output feature) dimension.
    """

    tile_m: int
    tile_k: int
    tile_n: int


@dataclasses.dataclass(frozen=True)
class Dimensions:
    """Problem dimensions for the v3 grouped matmul kernel.

    Attributes:
        size_m: Total number of token rows across all groups.
        size_k: Contraction dimension (input feature width).
        size_n: Output dimension (output feature width; doubled when
            ``fuse_act`` is not None and the combined gate+up weight is passed).
        size_group: Number of groups processed by this shard (``rhs.shape[0]``).
        size_lhs_group: Total number of groups in ``group_sizes``
            (may be larger than ``size_group`` for sharded runs).
        size_lhs_sublane: TPU sublane tiling for the LHS dtype, capped to
            ``size_m``.  Used to align row slices to HBM DMA boundaries.
    """

    size_m: int
    size_k: int
    size_n: int
    size_group: int
    size_lhs_group: int
    size_lhs_sublane: int


@dataclasses.dataclass(frozen=True)
class InputConfigs:
    """Per-operand (LHS or RHS) quantisation and dtype configuration.

    Attributes:
        quant_dtype: The quantised storage dtype (e.g. ``jnp.float8_e4m3fn``
            for FP8 LHS quantisation, or ``rhs.dtype`` for quantised weights).
            None for unquantised operands.
        quant_block_size: Number of elements per quantisation block along K.
            None for unquantised operands.
        dtype: Storage dtype of the operand in HBM.
        has_bias: Whether this operand has an additive per-group bias tensor.
        has_scale: Whether this operand has a per-block scale tensor.
    """

    quant_dtype: jnp.dtype | None
    quant_block_size: int | None
    dtype: jnp.dtype
    has_bias: bool = False
    has_scale: bool = False

    @property
    def should_bitcast(self) -> bool:
        """Return True when the dtype requires a bitcast (sub-byte elements)."""
        bits = jax.dtypes.itemsize_bits(self.dtype)
        return bits < 8


@dataclasses.dataclass(frozen=True)
class GmmConfigs:
    """Full configuration bundle for one grouped matmul v3 kernel launch.

    Attributes:
        tiles: Static tile dimensions ``(tile_m, tile_k, tile_n)``.
        dims: Problem dimensions ``(size_m, size_k, size_n, ...)``.
        lhs_cfgs: LHS quantisation and dtype settings.
        rhs_cfgs: RHS quantisation and dtype settings.
        out_dtype: Dtype of the output tensor written to HBM.
        acc_dtype: Dtype of the VMEM accumulator (float32 or bfloat16).
        zero_init: If True the kernel zeroes output rows outside the active
            compute range via async DMA before beginning the pipeline.
        fuse_act: Optional fused activation identifier: ``"silu"``,
            ``"gelu"``, or ``"swigluoai"``.  None for no activation.
    """

    tiles: TileSizes
    dims: Dimensions
    lhs_cfgs: InputConfigs
    rhs_cfgs: InputConfigs
    out_dtype: jnp.dtype
    acc_dtype: jnp.dtype
    zero_init: bool
    fuse_act: str | None

    @property
    def num_quant_blocks_per_tile_k(self) -> int:
        """Number of quantisation blocks per tile along the K dimension."""
        return pl.cdiv(self.tiles.tile_k, self.rhs_cfgs.quant_block_size)

    @property
    def out_size_n(self) -> int:
        """Actual output N dimension (halved when a fused activation is active)."""
        if self.fuse_act is None:
            return self.dims.size_n
        return self.dims.size_n // 2


TileFn = Callable[[Dimensions, InputConfigs, InputConfigs, int, str | None], TileSizes]


class IndexMaps:
    """Pallas ``BlockSpec`` index-map callables for the v3 GMM kernel.

    Each method returns a tuple of array indices that selects the correct
    HBM tile for a given ``(n_id, gm_id, k_id)`` grid point.  The mapping
    is computed from ``MetadataRef`` which is populated at kernel entry by
    ``fill_metadata``.

    Args:
        metadata_ref: SMEM metadata populated by ``fill_metadata``.
        cfgs: Full kernel configuration bundle.
    """

    def __init__(self, metadata_ref: MetadataRef, cfgs: GmmConfigs):
        self.metadata_ref = metadata_ref
        self.cfgs = cfgs

    def lhs_index_map(self, _: jax.Array, gm_id: jax.Array, k_id: jax.Array):
        """Return the LHS ``BoundedSlice`` tile index for grid step ``gm_id``.

        Returns a ``(pl.ds(row_start, row_size), 0, k_id)`` tuple selecting
        the rows belonging to the current group-step, aligned to
        ``size_lhs_sublane`` boundaries.
        """
        m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
        m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

        row_start = m_start // self.cfgs.dims.size_lhs_sublane
        row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
        row_size = row_end - row_start
        return (pl.ds(row_start, row_size), 0, k_id)

    def rhs_weight_index_map(self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array):
        """Return the RHS weight tile index ``(group_id, k_id, n_id)``."""
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        return (group_id, k_id, n_id)

    def rhs_bias_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array):
        """Return the RHS bias tile index ``(group_id, 0, n_id)``."""
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        return (group_id, 0, n_id)

    def rhs_scale_index_map(self, n_id: jax.Array, gm_id: jax.Array, k_id: jax.Array):
        """Return the RHS scale tile index ``(group_id, b_tile_id, 0, n_id)``."""
        group_id = self.metadata_ref.gm_id_to_group_id[gm_id]
        k_row = k_id * self.cfgs.tiles.tile_k
        b_row = k_row // self.cfgs.rhs_cfgs.quant_block_size
        b_tile_id = b_row // self.cfgs.num_quant_blocks_per_tile_k
        return (group_id, b_tile_id, 0, n_id)

    def out_index_map(self, n_id: jax.Array, gm_id: jax.Array, _: jax.Array):
        """Return the output ``BoundedSlice`` tile index for grid step ``gm_id``.

        For intermediate grid steps the row slice is capped to the sublane
        boundary so partial rows are not written prematurely.  For the last
        step the full ceil-aligned slice is used.
        """
        is_last_gm = gm_id == (pl.num_programs(1) - 1)
        m_start = self.metadata_ref.gm_id_to_m_offset[gm_id]
        m_end = self.metadata_ref.gm_id_to_m_offset[gm_id + 1]

        row_start = m_start // self.cfgs.dims.size_lhs_sublane
        capped_row_end = m_end // self.cfgs.dims.size_lhs_sublane
        last_row_end = pl.cdiv(m_end, self.cfgs.dims.size_lhs_sublane)
        row_end = jnp.where(is_last_gm, last_row_end, capped_row_end)
        row_size = row_end - row_start
        return (pl.ds(row_start, row_size), 0, n_id)


def generate_block_specs(
    metadata_ref: MetadataRef,
    cfgs: GmmConfigs,
) -> tuple[tuple[pl.BlockSpec, WeightsRef], pl.BlockSpec]:
    """Build Pallas ``BlockSpec`` descriptors for LHS, RHS, and output tensors.

    Called inside ``kernel_main`` after ``fill_metadata`` has populated
    ``metadata_ref``.  The returned specs are passed to ``emit_pipeline``.

    LHS block shape: ``(BoundedSlice(tile_m // size_lhs_sublane), size_lhs_sublane, tile_k)``
    RHS weight block shape: ``(None, tile_k_rhs, tile_n)`` where ``tile_k_rhs``
        may be smaller than ``tile_k`` for packed sub-byte dtypes.
    Output block shape: mirrors the LHS block shape for matching row slices.

    Args:
        metadata_ref: Populated SMEM metadata (group/row mappings).
        cfgs: Full kernel configuration bundle.

    Returns:
        ``((lhs_block_spec, rhs_block_spec), out_block_spec)`` where
        ``rhs_block_spec`` is a ``WeightsRef`` pytree of ``BlockSpec`` nodes
        (weight, optional scale, optional bias).
    """
    index_map = IndexMaps(metadata_ref, cfgs)
    bounded_slice_gm = pl.BoundedSlice(cfgs.tiles.tile_m // cfgs.dims.size_lhs_sublane)

    lhs_block_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_k),
        index_map.lhs_index_map,
    )

    tile_k_rhs = cfgs.tiles.tile_k
    if cfgs.rhs_cfgs.should_bitcast:
        packing = pl.cdiv(32, jax.dtypes.itemsize_bits(cfgs.rhs_cfgs.dtype))
        tile_k_rhs //= packing

    rhs_weight_spec = pl.BlockSpec(
        (None, tile_k_rhs, cfgs.tiles.tile_n),
        index_map.rhs_weight_index_map,
        pipeline_mode=pl.Buffered(buffer_count=3),
    )
    rhs_scale_block_spec = rhs_bias_block_spec = None
    if cfgs.rhs_cfgs.has_bias:
        rhs_bias_block_spec = pl.BlockSpec(
            (None, 1, cfgs.tiles.tile_n),
            index_map.rhs_bias_index_map,
        )
    if cfgs.rhs_cfgs.has_scale:
        rhs_scale_block_spec = pl.BlockSpec(
            (None, cfgs.num_quant_blocks_per_tile_k, 1, cfgs.tiles.tile_n),
            index_map.rhs_scale_index_map,
        )

    rhs_block_spec = WeightsRef(
        weight=rhs_weight_spec,
        scale=rhs_scale_block_spec,
        bias=rhs_bias_block_spec,
    )

    out_block_spec = pl.BlockSpec(
        (bounded_slice_gm, cfgs.dims.size_lhs_sublane, cfgs.tiles.tile_n),
        index_map.out_index_map,
    )

    return (lhs_block_spec, rhs_block_spec), out_block_spec


def inner_kernel(
    tiled_lhs_ref: jax.Array,
    tiled_rhs_ref: RhsRef,
    tiled_out_ref: jax.Array,
    partial_out_ref: jax.Array,
    acc_ref: jax.Array,
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
):
    """Inner pipeline body invoked by ``emit_pipeline`` for each tile.

    Called with the current LHS and RHS tiles already resident in VMEM.
    Dispatches to one of four specialised ``_matmul`` variants depending on
    whether this is the first K-step, the last K-step, both, or neither.

    On the last K-step:
    - Optionally dequantises the RHS weight tile using per-block scales.
    - Adds an optional per-group bias.
    - Applies a fused activation (SiLU / GeLU / SwiGLUoai) if configured.
    - Applies a causal row mask derived from the group's ``m_start``/``m_end``
      offsets, zeroing rows outside the active group slice.
    - Accumulates partial output rows that straddle sublane boundaries into
      ``partial_out_ref`` for the next grid step.

    On intermediate K-steps the partial result is accumulated in ``acc_ref``
    (VMEM) without writing to HBM.

    Args:
        tiled_lhs_ref: Current LHS tile in VMEM.
        tiled_rhs_ref: Current RHS tile bundle (weight + optional scale/bias).
        tiled_out_ref: Output tile in VMEM (written on the last K-step).
        partial_out_ref: VMEM scratch for sub-sublane partial rows.
        acc_ref: VMEM float32 accumulator ``[tile_m, tile_n]``.
        metadata_ref: Populated group/row metadata.
        cfgs: Full kernel configuration bundle.
    """

    def _matmul(is_first_k_step: bool, is_last_k_step: bool):
        tiled_lhs = tiled_lhs_ref.reshape(-1, cfgs.tiles.tile_k)[...]
        tiled_rhs = tiled_rhs_ref.get_weight()
        if cfgs.rhs_cfgs.should_bitcast:
            tiled_rhs = pltpu.bitcast(tiled_rhs, cfgs.rhs_cfgs.dtype)
        rhs_tile_n = tiled_rhs.shape[1]

        valid_k = cfgs.dims.size_k % cfgs.tiles.tile_k
        if is_last_k_step and valid_k != 0:
            mask_rhs = lax.broadcasted_iota(jnp.int32, tiled_rhs.shape, 0) < valid_k
            tiled_rhs = jnp.where(mask_rhs, tiled_rhs, 0)

        acc_list = []
        if cfgs.lhs_cfgs.quant_dtype is None:
            mxu_size = pltpu.get_tpu_info().mxu_column_size
            rhs_qbs = cfgs.rhs_cfgs.quant_block_size
            for start_n in range(0, rhs_tile_n, mxu_size):
                end_n = min(rhs_tile_n, start_n + mxu_size)
                col_size = end_n - start_n

                acc_n = jnp.zeros((cfgs.tiles.tile_m, col_size), dtype=acc_ref.dtype)
                for b_id in range(cfgs.num_quant_blocks_per_tile_k):
                    k_start = b_id * rhs_qbs
                    k_end = k_start + rhs_qbs

                    block_acc = jnp.matmul(
                        tiled_lhs[:, k_start:k_end],
                        tiled_rhs[k_start:k_end, start_n:end_n],
                        preferred_element_type=jnp.float32,
                    ).astype(acc_ref.dtype)

                    if cfgs.rhs_cfgs.has_scale:
                        tiled_rhs_scale = tiled_rhs_ref.get_scale()
                        block_acc *= tiled_rhs_scale[b_id, :, start_n:end_n].astype(acc_ref.dtype)

                    acc_n += block_acc
                acc_list.append(acc_n)
        else:
            lhs_q_dtype = cfgs.lhs_cfgs.quant_dtype
            q_block_size = cfgs.lhs_cfgs.quant_block_size

            if jnp.issubdtype(lhs_q_dtype, jnp.floating):
                dtype_max = float(jnp.finfo(lhs_q_dtype).max)
                preferred_element_type = jnp.float32
            else:
                dtype_max = float(jnp.iinfo(lhs_q_dtype).max)
                preferred_element_type = jnp.int32

            mxu_size = pltpu.get_tpu_info().mxu_column_size
            for start_n in range(0, rhs_tile_n, mxu_size):
                end_n = min(rhs_tile_n, start_n + mxu_size)
                col_size = end_n - start_n

                acc_n = jnp.zeros((cfgs.tiles.tile_m, col_size), dtype=acc_ref.dtype)
                for start_k in range(0, cfgs.tiles.tile_k, q_block_size):
                    end_k = min(cfgs.tiles.tile_k, start_k + q_block_size)

                    block_lhs = tiled_lhs[:, start_k:end_k]
                    block_rhs = tiled_rhs[start_k:end_k, start_n:end_n]

                    block_abs_max = jnp.max(jnp.abs(block_lhs), axis=1, keepdims=True)
                    block_scale = block_abs_max / dtype_max
                    block_scale_inv = jnp.where(block_scale == 0, 0, 1 / block_scale)
                    block_lhs_q = (block_lhs * block_scale_inv).astype(lhs_q_dtype)

                    block_acc = jnp.matmul(
                        block_lhs_q,
                        block_rhs,
                        preferred_element_type=preferred_element_type,
                    ).astype(acc_ref.dtype)
                    block_acc *= block_scale.astype(acc_ref.dtype)

                    if cfgs.rhs_cfgs.has_scale:
                        b_id = start_k // cfgs.rhs_cfgs.quant_block_size
                        rhs_scale_slice = tiled_rhs_ref.get_scale()
                        block_acc *= rhs_scale_slice[b_id, :, start_n:end_n].astype(acc_ref.dtype)

                    acc_n += block_acc
                acc_list.append(acc_n)
        acc = jnp.concatenate(acc_list, axis=1)

        if not is_first_k_step:
            acc += acc_ref[...]

        if is_last_k_step:
            if cfgs.rhs_cfgs.has_bias:
                tiled_rhs_bias = tiled_rhs_ref.get_bias()
                acc += tiled_rhs_bias.astype(acc.dtype)

            acc = apply_act_fn(acc, cfgs.fuse_act)

            gm_id = pl.program_id(1)
            m_start = metadata_ref.gm_id_to_m_offset[gm_id]
            m_end = metadata_ref.gm_id_to_m_offset[gm_id + 1]
            m_offset = m_start - m_start % cfgs.dims.size_lhs_sublane

            m_start_local = m_start - m_offset
            m_end_local = m_end - m_offset

            iota = lax.broadcasted_iota(jnp.int32, acc.shape, 0)
            mask = jnp.logical_and(m_start_local <= iota, iota < m_end_local)
            acc_masked = jnp.where(mask, acc, 0).reshape(tiled_out_ref.shape)
            tiled_out_ref[...] = acc_masked.astype(tiled_out_ref.dtype)

            partial_out_zeros = jnp.zeros_like(partial_out_ref)
            tiled_out_ref[0] += jnp.where(gm_id == 0, partial_out_zeros, partial_out_ref[...])

            last_row = m_end_local // cfgs.dims.size_lhs_sublane
            partial_out_ref[...] = jnp.where(
                m_end_local % cfgs.dims.size_lhs_sublane == 0,
                partial_out_zeros,
                tiled_out_ref[last_row],
            )
        else:
            acc_ref[...] = acc

    @jax.named_scope("matmul_first_last")
    def matmul_first_last():
        _matmul(is_first_k_step=True, is_last_k_step=True)

    @jax.named_scope("matmul_first")
    def matmul_first():
        _matmul(is_first_k_step=True, is_last_k_step=False)

    @jax.named_scope("matmul")
    def matmul():
        _matmul(is_first_k_step=False, is_last_k_step=False)

    @jax.named_scope("matmul_last")
    def matmul_last():
        _matmul(is_first_k_step=False, is_last_k_step=True)

    num_k = pl.num_programs(2)
    k_id = pl.program_id(2)
    is_first_k_step = k_id == 0
    is_last_k_step = k_id == (num_k - 1)

    lax.cond(
        is_first_k_step,
        lambda: lax.cond(is_last_k_step, matmul_first_last, matmul_first),
        lambda: lax.cond(is_last_k_step, matmul_last, matmul),
    )


def fill_metadata(
    lhs_group_sizes_ref: jax.Array,
    group_offset_ref: jax.Array,
    metadata_ref: MetadataRef,
    *,
    cfgs: GmmConfigs,
) -> jax.Array:
    """Populate SMEM metadata for the grouped matmul grid.

    Runs once per kernel invocation (called from ``kernel_main``) to build
    the ``gm_id_to_group_id`` and ``gm_id_to_m_offset`` mappings that
    ``inner_kernel`` and ``IndexMaps`` use to look up the correct HBM tiles.

    The logic iterates over groups ``[group_offset, group_offset + size_group)``
    and, for each non-empty group, subdivides its token range into
    ``tile_m``-sized steps.  Steps that start mid-sublane are given a smaller
    ``tm_size`` so that the first step covers only up to the next aligned
    sublane boundary.

    Args:
        lhs_group_sizes_ref: Scalar-prefetch ref ``[size_lhs_group]`` int32.
        group_offset_ref: Scalar-prefetch ref ``[1]`` int32.
        metadata_ref: SMEM struct to write ``gm_id_to_group_id`` and
            ``gm_id_to_m_offset`` into.
        cfgs: Full kernel configuration bundle.

    Returns:
        ``num_gm``: the total number of grid steps written into ``metadata_ref``.
    """
    group_offset = group_offset_ref[0]
    max_num_group = group_offset + cfgs.dims.size_group
    metadata_ref.gm_id_to_m_offset[0] = 0

    @jax.named_scope("inner_tm_loop")
    def inner_tm_loop(tm_id, curr_m_offset, *, end_m_offset, group_id):
        local_offset = curr_m_offset % cfgs.dims.size_lhs_sublane
        tm_size = jnp.minimum(cfgs.tiles.tile_m - local_offset, end_m_offset - curr_m_offset)

        metadata_ref.gm_id_to_group_id[tm_id] = group_id
        next_m_offset = curr_m_offset + tm_size
        metadata_ref.gm_id_to_m_offset[tm_id] = curr_m_offset
        metadata_ref.gm_id_to_m_offset[tm_id + 1] = next_m_offset
        return next_m_offset

    @jax.named_scope("outer_group_loop")
    def outer_group_loop(lhs_group_id, carry):
        num_gm, start_m_offset = carry

        group_id = lhs_group_id - group_offset
        group_size = lhs_group_sizes_ref[lhs_group_id]
        end_m_offset = start_m_offset + group_size

        local_offset = start_m_offset % cfgs.dims.size_lhs_sublane
        aligned_group_size = group_size + local_offset
        curr_num_gm = pl.cdiv(aligned_group_size, cfgs.tiles.tile_m)

        should_process = jnp.logical_and(group_size > 0, group_id >= 0)
        curr_num_gm = jnp.where(should_process, curr_num_gm, 0)
        next_num_gm = num_gm + curr_num_gm

        tm_loop_fn = functools.partial(inner_tm_loop, end_m_offset=end_m_offset, group_id=group_id)
        lax.fori_loop(num_gm, next_num_gm, tm_loop_fn, start_m_offset)
        return next_num_gm, end_m_offset

    num_gm, _ = lax.fori_loop(0, max_num_group, outer_group_loop, (0, 0))
    return num_gm


def zero_out_start(
    out_ref: jax.Array,
    zero_ref: jax.Array,
    semaphore_ref: jax.Array,
    metadata_ref: MetadataRef,
    num_gm: jax.Array,
    *,
    dims: Dimensions,
):
    """Asynchronously zero output rows outside the active compute range.

    Issues DMA copies from a pre-zeroed VMEM buffer (``zero_ref``) to the
    HBM rows that will not be touched by the pipeline.  This covers two
    regions:

    - Left region: rows 0 .. ``compute_start // size_lhs_sublane``
    - Right region: rows ``ceil(compute_end / size_lhs_sublane)`` .. end

    Copies are issued with ``priority=1`` to run concurrently with the
    pipeline compute.  ``zero_out_end`` must be called after the pipeline
    to wait for all outstanding DMAs.

    Args:
        out_ref: HBM output tensor ``[size_m, aligned_n]``.
        zero_ref: VMEM zero buffer ``[tile_zero_m, num_lanes]``.
        semaphore_ref: DMA semaphore ``[1]``.
        metadata_ref: Populated metadata (provides ``compute_start/end``).
        num_gm: Total number of active grid steps.
        dims: Problem dimensions.

    Returns:
        ``zero_size``: number of rows (in sublane units) actually zeroed,
        forwarded to ``zero_out_end`` as the wait bound.
    """
    num_lanes = pltpu.get_tpu_info().num_lanes
    assert num_lanes == zero_ref.shape[-1]
    zero_ref[...] = jnp.zeros_like(zero_ref)

    zero_dma = zero_ref.reshape(-1, dims.size_lhs_sublane, num_lanes)
    out_dma = out_ref.reshape(-1, dims.size_lhs_sublane, out_ref.shape[-1])
    row_size = zero_dma.shape[0]

    compute_start = metadata_ref.gm_id_to_m_offset[0]
    compute_end = metadata_ref.gm_id_to_m_offset[num_gm]

    left_zero_start = 0
    left_zero_end = compute_start // dims.size_lhs_sublane
    left_zero_size = left_zero_end - left_zero_start
    left_num_loops = pl.cdiv(left_zero_size, row_size)

    right_zero_start = pl.cdiv(compute_end, dims.size_lhs_sublane)
    right_zero_end = out_dma.shape[0]
    right_zero_size = right_zero_end - right_zero_start
    right_num_loops = pl.cdiv(right_zero_size, row_size)

    def fill_zero(i, zero_size, *, start, end):
        dma_start = start + i * row_size
        dma_end = jnp.minimum(dma_start + row_size, end)
        dma_size = dma_end - dma_start

        for n_start in range(0, out_dma.shape[-1], num_lanes):
            n_end = n_start + num_lanes
            pltpu.make_async_copy(
                src_ref=zero_dma.at[pl.ds(0, dma_size)],
                dst_ref=out_dma.at[pl.ds(dma_start, dma_size), :, n_start:n_end],
                sem=semaphore_ref.at[0],
            ).start(priority=1)

        return zero_size + dma_size

    @jax.named_scope("left_fill_zero")
    def left_fill_zero(i, zero_size):
        return fill_zero(i, zero_size, start=left_zero_start, end=left_zero_end)

    @jax.named_scope("right_fill_zero")
    def right_fill_zero(i, zero_size):
        return fill_zero(i, zero_size, start=right_zero_start, end=right_zero_end)

    zero_size = lax.fori_loop(0, left_num_loops, left_fill_zero, 0)
    zero_size = lax.fori_loop(0, right_num_loops, right_fill_zero, zero_size)
    return zero_size


def zero_out_end(
    out_ref: jax.Array,
    semaphore_ref: jax.Array,
    zero_size: jax.Array,
    *,
    dims: Dimensions,
):
    """Wait for the zero-fill DMA copies initiated by ``zero_out_start``.

    Issues a self-copy (src == dst) of ``zero_size`` rows as a barrier
    that the DMA engine resolves only after all preceding DMA ops on the
    same semaphore have completed.

    Args:
        out_ref: HBM output tensor (same reference passed to ``zero_out_start``).
        semaphore_ref: DMA semaphore ``[1]``.
        zero_size: Number of sublane-aligned rows to wait on.
        dims: Problem dimensions.
    """
    out_dma = out_ref.reshape(-1, dims.size_lhs_sublane, out_ref.shape[-1])
    pltpu.make_async_copy(
        src_ref=out_dma.at[pl.ds(0, zero_size)],
        dst_ref=out_dma.at[pl.ds(0, zero_size)],
        sem=semaphore_ref.at[0],
    ).wait()


def kernel_main(
    lhs_group_sizes_ref: jax.Array,
    group_offset_ref: jax.Array,
    lhs_ref: jax.Array,
    rhs_ref: WeightsRef,
    out_ref: jax.Array,
    partial_out_ref: jax.Array,
    acc_ref: jax.Array,
    metadata_ref: MetadataRef,
    zero_ref: jax.Array | None,
    semaphore_ref: jax.Array | None,
    *,
    cfgs: GmmConfigs,
):
    """Top-level Pallas kernel body for grouped matmul v3.

    Orchestrates the full kernel execution:

    1. Optionally bitcasts the RHS weight to ``uint32`` for packed dtypes.
    2. Calls ``fill_metadata`` to build the SMEM group/row mappings.
    3. If ``cfgs.zero_init``: starts async DMA zero-fill of uncomputed rows.
    4. Calls ``generate_block_specs`` to get tile index maps.
    5. If ``cfgs.fuse_act`` is set: wraps RHS in a ``FusedWeightsRef``.
    6. Calls ``pltpu.emit_pipeline`` with ``inner_kernel`` as the body.
    7. If ``cfgs.zero_init``: waits for the zero-fill DMAs to complete.

    Scalar prefetch refs ``lhs_group_sizes_ref`` and ``group_offset_ref``
    are read by ``fill_metadata`` and are *not* passed through
    ``emit_pipeline``.  All other refs (``lhs_ref``, ``rhs_ref``,
    ``out_ref``, ``partial_out_ref``, ``acc_ref``, ``metadata_ref``,
    ``zero_ref``, ``semaphore_ref``) are either VMEM scratch shapes or HBM
    tensors.

    Args:
        lhs_group_sizes_ref: Scalar-prefetch int32 array ``[size_lhs_group]``.
        group_offset_ref: Scalar-prefetch int32 array ``[1]``.
        lhs_ref: HBM LHS tensor ``[size_m, size_k]``.
        rhs_ref: HBM RHS bundle (weight + optional scale/bias).
        out_ref: HBM output tensor ``[size_m, aligned_n]``.
        partial_out_ref: VMEM scratch for partial sublane rows
            ``[size_lhs_sublane, tile_n]``.
        acc_ref: VMEM float32 accumulator ``[tile_m, acc_cols]``.
        metadata_ref: SMEM metadata scratch.
        zero_ref: VMEM zero buffer (or None when ``zero_init=False``).
        semaphore_ref: DMA semaphore (or None when ``zero_init=False``).
        cfgs: Full kernel configuration bundle.
    """
    num_k = pl.cdiv(cfgs.dims.size_k, cfgs.tiles.tile_k)
    num_n = pl.cdiv(cfgs.out_size_n, cfgs.tiles.tile_n)

    if cfgs.rhs_cfgs.should_bitcast:
        rhs_weight = rhs_ref.weight.bitcast(jnp.uint32)
        rhs_ref = dataclasses.replace(rhs_ref, weight=rhs_weight)

    num_gm = fill_metadata(lhs_group_sizes_ref, group_offset_ref, metadata_ref, cfgs=cfgs)

    if cfgs.zero_init:
        zero_size = zero_out_start(
            out_ref,
            zero_ref,
            semaphore_ref,
            metadata_ref,
            num_gm,
            dims=cfgs.dims,
        )

    (lhs_spec, rhs_spec), out_spec = generate_block_specs(metadata_ref, cfgs)

    if cfgs.fuse_act is not None:
        rhs_up_ref = jax.tree.map(lambda x: x.at[..., cfgs.out_size_n :], rhs_ref)
        rhs_ref = FusedWeightsRef(gate=rhs_ref, up=rhs_up_ref)
        rhs_spec = FusedWeightsRef(gate=rhs_spec, up=rhs_spec)

    pipeline_fn = pltpu.emit_pipeline(
        functools.partial(inner_kernel, cfgs=cfgs),
        grid=(num_n, num_gm, num_k),
        in_specs=(lhs_spec, rhs_spec),
        out_specs=out_spec,
    )

    lhs_in = lhs_ref.reshape(-1, cfgs.dims.size_lhs_sublane, lhs_ref.shape[-1])
    out_in = out_ref.reshape(-1, cfgs.dims.size_lhs_sublane, out_ref.shape[-1])
    scratches = [partial_out_ref, acc_ref, metadata_ref]
    pipeline_fn(lhs_in, rhs_ref, out_in, scratches=scratches)

    if cfgs.zero_init:
        zero_out_end(out_ref, semaphore_ref, zero_size, dims=cfgs.dims)


def calculate_tiling(
    dims: Dimensions,
    lhs_cfgs: InputConfigs,
    rhs_cfgs: InputConfigs,
    vmem_limit_bytes: int,
    fuse_act: str | None = None,
) -> TileSizes:
    """Automatically choose tile sizes for the v3 grouped matmul kernel.

    Selects ``tile_m``, ``tile_k``, and ``tile_n`` to fit the RHS weight,
    scale, and bias tiles within ``vmem_limit_bytes / num_rhs_buffers``
    (currently 3 RHS pipeline buffers).  The N dimension is first reduced
    by increasing ``num_n_tiles``, and if still over budget the K dimension
    is reduced by increasing ``num_k_tiles``.  Both ``tile_n`` and ``tile_k``
    are kept as multiples of ``num_lanes`` for TPU alignment.

    ``tile_m`` is chosen based on LHS and RHS dtypes relative to the
    bf16/bf16 baseline of 128.

    Args:
        dims: Problem dimensions.
        lhs_cfgs: LHS dtype / quantisation settings.
        rhs_cfgs: RHS dtype / quantisation settings.
        vmem_limit_bytes: Available VMEM budget in bytes (from
            ``pltpu.get_tpu_info().vmem_capacity_bytes * 0.9`` by default).
        fuse_act: Optional activation name; halves the effective N dimension
            when not None so that the combined gate+up tile still fits.

    Returns:
        ``TileSizes(tile_m, tile_k, tile_n)``.

    Raises:
        ValueError: If no valid tile sizes can be found for the given
            dimensions and VMEM budget.
    """
    lhs_dtype = lhs_cfgs.quant_dtype or lhs_cfgs.dtype
    rhs_dtype = rhs_cfgs.dtype

    lhs_bits = jax.dtypes.itemsize_bits(lhs_dtype)
    rhs_bits = jax.dtypes.itemsize_bits(rhs_dtype)

    bf16_bf16_tile_m = 128
    lhs_mod = min(pl.cdiv(16, lhs_bits), 2)
    rhs_mod = min(pl.cdiv(16, rhs_bits), 2)
    tile_m = bf16_bf16_tile_m * lhs_mod // rhs_mod
    tile_m = min(tile_m, dims.size_m)

    num_rhs_buffers = 3
    rhs_vmem_target = vmem_limit_bytes // num_rhs_buffers
    base_rhs_size_bytes = dims.size_k * dims.size_n * rhs_bits // 8

    tile_n_limit = pltpu.get_tpu_info().mxu_column_size * 2
    tile_n_limit = min(tile_n_limit, dims.size_n)

    size_n_per_rhs = dims.size_n
    if fuse_act is not None:
        size_n_per_rhs //= 2
        tile_n_limit //= 2

    def _is_tile_k_quant_block_compatible(tk: int) -> bool:
        if tk % rhs_cfgs.quant_block_size != 0 and rhs_cfgs.quant_block_size % tk != 0:
            return False
        return True

    num_k_tiles = num_n_tiles = 1
    num_lanes = pltpu.get_tpu_info().num_lanes
    tile_k = align_to(dims.size_k, num_lanes)
    tile_n = align_to(size_n_per_rhs, num_lanes)

    while pl.cdiv(base_rhs_size_bytes, num_n_tiles) > rhs_vmem_target and tile_n > tile_n_limit:
        num_n_tiles += 1
        tile_n = align_to(size_n_per_rhs, num_n_tiles * num_lanes) // num_n_tiles

    if tile_n < tile_n_limit:
        num_n_tiles -= 1
        tile_n = align_to(size_n_per_rhs, num_n_tiles * num_lanes) // num_n_tiles
        base_rhs_size_bytes = pl.cdiv(base_rhs_size_bytes, num_n_tiles)
        while pl.cdiv(base_rhs_size_bytes, num_k_tiles) > rhs_vmem_target or not _is_tile_k_quant_block_compatible(
            tile_k
        ):
            num_k_tiles += 1
            tile_k = align_to(dims.size_k, num_k_tiles * num_lanes) // num_k_tiles

    if tile_n == 0 or tile_k == 0:
        raise ValueError(f"Could not find valid tile sizes for {dims=} and {rhs_vmem_target=}.")

    return TileSizes(tile_m=tile_m, tile_k=tile_k, tile_n=tile_n)


def validate_inputs(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    fuse_act: str | None = None,
) -> Dimensions:
    """Validate inputs and return problem ``Dimensions`` for the v3 kernel.

    Checks that shapes are mutually consistent:
    - ``lhs.shape == (size_m, size_k)``
    - ``rhs.shape == (size_group, size_k, size_n)``
    - ``rhs_bias.shape == (size_group, 1, size_n)`` if provided
    - ``rhs_scale.shape == (size_group, num_blocks, 1, size_n)`` if provided
      and ``size_k % num_blocks == 0``
    - ``group_offset.shape == (1,)``
    - When ``fuse_act`` is enabled, ``size_n`` must be divisible by
      ``2 * num_lanes``.

    Args:
        lhs: Token features ``[size_m, size_k]``.
        rhs: Per-group weights ``[size_group, size_k, size_n]``.
        rhs_scale: Optional quantisation scale.
        rhs_bias: Optional per-group bias.
        group_sizes: Token counts per group ``[size_lhs_group]``, int32.
        group_offset: Shard offset ``[1]``, int32.
        fuse_act: Optional activation name; triggers additional N-divisibility
            check.

    Returns:
        Populated ``Dimensions`` dataclass.

    Raises:
        AssertionError: On any shape mismatch.
        ValueError: When ``fuse_act`` N-divisibility constraint is not met.
    """
    size_m = lhs.shape[0]
    size_group, size_k, size_n = rhs.shape
    size_lhs_group = group_sizes.shape[0]

    assert size_group <= size_lhs_group
    assert lhs.shape == (size_m, size_k)
    assert rhs.shape == (size_group, size_k, size_n)
    if rhs_bias is not None:
        assert rhs_bias.shape == (size_group, 1, size_n)
    if rhs_scale is not None:
        num_quant_blocks = rhs_scale.shape[1]
        assert rhs_scale.shape == (size_group, num_quant_blocks, 1, size_n)
        assert size_k % num_quant_blocks == 0

    assert group_offset.shape == (1,)

    size_lhs_sublane = pltpu.get_tpu_info().get_sublane_tiling(lhs.dtype)
    size_lhs_sublane = min(size_lhs_sublane, size_m)
    if fuse_act is not None:
        num_lanes = pltpu.get_tpu_info().num_lanes
        if size_n % (2 * num_lanes) != 0:
            raise ValueError(f"{size_n=} should be divisible by 2 * num_lanes when fuse_act is enabled.")

    return Dimensions(
        size_m=size_m,
        size_k=size_k,
        size_n=size_n,
        size_group=size_group,
        size_lhs_group=size_lhs_group,
        size_lhs_sublane=size_lhs_sublane,
    )


def get_cost_estimate(cfgs: GmmConfigs):
    """Build a ``pl.CostEstimate`` for the given v3 kernel configuration.

    Reports the theoretical FLOP count (``2 * M * K * N``) and the total
    bytes accessed by LHS, RHS weight + optional scale/bias, and output
    tensors.  The estimate is consumed by the XLA cost model for scheduling
    and performance analysis.

    Args:
        cfgs: Full kernel configuration bundle.

    Returns:
        ``pl.CostEstimate(flops=..., bytes_accessed=..., transcendentals=0)``.
    """
    dims = cfgs.dims
    lhs_dtype = cfgs.lhs_cfgs.quant_dtype or cfgs.lhs_cfgs.dtype
    rhs_dtype = cfgs.rhs_cfgs.dtype

    rhs_bits = jax.dtypes.itemsize_bits(rhs_dtype)
    fp32_bytes = jnp.dtype(jnp.float32).itemsize

    flops = 2 * dims.size_m * dims.size_k * dims.size_n
    lhs_bytes = dims.size_m * dims.size_k * lhs_dtype.itemsize

    rhs_size = dims.size_group * dims.size_k * dims.size_n
    rhs_bytes = rhs_size * rhs_bits // 8
    if cfgs.rhs_cfgs.has_scale:
        num_quant_blocks = pl.cdiv(dims.size_k, cfgs.rhs_cfgs.quant_block_size)
        rhs_bytes += dims.size_group * num_quant_blocks * dims.size_n * fp32_bytes
    if cfgs.rhs_cfgs.has_bias:
        rhs_bytes += dims.size_group * dims.size_n * fp32_bytes

    out_bytes = dims.size_m * cfgs.out_size_n * cfgs.out_dtype.itemsize
    total_bytes = lhs_bytes + rhs_bytes + out_bytes

    return pl.CostEstimate(
        flops=flops,
        bytes_accessed=total_bytes,
        transcendentals=0,
    )


def get_scope_name(cfgs: GmmConfigs) -> str:
    """Return a human-readable XProf scope name for the v3 kernel launch."""
    dims = cfgs.dims
    tiles = cfgs.tiles
    return (
        f"gmm_v3-g_{dims.size_group}-m_{dims.size_m}-k_{dims.size_k}-act_{cfgs.fuse_act}"
        f"-n_{dims.size_n}-tm_{tiles.tile_m}-tk_{tiles.tile_k}-tn_{tiles.tile_n}"
    )


def make_gmm_configs(
    lhs: jax.Array,
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    group_sizes: jax.Array,
    group_offset: jax.Array,
    *,
    tile_info: TileSizes | TileFn,
    vmem_limit_bytes: int | None,
    out_dtype: jnp.dtype | None,
    acc_dtype: jnp.dtype | None,
    maybe_quantize_lhs: bool,
    zero_initialize: bool,
    fuse_act: str | None = None,
):
    """Validate inputs and build the full ``GmmConfigs`` bundle.

    Derives ``InputConfigs`` for both LHS and RHS (including optional LHS
    on-the-fly quantisation for FP8/INT8 when the hardware supports it),
    resolves tiling (``TileSizes`` or ``TileFn``), and packages everything
    into a ``GmmConfigs`` dataclass consumed by ``grouped_matmulv3_pallas_impl``.

    Args:
        lhs: Token features ``[size_m, size_k]``.
        rhs: Per-group weights ``[size_group, size_k, size_n]``.
        rhs_scale: Optional quantisation scale tensor.
        rhs_bias: Optional per-group additive bias tensor.
        group_sizes: Token counts per group ``[size_lhs_group]``, int32.
        group_offset: Shard offset ``[1]``, int32.
        tile_info: Pre-computed ``TileSizes`` or a ``TileFn`` callable.
        vmem_limit_bytes: VMEM budget; passed to ``TileFn`` if ``tile_info``
            is callable.  None means the budget is not passed to the tiling
            function.
        out_dtype: Desired output dtype; defaults to ``lhs.dtype``.
        acc_dtype: Accumulator dtype; defaults to float32 (or bfloat16 when
            LHS quantisation is enabled).
        maybe_quantize_lhs: If True and RHS is quantised (has ``rhs_scale``),
            attempt LHS on-the-fly quantisation to FP8 or INT8.
        zero_initialize: If True the kernel will zero rows outside the active
            group range via async DMA.
        fuse_act: Optional fused activation (``"silu"``, ``"gelu"``,
            ``"swigluoai"``).

    Returns:
        ``GmmConfigs`` dataclass ready to pass to ``kernel_main``.
    """
    dims = validate_inputs(lhs, rhs, rhs_scale, rhs_bias, group_sizes, group_offset, fuse_act)

    if rhs_scale is not None:
        has_scale = True
        rhs_quant_dtype = rhs.dtype
        num_blocks = rhs_scale.shape[1]
        block_size = dims.size_k // num_blocks
    else:
        has_scale = False
        rhs_quant_dtype = None
        block_size = dims.size_k

    rhs_cfgs = InputConfigs(
        quant_dtype=rhs_quant_dtype,
        quant_block_size=block_size,
        dtype=rhs.dtype,
        has_bias=rhs_bias is not None,
        has_scale=has_scale,
    )

    lhs_q_dtype = None
    if maybe_quantize_lhs and rhs_quant_dtype is not None:
        is_rhs_float = jnp.issubdtype(rhs_quant_dtype, jnp.floating)
        tpu_info = pltpu.get_tpu_info()
        if is_rhs_float:
            if tpu_info.fp8_ops_per_second > 0:
                lhs_q_dtype = jnp.float8_e4m3fn.dtype
        else:
            if tpu_info.int8_ops_per_second > 0:
                lhs_q_dtype = jnp.int8.dtype

    lhs_cfgs = InputConfigs(
        quant_dtype=lhs_q_dtype,
        quant_block_size=512,
        dtype=lhs.dtype,
    )

    if out_dtype is None:
        out_dtype = lhs.dtype
    out_dtype = jnp.dtype(out_dtype)

    if acc_dtype is None:
        if lhs_cfgs.quant_dtype is None:
            acc_dtype = jnp.float32.dtype
        else:
            acc_dtype = jnp.bfloat16.dtype
    acc_dtype = jnp.dtype(acc_dtype)

    if isinstance(tile_info, TileSizes):
        tiles = tile_info
    else:
        tiles = tile_info(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act)

    return GmmConfigs(
        dims=dims,
        tiles=tiles,
        lhs_cfgs=lhs_cfgs,
        rhs_cfgs=rhs_cfgs,
        out_dtype=out_dtype,
        acc_dtype=acc_dtype,
        zero_init=zero_initialize,
        fuse_act=fuse_act,
    )


def get_metadata(cfgs: GmmConfigs):
    """Flatten ``GmmConfigs`` into a dict of scalar values for XProf metadata.

    Uses ``jax.tree_util.tree_leaves_with_path`` to walk the dataclass tree
    and converts ``jnp.dtype`` values to their string names.

    Args:
        cfgs: Full kernel configuration bundle.

    Returns:
        A flat ``dict[str, Any]`` suitable for passing as ``metadata`` to
        ``pl.pallas_call``.
    """
    cfgs_dict = dataclasses.asdict(cfgs)
    ret = {}
    for path, val in jax.tree_util.tree_leaves_with_path(cfgs_dict):
        key = jax.tree_util.keystr(path, simple=True, separator=".")
        if isinstance(val, jnp.dtype):
            val = val.name
        ret[key] = val
    return ret


@jax.jit(
    static_argnames=[
        "tile_info",
        "vmem_limit_bytes",
        "precision",
        "preferred_element_type",
        "acc_dtype",
        "maybe_quantize_lhs",
        "zero_initialize",
        "fuse_act",
        "interpret",
    ]
)
def grouped_matmulv3_pallas_impl(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    group_offset: jax.Array | None = None,
    *,
    tile_info: TileSizes | TileFn = calculate_tiling,
    vmem_limit_bytes: int | None = None,
    precision: jax.lax.Precision = jax.lax.Precision.DEFAULT,
    preferred_element_type: jnp.dtype | None = None,
    acc_dtype: jnp.dtype | None = None,
    maybe_quantize_lhs: bool = True,
    zero_initialize: bool = True,
    fuse_act: str | None = None,
    interpret: bool = False,
) -> jax.Array:
    """Core TPU Pallas grouped matmul v3 using ``pltpu.emit_pipeline``.

    Computes ``out[s_i:s_i+g_i, :] = lhs[s_i:s_i+g_i, :] @ rhs[i, :, :]``
    for each group ``i``, using an upstream-style metadata-driven pipeline
    that issues DMA zero-fills, grid metadata, and tiled matmul all in a
    single kernel launch.

    Grid layout:
        ``(num_n, num_gm, num_k)`` where ``num_gm`` is the total number of
        active grid steps derived from ``group_sizes`` and tile sizes.
        ``num_n`` and ``num_k`` tile the output N and contraction K dimensions
        respectively.  The N dimension is marked ``"parallel"`` in
        ``CompilerParams``.

    VMEM usage (scratch shapes):
        - ``partial_out_ref``: ``(size_lhs_sublane, tile_n)`` -- partial rows
          straddling sublane boundaries.
        - ``acc_ref``: ``(tile_m, acc_cols)`` -- fp32 or bf16 accumulator.
        - ``metadata_ref``: ``MetadataRef`` in SMEM -- group/row mappings.
        - ``zero_ref`` (optional): ``(tile_zero_m, num_lanes)`` -- DMA source
          for zeroing output rows.
        - ``semaphore_ref`` (optional): ``SemaphoreType.DMA((1,))`` for
          zero-fill DMA synchronisation.

    All inputs (``lhs``, ``rhs``, and optional ``rhs_scale`` / ``rhs_bias``)
    and the output are in HBM (``pltpu.HBM`` memory space).

    Args:
        lhs: Token features ``[size_m, size_k]``.
        rhs: Per-group weights ``[size_group, size_k, size_n]``.
        group_sizes: Token counts per group ``[size_lhs_group]``, int32.
        rhs_scale: Optional quantisation scale
            ``[size_group, num_blocks, 1, size_n]`` in float32 (cast
            internally).
        rhs_bias: Optional per-group bias ``[size_group, 1, size_n]`` in
            float32 (cast internally).
        group_offset: Shard offset scalar array ``[1]`` int32, or None
            (defaults to 0).
        tile_info: ``TileSizes`` or a ``TileFn`` callable.  Defaults to the
            automatic ``calculate_tiling`` heuristic.
        vmem_limit_bytes: VMEM budget override.  Defaults to 90 % of the
            device's ``vmem_capacity_bytes``.
        precision: Ignored (present for API compatibility).
        preferred_element_type: Output dtype; defaults to ``lhs.dtype``.
        acc_dtype: Accumulator dtype; defaults to float32 for unquantised
            inputs and bfloat16 for LHS-quantised paths.
        maybe_quantize_lhs: Attempt on-the-fly LHS quantisation when the RHS
            is already quantised and the hardware supports FP8/INT8.
        zero_initialize: Zero output rows outside the active group range.
        fuse_act: Optional fused activation: ``"silu"``, ``"gelu"``, or
            ``"swigluoai"``.  When set, ``rhs`` must have
            ``size_n == 2 * actual_output_n`` (gate + up concatenated).
        interpret: Run in Pallas interpreter mode (for debugging only).

    Returns:
        Output tensor ``[size_m, out_size_n]`` where ``out_size_n == size_n``
        (or ``size_n // 2`` when ``fuse_act`` is not None).
    """
    del precision

    if group_offset is None:
        group_offset = jnp.array([0], dtype=jnp.int32)
    else:
        if jnp.isscalar(group_offset):
            group_offset = group_offset[None]

    if vmem_limit_bytes is None:
        vmem_limit_bytes = int(pltpu.get_tpu_info().vmem_capacity_bytes * 0.9)

    cfgs = make_gmm_configs(
        lhs,
        rhs,
        rhs_scale,
        rhs_bias,
        group_sizes,
        group_offset,
        tile_info=tile_info,
        vmem_limit_bytes=vmem_limit_bytes,
        out_dtype=preferred_element_type,
        acc_dtype=acc_dtype,
        maybe_quantize_lhs=maybe_quantize_lhs,
        zero_initialize=zero_initialize,
        fuse_act=fuse_act,
    )
    dims = cfgs.dims
    tiles = cfgs.tiles

    rhs_scale_spec = rhs_bias_spec = None
    if rhs_scale is not None:
        rhs_scale = rhs_scale.astype(jnp.float32)
        rhs_scale_spec = pl.BlockSpec(memory_space=pltpu.HBM)
    if rhs_bias is not None:
        rhs_bias = rhs_bias.astype(jnp.float32)
        rhs_bias_spec = pl.BlockSpec(memory_space=pltpu.HBM)

    max_num_gm = dims.size_group + pl.cdiv(dims.size_m, tiles.tile_m) - 1
    acc_cols = 2 * tiles.tile_n if cfgs.fuse_act is not None else tiles.tile_n
    scratch_shapes = [
        pltpu.VMEM((dims.size_lhs_sublane, tiles.tile_n), cfgs.out_dtype),
        pltpu.VMEM((tiles.tile_m, acc_cols), cfgs.acc_dtype),
        MetadataRef(
            gm_id_to_group_id=pltpu.SMEM((max_num_gm,), jnp.int32),
            gm_id_to_m_offset=pltpu.SMEM((max_num_gm + 1,), jnp.int32),
        ),
    ]

    num_lanes = pltpu.get_tpu_info().num_lanes
    if cfgs.zero_init:
        target_zero_ref_bytes = 2 * 1024 * 1024
        out_bytes = jnp.dtype(cfgs.out_dtype).itemsize
        tile_zero_m = target_zero_ref_bytes // num_lanes // out_bytes
        tile_zero_m = min(tile_zero_m, dims.size_m)

        scratch_shapes += [
            pltpu.VMEM((tile_zero_m, num_lanes), cfgs.out_dtype),
            pltpu.SemaphoreType.DMA((1,)),
        ]
    else:
        scratch_shapes += [None, None]

    aligned_n = align_to(cfgs.out_size_n, num_lanes)
    out_init = jax.ShapeDtypeStruct((dims.size_m, aligned_n), cfgs.out_dtype)
    rhs_weights = WeightsRef(weight=rhs, scale=rhs_scale, bias=rhs_bias)

    return pl.pallas_call(
        functools.partial(kernel_main, cfgs=cfgs),
        out_shape=out_init,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                WeightsRef(
                    weight=pl.BlockSpec(memory_space=pltpu.HBM),
                    scale=rhs_scale_spec,
                    bias=rhs_bias_spec,
                ),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=scratch_shapes,
        ),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=vmem_limit_bytes,
            disable_bounds_checks=True,
        ),
        name=get_scope_name(cfgs),
        cost_estimate=get_cost_estimate(cfgs),
        metadata=get_metadata(cfgs),
        interpret=interpret,
    )(group_sizes, group_offset, lhs, rhs_weights)[:, : cfgs.out_size_n]


__all__ = (
    "Dimensions",
    "GmmConfigs",
    "InputConfigs",
    "MetadataRef",
    "TileFn",
    "TileSizes",
    "calculate_tiling",
    "grouped_matmulv3_pallas_impl",
)
