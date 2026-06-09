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

"""Tile-lang grouped matmul v3."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, DTypeLike, Float, Int

from ejkernel.errors import EjkernelRuntimeError
from ejkernel.kernels._pallas.tpu.grouped_matmul._interface import LutFn

from ..._registry import Backend, Platform, kernel_registry
from ._impl import grouped_matmulv3_tilelang


def _check_common_options(preferred_element_type, tiling, group_offset, interpret, precision):
    """Validate the options specific to ``grouped_matmulv3``.

    Args:
        preferred_element_type: Must be ``jnp.float32``; other dtypes are
            not yet supported.
        tiling: Must be a ``tuple[int, int, int]`` or ``None``.  LUT callables
            (``LutFn``) are not supported by the TileLang backend.
        group_offset: Accepted for API compatibility; currently not validated
            here (validation happens in the impl layer).
        interpret: Must be ``False``; interpreted mode is not supported.
        precision: Must be ``jax.lax.Precision.DEFAULT``.

    Raises:
        EjkernelRuntimeError: If any unsupported option is passed.
    """
    _ = group_offset
    if tiling is not None and not isinstance(tiling, tuple):
        raise EjkernelRuntimeError("tile-lang grouped_matmulv3 does not support LUT tiling callables.")
    if jnp.dtype(preferred_element_type) != jnp.float32:
        raise EjkernelRuntimeError("tile-lang grouped_matmulv3 v0 only supports preferred_element_type=jnp.float32.")
    if interpret:
        raise EjkernelRuntimeError("tile-lang grouped_matmulv3 v0 does not support interpret=True.")
    if precision != jax.lax.Precision.DEFAULT:
        raise EjkernelRuntimeError("tile-lang grouped_matmulv3 v0 does not yet support custom precision.")


@kernel_registry.register("grouped_matmulv3", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def grouped_matmulv3(
    lhs: Float[Array, "m k"],
    rhs: Float[Array, "num_groups k n"] | Float[Array, "num_groups n k"],
    group_sizes: Int[Array, "num_groups_or_shards"],
    preferred_element_type: DTypeLike = jnp.float32,
    tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128),
    group_offset: Int[Array, "..."] | None = None,
    existing_out: Float[Array, "m n"] | None = None,
    rhs_scale: Float[Array, "num_groups num_blocks 1 n"] | None = None,
    rhs_bias: Float[Array, "num_groups 1 n"] | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
) -> Float[Array, "m n"]:
    """Compute grouped matrix multiplication v3 on GPU via TileLang.

    Extends v1/v2 with optional block-wise weight scaling, per-group bias
    addition, and accumulation into an existing output tensor.  A full native
    VJP is provided for ``lhs``, ``rhs``, ``rhs_scale``, ``rhs_bias``, and
    ``existing_out``.

    The per-group row assignment follows the same ``group_sizes`` convention
    as v1: ``out[g_rows] = lhs[g_rows] @ rhs[g] * scale[g] + bias[g]``.

    Args:
        lhs: ``[m, k]`` float32 activation matrix.
        rhs: ``[num_groups, k, n]`` weight matrix, or
            ``[num_groups, n, k]`` when ``transpose_rhs=True``.
        group_sizes: ``[num_groups]`` or ``[num_shards]`` int32 row counts.
        preferred_element_type: Only ``jnp.float32`` accepted.
        tiling: ``(block_m, block_n, block_k)`` hint accepted for API
            compatibility; tile sizes are chosen automatically via
            :func:`~ejkernel.kernels._tilelang.grouped_matmulv3._impl._pick_tile`.
            LUT callables are not supported.
        group_offset: Optional int32 scalar or vector used to index into
            ``group_sizes`` for shard-local operation.  ``None`` means start
            from offset 0.
        existing_out: Optional ``[m, n]`` float32 tensor whose values are
            added element-wise to the matmul output.
        rhs_scale: Optional ``[num_groups, num_blocks, 1, n]`` float32
            block-wise per-column scale applied to ``rhs`` before the
            matmul.  ``num_blocks`` must divide ``k`` evenly.
        rhs_bias: Optional ``[num_groups, 1, n]`` float32 per-group bias
            added after the matmul (and after any ``rhs_scale``).
        transpose_rhs: If ``True``, ``rhs`` is stored as
            ``[num_groups, n, k]``.
        interpret: Must be ``False``; not supported.
        precision: Must be ``jax.lax.Precision.DEFAULT``; not yet supported.

    Returns:
        ``[m, n]`` float32 output tensor.

    Raises:
        EjkernelRuntimeError: If unsupported options are passed.
    """
    _check_common_options(preferred_element_type, tiling, group_offset, interpret, precision)

    if isinstance(tiling, tuple) and len(tiling) == 3:
        bm, bn, bk = int(tiling[0]), int(tiling[1]), int(tiling[2])
    else:
        bm, bn, bk = 128, 128, 64

    return grouped_matmulv3_tilelang(
        lhs,
        rhs,
        group_sizes,
        group_offset=group_offset,
        existing_out=existing_out,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        transpose_rhs=transpose_rhs,
        block_m=bm,
        block_n=bn,
        block_k=bk,
    )


__all__ = ["grouped_matmulv3"]
