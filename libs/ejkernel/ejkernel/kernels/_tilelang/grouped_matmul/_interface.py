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

"""Tile-lang grouped matmul (v1 + v2).

``grouped_matmulv2`` has the same numerics as v1 in the XLA reference —
the version split is about tiling heuristics, which the tile-lang tile
picker handles adaptively. Both names register here.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, DTypeLike, Float, Int

from ejkernel.errors import EjkernelRuntimeError
from ejkernel.kernels._pallas.tpu.grouped_matmul._interface import LutFn

from ..._registry import Backend, Platform, kernel_registry
from .._grouped_matmul_impl import grouped_matmul_trainable_tilelang


def _impl(lhs, rhs, group_sizes, group_offset, existing_out, transpose_rhs, *, block_m, block_n, block_k):
    """Thin shim that forwards to the shared v3-backed trainable kernel."""
    return grouped_matmul_trainable_tilelang(
        lhs,
        rhs,
        group_sizes,
        group_offset=group_offset,
        existing_out=existing_out,
        transpose_rhs=transpose_rhs,
        block_m=int(block_m),
        block_n=int(block_n),
        block_k=int(block_k),
    )


def _tile_from(tiling) -> tuple[int, int, int]:
    """Extract ``(block_m, block_n, block_k)`` from a tiling tuple.

    Falls back to constants when ``tiling`` is None or a non-tuple LUT
    function (the caller did not specify tiles). The operation layer
    supplies the authoritative tiles via cfg.
    """
    if isinstance(tiling, tuple) and len(tiling) == 3:
        return int(tiling[0]), int(tiling[1]), int(tiling[2])
    return 128, 128, 64


def _check_common_options(preferred_element_type, tiling, interpret, precision):
    """Validate options that apply to both ``grouped_matmul`` and ``grouped_matmulv2``.

    Args:
        preferred_element_type: Output accumulation dtype.  Only
            ``jnp.float32`` is currently accepted.
        tiling: Tile-size hint.  Accepted but ignored — tile sizes are
            determined automatically by the kernel.
        interpret: Must be ``False``; interpreted mode is not supported.
        precision: Must be ``jax.lax.Precision.DEFAULT``; custom precision
            is not yet supported.

    Raises:
        EjkernelRuntimeError: If any unsupported option is passed.
    """
    _ = tiling
    if jnp.dtype(preferred_element_type) != jnp.float32:
        raise EjkernelRuntimeError("tile-lang grouped_matmul v0 only supports preferred_element_type=jnp.float32.")
    if interpret:
        raise EjkernelRuntimeError("tile-lang grouped_matmul v0 does not support interpret=True.")
    if precision != jax.lax.Precision.DEFAULT:
        raise EjkernelRuntimeError("tile-lang grouped_matmul v0 does not yet support custom precision.")


@kernel_registry.register("grouped_matmul", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def grouped_matmul(
    lhs: Float[Array, "m k"],
    rhs: Float[Array, "num_groups k n"] | Float[Array, "num_groups n k"],
    group_sizes: Int[Array, "num_groups_or_shards"],
    preferred_element_type: DTypeLike = jnp.float32,
    tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128),
    group_offset: Int[Array, "..."] | None = None,
    existing_out: Float[Array, "m n"] | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
) -> Float[Array, "m n"]:
    """Compute a grouped matrix multiplication on GPU via TileLang.

    Computes ``out[g_rows] = lhs[g_rows] @ rhs[g]`` for each group ``g``,
    where ``g_rows`` is the contiguous row range assigned to group ``g``
    according to ``group_sizes``.

    Delegates to the v3 trainable kernel
    (:func:`~ejkernel.kernels._tilelang.grouped_matmulv3._impl.grouped_matmulv3_tilelang`).
    Native VJP is available via :func:`grouped_matmulv3_tilelang`.

    Args:
        lhs: ``[m, k]`` float32 activation matrix.
        rhs: ``[num_groups, k, n]`` weight matrix, or
            ``[num_groups, n, k]`` when ``transpose_rhs=True``.
        group_sizes: ``[num_groups]`` or ``[num_shards]`` int32 vector of
            per-group row counts (must sum to ``m``).
        preferred_element_type: Output dtype.  Only ``jnp.float32`` is
            currently accepted; passing another dtype raises
            ``EjkernelRuntimeError``.
        tiling: Accepted for API compatibility with the TPU Pallas backend but
            ignored here — tile sizes are chosen automatically.
        group_offset: Optional int32 scalar or vector used to index into
            ``group_sizes`` when operating on a shard.  ``None`` means
            start from offset 0.
        existing_out: Optional ``[m, n]`` float32 tensor whose values are
            added element-wise to the matmul result before writing the
            output (i.e. in-place accumulation).
        transpose_rhs: If ``True``, ``rhs`` is interpreted as
            ``[num_groups, n, k]`` instead of ``[num_groups, k, n]``.
        interpret: Must be ``False``; not supported by this backend.
        precision: Must be ``jax.lax.Precision.DEFAULT``; not yet supported.

    Returns:
        ``[m, n]`` float32 output tensor.

    Raises:
        EjkernelRuntimeError: If unsupported options are passed.
    """
    _check_common_options(preferred_element_type, tiling, interpret, precision)
    bm, bn, bk = _tile_from(tiling)
    return _impl(
        lhs,
        rhs,
        group_sizes,
        group_offset,
        existing_out,
        transpose_rhs,
        block_m=bm,
        block_n=bn,
        block_k=bk,
    )


@kernel_registry.register("grouped_matmulv2", Platform.TILELANG, Backend.GPU)
@jaxtyping.jaxtyped(typechecker=beartype)
def grouped_matmulv2(
    lhs: Float[Array, "m k"],
    rhs: Float[Array, "num_groups k n"] | Float[Array, "num_groups n k"],
    group_sizes: Int[Array, "num_groups_or_shards"],
    preferred_element_type: DTypeLike = jnp.float32,
    tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128),
    group_offset: Int[Array, "..."] | None = None,
    existing_out: Float[Array, "m n"] | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
) -> Float[Array, "m n"]:
    """Grouped matrix multiplication v2 — identical kernel to ``grouped_matmul`` v1.

    This entry exists so that callers using the ``grouped_matmulv2`` name in
    the XLA registry can transparently route to the same TileLang kernel.
    See :func:`grouped_matmul` for full argument documentation.

    Args:
        lhs: ``[m, k]`` float32 activation matrix.
        rhs: ``[num_groups, k, n]`` (or ``[num_groups, n, k]`` when
            ``transpose_rhs=True``) weight matrix.
        group_sizes: ``[num_groups]`` or ``[num_shards]`` int32 row counts.
        preferred_element_type: Only ``jnp.float32`` accepted.
        tiling: Accepted but ignored.
        group_offset: Optional shard offset into ``group_sizes``.
        existing_out: Optional ``[m, n]`` tensor to accumulate into.
        transpose_rhs: Treat ``rhs`` as ``[num_groups, n, k]``.
        interpret: Must be ``False``.
        precision: Must be ``jax.lax.Precision.DEFAULT``.

    Returns:
        ``[m, n]`` float32 output tensor.

    Raises:
        EjkernelRuntimeError: If unsupported options are passed.
    """
    _check_common_options(preferred_element_type, tiling, interpret, precision)
    bm, bn, bk = _tile_from(tiling)
    return _impl(
        lhs,
        rhs,
        group_sizes,
        group_offset,
        existing_out,
        transpose_rhs,
        block_m=bm,
        block_n=bn,
        block_k=bk,
    )


__all__ = ["grouped_matmul", "grouped_matmulv2"]
