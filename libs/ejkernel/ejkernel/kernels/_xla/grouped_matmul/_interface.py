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
"""Grouped matrix multiplication public interface for XLA backend.

This module registers ``grouped_matmul`` under the ``"grouped_matmul"`` and
``"grouped_matmulv2"`` keys in the ejkernel registry for the XLA platform.
The actual computation delegates to ``jax.lax.ragged_dot_general``, the
canonical XLA primitive for ragged grouped GEMM.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype

from ejkernel.kernels._pallas.tpu.grouped_matmul._interface import LutFn

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import Array, DTypeLike, Float, Int, jax, jnp
from ._xla_impl_fwd import grouped_matmul as _grouped_matmul_impl


@kernel_registry.register("grouped_matmul", Platform.XLA, Backend.ANY)
@kernel_registry.register("grouped_matmulv2", Platform.XLA, Backend.ANY)
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
    """Grouped Matrix Multiplication using XLA's ragged_dot_general primitive.

    Computes separate matrix products for each group of rows in ``lhs``,
    multiplying them against the corresponding group matrix in ``rhs``.
    For each group ``i``:
        ``out[start_i:end_i, :] = lhs[start_i:end_i, :] @ rhs[i, :, :]``
    where ``start_i`` and ``end_i`` are determined by the prefix sums of
    ``group_sizes``.

    The XLA backend delegates to ``jax.lax.ragged_dot_general``.  Optional
    ``tiling`` metadata is forwarded to XLA via ``set_xla_metadata`` to guide
    tile-size selection.

    Args:
        lhs: Left-hand side matrix. Shape: [m, k].  ``m`` is the total number
            of rows across all groups.
        rhs: Right-hand side per-group weight matrices.
            Shape: [num_groups, k, n] (or [num_groups, n, k] when
            ``transpose_rhs=True``).
        group_sizes: Number of ``lhs`` rows per group.
            Shape: [num_groups].  Must sum to ``m``.
        preferred_element_type: Accumulation and output dtype. Defaults to
            ``float32``.
        tiling: Hint for XLA tile sizes as ``(tm, tk, tn)``, a ``LutFn``
            callable ``(m, k, n) -> (tm, tk, tn) | None``, or ``None`` to let
            XLA choose.  Passed via ``set_xla_metadata``; no effect on
            correctness.
        group_offset: Starting group index for sharded execution.  Defaults to
            ``0``.
        existing_out: Optional output tensor to accumulate into.  If provided,
            the final result is ``ragged_dot_result + existing_out``.
            Shape: [m, n].
        transpose_rhs: If True, ``rhs`` is interpreted as [num_groups, n, k]
            and transposed before the matmul.
        interpret: Accepted for API compatibility with the Pallas backend;
            silently ignored in this XLA implementation.
        precision: JAX ``lax.Precision`` for the matrix multiplication.

    Returns:
        Output matrix of shape [m, n].

    Example:
        >>> lhs = jnp.ones((300, 64))
        >>> rhs = jnp.ones((3, 64, 32))
        >>> group_sizes = jnp.array([100, 150, 50], dtype=jnp.int32)
        >>> result = grouped_matmul(lhs, rhs, group_sizes)
        >>> result.shape
        (300, 32)
    """
    return _grouped_matmul_impl(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type,
        tiling,
        group_offset,
        existing_out,
        transpose_rhs,
        interpret,
        precision,
    )


__all__ = ("grouped_matmul",)
