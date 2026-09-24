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

"""Grouped matrix multiplication core implementation using XLA ragged_dot.

This module provides the core XLA implementation of grouped matrix
multiplication using ``jax.lax.ragged_dot_general``.  Different row slices
of ``lhs`` are multiplied against their corresponding group matrix in ``rhs``
according to the ``group_sizes`` partition vector.

The ``set_xla_metadata`` context manager is used to forward optional ``tiling``
hints to the XLA compiler, enabling tile-size optimisation without altering
numerical behaviour.
"""

from __future__ import annotations

import contextlib
import typing
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import xla_metadata
from jaxtyping import Array, DTypeLike, Float, Int

if typing.TYPE_CHECKING:
    from ejkernel.kernels._pallas.tpu.grouped_matmul._interface import LutFn

set_xla_metadata = xla_metadata.set_xla_metadata

# XLA:TPU lowers ``jax.lax.ragged_dot_general`` through ``ragged_dot_expander``,
# which tiles the ragged (m) dimension by the TPU sublane size (8).  When m is
# not a multiple of 8 the expander drops into a degenerate windowing whose
# ``max_ragged_dim_window_bound`` is 2, while the window it actually needs is
# ``ceil(m / num_groups)``.  For skewed MoE shapes (e.g. m=52, num_groups=8) on
# jax/jaxlib 0.10.0 this trips a *process-fatal* CHECK during backend
# compilation:
#     F ragged_dot_expander.cc:959] Check failed:
#       ragged_dim_window_bound <= max_ragged_dim_window_bound
#       Window too large 7 max_ragged_dim_window_bound 2
# Padding m up to a multiple of 8 moves the expander back into its regular
# tiling regime.  Verified on v5p safe for m % 8 == 0 across num_groups in
# {3, 4, 5, 8, 16, 60, 64, 128} and windows up to 270; the ``ragged_dot_tiling``
# metadata has no influence on this CHECK.
_RAGGED_DOT_M_ALIGNMENT = 8

# The native TPU GMM tiler on jax/jaxlib 0.11.2 with libtpu 0.0.48 also rejects
# an m=120 MoE projection (bf16 preferred output, no tiling hint):
#     expecting m % mt == 0, got 120 % 32
# With no explicit hint, align to 32 while retaining the older 8-row alignment.
# Explicit tiling keeps the original alignment: rounding m=40 to 64 would break
# a valid tm=40 hint (64 % 40 != 0).
_UNTILED_RAGGED_DOT_M_ALIGNMENT = 32


# Keep pad/dot/slice in one compilation. Separate eager primitive dispatch on
# JAX 0.11.2/libtpu 0.0.48 corrupts narrow TP4 outputs (local n=32), including
# nonfinite values from finite operands. An enclosing jit avoids that failure.
@partial(
    jax.jit,
    static_argnames=("preferred_element_type", "tiling", "transpose_rhs", "interpret", "precision"),
)
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
    """Grouped Matrix Multiplication via ``jax.lax.ragged_dot_general``.

    Computes per-group matrix products: for each group ``i``,
        ``out[start_i:end_i, :] = lhs[start_i:end_i, :] @ rhs[i, :, :]``
    where ``start_i`` / ``end_i`` are determined by prefix sums of
    ``group_sizes``.

    Tiling metadata is forwarded to the XLA compiler via
    ``set_xla_metadata(ragged_dot_tiling=...)`` but does not affect numerical
    results.  If ``precision is None``, it is overridden to
    ``lax.Precision.HIGHEST`` before the call.

    Args:
        lhs: Left-hand side matrix. Shape: [m, k].
        rhs: Per-group right-hand side matrices.
            Shape: [num_groups, k, n] (or [num_groups, n, k] when
            ``transpose_rhs=True``).
        group_sizes: Number of ``lhs`` rows per group.
            Shape: [num_groups].  Must sum to ``m``.
        preferred_element_type: Accumulation / output dtype.  Defaults to
            ``float32``.
        tiling: Optional ``(tm, tk, tn)`` tile-size hint forwarded as XLA
            metadata, a ``LutFn``, or ``None`` (no hint).  Does not change
            correctness.
        group_offset: Starting group index for sharded execution.  Defaults to
            ``0``.
        existing_out: Optional tensor to add to the result.
            If provided, ``output = ragged_dot_result + existing_out``.
            Shape: [m, n].
        transpose_rhs: If True, ``rhs`` is treated as [num_groups, n, k].
        interpret: Accepted for API compatibility; silently ignored.
        precision: JAX ``lax.Precision`` for the matmul.  Defaults to
            ``DEFAULT``; ``None`` is coerced to ``HIGHEST``.

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
    if precision is None:
        precision = jax.lax.Precision.HIGHEST
    if tiling is None:
        manager = contextlib.nullcontext()
    else:
        manager = set_xla_metadata(ragged_dot_tiling=",".join([str(t) for t in tiling]))

    # Align the ragged (m) dimension for the TPU expander, and for the native
    # GMM tiler only when no explicit hint was supplied (see constants above).
    # Keep group_sizes unchanged: the zero padding rows sit past its sum and
    # belong to no group. Strip them before adding existing_out.
    m_orig = lhs.shape[0]
    alignment = _UNTILED_RAGGED_DOT_M_ALIGNMENT if tiling is None else _RAGGED_DOT_M_ALIGNMENT
    ragged_pad = (-m_orig) % alignment
    if ragged_pad:
        lhs = jax.lax.pad(lhs, jnp.array(0.0, dtype=lhs.dtype), [(0, ragged_pad, 0), (0, 0, 0)])

    with manager:
        out = jax.lax.ragged_dot_general(
            lhs=lhs,
            rhs=rhs,
            group_sizes=group_sizes,
            precision=precision,
            preferred_element_type=preferred_element_type,
            group_offset=group_offset,
            ragged_dot_dimension_numbers=jax.lax.RaggedDotDimensionNumbers(
                dot_dimension_numbers=(((1,), (2,)) if transpose_rhs else ((1,), (1,)), ((), ())),
                lhs_ragged_dimensions=(0,),
                rhs_group_dimensions=(0,),
            ),
        )
    if ragged_pad:
        out = out[:m_orig]
    if existing_out is not None:
        out = out + jnp.asarray(existing_out, dtype=out.dtype)
    return out
