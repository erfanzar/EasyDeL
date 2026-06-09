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

"""Grouped Matrix Multiplication v3 (GMM v3) public interface for XLA backend.

This module extends the base grouped matmul (v1/v2) with two extra per-group
parameters: an optional block-wise scale (``rhs_scale``) and an optional bias
(``rhs_bias``), enabling block-float quantisation of the weight matrices.

Key differences from ``grouped_matmul``:
    - ``rhs_scale``: per-block scale tensor applied to ``rhs`` before the matmul.
    - ``rhs_bias``: per-group bias added to the output after the matmul.
    - A custom VJP (``_grouped_matmulv3_core``) ensures that gradients for all
      optional tensors (``rhs_scale``, ``rhs_bias``, ``existing_out``) are
      computed correctly.
    - When ``rhs_scale`` or ``rhs_bias`` is provided the forward path falls back
      to a vmap-based pure-JAX reference (``grouped_matmulv3_autodiff_reference``)
      rather than ``ragged_dot_general``, so that scale/bias are fused into the
      weight before the matmul.

Registered kernel keys: ``"grouped_matmulv3"`` (XLA platform, any backend).
"""

from __future__ import annotations

from functools import partial

import jaxtyping
from beartype import beartype

from ejkernel.kernels._pallas.tpu.grouped_matmul._interface import LutFn

from ..._registry import Backend, Platform, kernel_registry
from ..grouped_matmul._xla_impl_fwd import Array, DTypeLike, Float, Int, jax, jnp
from ..grouped_matmul._xla_impl_fwd import grouped_matmul as _grouped_matmul_impl


def _apply_rhs_scale_bias(
    rhs: jax.Array,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    *,
    transpose_rhs: bool,
) -> tuple[jax.Array, jax.Array | None]:
    """Pre-process ``rhs`` by applying optional block-wise scale and extracting bias.

    Handles the ``transpose_rhs`` layout normalisation and the optional
    block-float ``rhs_scale`` dequantisation in a backend-agnostic way.
    The resulting ``rhs_prepped`` is always in [num_groups, k, n] layout and
    has ``rhs_scale`` baked in.

    Args:
        rhs: Raw per-group weight tensor.  Either [num_groups, k, n] or, when
            ``transpose_rhs=True``, [num_groups, n, k].
        rhs_scale: Optional block-wise scale in
            shape [num_groups, num_blocks, 1, n].  Each block along the ``k``
            dimension shares one scale value broadcast over ``block_size = k //
            num_blocks`` rows.
        rhs_bias: Optional per-group bias in shape [num_groups, 1, n].
            Extracted and returned as a [num_groups, n] vector for downstream
            index-based broadcast.
        transpose_rhs: When True, ``rhs`` is transposed from [num_groups, n, k]
            to [num_groups, k, n] before scale is applied.

    Returns:
        Tuple of:
            - rhs_prepped: Scale-applied weight tensor [num_groups, k, n].
            - bias: Bias vector [num_groups, n], or None if ``rhs_bias`` is None.

    Raises:
        ValueError: If ``rhs_scale`` shape is incompatible with ``rhs``.
        ValueError: If ``rhs_bias`` shape is incompatible with ``rhs``.
    """
    rhs_prepped = rhs.swapaxes(1, 2) if transpose_rhs else rhs
    bias = None

    if rhs_scale is not None:
        if rhs_scale.ndim != 4 or rhs_scale.shape[2] != 1:
            raise ValueError("rhs_scale must have shape [num_groups, num_blocks, 1, n].")
        num_groups, size_k, size_n = rhs_prepped.shape
        if rhs_scale.shape[0] != num_groups or rhs_scale.shape[3] != size_n:
            raise ValueError("rhs_scale group/out dimensions must match rhs.")
        num_blocks = int(rhs_scale.shape[1])
        if size_k % num_blocks != 0:
            raise ValueError("rhs.shape[1] must be divisible by rhs_scale.shape[1].")
        block_size = size_k // num_blocks
        scale = jnp.repeat(rhs_scale[:, :, 0, :], block_size, axis=1)
        rhs_prepped = rhs_prepped * scale.astype(rhs_prepped.dtype)

    if rhs_bias is not None:
        if rhs_bias.ndim != 3 or rhs_bias.shape[1] != 1:
            raise ValueError("rhs_bias must have shape [num_groups, 1, n].")
        if rhs_bias.shape[0] != rhs_prepped.shape[0] or rhs_bias.shape[2] != rhs_prepped.shape[2]:
            raise ValueError("rhs_bias group/out dimensions must match rhs.")
        bias = rhs_bias[:, 0, :]

    return rhs_prepped, bias


def _active_group_ids(
    group_sizes: jax.Array,
    num_groups: int,
    total_rows: int,
    group_offset: jax.Array | None,
) -> jax.Array:
    """Build a per-row group-index vector for use with ``jax.vmap`` dispatch.

    For each row in ``lhs[0:total_rows]`` returns the index of the group it
    belongs to.  Used by the pure-JAX autodiff reference to select the
    appropriate ``rhs`` slice for each row.

    Args:
        group_sizes: Per-group row counts. Shape: [num_groups_or_shards].
        num_groups: Number of active groups to process (typically
            ``rhs.shape[0]``).
        total_rows: Total number of ``lhs`` rows (``m``).
        group_offset: Optional scalar (or 1-element array) indicating the
            starting offset into ``group_sizes`` for sharded execution.

    Returns:
        Integer array of shape [total_rows] where ``result[i]`` is the group
        index for row ``i``.
    """
    offset = (
        group_offset.reshape(-1)[0].astype(group_sizes.dtype)
        if group_offset is not None
        else jnp.array(0, dtype=group_sizes.dtype)
    )
    active_sizes = jax.lax.dynamic_slice_in_dim(group_sizes, offset, num_groups, axis=0)
    return jnp.repeat(
        jnp.arange(num_groups, dtype=group_sizes.dtype),
        active_sizes,
        total_repeat_length=total_rows,
    )


def grouped_matmulv3_autodiff_reference(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    preferred_element_type: DTypeLike = jnp.float32,
    tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128),
    group_offset: jax.Array | None = None,
    existing_out: jax.Array | None = None,
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
) -> jax.Array:
    """Pure-JAX vmap reference for GMM v3, used as the autodiff-compatible path.

    When ``rhs_scale`` or ``rhs_bias`` is non-None, ``ragged_dot_general``
    cannot track gradients through the quantisation parameters.  This function
    uses a ``jax.vmap`` over rows instead, keeping all operations differentiable
    by standard JAX autodiff.

    ``tiling`` and ``interpret`` are accepted but ignored (they only affect
    the Pallas backend).

    Args:
        lhs: [m, k] left-hand side matrix.
        rhs: [num_groups, k, n] (or [num_groups, n, k]) weight matrices.
        group_sizes: [num_groups] per-group row counts.
        preferred_element_type: Output dtype.
        tiling: Ignored in this reference implementation.
        group_offset: Optional starting group index for sharded execution.
        existing_out: Optional [m, n] accumulation tensor.
        rhs_scale: Optional [num_groups, num_blocks, 1, n] block-float scale.
        rhs_bias: Optional [num_groups, 1, n] per-group bias.
        transpose_rhs: If True, ``rhs`` is [num_groups, n, k].
        interpret: Ignored.
        precision: JAX matmul precision.

    Returns:
        Output matrix of shape [m, n].
    """
    del tiling, interpret
    rhs_prepped, bias = _apply_rhs_scale_bias(
        rhs,
        rhs_scale,
        rhs_bias,
        transpose_rhs=transpose_rhs,
    )
    group_ids = _active_group_ids(group_sizes, rhs_prepped.shape[0], lhs.shape[0], group_offset)
    out = jax.vmap(
        lambda row, mat: jnp.matmul(
            row,
            mat,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )
    )(lhs, rhs_prepped[group_ids])
    if bias is not None:
        out = out + bias[group_ids].astype(out.dtype)
    if existing_out is not None:
        out = out + jnp.asarray(existing_out, dtype=out.dtype)
    return out


def grouped_matmulv3_reference(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    preferred_element_type: DTypeLike = jnp.float32,
    tiling: tuple[int, int, int] | LutFn | None = (128, 128, 128),
    group_offset: jax.Array | None = None,
    existing_out: jax.Array | None = None,
    rhs_scale: jax.Array | None = None,
    rhs_bias: jax.Array | None = None,
    transpose_rhs: bool = False,
    interpret: bool = False,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.DEFAULT,
) -> jax.Array:
    """Forward pass for GMM v3 used by both XLA execution and the TPU backward helpers.

    Dispatches to one of two implementations:
        - ``grouped_matmulv3_autodiff_reference`` (vmap-based): when
          ``rhs_scale`` or ``rhs_bias`` is provided, so that quantisation
          parameters remain differentiable.
        - ``_grouped_matmul_impl`` (``ragged_dot_general``): when no scale/bias
          is set, taking advantage of the optimised XLA primitive.

    After the core matmul, any ``rhs_bias`` and ``existing_out`` are added
    in the output dtype.

    Args:
        lhs: [m, k] left-hand side matrix.
        rhs: [num_groups, k, n] (or [num_groups, n, k]) weight matrices.
        group_sizes: [num_groups] per-group row counts.
        preferred_element_type: Output dtype.
        tiling: Tile-size hint for ``ragged_dot_general`` via XLA metadata.
            Ignored when falling back to the vmap reference.
        group_offset: Optional starting group index for sharded execution.
        existing_out: Optional [m, n] accumulation tensor.
        rhs_scale: Optional [num_groups, num_blocks, 1, n] block-float scale.
        rhs_bias: Optional [num_groups, 1, n] per-group bias.
        transpose_rhs: If True, ``rhs`` is [num_groups, n, k].
        interpret: Accepted for API compatibility; ignored.
        precision: JAX matmul precision.

    Returns:
        Output matrix of shape [m, n].
    """
    if rhs_scale is not None or rhs_bias is not None:
        return grouped_matmulv3_autodiff_reference(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type,
            tiling,
            group_offset,
            existing_out,
            rhs_scale,
            rhs_bias,
            transpose_rhs,
            interpret,
            precision,
        )
    rhs_prepped, bias = _apply_rhs_scale_bias(
        rhs,
        rhs_scale,
        rhs_bias,
        transpose_rhs=transpose_rhs,
    )
    out = _grouped_matmul_impl(
        lhs,
        rhs_prepped,
        group_sizes,
        preferred_element_type,
        tiling,
        group_offset,
        existing_out=None,
        transpose_rhs=False,
        interpret=interpret,
        precision=precision,
    )
    if bias is not None:
        group_ids = _active_group_ids(group_sizes, bias.shape[0], lhs.shape[0], group_offset)
        out = out + bias[group_ids].astype(out.dtype)
    if existing_out is not None:
        out = out + jnp.asarray(existing_out, dtype=out.dtype)
    return out


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 9, 10, 11))
def _grouped_matmulv3_core(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    preferred_element_type: DTypeLike,
    tiling: tuple[int, int, int] | LutFn | None,
    group_offset: jax.Array | None,
    existing_out: jax.Array | None,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    transpose_rhs: bool,
    interpret: bool,
    precision: jax.lax.PrecisionLike,
) -> jax.Array:
    """GMM v3 core function with a custom VJP defined for stable gradient computation.

    ``preferred_element_type``, ``tiling``, ``transpose_rhs``, ``interpret``,
    and ``precision`` are non-differentiable static arguments (``nondiff_argnums``).

    The forward computation delegates to ``grouped_matmulv3_reference``.
    Gradients are computed via ``_grouped_matmulv3_bwd`` which uses the vmap-based
    autodiff reference to handle ``rhs_scale`` and ``rhs_bias`` gradients.

    Args:
        lhs: [m, k] left-hand side.
        rhs: [num_groups, k, n] or [num_groups, n, k] weight matrices.
        group_sizes: [num_groups] row partition.
        preferred_element_type: Non-diff output dtype.
        tiling: Non-diff XLA tile hint.
        group_offset: Optional shard offset.
        existing_out: Optional accumulation tensor.
        rhs_scale: Optional block-float scale.
        rhs_bias: Optional per-group bias.
        transpose_rhs: Non-diff transposition flag.
        interpret: Non-diff debug flag (ignored).
        precision: Non-diff matmul precision.

    Returns:
        Output matrix of shape [m, n].
    """
    return grouped_matmulv3_reference(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type,
        tiling,
        group_offset,
        existing_out,
        rhs_scale,
        rhs_bias,
        transpose_rhs,
        interpret,
        precision,
    )


def _grouped_matmulv3_fwd(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    preferred_element_type: DTypeLike,
    tiling: tuple[int, int, int] | LutFn | None,
    group_offset: jax.Array | None,
    existing_out: jax.Array | None,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    transpose_rhs: bool,
    interpret: bool,
    precision: jax.lax.PrecisionLike,
):
    """Forward rule for ``_grouped_matmulv3_core``'s custom VJP.

    Runs the forward computation and saves the inputs needed by the backward
    pass in the residual tuple.

    Returns:
        Tuple of (output, residuals) where residuals is
        ``(lhs, rhs, group_sizes, group_offset, existing_out, rhs_scale, rhs_bias)``.
    """
    out = grouped_matmulv3_reference(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type,
        tiling,
        group_offset,
        existing_out,
        rhs_scale,
        rhs_bias,
        transpose_rhs,
        interpret,
        precision,
    )
    return out, (lhs, rhs, group_sizes, group_offset, existing_out, rhs_scale, rhs_bias)


def _grouped_matmulv3_bwd(
    preferred_element_type: DTypeLike,
    tiling: tuple[int, int, int] | LutFn | None,
    transpose_rhs: bool,
    interpret: bool,
    precision: jax.lax.PrecisionLike,
    residual,
    grad: jax.Array,
):
    """Backward rule for ``_grouped_matmulv3_core``'s custom VJP.

    Computes gradients for all differentiable inputs:
        - ``grad_lhs``, ``grad_rhs``: from ``jax.vjp`` through
          ``grouped_matmulv3_autodiff_reference`` (pure JAX, always
          differentiable).
        - ``grad_rhs_scale``, ``grad_rhs_bias``: separate ``jax.vjp`` calls
          when those optional tensors are non-None.
        - ``grad_existing_out``: equal to ``grad`` when ``existing_out`` is
          non-None (addition is the identity in the backward).
        - ``grad_group_sizes``, ``grad_group_offset``: always ``None``
          (integer indices are not differentiable).

    Args:
        preferred_element_type: Non-diff argument from ``nondiff_argnums``.
        tiling: Non-diff argument from ``nondiff_argnums``.
        transpose_rhs: Non-diff argument from ``nondiff_argnums``.
        interpret: Non-diff argument from ``nondiff_argnums``.
        precision: Non-diff argument from ``nondiff_argnums``.
        residual: Saved tuple ``(lhs, rhs, group_sizes, group_offset,
            existing_out, rhs_scale, rhs_bias)``.
        grad: Upstream gradient with shape [m, n].

    Returns:
        Tuple of gradients ``(grad_lhs, grad_rhs, None, None, grad_existing_out,
        grad_rhs_scale, grad_rhs_bias)`` matching the differentiable positional
        args of ``_grouped_matmulv3_core``.
    """
    lhs, rhs, group_sizes, group_offset, existing_out, rhs_scale, rhs_bias = residual

    _, pullback = jax.vjp(
        lambda lhs, rhs: grouped_matmulv3_autodiff_reference(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type,
            tiling,
            group_offset,
            existing_out,
            rhs_scale,
            rhs_bias,
            transpose_rhs,
            interpret,
            precision,
        ),
        lhs,
        rhs,
    )
    grad_lhs, grad_rhs = pullback(grad)

    grad_existing_out = grad if existing_out is not None else None
    grad_rhs_scale = None
    grad_rhs_bias = None

    if rhs_scale is not None:
        _, pullback = jax.vjp(
            lambda scale: grouped_matmulv3_autodiff_reference(
                lhs,
                rhs,
                group_sizes,
                preferred_element_type,
                tiling,
                group_offset,
                existing_out,
                scale,
                rhs_bias,
                transpose_rhs,
                interpret,
                precision,
            ),
            rhs_scale,
        )
        (grad_rhs_scale,) = pullback(grad)

    if rhs_bias is not None:
        _, pullback = jax.vjp(
            lambda bias: grouped_matmulv3_autodiff_reference(
                lhs,
                rhs,
                group_sizes,
                preferred_element_type,
                tiling,
                group_offset,
                existing_out,
                rhs_scale,
                bias,
                transpose_rhs,
                interpret,
                precision,
            ),
            rhs_bias,
        )
        (grad_rhs_bias,) = pullback(grad)

    return grad_lhs, grad_rhs, None, None, grad_existing_out, grad_rhs_scale, grad_rhs_bias


_grouped_matmulv3_core.defvjp(_grouped_matmulv3_fwd, _grouped_matmulv3_bwd)


@kernel_registry.register("grouped_matmulv3", Platform.XLA, Backend.ANY)
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
    """Grouped Matrix Multiplication v3 with optional block-float scale and bias.

    Extends the base ``grouped_matmul`` with per-group ``rhs_scale`` and
    ``rhs_bias`` for block-float quantisation workflows.  Uses a custom VJP
    to ensure correct gradient flow through optional tensors.

    For each group ``i``:
        ``out[start_i:end_i, :] = (lhs @ rhs_dequant[i]) + rhs_bias[i]``
    where ``rhs_dequant[i]`` is ``rhs[i]`` dequantised by ``rhs_scale[i]``
    and group boundaries come from prefix sums of ``group_sizes``.

    When ``rhs_scale`` or ``rhs_bias`` is None the computation is equivalent
    to the base ``grouped_matmul`` XLA implementation.

    Args:
        lhs: Left-hand side matrix. Shape: [m, k].
        rhs: Per-group weight matrices.
            Shape: [num_groups, k, n] (or [num_groups, n, k] when
            ``transpose_rhs=True``).
        group_sizes: Number of ``lhs`` rows per group.
            Shape: [num_groups].  Must sum to ``m``.
        preferred_element_type: Accumulation and output dtype.
            Defaults to ``float32``.
        tiling: XLA tile-size hint as ``(tm, tk, tn)``, a ``LutFn``, or None.
            Ignored when ``rhs_scale`` / ``rhs_bias`` is provided (vmap path).
        group_offset: Optional scalar starting group index for sharded runs.
        existing_out: Optional [m, n] tensor to add to the result.
        rhs_scale: Optional block-float scale.
            Shape: [num_groups, num_blocks, 1, n].  ``num_blocks`` must evenly
            divide ``k`` (the inner dimension of ``rhs``).
        rhs_bias: Optional per-group output bias.
            Shape: [num_groups, 1, n].
        transpose_rhs: If True, ``rhs`` is [num_groups, n, k].
        interpret: Accepted for API compatibility; silently ignored.
        precision: JAX matmul precision.

    Returns:
        Output matrix of shape [m, n].

    Example:
        >>> lhs = jnp.ones((300, 64))
        >>> rhs = jnp.ones((3, 64, 32))
        >>> group_sizes = jnp.array([100, 150, 50], dtype=jnp.int32)
        >>> result = grouped_matmulv3(lhs, rhs, group_sizes)
        >>> result.shape
        (300, 32)
    """
    preferred_element_type = jnp.dtype(preferred_element_type) if preferred_element_type is not None else None
    return _grouped_matmulv3_core(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type,
        tiling,
        group_offset,
        existing_out,
        rhs_scale,
        rhs_bias,
        transpose_rhs,
        interpret,
        precision,
    )


__all__ = ("grouped_matmulv3", "grouped_matmulv3_reference")
