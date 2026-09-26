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
    - ``rhs_scale`` is folded into the ``[num_groups, k, n]`` weight before a
      single ``ragged_dot_general``; ``rhs_bias`` is added per row afterwards.
      ``group_offset`` is honoured by slicing the active ``group_sizes`` window
      (``ragged_dot_general`` itself does not implement ``group_offset``).
    - Rows past ``sum(active group_sizes)`` belong to no group: they are zero in
      the forward (no bias) and contribute nothing to any gradient.

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
        if num_blocks <= 0:
            raise ValueError("rhs_scale must have at least one quant block")
        if size_k % num_blocks != 0:
            raise ValueError("rhs.shape[1] must be divisible by rhs_scale.shape[1].")
        block_size = size_k // num_blocks
        scale = jnp.repeat(rhs_scale[:, :, 0, :], block_size, axis=1)
        if jnp.issubdtype(rhs_prepped.dtype, jnp.integer):
            # Fractional scales must not be truncated into the integer codes.
            # This path also supplies the differentiable backward reference.
            rhs_prepped = rhs_prepped.astype(jnp.float32)
        rhs_prepped = rhs_prepped * scale.astype(rhs_prepped.dtype)

    if rhs_bias is not None:
        if rhs_bias.ndim != 3 or rhs_bias.shape[1] != 1:
            raise ValueError("rhs_bias must have shape [num_groups, 1, n].")
        if rhs_bias.shape[0] != rhs_prepped.shape[0] or rhs_bias.shape[2] != rhs_prepped.shape[2]:
            raise ValueError("rhs_bias group/out dimensions must match rhs.")
        bias = rhs_bias[:, 0, :]

    return rhs_prepped, bias


def _active_group_sizes(
    group_sizes: jax.Array,
    num_groups: int,
    group_offset: jax.Array | None,
) -> jax.Array:
    """Return the ``num_groups`` group sizes starting at ``group_offset``.

    Rows are assigned to the active groups from row 0 onward (groups before
    ``group_offset`` do not shift the row offsets), matching the Pallas v3
    kernel's metadata.

    Args:
        group_sizes: Per-group row counts. Shape: [num_groups_or_shards].
        num_groups: Number of active groups to process (typically
            ``rhs.shape[0]``).
        group_offset: Optional scalar (or 1-element array) indicating the
            starting offset into ``group_sizes`` for sharded execution.

    Returns:
        Integer array of shape [num_groups].
    """
    offset = (
        group_offset.reshape(-1)[0].astype(group_sizes.dtype)
        if group_offset is not None
        else jnp.array(0, dtype=group_sizes.dtype)
    )
    return jax.lax.dynamic_slice_in_dim(group_sizes, offset, num_groups, axis=0)


def _active_group_ids(
    group_sizes: jax.Array,
    num_groups: int,
    total_rows: int,
    group_offset: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    """Build a per-row group-index vector and a row-validity mask.

    For each row in ``lhs[0:total_rows]`` returns the index of the group it
    belongs to. Rows past ``sum(active group sizes)`` belong to no group:
    ``jnp.repeat`` would assign them to the last group, so callers must mask
    them with the returned ``row_valid``.

    Args:
        group_sizes: Per-group row counts. Shape: [num_groups_or_shards].
        num_groups: Number of active groups to process (typically
            ``rhs.shape[0]``).
        total_rows: Total number of ``lhs`` rows (``m``).
        group_offset: Optional scalar (or 1-element array) indicating the
            starting offset into ``group_sizes`` for sharded execution.

    Returns:
        ``(group_ids, row_valid)``: integer ``[total_rows]`` group index per
        row and boolean ``[total_rows]`` mask that is False for tail rows.
    """
    active_sizes = _active_group_sizes(group_sizes, num_groups, group_offset)
    group_ids = jnp.repeat(
        jnp.arange(num_groups, dtype=group_sizes.dtype),
        active_sizes,
        total_repeat_length=total_rows,
    )
    return group_ids, _tail_row_mask(active_sizes, total_rows)


def _tail_row_mask(active_sizes: jax.Array, total_rows: int) -> jax.Array:
    """Boolean ``[total_rows]`` mask, False for rows past ``sum(active_sizes)``."""
    return jnp.arange(total_rows, dtype=active_sizes.dtype) < jnp.sum(active_sizes)


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
    """Pure-JAX per-row vmap reference for GMM v3 (testing / parity only).

    Gathers ``rhs[group_id]`` per row, i.e. materialises an ``[m, k, n]``
    tensor, so it is only suitable for small shapes. Production paths use
    :func:`grouped_matmulv3_reference` (a single ``ragged_dot_general``).
    Rows past ``sum(active group_sizes)`` produce zero output and gradient.

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
    group_ids, row_valid = _active_group_ids(group_sizes, rhs_prepped.shape[0], lhs.shape[0], group_offset)
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
    out = jnp.where(row_valid[:, None], out, jnp.zeros((), dtype=out.dtype))
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

    ``rhs_scale`` is folded into the ``[num_groups, k, n]`` weight, then one
    ``ragged_dot_general`` runs over the active ``group_sizes`` window (sliced
    at ``group_offset``, which ``ragged_dot_general`` does not implement
    itself). The whole function is differentiable by standard JAX autodiff
    w.r.t. ``lhs``, ``rhs``, ``rhs_scale`` and ``rhs_bias`` without ever
    materialising a per-row ``[m, k, n]`` weight gather.

    After the core matmul, any ``rhs_bias`` (valid rows only) and
    ``existing_out`` are added in the output dtype. Rows past
    ``sum(active group_sizes)`` are zero (before ``existing_out``).

    Args:
        lhs: [m, k] left-hand side matrix.
        rhs: [num_groups, k, n] (or [num_groups, n, k]) weight matrices.
        group_sizes: [num_groups] per-group row counts.
        preferred_element_type: Output dtype.
        tiling: Tile-size hint for ``ragged_dot_general`` via XLA metadata
            (tuple form, forwarded only when neither scale nor bias is set).
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
    rhs_prepped, bias = _apply_rhs_scale_bias(
        rhs,
        rhs_scale,
        rhs_bias,
        transpose_rhs=transpose_rhs,
    )
    lhs_mm = lhs
    if rhs_scale is not None:
        # The dequantised weight is floating (fp32 for integer codes); match the
        # promotion ``jnp.matmul`` would apply instead of relying on mixed-dtype
        # ragged_dot lowering.
        common_dtype = jnp.promote_types(lhs.dtype, rhs_prepped.dtype)
        lhs_mm = lhs.astype(common_dtype)
        rhs_prepped = rhs_prepped.astype(common_dtype)
    num_groups = rhs_prepped.shape[0]
    active_sizes = _active_group_sizes(group_sizes, num_groups, group_offset)
    # Forward the XLA tile hint only on the plain path (the only one that used ragged_dot before):
    # the scale/bias path is also reached from the Pallas backward with Pallas-tuned (or callable)
    # tiling, which is not a valid ragged_dot hint for arbitrary m.
    plain = rhs_scale is None and rhs_bias is None
    ragged_tiling = tiling if plain and isinstance(tiling, tuple) else None
    out = _grouped_matmul_impl(
        lhs_mm,
        rhs_prepped,
        active_sizes,
        preferred_element_type,
        ragged_tiling,
        None,
        existing_out=None,
        transpose_rhs=False,
        interpret=interpret,
        precision=precision,
    )
    if bias is not None:
        group_ids, _ = _active_group_ids(group_sizes, num_groups, lhs.shape[0], group_offset)
        out = out + bias[group_ids].astype(out.dtype)
    # Rows past ``sum(active_sizes)`` belong to no group. ragged_dot leaves them unspecified on TPU
    # (not guaranteed zero), and ``bias[group_ids]`` would give them the last group's bias: zero them.
    out = jnp.where(_tail_row_mask(active_sizes, lhs.shape[0])[:, None], out, jnp.zeros((), dtype=out.dtype))
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
        - ``grad_lhs``, ``grad_rhs``, ``grad_rhs_scale``, ``grad_rhs_bias``:
          one ``jax.vjp`` through the same ``ragged_dot_general`` forward
          (:func:`grouped_matmulv3_reference`), so forward and backward agree
          on every row, including tail rows past ``sum(group_sizes)`` (zero).
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
        lambda lhs, rhs, scale, bias: grouped_matmulv3_reference(
            lhs,
            rhs,
            group_sizes,
            preferred_element_type,
            tiling,
            group_offset,
            existing_out,
            scale,
            bias,
            transpose_rhs,
            interpret,
            precision,
        ),
        lhs,
        rhs,
        rhs_scale,
        rhs_bias,
    )
    grad_lhs, grad_rhs, grad_rhs_scale, grad_rhs_bias = pullback(grad)
    # grad_lhs is itself a ragged_dot output over the same groups: zero its (unspecified) tail rows.
    active_sizes = _active_group_sizes(group_sizes, rhs.shape[0], group_offset)
    grad_lhs = jnp.where(
        _tail_row_mask(active_sizes, lhs.shape[0])[:, None], grad_lhs, jnp.zeros((), dtype=grad_lhs.dtype)
    )

    grad_existing_out = grad if existing_out is not None else None
    return grad_lhs, grad_rhs, None, None, grad_existing_out, grad_rhs_scale, grad_rhs_bias


_grouped_matmulv3_core.defvjp(_grouped_matmulv3_fwd, _grouped_matmulv3_bwd)


@kernel_registry.register("grouped_matmulv3", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
def grouped_matmulv3(
    lhs: Float[Array, "m k"],
    rhs: (
        Float[Array, "num_groups k n"]
        | Float[Array, "num_groups n k"]
        | Int[Array, "num_groups k n"]
        | Int[Array, "num_groups n k"]
    ),
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
            Only a tuple hint is forwarded, and only when ``rhs_scale`` /
            ``rhs_bias`` are both None.
        group_offset: Optional scalar starting group index for sharded runs:
            ``lhs`` rows are assigned to groups
            ``group_offset .. group_offset + num_groups - 1`` from row 0.
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
