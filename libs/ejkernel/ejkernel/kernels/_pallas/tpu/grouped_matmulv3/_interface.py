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

"""Grouped Matrix Multiplication v3 for TPU using upstream-style Pallas kernels."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import jaxtyping
from beartype import beartype
from jaxtyping import Array, DTypeLike, Float, Int

from ...._registry import Backend, Platform, kernel_registry
from ...._xla.grouped_matmulv3._interface import _apply_rhs_scale_bias, grouped_matmulv3_reference
from ..grouped_matmul._pallas_impl import LutFn
from ..grouped_matmulv2._pallas_impl import grouped_matmul as back_grouped_matmul
from ..grouped_matmulv2._pallas_impl import transposed_grouped_matmul as back_tgrouped_matmul
from ._pallas_impl import TileSizes, calculate_tiling, grouped_matmulv3_pallas_impl


def _normalize_tiling(
    tiling: tuple[int, int, int] | LutFn | None,
    lhs: jax.Array,
    rhs: jax.Array,
) -> TileSizes | Callable:
    """Normalize ejkernel-style tiling spec into the v3 ``TileSizes | TileFn`` format.

    The ejkernel public API accepts ``(tm, tk, tn)`` tuples or ``LutFn``
    callables with signature ``(m, k, n) -> (tm, tk, tn) | None``.  The v3
    implementation expects either a ``TileSizes`` dataclass or a ``TileFn``
    with signature ``(dims, lhs_cfgs, rhs_cfgs, vmem_limit, fuse_act)``.
    ``None`` maps to the automatic ``calculate_tiling`` heuristic.

    Args:
        tiling: Tile spec in ejkernel format: a ``(tm, tk, tn)`` tuple, a
            ``LutFn`` callable, or ``None`` to use automatic tiling.
        lhs: LHS array (used only for type annotations; not inspected here).
        rhs: RHS array (used only for type annotations; not inspected here).

    Returns:
        A ``TileSizes`` instance or a ``TileFn`` callable compatible with
        ``grouped_matmulv3_pallas_impl``.
    """
    if tiling is None:
        return calculate_tiling
    if isinstance(tiling, tuple):
        return TileSizes(tile_m=int(tiling[0]), tile_k=int(tiling[1]), tile_n=int(tiling[2]))

    def _wrapped_tile_fn(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act):
        tile = tiling(dims.size_m, dims.size_k, dims.size_n)
        if tile is None:
            return calculate_tiling(dims, lhs_cfgs, rhs_cfgs, vmem_limit_bytes, fuse_act)
        return TileSizes(tile_m=int(tile[0]), tile_k=int(tile[1]), tile_n=int(tile[2]))

    return _wrapped_tile_fn


def _call_grouped_matmulv3(
    lhs: jax.Array,
    rhs: jax.Array,
    group_sizes: jax.Array,
    preferred_element_type: DTypeLike,
    tiling: tuple[int, int, int] | LutFn | None,
    group_offset: jax.Array | None,
    rhs_scale: jax.Array | None,
    rhs_bias: jax.Array | None,
    transpose_rhs: bool,
    interpret: bool,
    precision: jax.lax.PrecisionLike,
) -> tuple[jax.Array, jax.Array]:
    """Run the v3 Pallas kernel and return the output and the pre-processed RHS.

    Applies ``transpose_rhs`` by swapping axes 1 and 2 of ``rhs``, then
    delegates to ``grouped_matmulv3_pallas_impl``.

    Args:
        lhs: Token features ``[m, k]``.
        rhs: Per-group weights in ejkernel layout (before axis swap).
        group_sizes: Per-group token counts, int32 ``[num_groups]``.
        preferred_element_type: Output dtype.
        tiling: Tile spec (see ``_normalize_tiling``).
        group_offset: Scalar shard offset array or None.
        rhs_scale: Optional quantisation scale ``[num_groups, num_blocks, 1, n]``.
        rhs_bias: Optional per-group bias ``[num_groups, 1, n]``.
        transpose_rhs: Whether to swap axes 1/2 of ``rhs`` before the kernel.
        interpret: Run in Pallas interpreter mode.
        precision: Ignored; present for API consistency.

    Returns:
        ``(out, rhs_prepped)`` where ``rhs_prepped`` is the axis-swapped RHS
        stored for use in the backward pass.
    """
    del precision
    rhs_prepped = rhs.swapaxes(1, 2) if transpose_rhs else rhs
    out = grouped_matmulv3_pallas_impl(
        lhs=lhs,
        rhs=rhs_prepped,
        group_sizes=group_sizes,
        rhs_scale=rhs_scale,
        rhs_bias=rhs_bias,
        group_offset=group_offset,
        tile_info=_normalize_tiling(tiling, lhs, rhs_prepped),
        preferred_element_type=preferred_element_type,
        interpret=interpret,
    )
    return out, rhs_prepped


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
    """Custom-VJP wrapper for the v3 grouped matmul.

    This function is decorated with ``jax.custom_vjp`` so that
    ``_grouped_matmulv3_fwd`` / ``_grouped_matmulv3_bwd`` are used during
    automatic differentiation instead of JAX's default AD rules.
    Non-differentiable static configuration is captured via
    ``nondiff_argnums=(3, 4, 9, 10, 11)``.

    Args:
        lhs: Token features ``[m, k]``.
        rhs: Per-group weights (ejkernel layout).
        group_sizes: Token counts per group, int32 ``[num_groups]``.
        preferred_element_type: Output dtype (non-differentiable).
        tiling: Tile spec (non-differentiable).
        group_offset: Shard offset, or None.
        existing_out: Optional array to add to the output.
        rhs_scale: Optional quantisation scale.
        rhs_bias: Optional per-group bias.
        transpose_rhs: Whether to transpose RHS (non-differentiable).
        interpret: Interpreter mode flag (non-differentiable).
        precision: Precision hint (non-differentiable).

    Returns:
        Output tensor ``[m, n]``.
    """
    out, _ = _call_grouped_matmulv3(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type,
        tiling,
        group_offset,
        rhs_scale,
        rhs_bias,
        transpose_rhs,
        interpret,
        precision,
    )
    return out if existing_out is None else out + existing_out


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
    """Forward rule for the v3 grouped matmul custom VJP.

    Executes the v3 kernel and stashes residuals required by the backward pass.
    Residuals stored: ``(lhs, rhs, rhs_prepped, group_sizes, group_offset,
    rhs_scale, rhs_bias, rhs.shape[0], has_existing_out)`` where
    ``rhs_prepped`` is the potentially axis-swapped weight used by the kernel.

    Returns:
        ``(out, residuals)`` compatible with ``jax.custom_vjp`` protocol.
    """
    out, rhs_prepped = _call_grouped_matmulv3(
        lhs,
        rhs,
        group_sizes,
        preferred_element_type,
        tiling,
        group_offset,
        rhs_scale,
        rhs_bias,
        transpose_rhs,
        interpret,
        precision,
    )
    if existing_out is not None:
        out = out + existing_out
    return out, (
        lhs,
        rhs,
        rhs_prepped,
        group_sizes,
        group_offset,
        rhs_scale,
        rhs_bias,
        rhs.shape[0],
        existing_out is not None,
    )


def _grouped_matmulv3_bwd(
    preferred_element_type: DTypeLike,
    tiling: tuple[int, int, int] | LutFn | None,
    transpose_rhs: bool,
    interpret: bool,
    precision: jax.lax.PrecisionLike,
    residual,
    grad: jax.Array,
):
    """Backward rule for the v3 grouped matmul custom VJP.

    Computes gradients for all differentiable arguments:
    - ``grad_lhs``: uses the v2 ``back_grouped_matmul`` kernel with
      ``transpose_rhs=not transpose_rhs``.
    - ``grad_rhs``: uses the v2 ``back_tgrouped_matmul`` kernel.  If
      ``rhs_scale`` was provided the raw gradient is further multiplied by
      the expanded scale.
    - ``grad_rhs_scale`` and ``grad_rhs_bias``: computed via a full
      reference re-run through ``grouped_matmulv3_reference`` and JAX VJP.
      This path is slower but ensures numerical correctness.
    - ``grad_existing_out``: equal to ``grad`` when ``existing_out`` was
      provided (additive accumulation), or None otherwise.

    Non-differentiable static args ``(preferred_element_type, tiling,
    transpose_rhs, interpret, precision)`` are captured via
    ``nondiff_argnums``.

    Returns:
        7-tuple ``(grad_lhs, grad_rhs, None, None, grad_existing_out,
        grad_rhs_scale, grad_rhs_bias)`` matching the differentiable
        arguments of ``_grouped_matmulv3_core``.
    """
    (
        lhs,
        rhs,
        rhs_prepped,
        group_sizes,
        group_offset,
        rhs_scale,
        rhs_bias,
        num_actual_groups,
        has_existing_out,
    ) = residual

    rhs_effective, _ = _apply_rhs_scale_bias(
        rhs,
        rhs_scale,
        None,
        transpose_rhs=transpose_rhs,
    )
    resolved_tiling = tiling if isinstance(tiling, tuple) else (128, 128, 128)
    grad_lhs = back_grouped_matmul(
        grad,
        rhs_effective,
        group_sizes,
        lhs[0].dtype,
        resolved_tiling,
        input_buffer_count=2,
        group_offset=group_offset,
        transpose_rhs=not transpose_rhs,
        interpret=interpret,
    )
    grad_rhs = back_tgrouped_matmul(
        lhs.swapaxes(0, 1),
        grad,
        group_sizes,
        rhs_prepped.dtype,
        resolved_tiling,
        group_offset=group_offset,
        num_actual_groups=num_actual_groups,
        interpret=interpret,
    )
    if rhs_scale is not None:
        block_size = rhs_prepped.shape[1] // rhs_scale.shape[1]
        scale = jnp.repeat(rhs_scale[:, :, 0, :], block_size, axis=1)
        grad_rhs = grad_rhs * scale.astype(grad_rhs.dtype)
    grad_rhs = grad_rhs.swapaxes(1, 2) if transpose_rhs else grad_rhs

    grad_rhs_scale = None
    grad_rhs_bias = None
    if rhs_scale is not None:
        _, pullback = jax.vjp(
            lambda scale: grouped_matmulv3_reference(
                lhs,
                rhs,
                group_sizes,
                preferred_element_type,
                tiling,
                group_offset,
                None,
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
            lambda bias: grouped_matmulv3_reference(
                lhs,
                rhs,
                group_sizes,
                preferred_element_type,
                tiling,
                group_offset,
                None,
                rhs_scale,
                bias,
                transpose_rhs,
                interpret,
                precision,
            ),
            rhs_bias,
        )
        (grad_rhs_bias,) = pullback(grad)

    grad_existing_out = grad if has_existing_out else None
    return grad_lhs, grad_rhs, None, None, grad_existing_out, grad_rhs_scale, grad_rhs_bias


_grouped_matmulv3_core.defvjp(_grouped_matmulv3_fwd, _grouped_matmulv3_bwd)


@kernel_registry.register("grouped_matmulv3", Platform.PALLAS, Backend.TPU)
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
    """Grouped Matrix Multiplication v3 on TPU using the upstream emit_pipeline kernel.

    Performs the same per-group batched matmul as v1/v2 but with the v3 kernel
    that uses ``pltpu.emit_pipeline`` for pipelined execution, metadata-driven
    tile scheduling, and optional fused per-group RHS scale / bias for
    quantised weight formats.

    For each group ``i`` with token slice ``[s_i, s_i + g_i)``:

    .. code-block::

        out[s_i:s_i+g_i, :] = lhs[s_i:s_i+g_i, :] @ rhs_effective[i, :, :]

    where ``rhs_effective`` is derived from ``rhs``, ``rhs_scale``, and
    ``rhs_bias`` via ``_apply_rhs_scale_bias``.

    Args:
        lhs: Token features ``[m, k]``.
        rhs: Per-group weight tensor ``[num_groups, k, n]``, or
            ``[num_groups, n, k]`` when ``transpose_rhs=True``.
        group_sizes: Number of tokens per group, shape ``[num_groups_or_shards]``,
            dtype int32.
        preferred_element_type: Output dtype.  Defaults to float32.
        tiling: ``(tm, tk, tn)`` tile sizes, or a ``LutFn`` callable with
            signature ``(m, k, n) -> (tm, tk, tn) | None``.  ``None`` falls
            back to the v3 automatic tiling heuristic (``calculate_tiling``).
        group_offset: Scalar array giving the first active group index for
            sharded execution.  Defaults to 0.
        existing_out: If provided, the kernel output is added to this array
            element-wise before returning.  Shape ``[m, n]``.
        rhs_scale: Optional per-group block-wise scale tensor
            ``[num_groups, num_blocks, 1, n]``.  Used for quantised weights.
        rhs_bias: Optional per-group additive bias ``[num_groups, 1, n]``.
            Applied after scaling on the last k-step.
        transpose_rhs: If True, ``rhs`` has shape ``[num_groups, n, k]`` and
            the last two dimensions are swapped before the kernel runs.
        interpret: Run the Pallas kernel in interpreter mode for debugging.
            Significantly slower; do not use in production.
        precision: JAX precision hint.  Currently ignored by the v3 kernel
            (the kernel always computes in the dtype implied by the inputs and
            ``preferred_element_type``).

    Returns:
        Output tensor ``[m, n]`` of dtype ``preferred_element_type``.

    Note:
        - The backward pass falls back to the v2 Pallas kernel for ``grad_lhs``
          and ``grad_rhs``.  Gradients for ``rhs_scale`` and ``rhs_bias`` are
          computed via a full reference re-run through ``grouped_matmulv3_reference``
          which is slower.
        - ``precision`` is accepted but has no effect; pass it for API
          compatibility only.
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


__all__ = ("grouped_matmulv3",)
