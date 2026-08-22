# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Quantized ``ragged_dot`` — the mixture-of-experts contraction.

A ragged dot multiplies expert-sorted token rows ``[M, K]`` against a
stack of per-expert weights ``[G, K, N]``, with ``group_sizes`` saying how
many consecutive rows belong to each expert. On a large mixture-of-experts
model this single op carries the overwhelming majority of the parameters,
so quantizing it is what makes quantized training of such a model mean
anything at all.

**Per-expert scales.** Experts specialize, and their weight magnitudes
diverge accordingly; one shared range flattens the quiet ones. So the
weight is quantized channelwise on *both* the group axis and the output
axis, giving a ``[G, 1, N]`` scale — one scale per expert per output
channel, and bit-for-bit the layout EasyDeL's serving path already
stores.

That choice is what makes this more than a transcription of Qwix, whose
``ragged_dot_qt`` quantizes the weight with ``channelwise_axes=[2]``
only — a ``[1, 1, N]`` scale shared across every expert. Qwix can then
fold the scale into the cotangent with a plain broadcast; per-expert
scales cannot broadcast, because which scale a row needs depends on which
expert that row was routed to. The fix is a row-to-expert gather, derived
from ``group_sizes`` once and reused by the forward and both gradients.

Rows past ``sum(group_sizes)`` belong to no expert. ``ragged_dot`` writes
zeros there, and zero times any scale is still zero, so the gather is
clamped into range rather than masked.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from ..core._typing import Array, DType
from . import _numerics as numerics
from ._calibrate import (
    HowToQuantize,
    calibrate,
    compute_scale_zero_point,
    dequantize,
    quantize,
    quantize_with_scale_zero_point,
)
from ._qarray import MaybeQArray, QArray, generic_broadcast_op
from ._rules import QuantRule

__all__ = ["qragged_dot"]


def _row_group_ids(group_sizes: Array, num_rows: int) -> Array:
    """Map each row to the expert it was routed to.

    Args:
        group_sizes: Rows per expert, shape ``[G]``.
        num_rows: Total rows in the operand, which may exceed
            ``sum(group_sizes)`` when the dispatch buffer is padded.

    Returns:
        An int32 array of shape ``[num_rows]`` giving each row's expert
        index, clamped to the last expert for padding rows.
    """
    boundaries = jnp.cumsum(group_sizes)
    rows = jnp.arange(num_rows, dtype=boundaries.dtype)
    return jnp.clip(jnp.searchsorted(boundaries, rows, side="right"), 0, group_sizes.shape[0] - 1).astype(jnp.int32)


def _gather_group_scale(scale: Array, row_groups: Array) -> Array:
    """Expand a per-expert scale ``[G, 1, N]`` to per-row ``[M, N]``.

    Args:
        scale: The weight's scale, shape ``[G, 1, N]``.
        row_groups: Each row's expert index, shape ``[M]``.

    Returns:
        The scale aligned with the output rows, shape ``[M, N]``.
    """
    return jnp.take(jnp.squeeze(scale, axis=1), row_groups, axis=0)


def _how_to_quantize_lhs(qtype: numerics.QType, rule: QuantRule) -> HowToQuantize:
    """Describe how to quantize the activation rows ``[M, K]``.

    Args:
        qtype: Target quantized type.
        rule: The governing rule.

    Returns:
        A slicing decision with a per-row scale, subchannel-tiled on the
        contracted axis when the rule asks for it.
    """
    return HowToQuantize(
        qtype=qtype,
        channelwise_axes=() if rule.disable_channelwise_axes else (0,),
        tiled_axes={1: rule.tile_size} if rule.tile_size else {},
        calibration_method=rule.act_calibration_method,
        power_of_two_scale=rule.power_of_two_scale,
    )


def _how_to_quantize_rhs(qtype: numerics.QType, rule: QuantRule) -> HowToQuantize:
    """Describe how to quantize the stacked expert weights ``[G, K, N]``.

    Args:
        qtype: Target quantized type.
        rule: The governing rule.

    Returns:
        A slicing decision giving one scale per expert per output channel.
    """
    return HowToQuantize(
        qtype=qtype,
        channelwise_axes=() if rule.disable_channelwise_axes else (0, 2),
        tiled_axes={1: rule.tile_size} if rule.tile_size else {},
        calibration_method=rule.weight_calibration_method,
        power_of_two_scale=rule.power_of_two_scale,
    )


def _is_output_dequantizable(operand: MaybeQArray) -> bool:
    """Whether an operand permits contracting in the quantized type.

    Args:
        operand: A quantized or plain array.

    Returns:
        ``True`` when the operand is a quantized affine type whose
        contracted axis carries a single shared scale. A subchannel-tiled
        contraction cannot factor out of the sum, and a code book such as
        ``nf4`` has no arithmetic meaning before dequantization.
    """
    if not isinstance(operand, QArray):
        return not numerics.should_quantize(operand.dtype)
    if not numerics.can_dequantize_on_output(operand.qtype):
        return False
    return operand.scale.shape[1] == 1


def _ragged_dot_maybe_quantized(
    lhs: MaybeQArray,
    rhs: MaybeQArray,
    group_sizes: Array,
    *,
    precision: jax.lax.PrecisionLike,
    preferred_element_type: DType | None,
    group_offset: Array | None,
) -> Array:
    """Contract expert-sorted rows against stacked expert weights.

    Contracts in the quantized type and rescales the (much smaller) result
    when both operands allow it, and otherwise reconstructs the floats
    first and lets XLA fuse the dequantize into the matmul.

    Args:
        lhs: Token rows ``[M, K]``, quantized or plain.
        rhs: Stacked expert weights ``[G, K, N]``, quantized or plain.
        group_sizes: Rows per expert, shape ``[G]``.
        precision: Forwarded to :func:`jax.lax.ragged_dot`.
        preferred_element_type: Requested output dtype.
        group_offset: Forwarded to :func:`jax.lax.ragged_dot`.

    Returns:
        The contraction result ``[M, N]``, in floating point.
    """
    if not (_is_output_dequantizable(lhs) and _is_output_dequantizable(rhs)):
        lhs_float = dequantize(lhs) if isinstance(lhs, QArray) else lhs
        rhs_float = dequantize(rhs) if isinstance(rhs, QArray) else rhs
        return jax.lax.ragged_dot(
            lhs_float,
            rhs_float,
            group_sizes,
            precision=precision,
            preferred_element_type=preferred_element_type,
            group_offset=group_offset,
        )

    lhs_value = lhs.qvalue if isinstance(lhs, QArray) else lhs
    rhs_value = rhs.qvalue if isinstance(rhs, QArray) else rhs
    result_type = preferred_element_type
    if result_type is None:
        result_type = jnp.bfloat16
        for operand in (lhs, rhs):
            if isinstance(operand, QArray):
                result_type = jnp.result_type(result_type, operand.scale.dtype)
    accumulator = jnp.int32 if all("int" in jnp.dtype(v.dtype).name for v in (lhs_value, rhs_value)) else result_type

    out = jax.lax.ragged_dot(
        lhs_value,
        rhs_value,
        group_sizes,
        precision=precision,
        preferred_element_type=accumulator,
        group_offset=group_offset,
    ).astype(result_type)

    if isinstance(lhs, QArray):
        out = out * lhs.scale.astype(result_type)
    if isinstance(rhs, QArray):
        row_groups = _row_group_ids(group_sizes, lhs_value.shape[0])
        out = out * _gather_group_scale(rhs.scale, row_groups).astype(result_type)
    return out


def _quantize_across_axis(array: Array, how: HowToQuantize, axis_name: str | None) -> QArray:
    """Quantize an array whose contracted axis may be sharded.

    Inside a ``shard_map`` a local reduction over a sharded contraction
    sees one shard of what is logically a single tensor, so every rank
    would derive a different scale for the same weight. Reducing the
    calibration across the axis first restores the global statistic and
    keeps a quantization-aware training run matched to the single-scale
    layout it will later be served with.

    Args:
        array: The array to quantize.
        how: The slicing decision.
        axis_name: Mesh axis the contracted dimension is sharded over, or
            ``None`` when the calibration is already global.

    Returns:
        The quantized array.
    """
    if axis_name is None:
        return quantize(array, how)
    calibration = calibrate(array, how)
    calibration = jax.tree.map(lambda value: jax.lax.pmax(value, axis_name), calibration)
    scale, zero_point = compute_scale_zero_point(calibration, how.qtype, power_of_two_scale=how.power_of_two_scale)
    return quantize_with_scale_zero_point(array, how.qtype, scale, zero_point, how.noise_fn)


def _ragged_dot_qt_fwd(
    lhs: Array,
    rhs: Array,
    group_sizes: Array,
    rule: QuantRule,
    precision: jax.lax.PrecisionLike,
    preferred_element_type: DType | None,
    group_offset: Array | None,
    calibration_axis_name: str | None,
) -> tuple[Array, tuple[MaybeQArray, MaybeQArray, Array]]:
    """Quantize both operands, contract them, and save them as residuals.

    Args:
        lhs: Token rows ``[M, K]``.
        rhs: Stacked expert weights ``[G, K, N]``.
        group_sizes: Rows per expert.
        rule: The governing rule.
        precision: Forwarded to :func:`jax.lax.ragged_dot`.
        preferred_element_type: Requested output dtype.
        group_offset: Forwarded to :func:`jax.lax.ragged_dot`.
        calibration_axis_name: Mesh axis the contracted dimension is
            sharded over, or ``None``.

    Returns:
        ``(result, residuals)`` in the shape :func:`jax.custom_vjp` wants.
    """
    quantized_lhs: MaybeQArray = lhs
    quantized_rhs: MaybeQArray = rhs
    if rule.act_qtype is not None and numerics.should_quantize(lhs.dtype):
        quantized_lhs = _quantize_across_axis(
            lhs, _how_to_quantize_lhs(rule.act_qtype, rule), calibration_axis_name
        )
    if rule.weight_qtype is not None and numerics.should_quantize(rhs.dtype):
        quantized_rhs = _quantize_across_axis(
            rhs, _how_to_quantize_rhs(rule.weight_qtype, rule), calibration_axis_name
        )

    out = _ragged_dot_maybe_quantized(
        quantized_lhs,
        quantized_rhs,
        group_sizes,
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )
    return out, (quantized_lhs, quantized_rhs, group_sizes)


def _ragged_dot_qt_bwd(
    rule: QuantRule,
    precision: jax.lax.PrecisionLike,
    preferred_element_type: DType | None,
    group_offset: Array | None,
    calibration_axis_name: str | None,
    residuals: tuple[MaybeQArray, MaybeQArray, Array],
    cotangent: Array,
) -> tuple[Array, Array, None]:
    """Compute both operand gradients against the saved quantized residuals.

    Each gradient folds the *other* operand's scale into the cotangent
    before contracting, so the residual can be used as raw quantized
    values. For the weight's scale that fold is a row gather, because the
    scale a row needs depends on which expert the row was routed to.

    Args:
        rule: The governing rule.
        precision: Forwarded to the ragged dots.
        preferred_element_type: Requested output dtype.
        group_offset: Forwarded to the ragged dots.
        calibration_axis_name: Mesh axis the contracted dimension is
            sharded over, or ``None``.
        residuals: The quantized operands and group sizes from the forward.
        cotangent: The incoming gradient ``[M, N]``.

    Returns:
        Cotangents for ``(lhs, rhs, group_sizes)``; ``group_sizes`` is an
        integer routing array and gets ``None``.
    """
    lhs, rhs, group_sizes = residuals
    num_rows = lhs.shape[0]

    # --- gradient of the token rows: dlhs[m, k] = sum_n g[m, n] * rhs[e(m), k, n]
    lhs_cotangent = cotangent
    rhs_operand: MaybeQArray = rhs
    if isinstance(rhs, QArray):
        row_groups = _row_group_ids(group_sizes, num_rows)
        lhs_cotangent = lhs_cotangent * _gather_group_scale(rhs.scale, row_groups).astype(lhs_cotangent.dtype)
        rhs_operand = rhs.qvalue
    # ``[G, K, N] -> [G, N, K]``: the backward contracts the cotangent's N
    # against the weight's N. A QArray swaps every component in lock step,
    # so this is the same call either way.
    rhs_transposed = rhs_operand.swapaxes(1, 2)

    if rule.bwd_qtype is not None and numerics.should_quantize(lhs_cotangent.dtype):
        lhs_cotangent = quantize(
            lhs_cotangent,
            HowToQuantize(
                qtype=rule.bwd_qtype,
                channelwise_axes=() if rule.disable_channelwise_axes else (0,),
                calibration_method=rule.bwd_calibration_method,
            ),
        )
    dlhs = _ragged_dot_maybe_quantized(
        lhs_cotangent,
        rhs_transposed,
        group_sizes,
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )

    # --- gradient of the expert weights: drhs[e, k, n] = sum_{m in e} lhs[m, k] * g[m, n]
    rhs_cotangent = cotangent
    lhs_operand: MaybeQArray = lhs
    if isinstance(lhs, QArray):
        rhs_cotangent = generic_broadcast_op(jnp.multiply, rhs_cotangent, lhs.scale.astype(rhs_cotangent.dtype))
        lhs_operand = lhs.qvalue

    if rule.bwd_qtype is not None and numerics.should_quantize(rhs_cotangent.dtype):
        rhs_cotangent = dequantize(
            quantize(
                rhs_cotangent,
                HowToQuantize(
                    qtype=rule.bwd_qtype,
                    channelwise_axes=() if rule.disable_channelwise_axes else (1,),
                    tiled_axes={0: rule.bwd_weight_grad_tile_size} if rule.bwd_weight_grad_tile_size else {},
                    calibration_method=rule.bwd_calibration_method,
                ),
            )
        )
    if isinstance(lhs_operand, QArray):
        lhs_operand = dequantize(lhs_operand)
    elif lhs_operand.dtype != rhs_cotangent.dtype:
        lhs_operand = lhs_operand.astype(rhs_cotangent.dtype)

    drhs = jax.lax.ragged_dot_general(
        lhs_operand,
        rhs_cotangent,
        group_sizes,
        ragged_dot_dimension_numbers=jax.lax.RaggedDotDimensionNumbers(
            dot_dimension_numbers=(((0,), (0,)), ((), ())),
            lhs_ragged_dimensions=[0],
            rhs_group_dimensions=[],
        ),
        precision=precision,
        preferred_element_type=preferred_element_type,
        group_offset=group_offset,
    )
    return dlhs, drhs, None


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7))
def _ragged_dot_qt(
    lhs: Array,
    rhs: Array,
    group_sizes: Array,
    rule: QuantRule,
    precision: jax.lax.PrecisionLike,
    preferred_element_type: DType | None,
    group_offset: Array | None,
    calibration_axis_name: str | None,
) -> Array:
    """Quantized ragged dot with a custom, optionally quantized backward.

    Args:
        lhs: Token rows ``[M, K]``.
        rhs: Stacked expert weights ``[G, K, N]``.
        group_sizes: Rows per expert.
        rule: The governing rule.
        precision: Forwarded to :func:`jax.lax.ragged_dot`.
        preferred_element_type: Requested output dtype.
        group_offset: Forwarded to :func:`jax.lax.ragged_dot`.
        calibration_axis_name: Mesh axis the contracted dimension is
            sharded over, or ``None``.

    Returns:
        The contraction result ``[M, N]``.
    """
    out, _residuals = _ragged_dot_qt_fwd(
        lhs, rhs, group_sizes, rule, precision, preferred_element_type, group_offset, calibration_axis_name
    )
    return out


_ragged_dot_qt.defvjp(_ragged_dot_qt_fwd, _ragged_dot_qt_bwd)


def qragged_dot(
    lhs: Array,
    rhs: Array,
    group_sizes: Array,
    *,
    rule: QuantRule | None,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: DType | None = None,
    group_offset: Array | None = None,
    calibration_axis_name: str | None = None,
) -> Array:
    """Run a mixture-of-experts ragged dot under a quantization rule.

    Falls straight through to :func:`jax.lax.ragged_dot` when ``rule`` is
    ``None`` or does not quantize weights, so a call site can be written
    once and stay exact for unquantized models.

    The weight operand is always the stacked expert kernel and the left
    operand always the activation rows, which is what the op means; unlike
    :func:`~spectrax.quantization.qdot_general` there is nothing to
    declare.

    Args:
        lhs: Expert-sorted token rows ``[M, K]``.
        rhs: Stacked expert weights ``[G, K, N]``.
        group_sizes: Rows per expert, shape ``[G]``.
        rule: The governing rule, or ``None`` for full precision.
        precision: Forwarded to :func:`jax.lax.ragged_dot`.
        preferred_element_type: Requested output dtype.
        group_offset: Forwarded to :func:`jax.lax.ragged_dot`.
        calibration_axis_name: Mesh axis the contracted dimension is sharded
            over, when this runs inside a ``shard_map``. Without it each rank
            calibrates against its own shard and derives a different scale
            for what is logically one tensor. Passed as a name rather than a
            callable so it stays hashable and does not defeat the jit cache.

    Returns:
        The contraction result ``[M, N]``.

    Raises:
        ValueError: If the operands do not have the ragged-dot ranks.
    """
    if rule is None or rule.weight_qtype is None:
        return jax.lax.ragged_dot(
            lhs,
            rhs,
            group_sizes,
            precision=precision,
            preferred_element_type=preferred_element_type,
            group_offset=group_offset,
        )
    if lhs.ndim != 2 or rhs.ndim != 3:
        raise ValueError(
            f"qragged_dot expects token rows [M, K] and stacked experts [G, K, N], got {lhs.shape} and {rhs.shape}."
        )
    if rule.bwd_stochastic_rounding is not None:
        raise ValueError("Stochastic rounding is not implemented for ragged_dot; use bwd_stochastic_rounding=None.")
    return _ragged_dot_qt(
        lhs, rhs, group_sizes, rule, precision, preferred_element_type, group_offset, calibration_axis_name
    )
