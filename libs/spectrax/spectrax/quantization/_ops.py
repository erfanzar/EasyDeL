# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The quantization-aware ops layers call.

Two tiers, and the difference between them is what a layer gives up.

:func:`fake_quant` — **tier 1.** Round an array onto the quantized grid
and immediately reconstruct it, with an identity gradient. The matmul
that follows is whatever the layer already used: an ejkernel Pallas
kernel, a fused collective matmul, a grouped matmul with tuned tiles.
Nothing about the compute path changes, so this works on every backend
and every operand layout. What it buys is *the forward pass seeing
quantization error*, which is the whole point of quantization-aware
training; what it does not buy is speed — it adds a quantize and a
dequantize per step.

:func:`qdot_general` / :func:`qeinsum` — **tier 2.** Hand the contraction
itself over. The forward quantizes both operands and may contract in the
narrow type; the backward recomputes gradients against the *quantized*
residuals and can quantize the incoming cotangent too. This is the tier
that can run faster than bfloat16, and the tier that only applies where
the op really is a ``dot_general``.

Both tiers are straight-through estimators, but note how tier 2 achieves
it: not with ``stop_gradient`` around the rounding, but with a
:func:`jax.custom_vjp` whose backward is written directly in terms of the
quantized residuals. That is strictly more than an STE — it is what makes
a *quantized backward pass* expressible at all, since the cotangent is
never differentiated through in the first place.

Calibration deliberately happens outside the ``custom_vjp`` boundary. A
``custom_vjp`` cannot allocate variables or run side effects, so pulling
calibration out is what leaves room for static-range quantization, where
the scale comes from a running statistic rather than from this step's
values.

Ported from Google's Qwix (``qwix._src.core.dot_general_qt``,
Apache-2.0).
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..core._typing import Array, DType
from . import _numerics as numerics
from ._calibrate import (
    Calibration,
    HowToQuantize,
    calibrate,
    compute_scale_zero_point,
    dequantize,
    quantize,
    quantize_with_scale_zero_point,
)
from ._dot import dot_general as quantized_dot_general
from ._dot import how_to_quantize_for_dot
from ._qarray import MaybeQArray, QArray, generic_broadcast_op, tiled_axes, transpose_array
from ._rules import QuantRule

__all__ = [
    "fake_quant",
    "qdot_general",
    "qeinsum",
]


def fake_quant(
    array: Array,
    *,
    rule: QuantRule,
    contracting_axes: Sequence[int],
    is_weight: bool = True,
    tile_size: int | float | None | object = ...,
    calibration_transform: Callable[[Calibration], Calibration] | None = None,
) -> Array:
    """Round ``array`` onto its quantized grid and reconstruct it, gradient-transparently.

    The backend-agnostic tier. The returned array has the same shape and
    dtype as the input and can be fed to any downstream op — an ejkernel
    grouped matmul, a Pallas attention kernel, a fused collective matmul —
    which is what lets quantization-aware training cover matmuls that are
    not plain ``dot_general`` calls.

    Which axes get their own scale is derived from ``contracting_axes``,
    not guessed: every other axis is channelwise. That is what keeps a
    fused QKV or gate-up projection correct, since its output axis is
    non-contracted and therefore each fused sub-projection keeps an
    independent range.

    The gradient is exactly the identity. It is realised by adding a
    term that is exactly zero in the forward pass but has unit derivative,
    rather than by the more common ``x + stop_gradient(q - x)``, whose
    forward value can differ from ``q`` by an ulp in bfloat16.

    Args:
        array: The floating-point array to fake-quantize.
        rule: The governing rule. ``weight_qtype`` (or ``act_qtype`` when
            ``is_weight`` is false) selects the target type; ``None``
            there makes this a no-op.
        contracting_axes: Axes contracted by the downstream matmul.
        is_weight: Whether ``array`` is the weight operand. Selects
            between the rule's weight and activation settings.
        tile_size: Overrides ``rule.tile_size`` when given; pass ``None``
            to force per-channel scales. Defaults to the rule's value.
        calibration_transform: Optional hook applied to the calibration
            statistics before they become a scale. This is the seam for
            calibrations that are not purely local: inside a ``shard_map``
            whose *contracted* axis is sharded, a local reduction sees only
            one shard, so each rank would derive a different scale for what
            is logically one tensor. Passing a collective here (typically
            :func:`jax.lax.pmax` over the sharded axis) restores the global
            statistic. It is also the hook for static-range quantization,
            where the scale comes from a running average rather than from
            this step's values.

    Returns:
        An array equal to the dequantized quantization of ``array``, with
        the same dtype, and with a straight-through gradient. Returned
        unchanged when the rule does not quantize this operand or when
        the dtype is already too narrow to quantize.
    """
    qtype = rule.weight_qtype if is_weight else rule.act_qtype
    if qtype is None or not numerics.should_quantize(array.dtype):
        return array

    effective_tile = rule.tile_size if tile_size is ... else tile_size
    channelwise = () if rule.disable_channelwise_axes else tuple(sorted(set(range(array.ndim)) - set(contracting_axes)))
    how = HowToQuantize(
        qtype=qtype,
        channelwise_axes=channelwise,
        tiled_axes={contracting_axes[0]: effective_tile} if effective_tile and contracting_axes else {},
        calibration_method=rule.weight_calibration_method if is_weight else rule.act_calibration_method,
        power_of_two_scale=rule.power_of_two_scale,
    )
    calibration = calibrate(array, how)
    if calibration_transform is not None:
        calibration = calibration_transform(calibration)
    scale, zero_point = compute_scale_zero_point(calibration, qtype, power_of_two_scale=rule.power_of_two_scale)
    quantized = quantize_with_scale_zero_point(array, qtype, scale, zero_point, how.noise_fn)
    reconstructed = dequantize(quantized).astype(array.dtype)
    # Exactly `reconstructed` in the forward (adding an exact zero is
    # exact in IEEE arithmetic) and exactly the identity in the backward.
    return jax.lax.stop_gradient(reconstructed) + (array - jax.lax.stop_gradient(array))


def _backward_dimension_numbers(
    forward: jax.lax.DotDimensionNumbers,
    forward_ndims: tuple[int, int],
    *,
    for_lhs_grad: bool,
) -> tuple[jax.lax.DotDimensionNumbers, tuple[int, ...]]:
    """Derive the contraction that computes one operand's gradient.

    Both gradients of ``out = dot(lhs, rhs)`` are themselves
    ``dot_general`` calls against the incoming cotangent ``g``:
    ``dlhs = dot(g, rhs)`` and ``drhs = dot(g, lhs)``. The dimension
    numbers follow from ``g``'s layout, which ``dot_general`` fixes as
    ``batch axes + lhs remaining + rhs remaining``.

    Writing ``x`` for the operand being differentiated and ``y`` for the
    residual it is contracted against: ``g``'s batch axes are ``x``'s
    batch axes, and the axes to contract are ``y``'s remaining axes,
    which sit at a position determined by whether ``x`` was the left or
    the right operand. The result comes out ordered
    ``x_batch + x_remaining + x_contracting``, so a final transpose
    restores ``x``'s original axis order.

    Args:
        forward: Dimension numbers of the forward contraction.
        forward_ndims: Ranks of the forward ``(lhs, rhs)``.
        for_lhs_grad: Whether to derive ``dlhs`` (else ``drhs``).

    Returns:
        ``(dimension_numbers, transpose_axes)`` — the backward
        contraction, and the permutation restoring the operand's layout.
    """
    if for_lhs_grad:
        (x_ca, y_ca), (x_ba, y_ba) = forward
        x_ndim, y_ndim = forward_ndims
    else:
        (y_ca, x_ca), (y_ba, x_ba) = forward
        y_ndim, x_ndim = forward_ndims

    x_ra = tuple(sorted(set(range(x_ndim)) - set(x_ca) - set(x_ba)))
    y_ra = tuple(sorted(set(range(y_ndim)) - set(y_ca) - set(y_ba)))

    def consecutive(*groups: Sequence[int]) -> list[tuple[int, ...]]:
        """Assign consecutive index ranges to each group in turn.

        Args:
            *groups: Axis groups whose lengths set the range widths.

        Returns:
            One index tuple per group, laid out back to back from zero.
        """
        ranges: list[tuple[int, ...]] = []
        start = 0
        for group in groups:
            ranges.append(tuple(range(start, start + len(group))))
            start += len(group)
        return ranges

    if for_lhs_grad:
        g_ba, _, g_ca = consecutive(x_ba, x_ra, y_ra)
    else:
        g_ba, g_ca, _ = consecutive(x_ba, y_ra, x_ra)

    dimension_numbers = ((g_ca, y_ra), (g_ba, y_ba))
    x_ca_in_y_order = tuple(np.take(x_ca, np.argsort(y_ca))) if x_ca else ()
    transpose_axes = tuple(np.argsort(tuple(x_ba) + x_ra + x_ca_in_y_order))
    return dimension_numbers, transpose_axes


def _fold_residual_scale_into_cotangent(
    cotangent: Array,
    residual_scale: Array,
    dimension_numbers: jax.lax.DotDimensionNumbers,
) -> Array:
    """Move a residual's scale onto the cotangent before the backward dot.

    The forward pass quantized the residual with channelwise scales
    chosen for the *forward* contraction. The backward contraction
    contracts different axes, so those scales are in the wrong place. The
    cheap fix is to leave the residual as raw quantized values and
    multiply its scale into the cotangent instead — the scale is small,
    the cotangent is being quantized anyway, and this avoids requantizing
    the residual with a second set of axes.

    Args:
        cotangent: The incoming gradient, in the backward dot's left
            operand position.
        residual_scale: The residual's scale, in the residual's layout.
        dimension_numbers: The backward contraction.

    Returns:
        The cotangent with the residual's scale folded in.
    """
    (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
    lhs_to_rhs = dict(zip(lhs_ca, rhs_ca, strict=True)) | dict(zip(lhs_ba, rhs_ba, strict=True))
    aligned = transpose_array(residual_scale, [lhs_to_rhs.get(axis) for axis in range(cotangent.ndim)])
    return generic_broadcast_op(jnp.multiply, cotangent, aligned)


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class _DotQtConfig:
    """Resolved per-call numeric settings for :func:`_dot_general_qt`.

    Split out from :class:`~spectrax.quantization.QuantRule` because the
    rule is stated in terms of "weight" and "activation" while the op
    needs it in terms of "lhs" and "rhs", and which is which depends on
    the call site.

    Attributes:
        lhs_qtype: Type for the left operand, or ``None`` to leave it be.
        rhs_qtype: Type for the right operand, or ``None``.
        tile_size: Subchannel tile size on the contracted axis.
        lhs_calibration_method: Calibration for the left operand.
        rhs_calibration_method: Calibration for the right operand.
        dlhs_grad_qtype: Type for the cotangent when forming ``dlhs``.
        dlhs_grad_calibration_method: Its calibration method.
        dlhs_tile_size: Subchannel tiling when forming ``dlhs``.
        drhs_grad_qtype: Type for the cotangent when forming ``drhs``.
        drhs_grad_calibration_method: Its calibration method.
        drhs_tile_size: Subchannel tiling when forming ``drhs``.
        noise_key: PRNG key for stochastic rounding of cotangents, or
            ``None`` for deterministic rounding.
        channelwise_noise_axes: Axes receiving independent rounding noise.
        disable_channelwise_axes: Collapse non-contracted axes to one scale.
    """

    lhs_qtype: numerics.QType | None = None
    rhs_qtype: numerics.QType | None = None
    tile_size: int | float | None = None
    lhs_calibration_method: str = "absmax"
    rhs_calibration_method: str = "absmax"

    dlhs_grad_qtype: numerics.QType | None = None
    dlhs_grad_calibration_method: str = "absmax"
    dlhs_tile_size: int | float | None = None

    drhs_grad_qtype: numerics.QType | None = None
    drhs_grad_calibration_method: str = "absmax"
    drhs_tile_size: int | float | None = None

    noise_key: Array | None = None
    channelwise_noise_axes: tuple[int, ...] = (0,)
    disable_channelwise_axes: bool = False
    power_of_two_scale: bool = False
    lhs_block_size: int | None = None
    rhs_block_size: int | None = None


def _config_from_rule(
    rule: QuantRule,
    *,
    lhs_is_weight: bool,
    rhs_is_weight: bool,
    noise_key: Array | None,
) -> _DotQtConfig:
    """Translate a rule into per-operand settings for this call site.

    Args:
        rule: The governing rule.
        lhs_is_weight: Whether the left operand is a learned weight.
        rhs_is_weight: Whether the right operand is a learned weight.
        noise_key: PRNG key for stochastic rounding, or ``None``.

    Returns:
        The resolved configuration.

    Raises:
        ValueError: If both operands are declared to be weights, or if
            stochastic rounding is requested without a key.
    """
    if lhs_is_weight and rhs_is_weight:
        raise ValueError("Both operands cannot be weights; a quantized dot needs one activation side.")
    if rule.bwd_stochastic_rounding == "uniform" and noise_key is None:
        raise ValueError(
            "bwd_stochastic_rounding='uniform' needs a PRNG key. Pass key=... to the op "
            "(for example from the layer's Rngs)."
        )

    lhs_qtype = rule.weight_qtype if lhs_is_weight else rule.act_qtype
    rhs_qtype = rule.weight_qtype if rhs_is_weight else rule.act_qtype
    lhs_method = rule.weight_calibration_method if lhs_is_weight else rule.act_calibration_method
    rhs_method = rule.weight_calibration_method if rhs_is_weight else rule.act_calibration_method

    return _DotQtConfig(
        lhs_qtype=lhs_qtype,
        rhs_qtype=rhs_qtype,
        tile_size=rule.tile_size,
        lhs_calibration_method=lhs_method,
        rhs_calibration_method=rhs_method,
        dlhs_grad_qtype=rule.bwd_qtype,
        dlhs_grad_calibration_method=rule.bwd_calibration_method,
        # The weight-gradient tiling is applied to whichever backward dot
        # produces the weight's gradient -- that is, the one whose
        # *residual* is the activation.
        dlhs_tile_size=rule.bwd_weight_grad_tile_size if lhs_is_weight else None,
        drhs_grad_qtype=rule.bwd_qtype,
        drhs_grad_calibration_method=rule.bwd_calibration_method,
        drhs_tile_size=rule.bwd_weight_grad_tile_size if rhs_is_weight else None,
        noise_key=noise_key if rule.bwd_stochastic_rounding == "uniform" else None,
        channelwise_noise_axes=rule.channelwise_noise_axes,
        disable_channelwise_axes=rule.disable_channelwise_axes,
        power_of_two_scale=rule.power_of_two_scale,
        lhs_block_size=rule.weight_block_size if lhs_is_weight else None,
        rhs_block_size=rule.weight_block_size if rhs_is_weight else None,
    )


def _dot_general_qt_fwd(
    lhs: Array,
    rhs: Array,
    lhs_calibration: Calibration | None,
    rhs_calibration: Calibration | None,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    config: _DotQtConfig,
) -> tuple[Array, tuple[MaybeQArray, MaybeQArray]]:
    """Quantize the operands, contract them, and keep them as residuals.

    Saving the *quantized* operands rather than the originals is what
    makes the backward pass cheap and consistent: it never has to
    requantize, and it differentiates the same numbers the forward pass
    actually multiplied.

    Args:
        lhs: Left operand.
        rhs: Right operand.
        lhs_calibration: Precomputed statistics for the left operand, or
            ``None`` to leave it unquantized.
        rhs_calibration: Precomputed statistics for the right operand.
        dimension_numbers: The contraction.
        config: Resolved numeric settings.

    Returns:
        ``(result, residuals)`` in the shape :func:`jax.custom_vjp` wants.
    """
    if lhs_calibration is not None:
        scale, zero_point = compute_scale_zero_point(lhs_calibration, config.lhs_qtype)
        lhs = quantize_with_scale_zero_point(lhs, config.lhs_qtype, scale, zero_point)
    if rhs_calibration is not None:
        scale, zero_point = compute_scale_zero_point(rhs_calibration, config.rhs_qtype)
        rhs = quantize_with_scale_zero_point(rhs, config.rhs_qtype, scale, zero_point)
    return quantized_dot_general(lhs, rhs, dimension_numbers), (lhs, rhs)


def _dot_general_qt_bwd(
    dimension_numbers: jax.lax.DotDimensionNumbers,
    config: _DotQtConfig,
    residuals: tuple[MaybeQArray, MaybeQArray],
    cotangent: Array,
) -> tuple[Array, Array, None, None]:
    """Compute both operand gradients, optionally in a quantized type.

    Each gradient is a contraction of the cotangent against the other
    operand's saved residual. When ``bwd_qtype`` is set the cotangent is
    quantized first, which is the third of the three matmuls in a
    training step and therefore worth a third of the potential saving.

    Args:
        dimension_numbers: The forward contraction.
        config: Resolved numeric settings.
        residuals: The quantized operands saved by the forward pass.
        cotangent: The incoming gradient.

    Returns:
        Cotangents for ``(lhs, rhs, lhs_calibration, rhs_calibration)``;
        the calibrations are non-differentiable and get ``None``.
    """
    lhs, rhs = residuals

    def gradient_for(operand_cotangent: Array, residual: MaybeQArray, *, for_lhs_grad: bool) -> Array:
        """Contract the cotangent against one residual to get one gradient.

        Args:
            operand_cotangent: The incoming gradient.
            residual: The other operand, as saved by the forward pass.
            for_lhs_grad: Whether this produces ``dlhs`` (else ``drhs``).

        Returns:
            The gradient, transposed back into the operand's layout.
        """
        backward_dnums, transpose_axes = _backward_dimension_numbers(
            dimension_numbers, (lhs.ndim, rhs.ndim), for_lhs_grad=for_lhs_grad
        )
        if for_lhs_grad:
            grad_qtype = config.dlhs_grad_qtype
            grad_tile_size = config.dlhs_tile_size
            grad_method = config.dlhs_grad_calibration_method
        else:
            grad_qtype = config.drhs_grad_qtype
            grad_tile_size = config.drhs_tile_size
            grad_method = config.drhs_grad_calibration_method

        if grad_qtype and numerics.should_quantize(operand_cotangent.dtype):
            if isinstance(residual, QArray) and not tiled_axes(residual):
                # The residual's channelwise scales were chosen for the
                # forward contraction and sit on the wrong axes here.
                # Folding them into the cotangent is exact and avoids a
                # second quantization of the residual.
                assert residual.zero_point is None and residual.qtype == residual.qvalue.dtype
                operand_cotangent = _fold_residual_scale_into_cotangent(
                    operand_cotangent, residual.scale, backward_dnums
                )
                residual = residual.qvalue

            how = how_to_quantize_for_dot(
                dimension_numbers=backward_dnums,
                ndims=(operand_cotangent.ndim, residual.ndim),
                for_lhs=True,
                qtype=grad_qtype,
                tile_size=grad_tile_size,
                calibration_method=grad_method,
                disable_channelwise_axes=config.disable_channelwise_axes,
            )
            if config.noise_key is not None:
                how = dataclasses.replace(
                    how,
                    noise_fn=functools.partial(
                        numerics.uniform_noise,
                        key=config.noise_key,
                        channelwise_noise_axes=config.channelwise_noise_axes,
                    ),
                )
            operand_cotangent = quantize(operand_cotangent, how)

        return jax.lax.transpose(
            quantized_dot_general(operand_cotangent, residual, backward_dnums),
            transpose_axes,
        )

    return (
        gradient_for(cotangent, rhs, for_lhs_grad=True),
        gradient_for(cotangent, lhs, for_lhs_grad=False),
        None,
        None,
    )


@functools.partial(jax.custom_vjp, nondiff_argnums=(4, 5))
def _dot_general_qt(
    lhs: Array,
    rhs: Array,
    lhs_calibration: Calibration | None,
    rhs_calibration: Calibration | None,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    config: _DotQtConfig,
) -> Array:
    """Quantized ``dot_general`` with a custom, optionally quantized backward.

    Args:
        lhs: Left operand.
        rhs: Right operand.
        lhs_calibration: Statistics for the left operand, or ``None``.
        rhs_calibration: Statistics for the right operand, or ``None``.
        dimension_numbers: The contraction.
        config: Resolved numeric settings.

    Returns:
        The contraction's floating-point result.
    """
    result, _residuals = _dot_general_qt_fwd(
        lhs, rhs, lhs_calibration, rhs_calibration, dimension_numbers, config
    )
    return result


_dot_general_qt.defvjp(_dot_general_qt_fwd, _dot_general_qt_bwd)


def qdot_general(
    lhs: Array,
    rhs: Array,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    *,
    rule: QuantRule | None,
    lhs_is_weight: bool = False,
    rhs_is_weight: bool = True,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: DType | None = None,
    key: Array | None = None,
) -> Array:
    """Run a ``dot_general`` under a quantization rule.

    Falls straight through to :func:`jax.lax.dot_general` when ``rule`` is
    ``None`` or does not quantize weights, so a call site can be written
    once and stay exact for unquantized models.

    Unlike Qwix, which infers which operand is a weight by inspecting
    Flax's parameter boxing, this takes the answer from the caller. A
    layer always knows, and an explicit flag cannot be defeated by an
    operand that happens to have been reshaped on the way in.

    Args:
        lhs: Left operand.
        rhs: Right operand.
        dimension_numbers: The contraction.
        rule: The governing rule, or ``None`` for full precision.
        lhs_is_weight: Whether the left operand is a learned weight.
        rhs_is_weight: Whether the right operand is a learned weight.
        precision: Forwarded to :func:`jax.lax.dot_general` on the
            unquantized path.
        preferred_element_type: Forwarded output dtype request.
        key: PRNG key, required only for stochastic rounding.

    Returns:
        The contraction's result.
    """
    if rule is None or rule.weight_qtype is None:
        return jax.lax.dot_general(
            lhs,
            rhs,
            dimension_numbers,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )

    config = _config_from_rule(rule, lhs_is_weight=lhs_is_weight, rhs_is_weight=rhs_is_weight, noise_key=key)
    lhs_calibration = _calibration_for(lhs, rhs, dimension_numbers, config, for_lhs=True)
    rhs_calibration = _calibration_for(lhs, rhs, dimension_numbers, config, for_lhs=False)
    return _dot_general_qt(lhs, rhs, lhs_calibration, rhs_calibration, dimension_numbers, config)


def _calibration_for(
    lhs: Array,
    rhs: Array,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    config: _DotQtConfig,
    *,
    for_lhs: bool,
) -> Calibration | None:
    """Calibrate one operand ahead of the ``custom_vjp`` boundary.

    Args:
        lhs: Left operand.
        rhs: Right operand.
        dimension_numbers: The contraction.
        config: Resolved numeric settings.
        for_lhs: Whether to calibrate the left operand.

    Returns:
        The statistics, or ``None`` when this operand is not quantized or
        is already too narrow to quantize.
    """
    operand = lhs if for_lhs else rhs
    qtype = config.lhs_qtype if for_lhs else config.rhs_qtype
    if qtype is None or not numerics.should_quantize(operand.dtype):
        return None
    how = how_to_quantize_for_dot(
        dimension_numbers=dimension_numbers,
        ndims=(lhs.ndim, rhs.ndim),
        for_lhs=for_lhs,
        qtype=qtype,
        tile_size=config.tile_size,
        calibration_method=config.lhs_calibration_method if for_lhs else config.rhs_calibration_method,
        disable_channelwise_axes=config.disable_channelwise_axes,
        power_of_two_scale=config.power_of_two_scale,
        block_size=config.lhs_block_size if for_lhs else config.rhs_block_size,
    )
    return calibrate(operand, how)


def qeinsum(
    equation: str,
    lhs: Array,
    rhs: Array,
    *,
    rule: QuantRule | None,
    lhs_is_weight: bool = False,
    rhs_is_weight: bool = True,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: DType | None = None,
    key: Array | None = None,
) -> Array:
    """Run a two-operand :func:`jax.numpy.einsum` under a quantization rule.

    Implemented by handing :func:`jax.numpy.einsum` a custom
    ``_dot_general``, so the equation parsing, axis ordering and output
    layout stay exactly NumPy's. Tracing is disabled around the call
    because einsum's contraction planner must run in Python to hand us
    concrete dimension numbers.

    ``einsum`` is free to swap the operands while planning, so which side
    is the weight is re-derived inside the callback from the arrays it is
    actually given rather than assumed from the call site.

    Args:
        equation: A two-operand einsum equation.
        lhs: First operand.
        rhs: Second operand.
        rule: The governing rule, or ``None`` for full precision.
        lhs_is_weight: Whether ``lhs`` is a learned weight.
        rhs_is_weight: Whether ``rhs`` is a learned weight.
        precision: Forwarded to :func:`jax.numpy.einsum`.
        preferred_element_type: Forwarded output dtype request.
        key: PRNG key, required only for stochastic rounding.

    Returns:
        The einsum's result.
    """
    if rule is None or rule.weight_qtype is None:
        return jnp.einsum(
            equation,
            lhs,
            rhs,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )

    weight_id = id(rhs) if rhs_is_weight else (id(lhs) if lhs_is_weight else None)

    def dot(
        inner_lhs: Array,
        inner_rhs: Array,
        dimension_numbers: jax.lax.DotDimensionNumbers,
        precision: jax.lax.PrecisionLike = None,
        preferred_element_type: DType | None = None,
        **_: object,
    ) -> Array:
        """Contract the planner's operands under the rule.

        Args:
            inner_lhs: Left operand chosen by the einsum planner.
            inner_rhs: Right operand chosen by the einsum planner.
            dimension_numbers: The planned contraction.
            precision: Ignored; the quantized path sets its own.
            preferred_element_type: Forwarded output dtype request.
            **_: Other keyword arguments einsum may pass.

        Returns:
            The contraction's result.
        """
        del precision
        return qdot_general(
            inner_lhs,
            inner_rhs,
            dimension_numbers,
            rule=rule,
            lhs_is_weight=weight_id is not None and id(inner_lhs) == weight_id,
            rhs_is_weight=weight_id is not None and id(inner_rhs) == weight_id,
            preferred_element_type=preferred_element_type,
            key=key,
        )

    with jax.disable_jit():
        return jnp.einsum(
            equation,
            lhs,
            rhs,
            precision=precision,
            preferred_element_type=preferred_element_type,
            _dot_general=dot,
        )
