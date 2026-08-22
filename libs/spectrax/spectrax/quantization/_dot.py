# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
""":func:`jax.lax.dot_general` over quantized operands, with subchannel support.

:func:`dot_general` here accepts a :class:`~spectrax.quantization.QArray`
in either operand position and returns a plain floating-point array. It
picks between two strategies, and the choice is where most of the
performance lives:

**Dequantize on input** (``_dequantized_dot_general``). Reconstruct the
float values, then run an ordinary matmul. Sounds wasteful, but XLA fuses
the dequantize into the matmul's operand read, so when the *other*
operand is still bf16 the multiply has to happen in floating point
anyway. This is the right choice for weight-only quantization, for
subchannel tiles too small to amortize, and for ``"nf4"``, whose code
book has no arithmetic meaning.

**Dequantize on output** (``_quantized_dot_general``). Contract the
quantized values directly — an int8 x int8 matmul accumulating into
int32 — then apply the scales to the much smaller result. This is the
only path that actually runs low-precision arithmetic, and it requires
both operands to be quantized affine types.

Subchannel tiling is handled by rewriting the contraction rather than by
looping: a tiled contracted axis is split into ``(tile_count, tile_size)``
and ``tile_count`` is promoted to a *batch* axis, so one ``dot_general``
computes every tile's partial product and a final sum over the promoted
axes collapses them. :func:`loop_dot_general` provides the explicit-loop
alternative for backends that would rather not materialize the extra
batch axis.

Ported from Google's Qwix (``qwix._src.core.dot_general``, Apache-2.0).
"""

from __future__ import annotations

import itertools
from collections.abc import Collection, Sequence

import jax
import jax.numpy as jnp

from ..core._typing import Array, DType
from . import _numerics as numerics
from ._calibrate import HowToQuantize, dequantize
from ._numerics import QType
from ._qarray import (
    MaybeQArray,
    QArray,
    generic_broadcast_op,
    split_axis,
    tiled_axes,
    transpose_array,
    validate_qarray,
)

__all__ = [
    "MIN_TILE_SIZE_FOR_OUTPUT_DEQUANT",
    "accumulator_and_result_type",
    "dot_general",
    "how_to_quantize_for_dot",
    "loop_dot_general",
]


MIN_TILE_SIZE_FOR_OUTPUT_DEQUANT = 128
"""Shortest contracted tile still worth contracting in the quantized type.

Below this the promoted batch axis is long and the inner contraction
short, so the matmul degenerates into many tiny dots plus a large scale
multiply. Dequantizing on input is faster there, and dramatically faster
at ``tile_size=1`` (a per-channel scale on the contracted axis).
"""


def how_to_quantize_for_dot(
    *,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    ndims: tuple[int, int],
    for_lhs: bool,
    qtype: QType,
    tile_size: int | float | None,
    calibration_method: str,
    disable_channelwise_axes: bool = False,
    power_of_two_scale: bool = False,
    block_size: int | None = None,
) -> HowToQuantize:
    """Derive the slicing decision for one operand of a ``dot_general``.

    The rule is dictated by the contraction: an axis that survives into
    the output can carry its own scale for free, because that scale
    factors out of the sum. An axis that is contracted cannot — its scale
    would have to move inside the sum — so it gets one shared scale, or,
    when ``tile_size`` is set, one scale per tile with the tiles summed
    separately.

    Making every non-contracted axis channelwise by default is also what
    keeps fused projections honest: on a fused QKV weight the output axis
    is non-contracted, so Q, K and V columns get independent scales and
    never share a range.

    Args:
        dimension_numbers: The contraction, as passed to ``dot_general``.
        ndims: Ranks of ``(lhs, rhs)``.
        for_lhs: Whether to describe the left or right operand.
        qtype: Target quantized type.
        tile_size: Subchannel tile size for the contracted axis, or ``None``.
        calibration_method: Calibration method string.
        disable_channelwise_axes: Collapse non-contracted axes to a single
            shared scale instead of one per index.
        power_of_two_scale: Constrain scales to powers of two, as the
            microscaling formats require.
        block_size: Tile the *non-contracted* axes by this too, giving
            square blocks rather than one scale per output channel.
            DeepSeek-V3 quantizes weights on 128x128 blocks this way.
            Coarser than per-channel and therefore slightly less
            accurate, but it stores far fewer scales and lines the
            blocks up with the tensor-core tiles a kernel wants.

    Returns:
        The :class:`~spectrax.quantization.HowToQuantize` for that operand.
    """
    ndim = ndims[0] if for_lhs else ndims[1]
    contracting = dimension_numbers[0][0] if for_lhs else dimension_numbers[0][1]

    channelwise = () if disable_channelwise_axes else tuple(sorted(set(range(ndim)) - set(contracting)))
    tiled = {contracting[0]: tile_size} if tile_size else {}
    if block_size:
        # Square blocks: the non-contracted axes stop being per-channel
        # and are tiled at the same granularity as the contracted one.
        tiled |= {axis: block_size for axis in channelwise}
        channelwise = ()
    return HowToQuantize(
        qtype=qtype,
        channelwise_axes=channelwise,
        tiled_axes=tiled,
        calibration_method=calibration_method,
        power_of_two_scale=power_of_two_scale,
    )


def accumulator_and_result_type(
    *operands: MaybeQArray,
    preferred_element_type: DType | None,
) -> tuple[DType, DType]:
    """Choose the accumulation dtype and the output dtype for a quantized dot.

    The two differ for integer operands: an int8 x int8 contraction
    accumulates in int32 to avoid overflow but must be *reported* in the
    dtype the scales dequantize to, since that is what the caller's graph
    expects downstream.

    Types narrower than a byte have no promotion path in JAX, so they are
    widened explicitly — integers to int32, everything else to bfloat16 —
    rather than letting ``result_type`` fail.

    Args:
        *operands: The dot's operands, quantized or not.
        preferred_element_type: Caller's explicit output dtype, if any.

    Returns:
        ``(accumulator_dtype, result_dtype)``.
    """
    qvalue_dtypes: list[jnp.dtype] = []
    dequant_dtypes: list[jnp.dtype] = []
    for operand in operands:
        if isinstance(operand, QArray):
            qvalue_dtypes.append(jnp.dtype(operand.qvalue.dtype))
            dequant_dtypes.append(jnp.dtype(operand.scale.dtype))
        else:
            qvalue_dtypes.append(jnp.dtype(operand.dtype))
            dequant_dtypes.append(jnp.dtype(operand.dtype))

    result_type = preferred_element_type
    if result_type is None:
        widened = [
            (jnp.dtype(jnp.int32) if "int" in dtype.name else jnp.dtype(jnp.bfloat16)) if dtype.itemsize <= 1 else dtype
            for dtype in dequant_dtypes
        ]
        result_type = jnp.result_type(*widened)

    accumulator_type = result_type
    if all("int" in dtype.name for dtype in qvalue_dtypes):
        accumulator_type = jnp.int32
    return accumulator_type, result_type


def _scale_transposes(
    dimension_numbers: jax.lax.DotDimensionNumbers,
    ndims: tuple[int, int],
) -> tuple[list[int | None], list[int | None]]:
    """Compute how each operand's scale must be laid out against the output.

    ``dot_general`` emits ``batch axes + lhs remaining + rhs remaining``.
    A scale therefore keeps its batch and own-remaining axes and gains
    size-one placeholders where the *other* operand's remaining axes sit.

    Args:
        dimension_numbers: The contraction.
        ndims: Ranks of ``(lhs, rhs)``.

    Returns:
        The transpose orders for ``(lhs_scale, rhs_scale)``, in the form
        accepted by :func:`~spectrax.quantization.transpose_array`.
    """
    (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
    lhs_ra = sorted(set(range(ndims[0])) - set(lhs_ca) - set(lhs_ba))
    rhs_ra = sorted(set(range(ndims[1])) - set(rhs_ca) - set(rhs_ba))
    return (
        [*lhs_ba, *lhs_ra, *([None] * len(rhs_ra))],
        [*rhs_ba, *([None] * len(lhs_ra)), *rhs_ra],
    )


def _promote_tiles_to_batch(
    contracting: Sequence[int],
    batch: Sequence[int],
    tiled: Collection[int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Rewrite dimension numbers after tiled axes have been split.

    Splitting axis ``a`` into ``(tile_count, tile_size)`` shifts every
    later axis right by one and turns ``a`` into two axes: the tile count
    becomes a batch axis (so tiles are contracted independently) and the
    tile size becomes the new contracting axis.

    Tile-count axes are appended after the original batch axes, and both
    operands must use the same order for the pairing to line up.

    Args:
        contracting: Original contracting axes.
        batch: Original batch axes.
        tiled: Which contracting axes were split.

    Returns:
        ``(contracting, batch, sum_axes)`` in the split numbering, where
        ``sum_axes`` indexes the output axes to sum over to recombine
        the tiles.
    """
    new_ca = tuple(axis + sum(t <= axis for t in tiled) for axis in contracting)
    new_ba = [axis + sum(t < axis for t in tiled) for axis in batch]
    new_ba += [axis + sum(t < axis for t in tiled) for axis in contracting if axis in tiled]
    sum_axes = tuple(range(len(batch), len(new_ba)))
    return new_ca, tuple(new_ba), sum_axes


def _broadcast_axes(array: Array, shape: Sequence[int], axes: Collection[int]) -> Array:
    """Broadcast the given axes of ``array`` up to ``shape``.

    Used for zero points, which are stored per tile but must be
    materialized at full width to enter a contraction.

    Args:
        array: The array to broadcast.
        shape: Target shape to take extents from.
        axes: Axes to expand.

    Returns:
        The broadcast array.
    """
    target = list(array.shape)
    for axis in axes:
        target[axis] = shape[axis]
    return jnp.broadcast_to(array, target)


def _quantized_dot_general(
    lhs: MaybeQArray,
    rhs: MaybeQArray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    preferred_element_type: DType | None = None,
    **kwargs: object,
) -> Array:
    """Contract in the quantized type, then apply the scales to the output.

    The only path that performs genuine low-precision arithmetic. Handles
    subchannel tiling by promoting tile counts to batch axes and summing
    them afterwards, and handles an asymmetric operand by subtracting the
    zero point's contribution as a second contraction.

    Args:
        lhs: Left operand, quantized or plain.
        rhs: Right operand, quantized or plain.
        dimension_numbers: The contraction.
        preferred_element_type: Caller's explicit output dtype, if any.
        **kwargs: Forwarded to :func:`jax.lax.dot_general`.

    Returns:
        The dequantized result.

    Raises:
        ValueError: If both operands are asymmetric (the cross term is not
            representable as two corrections), or if contracted axes are
            tiled with mismatched tile sizes.
    """
    lhs_value, lhs_scale, lhs_zero, lhs_tiles = _unpack(lhs)
    rhs_value, rhs_scale, rhs_zero, rhs_tiles = _unpack(rhs)

    if lhs_zero is not None and rhs_zero is not None:
        raise ValueError("At most one operand of a quantized dot may be asymmetric.")

    (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers

    # Pair up tiled contracted axes. A tiled axis may legally meet an
    # untiled one -- the untiled side simply shares one scale across the
    # tiles -- but two tiled axes must agree on tile size.
    lhs_tiled_ca: dict[int, int] = {}
    rhs_tiled_ca: dict[int, int] = {}
    for left, right in zip(lhs_ca, rhs_ca, strict=True):
        left_size = lhs_tiles.get(left)
        right_size = rhs_tiles.get(right)
        if left_size and right_size and left_size != right_size:
            raise ValueError(
                f"Contracted axes must share a tile size: lhs axis {left} is tiled by {left_size} "
                f"but rhs axis {right} is tiled by {right_size}."
            )
        if left_size or right_size:
            size = left_size or right_size
            lhs_tiled_ca[left] = size
            rhs_tiled_ca[right] = size

    lhs_value = split_axis(lhs_value, lhs_tiled_ca)
    rhs_value = split_axis(rhs_value, rhs_tiled_ca)
    if lhs_zero is not None:
        lhs_zero = split_axis(lhs_zero, dict.fromkeys(lhs_tiled_ca, 1))
    if rhs_zero is not None:
        rhs_zero = split_axis(rhs_zero, dict.fromkeys(rhs_tiled_ca, 1))

    lhs_ca, lhs_ba, sum_axes = _promote_tiles_to_batch(lhs_ca, lhs_ba, lhs_tiled_ca)
    rhs_ca, rhs_ba, _ = _promote_tiles_to_batch(rhs_ca, rhs_ba, rhs_tiled_ca)
    dimension_numbers = ((lhs_ca, rhs_ca), (lhs_ba, rhs_ba))

    lhs_order, rhs_order = _scale_transposes(dimension_numbers, (lhs_value.ndim, rhs_value.ndim))
    if lhs_scale is not None:
        lhs_scale = transpose_array(split_axis(lhs_scale, dict.fromkeys(lhs_tiled_ca, 1)), lhs_order)
    if rhs_scale is not None:
        rhs_scale = transpose_array(split_axis(rhs_scale, dict.fromkeys(rhs_tiled_ca, 1)), rhs_order)

    accumulator, result_type = accumulator_and_result_type(lhs, rhs, preferred_element_type=preferred_element_type)

    out = jax.lax.dot_general(
        lhs_value,
        rhs_value,
        dimension_numbers=dimension_numbers,
        preferred_element_type=accumulator,
        **kwargs,
    )

    if lhs_zero is not None:
        out = generic_broadcast_op(
            jnp.subtract,
            out,
            jax.lax.dot_general(
                _broadcast_axes(lhs_zero, lhs_value.shape, (*lhs_ca, *lhs_ba)),
                rhs_value,
                dimension_numbers=dimension_numbers,
                preferred_element_type=accumulator,
                **kwargs,
            ),
        )
    if rhs_zero is not None:
        out = generic_broadcast_op(
            jnp.subtract,
            out,
            jax.lax.dot_general(
                lhs_value,
                _broadcast_axes(rhs_zero, rhs_value.shape, (*rhs_ca, *rhs_ba)),
                dimension_numbers=dimension_numbers,
                preferred_element_type=accumulator,
                **kwargs,
            ),
        )

    if lhs_scale is not None:
        out = generic_broadcast_op(jnp.multiply, out, lhs_scale)
    if rhs_scale is not None:
        out = generic_broadcast_op(jnp.multiply, out, rhs_scale)
    if sum_axes:
        out = jnp.sum(out, axis=sum_axes)
    return out.astype(result_type)


def _unpack(operand: MaybeQArray) -> tuple[Array, Array | None, Array | None, dict[int, int]]:
    """Split an operand into value, scale, zero point and tiling.

    Args:
        operand: A quantized or plain array.

    Returns:
        ``(value, scale, zero_point, tiled_axes)``, with the last three
        empty or ``None`` for a plain array.
    """
    if isinstance(operand, QArray):
        return operand.qvalue, operand.scale, operand.zero_point, tiled_axes(operand)
    return operand, None, None, {}


def _dequantized_dot_general(
    lhs: MaybeQArray,
    rhs: MaybeQArray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    **kwargs: object,
) -> Array:
    """Reconstruct float operands and run an ordinary matmul.

    Args:
        lhs: Left operand, quantized or plain.
        rhs: Right operand, quantized or plain.
        dimension_numbers: The contraction.
        **kwargs: Forwarded to :func:`jax.lax.dot_general`.

    Returns:
        The result of the floating-point contraction.
    """
    if isinstance(lhs, QArray):
        lhs = dequantize(lhs)
    if isinstance(rhs, QArray):
        rhs = dequantize(rhs)
    return jax.lax.dot_general(lhs, rhs, dimension_numbers, **kwargs)


def dot_general(
    lhs: MaybeQArray,
    rhs: MaybeQArray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    precision: jax.lax.PrecisionLike = None,
    preferred_element_type: DType | None = None,
    **kwargs: object,
) -> Array:
    """Contract two possibly-quantized operands, returning a float array.

    Chooses between dequantizing on input and dequantizing on output. See
    the module docstring for why the choice matters; the conditions that
    force the input path are:

    * either operand is still a wide float, so the multiply happens in
      floating point regardless and XLA can fuse the dequantize;
    * a quantized operand's type cannot be dequantized on the output
      (``"nf4"``, whose values are code-book indices);
    * a contracted axis is tiled below
      :data:`MIN_TILE_SIZE_FOR_OUTPUT_DEQUANT`, where the promoted batch
      axis costs more than the low-precision contraction saves.

    Args:
        lhs: Left operand, quantized or plain.
        rhs: Right operand, quantized or plain.
        dimension_numbers: The contraction.
        precision: Forwarded to :func:`jax.lax.dot_general`.
        preferred_element_type: Forwarded output dtype request.
        **kwargs: Forwarded to :func:`jax.lax.dot_general`.

    Returns:
        A floating-point array.
    """
    dequantize_on_input = False
    for operand, contracting in zip((lhs, rhs), dimension_numbers[0], strict=True):
        if not isinstance(operand, QArray):
            if numerics.should_quantize(operand.dtype):
                dequantize_on_input = True
                break
            continue

        validate_qarray(operand)
        if not numerics.can_dequantize_on_output(operand.qtype):
            dequantize_on_input = True
            break
        for axis in contracting:
            if operand.scale.shape[axis] > 1:
                tile = operand.qvalue.shape[axis] // operand.scale.shape[axis]
                if tile < MIN_TILE_SIZE_FOR_OUTPUT_DEQUANT:
                    dequantize_on_input = True
                    break
        if dequantize_on_input:
            break

    if dequantize_on_input:
        return _dequantized_dot_general(
            lhs,
            rhs,
            dimension_numbers,
            precision=precision,
            preferred_element_type=preferred_element_type,
            **kwargs,
        )
    return _quantized_dot_general(
        lhs,
        rhs,
        dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
        **kwargs,
    )


def loop_dot_general(
    lhs: MaybeQArray,
    rhs: MaybeQArray,
    dimension_numbers: jax.lax.DotDimensionNumbers,
    preferred_element_type: DType | None = None,
    **kwargs: object,
) -> Array:
    """Subchannel dot as an explicit Python loop over tiles.

    Mathematically identical to the tile-promoting path in
    :func:`dot_general`, but emits one ``dot_general`` per tile instead of
    one over a promoted batch axis. Useful where the extra batch axis is
    unwelcome — inside a Pallas kernel, or when the tile count is small
    enough that unrolling wins.

    Args:
        lhs: Left operand, quantized or plain. Must be symmetric.
        rhs: Right operand, quantized or plain. Must be symmetric.
        dimension_numbers: The contraction.
        preferred_element_type: Forwarded output dtype request.
        **kwargs: Forwarded to :func:`jax.lax.dot_general`.

    Returns:
        A floating-point array.

    Raises:
        AssertionError: If either operand carries a zero point.
        ValueError: If contracted axes are tiled with mismatched sizes.
    """
    lhs_value, lhs_scale, lhs_zero, lhs_tiles = _unpack(lhs)
    rhs_value, rhs_scale, rhs_zero, rhs_tiles = _unpack(rhs)
    assert lhs_zero is None and rhs_zero is None, "loop_dot_general does not support asymmetric operands."

    lhs_ca, rhs_ca = dimension_numbers[0]
    tile_counts: list[int] = []
    for left, right in zip(lhs_ca, rhs_ca, strict=True):
        left_size = lhs_tiles.get(left)
        right_size = rhs_tiles.get(right)
        if left_size and right_size and left_size != right_size:
            raise ValueError(
                f"Contracted axes must share a tile size: lhs axis {left} is tiled by {left_size} "
                f"but rhs axis {right} is tiled by {right_size}."
            )
        size = left_size or right_size
        tile_counts.append(lhs_value.shape[left] // size if size else 1)

    accumulator, result_type = accumulator_and_result_type(lhs, rhs, preferred_element_type=preferred_element_type)
    lhs_order, rhs_order = _scale_transposes(dimension_numbers, (lhs_value.ndim, rhs_value.ndim))

    def take(array: Array, axes: Sequence[int], indices: Sequence[int]) -> Array:
        """Slice out one tile along each contracted axis.

        Args:
            array: Value or scale to slice.
            axes: Contracted axes in ``array``'s numbering.
            indices: Tile index per contracted axis.

        Returns:
            The tile slice; axes of extent one are passed through whole,
            which is how a shared scale rides along with a tiled value.
        """
        selector: list[slice | int] = []
        for axis, extent in enumerate(array.shape):
            if axis not in axes or extent == 1:
                selector.append(slice(None))
                continue
            position = axes.index(axis)
            width = extent // tile_counts[position]
            selector.append(slice(indices[position] * width, (indices[position] + 1) * width))
        return array[tuple(selector)]

    total = None
    for indices in itertools.product(*map(range, tile_counts)):
        partial = jax.lax.dot_general(
            take(lhs_value, lhs_ca, indices),
            take(rhs_value, rhs_ca, indices),
            dimension_numbers=dimension_numbers,
            preferred_element_type=accumulator,
            **kwargs,
        )
        if lhs_scale is not None:
            partial = generic_broadcast_op(
                jnp.multiply, partial, transpose_array(take(lhs_scale, lhs_ca, indices), lhs_order)
            )
        if rhs_scale is not None:
            partial = generic_broadcast_op(
                jnp.multiply, partial, transpose_array(take(rhs_scale, rhs_ca, indices), rhs_order)
            )
        total = partial if total is None else total + partial

    assert total is not None
    return total.astype(result_type)
