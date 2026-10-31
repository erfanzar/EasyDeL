# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Calibration and the quantize/dequantize round trip.

Quantization is two decisions. *How* to slice the array — which axes get
their own scale, which are tiled, and how wide the tiles are — is
:class:`HowToQuantize`. *What* the scale should be, given that slicing,
is calibration: :func:`calibrate` reduces the array to per-tile
statistics and :func:`compute_scale_zero_point` turns those statistics
into a scale and (for asymmetric methods) a zero point.

Keeping the two apart is what makes static-range quantization possible:
the statistics from :func:`calibrate` are ordinary arrays, so they can be
averaged across steps, stored, or replaced before being handed to
:func:`compute_scale_zero_point`.

Supported calibration methods, spelled ``<method>[,<arg>...]``:

``absmax[,scale]``
    Symmetric, ranged on the largest magnitude. The optional factor
    clips the range, trading outlier fidelity for resolution in the bulk
    — ``"absmax,0.8"`` is a common choice for 4-bit weights.
``minmax[,scale]``
    Asymmetric, ranged on the true minimum and maximum, with the range
    forced to contain zero so that zero stays exactly representable.
``affine``
    Exact affine grid from each tile's true minimum and maximum, without
    expanding the range to contain zero. Uses unsigned integer codes and an
    unrounded floating-point zero point. A constant tile uses scale one.
``rms,scale``
    Symmetric, ranged on the root-mean-square times a required factor.
    Far more robust to outliers than absmax; the factor sets how many
    standard deviations are kept.
``fixed,max`` / ``fixed,min,max``
    A constant range, always per-tensor. Used for activations whose range
    is known a priori, and for tests that need a scale that does not move.

Ported from Google's Qwix (``qwix._src.core.qarray``, Apache-2.0).
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Collection, Mapping

import jax
import jax.numpy as jnp

from ..core._typing import Array
from . import _numerics as numerics
from ._numerics import NoiseFn, QType
from ._qarray import QArray, generic_broadcast_op, resolve_tile_size, split_axis

__all__ = [
    "Calibration",
    "HowToQuantize",
    "calibrate",
    "compute_scale_zero_point",
    "dequantize",
    "quantize",
    "quantize_with_scale_zero_point",
    "scale_shape",
]


Calibration = dict[str, Array]
"""Per-tile statistics.

Symmetric methods return ``absmax``; minmax returns ``min`` and ``max``.
Exact affine returns ``affine_neg_min`` and ``affine_max``. Storing the
negated minimum makes a tree-wide maximum collective correctly combine
both affine extrema across shards.
"""


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class HowToQuantize:
    """The slicing decision: which axes get their own scale, and at what granularity.

    Attributes:
        qtype: The target quantized type.
        channelwise_axes: Axes that get a scale per index. Equivalent to a
            tile size of one, but named separately because it is the
            default for every non-contracted axis and reads better at the
            call site.
        tiled_axes: Axes split into fixed-size tiles sharing a scale, as
            ``{axis: tile_size}``. An ``int`` is a literal size; a
            ``float`` means ``1 / tile_count``. This is subchannel
            quantization, and the axis it is applied to is normally the
            contracted one.
        calibration_method: A method string as documented in the module
            docstring.
        noise_fn: Optional noise source enabling stochastic rounding.
        power_of_two_scale: Snap each scale up to a power of two. This is
            what separates a microscaling format such as MXFP4 from
            "4-bit values with an arbitrary scale": the shared scale is an
            E8M0 exponent, so it must be exactly a power of two. It also
            makes rescaling exact, since multiplying by a power of two only
            shifts the exponent and cannot lose mantissa bits.
    """

    qtype: QType
    channelwise_axes: Collection[int] = ()
    tiled_axes: Mapping[int, int | float] = dataclasses.field(default_factory=dict)
    calibration_method: str = "absmax"
    noise_fn: NoiseFn | None = None
    power_of_two_scale: bool = False


def scale_shape(array_shape: tuple[int, ...], how: HowToQuantize) -> tuple[int, ...]:
    """Compute the scale shape implied by ``how`` for an array of ``array_shape``.

    Every axis contributes one entry: its full length when channelwise,
    its tile count when tiled, and ``1` otherwise (shared scale).

    Args:
        array_shape: Shape of the array to be quantized.
        how: The slicing decision.

    Returns:
        The scale's shape, of the same rank as ``array_shape``.

    Raises:
        ValueError: If an axis is listed as both channelwise and tiled, or
            if a tile size does not divide its axis.
    """
    overlap = set(how.channelwise_axes) & set(how.tiled_axes)
    if overlap:
        raise ValueError(f"Axes {sorted(overlap)} are both channelwise and tiled; pick one.")
    shape: list[int] = []
    for axis, dim in enumerate(array_shape):
        if axis in how.channelwise_axes:
            shape.append(dim)
        elif axis in how.tiled_axes:
            shape.append(dim // resolve_tile_size(dim, how.tiled_axes[axis]))
        else:
            shape.append(1)
    return tuple(shape)


def calibrate(array: Array, how: HowToQuantize) -> Calibration:
    """Reduce ``array`` to the per-tile statistics its scale is derived from.

    The reduction runs over every axis that is *not* channelwise; for a
    tiled axis it runs over the within-tile extent only, leaving the tile
    count intact. The returned statistics therefore already have the
    scale's shape, and can be averaged or persisted as-is.

    Args:
        array: The floating-point array to calibrate.
        how: The slicing decision, including the calibration method.

    Returns:
        ``{"absmax": ...}`` for symmetric methods, or
        ``{"min": ..., "max": ...}`` for minmax, or ``{"affine_neg_min":
        ..., "affine_max": ...}`` for exact affine. Every value has the
        shape returned by :func:`scale_shape`.

    Raises:
        ValueError: If the method string is unknown, if ``rms`` is used
            without its required factor, or if a ``fixed`` range is empty
            or does not contain zero.
    """
    reduce_axes: list[int] = []
    offset = 0
    for axis in range(array.ndim):
        if axis in how.channelwise_axes:
            continue
        if axis in how.tiled_axes:
            offset += 1
        reduce_axes.append(axis + offset)

    target = scale_shape(array.shape, how)
    split = split_axis(array, how.tiled_axes)

    method, *raw_args = how.calibration_method.lower().split(",")
    args = [float(a) for a in raw_args]

    if method == "absmax":
        absmax = jnp.max(jnp.abs(split), axis=reduce_axes, keepdims=True)
        if args:
            absmax = absmax * args[0]
        return {"absmax": absmax.reshape(target)}

    if method == "affine":
        if args:
            raise ValueError("The 'affine' calibration method takes no arguments.")
        # Validate the integer grid without importing any deployment framework.
        numerics.asymmetric_bound(how.qtype)
        low = jnp.min(split, axis=reduce_axes, keepdims=True)
        high = jnp.max(split, axis=reduce_axes, keepdims=True)
        return {"affine_neg_min": -low.reshape(target), "affine_max": high.reshape(target)}

    if method == "minmax":
        low = jnp.min(split, axis=reduce_axes, keepdims=True)
        high = jnp.max(split, axis=reduce_axes, keepdims=True)
        # Force the range to straddle zero so that an exact zero input
        # quantizes to an exact zero output -- padding and masked
        # positions depend on that.
        low = jnp.clip(low, max=0)
        high = jnp.clip(high, min=0)
        if args:
            low, high = low * args[0], high * args[0]
        return {"min": low.reshape(target), "max": high.reshape(target)}

    if method == "rms":
        if not args:
            raise ValueError("The 'rms' calibration method requires a factor, e.g. 'rms,3.0'.")
        rms = jnp.sqrt(jnp.mean(jnp.square(split), axis=reduce_axes, keepdims=True))
        return {"absmax": (rms * args[0]).reshape(target)}

    if method == "fixed":
        if len(args) not in (1, 2):
            raise ValueError("The 'fixed' calibration method takes 'fixed,max' or 'fixed,min,max'.")
        low, high = (-args[0], args[0]) if len(args) == 1 else (args[0], args[1])
        if low > 0 or high < 0 or low >= high:
            raise ValueError(f"A fixed range must be non-empty and contain zero; got [{low}, {high}].")
        per_tensor = tuple(1 for _ in target)
        if low + high == 0:
            return {"absmax": jnp.full(per_tensor, high, array.dtype)}
        return {"min": jnp.full(per_tensor, low, array.dtype), "max": jnp.full(per_tensor, high, array.dtype)}

    raise ValueError(
        f"Unknown calibration method {how.calibration_method!r}. Supported: absmax[,scale], minmax[,scale], affine, "
        "rms,scale, fixed[,min],max."
    )


def calibrate_across_axis(
    array: Array, how: HowToQuantize, axis_name: str | tuple[str, ...] | None, *, array_axis: int
) -> Calibration:
    """Calibrate global contraction tiles without gathering the operand.

    Complete tiles local to a shard remain independent. A tile crossing a
    shard boundary combines statistics only from pieces of that global tile,
    including when a boundary cuts a tile at a non-integral tile offset.

    Args:
        array: Rank-local operand.
        how: Quantization slicing and calibration settings.
        axis_name: Mesh axis (or ordered axes) partitioning ``array_axis``.
            ``None`` performs ordinary local calibration.
        array_axis: Operand axis partitioned over ``axis_name``.

    Returns:
        Statistics broadcastable to the local operand, on the global grid.
    """
    if axis_name is None or array_axis in how.channelwise_axes:
        return calibrate(array, how)
    method = how.calibration_method.lower().split(",")[0]
    if method == "fixed":
        return calibrate(array, dataclasses.replace(how, tiled_axes={}))
    if array_axis in how.tiled_axes:
        local_size = array.shape[array_axis]
        global_size = local_size * jax.lax.axis_size(axis_name)
        tile_size = resolve_tile_size(global_size, how.tiled_axes[array_axis])
        how = dataclasses.replace(how, tiled_axes={**how.tiled_axes, array_axis: tile_size})
        if local_size % tile_size == 0:
            return calibrate(array, how)
        piece_size = math.gcd(local_size, tile_size)
        piece_how = dataclasses.replace(how, tiled_axes={**how.tiled_axes, array_axis: piece_size})
        calibration = calibrate(array, piece_how)
        pieces_per_tile = tile_size // piece_size
        local_pieces = local_size // piece_size
        tile_ids = (jax.lax.axis_index(axis_name) * local_pieces + jnp.arange(local_pieces)) // pieces_per_tile
        for key, value in calibration.items():
            pieces = jax.lax.all_gather(value, axis_name, axis=array_axis, tiled=True)
            shape = (
                *pieces.shape[:array_axis],
                global_size // tile_size,
                pieces_per_tile,
                *pieces.shape[array_axis + 1 :],
            )
            pieces = pieces.reshape(shape)
            if method == "rms":
                combined = jnp.sqrt(jnp.mean(jnp.square(pieces), axis=array_axis + 1))
            else:
                combined = (jnp.min if key == "min" else jnp.max)(pieces, axis=array_axis + 1)
            calibration[key] = jnp.take(combined, tile_ids, axis=array_axis)
        return calibration
    calibration = calibrate(array, how)
    for key, value in calibration.items():
        if method == "rms":
            calibration[key] = jnp.sqrt(jax.lax.pmean(jnp.square(value), axis_name))
        else:
            calibration[key] = (jax.lax.pmin if key == "min" else jax.lax.pmax)(value, axis_name)
    return calibration


def _snap_up_to_power_of_two(scale: Array) -> Array:
    """Round each scale up to the next power of two.

    Rounding *up* rather than to nearest is deliberate: the scale divides
    the values, so a scale that is too small pushes them past the top of
    the quantized grid and clips. Rounding up only costs a little
    resolution.

    Args:
        scale: Positive scale factors.

    Returns:
        The scales, each raised to the next power of two.

    """
    return jnp.exp2(jnp.ceil(jnp.log2(scale)))


def compute_scale_zero_point(
    calibration: Calibration,
    qtype: QType,
    *,
    power_of_two_scale: bool = False,
) -> tuple[Array, Array | None]:
    """Turn calibration statistics into a scale and, if asymmetric, a zero point.

    A scale of exactly zero (an all-zero tile) is replaced by one so the
    division in :func:`quantize_with_scale_zero_point` cannot produce
    ``nan``; the quantized values are zero either way, so the substitution
    is exact rather than a fudge.

    Args:
        calibration: Statistics from :func:`calibrate`.
        qtype: The target quantized type.
        power_of_two_scale: Round each scale up to a power of two.

    Returns:
        ``(scale, zero_point)``, where ``zero_point`` is ``None`` for
        symmetric quantization, integer-valued for minmax, or an unrounded
        floating-point value for the unsigned exact-affine grid.

    Raises:
        ValueError: If the statistics dict has neither the symmetric nor
            the asymmetric key set.
    """
    if "affine_neg_min" in calibration and "affine_max" in calibration:
        qmin, qmax = numerics.asymmetric_bound(qtype)
        scale = (calibration["affine_max"] + calibration["affine_neg_min"]) / (qmax - qmin)
        scale = jnp.where(scale == 0, 1, scale)
        if power_of_two_scale:
            scale = _snap_up_to_power_of_two(scale)
        # Use with exact_affine=True in quantize_with_scale_zero_point.
        # Rounding/clipping this zero point would move shifted/constant tiles.
        return scale, calibration["affine_neg_min"] / scale

    if "min" in calibration and "max" in calibration:
        qmin, qmax = numerics.asymmetric_bound(qtype)
        scale = (calibration["max"] - calibration["min"]) / (qmax - qmin)
        scale = jnp.where(scale == 0, 1, scale)
        if power_of_two_scale:
            scale = _snap_up_to_power_of_two(scale)
        zero_point = numerics.convert_to(qmin - calibration["min"] / scale, qtype)
        return scale, zero_point

    if "absmax" in calibration:
        scale = calibration["absmax"] / numerics.symmetric_bound(qtype)
        scale = jnp.where(scale == 0, 1, scale)
        if power_of_two_scale:
            scale = _snap_up_to_power_of_two(scale)
        return scale, None

    raise ValueError(f"Unusable calibration {sorted(calibration)}; expected 'absmax' or both 'min' and 'max'.")


def quantize_with_scale_zero_point(
    array: Array,
    qtype: QType,
    scale: Array,
    zero_point: Array | None = None,
    noise_fn: NoiseFn | None = None,
    *,
    exact_affine: bool = False,
) -> QArray:
    """Quantize ``array`` using a scale and zero point that are already known.

    Split out from :func:`quantize` because static-range quantization
    computes the scale from a running statistic rather than from this
    call's data.

    The scale is cast to ``array``'s dtype first: :func:`dequantize` reads
    the reconstruction dtype off the scale, so letting a float32 scale
    ride along with a bfloat16 array would silently widen every
    downstream matmul.

    Args:
        array: The floating-point array to quantize.
        qtype: The target quantized type.
        scale: Per-tile scale, generic-broadcastable to ``array``.
        zero_point: Per-tile zero point for asymmetric quantization.
        noise_fn: Optional noise source enabling stochastic rounding.
        exact_affine: Use unsigned codes with an unrounded floating zero
            point, as returned by exact-affine calibration. Defaults to
            ``False``: signed grid, including caller-supplied floating zero
            points.

    Returns:
        The quantized array.

    Raises:
        ValueError: If ``array``'s dtype is too narrow to be quantized, or
            if the zero point's shape does not match the scale's.
    """
    if not numerics.should_quantize(array.dtype):
        raise ValueError(f"Refusing to quantize a {array.dtype} array; only bfloat16/float32/float64 are candidates.")
    if zero_point is not None and zero_point.shape != scale.shape:
        raise ValueError(f"zero_point shape {zero_point.shape} must match scale shape {scale.shape}.")
    if exact_affine and (zero_point is None or not jnp.issubdtype(zero_point.dtype, jnp.floating)):
        raise ValueError("Exact affine quantization requires an unrounded floating-point zero_point.")

    scale = scale.astype(array.dtype)
    qvalue = generic_broadcast_op(jnp.divide, array, scale)
    if zero_point is not None:
        qvalue = generic_broadcast_op(jnp.add, qvalue, zero_point.astype(qvalue.dtype))
        if exact_affine:
            qmin, qmax = numerics.asymmetric_bound(qtype)
            if noise_fn is not None:
                qvalue = qvalue.astype(jnp.float32) + noise_fn(qvalue.shape)
            bits = numerics.qtype_bits(qtype)
            storage = jnp.uint4 if bits <= 4 else jnp.uint8
            return QArray(
                qvalue=jnp.round(qvalue).clip(0, qmax - qmin).astype(storage),
                scale=scale,
                zero_point=zero_point,
                qtype=qtype,
            )
    return QArray(
        qvalue=numerics.convert_to(qvalue, qtype, noise_fn),
        scale=scale,
        zero_point=zero_point,
        qtype=qtype,
    )


def quantize(array: Array, how: HowToQuantize) -> QArray:
    """Quantize ``array`` with a scale derived from its own current values.

    This is dynamic-range quantization: calibrate, derive, quantize.

    Args:
        array: The floating-point array to quantize.
        how: The slicing decision and calibration method.

    Returns:
        The quantized array.
    """
    calibration = calibrate(array, how)
    scale, zero_point = compute_scale_zero_point(calibration, how.qtype, power_of_two_scale=how.power_of_two_scale)
    return quantize_with_scale_zero_point(
        array, how.qtype, scale, zero_point, how.noise_fn, exact_affine="affine_neg_min" in calibration
    )


def dequantize(array: QArray) -> Array:
    """Reconstruct the floating-point approximation of a quantized array.

    Args:
        array: The quantized array.

    Returns:
        ``(qvalue - zero_point) * scale``, in the scale's dtype.
    """
    # Reconstruct exact affine in at least fp32, even with bf16 metadata:
    # rounding a fractional zero-point subtraction before scaling moves the grid.
    exact_affine = (
        array.zero_point is not None
        and jnp.issubdtype(array.zero_point.dtype, jnp.floating)
        and jnp.issubdtype(array.qvalue.dtype, jnp.unsignedinteger)
    )
    compute_dtype = jnp.promote_types(array.scale.dtype, jnp.float32) if exact_affine else array.scale.dtype
    qvalue = numerics.convert_from(array.qvalue, array.qtype).astype(compute_dtype)
    if array.zero_point is not None:
        qvalue = generic_broadcast_op(jnp.subtract, qvalue, array.zero_point.astype(qvalue.dtype))
    return generic_broadcast_op(jnp.multiply, qvalue, array.scale.astype(compute_dtype)).astype(array.scale.dtype)
