# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Low-level numerics for training-time quantization.

This module answers three questions about a *quantized type* (``qtype``)
and nothing else:

* **Should this array be quantized at all?** — :func:`should_quantize`.
  Only ``bfloat16``/``float32``/``float64`` inputs are candidates; an
  array already stored in a narrow type is left alone.
* **What is the representable range?** — :func:`symmetric_bound` and
  :func:`asymmetric_bound`, which drive scale computation.
* **How do values move in and out of the type?** — :func:`convert_to`
  (round + clip + cast, optionally with stochastic rounding) and
  :func:`convert_from`.

A ``qtype`` is either a real JAX dtype (``jnp.int4``, ``jnp.int8``,
``jnp.float8_e4m3fn``, ``jnp.float8_e5m2``, ``jnp.float4_e2m1fn``) or one
of the string pseudo-types this module defines for widths JAX has no
dtype for: ``"int2"``, ``"int3"``, ``"int5"``, ``"int6"``, ``"int7"``,
and ``"nf4"``. Pseudo-integer values are *stored* in the smallest real
integer dtype that holds them (``jnp.int4`` up to 4 bits, ``jnp.int8``
beyond), so ``QArray.qtype`` and ``QArray.qvalue.dtype`` can differ.

Training-time quantization never bit-packs. Packing belongs to the
storage and kernel layers; here a 4-bit value is a real ``jnp.int4``
element, which is what keeps this module small and lets XLA fuse the
dequantize into the surrounding matmul.

The algorithms follow Google's Qwix (``qwix._src.core.numerics``,
Apache-2.0), reimplemented here so that spectrax — a foundation library
whose dependencies are limited to JAX, NumPy and a handful of I/O
packages — does not take on a Flax dependency to reach them.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TypeAlias

import jax
import jax.numpy as jnp

from ..core._typing import Array, ArrayLike, DType

__all__ = [
    "NoiseFn",
    "QType",
    "asymmetric_bound",
    "can_dequantize_on_output",
    "convert_from",
    "convert_to",
    "is_pseudo_qtype",
    "nf4_buckets",
    "qtype_bits",
    "qtype_name",
    "should_quantize",
    "storage_dtype",
    "symmetric_bound",
    "uniform_noise",
]


QType: TypeAlias = "DType | str"
"""A quantized type: a real JAX dtype or one of this module's pseudo-types."""

NoiseFn: TypeAlias = Callable[[Sequence[int]], Array]
"""Generates additive noise of a requested shape, for stochastic rounding."""


_PSEUDO_INT_QTYPES: dict[str, int] = {"int2": 2, "int3": 3, "int5": 5, "int6": 6, "int7": 7}
"""Integer widths that JAX has no dtype for, mapped to their bit count."""

_NF4 = "nf4"
"""4-bit NormalFloat, a non-uniform code book rather than an affine grid."""

_QUANTIZABLE_DTYPES: tuple[DType, ...] = (jnp.bfloat16, jnp.float32, jnp.float64)
"""Input dtypes that are wide enough to be worth quantizing."""


def qtype_name(qtype: QType) -> str:
    """Return a short, stable, human-readable name for ``qtype``.

    Used in error messages and in the repr of quantization rules, where
    ``jnp.int4`` should read as ``"int4"`` rather than as a dtype repr.

    Args:
        qtype: A real JAX dtype or one of the pseudo-types.

    Returns:
        The type's name, e.g. ``"int4"``, ``"nf4"`` or ``"float8_e4m3fn"``.
    """
    if isinstance(qtype, str):
        return qtype
    return jnp.dtype(qtype).name


def is_pseudo_qtype(qtype: QType) -> bool:
    """Whether ``qtype`` is one of this module's string pseudo-types.

    Args:
        qtype: A real JAX dtype or one of the pseudo-types.

    Returns:
        ``True`` for ``"int2"``/``"int3"``/``"int5"``/``"int6"``/``"int7"``
        and ``"nf4"``; ``False`` for every real dtype.
    """
    return isinstance(qtype, str) and (qtype in _PSEUDO_INT_QTYPES or qtype == _NF4)


def qtype_bits(qtype: QType) -> int:
    """Return the number of bits one value of ``qtype`` occupies logically.

    "Logically" matters: a ``"int3"`` value is *stored* in an ``int4``
    element but only carries 3 bits of information, and it is the logical
    width that determines the effective compression ratio a rule achieves.

    Args:
        qtype: A real JAX dtype or one of the pseudo-types.

    Returns:
        The logical bit width, e.g. ``4`` for ``jnp.int4`` and ``"nf4"``,
        ``8`` for ``jnp.int8`` and ``jnp.float8_e4m3fn``.

    Raises:
        ValueError: If ``qtype`` is a dtype wider than one byte, which is
            never a valid quantized type.
    """
    if isinstance(qtype, str):
        if qtype == _NF4:
            return 4
        if qtype in _PSEUDO_INT_QTYPES:
            return _PSEUDO_INT_QTYPES[qtype]
    dtype = jnp.dtype(qtype)
    if dtype == jnp.dtype(jnp.int4) or dtype == jnp.dtype(jnp.uint4):
        return 4
    if dtype.name in ("float4_e2m1fn",):
        return 4
    if dtype.itemsize > 1:
        raise ValueError(f"{qtype_name(qtype)} is too wide to be a quantized type.")
    return 8


def storage_dtype(qtype: QType) -> DType:
    """Return the real JAX dtype used to hold values of ``qtype``.

    Real dtypes map to themselves. Pseudo-integer types map to the
    narrowest signed integer dtype that holds their range, and ``"nf4"``
    maps to ``jnp.uint4`` because its values are code-book *indices*, not
    signed magnitudes.

    Args:
        qtype: A real JAX dtype or one of the pseudo-types.

    Returns:
        The dtype that :func:`convert_to` will produce.
    """
    if isinstance(qtype, str):
        if qtype == _NF4:
            return jnp.uint4
        bits = _PSEUDO_INT_QTYPES[qtype]
        return jnp.int4 if bits <= 4 else jnp.int8
    return jnp.dtype(qtype)


def should_quantize(dtype: DType) -> bool:
    """Whether an array of ``dtype`` is a candidate for quantization.

    Guards every quantization entry point so that re-quantizing an
    already-narrow array, or quantizing an integer index array, is a
    silent no-op rather than a corruption.

    Args:
        dtype: The dtype of the array under consideration.

    Returns:
        ``True`` only for ``bfloat16``, ``float32`` and ``float64``.
    """
    return jnp.dtype(dtype) in tuple(jnp.dtype(d) for d in _QUANTIZABLE_DTYPES)


def can_dequantize_on_output(qtype: QType) -> bool:
    """Whether a matmul may run in ``qtype`` and dequantize afterwards.

    Affine types satisfy ``x ~= q * scale``, so the scale factors out of
    a contraction and can be applied to the result. ``"nf4"`` is a
    non-uniform code book — its integer values are indices with no
    arithmetic meaning — so it must be dequantized *before* the matmul.

    Args:
        qtype: A real JAX dtype or one of the pseudo-types.

    Returns:
        ``True`` for every type except ``"nf4"``.
    """
    return qtype != _NF4


def asymmetric_bound(qtype: QType) -> tuple[float, float]:
    """Return ``(qmin, qmax)`` for asymmetric (zero-point) quantization.

    Only the two signed integer dtypes support asymmetric quantization;
    floating-point and code-book types have no meaningful integer zero
    point to shift.

    Args:
        qtype: A real JAX dtype or one of the pseudo-types.

    Returns:
        The inclusive integer range as a ``(min, max)`` pair of floats.

    Raises:
        ValueError: If ``qtype`` does not support asymmetric quantization.
    """
    if not isinstance(qtype, str):
        dtype = jnp.dtype(qtype)
        if dtype == jnp.dtype(jnp.int8):
            return (-128.0, 127.0)
        if dtype == jnp.dtype(jnp.int4):
            return (-8.0, 7.0)
    if isinstance(qtype, str) and qtype in _PSEUDO_INT_QTYPES:
        bits = _PSEUDO_INT_QTYPES[qtype]
        return (float(-(2 ** (bits - 1))), float(2 ** (bits - 1) - 1))
    raise ValueError(f"{qtype_name(qtype)} does not support asymmetric quantization; use a symmetric method.")


def symmetric_bound(qtype: QType) -> float:
    """Return the positive bound used to derive a symmetric scale.

    For integers the bound is extended by half a step
    (``2**(bits-1) - 0.5`` rather than ``2**(bits-1) - 1``) so that the
    extreme bucket is as wide as every other bucket instead of being a
    half-open endpoint. The correction is negligible at 8 bits and
    material at 2-4 bits, which is exactly where it matters.

    Args:
        qtype: A real JAX dtype or one of the pseudo-types.

    Returns:
        The largest magnitude representable, as a float.

    Raises:
        ValueError: If ``qtype`` is wider than one byte, which almost
            always means a compute dtype was passed by mistake.
    """
    if isinstance(qtype, str):
        if qtype == _NF4:
            return 1.0
        if qtype in _PSEUDO_INT_QTYPES:
            return 2 ** (_PSEUDO_INT_QTYPES[qtype] - 1) - 0.5
    dtype = jnp.dtype(qtype)
    if dtype.itemsize > 1:
        raise ValueError(
            f"Cannot use {qtype_name(qtype)} as a quantized type: it is {dtype.itemsize} bytes wide. "
            "Quantized types must fit in one byte (int4/int8/fp8/fp4/nf4)."
        )
    try:
        return float(jnp.finfo(dtype).max)
    except ValueError:
        return float(jnp.iinfo(dtype).max) + 0.5


def uniform_noise(
    shape: Sequence[int],
    *,
    key: Array,
    channelwise_noise_axes: Sequence[int] = (0,),
) -> Array:
    """Draw uniform ``[-0.5, 0.5)`` noise for stochastic rounding.

    Noise is generated at full size along ``channelwise_noise_axes`` and
    broadcast along the rest, which keeps the RNG cost proportional to a
    slice rather than to the whole tensor while still decorrelating the
    rounding error across channels.

    Args:
        shape: Shape the noise must broadcast to.
        key: PRNG key consumed for this draw.
        channelwise_noise_axes: Axes that receive independent noise; every
            other axis is broadcast from size one.

    Returns:
        A float32 array broadcastable to ``shape``, with values in
        ``[-0.5, 0.5)``.
    """
    noise_shape = [dim if axis in channelwise_noise_axes else 1 for axis, dim in enumerate(shape)]
    return jax.random.uniform(key, tuple(noise_shape), dtype=jnp.float32, minval=-0.5, maxval=0.5)


def convert_to(x: ArrayLike, qtype: QType, noise_fn: NoiseFn | None = None) -> Array:
    """Round, clip and cast ``x`` into ``qtype``.

    Integer targets round to nearest (or stochastically, when
    ``noise_fn`` is given) and rely on the cast to clip. Floating-point
    targets do not round — the cast already selects the nearest
    representable value — but *do* clip, because casting an out-of-range
    value to a saturating type such as ``float8_e4m3fn`` yields ``inf``
    or ``nan``. ``"nf4"`` bucketizes against its code book.

    Stochastic rounding is performed in float32 even when ``x`` is
    bfloat16: adding sub-unit noise to a bfloat16 value can be swallowed
    by its 8-bit mantissa, which would silently turn stochastic rounding
    back into deterministic rounding.

    Args:
        x: The scaled values to convert, typically ``array / scale``.
        qtype: Target quantized type.
        noise_fn: Optional noise source enabling stochastic rounding.
            Ignored for floating-point and code-book targets.

    Returns:
        An array in :func:`storage_dtype` of ``qtype``.
    """
    x = jnp.asarray(x)
    if isinstance(qtype, str):
        if qtype == _NF4:
            return _to_nf4(x)
        bits = _PSEUDO_INT_QTYPES[qtype]
        qmin, qmax = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
        if noise_fn is not None:
            x = x.astype(jnp.float32) + noise_fn(x.shape)
        return jnp.round(x).clip(qmin, qmax).astype(storage_dtype(qtype))

    dtype = jnp.dtype(qtype)
    try:
        finfo = jnp.finfo(dtype)
    except ValueError:
        finfo = None
    if finfo is None:
        if noise_fn is not None:
            x = x.astype(jnp.float32) + noise_fn(x.shape)
        return jnp.round(x).astype(dtype)
    return x.clip(float(finfo.min), float(finfo.max)).astype(dtype)


def convert_from(x: Array, qtype: QType) -> Array:
    """Undo :func:`convert_to`'s type mapping, leaving the scale applied.

    Affine types need no work — the stored integer *is* the quantized
    value, and the caller multiplies by the scale. ``"nf4"`` needs a
    code-book lookup because its stored values are indices.

    Args:
        x: The stored quantized values.
        qtype: The logical type ``x`` was produced with.

    Returns:
        Values on the quantized grid, still awaiting the scale.
    """
    if qtype == _NF4:
        return _from_nf4(x)
    return x


def nf4_buckets() -> Array:
    """Return the 16 NF4 code-book values.

    These are the information-theoretically optimal quantiles for a
    standard normal, from Appendix E of the QLoRA paper
    (https://arxiv.org/pdf/2305.14314). Built on demand rather than at
    import time so that importing spectrax never touches a JAX device.

    Returns:
        A float32 array of shape ``(16,)`` sorted ascending from -1 to 1.
    """
    return jnp.asarray(
        [
            -1.0,
            -0.6961928009986877,
            -0.5250730514526367,
            -0.39491748809814453,
            -0.28444138169288635,
            -0.18477343022823334,
            -0.09105003625154495,
            0.0,
            0.07958029955625534,
            0.16093020141124725,
            0.24611230194568634,
            0.33791524171829224,
            0.44070982933044434,
            0.5626170039176941,
            0.7229568362236023,
            1.0,
        ],
        dtype=jnp.float32,
    )


def _to_nf4(x: Array) -> Array:
    """Map values in ``[-1, 1]`` to their nearest NF4 code-book index.

    Implemented as a broadcast ``argmin`` over the 16 buckets rather than
    a ``vmap`` over elements: the code book is tiny, so one extra size-16
    axis is far cheaper than a per-element mapped call.

    Args:
        x: Scaled values, nominally within ``[-1, 1]``.

    Returns:
        Code-book indices stored as ``uint4``, shaped like ``x``.
    """
    buckets = nf4_buckets()
    distances = jnp.abs(x.astype(jnp.float32)[..., None] - buckets)
    return jnp.argmin(distances, axis=-1).astype(jnp.uint4)


def _from_nf4(x: Array) -> Array:
    """Map NF4 code-book indices back to their float values.

    Args:
        x: Code-book indices, typically ``uint4``.

    Returns:
        A float32 array of code-book values shaped like ``x``.
    """
    return nf4_buckets()[x.astype(jnp.int32)]
