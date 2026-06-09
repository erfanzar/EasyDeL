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

"""Floating-point lookup tables for microscaling quantization formats.

This module provides lookup tables for various low-precision floating-point
formats used in quantization:

- E2M1 (FP4): 2 exponent bits, 1 mantissa bit (used in MXFP4, NVFP4)
- E4M3 (FP8): 4 exponent bits, 3 mantissa bits (used in MXFP8, NVFP4 scales)
- NF4: 4-bit NormalFloat codebook optimized for normal distributions

The tables map integer codes to their float32 values, enabling efficient
vectorized dequantization via table lookup.
"""

from __future__ import annotations

from jax import numpy as jnp


def _decode_e2m1_codes(codes: jnp.ndarray, *, dtype: jnp.dtype) -> jnp.ndarray:
    """Decode E2M1 (FP4) codes to float values without a lookup table.

    Implements the E2M1 format (1 sign bit, 2 exponent bits, 1 mantissa bit,
    bias = 1) arithmetically.  Produces results identical to indexing into the
    table returned by :func:`_make_fp_table(exp_bits=2, mant_bits=1)`.

    Args:
        codes: Integer array of 4-bit E2M1 codes (values 0–15).
        dtype: Output floating-point dtype (e.g., ``jnp.float32``).

    Returns:
        Float array of the same shape as *codes*, with values in the E2M1
        representable range ``±{0, 0.5, 1, 1.5, 2, 3, 4, 6}``.
    """
    codes_u = codes.astype(jnp.uint32)
    sign = (codes_u >> 3) & jnp.uint32(0x1)
    exp = (codes_u >> 1) & jnp.uint32(0x3)
    mant = codes_u & jnp.uint32(0x1)

    mant_f = mant.astype(dtype) * jnp.asarray(0.5, dtype=dtype)
    exp_is_zero = exp == 0

    sub = mant_f
    norm = jnp.ldexp(jnp.asarray(1.0, dtype=dtype) + mant_f, exp.astype(jnp.int32) - 1)
    val = jnp.where(exp_is_zero, sub, norm)
    return jnp.where(sign.astype(bool), -val, val)


def _decode_e4m3_codes(codes: jnp.ndarray, *, dtype: jnp.dtype) -> jnp.ndarray:
    """Decode E4M3 (FP8) codes to float values without a lookup table.

    Implements the E4M3 format (1 sign bit, 4 exponent bits, 3 mantissa bits,
    bias = 7, ``nan_all_ones=True``) arithmetically.  Produces results
    identical to indexing into the table returned by
    :func:`_make_fp_table(exp_bits=4, mant_bits=3, nan_all_ones=True)`.

    The all-ones code (``exp=0xF``, ``mant=0x7``) decodes to NaN for both
    signs, matching the OCP MX specification.

    Args:
        codes: Integer array of 8-bit E4M3 codes (values 0–255).
        dtype: Output floating-point dtype (e.g., ``jnp.float32``).

    Returns:
        Float array of the same shape as *codes*.  The NaN code decodes to
        ``float("nan")``.
    """
    codes_u = codes.astype(jnp.uint32)
    sign = (codes_u >> 7) & jnp.uint32(0x1)
    exp = (codes_u >> 3) & jnp.uint32(0xF)
    mant = codes_u & jnp.uint32(0x7)

    bias = 7
    mant_f = mant.astype(dtype) * jnp.asarray(1.0 / 8.0, dtype=dtype)
    exp_is_zero = exp == 0
    exp_is_max = exp == 0xF

    sub = jnp.ldexp(mant_f, 1 - bias)
    norm = jnp.ldexp(jnp.asarray(1.0, dtype=dtype) + mant_f, exp.astype(jnp.int32) - bias)
    val = jnp.where(exp_is_zero, sub, norm)
    val = jnp.where(sign.astype(bool), -val, val)

    nan_mask = exp_is_max & (mant == 0x7)
    nan_val = jnp.asarray(jnp.nan, dtype=dtype)
    return jnp.where(nan_mask, nan_val, val)


def _make_fp_table(exp_bits: int, mant_bits: int, *, nan_all_ones: bool) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate a lookup table for a minifloat format.

    Builds a complete lookup table mapping all possible bit patterns to their
    float32 values for a custom floating-point format with the specified
    exponent and mantissa bit widths.

    The format is: [sign (1 bit)] [exponent (exp_bits)] [mantissa (mant_bits)]

    Args:
        exp_bits: Number of exponent bits.
        mant_bits: Number of mantissa bits.
        nan_all_ones: If True, the all-ones bit pattern (sign=1, exp=max, mant=max)
            is treated as NaN. This matches E4M3 semantics.

    Returns:
        Tuple of (value_table, nan_mask) where:
            - value_table: Float32 array of shape (2^(1+exp_bits+mant_bits),)
              mapping each code to its float value.
            - nan_mask: Boolean array indicating which codes are NaN.
    """
    bits = 1 + exp_bits + mant_bits
    codes = jnp.arange(2**bits, dtype=jnp.uint32)
    sign = (codes >> (exp_bits + mant_bits)) & 0x1
    exp = (codes >> mant_bits) & ((1 << exp_bits) - 1)
    mant = codes & ((1 << mant_bits) - 1)

    bias = (1 << (exp_bits - 1)) - 1
    mant_f = mant.astype(jnp.float32) / (2**mant_bits)

    exp_is_zero = exp == 0
    exp_is_max = exp == ((1 << exp_bits) - 1)

    sub = mant_f * jnp.exp2(1.0 - bias)
    norm = (1.0 + mant_f) * jnp.exp2(exp.astype(jnp.float32) - bias)
    val = jnp.where(exp_is_zero, sub, norm)
    val = jnp.where(sign == 1, -val, val)

    if nan_all_ones:
        nan_mask = exp_is_max & (mant == ((1 << mant_bits) - 1))
        val = jnp.where(nan_mask, jnp.nan, val)
    else:
        nan_mask = jnp.zeros_like(exp_is_max, dtype=bool)

    return val, nan_mask


def _build_threshold_map(codebook: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Build sorted-index and midpoint-boundary tensors for threshold quantization.

    Sorts the codebook once and precomputes decision boundaries as midpoints
    between consecutive sorted values.  The outputs are consumed by
    :func:`_quantize_to_codebook_threshold_from_map`.

    Args:
        codebook: 1-D float32 array of codebook values (e.g., NF4, E2M1, or
            E4M3 tables).

    Returns:
        Tuple of ``(sorted_idx, boundaries)`` where:
        - ``sorted_idx``: int32 array of shape ``(n,)`` mapping sorted
          position → original index.
        - ``boundaries``: float32 array of shape ``(n-1,)`` of midpoint
          decision boundaries.  Empty (shape ``(0,)``) when ``n <= 1``.
    """
    codebook = codebook.astype(jnp.float32)
    sorted_idx = jnp.argsort(codebook).astype(jnp.int32)
    sorted_vals = codebook[sorted_idx]
    if sorted_vals.shape[0] <= 1:
        boundaries = jnp.zeros((0,), dtype=jnp.float32)
    else:
        boundaries = ((sorted_vals[:-1] + sorted_vals[1:]) * 0.5).astype(jnp.float32)
    return sorted_idx, boundaries


def _build_nf4_table() -> jnp.ndarray:
    """Build the NF4 (NormalFloat 4-bit) codebook tensor.

    Returns a fixed float32 array of 16 codebook values sourced from the
    QLoRA paper (Dettmers et al., 2023).  Values span ``[-1.0, 1.0]`` and
    are spaced to minimize quantization error for data drawn from N(0, 1).

    Returns:
        float32 array of shape ``(16,)``.
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


def _get_e2m1_table() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Get the E2M1 (FP4) lookup table.

    E2M1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit (4 bits total).
    Range: ±{0, 0.5, 1, 1.5, 2, 3, 4, 6} (16 values).
    Used in MXFP4 and NVFP4 quantization modes.

    Returns:
        Tuple of (value_table, nan_mask). The table has 16 entries.
    """
    return _make_fp_table(2, 1, nan_all_ones=False)


def _get_e4m3_table() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Get the E4M3 (FP8) lookup table.

    E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits (8 bits total).
    Range: ±{0, ..., 448} with NaN at 0xFF. Used in MXFP8 and NVFP4 scales.

    Returns:
        Tuple of (value_table, nan_mask). The table has 256 entries.
    """
    return _make_fp_table(4, 3, nan_all_ones=True)


def _get_e4m3_table_q() -> jnp.ndarray:
    """Get E4M3 table with NaN replaced by infinity for quantization.

    When quantizing to E4M3, we want to avoid selecting the NaN code.
    By replacing NaN with inf in the lookup table, argmin-based quantization
    will never select it.

    Returns:
        E4M3 value table with NaN entries replaced by infinity.
    """
    table, nan_mask = _get_e4m3_table()
    return jnp.where(nan_mask, jnp.inf, table)


def _get_e2m1_max() -> jnp.ndarray:
    """Get the maximum representable absolute value in E2M1 format.

    Returns:
        Scalar float32 containing max(abs(E2M1 values)) = 6.0.
    """
    table, _ = _get_e2m1_table()
    return jnp.max(jnp.abs(table))


def _get_e4m3_max() -> jnp.ndarray:
    """Get the maximum representable absolute value in E4M3 format.

    Excludes NaN entries when computing the maximum.

    Returns:
        Scalar float32 containing max(abs(E4M3 values)) = 448.0.
    """
    table, nan_mask = _get_e4m3_table()
    return jnp.max(jnp.abs(jnp.where(nan_mask, 0.0, table)))


def _get_nf4_table() -> jnp.ndarray:
    """Get the NF4 (NormalFloat 4-bit) codebook.

    NF4 is a 4-bit quantization format optimized for normally distributed
    weights. The 16 codebook values are chosen to minimize quantization
    error for N(0, 1) distributed data.

    The values are asymmetric around zero to better match the typical
    distribution of neural network weights.

    Returns:
        Float32 array of shape (16,) containing the NF4 codebook values
        in range [-1, 1].

    References:
        QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023)
    """
    return _build_nf4_table()


def _get_e2m1_threshold_map() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Get cached threshold map (sorted_idx, boundaries) for E2M1 codebook."""
    return _build_threshold_map(_get_e2m1_table()[0])


def _get_e4m3_q_threshold_map() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Get cached threshold map (sorted_idx, boundaries) for E4M3-quant codebook."""
    return _build_threshold_map(_get_e4m3_table_q())


def _get_nf4_threshold_map() -> tuple[jnp.ndarray, jnp.ndarray]:
    """Get cached threshold map (sorted_idx, boundaries) for NF4 codebook."""
    return _build_threshold_map(_get_nf4_table())
