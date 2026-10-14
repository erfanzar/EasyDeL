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

"""Shared TPU Pallas core utilities for quantized matrix multiplication.

This module contains the low-level building blocks shared between the forward
and input-gradient TPU Pallas kernels for quantized matrix multiplication. It
provides bit-unpacking, dequantization, block normalization, legality checks,
predecoded weight caching, and a generic dense Pallas matmul.

Key Components:
    - Bit packing/unpacking: ``_unpack_packed_bits`` extracts 1-bit through
      8-bit affine values from packed 32-bit words
    - Dequantization: ``_dequantize_tile`` supports affine, NF4, MXFP4,
      MXFP8, NVFP4, and NVFP8 quantization modes
    - Predecode caching: ``get_predecoded_dense_weight`` materializes and
      caches dense bf16 weights from quantized representations with an
      LRU eviction policy
    - BlockSpec legality: ``is_packed_tpu_legal_forward`` and
      ``is_packed_tpu_legal_input_grad`` verify Mosaic TPU tiling constraints
    - Dense matmul: ``pallas_dense_matmul`` provides a tiled Pallas kernel
      for bf16 matrix multiplication with fp32 accumulation

Environment Variables:
    - EJKERNEL_QMM_TPU_PATH: Selects execution path for TPU QMM.
      TPU Pallas currently supports packed-only execution; "hybrid" and
      "predecode" are accepted as legacy aliases and normalize to "packed"
      (default: "packed")
    - EJKERNEL_QMM_TPU_PREDECODE_CACHE: Enable/disable predecode caching
      (default: True)
    - EJKERNEL_QMM_TPU_PREDECODE_CACHE_MAX_ITEMS: Max cached dense weights
      (default: 2)
    - EJKERNEL_QMM_TPU_MAX_PREDECODE_BYTES: Max bytes per predecode buffer
      (default: 256 MiB)
"""

from __future__ import annotations

import math
import os
import re
import threading
from collections import OrderedDict

import jax
import jax.numpy as jnp
from jax import core as jax_core
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_QMM_PATHS = frozenset(("packed", "hybrid", "predecode"))
_DEFAULT_PREDECODE_CACHE_MAX_ITEMS = 2
_DEFAULT_PREDECODE_MAX_BYTES = 256 * 1024 * 1024
_DEFAULT_TPU_VMEM_LIMIT_BYTES = 96 * 1024 * 1024
_DEFAULT_TPU_V7_VMEM_LIMIT_BYTES = 48 * 1024 * 1024

_PREDECODE_CACHE: OrderedDict[tuple, jax.Array] = OrderedDict()
_PREDECODE_CACHE_LOCK = threading.Lock()


def _ceil_div(a: int, b: int) -> int:
    """Compute ceiling division of a by b.

    Args:
        a: Dividend.
        b: Divisor (must be positive).

    Returns:
        The smallest integer >= a/b.
    """
    return (a + b - 1) // b


def _lcm(a: int, b: int) -> int:
    """Compute least common multiple for positive integers."""
    if a <= 0:
        return max(1, int(b))
    if b <= 0:
        return max(1, int(a))
    return abs(a * b) // math.gcd(a, b)


def _bit_aligned_values(bits: int) -> int:
    """Return the value count that starts and ends on a 32-bit word boundary."""
    return 32 // math.gcd(32, int(bits))


def _packed_words_for_values(values: int, bits: int) -> int:
    """Return the number of uint32 words needed for a packed bitstream."""
    return _ceil_div(int(values) * int(bits), 32)


def _pad_2d(x: jax.Array, pad0: int, pad1: int) -> jax.Array:
    """Zero-pad a 2D array along both dimensions.

    Args:
        x: Input 2D array.
        pad0: Number of rows to pad at the bottom.
        pad1: Number of columns to pad on the right.

    Returns:
        Padded array, or the original if both pads are zero.
    """
    if pad0 == 0 and pad1 == 0:
        return x
    return jnp.pad(x, ((0, pad0), (0, pad1)))


def _pad_2d_optional(x: jax.Array | None, pad0: int, pad1: int) -> jax.Array | None:
    """Zero-pad a 2D array if it is not None.

    Args:
        x: Input 2D array or None.
        pad0: Number of rows to pad at the bottom.
        pad1: Number of columns to pad on the right.

    Returns:
        Padded array, or None if x is None.
    """
    if x is None:
        return None
    return _pad_2d(x, pad0, pad1)


def _normalize_tpu_blocks(block_m: int, block_n: int, block_k: int) -> tuple[int, int, int]:
    """Round block dimensions up to TPU-friendly tile sizes.

    Ensures block_m is a multiple of 8 (>= 8) for the sublane dimension,
    and block_n/block_k are multiples of 128 (>= 128) for the lane dimension.
    These constraints are required by the Mosaic TPU compiler for valid
    BlockSpec tiling.

    Args:
        block_m: Desired M-dimension block size.
        block_n: Desired N-dimension block size.
        block_k: Desired K-dimension block size.

    Returns:
        Tuple of (block_m, block_n, block_k) rounded up to valid sizes.

    Raises:
        ValueError: If any block dimension is non-positive.
    """
    if block_m <= 0 or block_n <= 0 or block_k <= 0:
        raise ValueError("block_m/block_n/block_k must be positive.")
    block_m = max(8, _ceil_div(block_m, 8) * 8)
    block_n = max(128, _ceil_div(block_n, 128) * 128)
    block_k = max(128, _ceil_div(block_k, 128) * 128)
    return block_m, block_n, block_k


def _is_2d_blockspec_legal(block0: int, block1: int, dim0: int, dim1: int) -> bool:
    """Check whether a 2D BlockSpec satisfies Mosaic TPU lowering constraints.

    Mosaic requires the trailing dimension to be either the full dimension
    or a multiple of 128, and the second-to-trailing dimension to be either
    the full dimension or a multiple of 8.

    Args:
        block0: Block size for the first (sublane) dimension.
        block1: Block size for the second (lane) dimension.
        dim0: Full size of the first dimension after padding.
        dim1: Full size of the second dimension after padding.

    Returns:
        True if the block spec is legal for TPU lowering.
    """
    return (block1 == dim1 or block1 % 128 == 0) and (block0 == dim0 or block0 % 8 == 0)


def is_packed_tpu_legal_forward(
    x: jax.Array,
    w_q: jax.Array,
    scales: jax.Array,
    *,
    group_size: int,
    bits: int,
    block_m: int,
    block_n: int,
    block_k: int,
) -> bool:
    """Check if the packed fused TPU Pallas path is legal for forward matmul.

    Validates that quantization bit width, block dimensions, tensor shapes,
    and Mosaic tiling constraints are all satisfied for the packed (fused
    unpack + dequant + matmul) kernel path.

    Args:
        x: Activation tensor [M, K].
        w_q: Packed quantized weight tensor [K, ceil(N * bits / 32)].
        scales: Per-group scale tensor [K, N // group_size].
        group_size: Number of output elements per quantization group.
        bits: Quantization bit width, from 1 through 8.
        block_m: M-dimension tile size.
        block_n: N-dimension tile size.
        block_k: K-dimension tile size.

    Returns:
        True if all constraints are satisfied for the packed TPU path.
    """
    if bits < 1 or bits > 8:
        return False
    try:
        block_m, block_n, block_k = _normalize_tpu_blocks(block_m, block_n, block_k)
    except ValueError:
        return False

    value_alignment = _bit_aligned_values(bits)
    if block_n % value_alignment != 0 or block_n % group_size != 0:
        return False

    m, k = x.shape
    if w_q.shape[0] != k or scales.shape[0] != k:
        return False
    n = scales.shape[-1] * group_size
    if n <= 0:
        return False

    m_pad = _ceil_div(m, block_m) * block_m
    n_pad = _ceil_div(n, block_n) * block_n
    k_pad = _ceil_div(k, block_k) * block_k

    words_pad = _packed_words_for_values(n_pad, bits)
    groups_pad = n_pad // group_size
    if w_q.shape[1] > words_pad or scales.shape[1] > groups_pad:
        return False

    block_words = _packed_words_for_values(block_n, bits)
    block_groups = block_n // group_size
    return (
        _is_2d_blockspec_legal(block_m, block_k, m_pad, k_pad)
        and _is_2d_blockspec_legal(block_k, block_words, k_pad, words_pad)
        and _is_2d_blockspec_legal(block_k, block_groups, k_pad, groups_pad)
        and _is_2d_blockspec_legal(block_m, block_n, m_pad, n_pad)
    )


def is_packed_tpu_legal_input_grad(
    dy: jax.Array,
    w_q: jax.Array,
    scales: jax.Array,
    *,
    group_size: int,
    bits: int,
    block_m: int,
    block_n: int,
    block_k: int,
) -> bool:
    """Check if the packed fused TPU Pallas path is legal for the input gradient.

    Validates constraints for the backward pass dX = dY @ W^T using the
    packed kernel. The contraction axis is N (output dimension of the forward
    pass), so tiling constraints differ from the forward check.

    Args:
        dy: Upstream gradient tensor [M, N].
        w_q: Packed quantized weight tensor [K, ceil(N * bits / 32)].
        scales: Per-group scale tensor [K, N // group_size].
        group_size: Number of output elements per quantization group.
        bits: Quantization bit width, from 1 through 8.
        block_m: M-dimension tile size.
        block_n: N-dimension tile size.
        block_k: K-dimension tile size.

    Returns:
        True if all constraints are satisfied for the packed TPU dX path.
    """
    if bits < 1 or bits > 8:
        return False
    try:
        block_m, block_n, block_k = _normalize_tpu_blocks(block_m, block_n, block_k)
    except ValueError:
        return False

    value_alignment = _bit_aligned_values(bits)
    if block_n % value_alignment != 0 or block_n % group_size != 0:
        return False

    m, n = dy.shape
    k = w_q.shape[0]
    if scales.shape[0] != k:
        return False
    n_expected = scales.shape[-1] * group_size
    if n != n_expected:
        return False

    m_pad = _ceil_div(m, block_m) * block_m
    n_pad = _ceil_div(n, block_n) * block_n
    k_pad = _ceil_div(k, block_k) * block_k

    words_pad = _packed_words_for_values(n_pad, bits)
    groups_pad = n_pad // group_size
    if w_q.shape[1] > words_pad or scales.shape[1] > groups_pad:
        return False

    block_words = _packed_words_for_values(block_n, bits)
    block_groups = block_n // group_size
    return (
        _is_2d_blockspec_legal(block_m, block_n, m_pad, n_pad)
        and _is_2d_blockspec_legal(block_k, block_words, k_pad, words_pad)
        and _is_2d_blockspec_legal(block_k, block_groups, k_pad, groups_pad)
        and _is_2d_blockspec_legal(block_m, block_k, m_pad, k_pad)
    )


def _parse_bool_env(name: str, default: bool) -> bool:
    """Parse a boolean from an environment variable.

    Args:
        name: Environment variable name.
        default: Value to return when the variable is unset.

    Returns:
        True if the env var is set to a truthy string (1/true/yes/y/on),
        False otherwise.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_int_env(name: str, default: int) -> int:
    """Parse a non-negative integer from an environment variable.

    Args:
        name: Environment variable name.
        default: Value to return when the variable is unset or not parseable.

    Returns:
        Parsed integer clamped to >= 0, or *default* if unset/invalid.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(0, value)


def get_qmm_tpu_path() -> str:
    """Read the preferred QMM TPU execution path from the environment.

    Returns:
        Always "packed". Legacy values ("hybrid", "predecode") are accepted
        for backward compatibility and normalize to "packed".
    """
    path = os.getenv("EJKERNEL_QMM_TPU_PATH", "packed").strip().lower()
    if path not in _QMM_PATHS:
        return "packed"
    return "packed"


def get_predecode_cache_enabled() -> bool:
    """Check whether the predecoded dense weight LRU cache is enabled.

    Returns:
        True if caching is enabled (default: True).
    """
    return _parse_bool_env("EJKERNEL_QMM_TPU_PREDECODE_CACHE", True)


def get_predecode_cache_max_items() -> int:
    """Return the maximum number of items in the predecode LRU cache.

    Returns:
        Max cached entries (default: 2).
    """
    return _parse_int_env("EJKERNEL_QMM_TPU_PREDECODE_CACHE_MAX_ITEMS", _DEFAULT_PREDECODE_CACHE_MAX_ITEMS)


def get_predecode_max_bytes() -> int:
    """Return the maximum byte size allowed for a single predecoded weight buffer.

    Returns:
        Max bytes per predecoded buffer (default: 256 MiB).
    """
    return _parse_int_env("EJKERNEL_QMM_TPU_MAX_PREDECODE_BYTES", _DEFAULT_PREDECODE_MAX_BYTES)


def _get_tpu_version() -> int:
    """Return numeric TPU version, or -1 when unavailable."""
    try:
        kind = jax.devices()[0].device_kind
    except Exception:
        return -1
    match = re.match(r"^TPU[^\d]*(\d+)", kind)
    if match is None:
        return -1
    return int(match.group(1))


def get_qmm_tpu_vmem_limit_bytes() -> int:
    """Return VMEM compiler budget for TPU QMM kernels."""
    env_limit = _parse_int_env("EJKERNEL_QMM_TPU_VMEM_LIMIT_BYTES", -1)
    if env_limit > 0:
        return env_limit
    if _get_tpu_version() == 7:
        return _DEFAULT_TPU_V7_VMEM_LIMIT_BYTES
    return _DEFAULT_TPU_VMEM_LIMIT_BYTES


def estimate_qmm_tpu_vmem_limit_bytes(
    *,
    io_bytes: int,
    scratch_bytes: int,
    has_double_buffer: bool,
) -> int:
    """Estimate per-kernel VMEM usage and cap it to device budget."""
    estimated = 2 * (max(0, int(io_bytes)) + max(0, int(scratch_bytes)))
    if has_double_buffer:
        estimated += max(0, int(io_bytes))
    estimated = max(1, estimated)
    return min(estimated, get_qmm_tpu_vmem_limit_bytes())


def choose_packed_n_subtile(
    *,
    block_n: int,
    group_size: int,
    value_alignment: int,
    max_subtile: int = 256,
) -> int:
    """Pick an N-subtile width for packed kernels to reduce unpack/dequant pressure."""
    align = _lcm(group_size, value_alignment)
    target = min(block_n, max_subtile)
    target = max(align, (target // align) * align)
    n_subtile = target
    while n_subtile > align and block_n % n_subtile != 0:
        n_subtile -= align
    if n_subtile <= 0 or block_n % n_subtile != 0:
        return block_n
    return n_subtile


def _decode_e2m1(code: jax.Array) -> jax.Array:
    """Decode E2M1 (MXFP4) codes to float32.

    Args:
        code: Integer code array with values in [0, 7].

    Returns:
        Float32 array of decoded values.
    """
    c = code.astype(jnp.uint32)
    sign = ((c >> jnp.uint32(3)) & jnp.uint32(0x1)).astype(jnp.int32)
    exp = ((c >> jnp.uint32(1)) & jnp.uint32(0x3)).astype(jnp.int32)
    mant = (c & jnp.uint32(0x1)).astype(jnp.int32)
    mant_f = mant.astype(jnp.float32) * jnp.float32(0.5)
    subnormal = mant_f
    normal = (jnp.float32(1.0) + mant_f) * jnp.exp2((exp - jnp.int32(1)).astype(jnp.float32))
    vals = jnp.where(exp == 0, subnormal, normal)
    return jnp.where(sign == 0, vals, -vals)


def _decode_e4m3(code: jax.Array) -> jax.Array:
    """Decode E4M3 (FP8) codes to float32.

    Args:
        code: Integer code array with values in [0, 255].

    Returns:
        Float32 array of decoded values.
    """
    c = code.astype(jnp.uint32)
    sign = ((c >> jnp.uint32(7)) & jnp.uint32(0x1)).astype(jnp.int32)
    exp = ((c >> jnp.uint32(3)) & jnp.uint32(0xF)).astype(jnp.int32)
    mant = (c & jnp.uint32(0x7)).astype(jnp.int32)
    mant_f = mant.astype(jnp.float32) / jnp.float32(8.0)
    subnormal = mant_f * jnp.float32(0.015625)
    normal = (jnp.float32(1.0) + mant_f) * jnp.exp2((exp - jnp.int32(7)).astype(jnp.float32))
    vals = jnp.where(exp == 0, subnormal, normal)
    nan_mask = (exp == 0xF) & (mant == 0x7)
    vals = jnp.where(nan_mask, jnp.nan, vals)
    return jnp.where(sign == 0, vals, -vals)


def _decode_nf4(code: jax.Array) -> jax.Array:
    """Decode NF4 (NormalFloat4) codes to float32.

    Args:
        code: Integer code array with values in [0, 15].

    Returns:
        Float32 array of decoded values.
    """
    c = code.astype(jnp.int32)
    vals = jnp.where(
        c == 0,
        jnp.float32(-1.0),
        jnp.where(
            c == 1,
            jnp.float32(-0.6961928),
            jnp.where(
                c == 2,
                jnp.float32(-0.52507305),
                jnp.where(
                    c == 3,
                    jnp.float32(-0.3949175),
                    jnp.where(
                        c == 4,
                        jnp.float32(-0.28444138),
                        jnp.where(
                            c == 5,
                            jnp.float32(-0.18477343),
                            jnp.where(
                                c == 6,
                                jnp.float32(-0.091050036),
                                jnp.where(
                                    c == 7,
                                    jnp.float32(0.0),
                                    jnp.where(
                                        c == 8,
                                        jnp.float32(0.0795803),
                                        jnp.where(
                                            c == 9,
                                            jnp.float32(0.1609302),
                                            jnp.where(
                                                c == 10,
                                                jnp.float32(0.2461123),
                                                jnp.where(
                                                    c == 11,
                                                    jnp.float32(0.33791524),
                                                    jnp.where(
                                                        c == 12,
                                                        jnp.float32(0.44070983),
                                                        jnp.where(
                                                            c == 13,
                                                            jnp.float32(0.562617),
                                                            jnp.where(
                                                                c == 14,
                                                                jnp.float32(0.72295684),
                                                                jnp.float32(1.0),
                                                            ),
                                                        ),
                                                    ),
                                                ),
                                            ),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    return vals


def _unpack_packed_bits(words: jax.Array, bits: int) -> jax.Array:
    """Unpack 1-bit through 8-bit quantized values from packed 32-bit words.

    The packed representation is a contiguous little-endian bitstream. The
    output width is inferred from the number of whole values that fit in the
    provided words, so callers should pass word-aligned tiles whose value count
    is a multiple of ``32 / gcd(32, bits)``.

    Args:
        words: Packed weight array [K, N_packed] of uint32 words.
        bits: Quantization bit width, from 1 through 8.

    Returns:
        Unpacked uint32 code array [K, floor(N_packed * 32 / bits)].

    Raises:
        ValueError: If bits is outside 1..8.
    """
    if bits < 1 or bits > 8:
        raise ValueError("Packed bit unpack supports bits in [1, 8].")
    words = words.astype(jnp.uint32)
    if bits == 4:
        q = jnp.stack(
            [
                (words >> jnp.uint32(0)) & jnp.uint32(0xF),
                (words >> jnp.uint32(4)) & jnp.uint32(0xF),
                (words >> jnp.uint32(8)) & jnp.uint32(0xF),
                (words >> jnp.uint32(12)) & jnp.uint32(0xF),
                (words >> jnp.uint32(16)) & jnp.uint32(0xF),
                (words >> jnp.uint32(20)) & jnp.uint32(0xF),
                (words >> jnp.uint32(24)) & jnp.uint32(0xF),
                (words >> jnp.uint32(28)) & jnp.uint32(0xF),
            ],
            axis=-1,
        )
        return q.reshape(words.shape[0], words.shape[1] * 8)
    if bits == 8:
        q = jnp.stack(
            [
                (words >> jnp.uint32(0)) & jnp.uint32(0xFF),
                (words >> jnp.uint32(8)) & jnp.uint32(0xFF),
                (words >> jnp.uint32(16)) & jnp.uint32(0xFF),
                (words >> jnp.uint32(24)) & jnp.uint32(0xFF),
            ],
            axis=-1,
        )
        return q.reshape(words.shape[0], words.shape[1] * 4)
    out_width = words.shape[1] * 32 // bits
    value_idx = jnp.arange(out_width, dtype=jnp.uint32)
    bit_offsets = value_idx * jnp.uint32(bits)
    word_idx = bit_offsets // jnp.uint32(32)
    shifts = bit_offsets - word_idx * jnp.uint32(32)
    word0 = words[:, word_idx.astype(jnp.int32)]
    word1_idx = jnp.minimum(word_idx + jnp.uint32(1), jnp.uint32(max(0, words.shape[1] - 1)))
    word1 = words[:, word1_idx.astype(jnp.int32)]
    low_bits = jnp.minimum(jnp.uint32(32) - shifts, jnp.uint32(bits))
    high_bits = jnp.uint32(bits) - low_bits
    low_mask = (jnp.left_shift(jnp.uint32(1), low_bits) - jnp.uint32(1)).astype(jnp.uint32)
    high_mask = (jnp.left_shift(jnp.uint32(1), high_bits) - jnp.uint32(1)).astype(jnp.uint32)
    low = jnp.right_shift(word0, shifts) & low_mask
    high = word1 & high_mask
    return low | jnp.left_shift(high, low_bits)


def _unpack_bits_4_8(words: jax.Array, bits: int) -> jax.Array:
    """Backward-compatible alias for packed 1-bit through 8-bit unpacking."""
    return _unpack_packed_bits(words, bits)


def _expand_groups(values: jax.Array, group_size: int, width: int) -> jax.Array:
    """Repeat per-group values to match the full output width.

    Each group value is broadcast to ``group_size`` consecutive elements
    along the last axis, then truncated to ``width``.

    Args:
        values: Per-group array [K, num_groups].
        group_size: Number of output elements per group.
        width: Target output width (may be < num_groups * group_size).

    Returns:
        Expanded array [K, width].
    """
    groups = values.shape[-1]
    expanded = jnp.broadcast_to(values[..., :, None], (*values.shape, group_size))
    return expanded.reshape(values.shape[0], groups * group_size)[:, :width]


def _dequantize_tile(
    q: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None,
    mode: str,
    group_size: int,
) -> jax.Array:
    """Dequantize an unpacked code tile to float32 using per-group scales.

    Supports multiple quantization modes:
    - "affine": ``code * scale + bias`` (optional additive bias)
    - "nf4": NormalFloat4 lookup then multiply by scale
    - "mxfp4": E2M1 lookup with power-of-2 exponent scale
    - "mxfp8": E4M3 lookup with power-of-2 exponent scale
    - "nvfp4": E2M1 values with E4M3-decoded scale
    - "nvfp8": E4M3 values with E4M3-decoded scale

    Args:
        q: Unpacked code array [K, N] (uint32 codes).
        scales: Per-group scale array [K, N // group_size].
        biases: Optional per-group additive bias [K, N // group_size]
            (used only in "affine" mode).
        mode: Quantization mode string.
        group_size: Number of output elements per quantization group.

    Returns:
        Dequantized float32 weight tile [K, N].

    Raises:
        ValueError: If *mode* is not recognised.
    """
    width = q.shape[-1]
    if mode == "affine":
        vals = q.astype(jnp.int32).astype(jnp.float32)
        s = _expand_groups(scales.astype(jnp.float32), group_size, width)
        w = vals * s
        if biases is not None:
            b = _expand_groups(biases.astype(jnp.float32), group_size, width)
            w = w + b
        return w
    if mode == "nf4":
        vals = _decode_nf4(q)
        s = _expand_groups(scales.astype(jnp.float32), group_size, width)
        return vals * s
    if mode == "mxfp4":
        vals = _decode_e2m1(q)
        exp = scales.astype(jnp.int8).astype(jnp.int32).astype(jnp.float32)
        s = _expand_groups(jnp.exp2(exp), group_size, width)
        return vals * s
    if mode == "mxfp8":
        vals = _decode_e4m3(q)
        exp = scales.astype(jnp.int8).astype(jnp.int32).astype(jnp.float32)
        s = _expand_groups(jnp.exp2(exp), group_size, width)
        return vals * s
    if mode == "nvfp4":
        vals = _decode_e2m1(q)
        s_decoded = _decode_e4m3(scales.astype(jnp.uint32))
        s = _expand_groups(s_decoded, group_size, width)
        return vals * s
    if mode == "nvfp8":
        vals = _decode_e4m3(q)
        s_decoded = _decode_e4m3(scales.astype(jnp.uint32))
        s = _expand_groups(s_decoded, group_size, width)
        return vals * s
    raise ValueError(f"Unsupported quantization mode: {mode}")


def _predecode_dense_weight(
    w_q: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None,
    *,
    group_size: int,
    bits: int,
    mode: str,
) -> jax.Array:
    """Materialize a dense bfloat16 weight matrix from quantized representation.

    Unpacks the packed weight, dequantizes to float32 using the specified
    mode and scales, then casts to bfloat16 for TPU matmul.

    Args:
        w_q: Packed quantized weight [K, N_packed].
        scales: Per-group scales [K, N // group_size].
        biases: Optional per-group biases [K, N // group_size].
        group_size: Elements per quantization group.
        bits: Bit width, from 1 through 8.
        mode: Quantization mode (e.g. "affine", "nf4").

    Returns:
        Dense bfloat16 weight [K, N].

    Raises:
        ValueError: If bits is outside 1..8 or packed width is too small.
    """
    if bits < 1 or bits > 8:
        raise ValueError("TPU predecode path supports bits in [1, 8].")
    n = scales.shape[-1] * group_size
    q_full = _unpack_packed_bits(w_q, bits)
    if q_full.shape[-1] < n:
        raise ValueError("Packed weight width is smaller than scales-implied output width.")
    q = q_full[:, :n]
    w = _dequantize_tile(q, scales, biases, mode, group_size)
    return w.astype(jnp.bfloat16)


def _device_key(arr: jax.Array) -> tuple | None:
    """Extract a hashable device identifier from a JAX array for caching.

    Args:
        arr: JAX array whose device placement to identify.

    Returns:
        Tuple of (platform, device_id, device_str), or None if unavailable.
    """
    try:
        dev = arr.device()
    except Exception:
        try:
            devices = list(arr.devices())
            dev = devices[0] if devices else None
        except Exception:
            dev = None
    if dev is None:
        return None
    return (getattr(dev, "platform", None), getattr(dev, "id", None), str(dev))


def _is_tracer(x: object) -> bool:
    """Check whether *x* is a JAX abstract tracer (i.e. not a concrete value).

    Args:
        x: Object to check.

    Returns:
        True if *x* is a ``jax.core.Tracer`` instance.
    """
    return isinstance(x, jax_core.Tracer)


def _estimate_predecode_bytes(k: int, n: int) -> int:
    """Estimate the memory footprint of a predecoded dense bfloat16 weight.

    Args:
        k: Number of rows (contraction dimension).
        n: Number of columns (output dimension).

    Returns:
        Estimated byte count (k * n * 2 for bfloat16).
    """
    return int(k) * int(n) * jnp.dtype(jnp.bfloat16).itemsize


def get_predecoded_dense_weight(
    w_q: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None,
    *,
    group_size: int,
    bits: int,
    mode: str,
) -> jax.Array:
    """Get or compute a dense bfloat16 weight from quantized tensors.

    Uses a thread-safe LRU cache keyed on array identity, shape, dtype,
    device, and quantization parameters. When caching is disabled or the
    inputs are JAX tracers, the weight is computed eagerly without caching.

    Args:
        w_q: Packed quantized weight [K, N_packed].
        scales: Per-group scales [K, N // group_size].
        biases: Optional per-group biases (affine mode only).
        group_size: Elements per quantization group.
        bits: Quantization bit width, from 1 through 8.
        mode: Quantization mode string.

    Returns:
        Dense bfloat16 weight [K, N].

    Raises:
        ValueError: If the predecoded buffer would exceed the byte cap.
    """
    k = int(w_q.shape[0])
    n = int(scales.shape[-1]) * int(group_size)
    est_bytes = _estimate_predecode_bytes(k, n)
    max_bytes = get_predecode_max_bytes()
    if max_bytes > 0 and est_bytes > max_bytes:
        raise ValueError(
            f"Predecode buffer exceeds cap ({est_bytes} bytes > {max_bytes} bytes). "
            "Increase EJKERNEL_QMM_TPU_MAX_PREDECODE_BYTES or use packed/XLA path."
        )

    cache_allowed = get_predecode_cache_enabled()
    cache_allowed = cache_allowed and not _is_tracer(w_q) and not _is_tracer(scales)
    cache_allowed = cache_allowed and (biases is None or not _is_tracer(biases))
    if not cache_allowed:
        return _predecode_dense_weight(
            w_q,
            scales,
            biases,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )

    key = (
        _device_key(w_q),
        id(w_q),
        id(scales),
        id(biases) if biases is not None else None,
        w_q.shape,
        w_q.dtype,
        scales.shape,
        scales.dtype,
        biases.shape if biases is not None else None,
        biases.dtype if biases is not None else None,
        group_size,
        bits,
        mode,
    )

    with _PREDECODE_CACHE_LOCK:
        cached = _PREDECODE_CACHE.get(key)
        if cached is not None:
            _PREDECODE_CACHE.move_to_end(key)
            return cached

    decoded = _predecode_dense_weight(
        w_q,
        scales,
        biases,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    max_items = get_predecode_cache_max_items()
    if max_items <= 0:
        return decoded

    with _PREDECODE_CACHE_LOCK:
        _PREDECODE_CACHE[key] = decoded
        _PREDECODE_CACHE.move_to_end(key)
        while len(_PREDECODE_CACHE) > max_items:
            _PREDECODE_CACHE.popitem(last=False)
    return decoded


def pallas_dense_matmul(
    lhs: jax.Array,
    rhs: jax.Array,
    *,
    transpose_rhs: bool,
    block_m: int,
    block_n: int,
    block_k: int,
) -> jax.Array:
    """Tiled dense matrix multiplication on TPU via Pallas.

    Computes ``lhs @ rhs`` (or ``lhs @ rhs.T`` when transpose_rhs=True)
    using a 3D grid of (M, N, K) tiles with bfloat16 inputs and float32
    accumulation. Both operands are padded to multiples of the block sizes
    and the result is sliced back to the original dimensions.

    Grid strategy:
        - M and N dimensions are parallelised across TPU cores
        - K dimension is accumulated sequentially within each core
        - A VMEM scratch accumulator avoids repeated HBM round-trips

    Args:
        lhs: Left-hand-side activation [M, K].
        rhs: Right-hand-side weight [K, N] or [N, K] if transposed.
        transpose_rhs: If True, rhs has shape [N, K] and is transposed.
        block_m: Tile size for M dimension.
        block_n: Tile size for N dimension.
        block_k: Tile size for K (contraction) dimension.

    Returns:
        Float32 result [M, N].

    Raises:
        ValueError: If contraction dimensions do not match.
    """
    block_m, block_n, block_k = _normalize_tpu_blocks(block_m, block_n, block_k)

    m, k = lhs.shape
    if transpose_rhs:
        n = rhs.shape[0]
        k_rhs = rhs.shape[1]
    else:
        k_rhs = rhs.shape[0]
        n = rhs.shape[1]
    if k != k_rhs:
        raise ValueError("Dense Pallas matmul dimension mismatch on contraction axis.")

    m_pad = _ceil_div(m, block_m) * block_m
    n_pad = _ceil_div(n, block_n) * block_n
    k_pad = _ceil_div(k, block_k) * block_k

    lhs_pad = _pad_2d(lhs, m_pad - m, k_pad - k).astype(jnp.bfloat16)
    if transpose_rhs:
        rhs_pad = _pad_2d(rhs, n_pad - n, k_pad - k).astype(jnp.bfloat16)
        dot_dims = (((1,), (1,)), ((), ()))
        rhs_spec = pl.BlockSpec((block_n, block_k), lambda m_i, n_i, k_i: (n_i, k_i))
    else:
        rhs_pad = _pad_2d(rhs, k_pad - k, n_pad - n).astype(jnp.bfloat16)
        dot_dims = (((1,), (0,)), ((), ()))
        rhs_spec = pl.BlockSpec((block_k, block_n), lambda m_i, n_i, k_i: (k_i, n_i))

    num_m = m_pad // block_m
    num_n = n_pad // block_n
    num_k = k_pad // block_k

    def _kernel(lhs_ref, rhs_ref, out_ref, acc_ref):
        k_i = pl.program_id(2)

        @pl.when(k_i == 0)
        def _zero_acc():
            acc_ref[...] = jnp.zeros_like(acc_ref)

        acc_ref[...] += jax.lax.dot_general(
            lhs_ref[...].astype(jnp.bfloat16),
            rhs_ref[...].astype(jnp.bfloat16),
            dot_dims,
            preferred_element_type=jnp.float32,
        )

        @pl.when(k_i == num_k - 1)
        def _store():
            out_ref[...] = acc_ref[...]

    lhs_spec = pl.BlockSpec((block_m, block_k), lambda m_i, n_i, k_i: (m_i, k_i))
    out_spec = pl.BlockSpec((block_m, block_n), lambda m_i, n_i, k_i: (m_i, n_i))
    grid = (num_m, num_n, num_k)

    flops = 2 * m_pad * k_pad * n_pad
    lhs_bytes = m_pad * k_pad * jnp.dtype(jnp.bfloat16).itemsize
    rhs_bytes = n_pad * k_pad * jnp.dtype(jnp.bfloat16).itemsize
    out_bytes = m_pad * n_pad * jnp.dtype(jnp.float32).itemsize
    tile_lhs_bytes = block_m * block_k * jnp.dtype(jnp.bfloat16).itemsize
    tile_rhs_bytes = block_n * block_k * jnp.dtype(jnp.bfloat16).itemsize
    tile_out_bytes = block_m * block_n * jnp.dtype(jnp.float32).itemsize
    vmem_limit_bytes = estimate_qmm_tpu_vmem_limit_bytes(
        io_bytes=tile_lhs_bytes + tile_rhs_bytes + tile_out_bytes,
        scratch_bytes=tile_out_bytes,
        has_double_buffer=(num_m > 1 or num_n > 1 or num_k > 1),
    )
    cost_estimate = pl.CostEstimate(
        flops=flops,
        bytes_accessed=lhs_bytes + rhs_bytes + out_bytes,
        transcendentals=0,
    )

    out = pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((m_pad, n_pad), jnp.float32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=0,
            in_specs=[lhs_spec, rhs_spec],
            out_specs=out_spec,
            grid=grid,
            scratch_shapes=[pltpu.VMEM((block_m, block_n), jnp.float32)],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("parallel", "parallel", "arbitrary"),
            vmem_limit_bytes=vmem_limit_bytes,
        ),
        cost_estimate=cost_estimate,
    )(lhs_pad, rhs_pad)
    return out[:m, :n]


__all__ = (
    "_ceil_div",
    "_decode_e2m1",
    "_decode_e4m3",
    "_decode_nf4",
    "_dequantize_tile",
    "_normalize_tpu_blocks",
    "_pad_2d",
    "_pad_2d_optional",
    "_unpack_bits_4_8",
    "choose_packed_n_subtile",
    "estimate_qmm_tpu_vmem_limit_bytes",
    "get_predecoded_dense_weight",
    "get_qmm_tpu_path",
    "get_qmm_tpu_vmem_limit_bytes",
    "is_packed_tpu_legal_forward",
    "is_packed_tpu_legal_input_grad",
    "pallas_dense_matmul",
)
