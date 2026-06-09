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

"""CuTe DSL fused quantized matmul implementation.

This module provides CuTe DSL kernels that fuse bit unpacking,
dequantization, and GEMM for all quantization modes.  The default path
uses shared-memory tiling with register-level accumulation; a naive
scalar fallback is retained and selectable via the environment variable
``EJKERNEL_CUTE_QMM_USE_NAIVE=1``.
"""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Callable
from typing import Any

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import jax
import jax.numpy as jnp
from cutlass.cutlass_dsl import dsl_user_op
from jax import custom_batching

from ejkernel.callib._cute_call import cute_call
from ejkernel.callib._cute_ffi import build_cute_ffi_call, has_cute_ffi_support
from ejkernel.quantization._utils.qparams import (
    normalize_gemv_mode,
    normalize_revsplitk_mode,
    normalize_revsplitk_parts,
    select_qmm_kernel_family,
)

os.environ.setdefault("CUTE_DSL_ENABLE_TVM_FFI", "1")

_DEFAULT_BLOCK_M = 64
_DEFAULT_BLOCK_N = 64
_DEFAULT_BLOCK_K = 32
_MAX_THREADS_PER_BLOCK = 1024
_MIN_THREADS_PER_BLOCK = 64
_THREAD_TILE_M = 4
_THREAD_TILE_N = 4

_NF4_LUT_VALUES: tuple[float, ...] = (
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
)

_QMM_CALL_CACHE: dict[tuple[Any, ...], tuple[Callable[..., jax.Array], str]] = {}
_QMM_CALL_CACHE_LOCK = threading.Lock()


def _dtype_to_cutlass(dtype) -> type[cutlass.Numeric]:
    """Map a JAX/NumPy dtype to the corresponding CUTLASS scalar type."""
    dt = jnp.dtype(dtype)
    if dt == jnp.float16:
        return cutlass.Float16
    if dt == jnp.bfloat16:
        return cutlass.BFloat16
    if dt == jnp.float32:
        return cutlass.Float32
    if dt == jnp.int8:
        return cutlass.Int8
    if dt == jnp.uint8:
        return cutlass.Uint8
    if dt == jnp.int32:
        return cutlass.Int32
    if dt == jnp.uint32:
        return cutlass.Uint32
    raise TypeError(f"Unsupported dtype for CuTe DSL: {dt}")


def _fake_tensor(dtype: type[cutlass.Numeric], shape: tuple[int, ...]):
    """Create a compact fake tensor descriptor for CuTe compilation.

    Uses ``assumed_align=16`` (16 bytes = 128 bits) and stride
    divisibility ``128 // dtype.width`` so that kernels using
    ``CopyG2SOp`` with 128-bit cp.async copies pass MLIR alignment
    verification.  JAX/XLA guarantees ≥128-byte alignment for device
    buffers, and all JAX arrays have strides that are multiples of
    their element size.
    """
    rank = len(shape)
    stride_order = tuple(range(rank - 1, -1, -1))
    ft = cute.runtime.make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=stride_order,
        assumed_align=16,
    )
    if rank == 2:
        leading_dim = rank - 1
        divisibility = max(1, 128 // dtype.width)
        ft.mark_layout_dynamic(leading_dim=leading_dim)
        ft.mark_compact_shape_dynamic(
            mode=leading_dim,
            stride_order=stride_order,
            divisibility=divisibility,
        )
    return ft


@dsl_user_op
def _unpack_bits(
    packed: cute.Tensor,
    row: cutlass.Int32,
    elem_idx: cutlass.Int32,
    bits: int,
    *,
    loc=None,
):
    """Decode one quantized element from a packed uint32 weight row.

    Handles cross-word bit fields: when the *bits*-wide field starting at
    ``elem_idx * bits`` straddles a 32-bit word boundary, the function reads
    the low portion from ``packed[row, word0]`` and the high portion from
    ``packed[row, word0 + 1]``, then assembles the full value.

    Args:
        packed: 2-D CuTe tensor of ``uint32`` words, shape
            ``(num_rows, words_per_row)``.
        row: Row index into *packed*.
        elem_idx: Element index within the row (not the word index).
        bits: Bit-width of each quantized element (e.g. 4 or 8).
        loc: CuTe DSL source-location metadata (do not pass explicitly).

    Returns:
        Decoded quantized code as ``cutlass.Int32``.
    """
    bits_i = cutlass.Int32(bits)
    bit_offset = elem_idx * bits_i
    word_idx = bit_offset // cutlass.Int32(32)
    shift = bit_offset - word_idx * cutlass.Int32(32)
    shift_u = cutlass.Uint32(shift)

    word0 = cutlass.Uint32(packed[(row, word_idx)])
    split = (shift + bits_i) > cutlass.Int32(32)
    low_bits = cutlass.Int32(32) - shift
    low_bits = cutlass.select_(split, low_bits, bits_i)
    low_bits_u = cutlass.Uint32(low_bits)
    low_mask = (cutlass.Uint32(1) << low_bits_u) - cutlass.Uint32(1)
    low = (word0 >> shift_u) & low_mask

    word1_idx = word_idx + cutlass.Int32(1)
    max_word = cutlass.Int32(packed.shape[1] - 1)
    word1_idx = cutlass.select_(word1_idx > max_word, max_word, word1_idx)
    word1 = cutlass.Uint32(packed[(row, word1_idx)])

    high_bits = bits_i - low_bits
    high_bits_u = cutlass.Uint32(high_bits)
    high_mask = (cutlass.Uint32(1) << high_bits_u) - cutlass.Uint32(1)
    high = word1 & high_mask
    val = low | (high << low_bits_u)

    return cutlass.Int32(val)


@dsl_user_op
def _e2m1_value(code: cutlass.Int32, *, loc=None):
    """Decode an E2M1 4-bit float code into a float32 value.

    E2M1 is the MX-FP4 element format: 1 sign bit, 2 exponent bits,
    1 mantissa bit, with an exponent bias of 1.  Subnormals (exp == 0)
    are represented as ``sign * mantissa * 0.5``.

    Args:
        code: 4-bit integer code in the range ``[0, 15]``.
        loc: CuTe DSL source-location metadata (do not pass explicitly).

    Returns:
        Decoded value as ``cutlass.Float32``.
    """
    code_u = cutlass.Uint32(code)
    sign = (code_u >> cutlass.Uint32(3)) & cutlass.Uint32(1)
    exp = (code_u >> cutlass.Uint32(1)) & cutlass.Uint32(0x3)
    mant = code_u & cutlass.Uint32(0x1)

    mant_f = cutlass.Float32(mant) * cutlass.Float32(0.5)
    bias = cutlass.Int32(1)
    exp_f = cutlass.Float32(cutlass.Int32(exp) - bias)
    val = (cutlass.Float32(1.0) + mant_f) * cute.math.exp2(exp_f)

    val = cutlass.select_(exp == cutlass.Uint32(0), mant_f, val)
    val = cutlass.select_(sign != cutlass.Uint32(0), -val, val)
    return val


@dsl_user_op
def _e4m3_value(code: cutlass.Int32, *, loc=None):
    """Decode an E4M3 8-bit float code into a float32 value.

    E4M3 is the FP8 element format: 1 sign bit, 4 exponent bits, 3 mantissa
    bits, with an exponent bias of 7.  Subnormals (exp == 0) use
    ``sign * mantissa * 2^(1 - 7)``.  The special pattern ``exp = 0xF,
    mant = 0x7`` is decoded as ``NaN``.

    Args:
        code: 8-bit integer code in the range ``[0, 255]``.
        loc: CuTe DSL source-location metadata (do not pass explicitly).

    Returns:
        Decoded value as ``cutlass.Float32``.
    """
    code_u = cutlass.Uint32(code)
    sign = (code_u >> cutlass.Uint32(7)) & cutlass.Uint32(1)
    exp = (code_u >> cutlass.Uint32(3)) & cutlass.Uint32(0xF)
    mant = code_u & cutlass.Uint32(0x7)

    mant_f = cutlass.Float32(mant) * cutlass.Float32(1.0 / 8.0)
    bias = cutlass.Int32(7)
    exp_f = cutlass.Float32(cutlass.Int32(exp) - bias)
    val = (cutlass.Float32(1.0) + mant_f) * cute.math.exp2(exp_f)

    sub = mant_f * cute.math.exp2(cutlass.Float32(1 - 7))
    val = cutlass.select_(exp == cutlass.Uint32(0), sub, val)

    is_nan = (exp == cutlass.Uint32(0xF)) & (mant == cutlass.Uint32(0x7))
    val = cutlass.select_(is_nan, cutlass.Float32(float("nan")), val)

    val = cutlass.select_(sign != cutlass.Uint32(0), -val, val)
    return val


@dsl_user_op
def _nf4_value(code: cutlass.Int32, *, loc=None):
    """Decode an NF4 code into a float32 value using the canonical 16-entry LUT.

    NF4 (Normal Float 4) is the QLoRA quantization format; the 16 values are
    chosen to match the quantiles of a standard normal distribution, stored in
    :data:`_NF4_LUT_VALUES`.  This function implements the LUT as a chain of
    ``cutlass.select_`` calls, which the CuTe DSL lowers to predicated moves.

    Args:
        code: 4-bit integer code in the range ``[0, 15]``.
        loc: CuTe DSL source-location metadata (do not pass explicitly).

    Returns:
        Decoded NF4 value as ``cutlass.Float32``.
    """
    val = cutlass.Float32(0.0)
    val = cutlass.select_(code == cutlass.Int32(0), cutlass.Float32(-1.0), val)
    val = cutlass.select_(code == cutlass.Int32(1), cutlass.Float32(-0.6961928009986877), val)
    val = cutlass.select_(code == cutlass.Int32(2), cutlass.Float32(-0.5250730514526367), val)
    val = cutlass.select_(code == cutlass.Int32(3), cutlass.Float32(-0.39491748809814453), val)
    val = cutlass.select_(code == cutlass.Int32(4), cutlass.Float32(-0.28444138169288635), val)
    val = cutlass.select_(code == cutlass.Int32(5), cutlass.Float32(-0.18477343022823334), val)
    val = cutlass.select_(code == cutlass.Int32(6), cutlass.Float32(-0.09105003625154495), val)
    val = cutlass.select_(code == cutlass.Int32(7), cutlass.Float32(0.0), val)
    val = cutlass.select_(code == cutlass.Int32(8), cutlass.Float32(0.07958029955625534), val)
    val = cutlass.select_(code == cutlass.Int32(9), cutlass.Float32(0.16093020141124725), val)
    val = cutlass.select_(code == cutlass.Int32(10), cutlass.Float32(0.24611230194568634), val)
    val = cutlass.select_(code == cutlass.Int32(11), cutlass.Float32(0.33791524171829224), val)
    val = cutlass.select_(code == cutlass.Int32(12), cutlass.Float32(0.44070982933044434), val)
    val = cutlass.select_(code == cutlass.Int32(13), cutlass.Float32(0.5626170039176941), val)
    val = cutlass.select_(code == cutlass.Int32(14), cutlass.Float32(0.7229568362236023), val)
    val = cutlass.select_(code == cutlass.Int32(15), cutlass.Float32(1.0), val)
    return val


@dsl_user_op
def _exp2_from_e8m0(exp_code: cutlass.Uint32, *, loc=None):
    """Decode an E8M0 scale exponent and return ``2 ** (exp_code - bias)`` in fp32.

    E8M0 is the MX-series block-scale format: an 8-bit signed integer with no
    mantissa bits and no sign bit, interpreted as a biased exponent.  The value
    is converted to ``int8`` first so that signed arithmetic applies, yielding
    a power-of-two scale factor in float32.

    Args:
        exp_code: 8-bit unsigned exponent code.
        loc: CuTe DSL source-location metadata (do not pass explicitly).

    Returns:
        ``2 ** exp`` as ``cutlass.Float32``, where ``exp`` is the signed
        interpretation of *exp_code*.
    """
    exp_i8 = cutlass.Int8(exp_code)
    exp_f = cutlass.Float32(exp_i8)
    return cute.math.exp2(exp_f)


def _build_dequant_fn(*, mode: str, with_bias: bool):
    """Build a mode-specific dequantization callback used by fused QMM kernels.

    Returns a callable ``(row_idx, group_idx, q_code, scales[, biases]) -> Float32``
    that decodes a single quantized weight element according to the given
    quantization mode.  Supported modes: ``affine`` (with bias), ``nf4``,
    ``mxfp4``, ``mxfp8``, ``nvfp4``, ``nvfp8``.

    Args:
        mode: Quantization mode string (case-insensitive).
        with_bias: Whether the dequantization uses per-group bias (affine mode).

    Returns:
        A dequantization callable suitable for use inside CuTe DSL kernels.

    Raises:
        ValueError: If *mode* is not recognised.
    """
    mode = mode.lower()
    if with_bias and mode == "affine":

        def _dequant(row_idx, group_idx, q_code, scales, biases):
            scale = cutlass.Float32(scales[(row_idx, group_idx)])
            bias = cutlass.Float32(biases[(row_idx, group_idx)])
            return cutlass.Float32(q_code) * scale + bias

        return _dequant

    if mode == "nf4":

        def _dequant(row_idx, group_idx, q_code, scales, biases=None):
            scale = cutlass.Float32(scales[(row_idx, group_idx)])
            return _nf4_value(q_code) * scale

        return _dequant

    if mode == "mxfp4":

        def _dequant(row_idx, group_idx, q_code, scales, biases=None):
            exp_code = cutlass.Uint32(scales[(row_idx, group_idx)])
            return _e2m1_value(q_code) * _exp2_from_e8m0(exp_code)

        return _dequant

    if mode == "mxfp8":

        def _dequant(row_idx, group_idx, q_code, scales, biases=None):
            exp_code = cutlass.Uint32(scales[(row_idx, group_idx)])
            return _e4m3_value(q_code) * _exp2_from_e8m0(exp_code)

        return _dequant

    if mode == "nvfp4":

        def _dequant(row_idx, group_idx, q_code, scales, biases=None):
            scale_code = cutlass.Int32(scales[(row_idx, group_idx)])
            return _e2m1_value(q_code) * _e4m3_value(scale_code)

        return _dequant

    if mode == "nvfp8":

        def _dequant(row_idx, group_idx, q_code, scales, biases=None):
            scale_code = cutlass.Int32(scales[(row_idx, group_idx)])
            return _e4m3_value(q_code) * _e4m3_value(scale_code)

        return _dequant

    raise ValueError(f"Unsupported quantization mode for CuTe QMM: {mode}")


def _normalize_block(value: int | None, default: int) -> int:
    """Normalize a positive block size integer, falling back to *default* on error.

    Args:
        value: Requested block size, or ``None`` to use *default*.
        default: Value to use when *value* is ``None`` or cannot be converted.

    Returns:
        Positive integer block size, always at least 1.
    """
    try:
        block = int(default if value is None else value)
    except Exception:
        block = int(default)
    return max(1, block)


def _round_up_to_multiple(value: int, multiple: int) -> int:
    """Round *value* up to the nearest multiple of *multiple*."""
    return ((value + multiple - 1) // multiple) * multiple


def _smem_bytes(tile_m: int, tile_n: int, tile_k: int, dtype_bytes: int = 2) -> int:
    """Estimate peak shared-memory usage in bytes for a single tile pair.

    Uses the formula ``(tile_m * tile_k + tile_k * tile_n) * dtype_bytes + 1024``
    with a 1 KiB overhead for kernel bookkeeping and alignment padding.

    Args:
        tile_m: Tile height (output rows).
        tile_n: Tile width (output columns).
        tile_k: K-dimension tile depth.
        dtype_bytes: Bytes per element (default 2 for fp16/bf16).

    Returns:
        Estimated shared-memory requirement in bytes.
    """
    return (tile_m * tile_k + tile_k * tile_n) * dtype_bytes + 1024


def _use_naive_kernel() -> bool:
    """Return ``True`` when ``EJKERNEL_CUTE_QMM_USE_NAIVE=1`` is set in the environment."""
    raw = os.getenv("EJKERNEL_CUTE_QMM_USE_NAIVE", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _threads_for_tile(tile_m: int, tile_n: int) -> int:
    """Choose a practical CUDA thread count for the naive scalar kernel tile.

    Targets ``min(tile_m * tile_n, 512)`` threads, then clamps to
    ``[_MIN_THREADS_PER_BLOCK, _MAX_THREADS_PER_BLOCK]`` and rounds up
    to the nearest warp (multiple of 32).

    Args:
        tile_m: Tile height.
        tile_n: Tile width.

    Returns:
        Number of CUDA threads per thread block.
    """
    tile_area = max(1, int(tile_m) * int(tile_n))
    target = min(tile_area, 512)
    target = max(_MIN_THREADS_PER_BLOCK, target)
    target = min(_MAX_THREADS_PER_BLOCK, _round_up_to_multiple(target, 32))
    return target


def _infer_output_shape(
    *,
    x: jax.Array,
    scales: jax.Array,
    group_size: int,
    transpose: bool,
) -> tuple[int, int]:
    """Infer the ``(M, N)`` output shape of the fused QMM from input metadata.

    When the weight is not transposed the scales tensor has shape
    ``(K, num_groups)`` so ``N = scales.shape[1] * group_size``.
    When transposed the weight is ``(N, ...)`` so ``N = scales.shape[0]``.

    Args:
        x: Input activation, shape ``(M, K)``.
        scales: Per-group scale tensor.
        group_size: Number of weight elements per quantization group.
        transpose: Whether the weight matrix is transposed.

    Returns:
        Output shape ``(M, N)``.
    """
    m = int(x.shape[0])
    if transpose:
        n = int(scales.shape[0])
    else:
        n = int(scales.shape[1]) * int(group_size)
    return (m, n)


def _shape_key(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Convert an array shape to a hashable ``tuple[int, ...]`` cache key."""
    return tuple(int(d) for d in shape)


def _build_naive_qmm_host_fns(
    *,
    mode: str,
    bits: int,
    group_size: int,
    out_dtype: type[cutlass.Numeric],
    with_bias: bool,
    transpose: bool,
    tile_m: int,
    tile_n: int,
    tile_k: int,
):
    """Build naive scalar fused dequant+matmul CuTe host launchers.

    Creates a CuTe DSL kernel that fuses bit-unpacking, dequantization,
    and GEMM using a simple scalar accumulation strategy.  Each thread
    processes one or more ``(M, N)`` output elements in a strided loop,
    iterating over the K dimension in tiles.  This is the fallback path
    used when the tiled SMEM and MMA paths are not applicable.

    Parallelization: 2-D grid ``(ceil(M/tile_m), ceil(N/tile_n))`` with
    threads striding across the ``tile_m * tile_n`` tile area.

    Args:
        mode: Quantization mode (e.g. ``"affine"``, ``"nf4"``).
        bits: Bit-width of quantized weights (e.g. 4, 8).
        group_size: Number of elements per quantization group.
        out_dtype: CUTLASS output dtype (e.g. ``cutlass.BFloat16``).
        with_bias: Whether per-group dequantization bias is used.
        transpose: Whether the quantized weight matrix is transposed.
        tile_m: Output tile height.
        tile_n: Output tile width.
        tile_k: K-dimension tile size for the inner reduction loop.

    Returns:
        Tuple of ``(runtime_launcher, jax_stream_launcher)`` callables.
    """
    dequant = _build_dequant_fn(mode=mode, with_bias=with_bias)
    threads = _threads_for_tile(tile_m, tile_n)
    tile_area = tile_m * tile_n

    if with_bias:

        @cute.kernel
        def qmm_kernel(
            x: cute.Tensor,
            w_q: cute.Tensor,
            scales: cute.Tensor,
            biases: cute.Tensor,
            out: cute.Tensor,
        ):
            bidx, bidy, _ = cute.arch.block_idx()
            tidx, _, _ = cute.arch.thread_idx()

            m_dim = cutlass.Int32(out.shape[0])
            n_dim = cutlass.Int32(out.shape[1])
            k_dim = cutlass.Int32(x.shape[1])
            tile_m_i = cutlass.Int32(tile_m)
            tile_n_i = cutlass.Int32(tile_n)
            tile_k_i = cutlass.Int32(tile_k)
            tile_area_i = cutlass.Int32(tile_area)
            thread_stride = cutlass.Int32(threads)
            group_size_i = cutlass.Int32(group_size)

            work_idx = cutlass.Int32(tidx)
            while work_idx < tile_area_i:
                row_off = work_idx // tile_n_i
                col_off = work_idx - row_off * tile_n_i
                row = cutlass.Int32(bidx) * tile_m_i + row_off
                col = cutlass.Int32(bidy) * tile_n_i + col_off

                if row < m_dim and col < n_dim:
                    acc = cutlass.Float32(0.0)
                    k_start = cutlass.Int32(0)
                    while k_start < k_dim:
                        k_end = k_start + tile_k_i
                        k_end = cutlass.select_(k_end > k_dim, k_dim, k_end)
                        for kk in cutlass.range(k_start, k_end):
                            x_val = cutlass.Float32(x[(row, kk)])
                            w_val = cutlass.Float32(0.0)
                            if transpose:
                                w_row = col
                                group_idx = kk // group_size_i
                                q_code = _unpack_bits(w_q, w_row, kk, bits)
                                w_val = dequant(w_row, group_idx, q_code, scales, biases)
                            else:
                                w_row = kk
                                group_idx = col // group_size_i
                                q_code = _unpack_bits(w_q, w_row, col, bits)
                                w_val = dequant(w_row, group_idx, q_code, scales, biases)
                            acc += x_val * w_val
                        k_start = k_end
                    out[(row, col)] = out_dtype(acc)

                work_idx = work_idx + thread_stride

        @cute.jit
        def qmm_host_runtime(
            x: cute.Tensor,
            w_q: cute.Tensor,
            scales: cute.Tensor,
            biases: cute.Tensor,
            out: cute.Tensor,
        ):
            grid_x = (out.shape[0] + tile_m - 1) // tile_m
            grid_y = (out.shape[1] + tile_n - 1) // tile_n
            qmm_kernel(x, w_q, scales, biases, out).launch(
                grid=(grid_x, grid_y, 1),
                block=(threads, 1, 1),
            )

        @cute.jit
        def qmm_host_jax(
            stream: cuda.CUstream,
            x: cute.Tensor,
            w_q: cute.Tensor,
            scales: cute.Tensor,
            biases: cute.Tensor,
            out: cute.Tensor,
        ):
            grid_x = (out.shape[0] + tile_m - 1) // tile_m
            grid_y = (out.shape[1] + tile_n - 1) // tile_n
            qmm_kernel(x, w_q, scales, biases, out).launch(
                grid=(grid_x, grid_y, 1),
                block=(threads, 1, 1),
                stream=stream,
            )

        return qmm_host_runtime, qmm_host_jax

    @cute.kernel
    def qmm_kernel(
        x: cute.Tensor,
        w_q: cute.Tensor,
        scales: cute.Tensor,
        out: cute.Tensor,
    ):
        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        m_dim = cutlass.Int32(out.shape[0])
        n_dim = cutlass.Int32(out.shape[1])
        k_dim = cutlass.Int32(x.shape[1])
        tile_m_i = cutlass.Int32(tile_m)
        tile_n_i = cutlass.Int32(tile_n)
        tile_k_i = cutlass.Int32(tile_k)
        tile_area_i = cutlass.Int32(tile_area)
        thread_stride = cutlass.Int32(threads)
        group_size_i = cutlass.Int32(group_size)

        work_idx = cutlass.Int32(tidx)
        while work_idx < tile_area_i:
            row_off = work_idx // tile_n_i
            col_off = work_idx - row_off * tile_n_i
            row = cutlass.Int32(bidx) * tile_m_i + row_off
            col = cutlass.Int32(bidy) * tile_n_i + col_off

            if row < m_dim and col < n_dim:
                acc = cutlass.Float32(0.0)
                k_start = cutlass.Int32(0)
                while k_start < k_dim:
                    k_end = k_start + tile_k_i
                    k_end = cutlass.select_(k_end > k_dim, k_dim, k_end)
                    for kk in cutlass.range(k_start, k_end):
                        x_val = cutlass.Float32(x[(row, kk)])
                        w_val = cutlass.Float32(0.0)
                        if transpose:
                            w_row = col
                            group_idx = kk // group_size_i
                            q_code = _unpack_bits(w_q, w_row, kk, bits)
                            w_val = dequant(w_row, group_idx, q_code, scales)
                        else:
                            w_row = kk
                            group_idx = col // group_size_i
                            q_code = _unpack_bits(w_q, w_row, col, bits)
                            w_val = dequant(w_row, group_idx, q_code, scales)
                        acc += x_val * w_val
                    k_start = k_end
                out[(row, col)] = out_dtype(acc)

            work_idx = work_idx + thread_stride

    @cute.jit
    def qmm_host_runtime(
        x: cute.Tensor,
        w_q: cute.Tensor,
        scales: cute.Tensor,
        out: cute.Tensor,
    ):
        grid_x = (out.shape[0] + tile_m - 1) // tile_m
        grid_y = (out.shape[1] + tile_n - 1) // tile_n
        qmm_kernel(x, w_q, scales, out).launch(
            grid=(grid_x, grid_y, 1),
            block=(threads, 1, 1),
        )

    @cute.jit
    def qmm_host_jax(
        stream: cuda.CUstream,
        x: cute.Tensor,
        w_q: cute.Tensor,
        scales: cute.Tensor,
        out: cute.Tensor,
    ):
        grid_x = (out.shape[0] + tile_m - 1) // tile_m
        grid_y = (out.shape[1] + tile_n - 1) // tile_n
        qmm_kernel(x, w_q, scales, out).launch(
            grid=(grid_x, grid_y, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    return qmm_host_runtime, qmm_host_jax


def _build_tiled_qmm_host_fns(
    *,
    mode: str,
    bits: int,
    group_size: int,
    out_dtype: type[cutlass.Numeric],
    with_bias: bool,
    transpose: bool,
    tile_m: int,
    tile_n: int,
    tile_k: int,
):
    """Build tiled SMEM-based fused dequant+matmul CuTe host launchers.

    Creates a CuTe DSL kernel that uses shared-memory staging with explicit
    4x4 register tiling for the inner product.  Input activations (x) and
    dequantized weights are cooperatively loaded into shared memory tiles,
    then each thread computes a 4x4 register-resident accumulator block.
    For NF4 mode, a shared-memory lookup table is used for fast decoding.

    Parallelization: 2-D grid ``(ceil(M/tile_m), ceil(N/tile_n))`` with
    ``threads = (tile_n/4, tile_m/4)``.

    Args:
        mode: Quantization mode (e.g. ``"affine"``, ``"nf4"``).
        bits: Bit-width of quantized weights.
        group_size: Number of elements per quantization group.
        out_dtype: CUTLASS output dtype.
        with_bias: Whether per-group dequantization bias is used.
        transpose: Whether the quantized weight matrix is transposed.
        tile_m: Output tile height (must be divisible by 4).
        tile_n: Output tile width (must be divisible by 4).
        tile_k: K-dimension tile size for shared-memory staging.

    Returns:
        Tuple of ``(runtime_launcher, jax_stream_launcher)`` callables.

    Raises:
        ValueError: If tile dimensions are not divisible by the 4x4 thread tile
            or the resulting thread count is outside [32, 1024].
    """
    dequant = _build_dequant_fn(mode=mode, with_bias=with_bias)

    ttm = _THREAD_TILE_M
    ttn = _THREAD_TILE_N

    if tile_m % ttm != 0 or tile_n % ttn != 0:
        raise ValueError(f"Tile ({tile_m}, {tile_n}) must be divisible by thread tile ({ttm}, {ttn})")

    threads_y = tile_m // ttm
    threads_x = tile_n // ttn
    num_threads = threads_x * threads_y

    if num_threads > _MAX_THREADS_PER_BLOCK or num_threads < 32:
        raise ValueError(f"Thread count {num_threads} out of valid range [32, {_MAX_THREADS_PER_BLOCK}]")

    smem_x_elems = tile_m * tile_k
    smem_w_elems = tile_k * tile_n

    use_nf4_lut = mode.lower() == "nf4"

    if with_bias:

        @cute.kernel
        def qmm_tiled_kernel(
            x: cute.Tensor,
            w_q: cute.Tensor,
            scales: cute.Tensor,
            biases: cute.Tensor,
            out: cute.Tensor,
        ):
            bidx, bidy, _ = cute.arch.block_idx()
            tidx, tidy, _ = cute.arch.thread_idx()

            m_dim = cutlass.Int32(out.shape[0])
            n_dim = cutlass.Int32(out.shape[1])
            k_dim = cutlass.Int32(x.shape[1])
            block_m = cutlass.Int32(bidx) * cutlass.Int32(tile_m)
            block_n = cutlass.Int32(bidy) * cutlass.Int32(tile_n)

            sh_x_ptr = cute.arch.smem.alloc_smem(out_dtype, smem_x_elems)
            sh_x = cute.make_tensor(sh_x_ptr, cute.make_layout((tile_m, tile_k)))
            sh_w_ptr = cute.arch.smem.alloc_smem(out_dtype, smem_w_elems)
            sh_w = cute.make_tensor(sh_w_ptr, cute.make_layout((tile_k, tile_n)))
            nf4_lut_ptr = cute.arch.smem.alloc_smem(cutlass.Float32, 16)
            nf4_lut = cute.make_tensor(nf4_lut_ptr, cute.make_layout((16,)))

            flat_tid = cutlass.Int32(tidy) * cutlass.Int32(threads_x) + cutlass.Int32(tidx)
            num_thr = cutlass.Int32(num_threads)
            group_size_i = cutlass.Int32(group_size)

            if use_nf4_lut:
                if flat_tid < cutlass.Int32(16):
                    lut_val = cutlass.Float32(0.0)
                    for _li, _lv in enumerate(_NF4_LUT_VALUES):
                        lut_val = cutlass.select_(
                            flat_tid == cutlass.Int32(_li),
                            cutlass.Float32(_lv),
                            lut_val,
                        )
                    nf4_lut[(flat_tid,)] = lut_val
                cute.arch.sync_threads()

            a00 = cutlass.Float32(0.0)
            a01 = cutlass.Float32(0.0)
            a02 = cutlass.Float32(0.0)
            a03 = cutlass.Float32(0.0)
            a10 = cutlass.Float32(0.0)
            a11 = cutlass.Float32(0.0)
            a12 = cutlass.Float32(0.0)
            a13 = cutlass.Float32(0.0)
            a20 = cutlass.Float32(0.0)
            a21 = cutlass.Float32(0.0)
            a22 = cutlass.Float32(0.0)
            a23 = cutlass.Float32(0.0)
            a30 = cutlass.Float32(0.0)
            a31 = cutlass.Float32(0.0)
            a32 = cutlass.Float32(0.0)
            a33 = cutlass.Float32(0.0)

            k_start = cutlass.Int32(0)
            while k_start < k_dim:
                li = flat_tid
                while li < cutlass.Int32(smem_x_elems):
                    lr = li // cutlass.Int32(tile_k)
                    lc = li - lr * cutlass.Int32(tile_k)
                    gr = block_m + lr
                    gc = k_start + lc
                    gr_s = cutlass.select_(gr >= m_dim, cutlass.Int32(0), gr)
                    gc_s = cutlass.select_(gc >= k_dim, cutlass.Int32(0), gc)
                    xv = cutlass.Float32(x[(gr_s, gc_s)])
                    xv = cutlass.select_(gr >= m_dim, cutlass.Float32(0.0), xv)
                    xv = cutlass.select_(gc >= k_dim, cutlass.Float32(0.0), xv)
                    sh_x[(lr, lc)] = out_dtype(xv)
                    li = li + num_thr

                li = flat_tid
                while li < cutlass.Int32(smem_w_elems):
                    lk = li // cutlass.Int32(tile_n)
                    ln = li - lk * cutlass.Int32(tile_n)
                    gk = k_start + lk
                    gn = block_n + ln
                    wv = cutlass.Float32(0.0)
                    if gk < k_dim and gn < n_dim:
                        w_row = cutlass.Int32(0)
                        elem_idx = cutlass.Int32(0)
                        g_idx = cutlass.Int32(0)
                        if transpose:
                            w_row = gn
                            elem_idx = gk
                            g_idx = gk // group_size_i
                        else:
                            w_row = gk
                            elem_idx = gn
                            g_idx = gn // group_size_i
                        q_code = _unpack_bits(w_q, w_row, elem_idx, bits)
                        if use_nf4_lut:
                            sc = cutlass.Float32(scales[(w_row, g_idx)])
                            wv = cutlass.Float32(nf4_lut[(q_code,)]) * sc
                        else:
                            wv = dequant(w_row, g_idx, q_code, scales, biases)
                    sh_w[(lk, ln)] = out_dtype(wv)
                    li = li + num_thr

                cute.arch.sync_threads()

                bm = cutlass.Int32(tidy) * cutlass.Int32(ttm)
                bn = cutlass.Int32(tidx) * cutlass.Int32(ttn)
                for kk in cutlass.range(0, tile_k):
                    x0 = cutlass.Float32(sh_x[(bm, kk)])
                    x1 = cutlass.Float32(sh_x[(bm + cutlass.Int32(1), kk)])
                    x2 = cutlass.Float32(sh_x[(bm + cutlass.Int32(2), kk)])
                    x3 = cutlass.Float32(sh_x[(bm + cutlass.Int32(3), kk)])
                    w0 = cutlass.Float32(sh_w[(kk, bn)])
                    w1 = cutlass.Float32(sh_w[(kk, bn + cutlass.Int32(1))])
                    w2 = cutlass.Float32(sh_w[(kk, bn + cutlass.Int32(2))])
                    w3 = cutlass.Float32(sh_w[(kk, bn + cutlass.Int32(3))])
                    a00 += x0 * w0
                    a01 += x0 * w1
                    a02 += x0 * w2
                    a03 += x0 * w3
                    a10 += x1 * w0
                    a11 += x1 * w1
                    a12 += x1 * w2
                    a13 += x1 * w3
                    a20 += x2 * w0
                    a21 += x2 * w1
                    a22 += x2 * w2
                    a23 += x2 * w3
                    a30 += x3 * w0
                    a31 += x3 * w1
                    a32 += x3 * w2
                    a33 += x3 * w3

                cute.arch.sync_threads()
                k_start = k_start + cutlass.Int32(tile_k)

            bm_g = block_m + cutlass.Int32(tidy) * cutlass.Int32(ttm)
            bn_g = block_n + cutlass.Int32(tidx) * cutlass.Int32(ttn)
            if bm_g < m_dim and bn_g < n_dim:
                out[(bm_g, bn_g)] = out_dtype(a00)
            if bm_g < m_dim and bn_g + cutlass.Int32(1) < n_dim:
                out[(bm_g, bn_g + cutlass.Int32(1))] = out_dtype(a01)
            if bm_g < m_dim and bn_g + cutlass.Int32(2) < n_dim:
                out[(bm_g, bn_g + cutlass.Int32(2))] = out_dtype(a02)
            if bm_g < m_dim and bn_g + cutlass.Int32(3) < n_dim:
                out[(bm_g, bn_g + cutlass.Int32(3))] = out_dtype(a03)
            if bm_g + cutlass.Int32(1) < m_dim and bn_g < n_dim:
                out[(bm_g + cutlass.Int32(1), bn_g)] = out_dtype(a10)
            if bm_g + cutlass.Int32(1) < m_dim and bn_g + cutlass.Int32(1) < n_dim:
                out[(bm_g + cutlass.Int32(1), bn_g + cutlass.Int32(1))] = out_dtype(a11)
            if bm_g + cutlass.Int32(1) < m_dim and bn_g + cutlass.Int32(2) < n_dim:
                out[(bm_g + cutlass.Int32(1), bn_g + cutlass.Int32(2))] = out_dtype(a12)
            if bm_g + cutlass.Int32(1) < m_dim and bn_g + cutlass.Int32(3) < n_dim:
                out[(bm_g + cutlass.Int32(1), bn_g + cutlass.Int32(3))] = out_dtype(a13)
            if bm_g + cutlass.Int32(2) < m_dim and bn_g < n_dim:
                out[(bm_g + cutlass.Int32(2), bn_g)] = out_dtype(a20)
            if bm_g + cutlass.Int32(2) < m_dim and bn_g + cutlass.Int32(1) < n_dim:
                out[(bm_g + cutlass.Int32(2), bn_g + cutlass.Int32(1))] = out_dtype(a21)
            if bm_g + cutlass.Int32(2) < m_dim and bn_g + cutlass.Int32(2) < n_dim:
                out[(bm_g + cutlass.Int32(2), bn_g + cutlass.Int32(2))] = out_dtype(a22)
            if bm_g + cutlass.Int32(2) < m_dim and bn_g + cutlass.Int32(3) < n_dim:
                out[(bm_g + cutlass.Int32(2), bn_g + cutlass.Int32(3))] = out_dtype(a23)
            if bm_g + cutlass.Int32(3) < m_dim and bn_g < n_dim:
                out[(bm_g + cutlass.Int32(3), bn_g)] = out_dtype(a30)
            if bm_g + cutlass.Int32(3) < m_dim and bn_g + cutlass.Int32(1) < n_dim:
                out[(bm_g + cutlass.Int32(3), bn_g + cutlass.Int32(1))] = out_dtype(a31)
            if bm_g + cutlass.Int32(3) < m_dim and bn_g + cutlass.Int32(2) < n_dim:
                out[(bm_g + cutlass.Int32(3), bn_g + cutlass.Int32(2))] = out_dtype(a32)
            if bm_g + cutlass.Int32(3) < m_dim and bn_g + cutlass.Int32(3) < n_dim:
                out[(bm_g + cutlass.Int32(3), bn_g + cutlass.Int32(3))] = out_dtype(a33)

        @cute.jit
        def qmm_host_runtime(
            x: cute.Tensor,
            w_q: cute.Tensor,
            scales: cute.Tensor,
            biases: cute.Tensor,
            out: cute.Tensor,
        ):
            grid_x = (out.shape[0] + tile_m - 1) // tile_m
            grid_y = (out.shape[1] + tile_n - 1) // tile_n
            qmm_tiled_kernel(x, w_q, scales, biases, out).launch(
                grid=(grid_x, grid_y, 1),
                block=(threads_x, threads_y, 1),
            )

        @cute.jit
        def qmm_host_jax(
            stream: cuda.CUstream,
            x: cute.Tensor,
            w_q: cute.Tensor,
            scales: cute.Tensor,
            biases: cute.Tensor,
            out: cute.Tensor,
        ):
            grid_x = (out.shape[0] + tile_m - 1) // tile_m
            grid_y = (out.shape[1] + tile_n - 1) // tile_n
            qmm_tiled_kernel(x, w_q, scales, biases, out).launch(
                grid=(grid_x, grid_y, 1),
                block=(threads_x, threads_y, 1),
                stream=stream,
            )

        return qmm_host_runtime, qmm_host_jax

    @cute.kernel
    def qmm_tiled_kernel(
        x: cute.Tensor,
        w_q: cute.Tensor,
        scales: cute.Tensor,
        out: cute.Tensor,
    ):
        bidx, bidy, _ = cute.arch.block_idx()
        tidx, tidy, _ = cute.arch.thread_idx()

        m_dim = cutlass.Int32(out.shape[0])
        n_dim = cutlass.Int32(out.shape[1])
        k_dim = cutlass.Int32(x.shape[1])
        block_m = cutlass.Int32(bidx) * cutlass.Int32(tile_m)
        block_n = cutlass.Int32(bidy) * cutlass.Int32(tile_n)

        sh_x_ptr = cute.arch.smem.alloc_smem(out_dtype, smem_x_elems)
        sh_x = cute.make_tensor(sh_x_ptr, cute.make_layout((tile_m, tile_k)))
        sh_w_ptr = cute.arch.smem.alloc_smem(out_dtype, smem_w_elems)
        sh_w = cute.make_tensor(sh_w_ptr, cute.make_layout((tile_k, tile_n)))
        nf4_lut_ptr = cute.arch.smem.alloc_smem(cutlass.Float32, 16)
        nf4_lut = cute.make_tensor(nf4_lut_ptr, cute.make_layout((16,)))

        flat_tid = cutlass.Int32(tidy) * cutlass.Int32(threads_x) + cutlass.Int32(tidx)
        num_thr = cutlass.Int32(num_threads)
        group_size_i = cutlass.Int32(group_size)

        if use_nf4_lut:
            if flat_tid < cutlass.Int32(16):
                lut_val = cutlass.Float32(0.0)
                for _li, _lv in enumerate(_NF4_LUT_VALUES):
                    lut_val = cutlass.select_(
                        flat_tid == cutlass.Int32(_li),
                        cutlass.Float32(_lv),
                        lut_val,
                    )
                nf4_lut[(flat_tid,)] = lut_val
            cute.arch.sync_threads()

        a00 = cutlass.Float32(0.0)
        a01 = cutlass.Float32(0.0)
        a02 = cutlass.Float32(0.0)
        a03 = cutlass.Float32(0.0)
        a10 = cutlass.Float32(0.0)
        a11 = cutlass.Float32(0.0)
        a12 = cutlass.Float32(0.0)
        a13 = cutlass.Float32(0.0)
        a20 = cutlass.Float32(0.0)
        a21 = cutlass.Float32(0.0)
        a22 = cutlass.Float32(0.0)
        a23 = cutlass.Float32(0.0)
        a30 = cutlass.Float32(0.0)
        a31 = cutlass.Float32(0.0)
        a32 = cutlass.Float32(0.0)
        a33 = cutlass.Float32(0.0)

        k_start = cutlass.Int32(0)
        while k_start < k_dim:
            li = flat_tid
            while li < cutlass.Int32(smem_x_elems):
                lr = li // cutlass.Int32(tile_k)
                lc = li - lr * cutlass.Int32(tile_k)
                gr = block_m + lr
                gc = k_start + lc
                gr_s = cutlass.select_(gr >= m_dim, cutlass.Int32(0), gr)
                gc_s = cutlass.select_(gc >= k_dim, cutlass.Int32(0), gc)
                xv = cutlass.Float32(x[(gr_s, gc_s)])
                xv = cutlass.select_(gr >= m_dim, cutlass.Float32(0.0), xv)
                xv = cutlass.select_(gc >= k_dim, cutlass.Float32(0.0), xv)
                sh_x[(lr, lc)] = out_dtype(xv)
                li = li + num_thr

            li = flat_tid
            while li < cutlass.Int32(smem_w_elems):
                lk = li // cutlass.Int32(tile_n)
                ln = li - lk * cutlass.Int32(tile_n)
                gk = k_start + lk
                gn = block_n + ln
                wv = cutlass.Float32(0.0)
                if gk < k_dim and gn < n_dim:
                    w_row = cutlass.Int32(0)
                    elem_idx = cutlass.Int32(0)
                    g_idx = cutlass.Int32(0)
                    if transpose:
                        w_row = gn
                        elem_idx = gk
                        g_idx = gk // group_size_i
                    else:
                        w_row = gk
                        elem_idx = gn
                        g_idx = gn // group_size_i
                    q_code = _unpack_bits(w_q, w_row, elem_idx, bits)
                    if use_nf4_lut:
                        sc = cutlass.Float32(scales[(w_row, g_idx)])
                        wv = cutlass.Float32(nf4_lut[(q_code,)]) * sc
                    else:
                        wv = dequant(w_row, g_idx, q_code, scales)
                sh_w[(lk, ln)] = out_dtype(wv)
                li = li + num_thr

            cute.arch.sync_threads()

            bm = cutlass.Int32(tidy) * cutlass.Int32(ttm)
            bn = cutlass.Int32(tidx) * cutlass.Int32(ttn)
            for kk in cutlass.range(0, tile_k):
                x0 = cutlass.Float32(sh_x[(bm, kk)])
                x1 = cutlass.Float32(sh_x[(bm + cutlass.Int32(1), kk)])
                x2 = cutlass.Float32(sh_x[(bm + cutlass.Int32(2), kk)])
                x3 = cutlass.Float32(sh_x[(bm + cutlass.Int32(3), kk)])
                w0 = cutlass.Float32(sh_w[(kk, bn)])
                w1 = cutlass.Float32(sh_w[(kk, bn + cutlass.Int32(1))])
                w2 = cutlass.Float32(sh_w[(kk, bn + cutlass.Int32(2))])
                w3 = cutlass.Float32(sh_w[(kk, bn + cutlass.Int32(3))])
                a00 += x0 * w0
                a01 += x0 * w1
                a02 += x0 * w2
                a03 += x0 * w3
                a10 += x1 * w0
                a11 += x1 * w1
                a12 += x1 * w2
                a13 += x1 * w3
                a20 += x2 * w0
                a21 += x2 * w1
                a22 += x2 * w2
                a23 += x2 * w3
                a30 += x3 * w0
                a31 += x3 * w1
                a32 += x3 * w2
                a33 += x3 * w3

            cute.arch.sync_threads()
            k_start = k_start + cutlass.Int32(tile_k)

        bm_g = block_m + cutlass.Int32(tidy) * cutlass.Int32(ttm)
        bn_g = block_n + cutlass.Int32(tidx) * cutlass.Int32(ttn)
        if bm_g < m_dim and bn_g < n_dim:
            out[(bm_g, bn_g)] = out_dtype(a00)
        if bm_g < m_dim and bn_g + cutlass.Int32(1) < n_dim:
            out[(bm_g, bn_g + cutlass.Int32(1))] = out_dtype(a01)
        if bm_g < m_dim and bn_g + cutlass.Int32(2) < n_dim:
            out[(bm_g, bn_g + cutlass.Int32(2))] = out_dtype(a02)
        if bm_g < m_dim and bn_g + cutlass.Int32(3) < n_dim:
            out[(bm_g, bn_g + cutlass.Int32(3))] = out_dtype(a03)
        if bm_g + cutlass.Int32(1) < m_dim and bn_g < n_dim:
            out[(bm_g + cutlass.Int32(1), bn_g)] = out_dtype(a10)
        if bm_g + cutlass.Int32(1) < m_dim and bn_g + cutlass.Int32(1) < n_dim:
            out[(bm_g + cutlass.Int32(1), bn_g + cutlass.Int32(1))] = out_dtype(a11)
        if bm_g + cutlass.Int32(1) < m_dim and bn_g + cutlass.Int32(2) < n_dim:
            out[(bm_g + cutlass.Int32(1), bn_g + cutlass.Int32(2))] = out_dtype(a12)
        if bm_g + cutlass.Int32(1) < m_dim and bn_g + cutlass.Int32(3) < n_dim:
            out[(bm_g + cutlass.Int32(1), bn_g + cutlass.Int32(3))] = out_dtype(a13)
        if bm_g + cutlass.Int32(2) < m_dim and bn_g < n_dim:
            out[(bm_g + cutlass.Int32(2), bn_g)] = out_dtype(a20)
        if bm_g + cutlass.Int32(2) < m_dim and bn_g + cutlass.Int32(1) < n_dim:
            out[(bm_g + cutlass.Int32(2), bn_g + cutlass.Int32(1))] = out_dtype(a21)
        if bm_g + cutlass.Int32(2) < m_dim and bn_g + cutlass.Int32(2) < n_dim:
            out[(bm_g + cutlass.Int32(2), bn_g + cutlass.Int32(2))] = out_dtype(a22)
        if bm_g + cutlass.Int32(2) < m_dim and bn_g + cutlass.Int32(3) < n_dim:
            out[(bm_g + cutlass.Int32(2), bn_g + cutlass.Int32(3))] = out_dtype(a23)
        if bm_g + cutlass.Int32(3) < m_dim and bn_g < n_dim:
            out[(bm_g + cutlass.Int32(3), bn_g)] = out_dtype(a30)
        if bm_g + cutlass.Int32(3) < m_dim and bn_g + cutlass.Int32(1) < n_dim:
            out[(bm_g + cutlass.Int32(3), bn_g + cutlass.Int32(1))] = out_dtype(a31)
        if bm_g + cutlass.Int32(3) < m_dim and bn_g + cutlass.Int32(2) < n_dim:
            out[(bm_g + cutlass.Int32(3), bn_g + cutlass.Int32(2))] = out_dtype(a32)
        if bm_g + cutlass.Int32(3) < m_dim and bn_g + cutlass.Int32(3) < n_dim:
            out[(bm_g + cutlass.Int32(3), bn_g + cutlass.Int32(3))] = out_dtype(a33)

    @cute.jit
    def qmm_host_runtime(
        x: cute.Tensor,
        w_q: cute.Tensor,
        scales: cute.Tensor,
        out: cute.Tensor,
    ):
        grid_x = (out.shape[0] + tile_m - 1) // tile_m
        grid_y = (out.shape[1] + tile_n - 1) // tile_n
        qmm_tiled_kernel(x, w_q, scales, out).launch(
            grid=(grid_x, grid_y, 1),
            block=(threads_x, threads_y, 1),
        )

    @cute.jit
    def qmm_host_jax(
        stream: cuda.CUstream,
        x: cute.Tensor,
        w_q: cute.Tensor,
        scales: cute.Tensor,
        out: cute.Tensor,
    ):
        grid_x = (out.shape[0] + tile_m - 1) // tile_m
        grid_y = (out.shape[1] + tile_n - 1) // tile_n
        qmm_tiled_kernel(x, w_q, scales, out).launch(
            grid=(grid_x, grid_y, 1),
            block=(threads_x, threads_y, 1),
            stream=stream,
        )

    return qmm_host_runtime, qmm_host_jax


def _make_mma_smem_layout(
    tile_row: int,
    tile_col: int,
    dtype: type[cutlass.Numeric],
    row_major: bool = True,
    copy_bits: int = 128,
):
    """Create a bank-conflict-free swizzled SMEM layout for MMA A/B tiles.

    Follows the CUTLASS ``tensorop_gemm.py`` pattern: computes the number of
    swizzle bits from the major dimension and element width, caps at 3 bits
    (8-way XOR swizzle), then uses :func:`cute.make_composed_layout` to compose
    the swizzle with the base row/col-major atom, tiled to the full tile shape.

    Args:
        tile_row: Number of rows in the SMEM tile.
        tile_col: Number of columns in the SMEM tile.
        dtype: CUTLASS element dtype (used to compute swizzle bits from element width).
        row_major: If ``True``, the inner atom is row-major (stride=(major, 1)).
            If ``False``, the inner atom is column-major (stride=(1, major)).
        copy_bits: Bit width of the load atom (default 128 for cp.async).

    Returns:
        A CuTe ``ComposedLayout`` of shape ``(tile_row, tile_col)`` with
        XOR swizzling to avoid shared-memory bank conflicts.
    """
    major_size = tile_col if row_major else tile_row
    major_size = 64 if major_size >= 64 else major_size
    swizzle_bits = int(math.log2(major_size * dtype.width // copy_bits))
    swizzle_bits = min(swizzle_bits, 3)

    if row_major:
        atom_outer = cute.make_layout((8, major_size), stride=(major_size, 1))
    else:
        atom_outer = cute.make_layout((major_size, 8), stride=(1, major_size))

    atom = cute.make_composed_layout(cute.make_swizzle(swizzle_bits, 3, 3), 0, atom_outer)
    return cute.tile_to_shape(atom, (tile_row, tile_col), (0, 1))


def _make_mma_smem_layout_c(
    tile_m: int,
    tile_n: int,
    dtype: type[cutlass.Numeric],
    row_major: bool = True,
    copy_bits: int = 128,
):
    """Create a simple row/col-major SMEM layout for MMA epilogue C tile.

    Note: We use a non-swizzled layout here because ``cute.autovec_copy``
    calls ``right_inverse`` which does not support ``ComposedLayout`` (the
    type returned by ``make_composed_layout``).  The epilogue writeback is
    not performance-critical so the lack of swizzle is acceptable.
    """
    if row_major:
        return cute.make_layout((tile_m, tile_n), stride=(tile_n, 1))
    return cute.make_layout((tile_m, tile_n), stride=(1, tile_m))


_MMA_ATOM_M = 16
_MMA_ATOM_N = 8
_MMA_ATOM_K = 16
_MMA_ATOM_LAYOUT_M = 2
_MMA_ATOM_LAYOUT_N = 2
_MMA_ATOM_LAYOUT_K = 1
_MMA_NUM_THREADS = _MMA_ATOM_LAYOUT_M * _MMA_ATOM_LAYOUT_N * _MMA_ATOM_LAYOUT_K * 32
_MMA_TILED_M = _MMA_ATOM_LAYOUT_M * _MMA_ATOM_M
_MMA_TILED_N = _MMA_ATOM_LAYOUT_N * _MMA_ATOM_N * 2
_MMA_TILED_K = _MMA_ATOM_LAYOUT_K * _MMA_ATOM_K


def _validate_mma_tile(tile_m: int, tile_n: int, tile_k: int) -> bool:
    """Return ``True`` when tile dimensions are compatible with the MMA atom configuration.

    Requirements (derived from the ``MmaF16BF16Op`` atom and tiling layout):
        - ``tile_m >= _MMA_TILED_M`` (32) and divisible by it.
        - ``tile_n >= _MMA_TILED_N`` (32) and divisible by it.
        - ``tile_k`` divisible by ``_MMA_ATOM_K`` (16).
    """
    if tile_m < _MMA_TILED_M or tile_n < _MMA_TILED_N:
        return False
    if tile_m % _MMA_TILED_M != 0:
        return False
    if tile_n % _MMA_TILED_N != 0:
        return False
    if tile_k % _MMA_ATOM_K != 0:
        return False
    return True


def _use_tiled_scalar_kernel() -> bool:
    """Return ``True`` when ``EJKERNEL_CUTE_QMM_USE_TILED=1`` is set in the environment."""
    raw = os.getenv("EJKERNEL_CUTE_QMM_USE_TILED", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def _use_mma_single_stage() -> bool:
    """Return ``True`` when ``EJKERNEL_CUTE_QMM_USE_MMA_SINGLE=1`` is set in the environment."""
    raw = os.getenv("EJKERNEL_CUTE_QMM_USE_MMA_SINGLE", "").strip().lower()
    return raw in ("1", "true", "yes", "on")


_MMA_PIPELINE_STAGES = 3


def _make_staged_smem_layout(
    tile_row: int,
    tile_col: int,
    num_stages: int,
    dtype: type[cutlass.Numeric],
    row_major: bool = True,
    copy_bits: int = 128,
):
    """Extend a 2-D swizzled SMEM layout to 3-D ``(row, col, stages)`` for software pipelining.

    The 2-D base is built by :func:`_make_mma_smem_layout` and then tiled
    to include a pipeline-stage dimension, producing a
    ``ComposedLayout`` of shape ``(tile_row, tile_col, num_stages)``.

    Args:
        tile_row: Tile row count (M or N dimension).
        tile_col: Tile column count (K dimension).
        num_stages: Number of pipeline stages (SMEM circular buffer depth).
        dtype: CUTLASS element dtype for swizzle calculation.
        row_major: Whether the inner layout is row-major.
        copy_bits: Bit width of the cp.async copy atom (default 128).

    Returns:
        A 3-D CuTe ``ComposedLayout`` suitable for staged cp.async operations.
    """
    atom_2d = _make_mma_smem_layout(tile_row, tile_col, dtype, row_major, copy_bits)
    return cute.tile_to_shape(atom_2d, (tile_row, tile_col, num_stages), (0, 1, 2))


def _make_gmem_tiled_copy_x(
    out_dtype: type[cutlass.Numeric],
    tile_k: int,
    num_threads: int,
    copy_bits: int = 128,
):
    """Create a ``CopyG2SOp`` tiled copy for async GMEM-to-SMEM of *x* activations.

    Follows the ``_make_gmem_tiled_copy_AB`` pattern from the CUTLASS
    Ampere ``tensorop_gemm.py`` example.  *x* is row-major ``(M, K)``
    so the leading (contiguous) dimension is K.

    Uses 128-bit cp.async copies for maximum throughput.  The MLIR
    verifier requires matching alignment on the GMEM source; this is
    satisfied because ``_fake_tensor_from_shaped`` in the FFI layer
    sets ``assumed_align=16`` (16 bytes = 128 bits), and JAX/XLA
    guarantees at least 128-byte alignment for device buffers.

    Args:
        out_dtype: Element type for the SMEM tile.
        tile_k: K-dimension tile size.
        num_threads: Number of threads per thread block.
        copy_bits: Number of bits per cp.async copy (default 128).

    Returns:
        A ``TiledCopy`` object configured for async global-to-shared copies.
    """
    atom = cute.make_copy_atom(
        cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.cpasync.LoadCacheMode.GLOBAL),
        out_dtype,
        num_bits_per_copy=copy_bits,
    )
    copy_elems = copy_bits // out_dtype.width
    k_groups = tile_k // copy_elems
    thread_layout = cute.make_layout(
        (num_threads // k_groups, k_groups),
        stride=(k_groups, 1),
    )
    value_layout = cute.make_layout((1, copy_elems))
    return cute.make_tiled_copy_tv(atom, thread_layout, value_layout)


def _build_mma_qmm_host_fns(
    *,
    mode: str,
    bits: int,
    group_size: int,
    out_dtype: type[cutlass.Numeric],
    with_bias: bool,
    transpose: bool,
    tile_m: int,
    tile_n: int,
    tile_k: int,
):
    """Build MMA tensor-core fused dequant+matmul CuTe host launchers (SM80+).

    Creates a CuTe DSL kernel that leverages Ampere (SM80+) warp-level
    MMA instructions (``MmaF16BF16Op``) for the inner GEMM.  Input
    activations are cooperatively loaded into swizzled shared memory via
    scalar stores, quantized weights are dequantized on-the-fly and stored
    into shared memory in ``(N, K)`` layout, then SMEM-to-register copies
    use ``LdMatrix`` operations for bank-conflict-free loads.  The
    accumulator is written back through a shared-memory epilogue.

    This is a single-stage (non-pipelined) kernel variant; for overlapped
    data movement use ``_build_mma_pipelined_qmm_host_fns`` instead.

    Parallelization: 2-D grid ``(ceil(M/tile_m), ceil(N/tile_n))`` with
    128 threads (4 warps) per block.

    Args:
        mode: Quantization mode.
        bits: Bit-width of quantized weights.
        group_size: Quantization group size.
        out_dtype: CUTLASS output dtype.
        with_bias: Whether per-group bias is used.
        transpose: Whether the weight matrix is transposed.
        tile_m: Tile height (must be a multiple of 32).
        tile_n: Tile width (must be a multiple of 32).
        tile_k: K-dimension tile size (must be a multiple of 16).

    Returns:
        Tuple of ``(runtime_launcher, jax_stream_launcher)`` callables.

    Raises:
        ValueError: If tile dimensions are incompatible with the MMA atom
            configuration.
    """
    if not _validate_mma_tile(tile_m, tile_n, tile_k):
        raise ValueError(
            f"Tile ({tile_m}, {tile_n}, {tile_k}) incompatible with MMA config "
            f"(need multiples of {_MMA_TILED_M}, {_MMA_TILED_N}, {_MMA_ATOM_K})"
        )

    dequant = _build_dequant_fn(mode=mode, with_bias=with_bias)
    num_threads = _MMA_NUM_THREADS
    smem_x_elems = tile_m * tile_k
    smem_w_elems = tile_n * tile_k
    smem_c_elems = tile_m * tile_n
    use_nf4_lut = mode.lower() == "nf4"

    if with_bias:

        @cute.kernel
        def qmm_mma_kernel(
            x: cute.Tensor,
            w_q: cute.Tensor,
            scales: cute.Tensor,
            biases: cute.Tensor,
            out: cute.Tensor,
            sA_layout: cute.Layout,
            sB_layout: cute.Layout,
            sC_layout: cute.Layout,
            tiled_mma: cute.TiledMma,
            tiled_copy_s2r_A: cute.TiledCopy,
            tiled_copy_s2r_B: cute.TiledCopy,
        ):
            bidx, bidy, _ = cute.arch.block_idx()
            tidx, _, _ = cute.arch.thread_idx()

            m_dim = cutlass.Int32(out.shape[0])
            n_dim = cutlass.Int32(out.shape[1])
            k_dim = cutlass.Int32(x.shape[1])
            block_m = cutlass.Int32(bidx) * cutlass.Int32(tile_m)
            block_n = cutlass.Int32(bidy) * cutlass.Int32(tile_n)
            flat_tid = cutlass.Int32(tidx)
            num_thr = cutlass.Int32(num_threads)
            group_size_i = cutlass.Int32(group_size)

            smem = cutlass.utils.SmemAllocator()
            sA = smem.allocate_tensor(out_dtype, sA_layout, 16)
            sB = smem.allocate_tensor(out_dtype, sB_layout, 16)

            nf4_lut_ptr = cute.arch.smem.alloc_smem(cutlass.Float32, 16)
            nf4_lut = cute.make_tensor(nf4_lut_ptr, cute.make_layout((16,)))
            if use_nf4_lut:
                if flat_tid < cutlass.Int32(16):
                    lut_val = cutlass.Float32(0.0)
                    for _li, _lv in enumerate(_NF4_LUT_VALUES):
                        lut_val = cutlass.select_(
                            flat_tid == cutlass.Int32(_li),
                            cutlass.Float32(_lv),
                            lut_val,
                        )
                    nf4_lut[(flat_tid,)] = lut_val
                cute.arch.sync_threads()

            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)

            tCrA = tiled_mma.make_fragment_A(tCsA)
            tCrB = tiled_mma.make_fragment_B(tCsB)

            gC = cute.local_tile(out, tiler=(tile_m, tile_n), coord=(bidx, bidy))
            tCgC = thr_mma.partition_C(gC)
            tCrC = tiled_mma.make_fragment_C(tCgC)
            tCrC.fill(0.0)

            thr_s2r_A = tiled_copy_s2r_A.get_slice(tidx)
            tCsA_s2r = thr_s2r_A.partition_S(sA)
            tCrA_s2r = thr_s2r_A.retile(tCrA)
            thr_s2r_B = tiled_copy_s2r_B.get_slice(tidx)
            tCsB_s2r = thr_s2r_B.partition_S(sB)
            tCrB_s2r = thr_s2r_B.retile(tCrB)

            num_k_block = cute.size(tCrA, mode=[2])

            k_start = cutlass.Int32(0)
            while k_start < k_dim:
                li = flat_tid
                while li < cutlass.Int32(smem_x_elems):
                    lr = li // cutlass.Int32(tile_k)
                    lc = li - lr * cutlass.Int32(tile_k)
                    gr = block_m + lr
                    gc = k_start + lc
                    gr_s = cutlass.select_(gr >= m_dim, cutlass.Int32(0), gr)
                    gc_s = cutlass.select_(gc >= k_dim, cutlass.Int32(0), gc)
                    xv = cutlass.Float32(x[(gr_s, gc_s)])
                    xv = cutlass.select_(gr >= m_dim, cutlass.Float32(0.0), xv)
                    xv = cutlass.select_(gc >= k_dim, cutlass.Float32(0.0), xv)
                    sA[(lr, lc)] = out_dtype(xv)
                    li = li + num_thr

                li = flat_tid
                while li < cutlass.Int32(smem_w_elems):
                    ln = li // cutlass.Int32(tile_k)
                    lk = li - ln * cutlass.Int32(tile_k)
                    gk = k_start + lk
                    gn = block_n + ln
                    wv = cutlass.Float32(0.0)
                    if gk < k_dim and gn < n_dim:
                        w_row = cutlass.Int32(0)
                        elem_idx = cutlass.Int32(0)
                        g_idx = cutlass.Int32(0)
                        if transpose:
                            w_row = gn
                            elem_idx = gk
                            g_idx = gk // group_size_i
                        else:
                            w_row = gk
                            elem_idx = gn
                            g_idx = gn // group_size_i
                        q_code = _unpack_bits(w_q, w_row, elem_idx, bits)
                        if use_nf4_lut:
                            sc = cutlass.Float32(scales[(w_row, g_idx)])
                            wv = cutlass.Float32(nf4_lut[(q_code,)]) * sc
                        else:
                            wv = dequant(w_row, g_idx, q_code, scales, biases)
                    sB[(ln, lk)] = out_dtype(wv)
                    li = li + num_thr

                cute.arch.sync_threads()

                for kb in cutlass.range(num_k_block, unroll_full=True):
                    cute.copy(tiled_copy_s2r_A, tCsA_s2r[None, None, kb], tCrA_s2r[None, None, kb])
                    cute.copy(tiled_copy_s2r_B, tCsB_s2r[None, None, kb], tCrB_s2r[None, None, kb])
                for kb in cutlass.range(num_k_block, unroll_full=True):
                    cute.gemm(tiled_mma, tCrC, tCrA[None, None, kb], tCrB[None, None, kb], tCrC)

                cute.arch.sync_threads()
                k_start = k_start + cutlass.Int32(tile_k)

            tCrD = cute.make_fragment_like(tCrC, out_dtype)
            tCrD[None] = tCrC.load().to(out_dtype)

            sC = cute.make_tensor(cute.recast_ptr(sA.iterator, dtype=out_dtype), sC_layout)
            tCsC = thr_mma.partition_C(sC)
            cute.autovec_copy(tCrD, tCsC)
            cute.arch.sync_threads()

            li = flat_tid
            while li < cutlass.Int32(smem_c_elems):
                lr = li // cutlass.Int32(tile_n)
                lc = li - lr * cutlass.Int32(tile_n)
                gr = block_m + lr
                gc = block_n + lc
                if gr < m_dim and gc < n_dim:
                    out[(gr, gc)] = sC[(lr, lc)]
                li = li + num_thr

        @cute.jit
        def qmm_host_runtime(
            x: cute.Tensor,
            w_q: cute.Tensor,
            scales: cute.Tensor,
            biases: cute.Tensor,
            out: cute.Tensor,
        ):
            op = cute.nvgpu.warp.MmaF16BF16Op(out_dtype, cutlass.Float32, (_MMA_ATOM_M, _MMA_ATOM_N, _MMA_ATOM_K))
            atom_layout = cute.make_layout((_MMA_ATOM_LAYOUT_M, _MMA_ATOM_LAYOUT_N, _MMA_ATOM_LAYOUT_K))
            permutation_mnk = (_MMA_TILED_M, _MMA_TILED_N, _MMA_TILED_K)
            tiled_mma = cute.make_tiled_mma(op, atom_layout, permutation_mnk=permutation_mnk)

            sA_layout = _make_mma_smem_layout(tile_m, tile_k, out_dtype, row_major=True)
            sB_layout = _make_mma_smem_layout(tile_n, tile_k, out_dtype, row_major=True)
            sC_layout = _make_mma_smem_layout_c(tile_m, tile_n, out_dtype, row_major=True)

            atom_s2r_A = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), out_dtype)
            atom_s2r_B = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), out_dtype)
            tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
            tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)

            smem_bytes = (
                max(
                    cute.size_in_bytes(out_dtype, sC_layout),
                    cute.size_in_bytes(out_dtype, sA_layout) + cute.size_in_bytes(out_dtype, sB_layout),
                )
                + 128
            )

            grid_x = (out.shape[0] + tile_m - 1) // tile_m
            grid_y = (out.shape[1] + tile_n - 1) // tile_n
            qmm_mma_kernel(
                x,
                w_q,
                scales,
                biases,
                out,
                sA_layout,
                sB_layout,
                sC_layout,
                tiled_mma,
                tiled_copy_s2r_A,
                tiled_copy_s2r_B,
            ).launch(
                grid=(grid_x, grid_y, 1),
                block=(num_threads, 1, 1),
                smem=smem_bytes,
            )

        @cute.jit
        def qmm_host_jax(
            stream: cuda.CUstream,
            x: cute.Tensor,
            w_q: cute.Tensor,
            scales: cute.Tensor,
            biases: cute.Tensor,
            out: cute.Tensor,
        ):
            op = cute.nvgpu.warp.MmaF16BF16Op(out_dtype, cutlass.Float32, (_MMA_ATOM_M, _MMA_ATOM_N, _MMA_ATOM_K))
            atom_layout = cute.make_layout((_MMA_ATOM_LAYOUT_M, _MMA_ATOM_LAYOUT_N, _MMA_ATOM_LAYOUT_K))
            permutation_mnk = (_MMA_TILED_M, _MMA_TILED_N, _MMA_TILED_K)
            tiled_mma = cute.make_tiled_mma(op, atom_layout, permutation_mnk=permutation_mnk)

            sA_layout = _make_mma_smem_layout(tile_m, tile_k, out_dtype, row_major=True)
            sB_layout = _make_mma_smem_layout(tile_n, tile_k, out_dtype, row_major=True)
            sC_layout = _make_mma_smem_layout_c(tile_m, tile_n, out_dtype, row_major=True)

            atom_s2r_A = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), out_dtype)
            atom_s2r_B = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), out_dtype)
            tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
            tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)

            smem_bytes = (
                max(
                    cute.size_in_bytes(out_dtype, sC_layout),
                    cute.size_in_bytes(out_dtype, sA_layout) + cute.size_in_bytes(out_dtype, sB_layout),
                )
                + 128
            )

            grid_x = (out.shape[0] + tile_m - 1) // tile_m
            grid_y = (out.shape[1] + tile_n - 1) // tile_n
            qmm_mma_kernel(
                x,
                w_q,
                scales,
                biases,
                out,
                sA_layout,
                sB_layout,
                sC_layout,
                tiled_mma,
                tiled_copy_s2r_A,
                tiled_copy_s2r_B,
            ).launch(
                grid=(grid_x, grid_y, 1),
                block=(num_threads, 1, 1),
                smem=smem_bytes,
                stream=stream,
            )

        return qmm_host_runtime, qmm_host_jax

    @cute.kernel
    def qmm_mma_kernel(
        x: cute.Tensor,
        w_q: cute.Tensor,
        scales: cute.Tensor,
        out: cute.Tensor,
        sA_layout: cute.Layout,
        sB_layout: cute.Layout,
        sC_layout: cute.Layout,
        tiled_mma: cute.TiledMma,
        tiled_copy_s2r_A: cute.TiledCopy,
        tiled_copy_s2r_B: cute.TiledCopy,
    ):
        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        m_dim = cutlass.Int32(out.shape[0])
        n_dim = cutlass.Int32(out.shape[1])
        k_dim = cutlass.Int32(x.shape[1])
        block_m = cutlass.Int32(bidx) * cutlass.Int32(tile_m)
        block_n = cutlass.Int32(bidy) * cutlass.Int32(tile_n)
        flat_tid = cutlass.Int32(tidx)
        num_thr = cutlass.Int32(num_threads)
        group_size_i = cutlass.Int32(group_size)

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(out_dtype, sA_layout, 16)
        sB = smem.allocate_tensor(out_dtype, sB_layout, 16)

        nf4_lut_ptr = cute.arch.smem.alloc_smem(cutlass.Float32, 16)
        nf4_lut = cute.make_tensor(nf4_lut_ptr, cute.make_layout((16,)))
        if use_nf4_lut:
            if flat_tid < cutlass.Int32(16):
                lut_val = cutlass.Float32(0.0)
                for _li, _lv in enumerate(_NF4_LUT_VALUES):
                    lut_val = cutlass.select_(
                        flat_tid == cutlass.Int32(_li),
                        cutlass.Float32(_lv),
                        lut_val,
                    )
                nf4_lut[(flat_tid,)] = lut_val
            cute.arch.sync_threads()

        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)

        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        gC = cute.local_tile(out, tiler=(tile_m, tile_n), coord=(bidx, bidy))
        tCgC = thr_mma.partition_C(gC)
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)

        thr_s2r_A = tiled_copy_s2r_A.get_slice(tidx)
        tCsA_s2r = thr_s2r_A.partition_S(sA)
        tCrA_s2r = thr_s2r_A.retile(tCrA)
        thr_s2r_B = tiled_copy_s2r_B.get_slice(tidx)
        tCsB_s2r = thr_s2r_B.partition_S(sB)
        tCrB_s2r = thr_s2r_B.retile(tCrB)

        num_k_block = cute.size(tCrA, mode=[2])

        k_start = cutlass.Int32(0)
        while k_start < k_dim:
            li = flat_tid
            while li < cutlass.Int32(smem_x_elems):
                lr = li // cutlass.Int32(tile_k)
                lc = li - lr * cutlass.Int32(tile_k)
                gr = block_m + lr
                gc = k_start + lc
                gr_s = cutlass.select_(gr >= m_dim, cutlass.Int32(0), gr)
                gc_s = cutlass.select_(gc >= k_dim, cutlass.Int32(0), gc)
                xv = cutlass.Float32(x[(gr_s, gc_s)])
                xv = cutlass.select_(gr >= m_dim, cutlass.Float32(0.0), xv)
                xv = cutlass.select_(gc >= k_dim, cutlass.Float32(0.0), xv)
                sA[(lr, lc)] = out_dtype(xv)
                li = li + num_thr

            li = flat_tid
            while li < cutlass.Int32(smem_w_elems):
                ln = li // cutlass.Int32(tile_k)
                lk = li - ln * cutlass.Int32(tile_k)
                gk = k_start + lk
                gn = block_n + ln
                wv = cutlass.Float32(0.0)
                if gk < k_dim and gn < n_dim:
                    w_row = cutlass.Int32(0)
                    elem_idx = cutlass.Int32(0)
                    g_idx = cutlass.Int32(0)
                    if transpose:
                        w_row = gn
                        elem_idx = gk
                        g_idx = gk // group_size_i
                    else:
                        w_row = gk
                        elem_idx = gn
                        g_idx = gn // group_size_i
                    q_code = _unpack_bits(w_q, w_row, elem_idx, bits)
                    if use_nf4_lut:
                        sc = cutlass.Float32(scales[(w_row, g_idx)])
                        wv = cutlass.Float32(nf4_lut[(q_code,)]) * sc
                    else:
                        wv = dequant(w_row, g_idx, q_code, scales)
                sB[(ln, lk)] = out_dtype(wv)
                li = li + num_thr

            cute.arch.sync_threads()

            for kb in cutlass.range(num_k_block, unroll_full=True):
                cute.copy(tiled_copy_s2r_A, tCsA_s2r[None, None, kb], tCrA_s2r[None, None, kb])
                cute.copy(tiled_copy_s2r_B, tCsB_s2r[None, None, kb], tCrB_s2r[None, None, kb])
            for kb in cutlass.range(num_k_block, unroll_full=True):
                cute.gemm(tiled_mma, tCrC, tCrA[None, None, kb], tCrB[None, None, kb], tCrC)

            cute.arch.sync_threads()
            k_start = k_start + cutlass.Int32(tile_k)

        tCrD = cute.make_fragment_like(tCrC, out_dtype)
        tCrD[None] = tCrC.load().to(out_dtype)

        sC = cute.make_tensor(cute.recast_ptr(sA.iterator, dtype=out_dtype), sC_layout)
        tCsC = thr_mma.partition_C(sC)
        cute.autovec_copy(tCrD, tCsC)
        cute.arch.sync_threads()

        li = flat_tid
        while li < cutlass.Int32(smem_c_elems):
            lr = li // cutlass.Int32(tile_n)
            lc = li - lr * cutlass.Int32(tile_n)
            gr = block_m + lr
            gc = block_n + lc
            if gr < m_dim and gc < n_dim:
                out[(gr, gc)] = sC[(lr, lc)]
            li = li + num_thr

    @cute.jit
    def qmm_host_runtime(
        x: cute.Tensor,
        w_q: cute.Tensor,
        scales: cute.Tensor,
        out: cute.Tensor,
    ):
        op = cute.nvgpu.warp.MmaF16BF16Op(out_dtype, cutlass.Float32, (_MMA_ATOM_M, _MMA_ATOM_N, _MMA_ATOM_K))
        atom_layout = cute.make_layout((_MMA_ATOM_LAYOUT_M, _MMA_ATOM_LAYOUT_N, _MMA_ATOM_LAYOUT_K))
        permutation_mnk = (_MMA_TILED_M, _MMA_TILED_N, _MMA_TILED_K)
        tiled_mma = cute.make_tiled_mma(op, atom_layout, permutation_mnk=permutation_mnk)

        sA_layout = _make_mma_smem_layout(tile_m, tile_k, out_dtype, row_major=True)
        sB_layout = _make_mma_smem_layout(tile_n, tile_k, out_dtype, row_major=True)
        sC_layout = _make_mma_smem_layout_c(tile_m, tile_n, out_dtype, row_major=True)

        atom_s2r_A = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), out_dtype)
        atom_s2r_B = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), out_dtype)
        tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
        tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)

        smem_bytes = (
            max(
                cute.size_in_bytes(out_dtype, sC_layout),
                cute.size_in_bytes(out_dtype, sA_layout) + cute.size_in_bytes(out_dtype, sB_layout),
            )
            + 128
        )

        grid_x = (out.shape[0] + tile_m - 1) // tile_m
        grid_y = (out.shape[1] + tile_n - 1) // tile_n
        qmm_mma_kernel(
            x,
            w_q,
            scales,
            out,
            sA_layout,
            sB_layout,
            sC_layout,
            tiled_mma,
            tiled_copy_s2r_A,
            tiled_copy_s2r_B,
        ).launch(
            grid=(grid_x, grid_y, 1),
            block=(num_threads, 1, 1),
            smem=smem_bytes,
        )

    @cute.jit
    def qmm_host_jax(
        stream: cuda.CUstream,
        x: cute.Tensor,
        w_q: cute.Tensor,
        scales: cute.Tensor,
        out: cute.Tensor,
    ):
        op = cute.nvgpu.warp.MmaF16BF16Op(out_dtype, cutlass.Float32, (_MMA_ATOM_M, _MMA_ATOM_N, _MMA_ATOM_K))
        atom_layout = cute.make_layout((_MMA_ATOM_LAYOUT_M, _MMA_ATOM_LAYOUT_N, _MMA_ATOM_LAYOUT_K))
        permutation_mnk = (_MMA_TILED_M, _MMA_TILED_N, _MMA_TILED_K)
        tiled_mma = cute.make_tiled_mma(op, atom_layout, permutation_mnk=permutation_mnk)

        sA_layout = _make_mma_smem_layout(tile_m, tile_k, out_dtype, row_major=True)
        sB_layout = _make_mma_smem_layout(tile_n, tile_k, out_dtype, row_major=True)
        sC_layout = _make_mma_smem_layout_c(tile_m, tile_n, out_dtype, row_major=True)

        atom_s2r_A = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), out_dtype)
        atom_s2r_B = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), out_dtype)
        tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
        tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)

        smem_bytes = (
            max(
                cute.size_in_bytes(out_dtype, sC_layout),
                cute.size_in_bytes(out_dtype, sA_layout) + cute.size_in_bytes(out_dtype, sB_layout),
            )
            + 128
        )

        grid_x = (out.shape[0] + tile_m - 1) // tile_m
        grid_y = (out.shape[1] + tile_n - 1) // tile_n
        qmm_mma_kernel(
            x,
            w_q,
            scales,
            out,
            sA_layout,
            sB_layout,
            sC_layout,
            tiled_mma,
            tiled_copy_s2r_A,
            tiled_copy_s2r_B,
        ).launch(
            grid=(grid_x, grid_y, 1),
            block=(num_threads, 1, 1),
            smem=smem_bytes,
            stream=stream,
        )

    return qmm_host_runtime, qmm_host_jax


def _build_mma_pipelined_qmm_host_fns(
    *,
    mode: str,
    bits: int,
    group_size: int,
    out_dtype: type[cutlass.Numeric],
    with_bias: bool,
    transpose: bool,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    num_stages: int = _MMA_PIPELINE_STAGES,
):
    """Build pipelined MMA tensor-core fused dequant+matmul (SM80+ cp.async).

    Creates a CuTe DSL kernel that extends the single-stage MMA variant with
    a multi-stage software pipeline.  Input activations are loaded via
    ``cp.async`` (128-bit asynchronous global-to-shared-memory copies) into
    a circular buffer of ``num_stages`` shared-memory slots, overlapping
    data movement with MMA computation.  Quantized weights are dequantized
    cooperatively and stored into the corresponding SMEM stage.

    The pipeline structure follows the CUTLASS Ampere ``tensorop_gemm.py``
    pattern: prologue prefetches ``num_stages - 1`` tiles, then the mainloop
    issues future cp.async copies alongside MMA on already-loaded tiles.

    Parallelization: 2-D grid ``(ceil(M/tile_m), ceil(N/tile_n))`` with
    128 threads (4 warps) per block.

    Args:
        mode: Quantization mode.
        bits: Bit-width of quantized weights.
        group_size: Quantization group size.
        out_dtype: CUTLASS output dtype.
        with_bias: Whether per-group bias is used.
        transpose: Whether the weight matrix is transposed.
        tile_m: Tile height (must be a multiple of 32).
        tile_n: Tile width (must be a multiple of 32).
        tile_k: K-dimension tile size (must be a multiple of 16).
        num_stages: Number of pipeline stages for the SMEM circular buffer.

    Returns:
        Tuple of ``(runtime_launcher, jax_stream_launcher)`` callables.

    Raises:
        ValueError: If tile dimensions are incompatible with the MMA atom
            configuration.
    """
    if not _validate_mma_tile(tile_m, tile_n, tile_k):
        raise ValueError(
            f"Tile ({tile_m}, {tile_n}, {tile_k}) incompatible with MMA config "
            f"(need multiples of {_MMA_TILED_M}, {_MMA_TILED_N}, {_MMA_ATOM_K})"
        )

    dequant = _build_dequant_fn(mode=mode, with_bias=with_bias)
    num_threads = _MMA_NUM_THREADS
    smem_w_elems = tile_n * tile_k
    use_nf4_lut = mode.lower() == "nf4"

    if with_bias:

        @cute.kernel
        def qmm_pipe_kernel(
            x: cute.Tensor,
            w_q: cute.Tensor,
            scales: cute.Tensor,
            biases: cute.Tensor,
            out: cute.Tensor,
            sA_layout: cute.ComposedLayout,
            sB_layout: cute.ComposedLayout,
            sC_layout: cute.Layout,
            tiled_copy_g2s: cute.TiledCopy,
            tiled_mma: cute.TiledMma,
            tiled_copy_s2r_A: cute.TiledCopy,
            tiled_copy_s2r_B: cute.TiledCopy,
        ):
            bidx, bidy, _ = cute.arch.block_idx()
            tidx, _, _ = cute.arch.thread_idx()

            m_dim = cutlass.Int32(out.shape[0])
            n_dim = cutlass.Int32(out.shape[1])
            k_dim = cutlass.Int32(x.shape[1])
            block_m = cutlass.Int32(bidx) * cutlass.Int32(tile_m)
            block_n = cutlass.Int32(bidy) * cutlass.Int32(tile_n)
            flat_tid = cutlass.Int32(tidx)
            num_thr = cutlass.Int32(num_threads)
            group_size_i = cutlass.Int32(group_size)

            smem = cutlass.utils.SmemAllocator()
            sA = smem.allocate_tensor(out_dtype, sA_layout, 16)
            sB = smem.allocate_tensor(out_dtype, sB_layout, 16)

            nf4_lut_ptr = cute.arch.smem.alloc_smem(cutlass.Float32, 16)
            nf4_lut = cute.make_tensor(nf4_lut_ptr, cute.make_layout((16,)))
            if use_nf4_lut:
                if flat_tid < cutlass.Int32(16):
                    lut_val = cutlass.Float32(0.0)
                    for _li, _lv in enumerate(_NF4_LUT_VALUES):
                        lut_val = cutlass.select_(
                            flat_tid == cutlass.Int32(_li),
                            cutlass.Float32(_lv),
                            lut_val,
                        )
                    nf4_lut[(flat_tid,)] = lut_val
                cute.arch.sync_threads()

            gA = cute.local_tile(x, tiler=(tile_m, tile_k), coord=(bidx, None))
            gA = cute.make_tensor(gA.iterator.align(16), gA.layout)

            thr_g2s = tiled_copy_g2s.get_slice(tidx)
            tAgA = thr_g2s.partition_S(gA)
            tAgA = cute.make_tensor(tAgA.iterator.align(16), tAgA.layout)
            tAsA = thr_g2s.partition_D(sA)

            mcA = cute.make_identity_tensor(x.layout.shape)
            cA = cute.local_tile(mcA, tiler=(tile_m, tile_k), coord=(bidx, None))
            tAcA = thr_g2s.partition_S(cA)

            tApA = cute.make_rmem_tensor(
                cute.make_layout(
                    (
                        tAgA.shape[0][1],
                        cute.size(tAgA, mode=[1]),
                        cute.size(tAgA, mode=[2]),
                    ),
                    stride=(cute.size(tAgA, mode=[1]), 1, 0),
                ),
                cutlass.Boolean,
            )
            for rest_v in range(tApA.shape[0]):
                for m in range(tApA.shape[1]):
                    tApA[rest_v, m, 0] = cute.elem_less(tAcA[(0, rest_v), m, 0, 0][0], x.shape[0])

            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)

            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])

            gC = cute.local_tile(out, tiler=(tile_m, tile_n), coord=(bidx, bidy))
            tCgC = thr_mma.partition_C(gC)
            tCrC = tiled_mma.make_fragment_C(tCgC)
            tCrC.fill(0.0)

            thr_s2r_A = tiled_copy_s2r_A.get_slice(tidx)
            tCsA_s2r = thr_s2r_A.partition_S(sA)
            tCrA_s2r = thr_s2r_A.retile(tCrA)
            thr_s2r_B = tiled_copy_s2r_B.get_slice(tidx)
            tCsB_s2r = thr_s2r_B.partition_S(sB)
            tCrB_s2r = thr_s2r_B.retile(tCrB)

            num_k_block = cute.size(tCrA, mode=[2])
            k_tile_count = cute.size(tAgA, mode=[3])
            num_smem_stages = cute.size(tAsA, mode=[3])

            tAsA.fill(0)
            cute.arch.sync_threads()

            k_tile_index = cutlass.Int32(0)

            for k in range(tApA.shape[2]):
                if cute.elem_less(cutlass.Int32(-1), tAcA[0, 0, k, 0][1]):
                    cute.copy(
                        tiled_copy_g2s,
                        tAgA[None, None, k, k_tile_index],
                        tAsA[None, None, k, 0],
                        pred=tApA[None, None, k],
                    )
            k_tile_index = k_tile_index + cutlass.Int32(1)
            cute.arch.cp_async_commit_group()

            for stage in range(1, num_smem_stages - 1):
                if stage == k_tile_count:
                    tApA.fill(0)
                cute.copy(
                    tiled_copy_g2s,
                    tAgA[None, None, None, k_tile_index],
                    tAsA[None, None, None, stage],
                    pred=tApA,
                )
                k_tile_index = k_tile_index + cutlass.Int32(1)
                cute.arch.cp_async_commit_group()

            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            for stage in range(num_smem_stages - 1):
                k_start_s = cutlass.Int32(stage * tile_k)
                li = flat_tid
                while li < cutlass.Int32(smem_w_elems):
                    ln = li // cutlass.Int32(tile_k)
                    lk = li - ln * cutlass.Int32(tile_k)
                    gk = k_start_s + lk
                    gn = block_n + ln
                    wv = cutlass.Float32(0.0)
                    if gk < k_dim and gn < n_dim:
                        w_row = cutlass.Int32(0)
                        elem_idx = cutlass.Int32(0)
                        g_idx = cutlass.Int32(0)
                        if transpose:
                            w_row = gn
                            elem_idx = gk
                            g_idx = gk // group_size_i
                        else:
                            w_row = gk
                            elem_idx = gn
                            g_idx = gn // group_size_i
                        q_code = _unpack_bits(w_q, w_row, elem_idx, bits)
                        if use_nf4_lut:
                            sc = cutlass.Float32(scales[(w_row, g_idx)])
                            wv = cutlass.Float32(nf4_lut[(q_code,)]) * sc
                        else:
                            wv = dequant(w_row, g_idx, q_code, scales, biases)
                    sB[(ln, lk, stage)] = out_dtype(wv)
                    li = li + num_thr
                cute.arch.sync_threads()

            smem_pipe_read = 0
            smem_pipe_write = num_smem_stages - 1

            tCsA_p = tCsA_s2r[None, None, None, smem_pipe_read]
            tCsB_p = tCsB_s2r[None, None, None, smem_pipe_read]
            if num_k_block > 1:
                cute.arch.cp_async_wait_group(num_smem_stages - 2)
                cute.arch.sync_threads()
                cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, 0], tCrA_s2r[None, None, 0])
                cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, 0], tCrB_s2r[None, None, 0])

            for k_tile in range(k_tile_count):
                for kb in cutlass.range(num_k_block, unroll_full=True):
                    if kb == num_k_block - 1:
                        tCsA_p = tCsA_s2r[None, None, None, smem_pipe_read]
                        tCsB_p = tCsB_s2r[None, None, None, smem_pipe_read]
                        cute.arch.cp_async_wait_group(num_smem_stages - 2)
                        cute.arch.sync_threads()

                    kb_next = (kb + 1) % num_k_block
                    cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, kb_next], tCrA_s2r[None, None, kb_next])
                    cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, kb_next], tCrB_s2r[None, None, kb_next])

                    if kb == 0:
                        if k_tile + num_smem_stages - 1 < k_tile_count:
                            cute.copy(
                                tiled_copy_g2s,
                                tAgA[None, None, None, k_tile_index],
                                tAsA[None, None, None, smem_pipe_write],
                                pred=tApA,
                            )
                        k_tile_index = k_tile_index + cutlass.Int32(1)
                        cute.arch.cp_async_commit_group()

                        if k_tile + num_smem_stages - 1 < k_tile_count:
                            k_start_w = cutlass.Int32((k_tile + num_smem_stages - 1) * tile_k)
                            li_w = flat_tid
                            while li_w < cutlass.Int32(smem_w_elems):
                                ln_w = li_w // cutlass.Int32(tile_k)
                                lk_w = li_w - ln_w * cutlass.Int32(tile_k)
                                gk_w = k_start_w + lk_w
                                gn_w = block_n + ln_w
                                wv_w = cutlass.Float32(0.0)
                                if gk_w < k_dim and gn_w < n_dim:
                                    w_row_w = cutlass.Int32(0)
                                    elem_idx_w = cutlass.Int32(0)
                                    g_idx_w = cutlass.Int32(0)
                                    if transpose:
                                        w_row_w = gn_w
                                        elem_idx_w = gk_w
                                        g_idx_w = gk_w // group_size_i
                                    else:
                                        w_row_w = gk_w
                                        elem_idx_w = gn_w
                                        g_idx_w = gn_w // group_size_i
                                    q_code_w = _unpack_bits(w_q, w_row_w, elem_idx_w, bits)
                                    if use_nf4_lut:
                                        sc_w = cutlass.Float32(scales[(w_row_w, g_idx_w)])
                                        wv_w = cutlass.Float32(nf4_lut[(q_code_w,)]) * sc_w
                                    else:
                                        wv_w = dequant(w_row_w, g_idx_w, q_code_w, scales, biases)
                                sB[(ln_w, lk_w, smem_pipe_write)] = out_dtype(wv_w)
                                li_w = li_w + num_thr
                            cute.arch.sync_threads()

                        smem_pipe_write = smem_pipe_read
                        smem_pipe_read = (smem_pipe_read + 1) % num_smem_stages

                    cute.gemm(tiled_mma, tCrC, tCrA[None, None, kb], tCrB[None, None, kb], tCrC)

            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            tCrD = cute.make_fragment_like(tCrC, out_dtype)
            tCrD[None] = tCrC.load().to(out_dtype)

            sC = cute.make_tensor(cute.recast_ptr(sA.iterator, dtype=out_dtype), sC_layout)
            tCsC = thr_mma.partition_C(sC)
            cute.autovec_copy(tCrD, tCsC)
            cute.arch.sync_threads()

            li = flat_tid
            while li < cutlass.Int32(tile_m * tile_n):
                lr = li // cutlass.Int32(tile_n)
                lc = li - lr * cutlass.Int32(tile_n)
                gr = block_m + lr
                gc = block_n + lc
                if gr < m_dim and gc < n_dim:
                    out[(gr, gc)] = sC[(lr, lc)]
                li = li + num_thr

        @cute.jit
        def qmm_host_runtime(
            x: cute.Tensor,
            w_q: cute.Tensor,
            scales: cute.Tensor,
            biases: cute.Tensor,
            out: cute.Tensor,
        ):
            op = cute.nvgpu.warp.MmaF16BF16Op(out_dtype, cutlass.Float32, (_MMA_ATOM_M, _MMA_ATOM_N, _MMA_ATOM_K))
            atom_layout = cute.make_layout((_MMA_ATOM_LAYOUT_M, _MMA_ATOM_LAYOUT_N, _MMA_ATOM_LAYOUT_K))
            permutation_mnk = (_MMA_TILED_M, _MMA_TILED_N, _MMA_TILED_K)
            tiled_mma = cute.make_tiled_mma(op, atom_layout, permutation_mnk=permutation_mnk)

            sA_layout = _make_staged_smem_layout(tile_m, tile_k, num_stages, out_dtype, row_major=True)
            sB_layout = _make_staged_smem_layout(tile_n, tile_k, num_stages, out_dtype, row_major=True)
            sC_layout = _make_mma_smem_layout_c(tile_m, tile_n, out_dtype, row_major=True)

            tiled_copy_g2s = _make_gmem_tiled_copy_x(out_dtype, tile_k, num_threads)

            atom_s2r_A = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), out_dtype)
            atom_s2r_B = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), out_dtype)
            tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
            tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)

            smem_bytes = (
                max(
                    cute.size_in_bytes(out_dtype, sC_layout),
                    cute.size_in_bytes(out_dtype, sA_layout) + cute.size_in_bytes(out_dtype, sB_layout),
                )
                + 128
            )

            mA = cute.make_tensor(x.iterator, cute.make_layout((x.shape[0], x.shape[1]), stride=(x.shape[1], 1)))
            grid_x = (out.shape[0] + tile_m - 1) // tile_m
            grid_y = (out.shape[1] + tile_n - 1) // tile_n
            qmm_pipe_kernel(
                mA,
                w_q,
                scales,
                biases,
                out,
                sA_layout,
                sB_layout,
                sC_layout,
                tiled_copy_g2s,
                tiled_mma,
                tiled_copy_s2r_A,
                tiled_copy_s2r_B,
            ).launch(
                grid=(grid_x, grid_y, 1),
                block=(num_threads, 1, 1),
                smem=smem_bytes,
            )

        @cute.jit
        def qmm_host_jax(
            stream: cuda.CUstream,
            x: cute.Tensor,
            w_q: cute.Tensor,
            scales: cute.Tensor,
            biases: cute.Tensor,
            out: cute.Tensor,
        ):
            op = cute.nvgpu.warp.MmaF16BF16Op(out_dtype, cutlass.Float32, (_MMA_ATOM_M, _MMA_ATOM_N, _MMA_ATOM_K))
            atom_layout = cute.make_layout((_MMA_ATOM_LAYOUT_M, _MMA_ATOM_LAYOUT_N, _MMA_ATOM_LAYOUT_K))
            permutation_mnk = (_MMA_TILED_M, _MMA_TILED_N, _MMA_TILED_K)
            tiled_mma = cute.make_tiled_mma(op, atom_layout, permutation_mnk=permutation_mnk)

            sA_layout = _make_staged_smem_layout(tile_m, tile_k, num_stages, out_dtype, row_major=True)
            sB_layout = _make_staged_smem_layout(tile_n, tile_k, num_stages, out_dtype, row_major=True)
            sC_layout = _make_mma_smem_layout_c(tile_m, tile_n, out_dtype, row_major=True)

            tiled_copy_g2s = _make_gmem_tiled_copy_x(out_dtype, tile_k, num_threads)

            atom_s2r_A = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), out_dtype)
            atom_s2r_B = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), out_dtype)
            tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
            tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)

            smem_bytes = (
                max(
                    cute.size_in_bytes(out_dtype, sC_layout),
                    cute.size_in_bytes(out_dtype, sA_layout) + cute.size_in_bytes(out_dtype, sB_layout),
                )
                + 128
            )

            mA = cute.make_tensor(x.iterator, cute.make_layout((x.shape[0], x.shape[1]), stride=(x.shape[1], 1)))
            grid_x = (out.shape[0] + tile_m - 1) // tile_m
            grid_y = (out.shape[1] + tile_n - 1) // tile_n
            qmm_pipe_kernel(
                mA,
                w_q,
                scales,
                biases,
                out,
                sA_layout,
                sB_layout,
                sC_layout,
                tiled_copy_g2s,
                tiled_mma,
                tiled_copy_s2r_A,
                tiled_copy_s2r_B,
            ).launch(
                grid=(grid_x, grid_y, 1),
                block=(num_threads, 1, 1),
                smem=smem_bytes,
                stream=stream,
            )

        return qmm_host_runtime, qmm_host_jax

    @cute.kernel
    def qmm_pipe_kernel(
        x: cute.Tensor,
        w_q: cute.Tensor,
        scales: cute.Tensor,
        out: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        sC_layout: cute.Layout,
        tiled_copy_g2s: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        tiled_copy_s2r_A: cute.TiledCopy,
        tiled_copy_s2r_B: cute.TiledCopy,
    ):
        bidx, bidy, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()

        m_dim = cutlass.Int32(out.shape[0])
        n_dim = cutlass.Int32(out.shape[1])
        k_dim = cutlass.Int32(x.shape[1])
        block_m = cutlass.Int32(bidx) * cutlass.Int32(tile_m)
        block_n = cutlass.Int32(bidy) * cutlass.Int32(tile_n)
        flat_tid = cutlass.Int32(tidx)
        num_thr = cutlass.Int32(num_threads)
        group_size_i = cutlass.Int32(group_size)

        smem = cutlass.utils.SmemAllocator()
        sA = smem.allocate_tensor(out_dtype, sA_layout, 16)
        sB = smem.allocate_tensor(out_dtype, sB_layout, 16)

        nf4_lut_ptr = cute.arch.smem.alloc_smem(cutlass.Float32, 16)
        nf4_lut = cute.make_tensor(nf4_lut_ptr, cute.make_layout((16,)))
        if use_nf4_lut:
            if flat_tid < cutlass.Int32(16):
                lut_val = cutlass.Float32(0.0)
                for _li, _lv in enumerate(_NF4_LUT_VALUES):
                    lut_val = cutlass.select_(
                        flat_tid == cutlass.Int32(_li),
                        cutlass.Float32(_lv),
                        lut_val,
                    )
                nf4_lut[(flat_tid,)] = lut_val
            cute.arch.sync_threads()

        gA = cute.local_tile(x, tiler=(tile_m, tile_k), coord=(bidx, None))
        gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
        thr_g2s = tiled_copy_g2s.get_slice(tidx)
        tAgA = thr_g2s.partition_S(gA)
        tAgA = cute.make_tensor(tAgA.iterator.align(16), tAgA.layout)
        tAsA = thr_g2s.partition_D(sA)

        mcA = cute.make_identity_tensor(x.layout.shape)
        cA = cute.local_tile(mcA, tiler=(tile_m, tile_k), coord=(bidx, None))
        tAcA = thr_g2s.partition_S(cA)

        tApA = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tAgA.shape[0][1],
                    cute.size(tAgA, mode=[1]),
                    cute.size(tAgA, mode=[2]),
                ),
                stride=(cute.size(tAgA, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        for rest_v in range(tApA.shape[0]):
            for m in range(tApA.shape[1]):
                tApA[rest_v, m, 0] = cute.elem_less(tAcA[(0, rest_v), m, 0, 0][0], x.shape[0])

        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)

        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])

        gC = cute.local_tile(out, tiler=(tile_m, tile_n), coord=(bidx, bidy))
        tCgC = thr_mma.partition_C(gC)
        tCrC = tiled_mma.make_fragment_C(tCgC)
        tCrC.fill(0.0)

        thr_s2r_A = tiled_copy_s2r_A.get_slice(tidx)
        tCsA_s2r = thr_s2r_A.partition_S(sA)
        tCrA_s2r = thr_s2r_A.retile(tCrA)
        thr_s2r_B = tiled_copy_s2r_B.get_slice(tidx)
        tCsB_s2r = thr_s2r_B.partition_S(sB)
        tCrB_s2r = thr_s2r_B.retile(tCrB)

        num_k_block = cute.size(tCrA, mode=[2])
        k_tile_count = cute.size(tAgA, mode=[3])
        num_smem_stages = cute.size(tAsA, mode=[3])

        tAsA.fill(0)
        cute.arch.sync_threads()

        k_tile_index = cutlass.Int32(0)

        for k in range(tApA.shape[2]):
            if cute.elem_less(cutlass.Int32(-1), tAcA[0, 0, k, 0][1]):
                cute.copy(
                    tiled_copy_g2s,
                    tAgA[None, None, k, k_tile_index],
                    tAsA[None, None, k, 0],
                    pred=tApA[None, None, k],
                )
        k_tile_index = k_tile_index + cutlass.Int32(1)
        cute.arch.cp_async_commit_group()

        for stage in range(1, num_smem_stages - 1):
            if stage == k_tile_count:
                tApA.fill(0)
            cute.copy(
                tiled_copy_g2s,
                tAgA[None, None, None, k_tile_index],
                tAsA[None, None, None, stage],
                pred=tApA,
            )
            k_tile_index = k_tile_index + cutlass.Int32(1)
            cute.arch.cp_async_commit_group()

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        for stage in range(num_smem_stages - 1):
            k_start_s = cutlass.Int32(stage * tile_k)
            li = flat_tid
            while li < cutlass.Int32(smem_w_elems):
                ln = li // cutlass.Int32(tile_k)
                lk = li - ln * cutlass.Int32(tile_k)
                gk = k_start_s + lk
                gn = block_n + ln
                wv = cutlass.Float32(0.0)
                if gk < k_dim and gn < n_dim:
                    w_row = cutlass.Int32(0)
                    elem_idx = cutlass.Int32(0)
                    g_idx = cutlass.Int32(0)
                    if transpose:
                        w_row = gn
                        elem_idx = gk
                        g_idx = gk // group_size_i
                    else:
                        w_row = gk
                        elem_idx = gn
                        g_idx = gn // group_size_i
                    q_code = _unpack_bits(w_q, w_row, elem_idx, bits)
                    if use_nf4_lut:
                        sc = cutlass.Float32(scales[(w_row, g_idx)])
                        wv = cutlass.Float32(nf4_lut[(q_code,)]) * sc
                    else:
                        wv = dequant(w_row, g_idx, q_code, scales)
                sB[(ln, lk, stage)] = out_dtype(wv)
                li = li + num_thr
            cute.arch.sync_threads()

        smem_pipe_read = 0
        smem_pipe_write = num_smem_stages - 1

        tCsA_p = tCsA_s2r[None, None, None, smem_pipe_read]
        tCsB_p = tCsB_s2r[None, None, None, smem_pipe_read]
        if num_k_block > 1:
            cute.arch.cp_async_wait_group(num_smem_stages - 2)
            cute.arch.sync_threads()
            cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, 0], tCrA_s2r[None, None, 0])
            cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, 0], tCrB_s2r[None, None, 0])

        for k_tile in range(k_tile_count):
            for kb in cutlass.range(num_k_block, unroll_full=True):
                if kb == num_k_block - 1:
                    tCsA_p = tCsA_s2r[None, None, None, smem_pipe_read]
                    tCsB_p = tCsB_s2r[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(num_smem_stages - 2)
                    cute.arch.sync_threads()

                kb_next = (kb + 1) % num_k_block
                cute.copy(tiled_copy_s2r_A, tCsA_p[None, None, kb_next], tCrA_s2r[None, None, kb_next])
                cute.copy(tiled_copy_s2r_B, tCsB_p[None, None, kb_next], tCrB_s2r[None, None, kb_next])

                if kb == 0:
                    if k_tile + num_smem_stages - 1 < k_tile_count:
                        cute.copy(
                            tiled_copy_g2s,
                            tAgA[None, None, None, k_tile_index],
                            tAsA[None, None, None, smem_pipe_write],
                            pred=tApA,
                        )
                    k_tile_index = k_tile_index + cutlass.Int32(1)
                    cute.arch.cp_async_commit_group()

                    if k_tile + num_smem_stages - 1 < k_tile_count:
                        k_start_w = cutlass.Int32((k_tile + num_smem_stages - 1) * tile_k)
                        li_w = flat_tid
                        while li_w < cutlass.Int32(smem_w_elems):
                            ln_w = li_w // cutlass.Int32(tile_k)
                            lk_w = li_w - ln_w * cutlass.Int32(tile_k)
                            gk_w = k_start_w + lk_w
                            gn_w = block_n + ln_w
                            wv_w = cutlass.Float32(0.0)
                            if gk_w < k_dim and gn_w < n_dim:
                                w_row_w = cutlass.Int32(0)
                                elem_idx_w = cutlass.Int32(0)
                                g_idx_w = cutlass.Int32(0)
                                if transpose:
                                    w_row_w = gn_w
                                    elem_idx_w = gk_w
                                    g_idx_w = gk_w // group_size_i
                                else:
                                    w_row_w = gk_w
                                    elem_idx_w = gn_w
                                    g_idx_w = gn_w // group_size_i
                                q_code_w = _unpack_bits(w_q, w_row_w, elem_idx_w, bits)
                                if use_nf4_lut:
                                    sc_w = cutlass.Float32(scales[(w_row_w, g_idx_w)])
                                    wv_w = cutlass.Float32(nf4_lut[(q_code_w,)]) * sc_w
                                else:
                                    wv_w = dequant(w_row_w, g_idx_w, q_code_w, scales)
                            sB[(ln_w, lk_w, smem_pipe_write)] = out_dtype(wv_w)
                            li_w = li_w + num_thr
                        cute.arch.sync_threads()

                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read = (smem_pipe_read + 1) % num_smem_stages

                cute.gemm(tiled_mma, tCrC, tCrA[None, None, kb], tCrB[None, None, kb], tCrC)

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        tCrD = cute.make_fragment_like(tCrC, out_dtype)
        tCrD[None] = tCrC.load().to(out_dtype)

        sC = cute.make_tensor(cute.recast_ptr(sA.iterator, dtype=out_dtype), sC_layout)
        tCsC = thr_mma.partition_C(sC)
        cute.autovec_copy(tCrD, tCsC)
        cute.arch.sync_threads()

        li = flat_tid
        while li < cutlass.Int32(tile_m * tile_n):
            lr = li // cutlass.Int32(tile_n)
            lc = li - lr * cutlass.Int32(tile_n)
            gr = block_m + lr
            gc = block_n + lc
            if gr < m_dim and gc < n_dim:
                out[(gr, gc)] = sC[(lr, lc)]
            li = li + num_thr

    @cute.jit
    def qmm_host_runtime(
        x: cute.Tensor,
        w_q: cute.Tensor,
        scales: cute.Tensor,
        out: cute.Tensor,
    ):
        op = cute.nvgpu.warp.MmaF16BF16Op(out_dtype, cutlass.Float32, (_MMA_ATOM_M, _MMA_ATOM_N, _MMA_ATOM_K))
        atom_layout = cute.make_layout((_MMA_ATOM_LAYOUT_M, _MMA_ATOM_LAYOUT_N, _MMA_ATOM_LAYOUT_K))
        permutation_mnk = (_MMA_TILED_M, _MMA_TILED_N, _MMA_TILED_K)
        tiled_mma = cute.make_tiled_mma(op, atom_layout, permutation_mnk=permutation_mnk)

        sA_layout = _make_staged_smem_layout(tile_m, tile_k, num_stages, out_dtype, row_major=True)
        sB_layout = _make_staged_smem_layout(tile_n, tile_k, num_stages, out_dtype, row_major=True)
        sC_layout = _make_mma_smem_layout_c(tile_m, tile_n, out_dtype, row_major=True)

        tiled_copy_g2s = _make_gmem_tiled_copy_x(out_dtype, tile_k, num_threads)

        atom_s2r_A = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), out_dtype)
        atom_s2r_B = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), out_dtype)
        tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
        tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)

        smem_bytes = (
            max(
                cute.size_in_bytes(out_dtype, sC_layout),
                cute.size_in_bytes(out_dtype, sA_layout) + cute.size_in_bytes(out_dtype, sB_layout),
            )
            + 128
        )

        mA = cute.make_tensor(x.iterator, cute.make_layout((x.shape[0], x.shape[1]), stride=(x.shape[1], 1)))
        grid_x = (out.shape[0] + tile_m - 1) // tile_m
        grid_y = (out.shape[1] + tile_n - 1) // tile_n
        qmm_pipe_kernel(
            mA,
            w_q,
            scales,
            out,
            sA_layout,
            sB_layout,
            sC_layout,
            tiled_copy_g2s,
            tiled_mma,
            tiled_copy_s2r_A,
            tiled_copy_s2r_B,
        ).launch(
            grid=(grid_x, grid_y, 1),
            block=(num_threads, 1, 1),
            smem=smem_bytes,
        )

    @cute.jit
    def qmm_host_jax(
        stream: cuda.CUstream,
        x: cute.Tensor,
        w_q: cute.Tensor,
        scales: cute.Tensor,
        out: cute.Tensor,
    ):
        op = cute.nvgpu.warp.MmaF16BF16Op(out_dtype, cutlass.Float32, (_MMA_ATOM_M, _MMA_ATOM_N, _MMA_ATOM_K))
        atom_layout = cute.make_layout((_MMA_ATOM_LAYOUT_M, _MMA_ATOM_LAYOUT_N, _MMA_ATOM_LAYOUT_K))
        permutation_mnk = (_MMA_TILED_M, _MMA_TILED_N, _MMA_TILED_K)
        tiled_mma = cute.make_tiled_mma(op, atom_layout, permutation_mnk=permutation_mnk)

        sA_layout = _make_staged_smem_layout(tile_m, tile_k, num_stages, out_dtype, row_major=True)
        sB_layout = _make_staged_smem_layout(tile_n, tile_k, num_stages, out_dtype, row_major=True)
        sC_layout = _make_mma_smem_layout_c(tile_m, tile_n, out_dtype, row_major=True)

        tiled_copy_g2s = _make_gmem_tiled_copy_x(out_dtype, tile_k, num_threads)

        atom_s2r_A = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), out_dtype)
        atom_s2r_B = cute.make_copy_atom(cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4), out_dtype)
        tiled_copy_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
        tiled_copy_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)

        smem_bytes = (
            max(
                cute.size_in_bytes(out_dtype, sC_layout),
                cute.size_in_bytes(out_dtype, sA_layout) + cute.size_in_bytes(out_dtype, sB_layout),
            )
            + 128
        )

        mA = cute.make_tensor(x.iterator, cute.make_layout((x.shape[0], x.shape[1]), stride=(x.shape[1], 1)))
        grid_x = (out.shape[0] + tile_m - 1) // tile_m
        grid_y = (out.shape[1] + tile_n - 1) // tile_n
        qmm_pipe_kernel(
            mA,
            w_q,
            scales,
            out,
            sA_layout,
            sB_layout,
            sC_layout,
            tiled_copy_g2s,
            tiled_mma,
            tiled_copy_s2r_A,
            tiled_copy_s2r_B,
        ).launch(
            grid=(grid_x, grid_y, 1),
            block=(num_threads, 1, 1),
            smem=smem_bytes,
            stream=stream,
        )

    return qmm_host_runtime, qmm_host_jax


def _build_fused_qmm_host_fns(
    *,
    mode: str,
    bits: int,
    group_size: int,
    out_dtype: type[cutlass.Numeric],
    with_bias: bool,
    transpose: bool,
    tile_m: int,
    tile_n: int,
    tile_k: int,
):
    """Build fused dequant+matmul CuTe host launchers with automatic fallback.

    Tries kernel variants in decreasing performance order:
    pipelined MMA (cp.async) -> single-stage MMA -> tiled SMEM scalar -> naive
    scalar.  Environment variables can force a specific path:

    - ``EJKERNEL_CUTE_QMM_USE_NAIVE=1``: naive scalar only.
    - ``EJKERNEL_CUTE_QMM_USE_TILED=1``: tiled SMEM scalar.
    - ``EJKERNEL_CUTE_QMM_USE_MMA_SINGLE=1``: single-stage MMA (no pipeline).

    Args:
        mode: Quantization mode.
        bits: Bit-width of quantized weights.
        group_size: Quantization group size.
        out_dtype: CUTLASS output dtype.
        with_bias: Whether per-group bias is used.
        transpose: Whether the weight matrix is transposed.
        tile_m: Output tile height.
        tile_n: Output tile width.
        tile_k: K-dimension tile size.

    Returns:
        Tuple of ``(runtime_launcher, jax_stream_launcher)`` callables.
    """
    kw = dict(
        mode=mode,
        bits=bits,
        group_size=group_size,
        out_dtype=out_dtype,
        with_bias=with_bias,
        transpose=transpose,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
    )

    if _use_naive_kernel():
        return _build_naive_qmm_host_fns(**kw)

    if _use_tiled_scalar_kernel():
        try:
            return _build_tiled_qmm_host_fns(**kw)
        except Exception:
            return _build_naive_qmm_host_fns(**kw)

    if _use_mma_single_stage():
        if _validate_mma_tile(tile_m, tile_n, tile_k):
            try:
                return _build_mma_qmm_host_fns(**kw)
            except Exception:
                pass
        try:
            return _build_tiled_qmm_host_fns(**kw)
        except Exception:
            return _build_naive_qmm_host_fns(**kw)

    if _validate_mma_tile(tile_m, tile_n, tile_k):
        try:
            return _build_mma_pipelined_qmm_host_fns(**kw)
        except Exception:
            try:
                return _build_mma_qmm_host_fns(**kw)
            except Exception:
                pass

    try:
        return _build_tiled_qmm_host_fns(**kw)
    except Exception:
        return _build_naive_qmm_host_fns(**kw)


def _wrap_vmap_compatible_jax_call(call: Callable[..., jax.Array]) -> Callable[..., jax.Array]:
    """Wrap a JAX call with an explicit ``custom_vmap`` batching rule.

    CuTe primitive calls compiled through ``build_cute_ffi_call`` do not
    automatically provide a batching rule, so ``jax.vmap`` would fail.
    This wrapper keeps eager/JIT behavior unchanged (the inner ``call`` is
    invoked directly) while registering a ``def_vmap`` rule that maps the
    unbatched primitive across the leading batch axis via ``jax.lax.map``.

    Args:
        call: An unbatched JAX callable produced by :func:`_build_primitive_qmm_call`.

    Returns:
        A new callable with identical semantics but ``jax.vmap``-compatible.
    """

    @custom_batching.custom_vmap
    def _wrapped(*runtime_args):
        return call(*runtime_args)

    @_wrapped.def_vmap
    def _wrapped_vmap(axis_size, in_batched, *runtime_args):
        if not any(in_batched):
            return _wrapped(*runtime_args), False

        mapped_args = []
        for arg, is_batched in zip(runtime_args, in_batched, strict=False):
            if is_batched:
                mapped_args.append(arg)
            else:
                mapped_args.append(jnp.broadcast_to(arg, (axis_size, *arg.shape)))

        out = jax.lax.map(lambda args_i: call(*args_i), tuple(mapped_args))
        out_batched = jax.tree_util.tree_map(lambda _: True, out)
        return out, out_batched

    return _wrapped


def _build_primitive_qmm_call(
    *,
    jax_host_fn: Callable[..., None],
    out_shape: jax.ShapeDtypeStruct,
    use_vmap_wrapper: bool = True,
) -> Callable[..., jax.Array]:
    """Build a CuTe primitive fused-QMM call wrapper.

    Compiles the given CuTe host function into a JAX-callable primitive via
    ``build_cute_ffi_call`` and optionally wraps it with a ``custom_vmap``
    batching rule so that ``jax.vmap`` works out of the box.

    Args:
        jax_host_fn: A ``@cute.jit`` host launcher that accepts a CUDA stream
            and CuTe tensors.
        out_shape: Shape and dtype descriptor for the output array.
        use_vmap_wrapper: Whether to wrap the call with an explicit batching
            rule for ``jax.vmap`` compatibility.

    Returns:
        A callable that accepts JAX arrays and returns the QMM result.

    Raises:
        RuntimeError: If CuTe TVM-FFI support is not available or compilation fails.
    """
    if not has_cute_ffi_support():
        raise RuntimeError(
            "CUTE quantized_matmul requires CuTe primitive support. "
            "Ensure CuTe TVM-FFI support is available (install apache-tvm-ffi)."
        )

    try:
        call = build_cute_ffi_call(
            jax_host_fn,
            output_shape_dtype=out_shape,
            compile_options="--enable-tvm-ffi",
        )
        if use_vmap_wrapper:
            call = _wrap_vmap_compatible_jax_call(call)
    except Exception as exc:
        raise RuntimeError("Failed to build CuTe primitive call for CUTE quantized_matmul.") from exc
    return call


def _qmm_call_name(
    *,
    mode: str,
    bits: int,
    group_size: int,
    out_dtype: jnp.dtype,
    transpose: bool,
    with_bias: bool,
    kernel_family: str,
    revsplit_k_parts: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
) -> str:
    """Build a stable human-readable name for a CuTe fused-QMM call site/cache key.

    The name encodes all kernel-specialisation parameters so that the same
    string can be used as both the ``cute_call`` scope name for profiling and
    as part of the ``_QMM_CALL_CACHE`` key.
    """
    return (
        "cute_quantized_matmul_fused_"
        f"{mode.lower()}_b{int(bits)}_g{int(group_size)}_"
        f"{jnp.dtype(out_dtype).name}_{'t' if transpose else 'nt'}_"
        f"{'bias' if with_bias else 'nobias'}_"
        f"{kernel_family}_rsk{int(revsplit_k_parts)}_"
        f"bm{int(tile_m)}_bn{int(tile_n)}_bk{int(tile_k)}"
    )


def _array_device_key(arr: Any) -> tuple[str | None, int | None, str] | None:
    """Return a hashable device descriptor for *arr*, or ``None`` if unavailable.

    Used to make the QMM primitive cache device-aware so that the same
    compiled kernel is not reused across distinct physical devices.

    Returns:
        A ``(platform, device_id, str_repr)`` tuple, or ``None`` if *arr* is
        not a concrete ``jax.Array`` or its device cannot be determined.
    """
    if not isinstance(arr, jax.Array):
        return None
    try:
        dev = arr.device()
    except Exception:
        dev = None
    if dev is None:
        return None
    return (getattr(dev, "platform", None), getattr(dev, "id", None), str(dev))


def _qmm_primitive_cache_key(
    *,
    x: Any,
    w_q: Any,
    scales: Any,
    biases: Any | None,
    mode: str,
    bits: int,
    group_size: int,
    transpose: bool,
    out_dtype: jnp.dtype,
    with_bias: bool,
    kernel_family: str,
    revsplit_k_parts: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    out_shape: tuple[int, int],
) -> tuple[Any, ...]:
    """Build a hashable cache key for a compiled CuTe fused-QMM primitive.

    Encodes every axis of specialisation — quantization parameters, tile
    configuration, dtypes, tensor shapes, kernel family, and device
    placement — so that cached primitives are never reused for a different
    kernel variant or input configuration.

    Returns:
        A flat tuple of primitives suitable as a ``dict`` key.
    """
    return (
        mode.lower(),
        int(bits),
        int(group_size),
        bool(transpose),
        bool(with_bias),
        str(kernel_family),
        int(revsplit_k_parts),
        str(jnp.dtype(out_dtype)),
        int(tile_m),
        int(tile_n),
        int(tile_k),
        _shape_key(tuple(getattr(x, "shape", ()))),
        _shape_key(tuple(getattr(w_q, "shape", ()))),
        _shape_key(tuple(getattr(scales, "shape", ()))),
        _shape_key(tuple(getattr(biases, "shape", ()))) if biases is not None else None,
        str(jnp.dtype(getattr(x, "dtype", out_dtype))),
        str(jnp.dtype(getattr(w_q, "dtype", jnp.uint32))),
        str(jnp.dtype(getattr(scales, "dtype", jnp.float32))),
        str(jnp.dtype(getattr(biases, "dtype", jnp.float32))) if biases is not None else None,
        _shape_key(out_shape),
        _array_device_key(x),
    )


def get_cute_qmm_call(
    *,
    x: jax.Array,
    w_q: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None,
    mode: str,
    bits: int,
    group_size: int,
    transpose: bool,
    gemv_mode: str,
    revsplit_k: str,
    revsplit_k_parts: int | None,
    use_bf16: bool,
    block_m: int = _DEFAULT_BLOCK_M,
    block_n: int = _DEFAULT_BLOCK_N,
    block_k: int = _DEFAULT_BLOCK_K,
) -> Callable[..., jax.Array]:
    """Build a fused dequant+matmul CuTe primitive wrapper.

    This is the main entry point for CuTe-backed quantized matrix
    multiplication.  It selects the kernel family (fused GEMM or
    GEMV with reverse split-K), normalizes the explicit tile dimensions
    supplied by the operation executor, compiles the appropriate CuTe DSL
    kernel, and returns a callable that performs
    ``out = x @ dequant(w_q, scales[, biases])``.

    The compiled kernel and its CuTe FFI primitive are cached so that
    repeated calls with the same configuration reuse the compiled artefact.

    Args:
        x: Input activation tensor, shape ``[M, K]``.
        w_q: Packed quantized weights, shape depends on *transpose* and *bits*.
        scales: Per-group quantization scales.
        biases: Optional per-group quantization biases (affine mode).
        mode: Quantization mode (``"affine"``, ``"nf4"``, ``"mxfp4"``,
            ``"mxfp8"``, ``"nvfp4"``, ``"nvfp8"``).
        bits: Bit-width of quantized weights.
        group_size: Number of weight elements per quantization group.
        transpose: Whether the weight matrix is stored transposed.
        gemv_mode: GEMV dispatch mode hint (``"auto"``, ``"never"``, etc.).
        revsplit_k: Reverse split-K mode hint (``"auto"``, ``"never"``, etc.).
        revsplit_k_parts: Number of split-K partitions (if applicable).
        use_bf16: If True, use bfloat16 output; otherwise float16.
        block_m: Requested M tile size for the kernel grid.
        block_n: Requested N tile size for the kernel grid.
        block_k: Requested K tile size for the kernel grid.

    Returns:
        A callable ``(x, w_q, scales[, biases], *, out=None) -> jax.Array``
        that computes the fused dequantize + matmul operation.  If *out* is
        provided, the result is written into it.
    """
    out_dtype = jnp.bfloat16 if use_bf16 else jnp.float16
    out_cute_dtype = cutlass.BFloat16 if use_bf16 else cutlass.Float16
    with_bias = biases is not None
    gemv_mode = normalize_gemv_mode(gemv_mode)
    revsplit_k = normalize_revsplitk_mode(revsplit_k)
    revsplit_k_parts = normalize_revsplitk_parts(revsplit_k_parts)
    kernel_family, family_revsplit_parts = select_qmm_kernel_family(
        m=int(x.shape[0]),
        mode=mode.lower(),  # type: ignore[arg-type]
        bits=int(bits),
        gemv_mode=gemv_mode,
        revsplit_k=revsplit_k,
        revsplit_k_parts=revsplit_k_parts,
    )
    if kernel_family == "gemv_revsplitk":
        revsplit_parts_eff = 2 if family_revsplit_parts is None else int(family_revsplit_parts)
    else:
        revsplit_parts_eff = 0
    out_shape = _infer_output_shape(
        x=x,
        scales=scales,
        group_size=group_size,
        transpose=transpose,
    )
    out_struct = jax.ShapeDtypeStruct(out_shape, out_dtype)
    tile_m, tile_n, tile_k = (
        _normalize_block(block_m, _DEFAULT_BLOCK_M),
        _normalize_block(block_n, _DEFAULT_BLOCK_N),
        _normalize_block(block_k, _DEFAULT_BLOCK_K),
    )
    call_name = _qmm_call_name(
        mode=mode,
        bits=bits,
        group_size=group_size,
        out_dtype=jnp.dtype(out_dtype),
        transpose=transpose,
        with_bias=with_bias,
        kernel_family=kernel_family,
        revsplit_k_parts=revsplit_parts_eff,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
    )
    primitive_key = _qmm_primitive_cache_key(
        x=x,
        w_q=w_q,
        scales=scales,
        biases=biases,
        mode=mode,
        bits=bits,
        group_size=group_size,
        transpose=transpose,
        out_dtype=jnp.dtype(out_dtype),
        with_bias=with_bias,
        kernel_family=kernel_family,
        revsplit_k_parts=revsplit_parts_eff,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        out_shape=out_shape,
    )

    with _QMM_CALL_CACHE_LOCK:
        cached = _QMM_CALL_CACHE.get(primitive_key)
    if cached is None:
        _, jax_host_fn = _build_fused_qmm_host_fns(
            mode=mode,
            bits=bits,
            group_size=group_size,
            out_dtype=out_cute_dtype,
            with_bias=with_bias,
            transpose=transpose,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
        )
        qmm_call = _build_primitive_qmm_call(
            jax_host_fn=jax_host_fn,
            out_shape=out_struct,
            use_vmap_wrapper=True,
        )
        with _QMM_CALL_CACHE_LOCK:
            existing = _QMM_CALL_CACHE.get(primitive_key)
            if existing is None:
                _QMM_CALL_CACHE[primitive_key] = (qmm_call, call_name)
            else:
                qmm_call, call_name = existing
    else:
        qmm_call, call_name = cached

    def _call(x_arr, w_arr, scales_arr, biases_arr=None, *, out=None):
        x_cast = x_arr.astype(out_dtype)
        if with_bias:
            y = cute_call(
                x_cast,
                w_arr,
                scales_arr,
                biases_arr,
                call=qmm_call,
                out_shape=out_struct,
                name=call_name,
            )
        else:
            y = cute_call(
                x_cast,
                w_arr,
                scales_arr,
                call=qmm_call,
                out_shape=out_struct,
                name=call_name,
            )
        if out is not None:
            return out.at[:, :].set(y.astype(out.dtype))
        return y.astype(out_dtype)

    return _call
