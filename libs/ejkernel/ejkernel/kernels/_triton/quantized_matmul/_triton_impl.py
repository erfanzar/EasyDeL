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

"""Triton kernels for quantized matrix multiplication.

This module contains the low-level Triton GPU kernels for quantized matmul
operations. It provides optimized fused dequantization and matmul kernels
for "affine", "nf4", "mxfp4", "mxfp8", "nvfp4", and "nvfp8" quantization modes.

The kernels use split-K parallelism for improved performance on small M
dimensions and support both transposed (NxK) and non-transposed (KxN)
weight layouts.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Literal

import jax
import jax.numpy as jnp
import triton
import triton.language as tl

from ejkernel.callib import cdiv, strides_from_shape, triton_call
from ejkernel.quantization._utils.fp_tables import _get_e2m1_table, _get_e4m3_table, _get_nf4_table
from ejkernel.quantization._utils.qparams import (
    normalize_gemv_mode,
    normalize_revsplitk_mode,
    normalize_revsplitk_parts,
    resolve_qparams,
    select_qmm_kernel_family,
)

from ._triton_impl_gemv import quantized_matmul_triton_gemv

QuantizationMode = Literal["affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"]
GemvMode = Literal["auto", "on", "off"]
RevSplitKMode = Literal["auto", "on", "off"]


def _get_decode_tables() -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """Build decode lookup tables as local arrays (no global state)."""

    nf4_table = _get_nf4_table()
    e2m1_table, _ = _get_e2m1_table()
    e4m3_table, _ = _get_e4m3_table()
    e8m0_exp2_table = jnp.exp2(jnp.arange(256, dtype=jnp.uint8).astype(jnp.int8).astype(jnp.float32))
    return nf4_table, e2m1_table, e4m3_table, e8m0_exp2_table


@triton.jit
def _nf4_to_f32(x: tl.tensor, table_ptr) -> tl.tensor:
    """Convert 4-bit NF4 codes to float32 via table lookup."""
    return tl.load(table_ptr + x)


@triton.jit
def _e2m1_to_f32(x: tl.tensor, table_ptr) -> tl.tensor:
    """Convert 4-bit E2M1 codes to float32 via table lookup."""
    return tl.load(table_ptr + x)


@triton.jit
def _e4m3_to_f32(x: tl.tensor, table_ptr) -> tl.tensor:
    """Convert 8-bit E4M3 codes to float32 via table lookup."""
    return tl.load(table_ptr + x)


@triton.jit
def _unpack_packed_codes(word0, word1, shifts, BITS: tl.constexpr):
    """Decode packed uint32 codes for arbitrary affine bit-widths."""
    low_bits = tl.minimum(32 - shifts, BITS)
    high_bits = BITS - low_bits
    one = tl.full(shifts.shape, 1, tl.uint32)
    low_mask = (one << low_bits) - 1
    high_mask = (one << high_bits) - 1
    low = (word0 >> shifts) & low_mask
    high = word1 & high_mask
    return low | (high << low_bits)


@triton.jit
def qmm_dequant_nf4_kernel(
    Wq,
    Wscale,
    NF4_TABLE,
    N,
    K,
    O,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_or: tl.constexpr,
    stride_oc: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    """Dequantize a packed NF4-quantized weight tile to fp16/bf16.

    Grid: ``(cdiv(rows, BR), cdiv(cols, BC))``

    Unpacks ``VALUES_PER_WORD = 8`` 4-bit NF4 codes from each int32 word and
    converts them to float32 via ``NF4_TABLE`` lookup, then multiplies by the
    per-group scale ``Wscale``.  The output is cast to bfloat16 if
    ``OUT_BF16=True``, otherwise float16.

    Args:
        Wq: Packed quantized weight pointer.  When ``TRANSPOSE=False`` the
            shape is ``(K // VALUES_PER_WORD, N)`` (rows are K, cols are N);
            when ``TRANSPOSE=True`` the shape is ``(N // VALUES_PER_WORD, K)``.
        Wscale: Per-group scale pointer, shape ``(rows, groups)`` where
            ``groups = cols // GROUP_SIZE``.
        NF4_TABLE: Lookup table pointer for 16-entry NF4 float32 values.
        N: Output column count (output feature dimension).
        K: Output row count (input feature / reduction dimension).
        O: Output buffer pointer, shape ``(rows, cols)`` in dequantised layout.
        stride_wq0: Row stride of ``Wq``.
        stride_wq1: Column stride of ``Wq``.
        stride_ws0: Row stride of ``Wscale``.
        stride_ws1: Column stride of ``Wscale``.
        stride_or: Row stride of ``O``.
        stride_oc: Column stride of ``O``.
        GROUP_SIZE: Number of values sharing one scale (constexpr).
        VALUES_PER_WORD: Packed values per int32 storage word; equals 8 for
            4-bit NF4 (constexpr).
        BR: Tile rows per CTA (constexpr).
        BC: Tile columns per CTA (constexpr).
        TRANSPOSE: Whether weight is stored transposed (NxK layout)
            vs non-transposed (KxN layout) (constexpr).
        OUT_BF16: Output in bfloat16 when True, float16 when False (constexpr).
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_r = pid_r * BR + tl.arange(0, BR)
    offs_c = pid_c * BC + tl.arange(0, BC)

    if TRANSPOSE:
        r_mask = offs_r < N
        c_mask = offs_c < K
        word_offsets = offs_c // VALUES_PER_WORD
        word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
        shifts = (offs_c % VALUES_PER_WORD) * 4
        w_word0 = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = (w_word0 >> shifts[None, :]) & 0xF
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(K, GROUP_SIZE)
        ws = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0.0,
        )
        out = _nf4_to_f32(q.to(tl.int32), NF4_TABLE) * ws
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )
    else:
        r_mask = offs_r < K
        c_mask = offs_c < N
        word_offsets = offs_c // VALUES_PER_WORD
        word_mask = word_offsets < tl.cdiv(N, VALUES_PER_WORD)
        shifts = (offs_c % VALUES_PER_WORD) * 4
        w_word0 = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = (w_word0 >> shifts[None, :]) & 0xF
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(N, GROUP_SIZE)
        ws = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0.0,
        )
        out = _nf4_to_f32(q.to(tl.int32), NF4_TABLE) * ws
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )


@triton.jit
def qmm_dequant_affine4_kernel(
    Wq,
    Wscale,
    Wbias,
    N,
    K,
    O,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_wb0: tl.constexpr,
    stride_wb1: tl.constexpr,
    stride_or: tl.constexpr,
    stride_oc: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BITS: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    """Dequantize a packed affine-quantized weight tile to fp16/bf16.

    Grid: ``(cdiv(rows, BR), cdiv(cols, BC))``

    Unpacks affine codes from a contiguous bitstream with ``BITS`` bits per
    code and applies the per-group affine transform
    ``out = code * scale + bias``.
    The output is cast to bfloat16 if ``OUT_BF16=True``, otherwise float16.

    Args:
        Wq: Packed quantized weight pointer.  When ``TRANSPOSE=False`` the
            shape is ``(K // VALUES_PER_WORD, N)``; when ``TRANSPOSE=True``
            the shape is ``(N // VALUES_PER_WORD, K)``.
        Wscale: Per-group scale pointer, shape ``(rows, groups)`` where
            ``groups = cols // GROUP_SIZE``.
        Wbias: Per-group zero-point / bias pointer, same shape as ``Wscale``.
        N: Output column count (output feature dimension).
        K: Output row count (input feature / reduction dimension).
        O: Output buffer pointer in dequantised layout.
        stride_wq0: Row stride of ``Wq``.
        stride_wq1: Column stride of ``Wq``.
        stride_ws0: Row stride of ``Wscale``.
        stride_ws1: Column stride of ``Wscale``.
        stride_wb0: Row stride of ``Wbias``.
        stride_wb1: Column stride of ``Wbias``.
        stride_or: Row stride of ``O``.
        stride_oc: Column stride of ``O``.
        GROUP_SIZE: Number of values sharing one (scale, bias) pair (constexpr).
        VALUES_PER_WORD: Legacy packed values-per-word hint.
        BITS: Bits per affine code, from 1 through 8.
        BR: Tile rows per CTA (constexpr).
        BC: Tile columns per CTA (constexpr).
        TRANSPOSE: Whether weight is stored transposed (NxK) vs (KxN) (constexpr).
        OUT_BF16: Output in bfloat16 when True, float16 when False (constexpr).
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_r = pid_r * BR + tl.arange(0, BR)
    offs_c = pid_c * BC + tl.arange(0, BC)

    if TRANSPOSE:
        r_mask = offs_r < N
        c_mask = offs_c < K
        bit_offsets = offs_c * BITS
        word_offsets = bit_offsets // 32
        shifts = bit_offsets - word_offsets * 32
        n_words = tl.cdiv(K * BITS, 32)
        word_mask = word_offsets < n_words
        word_offsets1 = tl.minimum(word_offsets + 1, n_words - 1)
        w_word0 = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        w_word1 = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets1[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = _unpack_packed_codes(w_word0, w_word1, shifts[None, :], BITS)
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(K, GROUP_SIZE)
        ws = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0.0,
        )
        wb = tl.load(
            Wbias + offs_r[:, None] * stride_wb0 + group_idx[None, :] * stride_wb1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0.0,
        )
        out = q.to(ws.dtype) * ws + wb
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )
    else:
        r_mask = offs_r < K
        c_mask = offs_c < N
        bit_offsets = offs_c * BITS
        word_offsets = bit_offsets // 32
        shifts = bit_offsets - word_offsets * 32
        n_words = tl.cdiv(N * BITS, 32)
        word_mask = word_offsets < n_words
        word_offsets1 = tl.minimum(word_offsets + 1, n_words - 1)
        w_word0 = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        w_word1 = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets1[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = _unpack_packed_codes(w_word0, w_word1, shifts[None, :], BITS)
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(N, GROUP_SIZE)
        ws = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0.0,
        )
        wb = tl.load(
            Wbias + offs_r[:, None] * stride_wb0 + group_idx[None, :] * stride_wb1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0.0,
        )
        out = q.to(ws.dtype) * ws + wb
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )


@triton.jit
def qmm_dequant_affine8_kernel(
    Wq,
    Wscale,
    Wbias,
    N,
    K,
    O,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_wb0: tl.constexpr,
    stride_wb1: tl.constexpr,
    stride_or: tl.constexpr,
    stride_oc: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    """Dequantize a packed 8-bit affine-quantized weight tile to fp16/bf16.

    Grid: ``(cdiv(rows, BR), cdiv(cols, BC))``

    Unpacks ``VALUES_PER_WORD = 4`` 8-bit codes from each int32 word using an
    8-bit right-shift-and-mask pattern, then applies the per-group affine
    transform ``out = code * scale + bias``.  The output is cast to bfloat16
    if ``OUT_BF16=True``, otherwise float16.

    Args:
        Wq: Packed quantized weight pointer.  When ``TRANSPOSE=False`` the
            shape is ``(K // VALUES_PER_WORD, N)``; when ``TRANSPOSE=True``
            the shape is ``(N // VALUES_PER_WORD, K)``.
        Wscale: Per-group scale pointer, shape ``(rows, groups)`` where
            ``groups = cols // GROUP_SIZE``.
        Wbias: Per-group zero-point / bias pointer, same shape as ``Wscale``.
        N: Output column count (output feature dimension).
        K: Output row count (input feature / reduction dimension).
        O: Output buffer pointer in dequantised layout.
        stride_wq0: Row stride of ``Wq``.
        stride_wq1: Column stride of ``Wq``.
        stride_ws0: Row stride of ``Wscale``.
        stride_ws1: Column stride of ``Wscale``.
        stride_wb0: Row stride of ``Wbias``.
        stride_wb1: Column stride of ``Wbias``.
        stride_or: Row stride of ``O``.
        stride_oc: Column stride of ``O``.
        GROUP_SIZE: Number of values sharing one (scale, bias) pair (constexpr).
        VALUES_PER_WORD: Packed values per int32 word; equals 4 for 8-bit
            codes (constexpr).
        BR: Tile rows per CTA (constexpr).
        BC: Tile columns per CTA (constexpr).
        TRANSPOSE: Whether weight is stored transposed (NxK) vs (KxN) (constexpr).
        OUT_BF16: Output in bfloat16 when True, float16 when False (constexpr).
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_r = pid_r * BR + tl.arange(0, BR)
    offs_c = pid_c * BC + tl.arange(0, BC)

    if TRANSPOSE:
        r_mask = offs_r < N
        c_mask = offs_c < K
        word_offsets = offs_c // VALUES_PER_WORD
        word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
        shifts = (offs_c % VALUES_PER_WORD) * 8
        w_word = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = (w_word >> shifts[None, :]) & 0xFF
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(K, GROUP_SIZE)
        ws = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0.0,
        )
        wb = tl.load(
            Wbias + offs_r[:, None] * stride_wb0 + group_idx[None, :] * stride_wb1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0.0,
        )
        out = q.to(ws.dtype) * ws + wb
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )
    else:
        r_mask = offs_r < K
        c_mask = offs_c < N
        word_offsets = offs_c // VALUES_PER_WORD
        word_mask = word_offsets < tl.cdiv(N, VALUES_PER_WORD)
        shifts = (offs_c % VALUES_PER_WORD) * 8
        w_word = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = (w_word >> shifts[None, :]) & 0xFF
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(N, GROUP_SIZE)
        ws = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0.0,
        )
        wb = tl.load(
            Wbias + offs_r[:, None] * stride_wb0 + group_idx[None, :] * stride_wb1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0.0,
        )
        out = q.to(ws.dtype) * ws + wb
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )


@triton.jit
def qmm_dequant_mxfp4_kernel(
    Wq,
    Wscale,
    E2M1_TABLE,
    E8M0_TABLE,
    N,
    K,
    O,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_or: tl.constexpr,
    stride_oc: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    """Dequantize a packed MX-FP4 (E2M1) weight tile to fp16/bf16.

    Grid: ``(cdiv(rows, BR), cdiv(cols, BC))``

    Unpacks ``VALUES_PER_WORD = 8`` 4-bit E2M1 codes from each int32 word,
    then converts each code to float32 via ``E2M1_TABLE`` lookup and multiplies
    by the block-floating-point scale ``E8M0_TABLE[Wscale[...]]`` (an 8-bit
    exponent stored as a uint8 index into a precomputed exp2 table).

    Args:
        Wq: Packed quantized weight pointer.  When ``TRANSPOSE=False`` the
            shape is ``(K // VALUES_PER_WORD, N)``; when ``TRANSPOSE=True``
            the shape is ``(N // VALUES_PER_WORD, K)``.
        Wscale: Per-group E8M0 exponent code pointer, shape
            ``(rows, groups)`` where ``groups = cols // GROUP_SIZE``.
        E2M1_TABLE: Lookup table pointer for 16-entry E2M1 float32 values.
        E8M0_TABLE: Lookup table pointer mapping uint8 exponent codes to
            ``pow(2, exponent)`` float32 values (256 entries).
        N: Output column count (output feature dimension).
        K: Output row count (input feature / reduction dimension).
        O: Output buffer pointer in dequantised layout.
        stride_wq0: Row stride of ``Wq``.
        stride_wq1: Column stride of ``Wq``.
        stride_ws0: Row stride of ``Wscale``.
        stride_ws1: Column stride of ``Wscale``.
        stride_or: Row stride of ``O``.
        stride_oc: Column stride of ``O``.
        GROUP_SIZE: Number of values sharing one scale (constexpr).
        VALUES_PER_WORD: Packed values per int32 word; equals 8 for 4-bit
            codes (constexpr).
        BR: Tile rows per CTA (constexpr).
        BC: Tile columns per CTA (constexpr).
        TRANSPOSE: Whether weight is stored transposed (NxK) vs (KxN) (constexpr).
        OUT_BF16: Output in bfloat16 when True, float16 when False (constexpr).
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_r = pid_r * BR + tl.arange(0, BR)
    offs_c = pid_c * BC + tl.arange(0, BC)

    if TRANSPOSE:
        r_mask = offs_r < N
        c_mask = offs_c < K
        word_offsets = offs_c // VALUES_PER_WORD
        word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
        shifts = (offs_c % VALUES_PER_WORD) * 4
        w_word = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = (w_word >> shifts[None, :]) & 0xF
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(K, GROUP_SIZE)
        scale_codes = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0,
        ).to(tl.int32)
        scale = tl.load(E8M0_TABLE + scale_codes)
        out = _e2m1_to_f32(q.to(tl.int32), E2M1_TABLE) * scale
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )
    else:
        r_mask = offs_r < K
        c_mask = offs_c < N
        word_offsets = offs_c // VALUES_PER_WORD
        word_mask = word_offsets < tl.cdiv(N, VALUES_PER_WORD)
        shifts = (offs_c % VALUES_PER_WORD) * 4
        w_word = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = (w_word >> shifts[None, :]) & 0xF
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(N, GROUP_SIZE)
        scale_codes = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0,
        ).to(tl.int32)
        scale = tl.load(E8M0_TABLE + scale_codes)
        out = _e2m1_to_f32(q.to(tl.int32), E2M1_TABLE) * scale
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )


@triton.jit
def qmm_dequant_mxfp8_kernel(
    Wq,
    Wscale,
    E4M3_TABLE,
    E8M0_TABLE,
    N,
    K,
    O,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_or: tl.constexpr,
    stride_oc: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    """Dequantize a packed MX-FP8 (E4M3) weight tile to fp16/bf16.

    Grid: ``(cdiv(rows, BR), cdiv(cols, BC))``

    Unpacks ``VALUES_PER_WORD = 4`` 8-bit E4M3 codes from each int32 word,
    converts each code to float32 via ``E4M3_TABLE`` lookup, and multiplies
    by the block-floating-point scale ``E8M0_TABLE[Wscale[...]]``.

    Args:
        Wq: Packed quantized weight pointer.  When ``TRANSPOSE=False`` the
            shape is ``(K // VALUES_PER_WORD, N)``; when ``TRANSPOSE=True``
            the shape is ``(N // VALUES_PER_WORD, K)``.
        Wscale: Per-group E8M0 exponent code pointer, shape
            ``(rows, groups)`` where ``groups = cols // GROUP_SIZE``.
        E4M3_TABLE: Lookup table pointer for 256-entry E4M3 float32 values.
        E8M0_TABLE: Lookup table pointer mapping uint8 exponent codes to
            ``pow(2, exponent)`` float32 values (256 entries).
        N: Output column count (output feature dimension).
        K: Output row count (input feature / reduction dimension).
        O: Output buffer pointer in dequantised layout.
        stride_wq0: Row stride of ``Wq``.
        stride_wq1: Column stride of ``Wq``.
        stride_ws0: Row stride of ``Wscale``.
        stride_ws1: Column stride of ``Wscale``.
        stride_or: Row stride of ``O``.
        stride_oc: Column stride of ``O``.
        GROUP_SIZE: Number of values sharing one scale (constexpr).
        VALUES_PER_WORD: Packed values per int32 word; equals 4 for 8-bit
            codes (constexpr).
        BR: Tile rows per CTA (constexpr).
        BC: Tile columns per CTA (constexpr).
        TRANSPOSE: Whether weight is stored transposed (NxK) vs (KxN) (constexpr).
        OUT_BF16: Output in bfloat16 when True, float16 when False (constexpr).
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_r = pid_r * BR + tl.arange(0, BR)
    offs_c = pid_c * BC + tl.arange(0, BC)

    if TRANSPOSE:
        r_mask = offs_r < N
        c_mask = offs_c < K
        word_offsets = offs_c // VALUES_PER_WORD
        word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
        shifts = (offs_c % VALUES_PER_WORD) * 8
        w_word = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = (w_word >> shifts[None, :]) & 0xFF
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(K, GROUP_SIZE)
        scale_codes = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0,
        ).to(tl.int32)
        scale = tl.load(E8M0_TABLE + scale_codes)
        out = _e4m3_to_f32(q.to(tl.int32), E4M3_TABLE) * scale
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )
    else:
        r_mask = offs_r < K
        c_mask = offs_c < N
        word_offsets = offs_c // VALUES_PER_WORD
        word_mask = word_offsets < tl.cdiv(N, VALUES_PER_WORD)
        shifts = (offs_c % VALUES_PER_WORD) * 8
        w_word = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = (w_word >> shifts[None, :]) & 0xFF
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(N, GROUP_SIZE)
        scale_codes = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0,
        ).to(tl.int32)
        scale = tl.load(E8M0_TABLE + scale_codes)
        out = _e4m3_to_f32(q.to(tl.int32), E4M3_TABLE) * scale
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )


@triton.jit
def qmm_dequant_nvfp4_kernel(
    Wq,
    Wscale,
    E2M1_TABLE,
    E4M3_TABLE,
    N,
    K,
    O,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_or: tl.constexpr,
    stride_oc: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    """Dequantize a packed NV-FP4 (E2M1) weight tile to fp16/bf16.

    Grid: ``(cdiv(rows, BR), cdiv(cols, BC))``

    This is the NVIDIA FP4 variant (distinct from MX-FP4): the per-group
    scale is stored as an E4M3 floating-point code rather than a raw E8M0
    exponent, enabling finer-grained scale representation.

    Unpacks ``VALUES_PER_WORD = 8`` 4-bit E2M1 codes from each int32 word,
    converts each code to float32 via ``E2M1_TABLE`` lookup, and multiplies
    by the per-group scale ``E4M3_TABLE[Wscale[...]]``.

    Args:
        Wq: Packed quantized weight pointer.  When ``TRANSPOSE=False`` the
            shape is ``(K // VALUES_PER_WORD, N)``; when ``TRANSPOSE=True``
            the shape is ``(N // VALUES_PER_WORD, K)``.
        Wscale: Per-group E4M3 scale code pointer, shape
            ``(rows, groups)`` where ``groups = cols // GROUP_SIZE``.
        E2M1_TABLE: Lookup table pointer for 16-entry E2M1 float32 values.
        E4M3_TABLE: Lookup table pointer for 256-entry E4M3 float32 values
            used to decode the per-group scales.
        N: Output column count (output feature dimension).
        K: Output row count (input feature / reduction dimension).
        O: Output buffer pointer in dequantised layout.
        stride_wq0: Row stride of ``Wq``.
        stride_wq1: Column stride of ``Wq``.
        stride_ws0: Row stride of ``Wscale``.
        stride_ws1: Column stride of ``Wscale``.
        stride_or: Row stride of ``O``.
        stride_oc: Column stride of ``O``.
        GROUP_SIZE: Number of values sharing one scale (constexpr).
        VALUES_PER_WORD: Packed values per int32 word; equals 8 for 4-bit
            codes (constexpr).
        BR: Tile rows per CTA (constexpr).
        BC: Tile columns per CTA (constexpr).
        TRANSPOSE: Whether weight is stored transposed (NxK) vs (KxN) (constexpr).
        OUT_BF16: Output in bfloat16 when True, float16 when False (constexpr).
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_r = pid_r * BR + tl.arange(0, BR)
    offs_c = pid_c * BC + tl.arange(0, BC)

    if TRANSPOSE:
        r_mask = offs_r < N
        c_mask = offs_c < K
        word_offsets = offs_c // VALUES_PER_WORD
        word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
        shifts = (offs_c % VALUES_PER_WORD) * 4
        w_word = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = (w_word >> shifts[None, :]) & 0xF
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(K, GROUP_SIZE)
        scale_codes = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0,
        ).to(tl.int32)
        scale = tl.load(E4M3_TABLE + scale_codes)
        out = _e2m1_to_f32(q.to(tl.int32), E2M1_TABLE) * scale
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )
    else:
        r_mask = offs_r < K
        c_mask = offs_c < N
        word_offsets = offs_c // VALUES_PER_WORD
        word_mask = word_offsets < tl.cdiv(N, VALUES_PER_WORD)
        shifts = (offs_c % VALUES_PER_WORD) * 4
        w_word = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = (w_word >> shifts[None, :]) & 0xF
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(N, GROUP_SIZE)
        scale_codes = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0,
        ).to(tl.int32)
        scale = tl.load(E4M3_TABLE + scale_codes)
        out = _e2m1_to_f32(q.to(tl.int32), E2M1_TABLE) * scale
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )


@triton.jit
def qmm_dequant_nvfp8_kernel(
    Wq,
    Wscale,
    E4M3_TABLE,
    N,
    K,
    O,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_or: tl.constexpr,
    stride_oc: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BR: tl.constexpr,
    BC: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    """Dequantize a packed NV-FP8 (E4M3) weight tile to fp16/bf16.

    Grid: ``(cdiv(rows, BR), cdiv(cols, BC))``

    This is the NVIDIA FP8 variant: the per-group scale is a single E4M3
    floating-point value decoded via ``E4M3_TABLE``.

    Unpacks ``VALUES_PER_WORD = 4`` 8-bit E4M3 codes from each int32 word,
    converts each code to float32 via ``E4M3_TABLE`` lookup, and multiplies
    by the per-group scale ``E4M3_TABLE[Wscale[...]]``.

    Args:
        Wq: Packed quantized weight pointer.  When ``TRANSPOSE=False`` the
            shape is ``(K // VALUES_PER_WORD, N)``; when ``TRANSPOSE=True``
            the shape is ``(N // VALUES_PER_WORD, K)``.
        Wscale: Per-group E4M3 scale code pointer, shape
            ``(rows, groups)`` where ``groups = cols // GROUP_SIZE``.
        E4M3_TABLE: Lookup table pointer for 256-entry E4M3 float32 values,
            used for both weight values and scale decoding.
        N: Output column count (output feature dimension).
        K: Output row count (input feature / reduction dimension).
        O: Output buffer pointer in dequantised layout.
        stride_wq0: Row stride of ``Wq``.
        stride_wq1: Column stride of ``Wq``.
        stride_ws0: Row stride of ``Wscale``.
        stride_ws1: Column stride of ``Wscale``.
        stride_or: Row stride of ``O``.
        stride_oc: Column stride of ``O``.
        GROUP_SIZE: Number of values sharing one scale (constexpr).
        VALUES_PER_WORD: Packed values per int32 word; equals 4 for 8-bit
            codes (constexpr).
        BR: Tile rows per CTA (constexpr).
        BC: Tile columns per CTA (constexpr).
        TRANSPOSE: Whether weight is stored transposed (NxK) vs (KxN) (constexpr).
        OUT_BF16: Output in bfloat16 when True, float16 when False (constexpr).
    """
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_r = pid_r * BR + tl.arange(0, BR)
    offs_c = pid_c * BC + tl.arange(0, BC)

    if TRANSPOSE:
        r_mask = offs_r < N
        c_mask = offs_c < K
        word_offsets = offs_c // VALUES_PER_WORD
        word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
        shifts = (offs_c % VALUES_PER_WORD) * 8
        w_word = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = (w_word >> shifts[None, :]) & 0xFF
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(K, GROUP_SIZE)
        scale_codes = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0,
        ).to(tl.int32)
        scale = tl.load(E4M3_TABLE + scale_codes)
        out = _e4m3_to_f32(q.to(tl.int32), E4M3_TABLE) * scale
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )
    else:
        r_mask = offs_r < K
        c_mask = offs_c < N
        word_offsets = offs_c // VALUES_PER_WORD
        word_mask = word_offsets < tl.cdiv(N, VALUES_PER_WORD)
        shifts = (offs_c % VALUES_PER_WORD) * 8
        w_word = tl.load(
            Wq + offs_r[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
            mask=r_mask[:, None] & word_mask[None, :],
            other=0,
        )
        q = (w_word >> shifts[None, :]) & 0xFF
        group_idx = offs_c // GROUP_SIZE
        group_mask = group_idx < tl.cdiv(N, GROUP_SIZE)
        scale_codes = tl.load(
            Wscale + offs_r[:, None] * stride_ws0 + group_idx[None, :] * stride_ws1,
            mask=r_mask[:, None] & group_mask[None, :],
            other=0,
        ).to(tl.int32)
        scale = tl.load(E4M3_TABLE + scale_codes)
        out = _e4m3_to_f32(q.to(tl.int32), E4M3_TABLE) * scale
        out_ty = tl.bfloat16 if OUT_BF16 else tl.float16
        tl.store(
            O + offs_r[:, None] * stride_or + offs_c[None, :] * stride_oc,
            out.to(out_ty),
            mask=r_mask[:, None] & c_mask[None, :],
        )


def _zeroed_outputs_for_splitk(meta: dict) -> tuple[int, ...]:
    """Return output indices that should be zeroed for split-K kernels.

    When using split-K parallelism, the output buffer must be zeroed before
    the kernel runs because partial results are accumulated via atomic_add.

    Args:
        meta: Kernel metadata containing SPLIT_K configuration.

    Returns:
        Tuple of output indices to zero, or empty tuple if SPLIT_K == 1.
    """
    return (0,) if meta["SPLIT_K"] > 1 else ()


def _env_flag(name: str, default: str = "0") -> bool:
    """Return ``True`` if the named environment variable is a truthy string."""
    value = os.getenv(name, default)
    return value.lower() in {"1", "true", "yes", "y"}


def _parse_positive_int_env(name: str, default: int) -> int:
    """Parse a positive integer env var with a safe fallback."""
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    return max(1, value)


def _parse_nonnegative_int_env(name: str, default: int) -> int:
    """Parse a non-negative integer env var with a safe fallback."""
    raw = os.getenv(name)
    if raw is None:
        return int(default)
    try:
        value = int(raw)
    except ValueError:
        return int(default)
    return max(0, value)


def _parse_matmul_precision(value: str):
    """Map an ``EJKERNEL_QMM_MATMUL_PRECISION`` string to a ``jax.lax.Precision``.

    Accepted values (case-insensitive): ``"highest"``, ``"high"``,
    ``"fastest"``.  Falls back to ``Precision.DEFAULT`` for unknown strings.
    """
    value = value.lower()
    if value == "highest":
        return jax.lax.Precision.HIGHEST
    if value == "high":
        return jax.lax.Precision.HIGH
    if value == "fastest":
        return jax.lax.Precision.FASTEST
    return jax.lax.Precision.DEFAULT


def _parse_output_dtype(value: str):
    """Map a dtype string to a ``jnp`` dtype, or ``None`` for unrecognised values.

    Accepted strings (case-insensitive): ``"bf16"``/``"bfloat16"``,
    ``"fp16"``/``"float16"``, ``"fp32"``/``"float32"``.
    """
    value = value.lower()
    if value in {"bf16", "bfloat16"}:
        return jnp.bfloat16
    if value in {"fp16", "float16"}:
        return jnp.float16
    if value in {"fp32", "float32"}:
        return jnp.float32
    return None


@lru_cache(maxsize=1)
def _cuda_max_shared_mem_per_block_bytes() -> int | None:
    """Best-effort query of CUDA's max shared memory per block (opt-in limit).

    JAX's Triton runtime will attempt to launch autotune candidates as-is; when
    a config requires more shared memory than the device allows by default,
    JAX logs messages like "Unable to launch autotune config on device".

    Prefer querying `cudaDevAttrMaxSharedMemoryPerBlockOptin` (attribute 97) so
    we do not over-prune on GPUs where the Triton launcher opts-in to the larger
    shared-memory limit. Fall back to `cudaDevAttrMaxSharedMemoryPerBlock`
    (attribute 8) when the opt-in attribute isn't available.
    """
    try:
        import ctypes
    except Exception:
        return None

    lib = None
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
        try:
            lib = ctypes.CDLL(name)
            break
        except OSError:
            continue
    if lib is None:
        return None

    cudaGetDevice = getattr(lib, "cudaGetDevice", None)
    cudaDeviceGetAttribute = getattr(lib, "cudaDeviceGetAttribute", None)
    if cudaGetDevice is None or cudaDeviceGetAttribute is None:
        return None

    cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cudaGetDevice.restype = ctypes.c_int
    cudaDeviceGetAttribute.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
        ctypes.c_int,
    ]
    cudaDeviceGetAttribute.restype = ctypes.c_int

    dev = ctypes.c_int()
    if int(cudaGetDevice(ctypes.byref(dev))) != 0:
        return None

    def _get_attr(attr_id: int) -> int | None:
        val = ctypes.c_int()
        if int(cudaDeviceGetAttribute(ctypes.byref(val), attr_id, int(dev.value))) != 0:
            return None
        out = int(val.value)
        return out if out > 0 else None

    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97
    cudaDevAttrMaxSharedMemoryPerBlock = 8
    return _get_attr(cudaDevAttrMaxSharedMemoryPerBlockOptin) or _get_attr(cudaDevAttrMaxSharedMemoryPerBlock)


@lru_cache(maxsize=1)
def _qmm_smem_limit_bytes() -> int:
    """Return an estimate of usable shared memory per CTA for QMM kernels.

    This helper is used while constructing module-level autotune config lists,
    so it must not query JAX runtime backend state (which would eagerly
    initialize backends at import time).
    """
    default = 96 * 1024

    limit = _cuda_max_shared_mem_per_block_bytes()
    if limit is None:
        return default

    return min(default, int(limit))


def _qmm_estimated_smem_bytes(*, bm: int, bn: int, bk: int, num_stages: int) -> int:
    return int((bm * bk + bk * bn) * 2 * num_stages)


@triton.jit
def qmm_nf4_kernel(
    X,
    Wq,
    Wscale,
    NF4_TABLE,
    M,
    N,
    K,
    O,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_BF16: tl.constexpr = True,
    TRANSPOSE: tl.constexpr = True,
):
    """Fused NF4 dequantization and matrix multiplication Triton kernel.

    Performs x @ dequant(w) where w is packed in NF4 (4-bit NormalFloat) format.
    Each 32-bit word contains 8 NF4 codes that are decoded via table lookup,
    scaled by per-group scale factors, and multiplied with the activation tile.

    Supports both transposed (NxK) and non-transposed (KxN) weight layouts.
    Uses split-K parallelism with atomic accumulation when SPLIT_K > 1.

    Args:
        X: Input activation matrix pointer, shape (M, K).
        Wq: Packed NF4 weights pointer (uint32, 8 values per word).
        Wscale: Per-group scale factors pointer.
        NF4_TABLE: NF4 codebook lookup table pointer (16 float32 entries).
        M, N, K: Matrix dimensions.
        O: Output matrix pointer, shape (M, N).
        stride_*: Tensor stride parameters.
        GROUP_SIZE: Number of elements per quantization group.
        VALUES_PER_WORD: Number of quantized values per uint32 word (8 for NF4).
        BM, BK, BN: Block tile sizes for M, K, N dimensions.
        SPLIT_K: Split-K parallelism factor.
        USE_BF16: If True, use BF16 for dot product tiles; otherwise FP16.
        TRANSPOSE: If True, weights are in NxK layout; otherwise KxN.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BM, BN), tl.float32)
    dot_ty = tl.bfloat16 if USE_BF16 else tl.float16

    mask_bits = 0xF
    shifts = tl.arange(0, VALUES_PER_WORD) * 4
    if not TRANSPOSE:
        word_offsets_n = (pid_n * BN) // VALUES_PER_WORD + tl.arange(0, BN // VALUES_PER_WORD)
        word_mask_n = word_offsets_n < tl.cdiv(N, VALUES_PER_WORD)
        group_idx_n = offs_n // GROUP_SIZE

    for k0 in tl.range(0, K, BK * SPLIT_K, loop_unroll_factor=1):
        offs_k = k0 + pid_k * BK + tl.arange(0, BK)
        k_mask = offs_k < K

        x = tl.load(
            X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(dot_ty)

        if TRANSPOSE:
            word_offsets = (k0 + pid_k * BK) // VALUES_PER_WORD + tl.arange(0, BK // VALUES_PER_WORD)
            word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
            w_word = tl.load(
                Wq + offs_n[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
                mask=n_mask[:, None] & word_mask[None, :],
                other=0,
            )
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BN, BK))
            q = tl.trans(q)
            group_idx_k = offs_k // GROUP_SIZE
            ws = tl.load(
                Wscale + offs_n[None, :] * stride_ws0 + group_idx_k[:, None] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
        else:
            w_word = tl.load(
                Wq + offs_k[:, None] * stride_wq0 + word_offsets_n[None, :] * stride_wq1,
                mask=k_mask[:, None] & word_mask_n[None, :],
                other=0,
            )
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BK, BN))
            ws = tl.load(
                Wscale + offs_k[:, None] * stride_ws0 + group_idx_n[None, :] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )

        w = _nf4_to_f32(q.to(tl.int32), NF4_TABLE).to(dot_ty) * ws.to(dot_ty)
        acc = tl.dot(x, w, acc)

    if SPLIT_K == 1:
        tl.store(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )
    else:
        tl.atomic_add(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )


qmm_nf4_kernel_large = qmm_nf4_kernel


@triton.jit
def qmm_affine8_kernel(
    X,
    Wq,
    Wscale,
    Wbias,
    M,
    N,
    K,
    O,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_wb0: tl.constexpr,
    stride_wb1: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_BF16: tl.constexpr = True,
    TRANSPOSE: tl.constexpr = True,
    HAS_BIAS: tl.constexpr = True,
):
    """Fused affine dequantization and matrix multiplication Triton kernel (8-bit)."""

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BM, BN), tl.float32)
    dot_ty = tl.bfloat16 if USE_BF16 else tl.float16

    mask_bits = 0xFF
    shifts = tl.arange(0, VALUES_PER_WORD) * 8
    if not TRANSPOSE:
        word_offsets_n = (pid_n * BN) // VALUES_PER_WORD + tl.arange(0, BN // VALUES_PER_WORD)
        word_mask_n = word_offsets_n < tl.cdiv(N, VALUES_PER_WORD)
        group_idx_n = offs_n // GROUP_SIZE

    for k0 in tl.range(0, K, BK * SPLIT_K, loop_unroll_factor=1):
        offs_k = k0 + pid_k * BK + tl.arange(0, BK)
        k_mask = offs_k < K

        x = tl.load(
            X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(dot_ty)

        if TRANSPOSE:
            word_offsets = (k0 + pid_k * BK) // VALUES_PER_WORD + tl.arange(0, BK // VALUES_PER_WORD)
            word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
            w_word = tl.load(
                Wq + offs_n[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
                mask=n_mask[:, None] & word_mask[None, :],
                other=0,
            )
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BN, BK))
            q = tl.trans(q)
            group_idx_k = offs_k // GROUP_SIZE
            ws = tl.load(
                Wscale + offs_n[None, :] * stride_ws0 + group_idx_k[:, None] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            if HAS_BIAS:
                wb = tl.load(
                    Wbias + offs_n[None, :] * stride_wb0 + group_idx_k[:, None] * stride_wb1,
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0.0,
                )
        else:
            w_word = tl.load(
                Wq + offs_k[:, None] * stride_wq0 + word_offsets_n[None, :] * stride_wq1,
                mask=k_mask[:, None] & word_mask_n[None, :],
                other=0,
            )
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BK, BN))
            ws = tl.load(
                Wscale + offs_k[:, None] * stride_ws0 + group_idx_n[None, :] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            if HAS_BIAS:
                wb = tl.load(
                    Wbias + offs_k[:, None] * stride_wb0 + group_idx_n[None, :] * stride_wb1,
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0.0,
                )

        w = q.to(dot_ty) * ws.to(dot_ty)
        if HAS_BIAS:
            w = w + wb.to(dot_ty)

        acc = tl.dot(x, w, acc)

    if SPLIT_K == 1:
        tl.store(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )
    else:
        tl.atomic_add(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )


qmm_affine8_kernel_large = qmm_affine8_kernel


@triton.jit
def qmm_affine4_kernel(
    X,
    Wq,
    Wscale,
    Wbias,
    M,
    N,
    K,
    O,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_wb0: tl.constexpr,
    stride_wb1: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BITS: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_BF16: tl.constexpr = True,
    TRANSPOSE: tl.constexpr = True,
    HAS_BIAS: tl.constexpr = True,
):
    """Fused affine dequantization and matrix multiplication Triton kernel.

    Performs x @ dequant(w) where w is packed in affine quantization format.
    Dequantization applies: w_float = w_int * scale + bias (when HAS_BIAS)
    or w_float = w_int * scale (when not HAS_BIAS).

    Supports affine bit-widths from 1 through 8 with per-group scale and bias
    factors. Quantized values are packed into a contiguous uint32 bitstream.

    Uses split-K parallelism with atomic accumulation when SPLIT_K > 1.

    Args:
        X: Input activation matrix pointer, shape (M, K).
        Wq: Packed quantized weights pointer (uint32).
        Wscale: Per-group scale factors pointer.
        Wbias: Per-group bias factors pointer (ignored if HAS_BIAS=False).
        M, N, K: Matrix dimensions.
        O: Output matrix pointer, shape (M, N).
        stride_*: Tensor stride parameters.
        GROUP_SIZE: Number of elements per quantization group.
        VALUES_PER_WORD: Legacy packed values-per-word hint.
        BITS: Bits per affine code, from 1 through 8.
        BM, BK, BN: Block tile sizes for M, K, N dimensions.
        SPLIT_K: Split-K parallelism factor.
        USE_BF16: If True, use BF16 for dot product tiles; otherwise FP16.
        TRANSPOSE: If True, weights are in NxK layout; otherwise KxN.
        HAS_BIAS: If True, apply per-group bias during dequantization.
    """

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BM, BN), tl.float32)
    dot_ty = tl.bfloat16 if USE_BF16 else tl.float16

    if not TRANSPOSE:
        if BITS == 1 or BITS == 2 or BITS == 4 or BITS == 8:
            word_offsets_n = (pid_n * BN) // VALUES_PER_WORD + tl.arange(0, BN // VALUES_PER_WORD)
            word_mask_n = word_offsets_n < tl.cdiv(N, VALUES_PER_WORD)
        else:
            bit_offsets_n = offs_n * BITS
            word_offsets_n = bit_offsets_n // 32
            shifts_n = bit_offsets_n - word_offsets_n * 32
            n_words = tl.cdiv(N * BITS, 32)
            word_mask_n = word_offsets_n < n_words
            word_offsets_n1 = tl.minimum(word_offsets_n + 1, n_words - 1)
        group_idx_n = offs_n // GROUP_SIZE

    for k0 in tl.range(0, K, BK * SPLIT_K, loop_unroll_factor=1):
        offs_k = k0 + pid_k * BK + tl.arange(0, BK)
        k_mask = offs_k < K

        x = tl.load(
            X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(dot_ty)

        if TRANSPOSE:
            if BITS == 1 or BITS == 2 or BITS == 4 or BITS == 8:
                mask_bits = (1 << BITS) - 1
                shifts = tl.arange(0, VALUES_PER_WORD) * BITS
                word_offsets = (k0 + pid_k * BK) // VALUES_PER_WORD + tl.arange(0, BK // VALUES_PER_WORD)
                word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
                w_word = tl.load(
                    Wq + offs_n[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
                    mask=n_mask[:, None] & word_mask[None, :],
                    other=0,
                )
                q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
                q = tl.reshape(q, (BN, BK))
                q = tl.trans(q)
            else:
                bit_offsets_k = offs_k * BITS
                word_offsets = bit_offsets_k // 32
                shifts_k = bit_offsets_k - word_offsets * 32
                n_words = tl.cdiv(K * BITS, 32)
                word_mask = word_offsets < n_words
                word_offsets1 = tl.minimum(word_offsets + 1, n_words - 1)
                w_word0 = tl.load(
                    Wq + offs_n[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
                    mask=n_mask[:, None] & word_mask[None, :],
                    other=0,
                )
                w_word1 = tl.load(
                    Wq + offs_n[:, None] * stride_wq0 + word_offsets1[None, :] * stride_wq1,
                    mask=n_mask[:, None] & word_mask[None, :],
                    other=0,
                )
                q = _unpack_packed_codes(w_word0, w_word1, shifts_k[None, :], BITS)
                q = tl.trans(q)
            group_idx_k = offs_k // GROUP_SIZE
            ws = tl.load(
                Wscale + offs_n[None, :] * stride_ws0 + group_idx_k[:, None] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            if HAS_BIAS:
                wb = tl.load(
                    Wbias + offs_n[None, :] * stride_wb0 + group_idx_k[:, None] * stride_wb1,
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0.0,
                )
        else:
            if BITS == 1 or BITS == 2 or BITS == 4 or BITS == 8:
                mask_bits = (1 << BITS) - 1
                shifts = tl.arange(0, VALUES_PER_WORD) * BITS
                w_word = tl.load(
                    Wq + offs_k[:, None] * stride_wq0 + word_offsets_n[None, :] * stride_wq1,
                    mask=k_mask[:, None] & word_mask_n[None, :],
                    other=0,
                )
                q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
                q = tl.reshape(q, (BK, BN))
            else:
                w_word0 = tl.load(
                    Wq + offs_k[:, None] * stride_wq0 + word_offsets_n[None, :] * stride_wq1,
                    mask=k_mask[:, None] & word_mask_n[None, :],
                    other=0,
                )
                w_word1 = tl.load(
                    Wq + offs_k[:, None] * stride_wq0 + word_offsets_n1[None, :] * stride_wq1,
                    mask=k_mask[:, None] & word_mask_n[None, :],
                    other=0,
                )
                q = _unpack_packed_codes(w_word0, w_word1, shifts_n[None, :], BITS)
            ws = tl.load(
                Wscale + offs_k[:, None] * stride_ws0 + group_idx_n[None, :] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0.0,
            )
            if HAS_BIAS:
                wb = tl.load(
                    Wbias + offs_k[:, None] * stride_wb0 + group_idx_n[None, :] * stride_wb1,
                    mask=k_mask[:, None] & n_mask[None, :],
                    other=0.0,
                )

        w = q.to(dot_ty) * ws.to(dot_ty)
        if HAS_BIAS:
            w = w + wb.to(dot_ty)

        acc = tl.dot(x, w, acc)

    if SPLIT_K == 1:
        tl.store(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )
    else:
        tl.atomic_add(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )


qmm_affine4_kernel_large = qmm_affine4_kernel


@triton.jit
def qmm_mxfp4_kernel(
    X,
    Wq,
    Wscale,
    E2M1_TABLE,
    E8M0_TABLE,
    M,
    N,
    K,
    O,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_BF16: tl.constexpr = True,
    TRANSPOSE: tl.constexpr = True,
):
    """Fused MXFP4 dequantization and matrix multiplication Triton kernel."""

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BM, BN), tl.float32)
    dot_ty = tl.bfloat16 if USE_BF16 else tl.float16

    mask_bits = 0xF
    shifts = tl.arange(0, VALUES_PER_WORD) * 4
    if not TRANSPOSE:
        word_offsets_n = (pid_n * BN) // VALUES_PER_WORD + tl.arange(0, BN // VALUES_PER_WORD)
        word_mask_n = word_offsets_n < tl.cdiv(N, VALUES_PER_WORD)
        group_idx_n = offs_n // GROUP_SIZE

    for k0 in tl.range(0, K, BK * SPLIT_K, loop_unroll_factor=1):
        offs_k = k0 + pid_k * BK + tl.arange(0, BK)
        k_mask = offs_k < K

        x = tl.load(
            X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(dot_ty)

        if TRANSPOSE:
            word_offsets = (k0 + pid_k * BK) // VALUES_PER_WORD + tl.arange(0, BK // VALUES_PER_WORD)
            word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
            w_word = tl.load(
                Wq + offs_n[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
                mask=n_mask[:, None] & word_mask[None, :],
                other=0,
            )
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BN, BK))
            q = tl.trans(q)
            group_idx_k = offs_k // GROUP_SIZE
            scale_codes = tl.load(
                Wscale + offs_n[None, :] * stride_ws0 + group_idx_k[:, None] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0,
            )
        else:
            w_word = tl.load(
                Wq + offs_k[:, None] * stride_wq0 + word_offsets_n[None, :] * stride_wq1,
                mask=k_mask[:, None] & word_mask_n[None, :],
                other=0,
            )
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BK, BN))
            scale_codes = tl.load(
                Wscale + offs_k[:, None] * stride_ws0 + group_idx_n[None, :] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0,
            )

        scale_codes = scale_codes.to(tl.int32)
        scale = tl.load(E8M0_TABLE + scale_codes)
        w = _e2m1_to_f32(q.to(tl.int32), E2M1_TABLE).to(dot_ty) * scale.to(dot_ty)
        acc = tl.dot(x, w, acc)

    if SPLIT_K == 1:
        tl.store(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )
    else:
        tl.atomic_add(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )


qmm_mxfp4_kernel_large = qmm_mxfp4_kernel


@triton.jit
def qmm_mxfp8_kernel(
    X,
    Wq,
    Wscale,
    E4M3_TABLE,
    E8M0_TABLE,
    M,
    N,
    K,
    O,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_BF16: tl.constexpr = True,
    TRANSPOSE: tl.constexpr = True,
):
    """Fused MXFP8 dequantization and matrix multiplication Triton kernel."""

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BM, BN), tl.float32)
    dot_ty = tl.bfloat16 if USE_BF16 else tl.float16

    mask_bits = 0xFF
    shifts = tl.arange(0, VALUES_PER_WORD) * 8
    if not TRANSPOSE:
        word_offsets_n = (pid_n * BN) // VALUES_PER_WORD + tl.arange(0, BN // VALUES_PER_WORD)
        word_mask_n = word_offsets_n < tl.cdiv(N, VALUES_PER_WORD)
        group_idx_n = offs_n // GROUP_SIZE

    for k0 in tl.range(0, K, BK * SPLIT_K, loop_unroll_factor=1):
        offs_k = k0 + pid_k * BK + tl.arange(0, BK)
        k_mask = offs_k < K

        x = tl.load(
            X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(dot_ty)

        if TRANSPOSE:
            word_offsets = (k0 + pid_k * BK) // VALUES_PER_WORD + tl.arange(0, BK // VALUES_PER_WORD)
            word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
            w_word = tl.load(
                Wq + offs_n[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
                mask=n_mask[:, None] & word_mask[None, :],
                other=0,
            )
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BN, BK))
            q = tl.trans(q)
            group_idx_k = offs_k // GROUP_SIZE
            scale_codes = tl.load(
                Wscale + offs_n[None, :] * stride_ws0 + group_idx_k[:, None] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0,
            )
        else:
            w_word = tl.load(
                Wq + offs_k[:, None] * stride_wq0 + word_offsets_n[None, :] * stride_wq1,
                mask=k_mask[:, None] & word_mask_n[None, :],
                other=0,
            )
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BK, BN))
            scale_codes = tl.load(
                Wscale + offs_k[:, None] * stride_ws0 + group_idx_n[None, :] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0,
            )

        scale_codes = scale_codes.to(tl.int32)
        scale = tl.load(E8M0_TABLE + scale_codes)
        w = _e4m3_to_f32(q.to(tl.int32), E4M3_TABLE).to(dot_ty) * scale.to(dot_ty)
        acc = tl.dot(x, w, acc)

    if SPLIT_K == 1:
        tl.store(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )
    else:
        tl.atomic_add(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )


qmm_mxfp8_kernel_large = qmm_mxfp8_kernel


@triton.jit
def qmm_nvfp4_kernel(
    X,
    Wq,
    Wscale,
    E2M1_TABLE,
    E4M3_TABLE,
    M,
    N,
    K,
    O,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_BF16: tl.constexpr = True,
    TRANSPOSE: tl.constexpr = True,
):
    """Fused NVFP4 dequantization and matrix multiplication Triton kernel."""

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BM, BN), tl.float32)
    dot_ty = tl.bfloat16 if USE_BF16 else tl.float16

    mask_bits = 0xF
    shifts = tl.arange(0, VALUES_PER_WORD) * 4
    if not TRANSPOSE:
        word_offsets_n = (pid_n * BN) // VALUES_PER_WORD + tl.arange(0, BN // VALUES_PER_WORD)
        word_mask_n = word_offsets_n < tl.cdiv(N, VALUES_PER_WORD)
        group_idx_n = offs_n // GROUP_SIZE

    for k0 in tl.range(0, K, BK * SPLIT_K, loop_unroll_factor=1):
        offs_k = k0 + pid_k * BK + tl.arange(0, BK)
        k_mask = offs_k < K

        x = tl.load(
            X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(dot_ty)

        if TRANSPOSE:
            word_offsets = (k0 + pid_k * BK) // VALUES_PER_WORD + tl.arange(0, BK // VALUES_PER_WORD)
            word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
            w_word = tl.load(
                Wq + offs_n[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
                mask=n_mask[:, None] & word_mask[None, :],
                other=0,
            )
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BN, BK))
            q = tl.trans(q)
            group_idx_k = offs_k // GROUP_SIZE
            scale_codes = tl.load(
                Wscale + offs_n[None, :] * stride_ws0 + group_idx_k[:, None] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0,
            )
        else:
            w_word = tl.load(
                Wq + offs_k[:, None] * stride_wq0 + word_offsets_n[None, :] * stride_wq1,
                mask=k_mask[:, None] & word_mask_n[None, :],
                other=0,
            )
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BK, BN))
            scale_codes = tl.load(
                Wscale + offs_k[:, None] * stride_ws0 + group_idx_n[None, :] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0,
            )

        scale_codes = scale_codes.to(tl.int32)
        scale = tl.load(E4M3_TABLE + scale_codes)
        w = _e2m1_to_f32(q.to(tl.int32), E2M1_TABLE).to(dot_ty) * scale.to(dot_ty)
        acc = tl.dot(x, w, acc)

    if SPLIT_K == 1:
        tl.store(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )
    else:
        tl.atomic_add(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )


qmm_nvfp4_kernel_large = qmm_nvfp4_kernel


@triton.jit
def qmm_nvfp8_kernel(
    X,
    Wq,
    Wscale,
    E4M3_TABLE,
    M,
    N,
    K,
    O,
    stride_xm: tl.constexpr,
    stride_xk: tl.constexpr,
    stride_wq0: tl.constexpr,
    stride_wq1: tl.constexpr,
    stride_ws0: tl.constexpr,
    stride_ws1: tl.constexpr,
    stride_om: tl.constexpr,
    stride_on: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    VALUES_PER_WORD: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    BN: tl.constexpr,
    SPLIT_K: tl.constexpr,
    USE_BF16: tl.constexpr = True,
    TRANSPOSE: tl.constexpr = True,
):
    """Fused NVFP8 dequantization and matrix multiplication Triton kernel."""

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)

    m_mask = offs_m < M
    n_mask = offs_n < N

    acc = tl.zeros((BM, BN), tl.float32)
    dot_ty = tl.bfloat16 if USE_BF16 else tl.float16

    mask_bits = 0xFF
    shifts = tl.arange(0, VALUES_PER_WORD) * 8
    if not TRANSPOSE:
        word_offsets_n = (pid_n * BN) // VALUES_PER_WORD + tl.arange(0, BN // VALUES_PER_WORD)
        word_mask_n = word_offsets_n < tl.cdiv(N, VALUES_PER_WORD)
        group_idx_n = offs_n // GROUP_SIZE

    for k0 in tl.range(0, K, BK * SPLIT_K, loop_unroll_factor=1):
        offs_k = k0 + pid_k * BK + tl.arange(0, BK)
        k_mask = offs_k < K

        x = tl.load(
            X + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(dot_ty)

        if TRANSPOSE:
            word_offsets = (k0 + pid_k * BK) // VALUES_PER_WORD + tl.arange(0, BK // VALUES_PER_WORD)
            word_mask = word_offsets < tl.cdiv(K, VALUES_PER_WORD)
            w_word = tl.load(
                Wq + offs_n[:, None] * stride_wq0 + word_offsets[None, :] * stride_wq1,
                mask=n_mask[:, None] & word_mask[None, :],
                other=0,
            )
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BN, BK))
            q = tl.trans(q)
            group_idx_k = offs_k // GROUP_SIZE
            scale_codes = tl.load(
                Wscale + offs_n[None, :] * stride_ws0 + group_idx_k[:, None] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0,
            )
        else:
            w_word = tl.load(
                Wq + offs_k[:, None] * stride_wq0 + word_offsets_n[None, :] * stride_wq1,
                mask=k_mask[:, None] & word_mask_n[None, :],
                other=0,
            )
            q = (w_word[:, :, None] >> shifts[None, None, :]) & mask_bits
            q = tl.reshape(q, (BK, BN))
            scale_codes = tl.load(
                Wscale + offs_k[:, None] * stride_ws0 + group_idx_n[None, :] * stride_ws1,
                mask=k_mask[:, None] & n_mask[None, :],
                other=0,
            )

        scale_codes = scale_codes.to(tl.int32)
        scale = tl.load(E4M3_TABLE + scale_codes)
        w = _e4m3_to_f32(q.to(tl.int32), E4M3_TABLE).to(dot_ty) * scale.to(dot_ty)
        acc = tl.dot(x, w, acc)

    if SPLIT_K == 1:
        tl.store(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )
    else:
        tl.atomic_add(
            O + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
            acc,
            mask=m_mask[:, None] & n_mask[None, :],
        )


qmm_nvfp8_kernel_large = qmm_nvfp8_kernel


def _resolve_qparams(mode: str, group_size: int | None, bits: int | None) -> tuple[int, int]:
    """Resolve and validate quantization parameters for Triton kernels.

    Applies mode-specific defaults and validates that the parameters are
    compatible with the Triton kernel implementations.

    Args:
        mode: Quantization mode ("affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8").
        group_size: Number of elements per quantization group, or None for default.
        bits: Bit-width per quantized element, or None for default.

    Returns:
        Tuple of (resolved_group_size, resolved_bits).

    Raises:
        ValueError: If mode is not supported by Triton kernels.
        ValueError: If affine bits are outside 1..8.
        ValueError: If bits != 4 for nf4 mode.
        ValueError: If group_size/bits mismatch for explicit MXFP/NVFP modes.
    """
    _, group_size, bits, _ = resolve_qparams(mode, group_size, bits)
    return int(group_size), int(bits)


def _validate_shapes(
    x: jax.Array,
    w: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None,
    *,
    transpose: bool,
    group_size: int,
    bits: int,
) -> tuple[int, int, int]:
    """Validate input array shapes and extract matrix dimensions.

    Performs shape validation to ensure all inputs are compatible and
    extracts the M, K, N dimensions for the matmul operation.

    Args:
        x: Input activation matrix of shape (M, K).
        w: Packed uint32 weights. Shape depends on transpose setting.
        scales: Per-group scales array.
        biases: Per-group affine additive offsets (optional).
        transpose: If True, weights are in NxK layout; if False, KxN layout.
        group_size: Number of elements per quantization group.
        bits: Bit-width per quantized element.

    Returns:
        Tuple of (M, K, N) dimensions for the matmul operation.

    Raises:
        ValueError: If any input is not 2D.
        ValueError: If packed weight shape doesn't match expected dimensions.
        ValueError: If scales/affine-offset shapes are inconsistent.
    """
    if x.ndim != 2 or w.ndim != 2 or scales.ndim != 2:
        raise ValueError("x, w, and scales must be 2D arrays.")
    if biases is not None and biases.ndim != 2:
        raise ValueError("biases must be 2D when provided.")

    M, K = x.shape
    if transpose:
        N = w.shape[0]
        words_expected = math.ceil(K * bits / 32)
        if w.shape[1] != words_expected:
            raise ValueError("Packed weight shape does not match K dimension.")
        if scales.shape[0] != N:
            raise ValueError("scales first dimension must match N when transpose=True.")
        groups_expected = K // group_size
        if scales.shape[1] != groups_expected:
            raise ValueError("scales second dimension must match K/group_size.")
        if biases is not None and biases.shape != scales.shape:
            raise ValueError("biases shape must match scales.")
    else:
        if w.shape[0] != K:
            raise ValueError("Packed weight first dimension must match K when transpose=False.")
        groups_expected = scales.shape[1]
        N = groups_expected * group_size
        words_expected = math.ceil(N * bits / 32)
        if w.shape[1] != words_expected:
            raise ValueError("Packed weight shape does not match N dimension.")
        if scales.shape[0] != K:
            raise ValueError("scales first dimension must match K when transpose=False.")
        if biases is not None and biases.shape != scales.shape:
            raise ValueError("biases shape must match scales.")

    return M, K, N


def _validate_weight_shapes(
    w: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None,
    *,
    transpose: bool,
    group_size: int,
    bits: int,
) -> tuple[int, int]:
    """Validate weight/scales shapes and return (K, N)."""
    if w.ndim != 2 or scales.ndim != 2:
        raise ValueError("w and scales must be 2D arrays.")
    if biases is not None and biases.ndim != 2:
        raise ValueError("biases must be 2D when provided.")

    if transpose:
        N = w.shape[0]
        K = scales.shape[1] * group_size
        words_expected = math.ceil(K * bits / 32)
        if w.shape[1] != words_expected:
            raise ValueError("Packed weight shape does not match K dimension.")
        if scales.shape[0] != N:
            raise ValueError("scales first dimension must match N when transpose=True.")
        if biases is not None and biases.shape != scales.shape:
            raise ValueError("biases shape must match scales.")
    else:
        K = w.shape[0]
        N = scales.shape[1] * group_size
        words_expected = math.ceil(N * bits / 32)
        if w.shape[1] != words_expected:
            raise ValueError("Packed weight shape does not match N dimension.")
        if scales.shape[0] != K:
            raise ValueError("scales first dimension must match K when transpose=False.")
        if biases is not None and biases.shape != scales.shape:
            raise ValueError("biases shape must match scales.")

    return K, N


_DEFAULT_SPLIT_K_WHEN_NONE: int = 1


def quantized_matmul_dequant_triton(
    w: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None = None,
    *,
    transpose: bool = False,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
    use_bf16: bool = True,
) -> jax.Array:
    """Dequantize packed weights into BF16/FP16 for two-stage matmul."""
    nf4_table, e2m1_table, e4m3_table, e8m0_exp2_table = _get_decode_tables()
    mode = mode.lower()
    group_size, bits = _resolve_qparams(mode, group_size, bits)

    if mode == "affine" and biases is None:
        raise ValueError("affine quantized_matmul requires affine metadata.")
    if mode != "affine" and biases is not None:
        raise ValueError("affine metadata must be None for non-affine modes.")
    K, N = _validate_weight_shapes(
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
    )

    stride_wq0, stride_wq1 = strides_from_shape(w.shape)
    stride_ws0, stride_ws1 = strides_from_shape(scales.shape)

    out_dtype = jnp.bfloat16 if use_bf16 else jnp.float16
    if transpose:
        deq_shape = (N, K)
        stride_or, stride_oc = strides_from_shape(deq_shape)
        r_dim, c_dim = N, K
    else:
        deq_shape = (K, N)
        stride_or, stride_oc = strides_from_shape(deq_shape)
        r_dim, c_dim = K, N

    br = 128
    bc = 128

    def grid(META):
        return (cdiv(r_dim, META["BR"]), cdiv(c_dim, META["BC"]))

    if mode == "nf4":
        (w_deq,) = triton_call(
            w,
            scales,
            nf4_table,
            N,
            K,
            out_shape=[jax.ShapeDtypeStruct(shape=deq_shape, dtype=out_dtype)],
            grid=grid,
            kernel=qmm_dequant_nf4_kernel,
            stride_wq0=stride_wq0,
            stride_wq1=stride_wq1,
            stride_ws0=stride_ws0,
            stride_ws1=stride_ws1,
            stride_or=stride_or,
            stride_oc=stride_oc,
            GROUP_SIZE=group_size,
            VALUES_PER_WORD=8,
            BR=br,
            BC=bc,
            TRANSPOSE=transpose,
            OUT_BF16=use_bf16,
        )
    elif mode == "affine":
        stride_wb0, stride_wb1 = strides_from_shape(biases.shape) if biases is not None else (0, 0)
        (w_deq,) = triton_call(
            w,
            scales,
            biases,
            N,
            K,
            out_shape=[jax.ShapeDtypeStruct(shape=deq_shape, dtype=out_dtype)],
            grid=grid,
            kernel=qmm_dequant_affine4_kernel,
            stride_wq0=stride_wq0,
            stride_wq1=stride_wq1,
            stride_ws0=stride_ws0,
            stride_ws1=stride_ws1,
            stride_wb0=stride_wb0,
            stride_wb1=stride_wb1,
            stride_or=stride_or,
            stride_oc=stride_oc,
            GROUP_SIZE=group_size,
            VALUES_PER_WORD=max(1, 32 // bits),
            BITS=bits,
            BR=br,
            BC=bc,
            TRANSPOSE=transpose,
            OUT_BF16=use_bf16,
        )
    elif mode == "mxfp4":
        (w_deq,) = triton_call(
            w,
            scales,
            e2m1_table,
            e8m0_exp2_table,
            N,
            K,
            out_shape=[jax.ShapeDtypeStruct(shape=deq_shape, dtype=out_dtype)],
            grid=grid,
            kernel=qmm_dequant_mxfp4_kernel,
            stride_wq0=stride_wq0,
            stride_wq1=stride_wq1,
            stride_ws0=stride_ws0,
            stride_ws1=stride_ws1,
            stride_or=stride_or,
            stride_oc=stride_oc,
            GROUP_SIZE=group_size,
            VALUES_PER_WORD=8,
            BR=br,
            BC=bc,
            TRANSPOSE=transpose,
            OUT_BF16=use_bf16,
        )
    elif mode == "mxfp8":
        (w_deq,) = triton_call(
            w,
            scales,
            e4m3_table,
            e8m0_exp2_table,
            N,
            K,
            out_shape=[jax.ShapeDtypeStruct(shape=deq_shape, dtype=out_dtype)],
            grid=grid,
            kernel=qmm_dequant_mxfp8_kernel,
            stride_wq0=stride_wq0,
            stride_wq1=stride_wq1,
            stride_ws0=stride_ws0,
            stride_ws1=stride_ws1,
            stride_or=stride_or,
            stride_oc=stride_oc,
            GROUP_SIZE=group_size,
            VALUES_PER_WORD=4,
            BR=br,
            BC=bc,
            TRANSPOSE=transpose,
            OUT_BF16=use_bf16,
        )
    elif mode == "nvfp4":
        (w_deq,) = triton_call(
            w,
            scales,
            e2m1_table,
            e4m3_table,
            N,
            K,
            out_shape=[jax.ShapeDtypeStruct(shape=deq_shape, dtype=out_dtype)],
            grid=grid,
            kernel=qmm_dequant_nvfp4_kernel,
            stride_wq0=stride_wq0,
            stride_wq1=stride_wq1,
            stride_ws0=stride_ws0,
            stride_ws1=stride_ws1,
            stride_or=stride_or,
            stride_oc=stride_oc,
            GROUP_SIZE=group_size,
            VALUES_PER_WORD=8,
            BR=br,
            BC=bc,
            TRANSPOSE=transpose,
            OUT_BF16=use_bf16,
        )
    elif mode == "nvfp8":
        (w_deq,) = triton_call(
            w,
            scales,
            e4m3_table,
            N,
            K,
            out_shape=[jax.ShapeDtypeStruct(shape=deq_shape, dtype=out_dtype)],
            grid=grid,
            kernel=qmm_dequant_nvfp8_kernel,
            stride_wq0=stride_wq0,
            stride_wq1=stride_wq1,
            stride_ws0=stride_ws0,
            stride_ws1=stride_ws1,
            stride_or=stride_or,
            stride_oc=stride_oc,
            GROUP_SIZE=group_size,
            VALUES_PER_WORD=4,
            BR=br,
            BC=bc,
            TRANSPOSE=transpose,
            OUT_BF16=use_bf16,
        )
    else:
        raise ValueError(f"Unsupported mode for two-stage path: {mode}")

    return w_deq


def quantized_matmul_triton(
    x: jax.Array,
    w: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None = None,
    *,
    transpose: bool = False,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
    use_bf16: bool = True,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    num_warps: int | None = None,
    num_stages: int | None = None,
    split_k: int | None = None,
    gemv_mode: GemvMode = "auto",
    revsplit_k: RevSplitKMode = "auto",
    revsplit_k_parts: int | None = None,
) -> jax.Array:
    """Execute quantized matmul using Triton GPU kernels.

    This is the core Triton implementation that dispatches to the
    appropriate quantization kernel based on the mode parameter. The
    kernels perform fused dequantization and matmul for optimal performance.

    Args:
        x: Input activation matrix of shape (M, K) in float dtype.
        w: Packed uint32 weights. For transpose=True, shape is
            (N, ceil(K * bits / 32)). For transpose=False, shape is
            (K, ceil(N * bits / 32)).
        scales: Per-group scales. Shape is (N, K//group_size) for
            transpose=True or (K, N//group_size) for transpose=False.
        biases: Per-group affine additive offsets (required for affine mode only). Must have
            the same shape as scales.
        transpose: If True, weights are stored in NxK layout and the kernel
            computes x @ w.T. If False, weights are in KxN layout and the
            kernel computes x @ w. Default is False.
        group_size: Number of elements per quantization group. If None,
            uses mode defaults (affine/nf4: 64, mxfp4/mxfp8: 32, nvfp4/nvfp8: 16).
        bits: Bit-width per quantized element. If None, uses mode defaults
            (affine/nf4/mxfp4/nvfp4: 4, mxfp8/nvfp8: 8).
        mode: Quantization mode. One of "affine", "nf4", "mxfp4", "mxfp8",
            "nvfp4", "nvfp8".
        use_bf16: If True, use BF16 for dot product input tiles.
            If False, use FP16. Default is True.

    Returns:
        Matrix multiplication result of shape (M, N) in float32.

    Raises:
        ValueError: If mode is "affine" but affine metadata is missing.
        ValueError: If mode is not "affine" but affine metadata is provided.
        ValueError: If bits/group_size are invalid for the selected mode.
        ValueError: If input shapes are invalid or inconsistent.
    """
    mode = mode.lower()
    group_size, bits = _resolve_qparams(mode, group_size, bits)
    gemv_mode = normalize_gemv_mode(gemv_mode)
    revsplit_k = normalize_revsplitk_mode(revsplit_k)
    revsplit_k_parts = normalize_revsplitk_parts(revsplit_k_parts)

    if use_bf16 and getattr(x, "dtype", None) == jnp.float16:
        use_bf16 = False

    if mode == "affine" and biases is None:
        raise ValueError("affine quantized_matmul requires affine metadata.")
    if mode != "affine" and biases is not None:
        raise ValueError("affine metadata must be None for non-affine modes.")

    M, K, N = _validate_shapes(
        x,
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
    )

    kernel_family, family_revsplit_parts = select_qmm_kernel_family(
        m=int(M),
        mode=mode,  # type: ignore[arg-type]
        bits=bits,
        gemv_mode=gemv_mode,
        revsplit_k=revsplit_k,
        revsplit_k_parts=revsplit_k_parts,
    )
    if kernel_family == "gemm":
        split_k_selected = 1
    elif kernel_family in ("gemm_splitk", "gemv_splitk"):
        if split_k is None:
            split_k_selected = _DEFAULT_SPLIT_K_WHEN_NONE
        else:
            split_k_selected = max(1, int(split_k))
    else:
        split_k_selected = 2 if family_revsplit_parts is None else int(family_revsplit_parts)

    if split_k_selected not in {1, 2, 4, 8, 16}:
        raise ValueError("split_k must be one of {1,2,4,8,16}.")

    if kernel_family in {"gemv_splitk", "gemv_revsplitk"}:
        return quantized_matmul_triton_gemv(
            x,
            w,
            scales,
            biases,
            transpose=transpose,
            group_size=group_size,
            bits=bits,
            mode=mode,  # type: ignore[arg-type]
            kernel_family=kernel_family,  # type: ignore[arg-type]
            split_k=split_k_selected,
            revsplit_parts=family_revsplit_parts,
            block_n=block_n,
        )

    nf4_table, e2m1_table, e4m3_table, e8m0_exp2_table = _get_decode_tables()

    stride_xm, stride_xk = strides_from_shape(x.shape)
    stride_wq0, stride_wq1 = strides_from_shape(w.shape)
    stride_ws0, stride_ws1 = strides_from_shape(scales.shape)
    stride_om, stride_on = strides_from_shape((M, N))

    num_warps = int(num_warps) if num_warps is not None else 4
    num_stages = int(num_stages) if num_stages is not None else 3

    use_large_kernel = M >= 4096 and N >= 4096 and K >= 4096 and kernel_family in {"gemm", "gemm_splitk"}
    use_two_stage = _env_flag("EJKERNEL_QMM_TWO_STAGE", "1") and use_large_kernel

    if use_two_stage:
        out_dtype = jnp.bfloat16 if use_bf16 else jnp.float16
        output_dtype = jnp.bfloat16
        precision_env = os.getenv("EJKERNEL_QMM_MATMUL_PRECISION", "")
        if precision_env:
            matmul_precision = _parse_matmul_precision(precision_env)
        else:
            max_dim = max(M, N, K)
            if max_dim <= 2048:
                matmul_precision = jax.lax.Precision.FASTEST
            elif max_dim <= 4096:
                matmul_precision = jax.lax.Precision.HIGH
            else:
                matmul_precision = jax.lax.Precision.DEFAULT

        if transpose:
            deq_shape = (N, K)
            stride_or, stride_oc = strides_from_shape(deq_shape)
            r_dim, c_dim = N, K
        else:
            deq_shape = (K, N)
            stride_or, stride_oc = strides_from_shape(deq_shape)
            r_dim, c_dim = K, N

        br = 128
        bc = 128

        def grid(META):
            return (cdiv(r_dim, META["BR"]), cdiv(c_dim, META["BC"]))

        if mode == "nf4":
            (w_deq,) = triton_call(
                w,
                scales,
                nf4_table,
                N,
                K,
                out_shape=[jax.ShapeDtypeStruct(shape=deq_shape, dtype=out_dtype)],
                grid=grid,
                kernel=qmm_dequant_nf4_kernel,
                stride_wq0=stride_wq0,
                stride_wq1=stride_wq1,
                stride_ws0=stride_ws0,
                stride_ws1=stride_ws1,
                stride_or=stride_or,
                stride_oc=stride_oc,
                GROUP_SIZE=group_size,
                VALUES_PER_WORD=8,
                BR=br,
                BC=bc,
                TRANSPOSE=transpose,
                OUT_BF16=use_bf16,
            )
        elif mode == "affine":
            stride_wb0, stride_wb1 = strides_from_shape(biases.shape) if biases is not None else (0, 0)
            (w_deq,) = triton_call(
                w,
                scales,
                biases,
                N,
                K,
                out_shape=[jax.ShapeDtypeStruct(shape=deq_shape, dtype=out_dtype)],
                grid=grid,
                kernel=qmm_dequant_affine4_kernel,
                stride_wq0=stride_wq0,
                stride_wq1=stride_wq1,
                stride_ws0=stride_ws0,
                stride_ws1=stride_ws1,
                stride_wb0=stride_wb0,
                stride_wb1=stride_wb1,
                stride_or=stride_or,
                stride_oc=stride_oc,
                GROUP_SIZE=group_size,
                VALUES_PER_WORD=max(1, 32 // bits),
                BITS=bits,
                BR=br,
                BC=bc,
                TRANSPOSE=transpose,
                OUT_BF16=use_bf16,
            )
        elif mode == "mxfp4":
            (w_deq,) = triton_call(
                w,
                scales,
                e2m1_table,
                e8m0_exp2_table,
                N,
                K,
                out_shape=[jax.ShapeDtypeStruct(shape=deq_shape, dtype=out_dtype)],
                grid=grid,
                kernel=qmm_dequant_mxfp4_kernel,
                stride_wq0=stride_wq0,
                stride_wq1=stride_wq1,
                stride_ws0=stride_ws0,
                stride_ws1=stride_ws1,
                stride_or=stride_or,
                stride_oc=stride_oc,
                GROUP_SIZE=group_size,
                VALUES_PER_WORD=8,
                BR=br,
                BC=bc,
                TRANSPOSE=transpose,
                OUT_BF16=use_bf16,
            )
        elif mode == "mxfp8":
            (w_deq,) = triton_call(
                w,
                scales,
                e4m3_table,
                e8m0_exp2_table,
                N,
                K,
                out_shape=[jax.ShapeDtypeStruct(shape=deq_shape, dtype=out_dtype)],
                grid=grid,
                kernel=qmm_dequant_mxfp8_kernel,
                stride_wq0=stride_wq0,
                stride_wq1=stride_wq1,
                stride_ws0=stride_ws0,
                stride_ws1=stride_ws1,
                stride_or=stride_or,
                stride_oc=stride_oc,
                GROUP_SIZE=group_size,
                VALUES_PER_WORD=4,
                BR=br,
                BC=bc,
                TRANSPOSE=transpose,
                OUT_BF16=use_bf16,
            )
        elif mode == "nvfp4":
            (w_deq,) = triton_call(
                w,
                scales,
                e2m1_table,
                e4m3_table,
                N,
                K,
                out_shape=[jax.ShapeDtypeStruct(shape=deq_shape, dtype=out_dtype)],
                grid=grid,
                kernel=qmm_dequant_nvfp4_kernel,
                stride_wq0=stride_wq0,
                stride_wq1=stride_wq1,
                stride_ws0=stride_ws0,
                stride_ws1=stride_ws1,
                stride_or=stride_or,
                stride_oc=stride_oc,
                GROUP_SIZE=group_size,
                VALUES_PER_WORD=8,
                BR=br,
                BC=bc,
                TRANSPOSE=transpose,
                OUT_BF16=use_bf16,
            )
        elif mode == "nvfp8":
            (w_deq,) = triton_call(
                w,
                scales,
                e4m3_table,
                N,
                K,
                out_shape=[jax.ShapeDtypeStruct(shape=deq_shape, dtype=out_dtype)],
                grid=grid,
                kernel=qmm_dequant_nvfp8_kernel,
                stride_wq0=stride_wq0,
                stride_wq1=stride_wq1,
                stride_ws0=stride_ws0,
                stride_ws1=stride_ws1,
                stride_or=stride_or,
                stride_oc=stride_oc,
                GROUP_SIZE=group_size,
                VALUES_PER_WORD=4,
                BR=br,
                BC=bc,
                TRANSPOSE=transpose,
                OUT_BF16=use_bf16,
            )
        else:
            raise ValueError(f"Unsupported mode for two-stage path: {mode}")

        x_cast = x.astype(out_dtype)
        if transpose:
            dimension_numbers = (((1,), (1,)), ((), ()))
        else:
            dimension_numbers = (((1,), (0,)), ((), ()))
        out = jax.lax.dot_general(
            x_cast,
            w_deq,
            dimension_numbers=dimension_numbers,
            precision=matmul_precision,
            preferred_element_type=output_dtype,
        )
        return out.astype(jnp.bfloat16)

    if mode == "nf4":
        kernel = qmm_nf4_kernel_large if use_large_kernel else qmm_nf4_kernel
        (out,) = triton_call(
            x,
            w,
            scales,
            nf4_table,
            M,
            N,
            K,
            out_shape=[jax.ShapeDtypeStruct(shape=(M, N), dtype=jnp.float32)],
            grid=lambda META: (
                cdiv(M, META["BM"]),
                cdiv(N, META["BN"]),
                META["SPLIT_K"],
            ),
            kernel=kernel,
            zeroed_outputs=_zeroed_outputs_for_splitk,
            num_warps=num_warps,
            num_stages=num_stages,
            stride_xm=stride_xm,
            stride_xk=stride_xk,
            stride_wq0=stride_wq0,
            stride_wq1=stride_wq1,
            stride_ws0=stride_ws0,
            stride_ws1=stride_ws1,
            stride_om=stride_om,
            stride_on=stride_on,
            GROUP_SIZE=group_size,
            VALUES_PER_WORD=8,
            USE_BF16=use_bf16,
            TRANSPOSE=transpose,
            BM=block_m,
            BN=block_n,
            BK=block_k,
            SPLIT_K=split_k_selected,
        )
        return out.astype(jnp.bfloat16)
    if mode == "affine":
        stride_wb0, stride_wb1 = strides_from_shape(biases.shape) if biases is not None else (0, 0)
        bias_arg = biases if biases is not None else scales
        kernel = qmm_affine4_kernel_large if use_large_kernel else qmm_affine4_kernel

        (out,) = triton_call(
            x,
            w,
            scales,
            bias_arg,
            M,
            N,
            K,
            out_shape=[jax.ShapeDtypeStruct(shape=(M, N), dtype=jnp.float32)],
            grid=lambda META: (cdiv(M, META["BM"]), cdiv(N, META["BN"]), META["SPLIT_K"]),
            kernel=kernel,
            zeroed_outputs=_zeroed_outputs_for_splitk,
            num_warps=num_warps,
            num_stages=num_stages,
            stride_xm=stride_xm,
            stride_xk=stride_xk,
            stride_wq0=stride_wq0,
            stride_wq1=stride_wq1,
            stride_ws0=stride_ws0,
            stride_ws1=stride_ws1,
            stride_wb0=stride_wb0,
            stride_wb1=stride_wb1,
            stride_om=stride_om,
            stride_on=stride_on,
            GROUP_SIZE=group_size,
            VALUES_PER_WORD=max(1, 32 // bits),
            BITS=bits,
            USE_BF16=use_bf16,
            TRANSPOSE=transpose,
            HAS_BIAS=biases is not None,
            BM=block_m,
            BN=block_n,
            BK=block_k,
            SPLIT_K=split_k_selected,
        )
        return out.astype(jnp.bfloat16)

    if mode == "mxfp4":
        kernel = qmm_mxfp4_kernel_large if use_large_kernel else qmm_mxfp4_kernel
        (out,) = triton_call(
            x,
            w,
            scales,
            e2m1_table,
            e8m0_exp2_table,
            M,
            N,
            K,
            out_shape=[jax.ShapeDtypeStruct(shape=(M, N), dtype=jnp.float32)],
            grid=lambda META: (cdiv(M, META["BM"]), cdiv(N, META["BN"]), META["SPLIT_K"]),
            kernel=kernel,
            zeroed_outputs=_zeroed_outputs_for_splitk,
            num_warps=num_warps,
            num_stages=num_stages,
            stride_xm=stride_xm,
            stride_xk=stride_xk,
            stride_wq0=stride_wq0,
            stride_wq1=stride_wq1,
            stride_ws0=stride_ws0,
            stride_ws1=stride_ws1,
            stride_om=stride_om,
            stride_on=stride_on,
            GROUP_SIZE=group_size,
            VALUES_PER_WORD=8,
            USE_BF16=use_bf16,
            TRANSPOSE=transpose,
            BM=block_m,
            BN=block_n,
            BK=block_k,
            SPLIT_K=split_k_selected,
        )
        return out.astype(jnp.bfloat16)

    if mode == "mxfp8":
        kernel = qmm_mxfp8_kernel_large if use_large_kernel else qmm_mxfp8_kernel
        (out,) = triton_call(
            x,
            w,
            scales,
            e4m3_table,
            e8m0_exp2_table,
            M,
            N,
            K,
            out_shape=[jax.ShapeDtypeStruct(shape=(M, N), dtype=jnp.float32)],
            grid=lambda META: (cdiv(M, META["BM"]), cdiv(N, META["BN"]), META["SPLIT_K"]),
            kernel=kernel,
            zeroed_outputs=_zeroed_outputs_for_splitk,
            num_warps=num_warps,
            num_stages=num_stages,
            stride_xm=stride_xm,
            stride_xk=stride_xk,
            stride_wq0=stride_wq0,
            stride_wq1=stride_wq1,
            stride_ws0=stride_ws0,
            stride_ws1=stride_ws1,
            stride_om=stride_om,
            stride_on=stride_on,
            GROUP_SIZE=group_size,
            VALUES_PER_WORD=4,
            USE_BF16=use_bf16,
            TRANSPOSE=transpose,
            BM=block_m,
            BN=block_n,
            BK=block_k,
            SPLIT_K=split_k_selected,
        )
        return out.astype(jnp.bfloat16)

    if mode == "nvfp4":
        kernel = qmm_nvfp4_kernel_large if use_large_kernel else qmm_nvfp4_kernel
        (out,) = triton_call(
            x,
            w,
            scales,
            e2m1_table,
            e4m3_table,
            M,
            N,
            K,
            out_shape=[jax.ShapeDtypeStruct(shape=(M, N), dtype=jnp.float32)],
            grid=lambda META: (cdiv(M, META["BM"]), cdiv(N, META["BN"]), META["SPLIT_K"]),
            kernel=kernel,
            zeroed_outputs=_zeroed_outputs_for_splitk,
            num_warps=num_warps,
            num_stages=num_stages,
            stride_xm=stride_xm,
            stride_xk=stride_xk,
            stride_wq0=stride_wq0,
            stride_wq1=stride_wq1,
            stride_ws0=stride_ws0,
            stride_ws1=stride_ws1,
            stride_om=stride_om,
            stride_on=stride_on,
            GROUP_SIZE=group_size,
            VALUES_PER_WORD=8,
            USE_BF16=use_bf16,
            TRANSPOSE=transpose,
            BM=block_m,
            BN=block_n,
            BK=block_k,
            SPLIT_K=split_k_selected,
        )
        return out.astype(jnp.bfloat16)

    if mode == "nvfp8":
        kernel = qmm_nvfp8_kernel_large if use_large_kernel else qmm_nvfp8_kernel
        (out,) = triton_call(
            x,
            w,
            scales,
            e4m3_table,
            M,
            N,
            K,
            out_shape=[jax.ShapeDtypeStruct(shape=(M, N), dtype=jnp.float32)],
            grid=lambda META: (cdiv(M, META["BM"]), cdiv(N, META["BN"]), META["SPLIT_K"]),
            kernel=kernel,
            zeroed_outputs=_zeroed_outputs_for_splitk,
            num_warps=num_warps,
            num_stages=num_stages,
            stride_xm=stride_xm,
            stride_xk=stride_xk,
            stride_wq0=stride_wq0,
            stride_wq1=stride_wq1,
            stride_ws0=stride_ws0,
            stride_ws1=stride_ws1,
            stride_om=stride_om,
            stride_on=stride_on,
            GROUP_SIZE=group_size,
            VALUES_PER_WORD=4,
            USE_BF16=use_bf16,
            TRANSPOSE=transpose,
            BM=block_m,
            BN=block_n,
            BK=block_k,
            SPLIT_K=split_k_selected,
        )
        return out.astype(jnp.bfloat16)

    raise ValueError(f"Unsupported quantization mode for Triton: {mode}")
