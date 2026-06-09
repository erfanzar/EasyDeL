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

"""tile-lang prim_funcs for affine int8 quantized matmul (fwd + bwd-dx).

Layout (matches the simplest mode of
:func:`ejkernel.kernels._xla.quantized_matmul`):

* ``x``: ``(M, K)`` fp16/bf16 activations.
* ``w``: ``(N, K)`` int8 quantized weights.
* ``scales``: ``(N,)`` fp16/bf16 — per-output-channel symmetric scale; no
  zero point (zero point support can be layered on later by adding
  ``zeros: (N,) int32`` and shifting the dequant).
* Forward output: ``y = (x @ w.T) * scales``, shape ``(M, N)`` in the
  activation dtype.

Backward (gradient wrt ``x`` only; quantized weights are not trained):
    ``dx = (dy * scales) @ w``  (same dequant pipeline, mirrored).
"""

from __future__ import annotations

import jax.numpy as jnp
import tilelang.language as T


def _act_dtype_str(dtype) -> str:
    canonical = jnp.dtype(dtype)
    mapping = {
        jnp.dtype(jnp.float16): "float16",
        jnp.dtype(jnp.bfloat16): "bfloat16",
        jnp.dtype(jnp.float32): "float32",
    }
    if canonical not in mapping:
        raise TypeError(f"Unsupported activation dtype: {dtype}")
    return mapping[canonical]


def _compute_dtype_str(dtype, use_bf16: bool) -> str:
    canonical = jnp.dtype(dtype)
    if canonical == jnp.dtype(jnp.float32):
        return "float32"
    if use_bf16 and canonical == jnp.dtype(jnp.bfloat16):
        return "bfloat16"
    return "float16"


def _meta_dtype_str(dtype) -> str:
    canonical = jnp.dtype(dtype)
    mapping = {
        jnp.dtype(jnp.float16): "float16",
        jnp.dtype(jnp.bfloat16): "bfloat16",
        jnp.dtype(jnp.float32): "float32",
        jnp.dtype(jnp.uint8): "uint8",
    }
    if canonical not in mapping:
        raise TypeError(f"Unsupported quantized metadata dtype: {dtype}")
    return mapping[canonical]


def _decode_e2m1_value(code, accum):
    """Decode a 4-bit E2M1 (MXFP4) value to a floating-point scalar.

    E2M1 format: 1 sign bit, 2 exponent bits, 1 mantissa bit.
    Normal values: ``(-1)^sign * (1 + mant*0.5) * 2^(exp-1)``.
    Subnormal values (``exp == 0``): ``(-1)^sign * mant * 0.5``.

    Args:
        code: A scalar TileLang expression holding the raw 4-bit integer code.
        accum: The TileLang accumulator dtype string (e.g. ``"float32"``).

    Returns:
        A TileLang scalar expression in *accum* type.
    """
    code_u = T.Cast("uint32", code)
    sign = (code_u >> 3) & 1
    exp = (code_u >> 1) & 3
    mant = code_u & 1
    sign_bits = sign << 31
    exp_bits = (exp + 126) << 23
    mant_bits = mant << 22
    normal_bits = sign_bits | exp_bits | mant_bits
    sub = T.Cast(accum, mant) * 0.5
    sub_signed = T.if_then_else(sign != 0, -sub, sub)
    norm = T.Cast(accum, T.reinterpret(normal_bits, "float32"))
    return T.if_then_else(exp == 0, sub_signed, norm)


def _decode_e4m3_value(code, accum):
    """Decode an 8-bit E4M3 (FP8) value to a floating-point scalar.

    E4M3 format: 1 sign bit, 4 exponent bits, 3 mantissa bits.
    Normal values: ``(-1)^sign * (1 + mant/8) * 2^(exp-7)``.
    Subnormal values (``exp == 0``): ``(-1)^sign * (mant/8) * 2^(-6)``.

    Args:
        code: A scalar TileLang expression holding the raw 8-bit integer code.
        accum: The TileLang accumulator dtype string (e.g. ``"float32"``).

    Returns:
        A TileLang scalar expression in *accum* type.
    """
    code_u = T.Cast("uint32", code)
    sign = (code_u >> 7) & 1
    exp = (code_u >> 3) & 15
    mant = code_u & 7
    sign_bits = sign << 31
    exp_bits = (exp + 120) << 23
    mant_bits = mant << 20
    normal_bits = sign_bits | exp_bits | mant_bits
    sub = T.Cast(accum, mant) * 0.001953125
    sub_signed = T.if_then_else(sign != 0, -sub, sub)
    norm = T.Cast(accum, T.reinterpret(normal_bits, "float32"))
    return T.if_then_else(exp == 0, sub_signed, norm)


def _decode_nf4_value(code, accum):
    """Decode a 4-bit NF4 code to its lookup-table float value.

    NF4 uses a 16-entry lookup table of values from -1.0 to 1.0 distributed
    according to the normal distribution quantiles (as defined in the QLoRA
    paper, Dettmers et al. 2023).

    Args:
        code: A scalar TileLang expression holding the raw 4-bit integer code
            in ``[0, 15]``.
        accum: The TileLang accumulator dtype string (e.g. ``"float32"``).

    Returns:
        A TileLang scalar expression in *accum* type with the NF4 float value.
    """
    code_i = T.Cast("int32", code)
    even = (code_i & 1) == 0
    low2 = (code_i & 2) == 0
    low4 = (code_i & 4) == 0
    low8 = (code_i & 8) == 0
    p0 = T.if_then_else(even, -1.0, -0.6961928009986877)
    p1 = T.if_then_else(even, -0.5250730514526367, -0.39491748809814453)
    p2 = T.if_then_else(even, -0.28444138169288635, -0.18477343022823334)
    p3 = T.if_then_else(even, -0.09105003625154495, 0.0)
    p4 = T.if_then_else(even, 0.07958029955625534, 0.16093020141124725)
    p5 = T.if_then_else(even, 0.24611230194568634, 0.33791524171829224)
    p6 = T.if_then_else(even, 0.44070982933044434, 0.5626170039176941)
    p7 = T.if_then_else(even, 0.7229568362236023, 1.0)
    q0 = T.if_then_else(low2, p0, p1)
    q1 = T.if_then_else(low2, p2, p3)
    q2 = T.if_then_else(low2, p4, p5)
    q3 = T.if_then_else(low2, p6, p7)
    r0 = T.if_then_else(low4, q0, q1)
    r1 = T.if_then_else(low4, q2, q3)
    return T.Cast(accum, T.if_then_else(low8, r0, r1))


def _decode_e8m0_scale(scale_code, accum):
    """Decode an 8-bit E8M0 exponent-only scale (MXFP block scale) to a float.

    E8M0 represents a power-of-two scaling factor: value = 2^(raw - 127)
    where *raw* is interpreted as a signed integer (0-127 → positive exponents,
    128-255 → negative exponents after bias subtraction).

    Args:
        scale_code: A scalar TileLang expression holding the raw 8-bit code.
        accum: The TileLang accumulator dtype string.

    Returns:
        A TileLang scalar expression in *accum* type.
    """
    raw = T.Cast("int32", scale_code)
    signed = T.if_then_else(raw >= 128, raw - 256, raw)
    normal_bits = T.Cast("uint32", signed + 127) << 23
    sub_bits = T.if_then_else(
        signed == -127,
        T.Cast("uint32", 4194304),
        T.Cast("uint32", 2097152),
    )
    bits = T.if_then_else(signed < -126, sub_bits, normal_bits)
    return T.Cast(accum, T.reinterpret(bits, "float32"))


def _decode_quant_value(mode: str, code, scale_code, zero_code, accum):
    """Dispatch quantisation decoding to the appropriate mode-specific helper.

    This function is called inside kernel prim_func bodies (at Python/
    compile-time unrolling, not at GPU runtime) to produce a single float
    value from a raw quantised code and its associated metadata.

    Supported modes and their decoding rules:
        - ``"affine"``:  ``(code - zero) * scale`` (affine int quantisation).
        - ``"nf4"``:     NF4 table lookup, multiplied by the fp16/bf16 scale.
        - ``"mxfp4"``:   E2M1 code × E8M0 block scale.
        - ``"mxfp8"``:   E4M3 code × E8M0 block scale.
        - ``"nvfp4"``:   E2M1 code × E4M3 per-element scale.
        - ``"nvfp8"``:   E4M3 code × E4M3 per-element scale.

    Args:
        mode: Quantisation mode string (one of the six listed above).
        code: Raw quantised code (TileLang scalar expression).
        scale_code: Per-group scale or E8M0 exponent (TileLang expression).
        zero_code: Per-group zero point — only used for ``"affine"`` mode;
            ignored (pass ``0.0``) for all other modes.
        accum: TileLang accumulator dtype string (e.g. ``"float32"``).

    Returns:
        A TileLang scalar expression in *accum* type.

    Raises:
        ValueError: If *mode* is not one of the six supported modes.
    """
    if mode == "affine":
        scale = T.Cast(accum, scale_code)
        zero = T.Cast(accum, zero_code)
        return (T.Cast(accum, code) - zero) * scale
    if mode == "nf4":
        return _decode_nf4_value(code, accum) * T.Cast(accum, scale_code)
    if mode == "mxfp4":
        return _decode_e2m1_value(code, accum) * _decode_e8m0_scale(scale_code, accum)
    if mode == "mxfp8":
        return _decode_e4m3_value(code, accum) * _decode_e8m0_scale(scale_code, accum)
    if mode == "nvfp4":
        return _decode_e2m1_value(code, accum) * _decode_e4m3_value(scale_code, accum)
    if mode == "nvfp8":
        return _decode_e4m3_value(code, accum) * _decode_e4m3_value(scale_code, accum)
    raise ValueError(f"Unsupported quantized_matmul TileLang mode: {mode}")


def _load_packed_code(Wq, row, elem_idx, words: int, bits: int):
    """Load one packed uint32 code, including fields that cross word boundaries."""
    if bits in (1, 2, 4, 8):
        values_per_word = 32 // bits
        word_idx = elem_idx // values_per_word
        shift = (elem_idx - word_idx * values_per_word) * bits
        safe_word = T.min(word_idx, words - 1)
        word = T.Cast("uint32", Wq[row, safe_word])
        return (word >> shift) & T.Cast("uint32", (1 << bits) - 1)

    bit_offset = elem_idx * bits
    word_idx = bit_offset // 32
    shift = bit_offset - word_idx * 32
    safe_word = T.min(word_idx, words - 1)
    safe_word1 = T.min(word_idx + 1, words - 1)
    word0 = T.Cast("uint32", Wq[row, safe_word])
    word1 = T.Cast("uint32", Wq[row, safe_word1])
    low_bits = T.min(32 - shift, bits)
    high_bits = bits - low_bits
    one = T.Cast("uint32", 1)
    low_mask = (one << low_bits) - one
    high_mask = (one << high_bits) - one
    low = (word0 >> shift) & low_mask
    high = word1 & high_mask
    return low | (high << low_bits)


def _load_packed_code_kmajor(Wq, elem_idx, col, words: int, bits: int):
    """Load one packed uint32 code from ``Wq: (words, N)`` K-major layout."""
    if bits in (1, 2, 4, 8):
        values_per_word = 32 // bits
        word_idx = elem_idx // values_per_word
        shift = (elem_idx - word_idx * values_per_word) * bits
        safe_word = T.min(word_idx, words - 1)
        word = T.Cast("uint32", Wq[safe_word, col])
        return (word >> shift) & T.Cast("uint32", (1 << bits) - 1)

    bit_offset = elem_idx * bits
    word_idx = bit_offset // 32
    shift = bit_offset - word_idx * 32
    safe_word = T.min(word_idx, words - 1)
    safe_word1 = T.min(word_idx + 1, words - 1)
    word0 = T.Cast("uint32", Wq[safe_word, col])
    word1 = T.Cast("uint32", Wq[safe_word1, col])
    low_bits = T.min(32 - shift, bits)
    high_bits = bits - low_bits
    one = T.Cast("uint32", 1)
    low_mask = (one << low_bits) - one
    high_mask = (one << high_bits) - one
    low = (word0 >> shift) & low_mask
    high = word1 & high_mask
    return low | (high << low_bits)


def make_fwd_prim_func(
    *,
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
    dtype,
    threads: int = 128,
    num_stages: int = 2,
):
    """Build the forward ``@T.prim_func`` for affine int8 quantized matmul.

    Computes ``Y[m, n] = cast(sum_k(x[m,k] * w[n,k]) * scales[n], dtype)``.

    Grid: ``(ceildiv(N, BLOCK_N), ceildiv(M, BLOCK_M))``.

    Each CTA:
    1. Loads the per-channel scales for its ``BLOCK_N`` output columns into a
       register fragment.
    2. Iterates over the ``K`` dimension in tiles of ``BLOCK_K`` using a
       ``num_stages``-stage software pipeline, loading ``X`` slabs into shared
       memory and dequantising ``W`` from int8 to the activation dtype.
    3. Accumulates ``C_local += Xs @ Ws.T`` (float32 accumulator).
    4. Writes ``Y[m, n] = cast(C_local[i,j] * scales_local[j], dtype)``.

    Shared-memory usage: ``Xs: (BLOCK_M, BLOCK_K) dtype`` +
    ``Ws_i8: (BLOCK_N, BLOCK_K) int8`` + ``Ws_f: (BLOCK_N, BLOCK_K) dtype``.

    Args:
        m: Row count of the activation matrix (``M``).
        n: Column count of the weight matrix (``N``).
        k: Reduction dimension (``K``).
        block_m: Tile size along ``M`` (``BLOCK_M``).
        block_n: Tile size along ``N`` (``BLOCK_N``).
        block_k: Tile size along ``K`` (``BLOCK_K``).
        dtype: Activation dtype (float16, bfloat16, float32).
        threads: CUDA threads per CTA (default 128).
        num_stages: Software-pipeline stages (default 2).

    Returns:
        ``@T.prim_func`` with signature
        ``(X: [M, K] dtype, W: [N, K] int8, S: [N] dtype, Y: [M, N] dtype)``.
    """
    ts = _act_dtype_str(dtype)
    accum = "float32"
    BM, BN, BK = block_m, block_n, block_k

    @T.prim_func
    def qmm_fwd(
        X: T.Tensor((m, k), ts),
        W: T.Tensor((n, k), "int8"),
        S: T.Tensor((n,), ts),
        Y: T.Tensor((m, n), ts),
    ):
        with T.Kernel(T.ceildiv(n, BN), T.ceildiv(m, BM), threads=threads) as (bx, by):
            Xs = T.alloc_shared((BM, BK), ts)
            Ws_i8 = T.alloc_shared((BN, BK), "int8")
            Ws_f = T.alloc_shared((BN, BK), ts)
            C_local = T.alloc_fragment((BM, BN), accum)
            scales_local = T.alloc_fragment((BN,), accum)

            T.clear(C_local)

            for j in T.Parallel(BN):
                n_idx = bx * BN + j
                scales_local[j] = T.if_then_else(n_idx < n, T.Cast(accum, S[n_idx]), 0.0)

            for k_iter in T.Pipelined(T.ceildiv(k, BK), num_stages=num_stages):
                T.copy(
                    X[by * BM : (by + 1) * BM, k_iter * BK : (k_iter + 1) * BK],
                    Xs,
                )
                T.copy(
                    W[bx * BN : (bx + 1) * BN, k_iter * BK : (k_iter + 1) * BK],
                    Ws_i8,
                )
                for i, j in T.Parallel(BN, BK):
                    Ws_f[i, j] = T.Cast(ts, Ws_i8[i, j])
                T.gemm(Xs, Ws_f, C_local, transpose_B=True)

            for i, j in T.Parallel(BM, BN):
                m_idx = by * BM + i
                n_idx = bx * BN + j
                if (m_idx < m) & (n_idx < n):
                    Y[m_idx, n_idx] = T.Cast(ts, C_local[i, j] * scales_local[j])

    return qmm_fwd


def make_bwd_dx_prim_func(
    *,
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_k: int,
    block_n: int,
    dtype,
    threads: int = 128,
    num_stages: int = 2,
):
    """Build the dx-only backward ``@T.prim_func`` for affine int8 quantized matmul.

    Computes ``dX[m, k] = sum_n(dY[m, n] * scales[n] * w[n, k])``.

    Grid: ``(ceildiv(K, BLOCK_K), ceildiv(M, BLOCK_M))``.

    Each CTA iterates over the ``N`` dimension in tiles of ``BLOCK_N``, loading
    scaled ``dY`` slabs and dequantised ``W`` slabs into shared memory, then
    accumulating ``C_local += dYs_scaled @ Ws_f`` (no transpose).  The
    per-channel scale is applied to ``dY`` *before* the GEMM (inside the
    tile-loading loop).

    Note:
        Only the gradient with respect to ``X`` is computed; the weights ``W``
        and scales ``S`` are non-differentiable (quantised) in this context.

    Args:
        m: Row count of the activation matrix (``M``).
        n: Column count of the weight matrix (``N``).
        k: Reduction dimension (``K``).
        block_m: Tile size along ``M``.
        block_k: Tile size along ``K`` (output dimension of ``dX``).
        block_n: Tile size along ``N`` (reduction dimension of the backward
            GEMM).
        dtype: Activation dtype (float16, bfloat16, float32).
        threads: CUDA threads per CTA (default 128).
        num_stages: Software-pipeline stages (default 2).

    Returns:
        ``@T.prim_func`` with signature
        ``(dY: [M, N] dtype, W: [N, K] int8, S: [N] dtype, dX: [M, K] dtype)``.
    """
    ts = _act_dtype_str(dtype)
    accum = "float32"
    BM, BK, BN = block_m, block_k, block_n

    @T.prim_func
    def qmm_bwd_dx(
        dY: T.Tensor((m, n), ts),
        W: T.Tensor((n, k), "int8"),
        S: T.Tensor((n,), ts),
        dX: T.Tensor((m, k), ts),
    ):
        with T.Kernel(T.ceildiv(k, BK), T.ceildiv(m, BM), threads=threads) as (kx, by):
            dYs_scaled = T.alloc_shared((BM, BN), ts)
            Ws_i8 = T.alloc_shared((BN, BK), "int8")
            Ws_f = T.alloc_shared((BN, BK), ts)
            C_local = T.alloc_fragment((BM, BK), accum)
            scales_local = T.alloc_fragment((BN,), accum)
            dY_tile = T.alloc_fragment((BM, BN), accum)

            T.clear(C_local)

            for n_iter in T.Pipelined(T.ceildiv(n, BN), num_stages=num_stages):
                for j in T.Parallel(BN):
                    n_idx = n_iter * BN + j
                    scales_local[j] = T.if_then_else(n_idx < n, T.Cast(accum, S[n_idx]), 0.0)

                for i, j in T.Parallel(BM, BN):
                    m_idx = by * BM + i
                    n_idx = n_iter * BN + j
                    dY_tile[i, j] = T.if_then_else(
                        (m_idx < m) & (n_idx < n),
                        T.Cast(accum, dY[m_idx, n_idx]) * scales_local[j],
                        0.0,
                    )
                for i, j in T.Parallel(BM, BN):
                    dYs_scaled[i, j] = T.Cast(ts, dY_tile[i, j])

                T.copy(
                    W[n_iter * BN : (n_iter + 1) * BN, kx * BK : (kx + 1) * BK],
                    Ws_i8,
                )
                for i, j in T.Parallel(BN, BK):
                    Ws_f[i, j] = T.Cast(ts, Ws_i8[i, j])

                T.gemm(dYs_scaled, Ws_f, C_local)

            for i, j in T.Parallel(BM, BK):
                m_idx = by * BM + i
                k_idx = kx * BK + j
                if (m_idx < m) & (k_idx < k):
                    dX[m_idx, k_idx] = T.Cast(ts, C_local[i, j])

    return qmm_bwd_dx


def make_packed_fwd_prim_func(
    *,
    m: int,
    n: int,
    k: int,
    groups: int,
    words: int,
    transpose: bool,
    group_size: int,
    bits: int,
    mode: str,
    block_m: int,
    block_n: int,
    block_k: int,
    dtype,
    scale_dtype,
    use_bf16: bool,
    k_major: bool = False,
    threads: int = 128,
    num_stages: int = 2,
):
    """Build a packed quantized-matmul forward ``@T.prim_func``.

    Handles both affine (mode ``"affine"``) and non-affine (``"nf4"``,
    ``"mxfp4"``, ``"mxfp8"``, ``"nvfp4"``, ``"nvfp8"``) quantisation modes.
    Returns one of four inner ``@T.prim_func`` variants chosen by the
    ``(mode, transpose)`` combination at Python compile time.

    **Layout conventions**:
        - ``transpose=True`` (column-major / output-indexed weight packing):
          ``Wq: [N, words]``, scales/zeros: ``[N, groups]``.
          Packed bits along the ``K`` axis, output channels as rows.
        - ``transpose=False`` (row-major / input-indexed weight packing):
          ``Wq: [K, words]``, scales/zeros: ``[K, groups]``.
          Packed bits along the ``N`` axis, input channels as rows.

    **Dequantisation** is performed in-register during the tile-loading step;
    dequantised weights are stored in a shared-memory ``Ws: (B_leading, BK/BN)``
    tile of type *cts* (bf16 or fp16 per *use_bf16* / activation dtype).

    **Output** is always float32 (``accum = "float32"``); the caller casts
    as needed.  This differs from the legacy int8 kernel which outputs in
    *dtype* directly.

    Grid: ``(ceildiv(N, BLOCK_N), ceildiv(M, BLOCK_M))``.

    Args:
        m: Activation batch/row count (``M``).
        n: Output channel count (``N``).
        k: Input channel count (``K``).
        groups: Number of quantisation groups.
        words: Packed ``uint32`` words per weight row/column
            (``ceil(K * bits / 32)`` for column-major or
            ``ceil(N * bits / 32)`` for row-major).
        transpose: If ``True``, weights are packed column-major
            (output-channel indexed); if ``False``, row-major
            (input-channel indexed).
        group_size: Number of elements per quantisation group.
        bits: Bits per quantised value (1 through 8 for affine).
        mode: Quantisation mode string; one of ``"affine"``, ``"nf4"``,
            ``"mxfp4"``, ``"mxfp8"``, ``"nvfp4"``, ``"nvfp8"``.
        block_m: Tile size along ``M``.
        block_n: Tile size along ``N``.
        block_k: Tile size along ``K``.
        dtype: Activation dtype (float16, bfloat16, float32).
        scale_dtype: Scale/zero metadata dtype (same options as *dtype*, plus
            uint8 for mxfp scales).
        use_bf16: If ``True`` and *dtype* is bfloat16, use bfloat16 compute
            dtype (``cts = "bfloat16"``); otherwise fall back to float16.
        threads: CUDA threads per CTA (default 128).
        num_stages: Software-pipeline stages (default 2).

    Returns:
        A ``@T.prim_func``.  The exact signature depends on ``(mode, transpose)``:

        *Non-affine, column-major*:
        ``(X: [M,K], Wq: [N,words] uint32, S: [N,groups], Y: [M,N] f32)``.

        *Non-affine, row-major*:
        ``(X: [M,K], Wq: [K,words] uint32, S: [K,groups], Y: [M,N] f32)``.

        *Affine, column-major*:
        ``(X: [M,K], Wq: [N,words] uint32, S: [N,groups], Z: [N,groups], Y: [M,N] f32)``.

        *Affine, row-major*:
        ``(X: [M,K], Wq: [K,words] uint32, S: [K,groups], Z: [K,groups], Y: [M,N] f32)``.
    """
    ts = _act_dtype_str(dtype)
    mts = _meta_dtype_str(scale_dtype)
    cts = _compute_dtype_str(dtype, use_bf16)
    accum = "float32"
    BM, BN, BK = block_m, block_n, block_k
    meta_groups_per_tile = max(1, (BK + group_size - 1) // group_size)
    cache_group_meta = False
    if k_major:
        if mode != "affine" or not transpose:
            raise ValueError("TileLang K-major packed layout currently supports affine transpose=True forward only.")

        @T.prim_func
        def qmm_packed_fwd_col_kmajor(
            X: T.Tensor((m, k), ts),
            Wq: T.Tensor((words, n), "uint32"),
            S: T.Tensor((groups, n), mts),
            Z: T.Tensor((groups, n), mts),
            Y: T.Tensor((m, n), ts),
        ):
            with T.Kernel(T.ceildiv(n, BN), T.ceildiv(m, BM), threads=threads) as (bx, by):
                dtype_ref = T.alloc_fragment((1,), ts)
                meta_ref = T.alloc_fragment((1,), mts)
                Xs = T.alloc_shared((BM, BK), cts)
                Ws = T.alloc_shared((BN, BK), cts)
                C = T.alloc_fragment((BM, BN), accum)
                if cache_group_meta:
                    Sg = T.alloc_shared((BN, meta_groups_per_tile), mts)
                    Zg = T.alloc_shared((BN, meta_groups_per_tile), mts)
                dtype_ref[0] = T.Cast(ts, 0.0)
                meta_ref[0] = T.Cast(mts, 0.0)
                T.clear(C)

                for k_iter in T.Pipelined(T.ceildiv(k, BK), num_stages=num_stages):
                    tile_group_base = (k_iter * BK) // group_size
                    for i, j in T.Parallel(BM, BK):
                        m_idx = by * BM + i
                        k_idx = k_iter * BK + j
                        Xs[i, j] = T.if_then_else(
                            (m_idx < m) & (k_idx < k),
                            T.Cast(cts, X[m_idx, k_idx]),
                            T.Cast(cts, 0.0),
                        )
                    if cache_group_meta:
                        for i, g in T.Parallel(BN, meta_groups_per_tile):
                            n_idx = bx * BN + i
                            safe_n = T.min(n_idx, n - 1)
                            group_idx = T.min(tile_group_base + g, groups - 1)
                            Sg[i, g] = S[group_idx, safe_n]
                            Zg[i, g] = Z[group_idx, safe_n]
                        T.sync_threads()
                    if bits in (2, 4, 8) and BK % (32 // bits) == 0 and group_size % (32 // bits) == 0:
                        for i, j in T.Parallel(BN, BK // (32 // bits)):
                            n_idx = bx * BN + i
                            k_base = k_iter * BK + j * (32 // bits)
                            word_idx = k_base // (32 // bits)
                            safe_n = T.min(n_idx, n - 1)
                            safe_word = T.min(word_idx, words - 1)
                            packed = T.Cast("uint32", Wq[safe_word, safe_n])
                            group_idx = T.min(k_base // group_size, groups - 1)
                            if cache_group_meta:
                                local_group = T.min(T.max(group_idx - tile_group_base, 0), meta_groups_per_tile - 1)
                                scale = T.Cast(accum, Sg[i, local_group])
                                zero = T.Cast(accum, Zg[i, local_group])
                            else:
                                scale = T.Cast(accum, S[group_idx, safe_n])
                                zero = T.Cast(accum, Z[group_idx, safe_n])
                            for lane in T.serial(32 // bits):
                                local_k = j * (32 // bits) + lane
                                k_idx = k_base + lane
                                code = (packed >> (lane * bits)) & T.Cast("uint32", (1 << bits) - 1)
                                val = (T.Cast(accum, code) - zero) * scale
                                Ws[i, local_k] = T.if_then_else(
                                    (n_idx < n) & (word_idx < words) & (k_idx < k),
                                    T.Cast(cts, val),
                                    T.Cast(cts, 0.0),
                                )
                    else:
                        for i, j in T.Parallel(BN, BK):
                            n_idx = bx * BN + i
                            k_idx = k_iter * BK + j
                            group_idx = T.min(k_idx // group_size, groups - 1)
                            safe_n = T.min(n_idx, n - 1)
                            code = _load_packed_code_kmajor(Wq, k_idx, safe_n, words, bits)
                            if cache_group_meta:
                                local_group = T.min(T.max(group_idx - tile_group_base, 0), meta_groups_per_tile - 1)
                                scale = T.Cast(accum, Sg[i, local_group])
                                zero = T.Cast(accum, Zg[i, local_group])
                            else:
                                scale = T.Cast(accum, S[group_idx, safe_n])
                                zero = T.Cast(accum, Z[group_idx, safe_n])
                            val = (T.Cast(accum, code) - zero) * scale
                            Ws[i, j] = T.if_then_else(
                                (n_idx < n) & (k_idx < k),
                                T.Cast(cts, val),
                                T.Cast(cts, 0.0),
                            )
                    T.gemm(Xs, Ws, C, transpose_B=True)

                for i, j in T.Parallel(BM, BN):
                    m_idx = by * BM + i
                    n_idx = bx * BN + j
                    if (m_idx < m) & (n_idx < n):
                        Y[m_idx, n_idx] = C[i, j]

        return qmm_packed_fwd_col_kmajor

    if mode != "affine":
        if transpose:

            @T.prim_func
            def qmm_packed_fwd_col_nonaffine(
                X: T.Tensor((m, k), ts),
                Wq: T.Tensor((n, words), "uint32"),
                S: T.Tensor((n, groups), mts),
                Y: T.Tensor((m, n), ts),
            ):
                with T.Kernel(T.ceildiv(n, BN), T.ceildiv(m, BM), threads=threads) as (bx, by):
                    dtype_ref = T.alloc_fragment((1,), ts)
                    meta_ref = T.alloc_fragment((1,), mts)
                    Xs = T.alloc_shared((BM, BK), cts)
                    Ws = T.alloc_shared((BN, BK), cts)
                    C = T.alloc_fragment((BM, BN), accum)
                    dtype_ref[0] = T.Cast(ts, 0.0)
                    meta_ref[0] = T.Cast(mts, 0)
                    T.clear(C)

                    for k_iter in T.Pipelined(T.ceildiv(k, BK), num_stages=num_stages):
                        for i, j in T.Parallel(BM, BK):
                            m_idx = by * BM + i
                            k_idx = k_iter * BK + j
                            Xs[i, j] = T.if_then_else(
                                (m_idx < m) & (k_idx < k),
                                T.Cast(cts, X[m_idx, k_idx]),
                                T.Cast(cts, 0.0),
                            )
                        for i, j in T.Parallel(BN, BK):
                            n_idx = bx * BN + i
                            k_idx = k_iter * BK + j
                            group_idx = T.min(k_idx // group_size, groups - 1)
                            safe_n = T.min(n_idx, n - 1)
                            code = _load_packed_code(Wq, safe_n, k_idx, words, bits)
                            val = _decode_quant_value(mode, code, S[safe_n, group_idx], 0.0, accum)
                            Ws[i, j] = T.if_then_else(
                                (n_idx < n) & (k_idx < k),
                                T.Cast(cts, val),
                                T.Cast(cts, 0.0),
                            )
                        T.gemm(Xs, Ws, C, transpose_B=True)

                    for i, j in T.Parallel(BM, BN):
                        m_idx = by * BM + i
                        n_idx = bx * BN + j
                        if (m_idx < m) & (n_idx < n):
                            Y[m_idx, n_idx] = C[i, j]

            return qmm_packed_fwd_col_nonaffine

        @T.prim_func
        def qmm_packed_fwd_row_nonaffine(
            X: T.Tensor((m, k), ts),
            Wq: T.Tensor((k, words), "uint32"),
            S: T.Tensor((k, groups), mts),
            Y: T.Tensor((m, n), ts),
        ):
            with T.Kernel(T.ceildiv(n, BN), T.ceildiv(m, BM), threads=threads) as (bx, by):
                dtype_ref = T.alloc_fragment((1,), ts)
                meta_ref = T.alloc_fragment((1,), mts)
                Xs = T.alloc_shared((BM, BK), cts)
                Ws = T.alloc_shared((BK, BN), cts)
                C = T.alloc_fragment((BM, BN), accum)
                dtype_ref[0] = T.Cast(ts, 0.0)
                meta_ref[0] = T.Cast(mts, 0)
                T.clear(C)

                for k_iter in T.Pipelined(T.ceildiv(k, BK), num_stages=num_stages):
                    for i, j in T.Parallel(BM, BK):
                        m_idx = by * BM + i
                        k_idx = k_iter * BK + j
                        Xs[i, j] = T.if_then_else(
                            (m_idx < m) & (k_idx < k),
                            T.Cast(cts, X[m_idx, k_idx]),
                            T.Cast(cts, 0.0),
                        )
                    for i, j in T.Parallel(BK, BN):
                        k_idx = k_iter * BK + i
                        n_idx = bx * BN + j
                        group_idx = T.min(n_idx // group_size, groups - 1)
                        safe_k = T.min(k_idx, k - 1)
                        code = _load_packed_code(Wq, safe_k, n_idx, words, bits)
                        val = _decode_quant_value(mode, code, S[safe_k, group_idx], 0.0, accum)
                        Ws[i, j] = T.if_then_else(
                            (k_idx < k) & (n_idx < n),
                            T.Cast(cts, val),
                            T.Cast(cts, 0.0),
                        )
                    T.gemm(Xs, Ws, C)

                for i, j in T.Parallel(BM, BN):
                    m_idx = by * BM + i
                    n_idx = bx * BN + j
                    if (m_idx < m) & (n_idx < n):
                        Y[m_idx, n_idx] = C[i, j]

        return qmm_packed_fwd_row_nonaffine

    if transpose:

        @T.prim_func
        def qmm_packed_fwd_col(
            X: T.Tensor((m, k), ts),
            Wq: T.Tensor((n, words), "uint32"),
            S: T.Tensor((n, groups), mts),
            Z: T.Tensor((n, groups), mts),
            Y: T.Tensor((m, n), ts),
        ):
            with T.Kernel(T.ceildiv(n, BN), T.ceildiv(m, BM), threads=threads) as (bx, by):
                dtype_ref = T.alloc_fragment((1,), ts)
                meta_ref = T.alloc_fragment((1,), mts)
                Xs = T.alloc_shared((BM, BK), cts)
                Ws = T.alloc_shared((BN, BK), cts)
                C = T.alloc_fragment((BM, BN), accum)
                if cache_group_meta:
                    Sg = T.alloc_shared((BN, meta_groups_per_tile), mts)
                    Zg = T.alloc_shared((BN, meta_groups_per_tile), mts)
                dtype_ref[0] = T.Cast(ts, 0.0)
                meta_ref[0] = T.Cast(mts, 0.0)
                T.clear(C)

                for k_iter in T.Pipelined(T.ceildiv(k, BK), num_stages=num_stages):
                    tile_group_base = (k_iter * BK) // group_size
                    for i, j in T.Parallel(BM, BK):
                        m_idx = by * BM + i
                        k_idx = k_iter * BK + j
                        Xs[i, j] = T.if_then_else(
                            (m_idx < m) & (k_idx < k),
                            T.Cast(cts, X[m_idx, k_idx]),
                            T.Cast(cts, 0.0),
                        )
                    if cache_group_meta:
                        for i, g in T.Parallel(BN, meta_groups_per_tile):
                            n_idx = bx * BN + i
                            safe_n = T.min(n_idx, n - 1)
                            group_idx = T.min(tile_group_base + g, groups - 1)
                            Sg[i, g] = S[safe_n, group_idx]
                            Zg[i, g] = Z[safe_n, group_idx]
                        T.sync_threads()
                    for i, j in T.Parallel(BN, BK):
                        n_idx = bx * BN + i
                        k_idx = k_iter * BK + j
                        group_idx = T.min(k_idx // group_size, groups - 1)
                        safe_n = T.min(n_idx, n - 1)
                        code = _load_packed_code(Wq, safe_n, k_idx, words, bits)
                        if cache_group_meta:
                            local_group = T.min(T.max(group_idx - tile_group_base, 0), meta_groups_per_tile - 1)
                            scale = T.Cast(accum, Sg[i, local_group])
                            zero = T.Cast(accum, Zg[i, local_group])
                        else:
                            scale = T.Cast(accum, S[safe_n, group_idx])
                            zero = T.Cast(accum, Z[safe_n, group_idx])
                        val = (T.Cast(accum, code) - zero) * scale
                        Ws[i, j] = T.if_then_else(
                            (n_idx < n) & (k_idx < k),
                            T.Cast(cts, val),
                            T.Cast(cts, 0.0),
                        )
                    T.gemm(Xs, Ws, C, transpose_B=True)

                for i, j in T.Parallel(BM, BN):
                    m_idx = by * BM + i
                    n_idx = bx * BN + j
                    if (m_idx < m) & (n_idx < n):
                        Y[m_idx, n_idx] = C[i, j]

        return qmm_packed_fwd_col

    if (m % BM == 0) and (n % BN == 0) and (k % BK == 0):

        @T.prim_func
        def qmm_packed_fwd_row_exact(
            X: T.Tensor((m, k), ts),
            Wq: T.Tensor((k, words), "uint32"),
            S: T.Tensor((k, groups), mts),
            Z: T.Tensor((k, groups), mts),
            Y: T.Tensor((m, n), ts),
        ):
            with T.Kernel(T.ceildiv(n, BN), T.ceildiv(m, BM), threads=threads) as (bx, by):
                dtype_ref = T.alloc_fragment((1,), ts)
                meta_ref = T.alloc_fragment((1,), mts)
                shape_ref = T.alloc_fragment((1,), "int32")
                Xs = T.alloc_shared((BM, BK), cts)
                Ws = T.alloc_shared((BK, BN), cts)
                C = T.alloc_fragment((BM, BN), accum)
                dtype_ref[0] = T.Cast(ts, 0.0)
                meta_ref[0] = T.Cast(mts, 0.0)
                shape_ref[0] = words + groups
                T.clear(C)

                for k_iter in T.Pipelined(T.ceildiv(k, BK), num_stages=num_stages):
                    for i, j in T.Parallel(BM, BK):
                        Xs[i, j] = T.Cast(cts, X[by * BM + i, k_iter * BK + j])
                    for i, j in T.Parallel(BK, BN):
                        k_idx = k_iter * BK + i
                        n_idx = bx * BN + j
                        group_idx = n_idx // group_size
                        code = _load_packed_code(Wq, k_idx, n_idx, words, bits)
                        scale = T.Cast(accum, S[k_idx, group_idx])
                        zero = T.Cast(accum, Z[k_idx, group_idx])
                        val = (T.Cast(accum, code) - zero) * scale
                        Ws[i, j] = T.Cast(cts, val)
                    T.gemm(Xs, Ws, C)

                for i, j in T.Parallel(BM, BN):
                    Y[by * BM + i, bx * BN + j] = C[i, j]

        return qmm_packed_fwd_row_exact

    @T.prim_func
    def qmm_packed_fwd_row(
        X: T.Tensor((m, k), ts),
        Wq: T.Tensor((k, words), "uint32"),
        S: T.Tensor((k, groups), mts),
        Z: T.Tensor((k, groups), mts),
        Y: T.Tensor((m, n), ts),
    ):
        with T.Kernel(T.ceildiv(n, BN), T.ceildiv(m, BM), threads=threads) as (bx, by):
            dtype_ref = T.alloc_fragment((1,), ts)
            meta_ref = T.alloc_fragment((1,), mts)
            Xs = T.alloc_shared((BM, BK), cts)
            Ws = T.alloc_shared((BK, BN), cts)
            C = T.alloc_fragment((BM, BN), accum)
            dtype_ref[0] = T.Cast(ts, 0.0)
            meta_ref[0] = T.Cast(mts, 0.0)
            T.clear(C)

            for k_iter in T.Pipelined(T.ceildiv(k, BK), num_stages=num_stages):
                for i, j in T.Parallel(BM, BK):
                    m_idx = by * BM + i
                    k_idx = k_iter * BK + j
                    Xs[i, j] = T.if_then_else(
                        (m_idx < m) & (k_idx < k),
                        T.Cast(cts, X[m_idx, k_idx]),
                        T.Cast(cts, 0.0),
                    )
                for i, j in T.Parallel(BK, BN):
                    k_idx = k_iter * BK + i
                    n_idx = bx * BN + j
                    group_idx = T.min(n_idx // group_size, groups - 1)
                    safe_k = T.min(k_idx, k - 1)
                    code = _load_packed_code(Wq, safe_k, n_idx, words, bits)
                    scale = T.Cast(accum, S[safe_k, group_idx])
                    zero = T.Cast(accum, Z[safe_k, group_idx])
                    val = (T.Cast(accum, code) - zero) * scale
                    Ws[i, j] = T.if_then_else(
                        (k_idx < k) & (n_idx < n),
                        T.Cast(cts, val),
                        T.Cast(cts, 0.0),
                    )
                T.gemm(Xs, Ws, C)

            for i, j in T.Parallel(BM, BN):
                m_idx = by * BM + i
                n_idx = bx * BN + j
                if (m_idx < m) & (n_idx < n):
                    Y[m_idx, n_idx] = C[i, j]

    return qmm_packed_fwd_row


def make_packed_gemv_kmajor_prim_func(
    *,
    n: int,
    k: int,
    groups: int,
    words: int,
    group_size: int,
    bits: int,
    dtype,
    scale_dtype,
    block_n: int,
    block_k: int,
    threads: int = 128,
    num_stages: int = 2,
):
    """Build the serial native TileLang affine GEMV for K-major packed weights.

    This is the decode-only path for ``X: (1, K)`` and ``Wq: (words, N)``.
    It avoids the tensor-core GEMM kernel's padded-M overcompute while keeping
    unpack, dequantization, reduction, and masking inside one ``@T.prim_func``.
    """
    ts = _act_dtype_str(dtype)
    mts = _meta_dtype_str(scale_dtype)
    accum = "float32"
    BN, BK = block_n, block_k

    if bits == 8:
        BW = max(1, BK // 4)
        word_tiles = (words + BW - 1) // BW

        @T.prim_func
        def qmm_packed_gemv_kmajor_bit8_words(
            X: T.Tensor((1, k), ts),
            Wq: T.Tensor((words, n), "uint32"),
            S: T.Tensor((groups, n), mts),
            Z: T.Tensor((groups, n), mts),
            Y: T.Tensor((1, n), ts),
        ):
            with T.Kernel(T.ceildiv(n, BN), threads=threads) as bx:
                prod = T.alloc_fragment((BN, BW), accum)
                partial = T.alloc_fragment((BN,), accum)
                C = T.alloc_fragment((BN,), accum)
                dtype_ref = T.alloc_fragment((1,), ts)
                meta_ref = T.alloc_fragment((1,), mts)
                dtype_ref[0] = T.Cast(ts, 0.0)
                meta_ref[0] = T.Cast(mts, 0.0)
                T.clear(C)

                for word_iter in T.Pipelined(word_tiles, num_stages=num_stages):
                    for i, j in T.Parallel(BN, BW):
                        n_idx = bx * BN + i
                        word_idx = word_iter * BW + j
                        safe_n = T.min(n_idx, n - 1)
                        safe_word = T.min(word_idx, words - 1)
                        packed = T.Cast("uint32", Wq[safe_word, safe_n])
                        group_idx = T.min((word_idx * 4) // group_size, groups - 1)
                        scale = T.Cast(accum, S[group_idx, safe_n])
                        zero = T.Cast(accum, Z[group_idx, safe_n])
                        prod[i, j] = T.Cast(accum, 0.0)
                        for lane in T.serial(4):
                            k_idx = word_idx * 4 + lane
                            code = (packed >> (lane * 8)) & T.Cast("uint32", 0xFF)
                            val = (T.Cast(accum, code) - zero) * scale
                            prod[i, j] += T.if_then_else(
                                (n_idx < n) & (word_idx < words) & (k_idx < k),
                                T.Cast(accum, X[0, k_idx]) * val,
                                T.Cast(accum, 0.0),
                            )
                    T.reduce_sum(prod, partial, dim=1, clear=True)
                    for i in T.Parallel(BN):
                        C[i] += partial[i]

                for i in T.Parallel(BN):
                    n_idx = bx * BN + i
                    if n_idx < n:
                        Y[0, n_idx] = C[i]

        return qmm_packed_gemv_kmajor_bit8_words

    @T.prim_func
    def qmm_packed_gemv_kmajor(
        X: T.Tensor((1, k), ts),
        Wq: T.Tensor((words, n), "uint32"),
        S: T.Tensor((groups, n), mts),
        Z: T.Tensor((groups, n), mts),
        Y: T.Tensor((1, n), ts),
    ):
        with T.Kernel(T.ceildiv(n, BN), threads=threads) as bx:
            prod = T.alloc_fragment((BN, BK), accum)
            partial = T.alloc_fragment((BN,), accum)
            C = T.alloc_fragment((BN,), accum)
            dtype_ref = T.alloc_fragment((1,), ts)
            meta_ref = T.alloc_fragment((1,), mts)
            dtype_ref[0] = T.Cast(ts, 0.0)
            meta_ref[0] = T.Cast(mts, 0.0)
            T.clear(C)

            for k_iter in T.Pipelined(T.ceildiv(k, BK), num_stages=num_stages):
                for i, j in T.Parallel(BN, BK):
                    n_idx = bx * BN + i
                    k_idx = k_iter * BK + j
                    safe_n = T.min(n_idx, n - 1)
                    group_idx = T.min(k_idx // group_size, groups - 1)
                    code = _load_packed_code_kmajor(Wq, k_idx, safe_n, words, bits)
                    scale = T.Cast(accum, S[group_idx, safe_n])
                    zero = T.Cast(accum, Z[group_idx, safe_n])
                    val = (T.Cast(accum, code) - zero) * scale
                    prod[i, j] = T.if_then_else(
                        (n_idx < n) & (k_idx < k),
                        T.Cast(accum, X[0, k_idx]) * val,
                        T.Cast(accum, 0.0),
                    )
                T.reduce_sum(prod, partial, dim=1, clear=True)
                for i in T.Parallel(BN):
                    C[i] += partial[i]

            for i in T.Parallel(BN):
                n_idx = bx * BN + i
                if n_idx < n:
                    Y[0, n_idx] = C[i]

    return qmm_packed_gemv_kmajor


def make_packed_gemv_kmajor_split_prim_func(
    *,
    n: int,
    k: int,
    groups: int,
    words: int,
    group_size: int,
    bits: int,
    dtype,
    scale_dtype,
    block_n: int,
    block_k: int,
    threads: int = 128,
):
    """Build the split-K partial kernel for native TileLang affine GEMV."""
    ts = _act_dtype_str(dtype)
    mts = _meta_dtype_str(scale_dtype)
    accum = "float32"
    BN, BK = block_n, block_k
    k_tiles = (k + BK - 1) // BK

    if bits == 8 and group_size in (32, 64, 128) and BK == 64:
        BW = BK // 4
        groups_per_tile = BK // group_size if group_size <= BK else 1
        words_per_meta = BW // groups_per_tile

        @T.prim_func
        def qmm_packed_gemv_kmajor_split_bit8_grouped_words(
            X: T.Tensor((1, k), ts),
            Wq: T.Tensor((words, n), "uint32"),
            S: T.Tensor((groups, n), mts),
            Z: T.Tensor((groups, n), mts),
            P: T.Tensor((k_tiles, n), accum),
        ):
            with T.Kernel(T.ceildiv(n, BN), k_tiles, threads=threads) as (bx, by):
                C = T.alloc_fragment((BN,), accum)
                scale_v = T.alloc_fragment((BN,), accum)
                zero_v = T.alloc_fragment((BN,), accum)
                dtype_ref = T.alloc_fragment((1,), ts)
                meta_ref = T.alloc_fragment((1,), mts)
                dtype_ref[0] = T.Cast(ts, 0.0)
                meta_ref[0] = T.Cast(mts, 0.0)
                T.clear(C)

                for go in T.serial(groups_per_tile):
                    for i in T.Parallel(BN):
                        n_idx = bx * BN + i
                        safe_n = T.min(n_idx, n - 1)
                        if group_size <= BK:
                            group_idx = T.min(by * groups_per_tile + go, groups - 1)
                        else:
                            group_idx = T.min((by * BK) // group_size, groups - 1)
                        scale_v[i] = T.Cast(accum, S[group_idx, safe_n])
                        zero_v[i] = T.Cast(accum, Z[group_idx, safe_n])

                    for j in T.serial(words_per_meta):
                        for i in T.Parallel(BN):
                            n_idx = bx * BN + i
                            word_idx = by * BW + go * words_per_meta + j
                            safe_n = T.min(n_idx, n - 1)
                            safe_word = T.min(word_idx, words - 1)
                            packed = T.Cast("uint32", Wq[safe_word, safe_n])
                            for lane in T.serial(4):
                                k_idx = word_idx * 4 + lane
                                code = (packed >> (lane * 8)) & T.Cast("uint32", 0xFF)
                                val = (T.Cast(accum, code) - zero_v[i]) * scale_v[i]
                                C[i] += T.if_then_else(
                                    (n_idx < n) & (word_idx < words) & (k_idx < k),
                                    T.Cast(accum, X[0, k_idx]) * val,
                                    T.Cast(accum, 0.0),
                                )

                for i in T.Parallel(BN):
                    n_idx = bx * BN + i
                    if n_idx < n:
                        P[by, n_idx] = C[i]

        return qmm_packed_gemv_kmajor_split_bit8_grouped_words

    if bits == 8:
        BW = max(1, BK // 4)

        @T.prim_func
        def qmm_packed_gemv_kmajor_split_bit8_words(
            X: T.Tensor((1, k), ts),
            Wq: T.Tensor((words, n), "uint32"),
            S: T.Tensor((groups, n), mts),
            Z: T.Tensor((groups, n), mts),
            P: T.Tensor((k_tiles, n), accum),
        ):
            with T.Kernel(T.ceildiv(n, BN), k_tiles, threads=threads) as (bx, by):
                C = T.alloc_fragment((BN,), accum)
                dtype_ref = T.alloc_fragment((1,), ts)
                meta_ref = T.alloc_fragment((1,), mts)
                dtype_ref[0] = T.Cast(ts, 0.0)
                meta_ref[0] = T.Cast(mts, 0.0)
                T.clear(C)

                for j in T.serial(BW):
                    for i in T.Parallel(BN):
                        n_idx = bx * BN + i
                        word_idx = by * BW + j
                        safe_n = T.min(n_idx, n - 1)
                        safe_word = T.min(word_idx, words - 1)
                        packed = T.Cast("uint32", Wq[safe_word, safe_n])
                        group_idx = T.min((word_idx * 4) // group_size, groups - 1)
                        scale = T.Cast(accum, S[group_idx, safe_n])
                        zero = T.Cast(accum, Z[group_idx, safe_n])
                        for lane in T.serial(4):
                            k_idx = word_idx * 4 + lane
                            code = (packed >> (lane * 8)) & T.Cast("uint32", 0xFF)
                            val = (T.Cast(accum, code) - zero) * scale
                            C[i] += T.if_then_else(
                                (n_idx < n) & (word_idx < words) & (k_idx < k),
                                T.Cast(accum, X[0, k_idx]) * val,
                                T.Cast(accum, 0.0),
                            )
                for i in T.Parallel(BN):
                    n_idx = bx * BN + i
                    if n_idx < n:
                        P[by, n_idx] = C[i]

        return qmm_packed_gemv_kmajor_split_bit8_words

    if bits == 8:

        @T.prim_func
        def qmm_packed_gemv_kmajor_split_direct(
            X: T.Tensor((1, k), ts),
            Wq: T.Tensor((words, n), "uint32"),
            S: T.Tensor((groups, n), mts),
            Z: T.Tensor((groups, n), mts),
            P: T.Tensor((k_tiles, n), accum),
        ):
            with T.Kernel(T.ceildiv(n, BN), k_tiles, threads=threads) as (bx, by):
                C = T.alloc_fragment((BN,), accum)
                dtype_ref = T.alloc_fragment((1,), ts)
                meta_ref = T.alloc_fragment((1,), mts)
                dtype_ref[0] = T.Cast(ts, 0.0)
                meta_ref[0] = T.Cast(mts, 0.0)
                T.clear(C)

                for j in T.serial(BK):
                    for i in T.Parallel(BN):
                        n_idx = bx * BN + i
                        k_idx = by * BK + j
                        safe_n = T.min(n_idx, n - 1)
                        group_idx = T.min(k_idx // group_size, groups - 1)
                        code = _load_packed_code_kmajor(Wq, k_idx, safe_n, words, bits)
                        scale = T.Cast(accum, S[group_idx, safe_n])
                        zero = T.Cast(accum, Z[group_idx, safe_n])
                        val = (T.Cast(accum, code) - zero) * scale
                        C[i] += T.if_then_else(
                            (n_idx < n) & (k_idx < k),
                            T.Cast(accum, X[0, k_idx]) * val,
                            T.Cast(accum, 0.0),
                        )

                for i in T.Parallel(BN):
                    n_idx = bx * BN + i
                    if n_idx < n:
                        P[by, n_idx] = C[i]

        return qmm_packed_gemv_kmajor_split_direct

    @T.prim_func
    def qmm_packed_gemv_kmajor_split(
        X: T.Tensor((1, k), ts),
        Wq: T.Tensor((words, n), "uint32"),
        S: T.Tensor((groups, n), mts),
        Z: T.Tensor((groups, n), mts),
        P: T.Tensor((k_tiles, n), accum),
    ):
        with T.Kernel(T.ceildiv(n, BN), k_tiles, threads=threads) as (bx, by):
            prod = T.alloc_fragment((BN, BK), accum)
            partial = T.alloc_fragment((BN,), accum)
            dtype_ref = T.alloc_fragment((1,), ts)
            meta_ref = T.alloc_fragment((1,), mts)
            dtype_ref[0] = T.Cast(ts, 0.0)
            meta_ref[0] = T.Cast(mts, 0.0)

            for i, j in T.Parallel(BN, BK):
                n_idx = bx * BN + i
                k_idx = by * BK + j
                safe_n = T.min(n_idx, n - 1)
                group_idx = T.min(k_idx // group_size, groups - 1)
                code = _load_packed_code_kmajor(Wq, k_idx, safe_n, words, bits)
                scale = T.Cast(accum, S[group_idx, safe_n])
                zero = T.Cast(accum, Z[group_idx, safe_n])
                val = (T.Cast(accum, code) - zero) * scale
                prod[i, j] = T.if_then_else(
                    (n_idx < n) & (k_idx < k),
                    T.Cast(accum, X[0, k_idx]) * val,
                    T.Cast(accum, 0.0),
                )
            T.reduce_sum(prod, partial, dim=1, clear=True)
            for i in T.Parallel(BN):
                n_idx = bx * BN + i
                if n_idx < n:
                    P[by, n_idx] = partial[i]

    return qmm_packed_gemv_kmajor_split


def make_packed_gemv_kmajor_reduce_prim_func(
    *,
    n: int,
    k_tiles: int,
    dtype,
    block_n: int,
    threads: int = 128,
):
    """Build the native TileLang reduction kernel for split-K GEMV partials."""
    ts = _act_dtype_str(dtype)
    accum = "float32"
    BN = block_n

    @T.prim_func
    def qmm_packed_gemv_kmajor_reduce(
        P: T.Tensor((k_tiles, n), accum),
        Y: T.Tensor((1, n), ts),
    ):
        with T.Kernel(T.ceildiv(n, BN), threads=threads) as bx:
            C = T.alloc_fragment((BN,), accum)
            dtype_ref = T.alloc_fragment((1,), ts)
            dtype_ref[0] = T.Cast(ts, 0.0)
            T.clear(C)

            for split in T.serial(k_tiles):
                for i in T.Parallel(BN):
                    n_idx = bx * BN + i
                    if n_idx < n:
                        C[i] += P[split, n_idx]

            for i in T.Parallel(BN):
                n_idx = bx * BN + i
                if n_idx < n:
                    Y[0, n_idx] = C[i]

    return qmm_packed_gemv_kmajor_reduce


def make_packed_dequant_prim_func(
    *,
    n: int,
    k: int,
    groups: int,
    words: int,
    group_size: int,
    bits: int,
    mode: str,
    out_dtype,
    scale_dtype,
    block_k: int,
    block_n: int,
    transpose: bool = False,
    threads: int = 256,
):
    """Build a packed-weight dequantization ``@T.prim_func``.

    The produced kernel decodes packed weights into a dense ``(K, N)`` matrix
    suitable for a follow-up TileLang GEMM. It supports canonical row-major
    ``Wq: (K, words)`` and canonical column-major ``Wq: (N, words)`` layouts.
    """
    mts = _meta_dtype_str(scale_dtype)
    ots = _compute_dtype_str(out_dtype, jnp.dtype(out_dtype) == jnp.dtype(jnp.bfloat16))
    accum = "float32"
    BK, BN = block_k, block_n

    if mode == "affine":
        if transpose:

            @T.prim_func
            def qmm_packed_dequant_col(
                Wq: T.Tensor((n, words), "uint32"),
                S: T.Tensor((n, groups), mts),
                Z: T.Tensor((n, groups), mts),
                Wd: T.Tensor((k, n), ots),
            ):
                with T.Kernel(T.ceildiv(n, BN), T.ceildiv(k, BK), threads=threads) as (bx, by):
                    meta_ref = T.alloc_fragment((1,), mts)
                    meta_ref[0] = T.Cast(mts, 0.0)
                    for i, j in T.Parallel(BK, BN):
                        k_idx = by * BK + i
                        n_idx = bx * BN + j
                        group_idx = T.min(k_idx // group_size, groups - 1)
                        safe_n = T.min(n_idx, n - 1)
                        code = _load_packed_code(Wq, safe_n, k_idx, words, bits)
                        val = _decode_quant_value(mode, code, S[safe_n, group_idx], Z[safe_n, group_idx], accum)
                        if (k_idx < k) & (n_idx < n):
                            Wd[k_idx, n_idx] = T.Cast(ots, val)

            return qmm_packed_dequant_col

        @T.prim_func
        def qmm_packed_dequant_row(
            Wq: T.Tensor((k, words), "uint32"),
            S: T.Tensor((k, groups), mts),
            Z: T.Tensor((k, groups), mts),
            Wd: T.Tensor((k, n), ots),
        ):
            with T.Kernel(T.ceildiv(n, BN), T.ceildiv(k, BK), threads=threads) as (bx, by):
                meta_ref = T.alloc_fragment((1,), mts)
                meta_ref[0] = T.Cast(mts, 0.0)
                for i, j in T.Parallel(BK, BN):
                    k_idx = by * BK + i
                    n_idx = bx * BN + j
                    group_idx = T.min(n_idx // group_size, groups - 1)
                    safe_k = T.min(k_idx, k - 1)
                    code = _load_packed_code(Wq, safe_k, n_idx, words, bits)
                    val = _decode_quant_value(mode, code, S[safe_k, group_idx], Z[safe_k, group_idx], accum)
                    if (k_idx < k) & (n_idx < n):
                        Wd[k_idx, n_idx] = T.Cast(ots, val)

        return qmm_packed_dequant_row

    @T.prim_func
    def qmm_packed_dequant_row_nonaffine(
        Wq: T.Tensor((k, words), "uint32"),
        S: T.Tensor((k, groups), mts),
        Wd: T.Tensor((k, n), ots),
    ):
        with T.Kernel(T.ceildiv(n, BN), T.ceildiv(k, BK), threads=threads) as (bx, by):
            meta_ref = T.alloc_fragment((1,), mts)
            meta_ref[0] = T.Cast(mts, 0)
            for i, j in T.Parallel(BK, BN):
                k_idx = by * BK + i
                n_idx = bx * BN + j
                group_idx = T.min(n_idx // group_size, groups - 1)
                safe_k = T.min(k_idx, k - 1)
                code = _load_packed_code(Wq, safe_k, n_idx, words, bits)
                val = _decode_quant_value(mode, code, S[safe_k, group_idx], 0.0, accum)
                if (k_idx < k) & (n_idx < n):
                    Wd[k_idx, n_idx] = T.Cast(ots, val)

    return qmm_packed_dequant_row_nonaffine


def make_dense_fwd_prim_func(
    *,
    m: int,
    n: int,
    k: int,
    dtype,
    weight_dtype,
    block_m: int,
    block_n: int,
    block_k: int,
    copy_exact: bool = False,
    threads: int = 128,
    num_stages: int = 2,
):
    """Build a dense TileLang GEMM forward kernel for ``X @ Wd``.

    ``Wd`` is stored row-major as ``(K, N)`` and is expected to be produced by
    ``make_packed_dequant_prim_func``.
    """
    ts = _act_dtype_str(dtype)
    wts = _compute_dtype_str(weight_dtype, jnp.dtype(weight_dtype) == jnp.dtype(jnp.bfloat16))
    accum = "float32"
    BM, BN, BK = block_m, block_n, block_k

    if copy_exact and (m % BM == 0) and (n % BN == 0) and (k % BK == 0):

        @T.prim_func
        def qmm_dense_fwd_exact_copy(
            X: T.Tensor((m, k), ts),
            Wd: T.Tensor((k, n), wts),
            Y: T.Tensor((m, n), accum),
        ):
            with T.Kernel(T.ceildiv(n, BN), T.ceildiv(m, BM), threads=threads) as (bx, by):
                dtype_ref = T.alloc_fragment((1,), ts)
                weight_ref = T.alloc_fragment((1,), wts)
                Xs = T.alloc_shared((BM, BK), wts)
                Ws = T.alloc_shared((BK, BN), wts)
                C = T.alloc_fragment((BM, BN), accum)
                dtype_ref[0] = T.Cast(ts, 0.0)
                weight_ref[0] = T.Cast(wts, 0.0)
                T.clear(C)

                for k_iter in T.Pipelined(T.ceildiv(k, BK), num_stages=num_stages):
                    T.copy(
                        X[by * BM : (by + 1) * BM, k_iter * BK : (k_iter + 1) * BK],
                        Xs,
                    )
                    T.copy(
                        Wd[k_iter * BK : (k_iter + 1) * BK, bx * BN : (bx + 1) * BN],
                        Ws,
                    )
                    T.gemm(Xs, Ws, C)

                T.copy(C, Y[by * BM : (by + 1) * BM, bx * BN : (bx + 1) * BN])

        return qmm_dense_fwd_exact_copy

    if (m % BM == 0) and (n % BN == 0) and (k % BK == 0):

        @T.prim_func
        def qmm_dense_fwd_exact(
            X: T.Tensor((m, k), ts),
            Wd: T.Tensor((k, n), wts),
            Y: T.Tensor((m, n), accum),
        ):
            with T.Kernel(T.ceildiv(n, BN), T.ceildiv(m, BM), threads=threads) as (bx, by):
                dtype_ref = T.alloc_fragment((1,), ts)
                weight_ref = T.alloc_fragment((1,), wts)
                Xs = T.alloc_shared((BM, BK), wts)
                Ws = T.alloc_shared((BK, BN), wts)
                C = T.alloc_fragment((BM, BN), accum)
                dtype_ref[0] = T.Cast(ts, 0.0)
                weight_ref[0] = T.Cast(wts, 0.0)
                T.clear(C)

                for k_iter in T.Pipelined(T.ceildiv(k, BK), num_stages=num_stages):
                    for i, j in T.Parallel(BM, BK):
                        Xs[i, j] = T.Cast(wts, X[by * BM + i, k_iter * BK + j])
                    for i, j in T.Parallel(BK, BN):
                        Ws[i, j] = Wd[k_iter * BK + i, bx * BN + j]
                    T.gemm(Xs, Ws, C)

                for i, j in T.Parallel(BM, BN):
                    Y[by * BM + i, bx * BN + j] = C[i, j]

        return qmm_dense_fwd_exact

    @T.prim_func
    def qmm_dense_fwd(
        X: T.Tensor((m, k), ts),
        Wd: T.Tensor((k, n), wts),
        Y: T.Tensor((m, n), accum),
    ):
        with T.Kernel(T.ceildiv(n, BN), T.ceildiv(m, BM), threads=threads) as (bx, by):
            dtype_ref = T.alloc_fragment((1,), ts)
            weight_ref = T.alloc_fragment((1,), wts)
            Xs = T.alloc_shared((BM, BK), wts)
            Ws = T.alloc_shared((BK, BN), wts)
            C = T.alloc_fragment((BM, BN), accum)
            dtype_ref[0] = T.Cast(ts, 0.0)
            weight_ref[0] = T.Cast(wts, 0.0)
            T.clear(C)

            for k_iter in T.Pipelined(T.ceildiv(k, BK), num_stages=num_stages):
                for i, j in T.Parallel(BM, BK):
                    m_idx = by * BM + i
                    k_idx = k_iter * BK + j
                    Xs[i, j] = T.if_then_else(
                        (m_idx < m) & (k_idx < k),
                        T.Cast(wts, X[m_idx, k_idx]),
                        T.Cast(wts, 0.0),
                    )
                for i, j in T.Parallel(BK, BN):
                    k_idx = k_iter * BK + i
                    n_idx = bx * BN + j
                    Ws[i, j] = T.if_then_else(
                        (k_idx < k) & (n_idx < n),
                        Wd[k_idx, n_idx],
                        T.Cast(wts, 0.0),
                    )
                T.gemm(Xs, Ws, C)

            for i, j in T.Parallel(BM, BN):
                m_idx = by * BM + i
                n_idx = bx * BN + j
                if (m_idx < m) & (n_idx < n):
                    Y[m_idx, n_idx] = C[i, j]

    return qmm_dense_fwd


def make_dense_dx_prim_func(
    *,
    m: int,
    n: int,
    k: int,
    dtype,
    weight_dtype,
    block_m: int,
    block_k: int,
    block_n: int,
    threads: int = 128,
    num_stages: int = 2,
):
    """Build a dense TileLang ``dX = dY @ Wd.T`` kernel."""
    ts = _act_dtype_str(dtype)
    wts = _compute_dtype_str(weight_dtype, jnp.dtype(weight_dtype) == jnp.dtype(jnp.bfloat16))
    accum = "float32"
    BM, BK, BN = block_m, block_k, block_n

    @T.prim_func
    def qmm_dense_dx(
        dY: T.Tensor((m, n), accum),
        Wd: T.Tensor((k, n), wts),
        dX: T.Tensor((m, k), ts),
    ):
        with T.Kernel(T.ceildiv(k, BK), T.ceildiv(m, BM), threads=threads) as (kx, by):
            dtype_ref = T.alloc_fragment((1,), ts)
            weight_ref = T.alloc_fragment((1,), wts)
            dYs = T.alloc_shared((BM, BN), wts)
            Ws = T.alloc_shared((BK, BN), wts)
            C = T.alloc_fragment((BM, BK), accum)
            dtype_ref[0] = T.Cast(ts, 0.0)
            weight_ref[0] = T.Cast(wts, 0.0)
            T.clear(C)

            for n_iter in T.Pipelined(T.ceildiv(n, BN), num_stages=num_stages):
                for i, j in T.Parallel(BM, BN):
                    m_idx = by * BM + i
                    n_idx = n_iter * BN + j
                    dYs[i, j] = T.if_then_else(
                        (m_idx < m) & (n_idx < n),
                        T.Cast(wts, dY[m_idx, n_idx]),
                        T.Cast(wts, 0.0),
                    )
                for i, j in T.Parallel(BK, BN):
                    k_idx = kx * BK + i
                    n_idx = n_iter * BN + j
                    Ws[i, j] = T.if_then_else(
                        (k_idx < k) & (n_idx < n),
                        Wd[k_idx, n_idx],
                        T.Cast(wts, 0.0),
                    )
                T.gemm(dYs, Ws, C, transpose_B=True)

            for i, j in T.Parallel(BM, BK):
                m_idx = by * BM + i
                k_idx = kx * BK + j
                if (m_idx < m) & (k_idx < k):
                    dX[m_idx, k_idx] = T.Cast(ts, C[i, j])

    return qmm_dense_dx


def make_packed_bwd_dx_prim_func(
    *,
    m: int,
    n: int,
    k: int,
    groups: int,
    words: int,
    transpose: bool,
    group_size: int,
    bits: int,
    mode: str,
    block_m: int,
    block_k: int,
    block_n: int,
    dtype,
    scale_dtype,
    use_bf16: bool,
    threads: int = 128,
    num_stages: int = 2,
):
    """Build a packed quantized-matmul ``dX`` backward ``@T.prim_func``.

    Computes ``dX[m, k] = sum_n(dY[m, n] * w_dequant[n, k])`` (column-major)
    or ``dX[m, k] = sum_n(dY[m, n] * w_dequant[n, k])`` (row-major), where
    ``w_dequant`` is dequantised on-the-fly from the packed representation.

    Grid: ``(ceildiv(K, BLOCK_K), ceildiv(M, BLOCK_M))``.

    Accepts the same layout/mode combinations as ``make_packed_fwd_prim_func``;
    the dequantisation logic is identical to the forward pass.  The ``dY``
    input is expected in float32 (the forward output dtype).

    Note:
        For non-affine modes, ``zero_code`` is passed as ``0.0`` to
        ``_decode_quant_value``; the zero-point field is ignored.

    Args:
        m: Activation batch/row count.
        n: Output channel count.
        k: Input channel count.
        groups: Number of quantisation groups.
        words: Packed ``uint32`` words per weight row/column.
        transpose: If ``True``, weights are packed column-major.
        group_size: Elements per quantisation group.
        bits: Bits per quantised value.
        mode: Quantisation mode string.
        block_m: Tile size along ``M``.
        block_k: Tile size along ``K``.
        block_n: Tile size along ``N`` (reduction dimension).
        dtype: Activation dtype (output ``dX`` dtype).
        scale_dtype: Scale metadata dtype.
        use_bf16: Whether to use bfloat16 compute for bfloat16 activations.
        threads: CUDA threads per CTA (default 128).
        num_stages: Software-pipeline stages (default 2).

    Returns:
        A ``@T.prim_func`` that writes ``dX: [M, K]`` in *dtype*.  For
        non-affine modes the signature omits the zero tensor; for affine modes
        it includes it.  See ``make_packed_fwd_prim_func`` for detailed
        signature variants.
    """
    ts = _act_dtype_str(dtype)
    mts = _meta_dtype_str(scale_dtype)
    cts = _compute_dtype_str(dtype, use_bf16)
    accum = "float32"
    BM, BK, BN = block_m, block_k, block_n

    if mode != "affine":
        if transpose:

            @T.prim_func
            def qmm_packed_bwd_dx_col_nonaffine(
                dY: T.Tensor((m, n), accum),
                Wq: T.Tensor((n, words), "uint32"),
                S: T.Tensor((n, groups), mts),
                dX: T.Tensor((m, k), ts),
            ):
                with T.Kernel(T.ceildiv(k, BK), T.ceildiv(m, BM), threads=threads) as (kx, by):
                    meta_ref = T.alloc_fragment((1,), mts)
                    dYs = T.alloc_shared((BM, BN), cts)
                    Ws = T.alloc_shared((BN, BK), cts)
                    C = T.alloc_fragment((BM, BK), accum)
                    meta_ref[0] = T.Cast(mts, 0)
                    T.clear(C)

                    for n_iter in T.Pipelined(T.ceildiv(n, BN), num_stages=num_stages):
                        for i, j in T.Parallel(BM, BN):
                            m_idx = by * BM + i
                            n_idx = n_iter * BN + j
                            dYs[i, j] = T.if_then_else(
                                (m_idx < m) & (n_idx < n),
                                T.Cast(cts, dY[m_idx, n_idx]),
                                T.Cast(cts, 0.0),
                            )
                        for i, j in T.Parallel(BN, BK):
                            n_idx = n_iter * BN + i
                            k_idx = kx * BK + j
                            group_idx = T.min(k_idx // group_size, groups - 1)
                            safe_n = T.min(n_idx, n - 1)
                            code = _load_packed_code(Wq, safe_n, k_idx, words, bits)
                            val = _decode_quant_value(mode, code, S[safe_n, group_idx], 0.0, accum)
                            Ws[i, j] = T.if_then_else(
                                (n_idx < n) & (k_idx < k),
                                T.Cast(cts, val),
                                T.Cast(cts, 0.0),
                            )
                        T.gemm(dYs, Ws, C)

                    for i, j in T.Parallel(BM, BK):
                        m_idx = by * BM + i
                        k_idx = kx * BK + j
                        if (m_idx < m) & (k_idx < k):
                            dX[m_idx, k_idx] = T.Cast(ts, C[i, j])

            return qmm_packed_bwd_dx_col_nonaffine

        @T.prim_func
        def qmm_packed_bwd_dx_row_nonaffine(
            dY: T.Tensor((m, n), accum),
            Wq: T.Tensor((k, words), "uint32"),
            S: T.Tensor((k, groups), mts),
            dX: T.Tensor((m, k), ts),
        ):
            with T.Kernel(T.ceildiv(k, BK), T.ceildiv(m, BM), threads=threads) as (kx, by):
                meta_ref = T.alloc_fragment((1,), mts)
                dYs = T.alloc_shared((BM, BN), cts)
                Ws = T.alloc_shared((BN, BK), cts)
                C = T.alloc_fragment((BM, BK), accum)
                meta_ref[0] = T.Cast(mts, 0)
                T.clear(C)

                for n_iter in T.Pipelined(T.ceildiv(n, BN), num_stages=num_stages):
                    for i, j in T.Parallel(BM, BN):
                        m_idx = by * BM + i
                        n_idx = n_iter * BN + j
                        dYs[i, j] = T.if_then_else(
                            (m_idx < m) & (n_idx < n),
                            T.Cast(cts, dY[m_idx, n_idx]),
                            T.Cast(cts, 0.0),
                        )
                    for i, j in T.Parallel(BN, BK):
                        n_idx = n_iter * BN + i
                        k_idx = kx * BK + j
                        group_idx = T.min(n_idx // group_size, groups - 1)
                        safe_k = T.min(k_idx, k - 1)
                        code = _load_packed_code(Wq, safe_k, n_idx, words, bits)
                        val = _decode_quant_value(mode, code, S[safe_k, group_idx], 0.0, accum)
                        Ws[i, j] = T.if_then_else(
                            (n_idx < n) & (k_idx < k),
                            T.Cast(cts, val),
                            T.Cast(cts, 0.0),
                        )
                    T.gemm(dYs, Ws, C)

                for i, j in T.Parallel(BM, BK):
                    m_idx = by * BM + i
                    k_idx = kx * BK + j
                    if (m_idx < m) & (k_idx < k):
                        dX[m_idx, k_idx] = T.Cast(ts, C[i, j])

        return qmm_packed_bwd_dx_row_nonaffine

    if transpose:

        @T.prim_func
        def qmm_packed_bwd_dx_col(
            dY: T.Tensor((m, n), accum),
            Wq: T.Tensor((n, words), "uint32"),
            S: T.Tensor((n, groups), mts),
            Z: T.Tensor((n, groups), mts),
            dX: T.Tensor((m, k), ts),
        ):
            with T.Kernel(T.ceildiv(k, BK), T.ceildiv(m, BM), threads=threads) as (kx, by):
                meta_ref = T.alloc_fragment((1,), mts)
                dYs = T.alloc_shared((BM, BN), cts)
                Ws = T.alloc_shared((BN, BK), cts)
                C = T.alloc_fragment((BM, BK), accum)
                meta_ref[0] = T.Cast(mts, 0.0)
                T.clear(C)

                for n_iter in T.Pipelined(T.ceildiv(n, BN), num_stages=num_stages):
                    for i, j in T.Parallel(BM, BN):
                        m_idx = by * BM + i
                        n_idx = n_iter * BN + j
                        dYs[i, j] = T.if_then_else(
                            (m_idx < m) & (n_idx < n),
                            T.Cast(cts, dY[m_idx, n_idx]),
                            T.Cast(cts, 0.0),
                        )
                    for i, j in T.Parallel(BN, BK):
                        n_idx = n_iter * BN + i
                        k_idx = kx * BK + j
                        group_idx = T.min(k_idx // group_size, groups - 1)
                        safe_n = T.min(n_idx, n - 1)
                        code = _load_packed_code(Wq, safe_n, k_idx, words, bits)
                        scale = T.Cast(accum, S[safe_n, group_idx])
                        zero = T.Cast(accum, Z[safe_n, group_idx])
                        val = (T.Cast(accum, code) - zero) * scale
                        Ws[i, j] = T.if_then_else(
                            (n_idx < n) & (k_idx < k),
                            T.Cast(cts, val),
                            T.Cast(cts, 0.0),
                        )
                    T.gemm(dYs, Ws, C)

                for i, j in T.Parallel(BM, BK):
                    m_idx = by * BM + i
                    k_idx = kx * BK + j
                    if (m_idx < m) & (k_idx < k):
                        dX[m_idx, k_idx] = T.Cast(ts, C[i, j])

        return qmm_packed_bwd_dx_col

    @T.prim_func
    def qmm_packed_bwd_dx_row(
        dY: T.Tensor((m, n), accum),
        Wq: T.Tensor((k, words), "uint32"),
        S: T.Tensor((k, groups), mts),
        Z: T.Tensor((k, groups), mts),
        dX: T.Tensor((m, k), ts),
    ):
        with T.Kernel(T.ceildiv(k, BK), T.ceildiv(m, BM), threads=threads) as (kx, by):
            meta_ref = T.alloc_fragment((1,), mts)
            dYs = T.alloc_shared((BM, BN), cts)
            Ws = T.alloc_shared((BN, BK), cts)
            C = T.alloc_fragment((BM, BK), accum)
            meta_ref[0] = T.Cast(mts, 0.0)
            T.clear(C)

            for n_iter in T.Pipelined(T.ceildiv(n, BN), num_stages=num_stages):
                for i, j in T.Parallel(BM, BN):
                    m_idx = by * BM + i
                    n_idx = n_iter * BN + j
                    dYs[i, j] = T.if_then_else(
                        (m_idx < m) & (n_idx < n),
                        T.Cast(cts, dY[m_idx, n_idx]),
                        T.Cast(cts, 0.0),
                    )
                for i, j in T.Parallel(BN, BK):
                    n_idx = n_iter * BN + i
                    k_idx = kx * BK + j
                    group_idx = T.min(n_idx // group_size, groups - 1)
                    safe_k = T.min(k_idx, k - 1)
                    code = _load_packed_code(Wq, safe_k, n_idx, words, bits)
                    scale = T.Cast(accum, S[safe_k, group_idx])
                    zero = T.Cast(accum, Z[safe_k, group_idx])
                    val = (T.Cast(accum, code) - zero) * scale
                    Ws[i, j] = T.if_then_else(
                        (n_idx < n) & (k_idx < k),
                        T.Cast(cts, val),
                        T.Cast(cts, 0.0),
                    )
                T.gemm(dYs, Ws, C)

            for i, j in T.Parallel(BM, BK):
                m_idx = by * BM + i
                k_idx = kx * BK + j
                if (m_idx < m) & (k_idx < k):
                    dX[m_idx, k_idx] = T.Cast(ts, C[i, j])

    return qmm_packed_bwd_dx_row


def make_packed_bwd_meta_prim_func(
    *,
    m: int,
    n: int,
    k: int,
    groups: int,
    words: int,
    transpose: bool,
    group_size: int,
    bits: int,
    x_dtype,
    scale_dtype,
    threads: int = 128,
):
    """Build the affine packed scale/zero-point backward ``@T.prim_func``.

    Computes gradients with respect to the per-group affine parameters:

    - ``dS[n/k, g] = sum_{m,kk}  X[m, k_kk] * dY[m, n]  * centered_code``
    - ``dZ[n/k, g] = sum_{m,kk} -X[m, k_kk] * dY[m, n]  * scale``

    where ``centered_code = code - zero`` and the sums run over elements in
    quantisation group ``g``.

    Grid: ``(groups, N)`` (column-major) or ``(groups, K)`` (row-major).
    One CTA per ``(group, weight-row)`` pair.

    Note:
        *dY* is expected in float32.  Both *dS* and *dZ* are written in
        *scale_dtype*.

    Args:
        m: Activation batch/row count.
        n: Output channel count.
        k: Input channel count.
        groups: Number of quantisation groups.
        words: Packed ``uint32`` words per weight row.
        transpose: Column-major (``True``) or row-major (``False``) layout.
        group_size: Elements per quantisation group.
        bits: Bits per quantised value.
        x_dtype: Activation dtype (for loading ``X``).
        scale_dtype: Scale/zero dtype (for loading and writing metadata).
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(X, dY, Wq, S, Z, dS, dZ)`` — inputs in their respective dtypes;
        outputs ``dS`` and ``dZ`` in *scale_dtype*.
    """
    xts = _act_dtype_str(x_dtype)
    mts = _act_dtype_str(scale_dtype)
    accum = "float32"

    if transpose:

        @T.prim_func
        def qmm_packed_bwd_meta_col(
            X: T.Tensor((m, k), xts),
            dY: T.Tensor((m, n), accum),
            Wq: T.Tensor((n, words), "uint32"),
            S: T.Tensor((n, groups), mts),
            Z: T.Tensor((n, groups), mts),
            dS: T.Tensor((n, groups), mts),
            dZ: T.Tensor((n, groups), mts),
        ):
            with T.Kernel(groups, n, threads=threads) as (bg, bn):
                x_ref = T.alloc_fragment((1,), xts)
                dim_ref = T.alloc_fragment((1,), accum)
                sum_s = T.alloc_fragment((1,), accum)
                sum_z = T.alloc_fragment((1,), accum)
                x_ref[0] = T.Cast(xts, 0.0)
                dim_ref[0] = T.Cast(accum, k + words)
                sum_s[0] = 0.0
                sum_z[0] = 0.0
                scale = T.Cast(accum, S[bn, bg])
                zero = T.Cast(accum, Z[bn, bg])
                for kk in T.serial(group_size):
                    k_idx = bg * group_size + kk
                    code = T.Cast(accum, _load_packed_code(Wq, bn, k_idx, words, bits))
                    centered = code - zero
                    for mm in T.serial(m):
                        prod = T.Cast(accum, X[mm, k_idx]) * T.Cast(accum, dY[mm, bn])
                        sum_s[0] = sum_s[0] + prod * centered
                        sum_z[0] = sum_z[0] - prod * scale
                dS[bn, bg] = T.Cast(mts, sum_s[0])
                dZ[bn, bg] = T.Cast(mts, sum_z[0])

        return qmm_packed_bwd_meta_col

    @T.prim_func
    def qmm_packed_bwd_meta_row(
        X: T.Tensor((m, k), xts),
        dY: T.Tensor((m, n), accum),
        Wq: T.Tensor((k, words), "uint32"),
        S: T.Tensor((k, groups), mts),
        Z: T.Tensor((k, groups), mts),
        dS: T.Tensor((k, groups), mts),
        dZ: T.Tensor((k, groups), mts),
    ):
        with T.Kernel(groups, k, threads=threads) as (bg, bk):
            x_ref = T.alloc_fragment((1,), xts)
            dim_ref = T.alloc_fragment((1,), accum)
            sum_s = T.alloc_fragment((1,), accum)
            sum_z = T.alloc_fragment((1,), accum)
            x_ref[0] = T.Cast(xts, 0.0)
            dim_ref[0] = T.Cast(accum, n + words)
            sum_s[0] = 0.0
            sum_z[0] = 0.0
            scale = T.Cast(accum, S[bk, bg])
            zero = T.Cast(accum, Z[bk, bg])
            for jj in T.serial(group_size):
                n_idx = bg * group_size + jj
                code = T.Cast(accum, _load_packed_code(Wq, bk, n_idx, words, bits))
                centered = code - zero
                for mm in T.serial(m):
                    prod = T.Cast(accum, X[mm, bk]) * T.Cast(accum, dY[mm, n_idx])
                    sum_s[0] = sum_s[0] + prod * centered
                    sum_z[0] = sum_z[0] - prod * scale
            dS[bk, bg] = T.Cast(mts, sum_s[0])
            dZ[bk, bg] = T.Cast(mts, sum_z[0])

    return qmm_packed_bwd_meta_row


def make_packed_bwd_scale_prim_func(
    *,
    m: int,
    n: int,
    k: int,
    groups: int,
    words: int,
    transpose: bool,
    group_size: int,
    bits: int,
    mode: str,
    x_dtype,
    scale_dtype,
    threads: int = 128,
):
    """Build the non-affine packed scale backward ``@T.prim_func``.

    Currently only implemented for ``mode="nf4"``.  Computes::

        dS[n/k, g] = sum_{m, kk}  X[m, k_kk] * dY[m, n] * nf4_value(code)

    where ``nf4_value(code)`` is the NF4 lookup-table float value.

    Grid: ``(groups, N)`` (column-major) or ``(groups, K)`` (row-major).

    Note:
        *dY* is expected in float32.  ``dS`` is written in *scale_dtype*.

    Args:
        m: Activation batch/row count.
        n: Output channel count.
        k: Input channel count.
        groups: Number of quantisation groups.
        words: Packed ``uint32`` words per weight row.
        transpose: Column-major (``True``) or row-major (``False``) layout.
        group_size: Elements per quantisation group.
        bits: Bits per quantised value (must be 4 for NF4).
        mode: Quantisation mode — currently only ``"nf4"`` is supported.
        x_dtype: Activation dtype (for loading ``X``).
        scale_dtype: Scale metadata dtype.
        threads: CUDA threads per CTA (default 128).

    Returns:
        ``@T.prim_func`` with signature
        ``(X, dY, Wq, S, dS)`` — output ``dS`` in *scale_dtype*.

    Raises:
        ValueError: If *mode* is not ``"nf4"``.
    """
    if mode != "nf4":
        raise ValueError("packed non-affine scale backward is only differentiable for nf4 scales.")

    xts = _act_dtype_str(x_dtype)
    mts = _meta_dtype_str(scale_dtype)
    accum = "float32"

    if transpose:

        @T.prim_func
        def qmm_packed_bwd_scale_col(
            X: T.Tensor((m, k), xts),
            dY: T.Tensor((m, n), accum),
            Wq: T.Tensor((n, words), "uint32"),
            S: T.Tensor((n, groups), mts),
            dS: T.Tensor((n, groups), mts),
        ):
            with T.Kernel(groups, n, threads=threads) as (bg, bn):
                x_ref = T.alloc_fragment((1,), xts)
                meta_ref = T.alloc_fragment((1,), mts)
                dim_ref = T.alloc_fragment((1,), accum)
                sum_s = T.alloc_fragment((1,), accum)
                x_ref[0] = T.Cast(xts, 0.0)
                meta_ref[0] = T.Cast(mts, S[bn, bg])
                dim_ref[0] = T.Cast(accum, k + words)
                sum_s[0] = 0.0
                for kk in T.serial(group_size):
                    k_idx = bg * group_size + kk
                    code = _load_packed_code(Wq, bn, k_idx, words, bits)
                    code_val = _decode_nf4_value(code, accum)
                    for mm in T.serial(m):
                        prod = T.Cast(accum, X[mm, k_idx]) * T.Cast(accum, dY[mm, bn])
                        sum_s[0] = sum_s[0] + prod * code_val
                dS[bn, bg] = T.Cast(mts, sum_s[0])

        return qmm_packed_bwd_scale_col

    @T.prim_func
    def qmm_packed_bwd_scale_row(
        X: T.Tensor((m, k), xts),
        dY: T.Tensor((m, n), accum),
        Wq: T.Tensor((k, words), "uint32"),
        S: T.Tensor((k, groups), mts),
        dS: T.Tensor((k, groups), mts),
    ):
        with T.Kernel(groups, k, threads=threads) as (bg, bk):
            x_ref = T.alloc_fragment((1,), xts)
            meta_ref = T.alloc_fragment((1,), mts)
            dim_ref = T.alloc_fragment((1,), accum)
            sum_s = T.alloc_fragment((1,), accum)
            x_ref[0] = T.Cast(xts, 0.0)
            meta_ref[0] = T.Cast(mts, S[bk, bg])
            dim_ref[0] = T.Cast(accum, n + words)
            sum_s[0] = 0.0
            for jj in T.serial(group_size):
                n_idx = bg * group_size + jj
                code = _load_packed_code(Wq, bk, n_idx, words, bits)
                code_val = _decode_nf4_value(code, accum)
                for mm in T.serial(m):
                    prod = T.Cast(accum, X[mm, bk]) * T.Cast(accum, dY[mm, n_idx])
                    sum_s[0] = sum_s[0] + prod * code_val
            dS[bk, bg] = T.Cast(mts, sum_s[0])

    return qmm_packed_bwd_scale_row
