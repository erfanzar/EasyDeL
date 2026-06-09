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

"""Quantized matmul interface using XLA.

This module provides a JAX/XLA implementation of quantized matrix
multiplication. It supports the MLX-style packed uint32 layout and the
quantization modes implemented in ejkernel.quantization.

For performance, a blocked decode-and-matmul path is used when the
quantization parameters are compatible with fixed tiling. When the
inputs are not aligned to the tile sizes or the bit-width does not have a
blocked specialization, this falls back to dequantize+matmul.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from ejkernel.callib._ejit import ejit
from ejkernel.quantization._quants.quantizations import dequantize
from ejkernel.quantization._utils.bitpack import _unpack_bits
from ejkernel.quantization._utils.fp_tables import (
    _get_e2m1_table,
    _get_e4m3_table,
    _get_nf4_table,
)
from ejkernel.quantization._utils.qparams import (
    GemvMode,
    QuantizationAxis,
    RevSplitKMode,
    normalize_gemv_mode,
    normalize_revsplitk_mode,
    normalize_revsplitk_parts,
    resolve_qparams,
    resolve_runtime_axis_and_transpose,
    to_backend_mode,
)


def _resolve_qparams(mode: str, group_size: int | None, bits: int | None) -> tuple[str, int, int]:
    """Resolve quantization parameters and convert mode to backend representation.

    Normalizes the user-facing quantization mode string, group size, and bit-width
    into their canonical forms and then converts the mode to the backend-specific key.

    Args:
        mode: User-facing quantization mode string (e.g., "affine", "nf4").
        group_size: Number of elements per quantization group, or None for default.
        bits: Bit-width per quantized element, or None for mode default.

    Returns:
        Tuple of (backend_mode, group_size, bits) where backend_mode is the
        internal quantization key string.
    """
    mode, group_size, bits, _ = resolve_qparams(mode, group_size, bits)
    return to_backend_mode(mode, bits), group_size, bits


def _ceil_div(a: int, b: int) -> int:
    """Compute ceiling division of a by b."""
    return (a + b - 1) // b


def _lcm(a: int, b: int) -> int:
    """Compute least common multiple of two positive integers."""
    if a <= 0:
        return int(b)
    if b <= 0:
        return int(a)
    return abs(a * b) // math.gcd(a, b)


def _bit_aligned_values(bits: int) -> int:
    """Return the value count that starts and ends on a 32-bit word boundary."""
    return 32 // math.gcd(32, int(bits))


def _packed_words_for_values(values: int, bits: int) -> int:
    """Return the number of uint32 words needed for a packed bitstream."""
    return _ceil_div(int(values) * int(bits), 32)


def _pad_2d(x: jax.Array, pad0: int, pad1: int) -> jax.Array:
    """Pad a 2D array with zeros on the right and bottom edges.

    Args:
        x: Input 2D array.
        pad0: Padding for dimension 0 (rows).
        pad1: Padding for dimension 1 (columns).

    Returns:
        Padded array, or original array if no padding needed.
    """
    if pad0 == 0 and pad1 == 0:
        return x
    return jnp.pad(x, ((0, pad0), (0, pad1)))


def _pad_2d_optional(x: jax.Array | None, pad0: int, pad1: int) -> jax.Array | None:
    """Pad a 2D array if not None, otherwise return None."""
    if x is None:
        return None
    return _pad_2d(x, pad0, pad1)


def _decode_tile_affine(
    q: jax.Array,
    scale_tile: jax.Array,
    bias_tile: jax.Array | None,
) -> jax.Array:
    """Decode affine quantized tile using additive-bias form.

    Public affine metadata is ``(q - zero) * scale``. This helper operates on
    the equivalent additive form ``q * scale + bias`` where
    ``bias = -zero * scale``.

    Args:
        q: Quantized codes tile.
        scale_tile: Per-element scales (broadcast from per-group).
        bias_tile: Per-element additive bias terms (broadcast from per-group),
            or None.

    Returns:
        Dequantized weight tile.
    """
    out = q.astype(scale_tile.dtype) * scale_tile
    if bias_tile is not None:
        out = out + bias_tile
    return out


def _decode_tile_nf4(q: jax.Array, scale_tile: jax.Array) -> jax.Array:
    """Decode NF4 quantized tile using codebook lookup.

    Args:
        q: NF4 codes tile (0-15).
        scale_tile: Per-element absmax scales.

    Returns:
        Dequantized weight tile.
    """
    nf4_table = _get_nf4_table()
    vals = nf4_table[q.astype(jnp.int32)]
    return vals * scale_tile


def _decode_tile_mxfp4(q: jax.Array, scale_tile: jax.Array) -> jax.Array:
    """Decode MXFP4 (E2M1) quantized tile with E8M0 shared exponent.

    Args:
        q: E2M1 codes tile (0-15).
        scale_tile: E8M0 shared exponents as uint8.

    Returns:
        Dequantized weight tile.
    """
    e2m1_table, _ = _get_e2m1_table()
    vals = e2m1_table[q.astype(jnp.int32)]
    exp = scale_tile.astype(jnp.int8).astype(jnp.float32)
    scale = jnp.exp2(exp)
    return vals * scale


def _decode_tile_mxfp8(q: jax.Array, scale_tile: jax.Array) -> jax.Array:
    """Decode MXFP8 (E4M3) quantized tile with E8M0 shared exponent.

    Args:
        q: E4M3 codes tile (0-255).
        scale_tile: E8M0 shared exponents as uint8.

    Returns:
        Dequantized weight tile.
    """
    e4m3_table, _ = _get_e4m3_table()
    vals = e4m3_table[q.astype(jnp.int32)]
    exp = scale_tile.astype(jnp.int8).astype(jnp.float32)
    scale = jnp.exp2(exp)
    return vals * scale


def _decode_tile_nvfp4(q: jax.Array, scale_tile: jax.Array) -> jax.Array:
    """Decode NVFP4 (E2M1 with E4M3 scale) quantized tile.

    Args:
        q: E2M1 codes tile (0-15).
        scale_tile: E4M3 per-group scales as uint8.

    Returns:
        Dequantized weight tile.
    """
    e2m1_table, _ = _get_e2m1_table()
    e4m3_table, _ = _get_e4m3_table()
    vals = e2m1_table[q.astype(jnp.int32)]
    scale = e4m3_table[scale_tile.astype(jnp.int32)]
    return vals * scale


def _decode_tile_nvfp8(q: jax.Array, scale_tile: jax.Array) -> jax.Array:
    """Decode NVFP8 (E4M3 with E4M3 scale) quantized tile.

    Args:
        q: E4M3 codes tile (0-255).
        scale_tile: E4M3 per-group scales as uint8.

    Returns:
        Dequantized weight tile.
    """
    e4m3_table, _ = _get_e4m3_table()
    vals = e4m3_table[q.astype(jnp.int32)]
    scale = e4m3_table[scale_tile.astype(jnp.int32)]
    return vals * scale


def _dot_general(a: jax.Array, b: jax.Array) -> jax.Array:
    """Perform matrix multiplication with float32 output accumulation.

    Args:
        a: Left operand of shape (M, K).
        b: Right operand of shape (K, N).

    Returns:
        Result of shape (M, N) in float32.
    """
    return jax.lax.dot_general(
        a,
        b,
        dimension_numbers=(((1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )


def _blocked_quantized_matmul(
    x: jax.Array,
    w_q: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None,
    *,
    transpose: bool,
    group_size: int,
    bits: int,
    mode: str,
    block_m: int,
    block_n: int,
    block_k: int,
    use_bf16: bool,
) -> jax.Array:
    """Execute blocked quantized matmul with fused dequantization.

    This function performs tiled matrix multiplication where each tile's
    weights are dequantized on-the-fly before computing the dot product.
    This avoids materializing the full dequantized weight matrix in memory.

    The algorithm pads inputs to block boundaries, then iterates over tiles
    using nested fori_loops for M, N, and K dimensions.

    Args:
        x: Input activation matrix of shape (M, K).
        w_q: Packed uint32 weights.
        scales: Per-group scales array.
        biases: Per-group affine additive biases (derived from canonical
            ``zeros`` metadata), or None.
        transpose: If True, weights are in NxK layout; if False, KxN layout.
        group_size: Number of elements per quantization group.
        bits: Bit-width per quantized element.
        mode: Backend quantization key
            ("affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8").
        block_m: Block size for M dimension.
        block_n: Block size for N dimension.
        block_k: Block size for K dimension.
        use_bf16: If True, use BF16 for tile computations; if False, FP16.

    Returns:
        Matrix multiplication result of shape (M, N) in float32.

    Raises:
        ValueError: If block sizes are incompatible with group_size or bits.
    """
    mode = mode.lower()
    M, K = x.shape

    if transpose:
        N = w_q.shape[0]
    else:
        N = scales.shape[-1] * group_size

    value_alignment = _bit_aligned_values(bits)

    if transpose:
        if block_k % value_alignment != 0 or block_k % group_size != 0:
            raise ValueError("block_k must be word-aligned and a multiple of group_size for transpose=True")
    else:
        if block_n % value_alignment != 0 or block_n % group_size != 0:
            raise ValueError("block_n must be word-aligned and a multiple of group_size for transpose=False")

    M_pad = _ceil_div(M, block_m) * block_m
    N_pad = _ceil_div(N, block_n) * block_n
    K_pad = _ceil_div(K, block_k) * block_k

    x_pad = _pad_2d(x, M_pad - M, K_pad - K)

    if transpose:
        words_pad = _packed_words_for_values(K_pad, bits)
        w_q_pad = _pad_2d(w_q, N_pad - N, words_pad - w_q.shape[-1])
        groups_pad = K_pad // group_size
        scales_pad = _pad_2d(scales, N_pad - N, groups_pad - scales.shape[-1])
        biases_pad = _pad_2d_optional(biases, N_pad - N, groups_pad - scales.shape[-1])
    else:
        words_pad = _packed_words_for_values(N_pad, bits)
        w_q_pad = _pad_2d(w_q, K_pad - K, words_pad - w_q.shape[-1])
        groups_pad = N_pad // group_size
        scales_pad = _pad_2d(scales, K_pad - K, groups_pad - scales.shape[-1])
        biases_pad = _pad_2d_optional(biases, K_pad - K, groups_pad - scales.shape[-1])

    compute_dtype = jnp.bfloat16 if use_bf16 else jnp.float16
    out = jnp.zeros((M_pad, N_pad), dtype=jnp.float32)

    num_m = M_pad // block_m
    num_n = N_pad // block_n
    num_k = K_pad // block_k

    def load_q_tile(off_k: int, off_n: int) -> jax.Array:
        """Load and unpack a tile of quantized weight codes.

        Slices the packed uint32 weight array and unpacks it into
        individual quantized codes for the tile at (off_k, off_n).

        Args:
            off_k: Offset along the K dimension (reduction axis).
            off_n: Offset along the N dimension (output axis).

        Returns:
            Unpacked quantized codes of shape (block_k, block_n).
        """
        if transpose:
            word_start = off_k * bits // 32
            words_tile = _packed_words_for_values(block_k, bits)
            wq_tile = jax.lax.dynamic_slice(w_q_pad, (off_n, word_start), (block_n, words_tile))
            q = _unpack_bits(wq_tile, block_k, bits)
            q = jnp.swapaxes(q, 0, 1)
        else:
            word_start = off_n * bits // 32
            words_tile = _packed_words_for_values(block_n, bits)
            wq_tile = jax.lax.dynamic_slice(w_q_pad, (off_k, word_start), (block_k, words_tile))
            q = _unpack_bits(wq_tile, block_n, bits)
        return q

    def load_group_tile(off_k: int, off_n: int) -> tuple[jax.Array, jax.Array | None]:
        """Load per-group scales and affine additive biases for a weight tile.

        Slices the scales (and optionally affine additive biases) arrays and repeats
        them to match the tile dimensions for element-wise dequantization.

        Args:
            off_k: Offset along the K dimension.
            off_n: Offset along the N dimension.

        Returns:
            Tuple of (scale_tile, bias_tile) where scale_tile has shape
            (block_k, block_n) and bias_tile is the same shape or None.
        """
        if transpose:
            group_start = off_k // group_size
            groups_tile = block_k // group_size
            scale_tile = jax.lax.dynamic_slice(scales_pad, (off_n, group_start), (block_n, groups_tile))
            scale_tile = jnp.repeat(scale_tile, group_size, axis=1)
            scale_tile = jnp.swapaxes(scale_tile, 0, 1)
            bias_tile = None
            if biases_pad is not None:
                bias_tile = jax.lax.dynamic_slice(biases_pad, (off_n, group_start), (block_n, groups_tile))
                bias_tile = jnp.repeat(bias_tile, group_size, axis=1)
                bias_tile = jnp.swapaxes(bias_tile, 0, 1)
        else:
            group_start = off_n // group_size
            groups_tile = block_n // group_size
            scale_tile = jax.lax.dynamic_slice(scales_pad, (off_k, group_start), (block_k, groups_tile))
            scale_tile = jnp.repeat(scale_tile, group_size, axis=1)
            bias_tile = None
            if biases_pad is not None:
                bias_tile = jax.lax.dynamic_slice(biases_pad, (off_k, group_start), (block_k, groups_tile))
                bias_tile = jnp.repeat(bias_tile, group_size, axis=1)
        return scale_tile, bias_tile

    def decode_tile(off_k: int, off_n: int) -> jax.Array:
        """Dequantize a single weight tile using the configured quantization mode.

        Loads packed codes and group parameters, then applies the
        mode-specific decoding formula (affine/nf4/mxfp*/nvfp*).
        to produce a float tile.

        Args:
            off_k: Offset along the K dimension.
            off_n: Offset along the N dimension.

        Returns:
            Dequantized weight tile of shape (block_k, block_n) in compute_dtype.
        """
        q = load_q_tile(off_k, off_n)
        scale_tile, bias_tile = load_group_tile(off_k, off_n)
        if mode == "affine":
            w = _decode_tile_affine(q, scale_tile, bias_tile)
        elif mode == "nf4":
            w = _decode_tile_nf4(q, scale_tile)
        elif mode == "mxfp4":
            w = _decode_tile_mxfp4(q, scale_tile)
        elif mode == "mxfp8":
            w = _decode_tile_mxfp8(q, scale_tile)
        elif mode == "nvfp4":
            w = _decode_tile_nvfp4(q, scale_tile)
        elif mode == "nvfp8":
            w = _decode_tile_nvfp8(q, scale_tile)
        else:
            raise ValueError(f"Unsupported quantization mode: {mode}")
        if use_bf16:
            return w.astype(jnp.float16).astype(compute_dtype)
        return w.astype(compute_dtype)

    def k_loop(k_idx: int, acc: jax.Array, *, off_m: int, off_n: int) -> jax.Array:
        """Accumulate one K-dimension tile into the output accumulator.

        Loads the activation tile and dequantized weight tile, performs
        a dot product, and adds the result to the running accumulator.

        Args:
            k_idx: K-dimension block index.
            acc: Running accumulator of shape (block_m, block_n) in float32.
            off_m: M-dimension offset for the current tile.
            off_n: N-dimension offset for the current tile.

        Returns:
            Updated accumulator with this K-tile's contribution added.
        """
        off_k = k_idx * block_k
        x_tile = jax.lax.dynamic_slice(x_pad, (off_m, off_k), (block_m, block_k))
        if x_pad.dtype != jnp.float32:
            x_tile = x_tile.astype(compute_dtype)
        w_tile = decode_tile(off_k, off_n)
        acc = acc + _dot_general(x_tile, w_tile)
        return acc

    def n_loop(n_idx: int, out_buf: jax.Array) -> jax.Array:
        """Process one N-dimension block column across all M rows.

        Iterates over all M-dimension blocks and accumulates the full
        K-dimension reduction for each (M, N) output tile.

        Args:
            n_idx: N-dimension block index.
            out_buf: Output buffer of shape (M_pad, N_pad) being filled.

        Returns:
            Updated output buffer with this N column computed.
        """
        off_n = n_idx * block_n

        k_ids = jnp.arange(num_k, dtype=jnp.int32)

        def _decode_k(k_idx: jax.Array) -> jax.Array:
            off_k = k_idx * block_k
            return decode_tile(off_k, off_n)

        w_tiles = jax.lax.map(_decode_k, k_ids)

        def m_loop(m_idx: int, out_local: jax.Array) -> jax.Array:
            """Process one M-dimension block row for a fixed N column.

            Accumulates the dot product over all K-dimension tiles and
            writes the result to the output buffer.

            Args:
                m_idx: M-dimension block index.
                out_local: Output buffer being updated.

            Returns:
                Updated output buffer with this (M, N) tile written.
            """
            off_m = m_idx * block_m
            acc = jnp.zeros((block_m, block_n), dtype=jnp.float32)

            def k_body(idx: int, carry: jax.Array) -> jax.Array:
                """Inner loop body for K-dimension accumulation.

                Args:
                    idx: K-dimension block index.
                    carry: Running accumulator.

                Returns:
                    Updated accumulator.
                """
                off_k = idx * block_k
                x_tile = jax.lax.dynamic_slice(x_pad, (off_m, off_k), (block_m, block_k))
                if x_pad.dtype != jnp.float32:
                    x_tile = x_tile.astype(compute_dtype)
                w_tile = w_tiles[idx]
                return carry + _dot_general(x_tile, w_tile)

            acc = jax.lax.fori_loop(0, num_k, k_body, acc)
            out_local = jax.lax.dynamic_update_slice(out_local, acc, (off_m, off_n))
            return out_local

        out_buf = jax.lax.fori_loop(0, num_m, m_loop, out_buf)
        return out_buf

    out = jax.lax.fori_loop(0, num_n, n_loop, out)
    return out[:M, :N]


@ejit(
    static_argnames=[
        "transpose",
        "group_size",
        "bits",
        "mode",
        "block_m",
        "block_n",
        "block_k",
        "use_bf16",
        "allow_dense_fallback",
        "gemv_mode",
        "revsplit_k",
        "revsplit_k_parts",
    ],
)
def _operate(
    x,
    w,
    scales,
    biases,
    transpose,
    group_size,
    bits,
    mode,
    block_m,
    block_n,
    block_k,
    use_bf16,
    allow_dense_fallback,
    gemv_mode,
    revsplit_k,
    revsplit_k_parts,
):
    """Execute quantized matmul with automatic path selection.

    Attempts the blocked fused dequantize-and-matmul path first for
    4-bit and 8-bit modes with valid block sizes. If the fused path
    raises a ValueError (e.g., due to shape or tiling mismatches),
    falls back to a two-step dequantize+matmul approach.

    Args:
        x: Activation matrix ``[M, K]``.
        w: Packed uint32 weights.
        scales: Per-group scale parameters.
        biases: Per-group affine additive biases (``-zero * scale``), or
            ``None`` for non-affine modes.
        transpose: ``True`` → NxK weight layout; ``False`` → KxN.
        group_size: Elements per quantization group.
        bits: Bit-width per quantized element.
        mode: Backend quantization key (already normalised by ``_resolve_qparams``).
        block_m: M-dimension tile size.
        block_n: N-dimension tile size.
        block_k: K-dimension tile size.
        use_bf16: If ``True``, intermediate tiles use BF16; otherwise FP16.
        allow_dense_fallback: If ``False``, raises instead of falling through to
            the dense dequantize+matmul path.
        gemv_mode: Accepted for API compatibility; ignored (deleted on entry).
        revsplit_k: Accepted for API compatibility; ignored (deleted on entry).
        revsplit_k_parts: Accepted for API compatibility; ignored (deleted on entry).

    Returns:
        Result matrix ``[M, N]`` in float32.
    """
    del gemv_mode, revsplit_k, revsplit_k_parts

    can_fuse = bits in (4, 8)
    can_fuse = can_fuse and block_m > 0 and block_n > 0 and block_k > 0

    if can_fuse:
        try:
            return _blocked_quantized_matmul(
                x,
                w,
                scales,
                biases,
                transpose=transpose,
                group_size=group_size,
                bits=bits,
                mode=mode,
                block_m=block_m,
                block_n=block_n,
                block_k=block_k,
                use_bf16=use_bf16,
            )
        except ValueError as e:
            if not allow_dense_fallback:
                raise ValueError(
                    "XLA blocked quantized_matmul preconditions failed and dense dequantize+matmul fallback is "
                    "disabled (allow_dense_fallback=False)."
                ) from e

    if not allow_dense_fallback:
        raise ValueError(
            "XLA blocked quantized_matmul requires a legal blocked configuration; "
            "dense dequantize+matmul fallback is disabled (allow_dense_fallback=False)."
        )

    zeros = None
    if mode == "affine":
        if biases is None:
            raise ValueError("affine fallback dequantization requires affine metadata.")
        safe_scale = jnp.where(scales == 0, jnp.ones_like(scales), scales)
        zeros = -biases / safe_scale
    w_f = dequantize(w, scales, zeros, group_size=group_size, bits=bits, mode=mode)
    return x @ w_f.T if transpose else x @ w_f


def quantized_matmul(
    x: Float[Array, "m k"],
    w: Array,
    scales: Array,
    zeros: Array | None = None,
    transpose: bool = False,
    group_size: int | None = None,
    bits: int | None = None,
    mode: str = "affine",
    axis: QuantizationAxis | None = None,
    gemv_mode: GemvMode = "auto",
    revsplit_k: RevSplitKMode = "auto",
    revsplit_k_parts: int | None = None,
    *,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    use_bf16: bool = True,
    allow_dense_fallback: bool = True,
    num_warps: int | None = None,
    num_stages: int | None = None,
    split_k: int | None = None,
) -> Float[Array, "m n"]:
    """Compute quantized matrix multiplication using JAX/XLA.

    Attempts a blocked fused dequantize-and-matmul path for 4-bit and 8-bit
    modes, then falls back to a two-step ``dequantize + x @ w`` approach when
    the fused path's preconditions are not met (or when ``allow_dense_fallback``
    is ``True`` and the fused path raises ``ValueError``).

    For ``transpose=True`` the ``block_k`` tile is automatically rounded up to
    the nearest multiple of both the packed-word value alignment and
    ``group_size`` before JIT compilation if the default value is not already
    compatible.

    Args:
        x: Activation matrix ``[M, K]`` in a float dtype.
        w: Packed uint32 weights.  Shape:
            ``[N, ceil(K * bits / 32)]`` when ``transpose=True``, or
            ``[K, ceil(N * bits / 32)]`` when ``transpose=False``.
        scales: Per-group scales.  Shape and dtype depend on mode:

            - ``affine``/``nf4``: float, shape ``[N, K//gs]`` (T) or ``[K, N//gs]``.
            - ``mxfp4``/``mxfp8``: ``uint8`` E8M0 shared exponents.
            - ``nvfp4``/``nvfp8``: ``uint8`` E4M3 per-group scales.
        zeros: Per-group affine zero-points (canonical ``(q - zero) * scale``
            form).  Must match ``scales`` shape.  ``None`` for non-affine modes.
            Internally converted to additive bias ``-zero * scale``.
        transpose: If ``True``, weights are in NxK layout (compute ``x @ w.T``).
            If ``False``, weights are in KxN layout (compute ``x @ w``).
        group_size: Quantization group size.  Mode defaults:
            64 (affine/nf4), 32 (mxfp4/mxfp8), 16 (nvfp4/nvfp8).
        bits: Bit-width per element. Honoured for affine bits 1 through 8;
            ignored for other explicit modes.
        mode: User-facing quantization mode string (see ``_resolve_qparams`` for
            normalisation).
        axis: Optional ``QuantizationAxis``; overrides the effective
            ``transpose`` direction via ``resolve_runtime_axis_and_transpose``.
        gemv_mode: ``GemvMode`` hint.  Accepted but ignored on this path.
        revsplit_k: ``RevSplitKMode`` hint.  Accepted but ignored on this path.
        revsplit_k_parts: Split-K partition count.  Accepted but ignored.
        block_m: M-dimension tile size (default 128).
        block_n: N-dimension tile size (default 128).
        block_k: K-dimension tile size (default 64; auto-adjusted for
            ``transpose=True`` to satisfy packing/group-size constraints).
        use_bf16: ``True`` → intermediate tiles in BF16; ``False`` → FP16.
        allow_dense_fallback: If ``False``, raises instead of falling back to
            the dense dequantize+matmul path.
        num_warps: Triton-only; accepted for API compat, ignored here.
        num_stages: Triton-only; accepted for API compat, ignored here.
        split_k: Triton-only; accepted for API compat, ignored here.

    Returns:
        Result matrix ``[M, N]`` in float32.

    Raises:
        ValueError: If ``mode == "affine"`` and ``zeros`` is ``None``.
        ValueError: If ``mode != "affine"`` and ``zeros`` is not ``None``.
        ValueError: If ``allow_dense_fallback=False`` and the blocked fused
            path cannot be executed.
    """
    mode, group_size, bits = _resolve_qparams(mode, group_size, bits)
    _, transpose = resolve_runtime_axis_and_transpose(axis=axis, transpose=transpose)
    gemv_mode = normalize_gemv_mode(gemv_mode)
    revsplit_k = normalize_revsplitk_mode(revsplit_k)
    revsplit_k_parts = normalize_revsplitk_parts(revsplit_k_parts)

    if transpose:
        value_alignment = _bit_aligned_values(int(bits))
        if block_k % value_alignment != 0 or block_k % int(group_size) != 0:
            block_k = _lcm(int(block_k), int(value_alignment))
            block_k = _lcm(int(block_k), int(group_size))

    if mode == "affine":
        if zeros is None:
            raise ValueError("affine quantized_matmul requires `zeros`.")
        safe_scale = jnp.where(scales == 0, jnp.ones_like(scales), scales)
        affine_biases = -zeros * safe_scale
    else:
        if zeros is not None:
            raise ValueError("zeros must be None for non-affine modes.")
        affine_biases = None

    return _operate(
        x,
        w,
        scales,
        affine_biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        use_bf16=use_bf16,
        allow_dense_fallback=bool(allow_dense_fallback),
        gemv_mode=gemv_mode,
        revsplit_k=revsplit_k,
        revsplit_k_parts=revsplit_k_parts,
    )
