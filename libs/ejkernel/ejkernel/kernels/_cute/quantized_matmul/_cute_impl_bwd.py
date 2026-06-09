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

"""Backward CuTe implementation for quantized matrix multiplication.

This module provides :func:`quantized_matmul_input_grad`, which computes
the gradient of the loss with respect to the dense input activation *x*.
It first attempts to use the CuTe QMM forward kernel with flipped transpose
semantics; if that fails (e.g., due to unsupported layout), it falls back
to exact dequantization followed by ``jax.lax.dot_general``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ejkernel.quantization import dequantize

from ._cute_impl_fwd import quantized_matmul_forward


def quantized_matmul_input_grad(
    dy,
    w,
    scales,
    biases,
    *,
    transpose: bool,
    group_size: int | None,
    bits: int | None,
    mode: str,
    gemv_mode: str,
    revsplit_k: str,
    revsplit_k_parts: int | None,
    block_m: int,
    block_n: int,
    block_k: int,
    use_bf16: bool,
):
    """Compute gradient of the loss with respect to the dense input *x*.

    First attempts to reuse the CuTe QMM forward kernel by flipping the
    transpose flag. If that raises a ``ValueError`` (e.g., the transposed
    layout is unsupported for the given mode), the function falls back to
    dequantizing the packed weights to full precision and performing the
    transpose matmul via ``jax.lax.dot_general``.

    Args:
        dy: Upstream gradient of shape ``(M, N)``.
        w: Packed quantized weight matrix.
        scales: Per-group scale factors.
        biases: Per-group additive offsets (affine mode only).
        transpose: Whether the weight layout is transposed.
        group_size: Quantization group size.
        bits: Bit-width per quantized element.
        mode: Backend quantization mode string.
        gemv_mode: GEMV dispatch mode (unused in fallback path).
        revsplit_k: Reverse split-K dispatch mode (unused in fallback path).
        revsplit_k_parts: Split-K partitions (unused in fallback path).
        block_m: Tile size along the M dimension.
        block_n: Tile size along the N dimension.
        block_k: Tile size along the K dimension.
        use_bf16: Whether to use bfloat16 accumulation.

    Returns:
        Gradient tensor with shape ``(M, K)`` and dtype ``float32``.

    Raises:
        ValueError: If *biases* is ``None`` when mode is ``"affine"``.
    """
    try:
        return quantized_matmul_forward(
            dy,
            w,
            scales,
            biases,
            transpose=not transpose,
            group_size=group_size,
            bits=bits,
            mode=mode,
            gemv_mode=gemv_mode,
            revsplit_k=revsplit_k,
            revsplit_k_parts=revsplit_k_parts,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            use_bf16=use_bf16,
        )
    except ValueError:
        pass

    del gemv_mode, revsplit_k, revsplit_k_parts
    zeros = None
    if mode == "affine":
        if biases is None:
            raise ValueError("affine input grad requires affine metadata.")
        safe_scale = jnp.where(scales == 0, jnp.ones_like(scales), scales)
        zeros = -biases / safe_scale
    w_f = dequantize(w, scales, zeros, group_size=group_size, bits=bits, mode=mode)
    if transpose:
        dims = (((1,), (0,)), ((), ()))
    else:
        dims = (((1,), (1,)), ((), ()))
    return jax.lax.dot_general(dy, w_f, dims, preferred_element_type=jnp.float32)


__all__ = ("quantized_matmul_input_grad",)
