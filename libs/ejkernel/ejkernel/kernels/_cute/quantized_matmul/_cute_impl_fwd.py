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

"""Forward CuTe implementation for quantized matrix multiplication.

This module provides :func:`quantized_matmul_forward`, which resolves
quantization parameters, validates inputs, and dispatches to the CuTe DSL
fused kernel built by :func:`~._cute_impl.get_cute_qmm_call`. It is used
by the custom VJP forward rule in :mod:`~._interface`.
"""

from __future__ import annotations

from typing import Literal

from ejkernel.quantization._utils.qparams import resolve_qparams

from ._cute_impl import get_cute_qmm_call

QuantizationMode = Literal["affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"]
GemvMode = Literal["auto", "on", "off"]
RevSplitKMode = Literal["auto", "on", "off"]


def quantized_matmul_forward(
    x,
    w,
    scales,
    biases,
    *,
    transpose: bool,
    group_size: int | None,
    bits: int | None,
    mode: QuantizationMode,
    gemv_mode: GemvMode,
    revsplit_k: RevSplitKMode,
    revsplit_k_parts: int | None,
    block_m: int,
    block_n: int,
    block_k: int,
    use_bf16: bool,
):
    """Execute quantized matmul forward pass via the CuTe DSL fused kernel.

    Resolves quantization parameters, validates that the mode and biases
    are consistent, and dispatches to the cached CuTe DSL callable.

    Args:
        x: Input activation matrix of shape ``(M, K)``.
        w: Packed quantized weight matrix.
        scales: Per-group scale factors.
        biases: Per-group additive offsets (affine mode only).
        transpose: Whether the weight layout is transposed.
        group_size: Quantization group size.
        bits: Bit-width per quantized element.
        mode: Backend quantization mode string.
        gemv_mode: GEMV dispatch mode.
        revsplit_k: Reverse split-K dispatch mode.
        revsplit_k_parts: Number of split-K partitions.
        block_m: Tile size along the M dimension.
        block_n: Tile size along the N dimension.
        block_k: Tile size along the K dimension.
        use_bf16: Whether to use bfloat16 accumulation.

    Returns:
        Result matrix of shape ``(M, N)`` with the same dtype as *x*.

    Raises:
        ValueError: If biases are missing for affine mode or present for
            non-affine modes, or if the input is not on GPU.
    """
    mode_lower = mode.lower()
    _, group_size_resolved, bits_resolved, _ = resolve_qparams(mode_lower, group_size, bits)

    if mode_lower == "affine" and biases is None:
        raise ValueError("affine quantized_matmul requires affine metadata.")
    if mode_lower != "affine" and biases is not None:
        raise ValueError("affine metadata must be None for non-affine modes.")

    dev = None
    try:
        dev = x.device()
    except Exception:
        dev = None
    if dev is not None and getattr(dev, "platform", None) != "gpu":
        raise ValueError("CUTE quantized_matmul requires GPU backend.")

    call = get_cute_qmm_call(
        x=x,
        w_q=w,
        scales=scales,
        biases=biases,
        mode=mode_lower,
        bits=bits_resolved,
        group_size=group_size_resolved,
        transpose=transpose,
        gemv_mode=gemv_mode,
        revsplit_k=revsplit_k,
        revsplit_k_parts=revsplit_k_parts,
        use_bf16=use_bf16,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
    )
    if biases is not None:
        return call(x, w, scales, biases)
    return call(x, w, scales)


__all__ = ("quantized_matmul_forward",)
