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

"""Forward CUDA implementation for quantized matrix multiplication.

This module provides :func:`quantized_matmul_forward`, a thin JIT-compiled
wrapper around the low-level :func:`~._cuda_impl.quantized_matmul_cuda` FFI
call. It is used by the custom VJP forward rule in
:mod:`~._interface` and is not intended for direct external use.
"""

from __future__ import annotations

from typing import Literal

from ejkernel.callib._ejit import ejit

from ._cuda_impl import quantized_matmul_cuda

QuantizationMode = Literal["affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"]
GemvMode = Literal["auto", "on", "off"]
RevSplitKMode = Literal["auto", "on", "off"]


@ejit(
    static_argnames=[
        "transpose",
        "group_size",
        "bits",
        "mode",
        "gemv_mode",
        "revsplit_k",
        "revsplit_k_parts",
        "block_n",
        "block_k",
    ]
)
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
    block_n: int = 0,
    block_k: int = 0,
):
    """Execute quantized matmul forward pass via the CUDA FFI custom call.

    This function is JIT-compiled with ``@ejit`` and all quantization
    configuration arguments are marked as static. It simply forwards
    all arguments to :func:`~._cuda_impl.quantized_matmul_cuda`.

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

    Returns:
        Result matrix of shape ``(M, N)`` with the same dtype as *x*.
    """
    return quantized_matmul_cuda(
        x,
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        gemv_mode=gemv_mode,
        revsplit_k=revsplit_k,
        revsplit_k_parts=revsplit_k_parts,
        block_n=block_n,
        block_k=block_k,
    )


__all__ = ("quantized_matmul_forward",)
