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

"""Forward pass for the Triton quantized matmul backend.

This module implements ``quantized_matmul_forward`` as a clean, stateless
JIT-compiled wrapper around ``quantized_matmul_triton``.  Large square shapes
may optionally take the stateless two-stage dequantize-then-GEMM path inside
``quantized_matmul_triton`` controlled by pre-existing environment settings
(``EJKERNEL_QMM_TWO_STAGE`` and ``EJKERNEL_QMM_MATMUL_PRECISION``).
"""

from __future__ import annotations

from typing import Literal

from ejkernel.callib._ejit import ejit

from ._triton_impl import quantized_matmul_triton

QuantizationMode = Literal["affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"]
GemvMode = Literal["auto", "on", "off"]
RevSplitKMode = Literal["auto", "on", "off"]


@ejit(
    static_argnames=(
        "transpose",
        "group_size",
        "bits",
        "mode",
        "block_m",
        "block_n",
        "block_k",
        "use_bf16",
        "num_warps",
        "num_stages",
        "split_k",
        "gemv_mode",
        "revsplit_k",
        "revsplit_k_parts",
    )
)
def _quantized_matmul_triton_jit(
    x,
    w,
    scales,
    biases,
    *,
    transpose: bool,
    group_size: int | None,
    bits: int | None,
    mode: QuantizationMode,
    block_m: int,
    block_n: int,
    block_k: int,
    use_bf16: bool,
    num_warps: int | None,
    num_stages: int | None,
    split_k: int | None,
    gemv_mode: GemvMode,
    revsplit_k: RevSplitKMode,
    revsplit_k_parts: int | None,
):
    return quantized_matmul_triton(
        x,
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        use_bf16=use_bf16,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        split_k=split_k,
        gemv_mode=gemv_mode,
        revsplit_k=revsplit_k,
        revsplit_k_parts=revsplit_k_parts,
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
    block_m: int,
    block_n: int,
    block_k: int,
    use_bf16: bool,
    num_warps: int | None,
    num_stages: int | None,
    split_k: int | None,
    gemv_mode: GemvMode,
    revsplit_k: RevSplitKMode,
    revsplit_k_parts: int | None,
):
    """Execute quantized matmul forward pass using the stateless Triton path.

    Invokes ``quantized_matmul_triton`` through a module-level ``@ejit`` wrapper
    without any global array cache, avoiding JAX tracer leaks and Python
    ``id()`` reuse bugs while preserving ``x.dtype`` on output.

    Args:
        x: Activation matrix, shape (M, K).
        w: Packed quantized weight.
        scales: Per-group scale factors.
        biases: Per-group additive biases (affine mode) or ``None``.
        transpose: Weight transposition flag.
        group_size: Quantization group size.
        bits: Bits per quantized value.
        mode: Backend quantization mode string.
        block_m, block_n, block_k: Triton tile sizes.
        use_bf16: Use BF16 accumulation tiles.
        num_warps: Triton warps per program (``None`` = default).
        num_stages: Triton pipeline stages (``None`` = default).
        split_k: Split-K factor (``None`` = default).
        gemv_mode: GEMV kernel selection mode.
        revsplit_k: Reverse split-K mode.
        revsplit_k_parts: Reverse split-K parts.

    Returns:
        Output of shape (M, N) with the same dtype as ``x``.
    """
    out = _quantized_matmul_triton_jit(
        x,
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode.lower(),
        use_bf16=use_bf16,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
        split_k=split_k,
        gemv_mode=gemv_mode,
        revsplit_k=revsplit_k,
        revsplit_k_parts=revsplit_k_parts,
    )
    return out.astype(x.dtype)


__all__ = ("quantized_matmul_forward",)
