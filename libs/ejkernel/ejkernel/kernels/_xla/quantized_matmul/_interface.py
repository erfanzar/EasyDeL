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
"""Registry entry point for the XLA quantized matrix multiplication kernel.

Registers ``quantized_matmul`` under ``(Platform.XLA, Backend.ANY)`` and
delegates to the JAX/XLA blocked-dequantize implementation in
``_xla_impl_fwd``.  The ``tpu_path`` keyword is accepted for API
compatibility but is discarded before dispatch.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype

from ..._registry import Backend, Platform, kernel_registry
from ._xla_impl_fwd import Array, Float, GemvMode, QuantizationAxis, RevSplitKMode
from ._xla_impl_fwd import quantized_matmul as _quantized_matmul_impl


@kernel_registry.register("quantized_matmul", Platform.XLA, Backend.ANY)
@jaxtyping.jaxtyped(typechecker=beartype)
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
    tpu_path: str | None = None,
    allow_dense_fallback: bool = True,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
    use_bf16: bool = True,
    num_warps: int | None = None,
    num_stages: int | None = None,
    split_k: int | None = None,
) -> Float[Array, "m n"]:
    """Compute quantized matrix multiplication using XLA (registry entry point).

    Registry wrapper; see ``_xla_impl_fwd.quantized_matmul`` for the full
    algorithm description.

    Args:
        x: Activation matrix ``[M, K]`` in a float dtype.
        w: Packed uint32 weights.  Shape:
            ``[N, ceil(K / values_per_word)]`` when ``transpose=True``, or
            ``[K, ceil(N / values_per_word)]`` when ``transpose=False``.
        scales: Per-group scale parameters.  Shape and dtype depend on mode:
            ``float`` for affine/nf4; ``uint8`` E8M0 for mxfp*; ``uint8`` E4M3
            for nvfp*.
        zeros: Per-group zero-points (affine mode only).  Must be the same
            shape as ``scales`` and must be ``None`` for non-affine modes.
        transpose: ``True`` → weights are NxK (compute ``x @ w.T``);
            ``False`` → weights are KxN (compute ``x @ w``).
        group_size: Elements per quantization group.  Defaults:
            64 (affine/nf4), 32 (mxfp4/mxfp8), 16 (nvfp4/nvfp8).
        bits: Bit-width per quantized element.  Honoured for affine
            (``{4, 8}``); ignored for other explicit modes.
        mode: Dequantization formula:
            ``"affine"`` → ``(q - zero) * scale``;
            ``"nf4"`` → NF4 codebook;
            ``"mxfp4"``/``"mxfp8"`` → Microscaling FP4/FP8;
            ``"nvfp4"``/``"nvfp8"`` → NVIDIA Microscaling FP4/FP8.
        axis: Optional ``QuantizationAxis`` enum controlling the quantization
            axis; used to determine the effective ``transpose`` direction.
        gemv_mode: ``GemvMode`` hint for GEMV-specialised kernels.  Accepted
            for API compatibility; ignored on the XLA path.
        revsplit_k: ``RevSplitKMode`` hint.  Accepted for API compatibility;
            ignored on the XLA path.
        revsplit_k_parts: Number of split-K partitions.  Accepted for API
            compatibility; ignored on the XLA path.
        tpu_path: Accepted for API compatibility; discarded before dispatch.
        allow_dense_fallback: If ``False``, raises when the blocked fused path
            is illegal instead of silently falling back to dequantize+matmul.
        block_m: Tile size for the M dimension.  Default ``128``.
        block_n: Tile size for the N dimension.  Default ``128``.
        block_k: Tile size for the K dimension.  Default ``64``.  Automatically
            adjusted upward when ``transpose=True`` and the tile is not
            compatible with ``group_size`` or ``values_per_word``.
        use_bf16: ``True`` → intermediate tiles in BF16; ``False`` → FP16.
        num_warps: Triton-only; ignored on the XLA path.
        num_stages: Triton-only; ignored on the XLA path.
        split_k: Triton-only; ignored on the XLA path.

    Returns:
        Result matrix ``[M, N]`` in float32.

    Raises:
        ValueError: If ``mode == "affine"`` and ``zeros`` is ``None``.
        ValueError: If ``mode != "affine"`` and ``zeros`` is not ``None``.
        ValueError: If ``allow_dense_fallback=False`` and the blocked fused
            path preconditions are not satisfied.
    """
    del tpu_path
    return _quantized_matmul_impl(
        x,
        w,
        scales,
        zeros,
        transpose,
        group_size,
        bits,
        mode,
        axis,
        gemv_mode,
        revsplit_k,
        revsplit_k_parts,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        use_bf16=use_bf16,
        allow_dense_fallback=allow_dense_fallback,
        num_warps=num_warps,
        num_stages=num_stages,
        split_k=split_k,
    )


__all__ = ("quantized_matmul",)
