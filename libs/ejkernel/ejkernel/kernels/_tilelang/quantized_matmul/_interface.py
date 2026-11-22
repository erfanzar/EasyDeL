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

"""TileLang quantized matmul public interface.

Registers ``quantized_matmul`` for ``Platform.TILELANG / Backend.GPU``.

Routing logic (in priority order):
1. **Legacy int8 symmetric path**: ``mode="affine"``, ``w.dtype=int8``,
   ``zeros=None``, ``scales.ndim=1`` → calls ``quantized_matmul_tilelang``.
2. **Packed non-affine path**: ``mode in {"nf4","mxfp4","mxfp8","nvfp4","nvfp8"}``
   → calls ``quantized_matmul_packed_nonaffine_tilelang``.
3. **Packed affine path**: ``mode="affine"``, packed layout, ``zeros`` required
   → calls ``quantized_matmul_packed_tilelang``.

Input validation, axis/transpose normalisation and qparam resolution are
delegated to the shared utilities in
``ejkernel.quantization._utils.qparams``.
"""

from __future__ import annotations

import jaxtyping
from beartype import beartype
from jaxtyping import Array, Float

from ejkernel.errors import EjkernelRuntimeError
from ejkernel.quantization._utils.qparams import (
    GemvMode,
    QuantizationAxis,
    RevSplitKMode,
    normalize_gemv_mode,
    normalize_revsplitk_mode,
    normalize_revsplitk_parts,
    resolve_qparams,
    resolve_runtime_axis_and_transpose,
    validate_packed_quantized_matmul_layout,
)

from ..._registry import Backend, Platform, kernel_registry
from ._impl import (
    quantized_matmul_packed_nonaffine_tilelang,
    quantized_matmul_packed_tilelang,
    quantized_matmul_tilelang,
)


@kernel_registry.register("quantized_matmul", Platform.TILELANG, Backend.GPU)
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
    """TileLang GPU quantized matmul dispatcher.

    Resolves quantisation parameters and dispatches to the appropriate kernel
    variant.  Supports both the legacy int8 symmetric interface and the packed
    multi-bit interface.

    Args:
        x: ``[m, k]`` float activation matrix (fp16 / bf16 / fp32).
        w: Weight matrix.  For the legacy path: ``[n, k]`` int8.  For packed
            paths: ``uint32`` packed tensor.
        scales: Per-group or per-channel scale.  Legacy path: ``[n]`` rank-1.
            Packed paths: ``[n, groups]`` or ``[k, groups]``.
        zeros: Per-group zero-point for affine quantisation.  Required for the
            packed affine path; ``None`` for the legacy path and non-affine
            modes.
        transpose: If ``True``, weight is packed in column-major
            (output-channel indexed) layout.
        group_size: Quantisation group size.  Resolved to a mode-specific
            default when ``None``.
        bits: Bits per quantised value.  Resolved to a mode-specific default
            when ``None``.
        mode: Quantisation mode.  One of ``"affine"`` (default), ``"nf4"``,
            ``"mxfp4"``, ``"mxfp8"``, ``"nvfp4"``, ``"nvfp8"``.
        axis: Quantisation axis — ``"row"`` (default) or ``"col"``.  Affects
            which dimension the weights are indexed along.
        gemv_mode: GeMV hint (``"auto"`` / ``"on"`` / ``"off"``); accepted for
            API compatibility but currently ignored by this backend.
        revsplit_k: Reverse split-K hint; accepted but currently ignored.
        revsplit_k_parts: Reverse split-K part count; accepted but currently
            ignored.
        tpu_path: Not supported on the TileLang GPU backend; raises
            ``EjkernelRuntimeError`` if provided.
        allow_dense_fallback: Accepted for API compatibility; currently
            ignored.
        block_m: M-axis tile size (default 128; clamped internally).
        block_n: N-axis tile size (default 128; clamped internally).
        block_k: K-axis tile size (default 64; clamped internally).
        use_bf16: Use bfloat16 compute when activations are bfloat16.
        num_warps: Warp count per CTA; ``None`` defaults to 4 (128 threads).
        num_stages: Pipeline stages; ``None`` defaults to 2.
        split_k: Split-K factor.  Only ``None`` and ``1`` are supported;
            raises ``EjkernelRuntimeError`` for other values.

    Returns:
        ``[m, n]`` output tensor.  The dtype is *x*'s dtype for the legacy
        int8 path; float32 for all packed paths.

    Raises:
        EjkernelRuntimeError: On unsupported options or invalid layouts.
    """
    mode_n, group_size_n, bits_n, _ = resolve_qparams(mode, group_size, bits)
    runtime_axis, transpose_n = resolve_runtime_axis_and_transpose(axis=axis, transpose=transpose)
    normalize_gemv_mode(gemv_mode)
    normalize_revsplitk_mode(revsplit_k)
    normalize_revsplitk_parts(revsplit_k_parts)

    if tpu_path is not None:
        raise EjkernelRuntimeError("tile-lang quantized_matmul on TileLang GPU does not support `tpu_path`.")
    if split_k not in (None, 1):
        raise EjkernelRuntimeError("tile-lang quantized_matmul does not yet support split_k > 1.")
    if block_m <= 0 or block_n <= 0 or block_k <= 0:
        raise EjkernelRuntimeError("tile-lang quantized_matmul block sizes must be positive.")
    if num_stages is not None and num_stages <= 0:
        raise EjkernelRuntimeError("tile-lang quantized_matmul num_stages must be positive when provided.")
    if num_warps is not None and num_warps <= 0:
        raise EjkernelRuntimeError("tile-lang quantized_matmul num_warps must be positive when provided.")
    legacy_int8 = mode_n == "affine" and str(w.dtype) == "int8" and zeros is None and scales.ndim == 1
    if legacy_int8:
        if bits_n != 8 or transpose_n or runtime_axis != "row":
            raise EjkernelRuntimeError(
                "tile-lang legacy int8 quantized_matmul requires bits=8, transpose=False and axis='row'."
            )
        return quantized_matmul_tilelang(x, w, scales)

    if mode_n == "affine" and zeros is None:
        raise EjkernelRuntimeError("affine tile-lang quantized_matmul requires packed `zeros` metadata.")
    if x.ndim != 2:
        raise EjkernelRuntimeError(f"tile-lang quantized_matmul expects rank-2 x, got shape {x.shape}.")
    validate_packed_quantized_matmul_layout(
        x,
        w,
        scales,
        zeros,
        mode=mode_n,
        group_size=group_size_n,
        bits=bits_n,
        axis=runtime_axis,
        transpose=transpose_n,
    )

    _ = allow_dense_fallback
    if mode_n != "affine":
        return quantized_matmul_packed_nonaffine_tilelang(
            x,
            w,
            scales,
            mode=mode_n,
            transpose=transpose_n,
            bits=bits_n,
            group_size=group_size_n,
            use_bf16=use_bf16,
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_stages=num_stages,
            num_warps=num_warps,
        )

    return quantized_matmul_packed_tilelang(
        x,
        w,
        scales,
        zeros,
        transpose=transpose_n,
        bits=bits_n,
        group_size=group_size_n,
        use_bf16=use_bf16,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_stages=num_stages,
        num_warps=num_warps,
    )


__all__ = ["quantized_matmul"]
