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

This module implements ``quantized_matmul_forward``, which chooses between
two execution paths depending on problem shape and environment variables:

Two-stage path (``EJKERNEL_QMM_TWO_STAGE=1``, enabled by default):
    For shapes where M, N, K >= 4096, the weight is first dequantized once
    (via ``quantized_matmul_dequant_triton``) and then multiplied against the
    activation using ``jax.lax.dot_general``.  The dequantized weight tensor
    can be cached across calls via an LRU dict (``EJKERNEL_QMM_DEQUANT_CACHE``).

Fused kernel path (fallback or when ``EJKERNEL_QMM_TWO_STAGE=0``):
    Invokes ``quantized_matmul_triton`` which fuses dequantization and matmul
    in a single autotuned Triton kernel.

Environment variables:

* ``EJKERNEL_QMM_TWO_STAGE`` (default ``"1"``): enable two-stage path.
* ``EJKERNEL_QMM_DEQUANT_CACHE`` (default ``"1"``): enable weight cache.
* ``EJKERNEL_QMM_DEQUANT_CACHE_MAX_ITEMS`` (default ``"2"``): max cache size.
* ``EJKERNEL_QMM_MATMUL_PRECISION``: override JAX dot precision
  (``"fastest"``, ``"high"``, ``"highest"``).
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from typing import Literal

import jax
import jax.numpy as jnp

from ejkernel.callib._ejit import ejit

from ._triton_impl import (
    _parse_matmul_precision,
    _resolve_qparams,
    quantized_matmul_dequant_triton,
    quantized_matmul_triton,
)

QuantizationMode = Literal["affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"]
GemvMode = Literal["auto", "on", "off"]
RevSplitKMode = Literal["auto", "on", "off"]

_QMM_DEQUANT_CACHE: OrderedDict[tuple, jax.Array] = OrderedDict()
_QMM_DEQUANT_CACHE_LOCK = threading.Lock()


def _device_key(arr: jax.Array) -> tuple | None:
    """Return a hashable device identifier for ``arr``, or ``None`` on failure."""
    try:
        dev = arr.device()
    except Exception:
        return None
    if dev is None:
        return None
    return (getattr(dev, "platform", None), getattr(dev, "id", None), str(dev))


def _dequant_cache_key(
    w: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None,
    out_dtype,
    group_size: int,
    bits: int,
    mode: str,
    transpose: bool,
) -> tuple:
    """Build a hashable cache key for the dequantized-weight LRU cache.

    The key incorporates array identity (``id``), shape, dtype, and all
    quantization hyper-parameters so that re-used weights with identical
    metadata hit the cache.
    """
    return (
        _device_key(w),
        id(w),
        id(scales),
        id(biases) if biases is not None else None,
        w.shape,
        w.dtype,
        scales.shape,
        scales.dtype,
        biases.shape if biases is not None else None,
        biases.dtype if biases is not None else None,
        out_dtype,
        group_size,
        bits,
        mode,
        transpose,
    )


@ejit(static_argnames=["transpose", "group_size", "bits", "mode", "use_bf16"])
def _dequant_jit(
    w,
    scales,
    biases,
    *,
    transpose: bool,
    group_size: int | None,
    bits: int | None,
    mode: QuantizationMode,
    use_bf16: bool,
):
    return quantized_matmul_dequant_triton(
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode,
        use_bf16=use_bf16,
    )


@ejit(static_argnames=["transpose", "output_dtype", "precision"])
def _two_stage_matmul(
    x,
    w_deq,
    *,
    transpose: bool,
    output_dtype,
    precision,
):
    if transpose:
        dimension_numbers = (((1,), (1,)), ((), ()))
    else:
        dimension_numbers = (((1,), (0,)), ((), ()))
    return jax.lax.dot_general(
        x,
        w_deq,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=output_dtype,
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
    """Execute quantized matmul forward pass, choosing the best kernel path.

    For large shapes (M, N, K >= 4096) when ``EJKERNEL_QMM_TWO_STAGE=1``,
    dequantizes the weight separately and runs a standard BF16 GEMM.  The
    dequantized weight is stored in an LRU cache (up to
    ``EJKERNEL_QMM_DEQUANT_CACHE_MAX_ITEMS`` entries) to amortize cost across
    consecutive calls with the same weight.

    For all other shapes (or when the two-stage path is disabled), falls back
    to the autotuned fused ``quantized_matmul_triton`` kernel.

    The output is always cast to **bfloat16** before returning.

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
        num_warps: Triton warps per program (``None`` = autotuned).
        num_stages: Triton pipeline stages (``None`` = autotuned).
        split_k: Split-K factor (``None`` = autotuned).
        gemv_mode: GEMV kernel selection mode.
        revsplit_k: Reverse split-K mode.
        revsplit_k_parts: Reverse split-K parts.

    Returns:
        Output of shape (M, N) in bfloat16.
    """
    mode_lower = mode.lower()
    use_two_stage = os.getenv("EJKERNEL_QMM_TWO_STAGE", "1").lower() in {"1", "true", "yes", "y"}
    use_cache = os.getenv("EJKERNEL_QMM_DEQUANT_CACHE", "1").lower() in {"1", "true", "yes", "y"}

    if use_two_stage and use_cache:
        group_size_resolved, bits_resolved = _resolve_qparams(mode_lower, group_size, bits)
        m = x.shape[0]
        k = x.shape[1]
        if transpose:
            n = w.shape[0]
        else:
            n = scales.shape[1] * group_size_resolved

        use_large_kernel = m >= 4096 and n >= 4096 and k >= 4096
        if use_large_kernel:
            out_dtype = jnp.bfloat16 if use_bf16 else jnp.float16
            output_dtype = jnp.bfloat16
            precision_env = os.getenv("EJKERNEL_QMM_MATMUL_PRECISION", "")
            if precision_env:
                matmul_precision = _parse_matmul_precision(precision_env)
            else:
                max_dim = max(m, n, k)
                if max_dim <= 2048:
                    matmul_precision = jax.lax.Precision.FASTEST
                elif max_dim <= 4096:
                    matmul_precision = jax.lax.Precision.HIGH
                else:
                    matmul_precision = jax.lax.Precision.DEFAULT
            cache_limit = int(os.getenv("EJKERNEL_QMM_DEQUANT_CACHE_MAX_ITEMS", "2"))

            w_deq = None
            cache_key = None
            if cache_limit > 0:
                cache_key = _dequant_cache_key(
                    w,
                    scales,
                    biases,
                    out_dtype,
                    group_size_resolved,
                    bits_resolved,
                    mode_lower,
                    transpose,
                )
                with _QMM_DEQUANT_CACHE_LOCK:
                    w_deq = _QMM_DEQUANT_CACHE.get(cache_key)
                    if w_deq is not None:
                        _QMM_DEQUANT_CACHE.move_to_end(cache_key)

            if w_deq is None:
                w_deq = _dequant_jit(
                    w,
                    scales,
                    biases,
                    transpose=transpose,
                    group_size=group_size_resolved,
                    bits=bits_resolved,
                    mode=mode_lower,
                    use_bf16=use_bf16,
                )
                if cache_key is not None:
                    with _QMM_DEQUANT_CACHE_LOCK:
                        _QMM_DEQUANT_CACHE[cache_key] = w_deq
                        _QMM_DEQUANT_CACHE.move_to_end(cache_key)
                        while len(_QMM_DEQUANT_CACHE) > cache_limit:
                            _QMM_DEQUANT_CACHE.popitem(last=False)

            x_cast = x.astype(out_dtype)
            out = _two_stage_matmul(
                x_cast,
                w_deq,
                transpose=transpose,
                output_dtype=output_dtype,
                precision=matmul_precision,
            )
            return out.astype(jnp.bfloat16)

    out = ejit(
        func=quantized_matmul_triton,
        static_argnames=[
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
        ],
    )(
        x,
        w,
        scales,
        biases,
        transpose=transpose,
        group_size=group_size,
        bits=bits,
        mode=mode_lower,
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
    return out.astype(jnp.bfloat16)


__all__ = ("quantized_matmul_forward",)
