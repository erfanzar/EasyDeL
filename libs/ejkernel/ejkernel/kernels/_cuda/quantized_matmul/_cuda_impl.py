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

"""Low-level CUDA implementation of quantized matrix multiplication.

This module contains the direct JAX FFI binding to the ``ejk_qmm_cuda``
custom call. It handles:

* One-time registration of the CUDA shared library via :func:`_register_cuda_qmm`.
* Mapping string quantization mode names to integer IDs consumed by the
  CUDA kernel.
* Input validation (shapes, bit-widths, and mode constraints).
* Construction and dispatch of the FFI call that executes the CUDA kernel.

The main entry point is :func:`quantized_matmul_cuda`, which is invoked by
the higher-level :func:`~ejkernel.kernels._cuda.quantized_matmul._interface.quantized_matmul`.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from typing import Literal

import jax
import jax.ffi as ffi
import jax.numpy as jnp

from ejkernel.quantization._utils.qparams import (
    GemvMode,
    RevSplitKMode,
    normalize_gemv_mode,
    normalize_revsplitk_mode,
    normalize_revsplitk_parts,
    resolve_qparams,
    select_qmm_kernel_family,
    to_backend_mode,
)

from ._build import build_cuda_lib

QuantizationMode = Literal["affine", "nf4", "mxfp4", "mxfp8", "nvfp4", "nvfp8"]
"""Type alias for supported quantization mode strings."""


def _gemv_mode_to_id(mode: GemvMode) -> int:
    """Convert a GEMV mode string to the integer ID expected by the CUDA kernel.

    Args:
        mode: One of ``"auto"``, ``"on"``, or ``"off"``.

    Returns:
        Integer identifier: 0 for auto, 1 for on, 2 for off.
    """
    if mode == "auto":
        return 0
    if mode == "on":
        return 1
    return 2


def _revsplit_mode_to_id(mode: RevSplitKMode) -> int:
    """Convert a reverse split-K mode string to the integer ID expected by the CUDA kernel.

    Args:
        mode: One of ``"auto"``, ``"on"``, or ``"off"``.

    Returns:
        Integer identifier: 0 for auto, 1 for on, 2 for off.
    """
    if mode == "auto":
        return 0
    if mode == "on":
        return 1
    return 2


@lru_cache(maxsize=1)
def _register_cuda_qmm() -> None:
    """Build and register the ``ejk_qmm_cuda`` FFI target with JAX.

    On the first call this function:

    1. Compiles the CUDA shared library (if not already cached) via
       :func:`~._build.build_cuda_lib`.
    2. Loads it with :mod:`ctypes` to obtain the ``ejk_qmm_cuda`` symbol.
    3. Registers that symbol as a JAX FFI target on the ``"gpu"`` platform.

    Subsequent calls are no-ops thanks to :func:`functools.lru_cache`.

    Raises:
        RuntimeError: If the CUDA library cannot be built or loaded.
    """
    lib_path = build_cuda_lib()
    lib = ctypes.CDLL(str(lib_path))
    handler = lib.ejk_qmm_cuda
    ffi.register_ffi_target(
        "ejk_qmm_cuda",
        ffi.pycapsule(handler),
        platform="gpu",
        api_version=1,
    )


def _mode_to_id(mode: str) -> int:
    """Convert a quantization mode name to the integer ID expected by the CUDA kernel.

    Mapping:
        ========  ==
        Mode      ID
        ========  ==
        affine     0
        nf4        1
        mxfp4      2
        mxfp8      3
        nvfp4      4
        nvfp8      5
        ========  ==

    Args:
        mode: Case-insensitive quantization mode name.

    Returns:
        Integer identifier consumed by the CUDA kernel.

    Raises:
        ValueError: If *mode* is not one of the supported names.
    """
    mode = mode.lower()
    if mode == "affine":
        return 0
    if mode == "nf4":
        return 1
    if mode == "mxfp4":
        return 2
    if mode == "mxfp8":
        return 3
    if mode == "nvfp4":
        return 4
    if mode == "nvfp8":
        return 5
    raise ValueError("CUDA quantized_matmul supports {'affine','nf4','mxfp4','mxfp8','nvfp4','nvfp8'}.")


def _expected_words(n: int, bits: int) -> int:
    """Compute the number of 32-bit words required to store *n* elements at *bits* each.

    This is the expected packed width (second dimension) for the packed
    weight tensor ``w``. Each ``uint32`` word packs ``floor(32 / bits)``
    quantized values.

    Args:
        n: Number of output features (columns).
        bits: Bit-width per quantized element (4 or 8).

    Returns:
        Number of ``uint32`` words per row of the packed weight matrix.
    """
    return int((n * bits + 31) // 32)


def quantized_matmul_cuda(
    x: jax.Array,
    w: jax.Array,
    scales: jax.Array,
    biases: jax.Array | None = None,
    *,
    transpose: bool = False,
    group_size: int | None = None,
    bits: int | None = None,
    mode: QuantizationMode = "affine",
    gemv_mode: GemvMode = "auto",
    revsplit_k: RevSplitKMode = "auto",
    revsplit_k_parts: int | None = None,
    block_n: int = 0,
    block_k: int = 0,
) -> jax.Array:
    """Execute quantized matrix multiplication on CUDA via JAX FFI.

    Performs ``x @ dequantize(w, scales, zeros)`` entirely on the GPU
    using a custom CUDA kernel registered through JAX's FFI mechanism.
    Weights are stored in a packed ``uint32`` format and dequantized
    on-the-fly during kernel execution.

    The function validates all inputs, resolves default values for
    *bits* and *group_size* via :func:`resolve_qparams`, selects the
    appropriate kernel family (GEMV or GEMM with optional reverse split-K),
    registers the CUDA kernel (on the first call), and dispatches the
    FFI call.

    Args:
        x: Input activation matrix of shape ``(M, K)`` with dtype
            ``float16``, ``bfloat16``, or ``float32``.
        w: Packed quantized weight matrix (``uint32``). Shape depends on
            *transpose*:

            * ``transpose=False``: ``(K, ceil(N * bits / 32))``
            * ``transpose=True``:  ``(N, ceil(K * bits / 32))``
        scales: Per-group scale factors of rank 2. Shape depends on
            *transpose*:

            * ``transpose=False``: ``(K, N // group_size)``
            * ``transpose=True``:  ``(N, K // group_size)``
        biases: Internal additive offsets derived from affine ``zeros``
            metadata, with the same shape as *scales*. Must be ``None``
            for non-affine modes; for affine mode it may also be ``None``
            if no zero-point correction is needed.
        transpose: Whether the weight matrix is stored in ``(N, K)``
            transposed layout. When ``True``, the kernel dequantizes *w*
            into a ``(K, N)`` buffer before running GEMM.
        group_size: Number of weight elements per quantization group.
            Defaults: 64 for ``affine``/``nf4``; 32 for
            ``mxfp4``/``mxfp8``; 16 for ``nvfp4``/``nvfp8``.
        bits: Bit-width per quantized element. Meaningful only for
            ``affine`` (supported values: 4 or 8). Ignored for other modes
            which have fixed bit-widths.
        mode: Backend quantization mode string, already resolved to a
            backend-specific form (e.g., ``"affine4b"``). See
            :func:`to_backend_mode` for the mapping from canonical names.
        gemv_mode: GEMV dispatch strategy:
            ``"auto"`` -- kernel chooses based on *M*; ``"on"`` -- force
            GEMV path; ``"off"`` -- force GEMM path.
        revsplit_k: Reverse split-K strategy:
            ``"auto"`` -- kernel decides; ``"on"`` -- force split-K;
            ``"off"`` -- disable split-K.
        revsplit_k_parts: Number of split-K partitions when *revsplit_k*
            is ``"on"`` or the GEMV+split-K family is selected. ``None``
            defers the choice to the kernel selector.

    Returns:
        Result matrix of shape ``(M, N)`` with the same dtype as *x*.

    Raises:
        ValueError: If *bits* is outside [1, 8] or incompatible with the
            chosen *mode*.
        ValueError: If *scales* is not rank-2 or its shape is inconsistent
            with *x* and *w*.
        ValueError: If the packed weight width does not match the expected
            number of ``uint32`` words for the given *N* (or *K*) and
            *bits*.
        ValueError: If the K dimensions of *x* and *w* disagree.
        ValueError: If *x* dtype is not ``float16``, ``bfloat16``, or
            ``float32``.
    """

    mode, group_size, bits, _ = resolve_qparams(mode, group_size, bits)
    gemv_mode = normalize_gemv_mode(gemv_mode)
    revsplit_k = normalize_revsplitk_mode(revsplit_k)
    revsplit_k_parts = normalize_revsplitk_parts(revsplit_k_parts)
    mode = to_backend_mode(mode, bits)
    mode_id = _mode_to_id(mode)
    group_size = int(group_size)
    bits = int(bits)
    m, k = int(x.shape[0]), int(x.shape[1])
    kernel_family, family_revsplit_parts = select_qmm_kernel_family(
        m=m,
        mode=mode,
        bits=bits,
        gemv_mode=gemv_mode,
        revsplit_k=revsplit_k,
        revsplit_k_parts=revsplit_k_parts,
    )
    if kernel_family == "gemv_revsplitk":
        revsplit_k_parts = 2 if family_revsplit_parts is None else int(family_revsplit_parts)
    else:
        revsplit_k_parts = 0 if revsplit_k_parts is None else int(revsplit_k_parts)
    if scales.ndim != 2:
        raise ValueError("CUDA quantized_matmul scales must be rank-2.")
    if biases is not None and (biases.ndim != 2 or biases.shape != scales.shape):
        raise ValueError("CUDA quantized_matmul biases shape must match scales shape.")
    if transpose:
        if k % group_size != 0:
            raise ValueError("CUDA quantized_matmul K must be divisible by group_size for transpose=True.")
        expected_words = _expected_words(k, bits)
        if (
            scales.shape[1] == k // group_size
            and int(w.shape[1]) == expected_words
            and scales.shape[0] == int(w.shape[0])
        ):
            n_groups = int(scales.shape[1])
            n = int(w.shape[0])
        elif (
            scales.shape[0] == k // group_size
            and int(w.shape[0]) == expected_words
            and scales.shape[1] == int(w.shape[1])
        ):
            n_groups = int(scales.shape[0])
            n = int(w.shape[1])
        else:
            raise ValueError(
                "CUDA quantized_matmul transpose=True expects either "
                "w=(N, ceil(K*bits/32)), scales=(N, K/group_size), or "
                "k-major w=(ceil(K*bits/32), N), scales=(K/group_size, N)."
            )
    else:
        n_groups = int(scales.shape[1])
        n = n_groups * group_size
        if scales.shape[0] != k:
            raise ValueError("CUDA quantized_matmul scales shape must be (K, N/group_size).")
        expected_words = _expected_words(n, bits)
        k_w = int(w.shape[0])
        if int(w.shape[1]) != expected_words:
            raise ValueError("CUDA quantized_matmul packed weight shape does not match N and bits.")
        if k != k_w:
            raise ValueError("Input K dimension does not match packed weight K.")

    out_dtype = jnp.dtype(x.dtype)
    if out_dtype not in (jnp.float16, jnp.bfloat16, jnp.float32):
        raise ValueError("CUDA quantized_matmul supports float16/float32/bfloat16 outputs.")
    out_shape = jax.ShapeDtypeStruct((m, n), out_dtype)
    _register_cuda_qmm()

    call = ffi.ffi_call("ejk_qmm_cuda", out_shape, vmap_method="sequential")

    attrs = {
        "group_size": group_size,
        "bits": bits,
        "mode": mode_id,
        "transpose": int(transpose),
        "gemv_mode": _gemv_mode_to_id(gemv_mode),
        "revsplit_k": _revsplit_mode_to_id(revsplit_k),
        "revsplit_k_parts": int(revsplit_k_parts),
        "block_n": int(block_n),
        "block_k": int(block_k),
    }

    if biases is None:
        return call(x, w, scales, **attrs)
    return call(x, w, scales, biases, **attrs)
