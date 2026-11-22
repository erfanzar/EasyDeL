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


"""Kernel implementations for various platforms and backends.

This module provides a unified interface to kernel implementations across
different hardware platforms (GPU, TPU) and implementation frameworks
(Triton, CuTe, Pallas, XLA, CUDA).

The kernel system supports:
- Multi-platform implementations with automatic selection
- Priority-based kernel selection
- Hardware-specific optimizations
- Unified API across different backends

Submodules:
    cuda: CUDA-specific kernel implementations
    cute: CUTLASS CuTe DSL kernel implementations
    pallas: Pallas kernel implementations for TPU/GPU
    triton: Triton kernel implementations for GPU
    xla: XLA-based kernel implementations

Key Components:
    Backend: Enumeration of supported hardware backends
    Platform: Enumeration of supported implementation platforms
    kernel_registry: Central registry for kernel registration and lookup

Example:
    >>> from ejkernel.kernels import kernel_registry, Platform, Backend
    >>>
    >>> kernel = kernel_registry.get(
    ...     "flash_attention",
    ...     platform=Platform.TRITON,
    ...     backend=Backend.GPU
    ... )
    >>>
    >>> output = kernel(query, key, value)
"""

import importlib.util as _importlib_util

from . import _cuda as cuda
from . import _pallas as pallas
from . import _xla as xla
from ._registry import Backend, Platform, kernel_registry


def _is_optional_cute_dependency_error(err: ModuleNotFoundError) -> bool:
    """Return True when a CuTe import failed due to optional GPU deps.

    CuTe kernels require CUTLASS and CUDA Python bindings.  Missing either
    should not fail importing ``ejkernel``; we simply disable the CuTe backend.
    """
    name = err.name or ""
    return name == "cutlass" or name.startswith("cutlass.") or name == "cuda" or name.startswith("cuda.")


try:
    build_cuda_libs = cuda.build_cuda_libs
except Exception:  # pragma: no cover

    def build_cuda_libs():
        """No-op fallback for building CUDA libraries when the CUDA backend is unavailable.

        This stub is used when the native CUDA module fails to load,
        allowing imports to succeed without a CUDA toolkit installation.

        Returns:
            None: Always returns None since no build can be performed.
        """
        return None  # type: ignore[assignment]


try:
    from . import _triton as triton
except ModuleNotFoundError as err:  # pragma: no cover
    if err.name not in {"triton", "jax.experimental.pallas.triton"} and not (
        isinstance(err.name, str) and err.name.startswith("triton")
    ):
        raise
    triton = None  # type: ignore[assignment]

_has_cutlass = _importlib_util.find_spec("cutlass") is not None
if _has_cutlass:
    try:
        from . import _cute as cute
    except ModuleNotFoundError as err:  # pragma: no cover
        if not _is_optional_cute_dependency_error(err):
            raise
        cute = None  # type: ignore[assignment]
else:  # pragma: no cover
    cute = None  # type: ignore[assignment]


def _is_optional_tilelang_dependency_error(err: ModuleNotFoundError) -> bool:
    """Return True when a tile-lang import failed due to optional GPU deps.

    tile-lang kernels require the ``tilelang`` package, ``jax_tvm_ffi``, and a
    CUDA-enabled jaxlib. Missing any of these should not break ``import
    ejkernel``; we simply disable the tile-lang backend.
    """
    name = err.name or ""
    return (
        name == "tilelang"
        or name.startswith("tilelang.")
        or name == "jax_tvm_ffi"
        or name.startswith("jax_tvm_ffi.")
        or name == "tvm_ffi"
        or name.startswith("tvm_ffi.")
    )


_has_tilelang = _importlib_util.find_spec("tilelang") is not None
if _has_tilelang:
    try:
        from . import _tilelang as tilelang
    except ModuleNotFoundError as err:  # pragma: no cover
        if not _is_optional_tilelang_dependency_error(err):
            raise
        tilelang = None  # type: ignore[assignment]
else:  # pragma: no cover
    tilelang = None  # type: ignore[assignment]


__all__ = [
    "Backend",
    "Platform",
    "build_cuda_libs",
    "cuda",
    "kernel_registry",
    "pallas",
    "xla",
]
if triton is not None:
    __all__.append("triton")
if cute is not None:
    __all__.append("cute")
if tilelang is not None:
    __all__.append("tilelang")
__all__ = tuple(__all__)
