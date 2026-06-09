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


"""Calling library for Triton, CuTe, tile-lang, and JAX interoperability.

This module provides utilities for integrating GPU/TPU kernels with JAX,
including JIT compilation decorators, type conversions, and helper functions
for kernel development. It bridges the gap between the different kernel DSLs
(Triton, CUTLASS CuTe, tile-lang, Pallas) and JAX's functional array
programming paradigm.

Key Components:
    - ejit: Enhanced JIT decorator for JAX functions with persistent caching
    - triton_call: Interface for calling Triton kernels from JAX
    - cute_call: Interface for calling CuTe DSL kernels from JAX
    - build_tilelang_call: Compile/autotune tile-lang prim-func factories
    - tilelang_call: Interface for calling tile-lang kernels from JAX
    - buffered_pallas_call: TPU-optimized Pallas kernel execution
    - Type conversion utilities for Triton/JAX compatibility
    - Mathematical helper functions for kernel development

Functions:
    ejit: Enhanced JIT compilation with disk caching
    triton_call: Execute Triton kernels from JAX
    cute_call: Execute CuTe DSL kernels from JAX
    build_tilelang_call: Build primitive-backed tile-lang calls
    tilelang_call: Execute tile-lang kernels from JAX
    buffered_pallas_call: Create buffered Pallas calls for TPU
    get_triton_type: Get Triton type string for objects
    cdiv: Ceiling division operation
    next_power_of_2: Find next power of 2
    strides_from_shape: Calculate array strides

Example:
    >>> from ejkernel.callib import ejit, triton_call, cute_call, tilelang_call, cdiv
    >>>
    >>> @ejit
    ... def fast_matmul(a, b):
    ...     return a @ b
    >>>
    >>> # Ceiling division for grid calculations
    >>> tiles = cdiv(seq_len, block_size)
"""

from __future__ import annotations

from typing import Any

from ._ejit import ejit
from ._pallas_call import buffered_pallas_call
from ._utils import cdiv, next_power_of_2, strides_from_shape


def _raise_triton_unavailable(err: Exception) -> None:
    """Raise an error indicating Triton is not available.

    This helper function provides a consistent error message when attempting
    to use Triton functionality without proper GPU support installed.

    Args:
        err: The original import error that caused Triton to be unavailable.

    Raises:
        ValueError: Always raised with installation instructions.
    """
    raise ValueError(
        "`triton_call` is only available when GPU Triton support is installed "
        "(install `ejkernel[gpu]` and use a CUDA/ROCm-enabled `jaxlib`)."
    ) from err


def _raise_cute_unavailable(err: Exception) -> None:
    """Raise an error indicating CuTe is not available.

    This helper function provides a consistent error message when attempting
    to use CuTe functionality without proper GPU support installed.

    Args:
        err: The original import error that caused CuTe to be unavailable.

    Raises:
        ValueError: Always raised with installation instructions.
    """
    raise ValueError(
        "`cute_call` is only available when CUTLASS CuTe is installed "
        "(install `ejkernel[gpu]` and use a CUDA-enabled `jaxlib`)."
    ) from err


try:
    from ._triton_call import get_triton_type, triton_call
except (ImportError, ModuleNotFoundError) as _triton_import_error:  # pragma: no cover

    def get_triton_type(obj: Any) -> str:  # type: ignore[override]
        """Get Triton type string for an object (unavailable fallback).

        This stub is active when Triton is not installed. It always raises
        an error directing the user to install GPU support.

        Args:
            obj: Object to get the Triton type for.

        Raises:
            ValueError: Always, with installation instructions.
        """
        _raise_triton_unavailable(_triton_import_error)

    def triton_call(*args: Any, **kwargs: Any):  # type: ignore[override]
        """Call a Triton kernel from JAX (unavailable fallback).

        This stub is active when Triton is not installed. It always raises
        an error directing the user to install GPU support.

        Args:
            *args: Ignored positional arguments.
            **kwargs: Ignored keyword arguments.

        Raises:
            ValueError: Always, with installation instructions.
        """
        _raise_triton_unavailable(_triton_import_error)


try:
    from ._cute_call import cute_call
except (ImportError, ModuleNotFoundError) as _cute_import_error:  # pragma: no cover

    def cute_call(*args: Any, **kwargs: Any):  # type: ignore[override]
        """Call a CuTe kernel from JAX (unavailable fallback).

        This stub is active when CuTe is not installed. It always raises
        an error directing the user to install GPU support.

        Args:
            *args: Ignored positional arguments.
            **kwargs: Ignored keyword arguments.

        Raises:
            ValueError: Always, with installation instructions.
        """
        _raise_cute_unavailable(_cute_import_error)


def _raise_tilelang_unavailable(err: Exception) -> None:
    """Raise an error indicating tile-lang is not available.

    Args:
        err: The original import error that caused tile-lang to be unavailable.

    Raises:
        ValueError: Always raised with installation instructions.
    """
    raise ValueError(
        "`tilelang_call` is only available when tile-lang is installed "
        "(install `tilelang` and use a CUDA-enabled `jaxlib`)."
    ) from err


try:
    from ._tilelang_call import build_tilelang_call, tilelang_call
except (ImportError, ModuleNotFoundError) as _tilelang_import_error:  # pragma: no cover

    def build_tilelang_call(*args: Any, **kwargs: Any):  # type: ignore[override]
        """Build a tile-lang kernel call from JAX (unavailable fallback).

        This stub is active when tile-lang is not installed. It always raises
        an error directing the user to install tile-lang.

        Args:
            *args: Ignored positional arguments.
            **kwargs: Ignored keyword arguments.

        Raises:
            ValueError: Always, with installation instructions.
        """
        _raise_tilelang_unavailable(_tilelang_import_error)

    def tilelang_call(*args: Any, **kwargs: Any):  # type: ignore[override]
        """Call a tile-lang kernel from JAX (unavailable fallback).

        This stub is active when tile-lang is not installed. It always raises
        an error directing the user to install tile-lang.

        Args:
            *args: Ignored positional arguments.
            **kwargs: Ignored keyword arguments.

        Raises:
            ValueError: Always, with installation instructions.
        """
        _raise_tilelang_unavailable(_tilelang_import_error)


__all__ = (
    "buffered_pallas_call",
    "build_tilelang_call",
    "cdiv",
    "cute_call",
    "ejit",
    "get_triton_type",
    "next_power_of_2",
    "strides_from_shape",
    "tilelang_call",
    "triton_call",
)
