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


"""ejKernel: High-performance kernel library for JAX.

This package provides a collection of optimized kernels for deep learning operations,
with support for multiple implementation backends (XLA, Triton, Pallas/GPU, Pallas/TPU,
CUDA/FFI, CUTE-DSL, TileLang) across GPU and TPU platforms. The library focuses on
efficient attention mechanisms (flash attention, sparse attention, ring attention, MLA,
ragged paged attention, etc.), quantized matrix multiplications, and other
performance-critical computations.

Key Features:
    - Multi-backend support (XLA, Triton, Pallas, CUDA, CUTE, TileLang) for optimal
      performance across hardware targets.
    - Platform-specific optimizations for NVIDIA/AMD GPUs and Google TPUs.
    - Extensive collection of attention kernels (Flash, BlockSparse, Ring, MLA, etc.).
    - Priority-based automatic kernel selection via a central ``KernelRegistry``.
    - Modular architecture: each backend lives under ``ejkernel.kernels._<backend>``.

Public API:
    Backend: Enum of hardware backends (GPU, TPU, CPU, ANY, MPS).
    Platform: Enum of implementation platforms (TRITON, PALLAS, CUDA, CUTE, TILELANG, XLA).
    EjkernelRuntimeError: Base runtime error for unsupported operations.
    kernel_registry: Global singleton ``KernelRegistry`` used by all built-in kernels.
    kernels: Sub-package containing all kernel implementations.
    modules: High-level module wrappers around the low-level kernels.
    utils: Utility functions (math helpers, test data generators, sharding helpers).
    errors: Custom exception types.
    xla_utils: XLA/JAX-level helper utilities.

Note:
    At import time, the following environment variables are set if not already present:
        - ``TF_GPU_ALLOCATOR=cuda_malloc_async`` for improved GPU memory allocation.
        - ``CUTE_DSL_ENABLE_TVM_FFI=1`` to enable TVM FFI support in CuTe DSL.
    ``kernel_registry.validate_signatures(None)`` is also called at import time to emit
    warnings for any registered implementations with mismatched signatures.

Example:
    >>> import ejkernel
    >>> from ejkernel import Backend, Platform
    >>>
    >>> impl = ejkernel.kernel_registry.get(
    ...     "flash_attention",
    ...     platform=Platform.TRITON,
    ...     backend=Backend.GPU,
    ... )
    >>> output = impl(query, key, value)
"""

import os as _os

_os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
_os.environ.setdefault("CUTE_DSL_ENABLE_TVM_FFI", "1")

__version__ = "0.0.80.3"

from . import errors, kernels, modules, types, utils, xla_utils
from .errors import EjkernelRuntimeError
from .kernels import Backend, Platform, kernel_registry

kernel_registry.validate_signatures(None)

__all__ = (
    "Backend",
    "EjkernelRuntimeError",
    "Platform",
    "errors",
    "kernel_registry",
    "kernels",
    "modules",
    "types",
    "utils",
    "xla_utils",
)
