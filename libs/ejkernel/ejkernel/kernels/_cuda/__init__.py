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


"""CUDA kernel implementations for NVIDIA GPUs.

This package provides native CUDA kernel implementations compiled as shared
libraries for high-performance GPU execution within the ejkernel framework.
Each submodule wraps a dedicated CUDA kernel (typically built from C++/CUDA
source code located under ``csrc/``) and exposes it through the unified
ejkernel kernel registry.

Submodules:
    blocksparse_attention: Block-sparse attention kernel for efficient
        attention computation with structured sparsity patterns.
    flash_attention: FlashAttention kernel providing memory-efficient,
        IO-aware exact attention.
    quantized_matmul: Quantized matrix multiplication kernel supporting
        reduced-precision (e.g., INT8/INT4) GEMM operations.
    ragged_page_attention_v3: Ragged paged-attention kernel (v3) for
        variable-length sequence decoding with a paged KV-cache.
    unified_attention: Unified attention kernel that consolidates
        multiple attention strategies behind a single interface.

The convenience function :func:`build_cuda_libs` compiles every CUDA
shared library in one call, which is useful during initial setup or
after modifying the C++/CUDA source files.

Note:
    Additional GPU kernels are implemented via Triton in the
    ``_triton`` subpackage and via Pallas in the ``_pallas``
    subpackage. Prefer those backends when a pure-Python kernel
    definition is sufficient.

See Also:
    ejkernel.kernels._triton: Triton-based GPU kernel implementations.
    ejkernel.kernels._xla: XLA/Pallas-based kernel implementations.
"""

from . import blocksparse_attention as blocksparse_attention
from . import flash_attention as flash_attention
from . import quantized_matmul as quantized_matmul
from . import ragged_page_attention_v3 as ragged_page_attention_v3
from . import unified_attention as unified_attention


def build_cuda_libs() -> None:
    """Build all CUDA shared libraries for the detected GPU architecture.

    Compiles every native CUDA kernel in this package by invoking the
    ``build_cuda_lib`` helper defined in each submodule's ``_build``
    module. The build order is:

    1. flash_attention
    2. blocksparse_attention
    3. quantized_matmul
    4. ragged_page_attention_v3
    5. unified_attention

    Each builder auto-detects the current GPU's compute capability and
    produces a shared library (``.so``) that the corresponding Python
    wrapper will load at runtime via ``ctypes`` or a similar mechanism.

    Raises:
        RuntimeError: If the CUDA toolkit is not installed or the
            detected GPU architecture is unsupported by the kernel
            source.
        FileNotFoundError: If the expected C++/CUDA source files
            under ``csrc/`` are missing.

    Example:
        Build all libraries once after cloning the repository::

            from ejkernel.kernels._cuda import build_cuda_libs
            build_cuda_libs()
    """
    from .blocksparse_attention._build import build_cuda_lib as build_blocksparse
    from .flash_attention._build import build_cuda_lib as build_flash
    from .quantized_matmul._build import build_cuda_lib as build_qmm
    from .ragged_page_attention_v3._build import build_cuda_lib as build_rpa3
    from .unified_attention._build import build_cuda_lib as build_unified

    build_flash()
    build_blocksparse()
    build_qmm()
    build_rpa3()
    build_unified()


__all__ = [
    "blocksparse_attention",
    "build_cuda_libs",
    "flash_attention",
    "quantized_matmul",
    "ragged_page_attention_v3",
    "unified_attention",
]
