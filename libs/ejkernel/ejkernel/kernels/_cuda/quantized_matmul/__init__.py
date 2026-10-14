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

"""CUDA quantized matrix multiplication kernels.

This subpackage provides a CUDA-backed implementation of quantized matrix
multiplication for the ejKernel framework. It exposes the :func:`quantized_matmul`
entry point, which performs matrix multiplication between a float input tensor and
packed quantized weights using a custom CUDA kernel invoked via JAX FFI.

Supported quantization modes:
    - **affine**: Grouped affine quantization with bits from 1 through 8.
    - **nf4**: 4-bit NormalFloat codebook quantization.
    - **mxfp4/mxfp8**: Microscaling FP4/FP8 modes (group size 32).
    - **nvfp4/nvfp8**: NVIDIA FP4/FP8 modes (group size 16).

The CUDA kernel is built on first use from C++/CUDA sources located in the
repository's ``csrc/quantized_matmul/`` directory using CMake. Build artifacts
are cached and automatically invalidated when source files change.

Example:
    >>> from ejkernel.kernels._cuda.quantized_matmul import quantized_matmul
    >>> result = quantized_matmul(x, packed_w, scales, zeros, mode="affine")
"""

from ._interface import quantized_matmul

__all__ = ["quantized_matmul"]
