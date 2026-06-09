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

"""XLA backend for quantized matrix multiplication.

This submodule provides an XLA/JAX implementation of quantized matrix
multiplication supporting affine, NF4, MXFP4/8, and NVFP4/8 quantization.

When possible, a blocked algorithm with fused on-the-fly dequantization
is used to avoid materializing the full weight matrix. For incompatible
shapes or bit-widths, it falls back to a two-step dequantize+matmul path.

Key Features:
    - Explicit quantization modes: affine, nf4, mxfp4, mxfp8, nvfp4, nvfp8
    - Blocked tiled matmul with fused dequantization
    - Automatic fallback for unsupported configurations
    - Configurable tile sizes and compute precision
"""

from ._interface import quantized_matmul

__all__ = ["quantized_matmul"]
