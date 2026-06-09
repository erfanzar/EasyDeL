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

"""Triton GPU backend for quantized matrix multiplication.

This submodule exposes fused dequantization + GEMM kernels for several
quantization formats:  affine (4-bit and 8-bit), NF4, MX-FP4, MX-FP8,
NV-FP4, and NV-FP8.

The public entry point is ``quantized_matmul``, which delegates to one of
several Triton kernel families depending on problem shape and the chosen
quantization mode:

* **GEMM path** (``quantized_matmul_triton``): tiled GEMM with explicit
  operation-selected block sizes and split-K parallelism.
* **Two-stage path**: weight tensor is dequantized once (optionally cached)
  and then multiplied via ``jax.lax.dot_general`` — used for large square
  shapes when the env var ``EJKERNEL_QMM_TWO_STAGE=1``.
* **GEMV split-K / reverse-split-K paths** (``quantized_matmul_triton_gemv``):
  dedicated M==1 kernels for decode-phase throughput.

Environment variables that influence kernel selection:

* ``EJKERNEL_QMM_TWO_STAGE`` (default ``"1"``): enable two-stage path for
  large kernels.
* ``EJKERNEL_QMM_DEQUANT_CACHE`` (default ``"1"``): cache dequantized weight
  tensors across calls.
* ``EJKERNEL_QMM_DEQUANT_CACHE_MAX_ITEMS`` (default ``"2"``): max cached items.
* ``EJKERNEL_QMM_MATMUL_PRECISION``: override JAX dot precision
  (``"fastest"``, ``"high"``, ``"highest"``).
"""

from ._interface import quantized_matmul

__all__ = ["quantized_matmul"]
