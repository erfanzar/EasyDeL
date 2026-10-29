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

"""CuTe DSL quantized matrix multiplication implementation.

Exposes :func:`quantized_matmul`, the registry entry point for the CuTe
platform.  The implementation fuses bit-unpacking, dequantization, and GEMM
inside a single CuTe DSL GPU kernel.  Multiple kernel families are available
(naive scalar, tiled SMEM, single-stage MMA, pipelined MMA) and are selected
automatically based on tile dimensions and optional environment-variable
overrides.  See ``_cute_impl.py`` for the full dispatch logic.
"""

from ._interface import quantized_matmul

__all__ = ["quantized_matmul"]
