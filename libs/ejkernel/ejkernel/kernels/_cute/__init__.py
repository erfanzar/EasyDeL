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

"""CuTe DSL kernel implementations for NVIDIA GPUs.

This package provides CUTLASS CuTe DSL kernels integrated with JAX via
custom calls. Implementations are registered with the kernel registry
under the "cute" platform.
"""

from .chunked_prefill_paged_decode import chunked_prefill_paged_decode
from .flash_attention import flash_attention
from .quantized_matmul import quantized_matmul
from .unified_attention import unified_attention

__all__ = [
    "chunked_prefill_paged_decode",
    "flash_attention",
    "quantized_matmul",
    "unified_attention",
]
