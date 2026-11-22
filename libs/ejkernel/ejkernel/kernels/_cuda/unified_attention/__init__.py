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

"""CUDA unified attention kernel subpackage.

Provides a CUDA-accelerated implementation of the unified attention kernel,
which performs paged key-value cache attention with support for causal masking,
sliding window attention, and logits soft-capping. The kernel operates on
paged (block-table-indexed) KV caches and is designed for efficient inference
serving with variable-length sequences.

The primary entry point is :func:`unified_attention`, which is registered
with the ejkernel kernel registry under the ``"unified_attention"`` name
for the CUDA platform and GPU backend.

Modules:
    _build: CMake-based build system for compiling the CUDA shared library.
    _cuda_impl: Low-level FFI binding that invokes the compiled CUDA kernel.
    _interface: High-level, type-checked interface registered with the kernel registry.

Typical usage::

    from ejkernel.kernels._cuda.unified_attention import unified_attention

    output = unified_attention(
        queries, key_cache, value_cache,
        kv_lens, block_tables, query_start_loc,
        softmax_scale=0.125, causal=True,
    )

Note:
    This kernel requires a CUDA-capable GPU and the CMake/CUDA toolkit for
    building the shared library on first use.
"""

from ._interface import unified_attention

__all__ = ["unified_attention"]
