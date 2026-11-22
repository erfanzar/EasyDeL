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

"""CUDA ragged paged attention v3 kernel subpackage.

This subpackage provides a CUDA-based implementation of the ragged paged
attention v3 kernel, designed for efficient attention computation with
variable-length (ragged) sequences and paged key-value caches. It is
registered with the ejKernel kernel registry under the
``ragged_page_attention_v3`` name for the CUDA platform and GPU backend.

The subpackage handles:
    - Building the CUDA shared library from C/CUDA source via CMake.
    - Registering the compiled kernel as a JAX FFI target.
    - Exposing a type-checked, JIT-compatible public interface.

Typical usage::

    from ejkernel.kernels._cuda.ragged_page_attention_v3 import (
        ragged_page_attention_v3,
    )

    output, updated_kv_cache = ragged_page_attention_v3(
        queries, keys, values, kv_cache,
        kv_lens, block_tables, query_start_loc, distribution,
        softmax_scale=1.0,
    )
"""

from ._interface import ragged_page_attention_v3

__all__ = ("ragged_page_attention_v3",)
