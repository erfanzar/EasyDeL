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


"""Pallas/Triton GPU backend for ragged decode attention.

Provides a single-token decode attention kernel for variable-length (ragged)
KV caches.  Each batch element may have a different valid key/value range
specified by per-element ``sequence_start`` / ``sequence_end`` indices.

The kernel tiles computation over both heads (``block_size_heads``) and keys
(``block_size_keys``) and can split the key sequence across multiple GPU
thread-block groups (``num_key_splits``) for additional parallelism.
Supports MHA, MQA, and GQA via query-head grouping.

Public API:
    ragged_decode_attention: Registered under
        ``Platform.PALLAS / Backend.GPU`` in the kernel registry.
"""

from ._interface import ragged_decode_attention

__all__ = ("ragged_decode_attention",)
