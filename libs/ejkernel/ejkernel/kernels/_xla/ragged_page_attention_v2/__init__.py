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


"""XLA backend for Ragged Page Attention V2.

This submodule provides XLA-optimized implementation of ragged page attention
for variable-length sequences with paged key-value cache management.

Key Features:
    - Variable-length sequence support via cu_seqlens
    - Paged KV cache with flexible page sizes
    - Optimized for mixed prefill and decode phases
    - Efficient handling of ragged batch shapes
"""

from ._interface import ragged_page_attention_v2

__all__ = ["ragged_page_attention_v2"]
