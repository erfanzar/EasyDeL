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


"""XLA backend for Paged Attention with KV cache.

This submodule provides XLA-optimized implementation of paged attention
for efficient LLM serving with paged key-value cache management.

Key Features:
    - Paged KV cache for memory-efficient serving
    - Variable page sizes for flexible memory management
    - Support for batch decoding with different sequence lengths
    - Integration with vLLM-style page table management
"""

from ._interface import page_attention

__all__ = ["page_attention"]
