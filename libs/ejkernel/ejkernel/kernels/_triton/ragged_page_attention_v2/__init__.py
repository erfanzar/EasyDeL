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


"""Triton backend for Ragged Paged Attention v2.

This submodule provides GPU-optimized paged attention v2 for variable-length
batches using Triton kernels with FlashAttention-style online softmax.

Key Features:
    - Multiple query tokens per sequence support
    - Block-based KV cache with page tables
    - FlashAttention-style memory efficiency
    - Optimized for mixed workloads
"""

from ._interface import ragged_page_attention_v2

__all__ = ("ragged_page_attention_v2",)
