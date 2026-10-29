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


"""XLA backend for Flash Attention with tiled computation.

This submodule provides an XLA-optimized implementation of Flash Attention
that achieves memory-efficient attention through tiled computation.

Key Features:
    - Memory-efficient O(N) space complexity (vs O(N^2) for standard attention)
    - Tiled forward and backward passes
    - Support for causal masking and sliding window
    - Optional dropout and attention bias
    - Segment ID support for packed sequences
    - Custom gradient implementation for numerical stability

Algorithm:
    The implementation processes attention in tiles/chunks, computing
    softmax incrementally to avoid materializing the full NxN attention matrix.

Reference:
    FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
    https://arxiv.org/abs/2205.14135
"""

from ._interface import _make_core_func, flash_attention

__all__ = ["_make_core_func", "flash_attention"]
