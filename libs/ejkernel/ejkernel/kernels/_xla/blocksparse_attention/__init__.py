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


"""XLA backend for Block Sparse Attention.

This submodule provides XLA-optimized implementation of block-sparse
attention with configurable sparsity patterns.

Key Features:
    - Block-wise sparse attention patterns
    - Configurable block sizes
    - Support for custom sparsity masks
    - Memory-efficient computation for long sequences
"""

from ._interface import blocksparse_attention

__all__ = ("blocksparse_attention",)
