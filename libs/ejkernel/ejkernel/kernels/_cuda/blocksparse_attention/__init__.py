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

"""CUDA block-sparse attention kernel subpackage.

This subpackage provides a CUDA-backed implementation of block-sparse attention.
Block-sparse attention exploits the sparsity pattern in attention masks to skip
computation on masked-out blocks, yielding significant speedups for long
sequences with structured sparsity (e.g., causal, sliding-window, or custom
sparse patterns).

The forward pass is computed by a compiled CUDA kernel loaded via JAX FFI,
while the backward pass uses a CUDA-side dense analytical fallback that
preserves block-sparse semantics for query/key/value gradients.

Public API:
    blocksparse_attention: High-level entry point registered in the kernel
        registry for the CUDA/GPU backend.
    _blocksparse_attention_bhtd_fwd: Forward pass function used by the
        custom VJP rule.
    _blocksparse_attention_bhtd_bwd: Backward pass function used by the
        custom VJP rule.
"""

from ._interface import (
    _blocksparse_attention_bhtd_bwd,
    _blocksparse_attention_bhtd_fwd,
    blocksparse_attention,
)

__all__ = [
    "_blocksparse_attention_bhtd_bwd",
    "_blocksparse_attention_bhtd_fwd",
    "blocksparse_attention",
]
