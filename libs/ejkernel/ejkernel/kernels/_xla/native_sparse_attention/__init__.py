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


"""XLA backend for Native Sparse Attention.

This submodule provides XLA-optimized implementation of native sparse
attention using configurable sparsity patterns like DeepSeek NSA.

Key Features:
    - Configurable block-sparse attention patterns
    - Sliding window with compressed global tokens
    - Support for both prefill and decode phases
    - Custom VJP for efficient gradient computation

Reference:
    DeepSeek-V3 Technical Report
    https://arxiv.org/abs/2412.19437
"""

from ._interface import (
    _sparse_attention_bwd_vjp as native_sparse_attention_xla_bwd,
)
from ._interface import (
    _sparse_attention_fwd_vjp as native_sparse_attention_xla_fwd,
)
from ._interface import (
    apply_native_sparse_attention,
    native_sparse_attention,
)

__all__ = [
    "apply_native_sparse_attention",
    "native_sparse_attention",
    "native_sparse_attention_xla_bwd",
    "native_sparse_attention_xla_fwd",
]
