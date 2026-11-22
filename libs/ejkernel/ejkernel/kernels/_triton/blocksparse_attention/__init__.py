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


"""Blocksparse Attention implementation using Triton kernels.

This module provides efficient blocksparse attention computation with
custom JAX gradients and Triton acceleration. Supports various sparsity
patterns including local attention and vertical stride patterns.
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
