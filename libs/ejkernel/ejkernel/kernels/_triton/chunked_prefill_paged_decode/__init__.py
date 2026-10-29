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

"""Triton backend for Chunked Prefill + Paged Decode Attention.

This submodule provides GPU-optimized attention for mixed prefill and decode
workloads using Triton kernels. It updates a block-tabled KV cache with new
keys/values, then computes unified paged attention outputs for packed queries.

Key Features:
    - Unified handling of chunked prefill and decode in single operation
    - In-place KV cache update with block table management
    - Support for causal masking and sliding window attention
    - ALiBi positional encoding support
    - Logit soft-capping for numerical stability
"""

from ._interface import chunked_prefill_paged_decode

__all__ = ("chunked_prefill_paged_decode",)
