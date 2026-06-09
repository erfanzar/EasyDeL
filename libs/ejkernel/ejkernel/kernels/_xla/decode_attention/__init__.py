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

"""XLA backend for paged decode attention.

This submodule provides XLA-optimized implementation of decode-phase
attention optimized for single-token autoregressive generation.

Key Features:
    - Optimized for seq_len_q=1 (single token generation)
    - Paged KV buffer with indirect token addressing
    - Variable context lengths per request
    - Efficient memory layout for serving workloads

Algorithm:
    Decode attention computes attention for one query token against
    the full KV cache built during prefill and previous decode steps.
    Uses req_to_tokens for flexible memory management across requests.
"""

from ._interface import decode_attention

__all__ = ("decode_attention",)
