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

"""Triton backend for RWKV-7 DPLR Recurrence.

This submodule provides GPU-optimized RWKV-7 attention using Triton kernels.
RWKV-7 introduces Diagonal Plus Low-Rank (DPLR) parameterization for state
transitions, enabling more expressive state dynamics with O(N) complexity.

Key Features:
    - O(N) time complexity with DPLR state transitions
    - Low-rank updates (a, b) for flexible information routing
    - Multiplicative parameterization variant (kk, a) via rwkv7_mul
    - Variable sequence length support via cumulative lengths
    - Bidirectional processing and custom JAX VJP support
"""

from ._interface import rwkv7, rwkv7_mul

__all__ = [
    "rwkv7",
    "rwkv7_mul",
]
