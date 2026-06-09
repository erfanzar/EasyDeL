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

"""Triton backend for RWKV-6 Linear Attention Recurrence.

This submodule provides GPU-optimized RWKV-6 linear attention using Triton
kernels. RWKV-6 extends RWKV-4 with multi-head architecture and per-timestep
time-decay parameters for more expressive sequence modeling.

Key Features:
    - O(N) time complexity with multi-head recurrent state
    - Per-timestep, per-head decay rates for flexible information routing
    - Variable sequence length support via cumulative lengths
    - Bidirectional processing via reverse flag
    - Custom Triton kernel with JAX VJP support
"""

from ._interface import rwkv6

__all__ = [
    "rwkv6",
]
