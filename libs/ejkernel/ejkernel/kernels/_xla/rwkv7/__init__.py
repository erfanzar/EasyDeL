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

"""XLA backend for RWKV-7 recurrence.

This submodule provides XLA-optimized implementation of RWKV-7's
recurrence mechanism with multiplicative state updates.

Key Features:
    - O(N) complexity through linear recurrence
    - Multiplicative state update for enhanced expressivity
    - Separate additive (rwkv7) and multiplicative (rwkv7_mul) variants
    - Improved modeling of long-range dependencies

Variants:
    - rwkv7: Standard RWKV-7 with additive updates
    - rwkv7_mul: Multiplicative variant with enhanced capacity

Algorithm:
    RWKV-7 extends DPLR with multiplicative mixing:
        h_t = diag(w_t) * h_{t-1} * a_t + k_t^T * v_t
        y_t = sigmoid(r_t) * (h_t @ q_t)
    where a_t provides multiplicative modulation.
"""

from ._interface import rwkv7, rwkv7_mul

__all__ = [
    "rwkv7",
    "rwkv7_mul",
]
