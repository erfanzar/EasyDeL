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

"""XLA backend for RWKV-6 recurrence.

This submodule provides a pure JAX/XLA implementation of the RWKV-6
multi-head time-mix recurrence.

Key Features:
    - O(N) complexity through linear recurrence
    - Multi-head structure with per-head [K, V] state matrices
    - Per-timestep data-dependent decay (w is input-dependent, unlike RWKV-4)
    - Bonus term u that boosts the current token's contribution
    - Efficient state caching for incremental inference
    - Variable-length packed-sequence mode (cu_seqlens)

Algorithm:
    RWKV-6 extends RWKV-4/5 with multi-head state and input-dependent decay:
        kv_t = k_t^T ⊗ v_t            (outer product, [H, K, V])
        o_t  = r_t^T @ (h_{t-1} + kv_t * u)  (query with current-token bonus)
        h_t  = h_{t-1} * exp(w_t) + kv_t     (exponential decay update)
    where w_t is sequence-position-dependent (shape [B, T, H, K]).

Reference:
    RWKV-6 (BlinkDL): https://github.com/BlinkDL/RWKV-LM
"""

from ._interface import rwkv6

__all__ = [
    "rwkv6",
]
