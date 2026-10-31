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

"""XLA backend for RWKV-4 time-mix recurrence.

This submodule provides a pure JAX/XLA implementation of the RWKV-4
time-mix mechanism — a linear-complexity alternative to self-attention.

Key Features:
    - O(N) complexity through per-channel recurrence
    - Numerically stable via log-sum-exp (eps) state tracking
    - Global learned decay w and current-token bonus u
    - State caching for autoregressive generation

Algorithm:
    RWKV-4 maintains per-channel (alpha, beta, eps) state and computes:
        tau_t   = max(u + k_t, eps_{t-1})
        wkv_t   = (exp(eps_{t-1} - tau_t) * alpha_{t-1} + exp(u + k_t - tau_t) * v_t)
                / (exp(eps_{t-1} - tau_t) * beta_{t-1}  + exp(u + k_t - tau_t))
        eps_t   = max(w + eps_{t-1}, k_t)
        alpha_t = exp(w + eps_{t-1} - eps_t) * alpha_{t-1} + exp(k_t - eps_t) * v_t
        beta_t  = exp(w + eps_{t-1} - eps_t) * beta_{t-1}  + exp(k_t - eps_t)
    where w = -exp(w_param) is the channel-wise learned decay.

Reference:
    RWKV: Reinventing RNNs for the Transformer Era
    https://arxiv.org/abs/2305.13048
"""

from ._interface import rwkv4

__all__ = [
    "rwkv4",
]
