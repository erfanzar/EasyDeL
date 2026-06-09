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

"""Tile-lang DeepSeek Sparse Attention (DSA) — native fused kernels (forward + backward).

The forward pass runs two native ``@T.prim_func`` kernels:
1. A Lightning-Indexer kernel that computes the per-sequence attention scores
   for the sparse indexer heads.
2. A fused sparse-MLA attention kernel that reconstructs ``K`` / ``V`` from
   the compressed latent ``key_value`` inside the attention loop via native
   GEMM kernels.

Gradients flow through ``query``, ``key_value``, ``w_kc`` and ``w_vc`` via
``jax.custom_vjp`` wrappers around native tile-lang GEMM kernels.  The
top-k selection in the indexer path is integer and is correctly
stop-gradient.

See :mod:`._impl` for the implementation and :mod:`._kernel` for the
``@T.prim_func`` factories.
"""

from ._interface import deepseek_attn

__all__ = ["deepseek_attn"]
