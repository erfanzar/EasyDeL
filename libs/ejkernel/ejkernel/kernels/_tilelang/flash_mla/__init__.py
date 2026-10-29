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

"""Tile-lang Flash Multi-head Latent Attention (forward + backward).

Provides :func:`flash_mla` registered against ``Platform.TILELANG``.  The
implementation reconstructs ``K`` and ``V`` from the compressed latent via
:func:`._kv_recon` (native tile-lang GEMM with VJP), optionally packs RoPE
score tails via :func:`._pack_shared_tail`, and delegates to
:func:`flash_attention_tilelang` for the attention core.
"""

from ._interface import flash_mla

__all__ = ["flash_mla"]
