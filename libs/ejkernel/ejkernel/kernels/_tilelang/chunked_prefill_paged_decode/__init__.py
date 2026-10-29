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

"""Tile-lang fused chunked-prefill + paged decode attention.

The kernel simultaneously:
1. Writes new ``K`` / ``V`` tokens into the paged KV cache from the
   ragged ``keys`` / ``values`` input.
2. Runs causal paged attention over the full ``[0, kv_len)`` context for
   every query token using the updated cache.

See :mod:`._interface` for the public API and :mod:`._kernel` for the
``@T.prim_func`` implementation.
"""

from ._interface import chunked_prefill_paged_decode

__all__ = ["chunked_prefill_paged_decode"]
