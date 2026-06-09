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

"""XLA backend for chunked prefill + paged decode attention.

Exposes ``chunked_prefill_paged_decode``, which handles a packed batch of
sequences where each sequence may be in either the prefill phase (multiple
query tokens) or the decode phase (a single query token).

Algorithm (XLA reference):
    1. Insert the incoming ``keys`` / ``values`` for every sequence into
       the block-tabled KV cache via ``lax.fori_loop`` scatter operations.
    2. Call ``unified_attention`` over the updated cache to produce the
       attention output for all query tokens.

The XLA implementation is correct but slow: the ``fori_loop`` cache-update
and the dense attention materialize O(seq_len²) work.  It exists as the
numerical reference for the Triton GPU implementation.

Note:
    Only causal attention is supported (``causal=False`` raises
    ``NotImplementedError``).
"""

from ._interface import chunked_prefill_paged_decode

__all__ = ("chunked_prefill_paged_decode",)
