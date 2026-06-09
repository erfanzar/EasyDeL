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

"""Tile-lang vLLM-style single-token paged decode attention.

Provides :func:`decode_attention` registered against ``Platform.TILELANG``.
Internally routes to a paged FlashDecoding kernel (one CTA per
``(batch, head)``, page-table lookup, online softmax, LSE output).
"""

from ._interface import decode_attention

__all__ = ["decode_attention"]
