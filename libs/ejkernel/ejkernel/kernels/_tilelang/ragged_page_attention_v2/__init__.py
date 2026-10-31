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

"""TileLang backend for ragged paged attention v2.

Exposes :func:`ragged_page_attention_v2`, a GPU kernel that handles variable-
length prefill and decode requests against a paged KV cache whose pages store
interleaved K and V heads along the ``num_combined_kv_heads`` axis.
"""

from ._interface import ragged_page_attention_v2

__all__ = ["ragged_page_attention_v2"]
