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

"""TileLang backend for ragged paged attention v3.

Exposes :func:`ragged_page_attention_v3`, a fused GPU kernel that writes new
K/V tokens into the paged cache and then immediately runs causal paged
attention.  The KV cache uses a packed layout with an explicit ``kv_packing``
dimension for sub-word dtype packing (e.g. two float16 elements per int32).
"""

from ._interface import ragged_page_attention_v3

__all__ = ["ragged_page_attention_v3"]
