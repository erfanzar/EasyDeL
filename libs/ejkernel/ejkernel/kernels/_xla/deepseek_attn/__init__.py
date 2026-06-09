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


"""XLA backend for DeepSeek Sparse Attention (DSA).

Implements DeepSeek-V3.2's sparse attention mechanism that combines MLA
with a Lightning Indexer for dynamic top-k token selection.

Reference:
    DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models
    https://arxiv.org/abs/2512.02556
"""

from ._interface import deepseek_attn

__all__ = ("deepseek_attn",)
