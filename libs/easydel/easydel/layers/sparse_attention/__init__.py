# Copyright 2026 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Budgeted sparse-attention token indexers.

An indexer sits beside a full-attention layer and decides, per query, which
past tokens the softmax may see: it scores compressed representations of the
prefix and returns a boolean/additive mask with ``-1``-padded top-k
selections. This is the ``layers/`` home for the pattern the model zoo
previously implemented per family (DeepSeek-V4 per-entry scorer, GLM-MoE-DSA,
MiniMax-M3-VL block scorer) -- see ``.claude/projects/qwen4-port.md`` Tier 2a.

The first occupant is :class:`BlockTopKIndexer`, the Qwen4-Exp QSA indexer:
block-granular (``compress_ratio`` consecutive visible tokens are mean-pooled
into one block key), with a fused ``index_qk_proj``, per-head layernorms,
partial RoPE (queries at their own positions, pooled block keys at the block's
start position), ``relu(q . k)`` scores summed over heads, and a static
``budget``-sized top-k over blocks plus the always-visible incomplete tail
block.
"""

from ._block_topk import BlockTopKIndexer, apply_partial_rope

__all__ = ("BlockTopKIndexer", "apply_partial_rope")
