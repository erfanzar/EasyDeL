# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""Unified dynamic sparse-attention indexer layer.

An indexer sits beside a full-attention layer and decides, per query, which
past tokens the softmax may see. Historically every model family implemented
its own variant (DeepSeek-V4 Lightning indexer, GLM-MoE-DSA per-token top-k,
GLM-5-Next k-pool, MiniMax-M3 block-sparse, Qwen4-Exp block top-k) with
near-identical plumbing: projections, per-key norms, partial RoPE, causal +
padding visibility, top-k with ``-1`` padding, packed per-token cache state,
straight-through score proxies and vacuous-selection fast paths.

:class:`SparseIndexer` unifies all of that behind one dynamic module driven by
:class:`IndexerConfig`, so adding an indexer to a future model is a
constructor call instead of a few hundred lines. Submodule names
(``wq_b`` / ``wk`` / ``k_norm`` / ``weights_proj`` and the k-pool
``index_kpool_compress_ape`` / ``index_kpool_compress_gate`` parameters)
match the published GLM checkpoints so families can adopt the layer without
renaming loaded weights.

Selection strategies:

- ``kind="token"``: score every key token directly (GLM-MoE-DSA, DeepSeek-V3/V2
  Lightning-style, after optional external compression).
- ``kind="pool"``: group keys into pools of ``kpool_size`` starting at the
  first valid token, summarise each pool with a softmax-gated learned average
  (``ape`` position bias), top-k over pools and expand back to token indices
  with the optional always-selected tail pool (GLM-5-Next).

Shared knobs cover the observed zoo variance: score activation
(``none`` / ``relu``), head reduction (``weighted`` / ``uniform``), query
source (``q_lora`` residual vs raw ``hidden``), RoPE style (``split_half`` /
``interleaved`` / ``none``) with width truncation, packed cache layout
(``keys`` / ``key_gate_valid`` / ``none``), straight-through gradient
suppression, and the ``prev_topk_indices`` carry for ``shared`` indexer
layers that reuse another layer's selection.
"""

from ._config import IndexerConfig, IndexerKind
from ._indexer import IndexerOutput, SparseIndexer
from ._primitives import (
    indices_to_bool_mask,
    is_vacuous_selection,
    ste_score_proxy,
    topk_select,
    visible_causal_mask,
)

__all__ = (
    "IndexerConfig",
    "IndexerKind",
    "IndexerOutput",
    "SparseIndexer",
    "apply_indexer_rope",
    "indices_to_bool_mask",
    "is_vacuous_selection",
    "ste_score_proxy",
    "topk_select",
    "visible_causal_mask",
)


def __getattr__(name: str):
    # rope helpers live in a leaf module without heavy deps; lazy-import to
    # keep ``import easydel.layers.indexer`` light on non-indexer users.
    if name == "apply_indexer_rope":
        from ._rope import apply_indexer_rope

        return apply_indexer_rope
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
