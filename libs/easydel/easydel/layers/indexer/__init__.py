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

"""Composable sparse-attention indexer layers.

:class:`BaseIndexer` and :class:`IndexerSelection` define shared ranking and
attention-consumer contracts. Concrete strategies keep their own projection
layout, compression, cache protocol and gradient policy. Static
:class:`SelectionSpec` metadata distinguishes raw-token offsets from
compressed-entry offsets and shared selections from per-group selections.

:class:`SparseIndexer` implements GLM-style token and learned-pool strategies;
it is not a replacement for mean-key pooling, max-score pooling or two-series
compressed-entry indexing. Model adapters retain checkpoint-native parameter
names without nesting a second parameter-owning implementation module.

Cache adapters own request/physical-page mapping; indexers return logical
selection offsets. The presence of a shared selection contract does not imply
that every strategy supports cached decode, packed training or shared-layer
reuse. Each implementation preserves its explicit capability checks.
"""

from ._block_max import BlockMaxIndexer, BlockMaxIndexerConfig
from ._block_topk import BlockTopKIndexer
from ._compressed import CompressedIndexer, CompressedIndexerAdapter, CompressedIndexerConfig, CompressedIndexerScorer
from ._config import IndexerConfig, IndexerKind
from ._indexer import IndexerOutput, SparseIndexer
from ._primitives import (
    indices_to_bool_mask,
    is_vacuous_selection,
    ste_score_proxy,
    topk_select,
    topk_selection_mask,
    visible_causal_mask,
)
from ._selection import BaseIndexer, IndexerSelection, SelectionSpec
from ._token import TokenIndexer, TokenIndexerConfig

__all__ = (
    "BaseIndexer",
    "BlockMaxIndexer",
    "BlockMaxIndexerConfig",
    "BlockTopKIndexer",
    "CompressedIndexer",
    "CompressedIndexerAdapter",
    "CompressedIndexerConfig",
    "CompressedIndexerScorer",
    "IndexerConfig",
    "IndexerKind",
    "IndexerOutput",
    "IndexerSelection",
    "SelectionSpec",
    "SparseIndexer",
    "TokenIndexer",
    "TokenIndexerConfig",
    "apply_indexer_rope",
    "indices_to_bool_mask",
    "is_vacuous_selection",
    "ste_score_proxy",
    "topk_select",
    "topk_selection_mask",
    "visible_causal_mask",
)


def __getattr__(name: str):
    # rope helpers live in a leaf module without heavy deps; lazy-import to
    # keep ``import easydel.layers.indexer`` light on non-indexer users.
    if name == "apply_indexer_rope":
        from ._rope import apply_indexer_rope

        return apply_indexer_rope
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
