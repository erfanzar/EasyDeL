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
from ._block_topk import BlockTopKIndexer, apply_partial_rope
from ._compressed import CompressedIndexer, CompressedIndexerAdapter, CompressedIndexerConfig, CompressedIndexerScorer
from ._config import IndexerConfig, IndexerKind
from ._distill import indexer_distillation_loss
from ._indexer import IndexerLayerNorm, IndexerOutput, SparseIndexer
from ._paged import (
    bounded_block_max_topk,
    bounded_candidate_entry_topk,
    bounded_entry_topk,
    paged_pool_select,
    paged_token_layout,
    paged_token_select,
    running_entry_topk,
    slot_buffer_pages,
    slot_entry_topk,
)
from ._paged_block_topk import paged_block_topk
from ._primitives import (
    CompactLayout,
    compact_valid_layout,
    indices_to_bool_mask,
    is_vacuous_selection,
    permute_tokens,
    ste_score_proxy,
    topk_select,
    topk_selection_mask,
    visible_causal_mask,
)
from ._selection import BaseIndexer, IndexerSelection, ScoreRowSplit, SelectionSpec
from ._shared_key import (
    IndexerScoreInputs,
    SharedKeyIndexer,
    SharedKeyIndexerConfig,
    select_candidate_blocks,
    selection_bias,
    shared_key_scores,
    warn_unsupported_scoring_kernel,
)
from ._token import TokenIndexer, TokenIndexerConfig
from ._training import AUTO_SPARSE_MIN_SEQ, pool_bitmap_tail_error, use_sparse_training_attention

__all__ = (
    "AUTO_SPARSE_MIN_SEQ",
    "BaseIndexer",
    "BlockMaxIndexer",
    "BlockMaxIndexerConfig",
    "BlockTopKIndexer",
    "CompactLayout",
    "CompressedIndexer",
    "CompressedIndexerAdapter",
    "CompressedIndexerConfig",
    "CompressedIndexerScorer",
    "IndexerConfig",
    "IndexerKind",
    "IndexerLayerNorm",
    "IndexerOutput",
    "IndexerScoreInputs",
    "IndexerSelection",
    "ScoreRowSplit",
    "SelectionSpec",
    "SharedKeyIndexer",
    "SharedKeyIndexerConfig",
    "SparseIndexer",
    "TokenIndexer",
    "TokenIndexerConfig",
    "apply_indexer_rope",
    "apply_partial_rope",
    "bounded_block_max_topk",
    "bounded_candidate_entry_topk",
    "bounded_entry_topk",
    "compact_valid_layout",
    "indexer_distillation_loss",
    "indices_to_bool_mask",
    "is_vacuous_selection",
    "paged_block_topk",
    "paged_pool_select",
    "paged_token_layout",
    "paged_token_select",
    "permute_tokens",
    "pool_bitmap_tail_error",
    "running_entry_topk",
    "select_candidate_blocks",
    "selection_bias",
    "shared_key_scores",
    "slot_buffer_pages",
    "slot_entry_topk",
    "ste_score_proxy",
    "topk_select",
    "topk_selection_mask",
    "use_sparse_training_attention",
    "visible_causal_mask",
    "warn_unsupported_scoring_kernel",
)


def __getattr__(name: str):
    # rope helpers live in a leaf module without heavy deps; lazy-import to
    # keep ``import easydel.layers.indexer`` light on non-indexer users.
    if name == "apply_indexer_rope":
        from ._rope import apply_indexer_rope

        return apply_indexer_rope
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
