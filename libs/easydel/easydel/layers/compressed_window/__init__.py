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

"""Building blocks for sliding-window + compressed-KV attention.

:class:`GatedPoolCompressor` pools ``ratio`` tokens into one KV latent (stateless and incremental). The ring-window
helpers write and read the sliding-window ring and gather indexer-selected compressed entries. Indexer scoring and
candidate-block selection live in :mod:`easydel.layers.indexer`.
"""

from ._compressor import (
    BlockGroupLayout,
    GatedPoolCompressor,
    GatedPoolCompressorConfig,
    block_group_layout,
    gather_group_members,
    last_closed_overlap,
    overlap_group_slots,
    record_stream_trace,
    refresh_group_state,
    trace_compressor_stream,
)
from ._packed import (
    PACKED_PREFILL_BLOCK,
    grid_to_tokens,
    occupied_columns,
    packed_block_size,
    run_column_blocks,
    tokens_to_grid,
)
from ._window import (
    block_sink_attention,
    block_token_bias,
    gather_selected_entries,
    ring_decode_write,
    ring_prefill_write,
    ring_slot_positions,
    ring_visibility_bias,
    sink_attention,
)

__all__ = (
    "PACKED_PREFILL_BLOCK",
    "BlockGroupLayout",
    "GatedPoolCompressor",
    "GatedPoolCompressorConfig",
    "block_group_layout",
    "block_sink_attention",
    "block_token_bias",
    "gather_group_members",
    "gather_selected_entries",
    "grid_to_tokens",
    "last_closed_overlap",
    "occupied_columns",
    "overlap_group_slots",
    "packed_block_size",
    "record_stream_trace",
    "refresh_group_state",
    "ring_decode_write",
    "ring_prefill_write",
    "ring_slot_positions",
    "ring_visibility_bias",
    "run_column_blocks",
    "sink_attention",
    "tokens_to_grid",
    "trace_compressor_stream",
)
