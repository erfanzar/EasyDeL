# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ....layers.caching.paged_attention import PagedAttentionMetadata
from .scheduler import BatchInfo
from .sequence import SequenceState


@dataclass
class ModelInput:
    """Efficient model input representation with minimal copying."""

    input_ids: jax.Array
    position_ids: jax.Array
    attention_metadata: PagedAttentionMetadata


class ModelIOProcessor:
    """Optimized model I/O processing with pre-allocated buffers."""

    def __init__(self, block_size: int, max_batch_size: int, max_blocks: int):
        self.block_size = block_size
        self.max_batch_size = max_batch_size
        self.max_blocks = max_blocks
        self._slot_buffer = np.zeros(max_batch_size * block_size, dtype=np.int32)
        self._position_buffer = np.zeros(max_batch_size * block_size, dtype=np.int32)
        self._token_buffer = np.zeros(max_batch_size * block_size, dtype=np.int32)
        self._table_buffer = np.full((max_batch_size, max_blocks), -1, dtype=np.int32)

    def create_model_input(self, batch: BatchInfo) -> ModelInput:
        """Create model input with minimal allocations."""
        if batch.is_prefill:
            return self._create_prefill_input(batch.sequences)
        else:
            return self._create_decode_input(batch.sequences)

    def _create_prefill_input(self, sequences: list[SequenceState]) -> ModelInput:
        """Optimized prefill input creation."""
        total_tokens = sum(seq.num_tokens for seq in sequences)
        tokens = self._token_buffer[:total_tokens]
        positions = self._position_buffer[:total_tokens]
        slots = self._slot_buffer[:total_tokens]
        cu_seqlens = np.zeros(len(sequences) + 1, dtype=np.int32)

        offset = 0
        max_seqlen = 0

        for i, seq in enumerate(sequences):
            seq_len = seq.num_tokens
            seq_tokens = seq.all_tokens
            tokens[offset : offset + seq_len] = seq_tokens
            positions[offset : offset + seq_len] = np.arange(seq_len)
            for j in range(seq_len):
                block_idx = j // self.block_size
                block_offset = j % self.block_size
                slots[offset + j] = seq.block_table[block_idx] * self.block_size + block_offset

            cu_seqlens[i + 1] = cu_seqlens[i] + seq_len
            max_seqlen = max(max_seqlen, seq_len)
            offset += seq_len

        metadata = PagedAttentionMetadata(
            is_prefill=True,
            slot_mapping=jnp.array(slots),
            cu_seqlens_q=jnp.array(cu_seqlens),
            cu_seqlens_k=jnp.array(cu_seqlens),
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
        )

        return ModelInput(jnp.array(tokens), jnp.array(positions), metadata)

    def _create_decode_input(self, sequences: list[SequenceState]) -> ModelInput:
        """Optimized decode input creation."""
        batch_size = len(sequences)
        tokens = self._token_buffer[:batch_size]
        positions = self._position_buffer[:batch_size]
        slots = self._slot_buffer[:batch_size]
        context_lens = np.zeros(batch_size, dtype=np.int32)
        max_blocks = max(len(seq.block_table) for seq in sequences)
        block_tables = self._table_buffer[:batch_size, :max_blocks]
        block_tables.fill(-1)

        for i, seq in enumerate(sequences):
            tokens[i] = seq.all_tokens[-1]
            positions[i] = seq.num_tokens - 1
            context_lens[i] = seq.num_tokens
            last_block_idx = (seq.num_tokens - 1) // self.block_size
            last_block_offset = (seq.num_tokens - 1) % self.block_size
            slots[i] = seq.block_table[last_block_idx] * self.block_size + last_block_offset
            block_tables[i, : len(seq.block_table)] = seq.block_table

        metadata = PagedAttentionMetadata(
            is_prefill=False,
            slot_mapping=jnp.array(slots),
            block_tables=jnp.array(block_tables),
            context_lens=jnp.array(context_lens),
        )

        return ModelInput(jnp.array(tokens), jnp.array(positions), metadata)


__all__ = ("ModelIOProcessor", "ModelInput")
