# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

from __future__ import annotations

import jax.numpy as jnp
from eformer.pytree import auto_pytree
from jax import lax

from easydel.layers.caching import PagesMetadata
from easydel.utils import ejit


@auto_pytree
class PageManager:
    """
    Manages page allocation for sequences in a paged attention system.

    Pages are fixed-size blocks of memory that store KV cache data.
    Each sequence is allocated pages as needed to store its tokens.
    """

    sequence_page_table: jnp.ndarray  # [max_sequences, pages_per_sequence] - which pages belong to each sequence
    page_ownership: jnp.ndarray  # [num_pages] - which sequence owns each page (-1 if free)
    sequence_lengths: jnp.ndarray  # [max_sequences] - current token count for each sequence (-1 if inactive)
    page_size: int  # Number of tokens per page

    @property
    def max_sequences(self) -> int:
        """Maximum number of concurrent sequences."""
        return self.sequence_page_table.shape[0]

    @property
    def free_pages(self) -> int:
        """Number of unallocated pages."""
        return jnp.sum(self.page_ownership < 0)

    @staticmethod
    def create(
        num_pages: int,
        max_sequences: int,
        page_size: int,
        pages_per_sequence: int,
    ) -> PageManager:
        """
        Create a new PageManager instance.

        Args:
            num_pages: Total number of pages available in the system
            max_sequences: Maximum number of concurrent sequences
            page_size: Number of tokens that fit in each page
            pages_per_sequence: Maximum pages allocatable to each sequence
        """
        return PageManager(
            sequence_page_table=jnp.full((max_sequences, pages_per_sequence), -1, dtype=jnp.int32),
            page_ownership=jnp.full((num_pages,), -1, dtype=jnp.int32),
            sequence_lengths=jnp.full((max_sequences,), -1, dtype=jnp.int32),
            page_size=page_size,
        )

    @ejit(donate_argnums=0)
    def allocate_sequence_slot(self) -> tuple[PageManager, int]:
        """
        Find and allocate a slot for a new sequence.

        Returns:
            Updated PageManager and the allocated sequence ID (-1 if no slots available)
        """
        seq_id = jnp.argmin(self.sequence_lengths)
        is_available = self.sequence_lengths[seq_id] < 0
        seq_id = jnp.where(is_available, seq_id, -1)
        new_lengths = jnp.where(seq_id >= 0, self.sequence_lengths.at[seq_id].set(0), self.sequence_lengths)

        return PageManager(
            sequence_page_table=self.sequence_page_table,
            page_ownership=self.page_ownership,
            sequence_lengths=new_lengths,
            page_size=self.page_size,
        ), seq_id

    @ejit
    def allocate_pages_for_tokens(self, token_sequence_ids: jnp.ndarray) -> tuple[PageManager, PagesMetadata]:
        """
        Allocate pages for a batch of tokens and their associated sequences.

        Args:
            token_sequence_ids: Array of sequence IDs, one per token

        Returns:
            Updated PageManager and metadata for accessing the allocated pages
        """
        token_sequence_ids = jnp.where(token_sequence_ids < 0, self.max_sequences, token_sequence_ids)
        unique_sequences, inverse_indices = jnp.unique(
            token_sequence_ids,
            return_inverse=True,
            size=self.max_sequences,
            fill_value=self.max_sequences,
        )

        token_counts = jnp.zeros(self.max_sequences, dtype=jnp.int32)
        token_counts = token_counts.at[inverse_indices].add(1)
        token_counts = jnp.where(unique_sequences >= self.max_sequences, 0, token_counts)

        current_lengths = jnp.where(self.sequence_lengths < 0, 0, self.sequence_lengths)
        new_lengths = current_lengths.at[unique_sequences].add(token_counts, mode="drop")
        new_lengths = jnp.where(self.sequence_lengths >= 0, new_lengths, -1)

        pages_needed_new = (new_lengths + self.page_size - 1) // self.page_size
        pages_needed_old = (self.sequence_lengths + self.page_size - 1) // self.page_size

        updated_page_table, updated_ownership = self._allocate_pages_batch(
            unique_sequences,
            pages_needed_old,
            pages_needed_new,
        )

        updated_manager = PageManager(
            sequence_page_table=updated_page_table,
            page_ownership=updated_ownership,
            sequence_lengths=new_lengths,
            page_size=self.page_size,
        )

        metadata = self._build_page_metadata(
            unique_sequences,
            self.sequence_lengths,
            updated_manager,
            token_counts,
            token_sequence_ids,
        )

        return updated_manager, metadata

    def _allocate_pages_batch(
        self,
        sequences: jnp.ndarray,
        old_page_counts: jnp.ndarray,
        new_page_counts: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Allocate pages for multiple sequences in batch."""

        def allocate_for_sequence(seq_id, pages_data):
            page_manager, ownership = pages_data
            old_count = old_page_counts[seq_id]
            new_count = new_page_counts[seq_id]

            def allocate_single_page(page_idx, state):
                page_manager, ownership = state
                free_page = jnp.argmin(ownership)
                ownership = ownership.at[free_page].set(seq_id)
                page_manager = page_manager.at[seq_id, page_idx].set(free_page)
                return page_manager, ownership

            return lax.fori_loop(old_count, new_count, allocate_single_page, (page_manager, ownership))

        def process_sequence(i, pages_data):
            seq_id = sequences[i]
            is_valid = jnp.logical_and(seq_id >= 0, seq_id < self.max_sequences)
            return lax.cond(is_valid, lambda data: allocate_for_sequence(seq_id, data), lambda data: data, pages_data)

        return lax.fori_loop(0, len(sequences), process_sequence, (self.sequence_page_table, self.page_ownership))

    def _build_page_metadata(
        self,
        active_sequences: jnp.ndarray,
        old_lengths: jnp.ndarray,
        updated_manager: PageManager,
        token_counts: jnp.ndarray,
        token_sequence_ids: jnp.ndarray,
    ) -> PagesMetadata:
        """Build metadata for accessing pages in the cache."""
        is_valid = jnp.logical_and(active_sequences >= 0, active_sequences < self.max_sequences)
        safe_sequences = jnp.where(is_valid, active_sequences, 0)
        page_tables = updated_manager.sequence_page_table[safe_sequences]
        page_tables = jnp.where(is_valid[:, None], page_tables, -1)
        sequence_lengths = updated_manager.sequence_lengths[safe_sequences]
        sequence_lengths = jnp.where(is_valid, sequence_lengths, -1)
        slot_mapping = self._compute_slot_mapping(token_sequence_ids, old_lengths, updated_manager)
        query_start_locations = jnp.concatenate(
            [
                jnp.zeros(1, dtype=jnp.int32),
                jnp.cumsum(token_counts, dtype=jnp.int32),
            ]
        )
        position_ids = self._compute_position_ids(token_sequence_ids, updated_manager.sequence_lengths)

        return PagesMetadata(
            pages_tables=page_tables,
            context_lens=sequence_lengths,
            query_start_loc=query_start_locations,
            num_seqs=jnp.sum(is_valid),
            slot_mapping=slot_mapping,
            position_ids=position_ids,
            page_size=self.page_size,
        )

    def _compute_slot_mapping(
        self,
        token_sequence_ids: jnp.ndarray,
        old_lengths: jnp.ndarray,
        updated_manager: PageManager,
    ) -> jnp.ndarray:
        """Compute the cache slot destination for each token."""

        slot_mapping = jnp.full(token_sequence_ids.shape, -1, dtype=jnp.int32)
        sequence_positions = jnp.where(old_lengths < 0, 0, old_lengths)

        def map_token(i, state):
            mapping, positions = state
            seq_id = token_sequence_ids[i]

            def compute_slot(state):
                mapping, positions = state
                pos = positions[seq_id]
                page_idx = pos // self.page_size
                page_offset = pos % self.page_size
                page_num = updated_manager.sequence_page_table[seq_id, page_idx]
                slot = jnp.where(page_num < 0, -1, page_num * self.page_size + page_offset)
                mapping = mapping.at[i].set(slot)
                positions = positions.at[seq_id].add(1)
                return mapping, positions

            return lax.cond(seq_id >= 0, compute_slot, lambda s: s, (mapping, positions))

        slot_mapping, _ = lax.fori_loop(0, len(token_sequence_ids), map_token, (slot_mapping, sequence_positions))

        return slot_mapping

    def _compute_position_ids(self, token_sequence_ids: jnp.ndarray, sequence_lengths: jnp.ndarray) -> jnp.ndarray:
        """Compute position IDs for each token within its sequence."""

        token_indices = jnp.arange(len(token_sequence_ids))
        is_new_sequence = jnp.concatenate([jnp.array([True]), token_sequence_ids[1:] != token_sequence_ids[:-1]])
        segment_starts = token_indices * is_new_sequence.astype(jnp.int32)
        segment_starts = jnp.maximum.accumulate(segment_starts)
        relative_positions = token_indices - segment_starts
        sequence_start_positions = jnp.where(
            token_sequence_ids >= 0,
            sequence_lengths[token_sequence_ids]
            - jnp.sum(token_sequence_ids[None, :] == token_sequence_ids[:, None], axis=1),
            -1,
        )
        position_ids = sequence_start_positions + relative_positions
        position_ids = jnp.where(token_sequence_ids < 0, -1, position_ids)
        return position_ids

    @ejit(donate_argnums=0)
    def release_sequence_slot(self, sequence_id: int) -> PageManager:
        """
        Release all pages allocated to a sequence and mark it as inactive.

        Args:
            sequence_id: ID of the sequence to release

        Returns:
            Updated PageManager with freed resources
        """
        return PageManager(
            sequence_page_table=self.sequence_page_table.at[sequence_id].set(-1),
            page_ownership=jnp.where(self.page_ownership == sequence_id, -1, self.page_ownership),
            sequence_lengths=self.sequence_lengths.at[sequence_id].set(-1),
            page_size=self.page_size,
        )

    def get_memory_stats(self) -> dict:
        """Get current memory usage statistics."""
        num_pages = self.page_ownership.shape[0]
        free_pages = self.free_pages
        used_pages = num_pages - free_pages

        return {
            "num_pages": num_pages,
            "used_pages": used_pages,
            "free_pages": free_pages,
            "utilization": used_pages / num_pages if num_pages > 0 else 0,
            "max_sequences": self.max_sequences,
        }
