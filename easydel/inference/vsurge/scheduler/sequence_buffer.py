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

import jax
from eformer.pytree import auto_pytree
from jax import lax
from jax import numpy as jnp
from jaxtyping import Array


def copy_to_dest(dest: Array, start: int, src: Array, num_to_copy: int) -> Array:
    src_len = src.shape[0]
    dest_len = dest.shape[0]

    src_indices = jnp.arange(src_len)
    src_indices = jnp.where(src_indices < num_to_copy, src_indices, src_len)

    dest_indices = jnp.arange(dest_len)
    valid_dest_indices_mask = (dest_indices >= start) & (dest_indices < start + num_to_copy) & (dest_indices < dest_len)

    src_idx_for_dest = dest_indices - start
    src_indices_for_gather = jnp.where(valid_dest_indices_mask, src_idx_for_dest, src_len)
    target_dtype = src.dtype
    if target_dtype == jnp.int64:
        target_dtype = jnp.int32
    padded_src = jnp.concatenate([src, jnp.full((1,), -1, dtype=target_dtype)])
    gathered_values = padded_src[src_indices_for_gather]

    result = jnp.where(valid_dest_indices_mask, gathered_values, dest)
    return result


@auto_pytree
class BatchedSequences:
    """Packed sequence for efficient processing"""

    tokens: Array
    sequence_ids: Array
    num_tokens: Array
    is_boundary: Array

    def boundary_indices(self, max_boundaries: int) -> Array:
        """Get sequence boundary indices"""
        total_positions = self.is_boundary.shape[0]
        indices_all = jnp.arange(total_positions, dtype=jnp.int32)
        pad_key_value = total_positions
        sort_keys = jnp.where(self.is_boundary, indices_all, pad_key_value)
        sorted_indices = jnp.argsort(sort_keys)
        candidate_indices = sorted_indices[:total_positions]
        is_valid_candidate = self.is_boundary[candidate_indices]
        output_indices = jnp.arange(max_boundaries, dtype=jnp.int32)
        selected_candidate_idx = jnp.where(output_indices < total_positions, output_indices, total_positions)
        padded_candidates = jnp.concatenate([candidate_indices, jnp.array([-1], dtype=jnp.int32)])
        padded_validity = jnp.concatenate([is_valid_candidate, jnp.array([False], dtype=jnp.bool_)])
        final_selected_indices = padded_candidates[selected_candidate_idx]
        final_is_valid = padded_validity[selected_candidate_idx]
        result = jnp.where(final_is_valid, final_selected_indices, -1)
        return result


@auto_pytree
class SequenceBufferState:
    """JIT-compatible scheduler for token management"""

    generated_tokens: Array
    generated_sequence_ids: Array
    num_generated_tokens: Array
    queued_tokens: Array
    queued_sequence_ids: Array
    num_queued_tokens: Array

    @staticmethod
    def init(max_tokens: int) -> SequenceBufferState:
        """Initialize empty SequenceBufferState"""
        return SequenceBufferState(
            generated_tokens=jnp.full((max_tokens,), -1, dtype=jnp.int32),
            generated_sequence_ids=jnp.full((max_tokens,), -1, dtype=jnp.int32),
            num_generated_tokens=jnp.array(0, dtype=jnp.int32),
            queued_tokens=jnp.full((max_tokens,), -1, dtype=jnp.int32),
            queued_sequence_ids=jnp.full((max_tokens,), -1, dtype=jnp.int32),
            num_queued_tokens=jnp.array(0, dtype=jnp.int32),
        )

    def dequeue_tokens_with_return(self, num_to_dequeue: int) -> tuple[SequenceBufferState, Array, Array, int]:
        """Remove tokens from the front of the queue and return them.

        Args:
            num_to_dequeue: Number of tokens to remove from the queue

        Returns:
            Tuple of:
            - Updated SequenceBufferState with tokens removed
            - Dequeued tokens array
            - Dequeued sequence IDs array
            - Actual number of tokens dequeued
        """
        actual_dequeue = jnp.minimum(num_to_dequeue, self.num_queued_tokens)
        dequeued_tokens = jnp.where(jnp.arange(self.queued_tokens.shape[0]) < actual_dequeue, self.queued_tokens, -1)
        dequeued_sequence_ids = jnp.where(
            jnp.arange(self.queued_sequence_ids.shape[0]) < actual_dequeue,
            self.queued_sequence_ids,
            -1,
        )

        new_state = self.dequeue_tokens(actual_dequeue)

        return new_state, dequeued_tokens, dequeued_sequence_ids, actual_dequeue

    def enqueue_tokens(self, new_tokens: Array, new_sequence_ids: Array, num_new_tokens: int) -> SequenceBufferState:
        """Add tokens to queue"""

        return SequenceBufferState(
            generated_tokens=self.generated_tokens,
            generated_sequence_ids=self.generated_sequence_ids,
            num_generated_tokens=self.num_generated_tokens,
            queued_tokens=copy_to_dest(self.queued_tokens, self.num_queued_tokens, new_tokens, num_new_tokens),
            queued_sequence_ids=copy_to_dest(
                self.queued_sequence_ids, self.num_queued_tokens, new_sequence_ids, num_new_tokens
            ),
            num_queued_tokens=self.num_queued_tokens + num_new_tokens,
        )

    def dequeue_tokens(self, num_to_dequeue: int) -> SequenceBufferState:
        """Remove tokens from the front of the queue.

        Args:
            num_to_dequeue: Number of tokens to remove from the queue

        Returns:
            Updated SequenceBufferState with tokens removed
        """
        num_to_dequeue = jnp.minimum(num_to_dequeue, self.num_queued_tokens)
        remaining_tokens = self.num_queued_tokens - num_to_dequeue
        buffer_size = self.queued_tokens.shape[0]
        indices = jnp.arange(buffer_size)
        shifted_indices = indices + num_to_dequeue
        valid_mask = shifted_indices < buffer_size
        gather_indices = jnp.where(valid_mask, shifted_indices, buffer_size)
        padded_tokens = jnp.concatenate([self.queued_tokens, jnp.array([-1], dtype=jnp.int32)])
        padded_seq_ids = jnp.concatenate([self.queued_sequence_ids, jnp.array([-1], dtype=jnp.int32)])
        new_tokens = padded_tokens[gather_indices]
        new_sequence_ids = padded_seq_ids[gather_indices]
        new_tokens = jnp.where(indices < remaining_tokens, new_tokens, -1)
        new_sequence_ids = jnp.where(indices < remaining_tokens, new_sequence_ids, -1)
        return SequenceBufferState(
            generated_tokens=self.generated_tokens,
            generated_sequence_ids=self.generated_sequence_ids,
            num_generated_tokens=self.num_generated_tokens,
            queued_tokens=new_tokens,
            queued_sequence_ids=new_sequence_ids,
            num_queued_tokens=remaining_tokens,
        )

    def dequeue_tokens_for_sequences(self, sequence_ids_to_remove: Array) -> SequenceBufferState:
        """Remove all queued tokens belonging to specific sequences.

        Args:
            sequence_ids_to_remove: Array of sequence IDs whose tokens should be removed

        Returns:
            Updated SequenceBufferState with specified sequences' tokens removed
        """
        buffer_size = self.queued_tokens.shape[0]
        keep_mask = jnp.ones(buffer_size, dtype=jnp.bool_)
        for seq_id in sequence_ids_to_remove:
            keep_mask = keep_mask & (self.queued_sequence_ids != seq_id)
        valid_mask = jnp.arange(buffer_size) < self.num_queued_tokens
        keep_mask = keep_mask & valid_mask
        keep_indices = jnp.where(keep_mask, jnp.arange(buffer_size), buffer_size)
        sorted_indices = jnp.sort(keep_indices)
        padded_tokens = jnp.concatenate([self.queued_tokens, jnp.array([-1], dtype=jnp.int32)])
        padded_seq_ids = jnp.concatenate([self.queued_sequence_ids, jnp.array([-1], dtype=jnp.int32)])
        new_tokens = padded_tokens[sorted_indices]
        new_sequence_ids = padded_seq_ids[sorted_indices]
        new_count = jnp.sum(keep_mask)
        position_mask = jnp.arange(buffer_size) < new_count
        new_tokens = jnp.where(position_mask, new_tokens, -1)
        new_sequence_ids = jnp.where(position_mask, new_sequence_ids, -1)

        return SequenceBufferState(
            generated_tokens=self.generated_tokens,
            generated_sequence_ids=self.generated_sequence_ids,
            num_generated_tokens=self.num_generated_tokens,
            queued_tokens=new_tokens,
            queued_sequence_ids=new_sequence_ids,
            num_queued_tokens=new_count,
        )

    def update_after_sampling(
        self,
        new_tokens: Array,
        new_token_sequence_ids: Array,
        num_new_tokens: int,
    ) -> SequenceBufferState:
        """Update after generating new tokens"""

        updated_gen = SequenceBufferState(
            generated_tokens=copy_to_dest(self.generated_tokens, self.num_generated_tokens, new_tokens, num_new_tokens),
            generated_sequence_ids=copy_to_dest(
                self.generated_sequence_ids,
                self.num_generated_tokens,
                new_token_sequence_ids,
                num_new_tokens,
            ),
            num_generated_tokens=self.num_generated_tokens + num_new_tokens,
            queued_tokens=self.queued_tokens,
            queued_sequence_ids=self.queued_sequence_ids,
            num_queued_tokens=self.num_queued_tokens,
        )

        updated = updated_gen.enqueue_tokens(new_tokens, new_token_sequence_ids, num_new_tokens)
        return updated

    def pack_next_sequence(self, max_tokens: int) -> tuple[SequenceBufferState, BatchedSequences]:
        """Pack tokens for processing"""
        buffer_size = self.queued_tokens.shape[0]
        num_to_pack = jnp.minimum(self.num_queued_tokens, max_tokens)

        tokens = jnp.where(jnp.arange(buffer_size) < num_to_pack, self.queued_tokens, -1)
        sequence_ids = jnp.where(jnp.arange(buffer_size) < num_to_pack, self.queued_sequence_ids, -1)

        rolled_tokens = jnp.roll(self.queued_tokens, -num_to_pack)
        rolled_sequence_ids = jnp.roll(self.queued_sequence_ids, -num_to_pack)

        idx = jnp.arange(buffer_size)
        mask = idx >= (buffer_size - num_to_pack)
        filler_tokens = jnp.where(mask, -1, rolled_tokens)
        filler_sequence_ids = jnp.where(mask, -1, rolled_sequence_ids)

        new_scheduler = SequenceBufferState(
            generated_tokens=self.generated_tokens,
            generated_sequence_ids=self.generated_sequence_ids,
            num_generated_tokens=self.num_generated_tokens,
            queued_tokens=filler_tokens,
            queued_sequence_ids=filler_sequence_ids,
            num_queued_tokens=self.num_queued_tokens - num_to_pack,
        )

        next_sequence_ids = jnp.roll(sequence_ids, -1)
        is_boundary = (sequence_ids != next_sequence_ids) & (sequence_ids != -1)

        last_packed_idx = num_to_pack - 1
        next_after_last_packed = rolled_sequence_ids[0]

        def compute_boundary_last_true():
            last_packed_seq_id = sequence_ids[last_packed_idx]
            return (last_packed_seq_id != -1) & (
                (next_after_last_packed == -1) | (last_packed_seq_id != next_after_last_packed)
            )

        def compute_boundary_last_false():
            return False

        boundary_last = lax.cond(num_to_pack > 0, compute_boundary_last_true, compute_boundary_last_false)
        is_boundary = lax.cond(
            num_to_pack > 0,
            lambda: is_boundary.at[last_packed_idx].set(boundary_last),
            lambda: is_boundary,
        )

        return new_scheduler, BatchedSequences(
            tokens=tokens,
            sequence_ids=sequence_ids,
            num_tokens=num_to_pack,
            is_boundary=is_boundary,
        )

    def get_tokens(self, sequence_ids: Array, max_tokens: int) -> tuple[SequenceBufferState, Array]:
        """Extract generated tokens for specific sequences"""

        pos_idx = jnp.arange(self.generated_tokens.shape[0], dtype=jnp.int32)
        valid_mask = pos_idx < self.num_generated_tokens

        seq_matches = (self.generated_sequence_ids[None, :] == sequence_ids[:, None]) & valid_mask[None, :]
        cumsum_matches = jnp.cumsum(seq_matches.astype(jnp.int32), axis=1)
        take_mask_per_seq = seq_matches & (cumsum_matches <= max_tokens)
        removal_mask = jnp.any(take_mask_per_seq, axis=0)
        keep_mask = valid_mask & (~removal_mask)

        def gather_row(args):
            mask_row, _ = args
            idx = jnp.argsort(jnp.where(mask_row, pos_idx, self.generated_tokens.shape[0] + pos_idx))[:max_tokens]
            good = mask_row[idx]
            return jnp.where(good, self.generated_tokens[idx], -1)

        gathered = jax.vmap(gather_row)((take_mask_per_seq, sequence_ids))
        order_keep = jnp.argsort(jnp.where(keep_mask, pos_idx, self.generated_tokens.shape[0] + pos_idx))
        new_num = keep_mask.sum()
        tail_mask = pos_idx >= new_num
        updated = SequenceBufferState(
            generated_tokens=jnp.where(tail_mask, -1, self.generated_tokens[order_keep]),
            generated_sequence_ids=jnp.where(tail_mask, -1, self.generated_sequence_ids[order_keep]),
            num_generated_tokens=new_num,
            queued_tokens=self.queued_tokens,
            queued_sequence_ids=self.queued_sequence_ids,
            num_queued_tokens=self.num_queued_tokens,
        )
        return updated, gathered
