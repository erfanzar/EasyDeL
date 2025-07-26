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

import enum
from functools import partial

import jax
import jax.numpy as jnp
from eformer.pytree import auto_pytree
from jaxtyping import Array

from easydel.inference.vsurge.scheduler.page_manager import PageManager

from .sequence_buffer import BatchedSequences, SequenceBufferState


class RequestPhase(enum.IntEnum):
    """Phase of request processing"""

    PREFILL = 0
    DECODE = 1
    COMPLETE = 2


@auto_pytree
class RequestState:
    """State of a single request"""

    request_id: Array
    sequence_id: Array
    tokens: Array
    current_position: Array
    total_tokens: Array
    phase: Array
    priority: Array
    generated_count: Array
    max_generate: Array
    is_active: Array


@auto_pytree
class RequestManager:
    """Manages multiple concurrent requests"""

    states: RequestState
    max_requests: int
    prefill_chunk_size: int
    decode_batch_size: int

    @staticmethod
    def create(
        max_requests: int,
        max_sequence_length: int,
        prefill_chunk_size: int = 256,
        decode_batch_size: int = 32,
    ) -> RequestManager:
        """Create a new RequestManager"""
        return RequestManager(
            states=RequestState(
                request_id=jnp.arange(max_requests, dtype=jnp.int32),
                sequence_id=jnp.full((max_requests,), -1, dtype=jnp.int32),
                tokens=jnp.full((max_requests, max_sequence_length), -1, dtype=jnp.int32),
                current_position=jnp.zeros((max_requests,), dtype=jnp.int32),
                total_tokens=jnp.zeros((max_requests,), dtype=jnp.int32),
                phase=jnp.full((max_requests,), RequestPhase.PREFILL, dtype=jnp.int32),
                priority=jnp.ones((max_requests,), dtype=jnp.float32),
                generated_count=jnp.zeros((max_requests,), dtype=jnp.int32),
                max_generate=jnp.zeros((max_requests,), dtype=jnp.int32),
                is_active=jnp.zeros((max_requests,), dtype=jnp.bool_),
            ),
            max_requests=max_requests,
            prefill_chunk_size=prefill_chunk_size,
            decode_batch_size=decode_batch_size,
        )

    def add_request(
        self,
        tokens: Array,
        max_generate: int = 256,
        priority: float = 1.0,
    ) -> tuple[RequestManager, int]:
        """
        Add a new request to the manager.

        This function:
        1. Finds an available slot
        2. Validates the slot can be used
        3. Prepares the token data
        4. Updates all state arrays
        5. Returns the new manager and request ID
        """
        request_id = jnp.argmin(self.states.is_active)
        slot_available = ~self.states.is_active[request_id]
        max_token_length = self.states.tokens.shape[1]
        input_length = jnp.minimum(tokens.shape[0], max_token_length)

        new_tokens = jnp.full(max_token_length, -1, dtype=jnp.int32)
        new_tokens = new_tokens.at[:input_length].set(tokens[:input_length])
        update_mask = jnp.arange(self.max_requests) == request_id
        new_sequence_id = jnp.where(update_mask & slot_available, -1, self.states.sequence_id)
        new_tokens_array = jnp.where(update_mask[:, None] & slot_available, new_tokens[None, :], self.states.tokens)
        new_current_position = jnp.where(update_mask & slot_available, 0, self.states.current_position)
        new_total_tokens = jnp.where(update_mask & slot_available, input_length, self.states.total_tokens)
        new_phase = jnp.where(update_mask & slot_available, RequestPhase.PREFILL, self.states.phase)
        new_priority = jnp.where(update_mask & slot_available, priority, self.states.priority)
        new_generated_count = jnp.where(update_mask & slot_available, 0, self.states.generated_count)
        new_max_generate = jnp.where(update_mask & slot_available, max_generate, self.states.max_generate)
        new_is_active = jnp.where(update_mask & slot_available, True, self.states.is_active)
        new_states = RequestState(
            request_id=self.states.request_id,
            sequence_id=new_sequence_id,
            tokens=new_tokens_array,
            current_position=new_current_position,
            total_tokens=new_total_tokens,
            phase=new_phase,
            priority=new_priority,
            generated_count=new_generated_count,
            max_generate=new_max_generate,
            is_active=new_is_active,
        )

        new_manager = RequestManager(
            states=new_states,
            max_requests=self.max_requests,
            prefill_chunk_size=self.prefill_chunk_size,
            decode_batch_size=self.decode_batch_size,
        )

        result_request_id = jnp.where(slot_available, request_id, -1)

        return new_manager, result_request_id

    def assign_sequence(self, request_id: int, sequence_id: int) -> RequestManager:
        """Assign a sequence ID to a request"""
        new_states = self.states
        new_states = jax.tree_util.tree_map(
            lambda x: x.at[request_id].set(sequence_id) if x is self.states.sequence_id else x,
            new_states,
        )
        return RequestManager(
            states=new_states,
            max_requests=self.max_requests,
            prefill_chunk_size=self.prefill_chunk_size,
            decode_batch_size=self.decode_batch_size,
        )

    def get_next_batch(
        self,
        max_batch_tokens: int,
        prefill_priority: float = 0.7,
    ) -> tuple[RequestManager, Array, Array, Array]:
        """
        Get next batch of tokens to process.

        Returns:
            - Updated RequestManager
            - Tokens to process
            - Sequence IDs for each token
            - Phase indicators for each token
        """

        prefill_budget = jnp.int32(max_batch_tokens * prefill_priority)
        decode_budget = max_batch_tokens - prefill_budget

        is_prefill = (self.states.phase == RequestPhase.PREFILL) & self.states.is_active
        is_decode = (self.states.phase == RequestPhase.DECODE) & self.states.is_active

        prefill_tokens, prefill_seq_ids, prefill_count, new_positions = self._schedule_prefill(
            is_prefill, prefill_budget
        )

        decode_tokens, decode_seq_ids, decode_count = self._schedule_decode(is_decode, decode_budget)

        total_tokens = prefill_count + decode_count
        batch_tokens = jnp.concatenate(
            [
                prefill_tokens[:prefill_count],
                decode_tokens[:decode_count],
                jnp.full((max_batch_tokens - total_tokens,), -1, dtype=jnp.int32),
            ]
        )

        batch_seq_ids = jnp.concatenate(
            [
                prefill_seq_ids[:prefill_count],
                decode_seq_ids[:decode_count],
                jnp.full((max_batch_tokens - total_tokens,), -1, dtype=jnp.int32),
            ]
        )

        batch_phases = jnp.concatenate(
            [
                jnp.full((prefill_count,), RequestPhase.PREFILL, dtype=jnp.int32),
                jnp.full((decode_count,), RequestPhase.DECODE, dtype=jnp.int32),
                jnp.full((max_batch_tokens - total_tokens,), -1, dtype=jnp.int32),
            ]
        )

        new_states = self.states
        new_states = jax.tree_util.tree_map(
            lambda x: jnp.where(x is self.states.current_position, jnp.where(is_prefill, new_positions, x), x),
            new_states,
        )

        prefill_complete = new_positions >= self.states.total_tokens
        new_states = jax.tree_util.tree_map(
            lambda x: jnp.where(
                x is self.states.phase, jnp.where(is_prefill & prefill_complete, RequestPhase.DECODE, x), x
            ),
            new_states,
        )

        return (
            RequestManager(
                states=new_states,
                max_requests=self.max_requests,
                prefill_chunk_size=self.prefill_chunk_size,
                decode_batch_size=self.decode_batch_size,
            ),
            batch_tokens,
            batch_seq_ids,
            batch_phases,
        )

    def _schedule_prefill(
        self,
        is_prefill: Array,
        budget: int,
    ) -> tuple[Array, Array, int, Array]:
        """Schedule prefill tokens within budget"""

        # Get prefill requests with their priorities
        prefill_mask = is_prefill & self.states.is_active

        if not jnp.any(prefill_mask):
            # No prefill requests
            empty_tokens = jnp.full((budget,), -1, dtype=jnp.int32)
            empty_seq_ids = jnp.full((budget,), -1, dtype=jnp.int32)
            return empty_tokens, empty_seq_ids, 0, self.states.current_position

        # Sort by priority (higher priority first)
        priorities = jnp.where(prefill_mask, self.states.priority, -jnp.inf)
        sorted_indices = jnp.argsort(-priorities)

        # Collect tokens from highest priority requests
        all_tokens = []
        all_seq_ids = []
        total_scheduled = 0
        new_positions = self.states.current_position.copy()

        for i in range(self.max_requests):
            if total_scheduled >= budget:
                break

            idx = sorted_indices[i]

            # Skip if not a valid prefill request
            if not prefill_mask[idx]:
                continue

            # Calculate how many tokens we can take from this request
            remaining_in_request = self.states.total_tokens[idx] - self.states.current_position[idx]
            remaining_in_budget = budget - total_scheduled
            chunk_size = jnp.minimum(jnp.minimum(remaining_in_request, self.prefill_chunk_size), remaining_in_budget)

            if chunk_size > 0:
                # Extract tokens from this request
                start_pos = self.states.current_position[idx]
                request_tokens = self.states.tokens[idx]

                # Get the chunk
                chunk_tokens = request_tokens[start_pos : start_pos + chunk_size]
                chunk_seq_ids = jnp.full((chunk_size,), self.states.sequence_id[idx], dtype=jnp.int32)

                all_tokens.append(chunk_tokens)
                all_seq_ids.append(chunk_seq_ids)
                total_scheduled += chunk_size

                # Update position for this request
                new_positions = new_positions.at[idx].add(chunk_size)

        # Concatenate all collected tokens
        if all_tokens:
            combined_tokens = jnp.concatenate(all_tokens)
            combined_seq_ids = jnp.concatenate(all_seq_ids)
        else:
            combined_tokens = jnp.array([], dtype=jnp.int32)
            combined_seq_ids = jnp.array([], dtype=jnp.int32)

        # Pad to budget size
        padded_tokens = jnp.full((budget,), -1, dtype=jnp.int32)
        padded_seq_ids = jnp.full((budget,), -1, dtype=jnp.int32)

        if len(combined_tokens) > 0:
            padded_tokens = padded_tokens.at[: len(combined_tokens)].set(combined_tokens)
            padded_seq_ids = padded_seq_ids.at[: len(combined_seq_ids)].set(combined_seq_ids)

        return padded_tokens, padded_seq_ids, total_scheduled, new_positions

    def _schedule_decode(
        self,
        is_decode: Array,
        budget: int,
    ) -> tuple[Array, Array, int]:
        """Schedule decode tokens within budget"""

        decode_indices = jnp.where(is_decode, jnp.arange(self.max_requests), self.max_requests)
        decode_indices = jnp.sort(decode_indices)[: self.decode_batch_size]

        valid_mask = decode_indices < self.max_requests
        num_decode = jnp.minimum(jnp.sum(valid_mask), budget)

        tokens = jnp.full((budget,), -1, dtype=jnp.int32)
        seq_ids = jnp.where(
            jnp.arange(budget) < num_decode,
            self.states.sequence_id[decode_indices[jnp.arange(budget) % self.decode_batch_size]],
            -1,
        )

        return tokens, seq_ids, num_decode

    def update_generated_tokens(self, sequence_ids: Array, new_tokens: Array) -> RequestManager:
        """Update generated token counts after decoding"""
        new_states = self.states

        for i in range(len(sequence_ids)):
            seq_id = sequence_ids[i]
            if seq_id >= 0:
                mask = (self.states.sequence_id == seq_id) & self.states.is_active
                request_idx = jnp.argmax(mask)

                if mask[request_idx]:

                    def _map(x, request_idx):
                        return x.at[request_idx].add(1) if x is self.states.generated_count else x

                    new_states = jax.tree_util.tree_map(partial(_map, request_idx=request_idx), new_states)
                    is_complete = new_states.generated_count[request_idx] >= new_states.max_generate[request_idx]

                    def _map(x, request_idx, is_complete):
                        return (
                            x.at[request_idx].set(jnp.where(is_complete, RequestPhase.COMPLETE, x[request_idx]))
                            if x is self.states.phase
                            else x
                        )

                    new_states = jax.tree_util.tree_map(
                        partial(_map, request_idx=request_idx, is_complete=is_complete),
                        new_states,
                    )

        return RequestManager(
            states=new_states,
            max_requests=self.max_requests,
            prefill_chunk_size=self.prefill_chunk_size,
            decode_batch_size=self.decode_batch_size,
        )

    def get_completed_requests(self) -> Array:
        """Get IDs of completed requests"""
        is_complete = (self.states.phase == RequestPhase.COMPLETE) & self.states.is_active
        return jnp.where(is_complete, self.states.request_id, -1)

    def complete_request(self, request_id: int) -> RequestManager:
        """Mark a request as inactive"""
        new_states = self.states
        new_states = jax.tree_util.tree_map(
            lambda x: x.at[request_id].set(False) if x is self.states.is_active else x, new_states
        )
        return RequestManager(
            states=new_states,
            max_requests=self.max_requests,
            prefill_chunk_size=self.prefill_chunk_size,
            decode_batch_size=self.decode_batch_size,
        )

    def has_active_requests(self) -> bool:
        """Check if there are any active requests"""
        return jnp.any(self.states.is_active)


@auto_pytree
class ContinuousBatchingEngine:
    """Main engine for continuous batching inference"""

    request_manager: RequestManager
    sequence_buffer: SequenceBufferState
    page_manager: PageManager
    max_batch_tokens: int
    prefill_priority: float

    @staticmethod
    def create(
        max_requests: int,
        max_sequence_length: int,
        max_batch_tokens: int,
        num_pages: int,
        page_size: int,
        max_num_pages_per_req: int,
        prefill_chunk_size: int = 256,
        decode_batch_size: int = 32,
        prefill_priority: float = 0.7,
    ) -> ContinuousBatchingEngine:
        """Create a new continuous batching engine"""

        request_manager = RequestManager.create(
            max_requests=max_requests,
            max_sequence_length=max_sequence_length,
            prefill_chunk_size=prefill_chunk_size,
            decode_batch_size=decode_batch_size,
        )

        sequence_buffer = SequenceBufferState.init(max_batch_tokens * 4)

        page_manager = PageManager.create(
            num_pages=num_pages,
            max_sequences=max_requests,
            page_size=page_size,
            max_num_pages_per_req=max_num_pages_per_req,
        )

        return ContinuousBatchingEngine(
            request_manager=request_manager,
            sequence_buffer=sequence_buffer,
            page_manager=page_manager,
            max_batch_tokens=max_batch_tokens,
            prefill_priority=prefill_priority,
        )

    def add_request(
        self,
        tokens: Array,
        max_generate: int = 256,
        priority: float = 1.0,
    ) -> tuple[ContinuousBatchingEngine, int]:
        """Add a new request to the engine"""

        new_request_manager, request_id = self.request_manager.add_request(tokens, max_generate, priority)

        if request_id >= 0:
            new_page_manager, sequence_id = self.page_manager.allocate_sequence_slot()

            if sequence_id >= 0:
                new_request_manager = new_request_manager.assign_sequence(request_id, sequence_id)

                return ContinuousBatchingEngine(
                    request_manager=new_request_manager,
                    sequence_buffer=self.sequence_buffer,
                    page_manager=new_page_manager,
                    max_batch_tokens=self.max_batch_tokens,
                    prefill_priority=self.prefill_priority,
                ), request_id
            else:
                new_request_manager = new_request_manager.complete_request(request_id)
                return self, -1

        return self, -1

    def step(self) -> tuple[ContinuousBatchingEngine, BatchedSequences, bool]:
        """Execute one step of continuous batching"""

        new_request_manager, batch_tokens, batch_seq_ids, batch_phases = self.request_manager.get_next_batch(
            self.max_batch_tokens, self.prefill_priority
        )

        has_work = jnp.any(batch_tokens >= 0)

        if not has_work:
            return self, None, False

        valid_tokens = batch_tokens[batch_tokens >= 0]
        valid_seq_ids = batch_seq_ids[batch_tokens >= 0]

        new_sequence_buffer = self.sequence_buffer.enqueue_tokens(
            valid_tokens,
            valid_seq_ids,
            len(valid_tokens),
        )

        new_sequence_buffer, sequence_batch = new_sequence_buffer.pack_next_sequence(self.max_batch_tokens)

        new_page_manager, metadata = self.page_manager.allocate_pages_for_tokens(sequence_batch.sequence_ids)

        new_sequence_buffer = new_sequence_buffer.dequeue_tokens(sequence_batch.num_tokens)

        return (
            ContinuousBatchingEngine(
                request_manager=new_request_manager,
                sequence_buffer=new_sequence_buffer,
                page_manager=new_page_manager,
                max_batch_tokens=self.max_batch_tokens,
                prefill_priority=self.prefill_priority,
            ),
            sequence_batch,
            True,
        )

    def update_after_generation(
        self,
        sequence_ids: Array,
        new_tokens: Array,
    ) -> ContinuousBatchingEngine:
        """Update state after model generation"""

        new_sequence_buffer = self.sequence_buffer.update_after_sampling(
            new_tokens,
            sequence_ids,
            len(new_tokens),
        )

        new_request_manager = self.request_manager.update_generated_tokens(
            sequence_ids,
            new_tokens,
        )

        return ContinuousBatchingEngine(
            request_manager=new_request_manager,
            sequence_buffer=new_sequence_buffer,
            page_manager=self.page_manager,
            max_batch_tokens=self.max_batch_tokens,
            prefill_priority=self.prefill_priority,
        )

    def get_completed_requests(self) -> tuple[ContinuousBatchingEngine, list[tuple[int, Array]]]:
        """Get completed requests and their generated tokens"""

        completed_ids = self.request_manager.get_completed_requests()
        completed_ids = completed_ids[completed_ids >= 0]

        results = []
        new_engine = self

        for request_id in completed_ids:
            sequence_id = new_engine.request_manager.states.sequence_id[request_id]

            if sequence_id >= 0:
                new_sequence_buffer, generated_tokens = new_engine.sequence_buffer.get_tokens(
                    jnp.array([sequence_id]), max_tokens=new_engine.request_manager.states.max_generate[request_id]
                )
                new_sequence_buffer = new_sequence_buffer.dequeue_tokens_for_sequences(jnp.array([sequence_id]))
                new_page_manager = new_engine.page_manager.release_sequence(sequence_id)
                new_request_manager = new_engine.request_manager.complete_request(request_id)
                new_engine = ContinuousBatchingEngine(
                    request_manager=new_request_manager,
                    sequence_buffer=new_sequence_buffer,
                    page_manager=new_page_manager,
                    max_batch_tokens=new_engine.max_batch_tokens,
                    prefill_priority=new_engine.prefill_priority,
                )

                results.append((request_id, generated_tokens[0]))

        return new_engine, results

    def has_active_requests(self) -> bool:
        """Check if there are any active requests"""
        return self.request_manager.has_active_requests()

    def get_stats(self) -> dict:
        """Get engine statistics"""
        active_count = jnp.sum(self.request_manager.states.is_active)
        prefill_count = jnp.sum(
            (self.request_manager.states.phase == RequestPhase.PREFILL) & self.request_manager.states.is_active
        )
        decode_count = jnp.sum(
            (self.request_manager.states.phase == RequestPhase.DECODE) & self.request_manager.states.is_active
        )

        return {
            "active_requests": int(active_count),
            "prefill_requests": int(prefill_count),
            "decode_requests": int(decode_count),
            "queued_tokens": int(self.sequence_buffer.num_queued_tokens),
            "generated_tokens": int(self.sequence_buffer.num_generated_tokens),
            **self.page_manager.get_memory_stats(),
        }
