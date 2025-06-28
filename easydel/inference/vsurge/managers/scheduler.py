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
from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field

from .block_manager import BlockManager
from .sequence import SequenceState, SequenceStatus


@dataclass(frozen=True)
class SchedulerConfig:
    """Immutable scheduler configuration."""

    max_batch_size: int
    max_sequence_length: int
    block_size: int
    num_blocks: int
    eos_token_ids: set[int]
    preemption_mode: str = "longest_first"


@dataclass
class BatchInfo:
    """Information about a scheduled batch."""

    sequences: list[SequenceState]
    is_prefill: bool
    total_tokens: int
    block_copies: dict[int, int] = field(default_factory=dict)


class PriorityScheduler:
    """Advanced scheduler with priority-based preemption."""

    def __init__(self, config: SchedulerConfig, block_manager: BlockManager):
        self.config = config
        self.block_manager = block_manager
        self.waiting_queue: list[tuple[float, int, SequenceState]] = []
        self.running_sequences: dict[int, SequenceState] = {}
        self.stats = defaultdict(int)

    def add_sequence(self, seq: SequenceState, priority: float = 0.0) -> None:
        """Add sequence with priority (lower = higher priority)."""
        heapq.heappush(self.waiting_queue, (priority, seq.metadata.seq_id, seq))
        self.stats["total_requests"] += 1

    def _calculate_preemption_score(self, seq: SequenceState) -> float:
        """Calculate preemption score based on configured mode."""
        if self.config.preemption_mode == "longest_first":
            return -len(seq.output_tokens)
        else:
            return seq.metadata.seq_id

    def _preempt_sequences(self, blocks_needed: int) -> list[SequenceState]:
        """Preempt running sequences to free blocks."""
        if not self.running_sequences:
            return []
        candidates = [(self._calculate_preemption_score(seq), seq) for seq in self.running_sequences.values()]
        candidates.sort()

        preempted = []
        blocks_freed = 0

        for _, seq in candidates:
            if blocks_freed >= blocks_needed:
                break
            for block_id in seq.block_table:
                self.block_manager.decrement_ref(block_id)
            blocks_freed += len(seq.block_table)
            seq.block_table.clear()
            seq.status = SequenceStatus.WAITING
            del self.running_sequences[seq.metadata.seq_id]
            self.add_sequence(seq, priority=-len(seq.output_tokens))

            preempted.append(seq)
            self.stats["preemptions"] += 1

        return preempted

    def _schedule_prefill_batch(self) -> BatchInfo:
        """Schedule sequences for prefill (initial processing)."""
        batch_sequences = []
        total_tokens = 0

        temp_waiting = []

        while self.waiting_queue and len(batch_sequences) < self.config.max_batch_size:
            priority, seq_id, seq = heapq.heappop(self.waiting_queue)

            if total_tokens + seq.num_tokens > self.config.max_batch_size:
                temp_waiting.append((priority, seq_id, seq))
                continue

            blocks_needed = seq.num_blocks
            available_blocks = self.block_manager.allocator.num_free

            if blocks_needed > available_blocks:
                # Try preemption
                self._preempt_sequences(blocks_needed - available_blocks)
                available_blocks = self.block_manager.allocator.num_free

            if blocks_needed <= available_blocks:
                try:
                    # Allocate blocks
                    seq.block_table = self.block_manager.allocate_blocks(blocks_needed)
                    seq.status = SequenceStatus.RUNNING
                    self.running_sequences[seq_id] = seq

                    batch_sequences.append(seq)
                    total_tokens += seq.num_tokens
                    self.stats["scheduled_prefills"] += 1
                except MemoryError:
                    temp_waiting.append((priority, seq_id, seq))
            else:
                temp_waiting.append((priority, seq_id, seq))

        # Re-add sequences that couldn't be scheduled
        for item in temp_waiting:
            heapq.heappush(self.waiting_queue, item)

        return BatchInfo(batch_sequences, is_prefill=True, total_tokens=total_tokens)

    def _schedule_decode_batch(self) -> BatchInfo:
        """Schedule sequences for decode (generation) step."""
        batch_sequences = []
        block_copies = {}

        # Sort by sequence ID for deterministic ordering
        sorted_running = sorted(self.running_sequences.values(), key=lambda s: s.metadata.seq_id)

        for seq in sorted_running:
            if len(batch_sequences) >= self.config.max_batch_size:
                break

            # Check if we need a new block
            if seq.num_tokens % self.config.block_size == 0:
                try:
                    # Copy-on-write for last block if needed
                    if seq.block_table:
                        last_block = seq.block_table[-1]
                        new_block, copied = self.block_manager.copy_on_write(last_block)
                        if copied:
                            block_copies[last_block] = new_block
                            seq.block_table[-1] = new_block

                    # Allocate new block
                    new_blocks = self.block_manager.allocate_blocks(1)
                    seq.block_table.extend(new_blocks)
                except MemoryError:
                    continue

            batch_sequences.append(seq)
            self.stats["scheduled_decodes"] += 1

        return BatchInfo(batch_sequences, is_prefill=False, total_tokens=len(batch_sequences), block_copies=block_copies)

    def schedule(self) -> BatchInfo | None:
        """Schedule next batch of sequences."""
        # Prioritize prefill if waiting sequences exist
        if self.waiting_queue:
            batch = self._schedule_prefill_batch()
            if batch.sequences:
                return batch

        # Schedule decode for running sequences
        if self.running_sequences:
            return self._schedule_decode_batch()

        return None

    def finish_sequences(self, batch: BatchInfo, next_tokens: list[int]) -> list[SequenceState]:
        """Process generated tokens and identify finished sequences."""
        finished = []

        for seq, token in zip(batch.sequences, next_tokens, strict=False):
            seq.append_token(token)

            # Check termination conditions
            if len(seq.output_tokens) >= seq.metadata.max_tokens or token in self.config.eos_token_ids:
                seq.status = SequenceStatus.FINISHED
                finished.append(seq)

                # Free resources
                del self.running_sequences[seq.metadata.seq_id]
                for block_id in seq.block_table:
                    self.block_manager.decrement_ref(block_id)
                seq.block_table.clear()

                self.stats["completed_sequences"] += 1

        return finished


__all__ = ("BatchInfo", "PriorityScheduler", "SchedulerConfig")
