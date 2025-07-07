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
import threading
import time
import typing
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .block_manager import BlockManager
from .sequence import SequenceState, SequenceStatus


@dataclass(frozen=True)
class SchedulerConfig:
    """Immutable scheduler configuration."""

    max_batch_size: int
    max_sequence_length: int
    block_size: int
    num_blocks: int
    eos_token_ids: list[int]
    preemption_mode: str = "longest_first"


@dataclass
class BatchInfo:
    """Information about a scheduled batch."""

    sequences: list[SequenceState]
    is_prefill: bool
    total_tokens: int
    block_copies: dict[int, int] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Provides a concise summary of the batch."""
        batch_type = "Prefill" if self.is_prefill else "Decode"
        return (
            f"BatchInfo("
            f"type={batch_type}, "
            f"num_sequences={len(self.sequences)}, "
            f"total_tokens={self.total_tokens}, "
            f"block_copies={len(self.block_copies)}"
            f")"
        )


class Scheduler:
    """Advanced scheduler with priority-based preemption."""

    def __init__(self, config: SchedulerConfig, block_manager: BlockManager):
        self.config = config
        self.block_manager = block_manager
        self.waiting_queue: list[tuple[float, int, SequenceState]] = []
        self.running_sequences: dict[int, SequenceState] = {}
        self.stats = defaultdict(int)
        self.request_times: dict[int, float] = {}
        self.completion_times: dict[int, float] = {}
        self.throughput_history: list[float] = []
        self._stats_lock = threading.Lock()

    def add_sequence(self, seq: SequenceState, priority: float = 0.0) -> None:
        """Add sequence with priority (lower = higher priority)."""
        with self._stats_lock:
            self.request_times[seq.metadata.seq_id] = time.time()
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
                self._preempt_sequences(blocks_needed - available_blocks)
                available_blocks = self.block_manager.allocator.num_free

            if blocks_needed <= available_blocks:
                try:
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

        for item in temp_waiting:
            heapq.heappush(self.waiting_queue, item)

        return BatchInfo(batch_sequences, is_prefill=True, total_tokens=total_tokens)

    def _schedule_decode_batch(self) -> BatchInfo:
        """Schedule sequences for decode (generation) step."""
        batch_sequences = []
        block_copies = {}
        sorted_running = sorted(self.running_sequences.values(), key=lambda s: s.metadata.seq_id)

        for seq in sorted_running:
            if len(batch_sequences) >= self.config.max_batch_size:
                break

            if seq.num_tokens % self.config.block_size == 0:
                try:
                    if seq.block_table:
                        last_block = seq.block_table[-1]
                        new_block, copied = self.block_manager.copy_on_write(last_block)
                        if copied:
                            block_copies[last_block] = new_block
                            seq.block_table[-1] = new_block

                    new_blocks = self.block_manager.allocate_blocks(1)
                    seq.block_table.extend(new_blocks)
                except MemoryError:
                    continue

            batch_sequences.append(seq)
            self.stats["scheduled_decodes"] += 1

        return BatchInfo(batch_sequences, is_prefill=False, total_tokens=len(batch_sequences), block_copies=block_copies)

    def schedule(self) -> BatchInfo | None:
        """Schedule next batch of sequences."""
        if self.waiting_queue:
            batch = self._schedule_prefill_batch()
            if batch.sequences:
                return batch

        if self.running_sequences:
            return self._schedule_decode_batch()

        return None

    def finish_sequences(self, batch: BatchInfo, next_tokens: list[int]) -> list[SequenceState]:
        """Process generated tokens and identify finished sequences."""
        finished = []
        for seq, token in zip(batch.sequences, next_tokens, strict=False):
            seq.append_token(token)
            if len(seq.output_tokens) >= seq.metadata.max_tokens or token in self.config.eos_token_ids:
                seq.status = SequenceStatus.FINISHED
                finished.append(seq)
                del self.running_sequences[seq.metadata.seq_id]
                for block_id in seq.block_table:
                    self.block_manager.decrement_ref(block_id)
                seq.block_table.clear()

                self.stats["completed_sequences"] += 1

        with self._stats_lock:
            current_time = time.time()
            for seq in finished:
                if seq.metadata.seq_id in self.request_times:
                    latency = current_time - self.request_times[seq.metadata.seq_id]
                    self.completion_times[seq.metadata.seq_id] = current_time
                    self.stats["total_latency"] += latency
                    self.stats["avg_latency"] = self.stats["total_latency"] / max(1, self.stats["completed_sequences"])
                    throughput = len(seq.output_tokens) / latency if latency > 0 else 0
                    self.throughput_history.append(throughput)
                    if len(self.throughput_history) > 1000:
                        self.throughput_history.pop(0)

        return finished

    def get_stats(self) -> dict[str, typing.Any]:
        """Get comprehensive statistics."""
        with self._stats_lock:
            stats = dict(self.stats)
            stats.update(
                {
                    "waiting_sequences": len(self.waiting_queue),
                    "running_sequences": len(self.running_sequences),
                    "avg_throughput": np.mean(self.throughput_history) if self.throughput_history else 0,
                    "memory_utilization": 1.0
                    - (self.block_manager.allocator.num_free / self.block_manager.allocator.num_blocks),
                    "total_blocks": self.block_manager.allocator.num_blocks,
                    "free_blocks": self.block_manager.allocator.num_free,
                }
            )
        return stats

    def __repr__(self) -> str:
        """Provides a high-level snapshot of the scheduler's state."""
        try:
            free = self.block_manager.allocator.num_free
            total = self.block_manager.allocator.num_blocks
            block_info = f"{free}/{total}"
        except AttributeError:
            block_info = "N/A"

        return (
            f"Scheduler("
            f"waiting={len(self.waiting_queue)}, "
            f"running={len(self.running_sequences)}, "
            f"free_blocks={block_info}, "
            f"preemption='{self.config.preemption_mode}'"
            f")"
        )


__all__ = ("BatchInfo", "Scheduler", "SchedulerConfig")
