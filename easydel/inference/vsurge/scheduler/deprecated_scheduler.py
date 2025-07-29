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

import dataclasses
import enum
import queue
import threading
import time

from ..utils import ActiveRequest


@enum.unique
class SchedulePolicy(enum.Enum):
    """Scheduling policies for request batching"""

    OFFLINE = enum.auto()
    ONLINE = enum.auto()
    ADAPTIVE = enum.auto()


@dataclasses.dataclass
class FIFOScheduleDecision:
    """Result of a simple scheduling decision"""

    prefill_requests: list[tuple[int, ActiveRequest]]  # (slot, request) pairs
    decode_slots: list[int]  # slots ready for decode
    should_decode: bool
    should_prefill: bool


class RequestPriority(enum.Enum):
    """Priority levels for request scheduling"""

    HIGH = 0
    NORMAL = 1
    LOW = 2


@dataclasses.dataclass
class BatchConfig:
    """Configuration for request batching"""

    min_batch_size: int = 1
    max_batch_wait_ms: float = 10.0
    force_batch_on_timeout: bool = True


class FIFOScheduler:
    """
    Simple FIFO scheduler for non-paged attention systems.
    Does NOT use chunked prefill - processes entire prompts at once.
    """

    def __init__(
        self,
        max_batch_size: int,
        max_prefill_batch_size: int,
        schedule_policy: SchedulePolicy = SchedulePolicy.ADAPTIVE,
        batch_config: BatchConfig | None = None,
        high_load_threshold: float = 0.8,
        max_decode_wait_iterations: int = 3,
    ):
        self.max_batch_size = max_batch_size
        self.max_prefill_batch_size = max_prefill_batch_size
        self.schedule_policy = schedule_policy
        self.high_load_threshold = high_load_threshold
        self.batch_config = batch_config or BatchConfig()
        self.max_decode_wait_iterations = max_decode_wait_iterations

        self.prefill_queue: queue.PriorityQueue = queue.PriorityQueue()

        # Active request tracking
        self.active_requests: dict[int, ActiveRequest] = {}  # slot -> request
        self.request_to_slot: dict[str, int] = {}  # request.id -> slot

        # Decode request tracking
        self.pending_decode_slots: set[int] = set()  # Slots waiting to decode
        self.decode_wait_counts: dict[int, int] = {}  # slot -> wait iterations

        self.pending_batch: list[tuple[float, int, ActiveRequest]] = []
        self.batch_start_time: float | None = None

        self.total_requests = 0
        self.completed_requests = 0

        self._lock = threading.Lock()

    def add_request(self, request: ActiveRequest, priority: RequestPriority = RequestPriority.NORMAL):
        """Add a new request to the scheduler"""
        with self._lock:
            self.total_requests += 1
            self.prefill_queue.put((priority.value, time.time(), request))

    def _find_available_slot(self) -> int:
        """Find an available slot for a new request"""
        for slot in range(self.max_batch_size):
            if slot not in self.active_requests or self.active_requests[slot] is None:
                return slot
        return -1

    def _should_form_batch(self) -> bool:
        """Determine if we should form a batch now"""
        if not self.pending_batch:
            return False

        # Check batch size
        if len(self.pending_batch) >= self.batch_config.min_batch_size:
            return True

        # Check timeout
        if self.batch_config.force_batch_on_timeout and self.batch_start_time:
            elapsed_ms = (time.time() - self.batch_start_time) * 1000
            if elapsed_ms >= self.batch_config.max_batch_wait_ms:
                return True

        return False

    def _update_pending_decode_slots(self):
        """Update the set of slots ready for decode"""
        self.pending_decode_slots.clear()

        for slot, request in self.active_requests.items():
            if request is None:
                continue

            # Check if ready for decode (has completed prefill)
            if hasattr(request, "prefill_result") and request.prefill_result is not None:
                self.pending_decode_slots.add(slot)

    def _get_starved_decode_slots(self) -> set[int]:
        """Get decode slots that have been waiting too long"""
        starved_slots = set()

        for slot in self.pending_decode_slots:
            wait_count = self.decode_wait_counts.get(slot, 0)
            if wait_count >= self.max_decode_wait_iterations:
                starved_slots.add(slot)

        return starved_slots

    def schedule(self) -> FIFOScheduleDecision:
        """Make scheduling decisions with decode request protection"""
        with self._lock:
            prefill_requests = []
            active_count = len([r for r in self.active_requests.values() if r is not None])
            load = active_count / self.max_batch_size

            # Determine effective policy
            if self.schedule_policy == SchedulePolicy.ADAPTIVE:
                effective_policy = SchedulePolicy.OFFLINE if load > self.high_load_threshold else SchedulePolicy.ONLINE
            else:
                effective_policy = self.schedule_policy

            # Update pending decode slots
            self._update_pending_decode_slots()

            # Check for starved decode requests
            starved_decode_slots = self._get_starved_decode_slots()

            # If we have starved decode requests, prioritize them
            if starved_decode_slots:
                # Reset wait counts for starved slots that will be processed
                for slot in starved_decode_slots:
                    self.decode_wait_counts[slot] = 0

                return FIFOScheduleDecision(
                    prefill_requests=[],
                    decode_slots=list(starved_decode_slots),
                    should_prefill=False,
                    should_decode=True,
                )

            # Calculate prefill budget (reduced if decode requests are waiting)
            base_prefill_budget = min(self.max_prefill_batch_size, self.max_batch_size - active_count)

            # Reduce prefill budget if decode requests are waiting
            if self.pending_decode_slots:
                prefill_budget = max(1, base_prefill_budget // 2)  # Allow at least 1 prefill
            else:
                prefill_budget = base_prefill_budget

            # Handle new requests
            if effective_policy == SchedulePolicy.ONLINE:
                self._process_new_requests_online(prefill_requests, prefill_budget)
            else:
                self._process_new_requests_offline(prefill_requests, prefill_budget)

            # Get decode slots to process
            decode_slots = list(self.pending_decode_slots)

            # Determine if we should decode
            should_decode = len(decode_slots) > 0

            if effective_policy == SchedulePolicy.OFFLINE and should_decode:
                # In offline mode, be less strict about batch size if requests are waiting
                min_decode_batch = self.max_batch_size * 0.5
                if any(self.decode_wait_counts.get(slot, 0) > 0 for slot in decode_slots):
                    min_decode_batch = max(1, self.max_batch_size * 0.25)  # Lower threshold for waiting requests

                if len(decode_slots) < min_decode_batch:
                    should_decode = False

            # Update wait counts for decode slots
            if should_decode:
                # Reset wait counts for slots that will be processed
                for slot in decode_slots:
                    self.decode_wait_counts[slot] = 0
            else:
                # Increment wait counts for slots that won't be processed
                for slot in decode_slots:
                    self.decode_wait_counts[slot] = self.decode_wait_counts.get(slot, 0) + 1

            return FIFOScheduleDecision(
                prefill_requests=prefill_requests,
                decode_slots=decode_slots if should_decode else [],
                should_prefill=len(prefill_requests) > 0,
                should_decode=should_decode,
            )

    def _process_new_requests_online(self, prefill_requests: list[tuple[int, ActiveRequest]], prefill_budget: int):
        """Process new requests immediately (online mode)"""
        temp_queue = []

        while not self.prefill_queue.empty() and len(prefill_requests) < prefill_budget:
            try:
                priority, timestamp, request = self.prefill_queue.get_nowait()

                # Find slot
                slot = self._find_available_slot()
                if slot < 0:
                    temp_queue.append((priority, timestamp, request))
                    break

                # Assign slot
                self.active_requests[slot] = request
                self.request_to_slot[request.id] = slot

                # Add to prefill list
                prefill_requests.append((slot, request))

            except queue.Empty:
                break

        # Put back unprocessed
        for item in temp_queue:
            self.prefill_queue.put(item)

    def _process_new_requests_offline(self, prefill_requests: list[tuple[int, ActiveRequest]], prefill_budget: int):
        """Batch requests before processing (offline mode)"""
        # Add to pending batch
        while not self.prefill_queue.empty() and len(self.pending_batch) < prefill_budget:
            try:
                priority, timestamp, request = self.prefill_queue.get_nowait()
                self.pending_batch.append((priority, timestamp, request))

                if self.batch_start_time is None:
                    self.batch_start_time = time.time()

            except queue.Empty:
                break

        # Check if we should form batch
        if self._should_form_batch():
            # Process the batch
            for priority, timestamp, request in self.pending_batch:
                if len(prefill_requests) >= prefill_budget:
                    # Put back in queue
                    self.prefill_queue.put((priority, timestamp, request))
                    continue

                slot = self._find_available_slot()
                if slot < 0:
                    self.prefill_queue.put((priority, timestamp, request))
                    continue

                # Assign slot
                self.active_requests[slot] = request
                self.request_to_slot[request.id] = slot

                # Add to prefill list
                prefill_requests.append((slot, request))

            # Clear pending batch
            self.pending_batch.clear()
            self.batch_start_time = None

    def complete_request(self, request_id: str):
        """Mark a request as completed and free resources"""
        with self._lock:
            if request_id in self.request_to_slot:
                slot = self.request_to_slot[request_id]

                # Free slot
                if slot in self.active_requests:
                    self.active_requests[slot] = None

                # Clean up tracking
                del self.request_to_slot[request_id]

                # Clean up decode tracking
                self.pending_decode_slots.discard(slot)
                self.decode_wait_counts.pop(slot, None)

                self.completed_requests += 1

    def force_batch(self):
        """Force pending batch to be processed (for idle periods)"""
        with self._lock:
            if self.pending_batch:
                self.batch_start_time = 0  # Force timeout

    def get_stats(self) -> dict:
        """Get scheduler statistics"""
        with self._lock:
            active_count = len([r for r in self.active_requests.values() if r is not None])
            pending_count = len(self.pending_batch)
            queue_size = self.prefill_queue.qsize()

            # Add decode wait stats
            max_decode_wait = max(self.decode_wait_counts.values()) if self.decode_wait_counts else 0
            avg_decode_wait = (
                sum(self.decode_wait_counts.values()) / len(self.decode_wait_counts) if self.decode_wait_counts else 0
            )

            return {
                "total_requests": self.total_requests,
                "completed_requests": self.completed_requests,
                "active_requests": active_count,
                "pending_batch_size": pending_count,
                "queue_size": queue_size,
                "pending_decode_slots": len(self.pending_decode_slots),
                "max_decode_wait": max_decode_wait,
                "avg_decode_wait": avg_decode_wait,
                "utilization": active_count / self.max_batch_size if self.max_batch_size > 0 else 0,
                "scheduler_type": "simple_fifo",
            }
