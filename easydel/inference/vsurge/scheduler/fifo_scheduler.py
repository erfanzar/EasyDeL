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
import queue
import threading
import time

from ..utils import ActiveRequest
from ._utils import RequestPriority, ScheduleDecision, SchedulePolicy


@dataclasses.dataclass
class BatchConfig:
    """Configuration for request batching"""

    min_batch_size: int = 1
    max_batch_wait_ms: float = 10.0
    force_batch_on_timeout: bool = True


class Scheduler:
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
    ):
        self.max_batch_size = max_batch_size
        self.max_prefill_batch_size = max_prefill_batch_size
        self.schedule_policy = schedule_policy
        self.high_load_threshold = high_load_threshold
        self.batch_config = batch_config or BatchConfig()

        self.prefill_queue: queue.PriorityQueue = queue.PriorityQueue()

        # Active request tracking
        self.active_requests: dict[int, ActiveRequest] = {}  # slot -> request
        self.request_to_slot: dict[str, int] = {}  # request.id -> slot

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

    def schedule(self) -> ScheduleDecision:
        """Make scheduling decisions for simple scheduler"""
        with self._lock:
            prefill_requests = []
            active_count = len([r for r in self.active_requests.values() if r is not None])
            load = active_count / self.max_batch_size

            # Determine effective policy
            if self.schedule_policy == SchedulePolicy.ADAPTIVE:
                effective_policy = SchedulePolicy.OFFLINE if load > self.high_load_threshold else SchedulePolicy.ONLINE
            else:
                effective_policy = self.schedule_policy

            # Calculate prefill budget
            prefill_budget = min(self.max_prefill_batch_size, self.max_batch_size - active_count)

            # Handle new requests
            if effective_policy == SchedulePolicy.ONLINE:
                # Process immediately
                self._process_new_requests_online(prefill_requests, prefill_budget)
            else:
                # Batch formation for offline mode
                self._process_new_requests_offline(prefill_requests, prefill_budget)

            # Handle decode scheduling
            decode_slots = []

            for slot, request in self.active_requests.items():
                if request is None:
                    continue

                # Check if ready for decode (has completed prefill)
                if (
                    hasattr(request, "prefill_result")
                    and request.prefill_result is not None
                    and slot not in [slot for slot, _ in prefill_requests]
                ):
                    decode_slots.append(slot)

            # Determine if we should decode
            should_decode = len(decode_slots) > 0

            if effective_policy == SchedulePolicy.OFFLINE and should_decode:
                # In offline mode, wait for more requests
                if len(decode_slots) < self.max_batch_size * 0.5:
                    should_decode = False

            return ScheduleDecision(
                prefill_requests=prefill_requests,
                decode_slots=decode_slots,
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

            return {
                "total_requests": self.total_requests,
                "completed_requests": self.completed_requests,
                "active_requests": active_count,
                "pending_batch_size": pending_count,
                "queue_size": queue_size,
                "utilization": active_count / self.max_batch_size if self.max_batch_size > 0 else 0,
                "scheduler_type": "simple_fifo",
            }
