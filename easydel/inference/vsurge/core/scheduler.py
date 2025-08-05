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

import queue
import typing as tp

import numpy as np

from ..utils import ActiveRequest

if tp.TYPE_CHECKING:
    from .engine import vEngine


class SchedulerAction(tp.NamedTuple):
    """Represents the actions decided by the Scheduler for the engine."""

    prefill_requests: list[ActiveRequest]  # Requests to prefill
    decode_slots: list[int]  # Slots with active requests to decode
    active_requests: dict[int, ActiveRequest]  # Current mapping of slot -> request for context


class Scheduler:
    """
    Schedules requests for prefilling and decoding in the vEngine.

    The scheduler manages request queues and decode slot assignments,
    determining which actions the engine should perform at each step.
    """

    def __init__(self, engine: vEngine):
        """
        Initializes the Scheduler.

        Args:
            engine: The vEngine instance this scheduler is managing requests for.
        """
        self._engine = engine
        self.max_concurrent_decodes = engine.max_concurrent_decodes
        self.samples_per_slot = engine.samples_per_slot

        self._prefill_backlog: queue.Queue[ActiveRequest] = queue.Queue()
        self._free_slots: set[int] = set(range(self.max_concurrent_decodes))
        self._live_requests: dict[int, ActiveRequest] = {}

    def add_request(self, request: ActiveRequest):
        """
        Adds a new request to the scheduler's prefill queue.

        Args:
            request (ActiveRequest): The request to be added.
        """
        if not isinstance(request, ActiveRequest):
            raise TypeError("Request must be of type ActiveRequest")
        self._prefill_backlog.put(request)

    def schedule(self) -> SchedulerAction:
        """
        Determines the next actions for the engine.

        Returns:
            SchedulerAction: Actions for prefill and decode.
        """
        prefill_requests = []
        decode_slots = list(self._live_requests.keys())
        active_requests = self._live_requests

        if not self._prefill_backlog.empty() and self.has_free_slot():
            try:
                request_to_prefill = self._prefill_backlog.get_nowait()
                prefill_requests.append(request_to_prefill)
            except queue.Empty:
                pass

        return SchedulerAction(
            prefill_requests=prefill_requests,
            decode_slots=decode_slots,
            active_requests=active_requests,
        )

    def insert_prefill_result(self, request: ActiveRequest, slot: int):
        """
        Inserts a prefilled request into a decode slot.

        Args:
            request (ActiveRequest): The prefilled request.
            slot (int): The decode slot index.

        Raises:
            ValueError: If the slot is not free.
        """
        if slot not in self._free_slots:
            raise ValueError(f"Slot {slot} is not free for insertion.")

        self._free_slots.remove(slot)
        if not hasattr(request, "complete") or request.complete is None:
            request.complete = np.zeros((self.samples_per_slot,), dtype=np.bool_)
        self._live_requests[slot] = request

    def free_slot(self, slot: int):
        """
        Frees a decode slot.

        Args:
            slot (int): The slot index to free.

        Raises:
            ValueError: If the slot is not currently occupied.
        """
        if slot not in self._live_requests:
            raise ValueError(f"Slot {slot} is not currently occupied.")
        del self._live_requests[slot]
        self._free_slots.add(slot)

    def get_free_slot_count(self) -> int:
        """Returns the number of currently free decode slots."""
        return len(self._free_slots)

    def get_active_request_count(self) -> int:
        """Returns the number of currently active requests."""
        return len(self._live_requests)

    def has_free_slot(self) -> bool:
        """Checks if there is at least one free decode slot."""
        return len(self._free_slots) > 0

    def has_prefill_work(self) -> bool:
        """Checks if there are requests waiting to be prefilled."""
        return not self._prefill_backlog.empty()

    def has_decode_work(self) -> bool:
        """Checks if there are active requests to decode."""
        return len(self._live_requests) > 0

    def get_live_request_slots(self) -> list[int]:
        """Returns a list of slot indices currently holding active requests."""
        return list(self._live_requests.keys())
