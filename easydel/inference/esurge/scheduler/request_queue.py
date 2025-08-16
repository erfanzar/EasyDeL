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

"""Request queue implementations for scheduling.

Provides different queueing strategies for managing inference requests,
including FCFS (first-come-first-served) and priority-based scheduling.

Classes:
    RequestQueue: Abstract base class for request queues
    FCFSRequestQueue: First-come-first-served queue
    PriorityRequestQueue: Priority-based request queue
    SchedulingPolicy: Enum of scheduling policies

Example:
    >>> queue = FCFSRequestQueue()
    >>> queue.add_request(request1)
    >>> queue.add_request(request2)
    >>> next_request = queue.pop_request()

    >>> priority_queue = PriorityRequestQueue()
    >>> priority_queue.add_request(high_priority_request)
    >>> priority_queue.add_request(low_priority_request)
    >>> next_request = priority_queue.pop_request()  # Gets high priority first
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from enum import Enum

from ..request import EngineRequest


class SchedulingPolicy(Enum):
    """Enum for scheduling policies.

    Attributes:
        FCFS: First-come-first-served scheduling.
        PRIORITY: Priority-based scheduling.
    """

    FCFS = "fcfs"
    PRIORITY = "priority"


class RequestQueue(ABC):
    """Abstract base class for request queues.

    Defines the interface for different request queueing strategies.
    Implementations must provide methods for adding, removing, and
    inspecting requests in the queue.
    """

    @abstractmethod
    def add_request(self, request: EngineRequest) -> None:
        """Add a request to the queue according to the policy.

        Args:
            request: The engine request to add.
        """
        pass

    @abstractmethod
    def pop_request(self) -> EngineRequest:
        """Pop a request from the queue according to the policy."""
        pass

    @abstractmethod
    def peek_request(self) -> EngineRequest:
        """Peek at the request at the front of the queue without removing it."""
        pass

    @abstractmethod
    def prepend_request(self, request: EngineRequest) -> None:
        """Prepend a request to the front of the queue."""
        pass

    @abstractmethod
    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        pass

    @abstractmethod
    def remove_request(self, request: EngineRequest) -> None:
        """Remove a specific request from the queue."""
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[EngineRequest]) -> None:
        """Remove multiple specific requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[EngineRequest]:
        """Iterate over the queue according to the policy."""
        pass

    @abstractmethod
    def __reversed__(self) -> Iterator[EngineRequest]:
        """Iterate over the queue in reverse order."""
        pass


class FCFSRequestQueue(deque[EngineRequest], RequestQueue):
    """A first-come-first-served queue that supports deque operations."""

    def add_request(self, request: EngineRequest) -> None:
        """Add a request to the queue according to FCFS policy."""
        self.append(request)

    def pop_request(self) -> EngineRequest:
        """Pop a request from the queue according to FCFS policy."""
        return self.popleft()

    def peek_request(self) -> EngineRequest:
        """Peek at the next request in the queue without removing it."""
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: EngineRequest) -> None:
        """Prepend a request to the front of the queue."""
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this
        queue."""
        self.extendleft(reversed(requests))

    def remove_request(self, request: EngineRequest) -> None:
        """Remove a specific request from the queue."""
        self.remove(request)

    def remove_requests(self, requests: Iterable[EngineRequest]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return super().__len__()

    def __iter__(self) -> Iterator[EngineRequest]:
        """Iterate over the queue according to FCFS policy."""
        return super().__iter__()

    def __reversed__(self) -> Iterator[EngineRequest]:
        """Iterate over the queue in reverse order."""
        return super().__reversed__()


class PriorityRequestQueue(RequestQueue):
    """
    A priority queue that supports heap operations.

    Requests with a smaller value of `priority` are processed first.
    If multiple requests have the same priority, the one with the earlier
    `arrival_time` is processed first.
    """

    def __init__(self) -> None:
        self._heap: list[tuple[int, float, EngineRequest]] = []

    def add_request(self, request: EngineRequest) -> None:
        """Add a request to the queue according to priority policy."""
        heapq.heappush(self._heap, (request.priority, request.arrival_time, request))

    def pop_request(self) -> EngineRequest:
        """Pop a request from the queue according to priority policy."""
        if not self._heap:
            raise IndexError("pop from empty heap")
        _, _, request = heapq.heappop(self._heap)
        return request

    def peek_request(self) -> EngineRequest:
        """Peek at the next request in the queue without removing it."""
        if not self._heap:
            raise IndexError("peek from empty heap")
        _, _, request = self._heap[0]
        return request

    def prepend_request(self, request: EngineRequest) -> None:
        """Add a request to the queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.

        Note: In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time)."""
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: EngineRequest) -> None:
        """Remove a specific request from the queue."""
        self._heap = [(p, t, r) for p, t, r in self._heap if r != request]
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[EngineRequest]) -> None:
        """Remove multiple specific requests from the queue."""
        requests_to_remove = set(requests)
        self._heap = [(p, t, r) for p, t, r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[EngineRequest]:
        """Iterate over the queue according to priority policy."""
        heap_copy = self._heap[:]
        while heap_copy:
            _, _, request = heapq.heappop(heap_copy)
            yield request

    def __reversed__(self) -> Iterator[EngineRequest]:
        """Iterate over the queue in reverse priority order."""
        return reversed(list(self))


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy."""
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
