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

The module includes:
    - Abstract RequestQueue base class defining the queue interface
    - FCFSRequestQueue for simple first-come-first-served ordering
    - PriorityRequestQueue for priority-based ordering using a heap
    - Factory function to create queues based on policy

Classes:
    RequestQueue: Abstract base class for request queues.
    FCFSRequestQueue: First-come-first-served queue implementation.
    PriorityRequestQueue: Priority-based request queue using a min-heap.
    SchedulingPolicy: Enum of scheduling policies.

Functions:
    create_request_queue: Factory function to create a queue based on policy.

Example:
    Basic FCFS queue usage::

        >>> queue = FCFSRequestQueue()
        >>> queue.add_request(request1)
        >>> queue.add_request(request2)
        >>> next_request = queue.pop_request()

    Priority queue usage::

        >>> priority_queue = PriorityRequestQueue()
        >>> priority_queue.add_request(high_priority_request)
        >>> priority_queue.add_request(low_priority_request)
        >>> next_request = priority_queue.pop_request()  # Gets high priority first

    Using the factory function::

        >>> queue = create_request_queue(SchedulingPolicy.PRIORITY)
        >>> queue.add_request(request)
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

    Defines the available scheduling policies that can be used to order
    requests in the scheduler's queue.

    Attributes:
        FCFS: First-come-first-served scheduling. Requests are processed
            in the order they arrive.
        PRIORITY: Priority-based scheduling. Requests with lower priority
            values are processed first. If priorities are equal, earlier
            arrivals are processed first.

    Example:
        >>> policy = SchedulingPolicy.FCFS
        >>> queue = create_request_queue(policy)
    """

    FCFS = "fcfs"
    """First-come-first-served scheduling policy."""

    PRIORITY = "priority"
    """Priority-based scheduling policy."""


class RequestQueue(ABC):
    """Abstract base class for request queues.

    Defines the interface for different request queueing strategies.
    Implementations must provide methods for adding, removing, and
    inspecting requests in the queue.

    All queue implementations must support:
        - Adding requests according to the policy
        - Popping the next request to process
        - Peeking at the next request without removal
        - Prepending requests (for preemption handling)
        - Removing specific requests
        - Standard container operations (bool, len, iter)

    Example:
        >>> class CustomQueue(RequestQueue):
        ...     def add_request(self, request):
        ...         # Implementation
        ...         pass
        ...     # ... implement other abstract methods
    """

    @abstractmethod
    def add_request(self, request: EngineRequest) -> None:
        """Add a request to the queue according to the policy.

        The request is inserted into the queue at the appropriate position
        based on the queue's scheduling policy (e.g., at the end for FCFS,
        or at the priority-appropriate position for priority queues).

        Args:
            request: The engine request to add to the queue.

        Example:
            >>> queue.add_request(new_request)
        """
        pass

    @abstractmethod
    def pop_request(self) -> EngineRequest:
        """Pop a request from the queue according to the policy.

        Removes and returns the next request to be processed according to
        the queue's scheduling policy.

        Returns:
            EngineRequest: The next request to process.

        Raises:
            IndexError: If the queue is empty.

        Example:
            >>> request = queue.pop_request()
        """
        pass

    @abstractmethod
    def peek_request(self) -> EngineRequest:
        """Peek at the request at the front of the queue without removing it.

        Returns the next request that would be returned by pop_request()
        without actually removing it from the queue.

        Returns:
            EngineRequest: The next request in the queue.

        Raises:
            IndexError: If the queue is empty.

        Example:
            >>> next_req = queue.peek_request()
            >>> # Queue unchanged, next_req is still in queue
        """
        pass

    @abstractmethod
    def prepend_request(self, request: EngineRequest) -> None:
        """Prepend a request to the front of the queue.

        For FCFS queues, this adds the request at the beginning. For priority
        queues, this is equivalent to add_request since order is determined
        by priority, not insertion order.

        This method is primarily used for handling preempted requests that
        need to be rescheduled with high priority.

        Args:
            request: The engine request to prepend.

        Example:
            >>> queue.prepend_request(preempted_request)
        """
        pass

    @abstractmethod
    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this queue.

        Transfers all requests from the source queue to the front of this
        queue. For FCFS queues, the order of the source requests is preserved.
        For priority queues, requests are merged according to priority.

        Args:
            requests: A RequestQueue containing requests to prepend.

        Example:
            >>> queue.prepend_requests(preempted_queue)
        """
        pass

    @abstractmethod
    def remove_request(self, request: EngineRequest) -> None:
        """Remove a specific request from the queue.

        Removes the given request from anywhere in the queue. This is used
        when a request is cancelled or finished before being processed.

        Args:
            request: The engine request to remove.

        Raises:
            ValueError: If the request is not in the queue.

        Example:
            >>> queue.remove_request(cancelled_request)
        """
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[EngineRequest]) -> None:
        """Remove multiple specific requests from the queue.

        Efficiently removes multiple requests from the queue in a single
        operation. Requests not in the queue are silently ignored.

        Args:
            requests: An iterable of engine requests to remove.

        Example:
            >>> queue.remove_requests([req1, req2, req3])
        """
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """Check if queue has any requests.

        Returns:
            bool: True if the queue contains at least one request,
                False otherwise.

        Example:
            >>> if queue:
            ...     request = queue.pop_request()
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Get number of requests in queue.

        Returns:
            int: The number of requests currently in the queue.

        Example:
            >>> print(f"Queue has {len(queue)} requests")
        """
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[EngineRequest]:
        """Iterate over the queue according to the policy.

        Returns an iterator that yields requests in the order they would
        be popped (according to the scheduling policy). Does not modify
        the queue.

        Returns:
            Iterator[EngineRequest]: Iterator over requests in queue order.

        Example:
            >>> for request in queue:
            ...     print(request.request_id)
        """
        pass

    @abstractmethod
    def __reversed__(self) -> Iterator[EngineRequest]:
        """Iterate over the queue in reverse order.

        Returns an iterator that yields requests in reverse order from
        what __iter__ would return.

        Returns:
            Iterator[EngineRequest]: Iterator over requests in reverse order.

        Example:
            >>> for request in reversed(queue):
            ...     print(request.request_id)
        """
        pass


class FCFSRequestQueue(deque[EngineRequest], RequestQueue):
    """A first-come-first-served queue that supports deque operations.

    This implementation uses a deque as the underlying data structure,
    providing O(1) operations for adding to the end and removing from
    the front.

    Requests are processed in the order they are added (FIFO). The queue
    inherits from both deque and RequestQueue, providing efficient
    implementations of all required operations.

    Example:
        >>> queue = FCFSRequestQueue()
        >>> queue.add_request(request1)
        >>> queue.add_request(request2)
        >>> first = queue.pop_request()  # Returns request1
        >>> second = queue.pop_request()  # Returns request2
    """

    def add_request(self, request: EngineRequest) -> None:
        """Add a request to the queue according to FCFS policy.

        The request is added to the end of the queue, maintaining FIFO order.

        Args:
            request: The engine request to add.

        Example:
            >>> queue.add_request(new_request)
        """
        self.append(request)

    def pop_request(self) -> EngineRequest:
        """Pop a request from the queue according to FCFS policy.

        Removes and returns the oldest request in the queue (the one
        added earliest).

        Returns:
            EngineRequest: The oldest request in the queue.

        Raises:
            IndexError: If the queue is empty.

        Example:
            >>> request = queue.pop_request()
        """
        return self.popleft()

    def peek_request(self) -> EngineRequest:
        """Peek at the next request in the queue without removing it.

        Returns the oldest request that would be returned by pop_request().

        Returns:
            EngineRequest: The oldest request in the queue.

        Raises:
            IndexError: If the queue is empty.

        Example:
            >>> next_req = queue.peek_request()
        """
        if not self:
            raise IndexError("peek from an empty queue")
        return self[0]

    def prepend_request(self, request: EngineRequest) -> None:
        """Prepend a request to the front of the queue.

        The request is added at the front, making it the next request
        to be popped.

        Args:
            request: The engine request to prepend.

        Example:
            >>> queue.prepend_request(preempted_request)
            >>> # preempted_request will now be returned by next pop_request()
        """
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Prepend all requests from another queue to the front of this queue.

        Requests from the source queue are added to the front of this queue
        while preserving their relative order.

        Args:
            requests: A RequestQueue containing requests to prepend.

        Example:
            >>> queue.prepend_requests(preempted_queue)
        """
        self.extendleft(reversed(requests))

    def remove_request(self, request: EngineRequest) -> None:
        """Remove a specific request from the queue.

        Removes the first occurrence of the request from the queue.

        Args:
            request: The engine request to remove.

        Raises:
            ValueError: If the request is not in the queue.

        Example:
            >>> queue.remove_request(cancelled_request)
        """
        self.remove(request)

    def remove_requests(self, requests: Iterable[EngineRequest]) -> None:
        """Remove multiple specific requests from the queue.

        Efficiently removes all specified requests from the queue by
        filtering and rebuilding. Requests not in the queue are silently
        ignored.

        Args:
            requests: An iterable of engine requests to remove.

        Example:
            >>> queue.remove_requests([req1, req2, req3])
        """
        requests_to_remove = set(requests)
        filtered_requests = [req for req in self if req not in requests_to_remove]
        # deque does not support in-place filtering, so we need to clear
        # and extend
        self.clear()
        self.extend(filtered_requests)

    def __bool__(self) -> bool:
        """Check if queue has any requests.

        Returns:
            bool: True if the queue is non-empty.
        """
        return len(self) > 0

    def __len__(self) -> int:
        """Get number of requests in queue.

        Returns:
            int: Number of requests in the queue.
        """
        return super().__len__()

    def __iter__(self) -> Iterator[EngineRequest]:
        """Iterate over the queue according to FCFS policy.

        Yields requests from oldest to newest (front to back).

        Returns:
            Iterator[EngineRequest]: Iterator over requests.
        """
        return super().__iter__()

    def __reversed__(self) -> Iterator[EngineRequest]:
        """Iterate over the queue in reverse order.

        Yields requests from newest to oldest (back to front).

        Returns:
            Iterator[EngineRequest]: Reverse iterator over requests.
        """
        return super().__reversed__()


class PriorityRequestQueue(RequestQueue):
    """A priority queue that supports heap operations.

    This implementation uses a min-heap where requests with smaller priority
    values are processed first. If multiple requests have the same priority,
    the one with the earlier arrival time is processed first.

    The heap stores tuples of (priority, arrival_time, request) to enable
    proper ordering. This provides O(log n) insertion and O(log n) removal
    of the minimum element.

    Attributes:
        _heap: Internal heap storage as list of (priority, time, request) tuples.

    Example:
        >>> queue = PriorityRequestQueue()
        >>> queue.add_request(low_priority_request)   # priority=10
        >>> queue.add_request(high_priority_request)  # priority=1
        >>> first = queue.pop_request()  # Returns high_priority_request
    """

    def __init__(self) -> None:
        """Initialize an empty priority queue.

        Creates an empty heap for storing requests.
        """
        self._heap: list[tuple[int, float, EngineRequest]] = []

    def add_request(self, request: EngineRequest) -> None:
        """Add a request to the queue according to priority policy.

        The request is inserted into the heap at the appropriate position
        based on its priority and arrival time.

        Args:
            request: The engine request to add. Must have ``priority`` and
                ``arrival_time`` attributes.

        Example:
            >>> queue.add_request(new_request)
        """
        heapq.heappush(self._heap, (request.priority, request.arrival_time, request))

    def pop_request(self) -> EngineRequest:
        """Pop a request from the queue according to priority policy.

        Removes and returns the highest-priority request (lowest priority
        value). If priorities are equal, returns the earliest arrival.

        Returns:
            EngineRequest: The highest-priority request.

        Raises:
            IndexError: If the queue is empty.

        Example:
            >>> request = queue.pop_request()
        """
        if not self._heap:
            raise IndexError("pop from empty heap")
        _, _, request = heapq.heappop(self._heap)
        return request

    def peek_request(self) -> EngineRequest:
        """Peek at the next request in the queue without removing it.

        Returns the highest-priority request without removing it.

        Returns:
            EngineRequest: The highest-priority request.

        Raises:
            IndexError: If the queue is empty.

        Example:
            >>> next_req = queue.peek_request()
        """
        if not self._heap:
            raise IndexError("peek from empty heap")
        _, _, request = self._heap[0]
        return request

    def prepend_request(self, request: EngineRequest) -> None:
        """Add a request to the queue according to priority policy.

        In a priority queue, there is no concept of prepending to the
        front. Requests are always ordered by (priority, arrival_time).
        This method is equivalent to add_request.

        Args:
            request: The engine request to add.

        Note:
            For priority queues, this behaves identically to add_request().
            The request will be placed according to its priority, not at
            the front.

        Example:
            >>> queue.prepend_request(preempted_request)
        """
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        """Add all requests from another queue according to priority policy.

        In a priority queue, there is no concept of prepending to the
        front. Requests are ordered by (priority, arrival_time). All
        requests from the source queue are merged into this queue.

        Args:
            requests: A RequestQueue containing requests to add.

        Note:
            For priority queues, this behaves as a merge operation. Each
            request is placed according to its priority, not at the front.

        Example:
            >>> queue.prepend_requests(other_queue)
        """
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: EngineRequest) -> None:
        """Remove a specific request from the queue.

        Removes the given request from the heap and re-heapifies. This is
        an O(n) operation due to the need to search and rebuild.

        Args:
            request: The engine request to remove.

        Example:
            >>> queue.remove_request(cancelled_request)
        """
        self._heap = [(p, t, r) for p, t, r in self._heap if r != request]
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[EngineRequest]) -> None:
        """Remove multiple specific requests from the queue.

        Efficiently removes all specified requests from the queue in a
        single pass, then re-heapifies.

        Args:
            requests: An iterable of engine requests to remove.

        Example:
            >>> queue.remove_requests([req1, req2, req3])
        """
        requests_to_remove = set(requests)
        self._heap = [(p, t, r) for p, t, r in self._heap if r not in requests_to_remove]
        heapq.heapify(self._heap)

    def __bool__(self) -> bool:
        """Check if queue has any requests.

        Returns:
            bool: True if the queue contains at least one request.
        """
        return bool(self._heap)

    def __len__(self) -> int:
        """Get number of requests in queue.

        Returns:
            int: Number of requests in the queue.
        """
        return len(self._heap)

    def __iter__(self) -> Iterator[EngineRequest]:
        """Iterate over the queue according to priority policy.

        Creates a copy of the heap and yields requests in priority order
        (highest priority first). Does not modify the original queue.

        Returns:
            Iterator[EngineRequest]: Iterator yielding requests in
                priority order.

        Example:
            >>> for request in queue:
            ...     print(f"{request.priority}: {request.request_id}")
        """
        heap_copy = self._heap[:]
        while heap_copy:
            _, _, request = heapq.heappop(heap_copy)
            yield request

    def __reversed__(self) -> Iterator[EngineRequest]:
        """Iterate over the queue in reverse priority order.

        Yields requests from lowest priority to highest priority.

        Returns:
            Iterator[EngineRequest]: Iterator yielding requests in
                reverse priority order.
        """
        return reversed(list(self))


def create_request_queue(policy: SchedulingPolicy) -> RequestQueue:
    """Create request queue based on scheduling policy.

    Factory function that creates the appropriate RequestQueue implementation
    based on the specified scheduling policy.

    Args:
        policy: The scheduling policy to use. Must be a SchedulingPolicy enum value.

    Returns:
        RequestQueue: A new queue instance of the appropriate type:
            - FCFSRequestQueue for SchedulingPolicy.FCFS
            - PriorityRequestQueue for SchedulingPolicy.PRIORITY

    Raises:
        ValueError: If an unknown scheduling policy is provided.

    Example:
        >>> queue = create_request_queue(SchedulingPolicy.FCFS)
        >>> priority_queue = create_request_queue(SchedulingPolicy.PRIORITY)
    """
    if policy == SchedulingPolicy.PRIORITY:
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue()
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")
