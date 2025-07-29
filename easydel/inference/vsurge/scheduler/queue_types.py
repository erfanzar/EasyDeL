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
import heapq
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Iterator
from functools import total_ordering
from typing import Generic, TypeVar

from ..request_type import EngineRequest

# Type variable for generic queue operations
T = TypeVar("T", bound=EngineRequest)


class RequestQueue(ABC, Generic[T]):
    """
    Abstract base class for request queues with different scheduling policies.

    This class defines the interface for managing queues of engine requests
    with various scheduling strategies. Implementations must handle request
    ordering, insertion, removal, and iteration according to their specific
    scheduling policy.

    Type Parameters:
        T: Type of requests stored in the queue (must be EngineRequest or subclass).
    """

    @abstractmethod
    def add_request(self, request: T) -> None:
        """
        Add a request to the queue according to the scheduling policy.

        Args:
            request: The request to add to the queue.
        """
        pass

    @abstractmethod
    def pop_request(self) -> T:
        """
        Remove and return the next request according to the scheduling policy.

        Returns:
            The next request to be processed.

        Raises:
            IndexError: If the queue is empty.
        """
        pass

    @abstractmethod
    def peek_request(self) -> T:
        """
        Return the next request without removing it from the queue.

        Returns:
            The next request to be processed.

        Raises:
            IndexError: If the queue is empty.
        """
        pass

    @abstractmethod
    def prepend_request(self, request: T) -> None:
        """
        Add a request to the front of the queue (highest priority).

        Note: For priority queues, this may still respect priority ordering.

        Args:
            request: The request to prepend.
        """
        pass

    @abstractmethod
    def prepend_requests(self, requests: RequestQueue[T] | Iterable[T]) -> None:
        """
        Add multiple requests to the front of the queue.

        Args:
            requests: Queue or iterable of requests to prepend.
        """
        pass

    @abstractmethod
    def remove_request(self, request: T) -> None:
        """
        Remove a specific request from the queue.

        Args:
            request: The request to remove.

        Raises:
            ValueError: If the request is not in the queue.
        """
        pass

    @abstractmethod
    def remove_requests(self, requests: Iterable[T]) -> None:
        """
        Remove multiple specific requests from the queue.

        Args:
            requests: Iterable of requests to remove.
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Remove all requests from the queue."""
        pass

    @abstractmethod
    def __bool__(self) -> bool:
        """
        Check if queue has any requests.

        Returns:
            True if the queue is not empty, False otherwise.
        """
        pass

    @abstractmethod
    def __len__(self) -> int:
        """
        Get the number of requests in the queue.

        Returns:
            Number of requests currently in the queue.
        """
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        """
        Iterate over requests in the queue according to the scheduling policy.

        Returns:
            Iterator over requests in processing order.
        """
        pass

    @abstractmethod
    def __reversed__(self) -> Iterator[T]:
        """
        Iterate over requests in reverse order.

        Returns:
            Iterator over requests in reverse processing order.
        """
        pass

    def is_empty(self) -> bool:
        """
        Check if the queue is empty.

        Returns:
            True if the queue is empty, False otherwise.
        """
        return len(self) == 0

    def get_all_requests(self) -> list[T]:
        """
        Get all requests in the queue as a list.

        Returns:
            List of all requests in processing order.
        """
        return list(self)

    def contains_request(self, request: T) -> bool:
        """
        Check if a specific request is in the queue.

        Args:
            request: The request to search for.

        Returns:
            True if the request is in the queue, False otherwise.
        """
        return request in self

    def __contains__(self, request: T) -> bool:
        """
        Check if a request is in the queue.

        Args:
            request: The request to search for.

        Returns:
            True if the request is in the queue, False otherwise.
        """
        return any(req == request for req in self)


class FCFSRequestQueue(deque[EngineRequest], RequestQueue[EngineRequest]):
    """
    First-Come-First-Served (FCFS) request queue implementation.

    This queue processes requests in the order they arrive (FIFO).
    It extends Python's deque for efficient operations at both ends.

    Time Complexities:
        - add_request: O(1)
        - pop_request: O(1)
        - peek_request: O(1)
        - remove_request: O(n)
        - remove_requests: O(n)
    """

    def __init__(self, maxlen: int | None = None) -> None:
        """
        Initialize the FCFS queue.

        Args:
            maxlen: Maximum queue length. If specified, old requests
                   are discarded when the limit is exceeded.
        """
        super().__init__(maxlen=maxlen)
        self._maxlen = maxlen

    def add_request(self, request: EngineRequest) -> None:
        """
        Add a request to the end of the queue (FCFS policy).

        Args:
            request: The request to add.
        """
        self.append(request)

    def pop_request(self) -> EngineRequest:
        """
        Remove and return the oldest request (FCFS policy).

        Returns:
            The oldest request in the queue.

        Raises:
            IndexError: If the queue is empty.
        """
        if not self:
            raise IndexError("Cannot pop from an empty queue")
        return self.popleft()

    def peek_request(self) -> EngineRequest:
        """
        Return the oldest request without removing it.

        Returns:
            The oldest request in the queue.

        Raises:
            IndexError: If the queue is empty.
        """
        if not self:
            raise IndexError("Cannot peek into an empty queue")
        return self[0]

    def prepend_request(self, request: EngineRequest) -> None:
        """
        Add a request to the front of the queue (highest priority).

        Args:
            request: The request to prepend.
        """
        self.appendleft(request)

    def prepend_requests(self, requests: RequestQueue[EngineRequest] | Iterable[EngineRequest]) -> None:
        """
        Add multiple requests to the front of the queue.

        Args:
            requests: Queue or iterable of requests to prepend.
                     Requests are added in reverse order to maintain
                     their relative ordering.
        """
        if isinstance(requests, RequestQueue):
            # For RequestQueue, iterate in reverse to maintain order
            self.extendleft(reversed(requests))
        else:
            # For other iterables, convert to list and reverse
            self.extendleft(reversed(list(requests)))

    def remove_request(self, request: EngineRequest) -> None:
        """
        Remove a specific request from the queue.

        Args:
            request: The request to remove.

        Raises:
            ValueError: If the request is not in the queue.
        """
        try:
            self.remove(request)
        except ValueError as e:
            raise ValueError(f"Request {request.request_id} not found in queue") from e

    def remove_requests(self, requests: Iterable[EngineRequest]) -> None:
        """
        Remove multiple specific requests from the queue efficiently.

        Args:
            requests: Iterable of requests to remove.
        """
        requests_to_remove = set(requests)
        if not requests_to_remove:
            return

        # Filter out requests to remove
        filtered_requests = [req for req in self if req not in requests_to_remove]

        # Replace queue contents
        self.clear()
        self.extend(filtered_requests)

    def clear(self) -> None:
        """Remove all requests from the queue."""
        super().clear()

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self) > 0

    def get_queue_stats(self) -> dict[str, int | float | int | None]:
        """
        Get statistics about the queue.

        Returns:
            Dictionary with queue statistics.
        """
        return {
            "length": len(self),
            "max_length": self._maxlen,
            "is_empty": self.is_empty(),
            "is_full": self._maxlen is not None and len(self) >= self._maxlen,
        }


@total_ordering
class PriorityRequest:
    """
    Wrapper for requests in priority queue to handle comparison.

    This wrapper ensures proper ordering in the priority queue by
    comparing priority first, then arrival time as a tiebreaker.
    """

    def __init__(self, request: EngineRequest) -> None:
        """
        Initialize the priority request wrapper.

        Args:
            request: The engine request to wrap.
        """
        self.request = request
        self.priority = request.priority
        self.arrival_time = request.arrival_time

    def __lt__(self, other: PriorityRequest) -> bool:
        """
        Compare requests for priority ordering.

        Lower priority values are considered higher priority.
        For equal priorities, earlier arrival times have higher priority.

        Args:
            other: Another PriorityRequest to compare against.

        Returns:
            True if this request has higher priority than other.
        """
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.arrival_time < other.arrival_time

    def __eq__(self, other: object) -> bool:
        """
        Check equality based on the wrapped request.

        Args:
            other: Another object to compare against.

        Returns:
            True if the wrapped requests are equal.
        """
        if not isinstance(other, PriorityRequest):
            return NotImplemented
        return self.request == other.request

    def __hash__(self) -> int:
        """
        Hash based on the wrapped request.

        Returns:
            Hash of the wrapped request.
        """
        return hash(self.request)

    def __repr__(self) -> str:
        """String representation of the priority request."""
        return f"PriorityRequest(priority={self.priority}, arrival_time={self.arrival_time}, request={self.request})"


class PriorityRequestQueue(RequestQueue[EngineRequest]):
    """
    Priority-based request queue implementation using a binary heap.

    Requests with lower priority values are processed first. For requests
    with the same priority, those with earlier arrival times are processed first.

    Time Complexities:
        - add_request: O(log n)
        - pop_request: O(log n)
        - peek_request: O(1)
        - remove_request: O(n)
        - remove_requests: O(n)
    """

    def __init__(self) -> None:
        """Initialize the priority queue."""
        self._heap: list[PriorityRequest] = []
        self._request_map: dict[EngineRequest, PriorityRequest] = {}

    def add_request(self, request: EngineRequest) -> None:
        """
        Add a request to the queue according to priority.

        Args:
            request: The request to add.
        """
        priority_request = PriorityRequest(request)
        heapq.heappush(self._heap, priority_request)
        self._request_map[request] = priority_request

    def pop_request(self) -> EngineRequest:
        """
        Remove and return the highest priority request.

        Returns:
            The highest priority request.

        Raises:
            IndexError: If the queue is empty.
        """
        if not self._heap:
            raise IndexError("Cannot pop from an empty priority queue")

        priority_request = heapq.heappop(self._heap)
        request = priority_request.request
        del self._request_map[request]
        return request

    def peek_request(self) -> EngineRequest:
        """
        Return the highest priority request without removing it.

        Returns:
            The highest priority request.

        Raises:
            IndexError: If the queue is empty.
        """
        if not self._heap:
            raise IndexError("Cannot peek into an empty priority queue")
        return self._heap[0].request

    def prepend_request(self, request: EngineRequest) -> None:
        """
        Add a request with the highest possible priority.

        Note: In a priority queue, prepending means giving the request
        the highest priority (lowest priority value).

        Args:
            request: The request to add with highest priority.
        """
        # Create a copy with minimum priority to ensure it's processed first
        high_priority_request = EngineRequest(
            request_id=request.request_id,
            prompt_token_ids=request.prompt_token_ids,
            output_token_ids=request.output_token_ids,
            priority=min(req.priority for req in self._request_map.keys()) - 1 if self._request_map else 0,
            arrival_time=request.arrival_time,
        )
        self.add_request(high_priority_request)

    def prepend_requests(self, requests: RequestQueue[EngineRequest] | Iterable[EngineRequest]) -> None:
        """
        Add multiple requests with high priority.

        Args:
            requests: Queue or iterable of requests to add with high priority.
        """
        for request in requests:
            self.prepend_request(request)

    def remove_request(self, request: EngineRequest) -> None:
        """
        Remove a specific request from the queue.

        Args:
            request: The request to remove.

        Raises:
            ValueError: If the request is not in the queue.
        """
        if request not in self._request_map:
            raise ValueError(f"Request {request.request_id} not found in priority queue")

        # Remove from map
        priority_request = self._request_map.pop(request)

        # Rebuild heap without the removed request
        self._heap = [pr for pr in self._heap if pr != priority_request]
        heapq.heapify(self._heap)

    def remove_requests(self, requests: Iterable[EngineRequest]) -> None:
        """
        Remove multiple specific requests from the queue.

        Args:
            requests: Iterable of requests to remove.
        """
        requests_to_remove = set(requests)
        if not requests_to_remove:
            return

        # Remove from map and collect priority requests to remove
        priority_requests_to_remove = set()
        for request in requests_to_remove:
            if request in self._request_map:
                priority_request = self._request_map.pop(request)
                priority_requests_to_remove.add(priority_request)

        # Rebuild heap without removed requests
        if priority_requests_to_remove:
            self._heap = [pr for pr in self._heap if pr not in priority_requests_to_remove]
            heapq.heapify(self._heap)

    def clear(self) -> None:
        """Remove all requests from the queue."""
        self._heap.clear()
        self._request_map.clear()

    def __bool__(self) -> bool:
        """Check if queue has any requests."""
        return len(self._heap) > 0

    def __len__(self) -> int:
        """Get number of requests in queue."""
        return len(self._heap)

    def __iter__(self) -> Iterator[EngineRequest]:
        """
        Iterate over requests in priority order.

        Note: This creates a copy of the heap to avoid modifying the original.

        Returns:
            Iterator over requests in priority order.
        """
        heap_copy = self._heap[:]
        while heap_copy:
            priority_request = heapq.heappop(heap_copy)
            yield priority_request.request

    def __reversed__(self) -> Iterator[EngineRequest]:
        """
        Iterate over requests in reverse priority order.

        Returns:
            Iterator over requests in reverse priority order.
        """
        # Convert to list and reverse for reverse iteration
        return reversed(list(self))

    def __contains__(self, request: EngineRequest) -> bool:
        """
        Check if a request is in the queue.

        Args:
            request: The request to search for.

        Returns:
            True if the request is in the queue.
        """
        return request in self._request_map

    def get_priority_stats(self) -> dict[str, int | float | list[int]]:
        """
        Get statistics about request priorities in the queue.

        Returns:
            Dictionary with priority statistics.
        """
        if not self._heap:
            return {
                "length": 0,
                "min_priority": None,
                "max_priority": None,
                "avg_priority": None,
                "priority_distribution": [],
            }

        priorities = [pr.priority for pr in self._heap]
        return {
            "length": len(self._heap),
            "min_priority": min(priorities),
            "max_priority": max(priorities),
            "avg_priority": sum(priorities) / len(priorities),
            "priority_distribution": sorted(set(priorities)),
        }

    def get_requests_by_priority(self, priority: int) -> list[EngineRequest]:
        """
        Get all requests with a specific priority.

        Args:
            priority: The priority level to filter by.

        Returns:
            List of requests with the specified priority.
        """
        return [pr.request for pr in self._heap if pr.priority == priority]


class SchedulingPolicy(enum.Enum):
    """
    Enumeration of available scheduling policies.

    Attributes:
        FCFS: First-Come-First-Served scheduling.
        PRIORITY: Priority-based scheduling.
    """

    FCFS = "fcfs"
    PRIORITY = "priority"

    def __str__(self) -> str:
        """String representation of the policy."""
        return self.value

    @classmethod
    def from_string(cls, policy_str: str) -> SchedulingPolicy:
        """
        Create a SchedulingPolicy from a string.

        Args:
            policy_str: String representation of the policy.

        Returns:
            SchedulingPolicy enum value.

        Raises:
            ValueError: If the policy string is not recognized.
        """
        policy_str = policy_str.lower().strip()
        for policy in cls:
            if policy.value == policy_str:
                return policy

        valid_policies = [p.value for p in cls]
        raise ValueError(f"Unknown scheduling policy: {policy_str}. Valid options: {valid_policies}")


def create_request_queue(policy: SchedulingPolicy | str, **kwargs) -> RequestQueue[EngineRequest]:
    """
    Factory function to create a request queue based on scheduling policy.

    Args:
        policy: The scheduling policy to use.
        **kwargs: Additional arguments passed to the queue constructor.

    Returns:
        RequestQueue instance configured with the specified policy.

    Raises:
        ValueError: If the policy is not recognized.

    Examples:
        >>> queue = create_request_queue(SchedulingPolicy.FCFS, maxlen=100)
        >>> queue = create_request_queue("priority")
        >>> queue = create_request_queue("fcfs")
    """
    if isinstance(policy, str):
        policy = SchedulingPolicy.from_string(policy)

    if policy == SchedulingPolicy.PRIORITY:
        if kwargs:
            raise ValueError("PriorityRequestQueue does not accept additional arguments")
        return PriorityRequestQueue()
    elif policy == SchedulingPolicy.FCFS:
        return FCFSRequestQueue(**kwargs)
    else:
        raise ValueError(f"Unknown scheduling policy: {policy}")


# Utility functions for queue management
def merge_queues(
    primary_queue: RequestQueue[EngineRequest], secondary_queue: RequestQueue[EngineRequest], policy: SchedulingPolicy
) -> RequestQueue[EngineRequest]:
    """
    Merge two request queues according to a scheduling policy.

    Args:
        primary_queue: The primary queue (higher priority).
        secondary_queue: The secondary queue (lower priority).
        policy: The scheduling policy for the merged queue.

    Returns:
        New queue containing all requests from both input queues.
    """
    merged_queue = create_request_queue(policy)

    # Add primary queue requests first
    for request in primary_queue:
        merged_queue.add_request(request)

    # Add secondary queue requests
    for request in secondary_queue:
        merged_queue.add_request(request)

    return merged_queue


def split_queue_by_condition(
    queue: RequestQueue[EngineRequest], condition: callable, policy: SchedulingPolicy
) -> tuple[RequestQueue[EngineRequest], RequestQueue[EngineRequest]]:
    """
    Split a queue into two queues based on a condition.

    Args:
        queue: The queue to split.
        condition: Function that takes a request and returns True/False.
        policy: The scheduling policy for the new queues.

    Returns:
        Tuple of (matching_queue, non_matching_queue).
    """
    matching_queue = create_request_queue(policy)
    non_matching_queue = create_request_queue(policy)

    for request in queue:
        if condition(request):
            matching_queue.add_request(request)
        else:
            non_matching_queue.add_request(request)

    return matching_queue, non_matching_queue


def get_queue_statistics(queue: RequestQueue[EngineRequest]) -> dict[str, int | float | str]:
    """
    Get comprehensive statistics about a request queue.

    Args:
        queue: The queue to analyze.

    Returns:
        Dictionary with queue statistics.
    """
    stats = {
        "length": len(queue),
        "is_empty": queue.is_empty(),
        "queue_type": type(queue).__name__,
    }

    if isinstance(queue, FCFSRequestQueue):
        stats.update(queue.get_queue_stats())
    elif isinstance(queue, PriorityRequestQueue):
        stats.update(queue.get_priority_stats())

    # Add request-level statistics if queue is not empty
    if queue:
        requests = list(queue)
        arrival_times = [req.arrival_time for req in requests]

        stats.update(
            {
                "oldest_request_age": max(arrival_times) - min(arrival_times) if len(arrival_times) > 1 else 0,
                "avg_arrival_time": sum(arrival_times) / len(arrival_times),
            }
        )

        # Add priority statistics for all queues
        if hasattr(requests[0], "priority"):
            priorities = [req.priority for req in requests]
            stats.update(
                {
                    "min_priority": min(priorities),
                    "max_priority": max(priorities),
                    "avg_priority": sum(priorities) / len(priorities),
                }
            )

    return stats


class QueueMonitor:
    """
    Monitor for tracking queue performance and behavior over time.
    """

    def __init__(self, queue: RequestQueue[EngineRequest]) -> None:
        """
        Initialize the queue monitor.

        Args:
            queue: The queue to monitor.
        """
        self.queue = queue
        self.total_requests_added = 0
        self.total_requests_removed = 0
        self.max_queue_length = 0
        self.total_wait_time = 0.0
        self.request_count_by_priority: dict[int, int] = {}

    def on_request_added(self, request: EngineRequest) -> None:
        """
        Called when a request is added to the queue.

        Args:
            request: The request that was added.
        """
        self.total_requests_added += 1
        self.max_queue_length = max(self.max_queue_length, len(self.queue))

        if hasattr(request, "priority"):
            priority = request.priority
            self.request_count_by_priority[priority] = self.request_count_by_priority.get(priority, 0) + 1

    def on_request_removed(self, request: EngineRequest, processing_time: float) -> None:
        """
        Called when a request is removed from the queue.

        Args:
            request: The request that was removed.
            processing_time: Time the request spent in the queue.
        """
        self.total_requests_removed += 1
        self.total_wait_time += processing_time

    def get_metrics(self) -> dict[str, int | float]:
        """
        Get monitoring metrics.

        Returns:
            Dictionary with monitoring metrics.
        """
        return {
            "total_requests_added": self.total_requests_added,
            "total_requests_removed": self.total_requests_removed,
            "current_queue_length": len(self.queue),
            "max_queue_length": self.max_queue_length,
            "avg_wait_time": (
                self.total_wait_time / self.total_requests_removed if self.total_requests_removed > 0 else 0.0
            ),
            "throughput": (self.total_requests_removed / (self.total_wait_time or 1.0)),
            "priority_distribution": dict(self.request_count_by_priority),
        }

    def reset(self) -> None:
        """Reset all monitoring metrics."""
        self.total_requests_added = 0
        self.total_requests_removed = 0
        self.max_queue_length = 0
        self.total_wait_time = 0.0
        self.request_count_by_priority.clear()


# Context manager for queue operations
class QueueTransaction:
    """
    Context manager for atomic queue operations.

    Allows multiple queue operations to be performed atomically,
    with rollback capability if an error occurs.
    """

    def __init__(self, queue: RequestQueue[EngineRequest]) -> None:
        """
        Initialize the transaction.

        Args:
            queue: The queue to operate on.
        """
        self.queue = queue
        self.original_state: list[EngineRequest] | None = None
        self.operations: list[tuple[str, tuple]] = []

    def __enter__(self) -> QueueTransaction:
        """Start the transaction by saving the current state."""
        self.original_state = list(self.queue)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        End the transaction.

        If an exception occurred, rollback to the original state.
        """
        if exc_type is not None and self.original_state is not None:
            # Rollback on exception
            self.queue.clear()
            for request in self.original_state:
                self.queue.add_request(request)

    def add_request(self, request: EngineRequest) -> None:
        """Add a request within the transaction."""
        self.queue.add_request(request)
        self.operations.append(("add", (request,)))

    def remove_request(self, request: EngineRequest) -> None:
        """Remove a request within the transaction."""
        self.queue.remove_request(request)
        self.operations.append(("remove", (request,)))

    def get_operations(self) -> list[tuple[str, tuple]]:
        """Get the list of operations performed in this transaction."""
        return self.operations.copy()
