# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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
"""Bounded completion-order coordination for asynchronous policy rollouts."""

from __future__ import annotations

import concurrent.futures
import queue
import threading
import time
import types
import typing as tp
from dataclasses import dataclass, field

SnapshotT = tp.TypeVar("SnapshotT")
PayloadT = tp.TypeVar("PayloadT")
ResultT = tp.TypeVar("ResultT")


def _immutable_metadata(metadata: tp.Mapping[str, tp.Any] | None) -> tp.Mapping[str, tp.Any]:
    """Return a shallow immutable metadata mapping."""

    return types.MappingProxyType({} if metadata is None else dict(metadata))


@dataclass(frozen=True, slots=True)
class PolicySnapshot(tp.Generic[SnapshotT]):
    """Versioned, caller-owned policy state used by rollout workers.

    The coordinator does not copy ``state`` because a model snapshot may be a
    large sharded pytree. Callers must supply an inference-safe immutable copy.

    Args:
        version: Monotonic policy version.
        state: Snapshot payload passed to the worker.
        metadata: Immutable diagnostics associated with the snapshot.
        created_at: Monotonic creation timestamp.
    """

    version: int
    state: SnapshotT
    metadata: tp.Mapping[str, tp.Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        """Validate version metadata."""

        if self.version < 0:
            raise ValueError(f"snapshot version must be non-negative, got {self.version}")
        object.__setattr__(self, "metadata", _immutable_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class RolloutTicket:
    """Stable identity and version metadata for one submitted rollout."""

    submission_id: int
    policy_version: int
    submitted_at: float
    metadata: tp.Mapping[str, tp.Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze request metadata."""

        object.__setattr__(self, "metadata", _immutable_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class CompletedRollout(tp.Generic[ResultT]):
    """A successful rollout returned in completion order."""

    value: ResultT
    ticket: RolloutTicket
    snapshot_created_at: float
    snapshot_metadata: tp.Mapping[str, tp.Any]
    started_at: float
    completed_at: float

    def __post_init__(self) -> None:
        """Freeze copied snapshot metadata."""

        object.__setattr__(self, "snapshot_metadata", _immutable_metadata(self.snapshot_metadata))

    @property
    def policy_version(self) -> int:
        """Policy version that produced the rollout."""

        return self.ticket.policy_version

    @property
    def queue_time(self) -> float:
        """Seconds spent waiting for a worker thread."""

        return max(0.0, self.started_at - self.ticket.submitted_at)

    @property
    def execution_time(self) -> float:
        """Seconds spent inside the rollout worker."""

        return max(0.0, self.completed_at - self.started_at)

    def staleness(self, current_policy_version: int) -> int:
        """Return non-negative policy-version staleness at consumption time."""

        return max(0, int(current_policy_version) - self.policy_version)


class AsyncRolloutError(RuntimeError):
    """Base error carrying the ticket for an unsuccessful rollout."""

    def __init__(self, message: str, ticket: RolloutTicket) -> None:
        super().__init__(message)
        self.ticket = ticket


class AsyncRolloutCancelled(AsyncRolloutError):
    """Raised when the next completion was cancelled before execution."""


class AsyncRolloutWorkerError(AsyncRolloutError):
    """Raised when a rollout worker fails."""

    def __init__(
        self,
        ticket: RolloutTicket,
        *,
        started_at: float,
        completed_at: float,
    ) -> None:
        super().__init__(
            f"rollout {ticket.submission_id} from policy version {ticket.policy_version} failed",
            ticket,
        )
        self.started_at = started_at
        self.completed_at = completed_at


@dataclass(frozen=True, slots=True)
class _WorkerSuccess(tp.Generic[ResultT]):
    """Internal successful worker result with timing."""

    value: ResultT
    started_at: float
    completed_at: float


class _WorkerFailure(Exception):
    """Internal wrapper retaining worker timing and original exception."""

    def __init__(self, exception: Exception, *, started_at: float, completed_at: float) -> None:
        super().__init__(str(exception))
        self.exception = exception
        self.started_at = started_at
        self.completed_at = completed_at


@dataclass(slots=True)
class _SubmissionRecord(tp.Generic[ResultT]):
    """Mutable coordinator bookkeeping for one future."""

    ticket: RolloutTicket
    snapshot_created_at: float
    snapshot_metadata: tp.Mapping[str, tp.Any]
    future: concurrent.futures.Future[_WorkerSuccess[ResultT]]


class BoundedAsyncRolloutCoordinator(tp.Generic[SnapshotT, PayloadT, ResultT]):
    """Run bounded rollout work and consume it in completion order.

    The bound applies to all submitted-but-unconsumed rollouts, including
    already completed results. This provides backpressure for both worker
    execution and the completion queue, preventing fast producers from
    accumulating unbounded model batches while the learner is busy.

    Args:
        worker: Callable receiving ``(snapshot, payload)``.
        max_workers: Number of worker threads.
        max_outstanding: Maximum submitted-but-unconsumed rollouts. Defaults to
            ``max_workers``.
        thread_name_prefix: Worker thread prefix.
    """

    def __init__(
        self,
        worker: tp.Callable[[PolicySnapshot[SnapshotT], PayloadT], ResultT],
        *,
        max_workers: int = 1,
        max_outstanding: int | None = None,
        thread_name_prefix: str = "online-rl-rollout",
    ) -> None:
        if max_workers <= 0:
            raise ValueError(f"max_workers must be positive, got {max_workers}")
        if max_outstanding is None:
            max_outstanding = max_workers
        if max_outstanding <= 0:
            raise ValueError(f"max_outstanding must be positive, got {max_outstanding}")
        if max_outstanding < max_workers:
            raise ValueError(f"max_outstanding ({max_outstanding}) cannot be smaller than max_workers ({max_workers})")

        self._worker = worker
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._capacity = threading.BoundedSemaphore(max_outstanding)
        self._completion_ids: queue.Queue[int] = queue.Queue(maxsize=max_outstanding)
        self._records: dict[int, _SubmissionRecord[ResultT]] = {}
        self._lock = threading.Lock()
        self._next_submission_id = 0
        self._closed = False

    @property
    def outstanding_count(self) -> int:
        """Number of submitted rollouts not yet consumed."""

        with self._lock:
            return len(self._records)

    @property
    def closed(self) -> bool:
        """Whether new submissions are disabled."""

        with self._lock:
            return self._closed

    def submit(
        self,
        payload: PayloadT,
        *,
        snapshot: PolicySnapshot[SnapshotT],
        metadata: tp.Mapping[str, tp.Any] | None = None,
        timeout: float | None = None,
    ) -> RolloutTicket:
        """Submit one rollout, blocking while the bounded queue is full.

        Args:
            payload: Request payload passed to the worker.
            snapshot: Inference-safe policy snapshot.
            metadata: Per-request metadata copied onto the returned ticket.
            timeout: Maximum seconds to wait for backpressure capacity. ``None``
                waits indefinitely.

        Returns:
            A stable ticket used for cancellation and diagnostics.

        Raises:
            RuntimeError: If the coordinator is closed.
            TimeoutError: If capacity is not available before ``timeout``.
            ValueError: If ``timeout`` is negative.
        """

        if timeout is not None and timeout < 0:
            raise ValueError(f"timeout must be non-negative, got {timeout}")
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot submit to a closed rollout coordinator")

        acquired = self._capacity.acquire() if timeout is None else self._capacity.acquire(timeout=timeout)
        if not acquired:
            raise TimeoutError("timed out waiting for asynchronous rollout capacity")

        with self._lock:
            if self._closed:
                self._capacity.release()
                raise RuntimeError("cannot submit to a closed rollout coordinator")
            submission_id = self._next_submission_id
            self._next_submission_id += 1
            ticket = RolloutTicket(
                submission_id=submission_id,
                policy_version=snapshot.version,
                submitted_at=time.monotonic(),
                metadata=_immutable_metadata(metadata),
            )
            future = self._executor.submit(self._run_worker, snapshot, payload)
            record = _SubmissionRecord(
                ticket=ticket,
                snapshot_created_at=snapshot.created_at,
                snapshot_metadata=snapshot.metadata,
                future=future,
            )
            self._records[submission_id] = record
            future.add_done_callback(
                lambda _future, completed_id=submission_id: self._completion_ids.put_nowait(completed_id)
            )
        return ticket

    def _run_worker(
        self,
        snapshot: PolicySnapshot[SnapshotT],
        payload: PayloadT,
    ) -> _WorkerSuccess[ResultT]:
        """Run the user worker and retain execution timing."""

        started_at = time.monotonic()
        try:
            value = self._worker(snapshot, payload)
        except Exception as exc:
            raise _WorkerFailure(exc, started_at=started_at, completed_at=time.monotonic()) from exc
        return _WorkerSuccess(value=value, started_at=started_at, completed_at=time.monotonic())

    def next_completed(self, *, timeout: float | None = None) -> CompletedRollout[ResultT]:
        """Consume the next rollout in completion order.

        Consuming a success, cancellation, or failure releases one unit of
        backpressure capacity.

        Args:
            timeout: Maximum seconds to wait. ``None`` waits indefinitely.

        Returns:
            The next successful rollout.

        Raises:
            AsyncRolloutCancelled: If the next completed future was cancelled.
            AsyncRolloutWorkerError: If its worker raised. The original
                exception is available as ``__cause__``.
            TimeoutError: If no completion arrives before ``timeout``.
            RuntimeError: If there is no outstanding work.
            ValueError: If ``timeout`` is negative.
        """

        if timeout is not None and timeout < 0:
            raise ValueError(f"timeout must be non-negative, got {timeout}")
        with self._lock:
            if not self._records:
                raise RuntimeError("no outstanding rollouts")

        try:
            submission_id = self._completion_ids.get() if timeout is None else self._completion_ids.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("timed out waiting for an asynchronous rollout") from exc

        with self._lock:
            record = self._records.pop(submission_id, None)
        if record is None:
            raise RuntimeError(f"completion for unknown rollout {submission_id}")
        self._capacity.release()

        if record.future.cancelled():
            raise AsyncRolloutCancelled(
                f"rollout {record.ticket.submission_id} from policy version "
                f"{record.ticket.policy_version} was cancelled",
                record.ticket,
            )
        try:
            outcome = record.future.result()
        except _WorkerFailure as failure:
            error = AsyncRolloutWorkerError(
                record.ticket,
                started_at=failure.started_at,
                completed_at=failure.completed_at,
            )
            raise error from failure.exception

        return CompletedRollout(
            value=outcome.value,
            ticket=record.ticket,
            snapshot_created_at=record.snapshot_created_at,
            snapshot_metadata=record.snapshot_metadata,
            started_at=outcome.started_at,
            completed_at=outcome.completed_at,
        )

    def cancel(self, ticket: RolloutTicket | int) -> bool:
        """Cancel a not-yet-running rollout.

        A successful cancellation remains a completion and must be consumed via
        :meth:`next_completed` before its backpressure slot is released.

        Args:
            ticket: Ticket or submission ID.

        Returns:
            ``True`` if the future was cancelled, otherwise ``False``.
        """

        submission_id = ticket if isinstance(ticket, int) else ticket.submission_id
        with self._lock:
            record = self._records.get(submission_id)
            if record is None:
                return False
            return record.future.cancel()

    def cancel_pending(self) -> int:
        """Cancel every future that has not started and return the count."""

        with self._lock:
            futures = [record.future for record in self._records.values()]
        return sum(future.cancel() for future in futures)

    def close(self, *, wait: bool = True, cancel_pending: bool = True) -> None:
        """Stop accepting work and release all outstanding capacity.

        Closing discards unconsumed completion records. With ``wait=True`` the
        method waits for already-running worker calls.

        Args:
            wait: Wait for running workers to finish.
            cancel_pending: Cancel executor tasks that have not started.
        """

        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=cancel_pending)
        with self._lock:
            outstanding = len(self._records)
            self._records.clear()
        for _ in range(outstanding):
            self._capacity.release()
        while True:
            try:
                self._completion_ids.get_nowait()
            except queue.Empty:
                break

    def __enter__(self) -> BoundedAsyncRolloutCoordinator[SnapshotT, PayloadT, ResultT]:
        """Return this coordinator as a context manager."""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: types.TracebackType | None,
    ) -> None:
        """Cancel pending tasks and wait for running workers on exit."""

        self.close(wait=True, cancel_pending=True)


__all__ = (
    "AsyncRolloutCancelled",
    "AsyncRolloutError",
    "AsyncRolloutWorkerError",
    "BoundedAsyncRolloutCoordinator",
    "CompletedRollout",
    "PolicySnapshot",
    "RolloutTicket",
)
