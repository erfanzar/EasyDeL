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

from .._shared.rollout_lifecycle import after_futures

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


def admission_is_feasible(
    deadlines: tp.Iterable[int],
    *,
    head_version: int,
    groups_per_update: int = 1,
) -> bool:
    """Return whether every outstanding rollout can still be consumed before its deadline.

    The trainer consumes ``groups_per_update`` rollouts per policy version,
    starting at ``head_version``, in earliest-deadline order. With the deadlines
    sorted ascending, the ``i``-th (1-based) deadline ``d_(i)`` is met iff
    ``d_(i) >= head_version`` and ``i <= groups_per_update * (d_(i) - head_version + 1)``.

    Args:
        deadlines: Last policy version at which each rollout may be consumed.
        head_version: Policy version the trainer consumes next.
        groups_per_update: Rollouts consumed per policy version.

    Returns:
        ``True`` if all deadlines are simultaneously satisfiable.

    Raises:
        ValueError: If ``groups_per_update`` is not positive.
    """

    if groups_per_update <= 0:
        raise ValueError(f"groups_per_update must be positive, got {groups_per_update}")
    for index, deadline in enumerate(sorted(int(value) for value in deadlines), start=1):
        if deadline < head_version or index > groups_per_update * (deadline - head_version + 1):
            return False
    return True


def min_admissible_version(
    outstanding_deadlines: tp.Iterable[int],
    *,
    head_version: int,
    max_staleness: int,
    groups_per_update: int = 1,
) -> int:
    """Return the earliest policy version at which a freshly anchored rollout is admissible.

    Assumes the outstanding rollouts are consumed in deadline order,
    ``groups_per_update`` per version, and that the retried rollout is
    anchored at the version it is resubmitted at (deadline
    ``version + max_staleness``).

    Args:
        outstanding_deadlines: Deadlines of the rollouts already admitted.
        head_version: Policy version the trainer consumes next.
        max_staleness: Allowed version lag of the retried rollout.
        groups_per_update: Rollouts consumed per policy version.

    Returns:
        The minimum version ``>= head_version`` at which to retry.

    Raises:
        ValueError: If ``max_staleness`` is negative or ``groups_per_update`` is not positive.
    """

    if max_staleness < 0:
        raise ValueError(f"max_staleness must be non-negative, got {max_staleness}")
    if groups_per_update <= 0:
        raise ValueError(f"groups_per_update must be positive, got {groups_per_update}")
    ordered = sorted(int(value) for value in outstanding_deadlines)
    version = int(head_version)
    while True:
        remaining = [value for value in ordered[groups_per_update * (version - head_version) :] if value >= version]
        candidate = [*remaining, version + max_staleness]
        if admission_is_feasible(candidate, head_version=version, groups_per_update=groups_per_update):
            return version
        version += 1


@dataclass(frozen=True, slots=True)
class RolloutTicket:
    """Stable identity and version metadata for one submitted rollout.

    Attributes:
        submission_id: Coordinator-unique submission number.
        policy_version: Snapshot version the rollout is anchored at.
        submitted_at: Monotonic submission timestamp.
        metadata: Immutable per-request metadata.
        deadline_version: Last policy version at which the rollout may be
            consumed, or ``None`` when deadline admission is disabled.
    """

    submission_id: int
    policy_version: int
    submitted_at: float
    metadata: tp.Mapping[str, tp.Any] = field(default_factory=dict)
    deadline_version: int | None = None

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


class AsyncRolloutAdmissionDeferred(RuntimeError):
    """Raised when a submission would miss its staleness deadline; the rollout was not started.

    Attributes:
        retry_at_version: Minimum policy version at which a rollout anchored at
            that version becomes admissible.
        deadline_version: Deadline the rejected submission would have had.
        head_version: Coordinator head version at rejection time.
    """

    def __init__(self, *, retry_at_version: int, deadline_version: int, head_version: int) -> None:
        super().__init__(
            f"rollout with deadline version {deadline_version} is infeasible at head version {head_version}; "
            f"retry from policy version {retry_at_version}"
        )
        self.retry_at_version = retry_at_version
        self.deadline_version = deadline_version
        self.head_version = head_version


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
    settled: bool = False


_CLOSED_SENTINEL = -1


class BoundedAsyncRolloutCoordinator(tp.Generic[SnapshotT, PayloadT, ResultT]):
    """Run bounded rollout work and consume it in completion order.

    The bound applies to all submitted-but-unconsumed rollouts, including
    already completed results. This provides backpressure for both worker
    execution and the completion queue, preventing fast producers from
    accumulating unbounded model batches while the learner is busy.

    Each admitted rollout holds exactly one capacity credit, returned exactly
    once when the rollout is settled: consumed by :meth:`next_completed`
    (success, cancellation, or worker error) or discarded by :meth:`close`.

    With ``max_staleness`` set, submissions use deadline admission: a rollout
    anchored at snapshot version ``v`` must be consumed by version
    ``v + max_staleness``. A submission is only started when every outstanding
    deadline stays satisfiable given the consumer takes ``groups_per_update``
    rollouts per version from :attr:`head_version` (see
    :func:`admission_is_feasible`); otherwise :class:`AsyncRolloutAdmissionDeferred`
    is raised and no work starts.

    Args:
        worker: Callable receiving ``(snapshot, payload)``.
        max_workers: Number of worker threads.
        max_outstanding: Maximum submitted-but-unconsumed rollouts. Defaults to
            ``max_workers``.
        thread_name_prefix: Worker thread prefix.
        max_staleness: Enables deadline admission when not ``None``.
        groups_per_update: Rollouts the consumer takes per policy version.
        head_version: Initial policy version the consumer takes next.
    """

    def __init__(
        self,
        worker: tp.Callable[[PolicySnapshot[SnapshotT], PayloadT], ResultT],
        *,
        max_workers: int = 1,
        max_outstanding: int | None = None,
        thread_name_prefix: str = "online-rl-rollout",
        max_staleness: int | None = None,
        groups_per_update: int = 1,
        head_version: int = 0,
    ) -> None:
        if max_workers <= 0:
            raise ValueError(f"max_workers must be positive, got {max_workers}")
        if max_outstanding is None:
            max_outstanding = max_workers
        if max_outstanding <= 0:
            raise ValueError(f"max_outstanding must be positive, got {max_outstanding}")
        if max_outstanding < max_workers:
            raise ValueError(f"max_outstanding ({max_outstanding}) cannot be smaller than max_workers ({max_workers})")
        if max_staleness is not None and max_staleness < 0:
            raise ValueError(f"max_staleness must be non-negative, got {max_staleness}")
        if groups_per_update <= 0:
            raise ValueError(f"groups_per_update must be positive, got {groups_per_update}")

        self._worker = worker
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix=thread_name_prefix,
        )
        self._max_outstanding = max_outstanding
        self._capacity = threading.BoundedSemaphore(max_outstanding)
        self._completion_ids: queue.Queue[int] = queue.Queue()
        self._records: dict[int, _SubmissionRecord[ResultT]] = {}
        self._lock = threading.Lock()
        self._next_submission_id = 0
        self._closed = False
        self._credits_in_use = 0
        self._max_staleness = max_staleness
        self._groups_per_update = int(groups_per_update)
        self._head_version = int(head_version)
        self._admission_deferred = 0

    @property
    def outstanding_count(self) -> int:
        """Number of submitted rollouts not yet consumed."""

        with self._lock:
            return len(self._records)

    @property
    def credits_in_use(self) -> int:
        """Capacity credits currently held by admitted, unsettled rollouts."""

        with self._lock:
            return self._credits_in_use

    @property
    def closed(self) -> bool:
        """Whether new submissions are disabled."""

        with self._lock:
            return self._closed

    @property
    def head_version(self) -> int:
        """Policy version the consumer takes next (deadline admission reference)."""

        with self._lock:
            return self._head_version

    @property
    def admission_deferred_count(self) -> int:
        """Number of submissions rejected by deadline admission."""

        with self._lock:
            return self._admission_deferred

    def advance_head(self, version: int) -> None:
        """Move the consumer head forward; the head never moves backwards.

        Args:
            version: Policy version the consumer takes next.
        """

        with self._lock:
            self._head_version = max(self._head_version, int(version))

    def _check_admission_locked(self, snapshot_version: int) -> int | None:
        """Raise :class:`AsyncRolloutAdmissionDeferred` if infeasible; return the deadline."""

        if self._max_staleness is None:
            return None
        deadline = int(snapshot_version) + self._max_staleness
        head = self._head_version
        # Rollouts already past their deadline are discarded by the consumer and occupy no update slot.
        outstanding = [
            record.ticket.deadline_version
            for record in self._records.values()
            if record.ticket.deadline_version is not None and record.ticket.deadline_version >= head
        ]
        if admission_is_feasible([*outstanding, deadline], head_version=head, groups_per_update=self._groups_per_update):
            return deadline
        self._admission_deferred += 1
        retry = min_admissible_version(
            outstanding,
            head_version=head,
            max_staleness=self._max_staleness,
            groups_per_update=self._groups_per_update,
        )
        raise AsyncRolloutAdmissionDeferred(retry_at_version=retry, deadline_version=deadline, head_version=head)

    def _settle_locked(self, submission_id: int) -> _SubmissionRecord[ResultT] | None:
        """Remove a record and return its credit exactly once. Caller holds ``_lock``."""

        record = self._records.pop(submission_id, None)
        if record is None or record.settled:
            return None
        record.settled = True
        self._credits_in_use -= 1
        self._capacity.release()
        return record

    def _on_future_done(self, submission_id: int) -> None:
        """Publish a completion unless the record was already settled by ``close``."""

        with self._lock:
            if submission_id in self._records:
                self._completion_ids.put_nowait(submission_id)

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
            snapshot: Inference-safe policy snapshot; its version is the
                rollout's anchor.
            metadata: Per-request metadata copied onto the returned ticket.
            timeout: Maximum seconds to wait for backpressure capacity. ``None``
                waits indefinitely.

        Returns:
            A stable ticket used for cancellation and diagnostics.

        Raises:
            AsyncRolloutAdmissionDeferred: If deadline admission is enabled and
                the rollout could not be consumed before its deadline. No work
                is started and no capacity is held.
            RuntimeError: If the coordinator is closed.
            TimeoutError: If capacity is not available before ``timeout``.
            ValueError: If ``timeout`` is negative.
        """

        if timeout is not None and timeout < 0:
            raise ValueError(f"timeout must be non-negative, got {timeout}")
        with self._lock:
            if self._closed:
                raise RuntimeError("cannot submit to a closed rollout coordinator")
            # Fail fast before blocking on capacity; re-checked atomically below.
            self._check_admission_locked(snapshot.version)

        acquired = self._capacity.acquire() if timeout is None else self._capacity.acquire(timeout=timeout)
        if not acquired:
            raise TimeoutError("timed out waiting for asynchronous rollout capacity")

        with self._lock:
            try:
                if self._closed:
                    raise RuntimeError("cannot submit to a closed rollout coordinator")
                deadline = self._check_admission_locked(snapshot.version)
                submission_id = self._next_submission_id
                ticket = RolloutTicket(
                    submission_id=submission_id,
                    policy_version=snapshot.version,
                    submitted_at=time.monotonic(),
                    metadata=_immutable_metadata(metadata),
                    deadline_version=deadline,
                )
                future = self._executor.submit(self._run_worker, snapshot, payload)
            except BaseException:
                self._capacity.release()
                raise
            self._next_submission_id += 1
            self._credits_in_use += 1
            self._records[submission_id] = _SubmissionRecord(
                ticket=ticket,
                snapshot_created_at=snapshot.created_at,
                snapshot_metadata=snapshot.metadata,
                future=future,
            )
        # Registered outside the lock: an already-finished future runs the callback inline.
        future.add_done_callback(lambda _future, completed_id=submission_id: self._on_future_done(completed_id))
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

        Consuming a success, cancellation, or failure settles the rollout and
        releases its backpressure credit exactly once.

        Args:
            timeout: Maximum seconds to wait. ``None`` waits indefinitely.

        Returns:
            The next successful rollout.

        Raises:
            AsyncRolloutCancelled: If the next completed future was cancelled.
            AsyncRolloutWorkerError: If its worker raised. The original
                exception is available as ``__cause__``.
            TimeoutError: If no completion arrives before ``timeout``.
            RuntimeError: If there is no outstanding work or the coordinator
                was closed while waiting.
            ValueError: If ``timeout`` is negative.
        """

        if timeout is not None and timeout < 0:
            raise ValueError(f"timeout must be non-negative, got {timeout}")
        with self._lock:
            if not self._records:
                raise RuntimeError("no outstanding rollouts")

        while True:
            try:
                submission_id = (
                    self._completion_ids.get() if timeout is None else self._completion_ids.get(timeout=timeout)
                )
            except queue.Empty as exc:
                raise TimeoutError("timed out waiting for an asynchronous rollout") from exc
            if submission_id == _CLOSED_SENTINEL:
                # Keep waking any other blocked consumer.
                self._completion_ids.put_nowait(_CLOSED_SENTINEL)
                raise RuntimeError("rollout coordinator was closed while waiting")
            with self._lock:
                record = self._settle_locked(submission_id)
            if record is not None:
                break

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
            future = record.future
        return future.cancel()

    def cancel_pending(self) -> int:
        """Cancel every future that has not started and return the count."""

        with self._lock:
            futures = [record.future for record in self._records.values()]
        return sum(future.cancel() for future in futures)

    def close(
        self,
        *,
        wait: bool = True,
        cancel_pending: bool = True,
        on_idle: tp.Callable[[], None] | None = None,
    ) -> None:
        """Stop accepting work and settle every outstanding rollout.

        Closing discards unconsumed completion records and returns their
        credits; a consumer blocked in :meth:`next_completed` is woken with
        ``RuntimeError``. Calling ``close`` again is a no-op. With
        ``wait=True`` the method waits for already-running worker calls.

        Args:
            wait: Wait for running workers to finish.
            cancel_pending: Cancel executor tasks that have not started.
            on_idle: Optional cleanup called after all running readers exit,
                including when ``wait=False`` returns before they finish.
        """

        with self._lock:
            if self._closed:
                return
            self._closed = True
            futures = [record.future for record in self._records.values()]
        self._executor.shutdown(wait=wait, cancel_futures=cancel_pending)
        if on_idle is not None:
            after_futures(futures, on_idle)
        with self._lock:
            for submission_id in tuple(self._records):
                self._settle_locked(submission_id)
            while True:
                try:
                    self._completion_ids.get_nowait()
                except queue.Empty:
                    break
            self._completion_ids.put_nowait(_CLOSED_SENTINEL)

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
    "AsyncRolloutAdmissionDeferred",
    "AsyncRolloutCancelled",
    "AsyncRolloutError",
    "AsyncRolloutWorkerError",
    "BoundedAsyncRolloutCoordinator",
    "CompletedRollout",
    "PolicySnapshot",
    "RolloutTicket",
    "admission_is_feasible",
    "min_admissible_version",
)
