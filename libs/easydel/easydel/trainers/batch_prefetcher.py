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

"""Ordered, multi-worker host-side batch prefetching for trainers.

The trainer-level producer pipeline (dataloader fetch + collation) runs on the
host and, for large global batches or heavy collators, can take longer than the
compiled device step. :class:`TrainBatchPrefetcher` hides that latency by
producing batches on background threads while the device is busy, with two
structural guarantees:

- **Deterministic order.** Fetching from the (shared, stateful) dataloader
  iterator is serialized through a ticket lock, so rows are consumed from the
  stream in exactly the order a serial loop would consume them, regardless of
  worker count. Collated batches are delivered in fetch order.
- **Bounded memory.** At most ``buffer_size`` batches are pending at any time;
  each pending batch holds a full collated global batch in host memory.

With ``worker_count > 1`` collation fans out across threads while fetch stays
serialized, so sustained producer throughput approaches
``max(fetch_time, collate_time / worker_count)`` per batch instead of
``fetch_time + collate_time``. The collator must be thread-safe when
``worker_count > 1``; fetch never runs concurrently, so the dataloader iterator
needs no locking of its own.

This module is trainer-flavor agnostic and accelerator agnostic: it touches no
JAX arrays and composes with any ``fetch_fn``/``data_collator`` pair.
"""

from __future__ import annotations

import collections
import collections.abc
import concurrent.futures
import threading
import typing as tp

from easydel.utils.helpers import capture_time  # pyright: ignore[reportPrivateLocalImportUsage]

__all__ = ("TrainBatchPrefetcher",)


class _PrefetcherClosed(Exception):
    """Raised inside worker tasks awakened after :meth:`TrainBatchPrefetcher.close`."""


class TrainBatchPrefetcher:
    """Ordered lookahead pipeline for host-side dataloader fetch and collation.

    Producer tasks are submitted with monotonically increasing tickets. Each
    task first waits for its ticket's turn to fetch (serializing iterator
    access and preserving stream order), then collates its raw batch in
    parallel with other workers. The consumer pops futures FIFO, so batches are
    delivered in ticket order even when collation completes out of order.

    Attributes:
        _fetch_fn: ``(data_iter, dataloader) -> (raw_batch, data_iter)``
            callable; typically ``BaseTrainer._get_next_batch``.
        _data_iter: Shared dataloader iterator, advanced only under the fetch
            ticket lock.
        _dataloader: Source dataloader, passed through to ``_fetch_fn`` for
            iterator re-creation on exhaustion.
        _data_collator: Callable applied to each raw batch. Must be
            thread-safe when ``worker_count > 1``.
        _worker_count: Number of producer threads.
        _buffer_size: Maximum number of pending (submitted, not yet consumed)
            batches.
        _pending: FIFO of outstanding futures, one per submitted ticket.
    """

    def __init__(
        self,
        *,
        fetch_fn: tp.Callable[
            [collections.abc.Iterator[tp.Any], collections.abc.Iterable[tp.Any]],
            tuple[tp.Any, collections.abc.Iterator[tp.Any]],
        ],
        data_iter: collections.abc.Iterator[tp.Any],
        dataloader: collections.abc.Iterable[tp.Any],
        data_collator: tp.Callable[[tp.Any], tp.Any],
        worker_count: int = 1,
        buffer_size: int | None = None,
    ) -> None:
        """Initialize the prefetcher and warm the pipeline to capacity.

        Args:
            fetch_fn: Callable pulling the next raw batch from
                ``(data_iter, dataloader)`` and returning
                ``(raw_batch, updated_iter)``.
            data_iter: Active dataloader iterator to read batches from.
            dataloader: Source dataloader; used by ``fetch_fn`` to re-create
                the iterator when it is exhausted.
            data_collator: Callable applied to each raw batch to produce the
                collated batch returned by :meth:`next`. Must be thread-safe
                when ``worker_count > 1``.
            worker_count: Parallel producer threads. Fetch stays serialized;
                only collation runs concurrently. Clamped to >= 1.
            buffer_size: Maximum pending batches. Defaults to
                ``worker_count``. Clamped to >= ``worker_count`` so every
                worker can hold a task.
        """
        worker_count = max(int(worker_count), 1)
        self._fetch_fn = fetch_fn
        self._data_iter = data_iter
        self._dataloader = dataloader
        self._data_collator = data_collator
        self._worker_count = worker_count
        self._buffer_size = max(int(buffer_size) if buffer_size is not None else worker_count, worker_count)
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="easydel-train-batch-prefetch",
        )
        self._pending: collections.deque[concurrent.futures.Future[tuple[tp.Any, float, float, tp.Any]]] = (
            collections.deque()
        )
        self._fetch_cond = threading.Condition()
        self._next_fetch_ticket = 0
        self._submitted_tickets = 0
        self._closed = False
        self._last_fetch_seconds: float | None = None
        self._last_collate_seconds: float | None = None
        # Iterator position as of the last batch actually DELIVERED via
        # :meth:`next`, distinct from ``_data_iter`` (advanced the instant a
        # ticket's fetch turn arrives, up to ``buffer_size`` ahead of
        # consumption). See :attr:`data_iter` for why the distinction matters.
        self._last_delivered_iter = data_iter
        self._top_up()

    def _load(self, ticket: int) -> tuple[tp.Any, float, float, tp.Any]:
        """Produce one batch: fetch in ticket order, then collate.

        Args:
            ticket: This task's position in the global fetch order.

        Returns:
            Tuple ``(batch, fetch_seconds, collate_seconds, iter_snapshot)``
            where ``iter_snapshot`` is the shared iterator's value immediately
            after this ticket's fetch.

        Raises:
            _PrefetcherClosed: If the prefetcher closed before this ticket's
                fetch turn arrived.
        """
        with self._fetch_cond:
            while ticket != self._next_fetch_ticket and not self._closed:
                self._fetch_cond.wait()
            if self._closed:
                raise _PrefetcherClosed
            try:
                with capture_time() as fetch_time:
                    raw_batch, self._data_iter = self._fetch_fn(self._data_iter, self._dataloader)
                fetch_seconds = float(fetch_time())
                iter_snapshot = self._data_iter
            finally:
                # Advance the turn even on a failed fetch so later tickets
                # don't deadlock waiting for one that will never complete.
                self._next_fetch_ticket = ticket + 1
                self._fetch_cond.notify_all()
        with capture_time() as collate_time:
            batch = self._data_collator(raw_batch)
        return batch, fetch_seconds, float(collate_time()), iter_snapshot

    def _top_up(self, limit: int | None = None) -> None:
        """Submit producer tasks until ``limit`` (default: full capacity) are pending."""
        target = self._buffer_size if limit is None else min(max(int(limit), 0), self._buffer_size)
        with self._fetch_cond:
            if self._closed:
                return
            while len(self._pending) < target:
                self._pending.append(self._executor.submit(self._load, self._submitted_tickets))
                self._submitted_tickets += 1

    def next(self, *, schedule_next: bool) -> tuple[tp.Any, float, float]:
        """Return the next batch in stream order, optionally refilling the pipeline.

        Args:
            schedule_next: When ``True``, top the pipeline back up to
                ``buffer_size`` pending batches after consuming this one.
                When ``False``, no new work is submitted, but batches already
                in flight keep producing (use :meth:`ensure_scheduled` to
                refill explicitly, e.g. for the bucket the rule selects next).

        Returns:
            Tuple ``(batch, wait_seconds, data_time)`` where ``wait_seconds``
            is the wall time spent blocking on the producer and ``data_time``
            is that batch's producer-side fetch + collate time.

        Raises:
            RuntimeError: If the prefetcher was closed before a batch was
                available.
        """
        with self._fetch_cond:
            future = self._pending.popleft() if self._pending else None
            closed = self._closed
        if future is None:
            if closed:
                raise RuntimeError("Batch prefetcher was closed before a batch was available.")
            self._top_up(limit=1)
            with self._fetch_cond:
                future = self._pending.popleft()
        with capture_time() as wait_time:
            batch, fetch_seconds, collate_seconds, iter_snapshot = future.result()
        self._last_fetch_seconds = fetch_seconds
        self._last_collate_seconds = collate_seconds
        with self._fetch_cond:
            self._last_delivered_iter = iter_snapshot
        if schedule_next:
            self._top_up()
        return batch, float(wait_time()), fetch_seconds + collate_seconds

    def ensure_scheduled(self) -> None:
        """Top the pipeline up to capacity if it has spare room.

        Used by the bucket-aware training loop to warm the *next* step's
        bucket (known ahead of time from the deterministic bucket rule) while
        the device is busy with the current step.
        """
        self._top_up()

    @property
    def data_iter(self) -> collections.abc.Iterator[tp.Any]:
        """Iterator position as of the last batch delivered via :meth:`next`.

        Deliberately not the latest FETCHED position. With readahead
        buffering, a ticket's fetch (and thus the shared iterator's advance)
        can complete up to ``buffer_size`` batches before the caller consumes
        it via :meth:`next`. A caller that persists this property as a resume
        position (e.g. after :meth:`close`) must see the position after the
        last batch it actually trained on, not the last one merely fetched
        into the buffer -- otherwise closing early silently skips the
        fetched-but-unconsumed batches still sitting in the buffer.
        """
        with self._fetch_cond:
            return self._last_delivered_iter

    @property
    def last_fetch_seconds(self) -> float | None:
        """Producer fetch time of the most recently consumed batch."""
        return self._last_fetch_seconds

    @property
    def last_collate_seconds(self) -> float | None:
        """Producer collate time of the most recently consumed batch."""
        return self._last_collate_seconds

    def close(self) -> None:
        """Cancel pending work and shut down the producer threads.

        Idempotent. Tasks that already fetched their rows (up to
        ``buffer_size`` of them) are discarded without being delivered, but
        this does *not* lose their place in the stream: :attr:`data_iter`
        reports the position as of the last batch actually delivered via
        :meth:`next`, not the latest fetch, so a caller that persists it as a
        resume position re-fetches exactly the discarded batches next time
        instead of silently skipping them.
        """
        with self._fetch_cond:
            self._closed = True
            pending = list(self._pending)
            self._pending.clear()
            self._fetch_cond.notify_all()
        for future in pending:
            future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
