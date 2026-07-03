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

"""Behavioral tests for the ordered multi-worker train-batch prefetcher.

These assert the public contract of ``TrainBatchPrefetcher``: batches come out
in exact stream order with no loss or duplication regardless of worker count,
fetch access to the shared iterator is serialized in stream order, collation
genuinely overlaps across workers, producer errors surface on ``next()``, and
a failed fetch does not deadlock later batches.
"""

from __future__ import annotations

import itertools
import threading
import time

import pytest
from easydel.trainers.batch_prefetcher import TrainBatchPrefetcher


def _simple_fetch(data_iter, dataloader):
    """Fetch mirroring BaseTrainer._get_next_batch's (batch, iterator) contract."""
    try:
        return next(data_iter), data_iter
    except StopIteration:
        data_iter = iter(dataloader)
        return next(data_iter), data_iter


def test_delivers_exact_stream_order_with_parallel_collation():
    n = 40
    stream = list(range(n))

    def collator(item):
        # Later items sleep less, so unordered parallel collation would
        # complete (and without ordering, deliver) later items first.
        time.sleep((n - item) * 0.002)
        return item * 10

    prefetcher = TrainBatchPrefetcher(
        fetch_fn=_simple_fetch,
        data_iter=iter(stream),
        dataloader=stream,
        data_collator=collator,
        worker_count=4,
        buffer_size=6,
    )
    try:
        got = [prefetcher.next(schedule_next=True)[0] for _ in range(n)]
    finally:
        prefetcher.close()
    assert got == [item * 10 for item in stream]


def test_fetch_order_is_serialized_across_workers():
    fetch_log: list[int] = []
    counter = itertools.count()

    def logging_fetch(data_iter, dataloader):
        value = next(counter)
        fetch_log.append(value)
        return value, data_iter

    prefetcher = TrainBatchPrefetcher(
        fetch_fn=logging_fetch,
        data_iter=iter(()),
        dataloader=(),
        data_collator=lambda batch: batch,
        worker_count=4,
        buffer_size=8,
    )
    try:
        got = [prefetcher.next(schedule_next=True)[0] for _ in range(30)]
    finally:
        prefetcher.close()
    assert got == list(range(30))
    assert fetch_log[:30] == sorted(fetch_log[:30])


def test_collation_overlaps_across_workers():
    active = 0
    peak = 0
    lock = threading.Lock()

    def collator(item):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return item

    prefetcher = TrainBatchPrefetcher(
        fetch_fn=_simple_fetch,
        data_iter=iter(range(12)),
        dataloader=range(12),
        data_collator=collator,
        worker_count=4,
        buffer_size=4,
    )
    try:
        for _ in range(12):
            prefetcher.next(schedule_next=True)
    finally:
        prefetcher.close()
    assert peak > 1, f"collation never overlapped (peak concurrency {peak})"


def test_collator_error_propagates_at_the_failing_batch():
    def collator(item):
        if item == 5:
            raise ValueError("boom on 5")
        return item

    prefetcher = TrainBatchPrefetcher(
        fetch_fn=_simple_fetch,
        data_iter=iter(range(10)),
        dataloader=range(10),
        data_collator=collator,
        worker_count=3,
        buffer_size=3,
    )
    try:
        for expected in range(5):
            assert prefetcher.next(schedule_next=True)[0] == expected
        with pytest.raises(ValueError, match="boom on 5"):
            prefetcher.next(schedule_next=True)
        # Later batches were produced independently and stay consumable.
        assert prefetcher.next(schedule_next=True)[0] == 6
    finally:
        prefetcher.close()


def test_failed_fetch_does_not_deadlock_later_batches():
    counter = itertools.count()

    def flaky_fetch(data_iter, dataloader):
        value = next(counter)
        if value == 2:
            raise RuntimeError("fetch failed for batch 2")
        return value, data_iter

    prefetcher = TrainBatchPrefetcher(
        fetch_fn=flaky_fetch,
        data_iter=iter(()),
        dataloader=(),
        data_collator=lambda batch: batch,
        worker_count=3,
        buffer_size=3,
    )
    try:
        assert prefetcher.next(schedule_next=True)[0] == 0
        assert prefetcher.next(schedule_next=True)[0] == 1
        with pytest.raises(RuntimeError, match="fetch failed for batch 2"):
            prefetcher.next(schedule_next=True)
        # Ticket 2 advanced despite the failure; batch 3 arrives.
        assert prefetcher.next(schedule_next=True)[0] == 3
    finally:
        prefetcher.close()


def test_on_demand_refill_when_not_scheduling_next():
    stream = list(range(8))
    prefetcher = TrainBatchPrefetcher(
        fetch_fn=_simple_fetch,
        data_iter=iter(stream),
        dataloader=stream,
        data_collator=lambda batch: batch,
        worker_count=2,
        buffer_size=2,
    )
    try:
        # Drain the warm pipeline plus more without ever scheduling; next()
        # must top up on demand instead of raising or hanging.
        got = [prefetcher.next(schedule_next=False)[0] for _ in range(5)]
        assert got == stream[:5]
        prefetcher.ensure_scheduled()
        assert prefetcher.next(schedule_next=False)[0] == stream[5]
    finally:
        prefetcher.close()


def test_close_is_idempotent_and_next_after_close_raises():
    prefetcher = TrainBatchPrefetcher(
        fetch_fn=_simple_fetch,
        data_iter=iter(range(100)),
        dataloader=range(100),
        data_collator=lambda batch: batch,
        worker_count=4,
        buffer_size=4,
    )
    assert prefetcher.next(schedule_next=True)[0] == 0
    prefetcher.close()
    prefetcher.close()
    assert prefetcher.data_iter is not None
    with pytest.raises(RuntimeError, match="closed"):
        prefetcher.next(schedule_next=True)


def test_producer_timings_reported():
    prefetcher = TrainBatchPrefetcher(
        fetch_fn=_simple_fetch,
        data_iter=iter(range(4)),
        dataloader=range(4),
        data_collator=lambda batch: batch,
    )
    try:
        batch, wait_seconds, data_time = prefetcher.next(schedule_next=False)
    finally:
        prefetcher.close()
    assert batch == 0
    assert wait_seconds >= 0.0
    assert data_time >= 0.0
    assert prefetcher.last_fetch_seconds is not None
    assert prefetcher.last_collate_seconds is not None
