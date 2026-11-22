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

"""A slow fetch must not block the consumer, and must not lose rows on close.

``TrainBatchPrefetcher.next`` has to be able to hand out an already-collated batch while a later
ticket is still inside ``fetch_fn``. Closing during a fetch must still record that row for replay,
exactly once and in stream order.
"""

from __future__ import annotations

import itertools
import threading

from easydel.trainers.batch_prefetcher import TrainBatchPrefetcher

_WAIT_SECONDS = 30.0


def _blocking_fetch(block_index: int, started: threading.Event, release: threading.Event):
    """Build a fetch that blocks on ``release`` during its ``block_index``-th call."""
    calls = itertools.count()

    def fetch(data_iter, dataloader):
        if next(calls) == block_index:
            started.set()
            assert release.wait(_WAIT_SECONDS), "test never released the blocked fetch"
        return next(data_iter), data_iter

    return fetch


def _make_prefetcher(fetch_fn, stream):
    return TrainBatchPrefetcher(
        fetch_fn=fetch_fn,
        data_iter=iter(stream),
        dataloader=stream,
        data_collator=lambda batch: batch * 10,
        worker_count=2,
        buffer_size=2,
    )


def test_consumer_gets_buffered_batch_while_a_fetch_is_in_progress():
    started = threading.Event()
    release = threading.Event()
    stream = list(range(10))
    prefetcher = _make_prefetcher(_blocking_fetch(1, started, release), stream)
    delivered: list[int] = []
    done = threading.Event()

    def consume():
        delivered.append(prefetcher.next(schedule_next=False)[0])
        done.set()

    consumer = threading.Thread(target=consume)
    try:
        assert started.wait(_WAIT_SECONDS), "second fetch never started"
        consumer.start()
        got_batch_during_fetch = done.wait(timeout=10.0)
    finally:
        release.set()
        consumer.join(timeout=_WAIT_SECONDS)
        prefetcher.close()

    assert got_batch_during_fetch, "consumer blocked while a later ticket was inside fetch_fn"
    assert delivered == [0]


def test_close_during_a_fetch_records_the_row_for_replay_once_in_order():
    started = threading.Event()
    release = threading.Event()
    stream = list(range(10))
    prefetcher = _make_prefetcher(_blocking_fetch(1, started, release), stream)
    closed = threading.Event()

    def close():
        prefetcher.close()
        closed.set()

    closer = threading.Thread(target=close)
    try:
        assert started.wait(_WAIT_SECONDS), "second fetch never started"
        assert prefetcher.next(schedule_next=False)[0] == 0
        closer.start()
    finally:
        release.set()
    assert closed.wait(_WAIT_SECONDS), "close() did not return after the in-flight fetch finished"
    closer.join(timeout=_WAIT_SECONDS)

    assert list(prefetcher.data_iter) == stream[1:]


def test_replay_after_close_has_no_duplicates_or_gaps_with_slow_fetch():
    started = threading.Event()
    release = threading.Event()
    stream = list(range(12))
    prefetcher = _make_prefetcher(_blocking_fetch(1, started, release), stream)
    try:
        assert started.wait(_WAIT_SECONDS), "second fetch never started"
        first = prefetcher.next(schedule_next=False)[0]
        release.set()
        second = prefetcher.next(schedule_next=False)[0]
    finally:
        release.set()
        prefetcher.close()

    assert [first, second] == [0, 10]
    assert list(prefetcher.data_iter) == stream[2:]
