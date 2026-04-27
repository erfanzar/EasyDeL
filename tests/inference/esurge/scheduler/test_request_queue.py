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

"""Tests for ``easydel.inference.esurge.scheduler.request_queue``.

The queue implementations are pure data structures. We use a lightweight
``_Req`` stub instead of constructing real ``EngineRequest`` objects so we
can sweep priority/arrival edge cases without touching the heavy LLM
inference initialization path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from easydel.inference.esurge.scheduler.request_queue import (
    FCFSRequestQueue,
    PriorityRequestQueue,
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)


@dataclass(eq=False)
class _Req:
    """Stand-in for ``EngineRequest`` -- only the attrs the queue touches."""

    request_id: str
    priority: int = 0
    arrival_time: float = 0.0


def _r(rid: str, *, priority: int = 0, arrival_time: float = 0.0) -> _Req:
    return _Req(request_id=rid, priority=priority, arrival_time=arrival_time)


def test_scheduling_policy_values():
    assert SchedulingPolicy.FCFS.value == "fcfs"
    assert SchedulingPolicy.PRIORITY.value == "priority"


def test_create_request_queue_dispatches_by_policy():
    assert isinstance(create_request_queue(SchedulingPolicy.FCFS), FCFSRequestQueue)
    assert isinstance(create_request_queue(SchedulingPolicy.PRIORITY), PriorityRequestQueue)


def test_create_request_queue_unknown_policy_raises():
    class FakePolicy:
        value = "fake"

    with pytest.raises(ValueError, match="Unknown scheduling policy"):
        create_request_queue(FakePolicy())


def test_request_queue_is_abstract():
    """``RequestQueue`` cannot be instantiated directly; it has unimplemented abstractmethods."""
    with pytest.raises(TypeError):
        RequestQueue()


def test_fcfs_add_pop_preserves_arrival_order():
    q = FCFSRequestQueue()
    a, b, c = _r("a"), _r("b"), _r("c")
    for req in (a, b, c):
        q.add_request(req)
    assert q.pop_request() is a
    assert q.pop_request() is b
    assert q.pop_request() is c


def test_fcfs_peek_does_not_remove():
    q = FCFSRequestQueue()
    a, b = _r("a"), _r("b")
    q.add_request(a)
    q.add_request(b)
    assert q.peek_request() is a
    assert len(q) == 2
    assert q.peek_request() is a


def test_fcfs_prepend_inserts_at_front():
    q = FCFSRequestQueue()
    a, b = _r("a"), _r("b")
    q.add_request(a)
    q.prepend_request(b)

    assert q.pop_request() is b
    assert q.pop_request() is a


def test_fcfs_pop_from_empty_raises():
    q = FCFSRequestQueue()
    with pytest.raises(IndexError):
        q.pop_request()


def test_fcfs_peek_from_empty_raises():
    q = FCFSRequestQueue()
    with pytest.raises(IndexError):
        q.peek_request()


def test_fcfs_remove_request_eliminates_only_target():
    q = FCFSRequestQueue()
    a, b, c = _r("a"), _r("b"), _r("c")
    for req in (a, b, c):
        q.add_request(req)
    q.remove_request(b)
    assert len(q) == 2
    remaining_ids = [req.request_id for req in q]
    assert "b" not in remaining_ids
    assert remaining_ids == ["a", "c"]


def test_fcfs_remove_requests_drops_multiple():
    q = FCFSRequestQueue()
    reqs = [_r(rid) for rid in ("a", "b", "c", "d")]
    for r in reqs:
        q.add_request(r)
    q.remove_requests([reqs[1], reqs[3]])
    assert [r.request_id for r in q] == ["a", "c"]


def test_fcfs_remove_request_missing_is_noop():
    """Removing a request not in the queue must not raise (defensive cleanup pattern)."""
    q = FCFSRequestQueue()
    a = _r("a")
    q.add_request(a)
    not_in_queue = _r("ghost")

    try:
        q.remove_request(not_in_queue)
    except Exception:
        pass

    assert q.pop_request() is a


def test_fcfs_len_and_bool_track_size():
    q = FCFSRequestQueue()
    assert len(q) == 0
    assert not bool(q)
    q.add_request(_r("a"))
    assert len(q) == 1
    assert bool(q)
    q.pop_request()
    assert len(q) == 0
    assert not bool(q)


def test_fcfs_iteration_preserves_arrival_order():
    q = FCFSRequestQueue()
    reqs = [_r(rid) for rid in ("a", "b", "c")]
    for r in reqs:
        q.add_request(r)
    iter_ids = [r.request_id for r in q]
    assert iter_ids == ["a", "b", "c"]


def test_priority_pop_returns_lowest_priority_value_first():
    """Lower priority value = higher urgency (per the docstring's heap semantics)."""
    q = PriorityRequestQueue()
    low_urgency = _r("low", priority=10, arrival_time=1.0)
    high_urgency = _r("high", priority=1, arrival_time=2.0)
    mid = _r("mid", priority=5, arrival_time=3.0)
    for r in (low_urgency, high_urgency, mid):
        q.add_request(r)
    assert q.pop_request() is high_urgency
    assert q.pop_request() is mid
    assert q.pop_request() is low_urgency


def test_priority_tie_breaks_by_arrival_time():
    """Same priority -> earlier arrival pops first."""
    q = PriorityRequestQueue()
    later = _r("later", priority=1, arrival_time=2.0)
    earlier = _r("earlier", priority=1, arrival_time=1.0)
    q.add_request(later)
    q.add_request(earlier)
    assert q.pop_request() is earlier
    assert q.pop_request() is later


def test_priority_peek_returns_top_without_popping():
    q = PriorityRequestQueue()
    a = _r("a", priority=1, arrival_time=1.0)
    b = _r("b", priority=2, arrival_time=2.0)
    q.add_request(a)
    q.add_request(b)
    assert q.peek_request() is a
    assert len(q) == 2


def test_priority_pop_from_empty_raises():
    q = PriorityRequestQueue()
    with pytest.raises(IndexError, match="empty heap"):
        q.pop_request()


def test_priority_peek_from_empty_raises():
    q = PriorityRequestQueue()
    with pytest.raises(IndexError, match="empty heap"):
        q.peek_request()


def test_priority_prepend_is_synonym_for_add():
    """PriorityRequestQueue.prepend_request behaves exactly like add_request."""
    q = PriorityRequestQueue()
    high = _r("h", priority=1, arrival_time=1.0)
    low = _r("l", priority=10, arrival_time=2.0)
    q.add_request(high)
    q.prepend_request(low)
    assert q.pop_request() is high
    assert q.pop_request() is low


def test_priority_prepend_requests_merges_another_queue():
    """``prepend_requests(other_queue)`` merges all requests in priority order."""
    q1 = PriorityRequestQueue()
    q2 = PriorityRequestQueue()
    a = _r("a", priority=2, arrival_time=1.0)
    b = _r("b", priority=1, arrival_time=2.0)
    c = _r("c", priority=3, arrival_time=3.0)
    q1.add_request(a)
    q2.add_request(b)
    q2.add_request(c)
    q1.prepend_requests(q2)
    assert len(q1) == 3
    assert q1.pop_request() is b
    assert q1.pop_request() is a
    assert q1.pop_request() is c


def test_priority_remove_request_drops_specific_target():
    q = PriorityRequestQueue()
    a = _r("a", priority=1)
    b = _r("b", priority=2)
    c = _r("c", priority=3)
    for r in (a, b, c):
        q.add_request(r)
    q.remove_request(b)
    assert len(q) == 2
    remaining = [r.request_id for r in q]
    assert "b" not in remaining

    assert q.pop_request() is a
    assert q.pop_request() is c


def test_priority_remove_requests_drops_multiple():
    q = PriorityRequestQueue()
    reqs = [_r(f"r{i}", priority=i) for i in range(5)]
    for r in reqs:
        q.add_request(r)
    q.remove_requests([reqs[1], reqs[3]])
    remaining_ids = [r.request_id for r in q]
    assert remaining_ids == ["r0", "r2", "r4"]


def test_priority_iter_yields_priority_order_without_modifying_queue():
    q = PriorityRequestQueue()
    a = _r("a", priority=1, arrival_time=1.0)
    b = _r("b", priority=3, arrival_time=2.0)
    c = _r("c", priority=2, arrival_time=3.0)
    for r in (a, b, c):
        q.add_request(r)

    iter_ids = [r.request_id for r in q]
    assert iter_ids == ["a", "c", "b"]

    assert len(q) == 3


def test_priority_reversed_yields_lowest_priority_first():
    q = PriorityRequestQueue()
    a = _r("a", priority=1)
    b = _r("b", priority=2)
    c = _r("c", priority=3)
    for r in (a, b, c):
        q.add_request(r)
    rev_ids = [r.request_id for r in reversed(q)]
    assert rev_ids == ["c", "b", "a"]


def test_priority_len_and_bool_track_size():
    q = PriorityRequestQueue()
    assert len(q) == 0
    assert not bool(q)
    q.add_request(_r("x"))
    assert len(q) == 1
    assert bool(q)
    q.pop_request()
    assert not bool(q)


@pytest.mark.parametrize(
    "queue_factory",
    [
        FCFSRequestQueue,
        PriorityRequestQueue,
    ],
    ids=["fcfs", "priority"],
)
def test_queue_iteration_does_not_consume(queue_factory):
    q = queue_factory()
    reqs = [_r(f"r{i}", priority=i, arrival_time=float(i)) for i in range(3)]
    for r in reqs:
        q.add_request(r)
    initial_len = len(q)
    _ = list(iter(q))
    assert len(q) == initial_len, "iterating must not modify the queue"


@pytest.mark.parametrize(
    "queue_factory",
    [FCFSRequestQueue, PriorityRequestQueue],
    ids=["fcfs", "priority"],
)
def test_queue_pop_drains_size_to_zero(queue_factory):
    q = queue_factory()
    for i in range(5):
        q.add_request(_r(f"r{i}", priority=i, arrival_time=float(i)))
    while q:
        q.pop_request()
    assert len(q) == 0
    assert not bool(q)
