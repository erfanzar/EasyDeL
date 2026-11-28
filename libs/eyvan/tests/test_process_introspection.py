# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Asking the control-plane process what it is doing, without stopping it.

A wedged scheduler presents identically to a healthy one from outside: the API
answers, health is green, the leader holds its lease, and nothing advances. The
stack is the only thing that differs, so the value of these is entirely in what
they say when the process is *busy* -- which is why the tests here run them
against live threads rather than a fixture.
"""

from __future__ import annotations

import threading
import tracemalloc

from eyvan.server.introspection import (
    MAX_STACK_FRAMES,
    allocations_as_json,
    capture_allocations,
    capture_thread_stacks,
)


def test_the_calling_thread_appears_with_this_test_in_its_stack() -> None:
    """The dump has to contain the frames an operator is looking for.

    A stack list that omits the caller would be a plausible-looking answer with
    the useful part missing, and nothing downstream could tell.
    """

    stacks = capture_thread_stacks()

    mine = threading.get_ident()
    stack = next(entry for entry in stacks if entry.thread_id == mine)
    assert any("test_the_calling_thread_appears" in frame for frame in stack.frames)


def test_a_blocked_worker_thread_is_visible_while_it_is_blocked() -> None:
    """The case the command exists for: a thread parked and not coming back.

    The thread is held on an event for the duration of the capture, so the
    assertion is about a stack taken *during* the block rather than after it --
    which is the only version of this that would diagnose a wedge.
    """

    release = threading.Event()
    entered = threading.Event()

    def park() -> None:
        entered.set()
        release.wait(timeout=10)

    worker = threading.Thread(target=park, name="eyvan-parked-probe")
    worker.start()
    try:
        assert entered.wait(timeout=10)
        stacks = capture_thread_stacks()
    finally:
        release.set()
        worker.join(timeout=10)

    parked = next(entry for entry in stacks if entry.name == "eyvan-parked-probe")
    assert any("park" in frame or "wait" in frame for frame in parked.frames)


def test_a_deep_stack_is_bounded_rather_than_unbounded(monkeypatch) -> None:
    """One recursive stack must not become the whole answer."""

    def recurse(depth: int) -> tuple:
        if depth:
            return recurse(depth - 1)
        return capture_thread_stacks()

    stacks = recurse(MAX_STACK_FRAMES * 2)

    mine = threading.get_ident()
    stack = next(entry for entry in stacks if entry.thread_id == mine)
    assert len(stack.frames) <= MAX_STACK_FRAMES


def test_allocations_report_whether_tracking_was_already_running() -> None:
    """The flag that decides whether the numbers mean anything.

    A snapshot from a tracker the request just started covers milliseconds, not
    the leak that has been growing since Tuesday -- and it looks reassuringly
    small, which is the wrong thing for it to look like.
    """

    was_tracing = tracemalloc.is_tracing()
    if was_tracing:  # pragma: no cover - depends on how the suite was invoked
        tracemalloc.stop()
    try:
        _sites, already_tracing = capture_allocations()
        assert already_tracing is False

        _sites, already_tracing_second = capture_allocations()
        assert already_tracing_second is True
    finally:
        if not was_tracing:
            tracemalloc.stop()


def test_the_allocation_document_carries_the_flag_beside_the_sites() -> None:
    """Serialised together, so no caller can read the sites without the caveat."""

    was_tracing = tracemalloc.is_tracing()
    try:
        document = allocations_as_json()
        assert "already_tracing" in document
        assert isinstance(document["sites"], list)
    finally:
        # The production endpoint intentionally starts a persistent tracker;
        # this test must not impose its allocation overhead on later tests.
        if not was_tracing:
            tracemalloc.stop()
