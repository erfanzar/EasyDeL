# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What the control-plane process is doing right now, from inside it.

A wedged scheduler is the hardest thing this system can present, because every
external signal looks the same: the API answers, the leader lease is held, and
nothing advances. The question is which frame the process is sitting in, and
until this existed the only way to ask was to attach a debugger to production.

Two answers, both from the standard library and both taken without stopping the
process:

* the stacks of every live thread, which is what identifies a wedge;
* the largest tracked allocations, which is what identifies a leak.

There is deliberately **no CPU profile here**. Doing that honestly for an async
server means sampling the event loop over a window, and ``cProfile`` measures
one thread's call counts instead -- on this process it would attribute
everything to the loop's ``select`` and read as an idle server under any load at
all. A profiler that confidently answers the wrong question is worse than the
absence of one, because the absence sends an operator to look elsewhere.
"""

from __future__ import annotations

import sys
import threading
import traceback
import tracemalloc
import typing as tp
from dataclasses import dataclass

#: Frames kept per thread. Deep enough to cross the framework boundary and reach
#: Eyvan's own code, bounded so one recursive stack cannot dominate the answer.
MAX_STACK_FRAMES = 40

#: Allocation sites reported. The tail of this distribution is noise; the head is
#: the leak.
MAX_ALLOCATION_SITES = 25


@dataclass(frozen=True, slots=True)
class ThreadStack:
    """One live thread, named as the operator would recognise it."""

    thread_id: int
    name: str
    frames: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AllocationSite:
    """One source location and what it is currently holding."""

    location: str
    size_bytes: int
    count: int


def capture_thread_stacks() -> tuple[ThreadStack, ...]:
    """Return every live thread's stack, deepest frame last.

    ``sys._current_frames`` is a snapshot rather than a stop-the-world pause, so
    a stack may be one frame stale by the time it is formatted. That is the right
    trade for a process that is *serving*: a consistent dump that suspended the
    scheduler to get itself would be an outage taken to investigate a stall.
    """

    names = {thread.ident: thread.name for thread in threading.enumerate()}
    stacks: list[ThreadStack] = []
    for thread_id, frame in sys._current_frames().items():
        formatted = traceback.format_stack(frame)[-MAX_STACK_FRAMES:]
        stacks.append(
            ThreadStack(
                thread_id=thread_id,
                name=names.get(thread_id, "unknown"),
                frames=tuple(line.rstrip("\n") for line in formatted),
            )
        )
    return tuple(sorted(stacks, key=lambda stack: (stack.name, stack.thread_id)))


def capture_allocations() -> tuple[tuple[AllocationSite, ...], bool]:
    """Return the largest tracked allocations and whether tracking was already on.

    The flag is the honest half. ``tracemalloc`` only sees allocations made after
    it starts, so a snapshot from a tracker this call had to switch on describes
    the last few milliseconds and not the leak that has been growing for a day.
    Reported as "just started" rather than presented as a memory profile, because
    a small clean snapshot is exactly what a leak looks like from that angle.
    """

    already_tracing = tracemalloc.is_tracing()
    if not already_tracing:
        tracemalloc.start()
    snapshot = tracemalloc.take_snapshot()
    sites = tuple(
        AllocationSite(
            location=f"{stat.traceback[0].filename}:{stat.traceback[0].lineno}",
            size_bytes=stat.size,
            count=stat.count,
        )
        for stat in snapshot.statistics("lineno")[:MAX_ALLOCATION_SITES]
    )
    return sites, already_tracing


def thread_stacks_as_json() -> list[dict[str, tp.Any]]:
    """Render thread stacks for the API without leaking frame objects."""

    return [
        {"thread_id": stack.thread_id, "name": stack.name, "frames": list(stack.frames)}
        for stack in capture_thread_stacks()
    ]


def allocations_as_json() -> dict[str, tp.Any]:
    """Render the allocation snapshot for the API, tracking state included."""

    sites, already_tracing = capture_allocations()
    return {
        "already_tracing": already_tracing,
        "sites": [{"location": site.location, "size_bytes": site.size_bytes, "count": site.count} for site in sites],
    }


__all__ = (
    "MAX_ALLOCATION_SITES",
    "MAX_STACK_FRAMES",
    "AllocationSite",
    "ThreadStack",
    "allocations_as_json",
    "capture_allocations",
    "capture_thread_stacks",
    "thread_stacks_as_json",
)
