# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Thread-local task profiling helpers for SpectraX MPMD runtime."""

from __future__ import annotations

import contextlib
import threading
import time
from collections.abc import Callable, Iterator

import jax

_PROFILER_STATE = threading.local()


def _active_profiler() -> "_Profiler | None":
    """Return the innermost active profiler on this thread, or ``None``.

    Uses thread-local storage so concurrent ``sxcall`` calls from
    different threads stay independent — useful for nested tests and
    any future multi-run orchestration.

    Returns:
        Return the innermost active profiler on this thread, or ``None``.
    """
    return getattr(_PROFILER_STATE, "active", None)


class _Profiler:
    """Per-task millisecond accumulator used by :func:`collect_task_times_ms`.

    Holds a flat ``task_name -> list[float_ms]`` dict that the
    :func:`_time_call` helper appends to whenever a labelled MPMD
    sub-task completes. One :class:`_Profiler` is active per thread at
    a time; nested ``collect_task_times_ms`` contexts share the outer
    profiler's dict so timings remain comparable.
    """

    def __init__(self) -> None:
        """Create a profiler with an empty ``task_name -> list[ms]`` map.

        The map is populated by :meth:`record` as :func:`sxcall`'s
        wrapped callables fire. A fresh profiler is constructed each
        time :func:`collect_task_times_ms` enters a new (non-nested)
        context.
        """
        self.times_ms: dict[str, list[float]] = {}

    def record(self, task_name: str, dt_ms: float) -> None:
        """Append a millisecond duration to the bucket for ``task_name``.

        The bucket is created on first use so callers do not need to
        register names ahead of time.

        Args:
            task_name: Profiler label (e.g. ``"stage0_fwd_mb3"``).
            dt_ms: Wall-clock duration of the task, including
                :func:`jax.block_until_ready`.
        """
        self.times_ms.setdefault(task_name, []).append(dt_ms)


@contextlib.contextmanager
def collect_task_times_ms() -> Iterator[dict[str, list[float]]]:
    """Record wall-clock milliseconds for each MPMD task in the body.

    Yields a ``dict[str, list[float]]`` that's filled as
    :func:`sxcall` executes: keys are task names like
    ``"stage0_fwd_mb3"`` / ``"stage2_bwd_i_mb0"``, values are a list
    of per-call durations in milliseconds (one entry per schedule
    action). :func:`jax.block_until_ready` is called before recording,
    so timings include actual device-side work, not just dispatch.

    Only one profiler may be active per thread at a time — nested
    calls share the outer profiler's dict.

    Example::

            with collect_task_times_ms() as times:
                loss, grads = sxcall(model, (x, y), mpmd_mesh=mm, ...)

            for name, ms in sorted(times.items()):
                print(f"{name}: {ms}")

    Returns:
        Result described by this helper.
    """
    outer = _active_profiler()
    if outer is not None:
        yield outer.times_ms
        return
    prof = _Profiler()
    _PROFILER_STATE.active = prof
    try:
        yield prof.times_ms
    finally:
        _PROFILER_STATE.active = None


def _time_call(
    task_name: str,
    fn: Callable[..., object],
    *args: object,
) -> object:
    """Invoke ``fn(*args)`` and record its wall time when a profiler is active.

    When no profiler is on the current thread the call is dispatched
    directly with no overhead. When one is active we wrap the call in
    :func:`time.perf_counter_ns` and :func:`jax.block_until_ready` so
    the recorded time reflects device-side work rather than just
    Python dispatch latency.

    Args:
        task_name: Profiler bucket label.
        fn: Callable to invoke.
        *args: Positional arguments for ``fn``.

    Returns:
        Whatever ``fn(*args)`` returned.
    """
    prof = _active_profiler()
    if prof is None:
        return fn(*args)
    t0 = time.perf_counter_ns()
    out = fn(*args)
    jax.block_until_ready(out)
    prof.record(task_name, (time.perf_counter_ns() - t0) / 1e6)
    return out
