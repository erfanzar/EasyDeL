# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hang watchdog for the scheduled MPMD dispatcher.

Example::

    with dispatch_watchdog(stall_after_s=300, on_stall=alert) as wd:
        train()

A daemon thread polls the open instrumentation spans and reports a stall when

* a schedule unit or transfer has been running for ``stall_after_s``, or
* something is in flight but no span has begun or ended for ``stall_after_s``
  (``compile_stall_after_s`` while a warm compile is running).

A :class:`StallReport` (with each stalled thread's Python stack, which shows
whether it is blocked in a transport-gate wait, a ``device_put``, ...) is
logged as a warning and passed to ``on_stall``. Reports are throttled to one
per ``stall_after_s``. The watchdog only observes; it never cancels work.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from ..._internal.logging import get_logger
from . import instrumentation as _instr

__all__ = ["DispatchWatchdog", "StallReport", "StalledSpan", "dispatch_watchdog"]

logger = get_logger("MPMD-Watchdog")

_WORK_KINDS = ("unit", "transfer")


@dataclass(frozen=True)
class StalledSpan:
    """One in-flight span at the time of a stall report."""

    kind: str
    name: str
    rank: int | None
    age_s: float
    thread_id: int
    thread_name: str
    stack: str | None = None
    meta: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class StallReport:
    """Snapshot of in-flight dispatcher work when a stall was detected.

    Attributes:
        reason: ``"span_age"`` (some unit/transfer exceeded the limit) or
            ``"no_progress"`` (nothing began or ended for the limit).
        idle_s: Seconds since the last span began or ended.
        spans: The stalled spans (for ``no_progress``: every open span).
        open_count: Number of open spans of any kind.
    """

    reason: str
    idle_s: float
    spans: tuple[StalledSpan, ...]
    open_count: int

    def format(self) -> str:
        """Multi-line human-readable rendering, including stacks."""
        lines = [
            f"SpectraX MPMD dispatch stall ({self.reason}): idle={self.idle_s:.1f}s "
            f"open_spans={self.open_count} stalled={len(self.spans)}"
        ]
        for s in self.spans:
            lines.append(f"  - {s.kind} {s.name!r} rank={s.rank} age={s.age_s:.1f}s thread={s.thread_name}")
            if s.stack:
                lines.extend("      " + line for line in s.stack.rstrip().splitlines())
        return "\n".join(lines)


class DispatchWatchdog:
    """Instrumentation observer that reports stalled dispatcher work.

    Args:
        stall_after_s: Age / inactivity limit in seconds.
        poll_interval_s: How often the daemon thread checks.
        on_stall: Optional callback receiving each :class:`StallReport`.
        include_stacks: Capture the Python stack of each stalled thread.
        compile_stall_after_s: Inactivity limit while a warm compile is in
            flight (default ``4 * stall_after_s``).
        log: Log each report as a warning.
    """

    block_until_ready = False

    def __init__(
        self,
        *,
        stall_after_s: float = 300.0,
        poll_interval_s: float = 10.0,
        on_stall: Callable[[StallReport], None] | None = None,
        include_stacks: bool = True,
        compile_stall_after_s: float | None = None,
        log: bool = True,
    ) -> None:
        if stall_after_s <= 0 or poll_interval_s <= 0:
            raise ValueError("stall_after_s and poll_interval_s must be positive")
        self.stall_after_s = float(stall_after_s)
        self.poll_interval_s = float(poll_interval_s)
        self.compile_stall_after_s = float(
            compile_stall_after_s if compile_stall_after_s is not None else 4 * stall_after_s
        )
        self.on_stall = on_stall
        self.include_stacks = bool(include_stacks)
        self.log = bool(log)
        self.reports: list[StallReport] = []
        self._last_activity_ns = time.perf_counter_ns()
        self._last_report_ns: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def on_begin(self, span: _instr.Span) -> None:
        """Record activity."""
        self._last_activity_ns = time.perf_counter_ns()

    def on_end(self, span: _instr.Span) -> None:
        """Record activity."""
        self._last_activity_ns = time.perf_counter_ns()

    def start(self) -> DispatchWatchdog:
        """Attach to the dispatcher and start the polling thread."""
        if self._thread is not None:
            return self
        self._stop.clear()
        self._last_activity_ns = time.perf_counter_ns()
        _instr.attach(self)
        self._thread = threading.Thread(target=self._run, name="spectrax-mpmd-watchdog", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop polling and detach."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, 2 * self.poll_interval_s))
            self._thread = None
        _instr.detach(self)

    def _run(self) -> None:
        while not self._stop.wait(self.poll_interval_s):
            try:
                self.check()
            except Exception:
                logger.exception("SpectraX MPMD watchdog check failed.")

    def snapshot(self) -> list[dict[str, object]]:
        """Return the currently open spans, oldest first."""
        now = time.perf_counter_ns()
        spans = sorted(_instr.open_spans(), key=lambda s: s.start_ns)
        return [
            {
                "kind": s.kind,
                "name": s.name,
                "rank": s.rank,
                "age_s": s.age_s(now),
                "thread": s.thread_name,
            }
            for s in spans
        ]

    def format_state(self) -> str:
        """Human-readable rendering of :meth:`snapshot`."""
        snap = self.snapshot()
        if not snap:
            return "no MPMD dispatch work in flight"
        return "\n".join(
            f"{s['kind']:<9} {s['name']!r} rank={s['rank']} age={s['age_s']:.1f}s thread={s['thread']}" for s in snap
        )

    def _stalled(self, span: _instr.Span, now_ns: int, frames: dict[int, object]) -> StalledSpan:
        stack = None
        if self.include_stacks:
            frame = frames.get(span.thread_id)
            if frame is not None:
                stack = "".join(traceback.format_stack(frame))
        meta = {k: v for k, v in span.meta.items() if k not in ("units", "deps")}
        return StalledSpan(
            kind=span.kind,
            name=span.name,
            rank=span.rank,
            age_s=span.age_s(now_ns),
            thread_id=span.thread_id,
            thread_name=span.thread_name,
            stack=stack,
            meta=meta,
        )

    def check(self, now_ns: int | None = None) -> StallReport | None:
        """Evaluate the stall rules once; emit and return a report if stalled."""
        now_ns = now_ns or time.perf_counter_ns()
        if self._last_report_ns is not None and (now_ns - self._last_report_ns) / 1e9 < self.stall_after_s:
            return None
        open_spans = sorted(_instr.open_spans(), key=lambda s: s.start_ns)
        if not open_spans:
            return None
        idle_s = (now_ns - self._last_activity_ns) / 1e9
        aged = [s for s in open_spans if s.kind in _WORK_KINDS and s.age_s(now_ns) >= self.stall_after_s]
        reason = None
        if aged:
            reason, chosen = "span_age", aged
        else:
            compiling = any(s.kind == "compile" for s in open_spans)
            limit = self.compile_stall_after_s if compiling else self.stall_after_s
            if idle_s >= limit:
                reason, chosen = "no_progress", open_spans
        if reason is None:
            return None
        frames = sys._current_frames() if self.include_stacks else {}
        report = StallReport(
            reason=reason,
            idle_s=idle_s,
            spans=tuple(self._stalled(s, now_ns, frames) for s in chosen),
            open_count=len(open_spans),
        )
        self._last_report_ns = now_ns
        self.reports.append(report)
        if self.log:
            logger.warning(report.format())
        if self.on_stall is not None:
            try:
                self.on_stall(report)
            except Exception:
                logger.exception("SpectraX MPMD watchdog on_stall callback failed.")
        return report


@contextlib.contextmanager
def dispatch_watchdog(
    *,
    stall_after_s: float = 300.0,
    poll_interval_s: float = 10.0,
    on_stall: Callable[[StallReport], None] | None = None,
    include_stacks: bool = True,
    compile_stall_after_s: float | None = None,
    log: bool = True,
) -> Iterator[DispatchWatchdog]:
    """Run a :class:`DispatchWatchdog` for the duration of the block."""
    wd = DispatchWatchdog(
        stall_after_s=stall_after_s,
        poll_interval_s=poll_interval_s,
        on_stall=on_stall,
        include_stacks=include_stacks,
        compile_stall_after_s=compile_stall_after_s,
        log=log,
    )
    wd.start()
    try:
        yield wd
    finally:
        wd.stop()
