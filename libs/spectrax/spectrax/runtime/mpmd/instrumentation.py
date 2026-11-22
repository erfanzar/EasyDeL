# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Opt-in instrumentation seam for the scheduled MPMD dispatcher.

The dispatcher calls :func:`begin` / :func:`end` around schedule dispatches,
schedule units, and cross-stage transfers. With no observer attached each hook
is a single tuple truthiness check and returns ``None``. Observers (the
schedule tracer and the hang watchdog) receive :class:`Span` objects through
``on_begin(span)`` and ``on_end(span)``.

Attaching an observer never changes which dispatcher runs. That is deliberate:
:func:`~spectrax.runtime.mpmd.collect_task_times_ms` forces the serial
dispatcher, which is how a threaded-dispatcher ordering bug once stayed hidden.
"""

from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass, field

from ..._internal.logging import get_logger

__all__ = [
    "Span",
    "active",
    "annotate",
    "attach",
    "begin",
    "begin_unit",
    "block_requested",
    "describe_unit",
    "detach",
    "end",
    "open_spans",
]

logger = get_logger("MPMD-Instrumentation")

_OBSERVERS: tuple[object, ...] = ()
_OBSERVERS_LOCK = threading.Lock()
_SPAN_IDS = itertools.count(1)
_THREAD_STATE = threading.local()
_OPEN: dict[int, Span] = {}
_OPEN_LOCK = threading.Lock()
_FAILED_OBSERVERS: set[int] = set()


@dataclass(eq=False)
class Span:
    """One timed region recorded by the dispatcher.

    Attributes:
        span_id: Process-unique id.
        kind: ``"dispatch"``, ``"compile"``, ``"unit"`` or ``"transfer"``.
        name: Human-readable label (e.g. ``"FWD s0 mb1"``).
        rank: Physical pipeline rank the span belongs to, if any.
        start_ns: ``time.perf_counter_ns()`` at begin.
        thread_id: ``threading.get_ident()`` of the thread that began the span.
        thread_name: Name of that thread.
        meta: Free-form JSON-friendly metadata.
        end_ns: ``time.perf_counter_ns()`` at end; ``None`` while open.
        error: ``"Type: message"`` when the region raised.
        parent_id: Enclosing span on the same thread, if any.
    """

    span_id: int
    kind: str
    name: str
    rank: int | None
    start_ns: int
    thread_id: int
    thread_name: str
    meta: dict[str, object] = field(default_factory=dict)
    end_ns: int | None = None
    error: str | None = None
    parent_id: int | None = None

    @property
    def duration_ns(self) -> int | None:
        """Elapsed nanoseconds, or ``None`` while the span is open."""
        return None if self.end_ns is None else self.end_ns - self.start_ns

    def age_s(self, now_ns: int | None = None) -> float:
        """Seconds since the span began (until its end, if ended)."""
        stop = self.end_ns if self.end_ns is not None else (now_ns or time.perf_counter_ns())
        return (stop - self.start_ns) / 1e9


def _stack() -> list[Span]:
    stack = getattr(_THREAD_STATE, "stack", None)
    if stack is None:
        stack = []
        _THREAD_STATE.stack = stack
    return stack


def attach(observer: object) -> None:
    """Attach ``observer`` (an object with ``on_begin`` / ``on_end``). Idempotent."""
    global _OBSERVERS
    with _OBSERVERS_LOCK:
        if not any(o is observer for o in _OBSERVERS):
            _OBSERVERS = (*_OBSERVERS, observer)


def detach(observer: object) -> None:
    """Detach ``observer``; a no-op if it is not attached."""
    global _OBSERVERS
    with _OBSERVERS_LOCK:
        _OBSERVERS = tuple(o for o in _OBSERVERS if o is not observer)
        if not _OBSERVERS:
            with _OPEN_LOCK:
                _OPEN.clear()


def active() -> bool:
    """Return whether any observer is attached."""
    return bool(_OBSERVERS)


def block_requested() -> bool:
    """Return whether an attached observer wants device-inclusive unit spans."""
    observers = _OBSERVERS
    return bool(observers) and any(getattr(o, "block_until_ready", False) for o in observers)


def open_spans() -> list[Span]:
    """Snapshot of spans that have begun but not ended (only while observed)."""
    with _OPEN_LOCK:
        return list(_OPEN.values())


def _notify(method: str, span: Span, observers: tuple[object, ...]) -> None:
    for observer in observers:
        fn = getattr(observer, method, None)
        if fn is None:
            continue
        try:
            fn(span)
        except Exception:
            key = id(observer)
            if key not in _FAILED_OBSERVERS:
                _FAILED_OBSERVERS.add(key)
                logger.exception("SpectraX MPMD instrumentation observer %r failed in %s.", observer, method)


def begin(kind: str, name: str, *, rank: int | None = None, **meta: object) -> Span | None:
    """Open a span; returns ``None`` (and does nothing) when unobserved."""
    observers = _OBSERVERS
    if not observers:
        return None
    thread = threading.current_thread()
    stack = _stack()
    span = Span(
        span_id=next(_SPAN_IDS),
        kind=kind,
        name=name,
        rank=rank,
        start_ns=time.perf_counter_ns(),
        thread_id=threading.get_ident(),
        thread_name=thread.name,
        meta=dict(meta),
        parent_id=stack[-1].span_id if stack else None,
    )
    stack.append(span)
    with _OPEN_LOCK:
        _OPEN[span.span_id] = span
    _notify("on_begin", span, observers)
    return span


def end(span: Span | None, error: BaseException | str | None = None) -> None:
    """Close ``span`` (``None`` is accepted and ignored)."""
    if span is None or span.end_ns is not None:
        return
    span.end_ns = time.perf_counter_ns()
    if error is not None:
        span.error = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
    stack = _stack()
    for i in range(len(stack) - 1, -1, -1):
        if stack[i] is span:
            del stack[i]
            break
    with _OPEN_LOCK:
        _OPEN.pop(span.span_id, None)
    _notify("on_end", span, _OBSERVERS)


def annotate(kind: str, **meta: object) -> None:
    """Merge ``meta`` into the innermost open span of ``kind`` on this thread."""
    if not _OBSERVERS:
        return
    stack = getattr(_THREAD_STATE, "stack", None)
    if not stack:
        return
    for span in reversed(stack):
        if span.kind == kind:
            span.meta.update(meta)
            return


def describe_unit(unit: object) -> tuple[str, dict[str, object]]:
    """Return ``(label, meta)`` for a ``_ScheduleUnit``."""
    phase = getattr(getattr(unit, "bwd_phase", None), "name", None)
    kind = getattr(unit, "kind", "action")
    fwd_logical, fwd_mb = getattr(unit, "fwd_logical", None), getattr(unit, "fwd_mb", None)
    bwd_logical, bwd_mb = getattr(unit, "bwd_logical", None), getattr(unit, "bwd_mb", None)
    if kind == "apply":
        label = f"APPLY r{unit.rank}"
    elif kind == "fused":
        label = f"FWD s{fwd_logical} mb{fwd_mb} + {phase} s{bwd_logical} mb{bwd_mb}"
    elif fwd_logical is not None:
        label = f"FWD s{fwd_logical} mb{fwd_mb}"
    else:
        label = f"{phase} s{bwd_logical} mb{bwd_mb}"
    meta: dict[str, object] = {
        "unit_index": int(unit.index),
        "row": int(unit.row),
        "unit_kind": kind,
        "virt": int(unit.virt),
        "fwd_logical": fwd_logical,
        "fwd_mb": fwd_mb,
        "bwd_logical": bwd_logical,
        "bwd_mb": bwd_mb,
        "bwd_phase": phase,
    }
    return label, meta


def begin_unit(unit: object, *, dispatch_span: Span | None = None) -> Span | None:
    """Open a ``"unit"`` span for ``unit``; ``None`` when unobserved."""
    if not _OBSERVERS:
        return None
    label, meta = describe_unit(unit)
    return begin(
        "unit",
        label,
        rank=int(unit.rank),
        dispatch_id=dispatch_span.span_id if dispatch_span is not None else None,
        **meta,
    )
