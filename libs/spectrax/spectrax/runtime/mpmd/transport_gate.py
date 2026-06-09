# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ordered host-side transfer gate used by scheduled MPMD dispatch."""

from __future__ import annotations

import contextlib
import contextvars
import logging
import threading
import time
from collections.abc import Callable, Iterator

logger = logging.getLogger(__name__)


class _OrderedScheduleTransportGate:
    """Deterministically order named schedule transfers while units run async."""

    def __init__(self, task_order: tuple[str, ...]) -> None:
        self._task_order = task_order
        self._task_set = set(task_order)
        self._task_positions = {name: idx for idx, name in enumerate(task_order)}
        self._position = 0
        self._condition = threading.Condition()

    def ready_for(self, task_names: tuple[str, ...]) -> bool:
        """Return whether a task's first ordered transfer is ready to enter."""
        with self._condition:
            if self._position >= len(self._task_order):
                return True
            for name in task_names:
                task_position = self._task_positions.get(name)
                if task_position is None or task_position < self._position:
                    continue
                return self._task_order[self._position] == name
            return True

    def next_task(self) -> str | None:
        """Return the next ordered transfer name, if any."""
        with self._condition:
            if self._position >= len(self._task_order):
                return None
            return self._task_order[self._position]

    def snapshot(self) -> dict[str, object]:
        """Return a stable diagnostic snapshot of the ordered gate."""
        with self._condition:
            next_task = self._task_order[self._position] if self._position < len(self._task_order) else None
            return {
                "position": self._position,
                "total": len(self._task_order),
                "next": next_task,
            }

    def position_for(self, task_names: tuple[str, ...]) -> int | None:
        """Return the earliest deterministic position for ``task_names``."""
        positions = [self._task_positions[name] for name in task_names if name in self._task_positions]
        return min(positions, default=None)

    def run(self, task_name: str | None, fn: Callable[[], object]) -> object:
        """Run ``fn`` when ``task_name`` is the next deterministic transfer."""
        if task_name is None or task_name not in self._task_set:
            return fn()
        task_position = self._task_positions.get(task_name)
        if task_position is None:
            return fn()
        started_wait = time.perf_counter()
        last_warning = started_wait
        with self._condition:
            if self._position > task_position:
                return fn()
            while (
                self._position <= task_position
                and self._position < len(self._task_order)
                and self._task_order[self._position] != task_name
            ):
                self._condition.wait(timeout=5.0)
                now = time.perf_counter()
                if now - last_warning >= 30.0:
                    logger.warning(
                        "SpectraX MPMD ordered transport gate waiting; task=%s next=%s position=%s total=%s waited_s=%.1f.",
                        task_name,
                        self._task_order[self._position] if self._position < len(self._task_order) else None,
                        self._position,
                        len(self._task_order),
                        now - started_wait,
                    )
                    last_warning = now
        try:
            return fn()
        finally:
            with self._condition:
                if self._position < len(self._task_order) and self._task_order[self._position] == task_name:
                    self._position += 1
                    self._condition.notify_all()

    def enter(self, task_name: str | None) -> "_OrderedScheduleTransportSlot | None":
        """Wait until ``task_name`` may launch and return an explicit release slot."""
        if task_name is None or task_name not in self._task_set:
            return None
        task_position = self._task_positions.get(task_name)
        if task_position is None:
            return None
        started_wait = time.perf_counter()
        last_warning = started_wait
        with self._condition:
            if self._position > task_position:
                return None
            while (
                self._position <= task_position
                and self._position < len(self._task_order)
                and self._task_order[self._position] != task_name
            ):
                self._condition.wait(timeout=5.0)
                now = time.perf_counter()
                if now - last_warning >= 30.0:
                    logger.warning(
                        "SpectraX MPMD ordered transport gate waiting; task=%s next=%s position=%s total=%s waited_s=%.1f.",
                        task_name,
                        self._task_order[self._position] if self._position < len(self._task_order) else None,
                        self._position,
                        len(self._task_order),
                        now - started_wait,
                    )
                    last_warning = now
        return _OrderedScheduleTransportSlot(self, task_name)

    def _release(self, task_name: str) -> None:
        """Advance past ``task_name`` once its ordered launch has been issued."""
        with self._condition:
            if self._position < len(self._task_order) and self._task_order[self._position] == task_name:
                self._position += 1
                self._condition.notify_all()


class _OrderedScheduleTransportSlot:
    """Idempotent release handle for one ordered transport launch."""

    def __init__(self, gate: _OrderedScheduleTransportGate, task_name: str) -> None:
        self._gate = gate
        self._task_name = task_name
        self._released = False
        self._lock = threading.Lock()

    def release(self) -> None:
        """Release the ordered slot at most once."""
        with self._lock:
            if self._released:
                return
            self._released = True
        self._gate._release(self._task_name)


_ORDERED_SCHEDULE_TRANSPORT_GATE: contextvars.ContextVar[_OrderedScheduleTransportGate | None] = contextvars.ContextVar(
    "spectrax_ordered_schedule_transport_gate", default=None
)
_ORDERED_SCHEDULE_TRANSPORT_SLOT: contextvars.ContextVar[_OrderedScheduleTransportSlot | None] = contextvars.ContextVar(
    "spectrax_ordered_schedule_transport_slot", default=None
)


@contextlib.contextmanager
def _ordered_schedule_transport_scope(
    gate: _OrderedScheduleTransportGate | None,
) -> Iterator[None]:
    """Install a deterministic transfer gate for the current schedule dispatch."""
    token = _ORDERED_SCHEDULE_TRANSPORT_GATE.set(gate)
    try:
        yield
    finally:
        _ORDERED_SCHEDULE_TRANSPORT_GATE.reset(token)
