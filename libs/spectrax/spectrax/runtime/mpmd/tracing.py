# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Schedule tracer for the scheduled MPMD dispatcher.

Records every dispatch, schedule unit, and cross-stage transfer as a span and
exports them as Chrome Trace Event JSON (opens in https://ui.perfetto.dev or
``chrome://tracing``), with one track per pipeline rank and flow arrows for
every unit dependency edge.

Example::

    with trace_schedule("/tmp/step.json") as tracer:
        loss, grads = sxvalue_and_grad(step_fn, argnums=(0,))(model, x, y)
    print(tracer.format_summary())

``block_until_ready=False`` (default) records host-dispatch spans: cheap, but
a unit span ends when its work is *enqueued*. ``True`` blocks each unit on its
outputs so spans are device-inclusive. The dispatcher mode never changes.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import os
import threading
import time
from collections.abc import Iterator

from ..schedules import Action, FusedTask, Phase, Schedule
from . import instrumentation as _instr

__all__ = ["ScheduleTracer", "format_schedule_grid", "trace_schedule"]

_PHASE_LETTER = {Phase.FWD: "F", Phase.BWD: "B", Phase.BWD_I: "I", Phase.BWD_W: "W"}
_FLOW_IDS = itertools.count(1)


def _jsonable(value: object, depth: int = 0) -> object:
    """Convert span metadata into JSON-safe values."""
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if depth > 4:
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_jsonable(v, depth + 1) for v in value]
    return str(value)


def _merged_busy_ns(intervals: list[tuple[int, int]]) -> int:
    """Total length of the union of ``[start, end)`` intervals."""
    total = 0
    cur_start = cur_end = None
    for start, end in sorted(intervals):
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        total += cur_end - cur_start
    return total


class ScheduleTracer:
    """Instrumentation observer that records ended spans.

    Attach with :func:`trace_schedule` rather than directly.

    Args:
        block_until_ready: Make unit spans device-inclusive.
        max_spans: Stop recording (and count drops) beyond this many spans.
    """

    def __init__(self, *, block_until_ready: bool = False, max_spans: int = 1_000_000) -> None:
        self.block_until_ready = bool(block_until_ready)
        self.max_spans = int(max_spans)
        self._spans: list[_instr.Span] = []
        self._lock = threading.Lock()
        self._dropped = 0
        self._origin_ns = time.perf_counter_ns()

    def on_begin(self, span: _instr.Span) -> None:
        """Ignore begins; spans are recorded once they end."""

    def on_end(self, span: _instr.Span) -> None:
        """Record one ended span."""
        with self._lock:
            if len(self._spans) < self.max_spans:
                self._spans.append(span)
            else:
                self._dropped += 1

    @property
    def spans(self) -> list[_instr.Span]:
        """Snapshot of recorded spans, ordered by start time."""
        with self._lock:
            return sorted(self._spans, key=lambda s: (s.start_ns, s.span_id))

    @property
    def dropped_spans(self) -> int:
        """Number of spans not recorded because ``max_spans`` was reached."""
        return self._dropped

    def clear(self) -> None:
        """Forget every recorded span."""
        with self._lock:
            self._spans.clear()
            self._dropped = 0
            self._origin_ns = time.perf_counter_ns()

    def summary(self) -> list[dict[str, object]]:
        """Return one statistics dict per recorded dispatch.

        Keys: ``name``, ``dispatcher``, ``schedule``, ``microbatches``,
        ``n_ranks``, ``units``, ``unit_errors``, ``wall_ms`` (whole dispatch),
        ``compile_ms`` (warm compile), ``exec_ms`` (first unit start to last
        unit end), ``per_rank_busy_ms``, ``bubble_fraction`` (idle share of
        ``n_ranks * exec_ms``, in ``[0, 1]``), and ``transfers`` (count,
        skipped, bytes, total ms, per-method counts).
        """
        spans = self.spans
        out: list[dict[str, object]] = []
        for dispatch in (s for s in spans if s.kind == "dispatch"):
            units = [s for s in spans if s.kind == "unit" and s.meta.get("dispatch_id") == dispatch.span_id]
            end_ns = dispatch.end_ns or dispatch.start_ns
            transfers = [s for s in spans if s.kind == "transfer" and dispatch.start_ns <= s.start_ns <= end_ns]
            compile_ns = sum(
                s.duration_ns or 0 for s in spans if s.kind == "compile" and s.parent_id == dispatch.span_id
            )
            n_ranks = int(dispatch.meta.get("n_ranks") or len({u.rank for u in units}) or 1)
            busy: dict[int, int] = {}
            for rank in sorted({u.rank for u in units if u.rank is not None}):
                busy[rank] = _merged_busy_ns([(u.start_ns, u.end_ns) for u in units if u.rank == rank])
            exec_ns = (max(u.end_ns for u in units) - min(u.start_ns for u in units)) if units else 0
            bubble = 0.0
            if exec_ns > 0:
                bubble = 1.0 - sum(busy.values()) / float(n_ranks * exec_ns)
                bubble = min(1.0, max(0.0, bubble))
            methods: dict[str, int] = {}
            for t in transfers:
                method = str(t.meta.get("method", "unknown"))
                methods[method] = methods.get(method, 0) + 1
            out.append(
                {
                    "name": dispatch.name,
                    "dispatcher": dispatch.meta.get("dispatcher"),
                    "schedule": dispatch.meta.get("schedule"),
                    "microbatches": dispatch.meta.get("microbatches"),
                    "n_ranks": n_ranks,
                    "units": len(units),
                    "planned_units": len(dispatch.meta.get("units") or ()),
                    "unit_errors": sum(1 for u in units if u.error),
                    "error": dispatch.error,
                    "wall_ms": (dispatch.duration_ns or 0) / 1e6,
                    "compile_ms": compile_ns / 1e6,
                    "exec_ms": exec_ns / 1e6,
                    "per_rank_busy_ms": {rank: ns / 1e6 for rank, ns in busy.items()},
                    "bubble_fraction": bubble,
                    "transfers": {
                        "count": len(transfers),
                        "skipped": sum(1 for t in transfers if t.meta.get("skipped")),
                        "bytes": sum(int(t.meta.get("nbytes", 0) or 0) for t in transfers),
                        "total_ms": sum(t.duration_ns or 0 for t in transfers) / 1e6,
                        "methods": methods,
                    },
                }
            )
        return out

    def format_summary(self) -> str:
        """Human-readable rendering of :meth:`summary`."""
        lines: list[str] = []
        summaries = self.summary()
        if not summaries:
            return "no scheduled MPMD dispatch recorded"
        for i, s in enumerate(summaries):
            tr = s["transfers"]
            lines.append(
                f"[{i}] {s['name']} dispatcher={s['dispatcher']} units={s['units']}/{s['planned_units']}"
                + (f" errors={s['unit_errors']}" if s["unit_errors"] else "")
            )
            lines.append(
                f"    wall={s['wall_ms']:.2f}ms compile={s['compile_ms']:.2f}ms exec={s['exec_ms']:.2f}ms "
                f"bubble={100 * s['bubble_fraction']:.1f}%"
            )
            busy = ", ".join(f"r{r}={ms:.2f}ms" for r, ms in s["per_rank_busy_ms"].items())
            lines.append(f"    busy: {busy or '-'}")
            lines.append(
                f"    transfers: {tr['count']} ({tr['skipped']} skipped) {tr['bytes'] / 2**20:.2f} MiB "
                f"{tr['total_ms']:.2f}ms methods={tr['methods']}"
            )
        if self._dropped:
            lines.append(f"(dropped {self._dropped} spans beyond max_spans={self.max_spans})")
        return "\n".join(lines)

    def to_chrome_trace(self) -> dict[str, object]:
        """Return the recording as a Chrome Trace Event dict (``traceEvents``)."""
        spans = self.spans
        origin = min((s.start_ns for s in spans), default=self._origin_ns)

        def us(ns: int) -> float:
            return (ns - origin) / 1e3

        def track_of(span: _instr.Span) -> tuple[int, str]:
            if span.kind in ("dispatch", "compile"):
                return (0, "dispatch")
            if span.kind == "transfer":
                dst = span.meta.get("dst_rank", span.rank)
                return (2000 + (dst if isinstance(dst, int) else 999), f"transfers -> rank {dst}")
            rank = span.rank if span.rank is not None else 999
            return (1000 + rank * 2, f"rank {rank}")

        lanes: dict[tuple[int, str], list[int]] = {}
        tids: dict[tuple[int, str, int], int] = {}
        events: list[dict[str, object]] = []
        span_tid: dict[int, int] = {}
        for span in spans:
            key = track_of(span)
            lane_ends = lanes.setdefault(key, [])
            end = span.end_ns or span.start_ns
            for lane, lane_end in enumerate(lane_ends):
                if lane_end <= span.start_ns:
                    lane_ends[lane] = end
                    break
            else:
                lane = len(lane_ends)
                lane_ends.append(end)
            tid_key = (key[0], key[1], lane)
            if tid_key not in tids:
                tids[tid_key] = len(tids) + 1
            tid = tids[tid_key]
            span_tid[span.span_id] = tid
            args = {k: v for k, v in span.meta.items() if k not in ("units", "deps")}
            if span.kind == "dispatch":
                args["unit_count"] = len(span.meta.get("units") or ())
                args["dep_edges"] = sum(len(v) for v in (span.meta.get("deps") or {}).values())
            args["thread"] = span.thread_name
            if span.error:
                args["error"] = span.error
            events.append(
                {
                    "name": span.name,
                    "cat": span.kind,
                    "ph": "X",
                    "ts": us(span.start_ns),
                    "dur": max(0.001, ((span.end_ns or span.start_ns) - span.start_ns) / 1e3),
                    "pid": 1,
                    "tid": tid,
                    "args": _jsonable(args),
                }
            )

        flow_count = 0
        for dispatch in (s for s in spans if s.kind == "dispatch"):
            by_unit = {
                s.meta.get("unit_index"): s
                for s in spans
                if s.kind == "unit" and s.meta.get("dispatch_id") == dispatch.span_id
            }
            for succ_idx, preds in (dispatch.meta.get("deps") or {}).items():
                succ = by_unit.get(int(succ_idx))
                if succ is None:
                    continue
                for pred_idx in preds:
                    pred = by_unit.get(int(pred_idx))
                    if pred is None or pred.end_ns is None:
                        continue
                    flow_id = next(_FLOW_IDS)
                    start_ts = max(us(pred.start_ns), us(pred.end_ns) - 0.0005)
                    events.append(
                        {
                            "name": "dep",
                            "cat": "dependency",
                            "ph": "s",
                            "id": flow_id,
                            "ts": start_ts,
                            "pid": 1,
                            "tid": span_tid[pred.span_id],
                        }
                    )
                    events.append(
                        {
                            "name": "dep",
                            "cat": "dependency",
                            "ph": "f",
                            "bp": "e",
                            "id": flow_id,
                            "ts": us(succ.start_ns),
                            "pid": 1,
                            "tid": span_tid[succ.span_id],
                        }
                    )
                    flow_count += 1

        events.append({"name": "process_name", "ph": "M", "pid": 1, "args": {"name": "spectrax MPMD schedule"}})
        for (sort_key, label, lane), tid in tids.items():
            name = label if lane == 0 else f"{label} ({lane + 1})"
            events.append({"name": "thread_name", "ph": "M", "pid": 1, "tid": tid, "args": {"name": name}})
            events.append(
                {
                    "name": "thread_sort_index",
                    "ph": "M",
                    "pid": 1,
                    "tid": tid,
                    "args": {"sort_index": sort_key * 16 + lane},
                }
            )
        return {
            "traceEvents": events,
            "displayTimeUnit": "ms",
            "otherData": {
                "tool": "spectrax.runtime.trace_schedule",
                "block_until_ready": self.block_until_ready,
                "dependency_flows": flow_count,
                "dropped_spans": self._dropped,
            },
        }

    def write(self, path: str | os.PathLike[str]) -> str:
        """Write :meth:`to_chrome_trace` as JSON to ``path`` and return the path."""
        path = os.fspath(path)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_chrome_trace(), f)
        return path


@contextlib.contextmanager
def trace_schedule(
    path: str | os.PathLike[str] | None = None,
    *,
    block_until_ready: bool = False,
    max_spans: int = 1_000_000,
) -> Iterator[ScheduleTracer]:
    """Trace scheduled MPMD dispatches run inside the ``with`` block.

    Args:
        path: If given, the Chrome trace JSON is written here on exit
            (also when the body raises, to help debug the failure).
        block_until_ready: Device-inclusive unit spans (see module docs).
        max_spans: Upper bound on recorded spans.

    Yields:
        The :class:`ScheduleTracer`; call :meth:`~ScheduleTracer.summary` or
        :meth:`~ScheduleTracer.to_chrome_trace` on it.
    """
    tracer = ScheduleTracer(block_until_ready=block_until_ready, max_spans=max_spans)
    _instr.attach(tracer)
    try:
        yield tracer
    finally:
        _instr.detach(tracer)
        if path is not None:
            tracer.write(path)


def _cell_label(cell: object, show_virtual: bool) -> str:
    """Render one schedule-grid cell, e.g. ``F3``, ``I0``, ``F2|B0``, ``.``."""
    if cell is None:
        return "."
    if isinstance(cell, FusedTask):
        return f"{_cell_label(cell.fwd, show_virtual)}|{_cell_label(cell.bwd, show_virtual)}"
    if isinstance(cell, Action):
        label = f"{_PHASE_LETTER.get(cell.phase, '?')}{cell.microbatch}"
        if show_virtual:
            label += f"v{cell.virtual_stage}"
        return label
    return "?"


def format_schedule_grid(schedule: Schedule, n_stages: int, *, legend: bool = True) -> str:
    """Render ``schedule.build(n_stages)`` as a text grid (rows = time steps).

    No devices are needed. Cell codes: ``F``/``B`` forward/backward,
    ``I``/``W`` split backward (input/weight grad), followed by the microbatch
    index, ``vK`` for virtual stage ``K`` on interleaved schedules,
    ``a|b`` for a fused cell, and ``.`` for an idle slot.

    Args:
        schedule: Any :class:`~spectrax.runtime.Schedule`.
        n_stages: Number of physical pipeline stages (ranks).
        legend: Append the bubble ratio and the legend.

    Returns:
        The rendered grid.
    """
    grid = schedule.build(n_stages)
    show_virtual = schedule.virtual_stages_per_rank() > 1
    rows = [[_cell_label(cell, show_virtual) for cell in row] for row in grid]
    width = max([len(c) for row in rows for c in row] + [len(f"r{n_stages - 1}"), 2])
    t_width = max(len(str(len(rows) - 1)), 1) + 1
    header = " " * (t_width + 1) + " ".join(f"r{s}".rjust(width) for s in range(n_stages))
    lines = [f"{type(schedule).__name__}(microbatches={schedule.microbatches}) on {n_stages} stage(s)", header]
    for t, row in enumerate(rows):
        cells = row + ["."] * (n_stages - len(row))
        lines.append(f"t{t}".rjust(t_width) + " " + " ".join(c.rjust(width) for c in cells))
    if legend:
        lines.append(
            f"steps={len(rows)} bubble_ratio={schedule.bubble_ratio(n_stages):.3f} "
            f"peak_activations={schedule.peak_activations(n_stages)}"
        )
        lines.append(
            "legend: F fwd, B bwd, I bwd-input, W bwd-weight, N=microbatch, vK virtual stage, a|b fused, . idle"
        )
    return "\n".join(lines)
