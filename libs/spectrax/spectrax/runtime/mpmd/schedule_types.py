# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Schedule unit payloads and runtime statistics for SpectraX MPMD dispatch."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Literal

from ..schedules import Phase


@dataclass(frozen=True)
class _ApplyPayload:
    """Payload for an APPLY unit (per-rank optimizer-apply step).

    APPLY is a *runtime* concept, not a schedule concept -- the schedule
    grid never contains APPLY entries. ``_build_schedule_units_from_plan``
    synthesizes one APPLY unit per ``(rank, virt)`` at the end of the unit
    list when the plan has compiled apply jits (see ``apply_jits`` in the
    plan dict). Each APPLY unit depends on every backward unit on its rank
    (so all that rank's local gradients are settled before the optimizer
    update fires), and is independent of backward work on other ranks --
    that's where the overlap with the rank-0 bwd tail comes from.

    The payload is intentionally small: the per-rank work (which jit to
    call, which leaves to update, which submesh) lives in the plan dict
    and is keyed by ``(rank, virt)`` at dispatch time.

    Attributes:
        rank: Physical pipeline rank whose stage-local leaves this apply
            unit updates.
        virt: Virtual sub-stage on ``rank`` (0 for flat / DualPipeV
            schedules where apply is per-rank, not per-virtual-stage).
    """

    rank: int
    virt: int


@dataclass(frozen=True)
class _ScheduleUnit:
    """One executable unit in a schedule-driven MPMD dispatch.

    A unit is the smallest granularity the schedule dispatcher fires:
    either a single :class:`Action` (``kind == "action"``), a fused
    forward+backward pair (``kind == "fused"``), or an optimizer-apply
    step (``kind == "apply"``) on a specific physical rank. Units are
    produced by walking ``schedule.build(n)`` (for fwd/bwd) and then
    appending one APPLY unit per ``(rank, virt)`` (when the plan has
    compiled apply jits). They form the nodes of the dependency DAG
    used by the async dispatcher.

    Attributes:
        index: Stable global ordering key (insertion order in the unit
            list). Used as the dependency-graph node id.
        row: Source row in the schedule grid; mostly diagnostic.
            APPLY units use a synthetic row at ``len(grid)`` so they
            sort after every fwd/bwd unit in deterministic dispatch.
        kind: Either ``"action"`` (a single :class:`Action`),
            ``"fused"`` (a :class:`FusedTask`), or ``"apply"`` (a
            stage-local optimizer-apply step).
        rank: Physical pipeline rank that owns this unit.
        virt: Virtual sub-stage on ``rank`` (always 0 for flat
            schedules with ``V == 1``).
        payload: The underlying :class:`Action`, :class:`FusedTask`, or
            :class:`_ApplyPayload`.
        fwd_logical: Logical stage index of the forward half (None for
            pure-backward and apply units).
        fwd_mb: Microbatch index of the forward half (None for
            pure-backward and apply units).
        bwd_logical: Logical stage index of the backward half (None
            for pure-forward and apply units).
        bwd_mb: Microbatch index of the backward half (None for
            pure-forward and apply units).
        bwd_phase: Specific backward phase (``Phase.BWD``,
            ``Phase.BWD_I``, or ``Phase.BWD_W``) or ``None`` for
            forward-only and apply units.
    """

    index: int
    row: int
    kind: Literal["action", "fused", "apply"]
    rank: int
    virt: int
    payload: object
    fwd_logical: int | None
    fwd_mb: int | None
    bwd_logical: int | None
    bwd_mb: int | None
    bwd_phase: Phase | None


class _ScheduleStatsCollector:
    """Non-blocking schedule runtime counters.

    Timings are host enqueue durations, not device completion times. They are
    useful for launch/dispatch critical-path analysis without adding per-task
    ``block_until_ready`` calls.
    """

    def __init__(
        self,
        *,
        dispatcher: str,
        unit_count: int | None = None,
        action_count: int | None = None,
        fused_count: int | None = None,
        window_count: int | None = None,
        fallback_reason: str | None = None,
        terminal_logical: int | None = None,
        eager_terminal_bwd: bool = False,
    ) -> None:
        """Initialize an empty stats collector for one ``sxcall`` invocation.

        Args:
            dispatcher: Tag identifying which scheduler path produced
                the units (e.g. ``"async"``, ``"sequential"``,
                ``"gpipe-vmap"``). Reported back in :meth:`as_dict` so
                downstream tooling can attribute timings.
            unit_count: Optional total number of schedule units
                planned. Reported as-is for sanity checking.
            action_count: Optional count of plain (non-fused) actions
                in the schedule.
            fused_count: Optional count of fwd+bwd fused units.
            window_count: Optional count of dependency windows the
                async dispatcher planned.
            fallback_reason: Optional human-readable string explaining
                why a faster path (e.g. GPipe vmap) was *not* taken.
                ``None`` when the fast path was used.
        """
        self.dispatcher = dispatcher
        self.unit_count = unit_count
        self.action_count = action_count
        self.fused_count = fused_count
        self.window_count = window_count
        self.fallback_reason = fallback_reason
        self.terminal_logical = terminal_logical
        self.eager_terminal_bwd = bool(eager_terminal_bwd)
        self.transfer_count = 0
        self.transfer_skipped_count = 0
        self.transfer_cache_hit_count = 0
        self.transfer_bytes = 0
        self.transfer_edges: dict[str, dict[str, int]] = {}
        self.transport_methods: dict[str, int] = {}
        self.boundary_shared_count = 0
        self.boundary_share_saved_count = 0
        self.per_rank_launch_count: dict[int, int] = {}
        self.per_rank_launch_enqueue_ms: dict[int, float] = {}
        self.per_rank_enqueue_ms: dict[int, float] = {}
        self.unit_enqueue_ms: dict[int, float] = {}
        self.gate_wait_ms_by_task: dict[str, float] = {}
        self.gate_wait_kind_ms: dict[str, float] = {}
        self.per_rank_gate_wait_ms: dict[int, float] = {}
        self.lock = threading.Lock()

    def record_launch(self, rank: int, elapsed_ms: float) -> None:
        """Increment the launch (top-level dispatch) counter for one rank.

        Distinct from :meth:`record_unit`: a launch is the wall time
        spent submitting *all* the rank's units to JAX during this
        sxcall, while a unit timing is per-individual-unit.

        Args:
            rank: Rank value consumed by this operation.
            elapsed_ms: Elapsed ms value consumed by this operation.
        """
        with self.lock:
            self.per_rank_launch_count[rank] = self.per_rank_launch_count.get(rank, 0) + 1
            self.per_rank_launch_enqueue_ms[rank] = self.per_rank_launch_enqueue_ms.get(rank, 0.0) + elapsed_ms

    def record_unit(self, unit_index: int, rank: int, elapsed_ms: float) -> None:
        """Record the host enqueue time for one schedule unit.

        ``elapsed_ms`` is the wall time spent in Python+XLA dispatch
                for this unit; it does *not* include device execution. Used
        to find dispatch hot-spots.

        Args:
            unit_index: Unit index value consumed by this operation.
            rank: Rank value consumed by this operation.
            elapsed_ms: Elapsed ms value consumed by this operation.
        """
        with self.lock:
            self.unit_enqueue_ms[unit_index] = elapsed_ms
            self.per_rank_enqueue_ms[rank] = self.per_rank_enqueue_ms.get(rank, 0.0) + elapsed_ms

    def record_gate_wait(self, task_name: str | None, rank: int | None, elapsed_ms: float, kind: str) -> None:
        """Record host time spent waiting for the ordered launch gate."""
        if elapsed_ms <= 0.0:
            return
        task = task_name or "<unnamed>"
        with self.lock:
            self.gate_wait_ms_by_task[task] = self.gate_wait_ms_by_task.get(task, 0.0) + elapsed_ms
            self.gate_wait_kind_ms[kind] = self.gate_wait_kind_ms.get(kind, 0.0) + elapsed_ms
            if rank is not None:
                self.per_rank_gate_wait_ms[rank] = self.per_rank_gate_wait_ms.get(rank, 0.0) + elapsed_ms

    def record_transfer(
        self,
        *,
        nbytes: int,
        skipped: bool,
        cache_hit: bool,
        src_rank: int | None,
        dst_rank: int | None,
    ) -> None:
        """Record one cross-rank transfer (or attempted transfer) plus its byte size.

        Tallies counts and bytes both globally and per
        ``"src->dst"`` edge. ``skipped=True`` means the transfer was
        elided (e.g. source and target sharding already matched);
        ``cache_hit=True`` means a sharding-decision cache lookup
        was reused. Both flags are also tracked separately so the
        downstream dashboard can show "% skipped".

        Args:
            nbytes: Nbytes value consumed by this operation.
            skipped: Skipped value consumed by this operation.
            cache_hit: Cache hit value consumed by this operation.
            src_rank: Src rank value consumed by this operation.
            dst_rank: Dst rank value consumed by this operation.
        """
        edge = f"{src_rank if src_rank is not None else '?'}->{dst_rank if dst_rank is not None else '?'}"
        with self.lock:
            self.transfer_count += 1
            self.transfer_bytes += int(nbytes)
            if skipped:
                self.transfer_skipped_count += 1
            if cache_hit:
                self.transfer_cache_hit_count += 1
            bucket = self.transfer_edges.setdefault(edge, {"count": 0, "bytes": 0, "skipped": 0})
            bucket["count"] += 1
            bucket["bytes"] += int(nbytes)
            if skipped:
                bucket["skipped"] += 1

    def record_transport_method(self, method: str) -> None:
        """Record the concrete transfer implementation selected for one edge."""
        with self.lock:
            self.transport_methods[method] = self.transport_methods.get(method, 0) + 1

    def record_boundary_sharing(self, *, saved: int) -> None:
        """Record how many duplicate forward-boundary transfers were avoided."""
        with self.lock:
            self.boundary_shared_count += 1
            self.boundary_share_saved_count += max(0, int(saved))

    def as_dict(
        self, deps: dict[int, set[int]] | None = None, units: list[_ScheduleUnit] | None = None
    ) -> dict[str, object]:
        """Render the recorded counters as a JSON-friendly dict.

        Optional ``deps``/``units`` compute a critical-path timing
        and per-phase / per-rank-phase breakdowns by walking the
        dependency DAG with memoization: each unit's longest-path
        finish time is the max over its predecessors' finish times
        plus its own enqueue time. Without ``deps``/``units`` the
        result still includes raw counters but no critical path.

        Args:
            deps: Optional mapping ``unit_index -> {predecessor indices}``.
            units: Optional list of all units (in any order); needed
                to look up phase metadata.

        Returns:
            A nested dict with the schedule's totals, per-rank
            timings, per-phase timings, and the top-N highest-cost
            units (capped at 16) sorted by enqueue time.
        """
        per_rank_critical_path_ms: dict[int, float] = {}
        critical_path_ms = 0.0
        per_phase_enqueue_ms: dict[str, float] = {}
        per_rank_phase_enqueue_ms: dict[str, float] = {}
        top_unit_enqueue_ms: list[dict[str, object]] = []
        total_unit_enqueue_ms = 0.0
        total_launch_enqueue_ms = 0.0
        total_gate_wait_ms = sum(float(value) for value in self.gate_wait_ms_by_task.values())
        top_gate_wait_ms = [
            {"task": task, "elapsed_ms": round(float(elapsed), 3)}
            for task, elapsed in sorted(self.gate_wait_ms_by_task.items(), key=lambda item: item[1], reverse=True)[:16]
        ]
        if deps is not None and units is not None:
            {unit.index: unit for unit in units}
            memo: dict[int, float] = {}

            def cp(idx: int) -> float:
                """Memoised critical-path (longest-finish) time for unit ``idx``.

                The critical path of any unit is the maximum critical
                path of its predecessors plus the unit's own enqueue
                time. Walking the DAG with memoisation collapses the
                recursion to ``O(|edges|)`` even when the graph has
                wide fan-in.

                Args:
                    idx: Idx value consumed by this operation.

                Returns:
                    Result described by this helper.
                """
                if idx in memo:
                    return memo[idx]
                dep_best = max((cp(dep) for dep in deps.get(idx, ())), default=0.0)
                total = dep_best + self.unit_enqueue_ms.get(idx, 0.0)
                memo[idx] = total
                return total

            for unit in units:
                value = cp(unit.index)
                critical_path_ms = max(critical_path_ms, value)
                rank = unit.rank
                per_rank_critical_path_ms[rank] = max(per_rank_critical_path_ms.get(rank, 0.0), value)

            def unit_phase(unit: _ScheduleUnit) -> str:
                """Classify a schedule unit by execution phase.

                Returns one of ``"fused"``, ``"fwd"``, the lowercased
                ``Phase.name`` for backward variants
                (e.g. ``"bwd"``, ``"bwd_w"``), or ``"unknown"`` if
                the unit doesn't carry phase metadata.

                Args:
                    unit: Unit value consumed by this operation.

                Returns:
                    Result described by this helper.
                """
                if unit.kind == "fused":
                    return unit.kind
                if (
                    self.eager_terminal_bwd
                    and self.terminal_logical is not None
                    and unit.fwd_logical == self.terminal_logical
                ):
                    return "terminal"
                if unit.fwd_logical is not None:
                    return "fwd"
                if unit.bwd_phase is not None:
                    return unit.bwd_phase.name.lower()
                return "unknown"

            def unit_label(unit: _ScheduleUnit) -> str:
                """Render a short human-readable label for one schedule unit.

                Format examples:

                                * ``r1/v0 fwd L2:mb3`` — forward of microbatch 3 on
                                  logical stage 2 at rank 1, virtual 0.
                                * ``r0/v0 bwd_w L0:mb1`` — weight-grad backward.
                                * ``r2/v0 fused L4:fwd_mb0+bwd_mb2`` — paired fused
                                  task.

                Args:
                    unit: Unit value consumed by this operation.

                Returns:
                    Result described by this helper.
                """
                phase = unit_phase(unit)
                loc = f"r{unit.rank}/v{unit.virt}"
                if unit.kind == "fused":
                    return f"{loc} fused L{unit.fwd_logical}:fwd_mb{unit.fwd_mb}+L{unit.bwd_logical}:bwd_mb{unit.bwd_mb}"
                if (
                    self.eager_terminal_bwd
                    and self.terminal_logical is not None
                    and unit.fwd_logical == self.terminal_logical
                ):
                    return f"{loc} terminal L{unit.fwd_logical}:fwd+bwd_mb{unit.fwd_mb}"
                if unit.fwd_logical is not None:
                    return f"{loc} fwd L{unit.fwd_logical}:mb{unit.fwd_mb}"
                return f"{loc} {phase} L{unit.bwd_logical}:mb{unit.bwd_mb}"

            for unit in units:
                elapsed = float(self.unit_enqueue_ms.get(unit.index, 0.0))
                total_unit_enqueue_ms += elapsed
                phase = unit_phase(unit)
                per_phase_enqueue_ms[phase] = per_phase_enqueue_ms.get(phase, 0.0) + elapsed
                rank_phase = f"r{unit.rank}:{phase}"
                per_rank_phase_enqueue_ms[rank_phase] = per_rank_phase_enqueue_ms.get(rank_phase, 0.0) + elapsed
                top_unit_enqueue_ms.append(
                    {
                        "index": unit.index,
                        "row": unit.row,
                        "rank": unit.rank,
                        "virt": unit.virt,
                        "phase": phase,
                        "fwd_logical": unit.fwd_logical,
                        "fwd_mb": unit.fwd_mb,
                        "bwd_logical": unit.bwd_logical,
                        "bwd_mb": unit.bwd_mb,
                        "elapsed_ms": round(elapsed, 3),
                        "label": unit_label(unit),
                    }
                )
            top_unit_enqueue_ms.sort(key=lambda item: item["elapsed_ms"], reverse=True)
            top_unit_enqueue_ms = top_unit_enqueue_ms[:16]
        total_launch_enqueue_ms = sum(float(value) for value in self.per_rank_launch_enqueue_ms.values())

        return {
            "dispatcher": self.dispatcher,
            "unit_count": self.unit_count,
            "action_count": self.action_count,
            "fused_count": self.fused_count,
            "window_count": self.window_count,
            "fallback_reason": self.fallback_reason,
            "transfer_count": self.transfer_count,
            "transfer_skipped_count": self.transfer_skipped_count,
            "transfer_cache_hit_count": self.transfer_cache_hit_count,
            "transfer_bytes": self.transfer_bytes,
            "transfer_edges": dict(sorted(self.transfer_edges.items())),
            "transport_methods": dict(sorted(self.transport_methods.items())),
            "boundary_shared_count": self.boundary_shared_count,
            "boundary_share_saved_count": self.boundary_share_saved_count,
            "total_launch_enqueue_ms": round(total_launch_enqueue_ms, 3),
            "total_unit_enqueue_ms": round(total_unit_enqueue_ms, 3),
            "total_gate_wait_ms": round(total_gate_wait_ms, 3),
            "per_rank_launch_count": dict(sorted(self.per_rank_launch_count.items())),
            "per_rank_launch_enqueue_ms": {k: round(v, 3) for k, v in sorted(self.per_rank_launch_enqueue_ms.items())},
            "per_rank_enqueue_ms": {k: round(v, 3) for k, v in sorted(self.per_rank_enqueue_ms.items())},
            "per_rank_gate_wait_ms": {k: round(v, 3) for k, v in sorted(self.per_rank_gate_wait_ms.items())},
            "gate_wait_kind_ms": {k: round(v, 3) for k, v in sorted(self.gate_wait_kind_ms.items())},
            "top_gate_wait_ms": top_gate_wait_ms,
            "per_phase_enqueue_ms": {k: round(v, 3) for k, v in sorted(per_phase_enqueue_ms.items())},
            "per_rank_phase_enqueue_ms": {k: round(v, 3) for k, v in sorted(per_rank_phase_enqueue_ms.items())},
            "top_unit_enqueue_ms": top_unit_enqueue_ms,
            "per_rank_critical_path_ms": {k: round(v, 3) for k, v in sorted(per_rank_critical_path_ms.items())},
            "critical_path_ms": round(critical_path_ms, 3),
        }
