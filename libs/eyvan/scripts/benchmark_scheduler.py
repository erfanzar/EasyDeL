# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Record synthetic scheduler timings and enforce explicit p99 budgets.

Measures snapshot construction, ordinary scheduling, and elastic packing.
This is a local CPU simulation: Spot is a modeled capacity class, not acquired
hardware. Receipts retain every sample, input sizes and interpreter/platform
identity; a small-sample p99 is not a claim about production tail latency.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter, process_time

from eyvan.domain import (
    CapacityCell,
    CapacityClass,
    PlacementMode,
    ResourceGang,
    ResourceVector,
    Scheduler,
    WorkerResourceScope,
    WorkloadIdentity,
    WorkloadSpec,
)


def snapshot(cell_count: int):
    """Deterministic mixed-width capacity without provider-specific inputs."""

    return tuple(
        CapacityCell(
            f"cell-{index:06d}",
            ResourceVector(cpu=1 + index % 3),
            CapacityClass.SPOT,
            worker_count=1 + index % 3,
            exclusive_workloads=True,
            unit_prices=ResourceVector(cpu=1),
        )
        for index in range(cell_count)
    )


def percentile(samples: list[float], fraction: float) -> float:
    return sorted(samples)[max(0, math.ceil(len(samples) * fraction) - 1)]


def _diagnostic_sample() -> tuple[float, tuple[int, ...]]:
    """Observe CPU and collection counters without forcing or disabling GC."""

    return process_time(), tuple(generation["collections"] for generation in gc.get_stats())


def benchmark(cell_count: int, iterations: int, *, diagnostics: bool = False) -> dict:
    identity = WorkloadIdentity("benchmark", "project", "runner")
    gang = (ResourceGang("workers", 1, ResourceVector(cpu=1)),)
    ordinary = tuple(WorkloadSpec(f"workload-{index}", identity, gang) for index in range(4))
    workers = min(32, cell_count)
    elastic = WorkloadSpec(
        "elastic",
        identity,
        gang,
        worker_count=workers,
        worker_resource_scope=WorkerResourceScope.PER_WORKER,
        placement=PlacementMode.ELASTIC,
    )
    samples: dict[str, list[float]] = {phase: [] for phase in ("snapshot", "schedule", "elastic_pack")}
    diagnostic_phases = {phase: {"cpu_samples_ms": [], "gc_collections": []} for phase in samples} if diagnostics else {}
    scheduler = Scheduler()
    # One unrecorded warm-up pass, then independent recorded decisions against
    # the same deterministic inputs. No provider, agent or database traffic.
    for iteration in range(iterations + 1):
        boundaries = [_diagnostic_sample()] if diagnostics else []
        started = perf_counter()
        cells = snapshot(cell_count)
        built = perf_counter()
        if diagnostics:
            boundaries.append(_diagnostic_sample())
        ordinary_result = scheduler.schedule(ordinary, cells)
        scheduled = perf_counter()
        if diagnostics:
            boundaries.append(_diagnostic_sample())
        elastic_result = scheduler.schedule((elastic,), cells)
        packed = perf_counter()
        if diagnostics:
            boundaries.append(_diagnostic_sample())
        if ordinary_result.infeasibilities or len(ordinary_result.placements) != len(ordinary):
            raise RuntimeError("ordinary benchmark scenario did not place every workload")
        if elastic_result.infeasibilities or len(elastic_result.placements) != 1:
            raise RuntimeError("elastic benchmark scenario did not place its workload")
        reserved = sum(int(lease.resources["cpu"]) for lease in elastic_result.placements[0].all_leases)
        if reserved != workers:
            raise RuntimeError("elastic benchmark scenario did not preserve its exact worker total")
        if iteration:
            samples["snapshot"].append((built - started) * 1000)
            samples["schedule"].append((scheduled - built) * 1000)
            samples["elastic_pack"].append((packed - scheduled) * 1000)
            if diagnostics:
                for index, phase in enumerate(samples):
                    before, after = boundaries[index], boundaries[index + 1]
                    diagnostic_phases[phase]["cpu_samples_ms"].append((after[0] - before[0]) * 1000)
                    diagnostic_phases[phase]["gc_collections"].append(
                        [end - start for start, end in zip(before[1], after[1], strict=True)]
                    )
    report = {
        "schema": "eyvan.scheduler-benchmark.v1",
        "observed_at": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "cells": cell_count,
        "iterations": iterations,
        "ordinary_workloads": len(ordinary),
        "elastic_workers": workers,
        "phases": {
            name: {
                "samples_ms": values,
                "p50_ms": percentile(values, 0.5),
                "p99_ms": percentile(values, 0.99),
            }
            for name, values in samples.items()
        },
        "violations": [],
    }
    if diagnostics:
        report["diagnostics"] = {
            "cpu_clock": "process_time",
            "note": "CPU and GC counter sampling adds overhead; wall-time budgets remain unchanged.",
            "phases": diagnostic_phases,
        }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=int, default=1000)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--budget-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--diagnostics", action="store_true", help="also observe process CPU time and GC collection counts"
    )
    args = parser.parse_args()
    if not 12 <= args.cells <= 100000 or not 1 <= args.iterations <= 1000:
        parser.error("cells must be 12..100000 and iterations must be 1..1000")
    budgets = None
    if args.budget_file:
        try:
            budgets = json.loads(args.budget_file.read_text())[str(args.cells)]
            if set(budgets) != {"snapshot", "schedule", "elastic_pack"}:
                raise ValueError("budget must cover every measured phase")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
                for value in budgets.values()
            ):
                raise ValueError("budgets must be finite, non-negative milliseconds")
        except (OSError, ValueError, KeyError, TypeError) as error:
            parser.error(f"invalid benchmark budget: {error}")
    report = benchmark(args.cells, args.iterations, diagnostics=args.diagnostics)
    if budgets is not None:
        report["violations"] = [
            f"{phase}: p99 {report['phases'][phase]['p99_ms']:.3f} ms exceeds {limit} ms"
            for phase, limit in budgets.items()
            if report["phases"][phase]["p99_ms"] > limit
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"receipt": str(args.output), "violations": report["violations"]}))
    return 1 if report["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
