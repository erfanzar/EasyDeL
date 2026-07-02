#!/usr/bin/env python
"""Generate a self-contained SpectraX pipeline-parallelism report.

The report has two data sources:

* schedule structure, derived directly from SpectraX schedule classes; and
* optional benchmark result files in JSON, JSONL, or CSV form.

It intentionally has no plotting dependency.  The output is one HTML file with
inline CSS and SVG, so it can be opened directly from disk.
"""

from __future__ import annotations

import argparse
import csv
import glob
import html
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from spectrax.runtime.schedules import (
    DualPipeV,
    Eager1F1B,
    FusedTask,
    GPipe,
    InterleavedH1,
    KimiK2,
    Phase,
    Std1F1B,
    ZeroBubbleH1,
)


@dataclass(frozen=True)
class Workload:
    name: str
    batch: int
    seq_len: int
    microbatches: int


@dataclass(frozen=True)
class PipelineConfig:
    name: str
    pp: int
    tp: int


@dataclass(frozen=True)
class ScheduleSpec:
    key: str
    label: str
    virtual_stages: int = 1
    stage_layout: str = "loop"
    split_backward: bool | None = None
    zero_bubble: bool | None = None

    def build(self, microbatches: int, terminal_backward_mode: str):
        if self.key == "gpipe":
            return GPipe(microbatches=microbatches, terminal_backward_mode=terminal_backward_mode)
        if self.key == "std1f1b":
            return Std1F1B(microbatches=microbatches, terminal_backward_mode=terminal_backward_mode)
        if self.key == "eager1f1b":
            return Eager1F1B(microbatches=microbatches, terminal_backward_mode=terminal_backward_mode)
        if self.key == "zerobubble_h1":
            return ZeroBubbleH1(microbatches=microbatches, terminal_backward_mode=terminal_backward_mode)
        if self.key == "interleaved_h1":
            return InterleavedH1(
                microbatches=microbatches,
                virtual_stages=self.virtual_stages,
                stage_layout=self.stage_layout,
                terminal_backward_mode=terminal_backward_mode,
            )
        if self.key == "kimi_k2":
            kwargs: dict[str, Any] = {
                "microbatches": microbatches,
                "virtual_stages": self.virtual_stages,
                "stage_layout": self.stage_layout,
                "terminal_backward_mode": terminal_backward_mode,
            }
            if self.split_backward is not None:
                kwargs["split_backward"] = self.split_backward
            return KimiK2(**kwargs)
        if self.key == "dualpipev":
            kwargs = {
                "microbatches": microbatches,
                "terminal_backward_mode": terminal_backward_mode,
            }
            if self.zero_bubble is not None:
                kwargs["zero_bubble"] = self.zero_bubble
            return DualPipeV(**kwargs)
        raise ValueError(f"unknown schedule key: {self.key}")


DEFAULT_WORKLOADS = (
    Workload("sft_smoke_bs4_s128", batch=4, seq_len=128, microbatches=4),
    Workload("longctx_bs32_s8192", batch=32, seq_len=8192, microbatches=8),
)

DEFAULT_CONFIGS = (
    PipelineConfig("pp2_tp2", pp=2, tp=2),
    PipelineConfig("pp4_tp1", pp=4, tp=1),
)

DEFAULT_SCHEDULES = (
    ScheduleSpec("gpipe", "GPipe"),
    ScheduleSpec("std1f1b", "Std1F1B"),
    ScheduleSpec("eager1f1b", "Eager1F1B"),
    ScheduleSpec("zerobubble_h1", "ZeroBubbleH1"),
    ScheduleSpec("interleaved_h1", "InterleavedH1(v=2)", virtual_stages=2),
    ScheduleSpec("kimi_k2", "KimiK2(v=2)", virtual_stages=2, stage_layout="loop"),
    ScheduleSpec("dualpipev", "DualPipeV"),
)

PHASE_COLORS = {
    "FWD": "#2f9e44",
    "BWD": "#1c7ed6",
    "BWD_I": "#7048e8",
    "BWD_W": "#f08c00",
    "FUSED": "#495057",
    "IDLE": "#f1f3f5",
    "ERROR": "#c92a2a",
}


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _cell_actions(cell: object) -> tuple[object, ...]:
    if cell is None:
        return ()
    if isinstance(cell, FusedTask):
        return cell.split()
    return (cell,)


def _phase_name(action: object) -> str:
    return getattr(getattr(action, "phase", None), "name", "?")


def _short_action(action: object) -> str:
    phase = _phase_name(action)
    mb = getattr(action, "microbatch", "?")
    return {"FWD": "F", "BWD": "B", "BWD_I": "BI", "BWD_W": "BW"}.get(phase, phase) + str(mb)


def _terminal_mode_note(phase: str, terminal_mode: str, scheduled_supported: bool) -> str:
    if phase == "FWD":
        return "loss+vjp" if terminal_mode == "eager" else "loss"
    if phase == "BWD":
        return "marker" if terminal_mode == "eager" else "vjp"
    if phase in {"BWD_I", "BWD_W"}:
        if terminal_mode == "scheduled" and not scheduled_supported:
            return "unsupported"
        return "marker" if terminal_mode == "eager" else phase.lower()
    return ""


def _terminal_split_phases(grid: list[list[object]], schedule: object, n_stages: int) -> set[str]:
    terminal_rank, terminal_virt = schedule.terminal_loc(n_stages)
    phases: set[str] = set()
    for row in grid:
        for rank, cell in enumerate(row):
            for action in _cell_actions(cell):
                if (rank, getattr(action, "virtual_stage", 0)) == (terminal_rank, terminal_virt):
                    phase = getattr(action, "phase", None)
                    if phase in (Phase.BWD_I, Phase.BWD_W):
                        phases.add(phase.name)
    return phases


def _schedule_metrics(spec: ScheduleSpec, workload: Workload, cfg: PipelineConfig, terminal_mode: str) -> dict[str, Any]:
    try:
        schedule = spec.build(workload.microbatches, terminal_mode)
        grid = schedule.build(cfg.pp)
        split_phases = _terminal_split_phases(grid, schedule, cfg.pp)
        scheduled_supported = not split_phases
        phase_counts: Counter[str] = Counter()
        occupied_cells = 0
        action_count = 0
        fused_cells = 0
        for row in grid:
            for cell in row:
                if cell is None:
                    continue
                occupied_cells += 1
                if isinstance(cell, FusedTask):
                    fused_cells += 1
                for action in _cell_actions(cell):
                    phase_counts[_phase_name(action)] += 1
                    action_count += 1
        total_cells = len(grid) * cfg.pp
        peak = schedule.peak_activations(cfg.pp)
        return {
            "ok": True,
            "schedule": schedule,
            "grid": grid,
            "rows": len(grid),
            "total_cells": total_cells,
            "occupied_cells": occupied_cells,
            "idle_cells": max(0, total_cells - occupied_cells),
            "idle_fraction": max(0.0, (total_cells - occupied_cells) / max(1, total_cells)),
            "action_count": action_count,
            "fused_cells": fused_cells,
            "peak_activations": peak,
            "phase_counts": dict(sorted(phase_counts.items())),
            "terminal_split_phases": sorted(split_phases),
            "scheduled_terminal_supported": scheduled_supported,
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "schedule": None,
            "grid": [],
            "rows": 0,
            "total_cells": 0,
            "occupied_cells": 0,
            "idle_cells": 0,
            "idle_fraction": math.nan,
            "action_count": 0,
            "fused_cells": 0,
            "peak_activations": None,
            "phase_counts": {},
            "terminal_split_phases": [],
            "scheduled_terminal_supported": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _timeline_svg(metrics: dict[str, Any], terminal_mode: str) -> str:
    if not metrics["ok"]:
        return f"<div class='error-box'>{_esc(metrics['error'])}</div>"
    schedule = metrics["schedule"]
    grid = metrics["grid"]
    n_stages = len(grid[0]) if grid else 0
    terminal_rank, terminal_virt = schedule.terminal_loc(n_stages)
    scheduled_supported = bool(metrics["scheduled_terminal_supported"])
    cell_w = 48
    cell_h = 30
    label_w = 72
    top_h = 28
    width = label_w + max(1, len(grid)) * cell_w + 20
    height = top_h + max(1, n_stages) * cell_h + 16
    parts = [
        f"<svg class='timeline-svg' viewBox='0 0 {width} {height}' role='img'>",
        "<text x='8' y='18' class='axis-label'>rank</text>",
    ]
    for t in range(len(grid)):
        x = label_w + t * cell_w + cell_w / 2
        if t % 2 == 0 or len(grid) <= 24:
            parts.append(f"<text x='{x:.1f}' y='18' text-anchor='middle' class='tick'>{t}</text>")
    for rank in range(n_stages):
        y = top_h + rank * cell_h
        parts.append(f"<text x='8' y='{y + 20}' class='axis-label'>r{rank}</text>")
        for t, row in enumerate(grid):
            x = label_w + t * cell_w
            cell = row[rank]
            actions = _cell_actions(cell)
            if not actions:
                parts.append(
                    f"<rect x='{x}' y='{y}' width='{cell_w - 2}' height='{cell_h - 2}' rx='3' fill='{PHASE_COLORS['IDLE']}' />"
                )
                continue
            if len(actions) == 1:
                action = actions[0]
                phase = _phase_name(action)
                color = PHASE_COLORS.get(phase, PHASE_COLORS["FUSED"])
                is_terminal = (rank, getattr(action, "virtual_stage", 0)) == (terminal_rank, terminal_virt)
                note = _terminal_mode_note(phase, terminal_mode, scheduled_supported) if is_terminal else ""
                stroke = "#212529" if is_terminal else "#ffffff"
                stroke_w = 2 if is_terminal else 1
                label = _short_action(action)
                if note:
                    label = f"{label} {note}"
                parts.append(
                    f"<rect x='{x}' y='{y}' width='{cell_w - 2}' height='{cell_h - 2}' rx='3' fill='{color}' "
                    f"stroke='{stroke}' stroke-width='{stroke_w}' />"
                )
                parts.append(
                    f"<text x='{x + (cell_w - 2) / 2:.1f}' y='{y + 18}' text-anchor='middle' class='cell-text'>{_esc(label)}</text>"
                )
                continue
            half = (cell_w - 2) / len(actions)
            labels = []
            for idx, action in enumerate(actions):
                phase = _phase_name(action)
                color = PHASE_COLORS.get(phase, PHASE_COLORS["FUSED"])
                is_terminal = (rank, getattr(action, "virtual_stage", 0)) == (terminal_rank, terminal_virt)
                stroke = "#212529" if is_terminal else "#ffffff"
                stroke_w = 2 if is_terminal else 1
                parts.append(
                    f"<rect x='{x + idx * half:.1f}' y='{y}' width='{half:.1f}' height='{cell_h - 2}' "
                    f"fill='{color}' stroke='{stroke}' stroke-width='{stroke_w}' />"
                )
                labels.append(_short_action(action))
            parts.append(
                f"<text x='{x + (cell_w - 2) / 2:.1f}' y='{y + 18}' text-anchor='middle' class='cell-text'>{_esc('+'.join(labels))}</text>"
            )
    parts.append("</svg>")
    return "\n".join(parts)


def _parse_workload(text: str) -> Workload:
    try:
        name, batch, seq_len, microbatches = text.split(":")
        return Workload(name, int(batch), int(seq_len), int(microbatches))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("workloads must be name:batch:seq_len:microbatches") from exc


def _load_json_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("results", "records", "runs"):
            nested = data.get(key)
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, dict)]
        return [data]
    return []


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _load_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _find_result_files(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        files.extend(Path(match) for match in matches)
    return sorted({path.resolve() for path in files if path.is_file()})


def _numeric(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def _list_of_numbers(value: object) -> list[float]:
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            number = _numeric(item)
            if number is not None:
                out.append(number)
        return out
    return []


def _infer_config_name(record: dict[str, Any]) -> str:
    name = record.get("config") or record.get("name") or record.get("run") or record.get("runner")
    if name:
        return str(name)
    dims = record.get("sharding_axis_dims")
    if isinstance(dims, list | tuple) and len(dims) >= 5:
        return f"pp{dims[0]}_tp{dims[4]}"
    return "unknown"


def _normalize_perf_record(record: dict[str, Any], source: Path) -> dict[str, Any]:
    metadata = record.get("metadata") if isinstance(record.get("metadata"), dict) else {}
    step_times = _list_of_numbers(record.get("step_times"))
    step_ms = (
        _numeric(record.get("step_ms")) or _numeric(record.get("mean_step_ms")) or _numeric(record.get("median_step_ms"))
    )
    if step_ms is None and step_times:
        step_ms = statistics.mean(step_times) * 1000.0 if max(step_times) < 10 else statistics.mean(step_times)
    median_step_ms = _numeric(record.get("median_step_ms")) or _numeric(record.get("step_ms"))
    if median_step_ms is None and step_times:
        median_step_ms = (
            statistics.median(step_times) * 1000.0 if max(step_times) < 10 else statistics.median(step_times)
        )
    if median_step_ms is None:
        median_step_ms = step_ms
    compile_ms = _numeric(record.get("compile_ms"))
    if compile_ms is None:
        compile_time = _numeric(record.get("compile_time"))
        compile_ms = compile_time * 1000.0 if compile_time is not None and compile_time < 1000 else compile_time
    return {
        "source": str(source),
        "config": _infer_config_name(record),
        "workload": record.get("workload") or metadata.get("workload") or metadata.get("model_size") or "unknown",
        "schedule": record.get("schedule") or record.get("schedule_name") or metadata.get("scheduler_kind") or "-",
        "terminal_mode": metadata.get("terminal_backward_mode") or record.get("terminal_backward_mode") or "unknown",
        "step_ms": step_ms,
        "median_step_ms": median_step_ms,
        "compile_ms": compile_ms,
        "loss": _numeric(record.get("loss"))
        or (_list_of_numbers(record.get("losses"))[-1] if _list_of_numbers(record.get("losses")) else None),
        "error": record.get("error"),
    }


def _load_perf_records(files: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in files:
        try:
            if path.suffix == ".json":
                raw = _load_json_records(path)
            elif path.suffix == ".jsonl":
                raw = _load_jsonl_records(path)
            elif path.suffix == ".csv":
                raw = _load_csv_records(path)
            else:
                raw = []
        except Exception:
            continue
        records.extend(_normalize_perf_record(record, path) for record in raw)
    return records


def _bar_svg(records: list[dict[str, Any]], metric: str, title: str) -> str:
    values = [(record, _numeric(record.get(metric))) for record in records if _numeric(record.get(metric)) is not None]
    if not values:
        return "<div class='empty'>No measured data for this metric.</div>"
    values.sort(key=lambda item: item[1] or 0.0)
    max_value = max(value for _record, value in values if value is not None)
    row_h = 30
    label_w = 260
    width = 920
    height = 44 + len(values) * row_h
    parts = [
        f"<svg class='bar-svg' viewBox='0 0 {width} {height}' role='img'>",
        f"<text x='8' y='22' class='chart-title'>{_esc(title)}</text>",
    ]
    for i, (record, value) in enumerate(values):
        assert value is not None
        y = 38 + i * row_h
        label = f"{record['config']} / {record['schedule']} / {record['terminal_mode']}"
        bar_w = (width - label_w - 80) * (value / max_value if max_value > 0 else 0.0)
        parts.append(f"<text x='8' y='{y + 18}' class='bar-label'>{_esc(label)}</text>")
        parts.append(f"<rect x='{label_w}' y='{y}' width='{bar_w:.1f}' height='20' rx='3' fill='#1971c2' />")
        parts.append(f"<text x='{label_w + bar_w + 8:.1f}' y='{y + 16}' class='bar-value'>{value:.2f}</text>")
    parts.append("</svg>")
    return "\n".join(parts)


def _structural_table(rows: list[dict[str, Any]]) -> str:
    body = []
    for row in rows:
        phases = ", ".join(f"{key}:{value}" for key, value in row["phase_counts"].items())
        split = ", ".join(row["terminal_split_phases"]) or "-"
        status = "ok" if row["ok"] else "invalid"
        scheduled = "yes" if row["scheduled_terminal_supported"] else "no"
        body.append(
            "<tr>"
            f"<td>{_esc(row['workload'])}</td>"
            f"<td>{_esc(row['config'])}</td>"
            f"<td>{_esc(row['terminal_mode'])}</td>"
            f"<td>{_esc(row['schedule_label'])}</td>"
            f"<td>{status}</td>"
            f"<td>{_esc(row['rows'])}</td>"
            f"<td>{row['idle_fraction'] * 100:.1f}%</td>"
            f"<td>{_esc(row['peak_activations'])}</td>"
            f"<td>{scheduled}</td>"
            f"<td>{_esc(split)}</td>"
            f"<td>{_esc(phases)}</td>"
            f"<td>{_esc(row['error'] or '')}</td>"
            "</tr>"
        )
    return (
        "<table class='data-table'><thead><tr>"
        "<th>workload</th><th>config</th><th>terminal</th><th>schedule</th><th>status</th>"
        "<th>rows</th><th>idle</th><th>peak act</th><th>scheduled terminal ok</th>"
        "<th>terminal split</th><th>phase counts</th><th>error</th>"
        "</tr></thead><tbody>" + "\n".join(body) + "</tbody></table>"
    )


def _perf_table(records: list[dict[str, Any]]) -> str:
    if not records:
        return "<div class='empty'>No JSON/JSONL/CSV benchmark result files were found.</div>"
    body = []
    for record in records:
        step_ms = "" if record["step_ms"] is None else f"{record['step_ms']:.3f}"
        compile_ms = "" if record["compile_ms"] is None else f"{record['compile_ms']:.1f}"
        loss = "" if record["loss"] is None else f"{record['loss']:.5g}"
        body.append(
            "<tr>"
            f"<td>{_esc(record['config'])}</td>"
            f"<td>{_esc(record['workload'])}</td>"
            f"<td>{_esc(record['schedule'])}</td>"
            f"<td>{_esc(record['terminal_mode'])}</td>"
            f"<td>{step_ms}</td>"
            f"<td>{compile_ms}</td>"
            f"<td>{loss}</td>"
            f"<td>{_esc(record['error'] or '')}</td>"
            f"<td>{_esc(Path(record['source']).name)}</td>"
            "</tr>"
        )
    return (
        "<table class='data-table'><thead><tr>"
        "<th>config</th><th>workload</th><th>schedule</th><th>terminal</th>"
        "<th>step ms</th><th>compile ms</th><th>loss</th><th>error</th><th>source</th>"
        "</tr></thead><tbody>" + "\n".join(body) + "</tbody></table>"
    )


def _style() -> str:
    return """
body { margin: 0; font: 14px/1.45 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #1f2933; background: #f7f9fb; }
header { padding: 28px 34px 18px; background: #102a43; color: #f8fafc; }
h1 { margin: 0 0 8px; font-size: 28px; letter-spacing: 0; }
h2 { margin: 0 0 14px; font-size: 19px; }
h3 { margin: 0 0 10px; font-size: 15px; }
main { padding: 22px 34px 44px; }
.subtle { color: #d9e2ec; }
.section { margin: 0 0 24px; padding: 18px; background: #ffffff; border: 1px solid #d9e2ec; border-radius: 8px; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-top: 16px; }
.card { padding: 12px 14px; background: #eef5ff; border: 1px solid #c6d8f5; border-radius: 8px; }
.card .k { color: #52606d; font-size: 12px; }
.card .v { font-weight: 700; font-size: 21px; }
.legend { display: flex; flex-wrap: wrap; gap: 10px 16px; margin-top: 10px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.swatch { width: 16px; height: 12px; border-radius: 2px; display: inline-block; }
.scroll { overflow: auto; border: 1px solid #e4e7eb; border-radius: 6px; background: #fbfdff; }
.timeline-svg, .bar-svg { display: block; min-width: 900px; width: 100%; height: auto; }
.axis-label { fill: #334e68; font-size: 12px; }
.tick { fill: #627d98; font-size: 10px; }
.cell-text { fill: #fff; font-size: 8px; font-weight: 700; pointer-events: none; }
.chart-title { fill: #102a43; font-size: 14px; font-weight: 700; }
.bar-label { fill: #334e68; font-size: 11px; }
.bar-value { fill: #102a43; font-size: 11px; font-weight: 700; }
.data-table { width: 100%; border-collapse: collapse; font-size: 12px; }
.data-table th, .data-table td { border-bottom: 1px solid #e4e7eb; padding: 7px 8px; text-align: left; vertical-align: top; }
.data-table th { position: sticky; top: 0; background: #f1f5f9; z-index: 1; }
details { margin: 10px 0; border: 1px solid #e4e7eb; border-radius: 8px; background: #fff; }
summary { cursor: pointer; padding: 10px 12px; font-weight: 700; }
.details-body { padding: 0 12px 12px; }
.empty, .error-box { padding: 12px; border-radius: 6px; background: #fff4e6; border: 1px solid #ffd8a8; color: #7c2d12; }
.error-box { background: #fff5f5; border-color: #ffc9c9; color: #9b1c1c; }
"""


def build_report(
    *,
    workloads: list[Workload],
    configs: list[PipelineConfig],
    schedules: tuple[ScheduleSpec, ...],
    perf_records: list[dict[str, Any]],
    result_files: list[Path],
) -> str:
    structural_rows: list[dict[str, Any]] = []
    detail_sections: list[str] = []
    for workload in workloads:
        for cfg in configs:
            for spec in schedules:
                for terminal_mode in ("eager", "scheduled"):
                    metrics = _schedule_metrics(spec, workload, cfg, terminal_mode)
                    row = {
                        **metrics,
                        "workload": workload.name,
                        "config": cfg.name,
                        "schedule_key": spec.key,
                        "schedule_label": spec.label,
                        "terminal_mode": terminal_mode,
                    }
                    structural_rows.append(row)
                    title = (
                        f"{workload.name} / {cfg.name} / {spec.label} / terminal={terminal_mode} "
                        f"(mb={workload.microbatches}, batch={workload.batch}, seq={workload.seq_len})"
                    )
                    detail_sections.append(
                        "<details>"
                        f"<summary>{_esc(title)}</summary>"
                        "<div class='details-body'>"
                        f"<div class='scroll'>{_timeline_svg(metrics, terminal_mode)}</div>"
                        "</div></details>"
                    )

    ok_count = sum(1 for row in structural_rows if row["ok"])
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    legend = "".join(
        f"<span><i class='swatch' style='background:{color}'></i>{_esc(name)}</span>"
        for name, color in PHASE_COLORS.items()
        if name != "ERROR"
    )
    source_list = "".join(f"<li>{_esc(path)}</li>" for path in result_files) or "<li>none</li>"
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>SpectraX Pipeline Report</title>
<style>{_style()}</style>
</head>
<body>
<header>
  <h1>SpectraX Pipeline Report</h1>
  <div class="subtle">Generated {generated}. Structure is derived from live SpectraX schedules; performance comes only from files listed below.</div>
  <div class="cards">
    <div class="card"><div class="k">workloads</div><div class="v">{len(workloads)}</div></div>
    <div class="card"><div class="k">PP configs</div><div class="v">{len(configs)}</div></div>
    <div class="card"><div class="k">schedule/mode rows</div><div class="v">{len(structural_rows)}</div></div>
    <div class="card"><div class="k">valid structure rows</div><div class="v">{ok_count}</div></div>
    <div class="card"><div class="k">perf rows</div><div class="v">{len(perf_records)}</div></div>
  </div>
</header>
<main>
  <section class="section">
    <h2>Legend</h2>
    <div class="legend">{legend}</div>
  </section>
  <section class="section">
    <h2>Performance Data</h2>
    <p>Result files loaded:</p>
    <ul>{source_list}</ul>
    {_bar_svg(perf_records, "step_ms", "Step time, lower is better")}
    {_bar_svg(perf_records, "compile_ms", "Compile/init time, lower is better")}
    <div class="scroll">{_perf_table(perf_records)}</div>
  </section>
  <section class="section">
    <h2>Schedule Structure Matrix</h2>
    <div class="scroll">{_structural_table(structural_rows)}</div>
  </section>
  <section class="section">
    <h2>Timelines</h2>
    {"".join(detail_sections)}
  </section>
</main>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tmp-files/spectrax-pipeline-visualization/report.html"),
        help="HTML report path.",
    )
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        help="JSON/JSONL/CSV result file or glob. Can be passed more than once.",
    )
    parser.add_argument(
        "--workload",
        action="append",
        type=_parse_workload,
        default=[],
        help="Additional/override workload as name:batch:seq_len:microbatches. If any are passed, defaults are replaced.",
    )
    args = parser.parse_args(argv)

    result_patterns = args.result or [
        "tmp-files/sft-pp-tp-comparison/**/*.json",
        "tmp-files/spectrax-pipeline-visualization/**/*.json",
        "tmp-files/spectrax-pipeline-visualization/**/*.jsonl",
        "tmp-files/spectrax-pipeline-visualization/**/*.csv",
    ]
    result_files = _find_result_files(result_patterns)
    perf_records = _load_perf_records(result_files)
    workloads = args.workload or list(DEFAULT_WORKLOADS)
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    report = build_report(
        workloads=workloads,
        configs=list(DEFAULT_CONFIGS),
        schedules=DEFAULT_SCHEDULES,
        perf_records=perf_records,
        result_files=result_files,
    )
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output}")
    print(f"  workloads: {len(workloads)}")
    print(f"  result files: {len(result_files)}")
    print(f"  perf rows: {len(perf_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
