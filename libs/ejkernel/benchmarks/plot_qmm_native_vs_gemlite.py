#!/usr/bin/env python3
# Copyright 2026 The EasyDeL/ejKernel Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render QMM native-vs-GemLite benchmark plots from the verified CSV."""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

DEFAULT_PLOT_DIR = Path("benchmark_results/qmm_native_vs_gemlite")
DEFAULT_CSV = DEFAULT_PLOT_DIR / "qmm_verified_results.csv"
BENCHMARK_SCRIPT = Path(__file__).with_name("benchmark_quantized_matmul_native_vs_gemlite.py")

BACKENDS = ("gemlite", "cuda", "tilelang")
NATIVE_BACKENDS = ("cuda", "tilelang")
BACKEND_LABELS = {"gemlite": "GemLite", "cuda": "CUDA", "tilelang": "TileLang"}
BACKEND_COLORS = {"gemlite": "#4B5563", "cuda": "#2563EB", "tilelang": "#D97706"}
BIT_COLORS = {1: "#0F766E", 2: "#2563EB", 4: "#7C3AED", 8: "#DC2626"}

DEFAULT_BENCHMARK_WORKLOADS = (
    {
        "name": "original_target_m4096_k8192_n4096",
        "m": 4096,
        "k": 8192,
        "n": 4096,
        "bits": "2",
        "groups": "128",
    },
    {
        "name": "decode_qwen_gateup_m1_k4096_n24576",
        "m": 1,
        "k": 4096,
        "n": 24576,
        "bits": "1,2,4,8",
        "groups": "32,64,128",
        "inner_iters": 100,
    },
    {
        "name": "prefill_qwen_gateup_m8192_k4096_n24576",
        "m": 8192,
        "k": 4096,
        "n": 24576,
        "bits": "1,2,4,8",
        "groups": "32,64,128",
    },
)


@dataclass(frozen=True)
class Result:
    workload: str
    bits: int
    group_size: int
    backend: str
    mean_ms: float
    speedup_vs_gemlite: float | None


def _read_results(path: Path) -> list[Result]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[Result] = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            speedup = row["speedup_vs_gemlite"]
            rows.append(
                Result(
                    workload=row["workload"],
                    bits=int(row["bits"]),
                    group_size=int(row["group_size"]),
                    backend=row["backend"],
                    mean_ms=float(row["mean_ms"]),
                    speedup_vs_gemlite=None if speedup == "" else float(speedup),
                )
            )
    return rows


def _run_verified_benchmarks(args: argparse.Namespace) -> None:
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    if args.csv.exists():
        args.csv.unlink()

    env = os.environ.copy()
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    command_log = [
        f"# Generated at {datetime.now(UTC).isoformat()}",
        f"# CSV: {args.csv}",
        "# Environment overrides:",
        f"XLA_PYTHON_CLIENT_PREALLOCATE={env.get('XLA_PYTHON_CLIENT_PREALLOCATE', '')}",
        "",
    ]

    for workload in DEFAULT_BENCHMARK_WORKLOADS:
        inner_iters = (
            args.benchmark_inner_iters if args.benchmark_inner_iters > 0 else int(workload.get("inner_iters", 1))
        )
        cmd = [
            sys.executable,
            str(BENCHMARK_SCRIPT),
            "--m",
            str(workload["m"]),
            "--k",
            str(workload["k"]),
            "--n",
            str(workload["n"]),
            "--bits",
            str(workload["bits"]),
            "--group-size",
            str(workload["groups"]),
            "--modes",
            "affine",
            "--axes",
            "col",
            "--dtype",
            args.benchmark_dtype,
            "--backends",
            args.benchmark_backends,
            "--warmup",
            str(args.benchmark_warmup),
            "--iters",
            str(args.benchmark_iters),
            "--inner-iters",
            str(inner_iters),
            "--seed",
            str(args.benchmark_seed),
            "--native-layout",
            args.native_layout,
            "--gemlite-matmul-type",
            "best",
            "--workload-name",
            str(workload["name"]),
            "--csv-output",
            str(args.csv),
            "--csv-append",
        ]
        command_log.append(shlex.join(cmd))
        print(f"Running benchmark workload: {workload['name']}", flush=True)
        subprocess.run(cmd, check=True, env=env)

    command_path = args.csv.with_suffix(".commands.txt")
    command_path.write_text("\n".join(command_log) + "\n", encoding="utf-8")
    print(f"Wrote benchmark command log: {command_path}")


def _ensure_results_csv(args: argparse.Namespace) -> None:
    if args.refresh_benchmark or not args.csv.exists():
        if args.no_run_benchmark:
            raise SystemExit(
                f"Missing benchmark CSV: {args.csv}\n"
                "Run without --no-run-benchmark to generate it, or pass --csv pointing at an existing CSV."
            )
        _run_verified_benchmarks(args)


def _index_results(rows: list[Result]) -> dict[tuple[str, int, int, str], Result]:
    return {(row.workload, row.bits, row.group_size, row.backend): row for row in rows}


def _workload_for(rows: list[Result], *prefixes: str) -> str:
    matches = sorted({row.workload for row in rows if any(row.workload.startswith(prefix) for prefix in prefixes)})
    if not matches:
        joined = ", ".join(repr(prefix) for prefix in prefixes)
        raise ValueError(f"no workload starting with {joined} in CSV")
    return matches[0]


def _bits_and_groups(rows: list[Result], workload: str) -> tuple[list[int], list[int]]:
    bits = sorted({row.bits for row in rows if row.workload == workload})
    groups = sorted({row.group_size for row in rows if row.workload == workload})
    return bits, groups


def _speedup_matrix(
    idx: dict[tuple[str, int, int, str], Result],
    workload: str,
    backend: str,
    bits: list[int],
    groups: list[int],
) -> np.ndarray:
    values = np.full((len(bits), len(groups)), np.nan)
    for row_i, bit in enumerate(bits):
        for col_i, group_size in enumerate(groups):
            result = idx[(workload, bit, group_size, backend)]
            values[row_i, col_i] = result.speedup_vs_gemlite or 1.0
    return values


def _latencies(
    idx: dict[tuple[str, int, int, str], Result],
    workload: str,
    bits: list[int],
    group_size: int,
    backend: str,
) -> list[float]:
    return [idx[(workload, bit, group_size, backend)].mean_ms for bit in bits]


def _setup_style() -> None:
    plt.switch_backend("Agg")
    plt.rcParams.update(
        {
            "axes.edgecolor": "#D1D5DB",
            "axes.grid": True,
            "axes.labelcolor": "#111827",
            "axes.linewidth": 0.8,
            "axes.titlecolor": "#111827",
            "figure.facecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
        }
    )


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
    fig.savefig(paths[0], dpi=240)
    fig.savefig(paths[1])
    plt.close(fig)
    return paths


def _annotate_thresholds(ax: plt.Axes, *, target: float = 2.0) -> None:
    ax.axhline(1.0, color="#6B7280", linewidth=0.9, alpha=0.65)
    ax.axhline(target, color="#DC2626", linestyle=(0, (4, 3)), linewidth=1.1, alpha=0.85)
    ax.text(
        0.99,
        target,
        f"{target:.0f}x target",
        color="#991B1B",
        fontsize=8,
        ha="right",
        va="bottom",
        transform=ax.get_yaxis_transform(),
    )


def _plot_speedup_heatmap(
    rows: list[Result],
    idx: dict[tuple[str, int, int, str], Result],
    workload: str,
    title: str,
    out_dir: Path,
    stem: str,
) -> list[Path]:
    bits, groups = _bits_and_groups(rows, workload)
    matrices = {backend: _speedup_matrix(idx, workload, backend, bits, groups) for backend in NATIVE_BACKENDS}
    vmax = max(2.5, *(float(np.nanmax(matrix)) for matrix in matrices.values()))
    cmap = LinearSegmentedColormap.from_list("qmm_speedup", ["#FEF2F2", "#FDE68A", "#34D399", "#2563EB"])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.7), constrained_layout=True)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.03)
    last_image = None
    for ax, backend in zip(axes, NATIVE_BACKENDS, strict=True):
        matrix = matrices[backend]
        last_image = ax.imshow(matrix, cmap=cmap, vmin=1.0, vmax=vmax, aspect="auto")
        ax.set_title(f"{BACKEND_LABELS[backend]} speedup vs GemLite", fontsize=11, fontweight="bold")
        ax.set_xticks(np.arange(len(groups)), [str(group) for group in groups])
        ax.set_yticks(np.arange(len(bits)), [str(bit) for bit in bits])
        ax.set_xlabel("Group size")
        ax.set_ylabel("Bits")
        ax.set_xticks(np.arange(-0.5, len(groups), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(bits), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.7)
        ax.grid(which="major", visible=False)
        for row_i, _bit in enumerate(bits):
            for col_i, _group_size in enumerate(groups):
                value = matrix[row_i, col_i]
                text_color = "white" if value >= vmax * 0.72 else "#111827"
                ax.text(col_i, row_i, f"{value:.2f}x", ha="center", va="center", color=text_color, fontweight="bold")
    if last_image is not None:
        cbar = fig.colorbar(last_image, ax=axes, shrink=0.86, pad=0.02)
        cbar.set_label("Speedup, higher is better")
    return _save(fig, out_dir, stem)


def _plot_speedup_lines(
    rows: list[Result],
    idx: dict[tuple[str, int, int, str], Result],
    workload: str,
    title: str,
    out_dir: Path,
    stem: str,
) -> list[Path]:
    bits, groups = _bits_and_groups(rows, workload)
    max_value = 0.0
    for backend in NATIVE_BACKENDS:
        matrix = _speedup_matrix(idx, workload, backend, bits, groups)
        max_value = max(max_value, float(np.nanmax(matrix)))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True, constrained_layout=True)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.03)
    for ax, backend in zip(axes, NATIVE_BACKENDS, strict=True):
        for bit in bits:
            ys = [idx[(workload, bit, group, backend)].speedup_vs_gemlite or 1.0 for group in groups]
            ax.plot(
                groups,
                ys,
                color=BIT_COLORS.get(bit, "#111827"),
                marker="o",
                linewidth=2.2,
                markersize=5.5,
                label=f"{bit}-bit",
            )
            for group, value in zip(groups, ys, strict=True):
                ax.text(group, value + 0.06, f"{value:.2f}x", ha="center", fontsize=7.5, color="#374151")
        _annotate_thresholds(ax)
        ax.set_title(BACKEND_LABELS[backend], fontsize=11, fontweight="bold")
        ax.set_xlabel("Group size")
        ax.set_xticks(groups)
        ax.set_ylim(0.8, max(2.25, max_value * 1.18))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Speedup vs GemLite")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(bits), bbox_to_anchor=(0.5, -0.04))
    return _save(fig, out_dir, stem)


def _plot_latency_bars(
    rows: list[Result],
    idx: dict[tuple[str, int, int, str], Result],
    workload: str,
    title: str,
    out_dir: Path,
    stem: str,
) -> list[Path]:
    bits, groups = _bits_and_groups(rows, workload)
    max_latency = max(row.mean_ms for row in rows if row.workload == workload)
    fig, axes = plt.subplots(1, len(groups), figsize=(13.5, 4.9), sharey=True, constrained_layout=True)
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.03)
    x = np.arange(len(bits), dtype=float)
    width = 0.24
    offsets = {"gemlite": -width, "cuda": 0.0, "tilelang": width}
    for ax, group_size in zip(axes, groups, strict=True):
        for backend in BACKENDS:
            vals = _latencies(idx, workload, bits, group_size, backend)
            bars = ax.bar(
                x + offsets[backend],
                vals,
                width=width,
                label=BACKEND_LABELS[backend],
                color=BACKEND_COLORS[backend],
                alpha=0.94,
            )
            for bar, value in zip(bars, vals, strict=True):
                label = f"{value:.3f}" if value < 1 else f"{value:.2f}"
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + max_latency * 0.015,
                    label,
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=7,
                    color="#374151",
                )
        ax.set_title(f"Group size {group_size}", fontsize=11, fontweight="bold")
        ax.set_xticks(x, [f"{bit}-bit" for bit in bits])
        ax.set_xlabel("Quantization")
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max_latency * 1.2)
    axes[0].set_ylabel("Mean latency (ms), lower is better")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    return _save(fig, out_dir, stem)


def _plot_lowbit_summary(
    rows: list[Result],
    idx: dict[tuple[str, int, int, str], Result],
    out_dir: Path,
) -> list[Path]:
    workloads = [
        (_workload_for(rows, "decode_"), "Decode GEMV, M=1"),
        (_workload_for(rows, "prefill_", "prefill8k_"), "Prefill, M=8192"),
    ]
    low_bits = [1, 2]
    max_value = 0.0
    for workload, _ in workloads:
        _, groups = _bits_and_groups(rows, workload)
        for backend in NATIVE_BACKENDS:
            for bit in low_bits:
                vals = [idx[(workload, bit, group, backend)].speedup_vs_gemlite or 1.0 for group in groups]
                max_value = max(max_value, *vals)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharex=True, constrained_layout=True)
    fig.suptitle("1-bit and 2-bit QMM speedup vs GemLite", fontsize=15, fontweight="bold", y=1.03)
    for row_i, (workload, workload_label) in enumerate(workloads):
        _, groups = _bits_and_groups(rows, workload)
        for col_i, bit in enumerate(low_bits):
            ax = axes[row_i, col_i]
            for backend in NATIVE_BACKENDS:
                vals = [idx[(workload, bit, group, backend)].speedup_vs_gemlite or 1.0 for group in groups]
                ax.plot(
                    groups,
                    vals,
                    color=BACKEND_COLORS[backend],
                    marker="o",
                    linewidth=2.4,
                    markersize=5.5,
                    label=BACKEND_LABELS[backend],
                )
                for group, value in zip(groups, vals, strict=True):
                    ax.text(group, value + 0.07, f"{value:.2f}x", ha="center", fontsize=7.5, color="#374151")
            _annotate_thresholds(ax)
            ax.set_title(f"{workload_label} - {bit}-bit", fontsize=11, fontweight="bold")
            ax.set_xticks(groups)
            ax.set_xlabel("Group size")
            ax.set_ylim(0.8, max(2.25, max_value * 1.16))
            ax.grid(axis="x", visible=False)
    axes[0, 0].set_ylabel("Speedup vs GemLite")
    axes[1, 0].set_ylabel("Speedup vs GemLite")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
    return _save(fig, out_dir, "lowbit_1_2_speedup_summary")


def _plot_original_target(
    rows: list[Result],
    out_dir: Path,
) -> list[Path]:
    workload = _workload_for(rows, "original_target_")
    values = {row.backend: row.mean_ms for row in rows if row.workload == workload}
    gemlite = values["gemlite"]
    fig, ax = plt.subplots(figsize=(7.0, 4.7), constrained_layout=True)
    labels = [BACKEND_LABELS[backend] for backend in BACKENDS]
    ys = [values[backend] for backend in BACKENDS]
    bars = ax.bar(labels, ys, color=[BACKEND_COLORS[backend] for backend in BACKENDS], width=0.55)
    ax.axhline(gemlite / 1.5, color="#DC2626", linestyle=(0, (4, 3)), linewidth=1.3, label="GemLite / 1.5")
    ax.axhline(gemlite / 2.0, color="#0F766E", linestyle=(0, (2, 2)), linewidth=1.1, label="GemLite / 2.0")
    for backend, bar, value in zip(BACKENDS, bars, ys, strict=True):
        if backend == "gemlite":
            label = f"{value:.3f} ms"
        else:
            label = f"{value:.3f} ms\n{gemlite / value:.2f}x"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + gemlite * 0.025,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="#111827",
        )
    ax.set_title("Original 2-bit target: 4096x8192 @ 8192x4096, gs=128", fontsize=13, fontweight="bold")
    ax.set_ylabel("Mean latency (ms), lower is better")
    ax.set_ylim(0, gemlite * 1.2)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    return _save(fig, out_dir, "original_target_latency")


def _write_readme(out_dir: Path, csv_path: Path, generated: list[Path]) -> None:
    grouped: dict[str, list[str]] = defaultdict(list)
    for path in sorted(generated):
        grouped[path.suffix.lstrip(".")].append(path.name)
    lines = [
        "# QMM Native vs GemLite Plots",
        "",
        "Data source: CSV emitted by `benchmark_quantized_matmul_native_vs_gemlite.py`.",
        "Native runs used `strict_fuse=True`, `allow_dense_fallback=False`, affine col quantization, and",
        "`native_layout=kmajor`. GemLite was packed through `GemLiteLinearTriton.pack(...)` from the same",
        "raw quantized values. The GemLite baseline in the CSV uses the fastest measured valid GemLite",
        "forward path for each case.",
        "",
        "## Location",
        "",
        f"- Plot directory: `{out_dir}`",
        f"- Data: `{csv_path}`",
        "",
        "## Reproduce",
        "",
        "```bash",
        f"python benchmarks/plot_qmm_native_vs_gemlite.py --refresh-benchmark --csv {csv_path} --out-dir {out_dir}",
        "```",
        "",
        "## Professional Plot Set",
        "",
    ]
    for suffix in ("png", "pdf"):
        if suffix in grouped:
            lines.append(f"### {suffix.upper()}")
            lines.extend(f"- `{name}`" for name in grouped[suffix])
            lines.append("")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Verified benchmark CSV.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PLOT_DIR, help="Directory for rendered plots.")
    parser.add_argument(
        "--refresh-benchmark",
        action="store_true",
        help="Run the QMM vs GemLite benchmark and overwrite --csv before plotting.",
    )
    parser.add_argument(
        "--no-run-benchmark",
        action="store_true",
        help="Do not auto-run benchmarks when --csv is missing.",
    )
    parser.add_argument("--benchmark-warmup", type=int, default=5)
    parser.add_argument("--benchmark-iters", type=int, default=30)
    parser.add_argument(
        "--benchmark-inner-iters",
        type=int,
        default=0,
        help="Override repeated launches per timing sample. Default 0 uses workload-specific values.",
    )
    parser.add_argument("--benchmark-seed", type=int, default=0)
    parser.add_argument("--benchmark-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--benchmark-backends", default="gemlite,cuda,tilelang")
    parser.add_argument("--native-layout", choices=("row", "kmajor"), default="kmajor")
    args = parser.parse_args()

    _ensure_results_csv(args)
    _setup_style()
    rows = _read_results(args.csv)
    idx = _index_results(rows)
    decode = _workload_for(rows, "decode_")
    prefill = _workload_for(rows, "prefill_", "prefill8k_")

    generated: list[Path] = []
    generated += _plot_speedup_heatmap(
        rows,
        idx,
        decode,
        "Decode GEMV speedup, Qwen gate/up shape M=1 K=4096 N=24576",
        args.out_dir,
        "decode_speedup_heatmap",
    )
    generated += _plot_speedup_lines(
        rows,
        idx,
        decode,
        "Decode GEMV speedup vs GemLite",
        args.out_dir,
        "decode_speedup_lines",
    )
    generated += _plot_latency_bars(
        rows,
        idx,
        decode,
        "Decode GEMV latency, Qwen gate/up shape M=1 K=4096 N=24576",
        args.out_dir,
        "decode_latency_bars",
    )
    generated += _plot_speedup_heatmap(
        rows,
        idx,
        prefill,
        "Prefill speedup, Qwen gate/up shape M=8192 K=4096 N=24576",
        args.out_dir,
        "prefill_speedup_heatmap",
    )
    generated += _plot_speedup_lines(
        rows,
        idx,
        prefill,
        "Prefill speedup vs GemLite",
        args.out_dir,
        "prefill_speedup_lines",
    )
    generated += _plot_latency_bars(
        rows,
        idx,
        prefill,
        "Prefill latency, Qwen gate/up shape M=8192 K=4096 N=24576",
        args.out_dir,
        "prefill_latency_bars",
    )
    generated += _plot_lowbit_summary(rows, idx, args.out_dir)
    generated += _plot_original_target(rows, args.out_dir)
    _write_readme(args.out_dir, args.csv, generated)

    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
