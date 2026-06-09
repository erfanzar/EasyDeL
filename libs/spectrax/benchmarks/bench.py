# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""spectrax vs flax.nnx benchmark harness.

Usage::

    python -m benchmarks.bench --cases all --device cpu --out results/

Produces ``benchmarks/results/run_<timestamp>.json`` plus
``benchmarks/results/latest.{json,md}``. Cases are chosen from
``{graph_seam, transforms, e2e, rng, all}``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _git_sha() -> str:
    """Return the short git SHA of the current HEAD, or ``'unknown'``."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def _time_fn(fn: Callable, n_warmup: int, n_iters: int) -> dict:
    """Run ``fn`` and record per-call wall times in nanoseconds.

    Args:
        fn: Zero-argument callable to benchmark.
        n_warmup: Number of warm-up calls before timing.
        n_iters: Number of timed iterations.

    Returns:
        Dictionary with ``median_ms``, ``p05_ms``, ``p95_ms``,
        ``n_iters``, and ``n_warmup``.
    """
    for _ in range(n_warmup):
        fn()
    times = []
    for _ in range(n_iters):
        t0 = time.perf_counter_ns()
        fn()
        t1 = time.perf_counter_ns()
        times.append(t1 - t0)
    times.sort()
    n = len(times)

    def pct(p):
        """Return the ``p``-th percentile of the sorted times in milliseconds."""
        idx = min(n - 1, max(0, int(p * (n - 1))))
        return times[idx] / 1e6

    return {
        "median_ms": times[n // 2] / 1e6,
        "p05_ms": pct(0.05),
        "p95_ms": pct(0.95),
        "n_iters": n_iters,
        "n_warmup": n_warmup,
    }


def _build_cases(selection: set[str]) -> dict[str, tuple[Callable, Callable]]:
    """Build the benchmark case dictionary from the selected case names.

    Args:
        selection: Set of case keys to include (e.g. ``{"graph_seam", "e2e"}``).

    Returns:
        Mapping from case name to ``(spectrax_fn, nnx_fn)`` pairs.
    """
    cases: dict[str, tuple[Callable, Callable]] = {}
    if "graph_seam" in selection or "all" in selection:
        from .cases import graph_seam

        cases.update(graph_seam.build())
    if "transforms" in selection or "all" in selection:
        from .cases import transforms as _transforms

        cases.update(_transforms.build())
    if "e2e" in selection or "all" in selection:
        from .cases import e2e

        cases.update(e2e.build())
    if "rng" in selection or "all" in selection:
        from .cases import rng

        cases.update(rng.build())
    if "large" in selection:
        from .cases import large

        cases.update(large.build())
    return cases


def _versions() -> dict[str, str]:
    """Return a dictionary of installed library versions."""
    import flax
    import jax

    import spectrax

    return {
        "jax_version": jax.__version__,
        "flax_version": flax.__version__,
        "spectrax_version": getattr(spectrax, "__version__", "unknown"),
    }


def _classify(case: str) -> str:
    """Return a phase label for the JSON rows.

    Args:
        case: Benchmark case name string.

    Returns:
        Phase classification string (e.g. ``"graph_seam"``, ``"e2e"``).
    """
    if case.startswith(("mlp12x1024/", "xfmr_d512/")):
        return "graph_seam"
    if case.startswith(("jit_dispatch/", "grad/", "value_and_grad/", "vmap/", "scan/", "remat/")):
        return "transform_wiring"
    if case.startswith("train_step/"):
        return "e2e"
    if case.startswith("rng/"):
        return "rng"
    if case.startswith("xfmr_1b/"):
        return "large_e2e"
    return "other"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — parse args, run benchmarks, write results."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="all", help="comma-separated: graph_seam,transforms,e2e,rng,all")
    parser.add_argument("--device", default="cpu", choices=["cpu", "gpu", "tpu"])
    parser.add_argument("--out", default="results", help="output dir under benchmarks/")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--tag", default="run", help="file-name tag; 'baseline' writes baseline.json")
    parser.add_argument("--quick", action="store_true", help="fewer iters for quick iteration")
    args = parser.parse_args(argv)

    os.environ["JAX_PLATFORMS"] = args.device
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    warmup = 5 if args.quick else args.warmup
    iters = 50 if args.quick else args.iters

    selection = set(s.strip() for s in args.cases.split(",") if s.strip())
    cases = _build_cases(selection)

    versions = _versions()
    sha = _git_sha()

    rows: list[dict] = []
    summary: list[tuple[str, float, float, float]] = []

    for case, (spx_fn, nnx_fn) in cases.items():
        phase = _classify(case)
        gc.collect()
        spx_timing = _time_fn(spx_fn, warmup, iters)
        gc.collect()
        nnx_timing = _time_fn(nnx_fn, warmup, iters)
        rows.append(
            {
                "case": case,
                "library": "spectrax",
                "phase": phase,
                **spx_timing,
                "device": args.device,
                "spectrax_sha": sha,
                **versions,
            }
        )
        rows.append(
            {
                "case": case,
                "library": "nnx",
                "phase": phase,
                **nnx_timing,
                "device": args.device,
                "spectrax_sha": sha,
                **versions,
            }
        )
        ratio = spx_timing["median_ms"] / max(nnx_timing["median_ms"], 1e-9)
        summary.append((case, spx_timing["median_ms"], nnx_timing["median_ms"], ratio))
        print(
            f"  {case:40s}  spx={spx_timing['median_ms']:9.4f} ms  nnx={nnx_timing['median_ms']:9.4f} ms  ratio={ratio:5.2f}"
        )

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = f"{args.tag}_{ts}.json" if args.tag != "baseline" else "baseline.json"
    out_path = out_dir / name
    payload = {"generated_at": ts, "device": args.device, **versions, "spectrax_sha": sha, "rows": rows}
    out_path.write_text(json.dumps(payload, indent=2))

    latest_json = out_dir / "latest.json"
    latest_json.write_text(json.dumps(payload, indent=2))

    md_lines = [
        "# Benchmark summary",
        "",
        f"- generated: {ts}",
        f"- device: {args.device}",
        f"- spectrax sha: {sha}",
        f"- jax: {versions['jax_version']}  flax: {versions['flax_version']}",
        "",
        "| case | spx (ms) | nnx (ms) | ratio | PASS (<=1.05) |",
        "|------|---------:|---------:|------:|:--------------|",
    ]
    passing = 0
    faster = 0
    for c, sms, nms, r in summary:
        ok = "PASS" if r <= 1.05 else "FAIL"
        if r <= 1.05:
            passing += 1
        if r < 1.0:
            faster += 1
        md_lines.append(f"| {c} | {sms:.4f} | {nms:.4f} | {r:.3f} | {ok} |")
    md_lines.append("")
    md_lines.append(
        f"**Pass:** {passing}/{len(summary)} cases within 5% of nnx — **Strictly faster:** {faster}/{len(summary)}"
    )
    (out_dir / "latest.md").write_text("\n".join(md_lines))

    geomean = statistics.geometric_mean([max(r, 1e-6) for _, _, _, r in summary]) if summary else 1.0
    print(f"\nCASES={len(summary)}  PASS={passing}  FASTER={faster}  GEOMEAN_RATIO={geomean:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
