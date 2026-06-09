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

"""Run the registry-driven ejKernel benchmark matrix across every platform."""

from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import jax
from _op_benchmark_registry import SPECS, _build_algorithms, _ignored_platforms

import ejkernel.kernels  # noqa: F401
from ejkernel.benchmarks import Benchmark
from ejkernel.kernels._registry import kernel_registry


def _parse_list(value: str | None) -> list[str]:
    """Parse a comma-or-space separated environment value."""

    if not value:
        return []
    return [item.strip() for item in value.replace(",", " ").split() if item.strip()]


def _parse_int(name: str, default: int) -> int:
    """Read an integer from the environment."""

    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _wanted_ops() -> list[str]:
    """Return benchmark spec names selected for this run."""

    selected = _parse_list(os.getenv("EJKERNEL_BENCH_OPS"))
    skipped = set(_parse_list(os.getenv("EJKERNEL_BENCH_SKIP_OPS")))
    if not selected:
        return [name for name in sorted(SPECS) if name not in skipped]

    selected_specs: list[str] = []
    for item in selected:
        if item in SPECS:
            selected_specs.append(item)
            continue
        matches = [name for name, spec in SPECS.items() if spec.algorithm == item]
        selected_specs.extend(sorted(matches))
    return [name for name in selected_specs if name not in skipped]


def _wanted_platforms() -> set[str] | None:
    """Return selected platform names, or ``None`` for every available platform."""

    selected = _parse_list(os.getenv("EJKERNEL_BENCH_PLATFORMS"))
    if selected:
        return {name.lower() for name in selected}
    return None


def _limit_configs(configs: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Apply the run-level config cap. A non-positive limit means no cap."""

    if limit <= 0:
        return configs
    return configs[:limit]


def _result_row(result) -> dict[str, Any]:
    """Convert a ``BenchmarkResult`` to a JSON-friendly dict."""

    return {
        "algorithm": result.algorithm,
        "config": result.config,
        "mean_ms": result.mean_ms,
        "median_ms": result.median_ms,
        "min_ms": result.min_ms,
        "max_ms": result.max_ms,
        "std_ms": result.std_ms,
        "mean_ms_bwd": result.mean_ms_bwd,
        "median_ms_bwd": result.median_ms_bwd,
        "min_ms_bwd": result.min_ms_bwd,
        "max_ms_bwd": result.max_ms_bwd,
        "std_ms_bwd": result.std_ms_bwd,
    }


def _finite(value: float | None) -> bool:
    """Return True when a timing value is finite."""

    return value is not None and value != float("inf")


def _config_key(config: dict[str, Any]) -> str:
    """Build a stable JSON key for a benchmark config."""

    return json.dumps(config, sort_keys=True, default=str)


def _platform_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count successful rows per benchmarked platform."""

    counts: dict[str, int] = {}
    for row in rows:
        if _finite(row["mean_ms"]):
            counts[row["algorithm"]] = counts.get(row["algorithm"], 0) + 1
    return dict(sorted(counts.items()))


def _xla_ratios(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute per-platform speed ratios against XLA when both rows exist."""

    by_config: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        if not _finite(row["mean_ms"]):
            continue
        by_config.setdefault(_config_key(row["config"]), {})[row["algorithm"]] = row

    ratios: dict[str, list[float]] = {}
    wins: dict[str, int] = {}
    for platforms in by_config.values():
        xla_row = platforms.get("xla")
        if xla_row is None:
            continue
        for platform, row in platforms.items():
            if platform == "xla":
                continue
            ratio = xla_row["mean_ms"] / row["mean_ms"]
            ratios.setdefault(platform, []).append(ratio)
            wins[platform] = wins.get(platform, 0) + int(ratio >= 1.0)

    return {
        platform: {
            "compared": len(values),
            "wins_vs_xla": wins.get(platform, 0),
            "mean_speedup_vs_xla": sum(values) / len(values),
            "min_speedup_vs_xla": min(values),
            "max_speedup_vs_xla": max(values),
        }
        for platform, values in sorted(ratios.items())
    }


def _fastest_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count which platform is fastest per config."""

    by_config: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if _finite(row["mean_ms"]):
            by_config.setdefault(_config_key(row["config"]), []).append(row)

    winners: dict[str, int] = {}
    for group in by_config.values():
        if len(group) < 2:
            continue
        winner = min(group, key=lambda item: item["mean_ms"])["algorithm"]
        winners[winner] = winners.get(winner, 0) + 1
    return dict(sorted(winners.items()))


def _summarize_op(op_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize per-op results for the Markdown report."""

    successful_configs = {_config_key(row["config"]) for row in rows if _finite(row["mean_ms"])}
    failed_counts: dict[str, int] = {}
    for row in rows:
        if not _finite(row["mean_ms"]):
            failed_counts[row["algorithm"]] = failed_counts.get(row["algorithm"], 0) + 1

    return {
        "op": op_name,
        "platform_successes": _platform_counts(rows),
        "fastest_counts": _fastest_counts(rows),
        "xla_ratios": _xla_ratios(rows),
        "configs_with_success": len(successful_configs),
        "failed_counts": dict(sorted(failed_counts.items())),
    }


def _coverage() -> dict[str, Any]:
    """Report benchmark spec coverage against the loaded kernel registry."""

    registered = sorted(kernel_registry.list_algorithms())
    covered = sorted({spec.algorithm for spec in SPECS.values()})
    return {
        "registered_algorithms": registered,
        "benchmarked_algorithms": covered,
        "missing_benchmark_specs": sorted(set(registered) - set(covered)),
        "extra_benchmark_specs": sorted(set(covered) - set(registered)),
    }


def _format_counts(counts: dict[str, int]) -> str:
    """Format compact count dictionaries for Markdown tables."""

    if not counts:
        return "-"
    return ", ".join(f"{name}:{count}" for name, count in counts.items())


def _format_xla_ratios(ratios: dict[str, Any]) -> str:
    """Format XLA speedup summaries for Markdown tables."""

    if not ratios:
        return "-"
    return ", ".join(f"{platform}:{item['mean_speedup_vs_xla']:.3f}x" for platform, item in ratios.items())


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    """Write a compact Markdown summary next to the JSON payload."""

    coverage = payload["coverage"]
    lines = [
        "# ejKernel Benchmark Suite",
        "",
        f"- Timestamp: `{payload['metadata']['timestamp_utc']}`",
        f"- Backend: `{payload['metadata']['jax_backend']}`",
        f"- Devices: `{payload['metadata']['jax_devices']}`",
        f"- Warmup: `{payload['metadata']['warmup']}`",
        f"- Iterations: `{payload['metadata']['iterations']}`",
        f"- Config limit: `{payload['metadata']['config_limit']}`",
        f"- Platform filter: `{payload['metadata']['platform_filter']}`",
        f"- Registered algorithms: `{len(coverage['registered_algorithms'])}`",
        f"- Benchmarked algorithms: `{len(coverage['benchmarked_algorithms'])}`",
        f"- Missing benchmark specs: `{coverage['missing_benchmark_specs']}`",
        "",
        "| op | platform successes | fastest configs | xla speedups | failed rows |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in payload["summary"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["op"],
                    _format_counts(row["platform_successes"]),
                    _format_counts(row["fastest_counts"]),
                    _format_xla_ratios(row["xla_ratios"]),
                    _format_counts(row["failed_counts"]),
                ]
            )
            + " |"
        )
    if payload["failures"]:
        lines.extend(["", "## Runner Failures", ""])
        for op_name, error in sorted(payload["failures"].items()):
            lines.append(f"- `{op_name}`: {error}")
    path.write_text("\n".join(lines) + "\n")


def _payload(
    timestamp: str,
    warmup: int,
    iterations: int,
    config_limit: int,
    wanted_platforms: set[str] | None,
    summaries: list[dict[str, Any]],
    failures: dict[str, str],
    operations: dict[str, Any],
) -> dict[str, Any]:
    """Build the serializable benchmark payload."""

    return {
        "metadata": {
            "timestamp_utc": timestamp,
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "warmup": warmup,
            "iterations": iterations,
            "config_limit": "all" if config_limit <= 0 else config_limit,
            "platform_filter": "all" if wanted_platforms is None else sorted(wanted_platforms),
        },
        "coverage": _coverage(),
        "summary": summaries,
        "failures": failures,
        "operations": operations,
    }


def _write_outputs(json_path: Path, md_path: Path, payload: dict[str, Any]) -> None:
    """Write JSON and Markdown reports."""

    json_path.write_text(json.dumps(payload, indent=2, default=str))
    _write_markdown(md_path, payload)


def main() -> int:
    """Run the selected benchmark suite and save JSON plus Markdown reports."""

    output_dir = Path(os.getenv("EJKERNEL_BENCH_OUTPUT_DIR", "benchmark_results"))
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    config_limit = _parse_int("EJKERNEL_BENCH_CONFIG_LIMIT", 0)
    warmup = max(_parse_int("EJKERNEL_BENCH_WARMUP", 5), 1)
    iterations = max(_parse_int("EJKERNEL_BENCH_ITERS", 30), 1)
    wanted_platforms = _wanted_platforms()
    ignored = _ignored_platforms()
    summaries = []
    operations = {}
    failures = {}
    json_path = output_dir / f"kernel_suite_{timestamp}.json"
    md_path = output_dir / f"kernel_suite_{timestamp}.md"

    for op_name in _wanted_ops():
        spec = SPECS[op_name]
        limited = replace(spec, configs=_limit_configs(spec.configs, config_limit))
        algorithms = _build_algorithms(limited, ignore_platforms=ignored)
        if wanted_platforms is not None:
            algorithms = {name: fn for name, fn in algorithms.items() if name.lower() in wanted_platforms}
        if not algorithms:
            failures[op_name] = "no selected implementations"
            _write_outputs(
                json_path,
                md_path,
                _payload(timestamp, warmup, iterations, config_limit, wanted_platforms, summaries, failures, operations),
            )
            continue
        bench = Benchmark(
            algorithms=algorithms,
            configs=limited.configs,
            input_generator=limited.input_generator,
            warmup=warmup,
            iterations=iterations,
            bench_bwd=limited.bench_bwd,
            static_kwargs=limited.static_kwargs,
            unpack_inputs=True,
        )
        try:
            analysis = bench.run(verbose=False)
        except Exception as exc:
            failures[op_name] = f"{type(exc).__name__}: {exc}"
        else:
            rows = [_result_row(result) for result in bench.results]
            operations[op_name] = {
                "algorithm": limited.algorithm,
                "configs": limited.configs,
                "platforms": sorted(algorithms),
                "results": rows,
                "analysis": analysis,
            }
            summaries.append(_summarize_op(op_name, rows))
        _write_outputs(
            json_path,
            md_path,
            _payload(timestamp, warmup, iterations, config_limit, wanted_platforms, summaries, failures, operations),
        )

    print(f"saved {json_path}")
    print(f"saved {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
