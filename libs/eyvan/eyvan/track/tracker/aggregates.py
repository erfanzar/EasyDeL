# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Deterministic, database-independent histogram and episode aggregation math."""

from __future__ import annotations

import math
import re
import statistics
import typing as tp
from bisect import bisect_left, bisect_right

Record = dict[str, tp.Any]


def mean(values: tp.Sequence[float | int]) -> float | None:
    """Compute a finite mean without overflowing the intermediate sum."""

    if not values:
        return None
    scale = max(abs(value) for value in values)
    if scale == 0:
        return 0.0
    # Normalize magnitudes before summing: even summing max_float / 3 three
    # times can overflow because each division rounds upward.
    normalized = math.fsum(value / scale for value in values) / len(values)
    try:
        # Scaling can underflow a tiny residual after large values cancel.
        result = float(statistics.mean(values)) if normalized == 0 else scale * normalized
    except OverflowError:
        try:
            result = float(statistics.mean(values))
        except OverflowError:
            return None
    return result if math.isfinite(result) else None


def percentile(values: tp.Sequence[float | int], q: float) -> float | None:
    """Return the type-seven (linear interpolated) percentile of finite values."""

    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    fraction = position - lower
    upper = min(lower + 1, len(ordered) - 1)
    left, right = ordered[lower], ordered[upper]
    scale = max(abs(left), abs(right))
    if scale == 0:
        return 0.0
    normalized = math.fsum((left / scale * (1 - fraction), right / scale * fraction))
    try:
        result = scale * normalized
    except OverflowError:
        return None
    return result if math.isfinite(result) else None


def _finite_values(rows: tp.Sequence[Record], field: str) -> list[float | int]:
    return [
        row[field]
        for row in rows
        if isinstance(row[field], int) or (isinstance(row[field], float) and math.isfinite(row[field]))
    ]


def _success(rows: tp.Sequence[Record]) -> float | None:
    return mean([int(row["success"]) for row in rows if row["success"] is not None])


def _lag(row: Record) -> int | None:
    if row["policy_version"] is None or row["learner_version"] is None:
        return None
    return row["learner_version"] - row["policy_version"]


def natural_key(value: str | None) -> tuple[tp.Any, ...]:
    """Sort digit runs numerically, with null identities after all strings."""

    if value is None:
        return (1, (), "")
    parts = tuple((1, int(part)) if part.isdecimal() else (0, part) for part in re.split(r"(\d+)", value))
    return (0, parts, value)


def axis(row: Record, x: str) -> float | int | None:
    """Resolve the standard or named axis without substituting missing values."""

    return row["step"] if x == "step" else row["t"] if x == "time" else row["x"].get(x)


def histogram_grid(columns: list[Record], bins: int) -> tuple[list[float], list[Record]]:
    """Rebin selected snapshots onto a common grid by source-bin overlap."""

    if not columns:
        return [], []
    lo = min(row["edges"][0] for row in columns)
    hi = max(row["edges"][-1] for row in columns)
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    # Weighted endpoints do not overflow for finite edges spanning +/- float max.
    grid = [math.fsum((lo * (1 - i / bins), hi * (i / bins))) for i in range(bins + 1)]
    result = []
    for row in columns:
        counts = [0.0] * bins
        for left, right, count in zip(row["edges"], row["edges"][1:], row["counts"], strict=False):
            if count == 0:
                continue
            scale = max(abs(left), abs(right), 1.0)
            width = right / scale - left / scale
            # Visit only overlapping display bins, rather than multiplying
            # every source bin by every display bin on large heatmaps.
            start = max(0, bisect_right(grid, left) - 1)
            stop = min(bins, bisect_left(grid, right))
            for i in range(start, stop):
                overlap = min(right, grid[i + 1]) / scale - max(left, grid[i]) / scale
                if overlap > 0:
                    counts[i] += count * (overlap / width)
        result.append(
            {
                **{
                    key: row[key]
                    for key in ("step", "attempt", "n", "under", "over", "nonfinite", "min", "max", "mean", "std")
                },
                "x": row["display_x"],
                "counts": counts,
            }
        )
    return grid, result


def curve_buckets(rows: list[Record], *, metric: str, x: str, points: int) -> list[Record]:
    """Aggregate contiguous sorted episodes, retaining nonfinite episode counts."""

    ordered = sorted(
        [(value, row) for row in rows if (value := axis(row, x)) is not None],
        key=lambda pair: (
            pair[0],
            pair[1]["t"],
            pair[1]["attempt"],
            pair[1]["seq"],
            pair[1]["line"],
            pair[1]["writer"],
            pair[1]["rank"],
        ),
    )
    n = len(ordered)
    buckets = []
    b = min(points, n)
    for i in range(b):
        selected = ordered[i * n // b : (i + 1) * n // b]
        records = [row for _, row in selected]
        values = _finite_values(records, "ret" if metric == "return" else "length")
        xs = [value for value, _ in selected]
        buckets.append(
            {
                "x": mean(xs),
                "x_min": min(xs),
                "x_max": max(xs),
                "count": len(records),
                "mean": mean(values),
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                **{f"p{q}": percentile(values, q / 100) for q in (5, 25, 50, 75, 95)},
                "success_rate": _success(records),
            }
        )
    return buckets


def episode_order(row: Record) -> tuple[tp.Any, ...]:
    """Return chronological order with deterministic segment tie-breakers."""

    return (row["t"], row["attempt"], row["seq"], row["line"], row["writer"], row["rank"])


def _population_std(values: tp.Sequence[float | int]) -> float | None:
    if not values:
        return None
    scale = max(abs(value) for value in values)
    if scale == 0:
        return 0.0
    normalized = [value / scale for value in values]
    average = tp.cast(float, mean(normalized))
    try:
        result = scale * math.sqrt(tp.cast(float, mean([(value - average) ** 2 for value in normalized])))
    except OverflowError:
        return None
    return result if math.isfinite(result) else None


def _nullable(value: tp.Any) -> tuple[bool, tp.Any]:
    return (value is not None, value)


def eval_groups(rows: list[Record]) -> list[Record]:
    """Group evals by checkpoint, then version, then step; select one best group."""

    groups: dict[tuple[str, tp.Any], list[Record]] = {}
    for row in rows:
        if row["kind"] != "eval":
            continue
        field = (
            "checkpoint"
            if row["checkpoint"] is not None
            else "policy_version"
            if row["policy_version"] is not None
            else "step"
        )
        groups.setdefault((field, row[field]), []).append(row)
    result = []
    for episodes in groups.values():
        # A group may span versions/steps. Its earliest episode supplies those
        # representative labels rather than accidentally splitting the group.
        first = min(episodes, key=episode_order)
        values = _finite_values(episodes, "ret")
        result.append(
            {
                **{field: first[field] for field in ("checkpoint", "policy_version", "step")},
                "count": len(episodes),
                "mean_return": mean(values),
                "std_return": _population_std(values),
                "min_return": min(values) if values else None,
                "max_return": max(values) if values else None,
                "success_rate": _success(episodes),
                "mean_length": mean(_finite_values(episodes, "length")),
                "first_t": first["t"],
                "best": False,
            }
        )
    result.sort(key=lambda row: tuple(_nullable(row[field]) for field in ("step", "policy_version", "checkpoint")))
    if result:
        # Null means cannot beat finite observations. If every eval is
        # nonfinite, still honour "exactly one" by choosing the earliest group.
        best = min(result, key=lambda row: (row["mean_return"] is None, -(row["mean_return"] or 0), row["first_t"]))
        best["best"] = True
    return result


def breakdown(rows: list[Record], *, kind: str, last: int) -> Record:
    """Compute actor windows, environment aggregates, policy lag and eval groups."""

    actors: dict[str | None, list[Record]] = {}
    envs: dict[str | None, list[Record]] = {}
    for row in rows:
        if row["kind"] == kind:
            actors.setdefault(row["actor"], []).append(row)
            envs.setdefault(row["env"], []).append(row)
    actor_result = []
    for actor in sorted(actors, key=natural_key):
        all_rows = actors[actor]
        window = sorted(all_rows, key=episode_order)[-last:]
        values = _finite_values(window, "ret")
        lags = [lag for row in window if (lag := _lag(row)) is not None]
        latest = window[-1]
        actor_result.append(
            {
                "actor": actor,
                "count": len(all_rows),
                "window": len(window),
                "mean_return": mean(values),
                "p50_return": percentile(values, 0.5),
                "last_return": latest["ret"] if isinstance(latest["ret"], int | float) else None,
                "mean_length": mean(_finite_values(window, "length")),
                "success_rate": _success(window),
                "mean_lag": mean(lags),
                "max_lag": max(lags) if lags else None,
                "last_t": latest["t"],
            }
        )
    env_result = [
        {
            "env": env,
            "count": len(envs[env]),
            "mean_return": mean(_finite_values(envs[env], "ret")),
            "success_rate": _success(envs[env]),
        }
        for env in sorted(envs, key=natural_key)
    ]
    lags = [lag for row in rows if row["kind"] == "train" and (lag := _lag(row)) is not None]
    lag_result = {
        "counts": [lags.count(i) for i in range(16)],
        "over": sum(lag >= 16 for lag in lags),
        "negative": sum(lag < 0 for lag in lags),
        "count": len(lags),
        "p50": percentile(lags, 0.5),
        "p95": percentile(lags, 0.95),
        "max": max(lags) if lags else None,
    }
    return {"actors": actor_result, "envs": env_result, "lag": lag_result, "evals": eval_groups(rows)}


__all__ = ("axis", "breakdown", "curve_buckets", "episode_order", "histogram_grid", "mean", "natural_key", "percentile")
