# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Strict boundary validation for additive histogram and episode streams."""

from __future__ import annotations

import math
import re
import typing as tp
from itertools import pairwise

from eyvan.track._format import EPISODE_KINDS, MAX_HIST_BINS, decode_value

_METRIC = re.compile(r"[A-Za-z0-9_./-]{1,128}")


def metric(value: object) -> str:
    """Validate a metric key or named axis."""

    if not isinstance(value, str) or not _METRIC.fullmatch(value):
        raise ValueError("invalid metric or axis name")
    return value


def finite(value: object) -> float:
    """Require a finite real JSON number, excluding booleans."""

    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError("expected a finite number")
    return float(value)


def integer(value: object, *, minimum: int | None = None, maximum: int | None = None) -> int:
    """Require an integer within optional inclusive limits."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("expected an integer")
    if (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise ValueError("integer outside permitted range")
    return value


def text(value: object, maximum: int) -> str | None:
    """Require nullable bounded text, retaining empty strings as valid identities."""

    if value is not None and (not isinstance(value, str) or len(value) > maximum):
        raise ValueError("invalid text field")
    if value is not None:
        tp.cast(str, value).encode("utf-8")
    return tp.cast(str | None, value)


def axes(value: object) -> dict[str, float]:
    """Require a mapping of valid axis names to finite numbers."""

    if not isinstance(value, dict):
        raise ValueError("axes must be an object")
    return {metric(key): finite(number) for key, number in value.items()}


def _encoded(value: object) -> float | str:
    if isinstance(value, str):
        decode_value(value)
        return value
    return finite(value)


def validate_histogram(record: dict[str, object]) -> dict[str, tp.Any]:
    """Validate a complete histogram record, ignoring unknown additive fields."""

    required = {
        "key",
        "step",
        "t",
        "edges",
        "counts",
        "n",
        "under",
        "over",
        "nonfinite",
        "min",
        "max",
        "mean",
        "std",
    }
    if not required.issubset(record):
        raise ValueError("incomplete histogram record")
    edges, counts = record["edges"], record["counts"]
    if not isinstance(edges, list) or not isinstance(counts, list) or not 1 <= len(counts) <= MAX_HIST_BINS:
        raise ValueError("invalid histogram bins")
    if len(edges) != len(counts) + 1:
        raise ValueError("histogram edges and counts do not match")
    edges = [finite(value) for value in edges]
    if any(left >= right for left, right in pairwise(edges)):
        raise ValueError("histogram edges must increase strictly")
    counts = [integer(value, minimum=0) for value in counts]
    n = integer(record["n"], minimum=0)
    if n != sum(counts):
        raise ValueError("histogram n differs from sum of counts")
    summary = {key: finite(record[key]) if record[key] is not None else None for key in ("min", "max", "mean", "std")}
    if summary["std"] is not None and summary["std"] < 0:
        raise ValueError("histogram standard deviation is negative")
    if summary["min"] is not None and summary["max"] is not None and summary["min"] > summary["max"]:
        raise ValueError("histogram minimum exceeds maximum")
    return {
        "key": metric(record["key"]),
        "step": integer(record["step"], minimum=0, maximum=2**63 - 1),
        "t": finite(record["t"]),
        "edges": edges,
        "counts": counts,
        "n": n,
        **{key: integer(record[key], minimum=0) for key in ("under", "over", "nonfinite")},
        **summary,
        "x": axes(record.get("x", {})),
    }


def validate_episode(record: dict[str, object]) -> dict[str, tp.Any]:
    """Validate an episode record, preserving encoded nonfinite return values."""

    required = {
        "t",
        "step",
        "ret",
        "length",
        "success",
        "kind",
        "actor",
        "env",
        "seed",
        "policy_version",
        "learner_version",
        "checkpoint",
        "x",
        "extra",
    }
    if not required.issubset(record):
        raise ValueError("incomplete episode record")
    if record["kind"] not in EPISODE_KINDS:
        raise ValueError("invalid episode kind")
    if record["success"] is not None and not isinstance(record["success"], bool):
        raise ValueError("episode success must be boolean or null")
    extra = record["extra"]
    if not isinstance(extra, dict) or len(extra) > 32:
        raise ValueError("episode extra must contain at most 32 keys")
    # The episode contract specifies signed int|null for step/length/seed;
    # only version fields explicitly require nonnegative integers.
    step = record["step"]
    return {
        "t": finite(record["t"]),
        "step": integer(step, minimum=-(2**63), maximum=2**63 - 1) if step is not None else None,
        "ret": _encoded(record["ret"]),
        "kind": record["kind"],
        "success": record["success"],
        **{key: integer(record[key]) if record[key] is not None else None for key in ("length", "seed")},
        **{
            key: integer(record[key], minimum=0) if record[key] is not None else None
            for key in ("policy_version", "learner_version")
        },
        "actor": text(record["actor"], 64),
        "env": text(record["env"], 128),
        "checkpoint": text(record["checkpoint"], 256),
        "x": axes(record["x"]),
        "extra": {metric(key): _encoded(value) for key, value in extra.items()},
    }


__all__ = ("axes", "finite", "integer", "metric", "text", "validate_episode", "validate_histogram")
