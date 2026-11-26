# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""SQLite reads and replay resolution for histogram and episode queries."""

from __future__ import annotations

import json
import sqlite3
import typing as tp

from eyvan.track.tracker import aggregates, rich_sql
from eyvan.track.tracker.records import integer, metric, text
from eyvan.track.tracker.replay import later_frontiers

Record = dict[str, tp.Any]

SCHEMA = """
CREATE TABLE IF NOT EXISTS histogram_records (
    segment TEXT NOT NULL REFERENCES segments(segment), line INTEGER NOT NULL,
    metric TEXT NOT NULL, step INTEGER NOT NULL, t REAL NOT NULL, axes TEXT NOT NULL, payload TEXT NOT NULL,
    PRIMARY KEY(segment, line)
);
CREATE INDEX IF NOT EXISTS histogram_metric ON histogram_records(metric, segment, step);
CREATE TABLE IF NOT EXISTS episode_records (
    segment TEXT NOT NULL REFERENCES segments(segment), line INTEGER NOT NULL,
    step INTEGER, t REAL NOT NULL, axes TEXT NOT NULL, payload TEXT NOT NULL,
    PRIMARY KEY(segment, line)
);
"""


def exists(db: sqlite3.Connection, scope: tuple[str, ...]) -> bool:
    """Check run existence, including runs containing only invalid segments."""

    return (
        db.execute("SELECT 1 FROM segments WHERE org = ? AND project = ? AND run = ? LIMIT 1", scope).fetchone()
        is not None
    )


def episode_rows(db: sqlite3.Connection, scope: tuple[str, ...]) -> list[Record]:
    """Resolve replay frontiers across ranks, kinds and the additive rich streams."""

    rows = db.execute(
        """SELECT e.*, s.attempt, s.rank, s.writer, s.seq FROM episode_records e JOIN segments s USING(segment)
           WHERE s.org = ? AND s.project = ? AND s.run = ? ORDER BY s.attempt, s.seq, e.line, s.writer, s.rank""",
        scope,
    ).fetchall()
    later = later_frontiers(db, scope)
    return [
        {
            **json.loads(row["payload"]),
            **{key: row[key] for key in ("attempt", "rank", "writer", "seq", "line")},
            "replayed": row["step"] is not None
            and later[row["attempt"]] is not None
            and row["step"] >= later[row["attempt"]],
        }
        for row in rows
    ]


def validate_episode_query(kind: str | None, actor: str | None = None, env: str | None = None) -> None:
    """Validate shared kind and nullable identity filters."""

    if kind is not None and kind not in ("train", "eval"):
        raise ValueError("invalid episode kind")
    text(actor, 64)
    text(env, 128)


def _filtered(rows: list[Record], kind: str | None, actor: str | None, env: str | None) -> list[Record]:
    return [
        row
        for row in rows
        if (kind is None or row["kind"] == kind)
        and (actor is None or row["actor"] == actor)
        and (env is None or row["env"] == env)
    ]


def histograms(
    db: sqlite3.Connection,
    scope: tuple[str, ...],
    key: str,
    *,
    x: str,
    points: int,
    bins: int,
    rank: int | None,
) -> Record:
    """Select winning snapshots and evenly spaced columns, then rebin them."""

    if rank is None:
        row = db.execute(
            """SELECT MIN(rank) FROM histogram_records JOIN segments USING(segment)
               WHERE org = ? AND project = ? AND run = ? AND metric = ?""",
            (*scope, key),
        ).fetchone()
        rank = row[0] if row[0] is not None else 0
    rows = db.execute(
        """SELECT h.payload, h.step, s.attempt FROM histogram_records h JOIN segments s USING(segment)
           WHERE s.org = ? AND s.project = ? AND s.run = ? AND h.metric = ? AND s.rank = ?
           ORDER BY s.attempt, s.seq, h.line, s.writer""",
        (*scope, key, rank),
    ).fetchall()
    display = {row["step"]: {**json.loads(row["payload"]), "attempt": row["attempt"]} for row in rows}
    columns = [
        {**row, "display_x": value}
        for _, row in sorted(display.items())
        if (value := aggregates.axis(row, x)) is not None
    ]
    total = len(columns)
    if total > points:
        indices = [total - 1] if points == 1 else sorted({round(i * (total - 1) / (points - 1)) for i in range(points)})
        columns = [columns[i] for i in indices]
    edges, columns = aggregates.histogram_grid(columns, bins)
    return {
        "run_id": scope[2],
        "key": key,
        "rank": rank,
        "x": x,
        "edges": edges,
        "columns": columns,
        "total": total,
        "downsampled": total > points,
    }


def episodes(
    db: sqlite3.Connection,
    scope: tuple[str, ...],
    *,
    kind: str | None,
    actor: str | None,
    env: str | None,
    limit: int,
    offset: int,
    include_replayed: bool,
) -> Record:
    """Return filtered episodes newest first, with pagination after replay filtering."""

    return rich_sql.episode_page(
        db, scope, kind=kind, actor=actor, env=env, limit=limit, offset=offset, include_replayed=include_replayed
    )


def episode_curve(
    db: sqlite3.Connection,
    scope: tuple[str, ...],
    *,
    metric: str,
    kind: str,
    x: str,
    points: int,
    env: str | None,
    actor: str | None,
) -> Record:
    """Aggregate a filtered curve after global episode replay resolution."""

    rows = _filtered(episode_rows(db, scope), kind, actor, env)
    excluded = sum(row["replayed"] for row in rows)
    visible = [row for row in rows if not row["replayed"] and aggregates.axis(row, x) is not None]
    field = "ret" if metric == "return" else "length"
    return {
        "run_id": scope[2],
        "metric": metric,
        "kind": kind,
        "x": x,
        "buckets": aggregates.curve_buckets(visible, metric=metric, x=x, points=points),
        "total": len(visible),
        "nonfinite": sum(isinstance(row[field], str) for row in visible),
        "replayed_excluded": excluded,
    }


def episode_breakdown(db: sqlite3.Connection, scope: tuple[str, ...], *, kind: str, last: int) -> Record:
    """Aggregate actor/environment selection alongside global train lag and evals."""

    rows = episode_rows(db, scope)
    return {
        **aggregates.breakdown([row for row in rows if not row["replayed"]], kind=kind, last=last),
        "replayed_excluded": sum(row["replayed"] for row in rows),
    }


def validate_histogram_query(key: str, x: str, points: int, bins: int, rank: int | None) -> None:
    """Bound histogram query cardinalities and validate axes."""

    metric(key)
    metric(x)
    integer(points, minimum=1, maximum=1000)
    integer(bins, minimum=1, maximum=256)
    if rank is not None:
        integer(rank, minimum=0, maximum=9999)


__all__ = (
    "SCHEMA",
    "episode_breakdown",
    "episode_curve",
    "episode_rows",
    "episodes",
    "exists",
    "histograms",
    "validate_episode_query",
    "validate_histogram_query",
)
