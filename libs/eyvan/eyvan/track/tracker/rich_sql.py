# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Scoped rich-record selection without materializing off-page heavy payloads."""

from __future__ import annotations

import json
import sqlite3
import typing as tp

Record = dict[str, tp.Any]
SCHEMA = """
CREATE TABLE IF NOT EXISTS content_summaries (
    segment TEXT NOT NULL, line INTEGER NOT NULL, payload TEXT NOT NULL,
    PRIMARY KEY(segment, line),
    FOREIGN KEY(segment, line) REFERENCES content_records(segment, line) ON DELETE CASCADE
);
CREATE TRIGGER IF NOT EXISTS invalidate_content_summary AFTER UPDATE ON content_records
BEGIN DELETE FROM content_summaries WHERE segment = OLD.segment AND line = OLD.line; END;
"""
# Replay frontiers use only indexed steps, not JSON decoding. The window excludes
# the current attempt, and NULL frontiers never hide null-step observations.
_CTE = """
WITH scoped AS (
    SELECT * FROM segments WHERE org = ? AND project = ? AND run = ? AND error IS NULL
), frontiers AS (
    SELECT s.attempt, MIN(r.step) frontier FROM (
        SELECT segment, step FROM episode_records
        UNION ALL SELECT segment, step FROM content_records
    ) r JOIN scoped s USING(segment) GROUP BY s.attempt
), later AS (
    SELECT attempt, MIN(frontier) OVER (
        ORDER BY attempt DESC ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) frontier FROM frontiers
), records AS (
    SELECT r.*, q.payload summary, s.attempt, s.rank, s.writer, s.seq, s.stream,
           COALESCE(r.step >= later.frontier, 0) replayed
    FROM content_records r JOIN scoped s USING(segment)
    JOIN content_summaries q USING(segment, line) LEFT JOIN later USING(attempt)
)
"""
_ORDER = "t DESC, attempt DESC, seq DESC, line DESC, writer DESC, rank DESC"


def select(
    db: sqlite3.Connection,
    scope: tuple[str, ...],
    projection: str,
    where: str = "1",
    parameters: tuple[tp.Any, ...] = (),
    suffix: str = "",
) -> sqlite3.Cursor:
    """Execute internal SQL fragments with every external value parameterized."""

    return db.execute(_CTE + f"SELECT {projection} FROM records WHERE {where} " + suffix, (*scope, *parameters))


def decoded(row: sqlite3.Row, *, summary: bool = False) -> Record:
    """Detach one selected payload and its deterministic identity from SQLite."""

    return {
        **json.loads(row["summary" if summary else "payload"]),
        **{key: row[key] for key in ("attempt", "rank", "writer", "seq", "line")},
        "id": f"a{row['attempt']}-r{row['rank']}-{row['writer']}-{row['seq']}-{row['line']}",
        "replayed": bool(row["replayed"]),
        "_stream": row["stream"],
    }


def page(
    db: sqlite3.Connection,
    scope: tuple[str, ...],
    stream: str,
    *,
    where: str = "1",
    parameters: tuple[tp.Any, ...] = (),
    limit: int,
    offset: int,
    summary: bool = False,
    order: str = _ORDER,
) -> tuple[list[Record], int]:
    """Count and paginate in SQL before retrieving or decoding any payload."""

    where = "stream = ? AND " + where
    parameters = (stream, *parameters)
    total = select(db, scope, "COUNT(*)", where, parameters).fetchone()[0]
    payload = "summary" if summary else "payload"
    # LIMIT/OFFSET are values too; never interpolate user-provided integers.
    selected = select(
        db,
        scope,
        f"{payload}, attempt, rank, writer, seq, line, stream, replayed",
        where,
        (*parameters, limit, min(offset, 2**63 - 1)),
        f"ORDER BY {order} LIMIT ? OFFSET ?",
    )
    return [decoded(row, summary=summary) for row in selected], total


def detail(db: sqlite3.Connection, scope: tuple[str, ...], stream: str, identity: str) -> Record | None:
    """Select a single deterministic id before loading its full record."""

    attempt, rank, writer, seq, line = identity.split("-")
    if int(line) > 2**63 - 1:
        return None  # A syntactically valid id can still name an absent row.
    row = select(
        db,
        scope,
        "payload, attempt, rank, writer, seq, line, stream, replayed",
        "stream = ? AND attempt = ? AND rank = ? AND writer = ? AND seq = ? AND line = ?",
        (stream, int(attempt[1:]), int(rank[1:]), writer, int(seq), int(line)),
        "LIMIT 1",
    ).fetchone()
    return decoded(row) if row is not None else None


def episode_page(
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
    """Filter and paginate episode payloads after the shared SQL replay frontier."""

    sql = (
        _CTE
        + """, episode_selection AS (
        SELECT e.*, s.attempt, s.rank, s.writer, s.seq,
               COALESCE(e.step >= later.frontier, 0) replayed
        FROM episode_records e JOIN scoped s USING(segment) LEFT JOIN later USING(attempt)
    ) SELECT {projection} FROM episode_selection
      WHERE (? OR NOT replayed)
        AND (? IS NULL OR json_extract(payload, '$.kind') = ?)
        AND (? IS NULL OR json_extract(payload, '$.actor') = ?)
        AND (? IS NULL OR json_extract(payload, '$.env') = ?) """
    )
    parameters = (*scope, include_replayed, kind, kind, actor, actor, env, env)
    total = db.execute(sql.format(projection="COUNT(*)"), parameters).fetchone()[0]
    selected = db.execute(
        sql.format(projection="*") + "ORDER BY " + _ORDER + " LIMIT ? OFFSET ?",
        (*parameters, limit, min(offset, 2**63 - 1)),
    )
    return {
        "episodes": [
            {
                **json.loads(row["payload"]),
                "attempt": row["attempt"],
                "rank": row["rank"],
                "replayed": bool(row["replayed"]),
            }
            for row in selected
        ],
        "total": total,
    }


__all__ = ("SCHEMA", "decoded", "detail", "episode_page", "page", "select")
