# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Deterministic rich-record reads and same-run blob authorization."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import typing as tp

from eyvan.track._format import MEDIA_TYPES, SegmentName, blob_key, validate_sha256
from eyvan.track._store import TrackingStore, TrackingStoreError
from eyvan.track.tracker import aggregates, rich_sql
from eyvan.track.tracker.phase3_records import MAX_BLOB_BYTES
from eyvan.track.tracker.records import integer
from eyvan.track.tracker.replay import later_frontiers

Record = dict[str, tp.Any]
SCHEMA = (
    """
CREATE TABLE IF NOT EXISTS content_records (
    segment TEXT NOT NULL REFERENCES segments(segment), line INTEGER NOT NULL,
    step INTEGER, t REAL NOT NULL, payload TEXT NOT NULL,
    PRIMARY KEY(segment, line)
);
"""
    + rich_sql.SCHEMA
)
_ID = re.compile(r"a([0-9]+)-r([0-9]+)-([0-9a-f]{12})-([0-9]+)-([0-9]+)")


class BlobIntegrityError(RuntimeError):
    """The stored bytes do not match their content-addressed identity or size cap."""


def validate_id(value: str) -> str:
    """Require a canonical deterministic rich-record identity."""

    match = _ID.fullmatch(value)
    if match is None:
        raise ValueError("invalid tracking record id")
    attempt, rank, writer, seq, line = match.groups()
    name = SegmentName(int(attempt), int(rank), writer, int(seq))
    integer(int(line), minimum=0)
    if value != f"a{name.attempt}-r{name.rank}-{writer}-{name.seq}-{int(line)}":
        raise ValueError("non-canonical tracking record id")
    return value


def rows(db: sqlite3.Connection, scope: tuple[str, ...], stream: str | None = None) -> list[Record]:
    """Load rich records with deterministic identities and shared replay flags."""

    selected = db.execute(
        """SELECT r.*, s.attempt, s.rank, s.writer, s.seq, s.stream FROM content_records r
           JOIN segments s USING(segment) WHERE s.org = ? AND s.project = ? AND s.run = ?
           AND s.error IS NULL"""
        + (" AND s.stream = ?" if stream is not None else "")
        + " ORDER BY s.attempt, s.seq, r.line, s.writer, s.rank, s.stream",
        (*scope, stream) if stream is not None else scope,
    ).fetchall()
    later = later_frontiers(db, scope)
    return [
        {
            **json.loads(row["payload"]),
            **{key: row[key] for key in ("attempt", "rank", "writer", "seq", "line")},
            "id": f"a{row['attempt']}-r{row['rank']}-{row['writer']}-{row['seq']}-{row['line']}",
            "replayed": row["step"] is not None
            and later.get(row["attempt"]) is not None
            and row["step"] >= later[row["attempt"]],
            "_stream": row["stream"],
        }
        for row in selected
    ]


def detail_fields(db: sqlite3.Connection, scope: tuple[str, ...]) -> Record:
    """Return additive run-detail counts and keys over non-replayed records."""

    counts = {
        row[0]: row[1]
        for row in rich_sql.select(db, scope, "stream, COUNT(*)", "NOT replayed", suffix="GROUP BY stream")
    }
    return {
        "rollouts": counts.get("rollouts", 0),
        "completion_groups": counts.get("completions", 0),
        **{
            field: [
                row[0]
                for row in rich_sql.select(
                    db,
                    scope,
                    "DISTINCT json_extract(summary, '$.key') name",
                    "NOT replayed AND stream = ?",
                    (stream,),
                    "ORDER BY name",
                )
            ]
            for field, stream in (("table_keys", "tables"), ("media_keys", "media"))
        },
    }


class Availability:
    """Request-local lazy existence lookup; never download media to render a list."""

    def __init__(self, store: TrackingStore | None, scope: tuple[str, ...]) -> None:
        self.store = store
        self.prefix = "/".join(scope) + "/blobs/"
        self._keys: set[str] | None = None

    def reference(self, ref: Record | None) -> Record | None:
        """Annotate a blob reference from one prefix listing shared by the query."""

        if ref is None:
            return None
        if self._keys is None:
            try:
                self._keys = set(self.store.list(self.prefix)) if self.store is not None else set()
            except (TrackingStoreError, FileNotFoundError):
                self._keys = set()
        return {**ref, "available": self.prefix + ref["sha256"] in self._keys}


def _public(row: Record) -> Record:
    return {key: value for key, value in row.items() if key not in {"writer", "_stream", "seq", "line"}}


def rollouts(
    db: sqlite3.Connection,
    scope: tuple[str, ...],
    *,
    kind: str | None,
    actor: str | None,
    limit: int,
    offset: int,
    include_replayed: bool,
    store: TrackingStore | None,
) -> Record:
    """Return newest-first rollout summaries, omitting heavy tracks and actions."""

    selected, total = rich_sql.page(
        db,
        scope,
        "rollouts",
        where="(? OR NOT replayed) AND (? IS NULL OR json_extract(summary, '$.kind') = ?) "
        "AND (? IS NULL OR json_extract(summary, '$.actor') = ?)",
        parameters=(include_replayed, kind, kind, actor, actor),
        limit=limit,
        offset=offset,
        summary=True,
    )
    available = Availability(store, scope)
    result = []
    for row in selected:
        summary = {key: value for key, value in row.items() if key not in {"writer", "_stream", "_refs"}}
        summary["video"] = available.reference(row["video"])
        result.append(summary)
    return {"rollouts": result, "total": total}


def rollout_detail(
    db: sqlite3.Connection,
    scope: tuple[str, ...],
    rollout_id: str,
    store: TrackingStore | None,
) -> Record | None:
    """Return a full sampled trajectory, including replayed historical records."""

    row = rich_sql.detail(db, scope, "rollouts", rollout_id)
    return {**_public(row), "video": Availability(store, scope).reference(row["video"])} if row is not None else None


def _finite(values: tp.Iterable[tp.Any]) -> list[float | int]:
    return [
        value
        for value in values
        if isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)
    ]


def _std(values: tp.Sequence[float | int]) -> float | None:
    if not values:
        return None
    scale = max(abs(value) for value in values)
    if scale == 0:
        return 0.0
    normalized = [value / scale for value in values]
    average = tp.cast(float, aggregates.mean(normalized))
    result = scale * math.sqrt(tp.cast(float, aggregates.mean([(value - average) ** 2 for value in normalized])))
    return result if math.isfinite(result) else None


def _token_mean(responses: tp.Sequence[Record]) -> float | None:
    values = [response["tokens"] for response in responses if response["tokens"] is not None]
    try:
        return aggregates.mean(values)
    except OverflowError:
        # Tokens have no wire integer ceiling. Preserve the original values but
        # report an unrepresentable aggregate as unknown, never drop observations.
        return None


def completion_summary(row: Record) -> Record:
    """Compute finite-only response summaries with population standard deviation."""

    responses = row["responses"]
    rewards = _finite(response["reward"] for response in responses)
    std = _std(rewards)
    return {
        **{key: row[key] for key in ("id", "group", "step", "t", "attempt", "rank", "replayed")},
        "prompt": row["prompt"][:300],
        "k": len(responses),
        "mean_reward": aggregates.mean(rewards),
        "std_reward": std,
        "pass_rate": aggregates.mean([int(value > 0) for value in rewards]),
        "zero_variance": std is not None and std < 1e-8,
        "mean_tokens": _token_mean(responses),
        "truncated": sum(response["truncated"] is True for response in responses),
    }


def completions(
    db: sqlite3.Connection,
    scope: tuple[str, ...],
    *,
    step: int | None,
    limit: int,
    offset: int,
    sort: str,
    include_replayed: bool,
) -> Record:
    """Return completion group summaries and the visible run's distinct steps."""

    order = "t DESC, attempt DESC, seq DESC, line DESC, writer DESC, rank DESC"
    if sort == "zero_variance":
        order = "json_extract(summary, '$.zero_variance') DESC, " + order
    elif sort == "reward":
        order = "json_extract(summary, '$.mean_reward') IS NULL, json_extract(summary, '$.mean_reward') DESC, " + order
    summaries, total = rich_sql.page(
        db,
        scope,
        "completions",
        where="(? OR NOT replayed) AND (? IS NULL OR step = ?)",
        parameters=(include_replayed, step, step),
        limit=limit,
        offset=offset,
        summary=True,
        order=order,
    )
    return {
        "groups": [
            {
                key: value
                for key, value in row.items()
                if key not in {"writer", "seq", "line", "_stream", "_refs", "_stats"}
            }
            for row in summaries
        ],
        "total": total,
        "steps": [
            row[0]
            for row in rich_sql.select(
                db,
                scope,
                "DISTINCT step",
                "stream = 'completions' AND (? OR NOT replayed) AND step IS NOT NULL",
                (include_replayed,),
                "ORDER BY step LIMIT 2000",
            )
        ],
    }


def completion_detail(db: sqlite3.Connection, scope: tuple[str, ...], completion_id: str) -> Record | None:
    """Return the full prompt and responses alongside group summary fields."""

    row = rich_sql.detail(db, scope, "completions", completion_id)
    return {**completion_summary(row), **_public(row)} if row is not None else None


def completion_stats(db: sqlite3.Connection, scope: tuple[str, ...], *, points: int) -> Record:
    """Aggregate responses, then bucket distinct steps without losing any groups."""

    by_step: dict[int, list[Record]] = {}
    for selected in rich_sql.select(
        db,
        scope,
        "summary, attempt, rank, writer, seq, line, stream, replayed",
        "stream = 'completions' AND NOT replayed AND step IS NOT NULL",
    ):
        row = rich_sql.decoded(selected, summary=True)
        by_step.setdefault(row["step"], []).append(row)
    steps = sorted(by_step)
    n, b = len(steps), min(points, len(steps))
    buckets = []
    for i in range(b):
        selected = steps[i * n // b : (i + 1) * n // b]
        groups = [row for step in selected for row in by_step[step]]
        responses = [response for row in groups for response in row["_stats"]]
        components = sorted({component for response in responses for component in response["rewards"]})
        buckets.append(
            {
                "step": selected[0],
                "groups": len(groups),
                "mean_reward": aggregates.mean(_finite(response["reward"] for response in responses)),
                "zero_variance_frac": aggregates.mean([int(row["zero_variance"]) for row in groups]),
                "mean_tokens": _token_mean(responses),
                "truncated_frac": aggregates.mean(
                    [int(response["truncated"]) for response in responses if response["truncated"] is not None]
                ),
                "components": {
                    component: aggregates.mean(_finite(response["rewards"].get(component) for response in responses))
                    for component in components
                },
            }
        )
    return {"steps": buckets}


def tables(
    db: sqlite3.Connection,
    scope: tuple[str, ...],
    *,
    key: str | None,
    step: int | None,
    store: TrackingStore | None,
) -> Record | None:
    """Return table key counts or the winning snapshot of one named table."""

    where = "stream = 'tables' AND NOT replayed"
    if key is None:
        return {
            "keys": [
                {"key": row[0], "count": row[1], "last_step": row[2]}
                for row in rich_sql.select(
                    db,
                    scope,
                    "json_extract(summary, '$.key') name, COUNT(*), MAX(step)",
                    where,
                    suffix="GROUP BY name ORDER BY name",
                )
            ]
        }
    where += " AND json_extract(summary, '$.key') = ?"
    selected, _ = rich_sql.page(
        db,
        scope,
        "tables",
        where=where + " AND (? IS NULL OR step = ?)",
        parameters=(key, step, step),
        limit=1,
        offset=0,
        order="step IS NOT NULL DESC, step DESC, attempt DESC, seq DESC, line DESC, writer DESC, rank DESC",
    )
    if not selected:
        return None
    row = selected[0]
    available = Availability(store, scope)
    return {
        **{key: row[key] for key in ("key", "step", "t", "columns")},
        "rows": [
            [available.reference(cell) if isinstance(cell, dict) else cell for cell in cells] for cells in row["rows"]
        ],
        "steps": [
            item[0]
            for item in rich_sql.select(
                db,
                scope,
                "DISTINCT step",
                where + " AND step IS NOT NULL",
                (key,),
                "ORDER BY step",
            )
        ],
    }


def media(
    db: sqlite3.Connection,
    scope: tuple[str, ...],
    *,
    key: str | None,
    limit: int,
    offset: int,
    store: TrackingStore | None,
) -> Record:
    """Return newest-first non-replayed media with request-local availability."""

    selected, total = rich_sql.page(
        db,
        scope,
        "media",
        where="NOT replayed AND (? IS NULL OR json_extract(summary, '$.key') = ?)",
        parameters=(key, key),
        limit=limit,
        offset=offset,
    )
    available = Availability(store, scope)
    return {
        "media": [
            {
                **{key: row[key] for key in ("key", "step", "t", "caption")},
                "blob": available.reference(row["blob"]),
            }
            for row in selected
        ],
        "total": total,
    }


def _references(row: Record) -> tp.Iterator[Record]:
    if row["_stream"] == "rollouts" and row["video"] is not None:
        yield row["video"]
    elif row["_stream"] == "media":
        yield row["blob"]
    elif row["_stream"] == "tables":
        for cells in row["rows"]:
            yield from (cell for cell in cells if isinstance(cell, dict))


def blob_reference(db: sqlite3.Connection, scope: tuple[str, ...], digest: str) -> Record | None:
    """Authorize a digest using only detached, indexed references."""

    validate_sha256(digest)
    # References are a small projection, not full rollout tracks or table cells.
    row = db.execute(
        """SELECT json_extract(ref.value, '$.content_type'), json_extract(ref.value, '$.bytes')
           FROM content_summaries q JOIN content_records c USING(segment, line)
           JOIN segments s USING(segment), json_each(q.payload, '$._refs') ref
           WHERE s.org = ? AND s.project = ? AND s.run = ? AND s.error IS NULL
             AND json_extract(ref.value, '$.sha256') = ?
           ORDER BY s.attempt, s.seq, c.line, s.writer, s.rank, s.stream LIMIT 1""",
        (*scope, digest),
    ).fetchone()
    return {"content_type": row[0], "bytes": row[1]} if row is not None else None


def blob(
    reference: Record | None,
    scope: tuple[str, ...],
    digest: str,
    store: TrackingStore,
) -> tuple[bytes, str] | None:
    """Authorize this run's indexed reference, then verify bytes before serving.

    Historical replayed references remain authorized. Corrupt segments never
    contribute records; another run's digest is insufficient even if bytes were
    independently placed in this run's prefix.
    """

    validate_sha256(digest)
    if reference is None:
        return None
    content_type = reference.get("content_type")
    if not isinstance(content_type, str) or content_type not in MEDIA_TYPES:
        return None
    if not isinstance(reference.get("bytes"), int) or not 0 <= reference["bytes"] <= MAX_BLOB_BYTES:
        raise BlobIntegrityError("blob integrity check failed")
    key = f"{scope[0]}/{scope[1]}/{blob_key(scope[2], digest)}"
    try:
        bounded = getattr(store, "get_bounded", None)
        data = bounded(key, MAX_BLOB_BYTES + 1) if callable(bounded) else store.get(key)
    except FileNotFoundError:
        return None
    if len(data) > MAX_BLOB_BYTES or hashlib.sha256(data).hexdigest() != digest:
        raise BlobIntegrityError("blob integrity check failed")
    return data, content_type


def insert_summary(db: sqlite3.Connection, segment: str, line: int, stream: str, record: Record) -> None:
    """Persist a rebuildable lightweight query projection beside the full record."""

    if stream == "completions":
        summary = completion_summary({**record, "id": "", "attempt": 0, "rank": 0, "replayed": False})
        summary["_stats"] = [
            {key: response[key] for key in ("reward", "rewards", "tokens", "truncated")}
            for response in record["responses"]
        ]
    elif stream == "rollouts":
        summary = {key: value for key, value in record.items() if key not in {"tracks", "actions"}}
        summary["track_names"] = sorted(record["tracks"])
    elif stream == "tables":
        summary = {"key": record["key"]}
    else:
        summary = record.copy()
    summary["_refs"] = list(_references({**record, "_stream": stream}))
    db.execute("INSERT INTO content_summaries VALUES (?, ?, ?)", (segment, line, json.dumps(summary, allow_nan=False)))


def populate_summaries(db: sqlite3.Connection) -> None:
    """Upgrade a pre-projection disposable index one record at a time."""

    for row in db.execute(
        """SELECT c.segment, c.line, c.payload, s.stream FROM content_records c
           JOIN segments s USING(segment) LEFT JOIN content_summaries q USING(segment, line)
           WHERE q.segment IS NULL"""
    ):
        insert_summary(db, row["segment"], row["line"], row["stream"], json.loads(row["payload"]))


def annotate_availability(result: tp.Any, store: TrackingStore | None, scope: tuple[str, ...]) -> tp.Any:
    """Perform store listing only after all indexed results have been detached."""

    available = Availability(store, scope)

    def visit(value: tp.Any) -> tp.Any:
        if isinstance(value, dict):
            if "available" in value and "sha256" in value:
                return available.reference(value)
            return {key: visit(item) for key, item in value.items()}
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    return visit(result)


__all__ = (
    "SCHEMA",
    "BlobIntegrityError",
    "blob",
    "completion_detail",
    "completion_stats",
    "completions",
    "detail_fields",
    "media",
    "rollout_detail",
    "rollouts",
    "rows",
    "tables",
    "validate_id",
)
