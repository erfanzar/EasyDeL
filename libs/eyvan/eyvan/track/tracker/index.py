# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Rebuildable SQLite index over immutable tracking segments.

The database is disposable: all timestamps, ordering and values come from the
store, never from ingestion order. Each operation owns its connection and a
process-local lock; FastAPI's scan thread and concurrent readers cannot share a
transaction or accidentally observe a partially ingested segment.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import tempfile
import threading
import time
import typing as tp
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from eyvan.track._format import (
    FINISH_STATUSES,
    FORMAT_VERSION,
    META_OPS,
    STREAMS,
    SegmentName,
    decode_segment,
    decode_value,
    validate_run_id,
)
from eyvan.track._store import TrackingStore, TrackingStoreError
from eyvan.track.tracker import aggregates, phase2, phase3, phase3_records
from eyvan.track.tracker.phase3_index import PhaseThreeIndexMixin
from eyvan.track.tracker.records import integer, validate_episode, validate_histogram

_METRIC_KEY = re.compile(r"[A-Za-z0-9_./-]{1,128}")
_DROP_KEYS = ("keys", "rate", "invalid", "queue")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS segments (
    segment TEXT PRIMARY KEY, org TEXT NOT NULL, project TEXT NOT NULL, run TEXT NOT NULL,
    stream TEXT NOT NULL, attempt INTEGER NOT NULL, rank INTEGER NOT NULL,
    writer TEXT NOT NULL, seq INTEGER NOT NULL, error TEXT
);
CREATE INDEX IF NOT EXISTS segments_scope ON segments(org, project, run);
CREATE TABLE IF NOT EXISTS metadata (
    segment TEXT NOT NULL REFERENCES segments(segment), line INTEGER NOT NULL,
    t REAL NOT NULL, op TEXT NOT NULL, payload TEXT NOT NULL,
    PRIMARY KEY(segment, line)
);
CREATE TABLE IF NOT EXISTS scalar_records (
    segment TEXT NOT NULL REFERENCES segments(segment), line INTEGER NOT NULL,
    step INTEGER NOT NULL, t REAL NOT NULL, axes TEXT NOT NULL,
    PRIMARY KEY(segment, line)
);
CREATE TABLE IF NOT EXISTS scalars (
    segment TEXT NOT NULL REFERENCES segments(segment), line INTEGER NOT NULL,
    metric TEXT NOT NULL, step INTEGER NOT NULL, t REAL NOT NULL, value REAL,
    axes TEXT NOT NULL, PRIMARY KEY(segment, line, metric)
);
CREATE INDEX IF NOT EXISTS scalar_metric ON scalars(metric, segment, step);
CREATE TABLE IF NOT EXISTS scan_state (id INTEGER PRIMARY KEY CHECK(id = 1), last_scan_at REAL);
"""


@dataclass(frozen=True, slots=True)
class ScanResult:
    """Outcome of a scan; corrupt segments count as ingested, but not as data."""

    ingested_segments: int
    indexed_segments: int
    errors: int
    last_scan_at: float


def _finite(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError("expected a finite number")
    return float(value)


def _count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("expected a nonnegative integer")
    # SQLite INTEGER is signed 64-bit; reject the segment instead of stopping a scan.
    if value > 2**63 - 1:
        raise ValueError("integer exceeds index range")
    return value


def _metric(value: object) -> str:
    if not isinstance(value, str) or not _METRIC_KEY.fullmatch(value):
        raise ValueError("invalid metric or axis name")
    return value


def _validate_records(stream: str, records: list[dict[str, object]]) -> list[dict[str, tp.Any]]:
    validated: list[dict[str, tp.Any]] = []
    for record in records:
        if stream in phase3_records.STREAMS:
            validated.append(phase3_records.validate_record(stream, record))
            continue
        if stream == "hist":
            validated.append(validate_histogram(record))
            continue
        if stream == "episodes":
            validated.append(validate_episode(record))
            continue
        t = _finite(record.get("t"))
        if stream == "scalars":
            step = _count(record.get("step"))
            values = record.get("v")
            axes = record.get("x", {})
            if not isinstance(values, dict) or not isinstance(axes, dict):
                raise ValueError("scalar values and axes must be objects")
            validated.append(
                {
                    "step": step,
                    "t": t,
                    "v": {_metric(key): decode_value(value) for key, value in values.items()},
                    "x": {_metric(key): _finite(value) for key, value in axes.items()},
                }
            )
            continue
        op = record.get("op")
        if not isinstance(op, str):
            raise ValueError("metadata operation must be a string")
        if op not in META_OPS:
            continue  # Additive metadata operations do not break older readers.
        if op == "start":
            if record.get("fmt", FORMAT_VERSION) != FORMAT_VERSION:
                raise ValueError("unsupported tracking format version")
            for field in ("name", "group", "workload_id"):
                if record.get(field) is not None and not isinstance(record[field], str):
                    raise ValueError("invalid run identity metadata")
            if "tags" in record and (
                not isinstance(record["tags"], list) or not all(isinstance(tag, str) for tag in record["tags"])
            ):
                raise ValueError("run tags must be strings")
        elif op in {"config", "summary"}:
            if not isinstance(record.get(op), dict):
                raise ValueError(f"{op} must be an object")
        elif op == "finish":
            if record.get("status") not in FINISH_STATUSES:
                raise ValueError("invalid terminal run status")
        elif op == "stats":
            _count(record.get("records"))
            _count(record.get("spilled"))
            drops = record.get("dropped")
            if not isinstance(drops, dict):
                raise ValueError("dropped stats must be an object")
            for key in _DROP_KEYS:
                _count(drops.get(key, 0))
        # Reject non-standard JSON such as NaN even in nested metadata values.
        json.dumps(record, allow_nan=False, ensure_ascii=False).encode("utf-8")
        validated.append({**record, "t": t})
    return validated


def _segment_identity(key: str) -> tuple[str, str, str, str, SegmentName] | None:
    parts = key.split("/")
    if len(parts) != 5 or parts[3] not in STREAMS:
        return None
    org, project, run, stream, filename = parts
    try:
        for value in (org, project, run):
            validate_run_id(value)
        name = SegmentName.parse(filename)
    except ValueError:
        return None
    return (org, project, run, stream, name) if name is not None else None


class TrackingIndex(PhaseThreeIndexMixin):
    """Disposable, thread-safe tracking index backed by a local SQLite file."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path).expanduser().resolve()
        self._lock = threading.RLock()
        self._scan_lock = threading.Lock()
        with self._connection() as db:
            phase3.populate_summaries(db)

    @contextmanager
    def _connection(self) -> tp.Iterator[sqlite3.Connection]:
        with self._lock:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.db_path, timeout=30)
            connection.row_factory = sqlite3.Row
            try:
                connection.execute("PRAGMA foreign_keys = ON")
                connection.executescript(_SCHEMA + phase2.SCHEMA + phase3.SCHEMA)
                with connection:
                    yield connection
            finally:
                connection.close()

    def scan_state(self) -> dict[str, int | float | None]:
        """Return the remembered segment count and last successful scan time."""

        with self._connection() as db:
            row = db.execute("SELECT last_scan_at FROM scan_state WHERE id = 1").fetchone()
            return {
                "indexed_segments": db.execute("SELECT COUNT(*) FROM segments").fetchone()[0],
                "last_scan_at": row[0] if row else None,
            }

    def scan(self, store: TrackingStore) -> ScanResult:
        """Ingest unseen segment keys, isolating corrupt segments from valid ones.

        A download failure is retried on the next scan, not remembered as corrupt.
        Invalid compressed data or records are remembered with a credential-free
        diagnostic and never prevent other segments from being indexed.
        """

        ingested = errors = 0
        with self._scan_lock:
            with self._connection() as db:
                seen = {row[0] for row in db.execute("SELECT segment FROM segments")}
            for key in sorted(store.list()):
                if key in seen or (identity := _segment_identity(key)) is None:
                    continue
                org, project, run, stream, name = identity
                try:
                    data = store.get(key)
                except (TrackingStoreError, FileNotFoundError):
                    continue
                error = None
                try:
                    records = _validate_records(stream, decode_segment(data))
                except (ValueError, TypeError, OverflowError, RecursionError, zlib.error):
                    error = "corrupt or invalid tracking segment"
                    records = []
                    errors += 1
                with self._connection() as db:
                    if db.execute("SELECT 1 FROM segments WHERE segment = ?", (key,)).fetchone():
                        continue
                    db.execute(
                        "INSERT INTO segments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (key, org, project, run, stream, name.attempt, name.rank, name.writer, name.seq, error),
                    )
                    self._insert_records(db, key, stream, records)
                ingested += 1
                seen.add(key)
            now = time.time()
            with self._connection() as db:
                db.execute("INSERT OR REPLACE INTO scan_state VALUES (1, ?)", (now,))
                total = db.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
        return ScanResult(ingested, total, errors, now)

    @staticmethod
    def _insert_records(db: sqlite3.Connection, key: str, stream: str, records: list[dict[str, tp.Any]]) -> None:
        for line, r in enumerate(records):
            payload = json.dumps(r, allow_nan=False) if stream != "scalars" else ""
            if stream in phase3_records.STREAMS:
                db.execute("INSERT INTO content_records VALUES (?, ?, ?, ?, ?)", (key, line, r["step"], r["t"], payload))
                phase3.insert_summary(db, key, line, stream, r)
            elif stream == "meta":
                db.execute("INSERT INTO metadata VALUES (?, ?, ?, ?, ?)", (key, line, r["t"], r["op"], payload))
            elif stream == "hist":
                db.execute(
                    "INSERT INTO histogram_records VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (key, line, r["key"], r["step"], r["t"], json.dumps(r["x"]), payload),
                )
            elif stream == "episodes":
                db.execute(
                    "INSERT INTO episode_records VALUES (?, ?, ?, ?, ?, ?)",
                    (key, line, r["step"], r["t"], json.dumps(r["x"]), payload),
                )
            else:
                db.execute(
                    "INSERT INTO scalar_records VALUES (?, ?, ?, ?, ?)",
                    (key, line, r["step"], r["t"], json.dumps(r["x"])),
                )
                db.executemany(
                    "INSERT INTO scalars VALUES (?, ?, ?, ?, ?, ?, ?)",
                    [
                        (
                            key,
                            line,
                            metric,
                            r["step"],
                            r["t"],
                            value if math.isfinite(value) else None,
                            json.dumps(r["x"]),
                        )
                        for metric, value in r["v"].items()
                    ],
                )

    def rebuild(self, store: TrackingStore) -> ScanResult:
        """Clear only index data and reconstruct it from the authoritative store."""

        # Build beside the live database without holding its query lock. Readers
        # continue to see the complete old snapshot until atomic publication.
        with self._scan_lock:
            descriptor, temporary = tempfile.mkstemp(prefix=".tracking-rebuild-", dir=self.db_path.parent)
            os.close(descriptor)
            try:
                candidate = TrackingIndex(temporary)
                result = candidate.scan(store)
                with self._lock:
                    os.replace(temporary, self.db_path)
                return result
            finally:
                Path(temporary).unlink(missing_ok=True)

    @staticmethod
    def _scope(org: str, project: str, run: str | None = None) -> tuple[str, ...]:
        return tuple(validate_run_id(value) for value in (org, project, run) if value is not None)

    def list_runs(self, org: str, project: str, status: str | None = None, limit: int = 200) -> list[dict[str, tp.Any]]:
        """Return scoped run summaries, newest first, optionally filtered by status."""

        scope = self._scope(org, project)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        with self._connection() as db:
            runs = db.execute("SELECT DISTINCT run FROM segments WHERE org = ? AND project = ?", scope).fetchall()
            summaries = [self._detail(db, org, project, row[0], time.time(), include_detail=False) for row in runs]
        selected = [r for r in summaries if r is not None and (status is None or r["status"] == status)]
        selected.sort(key=lambda r: (-(r["updated_at"] if r["updated_at"] is not None else -math.inf), r["run_id"]))
        return selected[:limit]

    def run_detail(self, org: str, project: str, run_id: str) -> dict[str, tp.Any] | None:
        """Return one run's metadata and ingestion errors, or ``None`` if absent."""

        self._scope(org, project, run_id)
        with self._connection() as db:
            return self._detail(db, org, project, run_id, time.time(), include_detail=True)

    @staticmethod
    def _detail(
        db: sqlite3.Connection, org: str, project: str, run: str, now: float, *, include_detail: bool
    ) -> dict[str, tp.Any] | None:
        scope = (org, project, run)
        segments = db.execute(
            "SELECT * FROM segments WHERE org = ? AND project = ? AND run = ? ORDER BY segment", scope
        ).fetchall()
        if not segments:
            return None
        meta = db.execute(
            """SELECT m.*, s.attempt, s.rank, s.writer, s.seq FROM metadata m
                           JOIN segments s USING(segment) WHERE s.org = ? AND s.project = ? AND s.run = ?
                           ORDER BY m.t, s.attempt, s.seq, m.line, s.rank, s.writer""",
            scope,
        ).fetchall()
        scalar_stats = db.execute(
            """SELECT s.attempt, MIN(v.t) first_t, MAX(v.t) last_t,
                                    MIN(v.step) first_step, MAX(v.step) last_step
                                    FROM (
                                        SELECT segment, step, t FROM scalar_records
                                        UNION ALL SELECT segment, step, t FROM histogram_records
                                        UNION ALL SELECT segment, step, t FROM episode_records
                                        UNION ALL SELECT segment, step, t FROM content_records
                                    ) v JOIN segments s USING(segment)
                                    WHERE s.org = ? AND s.project = ? AND s.run = ? GROUP BY s.attempt""",
            scope,
        ).fetchall()
        attempts: dict[int, dict[str, tp.Any]] = {}
        times: dict[int, list[float]] = {}
        for segment in segments:
            attempt = segment["attempt"]
            attempts.setdefault(
                attempt, {"attempt": attempt, "started_at": None, "first_step": None, "last_step": None, "status": None}
            )
            times.setdefault(attempt, [])
        for row in scalar_stats:
            attempt = row["attempt"]
            times[attempt].extend([row["first_t"], row["last_t"]])
            attempts[attempt].update(first_step=row["first_step"], last_step=row["last_step"])
        config: dict[str, tp.Any] = {}
        summary: dict[str, tp.Any] = {}
        config_order: tuple[tp.Any, ...] = ()
        identity_order: dict[str, tuple[tp.Any, ...]] = {}
        identity: dict[str, tp.Any] = {"name": run, "group": None, "tags": [], "workload_id": None}
        writer_stats: dict[tuple[int, int, str], dict[str, tp.Any]] = {}
        for row in meta:
            record = json.loads(row["payload"])
            attempt, op = row["attempt"], row["op"]
            times[attempt].append(row["t"])
            order = (attempt, row["t"], row["seq"], row["line"], row["rank"], row["writer"])
            if op == "start":
                for key in identity:
                    if (
                        key in record
                        and (record[key] is not None or key != "workload_id")
                        and order > identity_order.get(key, ())
                    ):
                        identity[key] = record[key]
                        identity_order[key] = order
            elif op == "config" and order > config_order:
                config = record["config"]
                config_order = order
            elif op == "summary":
                summary.update(record["summary"])
            elif op == "finish":
                attempts[attempt]["status"] = record["status"]
            elif op == "stats":
                writer_stats[(attempt, row["rank"], row["writer"])] = record
        for attempt, entry in attempts.items():
            entry["started_at"] = min(times[attempt]) if times[attempt] else None
        all_times = [t for values in times.values() for t in values]
        latest = max(attempts)
        terminal = attempts[latest]["status"]
        last_t = max(times[latest]) if times[latest] else None
        status = terminal or ("running" if last_t is not None and 0 <= now - last_t <= 300 else "unknown")
        metric_rows = db.execute(
            """SELECT DISTINCT metric FROM scalars JOIN segments USING(segment)
                                    WHERE org = ? AND project = ? AND run = ? ORDER BY metric""",
            scope,
        ).fetchall()
        keys = [row[0] for row in metric_rows]
        last_steps = [r["last_step"] for r in attempts.values() if r["last_step"] is not None]
        result = {
            "run_id": run,
            "org_id": org,
            "project_id": project,
            **identity,
            "status": status,
            "attempts": len(attempts),
            "started_at": min(all_times) if all_times else None,
            "updated_at": max(all_times) if all_times else None,
            "last_step": max(last_steps) if last_steps else None,
            "keys_count": len(keys),
            "summary": summary,
        }
        if not include_detail:
            return result
        axes_rows = db.execute(
            """SELECT DISTINCT axes FROM scalar_records JOIN segments USING(segment)
                                  WHERE org = ? AND project = ? AND run = ?""",
            scope,
        ).fetchall()
        axes = sorted({axis for row in axes_rows for axis in json.loads(row[0])})
        stats = {
            "records": sum(s["records"] for s in writer_stats.values()),
            "dropped": {key: sum(s["dropped"].get(key, 0) for s in writer_stats.values()) for key in _DROP_KEYS},
            "spilled": sum(s["spilled"] for s in writer_stats.values()),
        }
        hist_keys = [
            row[0]
            for row in db.execute(
                """SELECT DISTINCT metric FROM histogram_records JOIN segments USING(segment)
                   WHERE org = ? AND project = ? AND run = ? ORDER BY metric""",
                scope,
            )
        ]
        episode_rows = phase2.episode_rows(db, scope)
        visible_episodes = [row for row in episode_rows if not row["replayed"]]
        return {
            **result,
            **phase3.detail_fields(db, scope),
            "hist_keys": hist_keys,
            "episodes": {kind: sum(row["kind"] == kind for row in visible_episodes) for kind in ("train", "eval")},
            "episode_axes": sorted({axis for row in visible_episodes for axis in row["x"]}),
            "envs": sorted({row["env"] for row in visible_episodes if row["env"] is not None}),
            "actors": len({row["actor"] for row in visible_episodes if row["actor"] is not None}),
            "config": config,
            "keys": keys,
            "axes": axes,
            "attempt_list": [attempts[attempt] for attempt in sorted(attempts)],
            "stats": stats,
            "errors": [{"segment": row["segment"], "error": row["error"]} for row in segments if row["error"]],
        }

    def histograms(
        self,
        org: str,
        project: str,
        run_id: str,
        key: str,
        x: str = "step",
        points: int = 200,
        bins: int = 64,
        rank: int | None = None,
    ) -> dict[str, tp.Any] | None:
        """Return winning histogram snapshots rebinned onto a common display grid."""

        scope = self._scope(org, project, run_id)
        phase2.validate_histogram_query(key, x, points, bins, rank)
        with self._connection() as db:
            if not phase2.exists(db, scope):
                return None
            return phase2.histograms(db, scope, key, x=x, points=points, bins=bins, rank=rank)

    def episodes(
        self,
        org: str,
        project: str,
        run_id: str,
        kind: str | None = None,
        actor: str | None = None,
        env: str | None = None,
        limit: int = 100,
        offset: int = 0,
        include_replayed: bool = False,
    ) -> dict[str, tp.Any] | None:
        """Return newest-first episode records with global replay flags and filters."""

        scope = self._scope(org, project, run_id)
        phase2.validate_episode_query(kind, actor, env)
        integer(limit, minimum=1, maximum=1000)
        integer(offset, minimum=0)
        if not isinstance(include_replayed, bool):
            raise ValueError("include_replayed must be boolean")
        with self._connection() as db:
            if not phase2.exists(db, scope):
                return None
            return phase2.episodes(
                db,
                scope,
                kind=kind,
                actor=actor,
                env=env,
                limit=limit,
                offset=offset,
                include_replayed=include_replayed,
            )

    def episode_curve(
        self,
        org: str,
        project: str,
        run_id: str,
        metric: str = "return",
        kind: str = "train",
        x: str = "step",
        points: int = 200,
        env: str | None = None,
        actor: str | None = None,
    ) -> dict[str, tp.Any] | None:
        """Return contiguous episode buckets with finite-only type-seven statistics."""

        scope = self._scope(org, project, run_id)
        phase2.validate_episode_query(kind, actor, env)
        if kind is None or metric not in ("return", "length"):
            raise ValueError("invalid episode curve metric or kind")
        _metric(x)
        integer(points, minimum=1, maximum=2000)
        with self._connection() as db:
            if not phase2.exists(db, scope):
                return None
            return phase2.episode_curve(db, scope, metric=metric, kind=kind, x=x, points=points, env=env, actor=actor)

    def episode_breakdown(
        self,
        org: str,
        project: str,
        run_id: str,
        kind: str = "train",
        last: int = 200,
    ) -> dict[str, tp.Any] | None:
        """Return actor windows, environment totals, train policy lag and eval groups."""

        scope = self._scope(org, project, run_id)
        phase2.validate_episode_query(kind)
        if kind is None:
            raise ValueError("episode breakdown kind is required")
        integer(last, minimum=1, maximum=5000)
        with self._connection() as db:
            if not phase2.exists(db, scope):
                return None
            return phase2.episode_breakdown(db, scope, kind=kind, last=last)

    def series(
        self,
        org: str,
        project: str,
        run_id: str,
        keys: tp.Sequence[str],
        x: str = "step",
        points: int = 1000,
        rank: int | None = None,
        include_replayed: bool = False,
        now: float | None = None,
    ) -> dict[str, tp.Any] | None:
        """Return raw or bucketed display values with optional overridden attempts.

        Within an attempt, sequence and line order win; writer id breaks exact
        cross-writer ties deterministically. Display points are ordered by step,
        including when a different x-axis is selected. ``now`` is accepted for
        callers using the same clock as run-status queries; series are timeless.
        """

        scope = self._scope(org, project, run_id)
        if isinstance(points, bool) or not isinstance(points, int) or not 1 <= points <= 5000:
            raise ValueError("points must be an integer from 1 through 5000")
        if not 1 <= len(keys) <= 32:
            raise ValueError("request between 1 and 32 metric keys")
        for key in keys:
            _metric(key)
        _metric(x)
        if rank is not None:
            _count(rank)
        with self._connection() as db:
            if (
                db.execute("SELECT 1 FROM segments WHERE org = ? AND project = ? AND run = ? LIMIT 1", scope).fetchone()
                is None
            ):
                return None
            result = [self._one_series(db, scope, key, x, points, rank, include_replayed) for key in keys]
        return {"run_id": run_id, "x": x, "series": result}

    @staticmethod
    def _one_series(
        db: sqlite3.Connection,
        scope: tuple[str, ...],
        key: str,
        x: str,
        points: int,
        rank: int | None,
        include_replayed: bool,
    ) -> dict[str, tp.Any]:
        if rank is None:
            row = db.execute(
                """SELECT MIN(rank) FROM scalars JOIN segments USING(segment)
                                WHERE org = ? AND project = ? AND run = ? AND metric = ?""",
                (*scope, key),
            ).fetchone()
            rank = row[0] if row[0] is not None else 0
        rows = db.execute(
            """SELECT v.*, s.attempt, s.seq, s.writer FROM scalars v JOIN segments s USING(segment)
                             WHERE s.org = ? AND s.project = ? AND s.run = ? AND v.metric = ? AND s.rank = ?
                             ORDER BY s.attempt, s.seq, v.line, s.writer""",
            (*scope, key, rank),
        ).fetchall()
        # First resolve same-attempt overwrites, then select the highest attempt
        # at each step. Missing custom axes never resurrect an older attempt.
        per_attempt: dict[int, dict[int, sqlite3.Row]] = {}
        for row in rows:
            per_attempt.setdefault(row["attempt"], {})[row["step"]] = row
        display: dict[int, sqlite3.Row] = {}
        for attempt in sorted(per_attempt):
            display.update(per_attempt[attempt])

        def x_value(row: sqlite3.Row) -> float | int | None:
            return row["step"] if x == "step" else row["t"] if x == "time" else json.loads(row["axes"]).get(x)

        values = [(axis, row["value"]) for _, row in sorted(display.items()) if (axis := x_value(row)) is not None]
        replayed = []
        boundaries = []
        for attempt, by_step in sorted(per_attempt.items()):
            attempt_values = [(axis, row) for _, row in sorted(by_step.items()) if (axis := x_value(row)) is not None]
            if attempt_values:
                boundaries.append({"attempt": attempt, "first_x": attempt_values[0][0]})
            if include_replayed:
                old = [[axis, row["value"]] for axis, row in attempt_values if display[row["step"]]["attempt"] > attempt]
                if old:
                    replayed.append({"attempt": attempt, "points": old})
        n = len(values)
        downsampled = n > points
        buckets = []
        if not downsampled:
            buckets = [[axis, value, value, value] for axis, value in values]
        else:
            for i in range(points):
                bucket = values[i * n // points : (i + 1) * n // points]
                finite = [value for _, value in bucket if value is not None]
                mean = aggregates.mean(finite)
                buckets.append([bucket[0][0], mean, min(finite) if finite else None, max(finite) if finite else None])
        return {
            "key": key,
            "rank": rank,
            "points": buckets,
            "downsampled": downsampled,
            "total": n,
            "nonfinite": sum(value is None for _, value in values),
            "attempts": boundaries,
            "replayed": replayed,
        }


__all__ = ("ScanResult", "TrackingIndex")
