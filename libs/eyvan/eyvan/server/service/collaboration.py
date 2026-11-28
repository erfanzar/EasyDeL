# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Durable SQLite store and access policy for collaborative runs, experiments, speedruns, and team notes."""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import typing as tp
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from eyvan.server.service.errors import InputError

VALID_VISIBILITIES: frozenset[str] = frozenset({"private", "shared", "team"})
VALID_EXPERIMENT_KINDS: frozenset[str] = frozenset({"experiment", "speedrun"})
VALID_GOALS: frozenset[str] = frozenset({"min", "max"})
VALID_NOTE_TARGETS: frozenset[str] = frozenset({"run", "experiment"})

_MAX_SCOPE_LEN = 128
_MAX_RUN_ID_LEN = 128
_MAX_NAME_LEN = 160
_MAX_SUBJECT_LEN = 120
_MAX_DESC_LEN = 4000
_MAX_METRIC_KEY_LEN = 128
_MAX_GROUP_LEN = 256
_MAX_NOTE_BODY_LEN = 4000
_MAX_SHARED_WITH = 64
_MAX_TAG_FILTER = 32
_MAX_TAG_LEN = 64

_UNSET: tp.Any = object()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_overlays (
    org_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    owner TEXT,
    visibility TEXT NOT NULL DEFAULT 'team',
    shared_with_json TEXT NOT NULL DEFAULT '[]',
    experiment_id TEXT,
    pinned INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL,
    updated_by TEXT NOT NULL,
    PRIMARY KEY (org_id, project_id, run_id)
);
CREATE INDEX IF NOT EXISTS run_overlays_scope ON run_overlays(org_id, project_id);
CREATE INDEX IF NOT EXISTS run_overlays_experiment ON run_overlays(org_id, project_id, experiment_id);

CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    name TEXT NOT NULL,
    subject TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT 'experiment',
    description TEXT NOT NULL DEFAULT '',
    metric_key TEXT,
    goal TEXT NOT NULL DEFAULT 'min',
    target_value REAL,
    group_filter TEXT,
    tag_filter_json TEXT NOT NULL DEFAULT '[]',
    visibility TEXT NOT NULL DEFAULT 'team',
    shared_with_json TEXT NOT NULL DEFAULT '[]',
    owner TEXT NOT NULL,
    pinned_baseline_run_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS experiments_scope ON experiments(org_id, project_id);

CREATE TABLE IF NOT EXISTS notes (
    id TEXT PRIMARY KEY,
    org_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    author TEXT NOT NULL,
    body TEXT NOT NULL,
    metric_ref TEXT,
    step_ref INTEGER,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS notes_target ON notes(org_id, project_id, target_type, target_id, created_at);
"""


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def validate_visibility(value: str) -> str:
    """Validate and normalize a run or experiment visibility level."""

    if not isinstance(value, str):
        raise InputError('visibility must be "private", "shared", or "team"')
    cleaned = value.strip().lower()
    if cleaned not in VALID_VISIBILITIES:
        raise InputError('visibility must be "private", "shared", or "team"')
    return cleaned


def validate_experiment_kind(value: str) -> str:
    """Validate and normalize an experiment kind (``experiment`` or ``speedrun``)."""

    if not isinstance(value, str):
        raise InputError('kind must be "experiment" or "speedrun"')
    cleaned = value.strip().lower()
    if cleaned not in VALID_EXPERIMENT_KINDS:
        raise InputError('kind must be "experiment" or "speedrun"')
    return cleaned


def validate_goal(value: str) -> str:
    """Validate and normalize an optimization direction (``min`` or ``max``)."""

    if not isinstance(value, str):
        raise InputError('goal must be "min" or "max"')
    cleaned = value.strip().lower()
    if cleaned not in VALID_GOALS:
        raise InputError('goal must be "min" or "max"')
    return cleaned


def validate_note_target_type(value: str) -> str:
    """Validate and normalize a note target type (``run`` or ``experiment``)."""

    if not isinstance(value, str):
        raise InputError('target_type must be "run" or "experiment"')
    cleaned = value.strip().lower()
    if cleaned not in VALID_NOTE_TARGETS:
        raise InputError('target_type must be "run" or "experiment"')
    return cleaned


def normalize_subjects(values: tp.Sequence[str], *, max_items: int = _MAX_SHARED_WITH) -> list[str]:
    """Strip, bound, deduplicate, and sort a list of principal subjects."""

    if isinstance(values, str):
        raise InputError("shared_with must be a list of subject strings")
    result: set[str] = set()
    for raw in values:
        if not isinstance(raw, str):
            raise InputError("shared_with entries must be strings")
        cleaned = raw.strip()
        if not cleaned:
            continue
        if len(cleaned) > _MAX_SCOPE_LEN:
            raise InputError(f"subject {cleaned[:32]!r}... exceeds {_MAX_SCOPE_LEN} characters")
        result.add(cleaned)
    if len(result) > max_items:
        raise InputError(f"shared_with accepts at most {max_items} subjects")
    return sorted(result)


def normalize_tags(values: tp.Sequence[str], *, max_items: int = _MAX_TAG_FILTER) -> list[str]:
    """Strip, bound, deduplicate, and sort a list of filter tags."""

    if isinstance(values, str):
        raise InputError("tag_filter must be a list of tag strings")
    result: set[str] = set()
    for raw in values:
        if not isinstance(raw, str):
            raise InputError("tag_filter entries must be strings")
        cleaned = raw.strip()
        if not cleaned:
            continue
        if len(cleaned) > _MAX_TAG_LEN:
            raise InputError(f"tag {cleaned[:32]!r}... exceeds {_MAX_TAG_LEN} characters")
        result.add(cleaned)
    if len(result) > max_items:
        raise InputError(f"tag_filter accepts at most {max_items} tags")
    return sorted(result)


def _validate_required_text(value: str, *, field_name: str, max_len: int) -> str:
    if not isinstance(value, str):
        raise InputError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise InputError(f"{field_name} must not be empty")
    if len(cleaned) > max_len:
        raise InputError(f"{field_name} must be at most {max_len} characters")
    return cleaned


def _validate_optional_text(
    value: str | None, *, field_name: str, max_len: int, allow_empty: bool = False
) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InputError(f"{field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        return "" if allow_empty else None
    if len(cleaned) > max_len:
        raise InputError(f"{field_name} must be at most {max_len} characters")
    return cleaned


def _validate_finite_float(value: float | None, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise InputError(f"{field_name} must be a finite number")
    return float(value)


def _validate_step_ref(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InputError("step_ref must be a non-negative integer")
    return value


def can_view(
    principal_subject: str,
    *,
    owner: str | None,
    visibility: str,
    shared_with: tp.Sequence[str],
    is_admin: bool,
) -> bool:
    """Return whether ``principal_subject`` may read a run or experiment."""

    if is_admin:
        return True
    if visibility == "team":
        return True
    if owner and owner == principal_subject:
        return True
    if visibility == "shared" and principal_subject in shared_with:
        return True
    return False


def can_manage(principal_subject: str, *, owner: str | None, is_admin: bool) -> bool:
    """Return whether ``principal_subject`` may mutate sharing or metadata for an owned object."""

    return bool(is_admin or (bool(owner) and owner == principal_subject))


def _overlay_from_row(row: sqlite3.Row) -> dict[str, tp.Any]:
    return {
        "org_id": row["org_id"],
        "project_id": row["project_id"],
        "run_id": row["run_id"],
        "owner": row["owner"],
        "visibility": row["visibility"],
        "shared_with": json.loads(row["shared_with_json"]),
        "experiment_id": row["experiment_id"],
        "pinned": bool(row["pinned"]),
        "updated_at": row["updated_at"],
        "updated_by": row["updated_by"],
    }


def _experiment_from_row(row: sqlite3.Row) -> dict[str, tp.Any]:
    return {
        "id": row["id"],
        "org_id": row["org_id"],
        "project_id": row["project_id"],
        "name": row["name"],
        "subject": row["subject"],
        "kind": row["kind"],
        "description": row["description"],
        "metric_key": row["metric_key"],
        "goal": row["goal"],
        "target_value": row["target_value"],
        "group_filter": row["group_filter"],
        "tag_filter": json.loads(row["tag_filter_json"]),
        "visibility": row["visibility"],
        "shared_with": json.loads(row["shared_with_json"]),
        "owner": row["owner"],
        "pinned_baseline_run_id": row["pinned_baseline_run_id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _note_from_row(row: sqlite3.Row) -> dict[str, tp.Any]:
    return {
        "id": row["id"],
        "org_id": row["org_id"],
        "project_id": row["project_id"],
        "target_type": row["target_type"],
        "target_id": row["target_id"],
        "author": row["author"],
        "body": row["body"],
        "metric_ref": row["metric_ref"],
        "step_ref": row["step_ref"],
        "created_at": row["created_at"],
    }


class CollaborationStore:
    """Thread-safe SQLite store for run overlays, experiments/speedruns, and team notes."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._lock = threading.RLock()
        self._is_memory = str(db_path) == ":memory:"
        self.db_path: Path | None = None if self._is_memory else Path(db_path).expanduser().resolve()
        self._memory_conn: sqlite3.Connection | None = None
        self._init_schema()

    def _init_schema(self) -> None:
        """Create schema and configure WAL mode once on store initialization."""

        with self._lock:
            if self._is_memory:
                conn = sqlite3.connect(":memory:", check_same_thread=False)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("PRAGMA busy_timeout = 30000")
                conn.executescript(_SCHEMA)
                self._memory_conn = conn
                return
            assert self.db_path is not None
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA journal_mode = WAL")
                conn.execute("PRAGMA synchronous = NORMAL")
                conn.execute("PRAGMA busy_timeout = 30000")
                conn.execute("PRAGMA foreign_keys = ON")
                conn.executescript(_SCHEMA)
            finally:
                conn.close()

    def close(self) -> None:
        """Close the underlying connection if running in-memory."""

        with self._lock:
            if self._memory_conn is not None:
                self._memory_conn.close()
                self._memory_conn = None

    @contextmanager
    def _connection(self) -> tp.Iterator[sqlite3.Connection]:
        with self._lock:
            if self._is_memory:
                if self._memory_conn is None:
                    self._init_schema()
                assert self._memory_conn is not None
                with self._memory_conn:
                    yield self._memory_conn
                return
            assert self.db_path is not None
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("PRAGMA busy_timeout = 30000")
                with conn:
                    yield conn
            finally:
                conn.close()

    # ------------------------------------------------------------------ run overlays

    def get_run_overlay(self, org_id: str, project_id: str, run_id: str) -> dict[str, tp.Any] | None:
        """Return one run's collaboration overlay, or ``None`` if none has been saved."""

        norm_org = _validate_required_text(org_id, field_name="org_id", max_len=_MAX_SCOPE_LEN)
        norm_project = _validate_required_text(project_id, field_name="project_id", max_len=_MAX_SCOPE_LEN)
        norm_run = _validate_required_text(run_id, field_name="run_id", max_len=_MAX_RUN_ID_LEN)
        with self._connection() as db:
            row = db.execute(
                "SELECT * FROM run_overlays WHERE org_id = ? AND project_id = ? AND run_id = ?",
                (norm_org, norm_project, norm_run),
            ).fetchone()
            return _overlay_from_row(row) if row is not None else None

    def list_run_overlays(self, org_id: str, project_id: str) -> dict[str, dict[str, tp.Any]]:
        """Return all run overlays for ``(org_id, project_id)`` keyed by ``run_id``."""

        norm_org = _validate_required_text(org_id, field_name="org_id", max_len=_MAX_SCOPE_LEN)
        norm_project = _validate_required_text(project_id, field_name="project_id", max_len=_MAX_SCOPE_LEN)
        with self._connection() as db:
            rows = db.execute(
                "SELECT * FROM run_overlays WHERE org_id = ? AND project_id = ?",
                (norm_org, norm_project),
            ).fetchall()
            return {row["run_id"]: _overlay_from_row(row) for row in rows}

    def upsert_run_overlay(
        self,
        org_id: str,
        project_id: str,
        run_id: str,
        *,
        updated_by: str,
        owner: str | None = _UNSET,
        visibility: str = _UNSET,
        shared_with: tp.Sequence[str] = _UNSET,
        experiment_id: str | None = _UNSET,
        pinned: bool = _UNSET,
    ) -> dict[str, tp.Any]:
        """Create or partially update a run's sharing, ownership, pinning, or experiment link."""

        norm_org = _validate_required_text(org_id, field_name="org_id", max_len=_MAX_SCOPE_LEN)
        norm_project = _validate_required_text(project_id, field_name="project_id", max_len=_MAX_SCOPE_LEN)
        norm_run = _validate_required_text(run_id, field_name="run_id", max_len=_MAX_RUN_ID_LEN)
        norm_updated_by = _validate_required_text(updated_by, field_name="updated_by", max_len=_MAX_SCOPE_LEN)

        with self._connection() as db:
            existing_row = db.execute(
                "SELECT * FROM run_overlays WHERE org_id = ? AND project_id = ? AND run_id = ?",
                (norm_org, norm_project, norm_run),
            ).fetchone()
            existing = _overlay_from_row(existing_row) if existing_row is not None else None

            resolved_owner = existing["owner"] if existing is not None else None
            if owner is not _UNSET:
                resolved_owner = _validate_optional_text(owner, field_name="owner", max_len=_MAX_SCOPE_LEN)

            resolved_visibility = existing["visibility"] if existing is not None else "team"
            if visibility is not _UNSET:
                resolved_visibility = validate_visibility(visibility)

            resolved_shared = existing["shared_with"] if existing is not None else []
            if shared_with is not _UNSET:
                resolved_shared = normalize_subjects(shared_with)

            resolved_exp_id = existing["experiment_id"] if existing is not None else None
            if experiment_id is not _UNSET:
                resolved_exp_id = _validate_optional_text(
                    experiment_id, field_name="experiment_id", max_len=_MAX_SCOPE_LEN
                )

            resolved_pinned = existing["pinned"] if existing is not None else False
            if pinned is not _UNSET:
                if not isinstance(pinned, bool):
                    raise InputError("pinned must be a boolean")
                resolved_pinned = pinned

            now = _utcnow_iso()
            db.execute(
                """INSERT INTO run_overlays
                   (org_id, project_id, run_id, owner, visibility, shared_with_json,
                    experiment_id, pinned, updated_at, updated_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(org_id, project_id, run_id) DO UPDATE SET
                       owner = excluded.owner,
                       visibility = excluded.visibility,
                       shared_with_json = excluded.shared_with_json,
                       experiment_id = excluded.experiment_id,
                       pinned = excluded.pinned,
                       updated_at = excluded.updated_at,
                       updated_by = excluded.updated_by""",
                (
                    norm_org,
                    norm_project,
                    norm_run,
                    resolved_owner,
                    resolved_visibility,
                    json.dumps(resolved_shared),
                    resolved_exp_id,
                    1 if resolved_pinned else 0,
                    now,
                    norm_updated_by,
                ),
            )
            row = db.execute(
                "SELECT * FROM run_overlays WHERE org_id = ? AND project_id = ? AND run_id = ?",
                (norm_org, norm_project, norm_run),
            ).fetchone()
            assert row is not None
            return _overlay_from_row(row)

    # ------------------------------------------------------------------ experiments

    def create_experiment(
        self,
        org_id: str,
        project_id: str,
        *,
        name: str,
        owner: str,
        subject: str = "",
        kind: str = "experiment",
        description: str = "",
        metric_key: str | None = None,
        goal: str = "min",
        target_value: float | None = None,
        group_filter: str | None = None,
        tag_filter: tp.Sequence[str] = (),
        visibility: str = "team",
        shared_with: tp.Sequence[str] = (),
        pinned_baseline_run_id: str | None = None,
    ) -> dict[str, tp.Any]:
        """Create and persist a collaborative experiment or speedrun record."""

        norm_org = _validate_required_text(org_id, field_name="org_id", max_len=_MAX_SCOPE_LEN)
        norm_project = _validate_required_text(project_id, field_name="project_id", max_len=_MAX_SCOPE_LEN)
        norm_name = _validate_required_text(name, field_name="name", max_len=_MAX_NAME_LEN)
        norm_owner = _validate_required_text(owner, field_name="owner", max_len=_MAX_SCOPE_LEN)
        norm_subject = (
            _validate_optional_text(subject, field_name="subject", max_len=_MAX_SUBJECT_LEN, allow_empty=True) or ""
        )
        norm_kind = validate_experiment_kind(kind)
        norm_desc = (
            _validate_optional_text(description, field_name="description", max_len=_MAX_DESC_LEN, allow_empty=True) or ""
        )
        norm_metric = _validate_optional_text(metric_key, field_name="metric_key", max_len=_MAX_METRIC_KEY_LEN)
        norm_goal = validate_goal(goal)
        norm_target = _validate_finite_float(target_value, field_name="target_value")
        norm_group = _validate_optional_text(group_filter, field_name="group_filter", max_len=_MAX_GROUP_LEN)
        norm_tags = normalize_tags(tag_filter)
        norm_visibility = validate_visibility(visibility)
        norm_shared = normalize_subjects(shared_with)
        norm_baseline = _validate_optional_text(
            pinned_baseline_run_id, field_name="pinned_baseline_run_id", max_len=_MAX_RUN_ID_LEN
        )

        exp_id = f"exp_{uuid.uuid4().hex[:12]}"
        now = _utcnow_iso()
        with self._connection() as db:
            db.execute(
                """INSERT INTO experiments
                   (id, org_id, project_id, name, subject, kind, description,
                    metric_key, goal, target_value, group_filter, tag_filter_json,
                    visibility, shared_with_json, owner, pinned_baseline_run_id,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    exp_id,
                    norm_org,
                    norm_project,
                    norm_name,
                    norm_subject,
                    norm_kind,
                    norm_desc,
                    norm_metric,
                    norm_goal,
                    norm_target,
                    norm_group,
                    json.dumps(norm_tags),
                    norm_visibility,
                    json.dumps(norm_shared),
                    norm_owner,
                    norm_baseline,
                    now,
                    now,
                ),
            )
            row = db.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
            assert row is not None
            return _experiment_from_row(row)

    def get_experiment(self, org_id: str, project_id: str, exp_id: str) -> dict[str, tp.Any] | None:
        """Return one experiment scoped to ``(org_id, project_id)``, or ``None``."""

        norm_org = _validate_required_text(org_id, field_name="org_id", max_len=_MAX_SCOPE_LEN)
        norm_project = _validate_required_text(project_id, field_name="project_id", max_len=_MAX_SCOPE_LEN)
        norm_exp = _validate_required_text(exp_id, field_name="exp_id", max_len=_MAX_SCOPE_LEN)
        with self._connection() as db:
            row = db.execute(
                "SELECT * FROM experiments WHERE id = ? AND org_id = ? AND project_id = ?",
                (norm_exp, norm_org, norm_project),
            ).fetchone()
            return _experiment_from_row(row) if row is not None else None

    def list_experiments(
        self,
        org_id: str,
        project_id: str,
        *,
        kind: str | None = None,
        subject: str | None = None,
        visibility: str | None = None,
    ) -> list[dict[str, tp.Any]]:
        """List experiments in ``(org_id, project_id)`` newest-updated first."""

        norm_org = _validate_required_text(org_id, field_name="org_id", max_len=_MAX_SCOPE_LEN)
        norm_project = _validate_required_text(project_id, field_name="project_id", max_len=_MAX_SCOPE_LEN)
        clauses = ["org_id = ?", "project_id = ?"]
        params: list[tp.Any] = [norm_org, norm_project]
        if kind is not None:
            clauses.append("kind = ?")
            params.append(validate_experiment_kind(kind))
        if subject is not None and subject.strip():
            clauses.append("LOWER(subject) = LOWER(?)")
            params.append(subject.strip())
        if visibility is not None:
            clauses.append("visibility = ?")
            params.append(validate_visibility(visibility))
        sql = f"SELECT * FROM experiments WHERE {' AND '.join(clauses)} ORDER BY updated_at DESC, id ASC"
        with self._connection() as db:
            rows = db.execute(sql, params).fetchall()
            return [_experiment_from_row(row) for row in rows]

    def update_experiment(
        self,
        org_id: str,
        project_id: str,
        exp_id: str,
        *,
        name: str = _UNSET,
        subject: str = _UNSET,
        kind: str = _UNSET,
        description: str = _UNSET,
        metric_key: str | None = _UNSET,
        goal: str = _UNSET,
        target_value: float | None = _UNSET,
        group_filter: str | None = _UNSET,
        tag_filter: tp.Sequence[str] = _UNSET,
        visibility: str = _UNSET,
        shared_with: tp.Sequence[str] = _UNSET,
        pinned_baseline_run_id: str | None = _UNSET,
    ) -> dict[str, tp.Any] | None:
        """Partially update an experiment; returns ``None`` if absent in scope."""

        norm_org = _validate_required_text(org_id, field_name="org_id", max_len=_MAX_SCOPE_LEN)
        norm_project = _validate_required_text(project_id, field_name="project_id", max_len=_MAX_SCOPE_LEN)
        norm_exp = _validate_required_text(exp_id, field_name="exp_id", max_len=_MAX_SCOPE_LEN)
        with self._connection() as db:
            existing_row = db.execute(
                "SELECT * FROM experiments WHERE id = ? AND org_id = ? AND project_id = ?",
                (norm_exp, norm_org, norm_project),
            ).fetchone()
            if existing_row is None:
                return None
            existing = _experiment_from_row(existing_row)

            resolved_name = (
                _validate_required_text(name, field_name="name", max_len=_MAX_NAME_LEN)
                if name is not _UNSET
                else existing["name"]
            )
            resolved_subject = (
                _validate_optional_text(subject, field_name="subject", max_len=_MAX_SUBJECT_LEN, allow_empty=True) or ""
                if subject is not _UNSET
                else existing["subject"]
            )
            resolved_kind = validate_experiment_kind(kind) if kind is not _UNSET else existing["kind"]
            resolved_desc = (
                _validate_optional_text(description, field_name="description", max_len=_MAX_DESC_LEN, allow_empty=True)
                or ""
                if description is not _UNSET
                else existing["description"]
            )
            resolved_metric = (
                _validate_optional_text(metric_key, field_name="metric_key", max_len=_MAX_METRIC_KEY_LEN)
                if metric_key is not _UNSET
                else existing["metric_key"]
            )
            resolved_goal = validate_goal(goal) if goal is not _UNSET else existing["goal"]
            resolved_target = (
                _validate_finite_float(target_value, field_name="target_value")
                if target_value is not _UNSET
                else existing["target_value"]
            )
            resolved_group = (
                _validate_optional_text(group_filter, field_name="group_filter", max_len=_MAX_GROUP_LEN)
                if group_filter is not _UNSET
                else existing["group_filter"]
            )
            resolved_tags = normalize_tags(tag_filter) if tag_filter is not _UNSET else existing["tag_filter"]
            resolved_vis = validate_visibility(visibility) if visibility is not _UNSET else existing["visibility"]
            resolved_shared = normalize_subjects(shared_with) if shared_with is not _UNSET else existing["shared_with"]
            resolved_baseline = (
                _validate_optional_text(
                    pinned_baseline_run_id, field_name="pinned_baseline_run_id", max_len=_MAX_RUN_ID_LEN
                )
                if pinned_baseline_run_id is not _UNSET
                else existing["pinned_baseline_run_id"]
            )

            now = _utcnow_iso()
            db.execute(
                """UPDATE experiments
                   SET name = ?, subject = ?, kind = ?, description = ?,
                       metric_key = ?, goal = ?, target_value = ?, group_filter = ?,
                       tag_filter_json = ?, visibility = ?, shared_with_json = ?,
                       pinned_baseline_run_id = ?, updated_at = ?
                   WHERE id = ? AND org_id = ? AND project_id = ?""",
                (
                    resolved_name,
                    resolved_subject,
                    resolved_kind,
                    resolved_desc,
                    resolved_metric,
                    resolved_goal,
                    resolved_target,
                    resolved_group,
                    json.dumps(resolved_tags),
                    resolved_vis,
                    json.dumps(resolved_shared),
                    resolved_baseline,
                    now,
                    norm_exp,
                    norm_org,
                    norm_project,
                ),
            )
            row = db.execute(
                "SELECT * FROM experiments WHERE id = ? AND org_id = ? AND project_id = ?",
                (norm_exp, norm_org, norm_project),
            ).fetchone()
            assert row is not None
            return _experiment_from_row(row)

    def delete_experiment(self, org_id: str, project_id: str, exp_id: str) -> bool:
        """Atomically delete an experiment, unlink run overlays, and remove its notes."""

        norm_org = _validate_required_text(org_id, field_name="org_id", max_len=_MAX_SCOPE_LEN)
        norm_project = _validate_required_text(project_id, field_name="project_id", max_len=_MAX_SCOPE_LEN)
        norm_exp = _validate_required_text(exp_id, field_name="exp_id", max_len=_MAX_SCOPE_LEN)
        with self._connection() as db:
            cursor = db.execute(
                "DELETE FROM experiments WHERE id = ? AND org_id = ? AND project_id = ?",
                (norm_exp, norm_org, norm_project),
            )
            if cursor.rowcount == 0:
                return False
            db.execute(
                "UPDATE run_overlays SET experiment_id = NULL WHERE org_id = ? AND project_id = ? AND experiment_id = ?",
                (norm_org, norm_project, norm_exp),
            )
            db.execute(
                "DELETE FROM notes WHERE org_id = ? AND project_id = ? AND target_type = 'experiment' AND target_id = ?",
                (norm_org, norm_project, norm_exp),
            )
            return True

    # ------------------------------------------------------------------ notes

    def create_note(
        self,
        org_id: str,
        project_id: str,
        *,
        target_type: str,
        target_id: str,
        author: str,
        body: str,
        metric_ref: str | None = None,
        step_ref: int | None = None,
    ) -> dict[str, tp.Any]:
        """Create and persist a team note attached to a run or experiment."""

        norm_org = _validate_required_text(org_id, field_name="org_id", max_len=_MAX_SCOPE_LEN)
        norm_project = _validate_required_text(project_id, field_name="project_id", max_len=_MAX_SCOPE_LEN)
        norm_target_type = validate_note_target_type(target_type)
        norm_target_id = _validate_required_text(target_id, field_name="target_id", max_len=_MAX_RUN_ID_LEN)
        norm_author = _validate_required_text(author, field_name="author", max_len=_MAX_SCOPE_LEN)
        norm_body = _validate_required_text(body, field_name="body", max_len=_MAX_NOTE_BODY_LEN)
        norm_metric = _validate_optional_text(metric_ref, field_name="metric_ref", max_len=_MAX_METRIC_KEY_LEN)
        norm_step = _validate_step_ref(step_ref)

        note_id = f"note_{uuid.uuid4().hex[:12]}"
        now = _utcnow_iso()
        with self._connection() as db:
            db.execute(
                """INSERT INTO notes
                   (id, org_id, project_id, target_type, target_id, author, body,
                    metric_ref, step_ref, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    note_id,
                    norm_org,
                    norm_project,
                    norm_target_type,
                    norm_target_id,
                    norm_author,
                    norm_body,
                    norm_metric,
                    norm_step,
                    now,
                ),
            )
            row = db.execute("SELECT * FROM notes WHERE id = ?", (note_id,)).fetchone()
            assert row is not None
            return _note_from_row(row)

    def get_note(self, org_id: str, project_id: str, note_id: str) -> dict[str, tp.Any] | None:
        """Return one note scoped to ``(org_id, project_id)``, or ``None``."""

        norm_org = _validate_required_text(org_id, field_name="org_id", max_len=_MAX_SCOPE_LEN)
        norm_project = _validate_required_text(project_id, field_name="project_id", max_len=_MAX_SCOPE_LEN)
        norm_note = _validate_required_text(note_id, field_name="note_id", max_len=_MAX_SCOPE_LEN)
        with self._connection() as db:
            row = db.execute(
                "SELECT * FROM notes WHERE id = ? AND org_id = ? AND project_id = ?",
                (norm_note, norm_org, norm_project),
            ).fetchone()
            return _note_from_row(row) if row is not None else None

    def list_notes(
        self,
        org_id: str,
        project_id: str,
        *,
        target_type: str | None = None,
        target_id: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, tp.Any]]:
        """List notes scoped to ``(org_id, project_id)``, newest first."""

        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise InputError("limit must be an integer between 1 and 1000")
        norm_org = _validate_required_text(org_id, field_name="org_id", max_len=_MAX_SCOPE_LEN)
        norm_project = _validate_required_text(project_id, field_name="project_id", max_len=_MAX_SCOPE_LEN)
        clauses = ["org_id = ?", "project_id = ?"]
        params: list[tp.Any] = [norm_org, norm_project]
        if target_type is not None:
            clauses.append("target_type = ?")
            params.append(validate_note_target_type(target_type))
        if target_id is not None:
            clauses.append("target_id = ?")
            params.append(_validate_required_text(target_id, field_name="target_id", max_len=_MAX_RUN_ID_LEN))
        sql = f"SELECT * FROM notes WHERE {' AND '.join(clauses)} ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        with self._connection() as db:
            rows = db.execute(sql, params).fetchall()
            return [_note_from_row(row) for row in rows]

    def delete_note(self, org_id: str, project_id: str, note_id: str) -> bool:
        """Delete one note scoped to ``(org_id, project_id)``."""

        norm_org = _validate_required_text(org_id, field_name="org_id", max_len=_MAX_SCOPE_LEN)
        norm_project = _validate_required_text(project_id, field_name="project_id", max_len=_MAX_SCOPE_LEN)
        norm_note = _validate_required_text(note_id, field_name="note_id", max_len=_MAX_SCOPE_LEN)
        with self._connection() as db:
            cursor = db.execute(
                "DELETE FROM notes WHERE id = ? AND org_id = ? AND project_id = ?",
                (norm_note, norm_org, norm_project),
            )
            return cursor.rowcount > 0

    def count_notes_by_target(self, org_id: str, project_id: str, target_type: str) -> dict[str, int]:
        """Return ``{target_id: count}`` for all notes of ``target_type`` in scope."""

        norm_org = _validate_required_text(org_id, field_name="org_id", max_len=_MAX_SCOPE_LEN)
        norm_project = _validate_required_text(project_id, field_name="project_id", max_len=_MAX_SCOPE_LEN)
        norm_target_type = validate_note_target_type(target_type)
        with self._connection() as db:
            rows = db.execute(
                """SELECT target_id, COUNT(*) AS cnt
                   FROM notes
                   WHERE org_id = ? AND project_id = ? AND target_type = ?
                   GROUP BY target_id""",
                (norm_org, norm_project, norm_target_type),
            ).fetchall()
            return {row["target_id"]: int(row["cnt"]) for row in rows}


__all__ = (
    "VALID_EXPERIMENT_KINDS",
    "VALID_GOALS",
    "VALID_NOTE_TARGETS",
    "VALID_VISIBILITIES",
    "CollaborationStore",
    "can_manage",
    "can_view",
    "normalize_subjects",
    "normalize_tags",
    "validate_experiment_kind",
    "validate_goal",
    "validate_note_target_type",
    "validate_visibility",
)
