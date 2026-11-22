# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

"""A shipped, indexed store of pre-tuned kernel configurations.

Autotuning is expensive and its results were only ever kept per machine, in
``~/ejkernel-presistent-cache``, so every fresh TPU VM started cold. The one
kernel that shipped tuned values -- ragged_page_attention_v3 -- did it as a
hand-maintained 4000-line Python dict with a schema unique to itself, which is
not a thing anyone wants to write forty more times.

This is the same idea as a **table**, not a literal: one SQLite file for every
kernel, keyed by what actually changes the answer, read with an indexed lookup
instead of parsing a megabyte of Python at import.

    (kernel, device, dtypes, shape_key)  ->  platform + config

**Platform is stored next to config on purpose.** Which backend wins is a
measured fact that varies by shape -- XLA ``ragged_dot`` beats the Pallas
grouped matmul at MoE shapes while Pallas wins for paged attention -- and until
now that knowledge lived in hand-written escapes at call sites
(``moe_force_xla_gmm``, ``block_m % 8 != 0 -> xla``) rather than anywhere a
lookup could find it. Each row also keeps the runner-up and its timing, so
"is Pallas or XLA faster here" is answerable with evidence rather than folklore.

Design notes:

* **SQLite from the stdlib.** No dependency, one file, and a lookup touches a
  few pages instead of the whole table. Opened read-only and lazily, so importing
  ejkernel costs nothing.
* **Never hand-edited.** Rows come from ``ejkernel.ops.tuned`` sweeps; the CLI
  can dump the table to text for review and merge two databases.
* **Bucketed shape keys.** Sizes are rounded down to powers of two before they
  become part of the key, so a table stays small and a shape that was never
  measured still finds the nearest measured neighbour.
"""

from __future__ import annotations

import json
import sqlite3
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA_VERSION = 2

#: Ships inside the wheel next to this module.
DEFAULT_DB_NAME = "tuned_kernels.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tuned (
    kernel     TEXT NOT NULL,
    device     TEXT NOT NULL,
    dtypes     TEXT NOT NULL,
    shape_key  TEXT NOT NULL,
    platform   TEXT NOT NULL,
    config     TEXT NOT NULL,
    ms         REAL,
    runner_up  TEXT,
    baseline   TEXT,
    source_id  INTEGER,
    PRIMARY KEY (kernel, device, dtypes, shape_key)
);
CREATE INDEX IF NOT EXISTS tuned_by_kernel_device ON tuned (kernel, device, dtypes);
-- Provenance is identical across every row of a sweep, so it is stored once and
-- referenced. Inlining it cost ~35% of the file for no information.
CREATE TABLE IF NOT EXISTS sources (id INTEGER PRIMARY KEY, json TEXT UNIQUE);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def bucket(value: int, *, minimum: int = 1) -> int:
    """Round a size down to a power of two, so keys generalize.

    A table keyed on exact sizes only ever answers questions it was asked
    before. Bucketing means a shape between two measured points resolves to the
    lower neighbour, which is the safe direction: a config tuned for fewer rows
    under-fills its tiles rather than over-running them.

    Args:
        value: Size to bucket (non-positive values clamp to *minimum*).
        minimum: Smallest bucket to emit.

    Returns:
        The largest power of two that is <= *value*, at least *minimum*.
    """
    v = int(value)
    if v < minimum:
        return int(minimum)
    return 1 << (v.bit_length() - 1)


def dtype_signature(**dtypes: tp.Any) -> str:
    """Canonical, order-independent dtype key, e.g. ``"kv=int8,q=bfloat16"``.

    Args:
        **dtypes: Named dtypes participating in the kernel's choice.

    Returns:
        Sorted ``name=dtype`` pairs joined by commas.
    """

    def _name(d: tp.Any) -> str:
        return getattr(d, "name", None) or getattr(getattr(d, "dtype", None), "name", None) or str(d)

    return ",".join(f"{k}={_name(v)}" for k, v in sorted(dtypes.items()))


def shape_signature(**dims: int) -> str:
    """Canonical bucketed shape key, e.g. ``"e=256,k=4096,m=8192,n=2048"``.

    Args:
        **dims: Named dimensions. Each is bucketed to a power of two.

    Returns:
        Sorted ``name=bucket`` pairs joined by commas.
    """
    return ",".join(f"{k}={bucket(v)}" for k, v in sorted(dims.items()))


@dataclass(frozen=True)
class TunedEntry:
    """One measured winner.

    Attributes:
        kernel: Kernel/operation identifier (e.g. ``"grouped_matmul"``).
        device: Device kind as reported by JAX (e.g. ``"TPU v5p"``).
        dtypes: :func:`dtype_signature` output.
        shape_key: :func:`shape_signature` output.
        platform: Winning backend (``"pallas"``, ``"xla"``, ``"triton"``, ...).
        config: Winning configuration as a plain JSON-able dict.
        ms: Measured milliseconds for the winner, when known.
        runner_up: The next-best ``{"platform", "config", "ms"}``, when known.
            This is what makes a platform choice auditable instead of folklore.
        baseline: What the kernel would have used with NO table
            (``{"config", "ms"}``). Without it a row only says one candidate
            beat another, which cannot answer whether tuning beat not tuning --
            the only question that justifies storing the row at all.
        provenance: Free-form record of how the row was produced.
    """

    kernel: str
    device: str
    dtypes: str
    shape_key: str
    platform: str
    config: dict[str, tp.Any] = field(default_factory=dict)
    ms: float | None = None
    runner_up: dict[str, tp.Any] | None = None
    baseline: dict[str, tp.Any] | None = None
    provenance: dict[str, tp.Any] | None = None

    def speedup_over_default(self) -> float | None:
        """How much the tuned choice beat the untuned default, or ``None``."""
        if not self.baseline or not self.ms:
            return None
        other = self.baseline.get("ms")
        return float(other) / float(self.ms) if other else None

    def speedup_over_runner_up(self) -> float | None:
        """How much the winner beat the next-best option, or ``None``."""
        if not self.runner_up or not self.ms:
            return None
        other = self.runner_up.get("ms")
        if not other:
            return None
        return float(other) / float(self.ms)


def default_db_path() -> Path:
    """Path to the table shipped inside the installed package."""
    return Path(__file__).with_name(DEFAULT_DB_NAME)


class TunedStore:
    """Read-only, lazily opened view over a tuned-kernel database.

    Safe to construct at import time: no file is touched until the first
    :meth:`lookup`, and a missing database degrades to "no entries" rather than
    raising, so a source checkout without a generated table still runs.
    """

    def __init__(self, path: str | Path | None = None):
        self._path = Path(path) if path is not None else default_db_path()
        self._conn: sqlite3.Connection | None = None
        self._checked = False
        self._available = False

    @property
    def path(self) -> Path:
        return self._path

    def available(self) -> bool:
        """Whether a readable table exists (does not raise)."""
        self._connect()
        return self._available

    def _connect(self) -> sqlite3.Connection | None:
        if self._checked:
            return self._conn
        self._checked = True
        try:
            if not self._path.exists():
                return None
            conn = sqlite3.connect(f"file:{self._path}?mode=ro", uri=True, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._conn = conn
            self._available = True
        except Exception:
            # A corrupt or unreadable table must never break a kernel call; the
            # caller falls through to autotune or its own default.
            self._conn = None
            self._available = False
        return self._conn

    def lookup(
        self,
        kernel: str,
        device: str,
        dtypes: str,
        shape_key: str,
        *,
        allow_nearest: bool = True,
    ) -> TunedEntry | None:
        """Find the tuned entry for a call, or ``None``.

        Tries the exact key first. With *allow_nearest*, falls back to the
        closest measured shape for the same (kernel, device, dtypes) — closest
        by agreeing on the most leading dimensions, then by smallest total
        distance in log space, preferring smaller measured shapes on ties.

        Args:
            kernel: Kernel identifier.
            device: Device kind.
            dtypes: :func:`dtype_signature` output.
            shape_key: :func:`shape_signature` output.
            allow_nearest: Whether to fall back to a neighbouring shape.

        Returns:
            The entry, or ``None`` when nothing is tabulated.
        """
        conn = self._connect()
        if conn is None:
            return None
        try:
            row = conn.execute(
                "SELECT * FROM tuned WHERE kernel=? AND device=? AND dtypes=? AND shape_key=?",
                (kernel, device, dtypes, shape_key),
            ).fetchone()
            if row is not None:
                return _row_to_entry(row, self._sources().get(row["source_id"]))
            if not allow_nearest:
                return None
            rows = conn.execute(
                "SELECT * FROM tuned WHERE kernel=? AND device=? AND dtypes=?",
                (kernel, device, dtypes),
            ).fetchall()
        except Exception:
            return None
        if not rows:
            return None
        want = _parse_shape(shape_key)
        best, best_score = None, None
        for row in rows:
            have = _parse_shape(row["shape_key"])
            if set(have) != set(want):
                continue
            score = _shape_distance(want, have)
            if best_score is None or score < best_score:
                best, best_score = row, score
        return _row_to_entry(best, self._sources().get(best["source_id"])) if best is not None else None

    def _sources(self) -> dict[int, dict]:
        conn = self._connect()
        if conn is None:
            return {}
        try:
            return {r[0]: json.loads(r[1]) for r in conn.execute("SELECT id, json FROM sources")}
        except Exception:
            return {}

    def entries(self, kernel: str | None = None) -> list[TunedEntry]:
        """All entries, optionally restricted to one kernel."""
        conn = self._connect()
        if conn is None:
            return []
        sql = "SELECT * FROM tuned"
        args: tuple = ()
        if kernel:
            sql += " WHERE kernel=?"
            args = (kernel,)
        sql += " ORDER BY kernel, device, dtypes, shape_key"
        try:
            sources = self._sources()
            return [_row_to_entry(r, sources.get(r["source_id"])) for r in conn.execute(sql, args)]
        except Exception:
            return []

    def kernels(self) -> list[str]:
        """Kernel identifiers present in the table."""
        conn = self._connect()
        if conn is None:
            return []
        try:
            return [r[0] for r in conn.execute("SELECT DISTINCT kernel FROM tuned ORDER BY kernel")]
        except Exception:
            return []

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
        self._conn = None
        self._checked = False
        self._available = False


def _has(row: sqlite3.Row, key: str) -> bool:
    """Whether a (possibly older-schema) row carries *key* with a value."""
    try:
        return row[key] is not None
    except (IndexError, KeyError):
        return False


def _row_to_entry(row: sqlite3.Row, provenance: dict | None = None) -> TunedEntry:
    return TunedEntry(
        kernel=row["kernel"],
        device=row["device"],
        dtypes=row["dtypes"],
        shape_key=row["shape_key"],
        platform=row["platform"],
        config=json.loads(row["config"]) if row["config"] else {},
        ms=row["ms"],
        runner_up=json.loads(row["runner_up"]) if row["runner_up"] else None,
        baseline=json.loads(row["baseline"]) if _has(row, "baseline") else None,
        provenance=provenance,
    )


def _parse_shape(shape_key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for part in shape_key.split(","):
        if not part:
            continue
        name, _, value = part.partition("=")
        try:
            out[name] = int(value)
        except ValueError:
            continue
    return out


def _shape_distance(want: dict[str, int], have: dict[str, int]) -> tuple[int, float]:
    """Distance in log space; smaller is closer, ties prefer smaller shapes."""
    total = 0.0
    for name, target in want.items():
        other = have.get(name, 1)
        total += abs(int(max(1, target)).bit_length() - int(max(1, other)).bit_length())
    mismatched = sum(1 for name, target in want.items() if have.get(name) != target)
    return (mismatched, total)


# --- writing -----------------------------------------------------------------


def open_for_write(path: str | Path) -> sqlite3.Connection:
    """Create/open a tuned database for writing and ensure its schema."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA)
    # Older tables predate the baseline column; add it rather than forcing a
    # regeneration of rows that are still perfectly good.
    cols = {r[1] for r in conn.execute("PRAGMA table_info(tuned)")}
    if "baseline" not in cols:
        conn.execute("ALTER TABLE tuned ADD COLUMN baseline TEXT")
    conn.execute(
        "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()
    return conn


def upsert(conn: sqlite3.Connection, entries: tp.Iterable[TunedEntry]) -> int:
    """Insert or replace *entries*; returns how many rows were written."""
    entries = list(entries)
    if not entries:
        return 0
    source_ids: dict[str, int] = {}
    for e in entries:
        if not e.provenance:
            continue
        blob = json.dumps(e.provenance, sort_keys=True)
        if blob in source_ids:
            continue
        conn.execute("INSERT OR IGNORE INTO sources (json) VALUES (?)", (blob,))
        row = conn.execute("SELECT id FROM sources WHERE json=?", (blob,)).fetchone()
        source_ids[blob] = int(row[0])
    rows = [
        (
            e.kernel,
            e.device,
            e.dtypes,
            e.shape_key,
            e.platform,
            json.dumps(e.config, sort_keys=True),
            e.ms,
            json.dumps(e.runner_up, sort_keys=True) if e.runner_up else None,
            json.dumps(e.baseline, sort_keys=True) if e.baseline else None,
            source_ids.get(json.dumps(e.provenance, sort_keys=True)) if e.provenance else None,
        )
        for e in entries
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO tuned "
        "(kernel, device, dtypes, shape_key, platform, config, ms, runner_up, baseline, source_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return len(rows)


def merge(dst: str | Path, *sources: str | Path, prefer_faster: bool = True) -> int:
    """Merge tuned databases into *dst*.

    Sweeps run on different machines and dates, so merging has to be defined.
    With *prefer_faster* an incoming row only replaces an existing one when it
    measured faster; otherwise last-writer-wins. Rows with no timing never
    displace a row that has one.

    Args:
        dst: Database to merge into (created when absent).
        *sources: Databases to merge from.
        prefer_faster: Keep the better-measured row on conflict.

    Returns:
        Number of rows written to *dst*.
    """
    conn = open_for_write(dst)
    existing = {(e.kernel, e.device, e.dtypes, e.shape_key): e for e in TunedStore(dst).entries()}
    incoming: list[TunedEntry] = []
    for src in sources:
        for entry in TunedStore(src).entries():
            key = (entry.kernel, entry.device, entry.dtypes, entry.shape_key)
            current = existing.get(key)
            if current is not None and prefer_faster:
                if entry.ms is None:
                    continue
                if current.ms is not None and current.ms <= entry.ms:
                    continue
            existing[key] = entry
            incoming.append(entry)
    written = upsert(conn, incoming)
    conn.close()
    return written
