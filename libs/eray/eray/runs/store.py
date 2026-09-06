# Copyright 2026 The EasyDeL/eray Author @erfanzar (Erfan Zare Chavoshi).
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

"""Run store (``~/.eray/runs.json``) and the single-watcher lease.

The lease is the load-bearing part: exactly one ``eray runs watch`` may
manage a store at a time. Two concurrent babysitters restarting each
other's jobs is not a hypothetical failure mode — it is precisely what
happens when a second hand-rolled watcher is started while a stale one is
alive — so the watcher refuses to start while a live holder exists, and a
dead holder's lease (pid gone, or expired) is stolen automatically.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import socket
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from .model import RunRecord

STATE_PATH = Path("~/.eray/runs.json").expanduser()
LEASE_PATH = Path("~/.eray/runs.lease").expanduser()

_DOC_VERSION = 1
LEASE_TTL_S = 120.0


class LeaseHeldError(RuntimeError):
    """Another live watcher holds the runs lease."""


def _atomic_write(path: Path, payload: dict) -> None:
    """Atomically replace ``path`` with JSON ``payload``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


class RunStore:
    """Flock-guarded JSON document of run records.

    Every read-modify-write goes through :meth:`_locked`, so a CLI
    ``stop``/``retry``/``add`` cannot be silently reverted by a watcher pass
    saving stale state on top of it. The watcher persists **per record**
    (:meth:`upsert`) rather than rewriting the whole document, so its writes
    only ever touch the record it just reconciled.

    Args:
        path: Override the store file (tests); defaults to
            :data:`STATE_PATH` read at call time so monkeypatching works.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        """The store file path (late-bound to honor monkeypatched defaults)."""
        return self._path or STATE_PATH

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold an exclusive advisory lock for a read-modify-write."""
        lock_path = self.path.with_suffix(".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _read(self) -> dict[str, RunRecord]:
        if not self.path.exists():
            return {}
        with open(self.path) as f:
            doc = json.load(f)
        return {name: RunRecord.from_dict(data) for name, data in (doc.get("runs") or {}).items()}

    def _write(self, records: dict[str, RunRecord]) -> None:
        _atomic_write(
            self.path,
            {"version": _DOC_VERSION, "runs": {name: rec.to_dict() for name, rec in sorted(records.items())}},
        )

    def load(self) -> dict[str, RunRecord]:
        """Load every run record (empty dict when the file is absent)."""
        with self._locked():
            return self._read()

    def save(self, records: dict[str, RunRecord]) -> None:
        """Persist the full record set atomically (prefer :meth:`upsert`)."""
        with self._locked():
            self._write(records)

    def upsert(self, record: RunRecord) -> None:
        """Insert or replace one record without touching the others."""
        with self._locked():
            records = self._read()
            records[record.spec.name] = record
            self._write(records)

    def mutate(self, name: str, fn: Callable[[RunRecord], None]) -> RunRecord:
        """Apply ``fn`` to one record under the lock and persist it.

        Args:
            name: Record name.
            fn: Mutator applied to the freshly-read record.

        Returns:
            The mutated record.

        Raises:
            KeyError: When the record does not exist.
        """
        with self._locked():
            records = self._read()
            if name not in records:
                raise KeyError(f"run {name!r} is not registered")
            fn(records[name])
            self._write(records)
            return records[name]

    def remove(self, name: str) -> bool:
        """Remove one record; True when it existed."""
        with self._locked():
            records = self._read()
            removed = records.pop(name, None) is not None
            if removed:
                self._write(records)
            return removed


def _pid_alive(pid: int) -> bool:
    """True when ``pid`` is a live process on this host."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


class Lease:
    """Single-host watcher lease file with TTL and dead-holder stealing.

    Args:
        path: Override the lease file (tests).
        ttl_s: Lease lifetime; :meth:`refresh` each tick keeps it alive.
    """

    def __init__(self, path: Path | None = None, ttl_s: float = LEASE_TTL_S) -> None:
        self._path = path
        self.ttl_s = ttl_s

    @property
    def path(self) -> Path:
        """The lease file path (late-bound like the store's)."""
        return self._path or LEASE_PATH

    def _read(self) -> dict | None:
        try:
            with open(self.path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def acquire(self, *, steal: bool = False, now: float | None = None) -> None:
        """Take the lease, or raise if a live holder exists.

        A holder is live when its lease has not expired AND (same host) its
        pid still exists. ``steal=True`` overrides — for a human who knows
        the holder is a zombie the heuristics missed.

        Raises:
            LeaseHeldError: When a live holder exists and ``steal`` is off.
        """
        now = time.time() if now is None else now
        current = self._read()
        if current and not steal:
            expired = float(current.get("expires", 0)) <= now
            same_host = current.get("host") == socket.gethostname()
            holder_alive = _pid_alive(int(current.get("pid", -1))) if same_host else True
            if not expired and holder_alive:
                raise LeaseHeldError(
                    f"runs watcher already running (pid {current.get('pid')} on {current.get('host')}, "
                    f"expires in {float(current.get('expires', 0)) - now:.0f}s); "
                    "stop it first or pass --steal"
                )
        self.refresh(now=now)

    def refresh(self, *, now: float | None = None) -> None:
        """Write/extend our ownership of the lease."""
        now = time.time() if now is None else now
        _atomic_write(
            self.path,
            {"pid": os.getpid(), "host": socket.gethostname(), "expires": now + self.ttl_s},
        )

    def release(self) -> None:
        """Drop the lease if we hold it."""
        current = self._read()
        if current and int(current.get("pid", -1)) == os.getpid():
            try:
                self.path.unlink()
            except FileNotFoundError:
                pass
