# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Owner-only supervision of the optional, rebuildable local run tracker."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import signal
import stat
import subprocess
import sys
import tempfile
import time
import typing as tp
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from eyvan.server.local import daemon
from eyvan.track._store import store_for_uri

DEFAULT_TRACKER_DIRECTORY = Path("~/.eyvan/tracker")
DEFAULT_TRACKING_STORE = Path("~/.eyvan/tracking")


class LocalTrackerError(RuntimeError):
    """Local tracker configuration, ownership or readiness could not be established."""


@dataclass(frozen=True, slots=True)
class TrackerRecord:
    """Non-secret process metadata; the bearer credential lives in a separate file."""

    url: str
    pid: int
    store: str
    db: str
    log: str
    token_path: str

    def as_json(self) -> dict[str, str | int]:
        """Return the stable on-disk record shape, without the token itself."""

        return asdict(self)


def _directory(directory: Path | None) -> Path:
    return (directory or DEFAULT_TRACKER_DIRECTORY).expanduser().absolute()


def _require_owned_directory(directory: Path) -> os.stat_result:
    metadata = directory.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or metadata.st_uid != os.geteuid():
        raise LocalTrackerError("tracker directory must be an owned directory, not a symlink")
    return metadata


def _require_private_directory(directory: Path) -> None:
    metadata = _require_owned_directory(directory)
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise LocalTrackerError("tracker directory must be an owner-only directory, not a symlink")


def _database_path(directory: Path, store: str) -> Path:
    """Keep each canonical store's rebuildable index separate, including after down."""

    digest = hashlib.sha256(store.encode("utf-8")).hexdigest()
    return directory / f"index-{digest}.sqlite3"


def read_tracker_record(directory: Path | None = None) -> TrackerRecord | None:
    """Read a private, well-formed loopback record; absent or stale JSON is not live."""

    root = _directory(directory)
    if not root.exists() and not root.is_symlink():
        return None
    _require_private_directory(root)
    path = root / "record.json"
    if not path.exists() and not path.is_symlink():
        return None
    daemon.require_owner_only_record_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or set(value) != {"url", "pid", "store", "db", "log", "token_path"}:
            return None
        record = TrackerRecord(**value)
        if (
            isinstance(record.pid, bool)
            or not isinstance(record.pid, int)
            or record.pid <= 0
            or any(
                not isinstance(item, str)
                for item in (record.url, record.store, record.db, record.log, record.token_path)
            )
        ):
            return None
        parsed = urlsplit(record.url)
        if (
            parsed.scheme != "http"
            or parsed.hostname != "127.0.0.1"
            or parsed.port is None
            or not 1 <= parsed.port <= 65535
            or parsed.path
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
        ):
            return None
        if (
            # Accept legacy records so an existing owned process can still be
            # inspected/stopped. New launches never reuse that unbound index.
            record.db not in {str(root / "index.sqlite3"), str(_database_path(root, record.store))}
            or record.log != str(root / "tracker.log")
            or record.token_path != str(root / "token")
        ):
            return None
        return record
    except (ValueError, TypeError):
        return None


def read_tracker_token(record: TrackerRecord) -> str:
    """Read an owner-only bearer token without including it in diagnostics."""

    path = Path(record.token_path)
    _require_private_directory(path.parent)
    daemon.require_owner_only_record_file(path)
    token = path.read_text(encoding="utf-8").strip()
    if not token or len(token) > 4096 or any(character.isspace() for character in token):
        raise LocalTrackerError("tracker token file is invalid; restart with `eyvan track up --local`")
    return token


def _write_private(path: Path, text: str) -> None:
    if path.exists() or path.is_symlink():
        daemon.require_owner_only_record_file(path)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def tracker_health(url: str) -> bool:
    """Recognize only a healthy tracker, without redirects or environment proxies."""

    try:
        with httpx.Client(timeout=0.5, trust_env=False, follow_redirects=False) as client:
            response = client.get(f"{url}/api/v1/tracking/health")
        payload = response.json()
        return (
            response.is_success
            and isinstance(payload, dict)
            and payload.get("status") == "ok"
            and payload.get("service") == "eyvan-tracker"
        )
    except (httpx.HTTPError, ValueError):
        return False


def launch_tracker(argv: tuple[str, ...], environment: dict[str, str], log: Path) -> int:
    """Launch this installation detached, with the token only in its environment."""

    with daemon.open_owner_only_log(log) as handle:
        child = subprocess.Popen(
            argv,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    return child.pid


def _argv(record: TrackerRecord) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "eyvan.track.tracker",
        "--store",
        record.store,
        "--db",
        record.db,
        "--host",
        "127.0.0.1",
        "--port",
        str(urlsplit(record.url).port),
    )


class LocalTrackerSupervisor:
    """Serialize tracker startup and stop only a process with verified exact argv.

    Launch, health, process inspection, signalling and clock boundaries are
    injectable so tests need not start a server or signal an actual process.
    """

    def __init__(
        self,
        *,
        directory: Path | None = None,
        launcher: tp.Callable[[tuple[str, ...], dict[str, str], Path], int] = launch_tracker,
        health_probe: tp.Callable[[str], bool] = tracker_health,
        process_liveness: tp.Callable[[int], bool] = daemon.process_is_alive,
        argument_reader: tp.Callable[[int], tuple[str, ...] | None] = daemon._read_process_arguments,
        process_terminator: tp.Callable[[int], None] | None = None,
        process_killer: tp.Callable[[int], None] = daemon.force_kill_process,
        clock: tp.Callable[[], float] = time.monotonic,
        sleeper: tp.Callable[[float], None] = time.sleep,
    ) -> None:
        self.directory = _directory(directory)
        self.launcher = launcher
        self.health_probe = health_probe
        self.process_liveness = process_liveness
        self.argument_reader = argument_reader
        self.process_terminator = process_terminator or (lambda pid: os.kill(pid, signal.SIGTERM))
        self.process_killer = process_killer
        self.clock = clock
        self.sleeper = sleeper

    def _matches(self, record: TrackerRecord) -> bool:
        arguments = self.argument_reader(record.pid)
        return arguments is not None and arguments[1:] == _argv(record)[1:]

    def status(self) -> dict[str, object]:
        """Return observed state and non-secret metadata, never infer health from a PID."""

        record = read_tracker_record(self.directory)
        if record is None:
            return {"status": "stopped", "record": None}
        if not self.process_liveness(record.pid):
            state = "stopped"
        elif not self._matches(record):
            state = "unknown"
        else:
            state = "running" if self.health_probe(record.url) else "unreachable"
        return {"status": state, "record": record.as_json()}

    def up(self, *, store_uri: str | None = None, port: int = 8095, timeout_s: float = 15.0) -> TrackerRecord:
        """Start one tracker, or reuse the already healthy owned process."""

        if not 1 <= port <= 65535:
            raise LocalTrackerError("tracker port must be between 1 and 65535")
        store = store_for_uri(store_uri or str(DEFAULT_TRACKING_STORE)).root
        # Preparation tightens permissions using chmod; reject existing symlinks
        # and foreign ownership before it can modify an unrelated target.
        if self.directory.exists() or self.directory.is_symlink():
            _require_owned_directory(self.directory)
        daemon.prepare_owner_only_directory(self.directory)
        _require_private_directory(self.directory)
        with daemon.owner_only_startup_lock(self.directory / "startup.lock"):
            record = read_tracker_record(self.directory)
            if record is not None and self.process_liveness(record.pid):
                if not self._matches(record):
                    raise LocalTrackerError("recorded tracker process identity changed; refusing to replace it")
                if record.store != store or urlsplit(record.url).port != port:
                    raise LocalTrackerError("a tracker with different settings is running; run `eyvan track down` first")
                read_tracker_token(record)
                if self.health_probe(record.url):
                    return record
                raise LocalTrackerError("owned tracker health check failed; inspect its log or run `eyvan track down`")
            url = f"http://127.0.0.1:{port}"
            if self.health_probe(url):
                raise LocalTrackerError("tracker port already serves an unowned tracker; choose another --port")
            token_path = self.directory / "token"
            token = secrets.token_urlsafe(32)
            _write_private(token_path, token + "\n")
            draft = TrackerRecord(
                url=url,
                pid=1,
                store=store,
                db=str(_database_path(self.directory, store)),
                log=str(self.directory / "tracker.log"),
                token_path=str(token_path),
            )
            environment = dict(os.environ)
            environment["EYVAN_TRACKER_TOKEN"] = token
            pid = self.launcher(_argv(draft), environment, Path(draft.log))
            record = TrackerRecord(
                url=draft.url, pid=pid, store=draft.store, db=draft.db, log=draft.log, token_path=draft.token_path
            )
            try:
                _write_private(self.directory / "record.json", json.dumps(record.as_json(), sort_keys=True) + "\n")
                deadline = self.clock() + timeout_s
                while self.process_liveness(pid):
                    if self.health_probe(url):
                        return record
                    remaining = deadline - self.clock()
                    if remaining <= 0:
                        break
                    self.sleeper(min(0.1, remaining))
                raise LocalTrackerError(f"tracker health check failed at {url}; inspect {record.log}")
            except BaseException:
                self._stop(record)
                raise

    def _stop(self, record: TrackerRecord) -> None:
        if self.process_liveness(record.pid):
            if not self._matches(record):
                raise LocalTrackerError("recorded tracker process identity changed; refusing to signal it")
            outcome = daemon.stop_verified_process(
                record.pid,
                process_liveness=self.process_liveness,
                process_verifier=lambda pid: self._matches(record),
                process_terminator=self.process_terminator,
                process_killer=self.process_killer,
                graceful_timeout_s=5.0,
                forced_timeout_s=2.0,
                poll_interval_s=0.1,
                clock=self.clock,
                sleeper=self.sleeper,
            )
            if outcome != daemon.VerifiedProcessStopOutcome.STOPPED:
                raise LocalTrackerError(f"tracker stop could not be verified: {outcome}")
        for path in (self.directory / "record.json", Path(record.token_path)):
            if path.exists() or path.is_symlink():
                daemon.require_owner_only_record_file(path)
                path.unlink()

    def down(self) -> None:
        """Stop the exact owned process and remove only its metadata and credential."""

        if not self.directory.exists():
            return
        _require_private_directory(self.directory)
        with daemon.owner_only_startup_lock(self.directory / "startup.lock"):
            record = read_tracker_record(self.directory)
            if record is not None:
                self._stop(record)
