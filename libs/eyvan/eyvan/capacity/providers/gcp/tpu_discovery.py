# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Read-only inventory of every TPU VM the operator's gcloud identity can see.

Discovery only ever issues GETs: it lists projects, lists each project's TPU
nodes, and writes a local owner-only index. It never SSHes into a host, adds a
tag, opens a firewall rule, or touches a fleet. Connecting a discovered TPU is a
separate, explicit step (``eyvan gcp connect``, or a first submission to its
fleet name), so a listing can never install software on somebody's machine.

The index is what makes a fleet name usable before anything is connected: the
name is derived deterministically from ``project/zone/name`` (the same
``attach_fleet_name`` connect uses), so the name printed by discovery is the
name the connected fleet will have.
"""

from __future__ import annotations

import json
import os
import stat
import threading
import time
import typing as tp
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from eyvan._internal.workspace import create_private_directory
from eyvan.capacity.providers.gcp.tpu_nodes import (
    TpuApiError,
    TpuNode,
    TpuNodeApi,
    TpuNodeParseError,
    attach_fleet_name,
)

#: Index format; bumped only for an incompatible change.
INDEX_VERSION = 1
#: How long an index answers "which TPUs exist" before a caller should refresh.
DEFAULT_MAX_AGE_S = 600.0
#: Projects listed at once. Each list is one or a few HTTP GETs.
DEFAULT_CONCURRENCY = 8
_MAX_INDEX_BYTES = 4 * 1024 * 1024


def default_index_path() -> Path:
    """Return ``~/.eyvan/attached/discovered.json`` beside the connected fleet configs."""

    return Path("~/.eyvan/attached/discovered.json").expanduser()


@dataclass(frozen=True, slots=True)
class DiscoveredTpu:
    """One TPU VM as discovery saw it, with the fleet name it is used by."""

    fleet: str
    project: str
    zone: str
    name: str
    node_id: str
    accelerator_type: str
    capacity_class: str
    state: str
    hosts: int

    @classmethod
    def from_node(cls, node: TpuNode) -> DiscoveredTpu:
        return cls(
            fleet=attach_fleet_name(project=node.project, zone=node.zone, name=node.name),
            project=node.project,
            zone=node.zone,
            name=node.name,
            node_id=node.node_id,
            accelerator_type=node.accelerator_type,
            capacity_class=node.capacity_class.value,
            state=node.state,
            hosts=len(node.hosts),
        )


@dataclass(frozen=True, slots=True)
class DiscoverySnapshot:
    """Everything one discovery pass saw, including what it could not read.

    ``skipped`` holds projects whose TPU inventory could not be read, with the
    reason. A project that was skipped is *unknown*, never "has no TPUs".
    """

    discovered_at: float
    tpus: tuple[DiscoveredTpu, ...] = ()
    skipped: dict[str, str] = field(default_factory=dict)
    projects: tuple[str, ...] = ()
    #: The gcloud account the listing ran as; ``None`` when it could not be named.
    account: str | None = None

    def age_s(self, now: float | None = None) -> float:
        return max(0.0, (time.time() if now is None else now) - self.discovered_at)

    def resolve(self, name: str) -> DiscoveredTpu | None:
        """Return the TPU a fleet name or a TPU node name refers to.

        A fleet name is unique by construction. A bare node name is accepted only
        when exactly one discovered TPU carries it; the same name in two
        projects or zones is not guessed between.
        """

        by_fleet = [tpu for tpu in self.tpus if tpu.fleet == name]
        if by_fleet:
            return by_fleet[0]
        by_name = [tpu for tpu in self.tpus if tpu.name == name]
        return by_name[0] if len(by_name) == 1 else None

    def ambiguous(self, name: str) -> tuple[DiscoveredTpu, ...]:
        """Return every TPU sharing node name ``name`` when there is more than one."""

        matches = tuple(tpu for tpu in self.tpus if tpu.name == name)
        return matches if len(matches) > 1 else ()

    def as_document(self) -> dict[str, tp.Any]:
        return {
            "version": INDEX_VERSION,
            "discovered_at": self.discovered_at,
            "projects": list(self.projects),
            "tpus": [asdict(tpu) for tpu in self.tpus],
            "skipped": dict(self.skipped),
            "account": self.account,
        }

    @classmethod
    def from_document(cls, document: object) -> DiscoverySnapshot:
        if not isinstance(document, dict) or document.get("version") != INDEX_VERSION:
            raise ValueError("unsupported TPU discovery index")
        raw = document.get("tpus")
        if not isinstance(raw, list):
            raise ValueError("TPU discovery index has no TPU list")
        tpus = tuple(DiscoveredTpu(**item) for item in raw if isinstance(item, dict))
        skipped = document.get("skipped")
        projects = document.get("projects")
        discovered_at = document.get("discovered_at")
        if not isinstance(discovered_at, (int, float)):
            raise ValueError("TPU discovery index has no timestamp")
        account = document.get("account")
        return cls(
            account=account if isinstance(account, str) and account else None,
            discovered_at=float(discovered_at),
            tpus=tpus,
            skipped={str(k): str(v) for k, v in skipped.items()} if isinstance(skipped, dict) else {},
            projects=tuple(str(p) for p in projects) if isinstance(projects, list) else (),
        )


def _skip_reason(exc: Exception) -> str:
    text = str(exc)
    if "has not been used" in text or "is disabled" in text or "SERVICE_DISABLED" in text:
        return "Cloud TPU API is not enabled"
    if isinstance(exc, TpuApiError) and exc.status == 403:
        return "no permission to list TPU nodes"
    return text[:300]


def discover(
    api: TpuNodeApi | None = None,
    *,
    projects: tp.Sequence[str] | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
    clock: tp.Callable[[], float] = time.time,
) -> DiscoverySnapshot:
    """List every visible project's TPU nodes. Read-only: GETs only.

    A project whose inventory cannot be read is recorded in ``skipped`` and the
    rest are still listed; failing to list the projects themselves raises,
    since then nothing at all is known.
    """

    api = api if api is not None else TpuNodeApi()
    project_ids = tuple(projects) if projects is not None else tuple(p.project_id for p in api.list_projects())

    def list_one(project: str) -> tuple[str, tuple[TpuNode, ...] | None, str | None]:
        try:
            nodes, unreachable = api.list_nodes(project)
        except (TpuApiError, TpuNodeParseError, ValueError) as exc:
            return project, None, _skip_reason(exc)
        note = f"unreachable locations: {', '.join(unreachable)}" if unreachable else None
        return project, nodes, note

    tpus: list[DiscoveredTpu] = []
    skipped: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(project_ids) or 1))) as pool:
        for project, nodes, note in pool.map(list_one, project_ids):
            if nodes is None:
                skipped[project] = note or "unreadable"
                continue
            if note:
                # Partial: some zones could not be asked. What was seen is kept;
                # the gap is recorded so "no TPU here" is never claimed for it.
                skipped[project] = note
            tpus.extend(DiscoveredTpu.from_node(node) for node in nodes)
    tpus.sort(key=lambda tpu: (tpu.project, tpu.zone, tpu.name))
    # Looked up on the type so a reader without an identity (a test double)
    # is simply "unknown account" rather than asked for an attribute it lacks.
    account = api.account() if callable(getattr(type(api), "account", None)) else None
    return DiscoverySnapshot(
        discovered_at=clock(), tpus=tuple(tpus), skipped=skipped, projects=project_ids, account=account
    )


@dataclass
class DiscoveryIndex:
    """Owner-only, atomically replaced JSON index of the last discovery pass."""

    path: Path = field(default_factory=default_index_path)
    #: Reader used when a refresh is not handed one; injected by tests.
    api: TpuNodeApi | None = None

    def load(self) -> DiscoverySnapshot | None:
        """Return the saved snapshot, or ``None`` when absent or unreadable."""

        try:
            descriptor = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        except OSError:
            return None
        try:
            with os.fdopen(descriptor, "r") as stream:
                metadata = os.fstat(stream.fileno())
                if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.getuid():
                    return None
                text = stream.read(_MAX_INDEX_BYTES + 1)
            if len(text) > _MAX_INDEX_BYTES:
                return None
            return DiscoverySnapshot.from_document(json.loads(text))
        except (OSError, ValueError, TypeError):
            return None

    def save(self, snapshot: DiscoverySnapshot) -> None:
        create_private_directory(self.path.parent)
        temporary = self.path.with_name(f".{self.path.name}-{uuid.uuid4().hex}")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            with os.fdopen(descriptor, "w") as stream:
                json.dump(snapshot.as_document(), stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def refresh(self, api: TpuNodeApi | None = None, **kwargs: tp.Any) -> DiscoverySnapshot:
        """Run discovery and save it; the saved index is replaced only on success."""

        snapshot = discover(api if api is not None else self.api, **kwargs)
        self.save(snapshot)
        return snapshot

    def fresh(self, *, max_age_s: float = DEFAULT_MAX_AGE_S, api: TpuNodeApi | None = None) -> DiscoverySnapshot:
        """Return the saved snapshot when younger than ``max_age_s``, else refresh it."""

        saved = self.load()
        if saved is not None and saved.age_s() <= max_age_s:
            return saved
        return self.refresh(api)


def refresh_in_background(
    index: DiscoveryIndex | None = None, *, max_age_s: float = DEFAULT_MAX_AGE_S
) -> threading.Thread | None:
    """Refresh a stale index on a daemon thread; never blocks or fails the caller.

    Used where Eyvan is opened, so the fleet names of every TPU in the account
    are already listed by the time anyone asks. A fresh index is left alone, and
    a failed refresh keeps the previous one: an unreadable account is unknown,
    not empty.
    """

    index = index if index is not None else DiscoveryIndex()
    saved = index.load()
    if saved is not None and saved.age_s() <= max_age_s:
        return None

    def run() -> None:
        try:
            index.refresh()
        except Exception:  # best effort; `eyvan gcp sync` reports the reason
            return

    thread = threading.Thread(target=run, name="eyvan-tpu-discovery", daemon=True)
    thread.start()
    return thread


#: How often a running local plane re-lists the account's TPUs.
DEFAULT_REFRESH_INTERVAL_S = 300.0


class PeriodicDiscovery:
    """Re-list every visible TPU on a fixed interval, for as long as a plane runs.

    A daemon thread, not an event-loop task: a refresh is a few blocking HTTP
    GETs, and neither serving requests nor shutting down should wait on them.
    A failed pass keeps the previous index (``refresh`` only replaces it on
    success) and is retried at the next interval.
    """

    def __init__(
        self,
        index: DiscoveryIndex | None = None,
        *,
        interval_s: float = DEFAULT_REFRESH_INTERVAL_S,
        on_error: tp.Callable[[Exception], None] | None = None,
    ) -> None:
        if not interval_s > 0:
            raise ValueError("interval_s must be positive")
        self.index = index if index is not None else DiscoveryIndex()
        self.interval_s = interval_s
        self.on_error = on_error
        self.passes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="eyvan-tpu-discovery-periodic", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = 0.0) -> None:
        """Ask the loop to end; never waits on an in-flight GCP request by default."""

        self._stop.set()
        if self._thread is not None and timeout:
            self._thread.join(timeout)

    def _run(self) -> None:
        # The first pass runs at once, so a freshly started plane lists TPUs
        # without waiting a whole interval.
        while not self._stop.is_set():
            try:
                self.index.refresh()
            except Exception as exc:  # best effort; the previous index stays
                if self.on_error is not None:
                    self.on_error(exc)
            self.passes += 1
            if self._stop.wait(self.interval_s):
                return


__all__ = (
    "DEFAULT_MAX_AGE_S",
    "DEFAULT_REFRESH_INTERVAL_S",
    "DiscoveredTpu",
    "DiscoveryIndex",
    "DiscoverySnapshot",
    "PeriodicDiscovery",
    "default_index_path",
    "discover",
    "refresh_in_background",
)
