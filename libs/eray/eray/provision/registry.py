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


"""Fleet registry: the shared memory of eray-managed clusters.

One versioned JSON document holding every registered cluster's desired state,
identity (stable node id, current queued-resource id + generation), and the
watcher's persisted FSM state. Two backends behind one optimistic-concurrency
interface:

- **GCS** (``gs://bucket/path/clusters.json``): the multi-operator backend.
  Reads pair ``gcloud storage cat`` with the object generation; writes use
  ``gcloud storage cp --if-generation-match`` so concurrent writers conflict
  instead of clobbering (compare-and-swap; :class:`ConflictError` → retry).
- **Local file** (``~/.eray/clusters.json``): default when no GCS URI is
  configured; ``fcntl.flock`` held across each read-modify-write.

A fleet-level *lease* marks the active watcher (holder + expiry, renewed each
tick); expired leases are stealable, so a crashed watcher never wedges the
fleet.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..gcp import gcloud, gcloud_json

SCHEMA_VERSION = 1
CONFIG_PATH = Path("~/.eray/config.json").expanduser()
DEFAULT_LOCAL_PATH = Path("~/.eray/clusters.json").expanduser()
LEASE_TTL_S = 120


class ConflictError(RuntimeError):
    """Another writer updated the registry between our read and write."""


def _empty_doc() -> dict:
    """A fresh registry document."""
    return {"version": SCHEMA_VERSION, "clusters": {}, "lease": None}


@dataclass
class ClusterRecord:
    """One managed cluster's identity, desired state, and watcher state.

    Attributes:
        name: Registry key; also the stable TPU node id (survives re-queues,
            so ``eray tpu connect -n <name>`` always resolves).
        kind: ``"qr"`` (queued-resource managed slice) or ``"launcher"``
            (Ray cluster-launcher config).
        project: GCP project id.
        zone: GCP zone.
        accelerator_type: e.g. ``"v5p-64"`` (qr kind).
        runtime_version: TPU runtime version override, or None for the
            generation default.
        capacity: ``"spot"`` / ``"on-demand"`` / ``"reserved"`` / ``"guaranteed"``.
        qr_id: Current queued-resource id (``{name}-r{generation}`` once the
            watcher has re-queued at least once; initially ``name`` when
            adopting a hand-created QR).
        generation: Re-queue counter; incremented on every capacity recreate.
        desired_state: ``"up"`` or ``"down"``.
        bootstrap_cmd: Opaque shell command fanned to all hosts before the
            first connect of each generation (None: hosts assumed ready).
        bootstrapped_generation: Last generation the bootstrap ran on.
        head_ip: Last known head internal IP.
        state: Watcher FSM state (``HEALTHY``, ``DEGRADED``, ``WAITING``, ...).
        intent: Write-ahead intent (``{"action", "target", "ts"}``) recorded
            before every mutating gcloud call; cleared after. Crash recovery
            re-describes the deterministic target to learn whether the action
            landed.
        job_snapshot: Last-healthy jobs snapshot (list of dicts) used for
            resubmission after preemption.
        recreate_ts: Bounded ring of recent recreate unix timestamps —
            budget windows survive watcher failover because they live here.
        config_path: Cluster-launcher YAML path (launcher kind).
        last_up_ts: Unix timestamp of the last successful bring-up (stamped
            by `eray autoscale up`; QR-kind clusters brought up via `eray
            fleet up`/`ensure` do not set this). None if never brought up.
        last_down_ts: Unix timestamp of the last successful teardown (`eray
            autoscale down`). None if never torn down. For launcher-kind
            clusters this is the only record of "when did it die" — `ray
            down` deletes the GCE instances outright, so GCP has nothing
            left to query after the fact.
        extra: Forward-compatible bag for fields newer eray versions add.
    """

    name: str
    kind: str = "qr"
    project: str | None = None
    zone: str | None = None
    accelerator_type: str | None = None
    runtime_version: str | None = None
    capacity: str = "spot"
    qr_id: str | None = None
    generation: int = 0
    desired_state: str = "up"
    bootstrap_cmd: str | None = None
    bootstrapped_generation: int = -1
    head_ip: str | None = None
    state: str = "UNKNOWN"
    intent: dict | None = None
    job_snapshot: list | None = None
    recreate_ts: list = field(default_factory=list)
    config_path: str | None = None
    last_up_ts: float | None = None
    last_down_ts: float | None = None
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize for the registry document."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ClusterRecord:
        """Deserialize tolerantly: unknown keys land in ``extra``.

        Args:
            data: A cluster entry from the registry document.

        Returns:
            The record; fields written by newer eray versions are preserved
            round-trip via ``extra``.
        """
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in data.items() if k in known}
        unknown = {k: v for k, v in data.items() if k not in known}
        record = cls(**kwargs)
        if unknown:
            record.extra = {**record.extra, **unknown}
        return record

    def next_qr_id(self) -> str:
        """The deterministic QR id for the next generation."""
        return f"{self.name}-r{self.generation + 1}"


class LocalBackend:
    """Local-file JSON document storage guarded by flock, atomic-written.

    The lock is held across the entire read-modify-write, so local updates
    never conflict; ConflictError is not raised by this backend. Storage
    is generic over the document shape via `empty_doc` — the fleet registry
    uses this with its own `_empty_doc` (the default), but any other
    eray module wanting a safe local JSON store (e.g. `provision.tunnel`'s
    tracked-process table) can reuse this instead of re-deriving the same
    flock + atomic-rename primitive.
    """

    def __init__(self, path: Path | str = DEFAULT_LOCAL_PATH, *, empty_doc: Callable[[], dict] | None = None):
        """Initialize the backend.

        Args:
            path: Storage file path (created on first write).
            empty_doc: Factory for the starting document when the file
                doesn't exist yet, or is empty. Defaults to the fleet
                registry's ``{"version", "clusters", "lease"}`` shape.
        """
        self.path = Path(path).expanduser()
        self._lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self._empty_doc = empty_doc or _empty_doc

    def read(self) -> tuple[dict, Any]:
        """Read the document.

        Returns:
            (document, token) — the token is unused for local storage.
        """
        if not self.path.exists():
            return self._empty_doc(), None
        return json.loads(self.path.read_text() or "{}") or self._empty_doc(), None

    def write(self, doc: dict, token: Any) -> None:
        """Write the document atomically (tmp + rename).

        Args:
            doc: The document to persist.
            token: Ignored (locking makes local writes conflict-free).
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=2, sort_keys=True))
        os.rename(tmp, self.path)

    def update(self, fn: Callable[[dict], None]) -> dict:
        """Apply a mutation under the file lock.

        Args:
            fn: Mutator invoked with the current document (mutates in place).

        Returns:
            The document as persisted.
        """
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._lock_path, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            doc, token = self.read()
            fn(doc)
            self.write(doc, token)
            return doc


class GcsBackend:
    """Registry storage in a GCS object with generation-match CAS.

    Uses the gcloud CLI only (eray convention): reads via
    ``gcloud storage cat`` + ``objects describe`` (for the generation);
    writes via ``gcloud storage cp --if-generation-match=<gen>`` (``0`` when
    the object must not exist yet). A generation mismatch means another
    writer won the race → :class:`ConflictError` → the CAS loop in
    :class:`ClusterRegistry` re-reads and retries.
    """

    def __init__(self, uri: str):
        """Initialize the backend.

        Args:
            uri: Full object URI, e.g. ``gs://bucket/eray/clusters.json``.

        Raises:
            ValueError: If the URI is not a gs:// object path.
        """
        if not uri.startswith("gs://"):
            raise ValueError(f"GCS registry URI must start with gs://, got {uri!r}")
        self.uri = uri

    def read(self) -> tuple[dict, Any]:
        """Read the document and its generation.

        Returns:
            (document, generation) — generation ``0`` when the object does
            not exist yet (a conditional create).
        """
        try:
            meta = gcloud_json(["storage", "objects", "describe", self.uri])
        except subprocess.CalledProcessError as exc:
            stderr = str(exc.stderr or "")
            if "404" in stderr or "not found" in stderr.lower() or "No URLs matched" in stderr:
                return _empty_doc(), 0
            raise
        generation = int(meta.get("generation", 0)) if isinstance(meta, dict) else 0
        try:
            body = gcloud(["storage", "cat", self.uri])
        except subprocess.CalledProcessError as exc:
            stderr = str(exc.stderr or "")
            if "404" in stderr or "No URLs matched" in stderr:
                return _empty_doc(), 0
            raise
        return (json.loads(body) if body.strip() else _empty_doc()), generation

    def write(self, doc: dict, token: Any) -> None:
        """Conditionally write the document.

        Args:
            doc: The document to persist.
            token: The generation observed at read time (``0`` = create).

        Raises:
            ConflictError: If the object's generation changed since the read.
        """
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            tmp.write(json.dumps(doc, indent=2, sort_keys=True))
            tmp_path = tmp.name
        try:
            gcloud(["storage", "cp", tmp_path, self.uri, f"--if-generation-match={int(token)}"])
        except subprocess.CalledProcessError as exc:
            stderr = str(exc.stderr or "")
            if "412" in stderr or "condition" in stderr.lower() or "Precondition" in stderr:
                raise ConflictError(f"registry {self.uri} changed concurrently") from None
            raise
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def update(self, fn: Callable[[dict], None], *, retries: int = 5) -> dict:
        """Apply a mutation with a compare-and-swap retry loop.

        Args:
            fn: Mutator invoked with the current document (mutates in place).
            retries: CAS attempts before giving up.

        Returns:
            The document as persisted.

        Raises:
            ConflictError: If every attempt lost the race.
        """
        last: ConflictError | None = None
        for attempt in range(retries):
            doc, token = self.read()
            fn(doc)
            try:
                self.write(doc, token)
                return doc
            except ConflictError as exc:
                last = exc
                time.sleep(min(2**attempt * 0.2, 3.0))
        raise last if last is not None else ConflictError("registry update failed")


class ClusterRegistry:
    """Typed operations over the fleet document, backend-agnostic."""

    def __init__(self, backend: LocalBackend | GcsBackend):
        """Initialize with a storage backend.

        Args:
            backend: Local or GCS backend instance.
        """
        self.backend = backend

    # ── construction ────────────────────────────────────────────

    @staticmethod
    def configured_uri() -> str | None:
        """The registry location from ``~/.eray/config.json``, if set."""
        if not CONFIG_PATH.exists():
            return None
        try:
            return (json.loads(CONFIG_PATH.read_text()) or {}).get("fleet_state_uri")
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def set_configured_uri(uri: str | None) -> None:
        """Persist the registry location pointer.

        Args:
            uri: gs:// object URI, local path, or None to reset to default.
        """
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        cfg = {}
        if CONFIG_PATH.exists():
            try:
                cfg = json.loads(CONFIG_PATH.read_text()) or {}
            except (OSError, json.JSONDecodeError):
                cfg = {}
        if uri:
            cfg["fleet_state_uri"] = uri
        else:
            cfg.pop("fleet_state_uri", None)
        tmp = CONFIG_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(cfg, indent=2))
        os.rename(tmp, CONFIG_PATH)

    @classmethod
    def from_config(cls) -> ClusterRegistry:
        """Build a registry from the configured location (default: local)."""
        uri = cls.configured_uri()
        if uri and uri.startswith("gs://"):
            return cls(GcsBackend(uri))
        if uri:
            return cls(LocalBackend(uri))
        return cls(LocalBackend())

    # ── reads ───────────────────────────────────────────────────

    def load(self) -> dict[str, ClusterRecord]:
        """All registered clusters.

        Returns:
            Mapping of name → record.
        """
        doc, _ = self.backend.read()
        return {name: ClusterRecord.from_dict(data) for name, data in (doc.get("clusters") or {}).items()}

    def get(self, name: str) -> ClusterRecord | None:
        """One cluster's record, or None."""
        return self.load().get(name)

    # ── writes (CAS through the backend) ────────────────────────

    def upsert(self, record: ClusterRecord) -> None:
        """Insert or replace a cluster record.

        Args:
            record: The record to persist.
        """

        def mutate(doc: dict) -> None:
            doc.setdefault("clusters", {})[record.name] = record.to_dict()

        self.backend.update(mutate)

    def mutate_record(self, name: str, fn: Callable[[ClusterRecord], None]) -> ClusterRecord:
        """Read-modify-write one record atomically.

        Args:
            name: Cluster name.
            fn: Mutator applied to the (fresh) record inside the CAS loop.

        Returns:
            The record as persisted.

        Raises:
            KeyError: If the cluster is not registered.
        """
        result: dict = {}

        def mutate(doc: dict) -> None:
            clusters = doc.setdefault("clusters", {})
            if name not in clusters:
                raise KeyError(f"cluster {name!r} is not registered")
            record = ClusterRecord.from_dict(clusters[name])
            fn(record)
            clusters[name] = record.to_dict()
            result["record"] = record

        self.backend.update(mutate)
        return result["record"]

    def remove(self, name: str) -> bool:
        """Remove a cluster record.

        Args:
            name: Cluster name.

        Returns:
            True if it existed.
        """
        existed: dict = {}

        def mutate(doc: dict) -> None:
            existed["yes"] = (doc.get("clusters") or {}).pop(name, None) is not None

        self.backend.update(mutate)
        return bool(existed.get("yes"))

    # ── watcher lease ───────────────────────────────────────────

    @staticmethod
    def _holder() -> str:
        """This process's lease identity."""
        return f"{socket.gethostname()}:{os.getpid()}"

    def acquire_lease(self, *, ttl: float = LEASE_TTL_S, now: float | None = None) -> bool:
        """Try to become (or remain) the fleet's active watcher.

        Args:
            ttl: Lease validity in seconds from now.
            now: Injected clock for tests.

        Returns:
            True if this process holds the lease after the call. An expired
            lease is stolen; a live lease held elsewhere is respected.
        """
        now = time.time() if now is None else now
        holder = self._holder()
        acquired: dict = {}

        def mutate(doc: dict) -> None:
            lease = doc.get("lease")
            if lease and lease.get("holder") != holder and float(lease.get("expires", 0)) > now:
                acquired["ok"] = False
                return
            doc["lease"] = {"holder": holder, "expires": now + ttl}
            acquired["ok"] = True

        self.backend.update(mutate)
        return bool(acquired.get("ok"))

    def release_lease(self) -> None:
        """Release the lease if this process holds it."""
        holder = self._holder()

        def mutate(doc: dict) -> None:
            lease = doc.get("lease")
            if lease and lease.get("holder") == holder:
                doc["lease"] = None

        self.backend.update(mutate)

    def lease_holder(self, *, now: float | None = None) -> str | None:
        """The current live lease holder, or None.

        Args:
            now: Injected clock for tests.
        """
        now = time.time() if now is None else now
        doc, _ = self.backend.read()
        lease = doc.get("lease")
        if lease and float(lease.get("expires", 0)) > now:
            return str(lease.get("holder"))
        return None
