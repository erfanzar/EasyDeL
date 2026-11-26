# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Durable, fenced state shared by Eyvan cell agents."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import time
import typing as tp
import uuid
from contextlib import contextmanager
from pathlib import Path
from threading import Event, RLock, Thread

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic import JsonValue as PydanticJsonValue

from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object, model_json
from eyvan._internal.subprocess_env import provider_cli_env
from eyvan._internal.workspace import create_private_directory
from eyvan.capacity import CapacityStatus
from eyvan.capacity.drivers.base import DriverBusyError
from eyvan.capacity.gcs_rest import GcsPreconditionFailed, GcsRestError, GcsRestObject

#: Attempts at reading a body whose metadata says it exists. Small on purpose:
#: this guards a transient link, not an outage, and every caller is on a
#: reconcile loop that will come back.
_READ_ATTEMPTS = 3

SCHEMA_VERSION = 1
SCHEMA_ID = "eyvan.fleet-state"
DEFAULT_LOCAL_PATH = Path("~/.eyvan/fleet-state.json").expanduser()
DocumentToken: tp.TypeAlias = int | None


#: One identity per *process*, not per store instance. The persisted lease
#: exists to exclude other processes; two stores inside one agent are not rival
#: writers. Separate identities would make an agent contend against its own
#: lease, refusing capacity mutations for the full TTL.
_PROCESS_OWNER = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex}"

#: One lock per process owner *and durable document*. Stores sharing both are
#: one writer and must share a lock; an unrelated fleet-state document must not
#: wait behind long provider I/O merely because the same agent process owns it.
#: Stores with distinct owners represent distinct agents and contend through the
#: durable lease instead. A per-instance lock gets same-document aliases wrong;
#: an owner-only lock serializes every fleet in the process.
_EXCLUSIVE_LOCKS: dict[tuple[str, str], RLock] = {}
_EXCLUSIVE_LOCKS_GUARD = RLock()


def _lock_for(owner: str, scope: str) -> RLock:
    """Return the process-wide lock guarding one owner's durable fleet scope."""

    with _EXCLUSIVE_LOCKS_GUARD:
        return _EXCLUSIVE_LOCKS.setdefault((owner, scope), RLock())


def _holder_is_this_host(holder: str) -> tuple[bool, int | None]:
    """Return whether a holder token names this host, and its PID if parseable."""

    parts = holder.split(":")
    if len(parts) != 3 or parts[0] != socket.gethostname():
        return False, None
    try:
        return True, int(parts[1])
    except ValueError:
        return True, None


def _holder_is_dead(holder: str) -> bool:
    """Return whether a holder is a process on this host that no longer exists.

    A dead process cannot be mid-mutation, so its lease is safe to break. Without
    this an agent restart freezes every capacity mutation until the TTL expires,
    which is an hour of a fleet doing nothing for no reason.

    Deliberately conservative: a holder on another host is never assumed dead,
    because nothing here can observe it.
    """

    same_host, pid = _holder_is_this_host(holder)
    if not same_host or pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    except OSError:
        return False
    return False


class ConflictError(RuntimeError):
    """Another writer changed a compare-and-swap document."""


class StaleFenceError(RuntimeError):
    """A mutation does not own the durable generation fence."""


class FleetCellNotRegisteredError(StaleFenceError):
    """A fenced mutation targeted a cell record that was removed."""


class FleetLeaseBusyError(DriverBusyError):
    """Another cell agent owns the fleet mutation lease."""


class FleetStateSchemaError(RuntimeError):
    """A fleet document is foreign, malformed, or from an unsupported version."""


class FleetStateUnavailableError(RuntimeError):
    """The fleet state store could not be reached, so its contents are unknown.

    Deliberately not a :class:`ConflictError`: nothing conflicted. A caller that
    retries a conflict is right to do so immediately, while a caller that cannot
    see the store has learned nothing and must not conclude anything about the
    cells it describes.
    """


def _empty_document() -> JsonObject:
    return {
        "schema": SCHEMA_ID,
        "version": SCHEMA_VERSION,
        "cells": {},
        "cell_tombstones": {},
        "workload_fences": {},
        "lease": None,
    }


def _validate_document(value: JsonValue) -> JsonObject:
    if not is_json_object(value):
        raise FleetStateSchemaError("Eyvan fleet state must be a JSON object")
    if not value:
        return _empty_document()
    if value.get("schema") != SCHEMA_ID:
        raise FleetStateSchemaError("fleet state is not an Eyvan document; use a dedicated EYVAN_FLEET_STATE_URI")
    if value.get("version") != SCHEMA_VERSION:
        raise FleetStateSchemaError(
            f"unsupported Eyvan fleet-state version {value.get('version')!r}; expected {SCHEMA_VERSION}"
        )
    for field_name in ("cells", "cell_tombstones", "workload_fences"):
        entries = value.get(field_name)
        if not isinstance(entries, dict):
            raise FleetStateSchemaError(f"Eyvan fleet-state field {field_name!r} must be an object")
        # A present but unreadable entry is unknown, never missing capacity or
        # unclaimed ownership. Reject before any read or mutation can skip it.
        if any(not isinstance(entry, dict) for entry in entries.values()):
            raise FleetStateSchemaError(f"Eyvan fleet-state field {field_name!r} entries must be objects")
    if value.get("lease") is not None and not isinstance(value.get("lease"), dict):
        raise FleetStateSchemaError("Eyvan fleet-state field 'lease' must be an object or null")
    return value


def _object_field(document: JsonObject, field_name: str) -> JsonObject:
    value = document.get(field_name)
    if not is_json_object(value):
        raise FleetStateSchemaError(f"Eyvan fleet-state field {field_name!r} must be an object")
    return value


def _required_generation(value: JsonObject, *, context: str) -> int:
    generation = value.get("generation")
    if isinstance(generation, bool) or not isinstance(generation, int):
        raise FleetStateSchemaError(f"{context} generation must be an integer")
    return generation


def _lease_expiry(lease: JsonObject) -> float:
    value = lease.get("expires", 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FleetStateSchemaError("fleet mutation lease expiry must be numeric")
    return float(value)


def _predecessors(fence: JsonObject) -> tuple[JsonObject, ...]:
    value = fence.get("predecessors", [])
    if not isinstance(value, list):
        raise FleetStateSchemaError("workload fence predecessors must be an array")
    if not all(is_json_object(item) for item in value):
        raise FleetStateSchemaError("workload fence predecessors must contain only objects")
    return tuple(dict(item) for item in value if is_json_object(item))


class DocumentBackend(tp.Protocol):
    """Atomic JSON document storage used by :class:`FleetStateStore`."""

    @property
    def mutation_scope(self) -> str:
        """Return a stable identity for the durable document this backend mutates."""

        ...

    def read(self) -> tuple[JsonObject, DocumentToken]: ...

    def update(self, mutate: tp.Callable[[JsonObject], None]) -> JsonObject: ...


class LocalBackend:
    """Atomic local JSON storage guarded by an advisory file lock."""

    def __init__(self, path: Path | str = DEFAULT_LOCAL_PATH):
        self.path = Path(path).expanduser()
        self._lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @property
    def mutation_scope(self) -> str:
        """Identify aliases of the same local fleet-state document."""

        return f"local:{self.path.resolve(strict=False)}"

    def read(self) -> tuple[JsonObject, None]:
        if not self.path.exists():
            return _empty_document(), None
        try:
            value = json.loads(self.path.read_text() or "{}")
        except json.JSONDecodeError as exc:
            raise FleetStateSchemaError(f"Eyvan fleet state at {self.path} is not valid JSON") from exc
        return _validate_document(value), None

    def _write(self, document: JsonObject) -> None:
        create_private_directory(self.path.parent)
        with tempfile.NamedTemporaryFile(
            "w",
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            json.dump(document, temporary, indent=2, sort_keys=True)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, self.path)

    def update(self, mutate: tp.Callable[[JsonObject], None]) -> JsonObject:
        import fcntl

        create_private_directory(self.path.parent)
        with self._lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            document, _ = self.read()
            mutate(document)
            self._write(document)
            return document


def _gcloud(args: tp.Sequence[str]) -> subprocess.CompletedProcess[str]:
    timeout_s = max(float(os.getenv("EYVAN_GCLOUD_STORAGE_TIMEOUT_S", "30")), 1.0)
    return subprocess.run(
        ["gcloud", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        env=provider_cli_env(),
    )


class GcsBackend:
    """GCS JSON storage using object generations for compare-and-swap."""

    def __init__(
        self,
        uri: str,
        *,
        runner: tp.Callable[[tp.Sequence[str]], subprocess.CompletedProcess[str]] = _gcloud,
        rest: GcsRestObject | None = None,
    ):
        if not uri.startswith("gs://") or uri.endswith("/"):
            raise ValueError("fleet state URI must be a gs:// object path")
        self.uri = uri
        self._runner = runner
        # The fast path. Every mutation here is a compare-and-swap, and the
        # subprocess form spends three interpreter startups on it -- describe,
        # cat, conditional copy. Avoiding that overhead matters because releases
        # serialize on this document. ``None`` builds the default reader; pass
        # ``False`` to force the
        # subprocess path.
        # Only when this backend is using the real gcloud transport. A caller
        # that injects a runner is saying "drive the object through this", and
        # silently preferring an API path would bypass it and let tests using
        # fake runners reach the network or mint real tokens.
        default_rest = GcsRestObject(uri) if runner is _gcloud else None
        self._rest: GcsRestObject | None = default_rest if rest is None else (rest or None)

    @property
    def mutation_scope(self) -> str:
        """Identify all backend instances targeting the same GCS object."""

        return f"gcs:{self.uri}"

    @staticmethod
    def _missing(result: subprocess.CompletedProcess[str]) -> bool:
        detail = f"{result.stdout}\n{result.stderr}".lower()
        return result.returncode != 0 and ("404" in detail or "not found" in detail or "no urls matched" in detail)

    def read(self) -> tuple[JsonObject, int]:
        described = self._runner(("storage", "objects", "describe", self.uri, "--format=json"))
        if self._missing(described):
            return _empty_document(), 0
        if described.returncode != 0:
            raise RuntimeError(described.stderr.strip() or f"could not describe {self.uri}")
        try:
            metadata = json.loads(described.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise FleetStateSchemaError(f"GCS metadata for {self.uri} is not valid JSON") from exc
        if not is_json_object(metadata):
            raise RuntimeError(f"GCS metadata for {self.uri} is not a JSON object")
        raw_generation = metadata.get("generation", 0)
        if isinstance(raw_generation, bool) or not isinstance(raw_generation, (int, str)):
            raise RuntimeError(f"GCS metadata for {self.uri} has an invalid generation")
        try:
            generation = int(raw_generation)
        except ValueError as exc:
            raise RuntimeError(f"GCS metadata for {self.uri} has an invalid generation") from exc
        # A read that fails on a flaky link is retried before it is believed.
        # `describe` has just proven the object exists, so a failed body read
        # may be transient rather than evidence that the object is absent.
        #
        # Refusing to treat that as an empty fleet (below) is necessary but not
        # sufficient -- without a retry, one blip still takes out every route
        # that reads fleet state, including the log fetch an operator needs to
        # diagnose the blip.
        # Read once, then retry between checks rather than checking inside the
        # loop: identical attempt count and backoff, and ``body`` is provably
        # bound afterwards, which the loop-first shape left to the reader.
        body = self._runner(("storage", "cat", self.uri))
        for attempt in range(_READ_ATTEMPTS - 1):
            if not self._missing(body) and body.returncode == 0 and body.stdout.strip():
                break
            time.sleep(min(0.2 * (2**attempt), 2.0))
            body = self._runner(("storage", "cat", self.uri))
        # `describe` has already proven the object exists at `generation`, so from
        # here an unreadable or blank body is a *failed read*, never an empty
        # fleet. Returning `_empty_document()` here says "this fleet has no
        # cells", and every caller believes it: `state.get(cell_id)` returns None,
        # `_ensure_owned` concludes the cell no longer belongs to its generation,
        # the controller records a fence conflict, and the autoscaler drains live
        # capacity. One flaky `gcloud storage cat` destroys the whole fleet's
        # worth of cells and the next one destroys their replacements.
        if self._missing(body) or body.returncode != 0:
            raise FleetStateUnavailableError(
                body.stderr.strip() or f"could not read {self.uri} at generation {generation}"
            )
        if not body.stdout.strip():
            raise FleetStateUnavailableError(
                f"{self.uri} exists at generation {generation} but read back empty; "
                "refusing to treat it as an empty fleet"
            )
        try:
            value = json.loads(body.stdout)
        except json.JSONDecodeError as exc:
            raise FleetStateSchemaError(f"Eyvan fleet state at {self.uri} is not valid JSON") from exc
        return _validate_document(value), generation

    def _write(self, document: JsonObject, generation: int) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as temporary:
            json.dump(document, temporary, indent=2, sort_keys=True)
            temporary_path = temporary.name
        try:
            result = self._runner(("storage", "cp", temporary_path, self.uri, f"--if-generation-match={generation}"))
        except subprocess.TimeoutExpired:
            # A conditional write that times out client-side is *indeterminate*:
            # the object may well have been replaced already. Reporting it as a
            # failure is what makes a control plane and its agent disagree about a
            # generation -- the agent says "my mutation failed", the object says
            # otherwise, and the next fenced command is rejected as stale, which
            # the control plane reads as a lost cell and destroys real capacity.
            #
            # So ask the object what happened rather than guessing. Only if the
            # write demonstrably did not land is this an error, and then it is a
            # typed unavailability, not a ConflictError -- nothing conflicted.
            if self._landed(document):
                return
            raise FleetStateUnavailableError(f"conditional write to {self.uri} timed out and did not land") from None
        finally:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
        if result.returncode == 0:
            return
        detail = f"{result.stdout}\n{result.stderr}".lower()
        if "412" in detail or "precondition" in detail or "condition" in detail:
            raise ConflictError(f"fleet state {self.uri} changed concurrently")
        raise RuntimeError(result.stderr.strip() or f"could not write {self.uri}")

    def _landed(self, document: JsonObject) -> bool:
        """Return whether a timed-out conditional write actually replaced the object."""

        try:
            stored, _generation = self.read()
        except (subprocess.TimeoutExpired, RuntimeError):
            # Still cannot see the object; refuse to claim either outcome.
            return False
        return stored == document

    def update(
        self,
        mutate: tp.Callable[[JsonObject], None],
        *,
        retries: int = 5,
    ) -> JsonObject:
        last_error: ConflictError | None = None
        for attempt in range(retries):
            document, generation = self._read_fast()
            mutate(document)
            try:
                self._write_fast(document, generation)
                return document
            except ConflictError as exc:
                last_error = exc
                time.sleep(min(0.2 * (2**attempt), 3.0))
        raise last_error or ConflictError("fleet state update failed")

    def _read_fast(self) -> tuple[JsonObject, int]:
        """Read over the API, falling back to the subprocess path.

        Both forms return the same document and the same generation, so nothing
        downstream can tell which answered. The fallback exists because a raw
        HTTPS call can be blocked where gcloud is configured to work, and
        because the token may not be mintable at all.
        """

        if self._rest is not None:
            try:
                body, generation = self._rest.read()
            except GcsRestError:
                # Not disabled. One failure is a token that aged out or a blip;
                # turning the fast path off for the life of the process would
                # pin every later mutation onto the subprocess, which may itself
                # be unavailable. Fall back for this call only.
                pass
            else:
                if body is None:
                    return _empty_document(), 0
                return _validate_document(body), generation
        return self.read()

    def _write_fast(self, document: JsonObject, generation: int) -> None:
        """Conditionally write over the API, falling back to the subprocess path.

        A refused precondition is a ``ConflictError`` exactly as the subprocess
        path reports it: the caller re-reads and reapplies, which is the
        compare-and-swap. Anything else drops to the fallback rather than
        failing the mutation, because an unavailable API is not a conflict.
        """

        if self._rest is not None:
            try:
                self._rest.write(document, generation)
                return
            except GcsPreconditionFailed as exc:
                raise ConflictError(str(exc)) from exc
            except GcsRestError:
                # Per call, not for the life of the process. See ``_read_fast``.
                pass
        self._write(document, generation)


class FleetCellRecord(BaseModel):
    """Provider identity and fenced desired state for one cell."""

    model_config = ConfigDict(extra="forbid")

    name: str
    project: str | None = None
    zone: str | None = None
    accelerator_type: str | None = None
    capacity_class: str = "spot"
    runtime_version: str | None = None
    qr_id: str | None = None
    provider_generation: int = 0
    desired_state: str = "up"
    state: CapacityStatus = CapacityStatus.UNOBSERVED
    head_ip: str | None = None
    execution_endpoint: str | None = None
    generation: int = 0
    fencing_token: str | None = None
    idempotency_keys: list[str] = Field(default_factory=list)
    metadata: dict[str, PydanticJsonValue] = Field(default_factory=dict)

    def to_dict(self) -> JsonObject:
        return model_json(self)

    @classmethod
    def from_dict(cls, value: tp.Mapping[str, JsonValue]) -> FleetCellRecord:
        known = set(cls.model_fields)
        data = {key: item for key, item in value.items() if key in known}
        unknown = {key: item for key, item in value.items() if key not in known}
        metadata = data.get("metadata", {})
        if not is_json_object(metadata):
            raise FleetStateSchemaError("fleet cell metadata must be a JSON object")
        data["metadata"] = {**metadata, **unknown}
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise FleetStateSchemaError("fleet cell record is malformed") from exc

    def next_qr_id(self) -> str:
        return f"{self.name}-r{self.provider_generation + 1}"


class _GangMutation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    idempotency_key: str = Field(min_length=1)
    generation: int = Field(ge=0, strict=True)
    fencing_token: str = Field(min_length=1)


class _GangMember(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cell_id: str = Field(min_length=1)
    agent_endpoint: str = Field(min_length=1)
    mutation: _GangMutation


class FleetStateStore:
    """Typed, fenced operations over a shared Eyvan fleet document."""

    #: How long a mutation lease outlives its holder's last renewal. The
    #: renewal thread inside ``exclusive`` refreshes every ttl/3 while a
    #: mutation runs, so a live holder never loses its lease however long the
    #: provider takes -- which is what makes a *short* default safe. The point
    #: of the bound is the holder that cannot renew: a crashed or fully wedged
    #: agent, on any host, freezes fleet mutations only until the TTL expires
    #: instead of for an hour. Same-host dead PIDs were already breakable
    #: (``_holder_is_dead``); this closes the cross-host gap, where nothing can
    #: observe the process at all.
    DEFAULT_LEASE_TTL_S: tp.ClassVar[float] = 300

    def __init__(self, backend: DocumentBackend, *, lease_ttl_s: float = DEFAULT_LEASE_TTL_S, owner: str | None = None):
        if lease_ttl_s <= 0:
            raise ValueError("fleet mutation lease TTL must be positive")
        self.backend = backend
        self.lease_ttl_s = lease_ttl_s
        #: Defaults to the process identity because several stores are built over
        #: one document inside a single agent -- the GCP driver alone builds two.
        #: An explicit owner is how a *different* agent is represented, including
        #: when a test needs to simulate one.
        self._owner = owner or _PROCESS_OWNER
        self._exclusive_lock = _lock_for(self._owner, backend.mutation_scope)

    @classmethod
    def from_uri(cls, uri: str | None, *, lease_ttl_s: float = 300, owner: str | None = None) -> FleetStateStore:
        if uri and "://" in uri:
            if uri.startswith("gs://"):
                return cls(GcsBackend(uri), lease_ttl_s=lease_ttl_s, owner=owner)
            scheme = uri.split("://", 1)[0]
            raise ValueError(f"unsupported fleet-state URI scheme {scheme!r}; use a gs:// URI or a filesystem path")
        return cls(LocalBackend(uri or DEFAULT_LOCAL_PATH), lease_ttl_s=lease_ttl_s, owner=owner)

    def _holder(self) -> str:
        return self._owner

    @property
    def owner(self) -> str:
        """This store's writer identity: ``host:pid:nonce`` unless a test names one."""

        return self._owner

    def load(self) -> dict[str, FleetCellRecord]:
        document, _ = self.backend.read()
        return {
            name: FleetCellRecord.from_dict(value)
            for name, value in _object_field(document, "cells").items()
            if is_json_object(value)
        }

    def get(self, name: str) -> FleetCellRecord | None:
        return self.load().get(name)

    def cell_tombstone(self, name: str) -> JsonObject | None:
        document, _ = self.backend.read()
        value = _object_field(document, "cell_tombstones").get(name)
        return dict(value) if is_json_object(value) else None

    def workload_fence(self, workload_id: str) -> JsonObject | None:
        document, _ = self.backend.read()
        value = _object_field(document, "workload_fences").get(workload_id)
        return dict(value) if is_json_object(value) else None

    @staticmethod
    def _reject_tombstone(document: JsonObject, name: str, generation: int | None) -> None:
        if generation is None:
            return
        tombstone = _object_field(document, "cell_tombstones").get(name)
        if not is_json_object(tombstone):
            return
        fenced_generation = _required_generation(tombstone, context=f"cell tombstone {name!r}")
        if generation <= fenced_generation:
            raise StaleFenceError(
                f"generation {generation} is not newer than fenced cell {name!r} generation {fenced_generation}"
            )

    def mutate_record(
        self,
        name: str,
        mutate: tp.Callable[[FleetCellRecord], None],
        *,
        generation: int | None = None,
    ) -> FleetCellRecord:
        result: dict[str, FleetCellRecord] = {}

        def update(document: JsonObject) -> None:
            self._reject_tombstone(document, name, generation)
            cells = _object_field(document, "cells")
            if name not in cells:
                raise FleetCellNotRegisteredError(f"cell {name!r} is not registered")
            value = cells[name]
            if not is_json_object(value):
                raise FleetStateSchemaError(f"fleet cell {name!r} must be an object")
            record = FleetCellRecord.from_dict(value)
            mutate(record)
            cells[name] = record.to_dict()
            result["record"] = record

        self.backend.update(update)
        return result["record"]

    def mutate_or_create_record(
        self,
        name: str,
        mutate: tp.Callable[[FleetCellRecord], None],
        *,
        generation: int | None = None,
    ) -> FleetCellRecord:
        result: dict[str, FleetCellRecord] = {}

        def update(document: JsonObject) -> None:
            self._reject_tombstone(document, name, generation)
            cells = _object_field(document, "cells")
            value = cells.get(name)
            if value is not None and not is_json_object(value):
                raise FleetStateSchemaError(f"fleet cell {name!r} must be an object")
            record = FleetCellRecord.from_dict(value) if value is not None else FleetCellRecord(name=name)
            mutate(record)
            cells[name] = record.to_dict()
            result["record"] = record

        self.backend.update(update)
        return result["record"]

    def drain_record(
        self,
        name: str,
        generation: int,
        fencing_token: str,
        mutate: tp.Callable[[FleetCellRecord], None],
    ) -> FleetCellRecord | None:
        result: dict[str, FleetCellRecord | None] = {}

        def update(document: JsonObject) -> None:
            tombstones = _object_field(document, "cell_tombstones")
            current = tombstones.get(name)
            self._validate_fence(name, generation, fencing_token, current, action="fenced")
            cells = _object_field(document, "cells")
            value = cells.get(name)
            if value is not None and not is_json_object(value):
                raise FleetStateSchemaError(f"fleet cell {name!r} must be an object")
            record = FleetCellRecord.from_dict(value) if value is not None else None
            if record is not None:
                # The tombstone check above is not enough on its own: a cell that
                # has never been drained or released has no tombstone, and
                # ``_validate_fence`` returns immediately when there is none. A
                # superseded controller could therefore drain a live cell a newer
                # incarnation owns -- taking healthy capacity out of service with
                # every other mutation on that cell correctly refused. Draining is
                # a mutation like any other, so it is fenced like one.
                self._validate_fence(
                    name,
                    generation,
                    fencing_token,
                    {"generation": record.generation, "fencing_token": record.fencing_token},
                    action="drained",
                )
                mutate(record)
                cells[name] = record.to_dict()
            tombstones[name] = {
                "generation": generation,
                "fencing_token": fencing_token,
                "state": CapacityStatus.DRAINING,
            }
            result["record"] = record

        self.backend.update(update)
        return result.get("record")

    def tombstone_record(
        self,
        name: str,
        generation: int,
        fencing_token: str,
        predicate: tp.Callable[[FleetCellRecord], bool],
        *,
        metadata: tp.Mapping[str, JsonValue] | None = None,
    ) -> bool:
        result: dict[str, bool] = {}

        def update(document: JsonObject) -> None:
            tombstones = _object_field(document, "cell_tombstones")
            self._validate_fence(name, generation, fencing_token, tombstones.get(name), action="released")
            cells = _object_field(document, "cells")
            value = cells.get(name)
            if is_json_object(value) and not predicate(FleetCellRecord.from_dict(value)):
                result["removed"] = False
                return
            cells.pop(name, None)
            extra = dict(metadata or {})
            if {"generation", "fencing_token"} & extra.keys():
                raise ValueError("cell tombstone metadata cannot replace its generation fence")
            tombstones[name] = {"generation": generation, "fencing_token": fencing_token, **extra}
            result["removed"] = True

        self.backend.update(update)
        return bool(result.get("removed"))

    def fence_missing_record(
        self,
        name: str,
        generation: int,
        fencing_token: str,
    ) -> bool:
        """Fence a known-missing cell without changing a replacement record.

        This is used after an external operation observes that its original
        cell vanished between durable state transitions. It prevents replay of
        that generation while leaving a concurrently-created replacement
        untouched. Returns ``True`` only when this call installed the fence.
        """

        result: dict[str, bool] = {}

        def update(document: JsonObject) -> None:
            tombstones = _object_field(document, "cell_tombstones")
            tombstone = tombstones.get(name)
            self._validate_fence(name, generation, fencing_token, tombstone, action="fenced")
            if is_json_object(tombstone):
                tombstone_generation = _required_generation(tombstone, context=f"fenced cell {name!r}")
                if tombstone_generation == generation and tombstone.get("fencing_token") == fencing_token:
                    result["fenced"] = False
                    return
            cells = _object_field(document, "cells")
            value = cells.get(name)
            if value is not None:
                if not is_json_object(value):
                    raise FleetStateSchemaError(f"fleet cell {name!r} must be an object")
                result["fenced"] = False
                return
            tombstones[name] = {"generation": generation, "fencing_token": fencing_token}
            result["fenced"] = True

        self.backend.update(update)
        return bool(result.get("fenced"))

    @staticmethod
    def _validate_fence(
        name: str,
        generation: int,
        fencing_token: str,
        current: JsonValue,
        *,
        action: str,
    ) -> None:
        if not is_json_object(current):
            return
        current_generation = _required_generation(current, context=f"{action} cell {name!r}")
        if generation < current_generation:
            raise StaleFenceError(
                f"generation {generation} is older than {action} cell {name!r} generation {current_generation}"
            )
        if generation == current_generation and current.get("fencing_token") != fencing_token:
            raise StaleFenceError(f"fencing token does not own {action} cell {name!r} generation {current_generation}")

    def claim_workload_fence(
        self,
        workload_id: str,
        generation: int,
        fencing_token: str,
        *,
        remote_id: str,
        cell_id: str,
    ) -> JsonObject:
        """Claim a single execution without discarding unresolved gang ownership.

        Single submission cannot retire gang ranks. Refuse both a current gang
        and an inherited gang predecessor, even for a newer generation, before
        changing the fence. Recovery must use the gang submission path, which
        preserves every member and positively verifies retirement quiescence.
        """

        claimed: dict[str, JsonObject] = {}

        def update(document: JsonObject) -> None:
            fences = _object_field(document, "workload_fences")
            current = fences.get(workload_id)
            if is_json_object(current):
                current_generation = _required_generation(current, context=f"workload {workload_id!r}")
                if generation < current_generation:
                    raise StaleFenceError(
                        f"generation {generation} is older than workload {workload_id!r} generation {current_generation}"
                    )
                if current.get("kind") == "gang" or any(
                    predecessor.get("kind") == "gang" for predecessor in _predecessors(current)
                ):
                    raise StaleFenceError(
                        f"single-cell submission does not own gang workload {workload_id!r} "
                        f"generation {current_generation}; use gang submission to retire its ranks"
                    )
                if generation == current_generation:
                    if current.get("fencing_token") != fencing_token:
                        raise StaleFenceError(
                            f"fencing token does not own workload {workload_id!r} generation {current_generation}"
                        )
                    for field_name, requested in (("remote_id", remote_id), ("cell_id", cell_id)):
                        persisted = current.get(field_name)
                        if persisted is not None and persisted != requested:
                            raise StaleFenceError(
                                f"{field_name} does not own workload {workload_id!r} generation {current_generation}"
                            )
                    claimed["fence"] = dict(current)
                    return

            predecessors: list[JsonObject] = []
            if is_json_object(current):
                raw_predecessors = current.get("predecessors")
                if isinstance(raw_predecessors, list):
                    predecessors.extend(dict(item) for item in raw_predecessors if is_json_object(item))
                if current.get("remote_id") and current.get("cell_id"):
                    prior_generation = _required_generation(current, context=f"workload {workload_id!r}")
                    prior_token = current.get("fencing_token")
                    prior_remote_id = current.get("remote_id")
                    prior_cell_id = current.get("cell_id")
                    if not all(isinstance(item, str) for item in (prior_token, prior_remote_id, prior_cell_id)):
                        raise FleetStateSchemaError(f"workload {workload_id!r} predecessor identity is malformed")
                    predecessors.append(
                        {
                            "generation": prior_generation,
                            "fencing_token": prior_token,
                            "remote_id": prior_remote_id,
                            "cell_id": prior_cell_id,
                        }
                    )
            unique = {
                (item["cell_id"], item["remote_id"]): item
                for item in predecessors
                if isinstance(item.get("cell_id"), str) and isinstance(item.get("remote_id"), str)
            }
            fence: JsonObject = {
                "generation": generation,
                "fencing_token": fencing_token,
                "remote_id": remote_id,
                "cell_id": cell_id,
                **({"predecessors": list(unique.values())} if unique else {}),
            }
            fences[workload_id] = fence
            claimed["fence"] = fence

        self.backend.update(update)
        return dict(claimed["fence"])

    @staticmethod
    def _canonical_gang_members(members: tp.Sequence[tp.Mapping[str, JsonValue]]) -> tuple[_GangMember, ...]:
        """Validate and normalize every independently fenced gang member."""

        normalized: list[_GangMember] = []
        for member in members:
            try:
                normalized.append(_GangMember.model_validate(member))
            except ValidationError as exc:
                raise ValueError("gang workload member is malformed") from exc
        if any(
            value != value.strip()
            for member in normalized
            for value in (
                member.cell_id,
                member.agent_endpoint,
                member.mutation.idempotency_key,
                member.mutation.fencing_token,
            )
        ):
            raise ValueError("gang workload member strings must be trimmed")
        if len(normalized) < 2:
            raise ValueError("gang workload submission requires at least two members")
        if len({member.cell_id for member in normalized}) != len(normalized):
            raise ValueError("gang workload member cell IDs must be unique")
        if len({member.agent_endpoint for member in normalized}) != 1:
            raise ValueError("gang workload members must use one agent endpoint")
        generations = {member.mutation.generation for member in normalized}
        if len(generations) != 1:
            raise ValueError("gang workload member generations must match")
        return tuple(sorted(normalized, key=lambda member: member.cell_id))

    def claim_gang_workload_fence(
        self,
        workload_id: str,
        *,
        remote_id: str,
        coordinator_cell_id: str,
        members: tp.Sequence[tp.Mapping[str, JsonValue]],
    ) -> JsonObject:
        """Claim one workload fence shared by a set of independently fenced cells."""

        normalized_members = self._canonical_gang_members(members)
        if coordinator_cell_id != normalized_members[0].cell_id:
            raise ValueError("native gang coordinator must be the lexicographically first member cell")
        if not isinstance(remote_id, str) or not remote_id or remote_id != remote_id.strip():
            raise ValueError("gang workload remote_id must be non-empty and trimmed")
        generation = normalized_members[0].mutation.generation
        serialized_members = [model_json(member) for member in normalized_members]
        claimed: dict[str, JsonObject] = {}

        def predecessor_from(current: JsonObject) -> JsonObject | None:
            if current.get("kind") == "gang":
                prior_members = current.get("members")
                if isinstance(prior_members, list):
                    prior_remote_id = current.get("remote_id")
                    prior_coordinator = current.get("coordinator_cell_id")
                    if not isinstance(prior_remote_id, str) or not isinstance(prior_coordinator, str):
                        raise FleetStateSchemaError("gang workload predecessor identity is malformed")
                    return {
                        "kind": "gang",
                        "remote_id": prior_remote_id,
                        "coordinator_cell_id": prior_coordinator,
                        "members": [dict(member) for member in prior_members if is_json_object(member)],
                    }
                return None
            if current.get("remote_id") and current.get("cell_id"):
                prior_remote_id = current.get("remote_id")
                prior_cell_id = current.get("cell_id")
                prior_token = current.get("fencing_token")
                if not all(isinstance(item, str) for item in (prior_remote_id, prior_cell_id, prior_token)):
                    raise FleetStateSchemaError("single workload predecessor identity is malformed")
                return {
                    "kind": "single",
                    "remote_id": prior_remote_id,
                    "cell_id": prior_cell_id,
                    "generation": _required_generation(current, context=f"workload {workload_id!r}"),
                    "fencing_token": prior_token,
                }
            return None

        def update(document: JsonObject) -> None:
            fences = _object_field(document, "workload_fences")
            current = fences.get(workload_id)
            if is_json_object(current):
                current_generation = _required_generation(current, context=f"workload {workload_id!r}")
                if generation < current_generation:
                    raise StaleFenceError(
                        f"generation {generation} is older than workload {workload_id!r} generation {current_generation}"
                    )
                if generation == current_generation:
                    if (
                        current.get("kind") != "gang"
                        or current.get("remote_id") != remote_id
                        or current.get("coordinator_cell_id") != coordinator_cell_id
                        or current.get("members") != serialized_members
                    ):
                        raise StaleFenceError(
                            f"native gang does not own workload {workload_id!r} generation {current_generation}"
                        )
                    claimed["fence"] = dict(current)
                    return

            predecessors: list[JsonObject] = []
            if is_json_object(current):
                raw_predecessors = current.get("predecessors")
                if isinstance(raw_predecessors, list):
                    predecessors.extend(dict(item) for item in raw_predecessors if is_json_object(item))
                prior = predecessor_from(current)
                if prior is not None:
                    predecessors.append(prior)
            unique = {
                (
                    str(item.get("kind", "single")),
                    str(item.get("remote_id", "")),
                    repr(item.get("members", item.get("cell_id"))),
                ): item
                for item in predecessors
                if item.get("remote_id")
            }
            fence: JsonObject = {
                "kind": "gang",
                "generation": generation,
                "remote_id": remote_id,
                "coordinator_cell_id": coordinator_cell_id,
                "members": serialized_members,
                **({"predecessors": list(unique.values())} if unique else {}),
            }
            fences[workload_id] = fence
            claimed["fence"] = fence

        self.backend.update(update)
        return dict(claimed["fence"])

    def clear_workload_predecessors(self, workload_id: str, generation: int, fencing_token: str) -> None:
        def update(document: JsonObject) -> None:
            current = _object_field(document, "workload_fences").get(workload_id)
            if not is_json_object(current):
                raise StaleFenceError(f"workload {workload_id!r} has no durable fence")
            current_generation = _required_generation(current, context=f"workload {workload_id!r}")
            if generation != current_generation or current.get("fencing_token") != fencing_token:
                raise StaleFenceError(
                    f"generation {generation} does not own workload {workload_id!r} generation {current_generation}"
                )
            current.pop("predecessors", None)

        self.backend.update(update)

    def clear_gang_workload_predecessors(
        self,
        workload_id: str,
        *,
        members: tp.Sequence[tp.Mapping[str, JsonValue]],
    ) -> None:
        """Clear predecessors only when the caller owns the complete gang fence."""

        normalized_members = self._canonical_gang_members(members)
        generation = normalized_members[0].mutation.generation
        serialized_members = [model_json(member) for member in normalized_members]

        def update(document: JsonObject) -> None:
            current = _object_field(document, "workload_fences").get(workload_id)
            if (
                not is_json_object(current)
                or current.get("kind") != "gang"
                or _required_generation(current, context=f"workload {workload_id!r}") != generation
                or current.get("members") != serialized_members
            ):
                raise StaleFenceError(f"native gang does not own workload {workload_id!r} generation {generation}")
            current.pop("predecessors", None)

        self.backend.update(update)

    def acquire_lease(self, *, ttl: float | None = None) -> bool:
        acquired: dict[str, bool] = {}
        owner = self._holder()
        now = time.time()

        def update(document: JsonObject) -> None:
            current = document.get("lease")
            if (
                is_json_object(current)
                and current.get("holder") != owner
                and _lease_expiry(current) > now
                and not _holder_is_dead(str(current.get("holder", "")))
            ):
                acquired["ok"] = False
                return
            expires = now + (ttl if ttl is not None else self.lease_ttl_s)
            if is_json_object(current) and current.get("holder") == owner:
                expires = max(expires, _lease_expiry(current))
            document["lease"] = {"holder": owner, "expires": expires}
            acquired["ok"] = True

        self.backend.update(update)
        return bool(acquired.get("ok"))

    def release_lease(self) -> None:
        owner = self._holder()

        def update(document: JsonObject) -> None:
            current = document.get("lease")
            if is_json_object(current) and current.get("holder") == owner:
                document["lease"] = None

        self.backend.update(update)

    def lease_holder(self) -> str | None:
        document, _ = self.backend.read()
        current = document.get("lease")
        if not is_json_object(current) or _lease_expiry(current) <= time.time():
            return None
        value = current.get("holder")
        return value if isinstance(value, str) and value else None

    def require_lease(self) -> None:
        """Fail if this store no longer owns a live fleet mutation lease."""

        if self.lease_holder() != self._holder():
            raise StaleFenceError("Eyvan fleet mutation lease ownership was lost")

    @contextmanager
    def exclusive(self):
        with self._exclusive_lock:
            if not self.acquire_lease():
                raise FleetLeaseBusyError("Eyvan fleet mutation lease is busy")
            stop_renewal = Event()
            renewal_errors: list[BaseException] = []

            def renew() -> None:
                interval = min(max(self.lease_ttl_s / 3, 0.001), 30.0)
                while not stop_renewal.wait(interval):
                    try:
                        acquired = self.acquire_lease()
                    except BaseException as exc:
                        renewal_errors.append(exc)
                        return
                    if stop_renewal.is_set():
                        self.release_lease()
                        return
                    if not acquired:
                        renewal_errors.append(StaleFenceError("Eyvan fleet mutation lease ownership was lost"))
                        return

            renewal = Thread(target=renew, name="eyvan-fleet-lease-renewal", daemon=True)
            renewal.start()
            try:
                yield
                if renewal_errors:
                    raise StaleFenceError("Eyvan fleet mutation lease renewal failed") from renewal_errors[0]
                self.require_lease()
            finally:
                stop_renewal.set()
                renewal.join()
                self.release_lease()

    @contextmanager
    def workload_submission(
        self,
        workload_id: str,
        cell_id: str,
        remote_id: str,
        generation: int,
        fencing_token: str,
    ):
        with self.exclusive():
            fence = self.claim_workload_fence(
                workload_id,
                generation,
                fencing_token,
                remote_id=remote_id,
                cell_id=cell_id,
            )
            yield _predecessors(fence)

    @contextmanager
    def gang_workload_submission(
        self,
        workload_id: str,
        *,
        remote_id: str,
        coordinator_cell_id: str,
        members: tp.Sequence[tp.Mapping[str, JsonValue]],
    ):
        """Hold the fleet mutation lease through a fenced multi-cell launch."""

        with self.exclusive():
            fence = self.claim_gang_workload_fence(
                workload_id,
                remote_id=remote_id,
                coordinator_cell_id=coordinator_cell_id,
                members=members,
            )
            yield _predecessors(fence)
