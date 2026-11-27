# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Native systemd transient-service executor for single-host Eyvan cells.

Each Eyvan submission receives its own deterministic transient service unit.
Systemd owns the process cgroup, so stopping a workload also stops background
children created by its shell entrypoint. Durable records retain the fencing
contract and prevent a missing unit from being mistaken for a successful run.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import selectors
import shlex
import stat
import subprocess
import tempfile
import time
import typing as tp
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from eyvan._internal.compat import UTC
from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object
from eyvan._internal.workspace import create_private_directory
from eyvan.capacity.drivers.base import (
    ExecutionWorkloadObservation,
    ExecutionWorkloadStatus,
    MutationContext,
    StaleMutationError,
)
from eyvan.secrets import redact_secret_material
from eyvan.worker.distributed_environment import without_distributed_topology_overrides
from eyvan.worker.runtime.log_records import LogPage, parse_journal_json
from eyvan.worker.runtime.preemption import (
    CHECKPOINT_RECEIPT_PATH_ENV,
    RECLAIM_NOTICE_PATH_ENV,
    read_checkpoint_receipt,
)
from eyvan.worker.runtime.secret_materialization import (
    SecretMaterializationError,
    materialize_secrets,
    secret_environment_keys,
)

_TERMINAL_STATES = frozenset(
    {
        ExecutionWorkloadStatus.SUCCEEDED,
        ExecutionWorkloadStatus.FAILED,
        ExecutionWorkloadStatus.STOPPED,
        ExecutionWorkloadStatus.LOST,
    }
)
_FINAL_STATES = _TERMINAL_STATES | {ExecutionWorkloadStatus.INDETERMINATE}
_COLLECTABLE_STATES = frozenset(
    {
        ExecutionWorkloadStatus.SUCCEEDED,
        ExecutionWorkloadStatus.FAILED,
        ExecutionWorkloadStatus.STOPPED,
    }
)
_DURABLE_STATES = _FINAL_STATES | {ExecutionWorkloadStatus.PENDING, ExecutionWorkloadStatus.RUNNING}
_RECOVERABLE_CGROUP_INDETERMINATE = "cannot observe systemd workload cgroup before completion"

# Indeterminacy that a later observation can genuinely resolve. INDETERMINATE is
# otherwise latched as final, which is right when the record itself is unusable
# but wrong when the runtime merely could not be reached at one instant: a stop
# that failed leaves a workload whose unit is still observable, and latching then
# makes the first transient failure permanent and strands the accelerator.
_UNRECOVERABLE_INDETERMINATE_FIELD = "indeterminate_unrecoverable"

#: ActiveState values that settle what happened to a workload on their own. A
#: unit reporting one of these is answering the question even if systemd could
#: not fully load the unit file.
_DEFINITIVE_ACTIVE_STATES = frozenset({"failed", "inactive"})

#: ExecMainCode values representing a normal process exit (``CLD_EXITED`` is
#: reported as ``1`` by ``systemctl show`` and ``exited`` by test doubles).
_EXITED_EXEC_MAIN_CODES = frozenset({"1", "exited"})


def _is_definitive_unloadable_unit(properties: tp.Mapping[str, str]) -> bool:
    """Return whether an unloadable unit still reports a definitive outcome.

    A unit whose ``LoadState`` is ``bad-setting``, ``error``, or ``masked``
    still reports its live runtime properties. Terminal ``ActiveState`` values
    (``failed``, ``inactive``) and a cleanly exited ``RemainAfterExit=yes``
    service (``ActiveState=active``, ``SubState=exited``, ``Result=success``,
    ``ExecMainCode`` in ``{1, exited}``, ``ExecMainStatus=0``) settle what
    happened to the main process; ``_status_from_properties`` still checks the
    cgroup on systemd 249 before declaring ``SUCCEEDED``. A still-running or
    ambiguous unloadable unit returns ``False`` so it stays unreadable.
    """

    active = properties.get("ActiveState", "").lower()
    if active in _DEFINITIVE_ACTIVE_STATES:
        return True
    if active != "active" or properties.get("SubState", "").lower() != "exited":
        return False
    return (
        properties.get("Result", "").lower() == "success"
        and properties.get("ExecMainCode", "").lower() in _EXITED_EXEC_MAIN_CODES
        and properties.get("ExecMainStatus", "").strip() == "0"
    )


#: Journal lines read when naming why a service died. Enough to hold any Python
#: traceback plus the systemd noise around it, small enough to keep the status
#: read cheap on a host that may be stopping a whole multi-host job at once.
_FAILURE_JOURNAL_LINES = 40

#: How much of the worker's closing line a failure message carries. The line is
#: the diagnosis; a novel-length stack frame helps nobody reading a status row.
_FAILURE_EXCERPT_LIMIT = 240

#: A traceback's closing line, e.g. ``ModuleNotFoundError: No module named
#: 'spectrax'``. Anchored and shape-checked rather than substring-matched so a
#: traceback *frame* line or incidental prose never poses as the reason.
_FAILURE_EXCEPTION_LINE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Interrupt|Exit|Timeout):")
#: An exception named mid-line. Python's own launcher failures read
#: ``/path/bin/python: Error while finding module specification for 'x'
#: (ModuleNotFoundError: No module named 'eyvan')`` -- the reason is there, just
#: not at the start of the line.
_FAILURE_EXCEPTION_ANYWHERE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception):")
#: systemd's own narration of the unit's lifecycle. It is always the last thing
#: in the journal of a failed unit, and it is never why the unit failed.
_SYSTEMD_LIFECYCLE_LINE = re.compile(
    r"^(?:\S+\.service: (?:Main process exited|Failed with result|Deactivated|Consumed|Scheduled restart)"
    # systemd names the unit it started or stopped -- a transient unit by its
    # command line or its ``.service`` name -- so a job's own "Starting epoch 3"
    # is not mistaken for it.
    r"|(?:Started|Starting|Stopped|Stopping) (?:/|\S+\.service\b))"
)


def _is_recoverable_indeterminate(record: tp.Mapping[str, JsonValue]) -> bool:
    """Return whether a later observation may still resolve this indeterminacy.

    Recoverable is the default. The line is between "the thing is gone" and "I
    could not look just now": a vanished unit or an unreadable record can never
    be resolved by another observation, but a failed inspection or a failed stop
    says nothing about the workload and must not latch -- latching it is what
    turns one transient error into a stranded accelerator.

    Recorded as an explicit field rather than inferred from the message, because
    ``stop`` rewrites ``message`` while leaving the status in place -- a text
    comparison silently stops matching at exactly the moment it is needed. The
    *unrecoverable* case is the one stored, so records written before this
    existed read as recoverable, which is the safe direction: recovery
    re-derives from reality instead of assuming an answer.
    """

    return record.get(_UNRECOVERABLE_INDETERMINATE_FIELD) is not True


_UNIT_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_ENVIRONMENT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_SYSTEMD_IDENTITY = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")
_ATOMIC_ENV_TEMPORARY = re.compile(r"\.(?P<digest>[0-9a-f]{64})\.env\.[A-Za-z0-9_-]+\Z")
_ATOMIC_JSON_TEMPORARY = re.compile(r"\.(?P<digest>[0-9a-f]{64})\.json\.[A-Za-z0-9_-]+\Z")
_STARTUP_LOCK_NAME = ".startup.lock"
_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
)


@dataclass(frozen=True, slots=True)
class SystemdCommandResult:
    """Result returned by the small systemd command-runner seam."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


SystemdCommandRunner = tp.Callable[[tuple[str, ...]], SystemdCommandResult]
CgroupProcessesReader = tp.Callable[[str], bool | None]
CgroupPidsReader = tp.Callable[[str], tuple[int, ...] | None]


class SystemdLogsUnavailableError(RuntimeError):
    """journalctl could not read a workload's journal output.

    A genuinely silent workload returns an empty string from ``logs()``; this
    error keeps a journal transport failure distinguishable from that case.
    """

    def __init__(self, unit_name: str, stderr: str) -> None:
        self.unit_name = unit_name
        self.stderr = redact_secret_material(stderr)
        detail = self.stderr or "journalctl failed"
        super().__init__(f"cannot read systemd journal for unit {unit_name!r}: {detail}")


class _InvalidDurableRecord(ValueError):
    """A persisted systemd record cannot safely describe a workload."""


@dataclass(frozen=True, slots=True)
class _LogicalWorkloadPredecessor:
    """Validated owner of the prior logical-workload generation."""

    generation: int
    fencing_token: str
    remote_id: str


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _default_state_dir() -> Path:
    configured = os.getenv("EYVAN_SYSTEMD_RUNTIME_DIR")
    if configured:
        return Path(configured).expanduser()
    agent_state = os.getenv("EYVAN_AGENT_STATE_DIR")
    if agent_state:
        return Path(agent_state).expanduser() / "systemd-runtime"
    return Path.home() / ".local" / "state" / "eyvan" / "systemd-runtime"


def _record_name(remote_id: str) -> str:
    return hashlib.sha256(remote_id.encode("utf-8")).hexdigest()


def systemd_unit_name(remote_id: str) -> str:
    """Return a deterministic, systemd-safe transient service unit name."""

    if not isinstance(remote_id, str) or not remote_id.strip():
        raise ValueError("systemd submission ID must be a non-empty string")
    normalized = _UNIT_SAFE.sub("-", remote_id.strip()).strip(".-") or "workload"
    normalized = normalized[:48].rstrip(".-") or "workload"
    digest = hashlib.sha256(remote_id.encode("utf-8")).hexdigest()[:16]
    return f"eyvan-{normalized}-{digest}.service"


_DURABLE_DIAGNOSTIC_FIELDS = frozenset({"detail", "error", "message", "stderr", "stdout", "unit_collection_error"})
_OUTPUT_SUPPRESSED = "output suppressed for secret-bearing execution"


def _contains_secret_scaffolding(value: JsonValue) -> bool:
    if isinstance(value, dict):
        return any(
            (isinstance(key, str) and key.startswith("EYVAN_SECRET_")) or _contains_secret_scaffolding(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_secret_scaffolding(item) for item in value)
    return False


def _record_secret_bearing(record: tp.Mapping[str, JsonValue]) -> bool:
    """Fail closed for malformed flags and detect N-1 dispatch records."""

    flag = record.get("secret_bearing")
    return (
        flag is True or (flag is not None and not isinstance(flag, bool)) or _contains_secret_scaffolding(dict(record))
    )


def _upgrade_legacy_secret_record(record: JsonObject) -> None:
    """Persist the N-1 scaffolding signal before removing capability fields."""

    record["secret_bearing"] = _record_secret_bearing(record)


def _sanitize_durable_value(
    value: JsonValue,
    *,
    field: str | None = None,
    suppress_diagnostics: bool = False,
) -> JsonValue:
    """Return JSON with secret dispatch scaffolding and diagnostics removed."""

    if isinstance(value, dict):
        return {
            key: _sanitize_durable_value(item, field=key, suppress_diagnostics=suppress_diagnostics)
            for key, item in value.items()
            if not key.startswith("EYVAN_SECRET_")
        }
    if isinstance(value, list):
        return [_sanitize_durable_value(item, field=field, suppress_diagnostics=suppress_diagnostics) for item in value]
    if isinstance(value, str) and field in _DURABLE_DIAGNOSTIC_FIELDS:
        return _OUTPUT_SUPPRESSED if suppress_diagnostics else redact_secret_material(value)
    return value


def _sanitize_durable_record(value: tp.Mapping[str, JsonValue]) -> JsonObject:
    copied: JsonObject = dict(value)
    _upgrade_legacy_secret_record(copied)
    return tp.cast(
        "JsonObject",
        _sanitize_durable_value(copied, suppress_diagnostics=_record_secret_bearing(copied)),
    )


def _open_directory_no_follow(directory: Path) -> int:
    """Open one directory without resolving a final-component symlink."""

    descriptor = os.open(directory, _DIRECTORY_OPEN_FLAGS)
    if not stat.S_ISDIR(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise RuntimeError("runtime state path is not a directory")
    return descriptor


def _ensure_private_directory(directory: Path) -> None:
    """Create or tighten one process-owned runtime-state directory safely."""

    create_private_directory(directory)
    try:
        descriptor = _open_directory_no_follow(directory)
    except OSError:
        raise RuntimeError("runtime state directory is unsafe") from None
    try:
        metadata = os.fstat(descriptor)
        if metadata.st_uid != os.geteuid():
            raise RuntimeError("runtime state directory is not owned by this process identity")
        os.fchmod(descriptor, 0o700)
    finally:
        os.close(descriptor)


def _fsync_directory(directory: Path) -> None:
    descriptor = _open_directory_no_follow(directory)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_unlink(path: Path) -> bool:
    """Unlink one file and make the directory entry removal crash-durable."""

    try:
        path.unlink()
    except FileNotFoundError:
        return False
    _fsync_directory(path.parent)
    return True


def _startup_lock_path(path: Path) -> Path:
    """Return the state-wide lock shared by atomic writers and startup."""

    return path.parent.parent / _STARTUP_LOCK_NAME


@contextmanager
def _atomic_writer_exclusion(path: Path) -> tp.Iterator[None]:
    """Keep startup's exclusive sweep outside one complete temp lifecycle."""

    # Production callers already hold the job or workload lock. Successful
    # acquisition of this shared lock is the writer lifecycle's linearization
    # boundary: it happens before temp creation and remains held through publish
    # or cleanup. Startup takes the same lock exclusively only while sweeping and
    # releases it before acquiring any record lock.
    with _file_lock(_startup_lock_path(path), shared=True):
        yield


def _atomic_write_json(path: Path, value: tp.Mapping[str, JsonValue]) -> None:
    _ensure_private_directory(path.parent)
    with _atomic_writer_exclusion(path):
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        published = False
        try:
            os.fchmod(descriptor, 0o600)
            stream = os.fdopen(descriptor, "w", encoding="utf-8")
            descriptor = -1
            with stream:
                json.dump(_sanitize_durable_record(value), stream, sort_keys=True, separators=(",", ":"))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            published = True
            _fsync_directory(path.parent)
        except BaseException:
            if not published:
                try:
                    _durable_unlink(temporary)
                except BaseException as cleanup_error:
                    raise RuntimeError("could not durably remove an atomic JSON temporary file") from cleanup_error
            # Once replace succeeds this path is a complete, fsynced JSON document.
            # Removing it after a directory-fsync error could erase the last usable
            # lifecycle record; leave that valid record in place and fail the caller.
            raise
        finally:
            if descriptor >= 0:
                os.close(descriptor)


def _systemd_environment_value(value: str) -> str:
    """Encode a value for EnvironmentFile, not Docker's literal env grammar."""

    if not (value.startswith(("'", '"')) or "\\" in value or value != value.strip(" \t")):
        return value
    # Double-quoted EnvironmentFile values preserve whitespace. Backslashes
    # escape only these syntax characters; dollar/backtick expansion does not
    # occur, and encoding them explicitly preserves their literal bytes.
    escaped = "".join("\\" + character if character in '\\"$`' else character for character in value)
    return '"' + escaped + '"'


def _escape_systemd_exec_arg(value: str) -> str:
    """Escape ``$`` for ``systemd-run`` ``ExecStart`` argv elements.

    PID 1 runs ``replace_env_arg()`` on every ``ExecStart`` argument before
    ``execve()``, expanding ``$VAR`` and ``${...}`` (including bash syntax such
    as ``${#UV[@]}``) unless every ``$`` is doubled to ``$$``. Specifier ``%``
    tokens are not expanded on D-Bus positional argv and are left untouched.
    """

    return value.replace("$", "$$")


#: Inline ``--setup`` JSON payloads larger than this or containing ``%`` are
#: staged into a file and passed via ``--setup-file`` rather than inline in
#: ``systemd-run`` ``ExecStart=`` argv. On systemd 249, re-parsing
#: ``/run/systemd/transient/*.service`` runs ``unit_full_printf()`` over
#: ``ExecStart=`` and flips ``LoadState`` to ``bad-setting`` on unescaped ``%``
#: specifiers or oversized unit settings (such as the ~16 KB ``EasyDeLSetup``
#: JSON containing bash ``${line%%;*}`` syntax).
_INLINE_SETUP_LIMIT_BYTES = 1024
_DEPENDENCY_ENTRY_MODULE = "eyvan.worker.runtime.dependency_entry"
_DEPENDENCY_ENTRY_SCRIPT_SUFFIX = "/dependency_entry.py"
_SETUP_SHELL_CONTROL_TOKENS = frozenset({"&", "&&", "|", "|&", ";", ";;", "(", ")"})
_SETUP_STANDALONE_REDIRECT = re.compile(r"\d*(>>?|<<?)|\d*>&\d*")


def _should_stage_setup_payload(payload: str) -> bool:
    """Return whether a ``dependency_entry --setup`` JSON payload should be staged to disk."""

    return "%" in payload or len(payload.encode("utf-8")) > _INLINE_SETUP_LIMIT_BYTES


def _extract_stageable_setup_payload(entrypoint: str) -> tuple[list[str], int, int, str] | None:
    """Locate an inline ``dependency_entry --setup`` argument that should be staged to a file.

    Returns ``(tokens, flag_index, end_index, payload)`` describing the
    ``--setup <json>`` (or ``--setup=<json>``) slice to replace with
    ``("--setup-file", path)``, or ``None`` when the entrypoint is not a
    ``dependency_entry`` invocation or its setup payload is small and free of ``%``.
    """

    try:
        parts = shlex.split(entrypoint)
    except ValueError:
        return None
    if any(part in _SETUP_SHELL_CONTROL_TOKENS or _SETUP_STANDALONE_REDIRECT.fullmatch(part) for part in parts):
        return None

    options_start: int | None = None
    for index, part in enumerate(parts):
        if part == "--":
            break
        if index >= 2 and parts[index - 1] == "-m" and part == _DEPENDENCY_ENTRY_MODULE:
            options_start = index + 1
            break
        if index >= 1 and part.endswith(_DEPENDENCY_ENTRY_SCRIPT_SUFFIX):
            options_start = index + 1
            break
    if options_start is None:
        return None

    cursor = options_start
    while cursor < len(parts) and parts[cursor] != "--":
        token = parts[cursor]
        if token == "--require":
            cursor += 2
            continue
        if token.startswith("--require="):
            cursor += 1
            continue
        if token == "--setup" and cursor + 1 < len(parts):
            payload = parts[cursor + 1]
            return (parts, cursor, cursor + 2, payload) if _should_stage_setup_payload(payload) else None
        if token.startswith("--setup="):
            payload = token.removeprefix("--setup=")
            return (parts, cursor, cursor + 1, payload) if _should_stage_setup_payload(payload) else None
        cursor += 1
    return None


def _ensure_setup_directory(directory: Path, *, mode: int) -> None:
    """Create or tighten one setup-staging directory without following symlinks."""

    if mode == 0o700:
        _ensure_private_directory(directory)
        return
    directory.mkdir(parents=True, exist_ok=True, mode=mode)
    try:
        descriptor = _open_directory_no_follow(directory)
    except OSError:
        raise RuntimeError("runtime setup directory is unsafe") from None
    try:
        metadata = os.fstat(descriptor)
        if metadata.st_uid != os.geteuid():
            raise RuntimeError("runtime setup directory is not owned by this process identity")
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)


def _atomic_write_setup_file(path: Path, payload: str, *, dir_mode: int, file_mode: int) -> None:
    """Atomically write one staged ``dependency_entry --setup-file`` JSON document."""

    _ensure_setup_directory(path.parent, mode=dir_mode)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    published = False
    try:
        os.fchmod(descriptor, file_mode)
        stream = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = -1
        with stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        published = True
        _fsync_directory(path.parent)
    except BaseException:
        if not published:
            try:
                _durable_unlink(temporary)
            except BaseException as cleanup_error:
                raise RuntimeError("could not durably remove a staged setup temporary file") from cleanup_error
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _write_environment_file(path: Path, environment: tp.Mapping[str, str], *, systemd: bool = False) -> None:
    """Atomically write one owner-only env file without line injection.

    Docker consumes literal assignment values. Native systemd callers opt into
    EnvironmentFile quoting; sharing that encoding would corrupt Docker values.
    """

    lines: list[str] = []
    for name, value in sorted(environment.items()):
        if not isinstance(name, str) or not _ENVIRONMENT_NAME.fullmatch(name):
            raise ValueError("environment variable names must be valid shell identifiers")
        if not isinstance(value, str) or "\x00" in value or "\n" in value or "\r" in value:
            raise ValueError(f"environment variable {name!r} must be a NUL- and newline-free string")
        encoded = _systemd_environment_value(value) if systemd else value
        lines.append(f"{name}={encoded}")
    _ensure_private_directory(path.parent)
    with _atomic_writer_exclusion(path):
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(temporary_name)
        published = False
        try:
            os.fchmod(descriptor, 0o600)
            stream = os.fdopen(descriptor, "w", encoding="utf-8")
            descriptor = -1
            with stream:
                stream.write("".join(f"{line}\n" for line in lines))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            published = True
            _fsync_directory(path.parent)
        except BaseException:
            # Unlike a JSON lifecycle record, this file may contain plaintext. A
            # failed post-rename durability barrier must not leave the published
            # path available to a workload or across the next process restart.
            cleanup_path = path if published else temporary
            try:
                _durable_unlink(cleanup_path)
            except BaseException as cleanup_error:
                # The unlink may already be visible but not crash-durable. Never
                # allow launch to continue on that ambiguity.
                raise RuntimeError("could not durably remove a failed workload environment file") from cleanup_error
            raise
        finally:
            if descriptor >= 0:
                os.close(descriptor)


def _load_json(path: Path) -> JsonObject | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not is_json_object(payload):
        raise RuntimeError(f"systemd execution record {path} is not a JSON object")
    return payload


def _persisted_status(remote_id: str, record: tp.Mapping[str, JsonValue]) -> ExecutionWorkloadStatus:
    """Parse one required durable status without inventing a fallback."""

    try:
        raw_status = record["status"]
    except KeyError as exc:
        raise _InvalidDurableRecord(f"systemd execution {remote_id!r} has no persisted status") from exc
    if not isinstance(raw_status, str):
        raise _InvalidDurableRecord(f"systemd execution {remote_id!r} has a non-string persisted status")
    try:
        status = ExecutionWorkloadStatus(raw_status)
    except ValueError as exc:
        raise _InvalidDurableRecord(
            f"systemd execution {remote_id!r} has unsupported persisted status {raw_status!r}"
        ) from exc
    if status not in _DURABLE_STATES:
        raise _InvalidDurableRecord(f"systemd execution {remote_id!r} has unsupported durable status {status.value!r}")
    return status


def _persisted_message(remote_id: str, record: tp.Mapping[str, JsonValue]) -> str:
    """Read the required durable message without coercing its type."""

    try:
        message = record["message"]
    except KeyError as exc:
        raise _InvalidDurableRecord(f"systemd execution {remote_id!r} has no persisted message") from exc
    if not isinstance(message, str):
        raise _InvalidDurableRecord(f"systemd execution {remote_id!r} has a non-string persisted message")
    return _OUTPUT_SUPPRESSED if _record_secret_bearing(record) else redact_secret_material(message)


def _persisted_remote_id(expected_remote_id: str, record: tp.Mapping[str, JsonValue]) -> None:
    """Verify that a hashed record still belongs to the requested execution."""

    try:
        remote_id = record["remote_id"]
    except KeyError as exc:
        raise _InvalidDurableRecord(f"systemd execution {expected_remote_id!r} has no persisted remote ID") from exc
    if not isinstance(remote_id, str) or not remote_id:
        raise _InvalidDurableRecord(f"systemd execution {expected_remote_id!r} has an invalid persisted remote ID")
    if remote_id != expected_remote_id:
        raise _InvalidDurableRecord(f"systemd execution record belongs to {remote_id!r}, not {expected_remote_id!r}")


def _durable_observation(remote_id: str, record: tp.Mapping[str, JsonValue]) -> ExecutionWorkloadObservation:
    """Return a complete observation from a validated durable record."""

    _persisted_remote_id(remote_id, record)
    status = _persisted_status(remote_id, record)
    message = _persisted_message(remote_id, record)
    if status is ExecutionWorkloadStatus.INDETERMINATE and not message:
        raise _InvalidDurableRecord(f"systemd execution {remote_id!r} has an indeterminate status without a reason")
    return ExecutionWorkloadObservation(
        remote_id=remote_id,
        status=status,
        message=message,
    )


def _known_persisted_status(record: tp.Mapping[str, JsonValue]) -> ExecutionWorkloadStatus | None:
    """Return an existing valid durable status without treating invalid data as a state."""

    try:
        return _persisted_status("<record>", record)
    except _InvalidDurableRecord:
        return None


def _required_record_integer(record: tp.Mapping[str, JsonValue], field: str) -> int:
    """Read a JSON integer field without accepting booleans or coercions."""

    try:
        value = record[field]
    except KeyError as exc:
        raise ValueError(f"record has no {field!r} field") from exc
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"record has a non-integer {field!r} field")
    return value


def _required_record_string(record: tp.Mapping[str, JsonValue], field: str) -> str:
    """Read a JSON string field without coercing an arbitrary value."""

    try:
        value = record[field]
    except KeyError as exc:
        raise ValueError(f"record has no {field!r} field") from exc
    if not isinstance(value, str):
        raise ValueError(f"record has a non-string {field!r} field")
    return value


@contextmanager
def _file_lock(path: Path, *, shared: bool = False) -> tp.Iterator[None]:
    _ensure_private_directory(path.parent)
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    locked = False
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_uid != os.geteuid() or metadata.st_nlink != 1:
            raise RuntimeError("runtime state lock is unsafe")
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
        locked = True
        yield
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _was_retired(record: tp.Mapping[str, JsonValue]) -> bool:
    """Return whether Eyvan itself retired this execution, as opposed to it ending."""

    return record.get("retired") is True


#: How many incarnations back a retire key stays bound. Each partial-launch
#: rollback carries a fresh key, so without a bound the replay ledger grows by
#: one entry per rollback for the life of the record. A duplicate retire that
#: arrives more than this many full retire-and-relaunch cycles late is no longer
#: recognized as a replay.
_RETIRE_KEY_INCARNATION_WINDOW = 8


def _incarnation(record: tp.Mapping[str, JsonValue]) -> int:
    """Read the relaunch counter, preserving records written before it existed."""

    value = record.get("incarnation", 0)
    if type(value) is not int or value < 0:
        raise RuntimeError("invalid systemd execution incarnation")
    return value


def _record_lock_path(record_path: Path) -> Path:
    return record_path.with_suffix(".lock")


def _sweep_atomic_temporary_directory(
    directory: Path,
    *,
    pattern: re.Pattern[str],
) -> None:
    """Crash-durably remove exact orphan temp names without following entries.

    The caller holds the state-wide startup lock exclusively, while every
    conforming atomic writer holds it shared for its whole temporary-file
    lifecycle. No writer can therefore create a matching entry after this
    snapshot and before the startup phase declares the directory clean. Hidden
    names outside the exact atomic-writer grammar fail closed.
    """

    descriptor = _open_directory_no_follow(directory)
    removed = False
    try:
        metadata = os.fstat(descriptor)
        if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise RuntimeError("runtime temporary directory is not private")
        with os.scandir(descriptor) as entries:
            names = sorted(entry.name for entry in entries)
        for name in names:
            if pattern.fullmatch(name) is None:
                if name.startswith("."):
                    raise RuntimeError("runtime temporary directory contains an unexpected hidden entry")
                continue
            try:
                entry_metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if (
                not stat.S_ISREG(entry_metadata.st_mode)
                or entry_metadata.st_uid != os.geteuid()
                or entry_metadata.st_nlink != 1
            ):
                raise RuntimeError("runtime atomic temporary entry is unsafe")
            try:
                os.unlink(name, dir_fd=descriptor)
            except FileNotFoundError:
                continue
            removed = True
    finally:
        try:
            # Also persist removals made before a later unexpected entry made
            # startup fail closed; every unlink this invocation performed must
            # have the same crash-durability guarantee.
            if removed:
                os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _read_boot_id() -> str | None:
    """Return the Linux boot ID when available.

    A transient systemd unit cannot survive a reboot. Persisting this marker
    lets callers record a definitive ``LOST`` result instead of treating the
    post-reboot missing unit as a completed job. Non-Linux hosts simply omit
    the marker.
    """

    try:
        value = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def _cgroup_pids(control_group: str) -> tuple[int, ...] | None:
    """Return the PIDs in a systemd cgroup, or ``None`` when it is unreadable.

    ``None`` is deliberately distinct from an empty tuple: an unreadable,
    missing, or malformed cgroup is ambiguous, not proof that nothing is
    running. Collapsing the two would let a background child be reported as a
    completed workload.
    """

    if not isinstance(control_group, str) or not control_group.startswith("/"):
        return None
    parts = tuple(part for part in control_group.split("/") if part)
    if not parts or any(part in {".", ".."} for part in parts):
        return None
    try:
        contents = (Path("/sys/fs/cgroup").joinpath(*parts) / "cgroup.procs").read_text(encoding="utf-8")
    except OSError:
        # None, not (): an unreadable cgroup is ambiguous, not proof that
        # nothing is running. Returning an empty tuple makes
        # _cgroup_has_processes report False, which _status_from_properties maps
        # to SUCCEEDED -- so a permission-denied or namespace-hidden cgroup on a
        # hung collective would be reported as a completed workload.
        return None
    pids: list[int] = []
    for line in contents.splitlines():
        candidate = line.strip()
        if not candidate:
            continue
        try:
            pids.append(int(candidate))
        except ValueError:
            # A non-numeric line means the file is not what we think it is;
            # treat the whole reading as ambiguous rather than partially trust it.
            return None
    return tuple(pids)


def _cgroup_has_processes(control_group: str) -> bool | None:
    """Return whether a systemd cgroup still has processes."""

    pids = _cgroup_pids(control_group)
    return None if pids is None else bool(pids)


def _environment(spec: tp.Mapping[str, JsonValue]) -> dict[str, str]:
    runtime_env = spec.get("runtime_env")
    if runtime_env is None:
        runtime_env = {}
    if not isinstance(runtime_env, dict):
        raise ValueError("native-systemd runtime_env must be a mapping")
    unsupported = sorted(key for key in runtime_env if key != "env_vars")
    if unsupported:
        raise ValueError(f"native-systemd execution does not support runtime_env field {unsupported[0]!r}")
    legacy = runtime_env.get("env_vars")
    neutral = spec.get("environment")
    if legacy is None:
        legacy = {}
    if neutral is None:
        neutral = {}
    if not isinstance(legacy, dict) or not isinstance(neutral, dict):
        raise ValueError("native-systemd environment variables must be a mapping")

    result: dict[str, str] = {}
    for raw_name, raw_value in (*legacy.items(), *neutral.items()):
        if not isinstance(raw_name, str) or not _ENVIRONMENT_NAME.fullmatch(raw_name):
            raise ValueError("native-systemd environment variable names must be valid shell identifiers")
        if not isinstance(raw_value, str):
            raise ValueError(f"native-systemd environment variable {raw_name!r} must have a string value")
        if "\n" in raw_value or "\r" in raw_value:
            raise ValueError(f"native-systemd environment variable {raw_name!r} must not contain a newline")
        result[raw_name] = raw_value
    result.setdefault("EYVAN_REPLICA_INDEX", "0")
    result.setdefault("EYVAN_REPLICA_COUNT", "1")
    return result


def _validate_gang(spec: tp.Mapping[str, JsonValue]) -> None:
    gang = spec.get("gang")
    if not isinstance(gang, dict):
        raise ValueError("native-systemd workloads require a gang resource contract")
    replicas = gang.get("replicas")
    if isinstance(replicas, bool) or not isinstance(replicas, int):
        raise ValueError("native-systemd gang replicas must be an integer")
    if replicas != 1:
        raise ValueError("native-systemd execution supports exactly one replica")
    if not isinstance(gang.get("per_replica"), dict):
        raise ValueError("native-systemd gang per_replica resources must be a mapping")
    topology = gang.get("required_topology")
    if topology is None:
        topology = {}
    if not isinstance(topology, dict):
        raise ValueError("native-systemd required topology must be a mapping")
    if topology:
        raise ValueError("native-systemd execution does not support topology constraints")


def _parse_properties(payload: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in payload.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key] = value
    return result


def _systemd_runner(command: tuple[str, ...], *, timeout_s: float) -> SystemdCommandResult:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout_s)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return SystemdCommandResult(returncode=127, stderr=str(exc))
    return SystemdCommandResult(
        returncode=result.returncode,
        stdout=result.stdout or "",
        stderr=result.stderr or "",
    )


def _systemd_journal_runner(command: tuple[str, ...], *, limit: int, timeout_s: float) -> SystemdCommandResult:
    """Read only the first requested journal entries with bounded pipe buffers.

    journalctl's finite --lines option selects the tail, including after a
    cursor. Forward continuation therefore needs --lines=all, but must stop
    ingestion itself rather than capturing the entire unread backlog.
    """

    maximum_bytes = 8 * 1024 * 1024
    output = bytearray()
    errors = bytearray()
    deadline = time.monotonic() + timeout_s
    process: subprocess.Popen[bytes] | None = None
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        assert process.stdout is not None and process.stderr is not None
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            lines = 0
            scanned = 0
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return SystemdCommandResult(returncode=127, stderr="journalctl page read timed out")
                for key, _events in selector.select(remaining):
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    if key.data == "stderr":
                        errors.extend(chunk)
                        del errors[:-8192]
                        continue
                    output.extend(chunk)
                    while True:
                        newline = output.find(b"\n", scanned)
                        if newline < 0:
                            break
                        scanned = newline + 1
                        lines += 1
                        if lines >= max(1, limit):
                            if scanned > maximum_bytes:
                                return SystemdCommandResult(returncode=127, stderr="journalctl page exceeds byte limit")
                            return SystemdCommandResult(returncode=0, stdout=output[:scanned].decode("utf-8", "replace"))
                    if len(output) > maximum_bytes:
                        return SystemdCommandResult(returncode=127, stderr="journalctl page exceeds byte limit")
        returncode = process.wait(timeout=max(0.001, deadline - time.monotonic()))
        return SystemdCommandResult(
            returncode=returncode,
            stdout=output.decode("utf-8", "replace"),
            stderr=errors.decode("utf-8", "replace"),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return SystemdCommandResult(returncode=127, stderr=str(exc))
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
            process.wait()
            if process.stdout is not None:
                process.stdout.close()
            if process.stderr is not None:
                process.stderr.close()


class SystemdTransientServiceExecutor:
    """Execute fenced native workloads as durable systemd transient services.

    The executor supports one local replica. On systemd 250 and later,
    ``ExitType=cgroup`` keeps a unit nonterminal until every workload process
    has exited, including background children created by its shell entrypoint.
    On older managers, Eyvan queries the unit's ``ControlGroup`` before
    accepting an exited main process as complete. ``RemainAfterExit=yes``
    keeps the completed unit observable until Eyvan persists its terminal
    state. ``KillMode=control-group`` makes `systemctl stop` terminate that
    whole cgroup.
    """

    def __init__(
        self,
        state_dir: str | os.PathLike[str] | None = None,
        *,
        command_runner: SystemdCommandRunner | None = None,
        boot_id_reader: tp.Callable[[], str | None] | None = None,
        cgroup_processes_reader: CgroupProcessesReader | None = None,
        cgroup_pids_reader: CgroupPidsReader | None = None,
        failure_journal_reader: tp.Callable[[str], str | None] | None = None,
        use_exit_type_cgroup: bool = False,
        workload_user: str | None = None,
        workload_group: str | None = None,
        workload_directory: str | os.PathLike[str] | None = None,
        base_environment: tp.Mapping[str, str] | None = None,
        command_timeout_s: float = 315.0,
        _defer_startup_recovery: bool = False,
    ) -> None:
        if command_timeout_s <= 0:
            raise ValueError("systemd command timeout must be positive")
        if not isinstance(_defer_startup_recovery, bool):
            raise ValueError("systemd startup recovery deferral must be a boolean")
        if not isinstance(use_exit_type_cgroup, bool):
            raise ValueError("systemd ExitType cgroup setting must be a boolean")
        for label, identity in (("workload user", workload_user), ("workload group", workload_group)):
            if identity is not None and (not isinstance(identity, str) or not _SYSTEMD_IDENTITY.fullmatch(identity)):
                raise ValueError(f"systemd {label} must be a valid local identity")
        if workload_group is not None and workload_user is None:
            raise ValueError("systemd workload group requires a workload user")
        if workload_directory is not None:
            if not isinstance(workload_directory, (str, os.PathLike)) or not Path(workload_directory).is_absolute():
                raise ValueError("systemd workload directory must be absolute")
        if base_environment is not None:
            for name, value in base_environment.items():
                if not isinstance(name, str) or not _ENVIRONMENT_NAME.fullmatch(name):
                    raise ValueError("systemd base environment variable names must be valid shell identifiers")
                if not isinstance(value, str) or any(character in value for character in ("\x00", "\n", "\r")):
                    raise ValueError(
                        f"systemd base environment variable {name!r} must be a NUL- and newline-free string"
                    )
        self.state_dir = Path(state_dir).expanduser() if state_dir is not None else _default_state_dir()
        self.jobs_dir = self.state_dir / "jobs"
        self.workloads_dir = self.state_dir / "workloads"
        self.env_dir = self.state_dir / "env"
        self.setup_dir = self.state_dir / "setup"
        self.notices_dir = self.state_dir / "notices"
        for directory in (self.state_dir, self.jobs_dir, self.workloads_dir, self.env_dir):
            _ensure_private_directory(directory)
        self._command_timeout_s = command_timeout_s
        self._custom_command_runner = command_runner is not None
        self._command_runner = command_runner or (
            lambda command: _systemd_runner(command, timeout_s=self._command_timeout_s)
        )
        self._boot_id_reader = boot_id_reader or _read_boot_id
        self._cgroup_pids_reader = cgroup_pids_reader or _cgroup_pids
        # Best-effort by injection as by nature: a journal that cannot be read
        # must leave the failure message untouched, never fail the status read.
        self._failure_journal_reader = failure_journal_reader or self._read_failure_journal
        # Derive the boolean view from the PID reader unless one is injected, so
        # a test that stubs only one of the two cannot see them disagree.
        self._cgroup_processes_reader = cgroup_processes_reader or (
            _cgroup_has_processes
            if cgroup_pids_reader is None
            else (lambda group: (lambda pids: None if pids is None else bool(pids))(cgroup_pids_reader(group)))
        )
        self._use_exit_type_cgroup = use_exit_type_cgroup
        self._workload_user = workload_user
        self._workload_group = workload_group
        self._workload_directory = str(workload_directory) if workload_directory is not None else None
        self._base_environment = dict(base_environment or {})
        if not _defer_startup_recovery:
            self._recover_startup()

    def _record_path(self, remote_id: str) -> Path:
        return self.jobs_dir / f"{_record_name(remote_id)}.json"

    def _workload_fence_path(self, workload_id: str) -> Path:
        return self.workloads_dir / f"{_record_name(workload_id)}.json"

    def _environment_file_path(self, remote_id: str) -> Path:
        return self.env_dir / f"{_record_name(remote_id)}.env"

    def _environment_file_paths(self, remote_id: str) -> tuple[Path, ...]:
        """Return every owner-only plaintext environment file for one run."""

        return (self._environment_file_path(remote_id),)

    def _notice_file_paths(self, remote_id: str) -> tuple[Path, Path, int]:
        """Return ``(notice_path, receipt_path, notice_mode)`` for one workload run."""

        digest = _record_name(remote_id)
        if self._workload_directory is not None:
            workload_root = Path(self._workload_directory)
            if not workload_root.is_symlink() and workload_root.is_dir():
                directory = workload_root / ".eyvan-notices"
                return directory / f"{digest}.reclaim.json", directory / f"{digest}.checkpoint.json", 0o644
        return self.notices_dir / f"{digest}.reclaim.json", self.notices_dir / f"{digest}.checkpoint.json", 0o600

    def _setup_file_paths(self, remote_id: str) -> tuple[Path, Path, int, int]:
        """Return ``(host_path, workload_path, dir_mode, file_mode)`` for a staged setup JSON.

        On native TPU/GPU hosts ``self.state_dir`` is ``0700`` owned by the
        root worker daemon, so a workload running as ``workload_user`` cannot
        traverse it. When ``workload_directory`` exists on disk (already
        ``0750`` owned by ``workload_user``), stage under
        ``<workload_directory>/.eyvan-setup`` with ``0755``/``0644`` so the
        workload user can read it while other host users cannot enter
        ``workload_directory``. Otherwise stage privately under ``self.setup_dir``.
        """

        filename = f"{_record_name(remote_id)}.json"
        if self._workload_directory is not None:
            workload_root = Path(self._workload_directory)
            if not workload_root.is_symlink() and workload_root.is_dir():
                staged = workload_root / ".eyvan-setup" / filename
                return staged, staged, 0o755, 0o644
        staged = self.setup_dir / filename
        return staged, staged, 0o700, 0o600

    def _setup_file_host_paths(self, remote_id: str) -> tuple[Path, ...]:
        """Return every candidate host path that may hold a staged setup file for ``remote_id``."""

        filename = f"{_record_name(remote_id)}.json"
        paths = [self.setup_dir / filename]
        if self._workload_directory is not None:
            paths.append(Path(self._workload_directory) / ".eyvan-setup" / filename)
        return tuple(paths)

    def _stage_setup_file(self, remote_id: str, spec: tp.Mapping[str, JsonValue]) -> None:
        """Stage a large or ``%``-bearing ``--setup`` payload to disk before launch."""

        entrypoint = spec.get("entrypoint")
        if not isinstance(entrypoint, str):
            return
        extracted = _extract_stageable_setup_payload(entrypoint)
        if extracted is None:
            return
        _parts, _start, _end, payload = extracted
        host_path, _workload_path, dir_mode, file_mode = self._setup_file_paths(remote_id)
        _atomic_write_setup_file(host_path, payload, dir_mode=dir_mode, file_mode=file_mode)

    def _rewrite_staged_setup_entrypoint(self, entrypoint: str, remote_id: str) -> str:
        """Replace a staged inline ``--setup <json>`` option with ``--setup-file <path>``."""

        extracted = _extract_stageable_setup_payload(entrypoint)
        if extracted is None:
            return entrypoint
        parts, start, end, _payload = extracted
        _host_path, workload_path, _dir_mode, _file_mode = self._setup_file_paths(remote_id)
        return shlex.join([*parts[:start], "--setup-file", str(workload_path), *parts[end:]])

    def _remove_setup_files(self, record: tp.Mapping[str, JsonValue], *, runtime_absent: bool = False) -> None:
        """Remove staged setup JSON files once the workload is terminal or absent.

        Unlike ``EnvironmentFile`` (read by PID 1 before ``execve``), a staged
        ``--setup-file`` is read by ``dependency_entry`` after the unit starts,
        so it must remain on disk while the unit is ``RUNNING``.
        """

        status = _known_persisted_status(record)
        if status not in _TERMINAL_STATES and not runtime_absent:
            return
        remote_id = record.get("remote_id")
        if not isinstance(remote_id, str) or not remote_id:
            return
        for path in self._setup_file_host_paths(remote_id):
            try:
                _durable_unlink(path)
            except OSError:
                raise RuntimeError("could not remove staged workload setup file") from None

    def _environment_temporary_directories(self) -> tuple[Path, ...]:
        """Return managed directories containing atomic plaintext env writes."""

        return (self.env_dir,)

    def _sweep_startup_temporaries(self) -> None:
        """Remove exact crash-orphan atomic writes before record recovery."""

        for directory in self._environment_temporary_directories():
            _sweep_atomic_temporary_directory(directory, pattern=_ATOMIC_ENV_TEMPORARY)
        _sweep_atomic_temporary_directory(self.jobs_dir, pattern=_ATOMIC_JSON_TEMPORARY)
        _sweep_atomic_temporary_directory(self.workloads_dir, pattern=_ATOMIC_JSON_TEMPORARY)

    def _recover_startup(self) -> None:
        """Eagerly sweep atomic orphans, then recover every durable job.

        Construction is the startup boundary. Requiring a later status request
        left plaintext indefinitely on idle workers after a restart. Startup
        takes the state-wide lock exclusively for the exact-name sweep; every
        atomic writer holds it shared from before temp creation through publish
        or cleanup. The global lock is released before any per-record lock is
        acquired, preserving the ordinary record-then-global lock order.
        """

        try:
            with _file_lock(self.state_dir / _STARTUP_LOCK_NAME):
                self._sweep_startup_temporaries()
        except Exception:
            raise RuntimeError("could not safely recover atomic runtime temporary files") from None

        for record_path in sorted(self.jobs_dir.glob("*.json")):
            if record_path.is_symlink():
                raise RuntimeError("could not safely recover a systemd execution record")
            with _file_lock(_record_lock_path(record_path)):
                try:
                    record = _load_json(record_path)
                except Exception:
                    raise RuntimeError("could not safely recover a systemd execution record") from None
                if record is None:
                    continue
                _upgrade_legacy_secret_record(record)
                remote_id = record.get("remote_id")
                if (
                    not isinstance(remote_id, str)
                    or not remote_id
                    or self._record_path(remote_id).name != record_path.name
                ):
                    # Sanitization still happens before failing startup, so an
                    # N-1 capability is not left merely because its owner record
                    # is malformed.
                    _atomic_write_json(record_path, record)
                    raise RuntimeError("could not safely recover a systemd execution record")
                _atomic_write_json(record_path, record)
            # ``status`` takes the same lock itself. Never nest flock calls:
            # on some platforms that would release the process-wide lock early.
            self.status(remote_id)

    def _remove_environment_files(
        self,
        record: JsonObject,
        *,
        runtime_absent: bool = False,
        environment_consumed: bool = False,
    ) -> None:
        """Crash-durably unlink plaintext when runtime evidence makes it safe.

        A durable terminal outcome, a positively absent runtime, or a positively
        running runtime (which has consumed its launch environment) is enough.
        An unavailable inspection is not.  This distinction permits cleanup
        while the workload result remains semantically INDETERMINATE without
        guessing that an unobservable process is gone.
        """

        status = _known_persisted_status(record)
        if status not in _TERMINAL_STATES and not runtime_absent and not environment_consumed:
            return
        remote_id = record.get("remote_id")
        if not isinstance(remote_id, str) or not remote_id:
            raise RuntimeError("cannot remove workload environment for a record without a remote ID")
        for path in self._environment_file_paths(remote_id):
            try:
                _durable_unlink(path)
            except OSError:
                raise RuntimeError("could not remove owner-only workload environment file") from None
        # Do not trust this flag to skip a later unlink: a stale file restored
        # beside a durable record must be removed again after a restart.
        record["environment_collected"] = True
        record["updated_at"] = _utcnow()

    @staticmethod
    def _resolved_launch_environment(environment: tp.Mapping[str, str]) -> dict[str, str]:
        """Replace capability scaffolding with resolved values in memory only."""

        resolved = materialize_secrets(environment)
        launch_environment = dict(environment)
        for key in secret_environment_keys(launch_environment):
            launch_environment.pop(key, None)
        launch_environment.update(resolved)
        return launch_environment

    @staticmethod
    def _validate_fence(remote_id: str, record: tp.Mapping[str, JsonValue], context: MutationContext) -> None:
        try:
            generation = _required_record_integer(record, "generation")
            token = _required_record_string(record, "fencing_token")
        except ValueError as exc:
            raise StaleMutationError(f"systemd execution {remote_id!r} has an invalid persisted generation") from exc
        if generation != context.generation:
            raise StaleMutationError(
                f"generation {context.generation} does not own systemd execution {remote_id!r} generation {generation}"
            )
        if not isinstance(token, str) or token != context.fencing_token:
            raise StaleMutationError(
                f"fencing token does not own systemd execution {remote_id!r} generation {generation}"
            )

    @staticmethod
    def _logical_workload_id(metadata: tp.Mapping[str, JsonValue]) -> str | None:
        value = metadata.get("eyvan_workload_id")
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("native-systemd eyvan_workload_id must be a non-empty string")
        return value

    def _run(self, command: tuple[str, ...]) -> SystemdCommandResult:
        try:
            return self._command_runner(command)
        except Exception as exc:
            return SystemdCommandResult(returncode=127, stderr=str(exc))

    def _show_unit(self, unit_name: str) -> tuple[str, dict[str, str] | None, str]:
        """Return ``(outcome, properties, detail)`` for one unit.

        ``outcome`` is ``present``, ``missing``, or ``unavailable``. The last
        category deliberately does not collapse to a finished workload.
        """

        result = self._run(
            (
                "systemctl",
                "show",
                "--no-pager",
                "--property=LoadState",
                "--property=ActiveState",
                "--property=SubState",
                "--property=Result",
                "--property=ExecMainCode",
                "--property=ExecMainStatus",
                "--property=ControlGroup",
                "--property=MainPID",
                unit_name,
            )
        )
        properties = _parse_properties(result.stdout)
        detail = redact_secret_material(result.stderr.strip())
        if properties.get("LoadState") == "not-found" or result.returncode == 4:
            return "missing", None, detail or "systemd unit is absent"
        if result.returncode != 0:
            return "unavailable", None, detail or "systemctl show did not return a loaded unit"
        if properties.get("LoadState") != "loaded":
            # A unit systemd declined to load fully -- ``bad-setting``, ``error``
            # -- still reports its live runtime properties, and a terminal state
            # is direct evidence about the workload rather than a failure to
            # observe one. Discarding ``failed``/``inactive`` cost a live v5p-16
            # its preemption, and discarding a cleanly exited ``RemainAfterExit``
            # unit (``ActiveState=active``, ``SubState=exited``, ``Result=success``,
            # ``ExecMainCode=1``, ``ExecMainStatus=0``) left a completed v6e-8
            # job stuck in RUNNING forever.
            if _is_definitive_unloadable_unit(properties):
                return "present", properties, detail
            return "unavailable", None, detail or "systemctl show did not return a loaded unit"
        return "present", properties, detail

    def _status_from_properties(
        self, record: tp.Mapping[str, JsonValue], properties: tp.Mapping[str, str]
    ) -> tuple[ExecutionWorkloadStatus, str]:
        active = properties.get("ActiveState", "").lower()
        substate = properties.get("SubState", "").lower()
        result = properties.get("Result", "").lower()
        main_status = properties.get("ExecMainStatus", "")
        stop_requested = self._record_boolean(record, "stop_requested")

        if active == "activating":
            return ExecutionWorkloadStatus.PENDING, "systemd service is starting"
        if active in {"active", "reloading"}:
            if substate == "exited":
                # Persisting this mode is necessary because a worker restart or
                # upgrade can change the current systemd capability while an
                # older v249-compatible unit remains active.
                if not self._record_boolean(record, "exit_type_cgroup"):
                    control_group = properties.get("ControlGroup", "")
                    # On systemd 249 an exited transient unit retains
                    # ActiveState=active because of RemainAfterExit=yes, but
                    # its ControlGroup is cleared once every process is gone.
                    # A background child keeps a non-empty cgroup, so an
                    # empty value is positive completion evidence rather than
                    # an unreadable cgroup.
                    if not control_group:
                        return ExecutionWorkloadStatus.SUCCEEDED, "systemd service completed successfully"
                    try:
                        cgroup_has_processes = self._cgroup_processes_reader(control_group)
                    except Exception:
                        cgroup_has_processes = None
                    if cgroup_has_processes is True:
                        return ExecutionWorkloadStatus.RUNNING, "systemd service still has workload cgroup processes"
                    if cgroup_has_processes is not False:
                        return ExecutionWorkloadStatus.INDETERMINATE, _RECOVERABLE_CGROUP_INDETERMINATE
                return ExecutionWorkloadStatus.SUCCEEDED, "systemd service completed successfully"
            return ExecutionWorkloadStatus.RUNNING, "systemd service is running"
        if active == "deactivating":
            return ExecutionWorkloadStatus.RUNNING, "systemd service is stopping"
        if active == "inactive":
            if stop_requested:
                return ExecutionWorkloadStatus.STOPPED, "systemd service stopped by Eyvan"
            if result in {"", "success"} and main_status in {"", "0"}:
                return self._failed(record, "systemd service became inactive before Eyvan observed completion")
            return self._failed(record, f"systemd service exited with result {result or 'unknown'}")
        if active == "failed":
            if stop_requested:
                return ExecutionWorkloadStatus.STOPPED, "systemd service stopped by Eyvan"
            return self._failed(record, f"systemd service failed with result {result or 'unknown'}")
        return ExecutionWorkloadStatus.INDETERMINATE, (
            f"systemd service reported unexpected ActiveState={active or 'missing'}"
        )

    def _failed(self, record: tp.Mapping[str, JsonValue], message: str) -> tuple[ExecutionWorkloadStatus, str]:
        """Report a dead service, naming what its own output said last.

        A bare systemd result tells the submitter *that* every rank died while
        the one sentence naming *why* -- the last line of a Python traceback,
        usually -- sits in a log stream nobody is forced to read. A live
        multi-host job once burned its entire provisioning cycle to report
        ``exit-code``: the workers had died on ``ModuleNotFoundError`` seconds
        after start. The unit's journal tail is read on a best-effort basis;
        an unreadable journal leaves the message exactly as it was.
        """

        excerpt = self._failure_excerpt(record)
        if excerpt:
            return ExecutionWorkloadStatus.FAILED, f"{message}; worker output ended with: {excerpt}"
        return ExecutionWorkloadStatus.FAILED, message

    def _failure_excerpt(self, record: tp.Mapping[str, JsonValue]) -> str | None:
        """Return the dying worker's own closing line, or ``None``.

        Tracebacks end with their exception line, so the last line that reads
        like one is the reason the process died; any other tail still earns
        its final non-empty line, which for a ``SystemExit`` or a shell error
        is the sentence that matters.
        """

        unit_name = record.get("unit_name")
        if not isinstance(unit_name, str) or not unit_name:
            return None
        tail = self._failure_journal_reader(unit_name)
        if not tail:
            return None
        lines = [
            stripped
            for line in tail.splitlines()
            if (stripped := line.strip()) and not _SYSTEMD_LIFECYCLE_LINE.match(stripped)
        ]
        if not lines:
            return None
        # A job that dies on a missing module names it mid-line, and systemd's
        # "Failed with result 'exit-code'" follows it. Prefer the exception line
        # over lifecycle noise so the excerpt retains the actual cause.
        excerpt = next(
            (line for line in reversed(lines) if _FAILURE_EXCEPTION_LINE.match(line)),
            next((line for line in reversed(lines) if _FAILURE_EXCEPTION_ANYWHERE.search(line)), lines[-1]),
        )
        return excerpt[:_FAILURE_EXCERPT_LIMIT] or None

    def _read_failure_journal(self, unit_name: str) -> str | None:
        """Read a failed unit's journal tail, redacted, or ``None``."""

        try:
            result = self._run(
                (
                    "journalctl",
                    "--no-pager",
                    "--output=cat",
                    f"--unit={unit_name}",
                    f"--lines={_FAILURE_JOURNAL_LINES}",
                )
            )
        except Exception:
            return None
        if result.returncode != 0:
            return None
        return redact_secret_material(result.stdout)

    @staticmethod
    def _record_boolean(record: tp.Mapping[str, JsonValue], field: str) -> bool:
        """Read one required lifecycle flag without truthiness coercion."""

        try:
            value = record[field]
        except KeyError as exc:
            raise _InvalidDurableRecord(f"systemd execution record has no {field!r} flag") from exc
        if not isinstance(value, bool):
            raise _InvalidDurableRecord(f"systemd execution record has a non-boolean {field!r} flag")
        return value

    @staticmethod
    def _is_final(status: ExecutionWorkloadStatus) -> bool:
        return status in _FINAL_STATES

    @staticmethod
    def _set_status(record: JsonObject, status: ExecutionWorkloadStatus, message: str) -> None:
        previous = _known_persisted_status(record)
        record["status"] = status.value
        record["message"] = message
        if status is not ExecutionWorkloadStatus.INDETERMINATE:
            record.pop(_UNRECOVERABLE_INDETERMINATE_FIELD, None)
        if status is ExecutionWorkloadStatus.RUNNING and record.get("started_at") is None:
            record["started_at"] = _utcnow()
        if status in _FINAL_STATES and previous not in _FINAL_STATES:
            record["finished_at"] = _utcnow()
        record["updated_at"] = _utcnow()

    def _indeterminacy_is_recoverable(self, record_path: Path) -> bool:
        """Return whether this record's INDETERMINATE can still be resolved.

        Reads the record rather than the observation, because recoverability is
        a property of how the record was marked -- an earlier failed stop leaves
        a workload whose unit is still observable and still stoppable.
        """

        with _file_lock(_record_lock_path(record_path)):
            record = _load_json(record_path)
        return record is not None and _is_recoverable_indeterminate(record)

    def _mark_indeterminate(self, record: JsonObject, message: str, *, recoverable: bool = True) -> None:
        current_status = _known_persisted_status(record)
        if current_status not in _FINAL_STATES or (
            current_status is ExecutionWorkloadStatus.INDETERMINATE and _is_recoverable_indeterminate(record)
        ):
            self._set_status(record, ExecutionWorkloadStatus.INDETERMINATE, message)
            if not recoverable:
                record[_UNRECOVERABLE_INDETERMINATE_FIELD] = True

    def _collect_terminal_unit(self, record: JsonObject) -> None:
        """Ask systemd to release a unit after its terminal record is durable."""

        if _known_persisted_status(record) not in _COLLECTABLE_STATES or record.get("unit_collected") is True:
            return
        unit_name = record.get("unit_name")
        if not isinstance(unit_name, str) or not unit_name:
            return
        stop = self._run(("systemctl", "stop", "--no-pager", unit_name))
        reset = self._run(("systemctl", "reset-failed", "--no-pager", unit_name))
        # A missing unit is already collected. Other errors remain retryable on
        # the next observation while the durable status remains authoritative.
        if stop.returncode in {0, 4} and reset.returncode in {0, 4}:
            record["unit_collected"] = True
            record.pop("unit_collection_error", None)
            return
        detail = stop.stderr.strip() or reset.stderr.strip() or "systemd did not collect terminal unit"
        record["unit_collection_error"] = redact_secret_material(detail)

    def _secure_terminal_collection(
        self,
        record_path: Path,
        record: JsonObject,
        *,
        runtime_absent: bool = False,
        environment_consumed: bool = False,
    ) -> None:
        """Persist the outcome before any destructive runtime collection."""

        # External-runtime observations and boot-loss observations reach this
        # helper with a newly derived outcome. A finally-only write cannot
        # survive process loss after collection removes the last runtime evidence.
        # Publish the sanitized record first; fail closed if durability fails.
        _atomic_write_json(record_path, record)
        try:
            self._remove_environment_files(
                record,
                runtime_absent=runtime_absent,
                environment_consumed=environment_consumed,
            )
            self._remove_setup_files(record, runtime_absent=runtime_absent)
            self._collect_terminal_unit(record)
        finally:
            # Persist sanitization and successful cleanup markers even if a
            # filesystem error makes the caller fail closed and retry.
            _atomic_write_json(record_path, record)

    def _recover_latched_indeterminate_environment(self, record_path: Path, record: JsonObject) -> None:
        """Clean plaintext on positive runtime evidence without changing outcome.

        INDETERMINATE remains semantically latched.  This probe answers only the
        narrower question of whether the launch environment is still needed.
        """

        runtime_observation = self._runtime_status(record)
        if runtime_observation is not None:
            status, _message = runtime_observation
            if status in _TERMINAL_STATES or status is ExecutionWorkloadStatus.RUNNING:
                self._secure_terminal_collection(record_path, record, environment_consumed=True)
            else:
                _atomic_write_json(record_path, record)
            return
        unit_name = record.get("unit_name")
        if not isinstance(unit_name, str) or not unit_name:
            _atomic_write_json(record_path, record)
            return
        outcome, properties, _detail = self._show_unit(unit_name)
        if outcome == "missing":
            self._secure_terminal_collection(record_path, record, runtime_absent=True)
            return
        if outcome == "present" and properties is not None:
            status, _message = self._status_from_properties(record, properties)
            if status in _TERMINAL_STATES or status is ExecutionWorkloadStatus.RUNNING:
                self._secure_terminal_collection(record_path, record, environment_consumed=True)
                return
        _atomic_write_json(record_path, record)

    def _boot_changed(self, record: tp.Mapping[str, JsonValue]) -> bool:
        recorded = record.get("boot_id")
        if not isinstance(recorded, str) or not recorded:
            return False
        try:
            current = self._boot_id_reader()
        except OSError:
            return False
        return isinstance(current, str) and bool(current) and current != recorded

    def _claim_logical_workload_locked(
        self,
        path: Path,
        workload_id: str,
        remote_id: str,
        context: MutationContext,
    ) -> None:
        """Persist one workload fence while its caller holds ``path``'s lock."""

        current = _load_json(path)
        predecessor: _LogicalWorkloadPredecessor | None = None
        if current is not None:
            try:
                generation = _required_record_integer(current, "generation")
                current_remote_id = _required_record_string(current, "remote_id")
                token = _required_record_string(current, "fencing_token")
            except ValueError as exc:
                raise StaleMutationError(f"logical workload {workload_id!r} has an invalid persisted fence") from exc
            if not current_remote_id or not token:
                raise StaleMutationError(f"logical workload {workload_id!r} has an invalid persisted fence")
            if context.generation < generation:
                raise StaleMutationError(
                    f"generation {context.generation} is older than logical workload {workload_id!r} "
                    f"generation {generation}"
                )
            if context.generation == generation:
                if token != context.fencing_token:
                    raise StaleMutationError(
                        f"fencing token does not own logical workload {workload_id!r} generation {generation}"
                    )
                if current_remote_id != remote_id:
                    raise StaleMutationError(
                        f"submission {remote_id!r} does not own logical workload {workload_id!r} generation {generation}"
                    )
                return
            predecessor = _LogicalWorkloadPredecessor(
                generation=generation,
                fencing_token=token,
                remote_id=current_remote_id,
            )

        if predecessor is not None:
            self.retire(
                predecessor.remote_id,
                MutationContext(
                    idempotency_key=f"supersede:{workload_id}:{context.generation}",
                    generation=predecessor.generation,
                    fencing_token=predecessor.fencing_token,
                ),
            )
        _atomic_write_json(
            path,
            {
                "workload_id": workload_id,
                "generation": context.generation,
                "fencing_token": context.fencing_token,
                "remote_id": remote_id,
                **(
                    {
                        "predecessor": {
                            "generation": predecessor.generation,
                            "fencing_token": predecessor.fencing_token,
                            "remote_id": predecessor.remote_id,
                        }
                    }
                    if predecessor is not None
                    else {}
                ),
                "updated_at": _utcnow(),
            },
        )

    def _record_fields(
        self,
        spec: tp.Mapping[str, JsonValue],
        remote_id: str,
        environment: tp.Mapping[str, str],
    ) -> tp.Mapping[str, JsonValue]:
        """Return runtime-specific durable fields for a newly submitted record.

        Subclasses use this narrow hook to persist external runtime ownership
        identifiers before the transient service is launched.  The base
        executor deliberately owns the fencing and lifecycle fields.
        """

        del spec, remote_id, environment
        return {}

    def _prepare_launch(self, record: tp.Mapping[str, JsonValue], spec: tp.Mapping[str, JsonValue]) -> None:
        """Write the workload environment file before its unit is queued.

        The merged launch environment travels as one owner-only 0600 file so
        unit argv stays free of values. ``EnvironmentFile=`` without
        systemd's ``-`` ignore prefix makes a missing file fail the launch
        loudly instead of silently starting without an environment.

        Declared secrets are resolved here, into that same file. It is already
        the owner-only private state this workload's environment lives in, so a
        value written into it inherits the protection rather than needing a
        second mechanism -- and the capability that fetched it is dropped, so
        the job receives what it asked for and not the means to ask again.
        """

        remote_id = record.get("remote_id")
        if not isinstance(remote_id, str) or not remote_id:
            raise RuntimeError("systemd execution record has no remote ID")
        launch_environment = self._resolved_launch_environment({**self._base_environment, **_environment(spec)})
        notice_path, receipt_path, _mode = self._notice_file_paths(remote_id)
        launch_environment.setdefault(RECLAIM_NOTICE_PATH_ENV, str(notice_path))
        launch_environment.setdefault(CHECKPOINT_RECEIPT_PATH_ENV, str(receipt_path))
        _write_environment_file(self._environment_file_path(remote_id), launch_environment, systemd=True)
        self._stage_setup_file(remote_id, spec)

    def _launch_command(
        self,
        unit_name: str,
        entrypoint: str,
        environment: tp.Mapping[str, str],
        *,
        record: tp.Mapping[str, JsonValue],
        spec: tp.Mapping[str, JsonValue],
    ) -> tuple[str, ...]:
        """Return the systemd launch command for one durable record."""

        del environment
        remote_id = record.get("remote_id")
        if not isinstance(remote_id, str) or not remote_id:
            raise RuntimeError("systemd execution record has no remote ID")
        entrypoint = self._rewrite_staged_setup_entrypoint(entrypoint, remote_id)
        raw_grace = spec.get("stop_grace_s")
        stop_grace_arg: tuple[str, ...] = ()
        if not isinstance(raw_grace, bool) and isinstance(raw_grace, (int, float)) and raw_grace > 0:
            stop_grace_arg = (f"--property=TimeoutStopSec={int(raw_grace)}",)
        return (
            "systemd-run",
            "--quiet",
            "--no-block",
            f"--unit={unit_name}",
            # Explicit specifier-free description: without one, systemd-run
            # derives Description= from the full command line, which on large
            # or %-bearing commands triggers unit_full_printf errors on reload.
            f"--description=Eyvan workload {unit_name}",
            "--service-type=exec",
            *(("--property=ExitType=cgroup",) if self._use_exit_type_cgroup else ()),
            "--property=RemainAfterExit=yes",
            "--property=KillMode=control-group",
            *stop_grace_arg,
            "--property=StandardOutput=journal",
            "--property=StandardError=journal",
            "--property=LimitMEMLOCK=infinity",
            "--property=LimitNOFILE=100000",
            "--property=NoNewPrivileges=yes",
            f"--property=EnvironmentFile={self._environment_file_path(remote_id)}",
            *((f"--property=User={self._workload_user}",) if self._workload_user is not None else ()),
            *((f"--property=Group={self._workload_group}",) if self._workload_group is not None else ()),
            *(
                (f"--property=WorkingDirectory={self._workload_directory}",)
                if self._workload_directory is not None
                else ()
            ),
            "/bin/sh",
            "-c",
            _escape_systemd_exec_arg(entrypoint),
        )

    def _runtime_status(self, record: JsonObject) -> tuple[ExecutionWorkloadStatus, str] | None:
        """Optionally observe an external runtime before inspecting systemd.

        ``None`` delegates status ownership to systemd.  A tuple is treated as
        authoritative for the running invocation and is persisted through the
        same state-transition rules as native systemd observations.
        """

        del record
        return None

    def _stop_runtime(self, record: tp.Mapping[str, JsonValue]) -> None:
        """Stop an external runtime before stopping its transient service."""

        del record

    def _after_systemd_stop(self, record: tp.Mapping[str, JsonValue]) -> None:
        """Finish external-runtime cleanup after the transient unit stops."""

        del record

    def _retire_terminal_runtime(self, record: tp.Mapping[str, JsonValue]) -> None:
        """Release external-runtime state retained after terminal observation."""

        del record

    def _complete_retirement(self, record_path: Path, record: JsonObject) -> None:
        """Finish cleanup under the record lock before permitting a relaunch."""

        record["retire_cleanup_pending"] = True
        record.pop("retired", None)
        _atomic_write_json(record_path, record)
        try:
            self._retire_terminal_runtime(record)
            self._remove_environment_files(record)
            self._remove_setup_files(record)
        except Exception:
            _atomic_write_json(record_path, record)
            raise
        record.pop("retire_cleanup_pending", None)
        record["retired"] = True
        _atomic_write_json(record_path, record)

    def _runtime_logs(self, record: tp.Mapping[str, JsonValue]) -> str | None:
        """Return external-runtime logs, or ``None`` to fall back to journald."""

        del record
        return None

    def submit(self, spec: tp.Mapping[str, JsonValue], context: MutationContext) -> str:
        """Launch one transient unit or return an owned durable replay."""

        entrypoint = spec.get("entrypoint")
        if not isinstance(entrypoint, str) or not entrypoint.strip():
            raise ValueError("native-systemd workloads require a non-empty entrypoint")
        _validate_gang(spec)
        environment = _environment(spec)
        secret_bearing = bool(secret_environment_keys(environment))
        durable_environment = {
            name: value for name, value in environment.items() if not name.startswith("EYVAN_SECRET_")
        }
        submitted_remote_id = spec.get("submission_id")
        if submitted_remote_id is None:
            remote_id = context.idempotency_key.strip()
        elif isinstance(submitted_remote_id, str):
            remote_id = submitted_remote_id.strip()
        else:
            raise ValueError("native-systemd submission ID must be a string")
        if not remote_id:
            raise ValueError("native-systemd submission ID must be non-empty")
        metadata = spec.get("metadata")
        if metadata is None:
            metadata = {}
        if not isinstance(metadata, dict):
            raise ValueError("native-systemd workload metadata must be a mapping")
        workload_id = self._logical_workload_id(metadata)

        record_path = self._record_path(remote_id)
        # Reject a reused attempt ID before touching the logical workload
        # fence. Otherwise a rejected generation could supersede the current
        # workload high-watermark without ever owning this record.
        with _file_lock(_record_lock_path(record_path)):
            existing = _load_json(record_path)
            if existing is not None:
                self._validate_fence(remote_id, existing, context)
                # Rewrite records from older agents without their secret
                # scaffolding. Terminal replays also retry plaintext cleanup.
                self._remove_environment_files(existing)
                self._remove_setup_files(existing)
                _atomic_write_json(record_path, existing)
                if existing.get("retire_cleanup_pending") is True:
                    self._complete_retirement(record_path, existing)
                if not _was_retired(existing):
                    return remote_id

        def reserve_and_launch() -> str:
            """Reserve and launch while any logical workload lock remains held."""

            # Retirement stops release the record lock during runtime I/O. Keep
            # relaunch excluded until that old-incarnation I/O has finished.
            with _file_lock(record_path.with_suffix(".retire-stop.lock")), _file_lock(_record_lock_path(record_path)):
                existing = _load_json(record_path)
                if existing is not None:
                    self._validate_fence(remote_id, existing, context)
                    self._remove_environment_files(existing)
                    self._remove_setup_files(existing)
                    _atomic_write_json(record_path, existing)
                    if existing.get("retire_cleanup_pending") is True:
                        self._complete_retirement(record_path, existing)
                    if not _was_retired(existing):
                        return remote_id
                    # A retired record is one Eyvan stopped on purpose -- a gang
                    # rolling back a partial launch -- so a submit naming it again
                    # asks for a run, not a replay. Replaying would return units
                    # already stopped by rollback. Fall through and replace them.
                now = _utcnow()
                try:
                    boot_id = self._boot_id_reader()
                except OSError:
                    boot_id = None
                # Runtime hooks receive only the durable subset. Capability
                # scaffolding remains in ``spec`` just long enough for
                # ``_prepare_launch`` to redeem it in memory.
                record_fields = dict(self._record_fields(spec, remote_id, durable_environment))
                unit_name = systemd_unit_name(remote_id)
                protected_fields = {
                    "remote_id",
                    "unit_name",
                    "status",
                    "message",
                    "generation",
                    "fencing_token",
                    "idempotency_key",
                    "incarnation",
                    "retire_keys",
                    "retired",
                    "retire_cleanup_pending",
                    "entrypoint",
                    "environment",
                    "metadata",
                    "secret_bearing",
                    "exit_type_cgroup",
                    "boot_id",
                    "stop_requested",
                    "created_at",
                    "updated_at",
                    "started_at",
                    "finished_at",
                }
                conflicting_fields = sorted(protected_fields & set(record_fields))
                if conflicting_fields:
                    raise ValueError(
                        f"systemd runtime record fields must not replace lifecycle field {conflicting_fields[0]!r}"
                    )
                retire_keys = existing.get("retire_keys", {}) if existing is not None else {}
                if not isinstance(retire_keys, dict):
                    raise RuntimeError("invalid systemd execution retirement keys")
                # Gang rollback reuses the submission key. Relaunch makes that
                # key refer to this new run; distinct retirement-command keys
                # stay bound to their old incarnations for late replay safety.
                next_incarnation = _incarnation(existing) + 1 if existing is not None else 0
                # Bounded by incarnation, not by count: the record is written with
                # sorted keys, so insertion order cannot say which entries are oldest.
                retire_keys = {
                    key: value
                    for key, value in retire_keys.items()
                    if key != context.idempotency_key
                    and type(value) is int
                    and value > next_incarnation - _RETIRE_KEY_INCARNATION_WINDOW
                }
                record: JsonObject = {
                    "remote_id": remote_id,
                    "unit_name": unit_name,
                    "status": ExecutionWorkloadStatus.PENDING.value,
                    "message": "waiting for systemd transient service",
                    "generation": context.generation,
                    "fencing_token": context.fencing_token,
                    "idempotency_key": context.idempotency_key,
                    "incarnation": next_incarnation,
                    "retire_keys": retire_keys,
                    "entrypoint": entrypoint,
                    "environment": durable_environment,
                    "metadata": dict(metadata),
                    "secret_bearing": secret_bearing,
                    "exit_type_cgroup": self._use_exit_type_cgroup,
                    "boot_id": boot_id if isinstance(boot_id, str) and boot_id else None,
                    "stop_requested": False,
                    "created_at": now,
                    "updated_at": now,
                    "started_at": None,
                    "finished_at": None,
                    **record_fields,
                }
                _atomic_write_json(record_path, record)
                try:
                    self._prepare_launch(record, spec)
                except Exception as exc:
                    detail = (
                        _OUTPUT_SUPPRESSED
                        if _record_secret_bearing(record)
                        else str(exc)
                        if isinstance(exc, SecretMaterializationError)
                        else "native runtime preparation failed"
                    )
                    self._set_status(
                        record,
                        ExecutionWorkloadStatus.FAILED,
                        f"failed to prepare native runtime: {detail}",
                    )
                    _atomic_write_json(record_path, record)
                    try:
                        self._remove_environment_files(record)
                        self._remove_setup_files(record)
                    finally:
                        _atomic_write_json(record_path, record)
                    raise RuntimeError(f"failed to launch systemd execution {remote_id!r}: {detail}") from None
                try:
                    launch_command = self._launch_command(
                        unit_name,
                        entrypoint,
                        durable_environment,
                        record=record,
                        spec=spec,
                    )
                except Exception:
                    detail = "native runtime launch command could not be constructed"
                    self._set_status(record, ExecutionWorkloadStatus.FAILED, detail)
                    _atomic_write_json(record_path, record)
                    try:
                        self._remove_environment_files(record)
                        self._remove_setup_files(record)
                    finally:
                        _atomic_write_json(record_path, record)
                    raise RuntimeError(f"failed to launch systemd execution {remote_id!r}: {detail}") from None
                result = self._run(launch_command)
                if result.returncode == 0:
                    return remote_id

                outcome, _properties, detail = self._show_unit(unit_name)
                launch_detail = (
                    _OUTPUT_SUPPRESSED
                    if _record_secret_bearing(record)
                    else redact_secret_material(result.stderr.strip() or detail or "systemd-run failed")
                )
                if outcome == "missing":
                    self._set_status(
                        record,
                        ExecutionWorkloadStatus.FAILED,
                        f"failed to launch systemd service: {launch_detail}",
                    )
                elif outcome == "unavailable":
                    self._mark_indeterminate(
                        record,
                        f"cannot determine whether systemd service launched: {launch_detail}",
                    )
                else:
                    # A non-zero client result can race a successfully queued unit.
                    # Keep its record pending and let normal observation resolve it.
                    record["updated_at"] = _utcnow()
                _atomic_write_json(record_path, record)
                if outcome == "present":
                    return remote_id
                if outcome == "missing":
                    try:
                        self._remove_environment_files(record)
                        self._remove_setup_files(record)
                    finally:
                        _atomic_write_json(record_path, record)
            raise RuntimeError(f"failed to launch systemd execution {remote_id!r}: {launch_detail}")

        if workload_id is None:
            return reserve_and_launch()
        workload_path = self._workload_fence_path(workload_id)
        # Do not release this lock between fence advancement and the durable
        # record reservation/systemd-run call. A newer generation must observe
        # and retire an actual prior attempt, never a fence with a launch gap.
        with _file_lock(_record_lock_path(workload_path)):
            self._claim_logical_workload_locked(workload_path, workload_id, remote_id, context)
            return reserve_and_launch()

    def status(self, remote_id: str) -> ExecutionWorkloadObservation:
        """Return durable status, failing closed when runtime ownership is lost."""

        record_path = self._record_path(remote_id)
        with _file_lock(_record_lock_path(record_path)):
            record = _load_json(record_path)
            if record is None:
                return ExecutionWorkloadObservation(
                    remote_id=remote_id,
                    status=ExecutionWorkloadStatus.NOT_FOUND,
                    message="systemd execution is absent",
                )
            _upgrade_legacy_secret_record(record)
            try:
                durable = _durable_observation(remote_id, record)
            except _InvalidDurableRecord as exc:
                message = _OUTPUT_SUPPRESSED if _record_secret_bearing(record) else f"invalid durable record: {exc}"
                self._set_status(record, ExecutionWorkloadStatus.INDETERMINATE, message)
                _atomic_write_json(record_path, record)
                return ExecutionWorkloadObservation(
                    remote_id=remote_id,
                    status=ExecutionWorkloadStatus.INDETERMINATE,
                    message=message,
                )

            recoverable_indeterminate = (
                durable.status is ExecutionWorkloadStatus.INDETERMINATE and _is_recoverable_indeterminate(record)
            )
            if durable.status in _TERMINAL_STATES:
                self._secure_terminal_collection(record_path, record)
                return durable
            if durable.status is ExecutionWorkloadStatus.INDETERMINATE and not recoverable_indeterminate:
                try:
                    self._recover_latched_indeterminate_environment(record_path, record)
                except Exception:
                    # Failure to prove absence/consumption retains plaintext and
                    # the conservative outcome; startup does not guess.
                    _atomic_write_json(record_path, record)
                return durable
            if self._boot_changed(record):
                self._set_status(
                    record,
                    ExecutionWorkloadStatus.LOST,
                    "host boot ID changed; transient systemd service was lost",
                )
                self._secure_terminal_collection(record_path, record, runtime_absent=True)
                return _durable_observation(remote_id, record)

            runtime_absent = False
            environment_consumed = False
            try:
                runtime_observation = self._runtime_status(record)
                if runtime_observation is not None:
                    status, message = runtime_observation
                    if _record_secret_bearing(record):
                        message = _OUTPUT_SUPPRESSED
                    else:
                        message = redact_secret_material(message)
                    if status is ExecutionWorkloadStatus.INDETERMINATE:
                        self._mark_indeterminate(record, message)
                    else:
                        self._set_status(record, status, message)
                    environment_consumed = status is ExecutionWorkloadStatus.RUNNING or status in _TERMINAL_STATES
                    self._secure_terminal_collection(
                        record_path,
                        record,
                        environment_consumed=environment_consumed,
                    )
                    return _durable_observation(remote_id, record)

                unit_name = record.get("unit_name")
                if not isinstance(unit_name, str) or not unit_name:
                    self._mark_indeterminate(record, "systemd execution record has no unit name", recoverable=False)
                    _atomic_write_json(record_path, record)
                    return _durable_observation(remote_id, record)
                outcome, properties, detail = self._show_unit(unit_name)
                if outcome == "missing":
                    self._mark_indeterminate(
                        record,
                        "systemd service disappeared before a terminal state was recorded",
                        recoverable=False,
                    )
                    runtime_absent = True
                elif outcome == "unavailable" or properties is None:
                    message = (
                        _OUTPUT_SUPPRESSED
                        if _record_secret_bearing(record)
                        else f"cannot observe systemd service: {redact_secret_material(detail)}"
                    )
                    self._mark_indeterminate(record, message)
                else:
                    status, message = self._status_from_properties(record, properties)
                    if _record_secret_bearing(record):
                        message = _OUTPUT_SUPPRESSED
                    if status is ExecutionWorkloadStatus.INDETERMINATE:
                        self._mark_indeterminate(record, message)
                    else:
                        self._set_status(record, status, message)
                    environment_consumed = status is ExecutionWorkloadStatus.RUNNING or status in _TERMINAL_STATES
            except _InvalidDurableRecord as exc:
                message = _OUTPUT_SUPPRESSED if _record_secret_bearing(record) else f"invalid durable record: {exc}"
                self._mark_indeterminate(record, message, recoverable=False)
            _atomic_write_json(record_path, record)
            self._secure_terminal_collection(
                record_path,
                record,
                runtime_absent=runtime_absent,
                environment_consumed=environment_consumed,
            )
            return _durable_observation(remote_id, record)

    def stop(self, remote_id: str, context: MutationContext) -> None:
        """Stop the workload's systemd cgroup through ``systemctl stop``.

        The record lock is released for the runtime I/O below, and the unit name
        is deterministic, so a relaunch landing in that gap would be stopped by
        this call. Holding the relaunch lock for the whole stop closes the gap.
        """

        with _file_lock(self._record_path(remote_id).with_suffix(".retire-stop.lock")):
            self._stop_locked(remote_id, context)

    def _stop_locked(self, remote_id: str, context: MutationContext) -> None:
        """Stop the workload; the caller already holds the relaunch lock.

        ``flock`` is per open file description, so taking that lock again here
        from ``retire`` would deadlock this process against itself.
        """

        record_path = self._record_path(remote_id)
        with _file_lock(_record_lock_path(record_path)):
            record = _load_json(record_path)
            if record is None:
                return
            self._validate_fence(remote_id, record, context)
            try:
                durable = _durable_observation(remote_id, record)
            except _InvalidDurableRecord as exc:
                self._set_status(record, ExecutionWorkloadStatus.INDETERMINATE, f"invalid durable record: {exc}")
                _atomic_write_json(record_path, record)
                raise RuntimeError(f"cannot safely stop ambiguous systemd execution {remote_id!r}") from exc
            if durable.status in _TERMINAL_STATES:
                self._secure_terminal_collection(record_path, record)
                return
            # What makes a stop unsafe is not knowing *what* to stop, so the unit
            # name is the gate -- not the status.  An INDETERMINATE record with a
            # known unit is the case where stopping is most necessary: the status
            # is INDETERMINATE precisely because an earlier stop failed, and
            # refusing here made that first failure permanent.  The workload could
            # then never be cancelled and its accelerator was held until an
            # operator intervened, which is how one v5p Multislice stayed leased
            # to a container that had already exited.
            unit_name = record.get("unit_name")
            if not isinstance(unit_name, str) or not unit_name:
                self._mark_indeterminate(record, "systemd execution record has no unit name", recoverable=False)
                _atomic_write_json(record_path, record)
                raise RuntimeError(f"cannot safely stop ambiguous systemd execution {remote_id!r}")
            record["stop_requested"] = True
            record["message"] = "systemd stop requested by Eyvan"
            record["updated_at"] = _utcnow()
            _atomic_write_json(record_path, record)

        try:
            self._stop_runtime(record)
        except Exception as exc:
            detail = _OUTPUT_SUPPRESSED if _record_secret_bearing(record) else redact_secret_material(str(exc))
            with _file_lock(_record_lock_path(record_path)):
                current = _load_json(record_path)
                if current is not None and _known_persisted_status(current) not in _FINAL_STATES:
                    self._mark_indeterminate(current, f"failed to stop native runtime: {detail}")
                    _atomic_write_json(record_path, current)
            raise RuntimeError(f"failed to stop native runtime for {remote_id!r}: {detail}") from None

        result = self._run(("systemctl", "stop", "--no-pager", unit_name))
        try:
            self._after_systemd_stop(record)
        except Exception as exc:
            detail = _OUTPUT_SUPPRESSED if _record_secret_bearing(record) else redact_secret_material(str(exc))
            with _file_lock(_record_lock_path(record_path)):
                current = _load_json(record_path)
                if current is not None and _known_persisted_status(current) not in _FINAL_STATES:
                    self._mark_indeterminate(current, f"failed to finish native runtime stop: {detail}")
                    _atomic_write_json(record_path, current)
            raise RuntimeError(f"failed to finish native runtime stop for {remote_id!r}: {detail}") from None

        # Both stop stages have now succeeded, which is direct positive evidence
        # about a workload whose record may still carry an INDETERMINATE latched
        # by an *earlier* attempt -- including one written by a build that
        # predates the recoverable flag. Let the verification below re-derive
        # from systemd and Docker instead of being vetoed by that stale unknown,
        # which is what kept a Multislice leased to an exited container.
        with _file_lock(_record_lock_path(record_path)):
            current = _load_json(record_path)
            if current is not None and _known_persisted_status(current) is ExecutionWorkloadStatus.INDETERMINATE:
                current.pop(_UNRECOVERABLE_INDETERMINATE_FIELD, None)
                _atomic_write_json(record_path, current)

        observed = self.status(remote_id)
        # Asked once and used twice: both refusals below are about whether
        # something might still be running, and a unit systemd does not hold is
        # not running -- including when ``systemctl stop`` reported a failure
        # because there was nothing there to stop.
        runtime_gone = self._runtime_is_gone(unit_name)
        if result.returncode == 0 and runtime_gone and observed.status is ExecutionWorkloadStatus.INDETERMINATE:
            # A successful stop followed by collection proves intentional
            # termination, not a successful workload exit. Keep mere absence
            # (including a failed stop) unknown; stop_requested alone is not
            # evidence that either runtime stop stage succeeded.
            with _file_lock(_record_lock_path(record_path)):
                current = _load_json(record_path)
                if current is not None:
                    self._validate_fence(remote_id, current, context)
                    if _known_persisted_status(current) is ExecutionWorkloadStatus.INDETERMINATE:
                        self._set_status(
                            current, ExecutionWorkloadStatus.STOPPED, "systemd stop completed before collection"
                        )
                        _atomic_write_json(record_path, current)
                        self._secure_terminal_collection(record_path, current, runtime_absent=True)
        if observed.status is ExecutionWorkloadStatus.INDETERMINATE and not runtime_gone:
            raise RuntimeError(f"cannot safely verify that systemd execution {remote_id!r} stopped")
        if result.returncode != 0 and observed.status not in _TERMINAL_STATES and not runtime_gone:
            detail = (
                _OUTPUT_SUPPRESSED
                if _record_secret_bearing(record)
                else redact_secret_material(result.stderr.strip()) or "unknown systemd error"
            )
            raise RuntimeError(f"systemctl stop failed for {remote_id!r}: {detail}")

    def _runtime_is_gone(self, unit_name: str) -> bool:
        """Return whether systemd holds no such unit at all.

        The two questions ``stop`` and ``status`` ask are not the same one.
        ``status`` reports what the workload *did*, and a transient unit that
        vanished before recording a terminal state leaves that genuinely unknown
        -- INDETERMINATE is the right answer and must stay. ``stop`` asks only
        whether anything is still running, and a unit systemd does not have is
        not running: a transient unit that does not exist cannot execute.

        Conflating the two cost a live v5p-16 multislice. The cancel stopped the
        unit, the unit disappeared before its exit was recorded, and every retry
        then re-derived the same unknown and refused to confirm the stop. The
        workload stayed ``cancelling``, its lease was never released, and four
        TPU VMs went on billing with nothing able to end them.

        A probe that cannot answer returns ``False`` -- an unreachable systemd is
        not evidence of absence, and the refusal above is correct there.
        """

        outcome, _properties, _detail = self._show_unit(unit_name)
        return outcome == "missing"

    def notify_reclaim(
        self,
        remote_id: str,
        *,
        notice: tp.Mapping[str, JsonValue],
        grace_s: int,
        context: MutationContext,
    ) -> None:
        """Write an atomic reclaim notice file and send ``SIGUSR1`` to the unit."""

        record_path = self._record_path(remote_id)
        notice_path, _receipt_path, notice_mode = self._notice_file_paths(remote_id)
        with _file_lock(_record_lock_path(record_path)):
            record = _load_json(record_path)
            if record is None:
                return
            self._validate_fence(remote_id, record, context)
            if _known_persisted_status(record) in _TERMINAL_STATES:
                return
            unit_name = record.get("unit_name")
            if not isinstance(unit_name, str) or not unit_name:
                return
            if notice_mode == 0o600:
                _ensure_private_directory(notice_path.parent)
            else:
                notice_path.parent.mkdir(parents=True, exist_ok=True)
                os.chmod(notice_path.parent, 0o755)
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{notice_path.name}.", dir=notice_path.parent)
            temporary = Path(temporary_name)
            try:
                os.fchmod(descriptor, notice_mode)
                with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                    json.dump(dict(notice), stream, sort_keys=True, separators=(",", ":"))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, notice_path)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            record["reclaim_notified"] = True
            record["stop_grace_s"] = max(int(grace_s), 1)
            record["message"] = "systemd graceful reclaim notice delivered"
            record["updated_at"] = _utcnow()
            _atomic_write_json(record_path, record)
        self._run(("systemctl", "kill", "--signal=SIGUSR1", "--kill-who=all", unit_name))

    def read_checkpoint_uri(self, remote_id: str) -> str | None:
        """Return any checkpoint URI written to this run's checkpoint receipt file."""

        _notice_path, receipt_path, _mode = self._notice_file_paths(remote_id)
        receipt = read_checkpoint_receipt(receipt_path)
        if receipt is None:
            return None
        uri = receipt.get("checkpoint_uri")
        return uri.strip() if isinstance(uri, str) and uri.strip() else None

    def retire(self, remote_id: str, context: MutationContext) -> None:
        """Stop a superseded cgroup and wait for a verified terminal record."""

        record_path = self._record_path(remote_id)
        with _file_lock(_record_lock_path(record_path)):
            record = _load_json(record_path)
            if record is None:
                return
            self._validate_fence(remote_id, record, context)
            incarnation = _incarnation(record)
            retire_keys = record.get("retire_keys", {})
            if not isinstance(retire_keys, dict):
                raise RuntimeError("invalid systemd execution retirement keys")
            if context.idempotency_key in retire_keys and retire_keys[context.idempotency_key] != incarnation:
                return
            # Bind at admission, not completion: an overlapping retirement may
            # finish this incarnation while this caller is observing it.
            retire_keys[context.idempotency_key] = incarnation
            record["retire_keys"] = retire_keys
            _atomic_write_json(record_path, record)

        try:
            timeout_s = max(float(os.getenv("EYVAN_SUPERSEDE_WAIT_TIMEOUT_S", "300")), 0.1)
        except ValueError:
            timeout_s = 300.0
        try:
            poll_s = max(float(os.getenv("EYVAN_SUPERSEDE_POLL_S", "0.1")), 0.01)
        except ValueError:
            poll_s = 0.1
        deadline = time.monotonic() + timeout_s
        while True:
            observed = self.status(remote_id)
            if observed.status in _TERMINAL_STATES:
                with _file_lock(_record_lock_path(record_path)):
                    record = _load_json(record_path)
                    if record is not None:
                        self._validate_fence(remote_id, record, context)
                        if (
                            _incarnation(record) != incarnation
                            or _known_persisted_status(record) not in _TERMINAL_STATES
                        ):
                            return
                        if not _was_retired(record):
                            self._complete_retirement(record_path, record)
                return
            if observed.status is ExecutionWorkloadStatus.INDETERMINATE and not self._indeterminacy_is_recoverable(
                record_path
            ):
                raise RuntimeError(f"cannot safely retire ambiguous systemd execution {remote_id!r}")
            try:
                with _file_lock(record_path.with_suffix(".retire-stop.lock")):
                    with _file_lock(_record_lock_path(record_path)):
                        current = _load_json(record_path)
                        if current is None or _incarnation(current) != incarnation:
                            return
                        self._validate_fence(remote_id, current, context)
                    self._stop_locked(remote_id, context)
            except Exception:
                # A stop that fails now may well succeed on the next poll: a
                # container can still be shutting down, and this loop already
                # has a deadline. Aborting on the first failure would fail the
                # supersession that called us, which is how one slow teardown
                # permanently blocked the next generation of a workload from
                # starting at all.
                if time.monotonic() >= deadline:
                    break
                time.sleep(poll_s)
                continue
            if time.monotonic() >= deadline:
                break
            time.sleep(poll_s)

        with _file_lock(_record_lock_path(record_path)):
            record = _load_json(record_path)
            if record is not None and _incarnation(record) != incarnation:
                # Another caller finished retiring the incarnation this one was
                # waiting on, and a relaunch replaced it: the replacement is not
                # this retirement's to judge.
                return
            if record is not None and _known_persisted_status(record) not in _FINAL_STATES:
                self._mark_indeterminate(record, "systemd retirement timed out; workload ownership is ambiguous")
                _atomic_write_json(record_path, record)
        raise RuntimeError(f"cannot safely retire systemd execution {remote_id!r} before its timeout")

    def checkpoint(
        self,
        remote_id: str,
        *,
        hook: str,
        output_uri: str | None,
        context: MutationContext,
    ) -> str:
        """Run a separately fenced checkpoint hook after source validation."""

        if not isinstance(hook, str) or not hook.strip():
            raise ValueError("native-systemd checkpoint hook must be non-empty")
        record_path = self._record_path(remote_id)
        with _file_lock(_record_lock_path(record_path)):
            source = _load_json(record_path)
            if source is None:
                raise RuntimeError(f"systemd execution {remote_id!r} is absent")
            self._validate_fence(remote_id, source, context)
            try:
                source_status = _durable_observation(remote_id, source).status
            except _InvalidDurableRecord as exc:
                self._set_status(source, ExecutionWorkloadStatus.INDETERMINATE, f"invalid durable record: {exc}")
                _atomic_write_json(record_path, source)
                raise RuntimeError(f"cannot checkpoint ambiguous systemd execution {remote_id!r}") from exc
            if source_status is ExecutionWorkloadStatus.INDETERMINATE:
                raise RuntimeError(f"cannot checkpoint ambiguous systemd execution {remote_id!r}")
        inherited = source.get("environment")
        source_environment = (
            {name: value for name, value in (inherited or {}).items() if isinstance(value, str)}
            if isinstance(inherited, dict)
            else {}
        )

        # Keep the hook's rank IDENTITY, drop the cluster-forming topology. The
        # source workload is still RUNNING while its hook executes, so
        # inheriting JAX_COORDINATOR_* would make rank 0's hook bind the port the
        # live training coordinator already holds. `without_distributed_topology_overrides`
        # is the same filter the slice applies before injecting its own values.
        environment = without_distributed_topology_overrides(source_environment)
        rank = _checkpoint_rank(source_environment)
        if rank is not None:
            environment["EYVAN_CHECKPOINT_RANK"] = str(rank)
        environment.update(
            {
                "EYVAN_SOURCE_JOB_ID": remote_id,
                # Per-rank destination. JAX distributed membership does not
                # serialize object-store writes, so a single shared URI would let
                # every rank race on one destination. The unsuffixed value stays
                # available as the base so a hook can still address the set.
                "EYVAN_CHECKPOINT_URI": _rank_checkpoint_uri(output_uri, rank),
                "EYVAN_CHECKPOINT_BASE_URI": output_uri or "",
                # Per-rank profile destination, on the same plumbing and for the
                # same reason. On a TPU host the only useful profile is
                # jax.profiler/libtpu tracing, which must start *inside* the
                # training process: it needs the JAX runtime's device handles,
                # there is no ptrace equivalent for TPU device state, and libtpu
                # holds an exclusive per-host lock. So Eyvan hands the callable a
                # place to write and gets out of the way -- a platform-driven
                # ``profile_task`` is a recorded non-goal, not an omission.
                "EYVAN_PROFILE_URI": _rank_profile_uri(output_uri, rank),
            }
        )
        return self.submit(
            {
                "entrypoint": hook,
                "submission_id": context.idempotency_key,
                "environment": environment,
                "gang": {"replicas": 1, "per_replica": {"cpu": 1}},
                "metadata": {"eyvan_operation": "checkpoint", "eyvan_source_job_id": remote_id},
            },
            context,
        )

    def logs(self, remote_id: str) -> str:
        """Return the unit's journal output without treating log absence as success.

        A missing durable record, or one that never recorded a unit, yields an
        empty string because no workload output can exist. A journalctl
        failure is not an empty log: it raises ``SystemdLogsUnavailableError``
        carrying the captured stderr so callers can tell a log transport
        failure apart from a genuinely silent workload.

        Raises:
            SystemdLogsUnavailableError: If journalctl could not read the
                unit's journal.
        """

        record_path = self._record_path(remote_id)
        with _file_lock(_record_lock_path(record_path)):
            record = _load_json(record_path)
            if record is None:
                return ""
            if _record_secret_bearing(record):
                return ""
            unit_name = record.get("unit_name")
            if not isinstance(unit_name, str) or not unit_name:
                return ""
            runtime_logs = self._runtime_logs(record)
            if runtime_logs is not None:
                return redact_secret_material(runtime_logs)
        result = self._run(("journalctl", "--no-pager", "--output=cat", f"--unit={unit_name}"))
        if result.returncode != 0:
            raise SystemdLogsUnavailableError(unit_name, result.stderr.strip())
        return redact_secret_material(result.stdout)

    def exec_in_workload(
        self,
        remote_id: str,
        command: tp.Sequence[str],
        *,
        timeout_s: float,
    ) -> JsonObject:
        """Run one diagnostic command beside a live workload and return its output.

        **Beside, not inside.** The exec'd process is its own transient unit,
        not a shell in the workload's cgroup, and the difference matters twice:
        a process inside that cgroup competes for the memory allocation the
        trainer is sized against, and its death can take the trainer down with
        it. A diagnostic that can kill the job it is diagnosing is not one.

        **Not a PTY.** An interactive session needs a bidirectional long-lived
        socket through five synchronous hops, and ``MutationContext`` has no
        representation for a *session* -- so a stale controller could hold a
        shell on a cell whose lease was revoked. One command, one result.

        The source job's identity travels in the environment so a tool can
        target it: ``process_status`` already exposes the control group and the
        PIDs, which is what makes ``py-spy dump --pid`` useful from out here.
        """

        arguments = [str(part) for part in command]
        if not arguments:
            raise ValueError("exec requires a command")
        record_path = self._record_path(remote_id)
        with _file_lock(_record_lock_path(record_path)):
            source = _load_json(record_path)
        if source is None:
            raise RuntimeError(f"systemd execution {remote_id!r} is absent")
        unit_name = source.get("unit_name")
        if _record_secret_bearing(source):
            return {
                "remote_id": remote_id,
                "unit_name": unit_name if isinstance(unit_name, str) else None,
                "command": [],
                "exit_code": None,
                "stdout": "",
                "stderr": "",
                "output_suppressed": True,
            }
        inherited = source.get("environment")
        source_environment = (
            {name: value for name, value in (inherited or {}).items() if isinstance(value, str)}
            if isinstance(inherited, dict)
            else {}
        )
        # Same filter the checkpoint hook applies, for the same reason: the
        # source workload is still running, so inheriting its coordinator
        # address would make this process try to join a cluster it is only
        # meant to look at.
        environment = without_distributed_topology_overrides(source_environment)
        environment["EYVAN_SOURCE_JOB_ID"] = remote_id
        control_group = source.get("control_group")
        if isinstance(control_group, str) and control_group:
            environment["EYVAN_SOURCE_CONTROL_GROUP"] = control_group
        # The environment and the deadline travel inside the systemd-run
        # invocation rather than as runner arguments: systemd owns the unit's
        # lifetime, so a bound it enforces survives this process dying, while a
        # subprocess timeout here would leave the unit running with nobody
        # waiting for it.
        invocation = [
            "systemd-run",
            "--quiet",
            "--pipe",
            "--collect",
            "--wait",
            f"--property=RuntimeMaxSec={max(1, int(timeout_s))}",
        ]
        invocation.extend(f"--setenv={name}={value}" for name, value in sorted(environment.items()))
        invocation.append("--")
        invocation.extend(_escape_systemd_exec_arg(part) for part in arguments)
        result = self._run(tuple(invocation))
        document: JsonObject = {
            "remote_id": remote_id,
            "unit_name": unit_name if isinstance(unit_name, str) else None,
            "command": arguments,
            "exit_code": result.returncode,
            "stdout": redact_secret_material(result.stdout),
            "stderr": redact_secret_material(result.stderr),
        }
        return document

    def log_page(self, remote_id: str, *, after: str = "", limit: int = 200) -> LogPage:
        """Return one resumable page of structured journal records.

        ``--output=json`` rather than ``--output=cat``: the same read, with the
        fields a string throws away -- priority, timestamp, PID, and journald's
        own ``__CURSOR`` to resume from. The cursor is opaque here and handed
        straight back to journald, which is the only thing that can interpret it.

        An unreadable journal returns a page marked ``unavailable`` rather than
        raising. ``logs()`` raises because its caller has no other way to tell a
        transport failure from a silent workload; a page can say so in a field,
        and a follower polling every second should not have to catch an
        exception to learn a host is briefly unreachable.
        """

        record_path = self._record_path(remote_id)
        with _file_lock(_record_lock_path(record_path)):
            record = _load_json(record_path)
            if record is None:
                return LogPage(unavailable=True, detail="no durable execution record")
            if _record_secret_bearing(record):
                return LogPage(unavailable=True, detail=_OUTPUT_SUPPRESSED)
            unit_name = record.get("unit_name")
            if not isinstance(unit_name, str) or not unit_name:
                return LogPage(unavailable=True, detail="execution record has no unit")
        # ``--all``: without it journald's JSON output replaces any field over
        # 4096 bytes with ``null``, and a long line -- ``print(jax.devices())``
        # on a 64-chip slice is 4.3 KB -- reached the follower as nothing at
        # all, while ``--output=cat`` reads showed it. The page is still bounded
        # by the runner's byte limit, not by this flag.
        arguments = [
            "journalctl",
            "--no-pager",
            "--output=json",
            "--all",
            f"--unit={unit_name}",
            f"--lines={max(1, limit)}",
        ]
        if after:
            # journald resumes *after* the cursor, so a follower never re-reads
            # the line it already has.
            arguments.extend((f"--after-cursor={after}", "--lines=all"))
        if self._custom_command_runner:
            # Injected runners own their transport buffering. Enforce the same
            # forward-entry window before parsing their returned payload.
            result = self._run(tuple(arguments))
        else:
            result = _systemd_journal_runner(tuple(arguments), limit=max(1, limit), timeout_s=self._command_timeout_s)
        if result.returncode != 0:
            return LogPage(
                unavailable=True,
                detail=redact_secret_material(result.stderr.strip()) or "journalctl failed",
            )
        payload = "\n".join(result.stdout.splitlines()[: max(1, limit)])
        return parse_journal_json(redact_secret_material(payload))

    def process_status(self, remote_id: str) -> JsonObject:
        """Return live process detail for one workload attempt.

        Built for the case this stack actually hits: a multi-host job that has
        stopped making progress. ``status`` alone reports RUNNING for both a
        healthy step loop and a hung collective, so this exposes the unit's main
        PID and the cgroup's process list to tell them apart.

        ``pids`` is ``None`` -- never ``[]`` -- when the cgroup could not be
        read, preserving the same ambiguity distinction the status path relies
        on.
        """

        record_path = self._record_path(remote_id)
        with _file_lock(_record_lock_path(record_path)):
            record = _load_json(record_path)
        if record is None:
            return {"remote_id": remote_id, "observed": False, "detail": "no durable execution record"}
        if _record_secret_bearing(record):
            return {
                "remote_id": remote_id,
                "observed": False,
                "detail": _OUTPUT_SUPPRESSED,
                "output_suppressed": True,
            }
        unit_name = record.get("unit_name")
        if not isinstance(unit_name, str) or not unit_name:
            return {"remote_id": remote_id, "observed": False, "detail": "execution record has no unit"}

        outcome, properties, detail = self._show_unit(unit_name)
        if outcome != "present" or properties is None:
            return {
                "remote_id": remote_id,
                "unit_name": unit_name,
                "observed": False,
                "detail": detail or f"systemd unit is {outcome}",
            }
        control_group = properties.get("ControlGroup", "")
        pids = self._cgroup_pids_reader(control_group) if control_group else None
        return {
            "remote_id": remote_id,
            "unit_name": unit_name,
            "observed": True,
            "active_state": properties.get("ActiveState", ""),
            "sub_state": properties.get("SubState", ""),
            "result": properties.get("Result", ""),
            "main_pid": _optional_int(properties.get("MainPID")),
            "exec_main_status": _optional_int(properties.get("ExecMainStatus")),
            "control_group": control_group,
            "pids": None if pids is None else list(pids),
            "process_count": None if pids is None else len(pids),
            "detail": redact_secret_material(detail),
        }


def _checkpoint_rank(environment: tp.Mapping[str, str]) -> int | None:
    """Return this host's rank from a source workload environment, if it has one.

    A single-host workload has no rank; ``None`` means "not part of a slice"
    rather than "rank 0", so its checkpoint destination stays unsuffixed.
    """

    # Only the slice executor injects these. EYVAN_REPLICA_INDEX is deliberately
    # excluded: `submit` sets it from the gang's replica count on every workload,
    # so it is 0 for an ordinary single-host job and would shadow the real rank.
    for name in ("EYVAN_DISTRIBUTED_PROCESS_ID", "JAX_PROCESS_ID"):
        raw = environment.get(name)
        if raw is None:
            continue
        try:
            rank = int(raw)
        except ValueError:
            continue
        if rank >= 0:
            return rank
    return None


def _rank_checkpoint_uri(output_uri: str | None, rank: int | None) -> str:
    """Return a per-rank checkpoint destination derived from a shared URI."""

    if not output_uri or rank is None:
        return output_uri or ""
    return f"{output_uri.rstrip('/')}/rank-{rank}"


def _rank_profile_uri(output_uri: str | None, rank: int | None) -> str:
    """Return a per-rank profile destination beside the checkpoint destination.

    Separate prefix, same base: a profile is not a checkpoint, and writing both
    under one path would make a resume read trace files as state.
    """

    if not output_uri:
        return ""
    base = f"{output_uri.rstrip('/')}/profiles"
    return base if rank is None else f"{base}/rank-{rank}"


def _optional_int(value: str | None) -> int | None:
    """Parse a systemd integer property without inventing a value."""

    if value is None or not value.strip():
        return None
    try:
        return int(value.strip())
    except ValueError:
        return None


__all__ = (
    "SystemdCommandResult",
    "SystemdLogsUnavailableError",
    "SystemdTransientServiceExecutor",
    "systemd_unit_name",
)
