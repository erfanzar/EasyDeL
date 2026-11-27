# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Persistent single-host execution backend for Eyvan development cells."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import time
import typing as tp
from contextlib import ExitStack, contextmanager
from datetime import datetime
from pathlib import Path
from threading import RLock

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from pydantic import JsonValue as PydanticJsonValue

from eyvan._internal.compat import UTC
from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object, model_json
from eyvan._internal.workspace import create_private_directory
from eyvan.capacity.drivers.base import (
    ExecutionWorkloadObservation,
    ExecutionWorkloadStatus,
    MutationContext,
    StaleMutationError,
)
from eyvan.storage._region_guard import (
    CROSS_REGION_TRANSFER_BANNED_EXIT_CODE,
    CROSS_REGION_TRANSFER_BANNED_MARKER,
    CROSS_REGION_TRANSFER_BANNED_REASON,
)
from eyvan.worker.runtime.log_records import LogPage, LogRecord
from eyvan.worker.runtime.preemption import (
    CHECKPOINT_RECEIPT_PATH_ENV,
    RECLAIM_NOTICE_PATH_ENV,
    read_checkpoint_receipt,
)

_LOCAL_LOG_CURSOR = re.compile(r"^b(0|[1-9][0-9]*)$")
_LOCAL_LOG_PAGE_MAX_BYTES = 8 * 1024 * 1024
_CROSS_REGION_BAN_LOG_TAIL_BYTES = 64 * 1024
_CROSS_REGION_BAN_DETAIL_LIMIT = 400


def _extract_cross_region_ban_detail(log_path: Path | None) -> str | None:
    """Read the tail of a local workload log and extract the cross-region ban marker detail."""

    if log_path is None:
        return None
    try:
        with log_path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - _CROSS_REGION_BAN_LOG_TAIL_BYTES))
            tail = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(tail.splitlines()):
        idx = line.find(CROSS_REGION_TRANSFER_BANNED_MARKER)
        if idx != -1:
            detail = line[idx + len(CROSS_REGION_TRANSFER_BANNED_MARKER) :].strip()
            detail = detail.replace("\r", " ").replace("\n", " ")[:_CROSS_REGION_BAN_DETAIL_LIMIT]
            return detail or CROSS_REGION_TRANSFER_BANNED_MARKER
    return None


def _format_cross_region_ban_message(log_path: Path | None) -> str:
    """Render the canonical terminal failure message for exit code 95."""

    detail = _extract_cross_region_ban_detail(log_path)
    if detail:
        return f"{CROSS_REGION_TRANSFER_BANNED_REASON}: {detail}"
    return (
        f"{CROSS_REGION_TRANSFER_BANNED_REASON}: workload terminated with exit code "
        f"{CROSS_REGION_TRANSFER_BANNED_EXIT_CODE}"
    )


_TERMINAL_STATES = frozenset(
    {
        ExecutionWorkloadStatus.SUCCEEDED,
        ExecutionWorkloadStatus.FAILED,
        ExecutionWorkloadStatus.STOPPED,
        ExecutionWorkloadStatus.LOST,
    }
)
_STARTUP_GRACE_S = 10.0
_OWNER_ABSENCE_GRACE_S = 1.0


class _LocalExecutionRecord(BaseModel):
    """Complete durable record written for a newly submitted local workload."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    remote_id: str
    status: ExecutionWorkloadStatus
    message: str
    generation: int = Field(ge=0, strict=True)
    fencing_token: str
    idempotency_key: str
    entrypoint: str
    environment: dict[str, str]
    metadata: dict[str, PydanticJsonValue]
    worker_pid: int | None
    worker_start_marker: str | None
    worker_token: str
    stop_requested: bool
    exit_code: int | None
    created_at: str
    updated_at: str
    started_at: str | None
    finished_at: str | None
    log_path: str
    owner_path: str
    command_pid: int | None = None
    command_start_marker: str | None = None
    group_quiescent: bool = False
    notice_path: str | None = None
    receipt_path: str | None = None
    stop_grace_s: float | None = None
    reclaim_notified: bool = False
    checkpoint_uri: str | None = None


class _LogicalWorkloadFence(BaseModel):
    """Strict durable high watermark for one logical workload."""

    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)

    generation: int = Field(ge=0)
    fencing_token: str
    remote_id: str


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _default_state_dir() -> Path:
    configured = os.getenv("EYVAN_LOCAL_RUNTIME_DIR")
    if configured:
        return Path(configured).expanduser()
    agent_state = os.getenv("EYVAN_AGENT_STATE_DIR")
    if agent_state:
        return Path(agent_state).expanduser() / "runtime"
    return Path.home() / ".local" / "state" / "eyvan" / "runtime"


def _environment_float(name: str, *, default: float) -> float:
    """Parse a numeric environment override, failing fast on a malformed value."""

    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a finite numeric value, got {raw!r}") from None
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite numeric value, got {raw!r}")
    return value


def _record_name(remote_id: str) -> str:
    return hashlib.sha256(remote_id.encode()).hexdigest()


def _atomic_write_json(path: Path, value: tp.Mapping[str, JsonValue]) -> None:
    create_private_directory(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _load_json(path: Path) -> JsonObject | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if not is_json_object(payload):
        raise RuntimeError(f"local execution record {path} is not a strict JSON object")
    return payload


@contextmanager
def _file_lock(path: Path):
    create_private_directory(path.parent)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _record_lock_path(record_path: Path) -> Path:
    return record_path.with_suffix(".lock")


def _read_record(record_path: Path) -> JsonObject | None:
    with _file_lock(_record_lock_path(record_path)):
        return _load_json(record_path)


def _mutate_record(
    record_path: Path,
    mutation: tp.Callable[[JsonObject], None],
) -> JsonObject | None:
    with _file_lock(_record_lock_path(record_path)):
        record = _load_json(record_path)
        if record is None:
            return None
        mutation(record)
        record["updated_at"] = _utcnow()
        _atomic_write_json(record_path, record)
        return record


def _persisted_status(
    remote_id: str,
    record: tp.Mapping[str, JsonValue],
) -> ExecutionWorkloadStatus:
    """Parse one required durable status without inventing a fallback."""

    try:
        raw_status = record["status"]
    except KeyError as exc:
        raise RuntimeError(f"local execution {remote_id!r} has no persisted status") from exc
    if not isinstance(raw_status, str):
        raise RuntimeError(f"local execution {remote_id!r} has a non-string persisted status")
    try:
        return ExecutionWorkloadStatus(raw_status)
    except ValueError as exc:
        raise RuntimeError(f"local execution {remote_id!r} has unsupported persisted status {raw_status!r}") from exc


def _persisted_message(remote_id: str, record: tp.Mapping[str, JsonValue]) -> str:
    """Read an optional durable status message without coercing its type."""

    try:
        message = record["message"]
    except KeyError:
        return ""
    if not isinstance(message, str):
        raise RuntimeError(f"local execution {remote_id!r} has a non-string persisted message")
    return message


def _owner_active(owner_path: Path) -> bool:
    create_private_directory(owner_path.parent)
    descriptor = os.open(owner_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


def _process_start_marker(pid: int) -> str | None:
    """Return a stable best-effort marker for a live process identity.

    The durable worker record can outlive the process that created it. A raw
    PID is therefore never sufficient authority to signal a process group.
    Linux exposes a monotonic start tick in ``/proc``; other POSIX hosts use
    the portable ``ps`` start timestamp as a conservative fallback.
    """

    if pid <= 0:
        return None
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        payload = stat_path.read_text(encoding="utf-8")
        _prefix, suffix = payload.rsplit(")", 1)
        fields = suffix.split()
        # ``suffix`` starts at /proc stat field 3; field 22 is starttime.
        return f"proc:{fields[19]}"
    except (FileNotFoundError, IndexError, OSError, ValueError):
        pass
    try:
        result = subprocess.run(
            ("ps", "-o", "lstart=", "-p", str(pid)),
            check=False,
            capture_output=True,
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    marker = result.stdout.strip()
    return f"ps:{marker}" if result.returncode == 0 and marker else None


def _arguments_carry_token(arguments: tp.Sequence[bytes], expected: bytes) -> bool:
    """Return whether any argument carries the nonce, substring included.

    The token is spelled ``--worker-token=<nonce>``, so it is part of an argument
    and never an argument by itself. Comparing it against whole ``/proc`` argv
    elements therefore never matched on Linux, and every ownership check there
    answered "not my worker": ``stop`` set the message and signalled nothing, and
    ``retire`` waited its full timeout before declaring the workload
    INDETERMINATE without ever sending SIGKILL. macOS has no ``/proc`` and fell
    through to the ``ps`` branch, which compares against the whole command line
    and matched -- so the platform Eyvan does not deploy on was the only one that
    could stop a workload.
    """

    return any(expected in argument for argument in arguments)


def _process_has_worker_token(pid: int, worker_token: str) -> bool:
    """Check that a live process still carries the worker launch nonce."""

    if pid <= 0 or not worker_token:
        return False
    expected = worker_token.encode()
    try:
        arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        return _arguments_carry_token(arguments, expected)
    except OSError:
        pass
    try:
        result = subprocess.run(
            ("ps", "-ww", "-o", "command=", "-p", str(pid)),
            check=False,
            capture_output=True,
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0 and worker_token in result.stdout


def _unreaped_process_is_running(pid: int) -> bool:
    """Return whether a direct child is still live without reaping it.

    The live executor needs an unreaped child to prove that its process-group
    ID cannot have been recycled. A transiently unavailable owner lock is not
    enough evidence to terminate that child or its descendants.
    """

    if pid <= 0:
        return False
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        payload = stat_path.read_text(encoding="utf-8")
        _prefix, suffix = payload.rsplit(")", 1)
        state = suffix.split(maxsplit=1)[0]
        return state != "Z"
    except (FileNotFoundError, IndexError, OSError, ValueError):
        pass
    try:
        result = subprocess.run(
            ("ps", "-o", "stat=", "-p", str(pid)),
            check=False,
            capture_output=True,
            text=True,
            timeout=1,
        )
    except (OSError, subprocess.SubprocessError):
        # Fail open: the caller still owns this unreaped child, and uncertain
        # liveness must not cause a process-group signal.
        return True
    state = result.stdout.strip()
    return result.returncode == 0 and bool(state) and not state.startswith("Z")


def _tracked_worker_process_group(record: tp.Mapping[str, JsonValue]) -> int | None:
    """Return the worker process group only when its persisted identity matches."""

    worker_pid = record.get("worker_pid")
    marker = record.get("worker_start_marker")
    worker_token = record.get("worker_token")
    if (
        not isinstance(worker_pid, int)
        or worker_pid <= 0
        or not isinstance(marker, str)
        or not marker
        or not isinstance(worker_token, str)
        or not worker_token
    ):
        return None
    if _process_start_marker(worker_pid) != marker or not _process_has_worker_token(worker_pid, worker_token):
        return None
    try:
        return worker_pid if os.getpgid(worker_pid) == worker_pid and os.getsid(worker_pid) == worker_pid else None
    except (PermissionError, ProcessLookupError):
        return None


def _group_has_live_members(process_group: int) -> bool:
    """Observe live group members; inspection failure is never quiescence.

    Zombies cannot execute and need not hold up retirement. The caller retains
    an unreaped group/session leader until signalling and observation finish.
    """

    result = subprocess.run(("ps", "-axo", "pgid=,stat="), capture_output=True, text=True, check=False, timeout=2)
    if result.returncode != 0:
        raise RuntimeError("native-local process-group inspection failed")
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2 or not fields[0].isdigit():
            raise RuntimeError("native-local process-group inspection was malformed")
        if int(fields[0]) == process_group and not fields[1].startswith("Z"):
            return True
    return False


def _quiesce_process_group(process_group: int, *, grace_s: float = 0.25) -> bool:
    """TERM/KILL an owned group and confirm no executable member remains."""

    if not _group_has_live_members(process_group):
        return True
    for signal_number, wait_s in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 2.0)):
        try:
            os.killpg(process_group, signal_number)
        except (PermissionError, ProcessLookupError):
            return not _group_has_live_members(process_group)
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline:
            if not _group_has_live_members(process_group):
                return True
            time.sleep(0.01)
    return not _group_has_live_members(process_group)


def _owned_command_group(record: tp.Mapping[str, JsonValue]) -> int | None:
    """Validate the gated command leader before crash/forced-stop cleanup."""

    pid = record.get("command_pid")
    marker = record.get("command_start_marker")
    worker_pid = record.get("worker_pid")
    if not isinstance(pid, int) or pid <= 0 or not isinstance(marker, str) or not marker:
        return None
    if _process_start_marker(pid) != marker:
        return None
    try:
        return pid if os.getpgid(pid) == pid and os.getsid(pid) == worker_pid else None
    except (PermissionError, ProcessLookupError):
        return None


def _owned_session_command_group(record: tp.Mapping[str, JsonValue], session: int) -> int | None:
    """Recover an orphan group when an unreaped direct worker pins its session.

    Init may already have reaped the command leader after the worker crashes.
    Validate a surviving member's current group/session, never the recorded
    group number alone. Reconstructed executors do not call this fallback.
    """

    group = record.get("command_pid")
    if not isinstance(group, int) or group <= 0 or record.get("worker_pid") != session:
        return None
    result = subprocess.run(("ps", "-axo", "pid=,pgid=,stat="), capture_output=True, text=True, check=False, timeout=2)
    if result.returncode != 0:
        raise RuntimeError("native-local process-group ownership inspection failed")
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) != 3 or not fields[0].isdigit() or not fields[1].isdigit():
            raise RuntimeError("native-local process-group ownership inspection was malformed")
        if int(fields[1]) != group or fields[2].startswith("Z"):
            continue
        pid = int(fields[0])
        try:
            if os.getpgid(pid) == group and os.getsid(pid) == session:
                return group
        except ProcessLookupError:
            continue
        return None
    return None


def _terminate_process_group(process_group: int, *, grace_s: float = 0.25) -> None:
    """Best-effort cleanup for commands orphaned by a crashed worker."""

    if process_group <= 0:
        return
    try:
        os.killpg(process_group, signal.SIGTERM)
    except (PermissionError, ProcessLookupError):
        return
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        try:
            os.killpg(process_group, 0)
        except (PermissionError, ProcessLookupError):
            return
        time.sleep(0.01)
    try:
        os.killpg(process_group, signal.SIGKILL)
    except (PermissionError, ProcessLookupError):
        pass


def _terminate_tracked_worker(record: tp.Mapping[str, JsonValue], *, grace_s: float = 0.25) -> bool:
    """Terminate a worker group only after proving its PID was not recycled."""

    process_group = _tracked_worker_process_group(record)
    if process_group is None:
        return False
    command_group = _owned_command_group(record)
    if command_group is not None:
        _quiesce_process_group(command_group, grace_s=grace_s)
    _terminate_process_group(process_group, grace_s=grace_s)
    return True


def _signal_tracked_worker(record: tp.Mapping[str, JsonValue], signal_number: int) -> bool:
    """Signal the verified worker; it forwards stops to its command subgroup."""

    process_group = _tracked_worker_process_group(record)
    if process_group is None:
        return False
    if signal_number == signal.SIGKILL and record.get("command_pid") is not None:
        command_group = _owned_command_group(record)
        if command_group is None or not _quiesce_process_group(command_group):
            return False
    try:
        # New workers own a separate command group and forward TERM themselves.
        # Group signalling would also kill their transient ps inspection child,
        # turning a normal stop into an unobservable cleanup failure.
        if "command_pid" in record:
            os.kill(process_group, signal_number)
        else:
            os.killpg(process_group, signal_number)
    except (PermissionError, ProcessLookupError):
        return False
    return True


def _environment(spec: tp.Mapping[str, JsonValue]) -> dict[str, str]:
    runtime_env = spec.get("runtime_env") or {}
    if not isinstance(runtime_env, tp.Mapping):
        raise ValueError("native-local runtime_env must be a mapping")
    unsupported = sorted(str(key) for key in runtime_env if key != "env_vars")
    if unsupported:
        raise ValueError(f"native-local execution does not support runtime_env field {unsupported[0]!r}")
    legacy = runtime_env.get("env_vars") or {}
    neutral = spec.get("environment") or {}
    if not isinstance(legacy, tp.Mapping) or not isinstance(neutral, tp.Mapping):
        raise ValueError("native-local environment variables must be a mapping")
    result: dict[str, str] = {}
    for raw_name, raw_value in (*legacy.items(), *neutral.items()):
        if not isinstance(raw_name, str) or not raw_name or raw_name != raw_name.strip():
            raise ValueError("native-local environment variable names must be non-empty strings")
        if not isinstance(raw_value, str):
            raise ValueError(f"native-local environment variable {raw_name!r} must have a string value")
        result[raw_name] = raw_value
    result.setdefault("EYVAN_REPLICA_INDEX", "0")
    result.setdefault("EYVAN_REPLICA_COUNT", "1")
    return result


def _validate_gang(spec: tp.Mapping[str, JsonValue]) -> None:
    gang = spec.get("gang")
    if not isinstance(gang, tp.Mapping):
        raise ValueError("native-local workloads require a gang resource contract")
    replicas = gang.get("replicas")
    if isinstance(replicas, bool) or not isinstance(replicas, int):
        raise ValueError("native-local gang replicas must be an integer")
    if replicas != 1:
        raise ValueError("native-local execution supports exactly one replica")
    if not isinstance(gang.get("per_replica"), tp.Mapping):
        raise ValueError("native-local gang per_replica resources must be a mapping")
    topology = gang.get("required_topology") or {}
    if not isinstance(topology, tp.Mapping):
        raise ValueError("native-local required topology must be a mapping")
    if topology:
        raise ValueError("native-local execution does not support topology constraints")


def _trim_incomplete_utf8_tail(data: bytes) -> bytes:
    """Drop a trailing incomplete UTF-8 sequence so force-split chunks do not corrupt codepoints."""

    if not data:
        return data
    end = len(data)
    index = end - 1
    continuations = 0
    while index >= 0 and continuations < 3 and (data[index] & 0xC0) == 0x80:
        index -= 1
        continuations += 1
    if index < 0:
        return data
    lead = data[index]
    if lead < 0x80:
        return data
    if (lead & 0xE0) == 0xC0:
        expected = 2
    elif (lead & 0xF0) == 0xE0:
        expected = 3
    elif (lead & 0xF8) == 0xF0:
        expected = 4
    else:
        return data
    if end - index < expected and index > 0:
        return data[:index]
    return data


def _strip_line_ending(raw_line: bytes, *, force_split: bool) -> bytes:
    """Remove a single line terminator without dropping payload bytes from force-split chunks."""

    if raw_line.endswith(b"\r\n"):
        return raw_line[:-2]
    if raw_line.endswith(b"\n"):
        return raw_line[:-1]
    if force_split:
        return raw_line
    return raw_line.rstrip(b"\r\n")


class LocalProcessExecutor:
    """Run fenced workloads as durable, detached processes on one host.

    This backend is intended for development and control-plane integration. It
    does not enforce CPU or memory limits and deliberately supports only a
    single replica.
    """

    def __init__(self, state_dir: str | os.PathLike[str] | None = None) -> None:
        self.state_dir = Path(state_dir).expanduser() if state_dir is not None else _default_state_dir()
        self.jobs_dir = self.state_dir / "jobs"
        self.logs_dir = self.state_dir / "logs"
        self.owners_dir = self.state_dir / "owners"
        self.workloads_dir = self.state_dir / "workloads"
        self.notices_dir = self.state_dir / "notices"
        for directory in (
            self.state_dir,
            self.jobs_dir,
            self.logs_dir,
            self.owners_dir,
            self.workloads_dir,
            self.notices_dir,
        ):
            create_private_directory(directory)
            directory.chmod(0o700)
        self._lock = RLock()
        # Retaining direct children until terminal state lets this live worker
        # safely clean a crashed child's process group: an unreaped child PID
        # cannot be recycled. Reconstructed executors fail closed because they
        # do not have that ownership proof.
        self._worker_processes: dict[str, subprocess.Popen[bytes]] = {}

    def _paths(self, remote_id: str) -> tuple[Path, Path, Path]:
        name = _record_name(remote_id)
        return self.jobs_dir / f"{name}.json", self.logs_dir / f"{name}.log", self.owners_dir / f"{name}.lock"

    def _notice_paths(self, remote_id: str) -> tuple[Path, Path]:
        name = _record_name(remote_id)
        return self.notices_dir / f"{name}.reclaim.json", self.notices_dir / f"{name}.checkpoint.json"

    def _workload_fence_path(self, workload_id: str) -> Path:
        return self.workloads_dir / f"{_record_name(workload_id)}.json"

    def _release_worker_process(self, remote_id: str) -> None:
        process = self._worker_processes.pop(remote_id, None)
        if process is not None:
            process.poll()

    def _owned_unreaped_worker(
        self,
        remote_id: str,
        record: tp.Mapping[str, JsonValue],
    ) -> subprocess.Popen[bytes] | None:
        process = self._worker_processes.get(remote_id)
        worker_pid = record.get("worker_pid")
        if process is None or not isinstance(worker_pid, int) or process.pid != worker_pid:
            return None
        return process

    @staticmethod
    def _logical_workload_id(metadata: tp.Mapping[str, JsonValue]) -> str | None:
        value = metadata.get("eyvan_workload_id")
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip():
            raise ValueError("native-local eyvan_workload_id must be a non-empty string")
        return value

    def _claim_logical_workload(
        self,
        workload_id: str,
        remote_id: str,
        context: MutationContext,
    ) -> None:
        """Fence one logical workload before its attempt process is launched.

        The cell-agent normally prevents an older attempt from reaching this
        worker. Persisting the same high watermark at the execution boundary
        closes the gap for delayed or directly replayed HTTP submissions. The
        caller must hold the logical workload lock through this claim and the
        attempt's record reservation and worker publication.
        """

        path = self._workload_fence_path(workload_id)
        current = _load_json(path)
        predecessor: _LogicalWorkloadFence | None = None
        if current is not None:
            try:
                persisted = _LogicalWorkloadFence.model_validate(current)
            except ValidationError as exc:
                raise StaleMutationError(f"logical workload {workload_id!r} has an invalid persisted fence") from exc
            if context.generation < persisted.generation:
                raise StaleMutationError(
                    f"generation {context.generation} is older than logical workload {workload_id!r} "
                    f"generation {persisted.generation}"
                )
            if context.generation == persisted.generation:
                if persisted.fencing_token != context.fencing_token:
                    raise StaleMutationError(
                        f"fencing token does not own logical workload {workload_id!r} generation {persisted.generation}"
                    )
                if persisted.remote_id != remote_id:
                    raise StaleMutationError(
                        f"submission {remote_id!r} does not own logical workload {workload_id!r} "
                        f"generation {persisted.generation}"
                    )
                return
            predecessor = persisted

        now = _utcnow()
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
                **({"predecessor": model_json(predecessor)} if predecessor is not None else {}),
                "updated_at": now,
            },
        )

    @staticmethod
    def _validate_fence(
        remote_id: str,
        record: tp.Mapping[str, JsonValue],
        context: MutationContext,
    ) -> None:
        generation = record.get("generation")
        if isinstance(generation, bool) or not isinstance(generation, int):
            raise StaleMutationError(f"local execution {remote_id!r} has an invalid persisted generation")
        token = record.get("fencing_token")
        if generation != context.generation:
            raise StaleMutationError(
                f"generation {context.generation} does not own local execution {remote_id!r} generation {generation}"
            )
        if not isinstance(token, str) or token != context.fencing_token:
            raise StaleMutationError(f"fencing token does not own local execution {remote_id!r} generation {generation}")

    def submit(self, spec: tp.Mapping[str, JsonValue], context: MutationContext) -> str:
        """Launch one detached process, or replay an owned submission."""

        entrypoint = spec.get("entrypoint")
        if not isinstance(entrypoint, str) or not entrypoint.strip():
            raise ValueError("native-local workloads require a non-empty entrypoint")
        _validate_gang(spec)
        environment = _environment(spec)
        # Keep per-job runtime environments and unpacked workspaces under the
        # executor's own state directory (or the operator's explicit override)
        # instead of letting dependency_entry / workspace_entry fall back into
        # whatever git repository the agent was launched from.
        runtime_env_dir = os.environ.get("EYVAN_RUNTIME_ENV_DIR")
        environment.setdefault(
            "EYVAN_RUNTIME_ENV_DIR",
            runtime_env_dir if runtime_env_dir else str(self.state_dir / "runtime-envs"),
        )
        workspace_dir = os.environ.get("EYVAN_WORKSPACE_DIR")
        environment.setdefault(
            "EYVAN_WORKSPACE_DIR",
            workspace_dir if workspace_dir else str(self.state_dir / "workspaces"),
        )
        submitted_remote_id = spec.get("submission_id")
        if submitted_remote_id is None:
            remote_id = context.idempotency_key.strip()
        elif isinstance(submitted_remote_id, str):
            remote_id = submitted_remote_id.strip()
        else:
            raise ValueError("native-local submission ID must be a string")
        if not remote_id:
            raise ValueError("native-local submission ID must be non-empty")
        metadata = spec.get("metadata") or {}
        if not is_json_object(metadata):
            raise ValueError("native-local workload metadata must be a JSON object")
        workload_id = self._logical_workload_id(metadata)

        record_path, log_path, owner_path = self._paths(remote_id)
        notice_path, receipt_path = self._notice_paths(remote_id)
        environment.setdefault(RECLAIM_NOTICE_PATH_ENV, str(notice_path))
        environment.setdefault(CHECKPOINT_RECEIPT_PATH_ENV, str(receipt_path))
        with self._lock, ExitStack() as publication_locks:
            # Reject a generation that reuses an existing attempt ID before
            # advancing the logical high watermark. Otherwise a rejected
            # request could fence out the valid running attempt.
            with _file_lock(_record_lock_path(record_path)):
                existing = _load_json(record_path)
                if existing is not None:
                    self._validate_fence(remote_id, existing, context)
            if workload_id is not None:
                # A successor must not retire an absent predecessor between
                # its logical claim and attempt publication. Hold the workload
                # lock through reservation and launch, always before taking an
                # attempt lock (including predecessor retirement).
                workload_path = self._workload_fence_path(workload_id)
                publication_locks.enter_context(_file_lock(_record_lock_path(workload_path)))
                self._claim_logical_workload(workload_id, remote_id, context)
            with _file_lock(_record_lock_path(record_path)):
                existing = _load_json(record_path)
                if existing is not None:
                    self._validate_fence(remote_id, existing, context)
                    return remote_id
                now = _utcnow()
                new_record = _LocalExecutionRecord(
                    remote_id=remote_id,
                    status=ExecutionWorkloadStatus.PENDING,
                    message="waiting for native-local worker",
                    generation=context.generation,
                    fencing_token=context.fencing_token,
                    idempotency_key=context.idempotency_key,
                    entrypoint=entrypoint,
                    environment=environment,
                    metadata=metadata,
                    worker_pid=None,
                    worker_start_marker=None,
                    worker_token=secrets.token_urlsafe(24),
                    stop_requested=False,
                    exit_code=None,
                    created_at=now,
                    updated_at=now,
                    started_at=None,
                    finished_at=None,
                    log_path=str(log_path),
                    owner_path=str(owner_path),
                    notice_path=str(notice_path),
                    receipt_path=str(receipt_path),
                )
                record = model_json(new_record)
                _atomic_write_json(record_path, record)
                # The detached worker only needs the persisted record. Keep
                # agent and native-worker credentials out of its process tree.
                worker_env = {
                    **{key: value for key, value in os.environ.items() if not key.startswith("EYVAN_")},
                    "PYTHONUNBUFFERED": "1",
                }
                try:
                    process = subprocess.Popen(
                        (
                            sys.executable,
                            "-m",
                            "eyvan.worker.runtime.local_worker",
                            "--record",
                            str(record_path),
                            "--log",
                            str(log_path),
                            "--owner",
                            str(owner_path),
                            f"--worker-token={new_record.worker_token}",
                        ),
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        close_fds=True,
                        start_new_session=True,
                        env=worker_env,
                    )
                except BaseException as exc:
                    record.update(
                        {
                            "status": ExecutionWorkloadStatus.FAILED.value,
                            "message": f"failed to launch native-local worker: {exc}",
                            "finished_at": _utcnow(),
                            "updated_at": _utcnow(),
                            "group_quiescent": True,
                        }
                    )
                    _atomic_write_json(record_path, record)
                    raise
                record["worker_pid"] = process.pid
                record["worker_start_marker"] = _process_start_marker(process.pid)
                record["updated_at"] = _utcnow()
                _atomic_write_json(record_path, record)
                self._worker_processes[remote_id] = process

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if _owner_active(owner_path):
                break
            current = _read_record(record_path)
            if current is not None and _persisted_status(remote_id, current) in _TERMINAL_STATES:
                break
            time.sleep(0.01)
        return remote_id

    def _normalize_abandoned(
        self,
        remote_id: str,
        record_path: Path,
        owner_path: Path,
    ) -> JsonObject | None:
        record = _read_record(record_path)
        if record is None:
            return None
        status = _persisted_status(remote_id, record)
        if status in _TERMINAL_STATES:
            if record.get("group_quiescent") is True:
                self._release_worker_process(remote_id)
                return record

            # Legacy terminal records observed only the direct shell. Never
            # turn that assertion into permission for a successor to overlap
            # descendants. An operator must quiesce ambiguous legacy attempts.
            def legacy_ambiguous(current: JsonObject) -> None:
                if current.get("group_quiescent") is not True:
                    current["status"] = ExecutionWorkloadStatus.INDETERMINATE.value
                    current["message"] = "native-local terminal record lacks process-group quiescence proof"

            return _mutate_record(record_path, legacy_ambiguous) or record
        if _owner_active(owner_path):
            return record
        owned_worker = self._owned_unreaped_worker(remote_id, record)
        # ``local_worker`` records RUNNING before it starts the workload child.
        # A direct worker that has already exited while the durable record is
        # still PENDING therefore cannot have left a command behind. Report
        # that as a launch failure immediately instead of waiting for the
        # absence grace and describing an unstarted workload as LOST.
        if status is ExecutionWorkloadStatus.PENDING and owned_worker is not None:
            exit_code = owned_worker.poll()
            if exit_code is not None:

                def launch_failed(current: JsonObject) -> None:
                    if (
                        _persisted_status(remote_id, current) is not ExecutionWorkloadStatus.PENDING
                        or current.get("started_at") is not None
                    ):
                        return
                    current["status"] = ExecutionWorkloadStatus.FAILED.value
                    current["message"] = f"native-local worker exited before recording readiness (exit code {exit_code})"
                    current["exit_code"] = exit_code
                    current["finished_at"] = _utcnow()
                    current["group_quiescent"] = True

                finalized = _mutate_record(record_path, launch_failed) or record
                if _persisted_status(remote_id, finalized) in _TERMINAL_STATES:
                    self._worker_processes.pop(remote_id, None)
                return finalized
        try:
            owner_absent_at = datetime.fromisoformat(str(record.get("owner_absent_at")))
            owner_absence_s = (datetime.now(UTC) - owner_absent_at).total_seconds()
        except (TypeError, ValueError):
            owner_absence_s = 0
        if owner_absence_s < _OWNER_ABSENCE_GRACE_S:

            def mark_owner_absent(current: JsonObject) -> None:
                current.setdefault("owner_absent_at", _utcnow())

            return _mutate_record(record_path, mark_owner_absent) or record
        try:
            created = datetime.fromisoformat(str(record["created_at"]))
            age = (datetime.now(UTC) - created).total_seconds()
        except (KeyError, TypeError, ValueError):
            age = _STARTUP_GRACE_S
        if status is ExecutionWorkloadStatus.PENDING and age < _STARTUP_GRACE_S:
            return record
        if owned_worker is not None and _unreaped_process_is_running(owned_worker.pid):
            # The direct worker remains live. Its advisory ownership lock can
            # briefly disappear around startup or process scheduling, but that
            # is not authority to kill an active process group.
            return record
        quiescent = False
        if owned_worker is not None:
            # Retain the dead direct worker (session leader) through subgroup
            # cleanup. The gate ensures an unpublished child cannot execute.
            command_pid = record.get("command_pid")
            command_group = _owned_command_group(record) or _owned_session_command_group(record, owned_worker.pid)
            if command_pid is None:
                quiescent = _quiesce_process_group(owned_worker.pid)
            elif command_group is not None:
                quiescent = _quiesce_process_group(command_group)
            elif isinstance(command_pid, int) and command_pid > 0:
                quiescent = not _group_has_live_members(command_pid)
            if quiescent:
                try:
                    owned_worker.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    quiescent = False
                else:
                    self._worker_processes.pop(remote_id, None)

        def abandon(current: JsonObject) -> None:
            if _persisted_status(remote_id, current) in _TERMINAL_STATES:
                return
            if quiescent:
                current["status"] = (
                    ExecutionWorkloadStatus.STOPPED.value
                    if bool(current.get("stop_requested"))
                    else ExecutionWorkloadStatus.LOST.value
                )
                current["message"] = (
                    "native-local worker stopped"
                    if bool(current.get("stop_requested"))
                    else "native-local worker crashed"
                )
            else:
                current["status"] = ExecutionWorkloadStatus.INDETERMINATE.value
                current["message"] = "native-local worker ownership is ambiguous; workload was not signalled"
            current["finished_at"] = _utcnow()
            current["group_quiescent"] = quiescent

        return _mutate_record(record_path, abandon) or record

    def status(self, remote_id: str) -> ExecutionWorkloadObservation:
        """Return a canonical status derived from durable worker state."""

        record_path, log_path, owner_path = self._paths(remote_id)
        record = self._normalize_abandoned(remote_id, record_path, owner_path)
        if record is None:
            return ExecutionWorkloadObservation(
                remote_id=remote_id,
                status=ExecutionWorkloadStatus.NOT_FOUND,
                message="local execution is absent",
            )
        status = _persisted_status(remote_id, record)
        if status in _TERMINAL_STATES:
            self._release_worker_process(remote_id)
        message = _persisted_message(remote_id, record)
        if (
            status is ExecutionWorkloadStatus.FAILED
            and record.get("exit_code") == CROSS_REGION_TRANSFER_BANNED_EXIT_CODE
            and not message.startswith(f"{CROSS_REGION_TRANSFER_BANNED_REASON}:")
        ):
            raw_log = record.get("log_path")
            resolved_log = Path(raw_log) if isinstance(raw_log, str) and raw_log.strip() else log_path
            message = _format_cross_region_ban_message(resolved_log)
        return ExecutionWorkloadObservation(
            remote_id=remote_id,
            status=status,
            message=message,
        )

    def stop(self, remote_id: str, context: MutationContext) -> None:
        """Request termination of the worker's detached process group."""

        record_path, _log_path, owner_path = self._paths(remote_id)
        with self._lock:
            record = _read_record(record_path)
            if record is None:
                return
            self._validate_fence(remote_id, record, context)
            if _persisted_status(remote_id, record) in _TERMINAL_STATES:
                return

            def request_stop(current: JsonObject) -> None:
                current["stop_requested"] = True
                current["message"] = "native-local stop requested"

            record = _mutate_record(record_path, request_stop) or record
            if _owner_active(owner_path):
                _signal_tracked_worker(record, signal.SIGTERM)

    def notify_reclaim(
        self,
        remote_id: str,
        *,
        notice: tp.Mapping[str, JsonValue],
        grace_s: int,
        context: MutationContext,
    ) -> None:
        """Write an atomic reclaim notice file and deliver a graceful signal."""

        record_path, _log_path, owner_path = self._paths(remote_id)
        default_notice_path, _default_receipt_path = self._notice_paths(remote_id)
        with self._lock:
            record = _read_record(record_path)
            if record is None:
                return
            self._validate_fence(remote_id, record, context)
            if _persisted_status(remote_id, record) in _TERMINAL_STATES:
                return
            raw_notice_path = record.get("notice_path")
            notice_path = (
                Path(raw_notice_path)
                if isinstance(raw_notice_path, str) and raw_notice_path.strip()
                else default_notice_path
            )
            _atomic_write_json(notice_path, notice)
            effective_grace = float(max(grace_s, 1))

            def record_reclaim_notice(current: JsonObject) -> None:
                current["stop_grace_s"] = effective_grace
                current["reclaim_notified"] = True
                current["stop_requested"] = True
                current["message"] = "native-local graceful reclaim notice delivered"

            record = _mutate_record(record_path, record_reclaim_notice) or record
            if _owner_active(owner_path):
                graceful_signal = getattr(signal, "SIGUSR1", signal.SIGTERM)
                _signal_tracked_worker(record, graceful_signal)

    def read_checkpoint_uri(self, remote_id: str) -> str | None:
        """Return any checkpoint URI persisted by the worker or acknowledged by the job."""

        record_path, _log_path, _owner_path = self._paths(remote_id)
        _default_notice_path, default_receipt_path = self._notice_paths(remote_id)
        record = _read_record(record_path)
        if record is not None:
            stored = record.get("checkpoint_uri")
            if isinstance(stored, str) and stored.strip():
                return stored.strip()
            raw_receipt_path = record.get("receipt_path")
            receipt_path = (
                Path(raw_receipt_path)
                if isinstance(raw_receipt_path, str) and raw_receipt_path.strip()
                else default_receipt_path
            )
        else:
            receipt_path = default_receipt_path
        receipt = read_checkpoint_receipt(receipt_path)
        if receipt is None:
            return None
        uri = receipt.get("checkpoint_uri")
        return uri.strip() if isinstance(uri, str) and uri.strip() else None

    def retire(self, remote_id: str, context: MutationContext) -> None:
        """Stop a superseded execution and wait until its process group exits."""

        record_path, _log_path, owner_path = self._paths(remote_id)
        record = _read_record(record_path)
        if record is None:
            return
        self._validate_fence(remote_id, record, context)
        self.stop(remote_id, context)
        timeout_s = max(_environment_float("EYVAN_SUPERSEDE_WAIT_TIMEOUT_S", default=300.0), 0.1)
        poll_s = max(_environment_float("EYVAN_SUPERSEDE_POLL_S", default=0.1), 0.01)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            status = self.status(remote_id).status
            if status in _TERMINAL_STATES:
                return
            if status is ExecutionWorkloadStatus.INDETERMINATE:
                # Stop polling -- an INDETERMINATE record is not re-derived, so
                # waiting cannot change it -- but fall through rather than
                # raising here. Below is the SIGKILL that forces a workload of
                # unknown state to terminal, and it is the recovery this very
                # condition exists to trigger; raising first made it
                # unreachable and left the successor unable to start.
                break
            time.sleep(poll_s)
        current = _read_record(record_path)
        if current is not None and _owner_active(owner_path):
            _signal_tracked_worker(current, signal.SIGKILL)

        def ambiguous(value: JsonObject) -> None:
            if _persisted_status(remote_id, value) in _TERMINAL_STATES:
                return
            value["stop_requested"] = True
            value["status"] = ExecutionWorkloadStatus.INDETERMINATE.value
            value["message"] = "native-local retirement timed out; workload ownership is ambiguous"
            value["finished_at"] = _utcnow()

        _mutate_record(record_path, ambiguous)
        raise RuntimeError(f"cannot safely retire native-local execution {remote_id!r} before its timeout")

    def checkpoint(
        self,
        remote_id: str,
        *,
        hook: str,
        output_uri: str | None,
        context: MutationContext,
    ) -> str:
        """Launch an external checkpoint hook after validating source ownership."""

        if not hook or not hook.strip():
            raise ValueError("native-local checkpoint hook must be non-empty")
        source_path, _log_path, _owner_path = self._paths(remote_id)
        source = _read_record(source_path)
        if source is None:
            raise RuntimeError(f"local execution {remote_id!r} is absent")
        self._validate_fence(remote_id, source, context)
        return self.submit(
            {
                "entrypoint": hook,
                "submission_id": context.idempotency_key,
                "environment": {
                    "EYVAN_SOURCE_JOB_ID": remote_id,
                    "EYVAN_CHECKPOINT_URI": output_uri or "",
                },
                "gang": {"replicas": 1, "per_replica": {"cpu": 1}},
                "metadata": {"eyvan_operation": "checkpoint", "eyvan_source_job_id": remote_id},
            },
            context,
        )

    def logs(self, remote_id: str) -> str:
        """Return the durable combined stdout/stderr stream."""

        _record_path, log_path, _owner_path = self._paths(remote_id)
        try:
            return log_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def log_page(self, remote_id: str, *, after: str = "", limit: int = 200) -> LogPage:
        """Return one byte-offset resumable page of log lines for a local execution.

        Uses ``b<byte_offset>`` cursors so live following, ``eyvan run``, and
        settlement log archiving stream incrementally from ``log_path`` without
        colliding with the legacy ``rank=`` / ``attempt:`` delimiter grammar.
        """

        record_path, log_path, _owner_path = self._paths(remote_id)
        record = _read_record(record_path)
        if record is None:
            return LogPage(unavailable=True, detail="no durable execution record")
        if not after:
            offset = 0
        else:
            match = _LOCAL_LOG_CURSOR.fullmatch(after)
            if match is None:
                return LogPage(unavailable=True, detail=f"invalid local log cursor {after!r}")
            offset = int(match.group(1))
        raw_pid = record.get("command_pid")
        pid = raw_pid if isinstance(raw_pid, int) and not isinstance(raw_pid, bool) and raw_pid > 0 else None
        is_terminal = _persisted_status(remote_id, record) in _TERMINAL_STATES
        max_lines = max(1, limit)
        try:
            with log_path.open("rb") as stream:
                size = os.fstat(stream.fileno()).st_size
                if offset > size:
                    return LogPage(
                        unavailable=True,
                        detail=f"local log cursor {after!r} exceeds log size ({size} bytes)",
                    )
                if offset == size:
                    return LogPage(records=(), next_cursor=after)
                stream.seek(offset)
                records: list[LogRecord] = []
                bytes_read = 0
                while len(records) < max_lines and bytes_read < _LOCAL_LOG_PAGE_MAX_BYTES:
                    remaining = _LOCAL_LOG_PAGE_MAX_BYTES - bytes_read
                    raw_line = stream.readline(remaining)
                    if not raw_line:
                        break
                    force_split = False
                    if not raw_line.endswith(b"\n"):
                        at_eof = offset + len(raw_line) >= size
                        if at_eof:
                            # Hold back a partial trailing line while the workload is still
                            # running so live polling does not split one line across records.
                            if not is_terminal:
                                break
                        elif records:
                            # This line crosses the page byte budget; stop here so the next
                            # page reads it cleanly from its start instead of splitting it.
                            break
                        else:
                            # Single line reaches or exceeds _LOCAL_LOG_PAGE_MAX_BYTES.
                            # Consume an immediately adjacent line terminator so exact-boundary
                            # lines do not emit a phantom blank line on the next page.
                            peek = stream.read(2)
                            if raw_line.endswith(b"\r") and peek.startswith(b"\n"):
                                raw_line += b"\n"
                            elif peek.startswith(b"\r\n"):
                                raw_line += b"\r\n"
                            elif peek.startswith(b"\n"):
                                raw_line += b"\n"
                            else:
                                raw_line = _trim_incomplete_utf8_tail(raw_line)
                                if raw_line.endswith(b"\r") and len(raw_line) > 1:
                                    raw_line = raw_line[:-1]
                                force_split = True
                    offset += len(raw_line)
                    bytes_read += len(raw_line)
                    payload = _strip_line_ending(raw_line, force_split=force_split)
                    records.append(
                        LogRecord(
                            message=payload.decode("utf-8", "replace"),
                            stream="unknown",
                            cursor=f"b{offset}",
                            rank=0,
                            pid=pid,
                        )
                    )
                    if force_split:
                        break
        except FileNotFoundError:
            if offset > 0:
                return LogPage(unavailable=True, detail="local workload log file is missing")
            return LogPage(records=(), next_cursor=after)
        except OSError as exc:
            return LogPage(unavailable=True, detail=f"could not read local workload log: {exc}")
        next_cursor = records[-1].cursor if records else after
        return LogPage(records=tuple(records), next_cursor=next_cursor)


__all__ = ("LocalProcessExecutor",)
