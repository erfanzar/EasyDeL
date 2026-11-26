# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Local-only supervision for Eyvan's zero-configuration control plane."""

from __future__ import annotations

import ctypes
import fcntl
import hashlib
import ipaddress
import json
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time
import typing as tp
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import httpx

from eyvan._internal.compat import StrEnum
from eyvan._internal.json_types import JsonObject, is_json_object
from eyvan._internal.workspace import create_private_directory
from eyvan.sdk.client.endpoint import DEFAULT_CONTROL_PLANE_URL, ControlPlaneEndpoint
from eyvan.server.local.profile import DEFAULT_LOCAL_ARTIFACT_TOKEN_KEY_PATH, DEFAULT_LOCAL_CONTROL_PLANE_PATH

DEFAULT_LOCAL_DAEMON_DIRECTORY = DEFAULT_LOCAL_CONTROL_PLANE_PATH.parent
DEFAULT_LOCAL_MANAGED_ARTIFACTS_PATH = DEFAULT_LOCAL_CONTROL_PLANE_PATH.parent / "managed-artifacts"
DEFAULT_LOCAL_CELL_AGENT_URL = "http://127.0.0.1:8090"


class LocalDaemonError(RuntimeError):
    """Base error for local Eyvan control-plane supervision."""


class LocalDaemonStateError(LocalDaemonError):
    """A local daemon metadata or log path is unsafe to use."""


class LocalDaemonStartupError(LocalDaemonError):
    """Eyvan could not start a usable local control-plane process."""

    def __init__(self, endpoint: ControlPlaneEndpoint, log_path: Path, reason: str) -> None:
        self.endpoint = endpoint
        self.log_path = log_path
        self.reason = reason
        super().__init__(f"Eyvan could not start its local control plane at {endpoint.url}: {reason}. See {log_path}.")


class LocalDaemonConfigurationError(LocalDaemonError):
    """A live owned daemon does not match the requested local runtime configuration."""


class LocalDaemonState(StrEnum):
    """The outcome of one local control-plane readiness check."""

    SKIPPED = "skipped"
    REUSED = "reused"
    STARTED = "started"
    STOPPED = "stopped"


@dataclass(frozen=True, slots=True)
class LocalDaemonTarget:
    """A literal HTTP loopback address that can safely host an Eyvan daemon."""

    endpoint: ControlPlaneEndpoint
    host: str
    port: int

    @classmethod
    def from_endpoint(cls, endpoint: ControlPlaneEndpoint) -> LocalDaemonTarget | None:
        """Return a launchable literal-loopback target, if this endpoint has one."""

        parsed = urlsplit(endpoint.url)
        if parsed.scheme != "http" or parsed.path or parsed.hostname is None:
            return None
        try:
            address = ipaddress.ip_address(parsed.hostname)
        except ValueError:
            return None
        if not address.is_loopback:
            return None
        port = parsed.port
        if port is None:
            port = 80
        if not 1 <= port <= 65535:
            return None
        return cls(endpoint=endpoint, host=address.compressed, port=port)


@dataclass(frozen=True, slots=True)
class LocalDaemonProfile:
    """Persistent local paths and public URL used by one daemon endpoint."""

    endpoint: ControlPlaneEndpoint
    state_path: Path
    artifact_token_key_path: Path
    artifact_store_path: Path

    @classmethod
    def for_target(cls, target: LocalDaemonTarget, *, profile_directory: Path) -> LocalDaemonProfile:
        """Return the stable default profile or a fully isolated endpoint-specific profile."""

        if target.endpoint.url == DEFAULT_CONTROL_PLANE_URL:
            return cls(
                endpoint=target.endpoint,
                state_path=DEFAULT_LOCAL_CONTROL_PLANE_PATH,
                artifact_token_key_path=DEFAULT_LOCAL_ARTIFACT_TOKEN_KEY_PATH,
                artifact_store_path=DEFAULT_LOCAL_MANAGED_ARTIFACTS_PATH,
            )
        return cls(
            endpoint=target.endpoint,
            state_path=profile_directory / "control-plane.json",
            artifact_token_key_path=profile_directory / "artifact-token.key",
            artifact_store_path=profile_directory / "managed-artifacts",
        )

    @property
    def artifact_store_uri(self) -> str:
        """Return the local managed-artifact root in the form accepted by Eyvan settings."""

        return str(self.artifact_store_path)


@dataclass(frozen=True, slots=True)
class LocalDaemonPaths:
    """Owner-only files used to supervise one or more local Eyvan endpoints."""

    directory: Path = DEFAULT_LOCAL_DAEMON_DIRECTORY

    @property
    def resolved_directory(self) -> Path:
        """Return the expanded directory holding daemon state."""

        return self.directory.expanduser()

    def record_path(self, target: LocalDaemonTarget) -> Path:
        """Return the endpoint-specific PID metadata path."""

        return self.resolved_directory / f"control-plane-{self._identifier(target)}.pid"

    def log_path(self, target: LocalDaemonTarget) -> Path:
        """Return the endpoint-specific control-plane log path."""

        return self.resolved_directory / f"control-plane-{self._identifier(target)}.log"

    def lock_path(self, target: LocalDaemonTarget) -> Path:
        """Return the endpoint-specific startup lock path."""

        return self.resolved_directory / f"control-plane-{self._identifier(target)}.lock"

    def agent_record_path(self, target: LocalDaemonTarget) -> Path:
        """Return the endpoint-specific PID record for an internally supervised cell agent."""

        return self.resolved_directory / f"cell-agent-{self._identifier(target)}.pid"

    def agent_log_path(self, target: LocalDaemonTarget) -> Path:
        """Return the endpoint-specific log path for an internally supervised cell agent."""

        return self.resolved_directory / f"cell-agent-{self._identifier(target)}.log"

    def agent_lock_path(self, target: LocalDaemonTarget) -> Path:
        """Return the endpoint-specific startup lock for an internally supervised cell agent."""

        return self.resolved_directory / f"cell-agent-{self._identifier(target)}.lock"

    def profile_directory(self, target: LocalDaemonTarget) -> Path:
        """Return the endpoint-specific profile directory for a non-default daemon."""

        return self.resolved_directory / "profiles" / self._identifier(target)

    def profile(self, target: LocalDaemonTarget) -> LocalDaemonProfile:
        """Return the persistent profile used by this target's spawned process."""

        return LocalDaemonProfile.for_target(target, profile_directory=self.profile_directory(target))

    @staticmethod
    def _identifier(target: LocalDaemonTarget) -> str:
        digest = hashlib.sha256(target.endpoint.url.encode("utf-8")).hexdigest()[:16]
        return f"{target.port}-{digest}"


@dataclass(frozen=True, slots=True)
class LocalDaemonRecord:
    """PID metadata for a daemon process created by this supervisor."""

    endpoint: ControlPlaneEndpoint
    pid: int
    runtime_profile_path: str | None = None
    #: The source digest of the build this plane was launched from. The cell
    #: agent is judged against it, not against whichever CLI happens to run.
    source_digest: str | None = None

    def as_json(self) -> JsonObject:
        """Encode the record without any non-JSON values."""

        document: JsonObject = {"endpoint": self.endpoint.url, "pid": self.pid}
        if self.runtime_profile_path is not None:
            document["runtime_profile_path"] = self.runtime_profile_path
        if self.source_digest is not None:
            document["source_digest"] = self.source_digest
        return document

    @classmethod
    def from_json(cls, value: JsonObject) -> LocalDaemonRecord | None:
        """Decode valid record data while treating malformed stale metadata as absent."""

        endpoint_value = value.get("endpoint")
        pid = value.get("pid")
        runtime_profile_path = value.get("runtime_profile_path")
        source_digest = value.get("source_digest")
        if (
            not isinstance(endpoint_value, str)
            or isinstance(pid, bool)
            or not isinstance(pid, int)
            or pid <= 0
            or (runtime_profile_path is not None and not isinstance(runtime_profile_path, str))
        ):
            return None
        try:
            endpoint = ControlPlaneEndpoint.parse(endpoint_value)
        except ValueError:
            return None
        return cls(
            endpoint=endpoint,
            pid=pid,
            runtime_profile_path=runtime_profile_path,
            source_digest=source_digest if isinstance(source_digest, str) else None,
        )


def runtime_profile_identifier(path: Path | None) -> str | None:
    """Return the stable local identity for an optional private runtime profile."""

    if path is None:
        return None
    return str(path.expanduser().resolve())


@dataclass(frozen=True, slots=True)
class LocalDaemonResult:
    """A completed readiness decision for one configured control-plane endpoint."""

    endpoint: ControlPlaneEndpoint
    state: LocalDaemonState
    pid: int | None
    log_path: Path | None


@dataclass(frozen=True, slots=True)
class LocalDaemonCommand:
    """One exact subprocess invocation for a loopback Eyvan control plane."""

    target: LocalDaemonTarget
    log_path: Path
    profile: LocalDaemonProfile
    runtime_profile_path: Path | None = None
    #: Where the server rewrites the owner record once it is actually serving.
    #: The supervisor writes this file first, from the PID it launched, so a
    #: process that dies before binding is still cleaned up. But only the server
    #: knows which PID ended up holding the port, and a record naming any other
    #: process makes every later command refuse with "not owned by the selected
    #: native GCP runtime profile" until it is edited by hand.
    record_path: Path | None = None
    #: Start the daemon with ``--team`` (team accounts). Ownership matching reads
    #: only ``--host``/``--port``/``--local-daemon``/``--runtime-profile``, so a
    #: daemon is recognized as owned whether or not it carries this flag.
    team: bool = False

    @property
    def argv(self) -> tuple[str, ...]:
        """Return the command that starts the local API server in this interpreter."""

        command = (
            sys.executable,
            "-m",
            "eyvan.server.api.app",
            "--host",
            self.target.host,
            "--port",
            str(self.target.port),
            "--local-daemon",
            "--public-url",
            self.profile.endpoint.url,
            "--local-state-path",
            str(self.profile.state_path),
            "--local-artifact-token-key-path",
            str(self.profile.artifact_token_key_path),
            "--artifact-store-uri",
            self.profile.artifact_store_uri,
            *(() if self.record_path is None else ("--local-daemon-record-path", str(self.record_path))),
            *(("--team",) if self.team else ()),
            "--log-level",
            "info",
        )
        if self.runtime_profile_path is None:
            return command
        return (*command[:-2], "--runtime-profile", str(self.runtime_profile_path), *command[-2:])


class LocalDaemonProcess(tp.Protocol):
    """The minimal process lifecycle needed by the daemon supervisor."""

    @property
    def pid(self) -> int:
        """Return the operating-system process identifier."""

        ...

    def poll(self) -> int | None:
        """Return an exit status when the process has stopped."""

        ...

    def terminate(self) -> None:
        """Ask the process to stop."""

        ...

    def kill(self) -> None:
        """Force the process to stop."""

        ...

    def wait(self, timeout: float | None = None) -> int:
        """Wait for the process to stop and return its exit status."""

        ...


class LocalDaemonLauncher(tp.Protocol):
    """Starts a local Eyvan server from an immutable command description."""

    def start(self, command: LocalDaemonCommand) -> LocalDaemonProcess:
        """Launch one control-plane process without waiting for readiness."""

        ...


HealthProbe: tp.TypeAlias = tp.Callable[[ControlPlaneEndpoint], bool]
Clock: tp.TypeAlias = tp.Callable[[], float]
Sleeper: tp.TypeAlias = tp.Callable[[float], None]
ProcessArgumentReader: tp.TypeAlias = tp.Callable[[int], tuple[str, ...] | None]
ProcessLiveness: tp.TypeAlias = tp.Callable[[int], bool]
ProcessTerminator: tp.TypeAlias = tp.Callable[[int], None]
ProcessKiller: tp.TypeAlias = tp.Callable[[int], None]
ProcessVerifier: tp.TypeAlias = tp.Callable[[int], bool]


class VerifiedProcessStopOutcome(StrEnum):
    """Result of stopping a process whose identity was checked before signalling it."""

    STOPPED = "stopped"
    PROCESS_CHANGED = "process_changed"
    TIMED_OUT = "timed_out"


def prepare_owner_only_directory(directory: Path) -> None:
    """Create an owner-only local state directory without following a file path."""

    create_private_directory(directory)
    try:
        os.chmod(directory, 0o700)
    except OSError:
        pass


def require_owner_only_regular_file(path: Path, *, description: str) -> None:
    """Reject a symlink, directory, or special file before reading or replacing it."""

    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise LocalDaemonStateError(f"local Eyvan {description} {path} must be a regular file")


def require_owner_only_record_file(path: Path) -> None:
    """Reject a PID record that another account can replace or read."""

    require_owner_only_regular_file(path, description="daemon PID record")
    metadata = path.lstat()
    if metadata.st_uid != os.geteuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise LocalDaemonStateError(
            f"local Eyvan daemon PID record {path} must be owned by the current user and not group- or world-accessible"
        )


def open_owner_only_log(path: Path) -> tp.BinaryIO:
    """Open an append-only owner-only log without following a pre-existing symlink."""

    prepare_owner_only_directory(path.parent)
    if path.exists():
        require_owner_only_regular_file(path, description="daemon log")
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC | os.O_NOFOLLOW,
            0o600,
        )
    except OSError as exc:
        raise LocalDaemonStateError(f"could not open local Eyvan daemon log {path}: {exc}") from exc
    try:
        os.fchmod(descriptor, 0o600)
        return os.fdopen(descriptor, "ab", closefd=True)
    except BaseException:
        os.close(descriptor)
        raise


@contextmanager
def owner_only_startup_lock(path: Path) -> tp.Iterator[None]:
    """Serialize startup attempts for one endpoint across local processes."""

    prepare_owner_only_directory(path.parent)
    if path.exists():
        require_owner_only_regular_file(path, description="daemon startup lock")
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    except OSError as exc:
        raise LocalDaemonStateError(f"could not open local Eyvan daemon startup lock {path}: {exc}") from exc
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


@contextmanager
def record_publication_lock(record_path: Path) -> tp.Iterator[None]:
    """Serialize the supervisor's fallback write with the served process's own write.

    Distinct from the endpoint startup lock, which the supervisor holds for the
    whole boot and which the child therefore cannot take. Without this, a child
    that published its record (and its build digest) between the supervisor's
    ownership read and its fallback write had that record replaced by one with
    an unknown digest.
    """

    with owner_only_startup_lock(record_path.with_name(record_path.name + ".publish.lock")):
        yield


def write_owner_only_record(path: Path, record: LocalDaemonRecord) -> None:
    """Atomically replace one PID record after validating the target file shape."""

    prepare_owner_only_directory(path.parent)
    if path.exists():
        require_owner_only_record_file(path)
    temporary_path: Path | None = None
    descriptor: int | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary_path = Path(temporary_name)
        os.fchmod(descriptor, 0o600)
        handle = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = None
        with handle:
            json.dump(record.as_json(), handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
        os.chmod(path, 0o600)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as exc:
        raise LocalDaemonStateError(f"could not persist local Eyvan daemon PID record {path}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def read_owner_only_record(path: Path) -> LocalDaemonRecord | None:
    """Read a valid record while ignoring stale malformed JSON metadata."""

    if not path.exists():
        return None
    require_owner_only_record_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not is_json_object(value):
        return None
    return LocalDaemonRecord.from_json(value)


def _serving_record(
    endpoint: str | ControlPlaneEndpoint,
    paths: LocalDaemonPaths | None,
) -> LocalDaemonRecord | None:
    """Return the owner-only record of a live, Eyvan-owned plane at ``endpoint``."""

    configured_endpoint = ControlPlaneEndpoint.parse(endpoint)
    target = LocalDaemonTarget.from_endpoint(configured_endpoint)
    if target is None:
        return None
    try:
        record = read_owner_only_record((paths or LocalDaemonPaths()).record_path(target))
    except (LocalDaemonError, OSError):
        return None
    if record is None or record.endpoint != configured_endpoint or not process_is_alive(record.pid):
        return None
    return record


def serving_runtime_profile_path(
    endpoint: str | ControlPlaneEndpoint,
    *,
    paths: LocalDaemonPaths | None = None,
) -> Path | None:
    """Return the runtime profile a live, Eyvan-owned loopback plane was started with.

    ``eyvan runtime deploy --profile PATH`` starts a plane under a profile other
    than the default, and records which one in the plane's owner-only PID
    record. Every later client on this machine -- the SDK's ``init()``, doctor --
    must follow that record rather than assume the default path: assuming it
    made ``init()`` refuse the very plane the operator had just deployed, and
    its advice ("restart the control plane") would have replaced that plane with
    one bound to a different project.

    Only a record naming this exact endpoint, a live PID, and a profile is an
    answer. Anything else is ``None`` and callers keep their default.
    """

    record = _serving_record(endpoint, paths)
    if record is None or record.runtime_profile_path is None:
        return None
    return Path(record.runtime_profile_path)


def serving_source_digest(
    endpoint: str | ControlPlaneEndpoint,
    *,
    paths: LocalDaemonPaths | None = None,
) -> str | None:
    """Return the build a live, Eyvan-owned loopback plane was launched from.

    The plane stages the worker wheel, so it -- not the CLI that happens to be
    running -- is the build its cell agent must match. Comparing with an edited
    CLI tree could replace only the agent while the plane and staged wheel keep
    the old build, causing new cells to be rejected as stale.
    ``None`` for a plane that predates the record field, or none at all.
    """

    record = _serving_record(endpoint, paths)
    return None if record is None else record.source_digest


def clear_owner_only_record(path: Path, *, pid: int) -> None:
    """Remove only the PID record for the process this supervisor just stopped."""

    record = read_owner_only_record(path)
    if record is None or record.pid != pid:
        return
    path.unlink()


def process_is_alive(pid: int) -> bool:
    """Check a recorded PID without ever sending it a signal.

    A zombie -- exited but not yet reaped by its parent -- still answers
    signal 0, but it is not running: it holds no port and executes nothing.
    The supervisor's own children become zombies exactly when a stop
    succeeds while the parent that launched them is still alive, so a
    zombie must read as gone or a verified stop never converges.
    """

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        completed = subprocess.run(
            ("ps", "-o", "state=", "-p", str(pid)),
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        completed = None
    if completed is None or completed.returncode != 0:
        # The process can exit between signal zero and ps. Recheck before
        # reporting a survivor; an unreadable but existing PID remains live.
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True
    return not completed.stdout.strip().startswith(("Z", "z"))


def _parse_darwin_process_arguments(data: bytes) -> tuple[str, ...] | None:
    """Decode KERN_PROCARGS2's argc, executable, padding, and NUL-delimited argv."""

    integer_size = ctypes.sizeof(ctypes.c_int)
    if len(data) <= integer_size:
        return None
    argc = int.from_bytes(data[:integer_size], sys.byteorder, signed=True)
    if argc <= 0 or argc > len(data):
        return None
    executable_end = data.find(b"\0", integer_size)
    if executable_end < 0:
        return None
    offset = executable_end + 1
    while offset < len(data) and data[offset] == 0:
        offset += 1
    arguments: list[str] = []
    for _ in range(argc):
        end = data.find(b"\0", offset)
        if end < 0:
            return None
        arguments.append(os.fsdecode(data[offset:end]))
        offset = end + 1
    return tuple(arguments)


def _read_darwin_process_arguments(pid: int) -> tuple[str, ...] | None:
    """Read argv through macOS sysctl, not ps's ambiguous display string."""

    try:
        sysctl = ctypes.CDLL(None, use_errno=True).sysctl
        sysctl.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        sysctl.restype = ctypes.c_int
        # CTL_KERN, KERN_ARGMAX supplies the buffer bound for KERN_PROCARGS2.
        argmax = ctypes.c_int()
        size = ctypes.c_size_t(ctypes.sizeof(argmax))
        if sysctl((ctypes.c_int * 2)(1, 8), 2, ctypes.byref(argmax), ctypes.byref(size), None, 0) != 0:
            return None
        if argmax.value <= 0:
            return None
        buffer = ctypes.create_string_buffer(argmax.value)
        size = ctypes.c_size_t(len(buffer))
        # CTL_KERN, KERN_PROCARGS2, pid returns argc followed by argv and env.
        if sysctl((ctypes.c_int * 3)(1, 49, pid), 3, buffer, ctypes.byref(size), None, 0) != 0:
            return None
        return _parse_darwin_process_arguments(buffer.raw[: size.value])
    except (AttributeError, OSError, ValueError):
        return None


def _read_process_arguments(pid: int) -> tuple[str, ...] | None:
    """Read true process argv; refuse ownership when the OS cannot supply it."""

    if pid <= 0:
        return None
    if sys.platform == "darwin":
        return _read_darwin_process_arguments(pid)
    if sys.platform.startswith("linux"):
        try:
            data = Path(f"/proc/{pid}/cmdline").read_bytes()
        except OSError:
            return None
        if not data or not data.endswith(b"\0"):
            return None
        return tuple(os.fsdecode(argument) for argument in data[:-1].split(b"\0"))
    # Never recover argument boundaries by interpreting a display as a shell.
    return None


def _terminate_process(pid: int) -> None:
    """Request graceful shutdown of one PID after the caller validated its identity."""

    os.kill(pid, signal.SIGTERM)


def force_kill_process(pid: int) -> None:
    """Force-stop one already verified local Eyvan process after its grace period expires."""

    os.kill(pid, signal.SIGKILL)


def _wait_for_process_exit(
    pid: int,
    *,
    process_liveness: ProcessLiveness,
    timeout_s: float,
    poll_interval_s: float,
    clock: Clock,
    sleeper: Sleeper,
) -> bool:
    """Wait for one PID to exit without sending any additional signals."""

    deadline = clock() + timeout_s
    while process_liveness(pid):
        remaining = deadline - clock()
        if remaining <= 0:
            return False
        sleeper(min(poll_interval_s, remaining))
    return True


def stop_verified_process(
    pid: int,
    *,
    process_liveness: ProcessLiveness,
    process_verifier: ProcessVerifier,
    process_terminator: ProcessTerminator,
    process_killer: ProcessKiller,
    graceful_timeout_s: float,
    forced_timeout_s: float,
    poll_interval_s: float,
    clock: Clock,
    sleeper: Sleeper,
) -> VerifiedProcessStopOutcome:
    """Stop a verified owned PID, escalating only after it ignores graceful shutdown.

    The process command is verified again immediately before ``SIGKILL``. This
    prevents a recycled PID from receiving a forced signal after the original
    Eyvan process has exited.
    """

    try:
        process_terminator(pid)
    except ProcessLookupError:
        return VerifiedProcessStopOutcome.STOPPED
    if _wait_for_process_exit(
        pid,
        process_liveness=process_liveness,
        timeout_s=graceful_timeout_s,
        poll_interval_s=poll_interval_s,
        clock=clock,
        sleeper=sleeper,
    ):
        return VerifiedProcessStopOutcome.STOPPED
    if not process_verifier(pid):
        return VerifiedProcessStopOutcome.PROCESS_CHANGED
    try:
        process_killer(pid)
    except ProcessLookupError:
        return VerifiedProcessStopOutcome.STOPPED
    if _wait_for_process_exit(
        pid,
        process_liveness=process_liveness,
        timeout_s=forced_timeout_s,
        poll_interval_s=poll_interval_s,
        clock=clock,
        sleeper=sleeper,
    ):
        return VerifiedProcessStopOutcome.STOPPED
    return VerifiedProcessStopOutcome.TIMED_OUT


def _single_option_value(arguments: tuple[str, ...], option: str) -> str | None:
    """Return an unambiguous command-line option value, if it is present exactly once."""

    values = tuple(arguments[index + 1] for index, value in enumerate(arguments[:-1]) if value == option)
    if len(values) != 1:
        return None
    return values[0]


def _matches_local_daemon_process(
    arguments: tuple[str, ...] | None,
    target: LocalDaemonTarget,
    *,
    runtime_profile_path: str | None,
    require_local_daemon: bool,
) -> bool:
    """Return whether a recorded PID still runs the exact expected Eyvan loopback API."""

    if arguments is None:
        return False
    has_module_invocation = any(
        arguments[index : index + 2] == ("-m", "eyvan.server.api.app") for index in range(len(arguments) - 1)
    )
    if not has_module_invocation:
        return False
    if _single_option_value(arguments, "--host") != target.host:
        return False
    if _single_option_value(arguments, "--port") != str(target.port):
        return False
    if require_local_daemon and "--local-daemon" not in arguments:
        return False
    if runtime_profile_path is not None and _single_option_value(arguments, "--runtime-profile") != runtime_profile_path:
        return False
    return True


@dataclass(frozen=True, slots=True)
class HttpLocalDaemonHealthProbe:
    """Recognize only a healthy Eyvan API response at a literal loopback endpoint."""

    timeout_s: float = 0.25

    def __call__(self, endpoint: ControlPlaneEndpoint) -> bool:
        """Return whether the endpoint currently serves the typed Eyvan health contract."""

        try:
            with httpx.Client(base_url=endpoint.url, timeout=self.timeout_s, trust_env=False) as client:
                response = client.get("/api/v1/health")
        except httpx.HTTPError:
            return False
        if not response.is_success:
            return False
        try:
            payload = response.json()
        except ValueError:
            return False
        return is_json_object(payload) and payload.get("status") == "ok" and payload.get("service") == "eyvan"


@dataclass(frozen=True, slots=True)
class SubprocessLocalDaemonLauncher:
    """Launch the current Eyvan installation as a detached loopback process."""

    def start(self, command: LocalDaemonCommand) -> LocalDaemonProcess:
        """Start the server and redirect all of its output to its owner-only log."""

        with open_owner_only_log(command.log_path) as log:
            return subprocess.Popen(
                command.argv,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )


def _default_clock() -> Clock:
    """Return the monotonic clock used by the default supervisor."""

    return time.monotonic


def _default_sleeper() -> Sleeper:
    """Return the blocking sleep function used by the default supervisor."""

    return time.sleep


@dataclass(frozen=True, slots=True)
class LocalDaemonSupervisor:
    """Start or reuse only a literal-loopback Eyvan server for a client endpoint."""

    paths: LocalDaemonPaths = field(default_factory=LocalDaemonPaths)
    launcher: LocalDaemonLauncher = field(default_factory=SubprocessLocalDaemonLauncher)
    health_probe: HealthProbe = field(default_factory=HttpLocalDaemonHealthProbe)
    #: How long a freshly started server is given to answer.
    #:
    #: Twenty seconds was the budget of a plane holding one local pool. Startup
    #: reconciles every cell the fleet has, so a project with a dozen applied
    #: fleets and cells mid-provision needs longer -- and the old failure was
    #: not "wait more", it was killing the server and leaving nothing running.
    startup_timeout_s: float = 90.0
    poll_interval_s: float = 0.1
    clock: Clock = field(default_factory=_default_clock)
    sleeper: Sleeper = field(default_factory=_default_sleeper)
    process_argument_reader: ProcessArgumentReader = _read_process_arguments
    process_liveness: ProcessLiveness = process_is_alive
    process_terminator: ProcessTerminator = _terminate_process
    process_killer: ProcessKiller = force_kill_process

    def __post_init__(self) -> None:
        if self.startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be positive")
        if self.poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")

    def ensure(
        self,
        endpoint: str | ControlPlaneEndpoint,
        *,
        runtime_profile_path: Path | None = None,
    ) -> LocalDaemonResult:
        """Start an absent local server or reuse a healthy one without touching remote endpoints."""

        configured_endpoint = ControlPlaneEndpoint.parse(endpoint)
        target = LocalDaemonTarget.from_endpoint(configured_endpoint)
        if target is None:
            return LocalDaemonResult(
                endpoint=configured_endpoint,
                state=LocalDaemonState.SKIPPED,
                pid=None,
                log_path=None,
            )
        expanded_runtime_profile_path = (
            runtime_profile_path.expanduser().resolve() if runtime_profile_path is not None else None
        )
        profile_identifier = runtime_profile_identifier(expanded_runtime_profile_path)
        record_path = self.paths.record_path(target)
        log_path = self.paths.log_path(target)
        existing = self._existing_result(configured_endpoint, record_path, log_path, profile_identifier)
        if existing is not None:
            return existing

        with owner_only_startup_lock(self.paths.lock_path(target)):
            existing = self._existing_result(configured_endpoint, record_path, log_path, profile_identifier)
            if existing is not None:
                return existing
            command = LocalDaemonCommand(
                target=target,
                log_path=log_path,
                profile=self.paths.profile(target),
                runtime_profile_path=expanded_runtime_profile_path,
                record_path=record_path,
            )
            try:
                process = self.launcher.start(command)
            except OSError as exc:
                raise LocalDaemonStartupError(configured_endpoint, log_path, str(exc)) from exc
            # A plane that is already serving owns this record, and a process
            # that has not proven it can serve must not take it. Overwriting here
            # is what destroyed a working plane's record: the loser of a startup
            # race stamped its own PID over the winner's, then correctly cleared
            # "its" record on the way out -- leaving a healthy plane with no
            # record at all, and every later command refusing to talk to it.
            #
            # Losing the race is not an error. The other process is a control
            # plane at the endpoint this one wanted, so stop and adopt it.
            read_owner_only_record(record_path)
            try:
                # A record already published by this child is authoritative,
                # including its independently computed source digest. The parent
                # cannot vouch for the child's build from its own cached identity.
                # Re-read under the lock the child publishes under: the earlier
                # record validation may have been overtaken by the child since.
                with record_publication_lock(record_path):
                    current = read_owner_only_record(record_path)
                    if (
                        current is not None
                        and current.pid != process.pid
                        and self.process_liveness(current.pid)
                        and _matches_local_daemon_process(
                            self.process_argument_reader(current.pid),
                            target,
                            runtime_profile_path=current.runtime_profile_path,
                            require_local_daemon=True,
                        )
                    ):
                        # A live PID alone may be a recycled unrelated process.
                        # Only a verified daemon can win this race. It need not
                        # answer another health probe while it is still booting.
                        stop_owned_process(process)
                        if (
                            current.endpoint == configured_endpoint
                            and current.runtime_profile_path == profile_identifier
                        ):
                            return LocalDaemonResult(
                                endpoint=configured_endpoint,
                                state=LocalDaemonState.REUSED,
                                pid=current.pid,
                                log_path=log_path,
                            )
                        raise LocalDaemonStartupError(
                            configured_endpoint,
                            log_path,
                            f"another Eyvan control plane claimed {configured_endpoint.url} while this one was "
                            f"starting, under runtime profile {current.runtime_profile_path}",
                        )
                    if current is None or current.pid != process.pid:
                        write_owner_only_record(
                            record_path,
                            LocalDaemonRecord(
                                endpoint=configured_endpoint,
                                pid=process.pid,
                                runtime_profile_path=profile_identifier,
                                source_digest=None,
                            ),
                        )
            except BaseException:
                stop_owned_process(process)
                raise
            return self._wait_for_ready(process, configured_endpoint, record_path, log_path)

    def restart(
        self,
        endpoint: str | ControlPlaneEndpoint,
        *,
        runtime_profile_path: Path,
    ) -> LocalDaemonResult:
        """Explicitly replace one verified owned daemon with a profile-aware daemon.

        This path is intentionally separate from :meth:`ensure`: normal Eyvan
        commands never stop a running process. A restart accepts only an
        owner-only Eyvan PID record for the requested loopback endpoint and
        checks the recorded PID's command before sending ``SIGTERM``.
        """

        configured_endpoint = ControlPlaneEndpoint.parse(endpoint)
        target = LocalDaemonTarget.from_endpoint(configured_endpoint)
        if target is None:
            raise LocalDaemonConfigurationError(
                "the native GCP runtime can restart only a literal HTTP loopback control plane"
            )
        expanded_runtime_profile_path = runtime_profile_path.expanduser().resolve()
        profile_identifier = runtime_profile_identifier(expanded_runtime_profile_path)
        assert profile_identifier is not None
        record_path = self.paths.record_path(target)
        log_path = self.paths.log_path(target)
        with owner_only_startup_lock(self.paths.lock_path(target)):
            record = read_owner_only_record(record_path)
            if record is None:
                if self.health_probe(configured_endpoint):
                    # Name the record and the remedy. This refusal is correct --
                    # stopping an unidentified process is worse than failing --
                    # but as a bare sentence it is a dead end: the operator
                    # cannot tell whether the record was never written, was
                    # removed, or names a server started outside the supervisor,
                    # and the only way forward was hand-editing JSON.
                    raise LocalDaemonConfigurationError(
                        f"a local Eyvan control plane is already listening at {configured_endpoint.url}, but no "
                        f"owner-only Eyvan PID record identifies it at {record_path}; refusing to stop a process "
                        "this supervisor cannot prove it owns. Either it was started outside Eyvan (stop it "
                        "yourself, then retry), or its record was removed while it kept running."
                    )
            elif record.endpoint != configured_endpoint:
                raise LocalDaemonConfigurationError(
                    f"the recorded Eyvan control-plane endpoint does not match {configured_endpoint.url}; "
                    "refusing to stop it"
                )
            elif self.process_liveness(record.pid):
                if record.runtime_profile_path not in {None, profile_identifier}:
                    raise LocalDaemonConfigurationError(
                        f"the local Eyvan control plane at {configured_endpoint.url} (pid {record.pid}) belongs to "
                        f"native GCP runtime profile {record.runtime_profile_path}, not the selected "
                        f"{profile_identifier}; refusing to stop it. Select that profile, or stop that control plane "
                        "before running a command against this one."
                    )
                self._stop_recorded_daemon(
                    record,
                    target=target,
                    log_path=log_path,
                    expected_runtime_profile_path=profile_identifier,
                )
                clear_owner_only_record(record_path, pid=record.pid)
            else:
                clear_owner_only_record(record_path, pid=record.pid)
        return self.ensure(configured_endpoint, runtime_profile_path=expanded_runtime_profile_path)

    def stop(self, endpoint: str | ControlPlaneEndpoint) -> LocalDaemonResult:
        """Stop the daemon this supervisor's profile directory records for an endpoint.

        The counterpart to :meth:`ensure` for callers that own the profile
        directory: a developer clearing a local plane, or a test harness whose
        isolated home recorded one. Like :meth:`restart`, it accepts only an
        owner-only Eyvan PID record for the requested loopback endpoint and
        checks the recorded PID's command before sending ``SIGTERM``, so a
        recycled PID naming an unrelated process is refused rather than
        killed. A missing record stops nothing: a plane this supervisor
        cannot prove it owns is not this supervisor's to stop.

        Args:
            endpoint: The literal HTTP loopback endpoint to stop.

        Returns:
            The stopped outcome for the endpoint.

        Raises:
            LocalDaemonConfigurationError: The endpoint is not a literal HTTP
                loopback one, or its record names another endpoint.
        """

        configured_endpoint = ControlPlaneEndpoint.parse(endpoint)
        target = LocalDaemonTarget.from_endpoint(configured_endpoint)
        if target is None:
            raise LocalDaemonConfigurationError("the local control plane can stop only a literal HTTP loopback endpoint")
        record_path = self.paths.record_path(target)
        log_path = self.paths.log_path(target)
        with owner_only_startup_lock(self.paths.lock_path(target)):
            record = read_owner_only_record(record_path)
            if record is not None:
                if record.endpoint != configured_endpoint:
                    raise LocalDaemonConfigurationError(
                        f"the recorded Eyvan control-plane endpoint does not match {configured_endpoint.url}; "
                        "refusing to stop it"
                    )
                if self.process_liveness(record.pid):
                    self._stop_recorded_daemon(
                        record,
                        target=target,
                        log_path=log_path,
                        expected_runtime_profile_path=record.runtime_profile_path or "",
                    )
                clear_owner_only_record(record_path, pid=record.pid)
        return LocalDaemonResult(
            endpoint=configured_endpoint,
            state=LocalDaemonState.STOPPED,
            pid=None,
            log_path=log_path,
        )

    def stop_started(self, result: LocalDaemonResult, *, runtime_profile_path: Path) -> None:
        """Stop only the exact profile-aware daemon started by this supervisor.

        This is used to roll back API startup when its paired local cell agent
        cannot become ready. Reused daemons are intentionally left alone.
        """

        if result.state is not LocalDaemonState.STARTED:
            return
        if result.pid is None:
            raise LocalDaemonConfigurationError("a started local Eyvan control plane must have a PID")
        configured_endpoint = result.endpoint
        target = LocalDaemonTarget.from_endpoint(configured_endpoint)
        if target is None:
            raise LocalDaemonConfigurationError(
                "the native GCP runtime can stop only a literal HTTP loopback control plane"
            )
        profile_identifier = runtime_profile_identifier(runtime_profile_path.expanduser().resolve())
        assert profile_identifier is not None
        record_path = self.paths.record_path(target)
        log_path = self.paths.log_path(target)
        with owner_only_startup_lock(self.paths.lock_path(target)):
            record = read_owner_only_record(record_path)
            if record is None or record.endpoint != configured_endpoint or record.pid != result.pid:
                raise LocalDaemonConfigurationError(
                    "the newly started local Eyvan control plane is no longer identified by its owner-only PID record"
                )
            if record.runtime_profile_path != profile_identifier:
                raise LocalDaemonConfigurationError(
                    "the newly started local Eyvan control plane is no longer using the selected native GCP "
                    "runtime profile"
                )
            if self.process_liveness(record.pid):
                self._stop_recorded_daemon(
                    record,
                    target=target,
                    log_path=log_path,
                    expected_runtime_profile_path=profile_identifier,
                )
            clear_owner_only_record(record_path, pid=record.pid)

    def _stop_recorded_daemon(
        self,
        record: LocalDaemonRecord,
        *,
        target: LocalDaemonTarget,
        log_path: Path,
        expected_runtime_profile_path: str,
    ) -> None:
        """Terminate a PID only after its command proves it is the recorded Eyvan daemon."""

        if not self.process_liveness(record.pid):
            return
        require_local_daemon = record.runtime_profile_path is not None
        expected_command_profile = expected_runtime_profile_path if require_local_daemon else None
        if not _matches_local_daemon_process(
            self.process_argument_reader(record.pid),
            target,
            runtime_profile_path=expected_command_profile,
            require_local_daemon=require_local_daemon,
        ):
            raise LocalDaemonConfigurationError(
                f"the recorded PID {record.pid} for {target.endpoint.url} is not the expected Eyvan control-plane "
                "process; refusing to stop it"
            )
        try:
            outcome = stop_verified_process(
                record.pid,
                process_liveness=self.process_liveness,
                process_verifier=lambda pid: _matches_local_daemon_process(
                    self.process_argument_reader(pid),
                    target,
                    runtime_profile_path=expected_command_profile,
                    require_local_daemon=require_local_daemon,
                ),
                process_terminator=self.process_terminator,
                process_killer=self.process_killer,
                graceful_timeout_s=min(self.startup_timeout_s, 10.0),
                forced_timeout_s=min(self.startup_timeout_s, 5.0),
                poll_interval_s=self.poll_interval_s,
                clock=self.clock,
                sleeper=self.sleeper,
            )
        except ProcessLookupError:
            return
        except PermissionError as exc:
            raise LocalDaemonConfigurationError(
                f"could not stop the recorded Eyvan control-plane PID {record.pid}: permission denied"
            ) from exc
        except OSError as exc:
            raise LocalDaemonConfigurationError(
                f"could not stop the recorded Eyvan control-plane PID {record.pid}: {exc}"
            ) from exc
        if outcome is VerifiedProcessStopOutcome.TIMED_OUT:
            raise LocalDaemonConfigurationError(
                f"the recorded Eyvan control-plane PID {record.pid} did not exit after forced restart. See {log_path}."
            )

    def _existing_result(
        self,
        endpoint: ControlPlaneEndpoint,
        record_path: Path,
        log_path: Path,
        runtime_profile_path: str | None,
    ) -> LocalDaemonResult | None:
        """Return a reusable result only after a live typed health response."""

        if not self.health_probe(endpoint):
            return None
        record = read_owner_only_record(record_path)
        pid = None
        if record is None or record.endpoint != endpoint or not process_is_alive(record.pid):
            if runtime_profile_path is not None:
                # Say which of the three causes it is. They need different
                # remedies and only one of them is about the profile, but all
                # three used to report "not owned by the selected native GCP
                # runtime profile" -- so a stale record naming a dead pid sent
                # the operator looking for a profile problem that did not exist,
                # and the way out was hand-editing the record's JSON.
                if record is None:
                    detail = f"no owner-only PID record exists at {record_path}"
                    remedy = "it was started outside Eyvan, or its record was removed while it kept running"
                elif record.endpoint != endpoint:
                    detail = f"the record at {record_path} names endpoint {record.endpoint.url}"
                    remedy = "that record belongs to a different control plane"
                elif record.runtime_profile_path == runtime_profile_path:
                    # A dead PID beside a healthy listener is a stale *record*,
                    # not a foreign process. The health probe above already got
                    # a typed Eyvan response from this endpoint and the record
                    # names this endpoint and this runtime profile, which is
                    # more evidence than the live-PID branch below ever checks
                    # -- that one never confirms the recorded PID is the process
                    # holding the port either.
                    #
                    # Refusing here helped nobody. The plane is Eyvan's, it is
                    # serving, and the only way past the refusal was editing the
                    # record's JSON by hand -- which is what the operator did
                    # three times in one session before this was fixed. The
                    # record goes stale on its own: a later start that fails
                    # overwrites a working plane's record with the PID it just
                    # launched, and then that PID dies.
                    #
                    # Adopted without rewriting. Guessing which PID holds a
                    # socket is not portable, and the serving process rewrites
                    # its own record at startup, so the stale value heals on the
                    # next restart rather than being invented here.
                    return LocalDaemonResult(
                        endpoint=endpoint, state=LocalDaemonState.REUSED, pid=None, log_path=log_path
                    )
                else:
                    detail = (
                        f"the record at {record_path} names pid {record.pid}, which is not running, and "
                        f"runtime profile {record.runtime_profile_path}"
                    )
                    remedy = (
                        "that record belongs to a control plane started with a different runtime profile; select "
                        "that profile, or stop that listener"
                    )
                raise LocalDaemonConfigurationError(
                    f"a local Eyvan control plane is already listening at {endpoint.url}, but this supervisor "
                    f"cannot prove it owns it: {detail}. {remedy.capitalize()}."
                )
        else:
            if record.runtime_profile_path != runtime_profile_path:
                raise LocalDaemonConfigurationError(
                    f"the local Eyvan control plane at {endpoint.url} is running with a different runtime profile"
                )
            pid = record.pid
        return LocalDaemonResult(endpoint=endpoint, state=LocalDaemonState.REUSED, pid=pid, log_path=log_path)

    def _wait_for_ready(
        self,
        process: LocalDaemonProcess,
        endpoint: ControlPlaneEndpoint,
        record_path: Path,
        log_path: Path,
    ) -> LocalDaemonResult:
        """Wait until the owned process serves Eyvan's health response or exits."""

        deadline = self.clock() + self.startup_timeout_s
        while True:
            if self.health_probe(endpoint):
                return LocalDaemonResult(
                    endpoint=endpoint,
                    state=LocalDaemonState.STARTED,
                    pid=process.pid,
                    log_path=log_path,
                )
            exit_code = process.poll()
            if exit_code is not None:
                clear_owner_only_record(record_path, pid=process.pid)
                raise LocalDaemonStartupError(endpoint, log_path, f"server process exited with status {exit_code}")
            remaining = deadline - self.clock()
            if remaining <= 0:
                # The process is alive and simply slower than the budget, which
                # on a real fleet is the ordinary case: startup reconciles every
                # cell, and each one that is provisioning or draining costs a
                # provider round trip. Killing it here destroyed a reconcile in
                # progress and left the operator with no control plane at all --
                # observed with TPU capacity ACTIVE and nothing left to release
                # it. The record stays written, so the next `ensure` adopts this
                # process once it answers rather than starting a second one.
                raise LocalDaemonStartupError(
                    endpoint,
                    log_path,
                    f"server did not answer within {self.startup_timeout_s:.0f}s and is still starting; "
                    "it was left running -- re-run this command to adopt it once it is healthy",
                )
            self.sleeper(min(self.poll_interval_s, remaining))


def stop_owned_process(process: LocalDaemonProcess) -> None:
    """Stop only the exact process handle created by a local Eyvan supervisor."""

    if process.poll() is not None:
        return
    try:
        process.terminate()
        process.wait(timeout=2.0)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
            process.wait(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired):
            return


def ensure_loopback_control_plane(
    endpoint: str | ControlPlaneEndpoint,
    *,
    runtime_profile_path: Path | None = None,
) -> LocalDaemonResult:
    """Ensure a local daemon only when the configured endpoint is literal HTTP loopback."""

    return LocalDaemonSupervisor().ensure(endpoint, runtime_profile_path=runtime_profile_path)


__all__ = (
    "DEFAULT_LOCAL_CELL_AGENT_URL",
    "DEFAULT_LOCAL_DAEMON_DIRECTORY",
    "DEFAULT_LOCAL_MANAGED_ARTIFACTS_PATH",
    "HealthProbe",
    "HttpLocalDaemonHealthProbe",
    "LocalDaemonCommand",
    "LocalDaemonConfigurationError",
    "LocalDaemonError",
    "LocalDaemonLauncher",
    "LocalDaemonPaths",
    "LocalDaemonProcess",
    "LocalDaemonProfile",
    "LocalDaemonRecord",
    "LocalDaemonResult",
    "LocalDaemonStartupError",
    "LocalDaemonState",
    "LocalDaemonStateError",
    "LocalDaemonSupervisor",
    "LocalDaemonTarget",
    "ProcessKiller",
    "SubprocessLocalDaemonLauncher",
    "VerifiedProcessStopOutcome",
    "clear_owner_only_record",
    "ensure_loopback_control_plane",
    "force_kill_process",
    "open_owner_only_log",
    "owner_only_startup_lock",
    "prepare_owner_only_directory",
    "process_is_alive",
    "read_owner_only_record",
    "record_publication_lock",
    "require_owner_only_record_file",
    "require_owner_only_regular_file",
    "runtime_profile_identifier",
    "serving_runtime_profile_path",
    "serving_source_digest",
    "stop_owned_process",
    "stop_verified_process",
    "write_owner_only_record",
)
