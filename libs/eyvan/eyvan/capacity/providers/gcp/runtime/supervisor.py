# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Private local supervision for Eyvan's native GCP API-and-agent runtime."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
import typing as tp
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from pydantic import ValidationError

from eyvan._internal.build_identity import build_identity
from eyvan._internal.compat import StrEnum
from eyvan.capacity.providers.gcp.runtime.profile import (
    DEFAULT_GCP_NATIVE_RUNTIME_PROFILE_PATH,
    GcpNativeRuntimeProfile,
    GcpNativeRuntimeProfileError,
    GcpNativeRuntimeProfileStore,
)
from eyvan.sdk.client.endpoint import ControlPlaneEndpoint
from eyvan.server.local.daemon import (
    DEFAULT_LOCAL_CELL_AGENT_URL,
    Clock,
    LocalDaemonConfigurationError,
    LocalDaemonError,
    LocalDaemonPaths,
    LocalDaemonProcess,
    LocalDaemonRecord,
    LocalDaemonResult,
    LocalDaemonState,
    LocalDaemonSupervisor,
    LocalDaemonTarget,
    ProcessKiller,
    Sleeper,
    VerifiedProcessStopOutcome,
    clear_owner_only_record,
    force_kill_process,
    open_owner_only_log,
    owner_only_startup_lock,
    process_is_alive,
    read_owner_only_record,
    runtime_profile_identifier,
    serving_runtime_profile_path,
    serving_source_digest,
    stop_owned_process,
    stop_verified_process,
    write_owner_only_record,
)
from eyvan.worker.agent.client import CellAgentClient, CellAgentUnavailableError
from eyvan.worker.agent.models import AgentBuild, AgentCapabilities

if tp.TYPE_CHECKING:
    from eyvan.capacity.fleet_config import EyvanFleetConfig

# Compatibility export for callers that imported the initial native-runtime name.
DEFAULT_LOCAL_GCP_AGENT_URL = DEFAULT_LOCAL_CELL_AGENT_URL


class GcpRuntimeError(LocalDaemonError):
    """Base error for a local native GCP runtime that cannot safely be used."""


class GcpRuntimeProfileNotPreparedError(GcpRuntimeError):
    """The owner has not staged the native worker package required by GCP workers."""


class LocalAgentConfigurationError(GcpRuntimeError):
    """A live local cell agent does not match the selected native GCP runtime profile."""


class LocalAgentStartupError(GcpRuntimeError):
    """Eyvan could not start a usable authenticated local cell agent."""

    def __init__(self, endpoint: ControlPlaneEndpoint, log_path: Path, reason: str) -> None:
        self.endpoint = endpoint
        self.log_path = log_path
        self.reason = reason
        super().__init__(f"Eyvan could not start its local GCP cell agent at {endpoint.url}: {reason}. See {log_path}.")


class LocalAgentState(StrEnum):
    """The result of one authenticated local cell-agent readiness decision."""

    REUSED = "reused"
    STARTED = "started"


@dataclass(frozen=True, slots=True)
class LocalAgentCommand:
    """One exact subprocess invocation for the private local GCP cell agent."""

    target: LocalDaemonTarget
    log_path: Path
    runtime_profile_path: Path

    @property
    def argv(self) -> tuple[str, ...]:
        """Return an invocation carrying a profile path but never its secrets."""

        return (
            sys.executable,
            "-m",
            "eyvan.worker.agent.cli",
            "--host",
            self.target.host,
            "--port",
            str(self.target.port),
            "--runtime-profile",
            str(self.runtime_profile_path),
            "--log-level",
            "info",
        )


class LocalAgentLauncher(tp.Protocol):
    """Starts one locally supervised cell-agent process."""

    def start(self, command: LocalAgentCommand) -> LocalDaemonProcess:
        """Launch a cell agent without waiting for its authenticated capability response."""

        ...


class LocalControlPlaneSupervisor(tp.Protocol):
    """Profile-aware local API supervision used by the native GCP runtime."""

    def ensure(
        self,
        endpoint: str | ControlPlaneEndpoint,
        *,
        runtime_profile_path: Path | None = None,
    ) -> LocalDaemonResult:
        """Start or reuse the private control plane for one selected profile."""

        ...

    def stop_started(self, result: LocalDaemonResult, *, runtime_profile_path: Path) -> None:
        """Stop only a daemon started by this supervision attempt."""

        ...


AgentCapabilitiesProbe: tp.TypeAlias = tp.Callable[[ControlPlaneEndpoint, str], AgentCapabilities | None]
ProcessArgumentReader: tp.TypeAlias = tp.Callable[[int], tuple[str, ...] | None]
ProcessLiveness: tp.TypeAlias = tp.Callable[[int], bool]
ProcessTerminator: tp.TypeAlias = tp.Callable[[int], None]


@dataclass(frozen=True, slots=True)
class LocalAgentResult:
    """A ready local cell-agent process and its authenticated capabilities."""

    endpoint: ControlPlaneEndpoint
    state: LocalAgentState
    pid: int
    log_path: Path
    capabilities: AgentCapabilities


class NativeGcpCellAgentSupervisor(tp.Protocol):
    """Private cell-agent supervision required by the native GCP runtime."""

    def ensure(
        self,
        profile_path: Path,
        profile: GcpNativeRuntimeProfile,
        *,
        reference_digest: str | None = None,
    ) -> LocalAgentResult:
        """Start or reuse the authenticated agent for one selected profile."""

        ...


class NativeGcpRuntimePreparer(tp.Protocol):
    """Prepare the local private runtime needed before applying native GCP pools."""

    def ensure(self, endpoint: str | ControlPlaneEndpoint) -> GcpRuntimeResult:
        """Start or reuse the local API-and-agent pair for native GCP execution."""

        ...


@dataclass(frozen=True, slots=True)
class SubprocessLocalAgentLauncher:
    """Start the current Eyvan installation as a detached local cell agent."""

    def start(self, command: LocalAgentCommand) -> LocalDaemonProcess:
        """Start the process with owner-only log output and no inherited terminal input."""

        with open_owner_only_log(command.log_path) as log:
            return subprocess.Popen(
                command.argv,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )


@dataclass(frozen=True, slots=True)
class HttpCellAgentCapabilitiesProbe:
    """Probe the authenticated capability contract without proxying loopback traffic."""

    timeout_s: float = 0.25

    def __call__(self, endpoint: ControlPlaneEndpoint, secret: str) -> AgentCapabilities | None:
        """Return capabilities only when the selected agent accepts the selected shared secret."""

        try:
            with httpx.Client(
                base_url=endpoint.url,
                headers={"Authorization": f"Bearer {secret}"},
                timeout=self.timeout_s,
                trust_env=False,
            ) as transport:
                return CellAgentClient(endpoint.url, client=transport).capabilities()
        except (CellAgentUnavailableError, httpx.HTTPError, ValidationError, ValueError):
            # "The agent did not answer" is the expected case here -- this probe
            # runs *while* an agent is being replaced. It became a typed error
            # rather than a bare httpx one when the client grew a single
            # transport seam, and omitting it aborted `eyvan runtime restart`
            # with the very condition the probe exists to tolerate.
            return None


def _default_clock() -> Clock:
    """Return the monotonic clock used by the default local-agent supervisor."""

    return time.monotonic


@dataclass(frozen=True, slots=True)
class HttpCellAgentBuildProbe:
    """Read the build identity of a running agent, tolerating its absence.

    An agent older than the ``/api/v1/build`` route answers ``None`` here, and
    an unknowable identity must never trigger a replacement: the guard exists
    to stop verified staleness, not to churn agents it cannot measure.
    """

    timeout_s: float = 0.5

    def __call__(self, endpoint: ControlPlaneEndpoint, secret: str) -> AgentBuild | None:
        """Return the running agent's build identity, or ``None`` when it cannot be read."""

        try:
            with httpx.Client(
                base_url=endpoint.url,
                headers={"Authorization": f"Bearer {secret}"},
                timeout=self.timeout_s,
                trust_env=False,
            ) as transport:
                return CellAgentClient(endpoint.url, client=transport).build()
        except (CellAgentUnavailableError, httpx.HTTPError, ValidationError, ValueError):
            return None


def _default_sleeper() -> Sleeper:
    """Return the blocking sleep function used by the default local-agent supervisor."""

    return time.sleep


def _read_local_agent_process_arguments(pid: int) -> tuple[str, ...] | None:
    """Read one process command without invoking a shell or exposing it to users."""

    try:
        completed = subprocess.run(
            ("ps", "-ww", "-p", str(pid), "-o", "command="),
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    command = completed.stdout.strip()
    if not command:
        return None
    try:
        return tuple(shlex.split(command))
    except ValueError:
        return None


def _terminate_local_agent_process(pid: int) -> None:
    """Request graceful shutdown after the caller verified the process identity."""

    os.kill(pid, signal.SIGTERM)


def _single_option_value(arguments: tuple[str, ...], option: str) -> str | None:
    """Return an unambiguous command-line option value, if it is present exactly once."""

    values = tuple(arguments[index + 1] for index, value in enumerate(arguments[:-1]) if value == option)
    if len(values) != 1:
        return None
    return values[0]


def _matches_local_agent_process(
    arguments: tuple[str, ...] | None,
    target: LocalDaemonTarget,
    *,
    runtime_profile_path: str,
) -> bool:
    """Return whether a recorded PID still runs the exact expected Eyvan cell agent."""

    if arguments is None:
        return False
    has_module_invocation = any(
        arguments[index : index + 2] == ("-m", "eyvan.worker.agent.cli") for index in range(len(arguments) - 1)
    )
    return (
        has_module_invocation
        and _single_option_value(arguments, "--host") == target.host
        and _single_option_value(arguments, "--port") == str(target.port)
        and _single_option_value(arguments, "--runtime-profile") == runtime_profile_path
    )


@dataclass(frozen=True, slots=True)
class LocalAgentSupervisor:
    """Start or reuse one authenticated local GCP cell agent for one private profile."""

    paths: LocalDaemonPaths = field(default_factory=LocalDaemonPaths)
    launcher: LocalAgentLauncher = field(default_factory=SubprocessLocalAgentLauncher)
    capability_probe: AgentCapabilitiesProbe = field(default_factory=HttpCellAgentCapabilitiesProbe)
    build_probe: tp.Callable[[ControlPlaneEndpoint, str], AgentBuild | None] = field(
        default_factory=HttpCellAgentBuildProbe
    )
    endpoint: ControlPlaneEndpoint = field(
        default_factory=lambda: ControlPlaneEndpoint.parse(DEFAULT_LOCAL_GCP_AGENT_URL)
    )
    startup_timeout_s: float = 20.0
    poll_interval_s: float = 0.1
    clock: Clock = field(default_factory=_default_clock)
    sleeper: Sleeper = field(default_factory=_default_sleeper)
    process_argument_reader: ProcessArgumentReader = _read_local_agent_process_arguments
    process_liveness: ProcessLiveness = process_is_alive
    process_terminator: ProcessTerminator = _terminate_local_agent_process
    process_killer: ProcessKiller = force_kill_process

    def __post_init__(self) -> None:
        if self.startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be positive")
        if self.poll_interval_s <= 0:
            raise ValueError("poll_interval_s must be positive")
        if LocalDaemonTarget.from_endpoint(self.endpoint) is None:
            raise ValueError("local GCP cell agent endpoint must be literal HTTP loopback")

    def ensure(
        self,
        profile_path: Path,
        profile: GcpNativeRuntimeProfile,
        *,
        reference_digest: str | None = None,
    ) -> LocalAgentResult:
        """Start the profile's local agent or reuse its authenticated capability contract.

        Reuse requires the running agent to be *the runtime's build*: a
        compatible agent compiled from other code is replaced (when this
        process owns it) rather than silently reused. Checking only profile and
        secret would keep reusing a long-lived daemon after its code changes.

        ``reference_digest`` is that build: the one the serving control plane
        was launched from, which is also the build it stages for workers.
        Without it the caller's own build is the reference, as before.
        """

        expanded_profile_path, profile_id = self._prepared_profile(profile_path, profile)
        target = self._target()
        record_path = self.paths.agent_record_path(target)
        log_path = self.paths.agent_log_path(target)
        with owner_only_startup_lock(self.paths.agent_lock_path(target)):
            # Reuse checks can themselves stop a stale build or clear a dead
            # record. Keep them under the same lock as replacement startup.
            self._replace_agent_running_stale_code(
                profile, profile_id, record_path, log_path, target, reference_digest=reference_digest
            )
            existing = self._existing_result(
                profile, profile_id, record_path, log_path, reference_digest=reference_digest
            )
            if existing is not None:
                return existing

            # Recheck after a stale agent's stop or a wait for an existing
            # agent before publishing a fresh replacement's ownership.
            self._replace_agent_running_stale_code(
                profile, profile_id, record_path, log_path, target, reference_digest=reference_digest
            )
            existing = self._existing_result(
                profile, profile_id, record_path, log_path, reference_digest=reference_digest
            )
            if existing is not None:
                return existing
            return self._start_locked(
                target=target,
                expanded_profile_path=expanded_profile_path,
                profile=profile,
                profile_id=profile_id,
                record_path=record_path,
                log_path=log_path,
                reference_digest=reference_digest,
            )

    def restart(self, profile_path: Path, profile: GcpNativeRuntimeProfile) -> LocalAgentResult:
        """Replace only a verified profile-owned cell agent with a freshly started one.

        This is intentionally separate from :meth:`ensure`: regular commands
        must retain an already healthy agent. A restart validates the
        owner-only PID record, its selected runtime profile, and the recorded
        process command before requesting termination.
        """

        expanded_profile_path, profile_id = self._prepared_profile(profile_path, profile)
        target = self._target()
        record_path = self.paths.agent_record_path(target)
        log_path = self.paths.agent_log_path(target)
        with owner_only_startup_lock(self.paths.agent_lock_path(target)):
            record = read_owner_only_record(record_path)
            if record is None:
                capabilities = self.capability_probe(self.endpoint, profile.agent_shared_secret)
                if capabilities is not None:
                    self._validate_capabilities(capabilities)
                    raise LocalAgentConfigurationError(
                        f"a cell agent is already listening at {self.endpoint.url}, but it is not owned by the "
                        "selected native GCP runtime profile; refusing to stop it"
                    )
            elif record.endpoint != self.endpoint:
                raise LocalAgentConfigurationError(
                    f"the recorded Eyvan cell-agent endpoint does not match {self.endpoint.url}; refusing to stop it"
                )
            elif self.process_liveness(record.pid):
                if record.runtime_profile_path != profile_id:
                    raise LocalAgentConfigurationError(
                        f"the local Eyvan cell agent at {self.endpoint.url} belongs to a different runtime profile; "
                        "refusing to stop it"
                    )
                self._stop_recorded_agent(
                    record,
                    target=target,
                    log_path=log_path,
                    expected_runtime_profile_path=profile_id,
                )
                clear_owner_only_record(record_path, pid=record.pid)
            else:
                capabilities = self.capability_probe(self.endpoint, profile.agent_shared_secret)
                if capabilities is not None:
                    self._validate_capabilities(capabilities)
                    raise LocalAgentConfigurationError(
                        f"a cell agent is already listening at {self.endpoint.url}, but its owner-only PID record "
                        "is stale; refusing to stop it"
                    )
                clear_owner_only_record(record_path, pid=record.pid)
            return self._start_locked(
                target=target,
                expanded_profile_path=expanded_profile_path,
                profile=profile,
                profile_id=profile_id,
                record_path=record_path,
                log_path=log_path,
            )

    def stop(self, *, runtime_profile_path: Path | None = None) -> int | None:
        """Stop the cell agent this profile recorded for this supervisor's endpoint.

        The counterpart to :meth:`ensure` for callers that own the profile
        directory: a developer clearing a local runtime, or a test harness
        whose isolated home recorded the agent. The same ownership proof
        :meth:`restart` demands -- an owner-only PID record naming this
        endpoint, whose recorded PID still runs the expected Eyvan cell agent
        under its recorded runtime profile -- precedes any signal, so a
        recycled PID naming an unrelated process is refused rather than
        killed. A missing record stops nothing: an agent this supervisor
        cannot prove it owns is not this supervisor's to stop.

        Args:
            runtime_profile_path: Optional runtime profile that the recorded
                agent must belong to in order to be stopped. When supplied and
                the recorded agent belongs to a different profile, the agent
                and its record are left untouched and ``None`` is returned.

        Returns:
            The PID of the live cell agent that was stopped, or ``None`` when
            no matching live cell agent was running.
        """

        target = self._target()
        record_path = self.paths.agent_record_path(target)
        log_path = self.paths.agent_log_path(target)
        expected_profile_id = runtime_profile_identifier(runtime_profile_path)
        with owner_only_startup_lock(self.paths.agent_lock_path(target)):
            record = read_owner_only_record(record_path)
            if record is None:
                return None
            if record.endpoint != self.endpoint:
                raise LocalAgentConfigurationError(
                    f"the recorded Eyvan cell-agent endpoint does not match {self.endpoint.url}; refusing to stop it"
                )
            if expected_profile_id is not None and record.runtime_profile_path != expected_profile_id:
                return None
            stopped_pid: int | None = None
            if self.process_liveness(record.pid):
                if record.runtime_profile_path is None:
                    raise LocalAgentConfigurationError(
                        f"the recorded Eyvan cell agent at {self.endpoint.url} names no runtime profile; "
                        "refusing to stop it"
                    )
                self._stop_recorded_agent(
                    record,
                    target=target,
                    log_path=log_path,
                    expected_runtime_profile_path=record.runtime_profile_path,
                )
                stopped_pid = record.pid
            clear_owner_only_record(record_path, pid=record.pid)
            return stopped_pid

    @staticmethod
    def _prepared_profile(
        profile_path: Path,
        profile: GcpNativeRuntimeProfile,
    ) -> tuple[Path, str]:
        """Validate the staged-worker requirement and derive the immutable profile identity."""

        if profile.native_worker_wheel_uri is None:
            raise GcpRuntimeProfileNotPreparedError(
                "the native GCP runtime has no staged Eyvan worker package; run `eyvan runtime init`"
            )
        expanded_profile_path = profile_path.expanduser().resolve()
        return expanded_profile_path, str(expanded_profile_path)

    def _start_locked(
        self,
        *,
        target: LocalDaemonTarget,
        expanded_profile_path: Path,
        profile: GcpNativeRuntimeProfile,
        profile_id: str,
        record_path: Path,
        log_path: Path,
        reference_digest: str | None = None,
    ) -> LocalAgentResult:
        """Launch and authenticate a fresh agent while holding its owner-only startup lock."""

        caller_digest = build_identity().source_digest
        if reference_digest is not None and caller_digest is not None and reference_digest != caller_digest:
            raise LocalAgentConfigurationError(
                "the serving control plane and this process run different builds; "
                "run 'eyvan runtime deploy --yes' to replace them together"
            )
        command = LocalAgentCommand(
            target=target,
            log_path=log_path,
            runtime_profile_path=expanded_profile_path,
        )
        try:
            process = self.launcher.start(command)
        except OSError as exc:
            raise LocalAgentStartupError(self.endpoint, log_path, str(exc)) from exc
        try:
            write_owner_only_record(
                record_path,
                LocalDaemonRecord(
                    endpoint=self.endpoint,
                    pid=process.pid,
                    runtime_profile_path=profile_id,
                ),
            )
            return self._wait_for_ready(process, profile, record_path, log_path, reference_digest=reference_digest)
        except BaseException as startup_error:
            # Only this attempt's exact child belongs to failure cleanup. An
            # existing agent's wait is deliberately outside this guard.
            try:
                stop_owned_process(process)
                if process.poll() is None:
                    startup_error.add_note(
                        f"new cell-agent PID {process.pid} did not exit after startup cleanup; "
                        f"any ownership record was retained. See {log_path}."
                    )
                else:
                    clear_owner_only_record(record_path, pid=process.pid)
            except BaseException as cleanup_error:
                startup_error.add_note(
                    f"startup cleanup for new cell-agent PID {process.pid} failed "
                    f"({type(cleanup_error).__name__}); any ownership record was retained. See {log_path}."
                )
            raise

    def _agent_runs_different_code(
        self,
        profile: GcpNativeRuntimeProfile,
        *,
        reference_digest: str | None = None,
    ) -> bool:
        """Return True only when the live agent provably runs a different build.

        An unknowable identity (route absent, digest uncomputable on either
        side) never counts as different, matching the replacement policy below.
        """

        return self._agent_digest_skew(profile, reference_digest=reference_digest) is not None

    def _agent_digest_skew(
        self,
        profile: GcpNativeRuntimeProfile,
        *,
        reference_digest: str | None = None,
    ) -> tuple[str, str] | None:
        """Return ``(running, reference)`` digests when they provably differ.

        The reference is ``reference_digest`` when the caller knows the
        runtime's build, else this process's own. An unknowable identity (route
        absent, digest uncomputable on either side) returns ``None``: it never
        counts as different.
        """

        probe = self.build_probe
        remote = probe(self.endpoint, profile.agent_shared_secret) if probe is not None else None
        if remote is None or remote.source_digest is None:
            return None
        reference = reference_digest or build_identity().source_digest
        if reference is None or reference == remote.source_digest:
            return None
        return remote.source_digest, reference

    def _replace_agent_running_stale_code(
        self,
        profile: GcpNativeRuntimeProfile,
        profile_id: str,
        record_path: Path,
        log_path: Path,
        target: LocalDaemonTarget,
        *,
        reference_digest: str | None = None,
    ) -> None:
        """Stop a profile-owned agent whose build identity differs from the runtime's.

        An agent this process does not own is refused loudly instead of stopped,
        pointing the operator at the restart command that validates ownership
        the same way.
        """

        skew = self._agent_digest_skew(profile, reference_digest=reference_digest)
        if skew is None:
            return
        remote_digest, current_digest = skew
        record = read_owner_only_record(record_path)
        if (
            record is None
            or record.endpoint != self.endpoint
            or not self.process_liveness(record.pid)
            or record.runtime_profile_path != profile_id
        ):
            raise LocalAgentConfigurationError(
                f"a cell agent built from different code is listening at {self.endpoint.url} "
                f"(running digest {remote_digest[:12]}, this process {current_digest[:12]}); "
                "it is not owned by the selected runtime profile, so it was not replaced automatically -- "
                "run 'eyvan runtime restart --yes' to replace it explicitly"
            )
        self._stop_recorded_agent(
            record,
            target=target,
            log_path=log_path,
            expected_runtime_profile_path=profile_id,
        )
        clear_owner_only_record(record_path, pid=record.pid)

    def _existing_result(
        self,
        profile: GcpNativeRuntimeProfile,
        profile_id: str,
        record_path: Path,
        log_path: Path,
        *,
        reference_digest: str | None = None,
    ) -> LocalAgentResult | None:
        """Return a compatible authenticated agent or reject a conflicting live process."""

        capabilities = self.capability_probe(self.endpoint, profile.agent_shared_secret)
        record = read_owner_only_record(record_path)
        if capabilities is None:
            if record is not None and record.endpoint == self.endpoint and self.process_liveness(record.pid):
                if record.runtime_profile_path != profile_id:
                    raise LocalAgentConfigurationError(
                        f"the local Eyvan cell agent at {self.endpoint.url} is running with a different runtime profile"
                    )
                return self._wait_for_existing_agent(
                    profile, profile_id, record, record_path, log_path, reference_digest=reference_digest
                )
            return None
        self._validate_capabilities(capabilities)
        if record is None or record.endpoint != self.endpoint or not self.process_liveness(record.pid):
            raise LocalAgentConfigurationError(
                f"a cell agent is already listening at {self.endpoint.url}, but it is not owned by the selected "
                "native GCP runtime profile"
            )
        if record.runtime_profile_path != profile_id:
            raise LocalAgentConfigurationError(
                f"the local Eyvan cell agent at {self.endpoint.url} is running with a different runtime profile"
            )
        if self._agent_runs_different_code(profile, reference_digest=reference_digest):
            # A compatible capability response must not bypass build identity
            # verification. Replacement stays under the caller's endpoint lock.
            self._replace_agent_running_stale_code(
                profile, profile_id, record_path, log_path, self._target(), reference_digest=reference_digest
            )
            return None
        return LocalAgentResult(
            endpoint=self.endpoint,
            state=LocalAgentState.REUSED,
            pid=record.pid,
            log_path=log_path,
            capabilities=capabilities,
        )

    def _wait_for_existing_agent(
        self,
        profile: GcpNativeRuntimeProfile,
        profile_id: str,
        record: LocalDaemonRecord,
        record_path: Path,
        log_path: Path,
        *,
        reference_digest: str | None = None,
    ) -> LocalAgentResult | None:
        """Wait for another supervisor's owned matching agent instead of racing a second bind.

        Returns ``None`` when the agent that became ready turns out to run a
        different build -- a skew-era relaunch, which is replaced rather than
        reused -- leaving the caller to start a fresh one.
        """

        deadline = self.clock() + self.startup_timeout_s
        while True:
            capabilities = self.capability_probe(self.endpoint, profile.agent_shared_secret)
            if capabilities is not None:
                self._validate_capabilities(capabilities)
                if self._agent_runs_different_code(profile, reference_digest=reference_digest):
                    self._replace_agent_running_stale_code(
                        profile, profile_id, record_path, log_path, self._target(), reference_digest=reference_digest
                    )
                    return None
                return LocalAgentResult(
                    endpoint=self.endpoint,
                    state=LocalAgentState.REUSED,
                    pid=record.pid,
                    log_path=log_path,
                    capabilities=capabilities,
                )
            if not self.process_liveness(record.pid):
                clear_owner_only_record(self.paths.agent_record_path(self._target()), pid=record.pid)
                raise LocalAgentStartupError(self.endpoint, log_path, "recorded agent process exited before readiness")
            remaining = deadline - self.clock()
            if remaining <= 0:
                raise LocalAgentStartupError(
                    self.endpoint,
                    log_path,
                    "recorded agent did not become ready before timeout",
                )
            self.sleeper(min(self.poll_interval_s, remaining))

    def _wait_for_ready(
        self,
        process: LocalDaemonProcess,
        profile: GcpNativeRuntimeProfile,
        record_path: Path,
        log_path: Path,
        *,
        reference_digest: str | None = None,
    ) -> LocalAgentResult:
        """Wait for authenticated capabilities and a non-conflicting build identity."""

        deadline = self.clock() + self.startup_timeout_s
        while True:
            capabilities = self.capability_probe(self.endpoint, profile.agent_shared_secret)
            if capabilities is not None:
                self._validate_capabilities(capabilities)
                if self._agent_runs_different_code(profile, reference_digest=reference_digest):
                    raise LocalAgentConfigurationError(
                        "the started cell agent runs a different build from the runtime; "
                        "run 'eyvan runtime deploy --yes' to replace them together"
                    )
                return LocalAgentResult(
                    endpoint=self.endpoint,
                    state=LocalAgentState.STARTED,
                    pid=process.pid,
                    log_path=log_path,
                    capabilities=capabilities,
                )
            exit_code = process.poll()
            if exit_code is not None:
                clear_owner_only_record(record_path, pid=process.pid)
                raise LocalAgentStartupError(self.endpoint, log_path, f"agent process exited with status {exit_code}")
            remaining = deadline - self.clock()
            if remaining <= 0:
                # _start_locked owns exact-child cleanup and only clears the
                # record once exit is verified, including on timeout.
                raise LocalAgentStartupError(self.endpoint, log_path, "agent did not become ready before timeout")
            self.sleeper(min(self.poll_interval_s, remaining))

    def _stop_recorded_agent(
        self,
        record: LocalDaemonRecord,
        *,
        target: LocalDaemonTarget,
        log_path: Path,
        expected_runtime_profile_path: str,
    ) -> None:
        """Terminate a PID only after its command proves it is the recorded Eyvan cell agent."""

        if not self.process_liveness(record.pid):
            return
        if not _matches_local_agent_process(
            self.process_argument_reader(record.pid),
            target,
            runtime_profile_path=expected_runtime_profile_path,
        ):
            raise LocalAgentConfigurationError(
                f"the recorded PID {record.pid} for {target.endpoint.url} is not the expected Eyvan cell-agent "
                "process; refusing to stop it"
            )
        try:
            outcome = stop_verified_process(
                record.pid,
                process_liveness=self.process_liveness,
                process_verifier=lambda pid: _matches_local_agent_process(
                    self.process_argument_reader(pid),
                    target,
                    runtime_profile_path=expected_runtime_profile_path,
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
            raise LocalAgentConfigurationError(
                f"could not stop the recorded Eyvan cell-agent PID {record.pid}: permission denied"
            ) from exc
        except OSError as exc:
            raise LocalAgentConfigurationError(
                f"could not stop the recorded Eyvan cell-agent PID {record.pid}: {exc}"
            ) from exc
        if outcome is VerifiedProcessStopOutcome.TIMED_OUT:
            raise LocalAgentConfigurationError(
                f"the recorded Eyvan cell-agent PID {record.pid} did not exit after forced restart. See {log_path}."
            )

    def _target(self) -> LocalDaemonTarget:
        """Return the validated local target used by every agent record path."""

        target = LocalDaemonTarget.from_endpoint(self.endpoint)
        assert target is not None
        return target

    @staticmethod
    def _validate_capabilities(capabilities: AgentCapabilities) -> None:
        """Reject a reachable agent that is not the native GCP executor requested by the profile."""

        if capabilities.provider != "gcp" or capabilities.executor != "eyvan-native":
            raise LocalAgentConfigurationError(
                "the local Eyvan cell agent must expose provider 'gcp' and executor 'eyvan-native' for native GCP runs"
            )


@dataclass(frozen=True, slots=True)
class GcpRuntimeResult:
    """A fully prepared local API-and-agent pair for one native GCP profile."""

    profile: GcpNativeRuntimeProfile
    control_plane: LocalDaemonResult
    agent: LocalAgentResult


@dataclass(frozen=True, slots=True)
class GcpNativeRuntimeSupervisor:
    """Own the private local control-plane and cell-agent processes for native GCP fleets."""

    profile_store: GcpNativeRuntimeProfileStore = field(default_factory=GcpNativeRuntimeProfileStore)
    control_plane: LocalControlPlaneSupervisor = field(default_factory=LocalDaemonSupervisor)
    agent: NativeGcpCellAgentSupervisor = field(default_factory=LocalAgentSupervisor)

    @classmethod
    def for_endpoint(cls, endpoint: str | ControlPlaneEndpoint) -> GcpNativeRuntimeSupervisor:
        """Supervise with the profile the plane at ``endpoint`` already serves, else the default.

        A plane started by ``eyvan runtime deploy --profile PATH`` records that
        path. Ensuring it with the default profile instead refused it as
        "running with a different runtime profile", so every command after the
        deploy -- ``eyvan up``, the SDK's ``init()`` -- failed against the plane
        the operator had just started.
        """

        return cls(profile_store=GcpNativeRuntimeProfileStore(runtime_profile_path_for(endpoint)))

    def ensure(self, endpoint: str | ControlPlaneEndpoint) -> GcpRuntimeResult:
        """Load one profile, then start/reuse its API and authenticated cell agent."""

        configured_endpoint = ControlPlaneEndpoint.parse(endpoint)
        if LocalDaemonTarget.from_endpoint(configured_endpoint) is None:
            raise GcpRuntimeError(
                "the native GCP runtime requires a literal HTTP loopback control-plane URL such as http://127.0.0.1:8080"
            )
        try:
            profile = self.profile_store.load()
        except FileNotFoundError as exc:
            raise GcpRuntimeProfileNotPreparedError(
                "the native GCP runtime is not initialized; run `eyvan runtime init`"
            ) from exc
        except GcpNativeRuntimeProfileError as exc:
            raise GcpRuntimeError(f"the private native GCP runtime profile is invalid: {exc}") from exc
        if profile.native_worker_wheel_uri is None:
            raise GcpRuntimeProfileNotPreparedError(
                "the native GCP runtime has no staged Eyvan worker package; run `eyvan runtime init`"
            )
        try:
            control_plane = self.control_plane.ensure(
                configured_endpoint,
                runtime_profile_path=self.profile_store.path,
            )
        except LocalDaemonConfigurationError as exc:
            raise GcpRuntimeError(
                f"{exc}. Restart the locally supervised Eyvan control plane, then retry the native GCP command"
            ) from exc
        try:
            serving = serving_source_digest(configured_endpoint)
            caller = build_identity().source_digest
            if serving is not None and caller is not None and serving != caller:
                # Replacing only the agent here split a runtime in two: agent on
                # the new build, plane and staged worker wheel on the old one,
                # and every new cell recycled as stale. The runtime moves as one.
                warnings.warn(
                    f"this Eyvan ({caller[:12]}) differs from the local runtime it is using "
                    f"({serving[:12]}); run 'eyvan runtime deploy --yes' to move the control plane, "
                    "cell agent and worker package to it together",
                    RuntimeWarning,
                    stacklevel=2,
                )
            agent = self.agent.ensure(self.profile_store.path, profile, reference_digest=serving)
        except BaseException as agent_error:
            if control_plane.state is LocalDaemonState.STARTED:
                try:
                    self.control_plane.stop_started(control_plane, runtime_profile_path=self.profile_store.path)
                except BaseException as cleanup_error:
                    if isinstance(agent_error, Exception) and isinstance(cleanup_error, LocalDaemonError):
                        raise GcpRuntimeError(
                            "Eyvan could not start its local GCP cell agent and could not stop the newly started "
                            f"control plane: {cleanup_error}"
                        ) from agent_error
                    # Startup cancellation owns the same rollback as an ordinary
                    # failure. Cleanup must never replace the original interrupt.
                    agent_error.add_note(
                        "Eyvan could not stop the newly started control plane during agent startup rollback: "
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            raise
        return GcpRuntimeResult(profile=profile, control_plane=control_plane, agent=agent)


def ensure_gcp_native_runtime(
    endpoint: str | ControlPlaneEndpoint,
    *,
    profile_path: Path | None = None,
) -> GcpRuntimeResult:
    """Ensure a private native GCP runtime for a literal loopback control plane.

    ``profile_path`` selects the profile the runtime must be serving; omitted,
    it is the profile an already-running plane at ``endpoint`` was deployed
    with, else the default ``~/.eyvan/gcp-native-runtime.json``.
    """

    if profile_path is None:
        return GcpNativeRuntimeSupervisor.for_endpoint(endpoint).ensure(endpoint)
    return GcpNativeRuntimeSupervisor(profile_store=GcpNativeRuntimeProfileStore(profile_path)).ensure(endpoint)


def runtime_profile_path_for(endpoint: str | ControlPlaneEndpoint | None) -> Path:
    """Return the profile the loopback plane at ``endpoint`` serves, else the default profile path."""

    if endpoint is not None:
        try:
            serving = serving_runtime_profile_path(endpoint)
        except ValueError:
            serving = None
        if serving is not None:
            return serving
    return DEFAULT_GCP_NATIVE_RUNTIME_PROFILE_PATH


def has_gcp_native_runtime_profile(endpoint: str | ControlPlaneEndpoint | None = None) -> bool:
    """Return whether a valid private native GCP runtime profile applies to ``endpoint``.

    That is the profile a running plane at ``endpoint`` serves, else the
    current user's default profile.
    """

    try:
        GcpNativeRuntimeProfileStore(runtime_profile_path_for(endpoint)).load()
    except FileNotFoundError:
        return False
    except GcpNativeRuntimeProfileError as exc:
        raise GcpRuntimeError(f"the private native GCP runtime profile is invalid: {exc}") from exc
    return True


def requires_native_gcp_runtime(config: EyvanFleetConfig) -> bool:
    """Return whether an Eyvan fleet needs the private local native GCP runtime pair."""

    if config.provider.type != "gcp":
        return False
    from eyvan.capacity.drivers.base import canonical_execution_backend

    for node_type in config.pools.values():
        if node_type.agent_endpoint is not None or config.agent_endpoint is not None:
            continue
        executor = canonical_execution_backend(node_type.executor or config.executor)
        capacity_backend = (node_type.capacity_backend or node_type.default_capacity_backend).strip().lower()
        capacity_backend = capacity_backend.replace("_", "-")
        if executor == "eyvan-native" and capacity_backend in {
            "gcp",
            "gcp-compute",
            "gcp-tpu",
            "gcp-attach",
            "compute",
            "tpu",
        }:
            return True
    return False


__all__ = (
    "DEFAULT_LOCAL_GCP_AGENT_URL",
    "AgentCapabilitiesProbe",
    "GcpNativeRuntimeSupervisor",
    "GcpRuntimeError",
    "GcpRuntimeProfileNotPreparedError",
    "GcpRuntimeResult",
    "HttpCellAgentBuildProbe",
    "HttpCellAgentCapabilitiesProbe",
    "LocalAgentCommand",
    "LocalAgentConfigurationError",
    "LocalAgentLauncher",
    "LocalAgentResult",
    "LocalAgentStartupError",
    "LocalAgentState",
    "LocalAgentSupervisor",
    "LocalControlPlaneSupervisor",
    "NativeGcpCellAgentSupervisor",
    "NativeGcpRuntimePreparer",
    "SubprocessLocalAgentLauncher",
    "ensure_gcp_native_runtime",
    "has_gcp_native_runtime_profile",
    "requires_native_gcp_runtime",
    "runtime_profile_path_for",
)
