# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Durable OCI workload execution for Eyvan-native workers.

The Docker daemon owns a container outside the transient service cgroup, so a
plain ``systemctl stop`` is insufficient: it can kill the foreground Docker
client while leaving the workload alive.  This executor keeps systemd's
fencing and durable records, but records a deterministic container identity
and explicitly observes and stops the daemon-managed workload.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import time
import typing as tp
from dataclasses import dataclass
from pathlib import Path

from eyvan._internal.json_types import JsonObject, JsonValue, is_json_object
from eyvan.capacity.drivers.base import ExecutionWorkloadStatus, MutationContext
from eyvan.records import (
    NativeContainerConfig,
    NativeContainerMount,
    native_container_mount_targets_overlap,
)
from eyvan.secrets import redact_secret_material
from eyvan.worker.runtime.oci_callable_stager import (
    OCI_CALLABLE_RUNNER_CONTAINER_DIRECTORY,
    OCI_CALLABLE_RUNNER_FILENAME,
    OCI_DEPENDENCY_ENTRY_FILENAME,
    OCI_WORKSPACE_ENTRY_FILENAME,
    stage_oci_callable_runner,
)
from eyvan.worker.runtime.systemd_transient import (
    _OUTPUT_SUPPRESSED,
    SystemdCommandRunner,
    SystemdTransientServiceExecutor,
    _durable_unlink,
    _ensure_private_directory,
    _environment,
    _escape_systemd_exec_arg,
    _file_lock,
    _load_json,
    _record_lock_path,
    _record_secret_bearing,
    _write_environment_file,
)

_CONTAINER_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_CONTAINER_ID = re.compile(r"[0-9a-f]{12,64}\Z")
_CALLABLE_WORKER_MODULE = "eyvan.worker.runtime.callable_worker"
#: ``-m`` modules the controller launches and the staged standalone file each
#: becomes inside the image. Wrappers nest (``--with`` around ``--workspace``
#: around a callable), so every occurrence is rewritten, not only the first.
_STAGED_MODULES = {
    _CALLABLE_WORKER_MODULE: OCI_CALLABLE_RUNNER_FILENAME,
    "eyvan.worker.runtime.dependency_entry": OCI_DEPENDENCY_ENTRY_FILENAME,
    "eyvan.worker.runtime.workspace_entry": OCI_WORKSPACE_ENTRY_FILENAME,
}
# Tokens that prove an entrypoint is a shell composition rather than the one
# callable-worker command. Quoted argument values survive shlex as ordinary
# tokens, so an exact-token comparison only rejects real shell control flow.
_SHELL_CONTROL_TOKENS = frozenset({"&", "&&", "|", "|&", ";", ";;", "(", ")"})
_STANDALONE_REDIRECT = re.compile(r"\d*(>>?|<<?)|\d*>&\d*")
_ACTIVE_CONTAINER_STATES = frozenset({"created", "paused", "restarting", "running", "removing"})

# How long a container may take to leave the active states after ``docker stop``
# returns. A ``--rm`` container is removed asynchronously by the daemon, and a
# TPU workload's processes can sit in uninterruptible driver calls while they
# release the accelerator, so the transition is not instantaneous.
_CONTAINER_STOP_SETTLE_S = 30.0
_CONTAINER_STOP_SETTLE_POLL_S = 0.5


def _env_seconds(name: str, default: float) -> float:
    """Read a finite, non-negative seconds override, falling back on any bad value."""

    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if math.isfinite(value) and value >= 0.0 else default


_HOST_ONLY_BASE_ENVIRONMENT = frozenset({"LOGNAME", "PATH", "PYTHONHOME", "SHELL", "USER", "VIRTUAL_ENV"})
_HOST_DOCKER_HOME = "/root"
_HOST_DOCKER_PATH = "/snap/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
_DEFAULT_WORKLOAD_DIRECTORY = "/var/lib/eyvan/workload"
_DEFAULT_MULTIHOST_DIRECTORY = "/opt/eyvan-native/multihost"
_OCI_CALLABLE_RUNNER_DIRECTORY_NAME = "oci-callable-runner"
_DOCKER_PULL_POLICIES = {
    "always": "always",
    "if-not-present": "missing",
    "never": "never",
}

_ContainerInspectionOutcome: tp.TypeAlias = tp.Literal["present", "missing", "unavailable"]


@dataclass(frozen=True, slots=True)
class _ContainerMount:
    source: str
    target: str
    read_only: bool


@dataclass(frozen=True, slots=True)
class _ContainerRuntime:
    image: str
    pull_policy: str
    working_dir: str
    privileged: bool
    mounts: tuple[_ContainerMount, ...]


@dataclass(frozen=True, slots=True)
class _ContainerState:
    """Validated lifecycle fields from one Docker inspection document."""

    name: str
    error: str
    exit_code: int | None


def _container_state(payload: tp.Mapping[str, JsonValue]) -> tuple[_ContainerState | None, str]:
    """Decode the Docker state boundary without status or string fallbacks."""

    try:
        raw_state = payload["State"]
    except KeyError:
        return None, "Docker inspection has no container State object"
    if not isinstance(raw_state, dict):
        return None, "Docker inspection has a non-object container State"
    try:
        raw_name = raw_state["Status"]
    except KeyError:
        return None, "Docker inspection State has no Status"
    if not isinstance(raw_name, str) or not raw_name.strip():
        return None, "Docker inspection State has a non-string Status"
    raw_error = raw_state.get("Error", "")
    if not isinstance(raw_error, str):
        return None, "Docker inspection State has a non-string Error"
    raw_exit_code = raw_state.get("ExitCode")
    if raw_exit_code is not None and (isinstance(raw_exit_code, bool) or not isinstance(raw_exit_code, int)):
        return None, "Docker inspection State has a non-integer ExitCode"
    return _ContainerState(
        name=raw_name.strip().lower(),
        # Docker's OCI error can quote environment-expanded command fragments.
        # Presence is useful; its untrusted text is not safe to return.
        error="Docker reported a container error" if raw_error.strip() else "",
        exit_code=raw_exit_code,
    ), ""


def docker_container_name(remote_id: str) -> str:
    """Return a deterministic, Docker-safe container name for one remote ID."""

    if not isinstance(remote_id, str) or not remote_id.strip():
        raise ValueError("container submission ID must be a non-empty string")
    normalized = _CONTAINER_SAFE.sub("-", remote_id.strip()).strip(".-") or "workload"
    normalized = normalized[:48].rstrip(".-") or "workload"
    digest = hashlib.sha256(remote_id.encode("utf-8")).hexdigest()[:16]
    return f"eyvan-container-{normalized}-{digest}"


def _absolute_path(value: str, label: str) -> str:
    if not value or "\x00" in value:
        raise ValueError(f"native OCI {label} must be a non-empty NUL-free absolute path")
    if not value.startswith("/") or value.startswith("//"):
        raise ValueError(f"native OCI {label} must be an absolute path")
    parts = value.split("/")
    if any(part in {".", ".."} for part in parts):
        raise ValueError(f"native OCI {label} must not contain dot path segments")
    if value == "/":
        raise ValueError(f"native OCI {label} must not be root")
    return value.rstrip("/")


def _mount_from_value(value: NativeContainerMount, *, label: str) -> _ContainerMount:
    source = _absolute_path(value.source, f"{label} source")
    target = _absolute_path(value.target, f"{label} target")
    read_only = value.read_only
    if "," in source or "," in target:
        raise ValueError(f"native OCI {label} paths must not contain commas")
    return _ContainerMount(source=source, target=target, read_only=read_only)


def _runtime_from_configuration(configuration: NativeContainerConfig, *, workload_directory: str) -> _ContainerRuntime:
    image = configuration.image
    if not image.strip() or "\x00" in image:
        raise ValueError("native OCI image must be a non-empty NUL-free string")
    pull_policy = configuration.pull_policy
    if pull_policy not in {"always", "if-not-present", "never"}:
        raise ValueError("native OCI pull_policy must be always, if-not-present, or never")
    working_dir = configuration.working_dir
    if working_dir is None:
        working_dir = workload_directory
    working_dir = _absolute_path(working_dir, "working_dir")
    privileged = configuration.privileged
    mounts = tuple(_mount_from_value(mount, label="mount") for mount in configuration.mounts)
    targets = [mount.target for mount in mounts]
    if len(targets) != len(set(targets)):
        raise ValueError("native OCI mounts must not contain duplicate targets")
    return _ContainerRuntime(
        image=image.strip(),
        pull_policy=pull_policy,
        working_dir=working_dir,
        privileged=privileged,
        mounts=mounts,
    )


def _gang_accelerators(spec: tp.Mapping[str, JsonValue]) -> tuple[bool, bool]:
    gang = spec.get("gang")
    if not isinstance(gang, dict):
        raise ValueError("native OCI workloads require a gang resource contract")
    per_replica = gang.get("per_replica")
    if not isinstance(per_replica, dict):
        raise ValueError("native OCI gang per_replica resources must be a mapping")

    def requested(name: str) -> bool:
        raw = per_replica.get(name, 0)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"native OCI gang {name} resources must be numeric")
        if raw < 0:
            raise ValueError(f"native OCI gang {name} resources must not be negative")
        return raw > 0

    return requested("gpu"), requested("tpu")


class DockerContainerExecutor(SystemdTransientServiceExecutor):
    """Run native workloads in OCI images through the local Docker daemon.

    Docker is intentionally invoked directly by a root-owned transient service.
    The container stays foreground-attached to that service for journal logs,
    but its lifecycle is always reconciled through Docker itself because the
    daemon does not place the container in the client's systemd cgroup.
    """

    def __init__(
        self,
        state_dir: str | os.PathLike[str] | None = None,
        *,
        container: NativeContainerConfig,
        artifact_mounts: tp.Sequence[str] = (),
        workload_directory: str | os.PathLike[str] = _DEFAULT_WORKLOAD_DIRECTORY,
        multihost_directory: str | os.PathLike[str] = _DEFAULT_MULTIHOST_DIRECTORY,
        base_environment: tp.Mapping[str, str] | None = None,
        command_runner: SystemdCommandRunner | None = None,
        boot_id_reader: tp.Callable[[], str | None] | None = None,
        cgroup_processes_reader: tp.Callable[[str], bool | None] | None = None,
        use_exit_type_cgroup: bool = False,
        command_timeout_s: float = 315.0,
        docker_stop_timeout_s: int = 30,
        verify_docker: bool = True,
    ) -> None:
        self._container_workload_directory = _absolute_path(str(workload_directory), "workload_directory")
        self._container_multihost_directory = _absolute_path(str(multihost_directory), "multihost_directory")
        self._runtime = _runtime_from_configuration(container, workload_directory=self._container_workload_directory)
        self._artifact_mounts = tuple(
            _absolute_path(destination, "artifact destination") for destination in artifact_mounts
        )
        if len(self._artifact_mounts) != len(set(self._artifact_mounts)):
            raise ValueError("native OCI artifact mounts must not contain duplicate destinations")
        if (
            not isinstance(docker_stop_timeout_s, int)
            or isinstance(docker_stop_timeout_s, bool)
            or docker_stop_timeout_s < 0
        ):
            raise ValueError("native OCI docker stop timeout must be a non-negative integer")
        if not isinstance(verify_docker, bool):
            raise ValueError("native OCI Docker verification setting must be a boolean")
        self._docker_stop_timeout_s = docker_stop_timeout_s

        # Container services must run as the host's Docker-capable service
        # identity.  Do not inherit the native host workload user or its venv.
        super().__init__(
            state_dir,
            command_runner=command_runner,
            boot_id_reader=boot_id_reader,
            cgroup_processes_reader=cgroup_processes_reader,
            use_exit_type_cgroup=use_exit_type_cgroup,
            workload_user=None,
            workload_group=None,
            workload_directory=None,
            base_environment=base_environment,
            command_timeout_s=command_timeout_s,
            _defer_startup_recovery=True,
        )
        self.containers_dir = self.state_dir / "containers"
        _ensure_private_directory(self.containers_dir)
        # Sweep Docker's plaintext atomic temporaries before staging any helper:
        # even a later construction failure must not strand a prior crash's value.
        self._recover_startup()
        self._oci_callable_runner_directory = self.state_dir / _OCI_CALLABLE_RUNNER_DIRECTORY_NAME
        self._oci_callable_runner = stage_oci_callable_runner(self._oci_callable_runner_directory)
        self._automatic_mounts = self._build_automatic_mounts()
        if verify_docker:
            self._verify_docker()

    def _build_automatic_mounts(self) -> tuple[_ContainerMount, ...]:
        mounts = [
            _ContainerMount(self._container_workload_directory, self._container_workload_directory, False),
            _ContainerMount(self._container_multihost_directory, self._container_multihost_directory, True),
            _ContainerMount(
                str(self._oci_callable_runner_directory),
                OCI_CALLABLE_RUNNER_CONTAINER_DIRECTORY,
                True,
            ),
            *(_ContainerMount(destination, destination, True) for destination in self._artifact_mounts),
            *self._runtime.mounts,
        ]
        targets = [mount.target for mount in mounts]
        for index, target in enumerate(targets):
            if any(native_container_mount_targets_overlap(target, previous) for previous in targets[:index]):
                raise ValueError("native OCI automatic and configured mounts must not overlap")
        for mount in mounts:
            if "," in mount.source or "," in mount.target:
                raise ValueError("native OCI mount paths must not contain commas")
        return tuple(mounts)

    def _container_entrypoint(self, entrypoint: str) -> str:
        """Route a controller-generated callable through the OCI-safe runner.

        The controller deliberately retains its normal native-worker command;
        only the OCI executor has to replace the host-only ``eyvan`` module
        with the staged standalone script. Matching compares the parsed argv
        tokens rather than the rendered string, so non-canonical quoting or
        spacing still rewrites, while shell compositions (chaining, pipes,
        redirections) are preserved verbatim.
        """

        try:
            parts = shlex.split(entrypoint)
        except ValueError:
            return entrypoint
        if not any(part in _STAGED_MODULES for part in parts):
            return entrypoint
        if any(part in _SHELL_CONTROL_TOKENS or _STANDALONE_REDIRECT.fullmatch(part) for part in parts):
            return entrypoint
        rewritten: list[str] = []
        changed = False
        for index, part in enumerate(parts):
            # Interpreter flags (``python -I -m ...``) may precede the module
            # pair; everything before ``-m`` is retained verbatim.
            if part in _STAGED_MODULES and index >= 2 and parts[index - 1] == "-m" and rewritten[-1:] == ["-m"]:
                rewritten[-1] = f"{OCI_CALLABLE_RUNNER_CONTAINER_DIRECTORY}/{_STAGED_MODULES[part]}"
                changed = True
                continue
            rewritten.append(part)
        return shlex.join(rewritten) if changed else entrypoint

    def _verify_docker(self) -> None:
        result = self._run(("docker", "version", "--format", "{{.Server.Version}}"))
        if result.returncode != 0:
            raise RuntimeError("native OCI runtime requires a working Docker daemon")
        if not result.stdout.strip():
            raise RuntimeError("native OCI runtime could not determine the Docker server version")

    def _container_name(self, remote_id: str) -> str:
        return docker_container_name(remote_id)

    def _cidfile_path(self, remote_id: str) -> Path:
        return self.containers_dir / f"{hashlib.sha256(remote_id.encode('utf-8')).hexdigest()}.cid"

    def _container_env_file_path(self, remote_id: str) -> Path:
        return self.containers_dir / f"{hashlib.sha256(remote_id.encode('utf-8')).hexdigest()}.env"

    def _environment_file_paths(self, remote_id: str) -> tuple[Path, ...]:
        # Include the base path for upgrade cleanup even though this executor
        # writes only Docker's env file today. Missing paths are idempotent.
        return (*super()._environment_file_paths(remote_id), self._container_env_file_path(remote_id))

    def _setup_file_paths(self, remote_id: str) -> tuple[Path, Path, int, int]:
        """Stage large or ``%``-bearing setup JSON inside the read-only OCI bridge mount."""

        filename = f"{hashlib.sha256(remote_id.encode('utf-8')).hexdigest()}.json"
        host_path = self.state_dir / _OCI_CALLABLE_RUNNER_DIRECTORY_NAME / "setup" / filename
        container_path = Path(OCI_CALLABLE_RUNNER_CONTAINER_DIRECTORY) / "setup" / filename
        return host_path, container_path, 0o755, 0o644

    def _setup_file_host_paths(self, remote_id: str) -> tuple[Path, ...]:
        filename = f"{hashlib.sha256(remote_id.encode('utf-8')).hexdigest()}.json"
        return (
            *super()._setup_file_host_paths(remote_id),
            self.state_dir / _OCI_CALLABLE_RUNNER_DIRECTORY_NAME / "setup" / filename,
        )

    def _environment_temporary_directories(self) -> tuple[Path, ...]:
        """Include Docker's owner-only env directory in construction recovery."""

        return (*super()._environment_temporary_directories(), self.containers_dir)

    def _record_fields(
        self,
        spec: tp.Mapping[str, JsonValue],
        remote_id: str,
        environment: tp.Mapping[str, str],
    ) -> tp.Mapping[str, JsonValue]:
        del spec, environment
        container_name = self._container_name(remote_id)
        return {
            "container_name": container_name,
            "container_image": self._runtime.image,
            "container_cidfile": str(self._cidfile_path(remote_id)),
        }

    def _container_environment(self, environment: tp.Mapping[str, str]) -> dict[str, str]:
        # The host native worker has a venv-oriented PATH/VIRTUAL_ENV and an
        # account identity that must not leak into an OCI image.  Workload and
        # distributed-JAX values remain explicit Docker environment entries.
        launch_environment = {
            name: value for name, value in self._base_environment.items() if name not in _HOST_ONLY_BASE_ENVIRONMENT
        }
        launch_environment.update(environment)
        launch_environment["HOME"] = self._container_workload_directory
        launch_environment["EYVAN_CONTAINER_RUNTIME"] = "1"
        return launch_environment

    @staticmethod
    def _mount_argument(mount: _ContainerMount) -> str:
        suffix = ",readonly" if mount.read_only else ""
        return f"type=bind,src={mount.source},dst={mount.target}{suffix}"

    def _inspect_container(self, reference: str) -> tuple[_ContainerInspectionOutcome, JsonObject | None, str]:
        result = self._run(("docker", "container", "inspect", "--format", "{{json .}}", reference))
        if result.returncode == 0:
            try:
                payload = json.loads(result.stdout.strip())
            except json.JSONDecodeError as exc:
                return "unavailable", None, f"Docker returned invalid container inspection JSON: {exc}"
            if not is_json_object(payload):
                return "unavailable", None, "Docker returned a non-object container inspection"
            return "present", payload, ""
        raw_detail = result.stderr.strip() or result.stdout.strip()
        if "no such container" in raw_detail.lower() or "no such object" in raw_detail.lower():
            return "missing", None, "Docker container is absent"
        return "unavailable", None, "Docker container inspect failed"

    def _container_record_reference(self, record: tp.Mapping[str, JsonValue]) -> tuple[str, ...]:
        references: list[str] = []
        name = record.get("container_name")
        if isinstance(name, str) and name:
            references.append(name)
        container_id = record.get("container_id")
        if isinstance(container_id, str) and _CONTAINER_ID.fullmatch(container_id) and container_id not in references:
            references.append(container_id)
        cidfile = record.get("container_cidfile")
        if isinstance(cidfile, str) and cidfile:
            try:
                value = Path(cidfile).read_text(encoding="utf-8").strip()
            except OSError:
                value = ""
            if _CONTAINER_ID.fullmatch(value) and value not in references:
                references.append(value)
        return tuple(references)

    def _observed_container(self, record: JsonObject) -> tuple[_ContainerInspectionOutcome, JsonObject | None, str]:
        unavailable_detail = ""
        for reference in self._container_record_reference(record):
            outcome, payload, detail = self._inspect_container(reference)
            if outcome == "present":
                if payload is None:
                    return "unavailable", None, "Docker returned an empty present container inspection"
                container_id = payload.get("Id")
                if isinstance(container_id, str) and _CONTAINER_ID.fullmatch(container_id):
                    record["container_id"] = container_id
                return outcome, payload, detail
            if outcome == "unavailable":
                unavailable_detail = detail
                break
        if unavailable_detail:
            return "unavailable", None, unavailable_detail
        return "missing", None, "Docker container is absent"

    def _prepare_launch(self, record: tp.Mapping[str, JsonValue], spec: tp.Mapping[str, JsonValue]) -> None:
        mutable_record: JsonObject = dict(record)
        outcome, _payload, detail = self._observed_container(mutable_record)
        if outcome == "present":
            raise RuntimeError("a container with this durable Eyvan name already exists")
        if outcome == "unavailable":
            raise RuntimeError(f"cannot inspect native OCI container before launch: {detail}")
        cidfile = record.get("container_cidfile")
        if not isinstance(cidfile, str) or not cidfile:
            raise RuntimeError("native OCI execution record has no container cidfile")
        try:
            _durable_unlink(Path(cidfile))
        except OSError:
            raise RuntimeError("could not clear stale native OCI cidfile") from None
        remote_id = record.get("remote_id")
        if not isinstance(remote_id, str) or not remote_id:
            raise RuntimeError("native OCI execution record has no remote ID")
        # The container environment travels as one owner-only 0600 env file so
        # neither the unit argv nor the Docker command line exposes values.
        launch_environment = self._resolved_launch_environment(self._container_environment(_environment(spec)))
        _write_environment_file(self._container_env_file_path(remote_id), launch_environment)
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
        container_name = record.get("container_name")
        cidfile = record.get("container_cidfile")
        remote_id = record.get("remote_id")
        if not isinstance(container_name, str) or not container_name:
            raise RuntimeError("native OCI execution record has no container name")
        if not isinstance(cidfile, str) or not cidfile:
            raise RuntimeError("native OCI execution record has no container cidfile")
        if not isinstance(remote_id, str) or not remote_id:
            raise RuntimeError("native OCI execution record has no remote ID")
        entrypoint = self._rewrite_staged_setup_entrypoint(entrypoint, remote_id)
        entrypoint = self._container_entrypoint(entrypoint)
        gpu_requested, tpu_requested = _gang_accelerators(spec)
        if tpu_requested and not self._runtime.privileged:
            raise ValueError("native OCI TPU workloads require runtime.container.privileged=true")
        if gpu_requested and tpu_requested:
            raise ValueError("native OCI workloads cannot request GPU and TPU resources together")
        del environment
        return (
            "systemd-run",
            "--quiet",
            "--no-block",
            f"--unit={unit_name}",
            f"--description=Eyvan workload {unit_name}",
            "--service-type=exec",
            *(("--property=ExitType=cgroup",) if self._use_exit_type_cgroup else ()),
            "--property=RemainAfterExit=yes",
            "--property=KillMode=control-group",
            "--property=StandardOutput=journal",
            "--property=StandardError=journal",
            "--property=LimitMEMLOCK=infinity",
            "--property=LimitNOFILE=100000",
            f"--setenv=HOME={_HOST_DOCKER_HOME}",
            f"--setenv=PATH={_HOST_DOCKER_PATH}",
            "docker",
            "run",
            "--pull",
            _DOCKER_PULL_POLICIES[self._runtime.pull_policy],
            "--rm",
            "--name",
            container_name,
            "--cidfile",
            cidfile,
            "--network",
            "host",
            "--workdir",
            self._runtime.working_dir,
            "--label",
            "eyvan.managed=true",
            "--label",
            f"eyvan.remote-id={remote_id}",
            "--label",
            "eyvan.worker.runtime=native-container",
            *(option for mount in self._automatic_mounts for option in ("--mount", self._mount_argument(mount))),
            "--env-file",
            str(self._container_env_file_path(remote_id)),
            *(("--gpus", "all") if gpu_requested else ()),
            *(("--privileged",) if self._runtime.privileged else ()),
            "--entrypoint",
            "/bin/sh",
            self._runtime.image,
            "-c",
            _escape_systemd_exec_arg(entrypoint),
        )

    def submit(self, spec: tp.Mapping[str, JsonValue], context: MutationContext) -> str:
        """Validate accelerator/container compatibility before durable launch.

        Every rejection enforced by ``_launch_command`` must also be enforced
        here. ``_launch_command`` runs after the durable PENDING record and env
        file are written, so a rejection that only lives there strands an
        execution record that no ``systemd-run`` will ever back: ``status``
        reports INDETERMINATE forever and ``stop``/``retire`` both refuse to act.
        """

        gpu_requested, tpu_requested = _gang_accelerators(spec)
        if tpu_requested and not self._runtime.privileged:
            raise ValueError("native OCI TPU workloads require runtime.container.privileged=true")
        if gpu_requested and tpu_requested:
            raise ValueError("native OCI workloads cannot request GPU and TPU resources together")
        return super().submit(spec, context)

    def process_status(self, remote_id: str) -> JsonObject:
        """Report container process detail from Docker, not from the host cgroup.

        The inherited systemd implementation reads the transient unit's cgroup,
        but the Docker daemon does not place the container in the client's
        cgroup -- that unit contains the ``docker run`` client, not the workload.
        Inheriting it would make a dead containerized trainer report
        ``observed: true`` with the client's PID: exactly the misdiagnosis this
        endpoint exists to prevent.
        """

        record_path = self._record_path(remote_id)
        with _file_lock(_record_lock_path(record_path)):
            record = _load_json(record_path)
        if record is None:
            return {"remote_id": remote_id, "observed": False, "detail": "no durable execution record"}
        if _record_secret_bearing(record):
            return {
                "remote_id": remote_id,
                "runtime": "native-container",
                "observed": False,
                "detail": _OUTPUT_SUPPRESSED,
                "output_suppressed": True,
            }

        outcome, payload, detail = self._observed_container(record)
        if outcome != "present" or payload is None:
            return {
                "remote_id": remote_id,
                "runtime": "native-container",
                "observed": False,
                "detail": detail or f"native OCI container is {outcome}",
            }
        state, state_error = _container_state(payload)
        if state is None:
            return {
                "remote_id": remote_id,
                "runtime": "native-container",
                "observed": False,
                "detail": state_error,
            }
        raw_state = payload.get("State")
        container_pid = raw_state.get("Pid") if isinstance(raw_state, dict) else None
        if isinstance(container_pid, bool) or not isinstance(container_pid, int) or container_pid <= 0:
            # Docker reports 0 for a stopped container; that is not a PID.
            container_pid = None
        return {
            "remote_id": remote_id,
            "runtime": "native-container",
            "observed": True,
            "container_name": record.get("container_name"),
            "state": state.name,
            "exit_code": state.exit_code,
            "main_pid": container_pid,
            # The host cgroup belongs to the docker client, so a container's
            # process list is not observable through it. Reported as unknown
            # rather than as an empty list.
            "pids": None,
            "process_count": None,
            "detail": state.error or detail,
        }

    def _runtime_status(self, record: JsonObject) -> tuple[ExecutionWorkloadStatus, str] | None:
        outcome, payload, detail = self._observed_container(record)
        if outcome == "missing":
            return None
        if outcome != "present" or payload is None:
            return ExecutionWorkloadStatus.INDETERMINATE, f"cannot observe native OCI container: {detail}"
        state, state_error = _container_state(payload)
        if state is None:
            return ExecutionWorkloadStatus.INDETERMINATE, state_error
        stop_requested = self._record_boolean(record, "stop_requested")
        if state.name in {"created", "restarting"}:
            return ExecutionWorkloadStatus.PENDING, f"native OCI container is {state.name}"
        if state.name in {"running", "paused"}:
            return ExecutionWorkloadStatus.RUNNING, f"native OCI container is {state.name}"
        if state.name == "removing":
            # The daemon is already tearing the container down; reporting it
            # as running would fail open while workload ownership is unclear.
            return ExecutionWorkloadStatus.INDETERMINATE, "native OCI container is being removed"
        if state.name == "exited":
            if state.exit_code is None:
                return ExecutionWorkloadStatus.INDETERMINATE, "Docker inspection State has no integer ExitCode"
            if stop_requested:
                return ExecutionWorkloadStatus.STOPPED, "native OCI container stopped by Eyvan"
            if state.exit_code == 0:
                return ExecutionWorkloadStatus.SUCCEEDED, "native OCI container completed successfully"
            return ExecutionWorkloadStatus.FAILED, (
                f"native OCI container exited with status {state.exit_code!r}{': ' + state.error if state.error else ''}"
            )
        if state.name == "dead":
            if stop_requested:
                return ExecutionWorkloadStatus.STOPPED, "native OCI container stopped by Eyvan"
            return ExecutionWorkloadStatus.FAILED, (
                f"native OCI container is dead{': ' + state.error if state.error else ''}"
            )
        return ExecutionWorkloadStatus.INDETERMINATE, (
            f"Docker reported unexpected native OCI container state {state.name}"
        )

    def _stop_runtime(self, record: tp.Mapping[str, JsonValue]) -> None:
        mutable_record: JsonObject = dict(record)
        outcome, payload, detail = self._observed_container(mutable_record)
        if outcome == "missing":
            return
        if outcome != "present" or payload is None:
            raise RuntimeError(f"cannot inspect native OCI container before stop: {detail}")
        state, state_error = _container_state(payload)
        if state is None:
            raise RuntimeError(state_error)
        if state.name not in _ACTIVE_CONTAINER_STATES:
            return
        references = self._container_record_reference(mutable_record)
        if not references:
            raise RuntimeError("native OCI execution record has no container reference")
        reference = references[0]
        if state.name == "created":
            result = self._run(("docker", "rm", "--force", reference))
            verify_outcome, _verify_payload, _verify_detail = self._observed_container(mutable_record)
            if verify_outcome == "missing":
                return
            if result.returncode != 0:
                raise RuntimeError("Docker could not remove the native OCI container")
            raise RuntimeError("Docker remove returned successfully but the created native OCI container still exists")
        result = self._run(("docker", "stop", "--time", str(self._docker_stop_timeout_s), reference))
        settled, _verify_detail = self._container_settled_after_stop(mutable_record)
        if settled:
            return
        if result.returncode != 0:
            raise RuntimeError("Docker could not stop the native OCI container")
        raise RuntimeError("Docker stop returned successfully but the native OCI container is still active")

    def _container_settled_after_stop(self, record: JsonObject) -> tuple[bool, str]:
        """Wait a bounded window for a stopped container to leave the active states.

        ``docker stop`` returns once the daemon has signalled the container, not
        once the container is gone, and Eyvan runs workload containers with
        ``--rm``.  A stopping container therefore passes through ``removing``,
        which ``_ACTIVE_CONTAINER_STATES`` counts as active -- so a single early
        inspection reports a container that is shutting down normally as one
        that refused to stop.

        Reading that as a hard failure is not merely noisy: the caller marks the
        durable record INDETERMINATE, and an INDETERMINATE record refuses every
        later stop, so the workload can never be cancelled and its accelerator is
        held until an operator intervenes.  Poll instead, and only report failure
        when the container is still active after the window.
        """

        detail = ""
        settle_s = _env_seconds("EYVAN_CONTAINER_STOP_SETTLE_S", _CONTAINER_STOP_SETTLE_S)
        poll_s = _env_seconds("EYVAN_CONTAINER_STOP_POLL_S", _CONTAINER_STOP_SETTLE_POLL_S)
        deadline = time.monotonic() + settle_s
        while True:
            outcome, payload, detail = self._observed_container(record)
            if outcome == "missing":
                return True, detail
            if outcome == "present" and payload is not None:
                state, _state_error = _container_state(payload)
                if state is not None and state.name not in _ACTIVE_CONTAINER_STATES:
                    return True, detail
            if time.monotonic() >= deadline:
                return False, detail
            time.sleep(poll_s)

    def _remove_remaining_container(self, record: tp.Mapping[str, JsonValue]) -> None:
        mutable_record: JsonObject = dict(record)
        outcome, _payload, detail = self._observed_container(mutable_record)
        if outcome == "missing":
            return
        if outcome != "present":
            raise RuntimeError(f"cannot inspect native OCI container before removal: {detail}")
        references = self._container_record_reference(mutable_record)
        if not references:
            raise RuntimeError("native OCI execution record has no container reference")
        result = self._run(("docker", "rm", "--force", references[0]))
        verify_outcome, _verify_payload, _verify_detail = self._observed_container(mutable_record)
        if verify_outcome == "missing":
            return
        if result.returncode != 0:
            raise RuntimeError("Docker could not remove the native OCI container")
        raise RuntimeError("Docker remove returned successfully but the native OCI container still exists")

    def _after_systemd_stop(self, record: tp.Mapping[str, JsonValue]) -> None:
        self._remove_remaining_container(record)

    def _retire_terminal_runtime(self, record: tp.Mapping[str, JsonValue]) -> None:
        self._remove_remaining_container(record)

    def _runtime_logs(self, record: tp.Mapping[str, JsonValue]) -> str | None:
        mutable_record: JsonObject = dict(record)
        outcome, _payload, _detail = self._observed_container(mutable_record)
        if outcome != "present":
            return None
        references = self._container_record_reference(mutable_record)
        if not references:
            return None
        result = self._run(("docker", "logs", references[0]))
        return redact_secret_material(result.stdout) if result.returncode == 0 else None


__all__ = ("DockerContainerExecutor", "docker_container_name")
