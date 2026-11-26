# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import hashlib
import json
import shlex
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from eyvan.capacity.drivers.base import ExecutionWorkloadStatus, MutationContext
from eyvan.records import NativeContainerConfig, NativeContainerMount
from eyvan.worker.runtime.container import DockerContainerExecutor, docker_container_name
from eyvan.worker.runtime.oci_callable_stager import (
    OCI_CALLABLE_RUNNER_CONTAINER_DIRECTORY,
    OCI_CALLABLE_RUNNER_FILENAME,
)
from eyvan.worker.runtime.systemd_transient import SystemdCommandResult


def _context(*, key: str = "submit-1", generation: int = 1, token: str = "token-1") -> MutationContext:
    return MutationContext(idempotency_key=key, generation=generation, fencing_token=token)


def _spec(
    *,
    submission_id: str = "workload-1",
    entrypoint: str = "python train.py",
    resources: dict[str, int] | None = None,
) -> dict:
    return {
        "entrypoint": entrypoint,
        "submission_id": submission_id,
        "environment": {
            "JAX_PROCESS_ID": "0",
            "PYTHONPATH": "/opt/eyvan-native/multihost",
            "WANDB_MODE": "offline",
        },
        "gang": {
            "replicas": 1,
            "per_replica": resources or {"cpu": 4},
            "required_topology": {},
        },
        "metadata": {"eyvan_workload_id": submission_id},
    }


class FakeDockerSystemd:
    """Stateful Docker and systemd seam for native OCI executor tests."""

    def __init__(
        self,
        *,
        images: set[str] | None = None,
        materialize_containers: bool = True,
        materialize_on_systemd_stop: bool = False,
    ) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.images = set(images or ())
        self.materialize_containers = materialize_containers
        self.materialize_on_systemd_stop = materialize_on_systemd_stop
        self.containers: dict[str, dict] = {}
        self.units: dict[str, dict[str, str]] = {}
        self.journal: dict[str, str] = {}
        self.deferred_containers: dict[str, tuple[str, Path]] = {}

    @staticmethod
    def _option(command: tuple[str, ...], option: str) -> str:
        return command[command.index(option) + 1]

    @staticmethod
    def _unit(command: tuple[str, ...]) -> str:
        return next(value.removeprefix("--unit=") for value in command if value.startswith("--unit="))

    def _container_for_reference(self, reference: str) -> dict | None:
        if reference in self.containers:
            return self.containers[reference]
        return next((container for container in self.containers.values() if container["Id"] == reference), None)

    def _materialize_container(self, name: str, cidfile: Path) -> None:
        container_id = f"{len(self.containers) + 1:064x}"
        cidfile.write_text(container_id, encoding="utf-8")
        self.containers[name] = {
            "Id": container_id,
            "Name": name,
            "Logs": f"{name} output\n",
            "State": {"Status": "running", "ExitCode": 0, "Error": ""},
        }

    def __call__(self, command: tuple[str, ...]) -> SystemdCommandResult:
        self.commands.append(command)
        if command[:2] == ("docker", "version"):
            return SystemdCommandResult(returncode=0, stdout="27.1.0\n")
        if command[:3] == ("docker", "image", "inspect"):
            image = command[-1]
            if image in self.images:
                return SystemdCommandResult(returncode=0, stdout="sha256:present\n")
            return SystemdCommandResult(returncode=1, stderr=f"Error: No such image: {image}")
        if command[:2] == ("docker", "pull"):
            self.images.add(command[-1])
            return SystemdCommandResult(returncode=0, stdout="pulled\n")
        if command[:3] == ("docker", "container", "inspect"):
            container = self._container_for_reference(command[-1])
            if container is None:
                return SystemdCommandResult(returncode=1, stderr=f"Error: No such container: {command[-1]}")
            return SystemdCommandResult(returncode=0, stdout=f"{json.dumps(container)}\n")
        if command[:2] == ("docker", "logs"):
            container = self._container_for_reference(command[-1])
            if container is None:
                return SystemdCommandResult(returncode=1, stderr="Error: No such container")
            return SystemdCommandResult(returncode=0, stdout=str(container["Logs"]))
        if command[:2] == ("docker", "stop"):
            container = self._container_for_reference(command[-1])
            if container is None:
                return SystemdCommandResult(returncode=1, stderr="Error: No such container")
            container["State"].update({"Status": "exited", "ExitCode": 137})
            return SystemdCommandResult(returncode=0, stdout=f"{container['Name']}\n")
        if command[:3] == ("docker", "rm", "--force"):
            container = self._container_for_reference(command[-1])
            if container is None:
                return SystemdCommandResult(returncode=1, stderr="Error: No such container")
            self.containers.pop(str(container["Name"]), None)
            return SystemdCommandResult(returncode=0, stdout=f"{container['Name']}\n")
        if command[0] == "systemd-run":
            unit_name = self._unit(command)
            docker_index = command.index("docker")
            docker = command[docker_index:]
            assert docker[:2] == ("docker", "run")
            name = self._option(docker, "--name")
            cidfile = Path(self._option(docker, "--cidfile"))
            if self.materialize_containers:
                self._materialize_container(name, cidfile)
            else:
                self.deferred_containers[unit_name] = (name, cidfile)
            self.units[unit_name] = {
                "LoadState": "loaded",
                "ActiveState": "active",
                "SubState": "running",
                "Result": "success",
                "ExecMainCode": "exited",
                "ExecMainStatus": "0",
                "ControlGroup": f"/system.slice/{unit_name}",
            }
            return SystemdCommandResult(returncode=0)
        if command[:2] == ("systemctl", "show"):
            unit_name = command[-1]
            properties = self.units.get(unit_name)
            if properties is None:
                return SystemdCommandResult(returncode=4, stdout="LoadState=not-found\n")
            return SystemdCommandResult(
                returncode=0,
                stdout="".join(f"{name}={value}\n" for name, value in properties.items()),
            )
        if command[:2] == ("systemctl", "stop"):
            unit_name = command[-1]
            if unit_name not in self.units:
                return SystemdCommandResult(returncode=4, stderr="unit is absent")
            if self.materialize_on_systemd_stop and unit_name in self.deferred_containers:
                name, cidfile = self.deferred_containers.pop(unit_name)
                self._materialize_container(name, cidfile)
            self.units[unit_name].update({"ActiveState": "inactive", "SubState": "dead", "Result": "success"})
            return SystemdCommandResult(returncode=0)
        if command[:2] == ("systemctl", "reset-failed"):
            self.units.pop(command[-1], None)
            return SystemdCommandResult(returncode=0)
        if command[0] == "journalctl":
            unit_name = next(value.removeprefix("--unit=") for value in command if value.startswith("--unit="))
            return SystemdCommandResult(returncode=0, stdout=self.journal.get(unit_name, ""))
        raise AssertionError(f"unexpected command: {command}")


def _runtime(
    *,
    pull_policy: str = "if-not-present",
    privileged: bool = False,
) -> NativeContainerConfig:
    return NativeContainerConfig(
        image="registry.example/easydel:latest",
        pull_policy=pull_policy,
        privileged=privileged,
        mounts=(NativeContainerMount(source="/datasets", target="/datasets", read_only=True),),
    )


def _executor(
    tmp_path: Path,
    daemon: FakeDockerSystemd,
    *,
    runtime: NativeContainerConfig | None = None,
) -> DockerContainerExecutor:
    return DockerContainerExecutor(
        tmp_path,
        container=runtime or _runtime(),
        artifact_mounts=("/opt/eyvan-runtime",),
        base_environment={
            "HF_HOME": "/var/lib/eyvan/workload/hf",
            "HOME": "/incorrect-host-home",
            "LOGNAME": "eyvan-workload",
            "PATH": "/opt/eyvan-native/venv/bin:/usr/bin",
            "USER": "eyvan-workload",
            "VIRTUAL_ENV": "/opt/eyvan-native/venv",
        },
        command_runner=daemon,
        boot_id_reader=lambda: "boot-a",
        use_exit_type_cgroup=True,
    )


def _record(executor: DockerContainerExecutor, remote_id: str) -> dict:
    return json.loads(executor._record_path(remote_id).read_text(encoding="utf-8"))


def test_docker_startup_durably_sweeps_a_crash_orphan_plaintext_environment_temporary(tmp_path: Path) -> None:
    containers = tmp_path / "containers"
    digest = hashlib.sha256(b"crashed-container-attempt").hexdigest()
    final = containers / f"{digest}.env"
    marker = "docker-crash-orphan-secret"
    script = r"""
import os
import sys
from pathlib import Path
from eyvan.worker.runtime import systemd_transient

path = Path(sys.argv[1])
marker = sys.argv[2]


def crash_before_publish(_source, _destination):
    os._exit(74)


systemd_transient.os.replace = crash_before_publish
systemd_transient._write_environment_file(path, {"TOKEN": marker})
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(final), marker],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 74, completed.stderr
    temporaries = list(containers.glob(f".{final.name}.*"))
    assert len(temporaries) == 1
    assert marker.encode() in temporaries[0].read_bytes()

    executor = _executor(tmp_path, FakeDockerSystemd(images={"registry.example/easydel:latest"}))

    assert executor.containers_dir == containers
    assert list(containers.glob(f".{final.name}.*")) == []
    assert not any(marker.encode() in candidate.read_bytes() for candidate in tmp_path.rglob("*") if candidate.is_file())


def test_container_name_is_deterministic_sanitized_and_collision_resistant() -> None:
    remote_id = "training/team-a/run 1/with:unsafe!characters"

    name = docker_container_name(remote_id)

    assert name == docker_container_name(remote_id)
    assert name.startswith("eyvan-container-training-team-a-run-1-with-unsafe-characters-")
    assert "/" not in name
    assert " " not in name
    assert docker_container_name("same prefix?") != docker_container_name("same prefix!")


def test_container_launch_mounts_runtime_and_preserves_only_container_safe_environment(tmp_path: Path) -> None:
    daemon = FakeDockerSystemd(images={"registry.example/easydel:latest"})
    executor = _executor(tmp_path, daemon)

    remote_id = executor.submit(_spec(resources={"gpu": 1}), _context())

    record = _record(executor, remote_id)
    launch = next(command for command in daemon.commands if command[0] == "systemd-run")
    docker = launch[launch.index("docker") :]

    assert record["container_name"] == docker_container_name(remote_id)
    assert record["container_image"] == "registry.example/easydel:latest"
    assert "--property=User=eyvan-workload" not in launch
    assert "--property=NoNewPrivileges=yes" not in launch
    assert "--setenv=HOME=/root" in launch
    assert "--setenv=PATH=/snap/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" in launch
    assert docker[:2] == ("docker", "run")
    assert ("--pull", "missing") == tuple(docker[docker.index("--pull") : docker.index("--pull") + 2])
    assert "--rm" in docker
    assert ("--network", "host") == tuple(docker[docker.index("--network") : docker.index("--network") + 2])
    assert ("--workdir", "/var/lib/eyvan/workload") == tuple(
        docker[docker.index("--workdir") : docker.index("--workdir") + 2]
    )
    assert "--gpus" in docker
    assert docker[docker.index("--gpus") + 1] == "all"
    assert "--privileged" not in docker
    assert "--entrypoint" in docker
    assert docker[docker.index("--entrypoint") + 1 : docker.index("--entrypoint") + 4] == (
        "/bin/sh",
        "registry.example/easydel:latest",
        "-c",
    )
    mounts = [docker[index + 1] for index, value in enumerate(docker) if value == "--mount"]
    assert "type=bind,src=/var/lib/eyvan/workload,dst=/var/lib/eyvan/workload" in mounts
    assert "type=bind,src=/opt/eyvan-native/multihost,dst=/opt/eyvan-native/multihost,readonly" in mounts
    assert "type=bind,src=/opt/eyvan-runtime,dst=/opt/eyvan-runtime,readonly" in mounts
    assert "type=bind,src=/datasets,dst=/datasets,readonly" in mounts
    runner_mount = next(mount for mount in mounts if f"dst={OCI_CALLABLE_RUNNER_CONTAINER_DIRECTORY},readonly" in mount)
    runner_source = Path(runner_mount.split(",")[1].removeprefix("src="))
    assert runner_source == tmp_path / "oci-callable-runner"
    assert (runner_source / OCI_CALLABLE_RUNNER_FILENAME).is_file()
    assert "/opt/eyvan-native/venv" not in runner_mount
    env = [docker[index + 1] for index, value in enumerate(docker) if value == "--env"]
    assert env == []
    # The container environment travels in an owner-only env file so values
    # never appear in the unit or Docker command lines.
    env_path = Path(docker[docker.index("--env-file") + 1])
    assert env_path.parent == tmp_path / "containers"
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600
    container_env = dict(line.split("=", 1) for line in env_path.read_text(encoding="utf-8").splitlines() if line)
    assert container_env["HOME"] == "/var/lib/eyvan/workload"
    assert container_env["EYVAN_CONTAINER_RUNTIME"] == "1"
    assert container_env["HF_HOME"] == "/var/lib/eyvan/workload/hf"
    assert container_env["JAX_PROCESS_ID"] == "0"
    assert container_env["PYTHONPATH"] == "/opt/eyvan-native/multihost"
    assert "VIRTUAL_ENV" not in container_env
    assert "PATH" not in container_env
    assert "USER" not in container_env
    assert "LOGNAME" not in container_env
    assert not any("WANDB_MODE" in option for option in launch)
    assert not any("offline" in option for option in launch)
    assert not any("HF_HOME" in option for option in launch)

    assert executor.status(remote_id).status is ExecutionWorkloadStatus.RUNNING
    assert _record(executor, remote_id)["container_id"] == daemon.containers[record["container_name"]]["Id"]
    assert executor.logs(remote_id) == f"{record['container_name']} output\n"


def test_container_constructor_recovers_n_minus_one_env_without_status_call(tmp_path: Path, monkeypatch) -> None:
    from eyvan.worker.runtime import systemd_transient

    capability = "eyvs1.container-legacy.capability"
    value = "container-secret-value"
    daemon = FakeDockerSystemd(images={"registry.example/easydel:latest"})
    first = _executor(tmp_path, daemon)
    spec = _spec()
    spec["environment"].update(
        {
            "EYVAN_SECRET_COUNT": "1",
            "EYVAN_SECRET_0_NAME": "HF_TOKEN",
            "EYVAN_SECRET_0_URL": "https://eyvan.test/api/v1/secrets/content",
            "EYVAN_SECRET_0_TOKEN": capability,
        }
    )
    monkeypatch.setattr(systemd_transient, "materialize_secrets", lambda _environment: {"HF_TOKEN": value})
    remote_id = first.submit(spec, _context())
    record = _record(first, remote_id)
    env_path = first._container_env_file_path(remote_id)
    assert value in env_path.read_text(encoding="utf-8")

    record.pop("secret_bearing")
    record["environment"]["EYVAN_SECRET_0_TOKEN"] = capability
    first._record_path(remote_id).write_text(json.dumps(record), encoding="utf-8")
    daemon.containers.pop(record["container_name"])
    daemon.units.pop(record["unit_name"])

    second = _executor(tmp_path, daemon)

    recovered_text = second._record_path(remote_id).read_text(encoding="utf-8")
    recovered = json.loads(recovered_text)
    assert recovered["status"] == ExecutionWorkloadStatus.INDETERMINATE.value
    assert recovered["secret_bearing"] is True
    assert not env_path.exists()
    assert capability not in recovered_text
    assert value not in recovered_text


def test_container_secret_output_and_docker_errors_are_suppressed_across_restart(tmp_path: Path, monkeypatch) -> None:
    from eyvan.worker.runtime import systemd_transient

    value = "embedded-container-secret"
    daemon = FakeDockerSystemd(images={"registry.example/easydel:latest"})
    first = _executor(tmp_path, daemon)
    spec = _spec()
    spec["environment"].update(
        {
            "EYVAN_SECRET_COUNT": "1",
            "EYVAN_SECRET_0_NAME": "HF_TOKEN",
            "EYVAN_SECRET_0_URL": "https://eyvan.test/api/v1/secrets/content",
            "EYVAN_SECRET_0_TOKEN": "eyvs1.container-output.capability",
        }
    )
    monkeypatch.setattr(systemd_transient, "materialize_secrets", lambda _environment: {"HF_TOKEN": value})
    remote_id = first.submit(spec, _context())
    record = _record(first, remote_id)
    container = daemon.containers[record["container_name"]]
    container["Logs"] = f"prefix-{value}-suffix\n"

    assert first.logs(remote_id) == ""
    assert first.log_page(remote_id).unavailable is True
    assert first.exec_in_workload(remote_id, ["print-secret"], timeout_s=1)["output_suppressed"] is True
    assert first.process_status(remote_id)["output_suppressed"] is True

    container["State"].update({"Status": "dead", "ExitCode": 1, "Error": f"runtime-prefix-{value}-suffix"})
    terminal = first.status(remote_id)
    assert terminal.status is ExecutionWorkloadStatus.FAILED
    assert value not in terminal.message
    assert not first._container_env_file_path(remote_id).exists()

    second = _executor(tmp_path, daemon)
    surfaces = {
        "logs": second.logs(remote_id),
        "page": second.log_page(remote_id).detail,
        "exec": second.exec_in_workload(remote_id, ["print-secret"], timeout_s=1),
        "process": second.process_status(remote_id),
        "status": second.status(remote_id).message,
        "record": second._record_path(remote_id).read_text(encoding="utf-8"),
    }
    assert value not in json.dumps(surfaces, default=str)
    assert surfaces["logs"] == ""
    assert surfaces["exec"]["output_suppressed"] is True


def test_container_rewrites_a_controller_callable_to_the_staged_oci_runner(tmp_path: Path) -> None:
    daemon = FakeDockerSystemd(images={"registry.example/easydel:latest"})
    executor = _executor(tmp_path, daemon)
    original_entrypoint = (
        "python -m eyvan.worker.runtime.callable_worker --uri gs://test-bucket/callables/invocation.pkl "
        f"--sha256 {'a' * 64} --python 3.13 --format-version 1 --result-uri gs://test-bucket/results/invocation.bin"
    )

    executor.submit(_spec(entrypoint=original_entrypoint), _context())

    launch = next(command for command in daemon.commands if command[0] == "systemd-run")
    rendered_entrypoint = launch[-1]
    parts = shlex.split(rendered_entrypoint)
    assert parts[:2] == [
        "python",
        f"{OCI_CALLABLE_RUNNER_CONTAINER_DIRECTORY}/{OCI_CALLABLE_RUNNER_FILENAME}",
    ]
    assert "eyvan.worker.runtime.callable_worker" not in parts
    assert parts[2:] == shlex.split(original_entrypoint)[3:]


def test_container_executor_rejects_a_direct_nested_mount_over_the_oci_runner(tmp_path: Path) -> None:
    daemon = FakeDockerSystemd(images={"registry.example/easydel:latest"})
    runtime = NativeContainerConfig(
        image="registry.example/easydel:latest",
        mounts=(
            NativeContainerMount(
                source="/mnt/other",
                target=f"{OCI_CALLABLE_RUNNER_CONTAINER_DIRECTORY}/{OCI_CALLABLE_RUNNER_FILENAME}",
            ),
        ),
    )

    with pytest.raises(ValueError, match="must not overlap"):
        _executor(tmp_path, daemon, runtime=runtime)


def test_container_rewrites_noncanonical_callable_quoting_to_the_oci_runner(tmp_path: Path) -> None:
    daemon = FakeDockerSystemd(images={"registry.example/easydel:latest"})
    executor = _executor(tmp_path, daemon)
    # Extra whitespace, a quoted interpreter path, and a quoted argument value
    # that contains shell-like characters must still match the single command.
    original_entrypoint = (
        "'/opt/eyvan-native/venv/bin/python'  -m   eyvan.worker.runtime.callable_worker "
        "--uri 'gs://test-bucket/callables/a>b.pkl' "
        f"--sha256 {'a' * 64} --python 3.13"
    )

    executor.submit(_spec(entrypoint=original_entrypoint), _context())

    launch = next(command for command in daemon.commands if command[0] == "systemd-run")
    rendered_entrypoint = launch[-1]
    parts = shlex.split(rendered_entrypoint)
    interpreter = "/opt/eyvan-native/venv/bin/python"
    runner_path = f"{OCI_CALLABLE_RUNNER_CONTAINER_DIRECTORY}/{OCI_CALLABLE_RUNNER_FILENAME}"
    assert parts[:2] == [interpreter, runner_path]
    assert "eyvan.worker.runtime.callable_worker" not in parts
    assert parts[2:] == shlex.split(original_entrypoint)[3:]


@pytest.mark.parametrize(
    "original_entrypoint",
    [
        "python -m eyvan.worker.runtime.callable_worker --help && echo complete",
        "python -m eyvan.worker.runtime.callable_worker --help | tee log",
        "python -m eyvan.worker.runtime.callable_worker --help > out.txt",
        "python -m eyvan.worker.runtime.callable_worker --help 2> err.txt; echo done",
    ],
)
def test_container_preserves_shell_composed_callable_entrypoints(tmp_path: Path, original_entrypoint: str) -> None:
    daemon = FakeDockerSystemd(images={"registry.example/easydel:latest"})
    executor = _executor(tmp_path, daemon)

    executor.submit(_spec(entrypoint=original_entrypoint), _context())

    launch = next(command for command in daemon.commands if command[0] == "systemd-run")
    assert launch[-1] == original_entrypoint


def test_container_removing_state_is_indeterminate_not_running(tmp_path: Path) -> None:
    daemon = FakeDockerSystemd(images={"registry.example/easydel:latest"})
    executor = _executor(tmp_path, daemon)
    remote_id = executor.submit(_spec(), _context())
    record = _record(executor, remote_id)
    daemon.containers[record["container_name"]]["State"].update({"Status": "removing"})

    observation = executor.status(remote_id)

    assert observation.status is ExecutionWorkloadStatus.INDETERMINATE
    assert "being removed" in observation.message


def test_container_status_stop_and_logs_follow_docker_before_systemd(tmp_path: Path) -> None:
    daemon = FakeDockerSystemd(images={"registry.example/easydel:latest"})
    executor = _executor(tmp_path, daemon)
    remote_id = executor.submit(_spec(), _context())
    record = _record(executor, remote_id)

    executor.stop(remote_id, _context())

    assert executor.status(remote_id).status is ExecutionWorkloadStatus.STOPPED
    docker_stop = ("docker", "stop", "--time", "30", record["container_name"])
    assert docker_stop in daemon.commands
    docker_stop_index = daemon.commands.index(docker_stop)
    systemd_stop_index = daemon.commands.index(("systemctl", "stop", "--no-pager", record["unit_name"]))
    assert docker_stop_index < systemd_stop_index


def test_container_retire_removes_a_created_container_before_stopping_systemd(tmp_path: Path) -> None:
    daemon = FakeDockerSystemd(images={"registry.example/easydel:latest"})
    executor = _executor(tmp_path, daemon)
    remote_id = executor.submit(_spec(), _context())
    record = _record(executor, remote_id)
    daemon.containers[record["container_name"]]["State"].update({"Status": "created", "ExitCode": 0})

    executor.retire(remote_id, _context())

    docker_rm = ("docker", "rm", "--force", record["container_name"])
    systemd_stop = ("systemctl", "stop", "--no-pager", record["unit_name"])
    assert docker_rm in daemon.commands
    assert record["container_name"] not in daemon.containers
    assert executor.status(remote_id).status is ExecutionWorkloadStatus.STOPPED
    assert daemon.commands.index(docker_rm) < daemon.commands.index(systemd_stop)


def test_container_stop_removes_container_created_during_async_pull_stop_race(tmp_path: Path) -> None:
    daemon = FakeDockerSystemd(materialize_containers=False, materialize_on_systemd_stop=True)
    executor = _executor(tmp_path, daemon)
    remote_id = executor.submit(_spec(submission_id="pull-stop-race"), _context())
    record = _record(executor, remote_id)

    executor.stop(remote_id, _context())

    systemd_stop = ("systemctl", "stop", "--no-pager", record["unit_name"])
    docker_rm = ("docker", "rm", "--force", record["container_name"])
    assert systemd_stop in daemon.commands
    assert docker_rm in daemon.commands
    assert daemon.commands.index(systemd_stop) < daemon.commands.index(docker_rm)
    assert record["container_name"] not in daemon.containers
    assert executor.status(remote_id).status is ExecutionWorkloadStatus.STOPPED


def test_container_retire_removes_a_dead_container_and_preserves_journal_logs(tmp_path: Path) -> None:
    daemon = FakeDockerSystemd(images={"registry.example/easydel:latest"})
    executor = _executor(tmp_path, daemon)
    remote_id = executor.submit(_spec(submission_id="dead-container"), _context())
    record = _record(executor, remote_id)
    daemon.containers[record["container_name"]]["State"].update(
        {"Status": "dead", "ExitCode": 137, "Error": "daemon failure"}
    )
    daemon.journal[record["unit_name"]] = "dead container logs\n"

    executor.retire(remote_id, _context())

    assert ("docker", "rm", "--force", record["container_name"]) in daemon.commands
    assert record["container_name"] not in daemon.containers
    assert executor.logs(remote_id) == "dead container logs\n"


def test_container_terminal_exit_uses_docker_exit_code_and_collects_unit(tmp_path: Path) -> None:
    daemon = FakeDockerSystemd(images={"registry.example/easydel:latest"})
    executor = _executor(tmp_path, daemon)
    remote_id = executor.submit(_spec(), _context())
    record = _record(executor, remote_id)
    daemon.containers[record["container_name"]]["State"].update({"Status": "exited", "ExitCode": 0})

    observation = executor.status(remote_id)

    assert observation.remote_id == remote_id
    assert observation.status is ExecutionWorkloadStatus.SUCCEEDED
    assert observation.message == "native OCI container completed successfully"
    assert ("systemctl", "reset-failed", "--no-pager", record["unit_name"]) in daemon.commands


def test_container_malformed_inspection_state_is_indeterminate_with_an_explicit_reason(tmp_path: Path) -> None:
    daemon = FakeDockerSystemd(images={"registry.example/easydel:latest"})
    executor = _executor(tmp_path, daemon)
    remote_id = executor.submit(_spec(), _context())
    record = _record(executor, remote_id)
    daemon.containers[record["container_name"]]["State"] = {"Status": 7, "ExitCode": 0, "Error": ""}

    observation = executor.status(remote_id)

    assert observation.status is ExecutionWorkloadStatus.INDETERMINATE
    assert observation.message == "Docker inspection State has a non-string Status"
    assert _record(executor, remote_id)["status"] == ExecutionWorkloadStatus.INDETERMINATE.value


def test_container_status_falls_back_to_systemd_after_docker_removes_a_completed_container(tmp_path: Path) -> None:
    daemon = FakeDockerSystemd(images={"registry.example/easydel:latest"})
    executor = _executor(tmp_path, daemon)
    remote_id = executor.submit(_spec(), _context())
    record = _record(executor, remote_id)
    daemon.containers.pop(record["container_name"])
    daemon.units[record["unit_name"]].update(
        {
            "ActiveState": "active",
            "SubState": "exited",
            "ControlGroup": "",
        }
    )

    observation = executor.status(remote_id)

    assert observation.status is ExecutionWorkloadStatus.SUCCEEDED
    assert observation["message"] == "systemd service completed successfully"


def test_container_submit_returns_while_the_systemd_unit_owns_an_async_image_pull(tmp_path: Path) -> None:
    daemon = FakeDockerSystemd(materialize_containers=False)
    executor = _executor(tmp_path, daemon)

    remote_id = executor.submit(_spec(submission_id="async-pull"), _context())
    observation = executor.status(remote_id)

    assert remote_id == "async-pull"
    assert observation.status is ExecutionWorkloadStatus.RUNNING
    assert observation.message == "systemd service is running"
    assert not any(command[:2] in {("docker", "pull"), ("docker", "image")} for command in daemon.commands)


def test_container_image_pull_policies_and_tpu_privilege_are_enforced(tmp_path: Path) -> None:
    daemon = FakeDockerSystemd(images={"registry.example/easydel:latest"})
    always = _executor(tmp_path / "always", daemon, runtime=_runtime(pull_policy="always", privileged=True))

    always.submit(_spec(submission_id="always", resources={"tpu": 16}), _context())

    launch = next(command for command in daemon.commands if command[0] == "systemd-run")
    assert ("--pull", "always") == tuple(launch[launch.index("--pull") : launch.index("--pull") + 2])
    assert "--privileged" in launch

    missing_daemon = FakeDockerSystemd()
    on_demand = _executor(tmp_path / "if-not-present", missing_daemon, runtime=_runtime())
    on_demand.submit(_spec(submission_id="if-not-present"), _context())
    on_demand_launch = next(command for command in missing_daemon.commands if command[0] == "systemd-run")
    assert ("--pull", "missing") == tuple(
        on_demand_launch[on_demand_launch.index("--pull") : on_demand_launch.index("--pull") + 2]
    )

    denied_daemon = FakeDockerSystemd(images={"registry.example/easydel:latest"})
    denied = _executor(tmp_path / "denied", denied_daemon, runtime=_runtime(privileged=False))
    with pytest.raises(ValueError, match="privileged=true"):
        denied.submit(_spec(submission_id="denied", resources={"tpu": 16}), _context())
    assert not any(command[0] == "systemd-run" for command in denied_daemon.commands)

    absent_daemon = FakeDockerSystemd()
    never = _executor(tmp_path / "never", absent_daemon, runtime=_runtime(pull_policy="never"))
    never.submit(_spec(submission_id="never"), _context())
    never_launch = next(command for command in absent_daemon.commands if command[0] == "systemd-run")
    assert ("--pull", "never") == tuple(never_launch[never_launch.index("--pull") : never_launch.index("--pull") + 2])
    assert not any(command[:2] in {("docker", "pull"), ("docker", "image")} for command in absent_daemon.commands)


def test_container_rejects_gpu_and_tpu_before_writing_a_durable_record(tmp_path: Path) -> None:
    """A GPU+TPU request must be rejected by ``submit``, not by ``_launch_command``.

    ``_launch_command`` runs after the durable PENDING record and env file are
    written, so a rejection that only lives there strands an execution record
    that no ``systemd-run`` will ever back: ``status`` reports INDETERMINATE
    forever and ``stop``/``retire`` both refuse to act on it.
    """

    daemon = FakeDockerSystemd(images={"registry.example/easydel:latest"})
    root = tmp_path / "gpu-and-tpu"
    executor = _executor(root, daemon, runtime=_runtime(privileged=True))
    before = sorted(path.relative_to(root).as_posix() for path in root.rglob("*")) if root.exists() else []

    with pytest.raises(ValueError, match="cannot request GPU and TPU resources together"):
        executor.submit(_spec(submission_id="gpu-and-tpu", resources={"gpu": 8, "tpu": 16}), _context())

    assert not any(command[0] == "systemd-run" for command in daemon.commands)
    after = sorted(path.relative_to(root).as_posix() for path in root.rglob("*")) if root.exists() else []
    assert after == before
