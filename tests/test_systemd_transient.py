# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from eyvan.capacity.drivers.base import ExecutionWorkloadStatus, MutationContext, StaleMutationError
from eyvan.worker.runtime.systemd_transient import (
    SystemdCommandResult,
    SystemdLogsUnavailableError,
    SystemdTransientServiceExecutor,
    systemd_unit_name,
)


def _context(*, key: str = "submit-1", generation: int = 1, token: str = "token-1") -> MutationContext:
    return MutationContext(idempotency_key=key, generation=generation, fencing_token=token)


def _spec(
    *,
    submission_id: str = "workload-1",
    workload_id: str | None = None,
    entrypoint: str = "python -c 'print(1)'",
) -> dict:
    return {
        "entrypoint": entrypoint,
        "submission_id": submission_id,
        "environment": {"EXPERIMENT": "systemd"},
        "gang": {"replicas": 1, "per_replica": {"cpu": 1}, "required_topology": {}},
        "metadata": {"eyvan_workload_id": workload_id or submission_id},
    }


class FakeSystemd:
    """Small stateful command runner that models the systemd lifecycle seam."""

    def __init__(self) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.units: dict[str, dict[str, str]] = {}
        self.launch_options: dict[str, set[str]] = {}
        self.background_children: dict[str, bool] = {}
        self.journal: dict[str, str] = {}
        self.fail_launch = False
        self.fail_journal = False
        #: Make `systemctl show` fail without saying the unit is absent -- an
        #: unreachable manager, which is not evidence that anything stopped.
        self.fail_show = False

    @staticmethod
    def _unit_from_run(command: tuple[str, ...]) -> str:
        return next(part.removeprefix("--unit=") for part in command if part.startswith("--unit="))

    def complete(self, unit_name: str, *, result: str = "success", status: str = "0") -> None:
        self.units[unit_name] = {
            "LoadState": "loaded",
            "ActiveState": "active",
            "SubState": "exited",
            "Result": result,
            "ExecMainCode": "exited",
            "ExecMainStatus": status,
            "ControlGroup": f"/system.slice/{unit_name}",
        }

    def remove(self, unit_name: str) -> None:
        self.units.pop(unit_name, None)

    def __call__(self, command: tuple[str, ...]) -> SystemdCommandResult:
        self.commands.append(command)
        if command[0] == "systemd-run":
            unit_name = self._unit_from_run(command)
            if self.fail_launch:
                return SystemdCommandResult(returncode=1, stderr="manager unavailable")
            self.units[unit_name] = {
                "LoadState": "loaded",
                "ActiveState": "active",
                "SubState": "running",
                "Result": "success",
                "ExecMainCode": "exited",
                "ExecMainStatus": "0",
                "ControlGroup": f"/system.slice/{unit_name}",
            }
            self.launch_options[unit_name] = set(command)
            self.background_children[unit_name] = True
            return SystemdCommandResult(returncode=0)
        if command[:2] == ("systemctl", "show"):
            unit_name = command[-1]
            if self.fail_show:
                return SystemdCommandResult(returncode=1, stderr="Failed to connect to bus")
            properties = self.units.get(unit_name)
            if properties is None:
                return SystemdCommandResult(returncode=4, stdout="LoadState=not-found\n")
            return SystemdCommandResult(
                returncode=0,
                stdout="".join(f"{key}={value}\n" for key, value in properties.items()),
            )
        if command[:2] == ("systemctl", "stop"):
            unit_name = command[-1]
            if unit_name not in self.units:
                return SystemdCommandResult(returncode=4, stderr="unit is absent")
            # This child becomes dead only when the production command declares
            # cgroup-wide kill semantics. The assertion below therefore covers
            # the property that makes a shell's `command &` safe to stop.
            if "--property=KillMode=control-group" in self.launch_options[unit_name]:
                self.background_children[unit_name] = False
            self.units[unit_name].update({"ActiveState": "inactive", "SubState": "dead", "Result": "success"})
            return SystemdCommandResult(returncode=0)
        if command[:2] == ("systemctl", "reset-failed"):
            unit_name = command[-1]
            if unit_name not in self.units:
                return SystemdCommandResult(returncode=4, stderr="unit is absent")
            self.units.pop(unit_name, None)
            return SystemdCommandResult(returncode=0)
        if command[0] == "journalctl":
            unit_name = next(part.removeprefix("--unit=") for part in command if part.startswith("--unit="))
            if self.fail_journal:
                return SystemdCommandResult(returncode=1, stderr="journal unavailable")
            return SystemdCommandResult(returncode=0, stdout=self.journal.get(unit_name, ""))
        raise AssertionError(f"unexpected systemd command: {command}")


def _record(executor: SystemdTransientServiceExecutor, remote_id: str) -> dict:
    return json.loads(executor._record_path(remote_id).read_text(encoding="utf-8"))


def _environment_file_from_launch(launch: tuple[str, ...]) -> Path:
    option = next(part for part in launch if part.startswith("--property=EnvironmentFile="))
    # No "-" ignore prefix: a missing environment file must fail the launch loudly.
    assert not option.startswith("--property=EnvironmentFile=-")
    return Path(option.removeprefix("--property=EnvironmentFile="))


def _environment_from_launch(launch: tuple[str, ...]) -> dict[str, str]:
    env_path = _environment_file_from_launch(launch)
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600
    return dict(line.split("=", 1) for line in env_path.read_text(encoding="utf-8").splitlines() if line)


def _write_crash_temporary(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def test_startup_durably_sweeps_exact_crash_orphan_environment_and_record_temporaries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from eyvan.worker.runtime import systemd_transient

    digest = hashlib.sha256(b"crashed-attempt").hexdigest()
    environment_temporary = tmp_path / "env" / f".{digest}.env.crash"
    job_temporary = tmp_path / "jobs" / f".{digest}.json.crash"
    workload_temporary = tmp_path / "workloads" / f".{digest}.json.crash"
    marker = b"crash-orphan-secret-plaintext"
    for temporary in (environment_temporary, job_temporary, workload_temporary):
        _write_crash_temporary(temporary, marker)

    real_fsync = os.fsync
    fsynced_directories: set[tuple[int, int]] = set()

    def tracking_fsync(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        if stat.S_ISDIR(metadata.st_mode):
            fsynced_directories.add((metadata.st_dev, metadata.st_ino))
        real_fsync(descriptor)

    monkeypatch.setattr(systemd_transient.os, "fsync", tracking_fsync)
    SystemdTransientServiceExecutor(
        tmp_path,
        command_runner=FakeSystemd(),
        boot_id_reader=lambda: "boot-a",
    )

    assert not environment_temporary.exists()
    assert not job_temporary.exists()
    assert not workload_temporary.exists()
    expected_directories = {
        (directory.stat().st_dev, directory.stat().st_ino)
        for directory in (tmp_path / "env", tmp_path / "jobs", tmp_path / "workloads")
    }
    assert expected_directories <= fsynced_directories
    assert not any(marker in candidate.read_bytes() for candidate in tmp_path.rglob("*") if candidate.is_file())


@pytest.mark.parametrize("unsafe_kind", ["symlink", "unexpected-name"])
def test_startup_refuses_unsafe_or_unrecognized_hidden_atomic_entries(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    digest = hashlib.sha256(b"unsafe-attempt").hexdigest()
    env_dir = tmp_path / "env"
    env_dir.mkdir(parents=True, mode=0o700)
    if unsafe_kind == "symlink":
        victim = tmp_path / "must-not-follow"
        victim.write_text("victim-value", encoding="utf-8")
        candidate = env_dir / f".{digest}.env.crash"
        candidate.symlink_to(victim)
    else:
        candidate = env_dir / f".{digest}.env.crash.extra"
        _write_crash_temporary(candidate, b"unknown-writer-plaintext")
        victim = None

    with pytest.raises(RuntimeError, match="atomic runtime temporary"):
        SystemdTransientServiceExecutor(
            tmp_path,
            command_runner=FakeSystemd(),
            boot_id_reader=lambda: "boot-a",
        )

    assert candidate.exists() or candidate.is_symlink()
    if victim is not None:
        assert victim.read_text(encoding="utf-8") == "victim-value"


def test_startup_waits_for_a_writer_linearized_before_temporary_creation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from eyvan.worker.runtime import systemd_transient

    digest = hashlib.sha256(b"active-attempt").hexdigest()
    final = tmp_path / "env" / f"{digest}.env"
    writer_holds_startup_lock = threading.Event()
    release_writer = threading.Event()
    constructor_waiting = threading.Event()
    scan_started = threading.Event()
    writer_errors: list[BaseException] = []
    constructor_errors: list[BaseException] = []
    real_mkstemp = systemd_transient.tempfile.mkstemp
    real_flock = systemd_transient.fcntl.flock
    real_sweep = systemd_transient._sweep_atomic_temporary_directory

    def blocking_mkstemp(*args, **kwargs):
        if threading.current_thread().name == "atomic-writer":
            # _atomic_writer_exclusion has acquired LOCK_SH before mkstemp.
            writer_holds_startup_lock.set()
            if not release_writer.wait(timeout=2):
                raise RuntimeError("writer barrier timed out")
        return real_mkstemp(*args, **kwargs)

    def observing_flock(descriptor: int, operation: int) -> None:
        if threading.current_thread().name == "startup-constructor" and operation == systemd_transient.fcntl.LOCK_EX:
            constructor_waiting.set()
        real_flock(descriptor, operation)

    def observing_sweep(directory: Path, *, pattern) -> None:
        scan_started.set()
        real_sweep(directory, pattern=pattern)

    monkeypatch.setattr(systemd_transient.tempfile, "mkstemp", blocking_mkstemp)
    monkeypatch.setattr(systemd_transient.fcntl, "flock", observing_flock)
    monkeypatch.setattr(systemd_transient, "_sweep_atomic_temporary_directory", observing_sweep)

    def write() -> None:
        try:
            systemd_transient._write_environment_file(final, {"TOKEN": "active-writer-value"})
        except BaseException as exc:
            writer_errors.append(exc)

    def construct() -> None:
        try:
            SystemdTransientServiceExecutor(tmp_path, command_runner=FakeSystemd(), boot_id_reader=lambda: "boot-a")
        except BaseException as exc:
            constructor_errors.append(exc)

    writer = threading.Thread(target=write, name="atomic-writer", daemon=True)
    writer.start()
    assert writer_holds_startup_lock.wait(timeout=2)
    assert list(final.parent.glob(f".{final.name}.*")) == []

    constructor = threading.Thread(target=construct, name="startup-constructor", daemon=True)
    constructor.start()
    assert constructor_waiting.wait(timeout=2)
    assert constructor.is_alive()
    assert not scan_started.is_set()
    assert list(final.parent.glob(f".{final.name}.*")) == []

    release_writer.set()
    writer.join(timeout=2)
    constructor.join(timeout=2)

    assert not writer.is_alive()
    assert not constructor.is_alive()
    assert writer_errors == []
    assert constructor_errors == []
    assert scan_started.is_set()
    assert final.read_text(encoding="utf-8") == "TOKEN=active-writer-value\n"
    assert list(final.parent.glob(f".{final.name}.*")) == []


def test_writer_attempting_after_the_startup_snapshot_cannot_create_a_late_temporary(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from eyvan.worker.runtime import systemd_transient

    digest = hashlib.sha256(b"after-snapshot-attempt").hexdigest()
    final = tmp_path / "env" / f"{digest}.env"
    snapshot_complete = threading.Event()
    release_snapshot = threading.Event()
    writer_waiting = threading.Event()
    writer_errors: list[BaseException] = []
    constructor_errors: list[BaseException] = []
    real_flock = systemd_transient.fcntl.flock
    real_sweep = systemd_transient._sweep_atomic_temporary_directory

    def snapshot_barrier(directory: Path, *, pattern) -> None:
        real_sweep(directory, pattern=pattern)
        if directory == tmp_path / "env":
            snapshot_complete.set()
            if not release_snapshot.wait(timeout=2):
                raise RuntimeError("startup snapshot barrier timed out")

    def observing_flock(descriptor: int, operation: int) -> None:
        if threading.current_thread().name == "after-snapshot-writer" and operation == systemd_transient.fcntl.LOCK_SH:
            writer_waiting.set()
        real_flock(descriptor, operation)

    monkeypatch.setattr(systemd_transient, "_sweep_atomic_temporary_directory", snapshot_barrier)
    monkeypatch.setattr(systemd_transient.fcntl, "flock", observing_flock)

    def construct() -> None:
        try:
            SystemdTransientServiceExecutor(tmp_path, command_runner=FakeSystemd(), boot_id_reader=lambda: "boot-a")
        except BaseException as exc:
            constructor_errors.append(exc)

    def write() -> None:
        try:
            systemd_transient._write_environment_file(final, {"TOKEN": "after-snapshot-value"})
        except BaseException as exc:
            writer_errors.append(exc)

    constructor = threading.Thread(target=construct, name="snapshot-constructor", daemon=True)
    constructor.start()
    assert snapshot_complete.wait(timeout=2)

    writer = threading.Thread(target=write, name="after-snapshot-writer", daemon=True)
    writer.start()
    assert writer_waiting.wait(timeout=2)
    assert writer.is_alive()
    assert list(final.parent.glob(f".{final.name}.*")) == []

    release_snapshot.set()
    writer.join(timeout=2)
    constructor.join(timeout=2)

    assert not writer.is_alive()
    assert not constructor.is_alive()
    assert writer_errors == []
    assert constructor_errors == []
    assert final.read_text(encoding="utf-8") == "TOKEN=after-snapshot-value\n"
    assert list(final.parent.glob(f".{final.name}.*")) == []


def test_concurrent_constructors_serialize_the_global_sweep_without_record_lock_deadlock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from eyvan.worker.runtime import systemd_transient

    first_inside_sweep = threading.Event()
    release_first = threading.Event()
    second_waiting = threading.Event()
    second_inside_sweep = threading.Event()
    errors: list[BaseException] = []
    active_sweeps = 0
    maximum_active_sweeps = 0
    counter_lock = threading.Lock()
    real_flock = systemd_transient.fcntl.flock
    real_sweep = systemd_transient._sweep_atomic_temporary_directory

    def observing_flock(descriptor: int, operation: int) -> None:
        if threading.current_thread().name == "constructor-two" and operation == systemd_transient.fcntl.LOCK_EX:
            second_waiting.set()
        real_flock(descriptor, operation)

    def sweep_barrier(directory: Path, *, pattern) -> None:
        nonlocal active_sweeps, maximum_active_sweeps
        if directory == tmp_path / "env":
            with counter_lock:
                active_sweeps += 1
                maximum_active_sweeps = max(maximum_active_sweeps, active_sweeps)
            try:
                if threading.current_thread().name == "constructor-one":
                    first_inside_sweep.set()
                    if not release_first.wait(timeout=2):
                        raise RuntimeError("constructor barrier timed out")
                else:
                    second_inside_sweep.set()
                real_sweep(directory, pattern=pattern)
            finally:
                with counter_lock:
                    active_sweeps -= 1
            return
        real_sweep(directory, pattern=pattern)

    monkeypatch.setattr(systemd_transient.fcntl, "flock", observing_flock)
    monkeypatch.setattr(systemd_transient, "_sweep_atomic_temporary_directory", sweep_barrier)

    def construct() -> None:
        try:
            SystemdTransientServiceExecutor(tmp_path, command_runner=FakeSystemd(), boot_id_reader=lambda: "boot-a")
        except BaseException as exc:
            errors.append(exc)

    first = threading.Thread(target=construct, name="constructor-one", daemon=True)
    second = threading.Thread(target=construct, name="constructor-two", daemon=True)
    first.start()
    assert first_inside_sweep.wait(timeout=2)
    second.start()
    assert second_waiting.wait(timeout=2)
    assert second.is_alive()
    assert not second_inside_sweep.is_set()
    assert maximum_active_sweeps == 1

    release_first.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert errors == []
    assert second_inside_sweep.is_set()
    assert maximum_active_sweeps == 1


@pytest.mark.parametrize(
    ("directory_name", "extension", "writer_kind"),
    [("env", "env", "environment"), ("jobs", "json", "json"), ("workloads", "json", "json")],
)
def test_constructor_sweeps_a_temporary_left_by_a_crashed_atomic_writer(
    tmp_path: Path,
    directory_name: str,
    extension: str,
    writer_kind: str,
) -> None:
    digest = hashlib.sha256(f"crashed-{directory_name}-writer".encode()).hexdigest()
    final = tmp_path / directory_name / f"{digest}.{extension}"
    marker = f"crashed-{directory_name}-atomic-writer-secret"
    script = r"""
import os
import sys
from pathlib import Path
from eyvan.worker.runtime import systemd_transient

path = Path(sys.argv[1])
kind = sys.argv[2]
marker = sys.argv[3]


def crash_before_publish(_source, _destination):
    os._exit(73)


systemd_transient.os.replace = crash_before_publish
if kind == "environment":
    systemd_transient._write_environment_file(path, {"TOKEN": marker})
else:
    systemd_transient._atomic_write_json(path, {"remote_id": "crashed-writer", "status": "pending", "marker": marker})
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, str(final), writer_kind, marker],
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert completed.returncode == 73, completed.stderr
    temporaries = list(final.parent.glob(f".{final.name}.*"))
    assert len(temporaries) == 1
    assert marker.encode() in temporaries[0].read_bytes()

    SystemdTransientServiceExecutor(tmp_path, command_runner=FakeSystemd(), boot_id_reader=lambda: "boot-a")

    assert list(final.parent.glob(f".{final.name}.*")) == []
    assert not any(marker.encode() in candidate.read_bytes() for candidate in tmp_path.rglob("*") if candidate.is_file())


def test_environment_writer_unlinks_the_final_path_when_post_rename_fsync_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from eyvan.worker.runtime import systemd_transient

    path = tmp_path / "env" / f"{'a' * 64}.env"
    real_fsync_directory = systemd_transient._fsync_directory
    calls = 0

    def fail_publish_once(directory: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("publish directory fsync failed")
        real_fsync_directory(directory)

    monkeypatch.setattr(systemd_transient, "_fsync_directory", fail_publish_once)
    with pytest.raises(OSError, match="publish directory fsync failed"):
        systemd_transient._write_environment_file(path, {"TOKEN": "post-rename-secret"})

    assert calls == 2
    assert not path.exists()
    assert list(path.parent.glob(f".{path.name}.*")) == []


def test_environment_writer_fails_closed_when_cleanup_fsync_also_fails(tmp_path: Path, monkeypatch) -> None:
    from eyvan.worker.runtime import systemd_transient

    path = tmp_path / "env" / f"{'b' * 64}.env"
    calls = 0

    def fail_every_directory_fsync(_directory: Path) -> None:
        nonlocal calls
        calls += 1
        raise OSError("directory fsync unavailable")

    monkeypatch.setattr(systemd_transient, "_fsync_directory", fail_every_directory_fsync)
    with pytest.raises(RuntimeError, match="durably remove a failed workload environment"):
        systemd_transient._write_environment_file(path, {"TOKEN": "cleanup-fsync-secret"})

    assert calls == 2
    assert not path.exists()
    assert list(path.parent.glob(f".{path.name}.*")) == []


def test_json_writer_keeps_a_complete_record_when_post_rename_fsync_fails(tmp_path: Path, monkeypatch) -> None:
    from eyvan.worker.runtime import systemd_transient

    path = tmp_path / "jobs" / f"{'c' * 64}.json"
    systemd_transient._atomic_write_json(path, {"remote_id": "last-good", "status": "pending"})

    def fail_directory_fsync(_directory: Path) -> None:
        raise OSError("record directory fsync failed")

    monkeypatch.setattr(systemd_transient, "_fsync_directory", fail_directory_fsync)
    with pytest.raises(OSError, match="record directory fsync failed"):
        systemd_transient._atomic_write_json(path, {"remote_id": "new-complete", "status": "running"})

    assert json.loads(path.read_text(encoding="utf-8"))["remote_id"] == "new-complete"
    assert list(path.parent.glob(f".{path.name}.*")) == []


def test_unit_names_are_deterministic_sanitized_and_collision_resistant() -> None:
    remote_id = "training/team-a/run 1/with:unsafe!characters"
    unit_name = systemd_unit_name(remote_id)

    assert unit_name == systemd_unit_name(remote_id)
    assert unit_name.startswith("eyvan-training-team-a-run-1-with-unsafe-characters-")
    assert unit_name.endswith(".service")
    assert "/" not in unit_name
    assert " " not in unit_name
    assert systemd_unit_name("same prefix?") != systemd_unit_name("same prefix!")


def test_submit_launches_a_contained_systemd_cgroup_and_stop_kills_background_child(tmp_path: Path) -> None:
    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(
        tmp_path,
        command_runner=systemd,
        boot_id_reader=lambda: "boot-a",
        use_exit_type_cgroup=True,
        workload_user="eyvan-workload",
        workload_group="eyvan-workload",
        workload_directory="/var/lib/eyvan/workload",
        base_environment={
            "HOME": "/var/lib/eyvan/workload",
            "PATH": "/opt/eyvan-native/venv/bin:/usr/bin",
            "VIRTUAL_ENV": "/opt/eyvan-native/venv",
        },
    )
    context = _context()
    remote_id = executor.submit(_spec(entrypoint="python train.py & wait"), context)
    record = _record(executor, remote_id)
    unit_name = record["unit_name"]

    launch = systemd.commands[0]
    assert launch[:3] == ("systemd-run", "--quiet", "--no-block")
    assert f"--unit={unit_name}" in launch
    assert "--service-type=exec" in launch
    assert "--property=ExitType=cgroup" in launch
    assert "--property=RemainAfterExit=yes" in launch
    assert "--property=KillMode=control-group" in launch
    assert "--property=LimitMEMLOCK=infinity" in launch
    assert "--property=LimitNOFILE=100000" in launch
    assert "--property=NoNewPrivileges=yes" in launch
    assert "--property=User=eyvan-workload" in launch
    assert "--property=Group=eyvan-workload" in launch
    assert "--property=WorkingDirectory=/var/lib/eyvan/workload" in launch
    assert not any(option == "--property=PrivateTmp=yes" for option in launch)
    assert "-lc" not in launch
    # The workload environment must not be visible in unit argv: systemctl
    # show, ps, and the journal would leak values to every host user.
    assert not any(option.startswith("--setenv=") for option in launch)
    assert not any("/opt/eyvan-native/venv" in option for option in launch)
    assert not any("EXPERIMENT" in option for option in launch)
    env_path = _environment_file_from_launch(launch)
    assert env_path.parent == tmp_path / "env"
    env = _environment_from_launch(launch)
    assert env["EXPERIMENT"] == "systemd"
    assert env["HOME"] == "/var/lib/eyvan/workload"
    assert env["PATH"] == "/opt/eyvan-native/venv/bin:/usr/bin"
    assert env["VIRTUAL_ENV"] == "/opt/eyvan-native/venv"
    assert env["EYVAN_REPLICA_INDEX"] == "0"
    assert env["EYVAN_REPLICA_COUNT"] == "1"
    assert launch[-3:] == ("/bin/sh", "-c", "python train.py & wait")
    assert systemd.background_children[unit_name] is True
    assert executor.status(remote_id).status is ExecutionWorkloadStatus.RUNNING

    executor.stop(remote_id, context)

    assert ("systemctl", "stop", "--no-pager", unit_name) in systemd.commands
    assert systemd.background_children[unit_name] is False
    assert executor.status(remote_id).status is ExecutionWorkloadStatus.STOPPED
    assert not env_path.exists()


def test_secret_scaffolding_never_reaches_the_durable_record(tmp_path: Path, monkeypatch) -> None:
    from eyvan.worker.runtime import systemd_transient

    capability = "eyvs1.durable.capability"
    value = "hf_durable_secret_value"
    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(tmp_path, command_runner=systemd, boot_id_reader=lambda: "boot-a")
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

    remote_id = executor.submit(spec, _context())

    serialized = executor._record_path(remote_id).read_text(encoding="utf-8")
    assert "EYVAN_SECRET_" not in serialized
    assert capability not in serialized
    assert value not in serialized
    launch = next(command for command in systemd.commands if command[0] == "systemd-run")
    environment = _environment_from_launch(launch)
    assert environment["HF_TOKEN"] == value
    assert not [name for name in environment if name.startswith("EYVAN_SECRET_")]

    # Idempotent replay also upgrades a record written by an older worker.
    legacy = _record(executor, remote_id)
    legacy["environment"]["EYVAN_SECRET_0_TOKEN"] = capability
    executor._record_path(remote_id).write_text(json.dumps(legacy), encoding="utf-8")
    assert executor.submit(spec, _context()) == remote_id
    assert "EYVAN_SECRET_" not in executor._record_path(remote_id).read_text(encoding="utf-8")


def test_terminal_recovery_unlinks_the_owner_only_environment_idempotently(tmp_path: Path, monkeypatch) -> None:
    from eyvan.worker.runtime import systemd_transient

    systemd = FakeSystemd()
    first = SystemdTransientServiceExecutor(
        tmp_path,
        command_runner=systemd,
        boot_id_reader=lambda: "boot-a",
        use_exit_type_cgroup=True,
    )
    spec = _spec()
    spec["environment"].update(
        {
            "EYVAN_SECRET_COUNT": "1",
            "EYVAN_SECRET_0_NAME": "HF_TOKEN",
            "EYVAN_SECRET_0_URL": "https://eyvan.test/api/v1/secrets/content",
            "EYVAN_SECRET_0_TOKEN": "eyvs1.recovery.capability",
        }
    )
    monkeypatch.setattr(
        systemd_transient,
        "materialize_secrets",
        lambda _environment: {"HF_TOKEN": "hf_recovery_secret"},
    )
    remote_id = first.submit(spec, _context())
    environment_path = first._environment_file_path(remote_id)
    assert stat.S_IMODE(environment_path.stat().st_mode) == 0o600
    systemd.complete(_record(first, remote_id)["unit_name"])

    # A fresh executor models the worker restarting before collecting terminal
    # state. The first recovery removes the file; later recoveries stay safe.
    second = SystemdTransientServiceExecutor(
        tmp_path,
        command_runner=systemd,
        boot_id_reader=lambda: "boot-a",
        use_exit_type_cgroup=True,
    )
    # Constructor recovery is eager; no status call is needed to remove it.
    assert not environment_path.exists()
    assert second.status(remote_id).status is ExecutionWorkloadStatus.SUCCEEDED
    third = SystemdTransientServiceExecutor(
        tmp_path,
        command_runner=systemd,
        boot_id_reader=lambda: "boot-a",
        use_exit_type_cgroup=True,
    )
    assert third.status(remote_id).status is ExecutionWorkloadStatus.SUCCEEDED
    assert not environment_path.exists()


def test_secret_bearing_output_is_suppressed_before_and_after_cleanup_and_restart(tmp_path: Path, monkeypatch) -> None:
    from eyvan.worker.runtime import systemd_transient

    value = "embedded-systemd-secret"
    systemd = FakeSystemd()
    first = SystemdTransientServiceExecutor(tmp_path, command_runner=systemd, boot_id_reader=lambda: "boot-a")
    spec = _spec()
    spec["environment"].update(
        {
            "EYVAN_SECRET_COUNT": "1",
            "EYVAN_SECRET_0_NAME": "HF_TOKEN",
            "EYVAN_SECRET_0_URL": "https://eyvan.test/api/v1/secrets/content",
            "EYVAN_SECRET_0_TOKEN": "eyvs1.output.capability",
        }
    )
    monkeypatch.setattr(systemd_transient, "materialize_secrets", lambda _environment: {"HF_TOKEN": value})
    remote_id = first.submit(spec, _context())
    record = _record(first, remote_id)
    environment_path = first._environment_file_path(remote_id)
    systemd.journal[record["unit_name"]] = f"prefix-{value}-suffix\n"

    assert first.logs(remote_id) == ""
    assert first.log_page(remote_id).records == ()
    assert first.log_page(remote_id).unavailable is True
    assert first.exec_in_workload(remote_id, ["print-secret"], timeout_s=1)["stdout"] == ""
    assert first.process_status(remote_id)["output_suppressed"] is True
    observation = first.status(remote_id)
    assert observation.status is ExecutionWorkloadStatus.RUNNING
    assert value not in observation.message
    assert not environment_path.exists()

    # The durable boolean, not retained plaintext, keeps restart behavior closed.
    second = SystemdTransientServiceExecutor(tmp_path, command_runner=systemd, boot_id_reader=lambda: "boot-a")
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
    assert value not in surfaces["record"]


def test_constructor_upgrades_n_minus_one_record_and_cleans_absent_indeterminate_runtime(
    tmp_path: Path, monkeypatch
) -> None:
    from eyvan.worker.runtime import systemd_transient

    capability = "eyvs1.legacy.capability"
    value = "hf_legacy_plaintext"
    systemd = FakeSystemd()
    first = SystemdTransientServiceExecutor(tmp_path, command_runner=systemd, boot_id_reader=lambda: "boot-a")
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
    record_path = first._record_path(remote_id)
    environment_path = first._environment_file_path(remote_id)
    legacy = _record(first, remote_id)
    legacy.pop("secret_bearing")
    legacy["environment"]["EYVAN_SECRET_COUNT"] = "1"
    legacy["environment"]["EYVAN_SECRET_0_TOKEN"] = capability
    record_path.write_text(json.dumps(legacy), encoding="utf-8")
    systemd.remove(legacy["unit_name"])

    SystemdTransientServiceExecutor(tmp_path, command_runner=systemd, boot_id_reader=lambda: "boot-a")

    recovered_text = record_path.read_text(encoding="utf-8")
    recovered = json.loads(recovered_text)
    assert recovered["status"] == ExecutionWorkloadStatus.INDETERMINATE.value
    assert recovered["secret_bearing"] is True
    assert not environment_path.exists()
    assert capability not in recovered_text
    assert value not in recovered_text


def test_failed_secret_launch_removes_plaintext_and_redacts_the_record(tmp_path: Path, monkeypatch) -> None:
    from eyvan.worker.runtime import systemd_transient

    capability = "eyvs1.failure.capability"
    value = "hf_failed_launch_secret"
    systemd = FakeSystemd()
    systemd.fail_launch = True
    executor = SystemdTransientServiceExecutor(tmp_path, command_runner=systemd, boot_id_reader=lambda: "boot-a")
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

    with pytest.raises(RuntimeError, match="failed to launch systemd execution") as captured:
        executor.submit(spec, _context())

    record_text = executor._record_path("workload-1").read_text(encoding="utf-8")
    assert not executor._environment_file_path("workload-1").exists()
    assert capability not in record_text
    assert value not in record_text
    assert capability not in str(captured.value)
    assert value not in str(captured.value)


def test_systemd_249_fallback_observes_the_cgroup_before_marking_an_exited_main_process_complete(tmp_path: Path) -> None:
    systemd = FakeSystemd()

    def cgroup_has_processes(control_group: str) -> bool | None:
        unit_name = control_group.rsplit("/", 1)[-1]
        return systemd.background_children.get(unit_name)

    executor = SystemdTransientServiceExecutor(
        tmp_path,
        command_runner=systemd,
        boot_id_reader=lambda: "boot-a",
        cgroup_processes_reader=cgroup_has_processes,
        use_exit_type_cgroup=False,
    )
    remote_id = executor.submit(_spec(entrypoint="python train.py &"), _context())
    unit_name = _record(executor, remote_id)["unit_name"]
    systemd.complete(unit_name)

    launch = systemd.commands[0]
    assert "--property=ExitType=cgroup" not in launch
    assert executor.status(remote_id).status is ExecutionWorkloadStatus.RUNNING

    systemd.background_children[unit_name] = False

    assert executor.status(remote_id).status is ExecutionWorkloadStatus.SUCCEEDED


def test_systemd_249_fallback_accepts_an_exited_unit_after_its_cgroup_is_released(tmp_path: Path) -> None:
    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(
        tmp_path,
        command_runner=systemd,
        boot_id_reader=lambda: "boot-a",
        cgroup_processes_reader=lambda _control_group: pytest.fail("released cgroups must not be probed"),
        use_exit_type_cgroup=False,
    )
    remote_id = executor.submit(_spec(), _context())
    unit_name = _record(executor, remote_id)["unit_name"]
    systemd.complete(unit_name)
    systemd.units[unit_name]["ControlGroup"] = ""

    observation = executor.status(remote_id)

    assert observation.status is ExecutionWorkloadStatus.SUCCEEDED
    assert "completed successfully" in observation["message"]


def test_systemd_249_rechecks_a_transiently_unobservable_cgroup(tmp_path: Path) -> None:
    systemd = FakeSystemd()
    readable = False
    executor = SystemdTransientServiceExecutor(
        tmp_path,
        command_runner=systemd,
        boot_id_reader=lambda: "boot-a",
        cgroup_processes_reader=lambda _control_group: False if readable else None,
        use_exit_type_cgroup=False,
    )
    remote_id = executor.submit(_spec(), _context())
    unit_name = _record(executor, remote_id)["unit_name"]
    systemd.complete(unit_name)

    assert executor.status(remote_id).status is ExecutionWorkloadStatus.INDETERMINATE
    readable = True

    assert executor.status(remote_id).status is ExecutionWorkloadStatus.SUCCEEDED


def test_systemd_249_recheckable_cgroup_ambiguity_becomes_terminal_when_the_unit_disappears(tmp_path: Path) -> None:
    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(
        tmp_path,
        command_runner=systemd,
        boot_id_reader=lambda: "boot-a",
        cgroup_processes_reader=lambda _control_group: None,
        use_exit_type_cgroup=False,
    )
    remote_id = executor.submit(_spec(), _context())
    unit_name = _record(executor, remote_id)["unit_name"]
    systemd.complete(unit_name)

    assert executor.status(remote_id).message == "cannot observe systemd workload cgroup before completion"
    systemd.remove(unit_name)

    observation = executor.status(remote_id)

    assert observation.status is ExecutionWorkloadStatus.INDETERMINATE
    assert observation.message == "systemd service disappeared before a terminal state was recorded"


def test_systemd_249_fallback_fails_closed_when_the_cgroup_cannot_be_read(tmp_path: Path) -> None:
    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(
        tmp_path,
        command_runner=systemd,
        boot_id_reader=lambda: "boot-a",
        cgroup_processes_reader=lambda _control_group: None,
        use_exit_type_cgroup=False,
    )
    remote_id = executor.submit(_spec(), _context())
    unit_name = _record(executor, remote_id)["unit_name"]
    systemd.complete(unit_name)

    observation = executor.status(remote_id)

    assert observation.status is ExecutionWorkloadStatus.INDETERMINATE
    assert "cannot observe" in observation["message"]


def test_missing_service_before_a_terminal_observation_is_indeterminate_and_persisted(tmp_path: Path) -> None:
    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(
        tmp_path,
        command_runner=systemd,
        boot_id_reader=lambda: "boot-a",
        use_exit_type_cgroup=True,
    )
    remote_id = executor.submit(_spec(), _context())
    unit_name = _record(executor, remote_id)["unit_name"]
    systemd.remove(unit_name)

    observation = executor.status(remote_id)

    assert observation.status is ExecutionWorkloadStatus.INDETERMINATE
    assert "disappeared" in observation["message"]
    assert _record(executor, remote_id)["status"] == ExecutionWorkloadStatus.INDETERMINATE.value
    with pytest.raises(RuntimeError, match="cannot safely retire ambiguous"):
        executor.retire(remote_id, _context())


def test_stopping_an_execution_whose_unit_is_already_gone_succeeds(tmp_path: Path) -> None:
    """A unit systemd does not hold is not running, and stop asks nothing else.

    This is what a live v5p-16 multislice cost. The cancel stopped the unit, the
    unit disappeared before its exit was recorded, and every retry re-derived
    the same unknown and refused to confirm the stop -- so the workload stayed
    `cancelling`, its lease was never released, and four TPU VMs went on billing
    with nothing able to end them.

    The outcome stays INDETERMINATE, which is honest: nobody knows whether that
    process succeeded. What it must not do is block the release.
    """

    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(
        tmp_path,
        command_runner=systemd,
        boot_id_reader=lambda: "boot-a",
        use_exit_type_cgroup=True,
    )
    remote_id = executor.submit(_spec(), _context())
    unit_name = _record(executor, remote_id)["unit_name"]
    systemd.remove(unit_name)
    assert executor.status(remote_id).status is ExecutionWorkloadStatus.INDETERMINATE

    executor.stop(remote_id, _context())

    assert executor.status(remote_id).status is ExecutionWorkloadStatus.INDETERMINATE


@pytest.mark.parametrize("stop_returncode", [0, 1])
def test_stop_collected_unit_records_only_a_confirmed_stop(tmp_path: Path, stop_returncode: int) -> None:
    """A successful stop survives collection without inventing a successful exit."""

    class CollectOnStop(FakeSystemd):
        def __call__(self, command: tuple[str, ...]) -> SystemdCommandResult:
            result = super().__call__(command)
            if command[:2] == ("systemctl", "stop") and result.returncode == 0:
                self.remove(command[-1])
                return SystemdCommandResult(returncode=stop_returncode)
            return result

    systemd = CollectOnStop()
    executor = SystemdTransientServiceExecutor(
        tmp_path, command_runner=systemd, boot_id_reader=lambda: "boot-a", use_exit_type_cgroup=True
    )
    remote_id = executor.submit(_spec(), _context())
    unit_name = _record(executor, remote_id)["unit_name"]
    executor.stop(remote_id, _context())

    assert unit_name not in systemd.units
    assert not systemd.background_children[unit_name]
    expected = ExecutionWorkloadStatus.STOPPED if stop_returncode == 0 else ExecutionWorkloadStatus.INDETERMINATE
    assert _record(executor, remote_id)["status"] == expected.value
    restarted = SystemdTransientServiceExecutor(
        tmp_path, command_runner=systemd, boot_id_reader=lambda: "boot-a", use_exit_type_cgroup=True
    )
    assert restarted.status(remote_id).status is expected
    if stop_returncode == 0:
        restarted.retire(remote_id, _context())


def test_a_stop_that_cannot_see_systemd_at_all_still_refuses(tmp_path: Path) -> None:
    """An unreachable manager is not evidence of absence.

    The relaxation above is narrow on purpose: "systemd says there is no such
    unit" and "systemd did not answer" are different facts, and only the first
    proves nothing is running.
    """

    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(
        tmp_path,
        command_runner=systemd,
        boot_id_reader=lambda: "boot-a",
        use_exit_type_cgroup=True,
    )
    remote_id = executor.submit(_spec(), _context())
    systemd.fail_show = True

    with pytest.raises(RuntimeError, match="cannot safely verify"):
        executor.stop(remote_id, _context())


def test_malformed_durable_status_is_indeterminate_with_an_explicit_reason(tmp_path: Path) -> None:
    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(
        tmp_path,
        command_runner=systemd,
        boot_id_reader=lambda: "boot-a",
        use_exit_type_cgroup=True,
    )
    remote_id = executor.submit(_spec(), _context())
    record_path = executor._record_path(remote_id)
    record = _record(executor, remote_id)
    record["status"] = "not-a-systemd-status"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    commands_before = len(systemd.commands)

    observation = executor.status(remote_id)

    assert observation.status is ExecutionWorkloadStatus.INDETERMINATE
    assert "unsupported persisted status" in observation.message
    assert _record(executor, remote_id)["status"] == ExecutionWorkloadStatus.INDETERMINATE.value
    assert len(systemd.commands) == commands_before


def test_terminal_record_survives_transient_unit_collection(tmp_path: Path) -> None:
    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(
        tmp_path,
        command_runner=systemd,
        boot_id_reader=lambda: "boot-a",
        use_exit_type_cgroup=True,
    )
    remote_id = executor.submit(_spec(), _context())
    unit_name = _record(executor, remote_id)["unit_name"]
    systemd.complete(unit_name)

    assert executor.status(remote_id).status is ExecutionWorkloadStatus.SUCCEEDED
    assert _record(executor, remote_id)["unit_collected"] is True
    assert ("systemctl", "reset-failed", "--no-pager", unit_name) in systemd.commands
    systemd.remove(unit_name)

    observation = executor.status(remote_id)

    assert observation.remote_id == remote_id
    assert observation.status is ExecutionWorkloadStatus.SUCCEEDED
    assert observation.message == "systemd service completed successfully"


def test_inactive_unit_without_a_prior_success_observation_is_not_reported_successful(tmp_path: Path) -> None:
    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(
        tmp_path,
        command_runner=systemd,
        boot_id_reader=lambda: "boot-a",
        use_exit_type_cgroup=True,
    )
    remote_id = executor.submit(_spec(), _context())
    unit_name = _record(executor, remote_id)["unit_name"]
    systemd.units[unit_name].update({"ActiveState": "inactive", "SubState": "dead", "Result": "success"})

    observation = executor.status(remote_id)

    assert observation.status is ExecutionWorkloadStatus.FAILED
    assert "before Eyvan observed completion" in observation["message"]


def test_boot_id_change_records_lost_without_querying_a_new_manager(tmp_path: Path) -> None:
    systemd = FakeSystemd()
    boot_id = {"value": "boot-a"}
    executor = SystemdTransientServiceExecutor(tmp_path, command_runner=systemd, boot_id_reader=lambda: boot_id["value"])
    remote_id = executor.submit(_spec(), _context())
    boot_id["value"] = "boot-b"
    commands_before = len(systemd.commands)

    observation = executor.status(remote_id)

    assert observation.status is ExecutionWorkloadStatus.LOST
    assert "boot ID changed" in observation["message"]
    assert len(systemd.commands) == commands_before


def test_submit_is_idempotent_and_rejects_stale_fences(tmp_path: Path) -> None:
    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(tmp_path, command_runner=systemd, boot_id_reader=lambda: "boot-a")
    spec = _spec()
    context = _context(generation=7, token="token-7")

    assert executor.submit(spec, context) == "workload-1"
    assert executor.submit(spec, context) == "workload-1"
    assert len([command for command in systemd.commands if command[0] == "systemd-run"]) == 1
    with pytest.raises(StaleMutationError, match="does not own"):
        executor.submit(spec, _context(generation=8, token="token-8"))
    with pytest.raises(StaleMutationError, match="fencing token"):
        executor.submit(spec, _context(generation=7, token="other"))


def test_newer_logical_attempt_stops_the_previous_cgroup_before_launching(tmp_path: Path, monkeypatch) -> None:
    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(tmp_path, command_runner=systemd, boot_id_reader=lambda: "boot-a")
    monkeypatch.setenv("EYVAN_SUPERSEDE_POLL_S", "0.01")
    first = executor.submit(_spec(submission_id="attempt-1", workload_id="training"), _context(generation=1))
    first_unit = _record(executor, first)["unit_name"]

    second = executor.submit(
        _spec(submission_id="attempt-2", workload_id="training"),
        _context(key="submit-2", generation=2, token="token-2"),
    )

    second_unit = _record(executor, second)["unit_name"]
    stop_index = systemd.commands.index(("systemctl", "stop", "--no-pager", first_unit))
    second_launch_index = systemd.commands.index(
        next(command for command in systemd.commands if f"--unit={second_unit}" in command)
    )
    assert systemd.background_children[first_unit] is False
    assert stop_index < second_launch_index


def test_concurrent_newer_generation_cannot_launch_during_the_prior_generation_fence_gap(tmp_path: Path) -> None:
    systemd = FakeSystemd()
    first_launch_entered = threading.Event()
    release_first_launch = threading.Event()
    second_launch_entered = threading.Event()
    failures: list[BaseException] = []
    results: dict[str, str] = {}

    def runner(command: tuple[str, ...]) -> SystemdCommandResult:
        if command[0] == "systemd-run":
            unit_name = FakeSystemd._unit_from_run(command)
            if "attempt-1" in unit_name:
                first_launch_entered.set()
                if not release_first_launch.wait(timeout=3):
                    raise TimeoutError("first launch was not released")
            if "attempt-2" in unit_name:
                second_launch_entered.set()
        return systemd(command)

    executor = SystemdTransientServiceExecutor(
        tmp_path,
        command_runner=runner,
        boot_id_reader=lambda: "boot-a",
        use_exit_type_cgroup=True,
    )

    def submit(name: str, spec: dict, context: MutationContext) -> None:
        try:
            results[name] = executor.submit(spec, context)
        except BaseException as exc:  # pragma: no cover - asserted by the caller
            failures.append(exc)

    first = threading.Thread(
        target=submit,
        args=("first", _spec(submission_id="attempt-1", workload_id="training"), _context(generation=1)),
    )
    first.start()
    assert first_launch_entered.wait(timeout=3)

    second = threading.Thread(
        target=submit,
        args=(
            "second",
            _spec(submission_id="attempt-2", workload_id="training"),
            _context(key="submit-2", generation=2, token="token-2"),
        ),
    )
    second.start()
    assert not second_launch_entered.wait(timeout=0.1)

    release_first_launch.set()
    first.join(timeout=3)
    second.join(timeout=3)

    assert not first.is_alive()
    assert not second.is_alive()
    assert failures == []
    assert results == {"first": "attempt-1", "second": "attempt-2"}
    launches = [FakeSystemd._unit_from_run(command) for command in systemd.commands if command[0] == "systemd-run"]
    assert launches[0].startswith("eyvan-attempt-1-")
    assert launches[1].startswith("eyvan-attempt-2-")
    workload_fence = json.loads(executor._workload_fence_path("training").read_text(encoding="utf-8"))
    assert workload_fence["generation"] == 2
    assert workload_fence["remote_id"] == "attempt-2"


def test_launch_failure_is_recorded_only_after_systemd_proves_unit_absent(tmp_path: Path) -> None:
    systemd = FakeSystemd()
    systemd.fail_launch = True
    executor = SystemdTransientServiceExecutor(tmp_path, command_runner=systemd, boot_id_reader=lambda: "boot-a")

    with pytest.raises(RuntimeError, match="failed to launch systemd execution"):
        executor.submit(_spec(), _context())

    assert _record(executor, "workload-1")["status"] == ExecutionWorkloadStatus.FAILED.value


def test_submit_rejects_multiline_environment_values_before_recording(tmp_path: Path) -> None:
    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(tmp_path, command_runner=systemd, boot_id_reader=lambda: "boot-a")
    spec = _spec()
    spec["environment"] = {"TOKEN": "first\nsecond"}

    with pytest.raises(ValueError, match="must not contain a newline"):
        executor.submit(spec, _context())

    assert not executor._record_path("workload-1").exists()
    assert not any(command[0] == "systemd-run" for command in systemd.commands)


def test_logs_distinguish_a_journal_failure_from_an_empty_log(tmp_path: Path) -> None:
    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(
        tmp_path,
        command_runner=systemd,
        boot_id_reader=lambda: "boot-a",
        use_exit_type_cgroup=True,
    )
    remote_id = executor.submit(_spec(), _context())
    unit_name = _record(executor, remote_id)["unit_name"]

    # A genuinely silent workload and a workload unknown to this host both
    # return an empty string; neither is a log transport failure.
    assert executor.logs(remote_id) == ""
    assert executor.logs("absent-execution") == ""

    systemd.journal[unit_name] = "real output\n"
    assert executor.logs(remote_id) == "real output\n"

    systemd.fail_journal = True
    with pytest.raises(SystemdLogsUnavailableError) as captured:
        executor.logs(remote_id)
    assert captured.value.unit_name == unit_name
    assert "journal unavailable" in captured.value.stderr


def test_a_failed_worker_names_its_own_dying_words_in_the_status(tmp_path: Path) -> None:
    """A dead rank must report why it died, not only that systemd noticed.

    A multi-host job once burned its whole provisioning cycle to answer
    ``systemd service failed with result exit-code``: every worker had died
    seconds after start on ``ModuleNotFoundError``, a fact printed to the log
    stream and absent from the status every consumer actually reads. The
    unit's journal tail is read on failure, and the last line that reads like
    an exception -- a traceback's closing line -- rides along in the message.
    """

    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(tmp_path, command_runner=systemd, boot_id_reader=lambda: "boot-a")
    remote_id = executor.submit(_spec(entrypoint="python -m device_test"), _context())
    unit_name = _record(executor, remote_id)["unit_name"]
    systemd.complete(unit_name, result="exit-code", status="1")
    systemd.units[unit_name].update({"ActiveState": "failed", "SubState": "failed", "ControlGroup": ""})
    systemd.journal[unit_name] = "\n".join(
        [
            "please install `torch` (cpu) if you want to use `AutoEasyDeLModel*.from_torch_pretrained`",
            "Traceback (most recent call last):",
            '  File "…/callable_protocol.py", line 629, in execute_callable',
            '  File "launch-test.py", line 20, in device_test',
            "ModuleNotFoundError: No module named 'spectrax'",
            "eyvan-device_test.service: Main process exited, code=exited, status=1/FAILURE",
        ]
    )

    observation = executor.status(remote_id)

    assert observation.status is ExecutionWorkloadStatus.FAILED
    assert "systemd service failed with result exit-code" in observation.message
    assert "ModuleNotFoundError: No module named 'spectrax'" in observation.message


def test_a_failed_worker_without_a_journal_keeps_the_plain_message(tmp_path: Path) -> None:
    """An unreadable journal degrades to exactly the old failure message."""

    systemd = FakeSystemd()
    systemd.fail_journal = True
    executor = SystemdTransientServiceExecutor(tmp_path, command_runner=systemd, boot_id_reader=lambda: "boot-a")
    remote_id = executor.submit(_spec(), _context())
    unit_name = _record(executor, remote_id)["unit_name"]
    systemd.complete(unit_name, result="exit-code", status="1")
    systemd.units[unit_name].update({"ActiveState": "failed", "SubState": "failed", "ControlGroup": ""})

    observation = executor.status(remote_id)

    assert observation.status is ExecutionWorkloadStatus.FAILED
    assert observation.message == "systemd service failed with result exit-code"


def test_a_failed_worker_names_a_non_traceback_last_word_too(tmp_path: Path) -> None:
    """``SystemExit`` and shell errors get their closing line, not silence."""

    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(tmp_path, command_runner=systemd, boot_id_reader=lambda: "boot-a")
    remote_id = executor.submit(_spec(), _context())
    unit_name = _record(executor, remote_id)["unit_name"]
    systemd.complete(unit_name, result="exit-code", status="1")
    systemd.units[unit_name].update({"ActiveState": "failed", "SubState": "failed", "ControlGroup": ""})
    systemd.journal[unit_name] = "venv/bin/python: No module named device_test\n"

    observation = executor.status(remote_id)

    assert observation.status is ExecutionWorkloadStatus.FAILED
    assert "No module named device_test" in observation.message


def test_systemds_own_closing_lines_are_never_the_reason(tmp_path: Path) -> None:
    """Prefer the launcher's diagnostic over systemd's generic closing lines when reporting failure."""

    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(tmp_path, command_runner=systemd, boot_id_reader=lambda: "boot-a")
    remote_id = executor.submit(_spec(), _context())
    unit_name = _record(executor, remote_id)["unit_name"]
    systemd.complete(unit_name, result="exit-code", status="1")
    systemd.units[unit_name].update({"ActiveState": "failed", "SubState": "failed", "ControlGroup": ""})
    systemd.journal[unit_name] = "\n".join(
        [
            "Started /bin/sh -c python -m eyvan.worker.runtime.dependency_entry --require einops==0.8.1 -- ...",
            "eyvan: /var/lib/eyvan/runtime-envs is not writable; caching this job's packages in ./.eyvan-runtime-envs",
            "/var/lib/eyvan/workload/.eyvan-runtime-envs/6a88/bin/python: Error while finding module specification "
            "for 'eyvan.worker.runtime.callable_worker' (ModuleNotFoundError: No module named 'eyvan')",
            f"{unit_name}: Main process exited, code=exited, status=1/FAILURE",
            f"{unit_name}: Failed with result 'exit-code'.",
        ]
    )

    observation = executor.status(remote_id)

    assert "ModuleNotFoundError: No module named 'eyvan'" in observation.message
    assert "Failed with result" not in observation.message.split("worker output ended with:", 1)[1]


def test_a_journal_of_nothing_but_systemd_lifecycle_keeps_the_plain_message(tmp_path: Path) -> None:
    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(tmp_path, command_runner=systemd, boot_id_reader=lambda: "boot-a")
    remote_id = executor.submit(_spec(), _context())
    unit_name = _record(executor, remote_id)["unit_name"]
    systemd.complete(unit_name, result="exit-code", status="1")
    systemd.units[unit_name].update({"ActiveState": "failed", "SubState": "failed", "ControlGroup": ""})
    systemd.journal[unit_name] = f"{unit_name}: Failed with result 'exit-code'.\n"

    observation = executor.status(remote_id)

    assert observation.message == "systemd service failed with result exit-code"


def test_a_jobs_own_starting_line_is_not_mistaken_for_systemd(tmp_path: Path) -> None:
    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(tmp_path, command_runner=systemd, boot_id_reader=lambda: "boot-a")
    remote_id = executor.submit(_spec(), _context())
    unit_name = _record(executor, remote_id)["unit_name"]
    systemd.complete(unit_name, result="exit-code", status="1")
    systemd.units[unit_name].update({"ActiveState": "failed", "SubState": "failed", "ControlGroup": ""})
    systemd.journal[unit_name] = "\n".join(
        ["loading data", "Stopping data loader: disk full", f"{unit_name}: Failed with result 'exit-code'."]
    )

    observation = executor.status(remote_id)

    assert "Stopping data loader: disk full" in observation.message


def test_a_retired_execution_is_launched_again_when_submitted_again(tmp_path: Path) -> None:
    """Relaunch an explicitly retired submission on retry rather than replay its stopped execution."""

    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(
        tmp_path, command_runner=systemd, boot_id_reader=lambda: "boot-a", use_exit_type_cgroup=True
    )
    remote_id = executor.submit(_spec(), _context())
    executor.retire(remote_id, _context())
    assert _record(executor, remote_id)["retired"] is True
    launches_before = sum(1 for command in systemd.commands if command[0] == "systemd-run")

    assert executor.submit(_spec(), _context()) == remote_id

    launches_after = sum(1 for command in systemd.commands if command[0] == "systemd-run")
    assert launches_after == launches_before + 1
    record = _record(executor, remote_id)
    assert "retired" not in record
    assert record["status"] not in {ExecutionWorkloadStatus.STOPPED.value, ExecutionWorkloadStatus.FAILED.value}


def test_retire_cleanup_failure_is_pending_and_submit_retries_it(tmp_path: Path, monkeypatch) -> None:
    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(tmp_path, command_runner=systemd, boot_id_reader=lambda: "boot-a")
    remote_id = executor.submit(_spec(), _context())
    systemd.complete(systemd_unit_name(remote_id))
    failures = True
    cleanup_calls = []

    def cleanup(record):
        cleanup_calls.append(record["remote_id"])
        if failures:
            raise RuntimeError("docker rm failed")

    monkeypatch.setattr(executor, "_retire_terminal_runtime", cleanup)
    with pytest.raises(RuntimeError, match="docker rm failed"):
        executor.retire(remote_id, _context(key="retire-1"))
    record = _record(executor, remote_id)
    assert record.get("retired") is not True
    assert record["retire_cleanup_pending"] is True
    launches = sum(command[0] == "systemd-run" for command in systemd.commands)
    with pytest.raises(RuntimeError, match="docker rm failed"):
        executor.submit(_spec(), _context())
    assert sum(command[0] == "systemd-run" for command in systemd.commands) == launches
    assert len(cleanup_calls) == 2
    failures = False
    executor.submit(_spec(), _context())
    assert len(cleanup_calls) == 3
    assert sum(command[0] == "systemd-run" for command in systemd.commands) == launches + 1
    assert _record(executor, remote_id)["incarnation"] == 1


def test_submission_key_can_retire_each_relaunched_incarnation(tmp_path: Path) -> None:
    """Each failed gang launch must be able to roll back using its submission key."""

    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(tmp_path, command_runner=systemd, boot_id_reader=lambda: "boot-a")
    context = _context()
    remote_id = executor.submit(_spec(), context)
    executor.retire(remote_id, context)
    assert _record(executor, remote_id)["retired"] is True
    executor.submit(_spec(), context)
    assert _record(executor, remote_id)["incarnation"] == 1
    stops = sum(command[:2] == ("systemctl", "stop") for command in systemd.commands)

    executor.retire(remote_id, context)

    record = _record(executor, remote_id)
    assert record["status"] == ExecutionWorkloadStatus.STOPPED.value
    assert record["retired"] is True
    assert record["incarnation"] == 1
    assert sum(command[:2] == ("systemctl", "stop") for command in systemd.commands) > stops


@pytest.mark.parametrize("legacy", [False, True])
def test_replayed_retire_does_not_stop_relaunched_incarnation(tmp_path: Path, legacy: bool) -> None:
    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(tmp_path, command_runner=systemd, boot_id_reader=lambda: "boot-a")
    remote_id = executor.submit(_spec(), _context())
    if legacy:
        record = _record(executor, remote_id)
        record.pop("incarnation", None)
        executor._record_path(remote_id).write_text(json.dumps(record))
    executor.retire(remote_id, _context(key="retire-1"))
    executor.submit(_spec(), _context())
    before = _record(executor, remote_id)
    stops = sum(command[:2] == ("systemctl", "stop") for command in systemd.commands)
    executor.retire(remote_id, _context(key="retire-1"))
    assert _record(executor, remote_id) == before
    assert sum(command[:2] == ("systemctl", "stop") for command in systemd.commands) == stops
    assert before["incarnation"] == 1
    executor.retire(remote_id, _context(key="retire-2"))
    assert _record(executor, remote_id)["retired"] is True


def test_retire_keys_are_bounded_to_recent_incarnations(tmp_path: Path) -> None:
    """Every rollback now carries a fresh key, so the replay ledger must not grow forever.

    It is pruned by incarnation, not by count: the record is written with sorted
    keys, so insertion order cannot say which entries are oldest.
    """

    from eyvan.worker.runtime.systemd_transient import _RETIRE_KEY_INCARNATION_WINDOW as window

    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(tmp_path, command_runner=systemd, boot_id_reader=lambda: "boot-a")
    remote_id = executor.submit(_spec(), _context())
    cycles = window * 3
    for index in range(cycles):
        executor.retire(remote_id, _context(key=f"retire-{index}"))
        executor.submit(_spec(), _context())

    record = _record(executor, remote_id)
    assert record["incarnation"] == cycles
    keys = record["retire_keys"]
    assert len(keys) <= window
    assert min(keys.values()) >= cycles - window

    # A replay from inside the window is still refused against the current run.
    stops = sum(command[:2] == ("systemctl", "stop") for command in systemd.commands)
    executor.retire(remote_id, _context(key=f"retire-{cycles - 1}"))
    assert _record(executor, remote_id) == record
    assert sum(command[:2] == ("systemctl", "stop") for command in systemd.commands) == stops


@pytest.mark.parametrize("terminal_observation", [False, True])
def test_retire_rechecks_incarnation_after_observation(tmp_path: Path, monkeypatch, terminal_observation: bool) -> None:
    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(tmp_path, command_runner=systemd, boot_id_reader=lambda: "boot-a")
    remote_id = executor.submit(_spec(), _context())
    if terminal_observation:
        systemd.complete(systemd_unit_name(remote_id))
    status = executor.status
    replacement = []
    stops = []

    def interleaved_status(remote_id):
        observed = status(remote_id)
        monkeypatch.setattr(executor, "status", status)
        executor.retire(remote_id, _context(key="concurrent-retire"))
        executor.submit(_spec(), _context())
        replacement.append(_record(executor, remote_id))
        stops.append(sum(command[:2] == ("systemctl", "stop") for command in systemd.commands))
        return observed

    monkeypatch.setattr(executor, "status", interleaved_status)
    executor.retire(remote_id, _context(key="original-retire"))
    assert _record(executor, remote_id) == replacement[0]
    assert sum(command[:2] == ("systemctl", "stop") for command in systemd.commands) == stops[0]


def test_a_launch_that_ended_by_itself_is_still_a_replay(tmp_path: Path) -> None:
    """At-least-once dispatch must not run a finished attempt twice."""

    systemd = FakeSystemd()
    executor = SystemdTransientServiceExecutor(
        tmp_path, command_runner=systemd, boot_id_reader=lambda: "boot-a", use_exit_type_cgroup=True
    )
    remote_id = executor.submit(_spec(), _context())
    systemd.complete(_record(executor, remote_id)["unit_name"])
    executor.status(remote_id)
    launches_before = sum(1 for command in systemd.commands if command[0] == "systemd-run")

    executor.submit(_spec(), _context())

    assert sum(1 for command in systemd.commands if command[0] == "systemd-run") == launches_before


def test_a_direct_stop_holds_the_relaunch_lock_through_runtime_io(tmp_path: Path) -> None:
    """A stop releases the record lock for its runtime I/O; a relaunch must not slip in.

    The unit name is deterministic, so a relaunch that landed between a stop's
    record read and its ``systemctl stop`` would be killed by that stop.
    """

    import fcntl

    seen: list[str] = []

    class Probe(SystemdTransientServiceExecutor):
        def _stop_runtime(self, record) -> None:
            path = self._record_path(record["remote_id"]).with_suffix(".retire-stop.lock")
            descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    seen.append("held")
                else:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                    seen.append("free")
            finally:
                os.close(descriptor)

    executor = Probe(tmp_path, command_runner=FakeSystemd(), boot_id_reader=lambda: "boot-a")
    remote_id = executor.submit(_spec(), _context())

    executor.stop(remote_id, _context(key="stop-1"))

    assert seen == ["held"]


def test_a_timed_out_retire_leaves_a_relaunched_incarnation_alone(tmp_path: Path, monkeypatch) -> None:
    """The timeout branch must not mark a replacement run INDETERMINATE."""

    from eyvan.worker.runtime import systemd_transient as module

    class StopFails(SystemdTransientServiceExecutor):
        fail = True

        def _stop_runtime(self, record) -> None:
            if self.fail:
                raise RuntimeError("runtime stop failed")

    executor = StopFails(tmp_path, command_runner=FakeSystemd(), boot_id_reader=lambda: "boot-a")
    remote_id = executor.submit(_spec(), _context())
    record_path = executor._record_path(remote_id)

    class Clock:
        calls = 0

        @classmethod
        def monotonic(cls) -> float:
            cls.calls += 1
            if cls.calls == 1:
                return 0.0
            if cls.calls == 2:
                # Between the failed stop and the deadline check, another caller
                # finishes retiring incarnation 0 and the dispatch retry relaunches.
                executor.fail = False
                record = module._load_json(record_path)
                record.update(status=ExecutionWorkloadStatus.STOPPED.value, retired=True)
                module._atomic_write_json(record_path, record)
                executor.submit(_spec(), _context())
            return 1e9

        @staticmethod
        def sleep(_seconds: float) -> None:
            return None

    monkeypatch.setattr(module, "time", Clock)

    executor.retire(remote_id, _context(key="retire-1"))

    record = _record(executor, remote_id)
    assert record["incarnation"] == 1
    assert record["status"] != ExecutionWorkloadStatus.INDETERMINATE.value
