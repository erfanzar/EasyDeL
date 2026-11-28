# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Wait for slow container teardown and allow a later stop to resolve an indeterminate record.

Containers using ``--rm`` can pass through the active ``removing`` state after
``docker stop`` succeeds. A transient teardown error must not permanently block
future stop attempts or release capacity while a container could still run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_container_executor import FakeDockerSystemd, _context, _executor, _record, _spec

from eyvan.capacity.drivers.base import ExecutionWorkloadStatus
from eyvan.worker.runtime.container import DockerContainerExecutor
from eyvan.worker.runtime.systemd_transient import SystemdCommandResult


class SlowTeardownDocker(FakeDockerSystemd):
    """A daemon whose ``--rm`` container lingers in ``removing`` after ``docker stop``.

    This is ordinary Docker behaviour, not a fault: removal is asynchronous, and
    a TPU workload's processes can sit in uninterruptible driver calls while they
    release the accelerator.
    """

    def __init__(self, *, removing_inspections: int, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._remaining_removing = removing_inspections
        self._stopped_names: set[str] = set()

    def __call__(self, command: tuple[str, ...]) -> SystemdCommandResult:
        if command[:2] == ("docker", "stop"):
            container = self._container_for_reference(command[-1])
            if container is not None:
                # The daemon accepts the stop and begins asynchronous removal.
                self.commands.append(command)
                self._stopped_names.add(str(container["Name"]))
                container["State"].update({"Status": "removing", "ExitCode": 137})
                return SystemdCommandResult(returncode=0, stdout=f"{container['Name']}\n")
        if command[:3] == ("docker", "container", "inspect"):
            container = self._container_for_reference(command[-1])
            if container is not None and str(container["Name"]) in self._stopped_names:
                if self._remaining_removing > 0:
                    self._remaining_removing -= 1
                else:
                    container["State"].update({"Status": "exited"})
        return super().__call__(command)


def _fast_settle(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the bounded settle window from making the suite wait on wall clock."""

    monkeypatch.setenv("EYVAN_CONTAINER_STOP_SETTLE_S", "5")
    monkeypatch.setenv("EYVAN_CONTAINER_STOP_POLL_S", "0")


def test_a_container_still_removing_after_stop_is_given_time_to_finish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact live failure: stop must not fail on a container mid-removal."""

    _fast_settle(monkeypatch)
    daemon = SlowTeardownDocker(removing_inspections=3, images={"registry.example/easydel:latest"})
    executor = _executor(tmp_path, daemon)
    remote_id = executor.submit(_spec(), _context())

    executor.stop(remote_id, _context())

    assert executor.status(remote_id).status is ExecutionWorkloadStatus.STOPPED


def test_a_container_that_never_stops_is_still_reported_as_a_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The settle window must not turn a genuinely stuck container into a success."""

    monkeypatch.setenv("EYVAN_CONTAINER_STOP_SETTLE_S", "0")
    monkeypatch.setenv("EYVAN_CONTAINER_STOP_POLL_S", "0")
    daemon = SlowTeardownDocker(removing_inspections=10_000, images={"registry.example/easydel:latest"})
    executor = _executor(tmp_path, daemon)
    remote_id = executor.submit(_spec(), _context())

    with pytest.raises(RuntimeError, match="still active"):
        executor.stop(remote_id, _context())


def test_stop_is_retryable_after_an_earlier_stop_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The defect that made the leak permanent, isolated from what triggered it.

    A record marked INDETERMINATE by a failed stop must still be stoppable: the
    unit name identifies what to stop, and the status only says the runtime's
    state is unknown. Refusing here is what held a v5p Multislice for 40 minutes
    after its container had exited.
    """

    _fast_settle(monkeypatch)
    daemon = SlowTeardownDocker(removing_inspections=10_000, images={"registry.example/easydel:latest"})
    executor = _executor(tmp_path, daemon)
    remote_id = executor.submit(_spec(), _context())
    monkeypatch.setenv("EYVAN_CONTAINER_STOP_SETTLE_S", "0")
    with pytest.raises(RuntimeError):
        executor.stop(remote_id, _context())
    assert executor.status(remote_id).status is ExecutionWorkloadStatus.INDETERMINATE

    # A later stop must succeed once the container finishes shutting down.
    daemon._remaining_removing = 0
    monkeypatch.setenv("EYVAN_CONTAINER_STOP_SETTLE_S", "5")

    executor.stop(remote_id, _context())

    record = json.loads(executor._record_path(remote_id).read_text(encoding="utf-8"))
    assert ("systemctl", "stop", "--no-pager", record["unit_name"]) in daemon.commands
    assert executor.status(remote_id).status is ExecutionWorkloadStatus.STOPPED


def test_a_record_poisoned_by_an_older_build_is_still_recoverable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Allow successful stop evidence to resolve legacy INDETERMINATE records without a recoverable flag."""

    _fast_settle(monkeypatch)
    daemon = SlowTeardownDocker(removing_inspections=0, images={"registry.example/easydel:latest"})
    executor = _executor(tmp_path, daemon)
    remote_id = executor.submit(_spec(), _context())
    record_path = executor._record_path(remote_id)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["status"] = ExecutionWorkloadStatus.INDETERMINATE.value
    record["message"] = "systemd stop requested by Eyvan"
    record.pop("indeterminate_unrecoverable", None)
    record_path.write_text(json.dumps(record), encoding="utf-8")

    executor.stop(remote_id, _context())

    assert executor.status(remote_id).status is ExecutionWorkloadStatus.STOPPED


def test_a_record_without_a_unit_name_is_still_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The safety property the refusal actually protects must survive the fix.

    Not knowing *which* unit to stop is the unsafe case, because ``systemctl
    stop`` would then target nothing or something else. That must keep raising.
    """

    _fast_settle(monkeypatch)
    daemon = SlowTeardownDocker(removing_inspections=0, images={"registry.example/easydel:latest"})
    executor = _executor(tmp_path, daemon)
    remote_id = executor.submit(_spec(), _context())
    record_path = executor._record_path(remote_id)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["unit_name"] = ""
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(RuntimeError, match="ambiguous systemd execution"):
        executor.stop(remote_id, _context())


def test_a_terminal_workload_stops_without_touching_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Idempotent cancel must stay free: a finished workload needs no stop at all."""

    _fast_settle(monkeypatch)
    daemon = SlowTeardownDocker(removing_inspections=0, images={"registry.example/easydel:latest"})
    executor = _executor(tmp_path, daemon)
    remote_id = executor.submit(_spec(), _context())
    executor.stop(remote_id, _context())
    before = len(daemon.commands)

    executor.stop(remote_id, _context())

    assert not [command for command in daemon.commands[before:] if command[:2] == ("docker", "stop")]


def test_the_executor_still_stops_a_container_that_exits_promptly(tmp_path: Path) -> None:
    """The common path must not depend on the settle window being configured."""

    daemon = FakeDockerSystemd(images={"registry.example/easydel:latest"})
    executor = _executor(tmp_path, daemon)
    remote_id = executor.submit(_spec(), _context())

    executor.stop(remote_id, _context())

    assert executor.status(remote_id).status is ExecutionWorkloadStatus.STOPPED
    assert isinstance(_record(executor, remote_id)["unit_name"], str)


def test_a_malformed_settle_override_falls_back_to_the_default() -> None:
    """A bad env value must not disable the window or crash the stop path."""

    from eyvan.worker.runtime.container import _CONTAINER_STOP_SETTLE_S, _env_seconds

    assert _env_seconds("EYVAN_NONEXISTENT_SETTLE", _CONTAINER_STOP_SETTLE_S) == _CONTAINER_STOP_SETTLE_S


def test_a_negative_settle_override_falls_back_to_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A negative window would make the poll loop meaningless."""

    from eyvan.worker.runtime.container import _env_seconds

    monkeypatch.setenv("EYVAN_CONTAINER_STOP_SETTLE_S", "-5")

    assert _env_seconds("EYVAN_CONTAINER_STOP_SETTLE_S", 30.0) == 30.0


def test_isinstance_guard_keeps_the_executor_type_stable(tmp_path: Path) -> None:
    """Guard the harness itself: the fake must still drive the real executor."""

    daemon = SlowTeardownDocker(removing_inspections=0, images={"registry.example/easydel:latest"})

    assert isinstance(_executor(tmp_path, daemon), DockerContainerExecutor)
