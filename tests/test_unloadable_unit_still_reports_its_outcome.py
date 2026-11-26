# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A unit systemd would not load fully still says what happened to the workload.

Found on a live 2x v5p-16 during a preemption. The victim's unit was::

    eyvan-wl_...-a2-g2.service  bad-setting  failed  failed

with no container running. ``LoadState`` was ``bad-setting`` rather than
``loaded``, so ``_show_unit`` reported the whole observation as *unavailable* --
discarding the ``ActiveState=failed`` sitting in the very same properties dict.

That is not a failure to observe. The unit is dead and systemd said so. Calling
it unreadable made the record INDETERMINATE, the stop phase could never verify,
the preemption command failed after eight attempts, and the victim stayed in
CHECKPOINTING while the production job that had preempted it never received the
capacity it was promised.
"""

from __future__ import annotations

import pytest
from test_systemd_transient import FakeSystemd, _context, _record, _spec

from eyvan.capacity.drivers.base import ExecutionWorkloadStatus
from eyvan.worker.runtime.systemd_transient import SystemdCommandResult, SystemdTransientServiceExecutor


class _LoadState(FakeSystemd):
    """A systemd whose unit reports a given LoadState and ActiveState."""

    def __init__(self, *, load_state: str, active_state: str, sub_state: str = "failed") -> None:
        super().__init__()
        self._load_state = load_state
        self._active_state = active_state
        self._sub_state = sub_state

    def __call__(self, command: tuple[str, ...]) -> SystemdCommandResult:
        if command[:2] == ("systemctl", "show"):
            self.commands.append(command)
            properties = {
                "LoadState": self._load_state,
                "ActiveState": self._active_state,
                "SubState": self._sub_state,
                "Result": "exit-code",
                "ExecMainCode": "exited",
                "ExecMainStatus": "1",
                "ControlGroup": "",
            }
            return SystemdCommandResult(
                returncode=0,
                stdout="".join(f"{name}={value}\n" for name, value in properties.items()),
            )
        return super().__call__(command)


def _executor(tmp_path, systemd) -> SystemdTransientServiceExecutor:
    return SystemdTransientServiceExecutor(
        tmp_path,
        command_runner=systemd,
        boot_id_reader=lambda: "boot-a",
        use_exit_type_cgroup=True,
    )


@pytest.mark.parametrize("load_state", ["bad-setting", "error", "masked"])
def test_a_dead_unit_is_terminal_even_when_it_would_not_load(tmp_path, load_state: str) -> None:
    """The live failure: ActiveState=failed is the answer, whatever LoadState says."""

    systemd = _LoadState(load_state=load_state, active_state="failed")
    executor = _executor(tmp_path, systemd)
    remote_id = executor.submit(_spec(), _context())

    observation = executor.status(remote_id)

    assert observation.status is ExecutionWorkloadStatus.FAILED
    assert observation.status is not ExecutionWorkloadStatus.INDETERMINATE


def test_an_inactive_unloadable_unit_is_also_terminal(tmp_path) -> None:
    """A stopped workload reports ``inactive``; that settles it too."""

    systemd = _LoadState(load_state="bad-setting", active_state="inactive", sub_state="dead")
    executor = _executor(tmp_path, systemd)
    remote_id = executor.submit(_spec(), _context())

    assert executor.status(remote_id).status in {
        ExecutionWorkloadStatus.STOPPED,
        ExecutionWorkloadStatus.FAILED,
        ExecutionWorkloadStatus.SUCCEEDED,
    }


def test_a_running_unit_that_would_not_load_is_still_unreadable(tmp_path) -> None:
    """The safety property must survive: ``active`` does not settle anything.

    A unit systemd could not load but which reports itself active is genuinely
    ambiguous -- that is the case the ``unavailable`` branch exists for, and
    treating it as terminal would let Eyvan declare a live workload finished.
    """

    systemd = _LoadState(load_state="bad-setting", active_state="active", sub_state="running")
    executor = _executor(tmp_path, systemd)
    remote_id = executor.submit(_spec(), _context())

    assert executor.status(remote_id).status is ExecutionWorkloadStatus.INDETERMINATE


def test_an_absent_unit_is_still_missing_not_terminal(tmp_path) -> None:
    """``not-found`` keeps its own meaning; this fix must not absorb it."""

    systemd = FakeSystemd()
    executor = _executor(tmp_path, systemd)
    remote_id = executor.submit(_spec(), _context())
    systemd.remove(_record(executor, remote_id)["unit_name"])

    assert executor.status(remote_id).status is ExecutionWorkloadStatus.INDETERMINATE


def test_a_failed_systemctl_call_is_still_unreadable(tmp_path) -> None:
    """A transport failure returns no properties at all, so nothing is settled."""

    class _Broken(FakeSystemd):
        def __call__(self, command: tuple[str, ...]) -> SystemdCommandResult:
            if command[:2] == ("systemctl", "show"):
                self.commands.append(command)
                return SystemdCommandResult(returncode=1, stderr="Failed to connect to bus")
            return super().__call__(command)

    executor = _executor(tmp_path, _Broken())
    remote_id = executor.submit(_spec(), _context())

    assert executor.status(remote_id).status is ExecutionWorkloadStatus.INDETERMINATE
