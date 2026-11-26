# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Which unknowns a later observation can resolve, and which it cannot.

INDETERMINATE is latched: once a record carries it, ``status`` returns it
verbatim rather than re-deriving, and the operations that could resolve it
refuse to run. That is correct for a workload nothing can ever learn about
again, and catastrophic for one that was merely unreadable for a moment -- it
turns a transient error into an accelerator nobody can reclaim.

The line is between *the thing is gone* and *I could not look just now*:

  unrecoverable   the record cannot name its unit; the record will not parse;
                  the unit disappeared before recording a terminal state
  recoverable     the unit could not be inspected; the cgroup could not be
                  read; the container could not be inspected; a stop failed

Getting this backwards is not hypothetical. Defaulting everything to
recoverable made a vanished unit poll for its full 300s retirement deadline
before failing, which alone took the suite from 25s to five and a half minutes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_systemd_transient import FakeSystemd, _context, _record, _spec

from eyvan.capacity.drivers.base import ExecutionWorkloadStatus
from eyvan.worker.runtime.systemd_transient import (
    _UNRECOVERABLE_INDETERMINATE_FIELD,
    SystemdCommandResult,
    SystemdTransientServiceExecutor,
    _is_recoverable_indeterminate,
)


def _executor(tmp_path: Path, systemd: FakeSystemd) -> SystemdTransientServiceExecutor:
    return SystemdTransientServiceExecutor(
        tmp_path,
        command_runner=systemd,
        boot_id_reader=lambda: "boot-a",
        use_exit_type_cgroup=True,
    )


def test_a_vanished_unit_is_not_recoverable(tmp_path: Path) -> None:
    """Nothing can later say whether that workload succeeded or was killed."""

    systemd = FakeSystemd()
    executor = _executor(tmp_path, systemd)
    remote_id = executor.submit(_spec(), _context())
    systemd.remove(_record(executor, remote_id)["unit_name"])

    assert executor.status(remote_id).status is ExecutionWorkloadStatus.INDETERMINATE
    assert _record(executor, remote_id)[_UNRECOVERABLE_INDETERMINATE_FIELD] is True


class UnreachableManagerSystemd(FakeSystemd):
    """A systemd whose manager cannot be reached, with its units still intact.

    Distinct from ``FakeSystemd.remove``: there the unit is gone, here it may be
    running perfectly and only the query failed. That is the whole distinction
    under test.
    """

    def __call__(self, command: tuple[str, ...]):  # type: ignore[no-untyped-def]
        if command[:2] == ("systemctl", "show"):
            self.commands.append(command)
            return SystemdCommandResult(returncode=1, stderr="Failed to connect to bus: No such file or directory")
        return super().__call__(command)


def test_a_unit_that_could_not_be_inspected_stays_recoverable(tmp_path: Path) -> None:
    """The unit may be perfectly alive; only the inspection failed."""

    systemd = UnreachableManagerSystemd()
    executor = _executor(tmp_path, systemd)
    remote_id = executor.submit(_spec(), _context())

    assert executor.status(remote_id).status is ExecutionWorkloadStatus.INDETERMINATE
    assert "cannot observe systemd service" in executor.status(remote_id).message
    assert _UNRECOVERABLE_INDETERMINATE_FIELD not in _record(executor, remote_id)


def test_a_record_with_no_unit_name_is_not_recoverable(tmp_path: Path) -> None:
    """Without a unit name there is nothing to stop, inspect, or wait for."""

    systemd = FakeSystemd()
    executor = _executor(tmp_path, systemd)
    remote_id = executor.submit(_spec(), _context())
    record_path = executor._record_path(remote_id)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["unit_name"] = ""
    record_path.write_text(json.dumps(record), encoding="utf-8")

    assert executor.status(remote_id).status is ExecutionWorkloadStatus.INDETERMINATE
    assert _record(executor, remote_id)[_UNRECOVERABLE_INDETERMINATE_FIELD] is True


@pytest.mark.parametrize(
    ("record", "recoverable"),
    [
        ({}, True),
        ({"message": "cannot observe systemd service: transport failed"}, True),
        ({"message": "failed to stop native runtime: docker was busy"}, True),
        ({_UNRECOVERABLE_INDETERMINATE_FIELD: True}, False),
        ({_UNRECOVERABLE_INDETERMINATE_FIELD: False}, True),
    ],
)
def test_the_predicate_reads_the_stored_flag_not_the_message(record: dict, recoverable: bool) -> None:
    """``stop`` rewrites ``message`` while leaving the status in place.

    A text comparison would therefore stop matching at exactly the moment it is
    needed, which is why recoverability is a stored field.
    """

    assert _is_recoverable_indeterminate(record) is recoverable


def test_a_record_from_a_build_without_the_flag_reads_as_recoverable() -> None:
    """The safe direction on upgrade: recovery re-derives from reality.

    An operator upgrading into this code has records written before the field
    existed. Treating those as unrecoverable would strand exactly the workloads
    the fix was written to free.
    """

    assert _is_recoverable_indeterminate({"status": "indeterminate", "message": "anything"}) is True


def test_leaving_indeterminate_clears_the_flag(tmp_path: Path) -> None:
    """A stale unrecoverable marker must not outlive the state it described."""

    executor = _executor(tmp_path, FakeSystemd())
    record: dict = {}
    executor._set_status(record, ExecutionWorkloadStatus.INDETERMINATE, "unknown")
    record[_UNRECOVERABLE_INDETERMINATE_FIELD] = True

    executor._set_status(record, ExecutionWorkloadStatus.STOPPED, "stopped")

    assert _UNRECOVERABLE_INDETERMINATE_FIELD not in record
