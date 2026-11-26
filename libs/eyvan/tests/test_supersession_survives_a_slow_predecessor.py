# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Retiring the predecessor when a logical workload is superseded.

``retire`` is what supersession calls on the previous generation before the next
one may run, so a retirement that cannot complete does not merely leak a unit --
it blocks the replacement from ever starting.

It had the same two defects the live 2x v5p-16 exposed in ``stop``, and one more
of its own:

* an INDETERMINATE record was refused outright, and INDETERMINATE latches, so a
  single earlier stop failure made the workload permanently un-supersedable;
* the refusal happened *before* any stop was attempted, so the one operation
  that could have resolved the ambiguity was never run;
* a stop that raised aborted the whole retirement on the first attempt, even
  though the loop already has a deadline and a poll interval -- a container that
  simply needed a few more seconds failed the supersession.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from test_container_executor import _context, _executor, _spec
from test_stop_survives_a_slow_container_teardown import SlowTeardownDocker

from eyvan.capacity.drivers.base import ExecutionWorkloadStatus


@pytest.fixture(autouse=True)
def _fast_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the bounded retirement loop off the wall clock."""

    monkeypatch.setenv("EYVAN_SUPERSEDE_WAIT_TIMEOUT_S", "5")
    monkeypatch.setenv("EYVAN_SUPERSEDE_POLL_S", "0.01")
    monkeypatch.setenv("EYVAN_CONTAINER_STOP_SETTLE_S", "0")
    monkeypatch.setenv("EYVAN_CONTAINER_STOP_POLL_S", "0")


class StubbornDocker(SlowTeardownDocker):
    """A container that needs more than one ``docker stop`` before it dies.

    Counting stops rather than inspections keeps the test deterministic: the
    interesting moment is *when retirement issues another stop*, not how many
    times something happened to be inspected on the way there.
    """

    def __init__(self, *, dies_after_stops: int, **kwargs: object) -> None:
        super().__init__(removing_inspections=10**9, **kwargs)  # type: ignore[arg-type]
        self._stops_remaining = dies_after_stops

    def __call__(self, command: tuple[str, ...]):  # type: ignore[no-untyped-def]
        if command[:2] == ("docker", "stop"):
            self._stops_remaining -= 1
            if self._stops_remaining <= 0:
                self._remaining_removing = 0
        return super().__call__(command)


def test_a_predecessor_marked_indeterminate_by_a_failed_stop_can_still_be_retired(tmp_path: Path) -> None:
    """The permanent block: one transient stop failure, then no generation may follow.

    The record is INDETERMINATE *and the container is still active* when
    retirement starts, which is the state the refusal used to reject outright --
    before attempting the one operation that resolves it.
    """

    daemon = StubbornDocker(dies_after_stops=2, images={"registry.example/easydel:latest"})
    executor = _executor(tmp_path, daemon)
    remote_id = executor.submit(_spec(), _context())
    with pytest.raises(RuntimeError):
        executor.stop(remote_id, _context())
    assert executor.status(remote_id).status is ExecutionWorkloadStatus.INDETERMINATE

    executor.retire(remote_id, _context())

    assert executor.status(remote_id).status in {
        ExecutionWorkloadStatus.STOPPED,
        ExecutionWorkloadStatus.FAILED,
        ExecutionWorkloadStatus.SUCCEEDED,
    }


def test_a_stop_that_fails_once_does_not_abort_the_retirement(tmp_path: Path) -> None:
    """The loop has a deadline; one slow teardown must not fail the supersession."""

    daemon = SlowTeardownDocker(removing_inspections=2, images={"registry.example/easydel:latest"})
    executor = _executor(tmp_path, daemon)
    remote_id = executor.submit(_spec(), _context())

    executor.retire(remote_id, _context())

    assert executor.status(remote_id).status is not ExecutionWorkloadStatus.RUNNING


def test_a_predecessor_that_never_dies_still_fails_the_retirement(tmp_path: Path) -> None:
    """Tolerating failures must not become ignoring them.

    A workload that is genuinely still running has to fail the supersession --
    letting it through would run two generations on one accelerator.
    """

    daemon = SlowTeardownDocker(removing_inspections=10**9, images={"registry.example/easydel:latest"})
    executor = _executor(tmp_path, daemon)
    remote_id = executor.submit(_spec(), _context())

    # Specifically the deadline error: retirement kept trying and ran out of
    # time, rather than giving up on the first failed stop.
    with pytest.raises(RuntimeError, match="before its timeout"):
        executor.retire(remote_id, _context())


def test_an_unrecoverably_ambiguous_record_is_still_refused(tmp_path: Path) -> None:
    """The safety property the refusal protects has to survive the fix.

    An INDETERMINATE that no later observation can resolve -- here a record with
    no usable unit name -- must keep raising rather than being retried forever.
    """

    daemon = SlowTeardownDocker(removing_inspections=0, images={"registry.example/easydel:latest"})
    executor = _executor(tmp_path, daemon)
    remote_id = executor.submit(_spec(), _context())
    record_path = executor._record_path(remote_id)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["status"] = ExecutionWorkloadStatus.INDETERMINATE.value
    record["message"] = "systemd execution record has no unit name"
    record["indeterminate_unrecoverable"] = True
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(RuntimeError, match="ambiguous systemd execution"):
        executor.retire(remote_id, _context())


def test_a_terminal_predecessor_retires_without_attempting_a_stop(tmp_path: Path) -> None:
    """The common supersession path stays cheap: nothing to stop, nothing stopped."""

    daemon = SlowTeardownDocker(removing_inspections=0, images={"registry.example/easydel:latest"})
    executor = _executor(tmp_path, daemon)
    remote_id = executor.submit(_spec(), _context())
    executor.stop(remote_id, _context())
    before = len(daemon.commands)

    executor.retire(remote_id, _context())

    assert not [command for command in daemon.commands[before:] if command[:2] == ("docker", "stop")]


#: Integration-scale (see the "slow" marker in pyproject.toml): excluded from
#: the fast iteration subset, `pytest tests -m "not slow"`.
pytestmark = pytest.mark.slow
