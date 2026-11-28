# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Retiring a local execution whose state has become ambiguous.

``LocalProcessExecutor.retire`` ends with a SIGKILL to the tracked process
group: the deliberate way to force a workload of unknown state to terminal so a
successor may start. The poll loop above it raised on INDETERMINATE, which
returned before that SIGKILL ever ran -- the recovery was guarded out by exactly
the condition it was written to recover from.

An INDETERMINATE record is not re-derived by ``status``, so continuing to poll
could not have helped either. The loop now stops polling and falls through to
the SIGKILL, then still reports the ambiguity.
"""

from __future__ import annotations

import json
import signal
import time
from pathlib import Path

import pytest
from test_local_runtime import _command, _context, _spec

from eyvan.capacity.drivers.base import ExecutionWorkloadObservation, ExecutionWorkloadStatus
from eyvan.worker.runtime.local_process import LocalProcessExecutor


def _record_path(executor: LocalProcessExecutor, remote_id: str) -> Path:
    record_path, _log_path, _owner_path = executor._paths(remote_id)
    return record_path


def test_an_ambiguous_local_execution_is_signalled_before_retirement_gives_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The SIGKILL must run. Before the fix, the early raise skipped it."""

    monkeypatch.setenv("EYVAN_SUPERSEDE_WAIT_TIMEOUT_S", "5")
    monkeypatch.setenv("EYVAN_SUPERSEDE_POLL_S", "0.01")
    executor = LocalProcessExecutor(tmp_path)
    remote_id = executor.submit(_spec(_command("import time; time.sleep(30)")), _context())

    record_path = _record_path(executor, remote_id)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["status"] = ExecutionWorkloadStatus.INDETERMINATE.value
    record["message"] = "native-local worker ownership is ambiguous; workload was not signalled"
    record_path.write_text(json.dumps(record), encoding="utf-8")

    # Stub the observation as well, as the test below does. The persisted
    # INDETERMINATE alone raced the real worker process starting up: on a
    # loaded runner (CI, Python 3.11 under coverage) the worker rewrote the
    # record and honoured stop_requested, so retire saw a terminal status and
    # returned without ever reaching the ambiguity path this test is about.
    monkeypatch.setattr(
        LocalProcessExecutor,
        "status",
        lambda self, rid: ExecutionWorkloadObservation(
            remote_id=rid,
            status=ExecutionWorkloadStatus.INDETERMINATE,
            message="native-local worker ownership is ambiguous; workload was not signalled",
        ),
    )

    signalled: list[int] = []
    monkeypatch.setattr(
        "eyvan.worker.runtime.local_process._signal_tracked_worker",
        lambda record, signal_number: signalled.append(signal_number) or True,
    )

    with pytest.raises(RuntimeError, match="before its timeout"):
        executor.retire(remote_id, _context())

    assert signal.SIGKILL in signalled, "retirement gave up without forcing the workload terminal"


def test_an_ambiguous_retirement_does_not_burn_the_whole_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Falling through must not mean polling a status that cannot change.

    ``status`` returns the persisted INDETERMINATE verbatim, so every extra poll
    is guaranteed to read the same answer. Waiting out a 300s default before
    recovering would make supersession unusable.

    The observation is stubbed rather than persisted: an earlier version wrote
    the record and launched a real process, so whether ``status`` re-derived the
    value before the loop read it depended on machine load. That test passed
    alone and failed in the full suite -- the signature this repository treats
    as a real defect rather than flake, and here it was the test's own timing
    dependence.
    """

    monkeypatch.setenv("EYVAN_SUPERSEDE_WAIT_TIMEOUT_S", "3600")
    monkeypatch.setenv("EYVAN_SUPERSEDE_POLL_S", "0.01")
    executor = LocalProcessExecutor(tmp_path)
    remote_id = executor.submit(_spec(_command("import time; time.sleep(30)")), _context())

    monkeypatch.setattr(
        LocalProcessExecutor,
        "status",
        lambda self, rid: ExecutionWorkloadObservation(
            remote_id=rid,
            status=ExecutionWorkloadStatus.INDETERMINATE,
            message="native-local worker ownership is ambiguous; workload was not signalled",
        ),
    )
    monkeypatch.setattr(
        "eyvan.worker.runtime.local_process._signal_tracked_worker",
        lambda record, signal_number: True,
    )

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="before its timeout"):
        executor.retire(remote_id, _context())

    # A one-hour deadline: returning at all proves the loop broke out.
    assert time.monotonic() - started < 60


def test_a_live_local_execution_is_still_retired_normally(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The ordinary supersession path must be untouched by the fix."""

    monkeypatch.setenv("EYVAN_SUPERSEDE_WAIT_TIMEOUT_S", "30")
    monkeypatch.setenv("EYVAN_SUPERSEDE_POLL_S", "0.01")
    executor = LocalProcessExecutor(tmp_path)
    remote_id = executor.submit(_spec(_command("import time; time.sleep(30)")), _context())

    executor.retire(remote_id, _context())

    assert executor.status(remote_id).status in {
        ExecutionWorkloadStatus.STOPPED,
        ExecutionWorkloadStatus.FAILED,
        ExecutionWorkloadStatus.SUCCEEDED,
    }
