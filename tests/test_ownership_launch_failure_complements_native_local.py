# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Ownership/launch-failure complements to native-local descendant regressions."""

from __future__ import annotations

import json
import os
import shlex
import signal
import sys
import time
from pathlib import Path

import pytest
from test_process_native_local_descendant_retirement import _alive, _spec, _wait_status

from eyvan.capacity.drivers.base import ExecutionWorkloadStatus, MutationContext
from eyvan.worker.runtime import LocalProcessExecutor, local_process, local_worker


def _worker_record(tmp_path: Path, command: str) -> tuple[Path, Path, Path]:
    record = tmp_path / "record.json"
    log = tmp_path / "log"
    owner = tmp_path / "owner.lock"
    record.write_text(
        json.dumps(
            {
                "worker_pid": os.getpid(),
                "worker_token": "test-worker-token",
                "entrypoint": command,
                "environment": {},
                "status": "PENDING",
                "stop_requested": False,
            }
        )
    )
    return record, log, owner


def _run_worker(paths: tuple[Path, Path, Path]) -> int:
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        return local_worker._run(*paths, "test-worker-token")
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def test_unpublished_command_group_never_executes_user_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executed = tmp_path / "executed"
    paths = _worker_record(
        tmp_path, shlex.join((sys.executable, "-c", f"from pathlib import Path; Path({str(executed)!r}).touch()"))
    )
    mutate = local_worker._mutate_record

    def refuse_publication(path, mutation):
        if mutation.__name__ == "command_owned":
            raise OSError("injected ownership publication failure")
        return mutate(path, mutation)

    monkeypatch.setattr(local_worker, "_mutate_record", refuse_publication)
    assert _run_worker(paths) == 1
    record = json.loads(paths[0].read_text())
    assert record["status"] == "FAILED"
    assert record["group_quiescent"] is True
    assert "publication failure" in record["message"]
    assert not executed.exists(), "command executed before its cleanup authority was persisted"


def test_terminal_write_failure_does_not_signal_an_already_reaped_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _worker_record(tmp_path, "exit 0")
    mutate = local_worker._mutate_record
    quiesce = local_worker._quiesce_process_group
    cleaned = []

    def fail_terminal_once(path, mutation):
        if mutation.__name__ == "finished":
            raise OSError("injected terminal persistence failure")
        return mutate(path, mutation)

    def record_cleanup(group):
        cleaned.append(group)
        return quiesce(group)

    monkeypatch.setattr(local_worker, "_mutate_record", fail_terminal_once)
    monkeypatch.setattr(local_worker, "_quiesce_process_group", record_cleanup)
    assert _run_worker(paths) == 1
    record = json.loads(paths[0].read_text())
    assert record["status"] == "FAILED"
    assert record["group_quiescent"] is True
    assert cleaned == [record["command_pid"]], "cleanup reused the group ID after its leader was reaped"


def test_group_inspection_failure_never_reports_quiescence(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(*_args, **_kwargs):
        raise OSError("injected process inspection outage")

    monkeypatch.setattr(local_process.subprocess, "run", unavailable)
    with pytest.raises(OSError, match="inspection outage"):
        local_process._quiesce_process_group(123456)


def test_legacy_terminal_record_cannot_admit_a_successor(tmp_path: Path) -> None:
    executor = LocalProcessExecutor(tmp_path / "state")
    record_path, _log, _owner = executor._paths("attempt-1")
    record_path.write_text(
        json.dumps({"status": "SUCCEEDED", "generation": 1, "fencing_token": "fence-1", "message": "legacy"})
    )
    fence = executor._workload_fence_path("writer-114-logical-workload")
    original_fence = {"remote_id": "attempt-1", "generation": 1, "fencing_token": "fence-1"}
    fence.write_text(json.dumps(original_fence))
    with pytest.raises(RuntimeError, match="cannot safely retire"):
        executor.submit(
            _spec("exit 0", "attempt-2"),
            MutationContext(idempotency_key="attempt-2", generation=2, fencing_token="fence-2"),
        )
    assert json.loads(record_path.read_text())["status"] == "INDETERMINATE"
    assert json.loads(fence.read_text()) == original_fence
    assert not executor._paths("attempt-2")[0].exists()


@pytest.mark.parametrize("leader_exited", [False, True])
def test_worker_crash_cleans_separate_term_ignoring_command_group(tmp_path: Path, leader_exited: bool) -> None:
    ready = tmp_path / "ready"
    sleeper = tmp_path / "owned-child.py"
    sleeper.write_text(
        "import os, signal, time\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(ready)!r}).write_text(str(os.getpid()))\ntime.sleep(60)\n"
    )
    executor = LocalProcessExecutor(tmp_path / "state")
    context = MutationContext(idempotency_key="attempt-1", generation=1, fencing_token="fence-1")
    remote_id = executor.submit(_spec(shlex.join((sys.executable, str(sleeper))) + " & wait", "attempt-1"), context)
    record_path = executor._paths(remote_id)[0]
    pid = None
    marker = None
    try:
        deadline = time.monotonic() + 8
        while (not ready.exists() or not ready.read_text()) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists() and ready.read_text(), executor.logs(remote_id)
        pid = int(ready.read_text())
        marker = local_process._process_start_marker(pid)
        record = json.loads(record_path.read_text())
        assert _alive(pid)
        assert os.getpgid(pid) == record["command_pid"] != record["worker_pid"]
        if leader_exited:
            # Freeze the owner before removing the shell so it cannot perform
            # cleanup itself. The retained direct worker still pins the session
            # identity even once the orphaned shell leader is reaped by init.
            os.kill(record["worker_pid"], signal.SIGSTOP)
            os.kill(record["command_pid"], signal.SIGKILL)
        os.kill(record["worker_pid"], signal.SIGKILL)
        if leader_exited:
            deadline = time.monotonic() + 3
            while local_process._process_start_marker(record["command_pid"]) is not None and time.monotonic() < deadline:
                time.sleep(0.02)
            assert local_process._process_start_marker(record["command_pid"]) is None
        _wait_status(executor, remote_id, ExecutionWorkloadStatus.LOST)
        assert not _alive(pid)
        assert json.loads(record_path.read_text())["group_quiescent"] is True
        assert remote_id not in executor._worker_processes
    finally:
        if pid is not None and marker is not None and local_process._process_start_marker(pid) == marker and _alive(pid):
            os.kill(pid, signal.SIGKILL)
        if remote_id in executor._worker_processes:
            record = json.loads(record_path.read_text())
            worker = executor._worker_processes[remote_id]
            if not local_process._unreaped_process_is_running(worker.pid):
                executor.status(remote_id)
            else:
                executor.retire(remote_id, context)


def test_new_worker_stop_does_not_signal_its_inspection_subprocesses(monkeypatch: pytest.MonkeyPatch) -> None:
    signals = []
    monkeypatch.setattr(local_process, "_tracked_worker_process_group", lambda _record: 123456)
    monkeypatch.setattr(local_process.os, "kill", lambda pid, sig: signals.append(("pid", pid, sig)))
    monkeypatch.setattr(local_process.os, "killpg", lambda pid, sig: signals.append(("group", pid, sig)))
    assert local_process._signal_tracked_worker({"command_pid": None}, signal.SIGTERM) is True
    assert signals == [("pid", 123456, signal.SIGTERM)]


pytestmark = pytest.mark.slow
