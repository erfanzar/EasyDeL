# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Real-process regressions for native-local descendant retirement."""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from eyvan._internal.json_types import JsonObject
from eyvan.capacity.drivers.base import ExecutionWorkloadStatus, MutationContext
from eyvan.worker.runtime import LocalProcessExecutor, local_process


def _alive(pid: int) -> bool:
    observation = subprocess.run(
        ("ps", "-o", "stat=", "-p", str(pid)), capture_output=True, text=True, check=False, timeout=2
    )
    state = observation.stdout.strip()
    return observation.returncode == 0 and bool(state) and not state.startswith("Z")


def _wait_status(executor: LocalProcessExecutor, remote_id: str, expected: ExecutionWorkloadStatus) -> None:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if executor.status(remote_id).status is expected:
            return
        time.sleep(0.02)
    pytest.fail(f"expected {expected.value}, got {executor.status(remote_id)}; logs={executor.logs(remote_id)!r}")


def _spec(entrypoint: str, attempt: str) -> JsonObject:
    return {
        "entrypoint": entrypoint,
        "submission_id": attempt,
        "environment": {},
        "gang": {"replicas": 1, "per_replica": {"cpu": 1}, "required_topology": {}},
        "metadata": {"eyvan_workload_id": "writer-114-logical-workload"},
    }


@pytest.mark.parametrize("boundary", ["terminal", "retire", "supersede", "stop"])
def test_native_local_terminal_boundary_requires_descendant_quiescence(tmp_path: Path, boundary: str) -> None:
    """A terminal shell must not release an ordinary live same-group child."""

    ready = tmp_path / "ready.pid"
    release = tmp_path / "release-shell"
    sleeper = tmp_path / "owned_sleeper.py"
    sleeper.write_text(
        "import os, pathlib, signal, time\n"
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if boundary == "stop" else "")
        + f"pathlib.Path({str(ready)!r}).write_text(str(os.getpid()))\n"
        + "time.sleep(60)\n",
        encoding="utf-8",
    )
    entrypoint = (
        f"{shlex.join((sys.executable, str(sleeper)))} & "
        f"while [ ! -s {shlex.quote(str(ready))} ]; do sleep 0.02; done; "
        f"while [ ! -e {shlex.quote(str(release))} ]; do sleep 0.02; done" + ("; wait" if boundary == "stop" else "")
    )
    executor = LocalProcessExecutor(tmp_path / "state")
    context = MutationContext(idempotency_key="attempt-1", generation=1, fencing_token="fence-1")
    pid = None
    marker = None
    remote_id = executor.submit(_spec(entrypoint, "attempt-1"), context)
    record_path, _log_path, _owner_path = executor._paths(remote_id)
    try:
        deadline = time.monotonic() + 8
        while (not ready.exists() or ready.stat().st_size == 0) and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists() and ready.stat().st_size > 0, executor.logs(remote_id)
        pid = int(ready.read_text())
        marker = local_process._process_start_marker(pid)
        assert marker is not None, "owned descendant did not retain an observable process identity"
        assert _alive(pid), "owned descendant did not reach its finite sleep"
        record = json.loads(record_path.read_text())
        assert os.getpgid(pid) == record.get("command_pid", record["worker_pid"]), (
            "regression must not escape the owned workload process group"
        )
        assert os.getsid(pid) == record["worker_pid"], "regression must remain in the owned worker session"
        release.touch()

        if boundary == "stop":
            executor.stop(remote_id, context)
            _wait_status(executor, remote_id, ExecutionWorkloadStatus.STOPPED)
        else:
            _wait_status(executor, remote_id, ExecutionWorkloadStatus.SUCCEEDED)
            if boundary == "retire":
                executor.retire(remote_id, context)
            elif boundary == "supersede":
                successor_observation = tmp_path / "successor-observation.txt"
                successor_source = (
                    "import pathlib, subprocess; "
                    f"r = subprocess.run(('ps', '-o', 'stat=', '-p', '{pid}'), "
                    "capture_output=True, text=True, check=False); "
                    "state = r.stdout.strip(); "
                    "alive = r.returncode == 0 and bool(state) and not state.startswith('Z'); "
                    f"pathlib.Path({str(successor_observation)!r}).write_text('alive' if alive else 'gone')"
                )
                successor = executor.submit(
                    _spec(shlex.join((sys.executable, "-c", successor_source)), "attempt-2"),
                    MutationContext(idempotency_key="attempt-2", generation=2, fencing_token="fence-2"),
                )
                _wait_status(executor, successor, ExecutionWorkloadStatus.SUCCEEDED)
                assert successor_observation.read_text() == "gone", (
                    "successor executed while predecessor child was alive"
                )
        assert not _alive(pid), f"{boundary} reported completion while owned descendant {pid} remained alive"
    finally:
        # Only signal identities created by this test. A terminal worker may
        # already have been reaped, so a raw persisted process-group ID is not
        # sufficient authority for cleanup.
        if record_path.exists():
            local_process._terminate_tracked_worker(json.loads(record_path.read_text()))
        if pid is not None and marker is not None and local_process._process_start_marker(pid) == marker:
            arguments = subprocess.run(
                ("ps", "-ww", "-o", "command=", "-p", str(pid)),
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
            )
            if str(sleeper) in arguments.stdout and _alive(pid):
                os.kill(pid, signal.SIGKILL)
        if pid is not None:
            deadline = time.monotonic() + 3
            while _alive(pid) and time.monotonic() < deadline:
                time.sleep(0.02)
            assert not _alive(pid), "test-owned descendant cleanup did not quiesce"


pytestmark = pytest.mark.slow
