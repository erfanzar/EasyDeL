# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import stat
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from eyvan._internal.json_types import JsonObject
from eyvan.capacity.drivers.base import (
    CapacityStatus,
    CellObservation,
    ExecutionWorkloadStatus,
    MutationContext,
    StaleMutationError,
)
from eyvan.worker.runtime import ConfiguredExecutionResolver, LocalProcessExecutor, local_process, local_worker


def _context(*, key: str = "submit-1", generation: int = 1, token: str = "token-1") -> MutationContext:
    return MutationContext(idempotency_key=key, generation=generation, fencing_token=token)


def _command(source: str) -> str:
    return shlex.join((sys.executable, "-c", source))


def _spec(
    entrypoint: str,
    *,
    submission_id: str = "workload-1",
    environment: dict[str, str] | None = None,
) -> JsonObject:
    environment_document: JsonObject = dict(environment or {})
    document: JsonObject = {
        "entrypoint": entrypoint,
        "submission_id": submission_id,
        "environment": environment_document,
        "gang": {"replicas": 1, "per_replica": {"cpu": 1}, "required_topology": {}},
        "metadata": {"eyvan_workload_id": submission_id},
    }
    return document


def _wait_for(
    executor: LocalProcessExecutor,
    remote_id: str,
    expected: set[ExecutionWorkloadStatus],
    *,
    timeout: float = 8,
) -> ExecutionWorkloadStatus:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = executor.status(remote_id).status
        if status in expected:
            return status
        time.sleep(0.02)
    pytest.fail(
        f"execution {remote_id!r} did not reach {sorted(status.value for status in expected)}; "
        f"got {executor.status(remote_id)!r}"
    )


def _process_alive(pid: int) -> bool:
    result = subprocess.run(("ps", "-o", "stat=", "-p", str(pid)), capture_output=True, text=True, check=False)
    state = result.stdout.strip()
    return result.returncode == 0 and bool(state) and not state.startswith("Z")


def test_runtime_import_and_native_resolution_do_not_import_ray(tmp_path: Path) -> None:
    code = "\n".join(
        (
            "import sys",
            "from eyvan.capacity.drivers.base import CapacityStatus, CellObservation",
            "from eyvan.worker.runtime import ConfiguredExecutionResolver, LocalProcessExecutor",
            "assert 'eyvan.worker.runtime.ray_jobs' not in sys.modules",
            f"resolver = ConfiguredExecutionResolver('native-local', state_dir={str(tmp_path)!r})",
            "driver = resolver.resolve('cell-a', CellObservation('cell-a', CapacityStatus.HEALTHY, 1))",
            "assert isinstance(driver, LocalProcessExecutor)",
            "assert 'eyvan.worker.runtime.ray_jobs' not in sys.modules",
        )
    )
    subprocess.run((sys.executable, "-c", code), check=True)


def test_python_commands_use_the_local_executor_interpreter_without_activation(tmp_path: Path) -> None:
    executor = LocalProcessExecutor(state_dir=tmp_path)
    command = shlex.join(("python", "-c", f"import sys; assert sys.prefix == {sys.prefix!r}"))
    remote_id = executor.submit(_spec(command, environment={"PATH": "/usr/bin:/bin"}), _context())
    outcome = _wait_for(executor, remote_id, {ExecutionWorkloadStatus.SUCCEEDED, ExecutionWorkloadStatus.FAILED})
    assert outcome is ExecutionWorkloadStatus.SUCCEEDED, executor.logs(remote_id)


def test_detached_local_worker_startup_does_not_import_the_control_plane() -> None:
    """Keep a cold worker launch independent of API and provider imports.

    A worker that exits before acquiring its durable owner lock used to be
    reported as ``LOST``. Eager package initializers made that possible under
    load because a detached worker imported the SDK, API, and GCP stack before
    it could record ``RUNNING``.
    """

    code = "\n".join(
        (
            "import sys",
            "import eyvan.worker.runtime.local_worker",
            "forbidden = {",
            "    'eyvan.server.api',",
            "    'eyvan.capacity.drivers.gcp',",
            "    'eyvan.server.persistence',",
            "    'eyvan.worker.runtime.native_worker',",
            "    'eyvan.worker.runtime.ray_jobs',",
            "    'eyvan.worker.runtime.systemd_transient',",
            "    'eyvan.sdk',",
            "}",
            "loaded = forbidden & set(sys.modules)",
            "assert not loaded, sorted(loaded)",
        )
    )

    subprocess.run((sys.executable, "-c", code), check=True)


def test_execution_backend_aliases_canonicalize_to_stable_names(tmp_path: Path) -> None:
    resolver = ConfiguredExecutionResolver("native_local", state_dir=tmp_path)

    assert resolver.backend == "native-local"
    assert isinstance(
        resolver.resolve("cell-a", CellObservation("cell-a", CapacityStatus.HEALTHY, 1)),
        LocalProcessExecutor,
    )


def test_local_process_persists_success_logs_and_environment_across_executor_restart(tmp_path: Path) -> None:
    executor = LocalProcessExecutor(tmp_path)
    context = _context()
    remote_id = executor.submit(
        _spec(
            _command(
                "import os; "
                "print(os.environ['EXPERIMENT']); "
                "print(os.environ['EYVAN_REPLICA_INDEX'] + '/' + os.environ['EYVAN_REPLICA_COUNT'])"
            ),
            environment={"EXPERIMENT": "native"},
        ),
        context,
    )

    assert _wait_for(executor, remote_id, {ExecutionWorkloadStatus.SUCCEEDED}) is ExecutionWorkloadStatus.SUCCEEDED
    restarted = LocalProcessExecutor(tmp_path)

    assert restarted.status(remote_id).status is ExecutionWorkloadStatus.SUCCEEDED
    assert restarted.logs(remote_id).splitlines() == ["native", "0/1"]


def test_local_process_accepts_a_worker_token_that_begins_with_a_hyphen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(local_process.secrets, "token_urlsafe", lambda _size: "-leading-hyphen-token")
    executor = LocalProcessExecutor(tmp_path)

    remote_id = executor.submit(_spec(_command("print('started')")), _context())

    assert _wait_for(executor, remote_id, {ExecutionWorkloadStatus.SUCCEEDED}) is ExecutionWorkloadStatus.SUCCEEDED
    assert executor.logs(remote_id).splitlines() == ["started"]


def test_local_process_submit_is_idempotent_and_rejects_stale_fences(tmp_path: Path) -> None:
    executor = LocalProcessExecutor(tmp_path)
    competing_executor = LocalProcessExecutor(tmp_path)
    context = _context(generation=7, token="token-7")
    spec = _spec(_command("print('once')"))

    with ThreadPoolExecutor(max_workers=2) as pool:
        submissions = (
            pool.submit(executor.submit, spec, context),
            pool.submit(competing_executor.submit, spec, context),
        )
        assert [submission.result() for submission in submissions] == ["workload-1", "workload-1"]
    assert executor.submit(spec, context) == "workload-1"
    with pytest.raises(StaleMutationError, match="does not own"):
        executor.submit(spec, _context(generation=8, token="token-8"))
    with pytest.raises(StaleMutationError, match="fencing token"):
        executor.submit(spec, _context(generation=7, token="other"))

    assert _wait_for(executor, "workload-1", {ExecutionWorkloadStatus.SUCCEEDED}) is ExecutionWorkloadStatus.SUCCEEDED
    assert executor.logs("workload-1").splitlines() == ["once"]


@pytest.mark.parametrize(
    ("updates", "match"),
    [
        ({"gang": {"replicas": 2, "per_replica": {"cpu": 1}}}, "exactly one replica"),
        ({"gang": {"replicas": 1, "per_replica": {"cpu": 1}, "required_topology": {"rack": "a"}}}, "topology"),
        ({"runtime_env": {"working_dir": "/tmp"}}, "runtime_env field"),
    ],
)
def test_local_process_rejects_capabilities_it_does_not_enforce(
    tmp_path: Path,
    updates: JsonObject,
    match: str,
) -> None:
    executor = LocalProcessExecutor(tmp_path)
    spec = _spec(_command("pass"))
    spec.update(updates)

    with pytest.raises(ValueError, match=match):
        executor.submit(spec, _context())


def test_local_process_rejects_a_non_string_submission_id(tmp_path: Path) -> None:
    executor = LocalProcessExecutor(tmp_path)
    spec = _spec(_command("pass"))
    spec["submission_id"] = 123

    with pytest.raises(ValueError, match="submission ID must be a string"):
        executor.submit(spec, _context())


def test_local_process_reports_failure_and_absent_executions(tmp_path: Path) -> None:
    executor = LocalProcessExecutor(tmp_path)

    missing = executor.status("missing")
    assert missing.remote_id == "missing"
    assert missing.status is ExecutionWorkloadStatus.NOT_FOUND
    assert missing.message == "local execution is absent"
    assert executor.logs("missing") == ""
    executor.retire("missing", _context())

    executor.submit(_spec(_command("raise SystemExit(7)")), _context())
    assert _wait_for(executor, "workload-1", {ExecutionWorkloadStatus.FAILED}) is ExecutionWorkloadStatus.FAILED
    assert "code 7" in executor.status("workload-1").message


def test_local_process_preserves_an_explicit_indeterminate_status(tmp_path: Path) -> None:
    executor = LocalProcessExecutor(tmp_path)
    remote_id = "ambiguous-workload"
    record_path, _log_path, _owner_path = executor._paths(remote_id)
    record_path.write_text(
        json.dumps(
            {
                "remote_id": remote_id,
                "status": ExecutionWorkloadStatus.INDETERMINATE.value,
                "message": "worker ownership could not be proven",
                "created_at": "2020-01-01T00:00:00+00:00",
            }
        )
    )

    observation = executor.status(remote_id)

    assert observation.remote_id == remote_id
    assert observation.status is ExecutionWorkloadStatus.INDETERMINATE
    assert observation.message == "worker ownership could not be proven"


@pytest.mark.parametrize(
    ("record", "match"),
    [
        ({"message": "missing"}, "has no persisted status"),
        ({"status": None, "message": "null"}, "non-string persisted status"),
        ({"status": "MYSTERY", "message": "unsupported"}, "unsupported persisted status 'MYSTERY'"),
    ],
)
def test_local_process_rejects_malformed_persisted_status(
    tmp_path: Path,
    record: dict,
    match: str,
) -> None:
    executor = LocalProcessExecutor(tmp_path)
    remote_id = "malformed-workload"
    record_path, _log_path, _owner_path = executor._paths(remote_id)
    record_path.write_text(json.dumps({"remote_id": remote_id, **record}))

    with pytest.raises(RuntimeError, match=match):
        executor.status(remote_id)


def test_local_process_reports_a_worker_exit_before_readiness_as_failed(tmp_path: Path) -> None:
    executor = LocalProcessExecutor(tmp_path)
    remote_id = "prelaunch-exit"
    record_path, log_path, owner_path = executor._paths(remote_id)
    worker = subprocess.Popen((sys.executable, "-c", "raise SystemExit(23)"), start_new_session=True)
    try:
        assert worker.wait(timeout=3) == 23
        record_path.write_text(
            json.dumps(
                {
                    "remote_id": remote_id,
                    "status": "PENDING",
                    "message": "waiting for native-local worker",
                    "worker_pid": worker.pid,
                    "created_at": "2020-01-01T00:00:00+00:00",
                    "started_at": None,
                    "finished_at": None,
                    "exit_code": None,
                    "log_path": str(log_path),
                    "owner_path": str(owner_path),
                }
            )
        )
        executor._worker_processes[remote_id] = worker

        observation = executor.status(remote_id)

        assert observation.status is ExecutionWorkloadStatus.FAILED
        assert "before recording readiness" in observation.message
        record = json.loads(record_path.read_text())
        assert record["exit_code"] == 23
    finally:
        executor._worker_processes.pop(remote_id, None)


def test_local_process_stop_terminates_the_detached_process_group(tmp_path: Path) -> None:
    executor = LocalProcessExecutor(tmp_path)
    context = _context()
    source = (
        "import subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "print(child.pid, flush=True); time.sleep(30)"
    )
    executor.submit(_spec(_command(source)), context)
    assert _wait_for(executor, "workload-1", {ExecutionWorkloadStatus.RUNNING}) is ExecutionWorkloadStatus.RUNNING
    deadline = time.monotonic() + 3
    child_pid = 0
    while time.monotonic() < deadline:
        logs = executor.logs("workload-1").strip()
        if logs:
            child_pid = int(logs.splitlines()[0])
            break
        time.sleep(0.02)
    assert child_pid > 0
    assert _process_alive(child_pid)

    executor.stop("workload-1", context)

    assert _wait_for(executor, "workload-1", {ExecutionWorkloadStatus.STOPPED}) is ExecutionWorkloadStatus.STOPPED
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and _process_alive(child_pid):
        time.sleep(0.02)
    assert not _process_alive(child_pid)


def test_local_process_restart_can_observe_and_retire_running_work(tmp_path: Path, monkeypatch) -> None:
    original = LocalProcessExecutor(tmp_path)
    context = _context(generation=3, token="token-3")
    original.submit(_spec(_command("import time; print('running', flush=True); time.sleep(30)")), context)
    assert _wait_for(original, "workload-1", {ExecutionWorkloadStatus.RUNNING}) is ExecutionWorkloadStatus.RUNNING

    restarted = LocalProcessExecutor(tmp_path)
    assert restarted.status("workload-1").status is ExecutionWorkloadStatus.RUNNING
    monkeypatch.setenv("EYVAN_SUPERSEDE_WAIT_TIMEOUT_S", "2")
    monkeypatch.setenv("EYVAN_SUPERSEDE_POLL_S", "0.02")

    restarted.retire("workload-1", context)

    assert restarted.status("workload-1").status is ExecutionWorkloadStatus.STOPPED


def test_local_process_worker_crash_cleans_up_surviving_process_group(tmp_path: Path) -> None:
    executor = LocalProcessExecutor(tmp_path)
    context = _context()
    source = (
        "import subprocess, sys, time; "
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "print(child.pid, flush=True); time.sleep(30)"
    )
    executor.submit(_spec(_command(source)), context)
    assert _wait_for(executor, "workload-1", {ExecutionWorkloadStatus.RUNNING}) is ExecutionWorkloadStatus.RUNNING
    deadline = time.monotonic() + 3
    child_pid = 0
    while time.monotonic() < deadline:
        logs = executor.logs("workload-1").strip()
        if logs:
            child_pid = int(logs.splitlines()[0])
            break
        time.sleep(0.02)
    record_path, _log_path, _owner_path = executor._paths("workload-1")
    worker_pid = int(json.loads(record_path.read_text())["worker_pid"])
    try:
        os.kill(worker_pid, signal.SIGKILL)
        assert _wait_for(executor, "workload-1", {ExecutionWorkloadStatus.LOST}) is ExecutionWorkloadStatus.LOST
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline and _process_alive(child_pid):
            time.sleep(0.02)
        assert not _process_alive(child_pid)
    finally:
        # The test may have already reaped the worker. Verify the persisted
        # identity before attempting cleanup so a recycled PID cannot signal a
        # later test's process group.
        record = json.loads(record_path.read_text()) if record_path.exists() else {}
        local_process._terminate_tracked_worker(record)


def test_local_process_does_not_kill_a_live_direct_worker_when_its_owner_lock_is_temporarily_unavailable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    executor = LocalProcessExecutor(tmp_path)
    context = _context()
    executor.submit(_spec(_command("import time; time.sleep(30)")), context)
    assert _wait_for(executor, "workload-1", {ExecutionWorkloadStatus.RUNNING}) is ExecutionWorkloadStatus.RUNNING
    record_path, _log_path, _owner_path = executor._paths("workload-1")
    record = json.loads(record_path.read_text())
    record["owner_absent_at"] = "2020-01-01T00:00:00+00:00"
    record_path.write_text(json.dumps(record))

    try:
        with monkeypatch.context() as patch:
            patch.setattr(local_process, "_owner_active", lambda _owner_path: False)
            assert executor.status("workload-1").status is ExecutionWorkloadStatus.RUNNING
    finally:
        executor.retire("workload-1", context)


def test_restarted_local_executor_fails_closed_when_a_crashed_worker_identity_is_ambiguous(tmp_path: Path) -> None:
    original = LocalProcessExecutor(tmp_path)
    context = _context()
    original.submit(_spec(_command("import time; time.sleep(30)")), context)
    assert _wait_for(original, "workload-1", {ExecutionWorkloadStatus.RUNNING}) is ExecutionWorkloadStatus.RUNNING
    record_path, _log_path, _owner_path = original._paths("workload-1")
    worker_pid = int(json.loads(record_path.read_text())["worker_pid"])
    os.kill(worker_pid, signal.SIGKILL)
    restarted = LocalProcessExecutor(tmp_path)

    try:
        deadline = time.monotonic() + 3
        status = ExecutionWorkloadStatus.RUNNING
        while time.monotonic() < deadline:
            status = restarted.status("workload-1").status
            if status is ExecutionWorkloadStatus.INDETERMINATE:
                break
            time.sleep(0.02)
        assert status is ExecutionWorkloadStatus.INDETERMINATE
    finally:
        # The original executor retains the direct child proof and can clean
        # the group without risking a recycled PID.
        deadline = time.monotonic() + 3
        status = ExecutionWorkloadStatus.RUNNING
        while time.monotonic() < deadline:
            status = original.status("workload-1").status
            if status is ExecutionWorkloadStatus.LOST:
                break
            time.sleep(0.02)
        assert status is ExecutionWorkloadStatus.LOST


def test_local_process_never_signals_a_reused_worker_pid(tmp_path: Path, monkeypatch) -> None:
    executor = LocalProcessExecutor(tmp_path)
    record_path, _log_path, _owner_path = executor._paths("reused-pid")
    record_path.write_text(
        json.dumps(
            {
                "remote_id": "reused-pid",
                "status": "RUNNING",
                "worker_pid": os.getpid(),
                "worker_start_marker": "stale-process-marker",
                "created_at": "2020-01-01T00:00:00+00:00",
                "owner_absent_at": "2020-01-01T00:00:00+00:00",
                "stop_requested": False,
            }
        )
    )
    signalled: list[int] = []
    monkeypatch.setattr(
        local_process, "_terminate_process_group", lambda process_group, **_kwargs: signalled.append(process_group)
    )

    status = executor.status("reused-pid")

    assert status.status is ExecutionWorkloadStatus.INDETERMINATE
    assert signalled == []


def test_local_process_checkpoint_runs_as_a_separately_fenced_operation(tmp_path: Path) -> None:
    executor = LocalProcessExecutor(tmp_path / "state")
    source_context = _context(generation=5, token="token-5")
    executor.submit(_spec(_command("import time; time.sleep(30)")), source_context)
    assert _wait_for(executor, "workload-1", {ExecutionWorkloadStatus.RUNNING}) is ExecutionWorkloadStatus.RUNNING
    output = tmp_path / "checkpoint.txt"
    checkpoint_context = _context(key="checkpoint-1", generation=5, token="token-5")
    hook = _command(f"import os, pathlib; pathlib.Path({str(output)!r}).write_text(os.environ['EYVAN_CHECKPOINT_URI'])")
    try:
        checkpoint_id = executor.checkpoint(
            "workload-1",
            hook=hook,
            output_uri="gs://checkpoints/workload-1",
            context=checkpoint_context,
        )

        assert checkpoint_id == "checkpoint-1"
        assert (
            _wait_for(executor, checkpoint_id, {ExecutionWorkloadStatus.SUCCEEDED}) is ExecutionWorkloadStatus.SUCCEEDED
        )
        assert output.read_text() == "gs://checkpoints/workload-1"
        with pytest.raises(StaleMutationError, match="does not own"):
            executor.checkpoint(
                "workload-1",
                hook=hook,
                output_uri=None,
                context=_context(key="checkpoint-2", generation=6, token="token-6"),
            )
    finally:
        executor.retire("workload-1", source_context)


@pytest.mark.parametrize("variable", ["EYVAN_SUPERSEDE_WAIT_TIMEOUT_S", "EYVAN_SUPERSEDE_POLL_S"])
@pytest.mark.parametrize("value", ["not-a-number", "inf"])
def test_local_process_retire_fails_fast_on_a_malformed_supersede_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
    value: str,
) -> None:
    """A malformed supersede timing override must name the offending variable."""

    executor = LocalProcessExecutor(tmp_path)
    context = _context()
    executor.submit(_spec(_command("import time; time.sleep(30)")), context)
    assert _wait_for(executor, "workload-1", {ExecutionWorkloadStatus.RUNNING}) is ExecutionWorkloadStatus.RUNNING
    for name in ("EYVAN_SUPERSEDE_WAIT_TIMEOUT_S", "EYVAN_SUPERSEDE_POLL_S"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(variable, value)

    try:
        with pytest.raises(ValueError, match=variable):
            executor.retire("workload-1", context)
    finally:
        monkeypatch.delenv(variable, raising=False)
        executor.retire("workload-1", context)
    assert executor.status("workload-1").status is ExecutionWorkloadStatus.STOPPED


def test_local_worker_creates_the_log_file_with_owner_only_permissions_from_birth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The workload log must never exist with group or other access bits set."""

    record_path = tmp_path / "record.json"
    log_path = tmp_path / "workload.log"
    owner_path = tmp_path / "owner.lock"
    record_path.write_text(
        json.dumps(
            {
                "remote_id": "workload-1",
                "status": "PENDING",
                "worker_pid": os.getpid(),
                "worker_token": "token-1",
                "stop_requested": False,
                "entrypoint": _command("print('hello-log')"),
                "environment": {},
            }
        )
    )
    birth_modes: dict[str, int] = {}
    real_open = os.open

    def recording_open(path, flags, mode=0o777, *args, **kwargs):
        descriptor = real_open(path, flags, mode, *args, **kwargs)
        if Path(path) == log_path and flags & os.O_CREAT:
            birth_modes["log"] = stat.S_IMODE(os.fstat(descriptor).st_mode)
        return descriptor

    monkeypatch.setattr(local_worker.os, "open", recording_open)
    previous_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        assert local_worker._run(record_path, log_path, owner_path, "token-1") == 0
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    assert "log" in birth_modes, "workload log was not created through a permission-restricted os.open"
    assert birth_modes["log"] & 0o077 == 0
    assert stat.S_IMODE(log_path.stat().st_mode) == 0o600
    record = json.loads(record_path.read_text())
    assert record["status"] == ExecutionWorkloadStatus.SUCCEEDED.value
    assert log_path.read_text().splitlines() == ["hello-log"]


def test_the_retired_ray_backend_refuses_with_an_actionable_message(tmp_path: Path) -> None:
    """Eyvan replaces Ray rather than dispatching to it.

    The name stays registered on purpose: a workload already persisted against
    ``ray-jobs`` has to fail with a sentence an operator can act on, not with an
    unknown-backend error that reads like a typo or a broken install.
    """

    resolver = ConfiguredExecutionResolver("ray-jobs", state_dir=tmp_path)
    observation = CellObservation("cell-b", CapacityStatus.HEALTHY, 1, metadata={"head_ip": "10.0.0.8"})

    with pytest.raises(RuntimeError, match="ray-jobs execution backend has been removed"):
        resolver.resolve("cell-b", observation)


def test_no_eyvan_module_imports_ray(tmp_path: Path) -> None:
    """The dependency is gone, not merely optional.

    An extra that nobody installs still shapes the package: it keeps a competitor
    in the dependency contract, in the audit surface, and in every reader's model
    of what Eyvan needs to run.
    """

    del tmp_path
    import pathlib

    package = pathlib.Path(__import__("eyvan").__file__).parent
    offenders = [
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        if re.search(r"^\s*(?:import ray\b|from ray[.\s])", path.read_text(encoding="utf-8"), re.MULTILINE)
    ]
    assert offenders == []


#: Integration-scale (see the "slow" marker in pyproject.toml): excluded from
#: the fast iteration subset, `pytest tests -m "not slow"`.
pytestmark = pytest.mark.slow
