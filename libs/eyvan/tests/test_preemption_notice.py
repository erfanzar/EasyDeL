# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for Eyvan's stdlib-only in-job preemption/reclaim notice protocol."""

from __future__ import annotations

import json
import os
import signal
import stat
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import eyvan
from eyvan.capacity.drivers.base import ExecutionWorkloadStatus, MutationContext
from eyvan.worker.runtime import progress as runtime_progress
from eyvan.worker.runtime.local_process import LocalProcessExecutor
from eyvan.worker.runtime.preemption import (
    CHECKPOINT_RECEIPT_PATH_ENV,
    RECLAIM_NOTICE_PATH_ENV,
    RESUME_CHECKPOINT_URI_ENV,
    PreemptionNotice,
    acknowledge_checkpoint,
    install_preemption_handler,
    preemption_notice,
    read_checkpoint_receipt,
    reclaim_requested,
    resume_checkpoint_uri,
    uninstall_preemption_handler,
    write_reclaim_notice,
)


@pytest.fixture(autouse=True)
def _reset_signal_handlers():
    uninstall_preemption_handler()
    yield
    uninstall_preemption_handler()


def test_lazy_exports_stay_stdlib_only_without_heavy_imports() -> None:
    """Importing the preemption helpers from ``eyvan`` must stay lightweight."""

    code = (
        "import sys, eyvan\n"
        "names = ('PreemptionNotice', 'preemption_notice', 'reclaim_requested', "
        "'install_preemption_handler', 'acknowledge_checkpoint', 'resume_checkpoint_uri')\n"
        "for n in names:\n"
        "    assert getattr(eyvan, n) is not None\n"
        "heavy = {'fastapi', 'httpx', 'pydantic', 'sqlalchemy'}\n"
        "loaded = sorted(heavy & set(sys.modules))\n"
        "assert not loaded, f'heavy modules imported: {loaded}'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr or result.stdout


def test_top_level_eyvan_reexports_all_six_symbols() -> None:
    assert eyvan.PreemptionNotice is PreemptionNotice
    assert eyvan.preemption_notice is preemption_notice
    assert eyvan.reclaim_requested is reclaim_requested
    assert eyvan.install_preemption_handler is install_preemption_handler
    assert eyvan.acknowledge_checkpoint is acknowledge_checkpoint
    assert eyvan.resume_checkpoint_uri is resume_checkpoint_uri


def test_preemption_notice_remaining_seconds_and_roundtrip() -> None:
    future_dt = datetime.now(UTC) + timedelta(seconds=120)
    notice = PreemptionNotice(
        reason="owner bob needs v5p-128",
        requested_by="bob",
        deadline_at=future_dt.isoformat(),
        deadline_epoch_s=future_dt.timestamp(),
        grace_seconds=600,
        policy="requeue",
        checkpoint_uri="gs://bucket/ckpt",
        workload_id="wl_123",
    )
    assert 100.0 <= notice.remaining_seconds <= 121.0
    restored = PreemptionNotice.from_dict(notice.to_dict())
    assert restored == notice

    past = PreemptionNotice(reason="expired", deadline_epoch_s=time.time() - 10.0, grace_seconds=600)
    assert past.remaining_seconds == 0.0

    iso_only = PreemptionNotice(
        reason="iso deadline",
        deadline_at=(datetime.now(UTC) + timedelta(seconds=45)).isoformat().replace("+00:00", "Z"),
    )
    assert 30.0 <= iso_only.remaining_seconds <= 46.0

    grace_only = PreemptionNotice(reason="grace only", grace_seconds=300)
    assert grace_only.remaining_seconds == 300.0


def test_notice_file_reading_and_malformed_file_tolerance(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    env = {RECLAIM_NOTICE_PATH_ENV: str(missing)}
    assert preemption_notice(env) is None
    assert reclaim_requested(env) is False

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not-json", encoding="utf-8")
    assert preemption_notice({RECLAIM_NOTICE_PATH_ENV: str(bad_json)}) is None
    assert reclaim_requested({RECLAIM_NOTICE_PATH_ENV: str(bad_json)}) is False

    non_dict = tmp_path / "list.json"
    non_dict.write_text("[1, 2, 3]", encoding="utf-8")
    assert preemption_notice({RECLAIM_NOTICE_PATH_ENV: str(non_dict)}) is None

    notice_path = tmp_path / "reclaim.json"
    deadline = datetime.now(UTC) + timedelta(seconds=600)
    expected = PreemptionNotice(
        reason="borrowed capacity reclaimed",
        requested_by="bob",
        deadline_at=deadline.isoformat(),
        deadline_epoch_s=deadline.timestamp(),
        grace_seconds=600,
        policy="stop",
        checkpoint_uri="gs://bucket/run-1",
        workload_id="wl_abc",
    )
    write_reclaim_notice(notice_path, expected)
    assert stat.S_IMODE(notice_path.stat().st_mode) == 0o600

    env_valid = {RECLAIM_NOTICE_PATH_ENV: str(notice_path)}
    assert reclaim_requested(env_valid) is True
    loaded = preemption_notice(env_valid)
    assert loaded == expected


def test_signal_handler_is_idempotent_chains_and_enriches_from_notice_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if not hasattr(signal, "SIGUSR1"):
        pytest.skip("SIGUSR1 not available on this platform")

    chained: list[int] = []
    callbacks: list[PreemptionNotice] = []
    previous = signal.getsignal(signal.SIGUSR1)

    def prior_handler(signum: int, _frame) -> None:
        chained.append(signum)

    signal.signal(signal.SIGUSR1, prior_handler)
    try:
        notice_path = tmp_path / "reclaim.json"
        deadline = datetime.now(UTC) + timedelta(seconds=600)
        write_reclaim_notice(
            notice_path,
            PreemptionNotice(
                reason="owner reclaim",
                requested_by="bob",
                deadline_at=deadline.isoformat(),
                deadline_epoch_s=deadline.timestamp(),
                grace_seconds=600,
                policy="requeue",
                workload_id="wl_live",
            ),
        )
        monkeypatch.setenv(RECLAIM_NOTICE_PATH_ENV, str(notice_path))

        # Installing twice with the same callback must remain idempotent.
        install_preemption_handler(callback=callbacks.append)
        install_preemption_handler(callback=callbacks.append)
        os.kill(os.getpid(), signal.SIGUSR1)

        assert reclaim_requested({}) is True
        active = preemption_notice()
        assert active is not None
        assert active.reason == "owner reclaim"
        assert active.requested_by == "bob"
        assert active.grace_seconds == 600
        assert len(callbacks) == 1
        assert callbacks[0] == active
        assert chained == [signal.SIGUSR1]
    finally:
        uninstall_preemption_handler()
        signal.signal(signal.SIGUSR1, previous)


def test_acknowledge_checkpoint_writes_0600_receipt_and_reports_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt_path = tmp_path / "sub" / "receipt.json"
    captured: list[tuple[str, str | None, dict[str, object]]] = []

    class _FakeResponse:
        def read(self) -> bytes:
            return b"{}"

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *_exc: object) -> bool:
            return False

    def _fake_urlopen(request, timeout=None, **_kwargs):
        del timeout
        token = request.get_header("X-eyvan-artifact-token")
        body = json.loads(request.data.decode("utf-8"))
        captured.append((request.full_url, token, body))
        return _FakeResponse()

    monkeypatch.setattr(runtime_progress.urllib.request, "urlopen", _fake_urlopen)

    env = {
        CHECKPOINT_RECEIPT_PATH_ENV: str(receipt_path),
        "EYVAN_PROGRESS_URL": "https://plane.example/api/v1/workloads/wl_1/progress-reports",
        "EYVAN_MANAGED_RESULT_TOKEN": "tok-123",
    }
    written = acknowledge_checkpoint("gs://eyvan-us-east5/run/step-42", step=42, environment=env)
    assert written == receipt_path
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o600
    receipt = read_checkpoint_receipt(receipt_path)
    assert receipt is not None
    assert receipt["checkpoint_uri"] == "gs://eyvan-us-east5/run/step-42"
    assert receipt["step"] == 42
    assert isinstance(receipt["recorded_at"], str)

    assert len(captured) == 1
    url, token, body = captured[0]
    assert url == "https://plane.example/api/v1/workloads/wl_1/progress-reports"
    assert token == "tok-123"
    assert body["step"] == 42
    assert "gs://eyvan-us-east5/run/step-42" in str(body["message"])

    # resume_checkpoint_uri falls back to receipt when RESUME env is absent, and prefers env when set.
    assert resume_checkpoint_uri(env) == "gs://eyvan-us-east5/run/step-42"
    assert (
        resume_checkpoint_uri({**env, RESUME_CHECKPOINT_URI_ENV: "gs://eyvan-us-east5/run/step-99"})
        == "gs://eyvan-us-east5/run/step-99"
    )

    with pytest.raises(ValueError, match="non-empty"):
        acknowledge_checkpoint("   ", environment={})
    with pytest.raises(ValueError, match="non-negative"):
        acknowledge_checkpoint("gs://b/c", step=-1, environment={})


def test_local_process_executor_delivers_graceful_reclaim_notice_and_captures_checkpoint(tmp_path: Path) -> None:
    """Real child process catches SIGUSR1, reads notice, acknowledges checkpoint, and exits cleanly."""

    executor = LocalProcessExecutor(state_dir=tmp_path / "runtime")
    ready_file = tmp_path / "ready.txt"
    observed_file = tmp_path / "observed.json"
    script = tmp_path / "trainer.py"
    script.write_text(
        "import json, os, pathlib, time\n"
        "import eyvan\n"
        "eyvan.install_preemption_handler()\n"
        f"pathlib.Path({str(ready_file)!r}).write_text('ready')\n"
        "deadline = time.monotonic() + 15.0\n"
        "while time.monotonic() < deadline:\n"
        "    notice = eyvan.preemption_notice()\n"
        "    if notice is not None:\n"
        f"        pathlib.Path({str(observed_file)!r}).write_text(json.dumps(notice.to_dict()))\n"
        "        eyvan.acknowledge_checkpoint('gs://eyvan-ckpt/job/step-128', step=128)\n"
        "        raise SystemExit(0)\n"
        "    time.sleep(0.02)\n"
        "raise SystemExit(2)\n",
        encoding="utf-8",
    )
    context = MutationContext(idempotency_key="sub-graceful-1", generation=1, fencing_token="tok-1")
    remote_id = executor.submit(
        {
            "entrypoint": f"{sys.executable} {script}",
            "submission_id": "sub-graceful-1",
            "environment": {"PYTHONPATH": str(Path(__file__).resolve().parents[1])},
            "gang": {"replicas": 1, "per_replica": {"cpu": 1}},
            "metadata": {"eyvan_workload_id": "wl_graceful"},
        },
        context,
    )

    wait_ready = time.monotonic() + 10.0
    while not ready_file.exists() and time.monotonic() < wait_ready:
        time.sleep(0.02)
    assert ready_file.exists(), executor.logs(remote_id)

    deadline_dt = datetime.now(UTC) + timedelta(seconds=600)
    executor.notify_reclaim(
        remote_id,
        notice={
            "reason": "owner bob reclaimed borrowed TPU",
            "requested_by": "bob",
            "deadline_at": deadline_dt.isoformat(),
            "deadline_epoch_s": deadline_dt.timestamp(),
            "grace_seconds": 600,
            "policy": "requeue",
            "checkpoint_uri": None,
            "workload_id": "wl_graceful",
        },
        grace_s=600,
        context=context,
    )

    wait_done = time.monotonic() + 10.0
    status = executor.status(remote_id)
    terminal_states = {ExecutionWorkloadStatus.STOPPED, ExecutionWorkloadStatus.FAILED}
    while status.status not in terminal_states and time.monotonic() < wait_done:
        time.sleep(0.02)
        status = executor.status(remote_id)

    assert status.status is ExecutionWorkloadStatus.STOPPED, f"{status} logs={executor.logs(remote_id)}"
    assert observed_file.exists()
    seen = json.loads(observed_file.read_text(encoding="utf-8"))
    assert seen["requested_by"] == "bob"
    assert seen["grace_seconds"] == 600
    assert seen["policy"] == "requeue"
    assert executor.read_checkpoint_uri(remote_id) == "gs://eyvan-ckpt/job/step-128"


def test_local_worker_escalates_to_sigkill_after_custom_reclaim_grace_expires(tmp_path: Path) -> None:
    """A job ignoring SIGUSR1/SIGTERM is killed once its per-job reclaim grace expires."""

    executor = LocalProcessExecutor(state_dir=tmp_path / "runtime")
    ready_file = tmp_path / "stubborn.ready"
    script = tmp_path / "stubborn.py"
    script.write_text(
        "import pathlib, signal, time\n"
        "signal.signal(signal.SIGUSR1, signal.SIG_IGN)\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"pathlib.Path({str(ready_file)!r}).write_text('ready')\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    context = MutationContext(idempotency_key="sub-stubborn-1", generation=1, fencing_token="tok-2")
    remote_id = executor.submit(
        {
            "entrypoint": f"{sys.executable} {script}",
            "submission_id": "sub-stubborn-1",
            "gang": {"replicas": 1, "per_replica": {"cpu": 1}},
            "metadata": {"eyvan_workload_id": "wl_stubborn"},
        },
        context,
    )

    wait_ready = time.monotonic() + 10.0
    while not ready_file.exists() and time.monotonic() < wait_ready:
        time.sleep(0.02)
    assert ready_file.exists()

    started = time.monotonic()
    executor.notify_reclaim(
        remote_id,
        notice={
            "reason": "deadline test",
            "requested_by": "bob",
            "deadline_at": None,
            "deadline_epoch_s": time.time() + 1.0,
            "grace_seconds": 1,
            "policy": "requeue",
            "checkpoint_uri": None,
            "workload_id": "wl_stubborn",
        },
        grace_s=1,
        context=context,
    )

    wait_done = time.monotonic() + 10.0
    status = executor.status(remote_id)
    while status.status is not ExecutionWorkloadStatus.STOPPED and time.monotonic() < wait_done:
        time.sleep(0.05)
        status = executor.status(remote_id)
    elapsed = time.monotonic() - started
    assert status.status is ExecutionWorkloadStatus.STOPPED
    assert elapsed >= 0.8
