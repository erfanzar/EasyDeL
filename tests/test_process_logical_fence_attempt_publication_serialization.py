# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Real-process regression for logical fence/attempt publication serialization."""

from __future__ import annotations

import json
import shlex
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from eyvan._internal.json_types import JsonObject
from eyvan.capacity.drivers.base import ExecutionWorkloadStatus, MutationContext
from eyvan.worker.runtime.local_process import LocalProcessExecutor


def _context(generation: int) -> MutationContext:
    return MutationContext(
        idempotency_key=f"submit-{generation}", generation=generation, fencing_token=f"token-{generation}"
    )


def _spec(generation: int) -> JsonObject:
    return {
        "entrypoint": shlex.join((sys.executable, "-c", "import time; time.sleep(60)")),
        "submission_id": f"attempt-{generation}",
        "gang": {"replicas": 1, "per_replica": {"cpu": 1}, "required_topology": {}},
        "metadata": {"eyvan_workload_id": "shared-workload"},
    }


def _wait_running(executor: LocalProcessExecutor, remote_id: str) -> None:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if executor.status(remote_id).status is ExecutionWorkloadStatus.RUNNING:
            return
        time.sleep(0.02)
    pytest.fail(f"{remote_id} did not start: {executor.status(remote_id)}")


def test_new_generation_cannot_retire_an_unpublished_predecessor(tmp_path: Path) -> None:
    """A successor must retire the real predecessor, not an absent attempt."""

    old_claimed = Event()
    resume_old = Event()
    new_started = Event()
    new_claimed = Event()

    class PausingExecutor(LocalProcessExecutor):
        def _claim_logical_workload(self, workload_id: str, remote_id: str, context: MutationContext) -> None:
            super()._claim_logical_workload(workload_id, remote_id, context)
            old_claimed.set()
            if not resume_old.wait(timeout=10):
                raise RuntimeError("test did not release predecessor publication")

    class SuccessorExecutor(LocalProcessExecutor):
        def submit(self, spec: JsonObject, context: MutationContext) -> str:
            new_started.set()
            return super().submit(spec, context)

        def _claim_logical_workload(self, workload_id: str, remote_id: str, context: MutationContext) -> None:
            super()._claim_logical_workload(workload_id, remote_id, context)
            new_claimed.set()

    old = PausingExecutor(tmp_path)
    new = SuccessorExecutor(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        try:
            first = pool.submit(old.submit, _spec(1), _context(1))
            assert old_claimed.wait(timeout=5), "predecessor never claimed its logical fence"
            second = pool.submit(new.submit, _spec(2), _context(2))
            assert new_started.wait(timeout=5), "successor submission never began"
            # On the buggy implementation the newer claim passes the paused
            # predecessor and launches immediately. With serialization it waits
            # for publication; release the predecessor after this bounded wait.
            if new_claimed.wait(timeout=2):
                assert second.result(timeout=8) == "attempt-2"
                _wait_running(new, "attempt-2")
            resume_old.set()
            assert first.result(timeout=8) == "attempt-1"
            assert second.result(timeout=8) == "attempt-2"
            _wait_running(new, "attempt-2")
            predecessor = old.status("attempt-1")
            successor = new.status("attempt-2")
            fence = json.loads(old._workload_fence_path("shared-workload").read_text())
            assert fence["generation"] == 2
            assert fence["remote_id"] == "attempt-2"
            assert predecessor.status is ExecutionWorkloadStatus.STOPPED, (
                f"unretired predecessor {predecessor}; successor {successor}; logical fence {fence}"
            )
        finally:
            resume_old.set()
            # Join submissions before cleanup so a released thread cannot launch
            # its command after the test has already attempted retirement.
            for future in (locals().get("first"), locals().get("second")):
                if future is not None:
                    try:
                        future.result(timeout=12)
                    except Exception:
                        pass
            try:
                old.retire("attempt-1", _context(1))
            finally:
                new.retire("attempt-2", _context(2))
            assert old.status("attempt-1").status in {
                ExecutionWorkloadStatus.STOPPED,
                ExecutionWorkloadStatus.NOT_FOUND,
            }
            assert new.status("attempt-2").status in {
                ExecutionWorkloadStatus.STOPPED,
                ExecutionWorkloadStatus.NOT_FOUND,
            }
