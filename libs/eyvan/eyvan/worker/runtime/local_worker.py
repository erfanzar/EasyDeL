# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Detached worker used by the Eyvan native-local execution backend."""

from __future__ import annotations

import argparse
import fcntl
import math
import os
import signal
import subprocess
import time
import typing as tp
from pathlib import Path
from types import FrameType

from eyvan._internal.json_types import JsonObject
from eyvan._internal.workspace import create_private_directory
from eyvan.capacity.drivers.base import ExecutionWorkloadStatus
from eyvan.storage._region_guard import CROSS_REGION_TRANSFER_BANNED_EXIT_CODE
from eyvan.worker.runtime.local_process import (
    _format_cross_region_ban_message,
    _group_has_live_members,
    _mutate_record,
    _process_start_marker,
    _quiesce_process_group,
    _read_record,
    _unreaped_process_is_running,
    _utcnow,
)
from eyvan.worker.runtime.preemption import read_checkpoint_receipt

# Default grace period between a SIGTERM stop request and SIGKILL escalation
# when the durable execution record does not declare a per-job ``stop_grace_s``.
_STOP_GRACE_S = 30.0


def _resolve_stop_grace_s(record_path: Path, *, default: float = _STOP_GRACE_S) -> float:
    """Return the per-job graceful stop window from the durable record, or ``default``."""

    current = _read_record(record_path)
    if current is None:
        return default
    raw = current.get("stop_grace_s")
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return default
    value = float(raw)
    return value if math.isfinite(value) and value > 0 else default


def _run(record_path: Path, log_path: Path, owner_path: Path, worker_token: str) -> int:
    create_private_directory(owner_path.parent)
    owner_descriptor = os.open(owner_path, os.O_CREAT | os.O_RDWR, 0o600)
    os.fchmod(owner_descriptor, 0o600)
    fcntl.flock(owner_descriptor, fcntl.LOCK_EX)
    child: subprocess.Popen[bytes] | None = None
    command_quiescent = False
    stop_requested = False
    reclaim_signaled = False
    escalated_stop = False

    def request_stop(signum: int, _frame: FrameType | None) -> None:
        nonlocal stop_requested, reclaim_signaled, escalated_stop
        stop_requested = True
        sigusr1 = getattr(signal, "SIGUSR1", None)
        if sigusr1 is not None and signum == sigusr1:
            reclaim_signaled = True
            forward_sig = sigusr1
        else:
            if reclaim_signaled:
                escalated_stop = True
            forward_sig = signal.SIGTERM
        if child is not None and not command_quiescent:
            try:
                os.killpg(child.pid, forward_sig)
            except ProcessLookupError:
                pass
            except PermissionError:
                if _group_has_live_members(child.pid):
                    raise

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if hasattr(signal, "SIGUSR1"):
        signal.signal(signal.SIGUSR1, request_stop)
    try:
        record = None
        # The parent persists this worker PID immediately after ``Popen``.
        # Under a busy controller host the child can start before that atomic
        # write is scheduled, so retain the normal startup grace rather than
        # treating a harmless publication delay as a failed workload.
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            candidate = _read_record(record_path)
            if (
                candidate is not None
                and candidate.get("worker_pid") == os.getpid()
                and candidate.get("worker_token") == worker_token
            ):
                record = candidate
                break
            time.sleep(0.01)
        if record is None:
            return 2
        if bool(record.get("stop_requested")):
            stop_requested = True

            def stopped_before_start(current: JsonObject) -> None:
                current["status"] = ExecutionWorkloadStatus.STOPPED.value
                current["message"] = "native-local worker stopped before command launch"
                current["exit_code"] = None
                current["finished_at"] = _utcnow()
                current["group_quiescent"] = True

            _mutate_record(record_path, stopped_before_start)
            return 0

        entrypoint = record.get("entrypoint")
        environment = record.get("environment") or {}
        if not isinstance(entrypoint, str) or not isinstance(environment, dict):
            raise RuntimeError("native-local execution record has an invalid command or environment")
        if entrypoint.startswith("python "):
            # The controller's managed-callable commands use a bare `python`.
            # Bind that default to the interpreter running this local executor,
            # not shell activation or an unrelated system interpreter. Explicit
            # interpreter paths and other shell commands remain untouched.
            import shlex
            import sys

            entrypoint = shlex.quote(sys.executable) + entrypoint[len("python") :]

        def running(current: JsonObject) -> None:
            current["status"] = ExecutionWorkloadStatus.RUNNING.value
            current["message"] = "native-local command is running"
            current["started_at"] = _utcnow()

        _mutate_record(record_path, running)
        create_private_directory(log_path.parent)
        # Create the log with owner-only permissions from birth: a plain open
        # followed by chmod leaves a window where workload output is readable
        # under the process umask. fchmod also tightens a pre-existing file,
        # whose mode ``O_CREAT`` leaves untouched.
        log_descriptor = os.open(log_path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        os.fchmod(log_descriptor, 0o600)
        with os.fdopen(log_descriptor, "ab", buffering=0) as log:
            # Service configuration, including native-worker credentials, must
            # never leak into the untrusted workload process. Workload-scoped
            # EYVAN_* values are persisted explicitly in ``environment``.
            inherited_environment = {key: value for key, value in os.environ.items() if not key.startswith("EYVAN_")}
            # The launch gate prevents command execution before subgroup
            # ownership is durably published. EOF (worker crash/publication
            # failure) makes the launcher exit without executing the command.
            gate_read, gate_write = os.pipe()
            try:
                child = subprocess.Popen(
                    (
                        "/bin/sh",
                        "-c",
                        'IFS= read -r permit && [ "$permit" = launch ] && exec </dev/null && exec /bin/sh -lc "$1"',
                        "eyvan-local-launch",
                        entrypoint,
                    ),
                    stdin=gate_read,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    close_fds=True,
                    process_group=0,
                    env={**inherited_environment, **environment},
                )
                marker = _process_start_marker(child.pid)
                if marker is None:
                    raise RuntimeError("native-local command identity could not be established")

                def command_owned(current: JsonObject) -> None:
                    if current.get("worker_token") != worker_token or current.get("worker_pid") != os.getpid():
                        raise RuntimeError("native-local worker ownership changed before launch")
                    current["command_pid"] = child.pid
                    current["command_start_marker"] = marker

                published = _mutate_record(record_path, command_owned)
                if published is None:
                    raise RuntimeError("native-local execution disappeared before command launch")
                stop_requested = stop_requested or bool(published.get("stop_requested"))
                if not stop_requested:
                    os.write(gate_write, b"launch\n")
            finally:
                os.close(gate_read)
                os.close(gate_write)
            if stop_requested:
                request_stop(getattr(signal, "SIGUSR1", signal.SIGTERM) if reclaim_signaled else signal.SIGTERM, None)
            stop_deadline = time.monotonic() + _resolve_stop_grace_s(record_path) if stop_requested else None
            # Do not poll/wait the leader yet: retaining it unreaped pins the
            # process-group identity through descendant TERM/KILL cleanup.
            while _unreaped_process_is_running(child.pid):
                if stop_requested:
                    if stop_deadline is None:
                        stop_deadline = time.monotonic() + _resolve_stop_grace_s(record_path)
                    elif escalated_stop:
                        # Controller explicitly escalated to SIGTERM after the
                        # reclaim grace window elapsed; bound remaining wait.
                        stop_deadline = min(stop_deadline, time.monotonic() + 2.0)
                        escalated_stop = False
                    if time.monotonic() >= stop_deadline:
                        break
                time.sleep(0.05)
            if not _quiesce_process_group(child.pid):
                raise RuntimeError("native-local command group did not quiesce")
            command_quiescent = True
            exit_code = child.wait()

        acknowledged_uri: str | None = None
        raw_receipt_path = record.get("receipt_path")
        if isinstance(raw_receipt_path, str) and raw_receipt_path.strip():
            receipt = read_checkpoint_receipt(raw_receipt_path)
            if receipt is not None:
                raw_uri = receipt.get("checkpoint_uri")
                if isinstance(raw_uri, str) and raw_uri.strip():
                    acknowledged_uri = raw_uri.strip()

        def finished(current: JsonObject) -> None:
            stopped = stop_requested or reclaim_signaled or bool(current.get("stop_requested"))
            current["status"] = (
                ExecutionWorkloadStatus.STOPPED.value
                if stopped
                else ExecutionWorkloadStatus.SUCCEEDED.value
                if exit_code == 0
                else ExecutionWorkloadStatus.FAILED.value
            )
            current["message"] = (
                "native-local command was stopped"
                if stopped
                else "native-local command completed"
                if exit_code == 0
                else _format_cross_region_ban_message(log_path)
                if exit_code == CROSS_REGION_TRANSFER_BANNED_EXIT_CODE
                else f"native-local command exited with code {exit_code}"
            )
            if acknowledged_uri is not None:
                current["checkpoint_uri"] = acknowledged_uri
            current["exit_code"] = exit_code
            current["finished_at"] = _utcnow()
            current["group_quiescent"] = True

        _mutate_record(record_path, finished)
        return 0
    except BaseException as exc:
        failure_message = str(exc)
        quiescent = child is None or command_quiescent
        if child is not None and not command_quiescent:
            try:
                quiescent = _quiesce_process_group(child.pid)
                if quiescent:
                    child.wait()
            except (OSError, RuntimeError, subprocess.SubprocessError):
                quiescent = False

        def failed(current: JsonObject) -> None:
            current["status"] = (
                ExecutionWorkloadStatus.FAILED.value if quiescent else ExecutionWorkloadStatus.INDETERMINATE.value
            )
            current["message"] = f"native-local worker failed: {failure_message}"
            current["finished_at"] = _utcnow()
            current["group_quiescent"] = quiescent

        _mutate_record(record_path, failed)
        return 1
    finally:
        fcntl.flock(owner_descriptor, fcntl.LOCK_UN)
        os.close(owner_descriptor)


def main(argv: tp.Sequence[str] | None = None) -> int:
    """Run one persisted native-local command."""

    parser = argparse.ArgumentParser(description="Run one Eyvan native-local workload")
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--owner", type=Path, required=True)
    parser.add_argument("--worker-token", required=True)
    args = parser.parse_args(argv)
    return _run(args.record, args.log, args.owner, args.worker_token)


if __name__ == "__main__":
    raise SystemExit(main())
