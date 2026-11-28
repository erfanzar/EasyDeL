# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bound acquisition, not merely parsing, in the external soak monitor."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
import typing as tp
from pathlib import Path

import pytest

from eyvan.sdk.commands import soak_monitor as monitor


def test_pid_file_acquisition_is_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "plane.pid"
    path.write_bytes(b"1" * 65_536)
    acquired: list[int] = []
    requests: list[int] = []
    real_open = Path.open

    class ObservedFile:
        def __init__(self, handle: tp.Any) -> None:
            self.handle = handle

        def __enter__(self) -> ObservedFile:
            self.handle.__enter__()
            return self

        def __exit__(self, *args: tp.Any) -> None:
            self.handle.__exit__(*args)

        def read(self, size: int = -1) -> bytes:
            requests.append(size)
            data = self.handle.read(size)
            acquired.append(len(data))
            return data

    def observed_open(file: Path, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        handle = real_open(file, *args, **kwargs)
        return ObservedFile(handle) if file == path else handle

    monkeypatch.setattr(Path, "open", observed_open)
    assert monitor.read_pid_file(path) is None
    assert acquired and max(acquired) <= monitor.MAX_PID_FILE_BYTES + 1, acquired
    assert all(0 < size <= monitor.MAX_PID_FILE_BYTES + 1 for size in requests), requests


@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_inventory_flood_is_bounded_and_child_reaped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stream: str
) -> None:
    marker = tmp_path / "child-completed"
    code = (
        "import os, pathlib, time; "
        "os.write(1, b'7\\n'); "
        f"os.write({1 if stream == 'stdout' else 2}, b' ' * 1048576); "
        "time.sleep(0.4); "
        f"pathlib.Path({str(marker)!r}).write_text('completed')"
    )
    children: list[subprocess.Popen[bytes]] = []
    pipe_fds: set[int] = set()
    acquired: dict[int, int] = {}
    requests: list[int] = []
    real_popen = subprocess.Popen
    real_read = os.read

    def observed_popen(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        child = real_popen(*args, **kwargs)
        children.append(child)
        for pipe in (child.stdout, child.stderr):
            if pipe is not None:
                pipe_fds.add(pipe.fileno())
        return child

    def observed_read(fd: int, size: int) -> bytes:
        data = real_read(fd, size)
        if fd in pipe_fds:
            requests.append(size)
            acquired[fd] = acquired.get(fd, 0) + len(data)
        return data

    monkeypatch.setattr(subprocess, "Popen", observed_popen)
    monkeypatch.setattr(os, "read", observed_read)
    result = monitor.measure_provider_inventory(shlex.join([sys.executable, "-c", code]), timeout_s=5.0)
    assert children and all(child.poll() is not None for child in children), "child must be reaped"
    assert acquired and max(acquired.values()) <= monitor.MAX_INVENTORY_OUTPUT_BYTES + 1, acquired
    assert requests and max(requests) <= monitor.MAX_INVENTORY_OUTPUT_BYTES + 1, requests
    assert not marker.exists(), "oversized child must be stopped before its later side effect"
    assert result is None


def test_inventory_timeout_stops_and_reaps_child(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    code = f"import os, pathlib, time; pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(30)"
    started = time.monotonic()
    assert monitor.measure_provider_inventory(shlex.join([sys.executable, "-c", code]), timeout_s=0.5) is None
    assert time.monotonic() - started < 5.0
    assert pid_file.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)


def test_inventory_still_scrubs_interpreter_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PYTHONPATH", "/not-for-the-provider")
    code = "import os; print(7 if 'PYTHONPATH' not in os.environ else 9)"
    assert monitor.measure_provider_inventory(shlex.join([sys.executable, "-c", code]), timeout_s=5.0) == 7


def test_general_process_probe_can_capture_more_than_inventory_ceiling() -> None:
    code = "print('f1\\n' * 3000, end='')"
    completed = monitor._run_command([sys.executable, "-c", code], timeout_s=5.0)
    assert completed is not None and completed.returncode == 0
    assert completed.stdout == "f1\n" * 3000


def test_inventory_accepts_small_stderr_diagnostics() -> None:
    code = "import sys; print(7); print('warning', file=sys.stderr)"
    assert monitor.measure_provider_inventory(shlex.join([sys.executable, "-c", code]), timeout_s=5.0) == 7


@pytest.mark.parametrize("extra_byte", [0, 1])
def test_inventory_acquisition_exact_byte_boundary(extra_byte: int) -> None:
    size = monitor.MAX_INVENTORY_OUTPUT_BYTES + extra_byte
    code = f"import os; os.write(1, b'7' + b' ' * {size - 1}); os.write(2, b' ' * {size})"
    result = monitor.measure_provider_inventory(shlex.join([sys.executable, "-c", code]), timeout_s=5.0)
    assert result == (7 if extra_byte == 0 else None)


def test_inventory_ceiling_is_bytes_not_decoded_characters() -> None:
    code = "import os; os.write(1, ('7' + '\\u2003' * 2000).encode('utf-8'))"
    assert monitor.measure_provider_inventory(shlex.join([sys.executable, "-c", code]), timeout_s=5.0) is None


def test_timeout_still_bounds_child_after_both_pipes_close(tmp_path: Path) -> None:
    pid_file = tmp_path / "closed-pipes-child.pid"
    code = (
        f"import os, pathlib, time; pathlib.Path({str(pid_file)!r}).write_text(str(os.getpid())); "
        "os.close(1); os.close(2); time.sleep(30)"
    )
    started = time.monotonic()
    assert monitor.measure_provider_inventory(shlex.join([sys.executable, "-c", code]), timeout_s=0.5) is None
    assert time.monotonic() - started < 5.0
    assert pid_file.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int(pid_file.read_text()), 0)
