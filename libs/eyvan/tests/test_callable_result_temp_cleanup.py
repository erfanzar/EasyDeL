# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""A failed local result write must not leak its temp file.

``UriArtifactTransfer.write`` publishes a result with temp-file-plus-rename;
the temp file is created with ``delete=False``, so every failed attempt used
to leave a ``.result-*`` orphan behind -- one per failed attempt, in a
directory nothing prunes.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from eyvan.worker.runtime.callable_protocol import UriArtifactTransfer, dump_callable, execute_callable


class _NoGcs:
    def read_object(self, bucket_name: str, object_name: str) -> bytes:
        raise AssertionError("not used")

    def write_object(self, bucket_name: str, object_name: str, payload: bytes) -> None:
        raise AssertionError("not used")


def test_a_failed_local_result_write_leaves_no_temp_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transfer = UriArtifactTransfer(_NoGcs(), {})
    target = tmp_path / "out" / "result.json"

    def broken_fsync(fd: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", broken_fsync)

    with pytest.raises(OSError, match="disk full"):
        transfer.write(str(target), b"payload")

    assert list((tmp_path / "out").iterdir()) == []


@pytest.mark.parametrize("failure", ("directory", "cleanup-error", "partial-write"))
def test_callable_executes_when_optional_cache_fill_and_cleanup_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], failure: str
) -> None:
    artifact = dump_callable(lambda: 7)
    source = tmp_path / "callable.pkl"
    source.write_bytes(artifact.payload)
    state = tmp_path / "worker-state"
    cache = state / "callable-cache"
    cache.mkdir(parents=True, mode=0o700)
    temporary = cache / f"{artifact.sha256}.pkl.{os.getpid()}.tmp"
    original_write = Path.write_bytes
    original_unlink = Path.unlink

    if failure == "directory":
        temporary.mkdir()
    else:

        def fail_write(path: Path, payload: bytes) -> int:
            if path == temporary:
                original_write(path, payload[:3])
                raise OSError("cache fill unavailable")
            return original_write(path, payload)

        monkeypatch.setattr(Path, "write_bytes", fail_write)
        if failure == "cleanup-error":

            def fail_unlink(path: Path, missing_ok: bool = False) -> None:
                if path == temporary:
                    raise OSError("cache cleanup unavailable")
                original_unlink(path, missing_ok=missing_ok)

            monkeypatch.setattr(Path, "unlink", fail_unlink)

    result = execute_callable(
        ["--uri", str(source), "--sha256", artifact.sha256, "--python", artifact.python],
        transfer=UriArtifactTransfer(_NoGcs(), {}),
        environment={"EYVAN_NATIVE_WORKER_STATE_DIR": str(state)},
    )

    assert result == 0
    assert capsys.readouterr().out == "7\n"
    assert not (cache / f"{artifact.sha256}.pkl").exists()
    if failure == "partial-write":
        assert not temporary.exists()
    else:
        assert temporary.exists()
