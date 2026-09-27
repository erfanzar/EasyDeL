# Copyright 2026 The EasyDeL/eFormer Author @erfanzar (Erfan Zare Chavoshi).
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Concurrent local index updates without TensorStore or device computation."""

import json
import multiprocessing
import os
import threading
from contextlib import nullcontext

import pytest


class _ArrayMetadataOnly:
    """Array-like leaf for the index; no JAX or NumPy computations."""

    shape = (2,)
    dtype = "float32"


class _NoOpManager:
    def serialize_with_paths(self, arrays, paths, on_commit_callback=None):
        if on_commit_callback is not None:
            on_commit_callback()


def _save_index(path, prefix, entered, release, started, reached_read=None, disable_lock=False):
    from eformer.paths import LocalPath
    from eformer.serialization import serialization as serialization_mod

    if entered is not None:
        original_replace = LocalPath.replace

        def pause_before_replace(self, target):
            entered.set()
            if not release.wait(timeout=20):
                raise TimeoutError("parent did not release first index writer")
            return original_replace(self, target)

        LocalPath.replace = pause_before_replace

    if started is not None:
        original_exists = LocalPath.exists

        def observe_index_read(self):
            if str(self) == f"{path}/tensorstore_index.json":
                reached_read.set()
            return original_exists(self)

        LocalPath.exists = observe_index_read
        if disable_lock:
            # Diagnostic only: show that this test fails against the unlocked path.
            serialization_mod._local_index_lock = lambda _path: nullcontext()
        started.set()
    serialization_mod.tree_serialize_leaves(path, {"weight": _ArrayMetadataOnly()}, _NoOpManager(), prefix=prefix)


def _run_two_index_writers(tmp_path, *, disable_lock):
    """Pause writer one before ``replace``; report whether writer two reached the index read meanwhile.

    Args:
        tmp_path: Checkpoint directory shared by both writer processes.
        disable_lock: Replace the second writer's index lock with a no-op.

    Returns:
        ``True`` if the second writer read the index while the first was paused, else ``False``.
    """
    ctx = multiprocessing.get_context("spawn")
    entered = ctx.Event()
    release = ctx.Event()
    started = ctx.Event()
    reached_read = ctx.Event()
    first = ctx.Process(target=_save_index, args=(str(tmp_path), "model", entered, release, None))
    second = ctx.Process(
        target=_save_index,
        args=(str(tmp_path), "tx", None, None, started, reached_read, disable_lock),
    )
    first.start()
    try:
        assert entered.wait(timeout=20), "first writer did not reach index replacement"
        second.start()
        assert started.wait(timeout=20), "second writer did not start"
        # Locked writers must not read while the first is paused; unlocked ones do so promptly.
        return reached_read.wait(timeout=5 if disable_lock else 1)
    finally:
        release.set()
        for process in (first, second):
            if process.pid is not None:
                process.join(timeout=20)
            if process.pid is not None and process.is_alive():
                process.terminate()
                process.join(timeout=5)


@pytest.mark.cpu_ok
def test_local_index_lock_negative_control_unlocked_writer_reads_concurrently(tmp_path):
    """Without the lock, the second writer reaches the index read while the first is paused.

    This proves the locked test below can fail: it observes the same event and
    only the lock separates the two outcomes.
    """
    from eformer.serialization import serialization as serialization_mod

    if serialization_mod.fcntl is None:
        pytest.skip("cross-process flock is unavailable")
    assert _run_two_index_writers(tmp_path, disable_lock=True), "unlocked second writer never read the index"


@pytest.mark.cpu_ok
def test_local_index_read_merge_replace_is_locked_across_processes(tmp_path):
    """A second prefix cannot reach index read until the first writer commits."""
    from eformer.serialization import serialization as serialization_mod

    if serialization_mod.fcntl is None:
        pytest.skip("cross-process flock is unavailable")
    ctx = multiprocessing.get_context("spawn")
    entered = ctx.Event()
    release = ctx.Event()
    started = ctx.Event()
    reached_read = ctx.Event()
    first = ctx.Process(target=_save_index, args=(str(tmp_path), "model", entered, release, None))
    second = ctx.Process(
        target=_save_index,
        args=(str(tmp_path), "tx", None, None, started, reached_read, False),
    )
    first.start()
    try:
        assert entered.wait(timeout=20), "first writer did not reach index replacement"
        second.start()
        assert started.wait(timeout=20), "second writer did not start"
        # This event fires at the actual exists/read site, not at process startup.
        # An unlocked writer reaches it while the first is paused before replace.
        assert not reached_read.wait(timeout=1), "second writer read the index while the first held its lock"
        release.set()
        assert reached_read.wait(timeout=20), "second writer never read the committed index"
        first.join(timeout=20)
        second.join(timeout=20)
        assert first.exitcode == 0
        assert second.exitcode == 0
        index = json.loads((tmp_path / "tensorstore_index.json").read_text())
        assert set(index["prefixes"]) == {"model", "tx"}
        assert index["prefixes"]["model"][0]["path"] == "model/weight"
        assert index["prefixes"]["tx"][0]["path"] == "tx/weight"
    finally:
        release.set()
        for process in (first, second):
            if process.pid is not None and process.is_alive():
                process.terminate()
            if process.pid is not None:
                process.join(timeout=5)


@pytest.mark.cpu_ok
def test_local_index_temp_names_are_unique_even_with_identical_host_pid_thread(tmp_path, monkeypatch):
    """Two writers on a shared mount with equal hostname/pid/thread id must not share a temp file."""
    from eformer.paths import LocalPath
    from eformer.serialization import serialization as serialization_mod

    monkeypatch.setattr(serialization_mod.os, "getpid", lambda: 4242)
    monkeypatch.setattr(serialization_mod.threading, "get_ident", lambda: 7)
    monkeypatch.setattr(serialization_mod.socket, "gethostname", lambda: "same-host")

    temp_names = []
    original_replace = LocalPath.replace

    def record_replace(self, target):
        temp_names.append(self.path.name)
        return original_replace(self, target)

    monkeypatch.setattr(LocalPath, "replace", record_replace)
    for prefix in ("model", "tx"):
        serialization_mod.tree_serialize_leaves(
            str(tmp_path), {"weight": _ArrayMetadataOnly()}, _NoOpManager(), prefix=prefix
        )

    assert len(temp_names) == 2
    assert temp_names[0] != temp_names[1]
    assert all(name.startswith("tensorstore_index.json.tmp-same-host-4242-7-") for name in temp_names)
    index = json.loads((tmp_path / "tensorstore_index.json").read_text())
    assert set(index["prefixes"]) == {"model", "tx"}
    assert not [p for p in tmp_path.iterdir() if ".tmp-" in p.name]


@pytest.mark.cpu_ok
def test_forked_child_resets_inherited_local_index_thread_lock():
    """Child must not inherit a lock held by a vanished thread at fork time."""
    from eformer.serialization import serialization as serialization_mod

    if not hasattr(os, "fork"):
        pytest.skip("os.fork is unavailable")
    lock = serialization_mod._LOCAL_INDEX_THREAD_LOCK
    held = threading.Event()
    release = threading.Event()

    def hold_lock():
        with lock:
            held.set()
            release.wait(timeout=10)

    holder = threading.Thread(target=hold_lock)
    holder.start()
    read_fd, write_fd = os.pipe()
    pid = None
    try:
        assert held.wait(timeout=5)
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            child_lock = serialization_mod._LOCAL_INDEX_THREAD_LOCK
            acquired = child_lock.acquire(timeout=0.5)
            if acquired:
                child_lock.release()
            os.write(write_fd, b"1" if acquired else b"0")
            os._exit(0)
        os.close(write_fd)
        release.set()
        assert os.read(read_fd, 1) == b"1"
    finally:
        release.set()
        holder.join(timeout=5)
        os.close(read_fd)
        if pid is not None and pid != 0:
            os.waitpid(pid, 0)
        if pid is None:
            os.close(write_fd)


@pytest.mark.cpu_ok
def test_local_path_replace_overwrites_existing_target_without_rename(tmp_path, monkeypatch):
    """The local index writer must use replace semantics, not Path.rename."""
    from pathlib import Path

    from eformer.paths import LocalPath

    src = tmp_path / "new-index.json"
    dst = tmp_path / "tensorstore_index.json"
    src.write_text("new")
    dst.write_text("old")

    def fail_rename(self, target):
        raise AssertionError("Path.rename is not an overwrite operation on Windows")

    monkeypatch.setattr(Path, "rename", fail_rename)
    LocalPath(src).replace(LocalPath(dst))
    assert dst.read_text() == "new"
    assert not src.exists()


@pytest.mark.cpu_ok
def test_index_write_creates_fresh_nested_local_checkpoint_directory(tmp_path):
    """A supplied manager need not pre-create the index's parent directory."""
    from eformer.serialization.serialization import tree_serialize_leaves

    checkpoint_dir = tmp_path / "new" / "nested" / "checkpoint"
    assert not checkpoint_dir.exists()
    tree_serialize_leaves(
        str(checkpoint_dir),
        {"weight": _ArrayMetadataOnly()},
        _NoOpManager(),
        prefix="model",
        write_index=True,
    )
    index = json.loads((checkpoint_dir / "tensorstore_index.json").read_text())
    assert index["prefixes"]["model"][0]["path"] == "model/weight"
