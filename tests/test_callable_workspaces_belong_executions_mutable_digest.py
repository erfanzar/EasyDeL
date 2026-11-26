# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Callable workspaces belong to executions, not a mutable digest cache."""

from __future__ import annotations

import hashlib
import hmac
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from eyvan._internal.workspace import build_workspace
from eyvan.worker.runtime import callable_protocol as protocol


@pytest.fixture(autouse=True)
def restore_process_state():
    """Materialization changes cwd and import paths; isolate every local proof."""

    cwd = Path.cwd()
    paths = sys.path[:]
    anchor = protocol._WORKSPACE_BASE_ANCHOR
    try:
        yield
    finally:
        os.chdir(cwd)
        sys.path[:] = paths
        protocol._WORKSPACE_BASE_ANCHOR = anchor


def _bundle(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    return build_workspace(source, respect_gitignore=False)


def _materialize(bundle, base: Path) -> Path:
    return protocol._materialize_workspace(
        bundle.payload,
        sha256=bundle.sha256,
        signature=None,
        signing_key=None,
        environment={"EYVAN_WORKSPACE_DIR": str(base)},
    )


def test_materialization_does_not_prune_a_live_completed_workspace(tmp_path: Path) -> None:
    """A real child retains its shipped module while a different bundle lands."""

    bundle = _bundle(tmp_path)
    base = tmp_path / "workspaces"
    base.mkdir()
    active = base / "000000000000"
    active.mkdir()
    (active / "helper.py").write_text("VALUE = 7\n", encoding="utf-8")
    (active / ".eyvan-extraction-complete").write_text("old-digest", encoding="utf-8")
    for index in range(1, 6):
        workspace = base / f"{index:012d}"
        workspace.mkdir()
        (workspace / ".eyvan-extraction-complete").write_text("other-digest", encoding="utf-8")
    script = (
        "from pathlib import Path; import sys; "
        "print('ready', flush=True); sys.stdin.readline(); "
        "print(Path('helper.py').read_text(), end='', flush=True)"
    )
    with subprocess.Popen(
        [sys.executable, "-B", "-c", script],
        cwd=active,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as child:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        try:
            incoming = _materialize(bundle, base)
            stdout, stderr = child.communicate("continue\n", timeout=10)
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate(timeout=10)
    assert (incoming / "helper.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert child.returncode == 0, stderr
    assert stdout == "VALUE = 7\n"
    assert active.is_dir()


@pytest.mark.parametrize("interleaving", ["before-marker", "before-extract"])
def test_concurrent_materializations_return_complete_trees(tmp_path: Path, monkeypatch, interleaving: str) -> None:
    """Control scheduling only; each contender uses the real hardened extractor."""

    bundle = _bundle(tmp_path)
    base = tmp_path / "workspaces"
    base.mkdir()
    extract = protocol._workspace_extractor()
    first_paused = threading.Event()
    second_paused = threading.Event()
    first_finished = threading.Event()
    release_first = threading.Event()
    release_second = threading.Event()

    def timed_extract(payload: bytes, destination: Path) -> int:
        if threading.current_thread().name.endswith("_0"):
            if interleaving == "before-marker":
                count = extract(payload, destination)
                first_paused.set()
                assert release_first.wait(10)
                return count
            first_paused.set()
            assert release_first.wait(10)
            return extract(payload, destination)
        second_paused.set()
        assert release_second.wait(10)
        return extract(payload, destination)

    monkeypatch.setattr(protocol, "_workspace_extractor", lambda: timed_extract)

    def first_attempt() -> Path:
        try:
            return _materialize(bundle, base)
        finally:
            first_finished.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(first_attempt)
        try:
            assert first_paused.wait(10)
            second = pool.submit(_materialize, bundle, base)
            assert second_paused.wait(10)
            release_first.set()
            assert first_finished.wait(10)
        finally:
            release_first.set()
            release_second.set()
        results = [first.result(timeout=10), second.result(timeout=10)]
    for workspace in results:
        assert (workspace / "helper.py").read_text(encoding="utf-8") == "VALUE = 1\n"
        assert (workspace / ".eyvan-extraction-complete").read_text(encoding="utf-8") == bundle.sha256


@pytest.mark.parametrize("mutation", ["rewrite", "delete", "module-import"])
def test_signed_retry_uses_submitted_files_after_failed_mutation(tmp_path: Path, mutation: str) -> None:
    """Two actual signed callable executions in fresh child processes share a base."""

    import json

    bundle = _bundle(tmp_path)
    attempt_marker = str(tmp_path / "first-attempt-failed")

    def run():
        from pathlib import Path

        helper = Path("helper.py")
        if mutation == "module-import":
            import helper as shipped_helper

            initial = f"VALUE = {shipped_helper.VALUE}\n"
        else:
            initial = helper.read_text(encoding="utf-8")
        marker = Path(attempt_marker)
        if not marker.exists():
            marker.write_text("failed", encoding="utf-8")
            if mutation == "delete":
                helper.unlink()
            else:
                helper.write_text("VALUE = 999\n", encoding="utf-8")
            raise RuntimeError("intentional first-attempt failure")
        return initial

    key = b"writer-110-local-test-key"
    artifact = protocol.dump_callable(run, signing_key=key)
    callable_path = tmp_path / "callable.pkl"
    workspace_path = tmp_path / "workspace.tar.gz"
    result_path = tmp_path / "result.pkl"
    callable_path.write_bytes(artifact.payload)
    workspace_path.write_bytes(bundle.payload)
    arguments = [
        "--uri",
        str(callable_path),
        "--sha256",
        artifact.sha256,
        "--python",
        artifact.python,
        "--signature",
        artifact.signature,
        "--workspace-uri",
        str(workspace_path),
        "--workspace-sha256",
        bundle.sha256,
        "--workspace-signature",
        hmac.new(key, bundle.payload, hashlib.sha256).hexdigest(),
        "--result-uri",
        str(result_path),
    ]
    environment = {"EYVAN_WORKSPACE_DIR": str(tmp_path / "workspaces"), "EYVAN_CALLABLE_SIGNING_KEY": key.decode()}
    # The transfer uses local files only. No URI in this test can contact a provider.
    script = (
        "import json; "
        "from eyvan.worker.runtime.callable_protocol import execute_callable, UriArtifactTransfer; "
        f"execute_callable(json.loads({json.dumps(arguments)!r}), transfer=UriArtifactTransfer(None, {{}}), "
        f"environment=json.loads({json.dumps(environment)!r}))"
    )
    checkout = Path(__file__).resolve().parent.parent
    assert Path(protocol.__file__).resolve().is_relative_to(checkout)
    first = subprocess.run(
        [sys.executable, "-B", "-c", script], cwd=checkout, capture_output=True, text=True, timeout=20
    )
    assert first.returncode != 0
    assert "intentional first-attempt failure" in first.stderr
    retry = subprocess.run(
        [sys.executable, "-B", "-c", script], cwd=checkout, capture_output=True, text=True, timeout=20
    )
    assert retry.returncode == 0, retry.stderr
    assert protocol.load_result(result_path.read_bytes(), signing_key=key) == "VALUE = 1\n"


def test_concurrent_cold_base_creation_succeeds(tmp_path: Path, monkeypatch) -> None:
    """Both callers see a missing base before either can create it."""

    base = tmp_path / "workspaces"
    barrier = threading.Barrier(2)
    mkdir = Path.mkdir

    def simultaneous_mkdir(path: Path, *args, **kwargs):
        if path == base:
            barrier.wait(timeout=10)
        return mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", simultaneous_mkdir)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(protocol._ensure_private_dir, base) for _ in range(2)]
        for future in futures:
            future.result(timeout=10)
    assert base.is_dir()
    assert base.stat().st_mode & 0o777 == 0o700


def test_failed_extraction_removes_only_its_owned_tree(tmp_path: Path) -> None:
    """Real invalid gzip is rejected; a pre-existing sibling remains untouched."""

    from gzip import BadGzipFile

    base = tmp_path / "workspaces"
    sibling = base / "live"
    sibling.mkdir(parents=True)
    (sibling / "helper.py").write_text("keep", encoding="utf-8")
    payload = b"not a gzip archive"
    with pytest.raises(BadGzipFile):
        protocol._materialize_workspace(
            payload,
            sha256=hashlib.sha256(payload).hexdigest(),
            signature=None,
            signing_key=None,
            environment={"EYVAN_WORKSPACE_DIR": str(base)},
        )
    assert list(base.iterdir()) == [sibling]
    assert (sibling / "helper.py").read_text(encoding="utf-8") == "keep"
