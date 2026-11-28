# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Writable execution trees must never be reused as verified bundle content."""

from __future__ import annotations

import hashlib
import hmac
import io
import os
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from eyvan._internal.workspace import WorkspaceError, build_workspace
from eyvan.worker.runtime import callable_protocol, workspace_entry


def _archive(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    return build_workspace(root)


def _import_result(workspace: Path) -> str:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import helper, importlib.util; print(helper.VALUE, importlib.util.find_spec('shadow') is None)",
        ],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.stdout.strip()


@pytest.mark.parametrize("path", ["command", "callable"])
@pytest.mark.parametrize("signed", [False, True])
def test_a_verified_bundle_gets_a_fresh_execution_tree(tmp_path: Path, monkeypatch, path: str, signed: bool) -> None:
    bundle = _archive(tmp_path)
    base = tmp_path / "executions"
    key = b"local-test-key" if signed else None
    signature = hmac.new(key, bundle.payload, hashlib.sha256).hexdigest() if key else None
    original_cwd = Path.cwd()
    original_path = list(sys.path)
    original_pythonpath = os.environ.get("PYTHONPATH")
    monkeypatch.setattr(callable_protocol, "_WORKSPACE_BASE_ANCHOR", None)
    monkeypatch.setattr(callable_protocol, "_ACTIVE_WORKSPACE_PYTHON_PATHS", ())

    def materialize() -> Path:
        common = dict(sha256=bundle.sha256, signature=signature, signing_key=key)
        if path == "command":
            return workspace_entry.materialize(bundle.payload, base=base, **common)
        return callable_protocol._materialize_workspace(
            bundle.payload, environment={"EYVAN_WORKSPACE_DIR": str(base)}, **common
        )

    try:
        first = materialize()
        assert _import_result(first) == "1 True"
        (first / "helper.py").write_text("VALUE = 999\n", encoding="utf-8")
        (first / "shadow.py").write_text("VALUE = 999\n", encoding="utf-8")
        second = materialize()
        # A new process exercises real import resolution, not sys.modules reuse.
        assert _import_result(second) == "1 True"
        assert second != first
        assert (first / "helper.py").read_text(encoding="utf-8") == "VALUE = 999\n"
        assert (first / "shadow.py").is_file(), "an active sibling's writable tree must not be removed"
        assert (second / workspace_entry.COMPLETE_MARKER).read_text(encoding="utf-8") == bundle.sha256
    finally:
        os.chdir(original_cwd)
        sys.path[:] = original_path
        if original_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = original_pythonpath


def test_concurrent_command_invocations_own_disjoint_trees(tmp_path: Path) -> None:
    bundle = _archive(tmp_path)
    base = tmp_path / "executions"

    def materialize(_: int) -> Path:
        return workspace_entry.materialize(
            bundle.payload, sha256=bundle.sha256, signature=None, signing_key=None, base=base
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        trees = list(pool.map(materialize, range(4)))
    assert len(set(trees)) == 4
    (trees[0] / "helper.py").write_text("VALUE = 999\n", encoding="utf-8")
    for tree in trees[1:]:
        assert _import_result(tree) == "1 True"
        assert tree.stat().st_mode & 0o777 == 0o700


def test_bad_digest_or_signature_never_creates_an_execution_tree(tmp_path: Path) -> None:
    bundle = _archive(tmp_path)
    base = tmp_path / "executions"
    with pytest.raises(ValueError, match="content-addressed digest"):
        workspace_entry.materialize(bundle.payload, sha256="0" * 64, signature=None, signing_key=None, base=base)
    with pytest.raises(ValueError, match="signature mismatch"):
        workspace_entry.materialize(
            bundle.payload, sha256=bundle.sha256, signature=None, signing_key=b"local-test-key", base=base
        )
    assert not base.exists()


def test_failed_extraction_removes_only_its_new_tree(tmp_path: Path) -> None:
    bundle = _archive(tmp_path)
    base = tmp_path / "executions"
    existing = workspace_entry.materialize(
        bundle.payload, sha256=bundle.sha256, signature=None, signing_key=None, base=base
    )
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        regular = tarfile.TarInfo("partial.py")
        regular.size = 1
        archive.addfile(regular, io.BytesIO(b"x"))
        link = tarfile.TarInfo("forbidden-link")
        link.type = tarfile.SYMTYPE
        link.linkname = "partial.py"
        archive.addfile(link)
    payload = buffer.getvalue()
    with pytest.raises(WorkspaceError, match="link"):
        workspace_entry.materialize(
            payload, sha256=hashlib.sha256(payload).hexdigest(), signature=None, signing_key=None, base=base
        )
    assert list(base.iterdir()) == [existing]
    assert _import_result(existing) == "1 True"
