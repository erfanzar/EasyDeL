# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Directory enumeration failures must not certify a partial package tree."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from eyvan._internal.build_identity import BuildIdentity, compute_build_identity


def _tree(root: Path, *, hidden: bool) -> Path:
    root.mkdir()
    (root / "readable.py").write_text("x = 1\n", encoding="utf-8")
    if hidden:
        (root / "blocked").mkdir()
        (root / "blocked" / "hidden.py").write_text("x = 2\n", encoding="utf-8")
    return root


def _assert_unknown(identity: BuildIdentity, control: BuildIdentity) -> None:
    assert identity.source_digest is None, identity.as_dict()
    assert identity.file_count == 0
    assert identity.digest_detail
    assert identity.matches(control) is None
    assert control.matches(identity) is None


def test_scandir_failure_cannot_certify_the_visible_subset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _tree(tmp_path / "package", hidden=True)
    control = compute_build_identity(_tree(tmp_path / "control", hidden=False))
    complete = compute_build_identity(root)
    assert complete.file_count == 2
    assert complete.matches(control) is False
    real_scandir = os.scandir

    def failing_scandir(path):
        if Path(path) == root / "blocked":
            raise PermissionError("credential-sensitive directory details")
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", failing_scandir)
    identity = compute_build_identity(root)

    _assert_unknown(identity, control)
    assert "credential-sensitive" not in identity.digest_detail
    assert "PermissionError" in identity.digest_detail


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX directory permissions")
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root can enumerate locked directories")
def test_real_unreadable_directory_cannot_certify_the_visible_subset(tmp_path: Path) -> None:
    root = _tree(tmp_path / "package", hidden=True)
    control = compute_build_identity(_tree(tmp_path / "control", hidden=False))
    before = compute_build_identity(root)
    (root / "blocked" / "hidden.py").write_text("x = 3\n", encoding="utf-8")
    after = compute_build_identity(root)
    assert before.matches(after) is False
    assert after.matches(control) is False
    blocked = root / "blocked"
    blocked.chmod(0o000)
    try:
        with pytest.raises(PermissionError):
            list(blocked.iterdir())
        identity = compute_build_identity(root)
    finally:
        blocked.chmod(0o700)

    _assert_unknown(identity, control)


def test_missing_root_does_not_certify_an_empty_package(tmp_path: Path) -> None:
    control = compute_build_identity(_tree(tmp_path / "control", hidden=False))
    identity = compute_build_identity(tmp_path / "missing")

    _assert_unknown(identity, control)
    assert "FileNotFoundError" in identity.digest_detail


def test_ignored_directory_is_not_scanned(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _tree(tmp_path / "package", hidden=False)
    control = compute_build_identity(root)
    ignored = root / "__pycache__"
    ignored.mkdir()
    (ignored / "readable.cpython-311.pyc").write_bytes(b"cache")
    real_scandir = os.scandir

    def failing_scandir(path):
        if Path(path) == ignored:
            raise PermissionError("ignored byte cache")
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", failing_scandir)
    identity = compute_build_identity(root)

    assert identity.file_count == 1
    assert identity.digest_detail == ""
    assert identity.matches(control) is True
