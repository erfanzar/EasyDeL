# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What a process reports about the code it is running.

Every Eyvan component reports one version string, so that string cannot tell an
operator whether the plane a proof polled and the agent that ran the work are the
same build. A proof has already been run against a stale agent and read as
evidence. These tests hold the property that would have caught it: the reported
identity changes when the code does, and says so rather than guessing when it
cannot be established.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from eyvan._internal.build_identity import BuildIdentity, build_identity, compute_build_identity


def _tree(root: Path, files: dict[str, str]) -> Path:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return root


def _identity(**overrides: object) -> BuildIdentity:
    fields: dict[str, object] = {
        "version": "0.0.1",
        "origin": "wheel",
        "origin_detail": "",
        "source_digest": "sha256:aa",
        "digest_detail": "",
        "file_count": 1,
        "python": "3.12.0",
        "platform": "linux-x86_64",
    }
    fields.update(overrides)
    return BuildIdentity(**fields)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- the digest


def test_identical_trees_agree(tmp_path: Path) -> None:
    """The whole point: same code, same answer, on two separate directories."""

    files = {"__init__.py": "x = 1\n", "capacity/data/shapes.yaml": "v5p-16: 2\n"}
    first = compute_build_identity(_tree(tmp_path / "a", files))
    second = compute_build_identity(_tree(tmp_path / "b", files))

    assert first.source_digest == second.source_digest
    assert first.file_count == 2
    assert first.matches(second) is True


def test_a_changed_source_file_changes_the_digest(tmp_path: Path) -> None:
    base = compute_build_identity(_tree(tmp_path / "a", {"__init__.py": "x = 1\n"}))
    changed = compute_build_identity(_tree(tmp_path / "b", {"__init__.py": "x = 2\n"}))

    assert base.source_digest != changed.source_digest
    assert base.matches(changed) is False


def test_a_changed_data_file_changes_the_digest(tmp_path: Path) -> None:
    """The accelerator catalog is data, and P1's whole criterion reads it.

    A digest over ``*.py`` alone would report two planes as the same build while
    one of them advertised a different host count per slice.
    """

    base = compute_build_identity(_tree(tmp_path / "a", {"data/shapes.yaml": "v5p-16:\n  hosts: 2\n"}))
    changed = compute_build_identity(_tree(tmp_path / "b", {"data/shapes.yaml": "v5p-16:\n  hosts: 4\n"}))

    assert base.source_digest != changed.source_digest


def test_moving_a_file_changes_the_digest(tmp_path: Path) -> None:
    """Content alone is not identity: the same bytes at another path is other code."""

    base = compute_build_identity(_tree(tmp_path / "a", {"pkg/one.py": "x = 1\n"}))
    moved = compute_build_identity(_tree(tmp_path / "b", {"pkg/two.py": "x = 1\n"}))

    assert base.source_digest != moved.source_digest


def test_a_byte_cache_is_not_part_of_the_identity(tmp_path: Path) -> None:
    """Two processes running identical code must not look different.

    ``__pycache__`` carries the compiling interpreter's magic number, so
    including it would report skew between a freshly started replica and one that
    has served a request.
    """

    plain = compute_build_identity(_tree(tmp_path / "a", {"m.py": "x = 1\n"}))
    cached = _tree(tmp_path / "b", {"m.py": "x = 1\n", "__pycache__/m.cpython-312.pyc": "\x00binary\n"})
    (cached / "stray.pyc").write_text("\x00", encoding="utf-8")

    assert compute_build_identity(cached).source_digest == plain.source_digest


def test_directory_order_does_not_matter(tmp_path: Path) -> None:
    """Two filesystems enumerate a directory differently; the digest may not."""

    forward = tmp_path / "a"
    reverse = tmp_path / "b"
    for name in ("a.py", "b.py", "c.py"):
        _tree(forward, {name: f"# {name}\n"})
    for name in ("c.py", "b.py", "a.py"):
        _tree(reverse, {name: f"# {name}\n"})

    assert compute_build_identity(forward).source_digest == compute_build_identity(reverse).source_digest


# --------------------------------------------------------------------------- unknown is an answer


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads unreadable files")
@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_an_unreadable_file_produces_no_digest_rather_than_a_partial_one(tmp_path: Path) -> None:
    """The defect this must not have.

    A digest over "everything except the awkward parts" compares equal across two
    trees that differ exactly in the file that was skipped -- which is the one
    conclusion a build digest exists to prevent. Absent, with a reason.
    """

    root = _tree(tmp_path / "a", {"readable.py": "x = 1\n", "locked.py": "secret = 1\n"})
    (root / "locked.py").chmod(0o000)
    try:
        identity = compute_build_identity(root)
    finally:
        (root / "locked.py").chmod(0o600)

    assert identity.source_digest is None
    assert "locked.py" in identity.digest_detail
    assert identity.file_count == 0


def test_an_unknown_digest_compares_to_neither_agreement_nor_skew() -> None:
    """Three answers, for the same reason the proof verdicts have three."""

    known = _identity(source_digest="sha256:aa")
    unknown = _identity(source_digest=None)

    assert known.matches(unknown) is None
    assert unknown.matches(known) is None
    assert unknown.matches(unknown) is None


def test_a_build_compares_against_a_plain_document() -> None:
    """The remote side arrives as decoded JSON, not as a dataclass."""

    known = _identity(source_digest="sha256:aa")

    assert known.matches({"source_digest": "sha256:aa"}) is True
    assert known.matches({"source_digest": "sha256:bb"}) is False
    assert known.matches({}) is None


# --------------------------------------------------------------------------- this process


def test_this_process_reports_a_complete_identity() -> None:
    """The installed package must be hashable as it actually ships."""

    identity = build_identity()

    assert identity.source_digest is not None, identity.digest_detail
    assert identity.file_count > 100
    assert identity.origin in {"wheel", "editable", "source"}
    assert identity.python.startswith(f"{sys.version_info.major}.{sys.version_info.minor}")


def test_the_identity_is_computed_once() -> None:
    """It is served from an HTTP route; the tree does not change under a process."""

    assert build_identity() is build_identity()


@pytest.mark.parametrize("which", ["plane", "agent"])
def test_building_the_app_pins_the_identity_before_the_tree_can_change(monkeypatch, which: str) -> None:
    """Pin build identity during app construction before an editable source tree can change."""

    from eyvan._internal import build_identity as module
    from eyvan.config import EyvanSettings

    calls: list[Path] = []
    real = module.compute_build_identity

    def counting(root: Path) -> BuildIdentity:
        calls.append(root)
        return real(root)

    monkeypatch.setattr(module, "compute_build_identity", counting)
    module.build_identity.cache_clear()
    try:
        if which == "plane":
            from eyvan.server.api.app import create_app

            create_app()
        else:
            from eyvan.worker.agent.app import create_agent_app

            class _Service:
                def close(self) -> None:
                    pass

            create_agent_app(_Service(), EyvanSettings(agent_shared_secret="agent-secret"))  # type: ignore[arg-type]
        assert len(calls) == 1
        build_identity()
        assert len(calls) == 1
    finally:
        module.build_identity.cache_clear()
