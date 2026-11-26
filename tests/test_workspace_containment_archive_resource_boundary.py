# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Workspace containment and archive resource-boundary regressions."""

from __future__ import annotations

import gzip
import hashlib
import io
import struct
import subprocess
import tarfile
from pathlib import Path

import pytest

from eyvan._internal import workspace


def _indexed_tree(root: Path) -> Path:
    """Give real Git a synthetic index without staging or committing anything."""

    root.mkdir()
    (root / "pkg").mkdir()
    (root / "pkg" / "module.py").write_bytes(b"ORIGINAL_PROJECT_BYTES")
    (root / "keep.py").write_bytes(b"KEEP_PROJECT_BYTES")
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
    entries = []
    for name in ("keep.py", "pkg/module.py"):
        path = root / name
        data = path.read_bytes()
        encoded = name.encode()
        oid = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).digest()
        entry = struct.pack(">10I", 0, 0, 0, 0, 0, 0, 0o100644, 0, 0, len(data)) + oid
        entry += struct.pack(">H", len(encoded)) + encoded + b"\0"
        entry += b"\0" * (-len(entry) % 8)
        entries.append(entry)
    index = b"DIRC" + struct.pack(">II", 2, len(entries)) + b"".join(entries)
    (root / ".git" / "index").write_bytes(index + hashlib.sha1(index).digest())
    listed = subprocess.run(["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True).stdout.split(
        b"\0"
    )
    assert b"pkg/module.py" in listed
    return root


def _contents(payload: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        contents = {}
        for member in archive:
            handle = archive.extractfile(member)
            assert handle is not None
            contents[member.name] = handle.read()
        return contents


def _outside(tmp_path: Path) -> Path:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "module.py").write_bytes(b"SYNTHETIC_PRIVATE_VALUE")
    return outside


def test_git_indexed_descendant_of_symlink_is_not_packaged(tmp_path: Path) -> None:
    root = _indexed_tree(tmp_path / "project")
    outside = _outside(tmp_path)
    (root / "leaf.py").symlink_to(outside / "module.py")
    assert _contents(workspace.build_workspace(root).payload) == {
        "keep.py": b"KEEP_PROJECT_BYTES",
        "pkg/module.py": b"ORIGINAL_PROJECT_BYTES",
    }
    (root / "pkg").rename(root / "original-pkg")
    (root / "pkg").symlink_to(outside, target_is_directory=True)

    contents = _contents(workspace.build_workspace(root).payload)

    assert "pkg/module.py" not in contents
    assert b"SYNTHETIC_PRIVATE_VALUE" not in contents.values()
    assert contents["keep.py"] == b"KEEP_PROJECT_BYTES"
    assert "leaf.py" not in contents


@pytest.mark.parametrize("replacement", ["parent", "leaf"])
def test_packaging_refuses_symlink_replacement_after_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, replacement: str
) -> None:
    root = _indexed_tree(tmp_path / "project")
    outside = _outside(tmp_path)
    collect = workspace._collect

    def replace_after_collection(*args, **kwargs):
        members = collect(*args, **kwargs)
        if replacement == "parent":
            (root / "pkg").rename(root / "original-pkg")
            (root / "pkg").symlink_to(outside, target_is_directory=True)
        else:
            (root / "pkg" / "module.py").rename(root / "pkg" / "original.py")
            (root / "pkg" / "module.py").symlink_to(outside / "module.py")
        return members

    monkeypatch.setattr(workspace, "_collect", replace_after_collection)
    with pytest.raises(workspace.WorkspaceError, match="safely read"):
        workspace.build_workspace(root)


def _metadata_archive(kind: str, metadata_bytes: int) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as archive:
        if kind == "pax":
            member = tarfile.TarInfo("tiny.py")
            member.pax_headers = {"comment": "x" * metadata_bytes}
        else:
            # Redundant './' components are safe after path normalization, but
            # tarfile must allocate the entire GNU long-name extension first.
            name = b"./" * (metadata_bytes // 2) + b"tiny.py\0"
            extension = tarfile.TarInfo("././@LongLink")
            extension.type = tarfile.GNUTYPE_LONGNAME
            extension.size = len(name)
            archive.addfile(extension, io.BytesIO(name))
            member = tarfile.TarInfo("tiny.py")
        member.size = 1
        archive.addfile(member, io.BytesIO(b"x"))
    return gzip.compress(raw.getvalue(), mtime=0)


@pytest.mark.parametrize("kind", ["pax", "gnu"])
def test_extension_metadata_cannot_bypass_decompressed_stream_budget(tmp_path: Path, kind: str) -> None:
    payload = _metadata_archive(kind, 1024 * 1024)
    assert len(payload) < 4096
    destination = tmp_path / "out"

    with pytest.raises(workspace.WorkspaceError, match="expands beyond"):
        workspace.extract_workspace(payload, destination, max_total_bytes=1, max_files=1)

    assert not (destination / "tiny.py").exists()


@pytest.mark.parametrize("kind", ["pax", "gnu"])
def test_small_extension_metadata_and_one_byte_file_remain_supported(tmp_path: Path, kind: str) -> None:
    destination = tmp_path / "out"
    assert workspace.extract_workspace(_metadata_archive(kind, 64), destination, max_total_bytes=1, max_files=1) == 1
    assert (destination / "tiny.py").read_bytes() == b"x"


def _nested_archive(*members: tuple[str, bytes | None]) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for name, content in members:
            info = tarfile.TarInfo(name)
            if content is None:
                info.type = tarfile.DIRTYPE
                archive.addfile(info)
            else:
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))
    return gzip.compress(raw.getvalue(), mtime=0)


@pytest.mark.parametrize("directory_member", [False, True])
def test_implicit_directories_cannot_bypass_entry_limit(tmp_path: Path, directory_member: bool) -> None:
    name = "/".join(f"d{index}" for index in range(40))
    member = (name, None) if directory_member else (name + "/tiny.py", b"x")
    destination = tmp_path / "out"

    with pytest.raises(workspace.WorkspaceError, match="more than 1 entries"):
        workspace.extract_workspace(_nested_archive(member), destination, max_files=1, max_total_bytes=1)

    assert not list(destination.iterdir())


@pytest.mark.parametrize("explicit_parent", [False, True])
def test_shared_implicit_directories_count_once(tmp_path: Path, explicit_parent: bool) -> None:
    members = [("a/b/one.py", b"x"), ("a/b/two.py", b"y")]
    if explicit_parent:
        members.insert(0, ("a", None))
    destination = tmp_path / "out"

    assert workspace.extract_workspace(_nested_archive(*members), destination, max_files=4, max_total_bytes=2) == 2
    assert len(list(destination.rglob("*"))) == 4
    assert (destination / "a" / "b" / "one.py").read_bytes() == b"x"
    assert (destination / "a" / "b" / "two.py").read_bytes() == b"y"


def test_shallow_file_with_parent_fits_two_entry_budget(tmp_path: Path) -> None:
    destination = tmp_path / "out"
    assert (
        workspace.extract_workspace(_nested_archive(("a/tiny.py", b"x")), destination, max_files=2, max_total_bytes=1)
        == 1
    )
    assert len(list(destination.rglob("*"))) == 2


def test_whole_stream_budget_includes_bytes_after_tar_end(tmp_path: Path) -> None:
    raw = gzip.decompress(_nested_archive(("tiny.py", b"x"))) + b"\0" * (1024 * 1024)
    payload = gzip.compress(raw, mtime=0)
    with pytest.raises(workspace.WorkspaceError, match="decompressed stream expands beyond"):
        workspace.extract_workspace(payload, tmp_path / "out", max_files=1, max_total_bytes=1)


def test_explicit_stream_budget_supports_metadata_without_widening_content_limit(tmp_path: Path) -> None:
    payload = _metadata_archive("pax", 16 * 1024)
    destination = tmp_path / "out"
    assert (
        workspace.extract_workspace(payload, destination, max_files=1, max_total_bytes=1, max_stream_bytes=64 * 1024)
        == 1
    )
    assert (destination / "tiny.py").read_bytes() == b"x"

    with pytest.raises(workspace.WorkspaceError, match="expands beyond the 1-byte limit"):
        workspace.extract_workspace(
            _nested_archive(("tiny.py", b"xy")), tmp_path / "too-big", max_total_bytes=1, max_stream_bytes=64 * 1024
        )
