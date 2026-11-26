# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Workspace bundle packaging determinism and hardened extraction.

The extraction tests build hostile archives by hand rather than through
``build_workspace``, because the packager can never produce them -- the point is
that a bundle arriving from anywhere else still cannot escape its destination.
"""

from __future__ import annotations

import gzip
import io
import os
import tarfile
from pathlib import Path

import pytest

from eyvan._internal.workspace import (
    DEFAULT_WORKSPACE_EXCLUDES,
    WorkspaceError,
    build_workspace,
    extract_workspace,
)


def _tree(root: Path) -> Path:
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("")
    (root / "pkg" / "train.py").write_text("print('train')\n")
    (root / "README.md").write_text("# project\n")
    return root


def _hostile_archive(*members: tuple[tarfile.TarInfo, bytes | None]) -> bytes:
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for info, payload in members:
                if payload is None:
                    archive.addfile(info)
                else:
                    info.size = len(payload)
                    archive.addfile(info, io.BytesIO(payload))
    return raw.getvalue()


def _regular(name: str, payload: bytes = b"x") -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name=name)
    info.type = tarfile.REGTYPE
    info.mode = 0o644
    return info, payload


# --------------------------------------------------------------------------- packaging


def test_build_workspace_is_deterministic_across_calls(tmp_path: Path) -> None:
    """The digest is the content address, so identical trees must dedupe."""

    first = build_workspace(_tree(tmp_path / "a"))
    second = build_workspace(_tree(tmp_path / "b"))

    assert first.sha256 == second.sha256
    assert first.payload == second.payload
    assert first.file_count == 3


def test_build_workspace_digest_changes_with_content(tmp_path: Path) -> None:
    root = _tree(tmp_path / "a")
    before = build_workspace(root)
    (root / "pkg" / "train.py").write_text("print('changed')\n")

    assert build_workspace(root).sha256 != before.sha256


def test_build_workspace_excludes_caches_and_vcs_by_default(tmp_path: Path) -> None:
    import subprocess

    root = _tree(tmp_path / "a")
    subprocess.run(["git", "init", "-q", str(root)], check=True, capture_output=True)
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "train.cpython-313.pyc").write_bytes(b"\x00")
    (root / "pkg" / "compiled.pyc").write_bytes(b"\x00")

    archive = build_workspace(root)
    names = _member_names(archive.payload)

    assert names == ["README.md", "pkg/__init__.py", "pkg/train.py"]
    assert ".git" in DEFAULT_WORKSPACE_EXCLUDES


def test_build_workspace_does_not_ship_credentials_that_sit_in_a_project(tmp_path: Path) -> None:
    """The exclude set is documented as covering secrets, and it did not.

    A bundle is uploaded to shared artifact storage and unpacked on a leased
    multi-tenant host, so shipping ``.env`` or ``.ssh/id_rsa`` is a credential
    leak, not an untidy archive.
    """

    root = _tree(tmp_path / "a")
    (root / ".env").write_text("OPENAI_API_KEY=sk-live-1234\n")
    (root / ".ssh").mkdir()
    (root / ".ssh" / "id_rsa").write_text("-----BEGIN OPENSSH PRIVATE KEY-----\n")
    (root / "service-account.json").write_text('{"private_key": "x"}')
    (root / "tls.pem").write_text("-----BEGIN CERTIFICATE-----\n")

    names = _member_names(build_workspace(root).payload)

    assert names == ["README.md", "pkg/__init__.py", "pkg/train.py"]


def test_egg_info_contents_are_excluded_not_just_the_directory_name(tmp_path: Path) -> None:
    """``.egg-info`` was in the *file suffix* set, which matched nothing.

    The files inside are named ``PKG-INFO`` and ``SOURCES.txt`` -- neither has
    that suffix, so build output shipped anyway.
    """

    root = _tree(tmp_path / "a")
    (root / "pkg.egg-info").mkdir()
    (root / "pkg.egg-info" / "PKG-INFO").write_text("Metadata-Version: 2.1\n")
    (root / "pkg.egg-info" / "SOURCES.txt").write_text("pkg/train.py\n")

    names = _member_names(build_workspace(root).payload)

    assert names == ["README.md", "pkg/__init__.py", "pkg/train.py"]


def test_build_workspace_honours_extra_excludes(tmp_path: Path) -> None:
    root = _tree(tmp_path / "a")
    (root / "checkpoints").mkdir()
    (root / "checkpoints" / "step-1.msgpack").write_bytes(b"\x00" * 16)

    archive = build_workspace(root, extra_excludes=["checkpoints"])

    assert "checkpoints/step-1.msgpack" not in _member_names(archive.payload)


def test_build_workspace_skips_symlinks_rather_than_following_them(tmp_path: Path) -> None:
    """Following a symlink would silently copy content from outside the tree."""

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("do not ship me")
    root = _tree(tmp_path / "a")
    (root / "link.txt").symlink_to(outside / "secret.txt")

    assert "link.txt" not in _member_names(build_workspace(root).payload)


def test_build_workspace_rejects_an_empty_or_missing_root(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(WorkspaceError, match="no packageable files"):
        build_workspace(empty)
    with pytest.raises(WorkspaceError, match="not a directory"):
        build_workspace(tmp_path / "absent")


def test_build_workspace_enforces_the_size_and_file_ceilings(tmp_path: Path) -> None:
    root = _tree(tmp_path / "a")
    (root / "big.bin").write_bytes(b"\x00" * 4096)

    with pytest.raises(WorkspaceError, match="above the"):
        build_workspace(root, max_bytes=64)
    with pytest.raises(WorkspaceError, match="more than 2 files"):
        build_workspace(root, max_files=2)


def _member_names(payload: bytes) -> list[str]:
    with gzip.GzipFile(fileobj=io.BytesIO(payload), mode="rb") as compressed:
        with tarfile.open(fileobj=compressed, mode="r|") as archive:
            return sorted(info.name for info in archive)


# --------------------------------------------------------------------------- round trip


def test_round_trip_restores_content_and_the_executable_bit(tmp_path: Path) -> None:
    root = _tree(tmp_path / "a")
    script = root / "run.sh"
    script.write_text("#!/bin/sh\necho hi\n")
    script.chmod(0o755)

    destination = tmp_path / "out"
    written = extract_workspace(build_workspace(root).payload, destination)

    assert written == 4
    assert (destination / "pkg" / "train.py").read_text() == "print('train')\n"
    assert (destination / "run.sh").stat().st_mode & 0o111


# --------------------------------------------------------------------------- extraction hardening


@pytest.mark.parametrize(
    ("name", "match"),
    (
        ("../escape.txt", "traverses outside"),
        ("pkg/../../escape.txt", "traverses outside"),
        ("/etc/passwd", "not a relative path"),
        ("", "not a relative path"),
    ),
)
def test_extract_rejects_traversal_and_absolute_members(tmp_path: Path, name: str, match: str) -> None:
    payload = _hostile_archive(_regular(name, b"owned"))

    with pytest.raises(WorkspaceError, match=match):
        extract_workspace(payload, tmp_path / "out")

    assert not (tmp_path / "escape.txt").exists()


def test_extract_rejects_a_symlink_member(tmp_path: Path) -> None:
    info = tarfile.TarInfo(name="link")
    info.type = tarfile.SYMTYPE
    info.linkname = "/etc/passwd"

    with pytest.raises(WorkspaceError, match="is a link"):
        extract_workspace(_hostile_archive((info, None)), tmp_path / "out")


def test_extract_rejects_a_hardlink_member(tmp_path: Path) -> None:
    info = tarfile.TarInfo(name="hard")
    info.type = tarfile.LNKTYPE
    info.linkname = "pkg/train.py"

    with pytest.raises(WorkspaceError, match="is a link"):
        extract_workspace(_hostile_archive((info, None)), tmp_path / "out")


def test_extract_rejects_special_files(tmp_path: Path) -> None:
    info = tarfile.TarInfo(name="fifo")
    info.type = tarfile.FIFOTYPE

    with pytest.raises(WorkspaceError, match="special file"):
        extract_workspace(_hostile_archive((info, None)), tmp_path / "out")


def test_extract_refuses_to_write_through_a_preexisting_symlink(tmp_path: Path) -> None:
    """A member with no '..' can still escape if the destination already links out."""

    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "out"
    destination.mkdir()
    (destination / "pkg").symlink_to(outside, target_is_directory=True)

    with pytest.raises(WorkspaceError, match="is not empty"):
        extract_workspace(_hostile_archive(_regular("pkg/train.py", b"owned")), destination)

    assert not (outside / "train.py").exists()


def test_a_member_that_resolves_outside_the_destination_is_rejected(tmp_path: Path) -> None:
    """Direct coverage for the resolution check the empty-destination rule fronts.

    The rule above stops this class of attack earlier, but the resolution check
    is the actual defence and must not be able to rot behind it.
    """

    from eyvan._internal.workspace import _safe_member_path

    outside = tmp_path / "outside"
    outside.mkdir()
    destination = tmp_path / "out"
    destination.mkdir()
    (destination / "pkg").symlink_to(outside, target_is_directory=True)

    with pytest.raises(WorkspaceError, match="escapes the workspace directory"):
        _safe_member_path("pkg/train.py", destination)


def test_extract_refuses_a_destination_that_already_has_content(tmp_path: Path) -> None:
    """Extraction never removes files, so reuse leaves stale modules importable.

    The destination becomes ``sys.path[0]`` for the callable. A module left by a
    previous bundle under a name the current bundle does not ship would keep
    resolving, silently changing what executes.
    """

    destination = tmp_path / "out"
    destination.mkdir()
    (destination / "train.py").write_text("raise SystemExit('stale bundle')\n")

    with pytest.raises(WorkspaceError, match="is not empty"):
        extract_workspace(_hostile_archive(_regular("other.py", b"ok")), destination)

    assert (destination / "train.py").read_text() == "raise SystemExit('stale bundle')\n"


def test_extract_will_not_write_through_a_hardlink_planted_in_the_destination(tmp_path: Path) -> None:
    """``Path.resolve`` cannot see hardlinks; only O_EXCL stops this one.

    A hardlink is indistinguishable from the original directory entry, so every
    path-resolution guard in this module passes it. ``open(target, "wb")`` would
    then truncate whatever the link points at, outside the workspace entirely.
    """

    from eyvan._internal.workspace import _create_exclusive

    secret = tmp_path / "outside" / "credentials"
    secret.parent.mkdir()
    secret.write_text("original contents")
    destination = tmp_path / "out"
    destination.mkdir()
    planted = destination / "train.py"
    os.link(secret, planted)

    with pytest.raises(WorkspaceError, match="already contains"):
        _create_exclusive(planted, 0o644)

    assert secret.read_text() == "original contents"


def test_extract_caps_total_expanded_bytes(tmp_path: Path) -> None:
    """A small archive must not be able to fill the host."""

    payload = _hostile_archive(_regular("big.bin", b"\x00" * 8192))

    with pytest.raises(WorkspaceError, match="expands beyond"):
        extract_workspace(payload, tmp_path / "out", max_total_bytes=1024)


def test_extract_caps_the_member_count(tmp_path: Path) -> None:
    payload = _hostile_archive(*(_regular(f"f{index}.txt") for index in range(8)))

    with pytest.raises(WorkspaceError, match="more than 3 entries"):
        extract_workspace(payload, tmp_path / "out", max_files=3)


def test_directory_members_count_against_the_cap(tmp_path: Path) -> None:
    """Directories carry no payload, so neither cap saw them.

    A tiny archive of nothing but directory entries therefore created unbounded
    inodes on a leased host without tripping the file count or the byte ceiling.
    """

    directories = []
    for index in range(8):
        info = tarfile.TarInfo(name=f"d{index}")
        info.type = tarfile.DIRTYPE
        directories.append((info, None))
    payload = _hostile_archive(*directories)

    with pytest.raises(WorkspaceError, match="more than 3 entries"):
        extract_workspace(payload, tmp_path / "out", max_files=3, max_total_bytes=1)


def _git_repo(root: Path, gitignore: str) -> None:
    """Initialize a real repository; git's own matching is what is under test."""

    import subprocess

    (root / ".gitignore").write_text(gitignore, encoding="utf-8")
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "test@example.com"],
        ["git", "config", "user.name", "test"],
        ["git", "add", "-A"],
        ["git", "commit", "-q", "-m", "initial"],
    ):
        subprocess.run(command, cwd=root, check=True, capture_output=True)


def _arcnames(archive) -> set[str]:
    import gzip as _gzip
    import io as _io
    import tarfile as _tarfile

    raw = _gzip.decompress(archive.payload)
    with _tarfile.open(fileobj=_io.BytesIO(raw)) as opened:
        return set(opened.getnames())


def test_a_gitignored_directory_is_not_shipped(tmp_path) -> None:
    """Without this a normal project ships its datasets and checkpoints.

    The 64 MB ceiling exists to keep a bundle to *source*. A repository that
    gitignores ``data/`` and ``checkpoints/`` -- which is every training
    repository -- would otherwise blow through it, and the resulting error names
    a size rather than the directories the user already declared they did not
    want.
    """

    (tmp_path / "train.py").write_text("print('hi')", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "shard-0.bin").write_bytes(b"x" * 4096)
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "step-100.msgpack").write_bytes(b"y" * 4096)
    _git_repo(tmp_path, "data/\ncheckpoints/\n")

    names = _arcnames(build_workspace(tmp_path))

    assert "train.py" in names
    assert not any(name.startswith("data/") for name in names), names
    assert not any(name.startswith("checkpoints/") for name in names), names


def test_an_untracked_but_unignored_file_is_shipped(tmp_path) -> None:
    """A file written since the last commit is still part of the workspace."""

    (tmp_path / "train.py").write_text("print('hi')", encoding="utf-8")
    _git_repo(tmp_path, "data/\n")
    (tmp_path / "new_module.py").write_text("VALUE = 1", encoding="utf-8")

    assert "new_module.py" in _arcnames(build_workspace(tmp_path))


def test_git_cannot_widen_what_leaves_the_machine(tmp_path) -> None:
    """A repository may *track* a secret; the bundle still must not carry it.

    ``.gitignore`` selects candidates, the mandatory excludes decide what ships.
    Reversing that order would let any repo with a committed ``.env`` publish it
    to shared artifact storage and unpack it on a multi-tenant host.
    """

    (tmp_path / "train.py").write_text("print('hi')", encoding="utf-8")
    (tmp_path / ".env").write_text("API_KEY=super-secret", encoding="utf-8")
    (tmp_path / "id_rsa").write_text("-----BEGIN PRIVATE KEY-----", encoding="utf-8")
    _git_repo(tmp_path, "")

    names = _arcnames(build_workspace(tmp_path))

    assert "train.py" in names
    assert ".env" not in names
    assert "id_rsa" not in names


def test_a_directory_that_is_not_a_repository_still_packages(tmp_path) -> None:
    """Not every workspace is a git checkout; that must not become an error."""

    (tmp_path / "train.py").write_text("print('hi')", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("hello", encoding="utf-8")

    assert _arcnames(build_workspace(tmp_path)) == {"train.py", "notes.txt"}


def test_opting_out_of_gitignore_ships_the_ignored_files(tmp_path) -> None:
    """The escape hatch has to actually work, or it is a lie in the signature."""

    (tmp_path / "train.py").write_text("print('hi')", encoding="utf-8")
    (tmp_path / "generated").mkdir()
    (tmp_path / "generated" / "weights.txt").write_text("data", encoding="utf-8")
    _git_repo(tmp_path, "generated/\n")

    names = _arcnames(build_workspace(tmp_path, respect_gitignore=False))

    assert "generated/weights.txt" in names


def test_an_ignored_workspace_requires_explicit_gitignore_opt_out(tmp_path) -> None:
    """An ignored root must not bypass nested exclusions without consent."""

    (tmp_path / "train.py").write_text("print('hi')", encoding="utf-8")
    _git_repo(tmp_path, "experiments/\n")
    root = tmp_path / "experiments" / "run-1"
    (root / "lib").mkdir(parents=True)
    (root / "lib" / "__init__.py").write_text("VALUE = 1", encoding="utf-8")
    (root / ".env").write_text("API_KEY=super-secret", encoding="utf-8")
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "lib.cpython-311.pyc").write_bytes(b"\0")

    (root / ".gitignore").write_text("tokens.txt\nprivate-config.json\n", encoding="utf-8")
    (root / "tokens.txt").write_text("not-a-real-token", encoding="utf-8")
    (root / "private-config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(WorkspaceError, match=r"repository ignores.*respect_gitignore=False"):
        build_workspace(root)

    names = _arcnames(build_workspace(root, respect_gitignore=False))
    assert names == {".gitignore", "lib/__init__.py", "tokens.txt", "private-config.json"}, names


def test_ignore_rules_inside_a_shipped_root_still_apply(tmp_path) -> None:
    """Git's nested exclusions still apply inside a non-ignored root."""

    root = tmp_path / "project"
    root.mkdir()
    (root / "train.py").write_text("print('hi')", encoding="utf-8")
    (root / "outputs").mkdir()
    (root / "outputs" / "metrics.json").write_text("{}", encoding="utf-8")
    _git_repo(root, "outputs/\n")

    assert _arcnames(build_workspace(root)) == {".gitignore", "train.py"}


def test_a_root_whose_every_file_is_excluded_says_so(tmp_path) -> None:
    """ "No packageable files" beside files on disk sent the user looking for a missing file."""

    (tmp_path / ".env").write_text("API_KEY=super-secret", encoding="utf-8")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"\0")

    with pytest.raises(WorkspaceError, match=r"has 2 file\(s\) on disk but none is packageable"):
        build_workspace(tmp_path)
