# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""What code a running Eyvan process is actually made of.

Every component of this system reports the same version string, because they
are one package and that string is set once per release. So "0.0.1" answers
nothing an operator asks during an upgrade: whether the plane a proof polled is
the plane the wheel was built for, whether an agent still runs yesterday's code,
whether the head and its workers agree. A hardware proof has already been run
against a stale agent and read as evidence.

The answer that does distinguish them is the content of the installed package.
:func:`build_identity` hashes every file the imported ``eyvan`` package is made
of -- Python sources and the data they read, which is why the accelerator
catalog counts -- and reports one digest for the lot. Two processes that report
the same digest are running the same code; two that differ are not, whatever
their versions say.

**A digest is either complete or absent.** A file that cannot be read makes the
whole digest ``None`` with a stated reason, because a digest computed over
"everything except the parts that were awkward" would compare equal across trees
that differ exactly there. That is the failure this exists to prevent, so it is
not one this module is allowed to have.

The walk is bounded and cached: a package tree is a few hundred small files, it
is read once per process, and nothing here touches the network or the database.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import typing as tp
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from pathlib import Path

from eyvan._version import __version__ as _VERSION

#: Directories and suffixes that are build products of the source beside them.
#: Excluded because a byte-cache differs between two processes running identical
#: code -- it carries the compiling interpreter's magic number and mtimes -- so
#: including it would report skew that does not exist.
_IGNORED_DIRECTORIES = frozenset({"__pycache__"})
_IGNORED_SUFFIXES = (".pyc", ".pyo")

#: Read size for hashing. Package files are small; this only bounds the memory
#: of an unexpectedly large one.
_CHUNK_BYTES = 1 << 20

#: A tree larger than this is not the package this was written for -- most
#: likely a working directory that swallowed a virtualenv or a dataset. Refusing
#: is better than spending minutes hashing it inside a request.
_MAX_FILES = 20_000


class Origin(tp.NamedTuple):
    """Where the imported package came from, and how confidently that is known."""

    #: ``wheel``, ``editable``, ``source`` or ``unknown``.
    kind: str
    #: The sentence explaining ``kind`` when it is not simply "installed".
    detail: str


@dataclass(frozen=True, slots=True)
class BuildIdentity:
    """One process's statement of what it is running.

    Attributes:
        version: The package version, which does not distinguish two builds.
        origin: Whether the code is an installed wheel, an editable checkout, or
            a source tree that was never installed.
        source_digest: ``sha256:<hex>`` over every file of the package, or
            ``None`` when it could not be computed in full.
        digest_detail: Why ``source_digest`` is ``None``, or ``""``.
        file_count: How many files the digest covers, or ``0``.
        python: The running interpreter's version.
        platform: The host platform string.
    """

    version: str
    origin: str
    origin_detail: str
    source_digest: str | None
    digest_detail: str
    file_count: int
    python: str
    platform: str

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "origin": self.origin,
            "origin_detail": self.origin_detail,
            "source_digest": self.source_digest,
            "digest_detail": self.digest_detail,
            "file_count": self.file_count,
            "python": self.python,
            "platform": self.platform,
        }

    def matches(self, other: BuildIdentity | tp.Mapping[str, object]) -> bool | None:
        """Return whether two builds are the same code, or ``None`` if unknown.

        Three answers rather than two, for the reason the proof verdicts have
        three: a build whose digest could not be computed is not evidence of
        agreement *or* of skew, and reporting either would be an invention. A
        caller that treats ``None`` as ``True`` has reintroduced exactly the
        stale-agent failure this module exists to catch.
        """

        mine = self.source_digest
        theirs = other.source_digest if isinstance(other, BuildIdentity) else other.get("source_digest")
        if not mine or not isinstance(theirs, str) or not theirs:
            return None
        return mine == theirs


def _package_root() -> Path:
    """Return the directory the ``eyvan`` package is imported from."""

    return Path(__file__).resolve().parent.parent


def _origin(root: Path) -> Origin:
    """Classify how the package at ``root`` was installed.

    ``direct_url.json`` is the packaging standard's own record of an editable
    install, so it is read rather than guessed at from path shapes -- a checkout
    that happens to sit beside ``site-packages`` is not thereby editable.
    """

    try:
        distribution = metadata.distribution("eyvan")
    except metadata.PackageNotFoundError:
        return Origin("source", f"no installed distribution provides this package; imported from {root}")
    except Exception as error:  # pragma: no cover - metadata backends vary
        return Origin("unknown", f"the installed distribution could not be read: {error}")
    try:
        raw = distribution.read_text("direct_url.json")
    except Exception:  # pragma: no cover - absent metadata is the common case
        raw = None
    if raw:
        try:
            direct = json.loads(raw)
        except ValueError:
            return Origin("unknown", "the distribution records an unreadable direct_url.json")
        info = direct.get("dir_info") if isinstance(direct, dict) else None
        if isinstance(info, dict) and info.get("editable") is True:
            return Origin("editable", f"an editable install of {direct.get('url', 'an unrecorded directory')}")
        return Origin("wheel", f"installed from {direct.get('url', 'an unrecorded location')}")
    return Origin("wheel", "installed from a package index or a built wheel")


def _digest_tree(root: Path) -> tuple[str | None, str, int]:
    """Hash every file under ``root``, or explain why it could not be done.

    The digest binds each file's path to its content, so moving a file changes
    the answer as much as editing it does. Paths are POSIX-normalized and sorted
    before hashing, so two identical trees on different filesystems -- whose
    directory order differs -- still agree.

    Returns:
        The ``sha256:``-prefixed digest, the reason it is absent, and how many
        files it covers.
    """

    def enumeration_failed(error: OSError) -> tp.NoReturn:
        raise error

    entries: list[Path] = []
    try:
        for directory, subdirectories, filenames in os.walk(root, followlinks=False, onerror=enumeration_failed):
            subdirectories[:] = sorted(name for name in subdirectories if name not in _IGNORED_DIRECTORIES)
            for filename in sorted(filenames):
                if filename.endswith(_IGNORED_SUFFIXES):
                    continue
                entries.append(Path(directory) / filename)
                if len(entries) > _MAX_FILES:
                    return None, f"the package tree at {root} holds more than {_MAX_FILES} files", 0
    except OSError as error:
        # An unscannable directory (including a missing root) must not silently
        # disappear from the digest. Avoid exposing arbitrary exception text.
        return None, f"the package tree could not be enumerated ({type(error).__name__})", 0

    digest = hashlib.sha256()
    for path in sorted(entries, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        file_hash = hashlib.sha256()
        try:
            with open(path, "rb") as handle:
                while chunk := handle.read(_CHUNK_BYTES):
                    file_hash.update(chunk)
        except OSError as error:
            # Not skipped. A digest over the readable subset would compare equal
            # to a tree that differs precisely in the file that was unreadable.
            return None, f"{relative} could not be read: {error}", 0
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.digest())
    return f"sha256:{digest.hexdigest()}", "", len(entries)


def compute_build_identity(root: Path) -> BuildIdentity:
    """Return the identity of the package rooted at ``root``.

    Split from :func:`build_identity` so the walk can be exercised against a
    constructed tree rather than against whatever the test process happens to
    have installed.
    """

    origin = _origin(root)
    source_digest, digest_detail, file_count = _digest_tree(root)
    try:
        version = metadata.version("eyvan")
    except metadata.PackageNotFoundError:
        version = _VERSION
    return BuildIdentity(
        version=version,
        origin=origin.kind,
        origin_detail=origin.detail,
        source_digest=source_digest,
        digest_detail=digest_detail,
        file_count=file_count,
        python=platform.python_version(),
        platform=f"{sys.platform}-{platform.machine()}",
    )


@lru_cache(maxsize=1)
def build_identity() -> BuildIdentity:
    """Return this process's build identity, computed once.

    Cached because it is served from an HTTP route and read by the CLI on every
    proof, and a process that is upgraded in place is restarted by every
    deployment path this package supports.

    The tree *can* change under a running process: an editable install is the
    working tree. The plane and the cell agent therefore call this while they are
    built, so each hashes the tree it started from. Computing on first use
    instead could hash different versions of the same tree and falsely mark
    healthy cells as running stale code.
    """

    return compute_build_identity(_package_root())


__all__ = ["BuildIdentity", "build_identity", "compute_build_identity"]
