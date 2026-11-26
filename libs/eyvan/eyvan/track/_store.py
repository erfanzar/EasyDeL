# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Object-store access shared by the in-job tracking writer and the tracker.

The store is the source of truth: a writer only ever *creates* immutable
segments, and a reader lists and fetches them. Two operations make that safe
without coordination:

* ``put`` is create-if-absent. Re-putting identical bytes (a replay from the
  spill directory after an ambiguous upload) succeeds; putting *different*
  bytes under an existing key is refused, because a segment name is only ever
  used for one payload.
* ``list`` is recursive and returns keys relative to the store root, sorted.

Standard library only: this module is imported inside training jobs.
"""

from __future__ import annotations

import os
import secrets
import stat
import typing as tp
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse


class TrackingStoreError(RuntimeError):
    """A store operation failed for a reason other than a missing key."""


class SegmentConflictError(TrackingStoreError):
    """A key already exists with different content; segments are immutable."""


class TrackingSizeLimitError(TrackingStoreError):
    """An object or HTTP response exceeds the bounded store-read budget."""


#: Store reads, including GCS HTTP bodies, stop after 256 MiB. Normal blobs are
#: capped by the writer at 64 MiB; gzip expansion is bounded separately by the
#: format decoder. Read in chunks so an absent/misleading length cannot bypass it.
MAX_OBJECT_BYTES = 256 * 1024 * 1024


def _read_bounded(handle: tp.BinaryIO) -> bytes:
    output = bytearray()
    while chunk := handle.read(min(1024 * 1024, MAX_OBJECT_BYTES - len(output) + 1)):
        output.extend(chunk)
        if len(output) > MAX_OBJECT_BYTES:
            raise TrackingSizeLimitError("tracking object exceeds the byte limit")
    return bytes(output)


@tp.runtime_checkable
class TrackingStore(tp.Protocol):
    """Minimal append-only object store used for run tracking."""

    @property
    def root(self) -> str:
        """Return the canonical URI of this store's root."""

        ...

    def put(self, key: str, data: bytes) -> None:
        """Create ``key`` with ``data``; idempotent for identical bytes.

        Raises:
            SegmentConflictError: If ``key`` exists with different bytes.
            TrackingStoreError: On any other storage failure.
        """

        ...

    def get(self, key: str) -> bytes:
        """Return the bytes stored at ``key``.

        Raises:
            FileNotFoundError: If ``key`` does not exist.
            TrackingStoreError: On any other storage failure.
        """

        ...

    def list(self, prefix: str = "") -> list[str]:
        """Return every key under ``prefix`` (recursive), sorted, relative to the root."""

        ...


def validate_key(key: str, *, allow_empty: bool = False) -> str:
    """Return a normalized relative POSIX key, refusing anything that escapes the root."""

    if not isinstance(key, str):
        raise ValueError("store key must be a string")
    if key == "" and allow_empty:
        return ""
    stripped = key.rstrip("/")
    parts = stripped.split("/")
    if (
        not stripped
        or stripped.startswith("/")
        or "\\" in stripped
        or any(part in {"", ".", ".."} for part in parts)
        or "\x00" in stripped
    ):
        raise ValueError(f"invalid store key {key!r}")
    return str(PurePosixPath(*parts))


class LocalTrackingStore:
    """A directory-backed store (``file://`` URIs and plain paths)."""

    def __init__(self, directory: str | os.PathLike[str]) -> None:
        self._directory = Path(directory).expanduser().resolve()

    @property
    def root(self) -> str:
        """Return the ``file://`` URI of the store root."""

        return self._directory.as_uri()

    @property
    def directory(self) -> Path:
        """Return the resolved local directory backing this store."""

        return self._directory

    @contextmanager
    def _parent(self, parts: list[str], *, create: bool = False) -> tp.Iterator[int]:
        """Open each directory without following links, keeping its descriptor pinned."""

        if create:
            self._directory.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        descriptor = os.open(self._directory, flags)
        try:
            for part in parts:
                if create:
                    try:
                        os.mkdir(part, dir_fd=descriptor)
                    except FileExistsError:
                        pass
                child = os.open(part, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            yield descriptor
        finally:
            os.close(descriptor)

    @staticmethod
    def _read_at(parent: int, name: str) -> bytes:
        """Read a regular leaf without following links or blocking on a FIFO."""

        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise TrackingStoreError("tracking object is not a regular file")
            return _read_bounded(handle)

    def put(self, key: str, data: bytes) -> None:
        """Atomically publish an immutable object relative to a no-follow parent fd."""

        parts = validate_key(key).split("/")
        try:
            with self._parent(parts[:-1], create=True) as parent:
                temp_name = f".tmp-{secrets.token_hex(12)}"
                descriptor = os.open(
                    temp_name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent
                )
                try:
                    with os.fdopen(descriptor, "wb") as handle:
                        handle.write(data)
                        handle.flush()
                        os.fsync(handle.fileno())
                    try:
                        os.link(temp_name, parts[-1], src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
                    except FileExistsError:
                        if self._read_at(parent, parts[-1]) != data:
                            raise SegmentConflictError("tracking object already exists with different content") from None
                finally:
                    os.unlink(temp_name, dir_fd=parent)
        except TrackingStoreError:
            raise
        except OSError:
            raise TrackingStoreError("could not write tracking object (symlinks are not allowed)") from None

    def get(self, key: str) -> bytes:
        """Read a regular object, refusing symlinks in any component of its key."""

        parts = validate_key(key).split("/")
        try:
            with self._parent(parts[:-1]) as parent:
                return self._read_at(parent, parts[-1])
        except FileNotFoundError:
            raise
        except OSError:
            raise TrackingStoreError("could not read tracking object (symlinks are not allowed)") from None

    def list(self, prefix: str = "") -> list[str]:
        """List regular files without following symlinks, skipping temporary files."""

        normalized = validate_key(prefix, allow_empty=True)
        keys: list[str] = []

        def walk(parent: int, relative: str) -> None:
            with os.scandir(parent) as entries:
                for entry in entries:
                    if entry.name.startswith(".tmp-"):
                        continue
                    key = f"{relative}/{entry.name}" if relative else entry.name
                    if entry.is_file(follow_symlinks=False):
                        keys.append(key)
                    elif entry.is_dir(follow_symlinks=False):
                        child = os.open(entry.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                        try:
                            walk(child, key)
                        finally:
                            os.close(child)

        try:
            with self._parent(normalized.split("/") if normalized else []) as parent:
                walk(parent, normalized)
        except FileNotFoundError:
            return []
        except OSError:
            raise TrackingStoreError("could not list tracking objects (symlinks are not allowed)") from None
        return sorted(keys)


def store_for_uri(uri: str) -> TrackingStore:
    """Return the store for ``uri``: ``gs://bucket/prefix``, ``file://`` or a local path."""

    if not isinstance(uri, str) or not uri.strip():
        raise ValueError("tracking store URI must be a non-empty string")
    value = uri.strip()
    if value.startswith("gs://"):
        from eyvan.track._gcs import GcsTrackingStore

        return GcsTrackingStore(value)
    if value.startswith("file://"):
        parsed = urlparse(value)
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError("file:// tracking URIs must be local")
        return LocalTrackingStore(unquote(parsed.path))
    if "://" in value:
        raise ValueError("unsupported tracking store URI scheme; use gs://, file:// or a path")
    return LocalTrackingStore(value)


def join_uri(root: str, *parts: str) -> str:
    """Join validated key segments onto a store URI or path without doubling slashes."""

    suffix = "/".join(validate_key(part) for part in parts if part)
    if not suffix:
        return root
    return f"{root.rstrip('/')}/{suffix}"


__all__ = (
    "MAX_OBJECT_BYTES",
    "LocalTrackingStore",
    "SegmentConflictError",
    "TrackingSizeLimitError",
    "TrackingStore",
    "TrackingStoreError",
    "join_uri",
    "store_for_uri",
    "validate_key",
)
