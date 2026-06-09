# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of EasyDeL.
#
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Minimal filesystem helpers for local and remote (GCS/S3) paths.

Replaces ``ePath`` with small fsspec-aware functions used only inside the
serialization module.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import fsspec.core


def is_remote_path(path: str) -> bool:
    """Return True when *path* points at a non-local backend.

    Args:
        path: Any path string, potentially with a URL scheme
            (e.g. ``"gs://bucket/path"``).

    Returns:
        ``True`` if fsspec detects a non-``"file"`` protocol, else ``False``.
    """
    protocol, _ = fsspec.core.split_protocol(str(path))
    return protocol is not None


def _get_fs(path: str):
    """Return (filesystem, plain_path) for *path*.

    For local paths filesystem is ``None`` and plain_path equals *path*.
    For remote paths the fsspec filesystem object and the path stripped of
    its scheme are returned.

    Args:
        path: Any path string.

    Returns:
        A 2-tuple ``(fs, plain_path)`` where *fs* is either an
        ``fsspec.AbstractFileSystem`` or ``None``, and *plain_path* is the
        path without URL scheme.
    """
    protocol, _ = fsspec.core.split_protocol(str(path))
    if protocol:
        fs, plain = fsspec.core.url_to_fs(str(path))
        return fs, plain
    return None, str(path)


def joinpath(a: str, *parts: str) -> str:
    """Join path components for both local and remote (URL) paths.

    Unlike :class:`pathlib.Path`, this preserves URL schemes such as
    ``"gs://"``.

    Args:
        a: Base path or URL.
        *parts: Additional path fragments to append.

    Returns:
        The joined path string.
    """
    result = a.rstrip("/")
    for b in parts:
        b = b.lstrip("/")
        result = f"{result}/{b}" if result and b else (result or b)
    return result


def mkdir(path: str, exist_ok: bool = True) -> None:
    """Create a directory (and parents) at *path*.

    Works for both local filesystems and remote object stores via fsspec.

    Args:
        path: Directory to create.
        exist_ok: If ``True``, do not raise when the directory already exists.
            Defaults to ``True``.
    """
    fs, plain = _get_fs(path)
    if fs is not None:
        fs.makedirs(plain, exist_ok=exist_ok)
    else:
        Path(plain).mkdir(parents=True, exist_ok=exist_ok)


def exists(path: str) -> bool:
    """Check whether *path* exists.

    Args:
        path: File or directory to check.

    Returns:
        ``True`` if the path exists, ``False`` otherwise.
    """
    fs, plain = _get_fs(path)
    if fs is not None:
        return fs.exists(plain)
    return Path(plain).exists()


def is_dir(path: str) -> bool:
    """Check whether *path* is a directory.

    Args:
        path: Path to check.

    Returns:
        ``True`` if *path* exists and is a directory, ``False`` otherwise.
    """
    fs, plain = _get_fs(path)
    if fs is not None:
        try:
            return fs.isdir(plain)
        except Exception:
            return False
    return Path(plain).is_dir()


def is_file(path: str) -> bool:
    """Check whether *path* is a regular file.

    Args:
        path: Path to check.

    Returns:
        ``True`` if *path* exists and is a file, ``False`` otherwise.
    """
    fs, plain = _get_fs(path)
    if fs is not None:
        try:
            return fs.isfile(plain)
        except Exception:
            return False
    return Path(plain).is_file()


def write_text(path: str, data: str, encoding: str = "utf-8") -> None:
    """Write a Unicode string to *path*.

    For remote paths the parent directory is created automatically.

    Args:
        path: Destination file path.
        data: String content to write.
        encoding: Text encoding. Defaults to ``"utf-8"``.
    """
    fs, plain = _get_fs(path)
    if fs is not None:
        parent = plain.rsplit("/", 1)[0]
        if parent:
            fs.makedirs(parent, exist_ok=True)
        with fs.open(plain, "w", encoding=encoding) as f:
            f.write(data)
    else:
        Path(plain).write_text(data, encoding=encoding)


def read_text(path: str, encoding: str = "utf-8") -> str:
    """Read the contents of *path* as a Unicode string.

    Args:
        path: File path to read.
        encoding: Text encoding. Defaults to ``"utf-8"``.

    Returns:
        The file contents.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    fs, plain = _get_fs(path)
    if fs is not None:
        with fs.open(plain, "r", encoding=encoding) as f:
            return f.read()
    return Path(plain).read_text(encoding=encoding)


def iterdir(path: str):
    """Yield full child paths (as strings) contained in *path*.

    Args:
        path: Directory to list.

    Yields:
        Absolute or fully-qualified child path strings.
    """
    fs, plain = _get_fs(path)
    if fs is not None:
        for entry in fs.listdir(plain):
            name = entry["name"] if isinstance(entry, dict) else entry
            if name.startswith(plain):
                yield name
            else:
                yield joinpath(plain, name)
    else:
        for entry in Path(plain).iterdir():
            yield str(entry)


def rm(path: str, recursive: bool = False) -> None:
    """Remove a file or directory at *path*.

    Args:
        path: File or directory to remove.
        recursive: If ``True``, remove directories recursively.
            Defaults to ``False``.
    """
    fs, plain = _get_fs(path)
    if fs is not None:
        try:
            fs.rm(plain, recursive=recursive)
        except FileNotFoundError:
            pass
    else:
        if recursive:
            shutil.rmtree(plain, ignore_errors=True)
        else:
            try:
                os.remove(plain)
            except FileNotFoundError:
                pass
