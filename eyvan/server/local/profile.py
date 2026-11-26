# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Owner-only paths and secrets used by a single-machine Eyvan profile."""

from __future__ import annotations

import os
import secrets
import stat
import tempfile
from pathlib import Path

from eyvan._internal.workspace import create_private_directory
from eyvan.config import DEFAULT_LOCAL_ARTIFACT_TOKEN_KEY_PATH, DEFAULT_LOCAL_CONTROL_PLANE_PATH


class LocalProfileSecretError(RuntimeError):
    """A local Eyvan secret path is missing, malformed, or unsafe."""


class LocalArtifactTokenKeyStore:
    """Create and load the stable artifact-transfer HMAC key for one local profile."""

    def __init__(self, path: Path | str = DEFAULT_LOCAL_ARTIFACT_TOKEN_KEY_PATH):
        self.path = Path(path).expanduser()

    def load_or_create(self) -> bytes:
        """Return one owner-only 32-byte key, creating it exactly once when absent."""

        create_private_directory(self.path.parent)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        try:
            return self._load()
        except FileNotFoundError:
            return self._create_or_load()

    def _load(self) -> bytes:
        """Load a previously created regular owner-only key file."""

        try:
            metadata = self.path.lstat()
        except FileNotFoundError:
            raise
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise LocalProfileSecretError(f"local artifact token key {self.path} must be a regular file")
        mode = stat.S_IMODE(metadata.st_mode)
        if mode & 0o077:
            raise LocalProfileSecretError(f"local artifact token key {self.path} must not be group- or world-readable")
        try:
            descriptor = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise LocalProfileSecretError(f"could not open local artifact token key {self.path}: {exc}") from exc
        try:
            chunks: list[bytes] = []
            size = 0
            while chunk := os.read(descriptor, 4096):
                size += len(chunk)
                if size > 32:
                    raise LocalProfileSecretError(f"local artifact token key {self.path} must contain exactly 32 bytes")
                chunks.append(chunk)
        finally:
            os.close(descriptor)
        key = b"".join(chunks)
        if len(key) != 32:
            raise LocalProfileSecretError(f"local artifact token key {self.path} must contain exactly 32 bytes")
        return key

    def _create_or_load(self) -> bytes:
        """Atomically create the key, accepting a concurrent creator's key instead."""

        key = secrets.token_bytes(32)
        try:
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        except OSError as exc:
            raise LocalProfileSecretError(f"could not create local artifact token key {self.path}: {exc}") from exc
        temporary = Path(temporary_name)
        try:
            try:
                offset = 0
                while offset < len(key):
                    written = os.write(descriptor, key[offset:])
                    if written <= 0:
                        raise LocalProfileSecretError(f"could not persist local artifact token key {self.path}")
                    offset += written
                os.fsync(descriptor)
            except OSError as exc:
                raise LocalProfileSecretError(f"could not persist local artifact token key {self.path}: {exc}") from exc
            finally:
                os.close(descriptor)
            try:
                # A hard link publishes complete bytes without replacing a concurrent winner.
                os.link(temporary, self.path)
            except FileExistsError:
                return self._load()
            except OSError as exc:
                raise LocalProfileSecretError(f"could not create local artifact token key {self.path}: {exc}") from exc
            self._fsync_parent()
            return key
        finally:
            temporary.unlink(missing_ok=True)

    def _fsync_parent(self) -> None:
        """Persist the directory entry created for a new key file."""

        try:
            descriptor = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
        except OSError as exc:
            message = f"could not sync local Eyvan profile directory {self.path.parent}: {exc}"
            raise LocalProfileSecretError(message) from exc
        try:
            os.fsync(descriptor)
        except OSError as exc:
            message = f"could not sync local Eyvan profile directory {self.path.parent}: {exc}"
            raise LocalProfileSecretError(message) from exc
        finally:
            os.close(descriptor)


__all__ = (
    "DEFAULT_LOCAL_ARTIFACT_TOKEN_KEY_PATH",
    "DEFAULT_LOCAL_CONTROL_PLANE_PATH",
    "LocalArtifactTokenKeyStore",
    "LocalProfileSecretError",
)
