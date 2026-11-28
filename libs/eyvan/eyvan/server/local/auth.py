# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Owner-only credentials for Eyvan's automatic loopback development mode."""

from __future__ import annotations

import base64
import os
import secrets
import stat
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from eyvan._internal.workspace import create_private_directory

DEFAULT_LOCAL_DEVELOPMENT_CREDENTIAL_PATH = Path("~/.eyvan/local-development-credential")
LOCAL_DEVELOPMENT_CREDENTIAL_HEADER = "X-Eyvan-Local-Credential"
#: The refusal a loopback plane answers when it holds a development credential
#: the caller did not present. Shared with the SDK so a client can recognize
#: exactly this refusal and diagnose it, instead of string-matching a guess.
LOCAL_DEVELOPMENT_CREDENTIAL_REQUIRED_DETAIL = "local Eyvan development credential is required"
_CREDENTIAL_BYTES = 32


class LocalDevelopmentCredentialError(RuntimeError):
    """A local Eyvan development credential is missing, malformed, or unsafe."""


@dataclass(frozen=True, slots=True)
class LocalDevelopmentCredential:
    """A bearer-equivalent secret for one owner's loopback Eyvan profile."""

    value: str = field(repr=False)

    def __post_init__(self) -> None:
        _validate_credential_value(self.value)

    def matches(self, presented: str | None) -> bool:
        """Return whether an HTTP header value matches this credential in constant time.

        The credential is canonical base64url, so a non-ASCII header can never
        match it -- and ``compare_digest`` answers non-ASCII ``str`` with
        ``TypeError``. A caller's garbage header must come back as a mismatch
        (401), never as a server error (500).
        """

        if presented is None:
            return False
        try:
            return secrets.compare_digest(self.value, presented)
        except TypeError:
            return False


class LocalDevelopmentCredentialStore:
    """Create and read the owner-only credential used by automatic loopback auth."""

    def __init__(self, path: Path | str = DEFAULT_LOCAL_DEVELOPMENT_CREDENTIAL_PATH):
        self.path = Path(path).expanduser()

    def load(self) -> LocalDevelopmentCredential:
        """Load the existing credential without creating profile state.

        Raises:
            FileNotFoundError: The automatic loopback server has not initialized this profile.
            LocalDevelopmentCredentialError: The credential file is malformed or not owner-only.
        """

        self._validate_existing_parent_directory()
        try:
            descriptor = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise LocalDevelopmentCredentialError(
                f"could not open local Eyvan development credential {self.path}: {exc}"
            ) from exc
        try:
            self._validate_file_metadata(os.fstat(descriptor))
            encoded = self._read_credential(descriptor)
        finally:
            os.close(descriptor)
        try:
            return LocalDevelopmentCredential(encoded.decode("ascii"))
        except UnicodeDecodeError as exc:
            raise LocalDevelopmentCredentialError(
                f"local Eyvan development credential {self.path} must contain ASCII data"
            ) from exc

    def _validate_existing_parent_directory(self) -> None:
        """Fail closed when another user could inspect the credential directory."""

        try:
            descriptor = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise LocalDevelopmentCredentialError(
                f"could not open local Eyvan profile directory {self.path.parent}: {exc}"
            ) from exc
        try:
            self._validate_profile_directory(os.fstat(descriptor), require_owner_only=True)
        finally:
            os.close(descriptor)

    def load_if_present(self) -> LocalDevelopmentCredential | None:
        """Return the existing credential, or ``None`` before a loopback server has initialized it."""

        try:
            return self.load()
        except FileNotFoundError:
            return None

    def load_or_create(self) -> LocalDevelopmentCredential:
        """Return the stable credential, creating it once with owner-only permissions."""

        self._prepare_parent_directory()
        try:
            return self.load()
        except FileNotFoundError:
            return self._create_or_load()

    def _prepare_parent_directory(self) -> None:
        """Create and validate the profile directory containing the secret."""

        parent = self.path.parent
        try:
            create_private_directory(parent)
            descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except OSError as exc:
            raise LocalDevelopmentCredentialError(
                f"could not prepare local Eyvan profile directory {parent}: {exc}"
            ) from exc
        try:
            self._validate_profile_directory(os.fstat(descriptor), require_owner_only=False)
            os.fchmod(descriptor, 0o700)
            self._validate_profile_directory(os.fstat(descriptor), require_owner_only=True)
        except OSError as exc:
            raise LocalDevelopmentCredentialError(
                f"could not secure local Eyvan profile directory {parent}: {exc}"
            ) from exc
        finally:
            os.close(descriptor)

    def _validate_profile_directory(self, metadata: os.stat_result, *, require_owner_only: bool) -> None:
        """Ensure the credential parent is a directory controlled by this OS user."""

        if not stat.S_ISDIR(metadata.st_mode):
            raise LocalDevelopmentCredentialError(
                f"local Eyvan profile directory {self.path.parent} must be a real directory"
            )
        if metadata.st_uid != os.getuid():
            raise LocalDevelopmentCredentialError(
                f"local Eyvan profile directory {self.path.parent} must be owned by this user"
            )
        if require_owner_only and stat.S_IMODE(metadata.st_mode) & 0o077:
            raise LocalDevelopmentCredentialError(
                f"local Eyvan profile directory {self.path.parent} must not be group- or world-accessible"
            )

    def _create_or_load(self) -> LocalDevelopmentCredential:
        """Atomically create a new credential or use a concurrent creator's credential."""

        credential = LocalDevelopmentCredential(_encode_credential(secrets.token_bytes(_CREDENTIAL_BYTES)))
        try:
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        except OSError as exc:
            raise LocalDevelopmentCredentialError(
                f"could not create local Eyvan development credential {self.path}: {exc}"
            ) from exc
        temporary = Path(temporary_name)
        try:
            try:
                self._write_credential(descriptor, credential.value.encode("ascii"))
                self._validate_file_metadata(os.fstat(descriptor))
            finally:
                os.close(descriptor)
            try:
                # A hard link publishes complete bytes without replacing a concurrent winner.
                os.link(temporary, self.path)
            except FileExistsError:
                return self.load()
            except OSError as exc:
                raise LocalDevelopmentCredentialError(
                    f"could not create local Eyvan development credential {self.path}: {exc}"
                ) from exc
            self._fsync_parent_directory()
            return credential
        finally:
            temporary.unlink(missing_ok=True)

    def _validate_file_metadata(self, metadata: os.stat_result) -> None:
        """Reject credential files that could be observed or replaced by another user."""

        if not stat.S_ISREG(metadata.st_mode):
            raise LocalDevelopmentCredentialError(
                f"local Eyvan development credential {self.path} must be a regular file"
            )
        if metadata.st_uid != os.getuid():
            raise LocalDevelopmentCredentialError(
                f"local Eyvan development credential {self.path} must be owned by this user"
            )
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise LocalDevelopmentCredentialError(
                f"local Eyvan development credential {self.path} must not be group- or world-readable"
            )

    def _read_credential(self, descriptor: int) -> bytes:
        """Read one bounded credential value from an already validated descriptor."""

        chunks: list[bytes] = []
        size = 0
        maximum_size = 128
        while chunk := os.read(descriptor, 4096):
            size += len(chunk)
            if size > maximum_size:
                raise LocalDevelopmentCredentialError(f"local Eyvan development credential {self.path} is too large")
            chunks.append(chunk)
        return b"".join(chunks)

    def _write_credential(self, descriptor: int, encoded: bytes) -> None:
        """Write and persist a credential to a new descriptor."""

        try:
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                if written <= 0:
                    raise LocalDevelopmentCredentialError(
                        f"could not persist local Eyvan development credential {self.path}"
                    )
                offset += written
            os.fsync(descriptor)
        except OSError as exc:
            raise LocalDevelopmentCredentialError(
                f"could not persist local Eyvan development credential {self.path}: {exc}"
            ) from exc

    def _fsync_parent_directory(self) -> None:
        """Persist the directory entry for a newly created credential."""

        try:
            descriptor = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except OSError as exc:
            raise LocalDevelopmentCredentialError(
                f"could not open local Eyvan profile directory {self.path.parent} for syncing: {exc}"
            ) from exc
        try:
            os.fsync(descriptor)
        except OSError as exc:
            raise LocalDevelopmentCredentialError(
                f"could not sync local Eyvan profile directory {self.path.parent}: {exc}"
            ) from exc
        finally:
            os.close(descriptor)


def _encode_credential(raw: bytes) -> str:
    """Encode one fixed-length random credential for safe use in an HTTP header."""

    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _validate_credential_value(value: str) -> None:
    """Validate the canonical fixed-length base64url credential representation."""

    try:
        decoded = base64.b64decode(value + "=", altchars=b"-_", validate=True)
    except (ValueError, UnicodeEncodeError) as exc:
        raise LocalDevelopmentCredentialError("local Eyvan development credential is malformed") from exc
    if len(decoded) != _CREDENTIAL_BYTES or _encode_credential(decoded) != value:
        raise LocalDevelopmentCredentialError("local Eyvan development credential is malformed")


__all__ = (
    "DEFAULT_LOCAL_DEVELOPMENT_CREDENTIAL_PATH",
    "LOCAL_DEVELOPMENT_CREDENTIAL_HEADER",
    "LOCAL_DEVELOPMENT_CREDENTIAL_REQUIRED_DETAIL",
    "LocalDevelopmentCredential",
    "LocalDevelopmentCredentialError",
    "LocalDevelopmentCredentialStore",
)
