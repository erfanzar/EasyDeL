# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Resolve operator-owned secret references without widening filesystem access.

A workload persists a reference, never a value.  Project policy authorizes
scheme/authority/path *segments* rather than raw string prefixes, and file
references are opened below the matched policy root with an ``openat``-style
component walk.  The policy check and the actual file open therefore cannot be
split by a path-resolution race.
"""

from __future__ import annotations

import base64
import binascii
import os
import re
import stat
import subprocess
import typing as tp
from dataclasses import dataclass
from urllib.parse import urlsplit

#: A secret is an environment value, not an artifact. Bounding it keeps both
#: the control-plane response and the worker's owner-only launch file small.
MAX_SECRET_BYTES: tp.Final = 1024 * 1024

_SECRET_CAPABILITY = re.compile(r"(?<![A-Za-z0-9_-])eyvs1\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+(?![A-Za-z0-9_-])")
_SECRET_REDACTION = "<redacted secret>"
_GCP_PROJECT = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]*\Z")
_GCP_SECRET = re.compile(r"[A-Za-z0-9_-]+\Z")
_GCP_VERSION = re.compile(r"(?:latest|[1-9][0-9]*)\Z")
_BASE64URL = re.compile(r"[A-Za-z0-9_-]*(?:={0,2})\Z")
_DIRECTORY_FLAGS = (
    os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)


class SecretUnavailable(RuntimeError):
    """Raised when a secret reference cannot be resolved safely."""


class SecretStore(tp.Protocol):
    """Read one secret value by reference."""

    def read(self, reference: str) -> bytes: ...


@dataclass(frozen=True, slots=True)
class _Reference:
    scheme: str
    authority: str
    segments: tuple[str, ...]


def _reference(reference: str, *, policy_root: bool) -> _Reference:
    """Parse the canonical subset of secret-reference URIs Eyvan accepts.

    Percent encoding is intentionally unsupported.  Decoding before the policy
    check makes ``%2f`` and ``%2e%2e`` change segment boundaries; decoding after
    it makes the policy and filesystem see different paths.  Refusing encoded
    forms leaves exactly one representation for every authorized path.
    """

    if not isinstance(reference, str) or not reference or reference != reference.strip():
        raise ValueError("secret references must be non-empty and trimmed")
    if any(ord(character) < 32 or ord(character) == 127 for character in reference):
        raise ValueError("secret references must not contain control characters")
    if "%" in reference or "\\" in reference:
        raise ValueError("secret references must not contain encoded or backslash escapes")
    parsed = urlsplit(reference)
    if parsed.query or parsed.fragment:
        raise ValueError("secret references must not contain a query or fragment")
    scheme = parsed.scheme.lower()
    if scheme == "file":
        # Validate the original spelling, not only urlsplit's normalized fields:
        # ``file:/x`` and ``file:///x`` otherwise parse to the same absolute path.
        # A fourth slash is a different spelling of that path on POSIX as well.
        if reference != f"file://{parsed.path}" or parsed.path.startswith("//"):
            raise ValueError("file secret references must use canonical file:/// absolute-path spelling")
        if parsed.netloc:
            raise ValueError("file secret references must not contain an authority")
        if not parsed.path.startswith("/"):
            raise ValueError("file secret references must use one absolute path")
        if parsed.path.endswith("/"):
            raise ValueError("file secret references must use canonical file paths without a trailing slash")
        raw_segments = parsed.path.split("/")[1:]
        if not raw_segments or any(not part or part in {".", ".."} for part in raw_segments):
            raise ValueError("file secret references must contain canonical path segments")
        return _Reference(scheme=scheme, authority="", segments=tuple(raw_segments))
    if scheme == "gcp-secret":
        if not _GCP_PROJECT.fullmatch(parsed.netloc):
            raise ValueError("gcp-secret references require a valid project authority")
        if not parsed.path.startswith("/") or parsed.path.endswith("/"):
            raise ValueError("gcp-secret references require a secret path")
        segments = tuple(parsed.path.split("/")[1:])
        if len(segments) not in {1, 2}:
            raise ValueError("gcp-secret references require name[/version]")
        if not _GCP_SECRET.fullmatch(segments[0]):
            raise ValueError("gcp-secret references contain an invalid secret name")
        if len(segments) == 2 and not _GCP_VERSION.fullmatch(segments[1]):
            raise ValueError("gcp-secret references contain an invalid version")
        return _Reference(scheme=scheme, authority=parsed.netloc, segments=segments)
    raise ValueError("secret reference uses an unknown scheme")


def validate_secret_reference(reference: str, *, policy_root: bool = False) -> None:
    """Validate one reference or configured policy root.

    Args:
        reference: URI to validate.
        policy_root: Whether the URI is an allow-list root rather than a value
            reference.

    Raises:
        ValueError: If the URI is ambiguous, unsupported, or non-canonical.
    """

    _reference(reference, policy_root=policy_root)


def matching_secret_reference_root(reference: str, allowed_roots: tp.Iterable[str]) -> str | None:
    """Return the most-specific scheme-aware segment root authorizing a URI.

    ``gcp-secret://p/training`` does not authorize
    ``gcp-secret://p/training-prod``; it authorizes the exact secret and, if
    present, its version child.  The same boundary rule makes
    ``file:///run/secrets`` authorize files beneath that directory without
    authorizing ``/run/secrets-old``.
    """

    try:
        candidate = _reference(reference, policy_root=False)
    except ValueError:
        return None
    matches: list[tuple[int, str]] = []
    for root_text in allowed_roots:
        try:
            root = _reference(root_text, policy_root=True)
        except (TypeError, ValueError):
            continue
        if (
            candidate.scheme == root.scheme
            and candidate.authority == root.authority
            and len(candidate.segments) >= len(root.segments)
            and candidate.segments[: len(root.segments)] == root.segments
        ):
            matches.append((len(root.segments), root_text))
    return max(matches, default=(0, None), key=lambda item: item[0])[1]


def secret_reference_allowed(reference: str, allowed_roots: tp.Iterable[str]) -> bool:
    """Return whether a configured root authorizes ``reference``."""

    return matching_secret_reference_root(reference, allowed_roots) is not None


def _read_at_most(descriptor: int, amount: int) -> bytes:
    """Read no more than ``amount`` bytes, tolerating short regular-file reads."""

    chunks: list[bytes] = []
    remaining = amount
    while remaining:
        chunk = os.read(descriptor, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


class FileSecretStore:
    """Race-safely read regular files beneath configured ``file://`` roots.

    Every component, including the configured root itself, is opened relative
    to an already-open directory descriptor with ``O_NOFOLLOW``.  Symlinks both
    within and out of the root are therefore rejected by the same operation that
    obtains the final file descriptor; there is no validate-then-open pathname
    window.
    """

    def __init__(self, *, allowed_roots: tp.Iterable[str] = (), max_bytes: int = MAX_SECRET_BYTES):
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 0:
            raise ValueError("file secret byte limit must be a non-negative integer")
        roots = tuple(allowed_roots)
        for root in roots:
            parsed = _reference(root, policy_root=True)
            if parsed.scheme != "file":
                raise ValueError("file secret roots must use the file scheme")
        self._allowed_roots = roots
        self._max_bytes = max_bytes

    def read(self, reference: str) -> bytes:
        """Open one regular file below its matched root and read limit + 1."""

        matched = matching_secret_reference_root(reference, self._allowed_roots)
        if matched is None:
            raise SecretUnavailable("file secret is outside configured roots")
        try:
            candidate = _reference(reference, policy_root=False)
            root = _reference(matched, policy_root=True)
            relative = candidate.segments[len(root.segments) :]
            if not relative:
                raise ValueError("file secret reference names its allowed directory")

            descriptor = os.open("/", _DIRECTORY_FLAGS)
            try:
                for component in (*root.segments, *relative[:-1]):
                    next_descriptor = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
                    os.close(descriptor)
                    descriptor = next_descriptor
                file_descriptor = os.open(relative[-1], _FILE_FLAGS, dir_fd=descriptor)
                try:
                    if not stat.S_ISREG(os.fstat(file_descriptor).st_mode):
                        raise OSError("file secret is not a regular file")
                    return _read_at_most(file_descriptor, self._max_bytes + 1)
                finally:
                    os.close(file_descriptor)
            finally:
                os.close(descriptor)
        except SecretUnavailable:
            raise
        except Exception:
            # Paths and kernel diagnostics are deliberately not reflected.  A
            # caller gets one retryable category at the HTTP boundary.
            raise SecretUnavailable("file secret could not be read safely") from None


def _decode_gcloud_payload(stdout: str) -> bytes:
    """Decode ``gcloud --format=get(payload.data)`` base64url wire output."""

    if not isinstance(stdout, str):
        raise ValueError("gcloud secret payload was not text")
    if stdout.endswith("\r\n"):
        encoded = stdout[:-2]
    elif stdout.endswith("\n"):
        encoded = stdout[:-1]
    else:
        encoded = stdout
    # Exactly one optional CLI line ending is accepted.  Internal/leading
    # whitespace would otherwise be ignored by permissive base64 decoders.
    if any(character.isspace() for character in encoded) or not _BASE64URL.fullmatch(encoded):
        raise ValueError("gcloud secret payload was not canonical base64url")
    unpadded = encoded.rstrip("=")
    supplied_padding = len(encoded) - len(unpadded)
    if "=" in unpadded or len(unpadded) % 4 == 1:
        raise ValueError("gcloud secret payload had invalid padding")
    required_padding = -len(unpadded) % 4
    # Both RFC 4648 base64url forms used by gcloud are accepted: no padding, or
    # the one exact canonical suffix. Partial, excess, and needless padding are
    # different wire spellings and are rejected rather than normalized.
    if supplied_padding not in {0, required_padding}:
        raise ValueError("gcloud secret payload had invalid padding")
    padded = unpadded + "=" * required_padding
    try:
        decoded = base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)
    except (UnicodeEncodeError, binascii.Error, ValueError):
        raise ValueError("gcloud secret payload was malformed") from None
    canonical_padded = base64.urlsafe_b64encode(decoded).decode("ascii")
    canonical_unpadded = canonical_padded.rstrip("=")
    if unpadded != canonical_unpadded or (supplied_padding and encoded != canonical_padded):
        raise ValueError("gcloud secret payload was not canonical base64url")
    return decoded


class GcpSecretManagerStore:
    """Read and decode a Google Secret Manager value through ``gcloud``."""

    def __init__(self, runner: tp.Callable[[tuple[str, ...]], subprocess.CompletedProcess[str]] | None = None):
        self._runner = runner or self._default_runner

    @staticmethod
    def _default_runner(command: tuple[str, ...]) -> subprocess.CompletedProcess[str]:
        # Interpreter-steering variables from the calling environment (a foreign
        # PYTHONPATH above all) crash gcloud's embedded Python; every provider
        # CLI spawn scrubs them.
        from eyvan._internal.subprocess_env import provider_cli_env

        return subprocess.run(command, capture_output=True, text=True, timeout=30, check=False, env=provider_cli_env())

    def read(self, reference: str) -> bytes:
        """Resolve ``gcp-secret://project/name[/version]``."""

        try:
            parsed = _reference(reference, policy_root=False)
            if parsed.scheme != "gcp-secret":
                raise ValueError("not a gcp-secret reference")
            name = parsed.segments[0]
            version = parsed.segments[1] if len(parsed.segments) == 2 else "latest"
            result = self._runner(
                (
                    "gcloud",
                    "secrets",
                    "versions",
                    "access",
                    version,
                    f"--secret={name}",
                    f"--project={parsed.authority}",
                    "--format=get(payload.data)",
                )
            )
            if result.returncode != 0:
                raise ValueError("gcloud secret access failed")
            payload = _decode_gcloud_payload(result.stdout)
            # Keep the provider boundary bounded for the API's exact 1 MiB
            # decision even though subprocess itself has already captured stdout.
            return payload[: MAX_SECRET_BYTES + 1]
        except Exception:
            # Provider stderr/stdout and arbitrary runner exceptions can contain
            # response fragments, request metadata, or even the value.  Normalize
            # every ordinary failure to one retryable category.
            raise SecretUnavailable("secret provider could not resolve the reference") from None


def redact_secret_material(value: str, *, secrets: tp.Iterable[str] = ()) -> str:
    """Remove secret capabilities and explicitly known values from text.

    Exact-value redaction is useful only while values are safely in memory.
    Durable secret-bearing executions use output suppression instead; plaintext
    is never persisted merely to make later redaction possible.
    """

    redacted = _SECRET_CAPABILITY.sub(_SECRET_REDACTION, str(value))
    for secret in sorted({secret for secret in secrets if secret}, key=len, reverse=True):
        redacted = redacted.replace(secret, _SECRET_REDACTION)
    return redacted


def secret_store_from_reference(reference: str, *, allowed_roots: tp.Iterable[str] = ()) -> SecretStore:
    """Select a backend after validating the URI's scheme and canonical form."""

    try:
        parsed = _reference(reference, policy_root=False)
    except ValueError:
        raise SecretUnavailable("secret reference is invalid or uses an unknown scheme") from None
    if parsed.scheme == "gcp-secret":
        return GcpSecretManagerStore()
    if parsed.scheme == "file":
        return FileSecretStore(allowed_roots=allowed_roots)
    raise SecretUnavailable("secret reference uses an unknown scheme")


def read_secret(reference: str, *, allowed_roots: tp.Iterable[str] = ()) -> bytes:
    """Resolve one secret under the same roots that authorized the request."""

    try:
        return secret_store_from_reference(reference, allowed_roots=allowed_roots).read(reference)
    except SecretUnavailable:
        raise
    except Exception:
        raise SecretUnavailable("secret store could not resolve the reference") from None


__all__ = (
    "MAX_SECRET_BYTES",
    "FileSecretStore",
    "GcpSecretManagerStore",
    "SecretStore",
    "SecretUnavailable",
    "matching_secret_reference_root",
    "read_secret",
    "redact_secret_material",
    "secret_reference_allowed",
    "secret_store_from_reference",
    "validate_secret_reference",
)
