# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Managed callable and result artifacts with scoped transfer authorization."""

from __future__ import annotations

import base64
import contextlib
import errno
import fcntl
import hashlib
import heapq
import hmac
import inspect
import json
import os
import re
import stat as stat_module
import subprocess
import tempfile
import threading
import time
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import url2pathname

import httpx

from eyvan._internal.compat import StrEnum
from eyvan._internal.subprocess_env import provider_cli_env
from eyvan._internal.workspace import create_private_directory
from eyvan.artifacts._capability_tokens import InvalidCapabilityToken, sign_claims, verify_signature
from eyvan.identity.rotation import CapabilityKeyring, StaticCredential

_CALLABLE_ID = re.compile(r"^[0-9a-f]{64}$")
_RESULT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

#: Width of a log chunk's zero-padded sequence, and the ceiling that width
#: implies. Fixed width is what makes lexicographic URI order numeric order, so
#: a writer can find its last chunk by listing rather than by probing. Six
#: digits is a million chunks for one attempt -- at any sane archive interval,
#: longer than any run and longer than any retention window.
LOG_CHUNK_SEQUENCE_DIGITS = 6
LOG_CHUNK_MAX_SEQUENCE = 10**LOG_CHUNK_SEQUENCE_DIGITS - 1
_LOG_CHUNK_SEQUENCE = re.compile(rf"^[0-9]{{{LOG_CHUNK_SEQUENCE_DIGITS}}}$")
_TOKEN_SEGMENT = re.compile(r"^[A-Za-z0-9_-]+$")
_TOKEN_VERSION = "v1"


class ManagedArtifactKind(StrEnum):
    """Kinds addressable through the managed artifact service."""

    CALLABLES = "callables"
    RESULTS = "results"
    WORKSPACES = "workspaces"
    #: Archived structured logs for one terminal attempt. The only durable copy:
    #: journald lives on the cell, and releasing a cell destroys it, so an
    #: attempt's output disappears exactly when the capacity bill stops -- which
    #: is also when someone starts asking what happened.
    LOGS = "logs"


#: Kinds whose object ID *is* the SHA-256 of the stored bytes, so the stored
#: payload is verified against its own URI on every write and read.
CONTENT_ADDRESSED_KINDS = frozenset({ManagedArtifactKind.CALLABLES, ManagedArtifactKind.WORKSPACES})


class ArtifactAction(StrEnum):
    """Operations authorized by a managed artifact transfer token."""

    READ = "read"
    WRITE = "write"


class InvalidManagedArtifactURI(ValueError):
    """Raised when a managed artifact URI is malformed or unsafe."""


class ManagedArtifactConflictError(RuntimeError):
    """Raised when an immutable managed object already has different bytes."""

    def __init__(
        self,
        message: str,
        *,
        current_metadata: ManagedArtifactObjectMetadata | None = None,
    ) -> None:
        super().__init__(message)
        self.current_metadata = current_metadata


class ManagedArtifactInventoryError(RuntimeError):
    """Raised when physical inventory cannot be observed conclusively.

    A list or metadata failure is never translated into an empty scope or a
    missing object. Callers must surface the inventory as inconclusive and must
    not reindex from a partial observation.
    """


class ManagedArtifactVersionMismatchError(RuntimeError):
    """Raised when an exact-version read observes a different live object."""

    def __init__(
        self,
        message: str,
        *,
        current_metadata: ManagedArtifactObjectMetadata | None = None,
    ) -> None:
        super().__init__(message)
        self.current_metadata = current_metadata


class ManagedArtifactOperationInconclusiveError(RuntimeError):
    """Raised when a provider operation cannot be classified safely."""


class ManagedArtifactWriteInconclusiveError(ManagedArtifactOperationInconclusiveError):
    """Raised when a conditional physical write has no conclusive outcome."""


class ManagedArtifactDeleteInconclusiveError(ManagedArtifactOperationInconclusiveError):
    """Raised when a conditional physical delete has no conclusive outcome."""


MAX_MANAGED_ARTIFACT_INVENTORY_PAGE_SIZE = 1000

_GCS_JSON_API_ROOT = "https://storage.googleapis.com/storage/v1"
_GCS_UPLOAD_API_ROOT = "https://storage.googleapis.com/upload/storage/v1"
#: What a JSON API create answers with: the stat fields plus GCS's own MD5 of
#: the bytes it stored, which is what lets a create be proven without a read.
_GCS_UPLOAD_FIELDS = "name,size,generation,md5Hash"
_GCS_INVENTORY_FIELDS = "items(name,size,generation),nextPageToken"
_GCS_STAT_FIELDS = "name,size,generation"
_GCS_INVENTORY_TIMEOUT_SECONDS = 30.0
_GCS_ACCESS_TOKEN_TIMEOUT_SECONDS = 10.0
_GCS_METADATA_TOKEN_TIMEOUT_SECONDS = 2.0
_GCS_TRANSFER_TIMEOUT_SECONDS = 30.0
#: How long one ambient access token is reused. Far inside the hour Google
#: issues them for, and a 401 drops it sooner.
_GCS_ACCESS_TOKEN_REUSE_SECONDS = 300.0
#: Sentinel for a 401 answer, distinct from ``None`` (an absent object).
_GCS_UNAUTHORIZED = object()
_GCS_METADATA_TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
_MAX_INVENTORY_CURSOR_BYTES = 512
_MAX_GCS_ACCESS_TOKEN_BYTES = 16 * 1024
_MAX_GCS_PAGE_TOKEN_BYTES = 16 * 1024
_MAX_GCS_OBJECT_NAME_BYTES = 1024
_MAX_GCS_INVENTORY_URL_BYTES = 16 * 1024
_MAX_GCS_INVENTORY_RESPONSE_BYTES = 2 * 1024 * 1024
_MAX_GCS_INTEGER_DIGITS = 20
_MAX_GCS_INVENTORY_SCAN_PAGES = 32


class InvalidArtifactToken(ValueError):
    """Raised when a managed artifact transfer token is invalid."""


class ExpiredArtifactToken(InvalidArtifactToken):
    """Raised when a validly signed managed artifact token has expired."""


@dataclass(frozen=True, slots=True)
class ManagedArtifactRef:
    """Canonical logical reference to one managed artifact."""

    kind: ManagedArtifactKind
    object_id: str

    @property
    def uri(self) -> str:
        """Return the canonical managed URI."""

        return f"eyvan://{self.kind.value}/{self.object_id}"


@dataclass(frozen=True, slots=True)
class ArtifactTokenClaims:
    """Verified scope carried by a managed artifact transfer token."""

    action: ArtifactAction
    artifact: ManagedArtifactRef
    org_id: str
    project_id: str
    issued_at: int
    expires_at: int | None
    workload_id: str | None = None


class ManagedArtifactStore(tp.Protocol):
    """Tenant-scoped storage boundary for managed callable and result bytes."""

    def put(self, uri: str, payload: bytes, *, org_id: str, project_id: str) -> str:
        """Atomically create ``uri`` or accept an identical existing object."""

        ...

    def get(self, uri: str, *, org_id: str, project_id: str) -> bytes:
        """Read ``uri`` from the specified tenant scope."""

        ...

    def exists(self, uri: str, *, org_id: str, project_id: str) -> bool:
        """Return whether ``uri`` exists in the specified tenant scope."""

        ...

    def delete(self, uri: str, *, org_id: str, project_id: str) -> bool:
        """Remove ``uri`` from the specified tenant scope.

        Returns whether an object was removed. Deleting an absent object is not an
        error: the collector is at-least-once, so a retry after a partial failure
        must be able to finish rather than raise on the half it already did.
        """

        ...


@dataclass(frozen=True, slots=True)
class ManagedArtifactObjectMetadata:
    """Authoritative metadata captured from one physical object observation.

    ``object_version`` identifies the exact provider generation or local file
    instance observed alongside ``size_bytes``. ``sha256`` is present when the
    backend observed the bytes through the exact descriptor/generation (writes
    and exact reads); provider metadata-only inventory may leave it ``None``.
    """

    uri: str
    size_bytes: int
    object_version: str
    sha256: str | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        parse_managed_uri(self.uri)
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int) or self.size_bytes < 0:
            raise ValueError("managed artifact inventory size must be non-negative")
        if not isinstance(self.object_version, str) or not self.object_version.strip():
            raise ValueError("managed artifact inventory object version must not be blank")
        if self.sha256 is not None and (not isinstance(self.sha256, str) or not _CALLABLE_ID.fullmatch(self.sha256)):
            raise ValueError("managed artifact observed digest must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class ManagedArtifactWrite:
    """Outcome of an immutable write and its exact physical object version.

    ``created`` is ``None`` only when a provider failure made authorship
    ambiguous but an exact read proved that the current object has identical
    bytes.
    """

    metadata: ManagedArtifactObjectMetadata
    created: bool | None

    def __post_init__(self) -> None:
        if self.created is not None and not isinstance(self.created, bool):
            raise TypeError("managed artifact write created must be a boolean or None")
        if self.metadata.sha256 is None:
            raise ValueError("managed artifact writes require a backend-observed SHA-256")


@dataclass(frozen=True, slots=True)
class ManagedArtifactRead:
    """Bytes read from exactly the physical version in ``metadata``."""

    payload: bytes
    metadata: ManagedArtifactObjectMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.payload, bytes):
            raise TypeError("managed artifact read payload must be bytes")
        if len(self.payload) != self.metadata.size_bytes:
            raise ValueError("managed artifact read payload size does not match its metadata")
        digest = hashlib.sha256(self.payload).hexdigest()
        if self.metadata.sha256 is None or not hmac.compare_digest(digest, self.metadata.sha256):
            raise ValueError("managed artifact read digest does not match its metadata")


class ManagedArtifactDeleteStatus(StrEnum):
    """Conclusive outcomes of one conditional physical deletion."""

    DELETED = "deleted"
    ABSENT = "absent"
    VERSION_MISMATCH = "version_mismatch"


@dataclass(frozen=True, slots=True)
class ManagedArtifactDeleteResult:
    """Outcome of deleting one exact physical object version."""

    status: ManagedArtifactDeleteStatus
    current_metadata: ManagedArtifactObjectMetadata | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.status, ManagedArtifactDeleteStatus):
            raise TypeError("managed artifact delete status must be a ManagedArtifactDeleteStatus")
        if self.status is ManagedArtifactDeleteStatus.VERSION_MISMATCH:
            if self.current_metadata is None:
                raise ValueError("version mismatch deletion results require current metadata")
        elif self.current_metadata is not None:
            raise ValueError("only version mismatch deletion results may include current metadata")


@tp.runtime_checkable
class ManagedArtifactVersionedStore(ManagedArtifactStore, tp.Protocol):
    """Physical store seam with exact versions and conditional deletion."""

    def put_versioned(
        self,
        uri: str,
        payload: bytes,
        *,
        org_id: str,
        project_id: str,
    ) -> ManagedArtifactWrite:
        """Create an immutable object or return its identical live version."""

        ...

    def get_versioned(
        self,
        uri: str,
        *,
        org_id: str,
        project_id: str,
        object_version: str | None = None,
    ) -> ManagedArtifactRead:
        """Read the current object, optionally requiring an exact version."""

        ...

    def delete_versioned(
        self,
        uri: str,
        *,
        org_id: str,
        project_id: str,
        object_version: str,
    ) -> ManagedArtifactDeleteResult:
        """Delete only the supplied exact physical object version."""

        ...


@dataclass(frozen=True, slots=True)
class ManagedArtifactInventoryPage:
    """One bounded page from exactly one known tenant prefix."""

    objects: tuple[ManagedArtifactObjectMetadata, ...]
    next_cursor: str | None = None

    def __post_init__(self) -> None:
        if self.next_cursor is not None and not self.next_cursor.strip():
            raise ValueError("managed artifact inventory cursor must not be blank")


@tp.runtime_checkable
class ManagedArtifactInventoryStore(ManagedArtifactStore, tp.Protocol):
    """The seam a dry-run inventory reconciler requires from every backend.

    Listing is scope-explicit and bounded. Errors propagate: a failed listing is
    inconclusive and must never be interpreted as an empty prefix. ``stat`` and
    page entries carry immutable object versions so a future reconciler can
    distinguish replacements without reading or deleting bytes.
    """

    def list_scope(
        self,
        *,
        org_id: str,
        project_id: str,
        cursor: str | None = None,
        limit: int = 1000,
    ) -> ManagedArtifactInventoryPage: ...

    def stat(self, uri: str, *, org_id: str, project_id: str) -> ManagedArtifactObjectMetadata | None: ...


def _validate_inventory_page_request(cursor: str | None, limit: int) -> str | None:
    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or not 1 <= limit <= MAX_MANAGED_ARTIFACT_INVENTORY_PAGE_SIZE
    ):
        raise ValueError(
            f"managed artifact inventory limit must be between 1 and {MAX_MANAGED_ARTIFACT_INVENTORY_PAGE_SIZE}"
        )
    if cursor is not None:
        if not isinstance(cursor, str) or not cursor.strip():
            raise ValueError("managed artifact inventory cursor must not be blank")
        try:
            cursor_size = len(cursor.encode("utf-8"))
        except UnicodeEncodeError:
            raise ValueError("managed artifact inventory cursor must be valid UTF-8") from None
        if cursor_size > _MAX_INVENTORY_CURSOR_BYTES:
            raise ValueError("managed artifact inventory cursor is too large")
        parse_managed_uri(cursor)
    return cursor


class CommandRunner(tp.Protocol):
    """Injectable subprocess boundary used by the GCS byte-transfer backend."""

    def __call__(
        self,
        args: tp.Sequence[str],
        *,
        check: bool,
        capture_output: bool,
        timeout: float,
    ) -> subprocess.CompletedProcess[bytes]: ...


def _default_gcloud_runner(
    args: tp.Sequence[str],
    *,
    check: bool,
    capture_output: bool,
    timeout: float,
) -> subprocess.CompletedProcess[bytes]:
    """Run ``gcloud`` without the caller's Python interpreter steering.

    The default used to be a bare ``subprocess.run``, which leaked a
    submitting process's ``PYTHONPATH`` into gcloud's own interpreter; see
    ``eyvan._internal.subprocess_env``.
    """

    return subprocess.run(args, check=check, capture_output=capture_output, timeout=timeout, env=provider_cli_env())


class _GCSInventoryTransport(tp.Protocol):
    """Injectable, read-only HTTP boundary for the GCS JSON inventory API."""

    def __call__(
        self,
        url: str,
        *,
        headers: tp.Mapping[str, str],
        timeout_seconds: float,
        maximum_response_bytes: int,
    ) -> tuple[int, bytes]: ...


class _GCSInventoryResponseTooLarge(RuntimeError):
    """Internal signal that a streaming response crossed its fixed body cap."""


def _httpx_gcs_inventory_transport(
    url: str,
    *,
    headers: tp.Mapping[str, str],
    timeout_seconds: float,
    maximum_response_bytes: int,
) -> tuple[int, bytes]:
    """Perform one non-redirecting, response-bounded GCS JSON API request."""

    timeout = httpx.Timeout(timeout_seconds)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            with client.stream("GET", url, headers=dict(headers)) as response:
                raw_length = response.headers.get("content-length")
                if raw_length is not None:
                    if len(raw_length) > _MAX_GCS_INTEGER_DIGITS or not raw_length.isascii() or not raw_length.isdigit():
                        raise _GCSInventoryResponseTooLarge
                    if int(raw_length) > maximum_response_bytes:
                        raise _GCSInventoryResponseTooLarge
                body = bytearray()
                for chunk in response.iter_bytes():
                    if len(body) + len(chunk) > maximum_response_bytes:
                        raise _GCSInventoryResponseTooLarge
                    body.extend(chunk)
                return response.status_code, bytes(body)
    except _GCSInventoryResponseTooLarge:
        raise
    except httpx.TimeoutException as exc:
        raise TimeoutError from exc
    except httpx.HTTPError as exc:
        raise OSError from exc


class _GCSUploadTransport(tp.Protocol):
    """Injectable HTTP boundary for one conditional JSON API object create."""

    def __call__(
        self,
        url: str,
        *,
        headers: tp.Mapping[str, str],
        content: bytes,
        timeout_seconds: float,
        maximum_response_bytes: int,
    ) -> tuple[int, bytes]: ...


def _httpx_gcs_upload_transport(
    url: str,
    *,
    headers: tp.Mapping[str, str],
    content: bytes,
    timeout_seconds: float,
    maximum_response_bytes: int,
) -> tuple[int, bytes]:
    """POST one simple-media upload, non-redirecting, with a bounded response body."""

    timeout = httpx.Timeout(timeout_seconds)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            with client.stream("POST", url, headers=dict(headers), content=content) as response:
                body = bytearray()
                for chunk in response.iter_bytes():
                    if len(body) + len(chunk) > maximum_response_bytes:
                        raise _GCSInventoryResponseTooLarge
                    body.extend(chunk)
                return response.status_code, bytes(body)
    except _GCSInventoryResponseTooLarge:
        raise
    except httpx.TimeoutException as exc:
        raise TimeoutError from exc
    except httpx.HTTPError as exc:
        raise OSError from exc


class _GCSDownloadTransport(tp.Protocol):
    """Injectable HTTP boundary for one conditional JSON API media read."""

    def __call__(
        self,
        url: str,
        *,
        headers: tp.Mapping[str, str],
        timeout_seconds: float,
    ) -> tuple[int, tp.Mapping[str, str], bytes]: ...


def _httpx_gcs_download_transport(
    url: str,
    *,
    headers: tp.Mapping[str, str],
    timeout_seconds: float,
) -> tuple[int, tp.Mapping[str, str], bytes]:
    """GET one object's media, non-redirecting; return status, lower-cased headers and body.

    The body is not capped, matching the ``gcloud storage cat`` it replaces.
    """

    timeout = httpx.Timeout(timeout_seconds)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            with client.stream("GET", url, headers=dict(headers)) as response:
                body = bytearray()
                for chunk in response.iter_raw():
                    body.extend(chunk)
                return (
                    response.status_code,
                    {key.lower(): value for key, value in response.headers.items()},
                    bytes(body),
                )
    except httpx.TimeoutException as exc:
        raise TimeoutError from exc
    except httpx.HTTPError as exc:
        raise OSError from exc


def _gcs_md5_header(value: str | None) -> str | None:
    """Return the base64 MD5 from an ``x-goog-hash`` header, if it carries one."""

    if not value:
        return None
    for part in value.split(","):
        name, separator, digest = part.strip().partition("=")
        if separator and name == "md5" and digest:
            return digest
    return None


def _gcloud_inventory_access_token() -> str:
    """Read an ephemeral OAuth token from the ambient gcloud/workload identity."""

    try:
        completed = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            check=False,
            capture_output=True,
            timeout=_GCS_ACCESS_TOKEN_TIMEOUT_SECONDS,
            env=provider_cli_env(),
        )
    except (OSError, subprocess.SubprocessError):
        raise ManagedArtifactInventoryError("ambient GCS credential lookup failed") from None
    if completed.returncode != 0:
        raise ManagedArtifactInventoryError("ambient GCS credential lookup failed")
    raw_token = completed.stdout
    if not isinstance(raw_token, bytes) or len(raw_token) > _MAX_GCS_ACCESS_TOKEN_BYTES:
        raise ManagedArtifactInventoryError("ambient GCS credential lookup returned an invalid access token")
    try:
        token = raw_token.decode("ascii").strip()
    except UnicodeDecodeError:
        raise ManagedArtifactInventoryError("ambient GCS credential lookup returned an invalid access token") from None
    return token


def _metadata_inventory_access_token() -> str:
    """Read an ephemeral token from the fixed Google workload metadata endpoint."""

    maximum_body_bytes = _MAX_GCS_ACCESS_TOKEN_BYTES * 2
    try:
        with httpx.Client(
            timeout=httpx.Timeout(_GCS_METADATA_TOKEN_TIMEOUT_SECONDS),
            follow_redirects=False,
            trust_env=False,
        ) as client:
            with client.stream(
                "GET",
                _GCS_METADATA_TOKEN_URL,
                headers={"Accept-Encoding": "identity", "Metadata-Flavor": "Google"},
            ) as response:
                if response.status_code != 200 or response.headers.get("metadata-flavor") != "Google":
                    raise ManagedArtifactInventoryError("ambient GCS credential lookup failed")
                body = bytearray()
                for chunk in response.iter_bytes():
                    if len(body) + len(chunk) > maximum_body_bytes:
                        raise ManagedArtifactInventoryError("ambient GCS credential lookup failed")
                    body.extend(chunk)
    except ManagedArtifactInventoryError:
        raise
    except (httpx.HTTPError, OSError):
        raise ManagedArtifactInventoryError("ambient GCS credential lookup failed") from None
    try:
        document = json.loads(bytes(body))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ManagedArtifactInventoryError("ambient GCS credential lookup failed") from None
    if not isinstance(document, dict) or set(document) - {"access_token", "expires_in", "token_type"}:
        raise ManagedArtifactInventoryError("ambient GCS credential lookup failed")
    token = document.get("access_token")
    if not isinstance(token, str):
        raise ManagedArtifactInventoryError("ambient GCS credential lookup failed")
    return token


def _ambient_gcs_inventory_access_token() -> str:
    """Use ambient gcloud credentials, then workload metadata when gcloud is absent."""

    try:
        return _gcloud_inventory_access_token()
    except ManagedArtifactInventoryError:
        try:
            return _metadata_inventory_access_token()
        except ManagedArtifactInventoryError:
            raise ManagedArtifactInventoryError("ambient GCS credential lookup failed") from None


def _validate_object_id(kind: ManagedArtifactKind, object_id: str) -> str:
    content_addressed = kind in CONTENT_ADDRESSED_KINDS
    pattern = _CALLABLE_ID if content_addressed else _RESULT_ID
    if not isinstance(object_id, str) or not pattern.fullmatch(object_id):
        expected = "a lowercase SHA-256 digest" if content_addressed else "a safe result identifier"
        raise InvalidManagedArtifactURI(f"{kind.value} object ID must be {expected}")
    return object_id


def callable_uri(sha256: str) -> str:
    """Build a canonical managed URI for a content-addressed callable."""

    return ManagedArtifactRef(
        ManagedArtifactKind.CALLABLES, _validate_object_id(ManagedArtifactKind.CALLABLES, sha256)
    ).uri


def workspace_uri(sha256: str) -> str:
    """Build a canonical managed URI for a content-addressed workspace bundle."""

    return ManagedArtifactRef(
        ManagedArtifactKind.WORKSPACES, _validate_object_id(ManagedArtifactKind.WORKSPACES, sha256)
    ).uri


def log_uri(workload_id: str, attempt: int, rank: int) -> str:
    """Build a canonical managed URI for one rank's archived logs.

    Keyed by attempt as well as workload: a retry produces different output from
    a different execution, and overwriting the predecessor's would destroy the
    record of the failure that caused the retry -- the one an operator is most
    likely to be looking for.
    """

    if attempt < 1 or rank < 0:
        raise InvalidManagedArtifactURI("log archives require a positive attempt and a non-negative rank")
    object_id = f"{workload_id}-a{attempt}-r{rank}"
    return ManagedArtifactRef(ManagedArtifactKind.LOGS, _validate_object_id(ManagedArtifactKind.LOGS, object_id)).uri


def log_manifest_uri(workload_id: str, attempt: int) -> str:
    """Build a canonical managed URI for one attempt's archive manifest.

    A first-class object beside the per-rank blobs rather than a line inside
    rank zero's own blob. An index that lives inside the first object it
    describes can only ever name what the writer *intends* to upload: rank zero
    is written before any other rank exists, so it claims objects that may still
    fail, and losing rank zero itself hides every rank that did land. A separate
    URI can be committed last, once every blob's fate is known, and is
    discoverable without reading any rank at all.

    ``-manifest`` cannot collide with ``-r{rank}`` because a rank suffix is
    always decimal digits.
    """

    if attempt < 1:
        raise InvalidManagedArtifactURI("log archive manifests require a positive attempt")
    object_id = f"{workload_id}-a{attempt}-manifest"
    return ManagedArtifactRef(ManagedArtifactKind.LOGS, _validate_object_id(ManagedArtifactKind.LOGS, object_id)).uri


def log_chunk_uri(workload_id: str, attempt: int, sequence: int) -> str:
    """Build a canonical managed URI for one incremental log chunk.

    Chunks are how an attempt's output survives losing its host. The terminal
    archive can only ever be written on the release path, so an attempt whose
    machine is reclaimed rather than released -- every preemption, which is the
    common case on spot -- reaches settlement with no journal left to read. A
    chunk is written while the cell is alive and is immutable once written, so
    what it holds is already safe when the host disappears.

    The sequence is zero-padded so that lexicographic URI order is numeric
    order. That is what lets the writer recover its position after a leadership
    change by listing the prefix and taking the last entry, rather than by
    probing every sequence from zero.

    ``-c{sequence}`` cannot collide with ``-r{rank}`` or ``-manifest``.
    """

    if attempt < 1 or sequence < 0:
        raise InvalidManagedArtifactURI("log chunks require a positive attempt and a non-negative sequence")
    if sequence > LOG_CHUNK_MAX_SEQUENCE:
        raise InvalidManagedArtifactURI(f"log chunk sequence must not exceed {LOG_CHUNK_MAX_SEQUENCE}")
    object_id = f"{log_chunk_object_prefix(workload_id, attempt)}{sequence:0{LOG_CHUNK_SEQUENCE_DIGITS}d}"
    return ManagedArtifactRef(ManagedArtifactKind.LOGS, _validate_object_id(ManagedArtifactKind.LOGS, object_id)).uri


def log_chunk_object_prefix(workload_id: str, attempt: int) -> str:
    """Return the object-ID prefix every chunk of one attempt shares."""

    if attempt < 1:
        raise InvalidManagedArtifactURI("log chunks require a positive attempt")
    return f"{workload_id}-a{attempt}-c"


def log_chunk_uri_prefix(workload_id: str, attempt: int) -> str:
    """Return the URI prefix every chunk of one attempt shares."""

    return f"eyvan://{ManagedArtifactKind.LOGS.value}/{log_chunk_object_prefix(workload_id, attempt)}"


def log_chunk_sequence(uri: str, workload_id: str, attempt: int) -> int | None:
    """Return the sequence a chunk URI names, or ``None`` if it names no chunk."""

    prefix = log_chunk_uri_prefix(workload_id, attempt)
    if not uri.startswith(prefix):
        return None
    suffix = uri[len(prefix) :]
    # ``str.isdigit`` is true of superscripts and other unicode digits that
    # ``int`` then parses into a sequence no writer ever minted.
    if not _LOG_CHUNK_SEQUENCE.fullmatch(suffix):
        return None
    return int(suffix)


def result_uri(result_id: str) -> str:
    """Build a canonical managed URI for a result object."""

    return ManagedArtifactRef(
        ManagedArtifactKind.RESULTS, _validate_object_id(ManagedArtifactKind.RESULTS, result_id)
    ).uri


def parse_managed_uri(uri: str) -> ManagedArtifactRef:
    """Parse a canonical ``eyvan://`` artifact URI without path normalization."""

    if not isinstance(uri, str):
        raise InvalidManagedArtifactURI("managed artifact URI must be a string")
    parsed = urlsplit(uri)
    if parsed.scheme != "eyvan" or parsed.query or parsed.fragment:
        raise InvalidManagedArtifactURI("managed artifact URI must use eyvan:// without query or fragment")
    try:
        kind = ManagedArtifactKind(parsed.netloc)
    except ValueError as exc:
        kinds = ", ".join(sorted(member.value for member in ManagedArtifactKind))
        raise InvalidManagedArtifactURI(f"managed artifact URI kind must be one of: {kinds}") from exc
    if not parsed.path.startswith("/") or parsed.path.count("/") != 1:
        raise InvalidManagedArtifactURI("managed artifact URI must contain exactly one object ID")
    object_id = _validate_object_id(kind, parsed.path[1:])
    reference = ManagedArtifactRef(kind, object_id)
    if uri != reference.uri:
        raise InvalidManagedArtifactURI("managed artifact URI must use its canonical form")
    return reference


def is_managed_uri(uri: str) -> bool:
    """Return whether ``uri`` is a canonical managed artifact URI.

    A predicate rather than a try/except at each call site: a workload may name
    a ``gs://`` or local result URI, and "not managed" is an ordinary answer
    there, not an error worth constructing an exception for.
    """

    try:
        parse_managed_uri(uri)
    except InvalidManagedArtifactURI:
        return False
    return True


def _validate_scope_value(value: str, name: str, *, max_length: int = 128) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_length:
        raise ValueError(f"{name} must be a non-empty string of at most {max_length} characters")
    return value


def storage_scope(org_id: str, project_id: str) -> str:
    """Return the opaque physical namespace for one organization project."""

    org_id = _validate_scope_value(org_id, "org_id")
    project_id = _validate_scope_value(project_id, "project_id")
    encoded = json.dumps((org_id, project_id), ensure_ascii=True, separators=(",", ":")).encode()
    return f"scope_{hashlib.sha256(encoded).hexdigest()}"


def _validate_payload(reference: ManagedArtifactRef, payload: bytes) -> bytes:
    if not isinstance(payload, bytes):
        raise TypeError("managed artifact payload must be bytes")
    if reference.kind in CONTENT_ADDRESSED_KINDS:
        digest = hashlib.sha256(payload).hexdigest()
        if not hmac.compare_digest(digest, reference.object_id):
            raise ValueError(f"{reference.kind.value} payload does not match its content-addressed URI")
    return payload


def _validate_loaded_payload(reference: ManagedArtifactRef, payload: bytes) -> bytes:
    if reference.kind in CONTENT_ADDRESSED_KINDS:
        digest = hashlib.sha256(payload).hexdigest()
        if not hmac.compare_digest(digest, reference.object_id):
            raise ValueError(f"stored {reference.kind.value} payload does not match its content-addressed URI")
    return payload


_OBJECT_EXTENSIONS = {
    ManagedArtifactKind.CALLABLES: ".pkl",
    ManagedArtifactKind.RESULTS: ".result",
    ManagedArtifactKind.WORKSPACES: ".tar.gz",
    # JSON lines, the same shape the live log page returns, so a reader needs
    # one parser for archived logs and streaming ones rather than two.
    ManagedArtifactKind.LOGS: ".jsonl",
}


def _relative_object_path(reference: ManagedArtifactRef, org_id: str, project_id: str) -> Path:
    extension = _OBJECT_EXTENSIONS[reference.kind]
    return Path("scopes", storage_scope(org_id, project_id), reference.kind.value, f"{reference.object_id}{extension}")


def _payload_digest(payload: bytes) -> bytes:
    return hashlib.sha256(payload).digest()


def _payloads_match(left: bytes, right: bytes) -> bool:
    return len(left) == len(right) and hmac.compare_digest(_payload_digest(left), _payload_digest(right))


def _parse_canonical_uint64(value: str, *, description: str, minimum: int) -> int:
    if (
        not value
        or len(value) > _MAX_GCS_INTEGER_DIGITS
        or not value.isascii()
        or not value.isdigit()
        or (len(value) > 1 and value.startswith("0"))
    ):
        raise ValueError(f"{description} must contain canonical unsigned integers")
    parsed = int(value)
    if parsed < minimum or parsed > (1 << 64) - 1:
        raise ValueError(f"{description} is outside the supported numeric range")
    return parsed


@dataclass(frozen=True, slots=True)
class _LocalObjectVersion:
    device: int
    inode: int
    ctime_ns: int
    generation_digest: str | None = None


_LOCAL_DELETE_TRANSACTION = re.compile(
    r"^\.delete-v1-(?P<layout>[12])-(?P<digest>[0-9a-f]{64})-"
    r"(?P<device>0|[1-9][0-9]*)-(?P<inode>0|[1-9][0-9]*)-(?P<ctime_ns>0|[1-9][0-9]*)$"
)
_LOCAL_DELETE_RECEIPT_ENTRIES = frozenset(
    {
        "cleanup-safe",
        "complete",
        "destroyed-anchor",
        "detached",
        "preserved-anchor",
        "replacement-anchor",
        "validated",
    }
)
_MAX_LOCAL_GENERATION_DIRECTORY_ENTRIES = 256
_MAX_LOCAL_DELETE_RECEIPT_ENTRIES = 16


@dataclass(frozen=True, slots=True)
class _LocalDeleteIntent:
    """Self-describing identity of one durable local deletion transaction."""

    layout: int
    digest: str
    device: int
    inode: int
    ctime_ns: int

    @property
    def directory_name(self) -> str:
        return f".delete-v1-{self.layout}-{self.digest}-{self.device}-{self.inode}-{self.ctime_ns}"

    @property
    def trash_directory_name(self) -> str:
        return self.directory_name.replace(".delete-", ".trash-", 1)

    @property
    def anchored(self) -> bool:
        return self.layout == 2

    def matches_version(self, version: _LocalObjectVersion) -> bool:
        return (
            self.device == version.device
            and self.inode == version.inode
            and self.ctime_ns == version.ctime_ns
            and (version.generation_digest is None or self.digest == version.generation_digest)
            and self.anchored == (version.generation_digest is not None)
        )


def _parse_local_delete_intent(directory_name: str) -> _LocalDeleteIntent:
    match = _LOCAL_DELETE_TRANSACTION.fullmatch(directory_name)
    if match is None:
        raise ValueError("local deletion transaction name is invalid")
    layout = int(match.group("layout"))
    digest = match.group("digest")
    device, inode, ctime_ns = (
        _parse_canonical_uint64(
            match.group(component),
            description="local deletion transaction",
            minimum=0,
        )
        for component in ("device", "inode", "ctime_ns")
    )
    return _LocalDeleteIntent(
        layout=layout,
        digest=digest,
        device=device,
        inode=inode,
        ctime_ns=ctime_ns,
    )


def _parse_local_trash_intent(directory_name: str) -> _LocalDeleteIntent:
    if not directory_name.startswith(".trash-"):
        raise ValueError("local deletion trash receipt name is invalid")
    intent = _parse_local_delete_intent(directory_name.replace(".trash-", ".delete-", 1))
    if intent.trash_directory_name != directory_name:
        raise ValueError("local deletion trash receipt name is invalid")
    return intent


def _parse_local_object_version(object_version: str) -> _LocalObjectVersion:
    if not isinstance(object_version, str) or not object_version.startswith("local:"):
        raise ValueError("local managed artifact version must use the local: prefix")
    components = object_version.split(":")
    if len(components) == 4:
        numeric = components[1:]
        generation_digest = None
    elif len(components) == 6 and components[1] == "2" and _CALLABLE_ID.fullmatch(components[2]):
        numeric = components[3:]
        generation_digest = components[2]
    else:
        raise ValueError("local managed artifact version must contain a supported immutable file identity")
    device, inode, ctime_ns = (
        _parse_canonical_uint64(component, description="local managed artifact version", minimum=0)
        for component in numeric
    )
    return _LocalObjectVersion(
        device=device,
        inode=inode,
        ctime_ns=ctime_ns,
        generation_digest=generation_digest,
    )


def _parse_gcs_object_version(object_version: str) -> int:
    if not isinstance(object_version, str) or not object_version.startswith("gcs:"):
        raise ValueError("GCS managed artifact version must use the gcs: prefix")
    components = object_version.split(":")
    if len(components) != 2:
        raise ValueError("GCS managed artifact version must contain exactly one generation")
    return _parse_canonical_uint64(
        components[1],
        description="GCS managed artifact generation",
        minimum=1,
    )


class LocalManagedArtifactStore:
    """Immutable local storage with durable generation deletion transactions.

    Canonical object paths remain regular files for compatibility. New writes
    also create an owner-only hard-link anchor outside ``scopes``. Conditional
    deletion renames the canonical entry into a deterministic transaction
    directory below that anchor, rehashes the exact detached descriptor, durably
    marks validation, and destroys the selected inode through ``ftruncate`` on
    that descriptor. It never unlinks the detached pathname while the transaction
    is active. A completed transaction is first archived under a private
    ``.trash-*`` name, then its fixed set of receipt links and markers is removed
    through no-follow directory-relative operations. A crash leaves only a
    resumable bounded receipt. A path replacement observed at any boundary is
    preserved before archival and, when valid, copied back as the new current
    generation.

    The configured root, ``.locks``, and ``.generations`` are a trusted operator
    boundary: processes able to write those hidden directories can forge store
    transactions and are therefore administrators, not artifact writers. Entries
    below ``scopes`` are still treated as hostile even when changed by the same
    UID. Opens are no-follow/nonblocking, every descriptor must be regular, and
    unrecognized hard links fail closed because an unseen alias defeats deletion
    accounting. Existing pre-anchor files remain readable and use the same
    descriptor-bound transaction when deleted.
    """

    _LOCK_DIRECTORY = ".locks"
    _LOCK_LAYOUT_DIRECTORY = "layout-v2"
    _LOCK_STRIPES = 16
    _GENERATION_DIRECTORY = ".generations"

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root).expanduser().resolve()
        create_private_directory(self.root)
        self._ensure_lock_layout()

    def _ensure_lock_layout(self) -> None:
        """Initialize empty roots only; never migrate a legacy flock domain."""

        maintenance_error = "local artifact lock layout requires quiesced operator maintenance; legacy files preserved"
        try:
            with self._open_directory((self._LOCK_DIRECTORY, self._LOCK_LAYOUT_DIRECTORY), create=False) as descriptor:
                self._validate_control_directory(descriptor)
        except FileNotFoundError:
            with self._open_directory((), create=False) as root_fd:
                with os.scandir(root_fd) as entries:
                    if next(entries, None) is not None:
                        raise InvalidManagedArtifactURI(maintenance_error) from None
                # A concurrent constructor may see this incomplete initialization
                # and refuse. It must never infer a legacy directory is safe.
                os.mkdir(self._LOCK_DIRECTORY, mode=0o700, dir_fd=root_fd)
                os.fsync(root_fd)
            with self._open_control_directory(self._LOCK_DIRECTORY, create=False) as lock_fd:
                os.mkdir(self._LOCK_LAYOUT_DIRECTORY, mode=0o700, dir_fd=lock_fd)
                os.fsync(lock_fd)
        allowed = {self._LOCK_LAYOUT_DIRECTORY, *(f"stripe-{index:02x}.lock" for index in range(self._LOCK_STRIPES))}
        with self._open_control_directory(self._LOCK_DIRECTORY, create=False) as lock_fd:
            with os.scandir(lock_fd) as entries:
                for entry in entries:
                    if entry.name not in allowed:
                        raise InvalidManagedArtifactURI(maintenance_error)

    @staticmethod
    def _directory_open_flags() -> int:
        return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)

    @staticmethod
    def _object_open_flags() -> int:
        return (
            os.O_RDONLY
            | os.O_NOFOLLOW
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOCTTY", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )

    @staticmethod
    def _translate_unsafe_path(exc: OSError) -> tp.NoReturn:
        if exc.errno in {
            errno.ELOOP,
            errno.ENOTDIR,
            errno.ENXIO,
            errno.ENODEV,
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }:
            raise InvalidManagedArtifactURI(
                "managed artifact storage path cannot contain symbolic links, special files, or non-directories"
            ) from exc
        raise exc

    @contextlib.contextmanager
    def _open_directory(self, components: tp.Sequence[str], *, create: bool) -> tp.Iterator[int]:
        """Open a root-confined directory chain without pathname re-resolution."""

        descriptors: list[int] = []
        try:
            try:
                descriptors.append(os.open(self.root, self._directory_open_flags()))
            except OSError as exc:
                self._translate_unsafe_path(exc)
            self._validate_control_directory(descriptors[-1])
            for component in components:
                if not component or component in {".", ".."} or "/" in component:
                    raise InvalidManagedArtifactURI("managed artifact storage path contains an unsafe component")
                if create:
                    try:
                        os.mkdir(component, mode=0o700, dir_fd=descriptors[-1])
                        os.fsync(descriptors[-1])
                    except FileExistsError:
                        pass
                try:
                    descriptor = os.open(component, self._directory_open_flags(), dir_fd=descriptors[-1])
                except OSError as exc:
                    self._translate_unsafe_path(exc)
                descriptors.append(descriptor)
            yield descriptors[-1]
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)

    @staticmethod
    def _validate_control_directory(descriptor: int) -> None:
        metadata = os.fstat(descriptor)
        effective_uid = getattr(os, "geteuid", lambda: metadata.st_uid)()
        if (
            not stat_module.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != effective_uid
            or metadata.st_mode & (stat_module.S_IWGRP | stat_module.S_IWOTH)
        ):
            raise InvalidManagedArtifactURI("managed artifact control directories must be owner-only real directories")

    @contextlib.contextmanager
    def _open_control_directory(self, name: str, *, create: bool) -> tp.Iterator[int]:
        with self._open_directory((name,), create=create) as descriptor:
            self._validate_control_directory(descriptor)
            yield descriptor

    @staticmethod
    def _object_key(relative: Path) -> str:
        return hashlib.sha256(relative.as_posix().encode("ascii")).hexdigest()

    @contextlib.contextmanager
    def _open_generation_directory(self, relative: Path, *, create: bool) -> tp.Iterator[int]:
        key = self._object_key(relative)
        with self._open_control_directory(self._GENERATION_DIRECTORY, create=create) as parent_fd:
            with self._open_directory((self._GENERATION_DIRECTORY, key), create=create) as descriptor:
                self._validate_control_directory(descriptor)
                identity = os.fstat(descriptor)
                yield descriptor
            # Mutation callers hold the stable stripe through descriptor close
            # and directory retirement. Read-only anchor lookups never retire.
            # rmdir itself proves emptiness; unknown receipts/anchors are kept.
            if create:
                current = os.stat(key, dir_fd=parent_fd, follow_symlinks=False)
                if (current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino):
                    raise ManagedArtifactDeleteInconclusiveError("local generation directory identity changed")
                try:
                    os.rmdir(key, dir_fd=parent_fd)
                except OSError as exc:
                    if exc.errno not in {errno.ENOTEMPTY, errno.EEXIST}:
                        raise
                else:
                    os.fsync(parent_fd)

    def _relative_object(self, reference: ManagedArtifactRef, org_id: str, project_id: str) -> Path:
        return _relative_object_path(reference, org_id, project_id)

    @contextlib.contextmanager
    def _object_lock(
        self,
        reference: ManagedArtifactRef,
        org_id: str,
        project_id: str,
    ) -> tp.Iterator[None]:
        """Serialize cooperating writers while pathname safety remains independent."""

        relative = self._relative_object(reference, org_id, project_id).as_posix().encode("ascii")
        stripe = int(hashlib.sha256(relative).hexdigest(), 16) % self._LOCK_STRIPES
        lock_name = f"stripe-{stripe:02x}.lock"
        with self._open_control_directory(self._LOCK_DIRECTORY, create=True) as lock_directory_fd:
            common_flags = (
                os.O_RDWR
                | os.O_NOFOLLOW
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_NOCTTY", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            while True:
                try:
                    descriptor = os.open(lock_name, common_flags, dir_fd=lock_directory_fd)
                    break
                except FileNotFoundError:
                    try:
                        descriptor = os.open(
                            lock_name,
                            common_flags | os.O_CREAT | os.O_EXCL,
                            0o600,
                            dir_fd=lock_directory_fd,
                        )
                        break
                    except FileExistsError:
                        continue
                    except OSError as exc:
                        self._translate_unsafe_path(exc)
                except OSError as exc:
                    self._translate_unsafe_path(exc)
            try:
                metadata = os.fstat(descriptor)
                if not stat_module.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                    raise InvalidManagedArtifactURI("managed artifact lock must be an unlinked regular file")
                fcntl.flock(descriptor, fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)

    @staticmethod
    def _regular_stat(
        descriptor: int,
        *,
        conflict_on_non_file: bool = False,
    ) -> os.stat_result:
        metadata = os.fstat(descriptor)
        if not stat_module.S_ISREG(metadata.st_mode):
            if conflict_on_non_file:
                raise ManagedArtifactConflictError("managed artifact target is not a regular file")
            raise InvalidManagedArtifactURI("managed artifact object must be a regular file")
        if metadata.st_nlink < 1:
            raise ManagedArtifactVersionMismatchError("managed artifact object was unlinked while observed")
        return metadata

    @staticmethod
    def _stable_stat_identity(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            metadata.st_mode,
            metadata.st_dev,
            metadata.st_ino,
            metadata.st_nlink,
            metadata.st_size,
            metadata.st_mtime_ns,
            metadata.st_ctime_ns,
        )

    @classmethod
    def _read_regular_descriptor(
        cls,
        descriptor: int,
        *,
        retain_payload: bool,
        conflict_on_non_file: bool = False,
    ) -> tuple[bytes | None, str, os.stat_result]:
        before = cls._regular_stat(descriptor, conflict_on_non_file=conflict_on_non_file)
        os.lseek(descriptor, 0, os.SEEK_SET)
        digest = hashlib.sha256()
        chunks: list[bytes] | None = [] if retain_payload else None
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            if chunks is not None:
                chunks.append(chunk)
        after = cls._regular_stat(descriptor, conflict_on_non_file=conflict_on_non_file)
        if cls._stable_stat_identity(after) != cls._stable_stat_identity(before):
            raise ManagedArtifactVersionMismatchError("managed artifact object changed while it was read")
        payload = b"".join(chunks) if chunks is not None else None
        return payload, digest.hexdigest(), after

    @classmethod
    def _open_object(cls, directory_fd: int, filename: str) -> int:
        try:
            return os.open(filename, cls._object_open_flags(), dir_fd=directory_fd)
        except OSError as exc:
            cls._translate_unsafe_path(exc)

    @staticmethod
    def _write_descriptor(descriptor: int, payload: bytes) -> None:
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("managed artifact generation write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)

    @staticmethod
    def _same_inode(left: os.stat_result, right: os.stat_result) -> bool:
        return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)

    def _has_exact_generation_anchor(
        self,
        relative: Path,
        digest: str,
        metadata: os.stat_result,
        *,
        conflict_on_unsafe: bool,
    ) -> bool:
        try:
            with self._open_generation_directory(relative, create=False) as generation_fd:
                descriptor = self._open_object(generation_fd, f"{digest}.artifact")
                try:
                    anchor = self._regular_stat(descriptor, conflict_on_non_file=conflict_on_unsafe)
                finally:
                    os.close(descriptor)
        except FileNotFoundError:
            anchor = None
        except InvalidManagedArtifactURI:
            if conflict_on_unsafe:
                raise ManagedArtifactConflictError("managed artifact generation anchor is unsafe") from None
            raise

        if anchor is not None and self._same_inode(anchor, metadata):
            if metadata.st_nlink != 2:
                message = "managed artifact generation has unexpected hard links"
                if conflict_on_unsafe:
                    raise ManagedArtifactConflictError(message)
                raise InvalidManagedArtifactURI(message)
            return True
        if metadata.st_nlink != 1:
            message = "managed artifact object has an unrecognized hard link"
            if conflict_on_unsafe:
                raise ManagedArtifactConflictError(message)
            raise InvalidManagedArtifactURI(message)
        return False

    def _metadata_from_observation(
        self,
        uri: str,
        relative: Path,
        digest: str,
        metadata: os.stat_result,
        *,
        conflict_on_unsafe: bool = False,
    ) -> ManagedArtifactObjectMetadata:
        anchored = self._has_exact_generation_anchor(
            relative,
            digest,
            metadata,
            conflict_on_unsafe=conflict_on_unsafe,
        )
        if anchored:
            version = f"local:2:{digest}:{metadata.st_dev}:{metadata.st_ino}:{metadata.st_ctime_ns}"
        else:
            version = f"local:{metadata.st_dev}:{metadata.st_ino}:{metadata.st_ctime_ns}"
        return ManagedArtifactObjectMetadata(
            uri=uri,
            size_bytes=metadata.st_size,
            object_version=version,
            sha256=digest,
        )

    def _read_descriptor(
        self,
        descriptor: int,
        uri: str,
        relative: Path,
        *,
        object_version: str | None = None,
        conflict_on_unsafe: bool = False,
    ) -> ManagedArtifactRead:
        raw_payload, digest, metadata = self._read_regular_descriptor(
            descriptor,
            retain_payload=True,
            conflict_on_non_file=conflict_on_unsafe,
        )
        assert raw_payload is not None
        observed = self._metadata_from_observation(
            uri,
            relative,
            digest,
            metadata,
            conflict_on_unsafe=conflict_on_unsafe,
        )
        if object_version is not None and observed.object_version != object_version:
            raise ManagedArtifactVersionMismatchError(
                "managed artifact object version does not match",
                current_metadata=observed,
            )
        return ManagedArtifactRead(payload=raw_payload, metadata=observed)

    def _observe_descriptor(
        self,
        descriptor: int,
        uri: str,
        relative: Path,
    ) -> ManagedArtifactObjectMetadata:
        _payload, digest, metadata = self._read_regular_descriptor(descriptor, retain_payload=False)
        return self._metadata_from_observation(uri, relative, digest, metadata)

    def _existing_write(
        self,
        directory_fd: int,
        relative: Path,
        uri: str,
        payload: bytes,
        *,
        created: bool | None = False,
    ) -> ManagedArtifactWrite | None:
        try:
            descriptor = self._open_object(directory_fd, relative.name)
        except FileNotFoundError:
            return None
        try:
            existing = self._read_descriptor(
                descriptor,
                uri,
                relative,
                conflict_on_unsafe=True,
            )
        finally:
            os.close(descriptor)
        if not _payloads_match(existing.payload, payload):
            raise ManagedArtifactConflictError(
                "managed object already contains different bytes",
                current_metadata=existing.metadata,
            )
        return ManagedArtifactWrite(metadata=existing.metadata, created=created)

    def _recover_generation_stage(self, generation_fd: int, filename: str, payload: bytes) -> None:
        """Remove only positively unpublished staging links under the object lock."""

        staging_name = f".pending-{filename}.tmp"
        try:
            descriptor = self._open_object(generation_fd, staging_name)
        except FileNotFoundError:
            return
        try:
            metadata = self._regular_stat(descriptor, conflict_on_non_file=True)
            if metadata.st_uid != os.geteuid() or metadata.st_mode & 0o077:
                raise ManagedArtifactWriteInconclusiveError("managed artifact staging ownership is unsafe")
            if metadata.st_nlink == 2:
                # Publication may have committed just before process death. The
                # other link must be the complete digest anchor, not an alias.
                anchor = self._open_object(generation_fd, filename)
                try:
                    anchored_payload, digest, anchored = self._read_regular_descriptor(
                        anchor, retain_payload=True, conflict_on_non_file=True
                    )
                finally:
                    os.close(anchor)
                if (
                    not self._same_inode(metadata, anchored)
                    or filename != f"{digest}.artifact"
                    or anchored_payload != payload
                ):
                    raise ManagedArtifactWriteInconclusiveError("managed artifact staging publication is ambiguous")
            elif metadata.st_nlink != 1:
                raise ManagedArtifactWriteInconclusiveError("managed artifact staging has unexpected hard links")
            current = os.stat(staging_name, dir_fd=generation_fd, follow_symlinks=False)
            if self._stable_stat_identity(current) != self._stable_stat_identity(metadata):
                raise ManagedArtifactWriteInconclusiveError("managed artifact staging was replaced during recovery")
            os.unlink(staging_name, dir_fd=generation_fd)
            os.fsync(generation_fd)
        finally:
            os.close(descriptor)

    def _open_or_create_generation(
        self,
        generation_fd: int,
        filename: str,
        payload: bytes,
    ) -> tuple[int, bool]:
        self._recover_generation_stage(generation_fd, filename, payload)
        created = False
        try:
            descriptor = self._open_object(generation_fd, filename)
        except FileNotFoundError:
            staging_name = f".pending-{filename}.tmp"
            flags = (
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_NOCTTY", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            try:
                descriptor = os.open(staging_name, flags, 0o600, dir_fd=generation_fd)
            except OSError as exc:
                self._translate_unsafe_path(exc)
            try:
                self._write_descriptor(descriptor, payload)
                os.fchmod(descriptor, 0o400)
                os.fsync(descriptor)
                try:
                    os.link(
                        staging_name,
                        filename,
                        src_dir_fd=generation_fd,
                        dst_dir_fd=generation_fd,
                        follow_symlinks=False,
                    )
                    created = True
                    os.fsync(generation_fd)
                except FileExistsError:
                    existing = self._open_object(generation_fd, filename)
                    os.close(descriptor)
                    descriptor = existing
                self._recover_generation_stage(generation_fd, filename, payload)
            except BaseException:
                os.close(descriptor)
                try:
                    self._recover_generation_stage(generation_fd, filename, payload)
                except (
                    OSError,
                    ManagedArtifactOperationInconclusiveError,
                    ManagedArtifactConflictError,
                    InvalidManagedArtifactURI,
                ):
                    pass
                raise
        try:
            observed_payload, observed_digest, metadata = self._read_regular_descriptor(
                descriptor,
                retain_payload=True,
                conflict_on_non_file=True,
            )
            assert observed_payload is not None
            if not _payloads_match(observed_payload, payload):
                raise ManagedArtifactConflictError("managed artifact generation contains different bytes")
            if filename != f"{observed_digest}.artifact":
                raise ManagedArtifactConflictError("managed artifact generation digest does not match its name")
            if metadata.st_nlink not in {1, 2}:
                raise ManagedArtifactConflictError("managed artifact generation has unexpected hard links")
            return descriptor, created
        except BaseException:
            os.close(descriptor)
            raise

    def _remove_generation_if_orphaned(
        self,
        generation_fd: int,
        filename: str,
        expected: os.stat_result,
    ) -> None:
        try:
            descriptor = self._open_object(generation_fd, filename)
        except FileNotFoundError:
            return
        try:
            current = self._regular_stat(descriptor)
            if not self._same_inode(current, expected):
                raise ManagedArtifactWriteInconclusiveError("local generation cleanup found a replacement")
            if current.st_nlink != 1:
                return
            os.unlink(filename, dir_fd=generation_fd)
            os.fsync(generation_fd)
        finally:
            os.close(descriptor)

    def put_versioned(
        self,
        uri: str,
        payload: bytes,
        *,
        org_id: str,
        project_id: str,
    ) -> ManagedArtifactWrite:
        """Create a scoped immutable generation and return its observed digest."""

        reference = parse_managed_uri(uri)
        payload = _validate_payload(reference, payload)
        relative = self._relative_object(reference, org_id, project_id)
        digest = hashlib.sha256(payload).hexdigest()
        with self._object_lock(reference, org_id, project_id):
            with self._open_directory(relative.parts[:-1], create=True) as directory_fd:
                with self._open_generation_directory(relative, create=True) as generation_fd:
                    try:
                        self._recover_delete_transactions(
                            generation_fd,
                            directory_fd,
                            relative,
                            reference,
                        )
                    except ManagedArtifactDeleteInconclusiveError as exc:
                        raise ManagedArtifactWriteInconclusiveError(
                            f"local deletion recovery blocked the managed artifact write: {exc}"
                        ) from exc
                    existing = self._existing_write(directory_fd, relative, reference.uri, payload)
                    if existing is not None:
                        return existing

                    generation_name = f"{digest}.artifact"
                    generation_descriptor, _anchor_created = self._open_or_create_generation(
                        generation_fd,
                        generation_name,
                        payload,
                    )
                    try:
                        generation_stat = self._regular_stat(
                            generation_descriptor,
                            conflict_on_non_file=True,
                        )
                        try:
                            os.link(
                                generation_name,
                                relative.name,
                                src_dir_fd=generation_fd,
                                dst_dir_fd=directory_fd,
                                follow_symlinks=False,
                            )
                        except FileExistsError:
                            existing = self._existing_write(directory_fd, relative, reference.uri, payload)
                            self._remove_generation_if_orphaned(
                                generation_fd,
                                generation_name,
                                generation_stat,
                            )
                            if existing is None:
                                raise ManagedArtifactWriteInconclusiveError(
                                    "local managed artifact create was inconclusive"
                                ) from None
                            return existing
                        except OSError as exc:
                            raise ManagedArtifactWriteInconclusiveError(
                                "local managed artifact create was inconclusive"
                            ) from exc

                        try:
                            os.fsync(directory_fd)
                            descriptor = self._open_object(directory_fd, relative.name)
                            try:
                                current_stat = self._regular_stat(descriptor, conflict_on_non_file=True)
                                if not self._same_inode(current_stat, generation_stat):
                                    raise ManagedArtifactWriteInconclusiveError(
                                        "local managed artifact was replaced during create"
                                    )
                                observed = self._read_descriptor(
                                    descriptor,
                                    reference.uri,
                                    relative,
                                    conflict_on_unsafe=True,
                                )
                            finally:
                                os.close(descriptor)
                        except Exception as exc:
                            self._remove_generation_if_orphaned(
                                generation_fd,
                                generation_name,
                                generation_stat,
                            )
                            if isinstance(exc, (ManagedArtifactConflictError, ManagedArtifactWriteInconclusiveError)):
                                raise
                            raise ManagedArtifactWriteInconclusiveError(
                                "local managed artifact create was inconclusive"
                            ) from exc
                        return ManagedArtifactWrite(metadata=observed.metadata, created=True)
                    finally:
                        os.close(generation_descriptor)

    def get_versioned(
        self,
        uri: str,
        *,
        org_id: str,
        project_id: str,
        object_version: str | None = None,
    ) -> ManagedArtifactRead:
        """Read one current local file, optionally requiring its exact identity."""

        reference = parse_managed_uri(uri)
        if object_version is not None:
            _parse_local_object_version(object_version)
        relative = self._relative_object(reference, org_id, project_id)
        try:
            with self._open_directory(relative.parts[:-1], create=False) as directory_fd:
                descriptor = self._open_object(directory_fd, relative.name)
                try:
                    observed = self._read_descriptor(
                        descriptor,
                        reference.uri,
                        relative,
                        object_version=object_version,
                    )
                finally:
                    os.close(descriptor)
        except FileNotFoundError:
            raise FileNotFoundError("managed artifact object is absent") from None
        try:
            payload = _validate_loaded_payload(reference, observed.payload)
        except ValueError as exc:
            raise ManagedArtifactConflictError(
                "managed artifact bytes do not match their content-addressed URI",
                current_metadata=observed.metadata,
            ) from exc
        return ManagedArtifactRead(payload=payload, metadata=observed.metadata)

    def put(self, uri: str, payload: bytes, *, org_id: str, project_id: str) -> str:
        """Compatibility wrapper returning the canonical logical URI."""

        return self.put_versioned(uri, payload, org_id=org_id, project_id=project_id).metadata.uri

    def get(self, uri: str, *, org_id: str, project_id: str) -> bytes:
        """Compatibility wrapper returning current scoped object bytes."""

        return self.get_versioned(uri, org_id=org_id, project_id=project_id).payload

    def exists(self, uri: str, *, org_id: str, project_id: str) -> bool:
        """Return whether a regular, safely linked scoped local object exists."""

        parse_managed_uri(uri)
        return self.stat(uri, org_id=org_id, project_id=project_id) is not None

    @contextlib.contextmanager
    def _inventory_scope_directory(self, org_id: str, project_id: str) -> tp.Iterator[int | None]:
        """Pin one root-confined scope directory until inventory finishes."""

        with contextlib.ExitStack() as stack:
            try:
                descriptor = stack.enter_context(
                    self._open_directory(("scopes", storage_scope(org_id, project_id)), create=False)
                )
            except FileNotFoundError:
                yield None
                return
            except (OSError, InvalidManagedArtifactURI) as exc:
                raise ManagedArtifactInventoryError(
                    "managed artifact inventory scope must contain only real directories"
                ) from exc
            yield descriptor

    def _local_inventory_metadata(
        self,
        directory_fd: int,
        filename: str,
        uri: str,
        relative: Path,
    ) -> ManagedArtifactObjectMetadata | None:
        """Observe metadata relative to a pinned, no-follow kind directory."""

        try:
            descriptor = os.open(filename, self._object_open_flags(), dir_fd=directory_fd)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise ManagedArtifactInventoryError("could not open managed artifact inventory object") from exc
        try:
            try:
                return self._observe_descriptor(descriptor, uri, relative)
            except (InvalidManagedArtifactURI, ManagedArtifactVersionMismatchError) as exc:
                raise ManagedArtifactInventoryError("managed artifact inventory object was unsafe or unstable") from exc
        finally:
            os.close(descriptor)

    @staticmethod
    def _reference_from_local_entry(kind: ManagedArtifactKind, filename: str) -> ManagedArtifactRef:
        extension = _OBJECT_EXTENSIONS[kind]
        if not filename.endswith(extension):
            raise ManagedArtifactInventoryError(f"managed artifact inventory found an unexpected file in {kind.value!r}")
        object_id = filename[: -len(extension)]
        try:
            reference = ManagedArtifactRef(kind, _validate_object_id(kind, object_id))
        except (InvalidManagedArtifactURI, ValueError) as exc:
            raise ManagedArtifactInventoryError(
                f"managed artifact inventory found a non-canonical object in {kind.value!r}"
            ) from exc
        if filename != f"{reference.object_id}{extension}":
            raise ManagedArtifactInventoryError(
                f"managed artifact inventory found a non-canonical object in {kind.value!r}"
            )
        return reference

    def list_scope(
        self,
        *,
        org_id: str,
        project_id: str,
        cursor: str | None = None,
        limit: int = MAX_MANAGED_ARTIFACT_INVENTORY_PAGE_SIZE,
    ) -> ManagedArtifactInventoryPage:
        """List a bounded page from exactly one explicitly supplied local scope."""

        cursor = _validate_inventory_page_request(cursor, limit)
        known = {kind.value: kind for kind in ManagedArtifactKind}

        def observed(scope_fd: int) -> tp.Iterator[ManagedArtifactObjectMetadata]:
            try:
                with os.scandir(scope_fd) as entries:
                    kinds = tuple(entries)
            except OSError as exc:
                raise ManagedArtifactInventoryError(f"could not list managed artifact inventory scope: {exc}") from exc
            for entry in kinds:
                if entry.name not in known or entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                    raise ManagedArtifactInventoryError("managed artifact inventory scope contains an unexpected entry")
                kind = known[entry.name]
                try:
                    kind_fd = os.open(entry.name, self._directory_open_flags(), dir_fd=scope_fd)
                except OSError as exc:
                    raise ManagedArtifactInventoryError(
                        f"could not open managed artifact inventory kind {kind.value!r}"
                    ) from exc
                try:
                    with os.scandir(kind_fd) as children:
                        for child in children:
                            if child.is_symlink() or not child.is_file(follow_symlinks=False):
                                raise ManagedArtifactInventoryError(
                                    f"managed artifact inventory kind {kind.value!r} contains a non-file entry"
                                )
                            reference = self._reference_from_local_entry(kind, child.name)
                            if cursor is not None and reference.uri <= cursor:
                                continue
                            relative = self._relative_object(reference, org_id, project_id)
                            metadata = self._local_inventory_metadata(kind_fd, child.name, reference.uri, relative)
                            if metadata is None:
                                raise ManagedArtifactInventoryError(
                                    "managed artifact disappeared while its inventory page was being observed"
                                )
                            yield metadata
                except OSError as exc:
                    raise ManagedArtifactInventoryError(
                        f"could not list managed artifact inventory kind {kind.value!r}"
                    ) from exc
                finally:
                    os.close(kind_fd)

        with self._inventory_scope_directory(org_id, project_id) as scope_fd:
            if scope_fd is None:
                return ManagedArtifactInventoryPage(objects=())
            page = heapq.nsmallest(limit + 1, observed(scope_fd), key=lambda item: item.uri)
        has_more = len(page) > limit
        selected = tuple(page[:limit])
        return ManagedArtifactInventoryPage(
            objects=selected,
            next_cursor=selected[-1].uri if has_more else None,
        )

    def stat(
        self,
        uri: str,
        *,
        org_id: str,
        project_id: str,
    ) -> ManagedArtifactObjectMetadata | None:
        """Observe one current local object's exact digest, size, and identity."""

        reference = parse_managed_uri(uri)
        relative = self._relative_object(reference, org_id, project_id)
        try:
            with self._open_directory(relative.parts[:-1], create=False) as directory_fd:
                descriptor = self._open_object(directory_fd, relative.name)
                try:
                    return self._observe_descriptor(descriptor, reference.uri, relative)
                finally:
                    os.close(descriptor)
        except FileNotFoundError:
            return None
        except ManagedArtifactVersionMismatchError as exc:
            raise ManagedArtifactInventoryError("managed artifact changed during metadata observation") from exc
        except OSError as exc:
            raise ManagedArtifactInventoryError("could not observe managed artifact inventory object") from exc

    @contextlib.contextmanager
    def _open_delete_transaction(
        self,
        generation_fd: int,
        intent: _LocalDeleteIntent,
        *,
        create: bool,
    ) -> tp.Iterator[int]:
        name = intent.directory_name
        if create:
            try:
                os.mkdir(name, mode=0o700, dir_fd=generation_fd)
                os.fsync(generation_fd)
            except FileExistsError:
                pass
            except OSError as exc:
                self._translate_unsafe_path(exc)
        try:
            descriptor = os.open(name, self._directory_open_flags(), dir_fd=generation_fd)
        except OSError as exc:
            self._translate_unsafe_path(exc)
        try:
            self._validate_control_directory(descriptor)
            yield descriptor
        finally:
            os.close(descriptor)

    @contextlib.contextmanager
    def _open_archived_delete_receipt(
        self,
        generation_fd: int,
        intent: _LocalDeleteIntent,
    ) -> tp.Iterator[int]:
        try:
            descriptor = os.open(
                intent.trash_directory_name,
                self._directory_open_flags(),
                dir_fd=generation_fd,
            )
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise ManagedArtifactDeleteInconclusiveError("local deletion trash receipt could not be opened") from exc
        try:
            try:
                self._validate_control_directory(descriptor)
            except InvalidManagedArtifactURI as exc:
                raise ManagedArtifactDeleteInconclusiveError("local deletion trash receipt is unsafe") from exc
            yield descriptor
        finally:
            os.close(descriptor)

    @staticmethod
    def _bounded_local_directory_names(
        directory_fd: int,
        *,
        limit: int,
        description: str,
    ) -> tuple[str, ...]:
        names: list[str] = []
        try:
            with os.scandir(directory_fd) as entries:
                for entry in entries:
                    if len(names) >= limit:
                        raise ManagedArtifactDeleteInconclusiveError(f"{description} exceeded its bounded scan limit")
                    names.append(entry.name)
        except ManagedArtifactDeleteInconclusiveError:
            raise
        except OSError as exc:
            raise ManagedArtifactDeleteInconclusiveError(f"{description} could not be listed") from exc
        return tuple(sorted(names))

    def _cleanup_archived_delete_receipt(self, generation_fd: int, intent: _LocalDeleteIntent) -> None:
        """Remove one completed receipt without opening or traversing its entries."""

        receipt_name = intent.trash_directory_name
        try:
            receipt_context = self._open_archived_delete_receipt(generation_fd, intent)
            with receipt_context as receipt_fd:
                names = self._bounded_local_directory_names(
                    receipt_fd,
                    limit=_MAX_LOCAL_DELETE_RECEIPT_ENTRIES,
                    description="local deletion trash receipt",
                )
                if any(name not in _LOCAL_DELETE_RECEIPT_ENTRIES for name in names):
                    raise ManagedArtifactDeleteInconclusiveError(
                        "local deletion trash receipt contains an unexpected entry"
                    )

                # Preflight the complete bounded set before removing anything. A
                # known name may be a symlink, FIFO, socket, or device: unlinkat
                # removes that directory entry itself and never opens or follows
                # it. Directories are retained because cleanup never recurses.
                for name in names:
                    try:
                        metadata = os.stat(name, dir_fd=receipt_fd, follow_symlinks=False)
                    except FileNotFoundError:
                        continue
                    except OSError as exc:
                        raise ManagedArtifactDeleteInconclusiveError(
                            "local deletion trash receipt entry could not be observed"
                        ) from exc
                    if stat_module.S_ISDIR(metadata.st_mode):
                        raise ManagedArtifactDeleteInconclusiveError(
                            "local deletion trash receipt contains a nested directory"
                        )

                # ``complete`` alone may describe a deliberately retained,
                # unvalidated inode. Only the stronger durable marker proves
                # that every payload-bearing receipt entry is either truncated
                # or has a validated replacement in the normal object layout.
                # The marker is unlinked last. If a crash has already removed
                # it, an empty/marker-only receipt is a partial safe cleanup;
                # any remaining payload-bearing name is retained instead.
                cleanup_safe = self._transaction_marker_exists(receipt_fd, "cleanup-safe")
                payload_entries = {
                    "destroyed-anchor",
                    "detached",
                    "preserved-anchor",
                    "replacement-anchor",
                }.intersection(names)
                if not cleanup_safe and payload_entries:
                    return
                cleanup_order = tuple(name for name in names if name != "cleanup-safe")
                if cleanup_safe:
                    cleanup_order += ("cleanup-safe",)

                for name in cleanup_order:
                    try:
                        os.unlink(name, dir_fd=receipt_fd)
                    except FileNotFoundError:
                        continue
                    except OSError as exc:
                        raise ManagedArtifactDeleteInconclusiveError(
                            "local deletion trash receipt entry could not be removed"
                        ) from exc
                try:
                    os.fsync(receipt_fd)
                except OSError as exc:
                    raise ManagedArtifactDeleteInconclusiveError(
                        "local deletion trash receipt cleanup could not be persisted"
                    ) from exc

                try:
                    opened_receipt = os.fstat(receipt_fd)
                except OSError as exc:
                    raise ManagedArtifactDeleteInconclusiveError(
                        "local deletion trash receipt descriptor could not be re-observed"
                    ) from exc
                try:
                    current_receipt = os.stat(receipt_name, dir_fd=generation_fd, follow_symlinks=False)
                except FileNotFoundError:
                    current_receipt = None
                except OSError as exc:
                    raise ManagedArtifactDeleteInconclusiveError(
                        "local deletion trash receipt could not be re-observed"
                    ) from exc
                if current_receipt is not None:
                    if not stat_module.S_ISDIR(current_receipt.st_mode) or not self._same_inode(
                        opened_receipt,
                        current_receipt,
                    ):
                        raise ManagedArtifactDeleteInconclusiveError(
                            "local deletion trash receipt was replaced during cleanup"
                        )
                    try:
                        os.rmdir(receipt_name, dir_fd=generation_fd)
                    except FileNotFoundError:
                        pass
                    except OSError as exc:
                        raise ManagedArtifactDeleteInconclusiveError(
                            "local deletion trash receipt could not be removed"
                        ) from exc
        except FileNotFoundError:
            return
        try:
            os.fsync(generation_fd)
        except OSError as exc:
            raise ManagedArtifactDeleteInconclusiveError(
                "local deletion trash receipt retirement could not be persisted"
            ) from exc

    def _archive_delete_transaction(self, generation_fd: int, intent: _LocalDeleteIntent) -> None:
        source = intent.directory_name
        destination = intent.trash_directory_name
        try:
            destination_stat = os.stat(destination, dir_fd=generation_fd, follow_symlinks=False)
        except FileNotFoundError:
            destination_stat = None
        except OSError as exc:
            raise ManagedArtifactDeleteInconclusiveError("local deletion trash receipt could not be observed") from exc
        if destination_stat is not None:
            if not stat_module.S_ISDIR(destination_stat.st_mode):
                raise ManagedArtifactDeleteInconclusiveError("local deletion trash receipt is unsafe")
            try:
                os.stat(source, dir_fd=generation_fd, follow_symlinks=False)
            except FileNotFoundError:
                self._cleanup_archived_delete_receipt(generation_fd, intent)
                return
            except OSError as exc:
                raise ManagedArtifactDeleteInconclusiveError("local deletion transaction could not be observed") from exc
            raise ManagedArtifactDeleteInconclusiveError(
                "local deletion transaction collides with an existing trash receipt"
            )
        try:
            os.rename(source, destination, src_dir_fd=generation_fd, dst_dir_fd=generation_fd)
            os.fsync(generation_fd)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise ManagedArtifactDeleteInconclusiveError("local deletion transaction could not be archived") from exc
        self._cleanup_archived_delete_receipt(generation_fd, intent)

    @staticmethod
    def _matches_delete_intent(metadata: os.stat_result, intent: _LocalDeleteIntent) -> bool:
        return (metadata.st_dev, metadata.st_ino) == (intent.device, intent.inode)

    def _matching_transaction_entry(
        self,
        directory_fd: int,
        name: str,
        intent: _LocalDeleteIntent,
    ) -> os.stat_result | None:
        try:
            descriptor = self._open_object(directory_fd, name)
        except FileNotFoundError:
            return None
        try:
            try:
                metadata = self._regular_stat(descriptor)
            except InvalidManagedArtifactURI:
                return None
            if not self._matches_delete_intent(metadata, intent):
                return None
            return metadata
        finally:
            os.close(descriptor)

    def _transaction_marker_exists(self, transaction_fd: int, name: str) -> bool:
        try:
            descriptor = self._open_object(transaction_fd, name)
        except FileNotFoundError:
            return False
        try:
            metadata = self._regular_stat(descriptor)
            if metadata.st_nlink != 1 or metadata.st_size != 0:
                raise ManagedArtifactDeleteInconclusiveError("local deletion transaction marker is unsafe")
            return True
        finally:
            os.close(descriptor)

    def _create_transaction_marker(self, transaction_fd: int, name: str) -> None:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | os.O_NOFOLLOW
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOCTTY", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(name, flags, 0o600, dir_fd=transaction_fd)
        except FileExistsError:
            if not self._transaction_marker_exists(transaction_fd, name):
                raise ManagedArtifactDeleteInconclusiveError("local deletion transaction marker disappeared") from None
            return
        except OSError as exc:
            self._translate_unsafe_path(exc)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(transaction_fd)

    def _current_read_from_directory(
        self,
        directory_fd: int,
        relative: Path,
        uri: str,
    ) -> ManagedArtifactRead | None:
        try:
            descriptor = self._open_object(directory_fd, relative.name)
        except FileNotFoundError:
            return None
        try:
            return self._read_descriptor(
                descriptor,
                uri,
                relative,
                conflict_on_unsafe=True,
            )
        finally:
            os.close(descriptor)

    def _canonical_matches_delete_intent(
        self,
        directory_fd: int,
        relative: Path,
        intent: _LocalDeleteIntent,
    ) -> bool:
        try:
            descriptor = self._open_object(directory_fd, relative.name)
        except FileNotFoundError:
            return False
        try:
            metadata = self._regular_stat(descriptor, conflict_on_non_file=True)
            return self._matches_delete_intent(metadata, intent)
        finally:
            os.close(descriptor)

    @staticmethod
    def _writable_object_open_flags() -> int:
        return (
            os.O_RDWR
            | os.O_NOFOLLOW
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOCTTY", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )

    def _open_writable_transaction_entry(
        self,
        directory_fd: int,
        name: str,
        intent: _LocalDeleteIntent,
    ) -> tuple[int, str, os.stat_result] | None:
        try:
            read_descriptor = self._open_object(directory_fd, name)
        except FileNotFoundError:
            return None
        try:
            before = self._regular_stat(read_descriptor)
            if not self._matches_delete_intent(before, intent):
                return None
            os.fchmod(read_descriptor, 0o600)
            try:
                descriptor = os.open(
                    name,
                    self._writable_object_open_flags(),
                    dir_fd=directory_fd,
                )
            except OSError as exc:
                self._translate_unsafe_path(exc)
        finally:
            os.close(read_descriptor)
        try:
            _payload, digest, metadata = self._read_regular_descriptor(
                descriptor,
                retain_payload=False,
            )
            if not self._matches_delete_intent(metadata, intent):
                os.close(descriptor)
                return None
            return descriptor, digest, metadata
        except Exception:
            os.close(descriptor)
            raise

    def _move_expected_anchor_into_transaction(
        self,
        generation_fd: int,
        transaction_fd: int,
        intent: _LocalDeleteIntent,
        destination: str,
    ) -> bool:
        if not intent.anchored:
            return False
        anchor_name = f"{intent.digest}.artifact"
        destination_stat = self._matching_transaction_entry(transaction_fd, destination, intent)
        if destination_stat is not None:
            source_stat = self._matching_transaction_entry(generation_fd, anchor_name, intent)
            if source_stat is not None:
                raise ManagedArtifactDeleteInconclusiveError(
                    "local deletion transaction retained duplicate generation anchors"
                )
            return True
        try:
            destination_lstat = os.stat(destination, dir_fd=transaction_fd, follow_symlinks=False)
        except FileNotFoundError:
            destination_lstat = None
        if destination_lstat is not None:
            raise ManagedArtifactDeleteInconclusiveError("local deletion transaction anchor receipt is unsafe")
        source_stat = self._matching_transaction_entry(generation_fd, anchor_name, intent)
        if source_stat is None:
            return False
        try:
            os.rename(
                anchor_name,
                destination,
                src_dir_fd=generation_fd,
                dst_dir_fd=transaction_fd,
            )
        except OSError as exc:
            raise ManagedArtifactDeleteInconclusiveError("local generation anchor could not be retired durably") from exc
        os.fsync(generation_fd)
        os.fsync(transaction_fd)
        return True

    def _move_replacement_anchor_into_transaction(
        self,
        generation_fd: int,
        transaction_fd: int,
        digest: str,
        metadata: os.stat_result,
    ) -> None:
        anchor_name = f"{digest}.artifact"
        destination = "replacement-anchor"
        try:
            destination_descriptor = self._open_object(transaction_fd, destination)
        except FileNotFoundError:
            destination_descriptor = None
        if destination_descriptor is not None:
            try:
                destination_stat = self._regular_stat(destination_descriptor)
                if not self._same_inode(destination_stat, metadata):
                    raise ManagedArtifactDeleteInconclusiveError(
                        "local replacement anchor receipt has a different identity"
                    )
            finally:
                os.close(destination_descriptor)
            try:
                source_descriptor = self._open_object(generation_fd, anchor_name)
            except FileNotFoundError:
                return
            try:
                source_stat = self._regular_stat(source_descriptor)
                if self._same_inode(source_stat, metadata):
                    raise ManagedArtifactDeleteInconclusiveError(
                        "local replacement retained duplicate generation anchors"
                    )
            finally:
                os.close(source_descriptor)
            # A different inode is the replacement copy created by an earlier
            # recovery attempt that crashed before recording completion.
            return

        try:
            source_descriptor = self._open_object(generation_fd, anchor_name)
        except FileNotFoundError:
            return
        try:
            source_stat = self._regular_stat(source_descriptor)
            if not self._same_inode(source_stat, metadata):
                return
        finally:
            os.close(source_descriptor)
        try:
            os.rename(
                anchor_name,
                destination,
                src_dir_fd=generation_fd,
                dst_dir_fd=transaction_fd,
            )
        except OSError as exc:
            raise ManagedArtifactDeleteInconclusiveError(
                "local replacement generation anchor could not be quarantined"
            ) from exc
        os.fsync(generation_fd)
        os.fsync(transaction_fd)

    def _read_detached_transaction_entry(
        self,
        transaction_fd: int,
    ) -> tuple[bytes, str, os.stat_result] | None:
        try:
            descriptor = self._open_object(transaction_fd, "detached")
        except FileNotFoundError:
            return None
        try:
            raw_payload, digest, metadata = self._read_regular_descriptor(
                descriptor,
                retain_payload=True,
            )
            assert raw_payload is not None
            return raw_payload, digest, metadata
        finally:
            os.close(descriptor)

    def _install_preserved_replacement(
        self,
        generation_fd: int,
        directory_fd: int,
        relative: Path,
        reference: ManagedArtifactRef,
        payload: bytes,
    ) -> ManagedArtifactObjectMetadata | None:
        try:
            payload = _validate_loaded_payload(reference, payload)
        except ValueError:
            return None
        digest = hashlib.sha256(payload).hexdigest()
        current = self._current_read_from_directory(directory_fd, relative, reference.uri)
        if current is not None:
            if current.metadata.sha256 != digest or not _payloads_match(current.payload, payload):
                return None
            # A raw scope entry is not itself durable replacement proof. Link
            # that exact pathname into the trusted generation directory, then
            # re-open and revalidate the same inode before cleanup may retire
            # the transaction copy. A concurrent C either remains canonical or
            # is anchored too; detached B is retained unless the exact B inode
            # has the normal two-link layout.
            generation_name = f"{digest}.artifact"
            try:
                os.link(
                    relative.name,
                    generation_name,
                    src_dir_fd=directory_fd,
                    dst_dir_fd=generation_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                pass
            except OSError as exc:
                raise ManagedArtifactDeleteInconclusiveError(
                    "preserved local replacement could not be anchored"
                ) from exc
            descriptor = self._open_object(directory_fd, relative.name)
            try:
                anchored = self._read_descriptor(
                    descriptor,
                    reference.uri,
                    relative,
                    conflict_on_unsafe=True,
                )
                if (
                    anchored.metadata.sha256 != digest
                    or not _payloads_match(anchored.payload, payload)
                    or not anchored.metadata.object_version.startswith(f"local:2:{digest}:")
                ):
                    return None
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            # Persist file contents before either directory entry.
            os.fsync(generation_fd)
            os.fsync(directory_fd)
            return anchored.metadata

        generation_name = f"{digest}.artifact"
        generation_descriptor, _created = self._open_or_create_generation(
            generation_fd,
            generation_name,
            payload,
        )
        try:
            generation_stat = self._regular_stat(
                generation_descriptor,
                conflict_on_non_file=True,
            )
            if generation_stat.st_nlink != 1:
                current = self._current_read_from_directory(directory_fd, relative, reference.uri)
                if current is not None and (
                    current.metadata.sha256 == digest and _payloads_match(current.payload, payload)
                ):
                    return current.metadata
                raise ManagedArtifactDeleteInconclusiveError(
                    "preserved local replacement has an ambiguous generation anchor"
                )
            try:
                os.link(
                    generation_name,
                    relative.name,
                    src_dir_fd=generation_fd,
                    dst_dir_fd=directory_fd,
                    follow_symlinks=False,
                )
            except FileExistsError:
                current = self._current_read_from_directory(directory_fd, relative, reference.uri)
                if current is None:
                    raise ManagedArtifactDeleteInconclusiveError(
                        "local replacement appeared without an observable current object"
                    ) from None
                return current.metadata
            except OSError as exc:
                raise ManagedArtifactDeleteInconclusiveError("local replacement could not be restored") from exc
            os.fsync(directory_fd)
            os.fsync(generation_fd)
            current = self._current_read_from_directory(directory_fd, relative, reference.uri)
            if current is None:
                raise ManagedArtifactDeleteInconclusiveError("restored local replacement disappeared")
            return current.metadata
        finally:
            os.close(generation_descriptor)

    def _preserve_special_detached_entry(
        self,
        transaction_fd: int,
        directory_fd: int,
        relative: Path,
    ) -> None:
        try:
            metadata = os.stat("detached", dir_fd=transaction_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        if stat_module.S_ISLNK(metadata.st_mode) or stat_module.S_ISDIR(metadata.st_mode):
            return
        try:
            os.link(
                "detached",
                relative.name,
                src_dir_fd=transaction_fd,
                dst_dir_fd=directory_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            return
        except OSError:
            return
        os.fsync(directory_fd)
        os.fsync(transaction_fd)

    def _restore_detached_replacement(
        self,
        generation_fd: int,
        transaction_fd: int,
        directory_fd: int,
        relative: Path,
        reference: ManagedArtifactRef,
        intent: _LocalDeleteIntent,
        *,
        force: bool,
    ) -> ManagedArtifactObjectMetadata | None:
        try:
            detached_lstat = os.stat("detached", dir_fd=transaction_fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        if not stat_module.S_ISREG(detached_lstat.st_mode):
            self._preserve_special_detached_entry(transaction_fd, directory_fd, relative)
            raise ManagedArtifactDeleteInconclusiveError(
                "unvalidated local replacement was retained in the deletion transaction"
            )
        observed = self._read_detached_transaction_entry(transaction_fd)
        if observed is None:
            return None
        payload, digest, metadata = observed
        if not force and self._matches_delete_intent(metadata, intent):
            return None
        self._move_replacement_anchor_into_transaction(
            generation_fd,
            transaction_fd,
            digest,
            metadata,
        )
        restored = self._install_preserved_replacement(
            generation_fd,
            directory_fd,
            relative,
            reference,
            payload,
        )
        if restored is None:
            raise ManagedArtifactDeleteInconclusiveError(
                "unvalidated local replacement was retained in the deletion transaction"
            )
        return restored

    def _complete_unvalidated_transaction(
        self,
        generation_fd: int,
        transaction_fd: int,
        directory_fd: int,
        relative: Path,
        reference: ManagedArtifactRef,
        intent: _LocalDeleteIntent,
    ) -> None:
        self._move_expected_anchor_into_transaction(
            generation_fd,
            transaction_fd,
            intent,
            "preserved-anchor",
        )
        self._restore_detached_replacement(
            generation_fd,
            transaction_fd,
            directory_fd,
            relative,
            reference,
            intent,
            force=True,
        )
        self._create_transaction_marker(transaction_fd, "complete")

    def _recover_delete_transaction(
        self,
        generation_fd: int,
        transaction_fd: int,
        directory_fd: int,
        relative: Path,
        reference: ManagedArtifactRef,
        intent: _LocalDeleteIntent,
    ) -> bool:
        """Finish one self-describing deletion intent without pathname deletion.

        Before ``validated`` exists, detached bytes must still hash to the
        intended generation (or its exact anchor must). Once that marker is
        durable, descriptor truncation may be repeated after any crash. A
        mismatched detached entry is never truncated: its bytes stay in the
        transaction receipt and a valid regular payload is copied back through
        the normal anchored-generation layout.
        """

        if self._transaction_marker_exists(transaction_fd, "complete"):
            return False

        validated = self._transaction_marker_exists(transaction_fd, "validated")
        candidate: tuple[int, str, os.stat_result] | None = None
        detached_stat = self._matching_transaction_entry(transaction_fd, "detached", intent)
        canonical_matches = self._canonical_matches_delete_intent(directory_fd, relative, intent)

        if not validated:
            candidate_directory_fd: int | None = None
            candidate_name: str | None = None
            required_links: int | None = None
            if detached_stat is not None:
                candidate_directory_fd = transaction_fd
                candidate_name = "detached"
                required_links = 2 if intent.anchored else 1
                if intent.anchored:
                    anchor_stat = self._matching_transaction_entry(
                        generation_fd,
                        f"{intent.digest}.artifact",
                        intent,
                    )
                    if anchor_stat is None or not self._same_inode(anchor_stat, detached_stat):
                        candidate_directory_fd = None
            elif intent.anchored and not canonical_matches:
                anchor_stat = self._matching_transaction_entry(
                    generation_fd,
                    f"{intent.digest}.artifact",
                    intent,
                )
                if anchor_stat is not None:
                    candidate_directory_fd = generation_fd
                    candidate_name = f"{intent.digest}.artifact"
                    required_links = 1
            elif canonical_matches:
                # The durable directory was created, but the canonical rename did
                # not commit. The caller can retry the detach with this intent.
                return False

            if candidate_directory_fd is not None and candidate_name is not None:
                candidate = self._open_writable_transaction_entry(
                    candidate_directory_fd,
                    candidate_name,
                    intent,
                )
                if candidate is not None:
                    descriptor, digest, metadata = candidate
                    try:
                        if digest == intent.digest and metadata.st_nlink == required_links:
                            self._create_transaction_marker(transaction_fd, "validated")
                            validated = True
                    finally:
                        # Ownership transfers to the truncation block only after
                        # validation is durable, including both marker fsyncs.
                        if not validated:
                            os.close(descriptor)
                            candidate = None

            if not validated:
                self._complete_unvalidated_transaction(
                    generation_fd,
                    transaction_fd,
                    directory_fd,
                    relative,
                    reference,
                    intent,
                )
                return True

        if candidate is None:
            candidates = (
                (generation_fd, f"{intent.digest}.artifact"),
                (transaction_fd, "detached"),
                (transaction_fd, "destroyed-anchor"),
            )
            for candidate_directory_fd, candidate_name in candidates:
                candidate = self._open_writable_transaction_entry(
                    candidate_directory_fd,
                    candidate_name,
                    intent,
                )
                if candidate is not None:
                    break

        if candidate is not None:
            descriptor, _digest, _metadata = candidate
            try:
                # This is the deletion primitive: it operates on the descriptor
                # validated after detachment, not on a pathname that can be
                # replaced between an fstat and unlink.
                os.ftruncate(descriptor, 0)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

        self._move_expected_anchor_into_transaction(
            generation_fd,
            transaction_fd,
            intent,
            "destroyed-anchor",
        )
        self._restore_detached_replacement(
            generation_fd,
            transaction_fd,
            directory_fd,
            relative,
            reference,
            intent,
            force=False,
        )
        # ``complete`` also describes conservative transactions that retain an
        # unvalidated payload. Cleanup requires this stronger marker, written
        # only after the selected inode was descriptor-truncated and every
        # observed replacement was durably restored through the normal layout.
        self._create_transaction_marker(transaction_fd, "cleanup-safe")
        self._create_transaction_marker(transaction_fd, "complete")
        return True

    def _delete_transaction_intents(
        self,
        generation_fd: int,
    ) -> tuple[tuple[_LocalDeleteIntent, ...], tuple[_LocalDeleteIntent, ...]]:
        names = self._bounded_local_directory_names(
            generation_fd,
            limit=_MAX_LOCAL_GENERATION_DIRECTORY_ENTRIES,
            description="local generation directory",
        )
        active: list[_LocalDeleteIntent] = []
        archived: list[_LocalDeleteIntent] = []
        for name in names:
            if name.startswith(".delete-"):
                parser = _parse_local_delete_intent
                destination = active
                description = "local deletion transaction"
            elif name.startswith(".trash-"):
                parser = _parse_local_trash_intent
                destination = archived
                description = "local deletion trash receipt"
            else:
                continue
            try:
                intent = parser(name)
            except ValueError as exc:
                raise ManagedArtifactDeleteInconclusiveError(f"{description} has an invalid identity") from exc
            try:
                metadata = os.stat(name, dir_fd=generation_fd, follow_symlinks=False)
            except OSError as exc:
                raise ManagedArtifactDeleteInconclusiveError(f"{description} could not be observed") from exc
            if not stat_module.S_ISDIR(metadata.st_mode):
                raise ManagedArtifactDeleteInconclusiveError(f"{description} is not a directory")
            destination.append(intent)
        if set(active).intersection(archived):
            raise ManagedArtifactDeleteInconclusiveError(
                "local deletion transaction collides with an existing trash receipt"
            )
        return tuple(active), tuple(archived)

    def _recover_delete_transactions(
        self,
        generation_fd: int,
        directory_fd: int,
        relative: Path,
        reference: ManagedArtifactRef,
    ) -> tuple[_LocalDeleteIntent, ...]:
        active, archived = self._delete_transaction_intents(generation_fd)
        completed: list[_LocalDeleteIntent] = []
        for intent in archived:
            self._cleanup_archived_delete_receipt(generation_fd, intent)
            completed.append(intent)
        for intent in active:
            with self._open_delete_transaction(generation_fd, intent, create=False) as transaction_fd:
                recovered = self._recover_delete_transaction(
                    generation_fd,
                    transaction_fd,
                    directory_fd,
                    relative,
                    reference,
                    intent,
                )
                is_complete = self._transaction_marker_exists(transaction_fd, "complete")
            if recovered and not is_complete:
                raise ManagedArtifactDeleteInconclusiveError(
                    "local deletion transaction recovery was not durably completed"
                )
            if is_complete:
                self._archive_delete_transaction(generation_fd, intent)
                completed.append(intent)
        return tuple(completed)

    def _retire_absent_generation(
        self,
        generation_fd: int,
        directory_fd: int,
        relative: Path,
        reference: ManagedArtifactRef,
        expected: _LocalObjectVersion,
    ) -> _LocalDeleteIntent | None:
        if expected.generation_digest is None:
            return None
        intent = _LocalDeleteIntent(
            layout=2,
            digest=expected.generation_digest,
            device=expected.device,
            inode=expected.inode,
            ctime_ns=expected.ctime_ns,
        )
        anchor = self._matching_transaction_entry(
            generation_fd,
            f"{intent.digest}.artifact",
            intent,
        )
        if anchor is None:
            return None
        with self._open_delete_transaction(generation_fd, intent, create=True) as transaction_fd:
            recovered = self._recover_delete_transaction(
                generation_fd,
                transaction_fd,
                directory_fd,
                relative,
                reference,
                intent,
            )
            is_complete = self._transaction_marker_exists(transaction_fd, "complete")
        if recovered and not is_complete:
            raise ManagedArtifactDeleteInconclusiveError("local deletion transaction recovery was not durably completed")
        if is_complete:
            self._archive_delete_transaction(generation_fd, intent)
            return intent
        return None

    def delete_versioned(
        self,
        uri: str,
        *,
        org_id: str,
        project_id: str,
        object_version: str,
    ) -> ManagedArtifactDeleteResult:
        """Delete only one descriptor-verified local generation."""

        reference = parse_managed_uri(uri)
        expected = _parse_local_object_version(object_version)
        relative = self._relative_object(reference, org_id, project_id)
        with self._object_lock(reference, org_id, project_id):
            try:
                with self._open_directory(relative.parts[:-1], create=True) as directory_fd:
                    with self._open_generation_directory(relative, create=True) as generation_fd:
                        completed = list(
                            self._recover_delete_transactions(
                                generation_fd,
                                directory_fd,
                                relative,
                                reference,
                            )
                        )
                        current = self._current_read_from_directory(
                            directory_fd,
                            relative,
                            reference.uri,
                        )
                        if current is None:
                            retired = self._retire_absent_generation(
                                generation_fd,
                                directory_fd,
                                relative,
                                reference,
                                expected,
                            )
                            if retired is not None:
                                completed.append(retired)
                            status = (
                                ManagedArtifactDeleteStatus.DELETED
                                if any(intent.matches_version(expected) for intent in completed)
                                else ManagedArtifactDeleteStatus.ABSENT
                            )
                            return ManagedArtifactDeleteResult(status)
                        if current.metadata.object_version != object_version:
                            return ManagedArtifactDeleteResult(
                                ManagedArtifactDeleteStatus.VERSION_MISMATCH,
                                current_metadata=current.metadata,
                            )
                        assert current.metadata.sha256 is not None
                        intent = _LocalDeleteIntent(
                            layout=2 if expected.generation_digest is not None else 1,
                            digest=current.metadata.sha256,
                            device=expected.device,
                            inode=expected.inode,
                            ctime_ns=expected.ctime_ns,
                        )
                        if expected.generation_digest is not None and intent.digest != expected.generation_digest:
                            raise ManagedArtifactDeleteInconclusiveError(
                                "local generation digest does not match its version"
                            )
                        with self._open_delete_transaction(
                            generation_fd,
                            intent,
                            create=True,
                        ) as transaction_fd:
                            if self._transaction_marker_exists(transaction_fd, "complete"):
                                raise ManagedArtifactDeleteInconclusiveError(
                                    "local deletion transaction identity was reused"
                                )
                            try:
                                os.stat("detached", dir_fd=transaction_fd, follow_symlinks=False)
                            except FileNotFoundError:
                                pass
                            else:
                                raise ManagedArtifactDeleteInconclusiveError(
                                    "local deletion transaction already has a detached entry"
                                )
                            try:
                                os.rename(
                                    relative.name,
                                    "detached",
                                    src_dir_fd=directory_fd,
                                    dst_dir_fd=transaction_fd,
                                )
                            except FileNotFoundError:
                                completed.extend(
                                    self._recover_delete_transactions(
                                        generation_fd,
                                        directory_fd,
                                        relative,
                                        reference,
                                    )
                                )
                            else:
                                os.fsync(directory_fd)
                                os.fsync(transaction_fd)
                                if self._recover_delete_transaction(
                                    generation_fd,
                                    transaction_fd,
                                    directory_fd,
                                    relative,
                                    reference,
                                    intent,
                                ):
                                    completed.append(intent)
                            transaction_complete = self._transaction_marker_exists(transaction_fd, "complete")
                        if transaction_complete:
                            self._archive_delete_transaction(generation_fd, intent)

                        replacement = self._current_read_from_directory(
                            directory_fd,
                            relative,
                            reference.uri,
                        )
                        if replacement is not None:
                            return ManagedArtifactDeleteResult(
                                ManagedArtifactDeleteStatus.VERSION_MISMATCH,
                                current_metadata=replacement.metadata,
                            )
                        return ManagedArtifactDeleteResult(ManagedArtifactDeleteStatus.DELETED)
            except ManagedArtifactDeleteInconclusiveError:
                raise
            except (InvalidManagedArtifactURI, ManagedArtifactConflictError, ManagedArtifactVersionMismatchError) as exc:
                raise ManagedArtifactDeleteInconclusiveError("local managed artifact deletion was inconclusive") from exc
            except OSError as exc:
                raise ManagedArtifactDeleteInconclusiveError("local managed artifact deletion was inconclusive") from exc

    def delete(self, uri: str, *, org_id: str, project_id: str) -> bool:
        """Compatibility wrapper that conditionally removes the observed object."""

        current = self.stat(uri, org_id=org_id, project_id=project_id)
        if current is None:
            return False
        deleted = self.delete_versioned(
            uri,
            org_id=org_id,
            project_id=project_id,
            object_version=current.object_version,
        )
        return deleted.status is ManagedArtifactDeleteStatus.DELETED


class GCSManagedArtifactStore:
    """Immutable GCS storage with a bounded, live-only JSON API inventory path."""

    def __init__(
        self,
        root_uri: str,
        *,
        runner: CommandRunner = _default_gcloud_runner,
        inventory_transport: _GCSInventoryTransport = _httpx_gcs_inventory_transport,
        access_token_provider: tp.Callable[[], str] = _ambient_gcs_inventory_access_token,
        inventory_timeout_seconds: float = _GCS_INVENTORY_TIMEOUT_SECONDS,
        upload_transport: _GCSUploadTransport | None = None,
        download_transport: _GCSDownloadTransport | None = None,
    ):
        if not isinstance(root_uri, str) or any(
            ord(character) < 0x20 or ord(character) == 0x7F for character in root_uri
        ):
            raise ValueError("GCS managed artifact root must be a canonical URI without control characters")
        parsed = urlsplit(root_uri.rstrip("/"))
        if parsed.scheme != "gs" or not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("GCS managed artifact storage requires a gs://bucket[/prefix] URI")
        if (
            "@" in parsed.netloc
            or ":" in parsed.netloc
            or "%" in parsed.path
            or any(character in f"{parsed.netloc}{parsed.path}" for character in "*?[]\r\n\0")
        ):
            raise ValueError("GCS managed artifact root must use a canonical, non-wildcard bucket and prefix")
        components = tuple(component for component in parsed.path.split("/") if component)
        if any(component in {".", ".."} for component in components):
            raise ValueError("GCS managed artifact root cannot contain relative path components")
        suffix = f"/{'/'.join(components)}" if components else ""
        self.root_uri = f"gs://{parsed.netloc}{suffix}"
        self._bucket = parsed.netloc
        self._root_object_prefix = "/".join(components)
        if not callable(runner) or not callable(inventory_transport) or not callable(access_token_provider):
            raise TypeError("GCS command runner, inventory transport, and access token provider must be callable")
        self._runner = runner
        try:
            runner_signature = inspect.signature(runner)
        except (TypeError, ValueError):
            self._runner_accepts_timeout = runner is subprocess.run
        else:
            self._runner_accepts_timeout = "timeout" in runner_signature.parameters or any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in runner_signature.parameters.values()
            )
        if (
            isinstance(inventory_timeout_seconds, bool)
            or not isinstance(inventory_timeout_seconds, (int, float))
            or not 0 < inventory_timeout_seconds <= 120
        ):
            raise ValueError("GCS inventory timeout must be between 0 and 120 seconds")
        self._inventory_transport = inventory_transport
        self._access_token_provider = access_token_provider
        self._inventory_timeout_seconds = float(inventory_timeout_seconds)
        # The ambient provider spawns `gcloud auth print-access-token` (~0.3 s),
        # and one immutable write stats its object about four times -- so every
        # archived rank paid over a second just for credentials, eight at once.
        # A token is reused for a bounded window and dropped on the first 401.
        self._token_lock = threading.Lock()
        self._cached_token: tuple[str, float] | None = None
        # Writes go through the JSON API whenever this store talks to real GCS.
        # A caller that injected its own gcloud runner without an upload
        # transport keeps the `gcloud storage cp` path it was written against.
        if upload_transport is None and runner is _default_gcloud_runner:
            upload_transport = _httpx_gcs_upload_transport
        if upload_transport is not None and not callable(upload_transport):
            raise TypeError("GCS upload transport must be callable")
        self._upload_transport = upload_transport
        # Same rule for exact reads: one generation-pinned JSON API GET instead
        # of a `gcloud storage cat` process between two stats.
        if download_transport is None and runner is _default_gcloud_runner:
            download_transport = _httpx_gcs_download_transport
        if download_transport is not None and not callable(download_transport):
            raise TypeError("GCS download transport must be callable")
        self._download_transport = download_transport

    def _run_gcloud(self, args: tp.Sequence[str]) -> subprocess.CompletedProcess[bytes]:
        """Run one transfer command with a production deadline.

        The timeout keyword is omitted only for legacy injected runners whose
        inspected signature predates the additive runner contract. The default
        runner path and all conforming runners always receive it.
        """

        arguments: dict[str, tp.Any] = {"check": False, "capture_output": True}
        if self._runner_accepts_timeout:
            arguments["timeout"] = _GCS_TRANSFER_TIMEOUT_SECONDS
        return self._runner(args, **arguments)

    def _object_uri(self, reference: ManagedArtifactRef, org_id: str, project_id: str) -> str:
        relative = _relative_object_path(reference, org_id, project_id).as_posix()
        return f"{self.root_uri}/{relative}"

    def _send_download(self, url: str, token: str) -> tuple[int, tp.Mapping[str, str], bytes]:
        download = tp.cast("_GCSDownloadTransport", self._download_transport)
        response = download(
            url,
            headers={"Accept-Encoding": "identity", "Authorization": f"Bearer {token}"},
            timeout_seconds=_GCS_TRANSFER_TIMEOUT_SECONDS,
        )
        if not isinstance(response, tuple) or len(response) != 3:
            raise ManagedArtifactInventoryError("GCS download transport returned an invalid response")
        status_code, headers, body = response
        if (
            isinstance(status_code, bool)
            or not isinstance(status_code, int)
            or not isinstance(headers, tp.Mapping)
            or not isinstance(body, bytes)
        ):
            raise ManagedArtifactInventoryError("GCS download transport returned an invalid response")
        return status_code, headers, body

    def _download_exact(
        self,
        reference: ManagedArtifactRef,
        *,
        org_id: str,
        project_id: str,
        object_version: str,
    ) -> ManagedArtifactRead | None:
        """Read exactly ``object_version`` in one request, or return ``None``.

        The stat / ``gcloud storage cat`` / stat sandwich proved that the bytes
        came from one current generation. ``ifGenerationMatch`` on the media GET
        makes GCS itself serve that generation or refuse, and the stored MD5 and
        length it reports are checked against the bytes received. Anything that
        does not prove the read -- a refusal, a missing or disagreeing checksum,
        a transcoded object, a transport error -- returns ``None``, and the
        caller performs the sandwich read exactly as before, which is also what
        produces the precise mismatch or absence error.
        """

        if self._download_transport is None:
            return None
        try:
            generation = str(_parse_gcs_object_version(object_version))
            object_name = self._object_name(reference, org_id, project_id)
            url = self._json_api_url(
                f"b/{quote(self._bucket, safe='')}/o/{quote(object_name, safe='')}",
                {"alt": "media", "ifGenerationMatch": generation},
            )
            token = self._access_token()
            status_code, headers, body = self._send_download(url, token)
            if status_code == 401:
                self._forget_access_token(token)
                status_code, headers, body = self._send_download(url, self._access_token())
        except Exception:
            return None
        lowered = {str(key).lower(): str(value) for key, value in headers.items()}
        stored_length = lowered.get("x-goog-stored-content-length", "")
        reported_md5 = _gcs_md5_header(lowered.get("x-goog-hash"))
        if (
            status_code != 200
            or lowered.get("x-goog-generation") != generation
            or lowered.get("x-goog-stored-content-encoding", "identity") != "identity"
            or not stored_length.isascii()
            or not stored_length.isdigit()
            or int(stored_length) != len(body)
            or reported_md5 is None
        ):
            return None
        expected_md5 = base64.b64encode(hashlib.md5(body, usedforsecurity=False).digest()).decode("ascii")
        if not hmac.compare_digest(reported_md5, expected_md5):
            return None
        metadata = ManagedArtifactObjectMetadata(
            uri=reference.uri,
            size_bytes=len(body),
            object_version=f"gcs:{generation}",
            sha256=hashlib.sha256(body).hexdigest(),
        )
        try:
            payload = _validate_loaded_payload(reference, body)
        except ValueError as exc:
            raise ManagedArtifactConflictError(
                "managed artifact bytes do not match their content-addressed URI",
                current_metadata=metadata,
            ) from exc
        return ManagedArtifactRead(payload=payload, metadata=metadata)

    def _get_object(self, object_uri: str) -> bytes:
        try:
            completed = self._run_gcloud(["gcloud", "storage", "cat", object_uri])
        except Exception:
            raise ManagedArtifactOperationInconclusiveError("GCS managed artifact read failed") from None
        if completed.returncode != 0 or not isinstance(completed.stdout, bytes):
            raise ManagedArtifactOperationInconclusiveError("GCS managed artifact read failed")
        return completed.stdout

    def get_versioned(
        self,
        uri: str,
        *,
        org_id: str,
        project_id: str,
        object_version: str | None = None,
    ) -> ManagedArtifactRead:
        """Read one exact current GCS generation with a metadata sandwich."""

        reference = parse_managed_uri(uri)
        if object_version is not None:
            _parse_gcs_object_version(object_version)
            exact = self._download_exact(reference, org_id=org_id, project_id=project_id, object_version=object_version)
            if exact is not None:
                return exact
        before = self.stat(reference.uri, org_id=org_id, project_id=project_id)
        if before is None:
            raise FileNotFoundError("managed artifact object is absent")
        if object_version is not None and before.object_version != object_version:
            raise ManagedArtifactVersionMismatchError(
                "managed artifact object version does not match",
                current_metadata=before,
            )
        try:
            payload = self._get_object(self._object_uri(reference, org_id, project_id))
        except ManagedArtifactOperationInconclusiveError:
            try:
                current = self.stat(reference.uri, org_id=org_id, project_id=project_id)
            except ManagedArtifactInventoryError:
                raise ManagedArtifactOperationInconclusiveError(
                    "GCS managed artifact read verification failed"
                ) from None
            if current is None:
                raise FileNotFoundError("managed artifact object is absent") from None
            if current.object_version != before.object_version:
                raise ManagedArtifactVersionMismatchError(
                    "managed artifact object changed while it was read",
                    current_metadata=current,
                ) from None
            raise
        try:
            after = self.stat(reference.uri, org_id=org_id, project_id=project_id)
        except ManagedArtifactInventoryError:
            raise ManagedArtifactOperationInconclusiveError("GCS managed artifact read verification failed") from None
        if after is None:
            raise FileNotFoundError("managed artifact object is absent")
        if after.object_version != before.object_version:
            raise ManagedArtifactVersionMismatchError(
                "managed artifact object changed while it was read",
                current_metadata=after,
            )
        if after != before:
            raise ManagedArtifactOperationInconclusiveError(
                "GCS returned inconsistent metadata for one object generation"
            )
        observed_metadata = ManagedArtifactObjectMetadata(
            uri=after.uri,
            size_bytes=after.size_bytes,
            object_version=after.object_version,
            sha256=hashlib.sha256(payload).hexdigest(),
        )
        try:
            payload = _validate_loaded_payload(reference, payload)
        except ValueError as exc:
            raise ManagedArtifactConflictError(
                "managed artifact bytes do not match their content-addressed URI",
                current_metadata=observed_metadata,
            ) from exc
        return ManagedArtifactRead(payload=payload, metadata=observed_metadata)

    def _matching_current_write(
        self,
        uri: str,
        payload: bytes,
        *,
        org_id: str,
        project_id: str,
        created: bool | None,
    ) -> ManagedArtifactWrite | None:
        """Return an identical stable current generation, retrying replacement races."""

        for _attempt in range(3):
            current = self.stat(uri, org_id=org_id, project_id=project_id)
            if current is None:
                return None
            try:
                observed = self.get_versioned(
                    uri,
                    org_id=org_id,
                    project_id=project_id,
                    object_version=current.object_version,
                )
            except (FileNotFoundError, ManagedArtifactVersionMismatchError):
                continue
            if not _payloads_match(observed.payload, payload):
                raise ManagedArtifactConflictError(
                    "managed object already contains different bytes",
                    current_metadata=observed.metadata,
                )
            return ManagedArtifactWrite(metadata=observed.metadata, created=created)
        raise ManagedArtifactWriteInconclusiveError("GCS managed artifact write verification was inconclusive")

    def put_versioned(
        self,
        uri: str,
        payload: bytes,
        *,
        org_id: str,
        project_id: str,
    ) -> ManagedArtifactWrite:
        """Conditionally create a GCS object and return its current generation."""

        reference = parse_managed_uri(uri)
        payload = _validate_payload(reference, payload)
        existing = self._matching_current_write(
            reference.uri,
            payload,
            org_id=org_id,
            project_id=project_id,
            created=False,
        )
        if existing is not None:
            return existing

        authored: ManagedArtifactWrite | None = None
        if self._upload_transport is not None:
            command_succeeded, authored = self._upload_object(reference, payload, org_id=org_id, project_id=project_id)
        else:
            command_succeeded = self._copy_object(reference, payload, org_id=org_id, project_id=project_id)
        if authored is not None:
            return authored

        try:
            # Anything the upload could not prove by checksum -- a failed or
            # timed-out create, a precondition race, an absent or disagreeing
            # checksum -- takes the exact byte read, exactly as before.
            observed = self._matching_current_write(
                reference.uri,
                payload,
                org_id=org_id,
                project_id=project_id,
                created=True if command_succeeded else None,
            )
        except ManagedArtifactConflictError:
            raise
        except (ManagedArtifactInventoryError, ManagedArtifactOperationInconclusiveError, OSError, ValueError):
            raise ManagedArtifactWriteInconclusiveError("GCS managed artifact write was inconclusive") from None
        if observed is not None:
            return observed
        raise ManagedArtifactWriteInconclusiveError("GCS managed artifact write was inconclusive")

    def _copy_object(self, reference: ManagedArtifactRef, payload: bytes, *, org_id: str, project_id: str) -> bool:
        """Conditionally create the object with ``gcloud storage cp``; return whether it answered success."""

        object_uri = self._object_uri(reference, org_id, project_id)
        temporary_path: Path | None = None
        command_succeeded = False
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temporary:
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            command = [
                "gcloud",
                "storage",
                "cp",
                "--if-generation-match=0",
                str(temporary_path),
                object_uri,
            ]
            try:
                completed = self._run_gcloud(command)
                command_succeeded = completed.returncode == 0
            except Exception:
                command_succeeded = False
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass
        return command_succeeded

    def _send_upload(self, url: str, payload: bytes, token: str) -> tuple[int, bytes]:
        upload = tp.cast("_GCSUploadTransport", self._upload_transport)
        response = upload(
            url,
            headers={
                "Accept": "application/json",
                "Accept-Encoding": "identity",
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/octet-stream",
            },
            content=payload,
            timeout_seconds=_GCS_TRANSFER_TIMEOUT_SECONDS,
            maximum_response_bytes=_MAX_GCS_INVENTORY_RESPONSE_BYTES,
        )
        if not isinstance(response, tuple) or len(response) != 2:
            raise ManagedArtifactInventoryError("GCS upload transport returned an invalid response")
        status_code, body = response
        if isinstance(status_code, bool) or not isinstance(status_code, int) or not isinstance(body, bytes):
            raise ManagedArtifactInventoryError("GCS upload transport returned an invalid response")
        return status_code, body

    def _upload_object(
        self,
        reference: ManagedArtifactRef,
        payload: bytes,
        *,
        org_id: str,
        project_id: str,
    ) -> tuple[bool, ManagedArtifactWrite | None]:
        """Conditionally create the object through the JSON API in one request.

        Replaces a ``gcloud storage cp`` process (~2 s) and the full
        ``gcloud storage cat`` that verified it (~2 s), which together made
        every archived log rank cost seconds. Returns whether the create was
        answered as this store's own, and the write when GCS's MD5 of the
        object it created matches the payload exactly. A create this store
        authored under ``ifGenerationMatch=0`` whose checksum and size agree is
        proven without reading the bytes back; every other outcome -- an error,
        a precondition race, a missing or disagreeing checksum -- returns no
        write, and the caller performs the exact byte read it always has.
        """

        object_name = self._object_name(reference, org_id, project_id)
        url = self._json_api_url(
            f"b/{quote(self._bucket, safe='')}/o",
            {"uploadType": "media", "name": object_name, "ifGenerationMatch": "0", "fields": _GCS_UPLOAD_FIELDS},
            root=_GCS_UPLOAD_API_ROOT,
        )
        try:
            token = self._access_token()
            status_code, body = self._send_upload(url, payload, token)
            if status_code == 401:
                # Same rule as inventory: a reused token may have lapsed. The
                # create never happened, so retrying it once is safe.
                self._forget_access_token(token)
                status_code, body = self._send_upload(url, payload, self._access_token())
        except Exception:
            return False, None
        if status_code != 200:
            return False, None
        try:
            decoded = self._decode_gcs_json(body)
            reported_md5 = decoded.pop("md5Hash", None)
            _name, metadata = self._metadata_from_gcs_item(
                decoded,
                org_id=org_id,
                project_id=project_id,
                expected_name=object_name,
            )
        except ManagedArtifactInventoryError:
            return True, None
        expected_md5 = base64.b64encode(hashlib.md5(payload, usedforsecurity=False).digest()).decode("ascii")
        if (
            not isinstance(reported_md5, str)
            or not hmac.compare_digest(reported_md5, expected_md5)
            or metadata.size_bytes != len(payload)
        ):
            return True, None
        return True, ManagedArtifactWrite(
            metadata=ManagedArtifactObjectMetadata(
                uri=metadata.uri,
                size_bytes=metadata.size_bytes,
                object_version=metadata.object_version,
                sha256=hashlib.sha256(payload).hexdigest(),
            ),
            created=True,
        )

    def put(self, uri: str, payload: bytes, *, org_id: str, project_id: str) -> str:
        """Compatibility wrapper returning the canonical logical URI."""

        return self.put_versioned(uri, payload, org_id=org_id, project_id=project_id).metadata.uri

    def get(self, uri: str, *, org_id: str, project_id: str) -> bytes:
        """Compatibility wrapper returning current scoped object bytes."""

        return self.get_versioned(uri, org_id=org_id, project_id=project_id).payload

    def exists(self, uri: str, *, org_id: str, project_id: str) -> bool:
        """Return whether a scoped current GCS object exists conclusively."""

        parse_managed_uri(uri)
        return self.stat(uri, org_id=org_id, project_id=project_id) is not None

    def _scope_object_name_prefix(self, org_id: str, project_id: str) -> str:
        suffix = f"scopes/{storage_scope(org_id, project_id)}/"
        prefix = f"{self._root_object_prefix}/{suffix}" if self._root_object_prefix else suffix
        self._validate_gcs_object_name(prefix, description="scope prefix")
        return prefix

    @staticmethod
    def _validate_gcs_object_name(name: str, *, description: str) -> str:
        if not isinstance(name, str) or not name:
            raise ManagedArtifactInventoryError(f"GCS inventory returned an invalid {description}")
        try:
            encoded = name.encode("utf-8")
        except UnicodeEncodeError:
            raise ManagedArtifactInventoryError(f"GCS inventory returned an invalid {description}") from None
        if len(encoded) > _MAX_GCS_OBJECT_NAME_BYTES:
            raise ManagedArtifactInventoryError(f"GCS inventory returned an oversized {description}")
        return name

    def _object_name(self, reference: ManagedArtifactRef, org_id: str, project_id: str) -> str:
        parsed = urlsplit(self._object_uri(reference, org_id, project_id))
        if parsed.scheme != "gs" or parsed.netloc != self._bucket:
            raise ManagedArtifactInventoryError("managed artifact inventory object escaped its configured GCS bucket")
        return self._validate_gcs_object_name(parsed.path.lstrip("/"), description="object name")

    def _reference_from_gcs_name(self, name: str, org_id: str, project_id: str) -> ManagedArtifactRef:
        name = self._validate_gcs_object_name(name, description="object name")
        prefix = self._scope_object_name_prefix(org_id, project_id)
        if not name.startswith(prefix):
            raise ManagedArtifactInventoryError("GCS inventory returned an object outside the requested scope prefix")
        relative = name[len(prefix) :]
        components = relative.split("/")
        if len(components) != 2:
            raise ManagedArtifactInventoryError("GCS inventory returned a non-canonical managed artifact path")
        try:
            kind = ManagedArtifactKind(components[0])
        except ValueError:
            raise ManagedArtifactInventoryError("GCS inventory returned an unknown managed artifact kind") from None
        filename = components[1]
        extension = _OBJECT_EXTENSIONS[kind]
        if not filename.endswith(extension):
            raise ManagedArtifactInventoryError("GCS inventory returned an object with an unexpected extension")
        object_id = filename[: -len(extension)]
        try:
            reference = ManagedArtifactRef(kind, _validate_object_id(kind, object_id))
        except (InvalidManagedArtifactURI, ValueError):
            raise ManagedArtifactInventoryError("GCS inventory returned a non-canonical managed artifact ID") from None
        if self._object_name(reference, org_id, project_id) != name:
            raise ManagedArtifactInventoryError("GCS inventory returned a non-canonical managed artifact path")
        return reference

    @staticmethod
    def _json_api_url(path: str, parameters: tp.Mapping[str, str], *, root: str = _GCS_JSON_API_ROOT) -> str:
        query = urlencode(tuple(parameters.items()), doseq=False, safe="", quote_via=quote)
        url = f"{root}/{path}?{query}"
        try:
            url_size = len(url.encode("ascii"))
        except UnicodeEncodeError:
            raise ManagedArtifactInventoryError("GCS inventory request URL was invalid") from None
        if url_size > _MAX_GCS_INVENTORY_URL_BYTES:
            raise ManagedArtifactInventoryError("GCS inventory request URL was too large")
        return url

    def _access_token(self) -> str:
        with self._token_lock:
            now = time.monotonic()
            if self._cached_token is not None and self._cached_token[1] > now:
                return self._cached_token[0]
            token = self._fresh_access_token()
            self._cached_token = (token, now + _GCS_ACCESS_TOKEN_REUSE_SECONDS)
            return token

    def _forget_access_token(self, token: str) -> None:
        with self._token_lock:
            if self._cached_token is not None and self._cached_token[0] == token:
                self._cached_token = None

    def _fresh_access_token(self) -> str:
        try:
            token = self._access_token_provider()
        except Exception:
            raise ManagedArtifactInventoryError("ambient GCS credential lookup failed") from None
        if not isinstance(token, str):
            raise ManagedArtifactInventoryError("ambient GCS credential lookup returned an invalid access token")
        try:
            encoded = token.encode("ascii")
        except UnicodeEncodeError:
            raise ManagedArtifactInventoryError(
                "ambient GCS credential lookup returned an invalid access token"
            ) from None
        if (
            not token
            or token != token.strip()
            or len(encoded) > _MAX_GCS_ACCESS_TOKEN_BYTES
            or any(character < 0x21 or character > 0x7E for character in encoded)
        ):
            raise ManagedArtifactInventoryError("ambient GCS credential lookup returned an invalid access token")
        return token

    def _request_inventory_json(self, url: str, *, absent_on_not_found: bool = False) -> bytes | None:
        token = self._access_token()
        payload = self._request_inventory_json_once(url, token, absent_on_not_found=absent_on_not_found)
        if payload is _GCS_UNAUTHORIZED:
            # A reused token can expire or be revoked inside its window. Drop it
            # and ask once more with a fresh one; a second 401 is the answer.
            self._forget_access_token(token)
            payload = self._request_inventory_json_once(
                url, self._access_token(), absent_on_not_found=absent_on_not_found
            )
            if payload is _GCS_UNAUTHORIZED:
                raise ManagedArtifactInventoryError("GCS inventory request failed with HTTP status 401")
        return tp.cast("bytes | None", payload)

    def _request_inventory_json_once(self, url: str, token: str, *, absent_on_not_found: bool) -> object:
        try:
            response = self._inventory_transport(
                url,
                headers={
                    "Accept": "application/json",
                    "Accept-Encoding": "identity",
                    "Authorization": f"Bearer {token}",
                },
                timeout_seconds=self._inventory_timeout_seconds,
                maximum_response_bytes=_MAX_GCS_INVENTORY_RESPONSE_BYTES,
            )
        except _GCSInventoryResponseTooLarge:
            raise ManagedArtifactInventoryError("GCS inventory response exceeded the bounded body size") from None
        except (TimeoutError, httpx.TimeoutException):
            raise ManagedArtifactInventoryError("GCS inventory request timed out") from None
        except Exception:
            raise ManagedArtifactInventoryError("GCS inventory request failed") from None
        if not isinstance(response, tuple) or len(response) != 2:
            raise ManagedArtifactInventoryError("GCS inventory transport returned an invalid response")
        status_code, payload = response
        if isinstance(status_code, bool) or not isinstance(status_code, int) or not 100 <= status_code <= 599:
            raise ManagedArtifactInventoryError("GCS inventory transport returned an invalid status")
        if not isinstance(payload, bytes) or len(payload) > _MAX_GCS_INVENTORY_RESPONSE_BYTES:
            raise ManagedArtifactInventoryError("GCS inventory response exceeded the bounded body size")
        if absent_on_not_found and status_code == 404:
            return None
        if status_code == 401:
            return _GCS_UNAUTHORIZED
        if status_code != 200:
            # The status is useful and bounded. Provider bodies and URLs may
            # contain secrets, tenant identifiers, or attacker-controlled text.
            raise ManagedArtifactInventoryError(f"GCS inventory request failed with HTTP status {status_code}")
        return payload

    @staticmethod
    def _decode_gcs_json(payload: bytes) -> dict[str, tp.Any]:
        def unique_object(pairs: list[tuple[str, tp.Any]]) -> dict[str, tp.Any]:
            value: dict[str, tp.Any] = {}
            for key, item in pairs:
                if key in value:
                    raise ValueError("duplicate JSON key")
                value[key] = item
            return value

        def reject_constant(_value: str) -> tp.NoReturn:
            raise ValueError("non-finite JSON number")

        try:
            decoded = json.loads(
                payload.decode("utf-8"),
                object_pairs_hook=unique_object,
                parse_constant=reject_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            raise ManagedArtifactInventoryError("GCS inventory returned malformed JSON") from None
        if not isinstance(decoded, dict):
            raise ManagedArtifactInventoryError("GCS inventory returned an invalid JSON document")
        return decoded

    @staticmethod
    def _parse_gcs_uint64(value: tp.Any, *, field: str, minimum: int) -> tuple[int, str]:
        if (
            not isinstance(value, str)
            or not value
            or len(value) > _MAX_GCS_INTEGER_DIGITS
            or not value.isascii()
            or not value.isdigit()
            or (len(value) > 1 and value.startswith("0"))
        ):
            raise ManagedArtifactInventoryError(f"GCS inventory returned an invalid {field}")
        parsed = int(value)
        if parsed < minimum or parsed > (1 << 64) - 1:
            raise ManagedArtifactInventoryError(f"GCS inventory returned an invalid {field}")
        return parsed, value

    def _metadata_from_gcs_item(
        self,
        item: tp.Any,
        *,
        org_id: str,
        project_id: str,
        expected_name: str | None = None,
    ) -> tuple[str, ManagedArtifactObjectMetadata]:
        if not isinstance(item, dict) or set(item) != {"name", "size", "generation"}:
            raise ManagedArtifactInventoryError("GCS inventory returned an invalid object metadata document")
        name = self._validate_gcs_object_name(item["name"], description="object name")
        if expected_name is not None and name != expected_name:
            raise ManagedArtifactInventoryError("GCS stat returned metadata for a different object")
        size_bytes, _size_text = self._parse_gcs_uint64(item["size"], field="object size", minimum=0)
        _generation, generation_text = self._parse_gcs_uint64(item["generation"], field="object generation", minimum=1)
        reference = self._reference_from_gcs_name(name, org_id, project_id)
        return (
            name,
            ManagedArtifactObjectMetadata(
                uri=reference.uri,
                size_bytes=size_bytes,
                object_version=f"gcs:{generation_text}",
            ),
        )

    def _parse_gcs_list_page(
        self,
        payload: bytes,
        *,
        org_id: str,
        project_id: str,
        maximum_items: int,
    ) -> tuple[list[tuple[str, ManagedArtifactObjectMetadata]], str | None]:
        decoded = self._decode_gcs_json(payload)
        if not set(decoded).issubset({"items", "nextPageToken"}):
            raise ManagedArtifactInventoryError("GCS inventory returned unexpected JSON fields")
        raw_items = decoded.get("items", [])
        if not isinstance(raw_items, list) or len(raw_items) > maximum_items:
            raise ManagedArtifactInventoryError("GCS inventory returned an invalid or unbounded object page")
        parsed: list[tuple[str, ManagedArtifactObjectMetadata]] = []
        names: set[str] = set()
        previous_name: str | None = None
        for item in raw_items:
            name, metadata = self._metadata_from_gcs_item(item, org_id=org_id, project_id=project_id)
            if name in names:
                raise ManagedArtifactInventoryError("GCS inventory returned duplicate object names")
            if previous_name is not None and name < previous_name:
                raise ManagedArtifactInventoryError("GCS inventory returned objects out of lexical order")
            names.add(name)
            previous_name = name
            parsed.append((name, metadata))

        next_page_token: str | None = None
        if "nextPageToken" in decoded:
            raw_token = decoded["nextPageToken"]
            if not isinstance(raw_token, str):
                raise ManagedArtifactInventoryError("GCS inventory returned an invalid page token")
            try:
                encoded_token = raw_token.encode("ascii")
            except UnicodeEncodeError:
                raise ManagedArtifactInventoryError("GCS inventory returned an invalid page token") from None
            if (
                not raw_token
                or raw_token != raw_token.strip()
                or len(encoded_token) > _MAX_GCS_PAGE_TOKEN_BYTES
                or any(character < 0x21 or character > 0x7E for character in encoded_token)
            ):
                raise ManagedArtifactInventoryError("GCS inventory returned an invalid page token")
            next_page_token = raw_token
        return parsed, next_page_token

    def list_scope(
        self,
        *,
        org_id: str,
        project_id: str,
        cursor: str | None = None,
        limit: int = MAX_MANAGED_ARTIFACT_INVENTORY_PAGE_SIZE,
    ) -> ManagedArtifactInventoryPage:
        """Return the smallest logical URIs after the public cursor, or fail closed.

        Appending a physical suffix changes prefix/punctuation ordering, so a
        filename-ordered provider page cannot be merged with the logical index.
        Scan the exact live scope with a fixed request budget and retain only
        ``limit+1`` logical candidates. No partial page escapes an exhausted or
        nonprogressing scan. Physical names and stored generations are unchanged.
        """

        cursor = _validate_inventory_page_request(cursor, limit)
        maximum = MAX_MANAGED_ARTIFACT_INVENTORY_PAGE_SIZE
        parameters = {
            "prefix": self._scope_object_name_prefix(org_id, project_id),
            "maxResults": str(maximum),
            "versions": "false",
            "softDeleted": "false",
            "projection": "noAcl",
            "fields": _GCS_INVENTORY_FIELDS,
        }
        bucket = quote(self._bucket, safe="")
        candidates: list[ManagedArtifactObjectMetadata] = []
        previous_name: str | None = None
        seen_tokens: set[str] = set()
        for _page in range(_MAX_GCS_INVENTORY_SCAN_PAGES):
            url = self._json_api_url(f"b/{bucket}/o", parameters)
            payload = self._request_inventory_json(url)
            assert payload is not None
            parsed, next_page_token = self._parse_gcs_list_page(
                payload,
                org_id=org_id,
                project_id=project_id,
                maximum_items=maximum,
            )
            if parsed and previous_name is not None and parsed[0][0] <= previous_name:
                raise ManagedArtifactInventoryError("GCS inventory page did not advance monotonically")
            if parsed:
                previous_name = parsed[-1][0]
            candidates = heapq.nsmallest(
                limit + 1,
                (
                    *candidates,
                    *(metadata for _name, metadata in parsed if cursor is None or metadata.uri > cursor),
                ),
                key=lambda metadata: metadata.uri,
            )
            if next_page_token is None:
                selected = tuple(candidates[:limit])
                return ManagedArtifactInventoryPage(
                    objects=selected,
                    next_cursor=selected[-1].uri if len(candidates) > limit else None,
                )
            if not parsed or next_page_token in seen_tokens:
                raise ManagedArtifactInventoryError("GCS inventory page token did not advance")
            seen_tokens.add(next_page_token)
            parameters["pageToken"] = next_page_token
        raise ManagedArtifactInventoryError("GCS inventory exhausted its bounded scan budget")

    def stat(
        self,
        uri: str,
        *,
        org_id: str,
        project_id: str,
    ) -> ManagedArtifactObjectMetadata | None:
        """GET metadata for exactly one current live object, never its versions."""

        reference = parse_managed_uri(uri)
        object_name = self._object_name(reference, org_id, project_id)
        bucket = quote(self._bucket, safe="")
        encoded_name = quote(object_name, safe="")
        url = self._json_api_url(
            f"b/{bucket}/o/{encoded_name}",
            {"projection": "noAcl", "softDeleted": "false", "fields": _GCS_STAT_FIELDS},
        )
        payload = self._request_inventory_json(url, absent_on_not_found=True)
        if payload is None:
            return None
        decoded = self._decode_gcs_json(payload)
        _name, metadata = self._metadata_from_gcs_item(
            decoded,
            org_id=org_id,
            project_id=project_id,
            expected_name=object_name,
        )
        return metadata

    def delete_versioned(
        self,
        uri: str,
        *,
        org_id: str,
        project_id: str,
        object_version: str,
    ) -> ManagedArtifactDeleteResult:
        """Delete only the supplied current GCS generation."""

        reference = parse_managed_uri(uri)
        generation = _parse_gcs_object_version(object_version)
        command = [
            "gcloud",
            "storage",
            "rm",
            f"--if-generation-match={generation}",
            self._object_uri(reference, org_id, project_id),
        ]
        try:
            completed = self._run_gcloud(command)
            command_succeeded = completed.returncode == 0
        except Exception:
            command_succeeded = False
        if command_succeeded:
            return ManagedArtifactDeleteResult(ManagedArtifactDeleteStatus.DELETED)

        try:
            current = self.stat(reference.uri, org_id=org_id, project_id=project_id)
        except (ManagedArtifactInventoryError, ManagedArtifactOperationInconclusiveError):
            raise ManagedArtifactDeleteInconclusiveError("GCS managed artifact deletion was inconclusive") from None
        if current is None:
            return ManagedArtifactDeleteResult(ManagedArtifactDeleteStatus.ABSENT)
        if current.object_version != object_version:
            try:
                replacement = self.get_versioned(
                    reference.uri,
                    org_id=org_id,
                    project_id=project_id,
                    object_version=current.object_version,
                )
            except (
                FileNotFoundError,
                ManagedArtifactInventoryError,
                ManagedArtifactOperationInconclusiveError,
                ManagedArtifactVersionMismatchError,
                OSError,
                ValueError,
            ):
                raise ManagedArtifactDeleteInconclusiveError("GCS managed artifact deletion was inconclusive") from None
            return ManagedArtifactDeleteResult(
                ManagedArtifactDeleteStatus.VERSION_MISMATCH,
                current_metadata=replacement.metadata,
            )
        raise ManagedArtifactDeleteInconclusiveError("GCS managed artifact deletion was inconclusive")

    def delete(self, uri: str, *, org_id: str, project_id: str) -> bool:
        """Compatibility wrapper that conditionally removes the observed object."""

        current = self.stat(uri, org_id=org_id, project_id=project_id)
        if current is None:
            return False
        deleted = self.delete_versioned(
            uri,
            org_id=org_id,
            project_id=project_id,
            object_version=current.object_version,
        )
        return deleted.status is ManagedArtifactDeleteStatus.DELETED


def managed_artifact_store_from_uri(uri: str) -> ManagedArtifactVersionedStore:
    """Create the version-aware managed backend selected by a server-owned root URI."""

    if uri.startswith("gs://"):
        return GCSManagedArtifactStore(uri)
    parsed = urlsplit(uri)
    if parsed.scheme == "file":
        if parsed.netloc not in ("", "localhost") or parsed.query or parsed.fragment:
            raise ValueError("file managed artifact URIs must be local paths without query or fragment")
        return LocalManagedArtifactStore(url2pathname(parsed.path))
    if parsed.scheme and not (len(parsed.scheme) == 1 and uri[1] == ":"):
        raise ValueError("managed artifact storage requires a gs:// URI, a file:// URI, or a local path")
    return LocalManagedArtifactStore(uri)


def _b64encode(payload: bytes) -> str:
    return base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")


def _b64decode(segment: str) -> bytes:
    if not _TOKEN_SEGMENT.fullmatch(segment):
        raise InvalidArtifactToken("invalid artifact transfer token")
    padding = "=" * (-len(segment) % 4)
    try:
        decoded = base64.b64decode(segment + padding, altchars=b"-_", validate=True)
    except (ValueError, TypeError) as exc:
        raise InvalidArtifactToken("invalid artifact transfer token") from exc
    if not hmac.compare_digest(_b64encode(decoded), segment):
        raise InvalidArtifactToken("invalid artifact transfer token")
    return decoded


def _constant_time_text_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left.encode(), right.encode())


class ArtifactTokenSigner:
    """Issue and verify scoped HMAC capabilities for managed object actions.

    ``secret`` is bytes or a :class:`~eyvan.identity.rotation.CapabilityKeyring`.
    The keyring form is what makes rotation survivable: tokens mint under the
    signing material, while verification walks every material still inside its
    retirement window, so outstanding capabilities outlive the swap.
    """

    def __init__(
        self,
        secret: bytes | CapabilityKeyring,
        *,
        default_ttl_seconds: int = 300,
        max_ttl_seconds: int = 3600,
        clock_skew_seconds: int = 0,
        clock: tp.Callable[[], float] = time.time,
    ):
        for value, name, allow_zero in (
            (default_ttl_seconds, "default_ttl_seconds", False),
            (max_ttl_seconds, "max_ttl_seconds", False),
            (clock_skew_seconds, "clock_skew_seconds", True),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if allow_zero else 1):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be a {qualifier} integer")
        if default_ttl_seconds > max_ttl_seconds:
            raise ValueError("default_ttl_seconds cannot exceed max_ttl_seconds")
        if isinstance(secret, bytes):
            if len(secret) < 32:
                raise ValueError("artifact token secret must contain at least 32 bytes")
            self._keyring: CapabilityKeyring = StaticCredential(secret)
        else:
            self._keyring = secret
        self._default_ttl_seconds = default_ttl_seconds
        self._max_ttl_seconds = max_ttl_seconds
        self._clock_skew_seconds = clock_skew_seconds
        self._clock = clock

    def _verified_payload(self, token: str) -> dict[str, object]:
        """Verify a token against every material the keyring still trusts.

        Newest first, so the common case costs one HMAC. Which generation a
        token failed against is never reported: that would tell a prober when
        a rotation happened.
        """

        # The shared verifier decodes segments without validating the base64
        # alphabet, so junk interleaved with a signature would otherwise be
        # discarded and the remainder verified. Tokens are canonical unpadded
        # base64url end to end; anything else is not one of ours.
        parts = token.split(".") if isinstance(token, str) else []
        if len(parts) != 3 or parts[0] != _TOKEN_VERSION:
            raise InvalidCapabilityToken("invalid artifact token")
        try:
            _b64decode(parts[1])
            _b64decode(parts[2])
        except InvalidArtifactToken as exc:
            raise InvalidCapabilityToken("invalid artifact token") from exc
        materials = self._keyring.verification_materials()
        if not materials:
            raise InvalidCapabilityToken("artifact token signer has no verifiable signing material")
        for material in materials:
            try:
                return verify_signature(material, _TOKEN_VERSION, token)
            except InvalidCapabilityToken:
                continue
        raise InvalidCapabilityToken("invalid artifact token")

    @staticmethod
    def _action(value: ArtifactAction | str) -> ArtifactAction:
        try:
            return ArtifactAction(value)
        except ValueError as exc:
            raise ValueError("artifact action must be read or write") from exc

    @staticmethod
    def _workload(value: str | None) -> str | None:
        return None if value is None else _validate_scope_value(value, "workload_id", max_length=256)

    def issue(
        self,
        action: ArtifactAction | str,
        uri: str,
        *,
        org_id: str,
        project_id: str,
        workload_id: str | None = None,
        ttl_seconds: int | None = None,
        expires: bool = True,
    ) -> str:
        """Issue a URL-safe token bound to one action, object, and tenant."""

        action = self._action(action)
        reference = parse_managed_uri(uri)
        org_id = _validate_scope_value(org_id, "org_id")
        project_id = _validate_scope_value(project_id, "project_id")
        workload_id = self._workload(workload_id)
        issued_at = int(self._clock())
        if expires:
            ttl_seconds = self._default_ttl_seconds if ttl_seconds is None else ttl_seconds
            if (
                isinstance(ttl_seconds, bool)
                or not isinstance(ttl_seconds, int)
                or not 1 <= ttl_seconds <= self._max_ttl_seconds
            ):
                raise ValueError(f"ttl_seconds must be an integer between 1 and {self._max_ttl_seconds}")
            expires_at: int | None = issued_at + ttl_seconds
        else:
            if ttl_seconds is not None:
                raise ValueError("non-expiring artifact tokens do not accept ttl_seconds")
            if action is not ArtifactAction.WRITE or workload_id is None:
                raise ValueError("non-expiring artifact tokens require a workload-bound write action")
            expires_at = None
        payload: dict[str, str | int | None] = {
            "action": action.value,
            "expires_at": expires_at,
            "issued_at": issued_at,
            "kind": reference.kind.value,
            "object_id": reference.object_id,
            "org_id": org_id,
            "project_id": project_id,
            "version": 1,
        }
        if workload_id is not None:
            payload["workload_id"] = workload_id
        return sign_claims(self._keyring.signing_material(), _TOKEN_VERSION, payload)

    def verify(
        self,
        token: str,
        action: ArtifactAction | str,
        uri: str,
        *,
        org_id: str,
        project_id: str,
        workload_id: str | None = None,
    ) -> ArtifactTokenClaims:
        """Verify signature, expiry, and the complete expected request scope."""

        action = self._action(action)
        reference = parse_managed_uri(uri)
        org_id = _validate_scope_value(org_id, "org_id")
        project_id = _validate_scope_value(project_id, "project_id")
        workload_id = self._workload(workload_id)
        try:
            payload = self._verified_payload(token)
        except InvalidCapabilityToken as exc:
            raise InvalidArtifactToken("invalid artifact transfer token") from exc
        required = {
            "action",
            "expires_at",
            "issued_at",
            "kind",
            "object_id",
            "org_id",
            "project_id",
            "version",
        }
        if not isinstance(payload, dict) or frozenset(payload) not in {
            frozenset(required),
            frozenset((*required, "workload_id")),
        }:
            raise InvalidArtifactToken("invalid artifact transfer token")
        try:
            claims_action = ArtifactAction(payload["action"])
            claims_kind = ManagedArtifactKind(payload["kind"])
            claims_object_id = _validate_object_id(claims_kind, payload["object_id"])
            claims_org = _validate_scope_value(payload["org_id"], "org_id")
            claims_project = _validate_scope_value(payload["project_id"], "project_id")
            claims_workload = self._workload(payload.get("workload_id"))
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidArtifactToken("invalid artifact transfer token") from exc
        issued_at = payload.get("issued_at")
        expires_at = payload.get("expires_at")
        if (
            isinstance(payload.get("version"), bool)
            or not isinstance(payload.get("version"), int)
            or payload.get("version") != 1
            or isinstance(issued_at, bool)
            or not isinstance(issued_at, int)
        ):
            raise InvalidArtifactToken("invalid artifact transfer token")
        if expires_at is None:
            if claims_action is not ArtifactAction.WRITE or claims_workload is None:
                raise InvalidArtifactToken("invalid artifact transfer token")
        elif (
            isinstance(expires_at, bool)
            or not isinstance(expires_at, int)
            or expires_at <= issued_at
            or expires_at - issued_at > self._max_ttl_seconds
        ):
            raise InvalidArtifactToken("invalid artifact transfer token")
        now = int(self._clock())
        if issued_at > now + self._clock_skew_seconds:
            raise InvalidArtifactToken("invalid artifact transfer token")
        if expires_at is not None and expires_at <= now - self._clock_skew_seconds:
            raise ExpiredArtifactToken("artifact transfer token has expired")
        scope_matches = (
            claims_action is action
            and claims_kind is reference.kind
            and _constant_time_text_equal(claims_object_id, reference.object_id)
            and _constant_time_text_equal(claims_org, org_id)
            and _constant_time_text_equal(claims_project, project_id)
            and (
                (claims_workload is None and workload_id is None)
                or (
                    claims_workload is not None
                    and workload_id is not None
                    and _constant_time_text_equal(claims_workload, workload_id)
                )
            )
        )
        if not scope_matches:
            raise InvalidArtifactToken("artifact transfer token scope does not match request")
        return ArtifactTokenClaims(
            action=claims_action,
            artifact=ManagedArtifactRef(claims_kind, claims_object_id),
            org_id=claims_org,
            project_id=claims_project,
            issued_at=issued_at,
            expires_at=expires_at,
            workload_id=claims_workload,
        )


__all__ = (
    "MAX_MANAGED_ARTIFACT_INVENTORY_PAGE_SIZE",
    "ArtifactAction",
    "ArtifactTokenClaims",
    "ArtifactTokenSigner",
    "CommandRunner",
    "ExpiredArtifactToken",
    "GCSManagedArtifactStore",
    "InvalidArtifactToken",
    "InvalidManagedArtifactURI",
    "LocalManagedArtifactStore",
    "ManagedArtifactConflictError",
    "ManagedArtifactDeleteInconclusiveError",
    "ManagedArtifactDeleteResult",
    "ManagedArtifactDeleteStatus",
    "ManagedArtifactInventoryError",
    "ManagedArtifactInventoryPage",
    "ManagedArtifactInventoryStore",
    "ManagedArtifactKind",
    "ManagedArtifactObjectMetadata",
    "ManagedArtifactOperationInconclusiveError",
    "ManagedArtifactRead",
    "ManagedArtifactRef",
    "ManagedArtifactStore",
    "ManagedArtifactVersionMismatchError",
    "ManagedArtifactVersionedStore",
    "ManagedArtifactWrite",
    "ManagedArtifactWriteInconclusiveError",
    "callable_uri",
    "is_managed_uri",
    "managed_artifact_store_from_uri",
    "parse_managed_uri",
    "result_uri",
    "storage_scope",
)
