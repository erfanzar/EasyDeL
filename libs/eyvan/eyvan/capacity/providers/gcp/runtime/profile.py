# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Owner-only local configuration for Eyvan's native GCP runtime."""

from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import stat
import tempfile
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from urllib.parse import urlsplit

from eyvan._internal.json_types import JsonObject, is_json_object
from eyvan._internal.workspace import create_private_directory

DEFAULT_GCP_NATIVE_RUNTIME_PROFILE_PATH = Path("~/.eyvan/gcp-native-runtime.json")
#: Where the profile-owned local agent keeps its fleet-state document when the
#: operator did not name one. A local plane and its local agent are the same
#: machine, so the document needs no bucket: keeping it on the local filesystem
#: means a GCS or CLI failure can never stop the agent from recording that
#: capacity it is already paying for exists. An
#: explicit ``EYVAN_FLEET_STATE_URI`` still wins.
DEFAULT_GCP_NATIVE_FLEET_STATE_PATH = Path("~/.eyvan/gcp-native-fleet-state.json")
GCP_NATIVE_RUNTIME_PROFILE_SCHEMA = "eyvan.gcp-native-runtime-profile"
GCP_NATIVE_RUNTIME_PROFILE_VERSION = 1

_GCP_PROJECT_ID = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
_GCS_BUCKET = re.compile(r"^[a-z0-9](?:[a-z0-9._-]{1,61}[a-z0-9])?$")
_WHEEL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WHEEL_FILENAME = re.compile(
    r"^(?P<distribution>[A-Za-z0-9]+(?:[A-Za-z0-9_.]*[A-Za-z0-9])?)-"
    r"(?P<version>[A-Za-z0-9]+(?:[A-Za-z0-9_.!+]*[A-Za-z0-9])?)"
    r"(?:-(?P<build>[0-9][A-Za-z0-9_]*))?-"
    r"(?P<python_tag>[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*)-"
    r"(?P<abi_tag>[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*)-"
    r"(?P<platform_tag>[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*)\.whl$"
)
_MAX_PROFILE_BYTES = 64 * 1024
_SECRET_BYTES = 32


class GcpNativeRuntimeProfileError(ValueError):
    """A native GCP runtime profile is invalid, unsafe, or cannot be persisted."""


@dataclass(frozen=True, slots=True)
class GcpNativeRuntimeProfile:
    """One owner's GCP runtime identity and private native-worker bootstrap secrets.

    This document is intentionally separate from public fleet and autoscale YAML.
    """

    project_id: str
    artifact_root_uri: str
    agent_shared_secret: str = field(repr=False)
    native_worker_root_secret: str = field(repr=False)
    native_worker_wheel_uri: str | None = None
    version: int = GCP_NATIVE_RUNTIME_PROFILE_VERSION

    def __post_init__(self) -> None:
        """Normalize and validate every persisted runtime profile value."""

        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, int)
            or self.version != GCP_NATIVE_RUNTIME_PROFILE_VERSION
        ):
            raise GcpNativeRuntimeProfileError(
                f"native GCP runtime profile version must be {GCP_NATIVE_RUNTIME_PROFILE_VERSION}"
            )
        object.__setattr__(self, "project_id", _validate_project_id(self.project_id))
        artifact_root_uri = _normalize_artifact_root_uri(self.artifact_root_uri)
        object.__setattr__(self, "artifact_root_uri", artifact_root_uri)
        object.__setattr__(
            self,
            "agent_shared_secret",
            _validate_secret(self.agent_shared_secret, "agent shared secret"),
        )
        object.__setattr__(
            self,
            "native_worker_root_secret",
            _validate_secret(self.native_worker_root_secret, "native worker root secret"),
        )
        object.__setattr__(
            self,
            "native_worker_wheel_uri",
            _normalize_native_worker_wheel_uri(self.native_worker_wheel_uri, artifact_root_uri=artifact_root_uri),
        )
        if self.agent_shared_secret == self.native_worker_root_secret:
            raise GcpNativeRuntimeProfileError("agent and native worker secrets must be distinct")

    def as_document(self) -> JsonObject:
        """Return the versioned JSON document persisted by the profile store."""

        return {
            "schema": GCP_NATIVE_RUNTIME_PROFILE_SCHEMA,
            "version": self.version,
            "project_id": self.project_id,
            "artifact_root_uri": self.artifact_root_uri,
            "agent_shared_secret": self.agent_shared_secret,
            "native_worker_root_secret": self.native_worker_root_secret,
            "native_worker_wheel_uri": self.native_worker_wheel_uri,
        }

    @property
    def fleet_state_uri(self) -> str:
        """Return the GCS state document URI derived from this profile's artifact root.

        This remains the right default for a *remote* head whose agent's disk is
        not durable. The profile-owned local agent defaults to
        ``DEFAULT_GCP_NATIVE_FLEET_STATE_PATH`` instead (see
        ``EyvanSettings.from_gcp_native_runtime_profile``).
        """

        return f"{self.artifact_root_uri}/control-plane/fleet-state.json"

    @property
    def native_worker_public_ingress_tag(self) -> str:
        """Return the stable opaque network tag for this profile's public workers.

        The HMAC keeps the durable worker-root secret out of both GCP resource
        names and logs while ensuring independent local profiles never contend
        for the same narrow ingress rule.
        """

        message = f"eyvan-native-worker-ingress:v1:{self.project_id}".encode()
        digest = hmac.new(self.native_worker_root_secret.encode(), message, sha256).hexdigest()[:24]
        return f"eyvan-nw-{digest}"

    @classmethod
    def from_document(cls, document: JsonObject) -> GcpNativeRuntimeProfile:
        """Validate one strict versioned JSON profile document."""

        expected_keys = {
            "schema",
            "version",
            "project_id",
            "artifact_root_uri",
            "agent_shared_secret",
            "native_worker_root_secret",
            "native_worker_wheel_uri",
        }
        unexpected = sorted(set(document).difference(expected_keys))
        if unexpected:
            raise GcpNativeRuntimeProfileError(
                f"native GCP runtime profile contains unsupported field(s): {', '.join(unexpected)}"
            )
        schema = _required_document_string(document, "schema")
        if schema != GCP_NATIVE_RUNTIME_PROFILE_SCHEMA:
            raise GcpNativeRuntimeProfileError(
                f"native GCP runtime profile schema must be {GCP_NATIVE_RUNTIME_PROFILE_SCHEMA!r}"
            )
        version = document.get("version")
        if isinstance(version, bool) or not isinstance(version, int):
            raise GcpNativeRuntimeProfileError("native GCP runtime profile version must be an integer")
        return cls(
            project_id=_required_document_string(document, "project_id"),
            artifact_root_uri=_required_document_string(document, "artifact_root_uri"),
            agent_shared_secret=_required_document_string(document, "agent_shared_secret"),
            native_worker_root_secret=_required_document_string(document, "native_worker_root_secret"),
            native_worker_wheel_uri=_optional_document_string(document, "native_worker_wheel_uri"),
            version=version,
        )


class GcpNativeRuntimeProfileStore:
    """Read and atomically write one owner-only native GCP runtime profile."""

    def __init__(self, path: Path | str = DEFAULT_GCP_NATIVE_RUNTIME_PROFILE_PATH):
        self.path = Path(path).expanduser()

    def load(self) -> GcpNativeRuntimeProfile:
        """Load a validated profile without creating any local state.

        Raises:
            FileNotFoundError: The owner has not initialized a native GCP runtime profile.
            GcpNativeRuntimeProfileError: The profile is unsafe, malformed, or unsupported.
        """

        self._validate_existing_parent_directory()
        descriptor = self._open_existing_profile()
        try:
            self._validate_profile_file(os.fstat(descriptor))
            payload = self._read_profile(descriptor)
        finally:
            os.close(descriptor)
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise GcpNativeRuntimeProfileError(
                f"could not decode native GCP runtime profile {self.path}: {exc}"
            ) from exc
        if not is_json_object(decoded):
            raise GcpNativeRuntimeProfileError(f"native GCP runtime profile {self.path} must be a JSON object")
        return GcpNativeRuntimeProfile.from_document(decoded)

    def save(self, profile: GcpNativeRuntimeProfile) -> Path:
        """Atomically replace the profile after validating its private storage location."""

        if not isinstance(profile, GcpNativeRuntimeProfile):
            raise GcpNativeRuntimeProfileError("native GCP runtime profile store requires a GcpNativeRuntimeProfile")
        self._prepare_parent_directory()
        self._validate_existing_profile_if_present()
        self._replace_atomically(_encode_profile(profile))
        return self.path

    def create(
        self,
        *,
        project_id: str,
        artifact_root_uri: str,
        native_worker_wheel_uri: str | None = None,
    ) -> GcpNativeRuntimeProfile:
        """Create a new profile with two independent cryptographically random secrets.

        The method never overwrites an existing profile, preventing an accidental
        credential rotation. Use :meth:`save` only for an explicit replacement.
        """

        agent_shared_secret = _new_secret()
        native_worker_root_secret = _new_secret()
        while native_worker_root_secret == agent_shared_secret:
            native_worker_root_secret = _new_secret()
        profile = GcpNativeRuntimeProfile(
            project_id=project_id,
            artifact_root_uri=artifact_root_uri,
            agent_shared_secret=agent_shared_secret,
            native_worker_root_secret=native_worker_root_secret,
            native_worker_wheel_uri=native_worker_wheel_uri,
        )
        self._prepare_parent_directory()
        self._create_atomically(_encode_profile(profile))
        return profile

    def load_or_create(
        self,
        *,
        project_id: str,
        artifact_root_uri: str,
        native_worker_wheel_uri: str | None = None,
    ) -> GcpNativeRuntimeProfile:
        """Load the stable profile, creating it once when it is absent."""

        normalized_project_id = _validate_project_id(project_id)
        normalized_artifact_root_uri = _normalize_artifact_root_uri(artifact_root_uri)
        normalized_native_worker_wheel_uri = _normalize_native_worker_wheel_uri(
            native_worker_wheel_uri,
            artifact_root_uri=normalized_artifact_root_uri,
        )
        try:
            existing = self.load()
        except FileNotFoundError:
            try:
                return self.create(
                    project_id=normalized_project_id,
                    artifact_root_uri=normalized_artifact_root_uri,
                    native_worker_wheel_uri=normalized_native_worker_wheel_uri,
                )
            except FileExistsError:
                existing = self.load()
        if (
            existing.project_id != normalized_project_id
            or existing.artifact_root_uri != normalized_artifact_root_uri
            or (
                normalized_native_worker_wheel_uri is not None
                and existing.native_worker_wheel_uri != normalized_native_worker_wheel_uri
            )
        ):
            raise GcpNativeRuntimeProfileError(
                "existing native GCP runtime profile does not match the requested project, artifact root, "
                "or worker wheel"
            )
        return existing

    def _prepare_parent_directory(self) -> None:
        """Create and secure the directory holding the profile secret document."""

        parent = self.path.parent
        try:
            create_private_directory(parent)
            descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except OSError as exc:
            raise GcpNativeRuntimeProfileError(
                f"could not prepare native GCP runtime profile directory {parent}: {exc}"
            ) from exc
        try:
            self._validate_profile_directory(os.fstat(descriptor), require_owner_only=False)
            os.fchmod(descriptor, 0o700)
            self._validate_profile_directory(os.fstat(descriptor), require_owner_only=True)
        except OSError as exc:
            raise GcpNativeRuntimeProfileError(
                f"could not secure native GCP runtime profile directory {parent}: {exc}"
            ) from exc
        finally:
            os.close(descriptor)

    def _validate_existing_parent_directory(self) -> None:
        """Reject a profile directory that another OS user can inspect or replace."""

        parent = self.path.parent
        try:
            descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise GcpNativeRuntimeProfileError(
                f"could not open native GCP runtime profile directory {parent}: {exc}"
            ) from exc
        try:
            self._validate_profile_directory(os.fstat(descriptor), require_owner_only=True)
        finally:
            os.close(descriptor)

    def _validate_profile_directory(self, metadata: os.stat_result, *, require_owner_only: bool) -> None:
        """Ensure the profile directory is a real directory owned by this OS user."""

        if not stat.S_ISDIR(metadata.st_mode):
            raise GcpNativeRuntimeProfileError(
                f"native GCP runtime profile directory {self.path.parent} must be a real directory"
            )
        if metadata.st_uid != os.getuid():
            raise GcpNativeRuntimeProfileError(
                f"native GCP runtime profile directory {self.path.parent} must be owned by this user"
            )
        if require_owner_only and stat.S_IMODE(metadata.st_mode) & 0o077:
            raise GcpNativeRuntimeProfileError(
                f"native GCP runtime profile directory {self.path.parent} must not be group- or world-accessible"
            )

    def _open_existing_profile(self) -> int:
        """Open the profile without following symlinks."""

        try:
            return os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise GcpNativeRuntimeProfileError(f"could not open native GCP runtime profile {self.path}: {exc}") from exc

    def _validate_existing_profile_if_present(self) -> None:
        """Validate an existing destination before a replacement write."""

        try:
            descriptor = self._open_existing_profile()
        except FileNotFoundError:
            return
        try:
            self._validate_profile_file(os.fstat(descriptor))
        finally:
            os.close(descriptor)

    def _validate_profile_file(self, metadata: os.stat_result) -> None:
        """Reject profile files that another OS user could observe or replace."""

        if not stat.S_ISREG(metadata.st_mode):
            raise GcpNativeRuntimeProfileError(f"native GCP runtime profile {self.path} must be a regular file")
        if metadata.st_uid != os.getuid():
            raise GcpNativeRuntimeProfileError(f"native GCP runtime profile {self.path} must be owned by this user")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise GcpNativeRuntimeProfileError(
                f"native GCP runtime profile {self.path} must not be group- or world-readable"
            )

    def _read_profile(self, descriptor: int) -> bytes:
        """Read one bounded profile document from an already validated descriptor."""

        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, 4096):
            total += len(chunk)
            if total > _MAX_PROFILE_BYTES:
                raise GcpNativeRuntimeProfileError(f"native GCP runtime profile {self.path} is too large")
            chunks.append(chunk)
        return b"".join(chunks)

    def _replace_atomically(self, payload: bytes) -> None:
        """Write a complete owner-only document before replacing the destination."""

        temporary_path = self._write_temporary(payload)
        try:
            self._validate_existing_profile_if_present()
            os.replace(temporary_path, self.path)
            temporary_path = None
            self._fsync_parent_directory()
        except OSError as exc:
            raise GcpNativeRuntimeProfileError(
                f"could not persist native GCP runtime profile {self.path}: {exc}"
            ) from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _create_atomically(self, payload: bytes) -> None:
        """Link a complete temporary document into place without replacing an existing profile."""

        temporary_path = self._write_temporary(payload)
        try:
            os.link(temporary_path, self.path)
        except FileExistsError as exc:
            self._validate_existing_profile_if_present()
            raise FileExistsError(f"native GCP runtime profile {self.path} already exists") from exc
        except OSError as exc:
            raise GcpNativeRuntimeProfileError(
                f"could not create native GCP runtime profile {self.path}: {exc}"
            ) from exc
        finally:
            temporary_path.unlink(missing_ok=True)
        self._fsync_parent_directory()

    def _write_temporary(self, payload: bytes) -> Path:
        """Persist an owner-only temporary profile document in the target directory."""

        descriptor = -1
        temporary_path: Path | None = None
        completed = False
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                dir=self.path.parent,
            )
            temporary_path = Path(temporary_name)
            os.fchmod(descriptor, 0o600)
            self._write_profile(descriptor, payload)
            os.close(descriptor)
            descriptor = -1
            completed = True
            return temporary_path
        except GcpNativeRuntimeProfileError:
            raise
        except OSError as exc:
            raise GcpNativeRuntimeProfileError(
                f"could not prepare native GCP runtime profile {self.path}: {exc}"
            ) from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if not completed and temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def _write_profile(self, descriptor: int, payload: bytes) -> None:
        """Write and sync a complete JSON document to one private file descriptor."""

        try:
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise GcpNativeRuntimeProfileError(f"could not persist native GCP runtime profile {self.path}")
                offset += written
            os.fsync(descriptor)
        except OSError as exc:
            raise GcpNativeRuntimeProfileError(
                f"could not persist native GCP runtime profile {self.path}: {exc}"
            ) from exc

    def _fsync_parent_directory(self) -> None:
        """Persist the directory entry after creating or replacing the profile."""

        try:
            descriptor = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except OSError as exc:
            raise GcpNativeRuntimeProfileError(
                f"could not open native GCP runtime profile directory {self.path.parent} for syncing: {exc}"
            ) from exc
        try:
            os.fsync(descriptor)
        except OSError as exc:
            raise GcpNativeRuntimeProfileError(
                f"could not sync native GCP runtime profile directory {self.path.parent}: {exc}"
            ) from exc
        finally:
            os.close(descriptor)


def default_artifact_root_uri(project_id: str) -> str:
    """Return the artifact root Eyvan manages for one project when none is named.

    The bucket name derives entirely from the project ID, which is already
    globally unique, so two operators of one project converge on the same root
    and two projects never collide. :func:`GcpRuntimeBootstrap.ensure_artifact_bucket`
    creates the bucket on first use -- an operator never hand-creates one.
    """

    return f"gs://eyvan-{_validate_project_id(project_id)}"


def _new_secret() -> str:
    """Generate one high-entropy ASCII secret suitable for private profile storage."""

    return secrets.token_urlsafe(_SECRET_BYTES)


def _validate_project_id(value: str) -> str:
    """Validate one canonical GCP project ID rather than a bucket URI or project number."""

    if not isinstance(value, str) or not _GCP_PROJECT_ID.fullmatch(value):
        raise GcpNativeRuntimeProfileError(
            "GCP project ID must be 6-30 lowercase letters, digits, or hyphens; it must start with a letter"
        )
    return value


def _validate_secret(value: str, name: str) -> str:
    """Require a long non-empty ASCII secret without exposing its value in errors."""

    if not isinstance(value, str) or value != value.strip() or not value.isascii() or "\x00" in value or len(value) < 32:
        raise GcpNativeRuntimeProfileError(
            f"native GCP runtime {name} must be an ASCII secret of at least 32 characters"
        )
    return value


def _normalize_artifact_root_uri(value: str) -> str:
    """Validate and canonicalize a GCS bucket or bucket-prefix URI."""

    bucket, path = _parse_gcs_uri(value, field_name="artifact root")
    return f"gs://{bucket}" if path is None else f"gs://{bucket}/{path}"


def validate_native_worker_wheel_filename(value: str) -> str:
    """Validate one concrete PEP 427 wheel filename without accepting path syntax.

    Native workers install this object directly with ``pip``. The content hash
    therefore belongs in a parent directory, not in place of the wheel's
    distribution/version/tag filename.
    """

    if (
        not isinstance(value, str)
        or value != value.strip()
        or not value.isascii()
        or "\x00" in value
        or "/" in value
        or "\\" in value
        or _WHEEL_FILENAME.fullmatch(value) is None
    ):
        raise GcpNativeRuntimeProfileError("native worker package must end with a concrete PEP 427 wheel filename")
    return value


def _normalize_native_worker_wheel_uri(value: str | None, *, artifact_root_uri: str) -> str | None:
    """Validate a staged wheel below the profile's immutable content-addressed GCS layout."""

    if value is None:
        return None
    bucket, path = _parse_gcs_uri(value, field_name="native worker package")
    root_bucket, root_path = _parse_gcs_uri(artifact_root_uri, field_name="artifact root")
    if path is None or bucket != root_bucket:
        raise GcpNativeRuntimeProfileError(
            "native worker package must be a content-addressed object below this profile's artifact root"
        )
    root_segments = () if root_path is None else tuple(root_path.split("/"))
    wheel_segments = tuple(path.split("/"))
    expected_prefix = (*root_segments, "runtime", "wheels")
    if len(wheel_segments) != len(expected_prefix) + 2 or wheel_segments[: len(expected_prefix)] != expected_prefix:
        raise GcpNativeRuntimeProfileError(
            "native worker package must use artifact-root/runtime/wheels/<sha256>/<wheel filename>.whl"
        )
    wheel_sha256, wheel_filename = wheel_segments[-2:]
    if _WHEEL_SHA256.fullmatch(wheel_sha256) is None:
        raise GcpNativeRuntimeProfileError("native worker package must use a lowercase SHA-256 directory")
    validate_native_worker_wheel_filename(wheel_filename)
    return f"gs://{bucket}/{path}"


def _parse_gcs_uri(value: str, *, field_name: str) -> tuple[str, str | None]:
    """Parse a conservative GCS URI without accepting wildcards or shell-unsafe forms."""

    if not isinstance(value, str) or value != value.strip() or not value or "\x00" in value:
        raise GcpNativeRuntimeProfileError(f"{field_name} must be a non-empty gs://bucket[/prefix] URI")
    parsed = urlsplit(value)
    if parsed.scheme != "gs" or not parsed.netloc or parsed.query or parsed.fragment:
        raise GcpNativeRuntimeProfileError(f"{field_name} must be a gs://bucket[/prefix] URI without query or fragment")
    bucket = parsed.netloc
    if not _GCS_BUCKET.fullmatch(bucket) or ".." in bucket or ".-" in bucket or "-." in bucket:
        raise GcpNativeRuntimeProfileError(f"{field_name} has an invalid GCS bucket name")
    raw_path = parsed.path.rstrip("/")
    if not raw_path:
        return bucket, None
    if not raw_path.startswith("/"):
        raise GcpNativeRuntimeProfileError(f"{field_name} must use a valid GCS object prefix")
    segments = raw_path[1:].split("/")
    if any(
        not segment or segment in {".", ".."} or any(character in segment for character in ("\\", "*", "?", "[", "]"))
        for segment in segments
    ):
        raise GcpNativeRuntimeProfileError(f"{field_name} must use a concrete GCS object prefix without wildcards")
    return bucket, "/".join(segments)


def _required_document_string(document: JsonObject, name: str) -> str:
    """Read one required string field from an untrusted JSON profile document."""

    value = document.get(name)
    if not isinstance(value, str):
        raise GcpNativeRuntimeProfileError(f"native GCP runtime profile field {name!r} must be a string")
    return value


def _optional_document_string(document: JsonObject, name: str) -> str | None:
    """Read one optional nullable string field from an untrusted JSON profile document."""

    value = document.get(name)
    if value is None:
        return None
    if not isinstance(value, str):
        raise GcpNativeRuntimeProfileError(f"native GCP runtime profile field {name!r} must be a string or null")
    return value


def _encode_profile(profile: GcpNativeRuntimeProfile) -> bytes:
    """Encode a validated profile as deterministic UTF-8 JSON."""

    return (json.dumps(profile.as_document(), indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")


__all__ = (
    "DEFAULT_GCP_NATIVE_FLEET_STATE_PATH",
    "DEFAULT_GCP_NATIVE_RUNTIME_PROFILE_PATH",
    "GCP_NATIVE_RUNTIME_PROFILE_SCHEMA",
    "GCP_NATIVE_RUNTIME_PROFILE_VERSION",
    "GcpNativeRuntimeProfile",
    "GcpNativeRuntimeProfileError",
    "GcpNativeRuntimeProfileStore",
    "default_artifact_root_uri",
    "validate_native_worker_wheel_filename",
)
