# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Persistent owner-only configuration store for ``~/.eyvan/storage.json``.

Standard library only. Writes are atomic (temp file + ``0o600`` + ``fsync`` +
``os.replace``) and reads refuse symlinks and oversized documents.
"""

from __future__ import annotations

import errno
import json
import os
import stat
import tempfile
import typing as tp
from dataclasses import dataclass, field
from pathlib import Path

from eyvan._internal.workspace import create_private_directory
from eyvan.storage._naming import (
    join_storage_path,
    parse_gcs_uri,
    validate_bucket_name,
    validate_location,
    validate_project_id,
    validate_region,
    validate_relative_path,
    validate_scope_segment,
)

DEFAULT_STORAGE_CONFIG_PATH = Path("~/.eyvan/storage.json")
STORAGE_CONFIG_SCHEMA = "eyvan.storage-config"
STORAGE_CONFIG_VERSION = 1
_MAX_CONFIG_BYTES = 256 * 1024


class StorageError(RuntimeError):
    """Base error for ``eyvan.storage`` operations."""


class StorageConfigError(StorageError, ValueError):
    """Raised when ``~/.eyvan/storage.json`` is malformed, unsafe, or cannot be written."""


class StorageNotConfiguredError(StorageError):
    """Raised when required storage is absent and creation was disabled."""


class StorageConsentRequiredError(StorageError):
    """Raised when creating a bucket requires explicit user consent or ``create=True``."""


class BucketNameConflictError(StorageError):
    """Raised when a candidate GCS bucket name is unavailable or owned by another project."""


class StorageAuthError(StorageError):
    """Raised when no valid GCP access token can be obtained (never exposes token data)."""


class StorageBackendError(StorageError):
    """Raised when a GCS JSON API call fails (never exposes credentials or raw bodies)."""


@dataclass(frozen=True, slots=True)
class RegionalBucketConfig:
    """Persisted regional storage bucket mapping for one GCP region."""

    bucket: str
    location: str
    uri: str = ""

    def __post_init__(self) -> None:
        bucket = validate_bucket_name(self.bucket)
        location = validate_region(self.location)
        expected_uri = f"gs://{bucket}"
        uri = self.uri or expected_uri
        parsed_bucket, parsed_prefix = parse_gcs_uri(uri)
        if parsed_bucket != bucket or parsed_prefix:
            raise StorageConfigError(
                f"regional bucket URI {uri!r} does not match canonical bucket root {expected_uri!r}"
            )
        object.__setattr__(self, "bucket", bucket)
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "uri", expected_uri)

    def as_document(self) -> dict[str, object]:
        """Serialize to a plain JSON-compatible dict."""

        return {
            "bucket": self.bucket,
            "uri": self.uri,
            "location": self.location,
        }

    @classmethod
    def from_document(
        cls,
        document: tp.Mapping[str, object],
        *,
        expected_region: str | None = None,
    ) -> RegionalBucketConfig:
        """Deserialize and validate one regional bucket entry."""

        if not isinstance(document, tp.Mapping):
            raise StorageConfigError("regional bucket entry must be a JSON object")
        bucket = _required_str(document, "bucket")
        location = _required_str(document, "location")
        uri = _optional_str(document, "uri") or f"gs://{bucket}"
        entry = cls(bucket=bucket, location=location, uri=uri)
        if expected_region is not None and entry.location != validate_region(expected_region):
            raise StorageConfigError(
                f"regional bucket key {expected_region!r} disagrees with entry location {entry.location!r}"
            )
        return entry


@dataclass(frozen=True, slots=True)
class MonitoringBucketConfig:
    """Persisted default monitoring bucket configuration for run tracking."""

    bucket: str
    location: str
    prefix: str = "tracking"
    uri: str = ""

    def __post_init__(self) -> None:
        bucket = validate_bucket_name(self.bucket)
        location = validate_location(self.location)
        prefix = validate_relative_path(self.prefix, allow_empty=True)
        expected_uri = f"gs://{bucket}/{prefix}" if prefix else f"gs://{bucket}"
        uri = self.uri or expected_uri
        parsed_bucket, parsed_prefix = parse_gcs_uri(uri)
        if parsed_bucket != bucket or parsed_prefix != prefix:
            raise StorageConfigError(f"monitoring bucket URI {uri!r} does not match canonical root {expected_uri!r}")
        object.__setattr__(self, "bucket", bucket)
        object.__setattr__(self, "location", location)
        object.__setattr__(self, "prefix", prefix)
        object.__setattr__(self, "uri", expected_uri)

    def tracking_uri(self, org: str = "default", project: str = "default") -> str:
        """Return the canonical tracking scope root ``gs://<bucket>/<prefix>/<org>/<project>``."""

        norm_org = validate_scope_segment(org, field_name="org")
        norm_project = validate_scope_segment(project, field_name="project")
        return join_storage_path(self.uri, norm_org, norm_project)

    def as_document(self) -> dict[str, object]:
        """Serialize to a plain JSON-compatible dict."""

        return {
            "bucket": self.bucket,
            "uri": self.uri,
            "location": self.location,
            "prefix": self.prefix,
        }

    @classmethod
    def from_document(cls, document: tp.Mapping[str, object]) -> MonitoringBucketConfig:
        """Deserialize and validate one monitoring bucket entry."""

        if not isinstance(document, tp.Mapping):
            raise StorageConfigError("monitoring bucket config must be a JSON object")
        bucket = _required_str(document, "bucket")
        location = _required_str(document, "location")
        raw_prefix = document.get("prefix", "tracking")
        if not isinstance(raw_prefix, str):
            raise StorageConfigError("monitoring prefix must be a string")
        uri = _optional_str(document, "uri") or (f"gs://{bucket}/{raw_prefix}" if raw_prefix else f"gs://{bucket}")
        return cls(bucket=bucket, location=location, prefix=raw_prefix, uri=uri)


@dataclass(frozen=True, slots=True)
class StorageConfig:
    """Top-level persisted Eyvan storage configuration (``~/.eyvan/storage.json``).

    Attributes:
        project: Default GCP project ID, if configured.
        monitoring: Single default monitoring bucket config across regions.
        regional: Mapping of GCP region name -> :class:`RegionalBucketConfig`.
        auto_create: Whether non-interactive calls with ``create=None`` may
            automatically create missing buckets.
        version: Document schema version (currently ``1``).
    """

    project: str | None = None
    monitoring: MonitoringBucketConfig | None = None
    regional: tp.Mapping[str, RegionalBucketConfig] = field(default_factory=dict)
    auto_create: bool = False
    version: int = STORAGE_CONFIG_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version != STORAGE_CONFIG_VERSION:
            raise StorageConfigError(f"storage config version must be {STORAGE_CONFIG_VERSION}")
        if not isinstance(self.auto_create, bool):
            raise StorageConfigError("storage config auto_create must be a boolean")
        project = validate_project_id(self.project) if self.project is not None else None
        object.__setattr__(self, "project", project)
        if self.monitoring is not None and not isinstance(self.monitoring, MonitoringBucketConfig):
            raise StorageConfigError("monitoring must be a MonitoringBucketConfig or None")
        normalized_regional: dict[str, RegionalBucketConfig] = {}
        for region_key, entry in self.regional.items():
            norm_region = validate_region(region_key)
            if not isinstance(entry, RegionalBucketConfig):
                raise StorageConfigError(f"regional entry for {region_key!r} must be a RegionalBucketConfig")
            if entry.location != norm_region:
                raise StorageConfigError(
                    f"regional key {norm_region!r} does not match bucket location {entry.location!r}"
                )
            normalized_regional[norm_region] = entry
        object.__setattr__(self, "regional", dict(sorted(normalized_regional.items())))

    def with_regional(
        self,
        entry: RegionalBucketConfig,
        *,
        project: str | None = None,
        auto_create: bool | None = None,
    ) -> StorageConfig:
        """Return a new config with ``entry`` recorded for ``entry.location``."""

        updated = dict(self.regional)
        updated[entry.location] = entry
        return StorageConfig(
            project=project if project is not None else self.project,
            monitoring=self.monitoring,
            regional=updated,
            auto_create=self.auto_create if auto_create is None else auto_create,
            version=self.version,
        )

    def with_monitoring(
        self,
        entry: MonitoringBucketConfig | None,
        *,
        project: str | None = None,
        auto_create: bool | None = None,
    ) -> StorageConfig:
        """Return a new config with ``monitoring`` updated."""

        return StorageConfig(
            project=project if project is not None else self.project,
            monitoring=entry,
            regional=dict(self.regional),
            auto_create=self.auto_create if auto_create is None else auto_create,
            version=self.version,
        )

    def as_document(self) -> dict[str, object]:
        """Serialize to the canonical versioned JSON document."""

        return {
            "schema": STORAGE_CONFIG_SCHEMA,
            "version": self.version,
            "project": self.project,
            "auto_create": self.auto_create,
            "monitoring": None if self.monitoring is None else self.monitoring.as_document(),
            "regional": {region: entry.as_document() for region, entry in sorted(self.regional.items())},
        }

    @classmethod
    def from_document(cls, document: tp.Mapping[str, object]) -> StorageConfig:
        """Parse and validate a decoded JSON object into a :class:`StorageConfig`."""

        if not isinstance(document, tp.Mapping):
            raise StorageConfigError("storage config document must be a JSON object")
        schema = document.get("schema", STORAGE_CONFIG_SCHEMA)
        if schema != STORAGE_CONFIG_SCHEMA:
            raise StorageConfigError(f"unsupported storage config schema {schema!r}")
        version = document.get("version", STORAGE_CONFIG_VERSION)
        if isinstance(version, bool) or not isinstance(version, int):
            raise StorageConfigError("storage config version must be an integer")
        project = _optional_str(document, "project")
        auto_create = document.get("auto_create", False)
        if not isinstance(auto_create, bool):
            raise StorageConfigError("storage config auto_create must be a boolean")
        raw_monitoring = document.get("monitoring")
        monitoring: MonitoringBucketConfig | None = None
        if raw_monitoring is not None:
            if not isinstance(raw_monitoring, tp.Mapping):
                raise StorageConfigError("storage config monitoring must be a JSON object or null")
            monitoring = MonitoringBucketConfig.from_document(raw_monitoring)
        raw_regional = document.get("regional", {})
        if not isinstance(raw_regional, tp.Mapping):
            raise StorageConfigError("storage config regional must be a JSON object")
        regional: dict[str, RegionalBucketConfig] = {}
        for region_key, raw_entry in raw_regional.items():
            if not isinstance(region_key, str):
                raise StorageConfigError("storage config regional keys must be strings")
            if not isinstance(raw_entry, tp.Mapping):
                raise StorageConfigError(f"regional entry for {region_key!r} must be a JSON object")
            entry = RegionalBucketConfig.from_document(raw_entry, expected_region=region_key)
            regional[entry.location] = entry
        return cls(
            project=project,
            monitoring=monitoring,
            regional=regional,
            auto_create=auto_create,
            version=version,
        )


class StorageConfigStore:
    """Read and atomically persist ``~/.eyvan/storage.json`` with ``0o600`` mode."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._explicit_path = Path(path).expanduser() if path is not None else None

    @property
    def path(self) -> Path:
        """Return the resolved path (expanding ``~`` dynamically when default)."""

        if self._explicit_path is not None:
            return self._explicit_path
        return DEFAULT_STORAGE_CONFIG_PATH.expanduser()

    def load_if_exists(self) -> StorageConfig | None:
        """Load the config file, returning ``None`` if it does not exist yet."""

        target = self.path
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(target, flags)
        except FileNotFoundError:
            return None
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise StorageConfigError(f"storage config {target} must not be a symlink") from None
            raise StorageConfigError(f"could not open storage config {target}: {exc.strerror or exc}") from None
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise StorageConfigError(f"storage config {target} must be a regular file")
            chunks: list[bytes] = []
            total = 0
            while chunk := os.read(descriptor, 4096):
                total += len(chunk)
                if total > _MAX_CONFIG_BYTES:
                    raise StorageConfigError(f"storage config {target} exceeds {_MAX_CONFIG_BYTES} bytes")
                chunks.append(chunk)
            payload = b"".join(chunks)
        finally:
            os.close(descriptor)
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise StorageConfigError(f"could not parse storage config {target}: {exc}") from None
        if not isinstance(decoded, dict):
            raise StorageConfigError(f"storage config {target} must be a JSON object")
        return StorageConfig.from_document(decoded)

    def load(self) -> StorageConfig:
        """Load the config file, or return an empty :class:`StorageConfig` if absent."""

        loaded = self.load_if_exists()
        return loaded if loaded is not None else StorageConfig()

    def save(self, config: StorageConfig) -> Path:
        """Atomically persist ``config`` with owner-only ``0o600`` permissions."""

        if not isinstance(config, StorageConfig):
            raise StorageConfigError("save() requires a StorageConfig instance")
        target = self.path
        parent = target.parent
        try:
            create_private_directory(parent, mode=0o700)
        except OSError as exc:
            raise StorageConfigError(f"could not create storage config directory {parent}: {exc}") from None
        if target.is_symlink():
            raise StorageConfigError(f"storage config {target} must not be a symlink")
        payload = (json.dumps(config.as_document(), indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
        descriptor = -1
        temp_path: Path | None = None
        try:
            descriptor, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=parent)
            temp_path = Path(temp_name)
            os.fchmod(descriptor, 0o600)
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                if written <= 0:
                    raise StorageConfigError(f"short write while saving storage config {target}")
                offset += written
            os.fsync(descriptor)
            os.close(descriptor)
            descriptor = -1
            if target.is_symlink():
                raise StorageConfigError(f"storage config {target} must not be a symlink")
            os.replace(temp_path, target)
            temp_path = None
            _fsync_dir_best_effort(parent)
            return target
        except StorageConfigError:
            raise
        except OSError as exc:
            raise StorageConfigError(f"could not save storage config {target}: {exc}") from None
        finally:
            if descriptor >= 0:
                os.close(descriptor)
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


def _fsync_dir_best_effort(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _required_str(document: tp.Mapping[str, object], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise StorageConfigError(f"storage config field {key!r} must be a non-empty string")
    return value


def _optional_str(document: tp.Mapping[str, object], key: str) -> str | None:
    value = document.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise StorageConfigError(f"storage config field {key!r} must be a string or null")
    stripped = value.strip()
    return stripped or None


__all__ = (
    "DEFAULT_STORAGE_CONFIG_PATH",
    "STORAGE_CONFIG_SCHEMA",
    "STORAGE_CONFIG_VERSION",
    "BucketNameConflictError",
    "MonitoringBucketConfig",
    "RegionalBucketConfig",
    "StorageAuthError",
    "StorageBackendError",
    "StorageConfig",
    "StorageConfigError",
    "StorageConfigStore",
    "StorageConsentRequiredError",
    "StorageError",
    "StorageNotConfiguredError",
)
