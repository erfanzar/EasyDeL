# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Bucket naming policies, GCP location validation, and safe GCS path helpers.

Standard library only so training jobs can import ``eyvan.storage`` without
pulling SDK, HTTP client, or control-plane dependencies.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from urllib.parse import urlsplit

#: Maximum GCS bucket name length permitted without DNS dotted names.
MAX_BUCKET_LENGTH = 63
MIN_BUCKET_LENGTH = 3

_ZONE_RE = re.compile(r"^[a-z]+-[a-z]+[0-9]+-[a-z][a-z0-9]*$")
_REGION_RE = re.compile(r"^[a-z]+-[a-z]+[0-9]+$")
_MULTI_REGION_RE = re.compile(r"^[a-z]{2,4}[0-9]?$")
_PROJECT_RE = re.compile(r"^[a-z][a-z0-9-]{4,28}[a-z0-9]$")
_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]$")
_GENERATED_BUCKET_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,61}[a-z0-9]$")
_SLUG_UNSAFE_RE = re.compile(r"[^a-z0-9_-]+")
_COLLAPSE_SEP_RE = re.compile(r"[-_]{2,}")
_SCOPE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_FORBIDDEN_PATH_CHARS = frozenset({"\\", "\x00", "?", "#"})


class StorageUri(str):
    """A validated ``gs://`` URI string supporting safe ``/`` path chaining.

    ``StorageUri`` is a :class:`str` subclass, so it compares equal to plain
    strings, serializes directly with :mod:`json`, and passes cleanly to any
    API expecting a ``str`` URI while also supporting ``handle / "a" / "b"``.
    """

    def __truediv__(self, part: str) -> StorageUri:
        return StorageUri(join_storage_path(str(self), part))


def validate_project_id(project: str) -> str:
    """Validate and normalize a GCP project ID.

    Args:
        project: Candidate GCP project ID.

    Returns:
        Normalized lowercase project ID.

    Raises:
        ValueError: If ``project`` is not a valid 6-30 character GCP project ID.
    """

    if not isinstance(project, str):
        raise ValueError("GCP project ID must be a string")
    cleaned = project.strip().lower()
    if cleaned.startswith("gs://"):
        raise ValueError(f"GCP project must be a project ID, not a GCS URI ({project!r})")
    if not _PROJECT_RE.fullmatch(cleaned):
        raise ValueError(
            f"invalid GCP project ID {project!r}: expected 6-30 lowercase letters, digits, or hyphens "
            "starting with a letter and ending with a letter or digit"
        )
    return cleaned


def validate_zone(zone: str) -> str:
    """Validate and normalize a GCP zone name (e.g. ``us-central2-b``).

    Args:
        zone: Candidate GCP zone name.

    Returns:
        Normalized lowercase zone name.

    Raises:
        ValueError: If ``zone`` does not match GCP zone syntax.
    """

    if not isinstance(zone, str):
        raise ValueError("GCP zone must be a string")
    cleaned = zone.strip().lower()
    if not _ZONE_RE.fullmatch(cleaned):
        raise ValueError(f"invalid GCP zone {zone!r}: expected a zone such as 'us-central2-b'")
    return cleaned


def zone_to_region(zone: str) -> str:
    """Derive a GCP region from a zone name (e.g. ``us-central2-b`` -> ``us-central2``).

    Args:
        zone: GCP zone name or full resource path (``projects/.../zones/<zone>``).

    Returns:
        Normalized GCP region name.

    Raises:
        ValueError: If ``zone`` cannot be parsed as a valid GCP zone.
    """

    if not isinstance(zone, str) or not zone.strip():
        raise ValueError("GCP zone must be a non-empty string")
    leaf = zone.strip().rstrip("/").rsplit("/", 1)[-1]
    validated = validate_zone(leaf)
    region = validated.rsplit("-", 1)[0]
    return validate_region(region)


def validate_region(region: str) -> str:
    """Validate and normalize a single-region GCP location (e.g. ``us-central2``).

    Args:
        region: Candidate GCP region name.

    Returns:
        Normalized lowercase region name.

    Raises:
        ValueError: If ``region`` is malformed or is a zone instead of a region.
    """

    if not isinstance(region, str):
        raise ValueError("GCP region must be a string")
    cleaned = region.strip().lower()
    if _ZONE_RE.fullmatch(cleaned):
        derived = cleaned.rsplit("-", 1)[0]
        raise ValueError(f"{region!r} is a GCP zone, not a region; pass zone={cleaned!r} or region={derived!r}")
    if not _REGION_RE.fullmatch(cleaned):
        raise ValueError(f"invalid GCP region {region!r}: expected a region such as 'us-central2'")
    return cleaned


def validate_location(location: str) -> str:
    """Validate a GCS bucket location (single region or multi/dual-region).

    Args:
        location: Candidate GCS bucket location such as ``us-central1`` or ``us``.

    Returns:
        Normalized lowercase location string.

    Raises:
        ValueError: If ``location`` is a zone or not a valid GCS location token.
    """

    if not isinstance(location, str):
        raise ValueError("GCS location must be a string")
    cleaned = location.strip().lower()
    if _ZONE_RE.fullmatch(cleaned):
        derived = cleaned.rsplit("-", 1)[0]
        raise ValueError(
            f"{location!r} is a GCP zone, not a bucket location; pass zone={cleaned!r} or location={derived!r}"
        )
    if _REGION_RE.fullmatch(cleaned) or _MULTI_REGION_RE.fullmatch(cleaned):
        return cleaned
    raise ValueError(f"invalid GCS bucket location {location!r}: expected a region (e.g. 'us-central1') or 'us'/'eu'")


def validate_bucket_name(bucket: str) -> str:
    """Validate a GCS bucket name (3-63 chars, lowercase letters/digits/._-).

    Args:
        bucket: Candidate GCS bucket name (without ``gs://``).

    Returns:
        Validated lowercase bucket name.

    Raises:
        ValueError: If the bucket name violates GCS naming rules.
    """

    if not isinstance(bucket, str):
        raise ValueError("GCS bucket name must be a string")
    cleaned = bucket.strip()
    if not cleaned or cleaned != cleaned.lower():
        raise ValueError(f"invalid GCS bucket name {bucket!r}: bucket names must be lowercase without whitespace")
    if cleaned.startswith("gs://"):
        raise ValueError(f"expected a GCS bucket name without 'gs://', got {bucket!r}")
    if (
        len(cleaned) < MIN_BUCKET_LENGTH
        or len(cleaned) > MAX_BUCKET_LENGTH
        or not _BUCKET_RE.fullmatch(cleaned)
        or ".." in cleaned
        or ".-" in cleaned
        or "-." in cleaned
        or "\x00" in cleaned
    ):
        raise ValueError(
            f"invalid GCS bucket name {bucket!r}: expected 3-63 lowercase letters, digits, '_', '-', or '.' "
            "starting and ending with a letter or digit"
        )
    return cleaned


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Parse ``gs://bucket[/prefix]`` into ``(bucket, normalized_prefix)``.

    Args:
        uri: GCS URI string.

    Returns:
        Tuple of validated bucket name and normalized relative prefix (or ``""``).

    Raises:
        ValueError: If ``uri`` is not a safe ``gs://`` URI.
    """

    if not isinstance(uri, str) or not uri.strip() or uri != uri.strip() or "\x00" in uri:
        raise ValueError("GCS URI must be a non-empty trimmed string")
    parsed = urlsplit(uri)
    if parsed.scheme != "gs" or not parsed.netloc or parsed.query or parsed.fragment:
        raise ValueError(f"invalid GCS URI {uri!r}: expected gs://bucket[/prefix] without query or fragment")
    bucket = validate_bucket_name(parsed.netloc)
    raw_path = parsed.path.lstrip("/").rstrip("/")
    if not raw_path:
        return bucket, ""
    return bucket, validate_relative_path(raw_path)


def validate_relative_path(path: str, *, allow_empty: bool = False) -> str:
    """Validate a relative POSIX object path, rejecting traversal and unsafe characters.

    Args:
        path: Relative object path or prefix.
        allow_empty: Whether ``""`` is accepted.

    Returns:
        Normalized relative POSIX path without leading or trailing slashes.

    Raises:
        ValueError: If ``path`` is absolute, contains ``..`` or ``.``, empty segments,
            backslashes, null bytes, or URL query/fragment delimiters.
    """

    if not isinstance(path, str):
        raise ValueError("storage path segment must be a string")
    if path == "" and allow_empty:
        return ""
    if not path or any(ch in path for ch in _FORBIDDEN_PATH_CHARS):
        raise ValueError(f"invalid storage path {path!r}")
    if path.startswith("/"):
        raise ValueError(f"storage path segment must be relative, got {path!r}")
    stripped = path.rstrip("/")
    if not stripped:
        raise ValueError(f"invalid storage path {path!r}")
    parts = stripped.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"invalid storage path {path!r}: '.' and '..' and empty segments are forbidden")
    return str(PurePosixPath(*parts))


def validate_scope_segment(value: str, *, field_name: str = "scope") -> str:
    """Validate a single tracking scope segment (org or project).

    Args:
        value: Candidate org or project segment.
        field_name: Field label for error reporting.

    Returns:
        Validated segment string.

    Raises:
        ValueError: If ``value`` is not a single safe segment.
    """

    if not isinstance(value, str) or not _SCOPE_SEGMENT_RE.fullmatch(value) or value in {".", ".."}:
        raise ValueError(
            f"invalid {field_name} {value!r}: use 1-128 letters, digits, '.', '_' or '-', "
            "starting with a letter or digit"
        )
    return value


def join_storage_path(base_uri: str, *parts: str) -> str:
    """Safely join relative object path segments onto a ``gs://`` URI.

    Args:
        base_uri: Base ``gs://bucket[/prefix]`` URI.
        *parts: Relative path segments to append.

    Returns:
        Canonical ``gs://`` URI with joined segments.

    Raises:
        ValueError: If ``base_uri`` or any segment is invalid or attempts path traversal.
    """

    bucket, prefix = parse_gcs_uri(base_uri)
    validated_parts = [validate_relative_path(part) for part in parts]
    all_parts = [part for part in (prefix, *validated_parts) if part]
    if not all_parts:
        return f"gs://{bucket}"
    return f"gs://{bucket}/{'/'.join(all_parts)}"


def slugify_project(project: str) -> str:
    """Convert a GCP project identifier into a valid lowercase GCS bucket slug.

    Args:
        project: GCP project ID or slug.

    Returns:
        Non-empty lowercase slug restricted to ``[a-z0-9_-]``.

    Raises:
        ValueError: If ``project`` yields an empty slug.
    """

    if not isinstance(project, str) or not project.strip():
        raise ValueError("project must be a non-empty string")
    lowered = project.strip().lower()
    cleaned = _SLUG_UNSAFE_RE.sub("-", lowered)
    cleaned = _COLLAPSE_SEP_RE.sub("-", cleaned).strip("-_")
    if not cleaned:
        raise ValueError(f"cannot derive a GCS bucket slug from project {project!r}")
    return cleaned


def _bounded_bucket_name(prefix: str, suffix: str, *, salt: str | None = None) -> str:
    """Combine ``prefix`` and ``suffix`` into a valid <=63-char GCS bucket name."""

    base = f"{prefix}_{suffix}" if suffix else prefix
    if salt is None and len(base) <= MAX_BUCKET_LENGTH and _GENERATED_BUCKET_RE.fullmatch(base):
        return validate_bucket_name(base)
    digest_input = f"{base}:{salt}" if salt is not None else base
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:6]
    max_head = MAX_BUCKET_LENGTH - len(digest) - 1
    head = base[:max_head].rstrip("-_")
    if not head or not head[0].isalnum():
        head = prefix[:max_head].rstrip("-_")
    candidate = f"{head}_{digest}"
    return validate_bucket_name(candidate)


@dataclass(frozen=True, slots=True)
class BucketNamingPolicy:
    """Customizable policy generating GCS bucket name candidates for Eyvan.

    Subclass or instantiate with custom prefixes to adapt naming conventions
    without changing resolution or consent logic.

    Attributes:
        regional_prefix: Prefix for per-region storage buckets (default ``"eyvan"``).
        monitoring_prefix: Prefix for the default monitoring bucket
            (default ``"eyvan_monitoring"``).
        default_monitoring_location: Fallback GCS location when no region is
            detected for the monitoring bucket (default ``"us-central1"``).
        default_monitoring_path_prefix: Default object prefix inside the
            monitoring bucket (default ``"tracking"``).
    """

    regional_prefix: str = "eyvan"
    monitoring_prefix: str = "eyvan_monitoring"
    default_monitoring_location: str = "us-central1"
    default_monitoring_path_prefix: str = "tracking"

    def __post_init__(self) -> None:
        for attr in ("regional_prefix", "monitoring_prefix"):
            val = getattr(self, attr)
            if not isinstance(val, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,30}", val):
                raise ValueError(f"{attr} must match [a-z0-9][a-z0-9_-]{{0,30}}, got {val!r}")
        object.__setattr__(self, "default_monitoring_location", validate_location(self.default_monitoring_location))
        object.__setattr__(
            self,
            "default_monitoring_path_prefix",
            validate_relative_path(self.default_monitoring_path_prefix, allow_empty=True),
        )

    def regional_candidates(self, region: str, *, project: str | None = None) -> tuple[str, ...]:
        """Return ordered bucket name candidates for regional storage in ``region``.

        Tries ``<regional_prefix>_<region>`` first (e.g. ``eyvan_us-central2``),
        then falls back to ``<regional_prefix>_<region>_<project_slug>`` (<=63 chars)
        and a deterministic hash-suffixed fallback if globally taken.
        """

        norm_region = validate_region(region)
        primary = _bounded_bucket_name(self.regional_prefix, norm_region)
        candidates: list[str] = [primary]
        if project:
            slug = slugify_project(project)
            scoped = _bounded_bucket_name(f"{self.regional_prefix}_{norm_region}", slug)
            if scoped not in candidates:
                candidates.append(scoped)
            hashed = _bounded_bucket_name(f"{self.regional_prefix}_{norm_region}", slug, salt=f"{project}:{norm_region}")
            if hashed not in candidates:
                candidates.append(hashed)
        return tuple(candidates)

    def monitoring_candidates(
        self,
        *,
        project: str | None = None,
        location: str | None = None,
    ) -> tuple[str, ...]:
        """Return ordered bucket name candidates for the default monitoring store.

        Uses ``<monitoring_prefix>_<project_slug>`` when ``project`` is known,
        with a deterministic collision-safe hash fallback if globally taken.
        """

        if location is not None:
            validate_location(location)
        candidates: list[str] = []
        if project:
            slug = slugify_project(project)
            primary = _bounded_bucket_name(self.monitoring_prefix, slug)
            candidates.append(primary)
            hashed = _bounded_bucket_name(self.monitoring_prefix, slug, salt=f"monitoring:{project}")
            if hashed not in candidates:
                candidates.append(hashed)
        else:
            candidates.append(validate_bucket_name(self.monitoring_prefix))
        return tuple(candidates)


DEFAULT_NAMING_POLICY = BucketNamingPolicy()

__all__ = (
    "DEFAULT_NAMING_POLICY",
    "MAX_BUCKET_LENGTH",
    "MIN_BUCKET_LENGTH",
    "BucketNamingPolicy",
    "StorageUri",
    "join_storage_path",
    "parse_gcs_uri",
    "slugify_project",
    "validate_bucket_name",
    "validate_location",
    "validate_project_id",
    "validate_region",
    "validate_relative_path",
    "validate_scope_segment",
    "validate_zone",
    "zone_to_region",
)
