# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pure-stdlib cross-region GCS transfer locality engine and live runtime guard.

This module has zero third-party and zero ``eyvan.*`` imports so it can be both
imported as ``eyvan.storage._region_guard`` and staged standalone as
``eyvan_region_guard.py`` next to ``sitecustomize.py`` on native TPU/GPU hosts
and OCI workers.

Security and safety invariants:
- Never leaks OAuth bearer tokens or raw HTTP response bodies in error messages.
- Refuses HTTP redirects on authenticated GCS metadata lookups.
- Reaches GCE metadata directly with ``ProxyHandler({})`` and requires the
  ``Metadata-Flavor: Google`` response header.
- Uses a thread-local re-entrancy bypass counter so metadata lookups performed
  by the guard itself can never recurse or trigger false violations.
- Restores every monkeypatched hook cleanly and idempotently in
  :func:`disable_cross_region_guard`.
"""

import http.client
import importlib.abc
import importlib.machinery
import inspect
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
import typing as tp
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

#: Dedicated process exit code emitted when a cross-region transfer ban kills a process.
CROSS_REGION_TRANSFER_BANNED_EXIT_CODE = 95

#: Single-line stderr marker emitted before raising or exiting on a cross-region transfer ban.
CROSS_REGION_TRANSFER_BANNED_MARKER = "eyvan-cross-region-transfer-banned"

#: Stable machine-readable prefix surfaced in workload failure reasons.
CROSS_REGION_TRANSFER_BANNED_REASON = "cross-region-transfer-banned"

#: Runtime-owned environment variables injected by Eyvan's control plane / dispatch path.
CROSS_REGION_POLICY_ENV = "EYVAN_CROSS_REGION_POLICY"
HOST_REGION_ENV = "EYVAN_HOST_REGION"
CROSS_REGION_ALLOWED_BUCKETS_ENV = "EYVAN_CROSS_REGION_ALLOWED_BUCKETS"
CROSS_REGION_ALLOW_MULTIREGION_ENV = "EYVAN_CROSS_REGION_ALLOW_MULTIREGION"
CROSS_REGION_KILL_ENV = "EYVAN_CROSS_REGION_KILL"

_VALID_POLICIES = frozenset({"allow", "warn", "ban"})
_METADATA_TOKEN_URL = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
_GCS_BUCKET_API_TEMPLATE = (
    "https://storage.googleapis.com/storage/v1/b/{bucket}?fields=location,locationType,customPlacementConfig"
)
_DEFAULT_METADATA_TIMEOUT_S = 1.5
_DEFAULT_LOOKUP_TIMEOUT_S = 5.0
_MAX_METADATA_RESPONSE_BYTES = 256 * 1024

_ZONE_RE = re.compile(r"^([a-z]+-[a-z]+[0-9]+)-[a-z][a-z0-9]*$")
_REGION_RE = re.compile(r"^[a-z]+-[a-z]+[0-9]+$")
_BUCKET_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]$")
_GCS_URI_RE = re.compile(r"gs://([a-z0-9][a-z0-9._-]{1,61}[a-z0-9])(?:/[^\s'\"`,;)\]}<>|&]*)?", re.IGNORECASE)
_GCS_JSON_API_BUCKET_RE = re.compile(r"^/(?:(?:upload|download)/)?storage/v\d+/b/([^/?#]+)(?:/|$)")

_GCS_PATH_HOSTS = frozenset({"storage.googleapis.com", "storage.cloud.google.com"})
_GCS_API_HOSTS = frozenset({"storage.googleapis.com", "www.googleapis.com", "storage.cloud.google.com"})
_GCS_RESERVED_FIRST_SEGMENTS = frozenset(
    {"storage", "upload", "download", "batch", "b", "o", "favicon.ico", "robots.txt"}
)

_PREDEFINED_DUAL_REGIONS: dict[str, frozenset[str]] = {
    "nam4": frozenset({"us-central1", "us-east1"}),
    "eur4": frozenset({"europe-north1", "europe-west4"}),
    "eur5": frozenset({"europe-west1", "europe-west2"}),
    "eur7": frozenset({"europe-west3", "europe-west6"}),
    "eur8": frozenset({"europe-west3", "europe-west6"}),
    "asia1": frozenset({"asia-northeast1", "asia-northeast2"}),
}

_MULTI_REGION_PREFIXES: dict[str, tuple[str, ...]] = {
    "us": ("us-",),
    "eu": ("europe-",),
    "asia": ("asia-",),
}

_WATCHED_MODULES = frozenset({"tensorstore", "gcsfs"})
_GCSFS_HOOKED_METHODS = ("open", "_call", "get", "put", "cat", "pipe")
_SHELL_EXECUTABLES = frozenset({"sh", "bash", "zsh", "dash"})

#: Process termination function used when ``kill_on_violation=True``.
#: Indirected at module level so tests can observe termination without killing pytest.
terminate_on_violation: Callable[[int], tp.NoReturn] = os._exit

#: HTTP transport callable: ``(method, url, headers, body) -> (status, response_headers, response_bytes)``.
_Transport = Callable[[str, str, dict[str, str], bytes | None], tuple[int, Mapping[str, str], bytes]]


@dataclass(frozen=True, slots=True)
class BucketLocation:
    """Resolved GCS bucket location metadata."""

    bucket: str
    location: str
    location_type: str = "unknown"
    data_locations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "bucket", self.bucket.strip().lower())
        object.__setattr__(self, "location", self.location.strip().lower())
        object.__setattr__(self, "location_type", self.location_type.strip().lower() or "unknown")
        object.__setattr__(
            self,
            "data_locations",
            tuple(item.strip().lower() for item in self.data_locations if isinstance(item, str) and item.strip()),
        )


@dataclass(frozen=True, slots=True)
class LocalityCheckResult:
    """Structured verdict for one GCS URI locality check."""

    allowed: bool
    host_region: str
    bucket: str
    bucket_location: str
    location_type: str
    uri: str
    operation: str
    reason: str


class CrossRegionTransferError(PermissionError):
    """Raised when a cross-region GCS transfer is blocked in ``ban`` mode."""

    def __init__(
        self,
        message: str = "",
        *,
        host_region: str = "",
        bucket: str = "",
        bucket_location: str = "",
        uri: str = "",
        operation: str = "",
        result: LocalityCheckResult | None = None,
    ) -> None:
        if result is not None:
            host_region = host_region or result.host_region
            bucket = bucket or result.bucket
            bucket_location = bucket_location or result.bucket_location
            uri = uri or result.uri
            operation = operation or result.operation
        operation = operation or "access"
        if not message:
            target = uri or (f"gs://{bucket}" if bucket else "<unknown>")
            message = (
                f"{CROSS_REGION_TRANSFER_BANNED_REASON}: host in {host_region or 'unknown'} "
                f"attempted {operation} on {target} (bucket {bucket!r} location: {bucket_location or 'unknown'})"
            )
        super().__init__(message)
        self.host_region = host_region
        self.bucket = bucket
        self.bucket_location = bucket_location
        self.uri = uri
        self.operation = operation
        self.result = result


@dataclass(frozen=True, slots=True)
class _GuardConfig:
    host_region: str
    mode: str
    allowed_buckets: frozenset[str]
    allow_multiregion: bool
    kill_on_violation: bool
    location_resolver: Callable[[str], BucketLocation | None] | None
    on_unknown: str
    on_violation: Callable[[LocalityCheckResult], None] | None
    terminate: Callable[[int], tp.NoReturn] | None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse HTTP redirects so bearer tokens are never forwarded to another host."""

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: tp.Any,
        code: int,
        msg: str,
        headers: tp.Any,
        newurl: str,
    ) -> None:
        return None


_GUARD_LOCAL = threading.local()
_GUARD_LOCK = threading.RLock()
_ACTIVE_GUARD: _GuardConfig | None = None

_LOCATION_CACHE_LOCK = threading.Lock()
_LOCATION_CACHE: dict[str, BucketLocation] = {}

_TOKEN_CACHE_LOCK = threading.Lock()
_CACHED_TOKEN: str | None = None
_CACHED_TOKEN_EXPIRES_AT: float = 0.0

_STDLIB_ORIGINALS: dict[str, tp.Any] = {}
_PATCHED_ATTRS: list[tuple[tp.Any, str, bool, tp.Any]] = []
_PATCHED_TARGETS: set[tuple[int, str]] = set()
_META_PATH_FINDER: importlib.abc.MetaPathFinder | None = None


def _is_bypassed() -> bool:
    return getattr(_GUARD_LOCAL, "bypass_depth", 0) > 0


@contextmanager
def _bypass_region_guard() -> tp.Iterator[None]:
    depth = getattr(_GUARD_LOCAL, "bypass_depth", 0)
    _GUARD_LOCAL.bypass_depth = depth + 1
    try:
        yield
    finally:
        _GUARD_LOCAL.bypass_depth = depth


def _is_bucket_checked_in_frame(bucket: str) -> bool:
    checked: set[str] | None = getattr(_GUARD_LOCAL, "checked_buckets", None)
    return checked is not None and bucket in checked


@contextmanager
def _mark_bucket_checked(bucket: str) -> tp.Iterator[None]:
    checked: set[str] | None = getattr(_GUARD_LOCAL, "checked_buckets", None)
    if checked is None:
        checked = set()
        _GUARD_LOCAL.checked_buckets = checked
    added = bucket not in checked
    if added:
        checked.add(bucket)
    try:
        yield
    finally:
        if added:
            checked.discard(bucket)


def _normalize_region(region_or_zone: str) -> str:
    """Normalize a GCP region or zone (including resource paths) to lowercase region."""

    if not isinstance(region_or_zone, str):
        return ""
    cleaned = region_or_zone.strip().lower()
    if not cleaned:
        return ""
    if "/" in cleaned:
        cleaned = cleaned.rstrip("/").rsplit("/", 1)[-1]
    match = _ZONE_RE.fullmatch(cleaned)
    if match is not None:
        return match.group(1)
    return cleaned


def _infer_location_type(location: str, data_locations: tuple[str, ...] = ()) -> str:
    loc = location.strip().lower()
    if data_locations or loc in _PREDEFINED_DUAL_REGIONS:
        return "dual-region"
    if loc in _MULTI_REGION_PREFIXES:
        return "multi-region"
    if _REGION_RE.fullmatch(loc):
        return "region"
    return "unknown"


def is_location_local_to_region(
    host_region: str,
    bucket_loc: BucketLocation,
    *,
    allow_multiregion: bool = True,
) -> bool:
    """Return whether ``bucket_loc`` is local to ``host_region`` under GCP locality rules.

    Args:
        host_region: GCP region (or zone, automatically normalized to region) of the compute host.
        bucket_loc: Resolved :class:`BucketLocation` metadata for the target bucket.
        allow_multiregion: Whether continental multi-region buckets (``us``, ``eu``, ``asia``)
            matching the host's continent are permitted. Set ``False`` for strict single-region mode.

    Returns:
        ``True`` if accessing ``bucket_loc`` from ``host_region`` is considered local.
    """

    norm_host = _normalize_region(host_region)
    if not norm_host:
        return False

    loc = bucket_loc.location.strip().lower()
    loc_type = bucket_loc.location_type.strip().lower()

    if loc_type == "dual-region":
        if bucket_loc.data_locations:
            return norm_host in {entry.strip().lower() for entry in bucket_loc.data_locations}
        regions = _PREDEFINED_DUAL_REGIONS.get(loc)
        return norm_host in regions if regions is not None else False

    if loc_type == "multi-region" or loc in _MULTI_REGION_PREFIXES:
        if not allow_multiregion:
            return False
        prefixes = _MULTI_REGION_PREFIXES.get(loc)
        if prefixes is None:
            return False
        return norm_host.startswith(prefixes)

    if loc_type == "region":
        return bool(loc) and loc == norm_host

    return False


def _header_get_ci(headers: Mapping[str, str], name: str) -> str | None:
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return value
    return None


def _valid_bearer_token(token: object) -> bool:
    return (
        isinstance(token, str)
        and bool(token)
        and "\r" not in token
        and "\n" not in token
        and " " not in token
        and "\t" not in token
    )


def _read_bounded(stream: tp.BinaryIO, limit: int = _MAX_METADATA_RESPONSE_BYTES) -> bytes:
    output = bytearray()
    while chunk := stream.read(min(65536, limit - len(output) + 1)):
        output.extend(chunk)
        if len(output) > limit:
            raise ValueError("response exceeded size limit")
    return bytes(output)


def _fetch_gce_metadata_token(*, timeout_s: float = _DEFAULT_METADATA_TIMEOUT_S) -> str | None:
    """Obtain an OAuth bearer token from GCE metadata without using HTTP proxies."""

    global _CACHED_TOKEN, _CACHED_TOKEN_EXPIRES_AT
    now = time.monotonic()
    with _TOKEN_CACHE_LOCK:
        if _CACHED_TOKEN is not None and now < _CACHED_TOKEN_EXPIRES_AT:
            return _CACHED_TOKEN

    with _bypass_region_guard():
        request = urllib.request.Request(
            _METADATA_TOKEN_URL,
            headers={"Metadata-Flavor": "Google"},
            method="GET",
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        try:
            with opener.open(request, timeout=timeout_s) as response:
                if response.status != 200:
                    return None
                flavor = _header_get_ci(dict(response.headers.items()), "Metadata-Flavor")
                if flavor is None or flavor.strip().lower() != "google":
                    return None
                raw = _read_bounded(response)
        except Exception:
            return None

    try:
        payload = json.loads(raw.decode("utf-8"))
        token = payload.get("access_token")
        expires_in = float(payload.get("expires_in", 300))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        return None
    if not _valid_bearer_token(token) or not math.isfinite(expires_in) or expires_in <= 0:
        return None

    ttl = max(1.0, expires_in - min(60.0, expires_in / 10.0))
    with _TOKEN_CACHE_LOCK:
        _CACHED_TOKEN = token
        _CACHED_TOKEN_EXPIRES_AT = time.monotonic() + ttl
    return token


def _default_gcs_get(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
    *,
    timeout_s: float = _DEFAULT_LOOKUP_TIMEOUT_S,
) -> tuple[int, Mapping[str, str], bytes]:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    opener = urllib.request.build_opener(_NoRedirect())
    try:
        with opener.open(request, timeout=timeout_s) as response:
            return response.status, dict(response.headers.items()), _read_bounded(response)
    except urllib.error.HTTPError as error:
        with error:
            try:
                payload = _read_bounded(error)
            except Exception:
                payload = b""
            return error.code, dict(error.headers.items()), payload


def clear_bucket_location_cache() -> None:
    """Clear process-level bucket location and metadata token caches."""

    global _CACHED_TOKEN, _CACHED_TOKEN_EXPIRES_AT
    with _LOCATION_CACHE_LOCK:
        _LOCATION_CACHE.clear()
    with _TOKEN_CACHE_LOCK:
        _CACHED_TOKEN = None
        _CACHED_TOKEN_EXPIRES_AT = 0.0


def prime_bucket_location_cache(bucket_location: BucketLocation) -> None:
    """Insert a resolved :class:`BucketLocation` into the process-level cache."""

    with _LOCATION_CACHE_LOCK:
        _LOCATION_CACHE[bucket_location.bucket] = bucket_location


def resolve_bucket_location(
    bucket: str,
    *,
    transport: _Transport | None = None,
    token_provider: Callable[[], str | None] | None = None,
    timeout_s: float = _DEFAULT_LOOKUP_TIMEOUT_S,
    use_cache: bool = True,
) -> BucketLocation | None:
    """Resolve a GCS bucket's location metadata via the GCS JSON API.

    Results are cached in a thread-safe process-level dictionary keyed by bucket
    name. All network requests execute under the guard's re-entrancy bypass so
    bucket-location checks can never infinitely recurse.

    Args:
        bucket: GCS bucket name (without ``gs://``).
        transport: Optional injectable HTTP transport for offline tests.
        token_provider: Optional callable returning an OAuth bearer token (or ``None``).
        timeout_s: HTTP timeout in seconds for the GCS metadata query.
        use_cache: Whether to read from and populate the process-level location cache.

    Returns:
        :class:`BucketLocation` on success, or ``None`` if lookup fails.
    """

    if not isinstance(bucket, str):
        return None
    norm_bucket = bucket.strip().lower()
    if not _BUCKET_NAME_RE.fullmatch(norm_bucket):
        return None

    if use_cache:
        with _LOCATION_CACHE_LOCK:
            cached = _LOCATION_CACHE.get(norm_bucket)
            if cached is not None:
                return cached

    with _bypass_region_guard():
        token: str | None = None
        try:
            if token_provider is not None:
                token = token_provider()
            else:
                token = _fetch_gce_metadata_token(timeout_s=min(timeout_s, _DEFAULT_METADATA_TIMEOUT_S))
        except Exception:
            token = None

        headers: dict[str, str] = {}
        if _valid_bearer_token(token):
            headers["Authorization"] = f"Bearer {token}"

        url = _GCS_BUCKET_API_TEMPLATE.format(bucket=urllib.parse.quote(norm_bucket, safe=""))
        try:
            if transport is not None:
                status, _resp_headers, payload = transport("GET", url, headers, None)
            else:
                status, _resp_headers, payload = _default_gcs_get("GET", url, headers, None, timeout_s=timeout_s)
        except Exception:
            return None

    if status != 200:
        return None

    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None

    raw_location = decoded.get("location")
    if not isinstance(raw_location, str) or not raw_location.strip():
        return None
    norm_location = raw_location.strip().lower()

    data_locations: tuple[str, ...] = ()
    custom_placement = decoded.get("customPlacementConfig")
    if isinstance(custom_placement, dict):
        raw_data_locations = custom_placement.get("dataLocations")
        if isinstance(raw_data_locations, (list, tuple)):
            data_locations = tuple(
                item.strip().lower() for item in raw_data_locations if isinstance(item, str) and item.strip()
            )

    raw_location_type = decoded.get("locationType")
    if isinstance(raw_location_type, str) and raw_location_type.strip():
        location_type = raw_location_type.strip().lower()
    else:
        location_type = _infer_location_type(norm_location, data_locations)

    resolved = BucketLocation(
        bucket=norm_bucket,
        location=norm_location,
        location_type=location_type,
        data_locations=data_locations,
    )
    if use_cache:
        with _LOCATION_CACHE_LOCK:
            _LOCATION_CACHE[norm_bucket] = resolved
    return resolved


def extract_gcs_uris(text: str) -> tuple[str, ...]:
    """Extract all unique ``gs://<bucket>[/path]`` URIs from ``text`` in appearance order."""

    if not isinstance(text, str) or "gs://" not in text.lower():
        return ()
    found: list[str] = []
    seen: set[str] = set()
    for match in _GCS_URI_RE.finditer(text):
        uri = match.group(0).rstrip(".:")
        if uri not in seen:
            seen.add(uri)
            found.append(uri)
    return tuple(found)


def extract_bucket_from_gcs_uri(uri: str) -> str | None:
    """Return the validated lowercase bucket name from a ``gs://`` URI, or ``None``."""

    if not isinstance(uri, str):
        return None
    cleaned = uri.strip()
    if not cleaned:
        return None
    parsed = urllib.parse.urlsplit(cleaned)
    if parsed.scheme.lower() != "gs":
        return None
    if parsed.username or parsed.password or ":" in parsed.netloc:
        return None
    bucket = parsed.netloc.strip().lower()
    if not _BUCKET_NAME_RE.fullmatch(bucket):
        return None
    return bucket


def _normalize_host(host: str | None) -> str:
    if not isinstance(host, str):
        return ""
    cleaned = host.strip().lower()
    if not cleaned:
        return ""
    if cleaned.startswith("["):
        end = cleaned.find("]")
        return cleaned[1:end] if end != -1 else cleaned
    return cleaned.split(":", 1)[0].rstrip(".")


def _is_bucket_location_metadata_query(path: str, query: str) -> bool:
    """Return whether ``path`` + ``query`` is a bucket-location-only metadata lookup."""

    if not query:
        return False
    match = re.fullmatch(r"/storage/v\d+/b/[^/?#]+/?", path)
    if match is None:
        return False
    params = urllib.parse.parse_qs(query, keep_blank_values=False)
    fields = params.get("fields")
    if not fields:
        return False
    tokens = {part.strip().lower() for value in fields for part in value.split(",") if part.strip()}
    return bool(tokens) and tokens <= {"location", "locationtype", "customplacementconfig"}


def extract_bucket_from_gcs_http_url(url: str, host: str | None = None) -> str | None:
    """Extract a GCS bucket name from a GCS JSON/XML HTTP URL or request path.

    Supports:
    - ``https://storage.googleapis.com/storage/v1/b/<bucket>/...``
    - ``https://storage.googleapis.com/upload/storage/v1/b/<bucket>/...``
    - ``https://storage.googleapis.com/download/storage/v1/b/<bucket>/...``
    - ``https://storage.googleapis.com/<bucket>/<object>``
    - ``https://<bucket>.storage.googleapis.com/<object>``
    - ``https://storage.cloud.google.com/<bucket>/<object>``
    - Relative paths paired with a GCS ``host`` (as passed to ``http.client.HTTPConnection.request``)

    Returns ``None`` for non-GCS hosts, and ignores bucket-location metadata
    queries while the guard's re-entrancy bypass is active.
    """

    if not isinstance(url, str):
        return None
    cleaned = url.strip()
    if not cleaned:
        return None

    parsed = urllib.parse.urlsplit(cleaned)
    if parsed.scheme.lower() in {"http", "https"}:
        effective_host = _normalize_host(parsed.hostname)
    else:
        effective_host = _normalize_host(host)
    if not effective_host:
        return None

    path = parsed.path or ""
    query = parsed.query or ""

    if effective_host.endswith(".storage.googleapis.com"):
        subdomain = effective_host[: -len(".storage.googleapis.com")]
        if _BUCKET_NAME_RE.fullmatch(subdomain):
            return subdomain
        return None

    if effective_host not in _GCS_API_HOSTS:
        return None

    if _is_bypassed() and _is_bucket_location_metadata_query(path, query):
        return None

    api_match = _GCS_JSON_API_BUCKET_RE.match(path)
    if api_match is not None:
        candidate = urllib.parse.unquote(api_match.group(1)).strip().lower()
        return candidate if _BUCKET_NAME_RE.fullmatch(candidate) else None

    if effective_host in _GCS_PATH_HOSTS:
        segments = [seg for seg in path.split("/") if seg]
        if segments:
            first = urllib.parse.unquote(segments[0]).strip().lower()
            if first not in _GCS_RESERVED_FIRST_SEGMENTS and _BUCKET_NAME_RE.fullmatch(first):
                return first

    return None


def check_gcs_uri_locality(
    uri: str,
    *,
    host_region: str,
    operation: str = "access",
    allowed_buckets: Iterable[str] = (),
    allow_multiregion: bool = True,
    location_resolver: Callable[[str], BucketLocation | None] | None = None,
    on_unknown: str = "allow",
    bucket_location: BucketLocation | None = None,
) -> LocalityCheckResult:
    """Evaluate whether accessing ``uri`` from ``host_region`` is local or cross-region.

    Args:
        uri: Target ``gs://`` URI or GCS HTTP URL.
        host_region: GCP region (or zone) of the host running the workload.
        operation: Short operation label for diagnostics (e.g. ``"urllib.get"``).
        allowed_buckets: Bucket names explicitly exempted from locality enforcement.
        allow_multiregion: Whether matching continental multi-regions are allowed.
        location_resolver: Optional bucket location lookup callable for tests or custom backends.
        on_unknown: Behavior when a bucket's location cannot be resolved (``"allow"`` or ``"ban"``).
        bucket_location: Optional pre-resolved :class:`BucketLocation` avoiding an API lookup.

    Returns:
        :class:`LocalityCheckResult` describing the allow/deny decision and reason.
    """

    norm_host = _normalize_region(host_region)
    bucket = extract_bucket_from_gcs_uri(uri) or extract_bucket_from_gcs_http_url(uri)
    if bucket is None:
        return LocalityCheckResult(
            allowed=True,
            host_region=norm_host,
            bucket="",
            bucket_location="",
            location_type="unknown",
            uri=uri,
            operation=operation,
            reason="not a GCS URI",
        )

    allowed_set = {item.strip().lower() for item in allowed_buckets if isinstance(item, str) and item.strip()}
    if bucket in allowed_set:
        return LocalityCheckResult(
            allowed=True,
            host_region=norm_host,
            bucket=bucket,
            bucket_location="allowlisted",
            location_type="allowlisted",
            uri=uri,
            operation=operation,
            reason=f"bucket {bucket!r} is explicitly allowlisted",
        )

    norm_on_unknown = on_unknown.strip().lower()
    if not norm_host:
        allow_unresolved = norm_on_unknown not in {"ban", "deny", "block", "reject"}
        return LocalityCheckResult(
            allowed=allow_unresolved,
            host_region="",
            bucket=bucket,
            bucket_location="unknown",
            location_type="unknown",
            uri=uri,
            operation=operation,
            reason="host region is unknown",
        )

    resolved = bucket_location
    if resolved is None:
        resolver = location_resolver or resolve_bucket_location
        with _bypass_region_guard():
            resolved = resolver(bucket)

    if resolved is None:
        allow_unresolved = norm_on_unknown not in {"ban", "deny", "block", "reject"}
        return LocalityCheckResult(
            allowed=allow_unresolved,
            host_region=norm_host,
            bucket=bucket,
            bucket_location="unknown",
            location_type="unknown",
            uri=uri,
            operation=operation,
            reason=f"bucket {bucket!r} location could not be resolved (on_unknown={norm_on_unknown})",
        )

    if is_location_local_to_region(norm_host, resolved, allow_multiregion=allow_multiregion):
        return LocalityCheckResult(
            allowed=True,
            host_region=norm_host,
            bucket=bucket,
            bucket_location=resolved.location,
            location_type=resolved.location_type,
            uri=uri,
            operation=operation,
            reason=f"bucket {bucket!r} ({resolved.location_type}:{resolved.location}) is local to {norm_host}",
        )

    return LocalityCheckResult(
        allowed=False,
        host_region=norm_host,
        bucket=bucket,
        bucket_location=resolved.location,
        location_type=resolved.location_type,
        uri=uri,
        operation=operation,
        reason=(
            f"host in {norm_host} attempted {operation} on {uri} "
            f"(bucket {bucket!r} location: {resolved.location}, type: {resolved.location_type})"
        ),
    )


def _sanitize_marker_field(value: str) -> str:
    return value.replace("\r", " ").replace("\n", " ").strip()


def _format_violation_marker(result: LocalityCheckResult) -> str:
    return (
        f"{CROSS_REGION_TRANSFER_BANNED_MARKER} "
        f"host_region={_sanitize_marker_field(result.host_region)} "
        f"bucket={_sanitize_marker_field(result.bucket)} "
        f"bucket_location={_sanitize_marker_field(result.bucket_location)} "
        f"operation={_sanitize_marker_field(result.operation)} "
        f"uri={_sanitize_marker_field(result.uri)}"
    )


def _emit_stderr_line(line: str) -> None:
    try:
        stream = sys.stderr
        if stream is None:
            return
        stream.write(f"{line}\n")
        stream.flush()
    except Exception:
        return


def _enforce_uri_with_config(
    uri: str,
    *,
    operation: str,
    config: _GuardConfig,
    bucket_location: BucketLocation | None = None,
) -> LocalityCheckResult:
    result = check_gcs_uri_locality(
        uri,
        host_region=config.host_region,
        operation=operation,
        allowed_buckets=config.allowed_buckets,
        allow_multiregion=config.allow_multiregion,
        location_resolver=config.location_resolver,
        on_unknown=config.on_unknown,
        bucket_location=bucket_location,
    )
    if result.allowed:
        return result

    if config.mode == "warn":
        _emit_stderr_line(
            "eyvan-cross-region-transfer-warning "
            f"host_region={_sanitize_marker_field(result.host_region)} "
            f"bucket={_sanitize_marker_field(result.bucket)} "
            f"bucket_location={_sanitize_marker_field(result.bucket_location)} "
            f"operation={_sanitize_marker_field(result.operation)} "
            f"uri={_sanitize_marker_field(result.uri)}"
        )
        if config.on_violation is not None:
            config.on_violation(result)
        return result

    _emit_stderr_line(_format_violation_marker(result))
    if config.on_violation is not None:
        config.on_violation(result)
    if config.kill_on_violation:
        killer = config.terminate if config.terminate is not None else terminate_on_violation
        killer(CROSS_REGION_TRANSFER_BANNED_EXIT_CODE)
    raise CrossRegionTransferError(result=result)


def is_cross_region_guard_active() -> bool:
    """Return whether a cross-region GCS transfer guard is currently armed."""

    return _ACTIVE_GUARD is not None


def enforce_gcs_access(
    uri: str,
    *,
    operation: str = "access",
    bucket_location: BucketLocation | None = None,
) -> LocalityCheckResult | None:
    """Enforce the currently active cross-region guard on ``uri``, if armed.

    Returns ``None`` immediately when no guard is active or the re-entrancy
    bypass flag is set.
    """

    config = _ACTIVE_GUARD
    if config is None or _is_bypassed():
        return None
    return _enforce_uri_with_config(
        uri,
        operation=operation,
        config=config,
        bucket_location=bucket_location,
    )


def _extract_gcs_targets_from_spec(spec: tp.Any, *, _depth: int = 0, _seen: set[int] | None = None) -> list[str]:
    """Recursively collect GCS URIs from a TensorStore spec (dict, JSON string, or Spec)."""

    if _depth > 32 or spec is None:
        return []
    if _seen is None:
        _seen = set()

    if isinstance(spec, str):
        uris = list(extract_gcs_uris(spec))
        stripped = spec.lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                decoded = None
            if decoded is not None:
                uris.extend(_extract_gcs_targets_from_spec(decoded, _depth=_depth + 1, _seen=_seen))
        return list(dict.fromkeys(uris))

    obj_id = id(spec)
    if obj_id in _seen:
        return []
    _seen.add(obj_id)

    uris: list[str] = []
    if isinstance(spec, Mapping):
        driver = spec.get("driver")
        bucket = spec.get("bucket")
        if isinstance(driver, str) and driver.strip().lower() in {"gcs", "gcs_grpc"} and isinstance(bucket, str):
            norm_bucket = bucket.strip().lower()
            if _BUCKET_NAME_RE.fullmatch(norm_bucket):
                raw_path = spec.get("path")
                rel = raw_path.lstrip("/") if isinstance(raw_path, str) else ""
                uris.append(f"gs://{norm_bucket}/{rel}" if rel else f"gs://{norm_bucket}")
        for value in spec.values():
            uris.extend(_extract_gcs_targets_from_spec(value, _depth=_depth + 1, _seen=_seen))
        return list(dict.fromkeys(uris))

    if isinstance(spec, (list, tuple)):
        for item in spec:
            uris.extend(_extract_gcs_targets_from_spec(item, _depth=_depth + 1, _seen=_seen))
        return list(dict.fromkeys(uris))

    to_json = getattr(spec, "to_json", None)
    if callable(to_json):
        try:
            json_value = to_json()
        except Exception:
            json_value = None
        if json_value is not None:
            uris.extend(_extract_gcs_targets_from_spec(json_value, _depth=_depth + 1, _seen=_seen))
    return list(dict.fromkeys(uris))


def _path_to_gcs_uri(raw: tp.Any, *, is_api_call: bool = False) -> str | None:
    if not isinstance(raw, str):
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    lowered = cleaned.lower()
    if lowered.startswith("gcs://"):
        cleaned = "gs://" + cleaned[6:]
        lowered = cleaned.lower()
    if lowered.startswith("gs://"):
        return cleaned if extract_bucket_from_gcs_uri(cleaned) is not None else None
    if lowered.startswith(("http://", "https://")):
        return cleaned if extract_bucket_from_gcs_http_url(cleaned) is not None else None

    trimmed = cleaned.lstrip("/")
    if is_api_call:
        if trimmed.startswith("b/"):
            rest = trimmed[2:]
            bucket = urllib.parse.unquote(rest.split("/", 1)[0]).strip().lower()
            if _BUCKET_NAME_RE.fullmatch(bucket):
                return f"gs://{bucket}"
        bucket = extract_bucket_from_gcs_http_url("/" + trimmed, host="storage.googleapis.com")
        if bucket is not None:
            return f"gs://{bucket}"
        return None

    bucket = trimmed.split("/", 1)[0].strip().lower()
    if _BUCKET_NAME_RE.fullmatch(bucket):
        return f"gs://{trimmed}"
    return None


def _collect_gcsfs_uris(raw: tp.Any, *, is_api_call: bool = False) -> list[str]:
    if isinstance(raw, str):
        uri = _path_to_gcs_uri(raw, is_api_call=is_api_call)
        return [uri] if uri is not None else []
    if isinstance(raw, Mapping):
        uris: list[str] = []
        for key in raw:
            uri = _path_to_gcs_uri(key, is_api_call=is_api_call)
            if uri is not None:
                uris.append(uri)
        return list(dict.fromkeys(uris))
    if isinstance(raw, (list, tuple)):
        uris = []
        for item in raw:
            uri = _path_to_gcs_uri(item, is_api_call=is_api_call)
            if uri is not None:
                uris.append(uri)
        return list(dict.fromkeys(uris))
    return []


def _extract_gcsfs_uris(method_name: str, args: tuple[tp.Any, ...], kwargs: Mapping[str, tp.Any]) -> list[str]:
    if method_name == "_call":
        target = args[1] if len(args) >= 2 else kwargs.get("path")
        return _collect_gcsfs_uris(target, is_api_call=True)
    if method_name == "put":
        target = args[1] if len(args) >= 2 else kwargs.get("rpath")
        return _collect_gcsfs_uris(target)
    if method_name == "get":
        target = args[0] if args else kwargs.get("rpath")
        return _collect_gcsfs_uris(target)
    target = args[0] if args else kwargs.get("path")
    return _collect_gcsfs_uris(target)


def _extract_subprocess_gcs_uris(args: tp.Any, executable: tp.Any = None) -> tuple[str, tuple[str, ...]]:
    """Inspect a ``subprocess.Popen`` command for ``gcloud storage`` / ``gsutil`` GCS URIs."""

    if isinstance(args, str):
        if "gs://" in args.lower() and ("gsutil" in args or ("gcloud" in args and "storage" in args)):
            return "subprocess.shell", extract_gcs_uris(args)
        return "", ()

    if isinstance(args, (bytes, bytearray, os.PathLike)):
        return "", ()

    if not isinstance(args, (list, tuple)) or not args:
        return "", ()

    tokens: list[str] = []
    for item in args:
        if isinstance(item, str):
            tokens.append(item)
        elif isinstance(item, os.PathLike):
            fspath = os.fspath(item)
            if isinstance(fspath, str):
                tokens.append(fspath)

    if not tokens:
        return "", ()

    exec_name = os.fspath(executable) if isinstance(executable, (str, os.PathLike)) else tokens[0]
    prog = os.path.basename(str(exec_name)).lower()
    if prog in {"gsutil", "gsutil.py"}:
        joined = " ".join(tokens[1:])
        return "subprocess.gsutil", extract_gcs_uris(joined)
    if prog in {"gcloud", "gcloud.py"}:
        head = [tok.lower() for tok in tokens[1:5]]
        if "storage" in head:
            joined = " ".join(tokens[1:])
            return "subprocess.gcloud", extract_gcs_uris(joined)
        return "", ()
    if prog in _SHELL_EXECUTABLES:
        for idx, tok in enumerate(tokens[1:], start=1):
            if tok.startswith("-") and not tok.startswith("--") and "c" in tok[1:]:
                if idx + 1 < len(tokens):
                    script = tokens[idx + 1]
                    if "gs://" in script.lower() and (
                        "gsutil" in script or ("gcloud" in script and "storage" in script)
                    ):
                        return f"subprocess.{prog}", extract_gcs_uris(script)
                break
    return "", ()


def _record_and_patch_attr(target: tp.Any, attr: str, replacement: tp.Any) -> None:
    key = (id(target), attr)
    if key in _PATCHED_TARGETS:
        return
    target_dict = getattr(target, "__dict__", {})
    had_own = attr in target_dict
    original = target_dict.get(attr) if had_own else None
    setattr(target, attr, replacement)
    _PATCHED_ATTRS.append((target, attr, had_own, original))
    _PATCHED_TARGETS.add(key)


def _wrap_tensorstore_callable(original: tp.Any, operation: str, *, is_classmethod: bool = False) -> tp.Any:
    if inspect.iscoroutinefunction(original):

        async def _async_ts_wrapper(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            config = _ACTIVE_GUARD
            if config is not None and not _is_bypassed():
                spec_idx = 1 if is_classmethod else 0
                spec = args[spec_idx] if len(args) > spec_idx else kwargs.get("spec")
                for uri in _extract_gcs_targets_from_spec(spec):
                    _enforce_uri_with_config(uri, operation=operation, config=config)
            return await original(*args, **kwargs)

        return _async_ts_wrapper

    def _sync_ts_wrapper(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        config = _ACTIVE_GUARD
        if config is not None and not _is_bypassed():
            spec_idx = 1 if is_classmethod else 0
            spec = args[spec_idx] if len(args) > spec_idx else kwargs.get("spec")
            for uri in _extract_gcs_targets_from_spec(spec):
                _enforce_uri_with_config(uri, operation=operation, config=config)
        return original(*args, **kwargs)

    return _sync_ts_wrapper


def _patch_tensorstore_module(module: tp.Any) -> None:
    if hasattr(module, "open") and callable(module.open):
        orig_open = module.open
        _record_and_patch_attr(module, "open", _wrap_tensorstore_callable(orig_open, "tensorstore.open"))

    kvstore = getattr(module, "KvStore", None)
    if kvstore is not None and hasattr(kvstore, "open"):
        raw = inspect.getattr_static(kvstore, "open")
        if isinstance(raw, staticmethod):
            wrapped = staticmethod(_wrap_tensorstore_callable(raw.__func__, "tensorstore.KvStore.open"))
        elif isinstance(raw, classmethod):
            wrapped = classmethod(
                _wrap_tensorstore_callable(raw.__func__, "tensorstore.KvStore.open", is_classmethod=True)
            )
        else:
            wrapped = staticmethod(_wrap_tensorstore_callable(raw, "tensorstore.KvStore.open"))
        _record_and_patch_attr(kvstore, "open", wrapped)


def _wrap_gcsfs_method(original: tp.Any, method_name: str) -> tp.Any:
    operation = f"gcsfs.{method_name}"
    if inspect.iscoroutinefunction(original):

        async def _async_gcsfs_wrapper(self: tp.Any, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            config = _ACTIVE_GUARD
            checked_buckets: list[str] = []
            if config is not None and not _is_bypassed():
                for uri in _extract_gcsfs_uris(method_name, args, kwargs):
                    bucket = extract_bucket_from_gcs_uri(uri) or extract_bucket_from_gcs_http_url(uri)
                    if bucket is not None and not _is_bucket_checked_in_frame(bucket):
                        _enforce_uri_with_config(uri, operation=operation, config=config)
                        checked_buckets.append(bucket)
            if not checked_buckets:
                return await original(self, *args, **kwargs)
            with _mark_bucket_checked(checked_buckets[0]):
                return await original(self, *args, **kwargs)

        return _async_gcsfs_wrapper

    def _sync_gcsfs_wrapper(self: tp.Any, *args: tp.Any, **kwargs: tp.Any) -> tp.Any:
        config = _ACTIVE_GUARD
        checked_buckets: list[str] = []
        if config is not None and not _is_bypassed():
            for uri in _extract_gcsfs_uris(method_name, args, kwargs):
                bucket = extract_bucket_from_gcs_uri(uri) or extract_bucket_from_gcs_http_url(uri)
                if bucket is not None and not _is_bucket_checked_in_frame(bucket):
                    _enforce_uri_with_config(uri, operation=operation, config=config)
                    checked_buckets.append(bucket)
        if not checked_buckets:
            return original(self, *args, **kwargs)
        with _mark_bucket_checked(checked_buckets[0]):
            return original(self, *args, **kwargs)

    return _sync_gcsfs_wrapper


def _patch_gcsfs_module(module: tp.Any) -> None:
    fs_cls = getattr(module, "GCSFileSystem", None)
    if fs_cls is None:
        return
    for method_name in _GCSFS_HOOKED_METHODS:
        if hasattr(fs_cls, method_name):
            orig = getattr(fs_cls, method_name)
            if callable(orig):
                _record_and_patch_attr(fs_cls, method_name, _wrap_gcsfs_method(orig, method_name))


def _patch_watched_module_if_active(fullname: str, module: tp.Any) -> None:
    with _GUARD_LOCK:
        if _ACTIVE_GUARD is None:
            return
        if fullname == "tensorstore":
            _patch_tensorstore_module(module)
        elif fullname == "gcsfs":
            _patch_gcsfs_module(module)


class _PatchingLoader(importlib.abc.Loader):
    """Wraps a real module loader to install region-guard hooks right after module exec."""

    def __init__(self, wrapped: importlib.abc.Loader, fullname: str) -> None:
        self._wrapped = wrapped
        self._fullname = fullname

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> tp.Any:
        create = getattr(self._wrapped, "create_module", None)
        if callable(create):
            return create(spec)
        return None

    def exec_module(self, module: tp.Any) -> None:
        exec_fn = getattr(self._wrapped, "exec_module", None)
        if callable(exec_fn):
            exec_fn(module)
        else:
            load_fn = getattr(self._wrapped, "load_module", None)
            if callable(load_fn):
                module = load_fn(self._fullname)
        _patch_watched_module_if_active(self._fullname, module)


class _LazyModulePatchFinder(importlib.abc.MetaPathFinder):
    """Non-intrusive ``sys.meta_path`` hook that patches ``tensorstore``/``gcsfs`` on import."""

    def __init__(self) -> None:
        self._local = threading.local()

    def find_spec(
        self,
        fullname: str,
        path: tp.Sequence[str] | None,
        target: tp.Any = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if fullname not in _WATCHED_MODULES or _ACTIVE_GUARD is None:
            return None
        if getattr(self._local, "reentering", False):
            return None
        self._local.reentering = True
        try:
            for finder in list(sys.meta_path):
                if finder is self:
                    continue
                find_spec = getattr(finder, "find_spec", None)
                if not callable(find_spec):
                    continue
                spec = find_spec(fullname, path, target)
                if spec is not None and spec.loader is not None:
                    spec.loader = _PatchingLoader(spec.loader, fullname)
                    return spec
            return None
        finally:
            self._local.reentering = False


def _install_hooks_locked() -> None:
    global _META_PATH_FINDER

    if "urllib_open" not in _STDLIB_ORIGINALS:
        orig_urllib_open = urllib.request.OpenerDirector.open
        _STDLIB_ORIGINALS["urllib_open"] = orig_urllib_open

        def _guarded_urllib_open(
            self: urllib.request.OpenerDirector,
            fullurl: str | urllib.request.Request,
            *args: tp.Any,
            **kwargs: tp.Any,
        ) -> tp.Any:
            config = _ACTIVE_GUARD
            checked_bucket: str | None = None
            if config is not None and not _is_bypassed():
                if isinstance(fullurl, urllib.request.Request):
                    url = fullurl.full_url
                    method = fullurl.get_method().lower()
                else:
                    url = str(fullurl)
                    data_arg = args[0] if args else kwargs.get("data")
                    method = "post" if data_arg is not None else "get"
                bucket = extract_bucket_from_gcs_http_url(url)
                if bucket is not None and not _is_bucket_checked_in_frame(bucket):
                    _enforce_uri_with_config(url, operation=f"urllib.{method}", config=config)
                    checked_bucket = bucket
            if checked_bucket is None:
                return orig_urllib_open(self, fullurl, *args, **kwargs)
            with _mark_bucket_checked(checked_bucket):
                return orig_urllib_open(self, fullurl, *args, **kwargs)

        urllib.request.OpenerDirector.open = _guarded_urllib_open  # type: ignore[method-assign]

    if "http_request" not in _STDLIB_ORIGINALS:
        orig_http_request = http.client.HTTPConnection.request
        _STDLIB_ORIGINALS["http_request"] = orig_http_request

        def _guarded_http_request(
            self: http.client.HTTPConnection,
            method: str,
            url: str,
            *args: tp.Any,
            **kwargs: tp.Any,
        ) -> tp.Any:
            config = _ACTIVE_GUARD
            if config is not None and not _is_bypassed():
                headers_arg = kwargs.get("headers") if "headers" in kwargs else (args[1] if len(args) >= 2 else None)
                host = getattr(self, "host", None)
                if not host and isinstance(headers_arg, Mapping):
                    host = _header_get_ci(headers_arg, "Host")
                bucket = extract_bucket_from_gcs_http_url(url, host=host)
                if bucket is not None and not _is_bucket_checked_in_frame(bucket):
                    scheme = "https" if isinstance(self, http.client.HTTPSConnection) else "http"
                    full_uri = url if url.startswith(("http://", "https://")) else f"{scheme}://{host}{url}"
                    _enforce_uri_with_config(full_uri, operation=f"http.{str(method).lower()}", config=config)
            return orig_http_request(self, method, url, *args, **kwargs)

        http.client.HTTPConnection.request = _guarded_http_request  # type: ignore[method-assign]

    if "popen_init" not in _STDLIB_ORIGINALS:
        orig_popen_init = subprocess.Popen.__init__
        _STDLIB_ORIGINALS["popen_init"] = orig_popen_init

        def _guarded_popen_init(
            self: subprocess.Popen[tp.Any],
            args: tp.Any,
            *pargs: tp.Any,
            **kwargs: tp.Any,
        ) -> None:
            config = _ACTIVE_GUARD
            if config is not None and not _is_bypassed():
                operation, uris = _extract_subprocess_gcs_uris(args, executable=kwargs.get("executable"))
                for uri in uris:
                    _enforce_uri_with_config(uri, operation=operation, config=config)
            orig_popen_init(self, args, *pargs, **kwargs)

        subprocess.Popen.__init__ = _guarded_popen_init  # type: ignore[method-assign]

    for mod_name in _WATCHED_MODULES:
        loaded = sys.modules.get(mod_name)
        if loaded is not None:
            if mod_name == "tensorstore":
                _patch_tensorstore_module(loaded)
            elif mod_name == "gcsfs":
                _patch_gcsfs_module(loaded)

    if _META_PATH_FINDER is None:
        _META_PATH_FINDER = _LazyModulePatchFinder()
        sys.meta_path.insert(0, _META_PATH_FINDER)


def _uninstall_hooks_locked() -> None:
    global _META_PATH_FINDER

    if "urllib_open" in _STDLIB_ORIGINALS:
        urllib.request.OpenerDirector.open = _STDLIB_ORIGINALS.pop("urllib_open")  # type: ignore[method-assign]
    if "http_request" in _STDLIB_ORIGINALS:
        http.client.HTTPConnection.request = _STDLIB_ORIGINALS.pop("http_request")  # type: ignore[method-assign]
    if "popen_init" in _STDLIB_ORIGINALS:
        subprocess.Popen.__init__ = _STDLIB_ORIGINALS.pop("popen_init")  # type: ignore[method-assign]

    for target, attr, had_own, original in reversed(_PATCHED_ATTRS):
        if had_own:
            setattr(target, attr, original)
        else:
            try:
                delattr(target, attr)
            except AttributeError:
                pass
    _PATCHED_ATTRS.clear()
    _PATCHED_TARGETS.clear()

    if _META_PATH_FINDER is not None:
        try:
            sys.meta_path.remove(_META_PATH_FINDER)
        except ValueError:
            pass
        _META_PATH_FINDER = None


def enable_cross_region_guard(
    *,
    host_region: str,
    mode: str = "ban",
    allowed_buckets: Iterable[str] = (),
    allow_multiregion: bool = True,
    kill_on_violation: bool = False,
    location_resolver: Callable[[str], BucketLocation | None] | None = None,
    on_unknown: str = "allow",
    on_violation: Callable[[LocalityCheckResult], None] | None = None,
    terminate: Callable[[int], tp.NoReturn] | None = None,
) -> bool:
    """Install live runtime hooks enforcing cross-region GCS transfer locality.

    Args:
        host_region: GCP region (or zone) of the host running this process.
        mode: Enforcement policy — ``"allow"`` (no-op), ``"warn"`` (log warning
            to stderr), or ``"ban"`` (emit marker and raise or kill process).
        allowed_buckets: Bucket names exempted from locality checks.
        allow_multiregion: Whether matching continental multi-regions are permitted.
        kill_on_violation: When ``True`` in ``"ban"`` mode, terminate the process
            immediately via ``terminate_on_violation(95)`` so ``except Exception``
            cannot swallow the ban.
        location_resolver: Optional bucket location lookup callable (defaults to
            :func:`resolve_bucket_location`).
        on_unknown: Policy when a bucket's location cannot be resolved (``"allow"`` or ``"ban"``).
        on_violation: Optional callback invoked with :class:`LocalityCheckResult` on violation.
        terminate: Optional process termination callback overriding :data:`terminate_on_violation`.

    Returns:
        ``True`` if hooks are armed, ``False`` if ``mode == "allow"`` or ``host_region`` is empty.
    """

    global _ACTIVE_GUARD
    norm_mode = mode.strip().lower() if isinstance(mode, str) else ""
    if norm_mode not in _VALID_POLICIES:
        raise ValueError(f"unsupported cross-region transfer policy {mode!r}; expected 'allow', 'warn', or 'ban'")
    norm_host = _normalize_region(host_region)

    with _GUARD_LOCK:
        if norm_mode == "allow" or not norm_host:
            _ACTIVE_GUARD = None
            _uninstall_hooks_locked()
            return False

        _ACTIVE_GUARD = _GuardConfig(
            host_region=norm_host,
            mode=norm_mode,
            allowed_buckets=frozenset(
                item.strip().lower() for item in allowed_buckets if isinstance(item, str) and item.strip()
            ),
            allow_multiregion=bool(allow_multiregion),
            kill_on_violation=bool(kill_on_violation),
            location_resolver=location_resolver,
            on_unknown=on_unknown.strip().lower() if isinstance(on_unknown, str) else "allow",
            on_violation=on_violation,
            terminate=terminate,
        )
        _install_hooks_locked()
        return True


def disable_cross_region_guard() -> None:
    """Disarm the cross-region GCS transfer guard and restore every patched hook.

    Safe to call multiple times or when no guard is armed.
    """

    global _ACTIVE_GUARD
    with _GUARD_LOCK:
        _ACTIVE_GUARD = None
        _uninstall_hooks_locked()


@contextmanager
def cross_region_guard(
    *,
    host_region: str,
    mode: str = "ban",
    allowed_buckets: Iterable[str] = (),
    allow_multiregion: bool = True,
    kill_on_violation: bool = False,
    location_resolver: Callable[[str], BucketLocation | None] | None = None,
    on_unknown: str = "allow",
    on_violation: Callable[[LocalityCheckResult], None] | None = None,
    terminate: Callable[[int], tp.NoReturn] | None = None,
) -> tp.Iterator[bool]:
    """Temporarily arm the cross-region GCS transfer guard within a ``with`` block."""

    global _ACTIVE_GUARD
    with _GUARD_LOCK:
        previous = _ACTIVE_GUARD
    armed = enable_cross_region_guard(
        host_region=host_region,
        mode=mode,
        allowed_buckets=allowed_buckets,
        allow_multiregion=allow_multiregion,
        kill_on_violation=kill_on_violation,
        location_resolver=location_resolver,
        on_unknown=on_unknown,
        on_violation=on_violation,
        terminate=terminate,
    )
    try:
        yield armed
    finally:
        with _GUARD_LOCK:
            if previous is None:
                _ACTIVE_GUARD = None
                _uninstall_hooks_locked()
            else:
                _ACTIVE_GUARD = previous
                _install_hooks_locked()


def arm_cross_region_guard_from_environment(
    environment: Mapping[str, str] | None = None,
    *,
    location_resolver: Callable[[str], BucketLocation | None] | None = None,
    on_unknown: str = "allow",
    on_violation: Callable[[LocalityCheckResult], None] | None = None,
    terminate: Callable[[int], tp.NoReturn] | None = None,
) -> bool:
    """Arm the cross-region GCS transfer guard from Eyvan's runtime environment variables.

    Reads ``EYVAN_CROSS_REGION_POLICY``, ``EYVAN_HOST_REGION``,
    ``EYVAN_CROSS_REGION_ALLOWED_BUCKETS``, ``EYVAN_CROSS_REGION_ALLOW_MULTIREGION``,
    and ``EYVAN_CROSS_REGION_KILL``.

    Args:
        environment: Environment mapping to read (defaults to ``os.environ``).
        location_resolver: Optional bucket location resolver override for tests.
        on_unknown: Policy when a bucket location cannot be resolved.
        on_violation: Optional violation callback.
        terminate: Optional process termination callback override.

    Returns:
        ``True`` if the guard was armed, ``False`` if inactive/unconfigured.
    """

    env = os.environ if environment is None else environment
    raw_policy = (env.get(CROSS_REGION_POLICY_ENV) or "").strip().lower()
    if not raw_policy or raw_policy == "allow":
        return False
    if raw_policy not in _VALID_POLICIES:
        raise ValueError(
            f"{CROSS_REGION_POLICY_ENV} must be 'allow', 'warn', or 'ban', got {env.get(CROSS_REGION_POLICY_ENV)!r}"
        )

    host_region = (env.get(HOST_REGION_ENV) or "").strip()
    if not host_region:
        return False

    raw_allowed = env.get(CROSS_REGION_ALLOWED_BUCKETS_ENV, "")
    allowed_buckets = tuple(item.strip() for item in raw_allowed.split(",") if item.strip())

    raw_multiregion = (env.get(CROSS_REGION_ALLOW_MULTIREGION_ENV) or "1").strip().lower()
    allow_multiregion = raw_multiregion not in {"0", "false", "no", "off"}

    raw_kill = (env.get(CROSS_REGION_KILL_ENV) or "0").strip().lower()
    kill_on_violation = raw_kill in {"1", "true", "yes", "on"}

    return enable_cross_region_guard(
        host_region=host_region,
        mode=raw_policy,
        allowed_buckets=allowed_buckets,
        allow_multiregion=allow_multiregion,
        kill_on_violation=kill_on_violation,
        location_resolver=location_resolver,
        on_unknown=on_unknown,
        on_violation=on_violation,
        terminate=terminate,
    )


__all__ = (
    "CROSS_REGION_ALLOWED_BUCKETS_ENV",
    "CROSS_REGION_ALLOW_MULTIREGION_ENV",
    "CROSS_REGION_KILL_ENV",
    "CROSS_REGION_POLICY_ENV",
    "CROSS_REGION_TRANSFER_BANNED_EXIT_CODE",
    "CROSS_REGION_TRANSFER_BANNED_MARKER",
    "CROSS_REGION_TRANSFER_BANNED_REASON",
    "HOST_REGION_ENV",
    "BucketLocation",
    "CrossRegionTransferError",
    "LocalityCheckResult",
    "arm_cross_region_guard_from_environment",
    "check_gcs_uri_locality",
    "clear_bucket_location_cache",
    "cross_region_guard",
    "disable_cross_region_guard",
    "enable_cross_region_guard",
    "enforce_gcs_access",
    "extract_bucket_from_gcs_http_url",
    "extract_bucket_from_gcs_uri",
    "extract_gcs_uris",
    "is_cross_region_guard_active",
    "is_location_local_to_region",
    "prime_bucket_location_cache",
    "resolve_bucket_location",
    "terminate_on_violation",
)
