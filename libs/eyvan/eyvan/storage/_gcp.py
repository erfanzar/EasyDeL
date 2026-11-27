# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""GCP/TPU environment detection and stdlib GCS JSON API bucket backend.

Standard library only. Zero new environment variables. Never leaks OAuth tokens
or raw HTTP response bodies in exception messages, and refuses HTTP redirects
on authenticated requests.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import threading
import time
import typing as tp
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import HTTPRedirectHandler, Request, build_opener

from eyvan._internal.subprocess_env import provider_cli_env
from eyvan.storage._config import (
    BucketNameConflictError,
    StorageAuthError,
    StorageBackendError,
    StorageConfigError,
    StorageConfigStore,
)
from eyvan.storage._naming import (
    validate_bucket_name,
    validate_location,
    validate_project_id,
    validate_region,
    validate_zone,
    zone_to_region,
)

#: HTTP transport callable: ``(method, url, headers, body) -> (status, response_headers, response_bytes)``.
Transport = tp.Callable[[str, str, dict[str, str], bytes | None], tuple[int, tp.Mapping[str, str], bytes]]
TokenProvider = tp.Callable[[], str]
MetadataFetcher = tp.Callable[[str, float], tuple[int, tp.Mapping[str, str], bytes] | None]
GcloudRunner = tp.Callable[[tuple[str, ...], float], tuple[int, str]]

_METADATA_IP_ROOT = "http://169.254.169.254/computeMetadata/v1"
_METADATA_HOST_ROOT = "http://metadata.google.internal/computeMetadata/v1"
_GCS_API_ROOT = "https://storage.googleapis.com/storage/v1"
_DEFAULT_METADATA_PROBE_TIMEOUT_S = 0.2
_DEFAULT_METADATA_READ_TIMEOUT_S = 1.5
_DEFAULT_GCLOUD_TIMEOUT_S = 2.0
_DEFAULT_GCLOUD_TOKEN_TIMEOUT_S = 15.0
_DEFAULT_API_TIMEOUT_S = 30.0
_MAX_API_RESPONSE_BYTES = 4 * 1024 * 1024
_MAX_LOCAL_FILE_BYTES = 256 * 1024
_GCLOUD_TOKEN_TTL_S = 300.0
_VALID_STORAGE_CLASSES = frozenset({"STANDARD", "NEARLINE", "COLDLINE", "ARCHIVE"})
_YAML_PROJECT_RE = re.compile(r"^\s*project\s*:\s*(?P<val>[^#\r\n]+?)\s*(?:#.*)?$")

_CONTEXT_CACHE_LOCK = threading.Lock()
_CONTEXT_CACHE: dict[tuple[str, str, bool, int | None, int | None, int | None], GcpContext] = {}
_METADATA_PROBE_DONE = False
_METADATA_PROBE_VALUE: GcpContext | None = None


@dataclass(frozen=True, slots=True)
class GcpContext:
    """Detected GCP/TPU runtime environment context.

    Attributes:
        on_gcp: Whether this process is running on a GCE/Cloud TPU host with a
            reachable metadata server.
        project: Detected GCP project ID, if any.
        zone: Detected GCP zone (e.g. ``us-central2-b``), if any.
        region: Detected or derived GCP region (e.g. ``us-central2``), if any.
        accelerator_type: Cloud TPU/accelerator type from instance metadata
            (e.g. ``v6e-8``), if available.
        source: Primary source that populated the context.
    """

    on_gcp: bool
    project: str | None = None
    zone: str | None = None
    region: str | None = None
    accelerator_type: str | None = None
    source: str = "none"


@dataclass(frozen=True, slots=True)
class BucketInfo:
    """Metadata describing one GCS bucket returned by a :class:`StorageBackend`."""

    name: str
    location: str
    project_number: str | None = None
    storage_class: str = "STANDARD"
    uniform_access: bool = True
    created: bool = False


@tp.runtime_checkable
class StorageBackend(tp.Protocol):
    """Pluggable bucket control backend used by ``eyvan.storage``."""

    def bucket_exists(self, bucket: str) -> bool:
        """Return whether ``bucket`` exists and is accessible."""

        ...

    def get_bucket(self, bucket: str) -> BucketInfo | None:
        """Return :class:`BucketInfo` if ``bucket`` exists, ``None`` if absent (404).

        Raises:
            BucketNameConflictError: If the bucket exists in another project or
                access is denied (HTTP 403).
            StorageBackendError: On any other transport or API error.
        """

        ...

    def create_bucket(
        self,
        bucket: str,
        *,
        project: str,
        location: str,
        storage_class: str = "STANDARD",
        uniform_access: bool = True,
    ) -> BucketInfo:
        """Create a GCS bucket in ``project`` and ``location``.

        Raises:
            BucketNameConflictError: If ``bucket`` is globally taken or creation is
                denied (HTTP 409/403).
            StorageBackendError: On any other transport or API error.
        """

        ...


class _NoRedirect(HTTPRedirectHandler):
    """Refuse redirects rather than forwarding bearer tokens to another host."""

    def redirect_request(
        self,
        req: Request,
        fp: tp.Any,
        code: int,
        msg: str,
        headers: tp.Any,
        newurl: str,
    ) -> None:
        return None


def _read_bounded(handle: tp.BinaryIO, limit: int = _MAX_API_RESPONSE_BYTES) -> bytes:
    output = bytearray()
    while chunk := handle.read(min(65536, limit - len(output) + 1)):
        output.extend(chunk)
        if len(output) > limit:
            raise StorageBackendError("GCS API response exceeded size limit")
    return bytes(output)


def _urllib_transport(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes | None,
    *,
    timeout_s: float = _DEFAULT_API_TIMEOUT_S,
) -> tuple[int, tp.Mapping[str, str], bytes]:
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with build_opener(_NoRedirect()).open(request, timeout=timeout_s) as response:
            return response.status, dict(response.headers.items()), _read_bounded(response)
    except HTTPError as error:
        with error:
            return error.code, dict(error.headers.items()), _read_bounded(error)


def _default_metadata_fetcher(url: str, timeout_s: float) -> tuple[int, tp.Mapping[str, str], bytes] | None:
    request = Request(url, headers={"Metadata-Flavor": "Google"}, method="GET")
    try:
        with build_opener(_NoRedirect()).open(request, timeout=timeout_s) as response:
            return response.status, dict(response.headers.items()), _read_bounded(response, limit=65536)
    except HTTPError as error:
        with error:
            return error.code, dict(error.headers.items()), _read_bounded(error, limit=65536)
    except Exception:
        return None


def _header_get_ci(headers: tp.Mapping[str, str], name: str) -> str | None:
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target:
            return value
    return None


def _is_google_metadata_response(headers: tp.Mapping[str, str]) -> bool:
    flavor = _header_get_ci(headers, "Metadata-Flavor")
    return flavor is not None and flavor.strip().lower() == "google"


def _default_gcloud_runner(args: tuple[str, ...], timeout_s: float) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            ("gcloud", *args),
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=provider_cli_env(),
        )
    except (OSError, subprocess.SubprocessError):
        return -1, ""
    return completed.returncode, completed.stdout.strip()


def _probe_gce_metadata(
    *,
    metadata_fetcher: MetadataFetcher | None = None,
    probe_timeout_s: float = _DEFAULT_METADATA_PROBE_TIMEOUT_S,
    read_timeout_s: float = _DEFAULT_METADATA_READ_TIMEOUT_S,
) -> GcpContext | None:
    """Fast, bounded GCE/TPU metadata probe; returns ``None`` off-GCP."""

    fetcher = metadata_fetcher or _default_metadata_fetcher
    base_root: str | None = None
    project_raw: str | None = None

    for idx, root in enumerate((_METADATA_IP_ROOT, _METADATA_HOST_ROOT)):
        started = time.monotonic()
        result = fetcher(f"{root}/project/project-id", probe_timeout_s)
        elapsed = time.monotonic() - started
        if result is not None:
            status, headers, body = result
            if status == 200 and _is_google_metadata_response(headers):
                base_root = root
                project_raw = body.decode("utf-8", errors="replace").strip()
                break
        # If the link-local IP already timed out, don't burn a second timeout on DNS.
        if idx == 0 and elapsed >= probe_timeout_s * 0.6:
            return None

    if base_root is None:
        return None

    project = _safe_project_id(project_raw)
    zone: str | None = None
    region: str | None = None
    accelerator_type: str | None = None

    zone_resp = fetcher(f"{base_root}/instance/zone", read_timeout_s)
    if zone_resp is not None and zone_resp[0] == 200 and _is_google_metadata_response(zone_resp[1]):
        raw_zone = zone_resp[2].decode("utf-8", errors="replace").strip()
        leaf = raw_zone.rstrip("/").rsplit("/", 1)[-1]
        zone = _safe_zone(leaf)
        if zone is not None:
            region = zone_to_region(zone)

    accel_resp = fetcher(f"{base_root}/instance/attributes/accelerator-type", read_timeout_s)
    if accel_resp is not None and accel_resp[0] == 200 and _is_google_metadata_response(accel_resp[1]):
        raw_accel = accel_resp[2].decode("utf-8", errors="replace").strip()
        if raw_accel and len(raw_accel) <= 64:
            accelerator_type = raw_accel
    else:
        tpu_env_resp = fetcher(f"{base_root}/instance/attributes/tpu-env", read_timeout_s)
        if tpu_env_resp is not None and tpu_env_resp[0] == 200 and _is_google_metadata_response(tpu_env_resp[1]):
            accelerator_type = _parse_tpu_env_accelerator(tpu_env_resp[2].decode("utf-8", errors="replace"))

    return GcpContext(
        on_gcp=True,
        project=project,
        zone=zone,
        region=region,
        accelerator_type=accelerator_type,
        source="gce-metadata",
    )


def _parse_tpu_env_accelerator(text: str) -> str | None:
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip().upper() == "ACCELERATOR_TYPE":
            cleaned = value.strip().strip("'\"")
            if cleaned and len(cleaned) <= 64:
                return cleaned
    return None


def _read_regular_file_bounded(path: Path) -> bytes | None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return None
    try:
        chunks: list[bytes] = []
        total = 0
        while chunk := os.read(descriptor, 4096):
            total += len(chunk)
            if total > _MAX_LOCAL_FILE_BYTES:
                return None
            chunks.append(chunk)
        return b"".join(chunks)
    except OSError:
        return None
    finally:
        os.close(descriptor)


def _read_native_runtime_project(home: Path) -> str | None:
    raw = _read_regular_file_bounded(home / ".eyvan" / "gcp-native-runtime.json")
    if raw is None:
        return None
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(decoded, dict):
        return None
    return _safe_project_id(decoded.get("project_id"))


def _read_autoscale_defaults_project(home: Path) -> str | None:
    raw = _read_regular_file_bounded(home / ".eyvan" / "autoscale" / "defaults.yaml")
    if raw is None:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    for line in text.splitlines():
        match = _YAML_PROJECT_RE.match(line)
        if match is not None:
            candidate = match.group("val").strip().strip("'\"")
            if candidate.lower() in {"", "null", "~", "none"}:
                return None
            return _safe_project_id(candidate)
    return None


def _safe_project_id(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip() or value.strip().lower() == "(unset)":
        return None
    try:
        return validate_project_id(value)
    except ValueError:
        return None


def _safe_zone(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip() or value.strip().lower() == "(unset)":
        return None
    try:
        return validate_zone(value)
    except ValueError:
        return None


def _safe_region(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip() or value.strip().lower() == "(unset)":
        return None
    try:
        return validate_region(value)
    except ValueError:
        return None


def _safe_mtime_ns(path: Path) -> int | None:
    try:
        return path.lstat().st_mtime_ns
    except OSError:
        return None


def _probe_gce_metadata_cached(
    *,
    refresh: bool = False,
    metadata_fetcher: MetadataFetcher | None = None,
    probe_timeout_s: float = _DEFAULT_METADATA_PROBE_TIMEOUT_S,
) -> GcpContext | None:
    """Run :func:`_probe_gce_metadata` and cache both positive and negative default probes."""

    global _METADATA_PROBE_DONE, _METADATA_PROBE_VALUE
    cacheable = metadata_fetcher is None and probe_timeout_s == _DEFAULT_METADATA_PROBE_TIMEOUT_S
    if cacheable and not refresh:
        with _CONTEXT_CACHE_LOCK:
            if _METADATA_PROBE_DONE:
                return _METADATA_PROBE_VALUE

    detected = _probe_gce_metadata(
        metadata_fetcher=metadata_fetcher,
        probe_timeout_s=probe_timeout_s,
    )
    if cacheable:
        with _CONTEXT_CACHE_LOCK:
            _METADATA_PROBE_VALUE = detected
            _METADATA_PROBE_DONE = True
    return detected


def clear_gcp_context_cache() -> None:
    """Clear the process-level caches used by :func:`detect_gcp_context`."""

    global _METADATA_PROBE_DONE, _METADATA_PROBE_VALUE
    with _CONTEXT_CACHE_LOCK:
        _CONTEXT_CACHE.clear()
        _METADATA_PROBE_DONE = False
        _METADATA_PROBE_VALUE = None


def detect_gcp_context(
    *,
    refresh: bool = False,
    config_path: Path | str | None = None,
    allow_gcloud: bool = True,
    metadata_timeout_s: float = _DEFAULT_METADATA_PROBE_TIMEOUT_S,
    gcloud_timeout_s: float = _DEFAULT_GCLOUD_TIMEOUT_S,
    metadata_fetcher: MetadataFetcher | None = None,
    gcloud_runner: GcloudRunner | None = None,
) -> GcpContext:
    """Detect the active GCP/TPU context quickly without mutating global state.

    Probes GCE metadata first with a tight bounded timeout (0.2s default) and
    caches negative off-GCP probes for the process lifetime. Off-GCP, falls back
    in read-only order to:

    1. ``~/.eyvan/storage.json``
    2. ``~/.eyvan/gcp-native-runtime.json``
    3. ``~/.eyvan/autoscale/defaults.yaml``
    4. Read-only ``gcloud config get-value`` (when ``allow_gcloud=True``)

    Default invocations are cached per user home and local config timestamps so
    repeated calls during training setup, API rendering, or CLI commands incur
    zero extra network or subprocess overhead.

    Args:
        refresh: Bypass and refresh the cached value.
        config_path: Optional override path for ``storage.json``.
        allow_gcloud: Whether read-only ``gcloud config get-value`` may be
            consulted off-GCP when local files omit project/zone/region.
        metadata_timeout_s: Bounded timeout for the initial metadata probe.
        gcloud_timeout_s: Bounded timeout for read-only ``gcloud`` calls.
        metadata_fetcher: Injectable metadata probe callback for offline tests.
        gcloud_runner: Injectable gcloud subprocess runner for offline tests.

    Returns:
        :class:`GcpContext` populated from the highest-priority available source.
    """

    home = Path.home()
    storage_store = StorageConfigStore(config_path)
    resolved_storage_path = storage_store.path
    use_cache = (
        metadata_fetcher is None
        and gcloud_runner is None
        and metadata_timeout_s == _DEFAULT_METADATA_PROBE_TIMEOUT_S
        and gcloud_timeout_s == _DEFAULT_GCLOUD_TIMEOUT_S
    )
    cache_key = (
        str(home),
        str(resolved_storage_path),
        bool(allow_gcloud),
        _safe_mtime_ns(resolved_storage_path),
        _safe_mtime_ns(home / ".eyvan" / "gcp-native-runtime.json"),
        _safe_mtime_ns(home / ".eyvan" / "autoscale" / "defaults.yaml"),
    )
    if use_cache and not refresh:
        with _CONTEXT_CACHE_LOCK:
            cached = _CONTEXT_CACHE.get(cache_key)
            if cached is not None:
                return cached

    detected = _probe_gce_metadata_cached(
        refresh=refresh,
        metadata_fetcher=metadata_fetcher,
        probe_timeout_s=metadata_timeout_s,
    )
    if detected is not None:
        if use_cache:
            with _CONTEXT_CACHE_LOCK:
                _CONTEXT_CACHE[cache_key] = detected
        return detected

    project: str | None = None
    zone: str | None = None
    region: str | None = None
    source = "none"

    try:
        storage_cfg = storage_store.load_if_exists()
    except StorageConfigError:
        storage_cfg = None
    if storage_cfg is not None:
        if storage_cfg.project is not None:
            project = storage_cfg.project
            source = "storage-config"
        if len(storage_cfg.regional) == 1:
            region = next(iter(storage_cfg.regional))
            if source == "none":
                source = "storage-config"

    if project is None:
        runtime_project = _read_native_runtime_project(home)
        if runtime_project is not None:
            project = runtime_project
            if source == "none":
                source = "gcp-native-runtime"

    if project is None:
        autoscale_project = _read_autoscale_defaults_project(home)
        if autoscale_project is not None:
            project = autoscale_project
            if source == "none":
                source = "autoscale-defaults"

    if allow_gcloud and (project is None or (zone is None and region is None)):
        runner = gcloud_runner or _default_gcloud_runner
        if project is None:
            rc, out = runner(("config", "get-value", "project"), gcloud_timeout_s)
            if rc < 0:
                allow_gcloud = False
            elif rc == 0:
                parsed_project = _safe_project_id(out)
                if parsed_project is not None:
                    project = parsed_project
                    if source == "none":
                        source = "gcloud"
        if allow_gcloud and zone is None:
            rc, out = runner(("config", "get-value", "compute/zone"), gcloud_timeout_s)
            if rc < 0:
                allow_gcloud = False
            elif rc == 0:
                parsed_zone = _safe_zone(out)
                if parsed_zone is not None:
                    zone = parsed_zone
                    region = zone_to_region(parsed_zone)
                    if source == "none":
                        source = "gcloud"
        if allow_gcloud and region is None:
            rc, out = runner(("config", "get-value", "compute/region"), gcloud_timeout_s)
            if rc == 0:
                parsed_region = _safe_region(out)
                if parsed_region is not None:
                    region = parsed_region
                    if source == "none":
                        source = "gcloud"

    result = GcpContext(
        on_gcp=False,
        project=project,
        zone=zone,
        region=region,
        accelerator_type=None,
        source=source,
    )
    if use_cache:
        with _CONTEXT_CACHE_LOCK:
            _CONTEXT_CACHE[cache_key] = result
    return result


def select_preferred_gcp_project(
    available_projects: tp.Sequence[str],
    *,
    config_path: Path | str | None = None,
    allow_gcloud: bool = True,
) -> str | None:
    """Pick the operator's active GCP project when present in ``available_projects``.

    Checks, in priority order, whether the project recorded in
    ``~/.eyvan/gcp-native-runtime.json``, ``~/.eyvan/storage.json``, or
    :func:`detect_gcp_context` appears in ``available_projects``, falling back to
    ``available_projects[0]`` (or ``None`` when empty).
    """

    if not available_projects:
        return None
    if len(available_projects) == 1:
        return available_projects[0]
    allowed = set(available_projects)
    home = Path.home()

    runtime_project = _read_native_runtime_project(home)
    if runtime_project in allowed:
        return runtime_project

    try:
        storage_cfg = StorageConfigStore(config_path).load_if_exists()
    except StorageConfigError:
        storage_cfg = None
    if storage_cfg is not None and storage_cfg.project in allowed:
        return storage_cfg.project

    detected = detect_gcp_context(config_path=config_path, allow_gcloud=allow_gcloud)
    if detected.project in allowed:
        return detected.project

    return available_projects[0]


class _DefaultGcpTokenProvider:
    """Thread-safe OAuth access token provider (GCE metadata -> gcloud fallback).

    Never includes tokens or subprocess diagnostics in raised errors.
    """

    def __init__(
        self,
        *,
        metadata_fetcher: MetadataFetcher | None = None,
        gcloud_runner: GcloudRunner | None = None,
        metadata_timeout_s: float = _DEFAULT_METADATA_PROBE_TIMEOUT_S,
        gcloud_timeout_s: float = _DEFAULT_GCLOUD_TOKEN_TIMEOUT_S,
    ) -> None:
        self._metadata_fetcher = metadata_fetcher or _default_metadata_fetcher
        self._gcloud_runner = gcloud_runner or _default_gcloud_runner
        self._metadata_timeout_s = metadata_timeout_s
        self._gcloud_timeout_s = gcloud_timeout_s
        self._lock = threading.Lock()
        self._cached_token: str | None = None
        self._expires_at = 0.0
        self._metadata_available: bool | None = None

    def invalidate(self) -> None:
        """Drop any cached token so the next call fetches a fresh credential."""

        with self._lock:
            self._cached_token = None
            self._expires_at = 0.0

    def __call__(self) -> str:
        with self._lock:
            now = time.monotonic()
            if self._cached_token is not None and now < self._expires_at:
                return self._cached_token

            token_and_ttl = self._try_metadata_locked()
            if token_and_ttl is None:
                token_and_ttl = self._try_gcloud_locked()
            if token_and_ttl is None:
                raise StorageAuthError(
                    "could not obtain a GCP access token from GCE metadata or `gcloud auth print-access-token`"
                )
            token, ttl = token_and_ttl
            self._cached_token = token
            self._expires_at = now + ttl
            return token

    def _try_metadata_locked(self) -> tuple[str, float] | None:
        if self._metadata_available is False:
            return None
        timeout = _DEFAULT_METADATA_READ_TIMEOUT_S if self._metadata_available is True else self._metadata_timeout_s
        url = f"{_METADATA_IP_ROOT}/instance/service-accounts/default/token"
        result = self._metadata_fetcher(url, timeout)
        if result is None:
            if self._metadata_available is None:
                self._metadata_available = False
            return None
        status, headers, body = result
        if status != 200 or not _is_google_metadata_response(headers):
            if self._metadata_available is None:
                self._metadata_available = False
            return None
        try:
            payload = json.loads(body.decode("utf-8"))
            token = payload["access_token"]
            expires_in = float(payload.get("expires_in", 300))
            if not _valid_bearer_token(token) or not math.isfinite(expires_in) or expires_in <= 0:
                return None
        except (KeyError, TypeError, ValueError, UnicodeError):
            return None
        self._metadata_available = True
        return token, max(1.0, expires_in - min(60.0, expires_in / 10.0))

    def _try_gcloud_locked(self) -> tuple[str, float] | None:
        rc, out = self._gcloud_runner(("auth", "print-access-token"), self._gcloud_timeout_s)
        if rc != 0:
            return None
        token = out.splitlines()[0].strip() if out else ""
        if not _valid_bearer_token(token):
            return None
        return token, _GCLOUD_TOKEN_TTL_S


def _valid_bearer_token(token: object) -> tp.TypeGuard[str]:
    return (
        isinstance(token, str)
        and bool(token)
        and "\r" not in token
        and "\n" not in token
        and " " not in token
        and "\t" not in token
    )


def default_gcp_token_provider(
    *,
    metadata_fetcher: MetadataFetcher | None = None,
    gcloud_runner: GcloudRunner | None = None,
) -> TokenProvider:
    """Return a cached stdlib GCP OAuth token provider (GCE metadata -> gcloud).

    Safe to pass to :class:`eyvan.track._gcs.GcsTrackingStore` so tracking
    commands work identically on Cloud TPU VMs and operator laptops.
    """

    return _DefaultGcpTokenProvider(
        metadata_fetcher=metadata_fetcher,
        gcloud_runner=gcloud_runner,
    )


class GcsBucketBackend:
    """Stdlib-only GCS JSON API backend for inspecting and creating buckets.

    Uses ``https://storage.googleapis.com/storage/v1/b`` with GCE metadata
    credentials on GCP hosts and ``gcloud auth print-access-token`` on laptops.
    Credentials and raw response bodies are never included in exceptions.

    Args:
        token_provider: Optional callable returning an OAuth access token.
        transport: Optional HTTP peer for offline tests.
        timeout_s: Request timeout in seconds for the default urllib transport.
    """

    def __init__(
        self,
        *,
        token_provider: TokenProvider | None = None,
        transport: Transport | None = None,
        timeout_s: float = _DEFAULT_API_TIMEOUT_S,
    ) -> None:
        self._default_token_provider = _DefaultGcpTokenProvider() if token_provider is None else None
        self._token_provider: TokenProvider = (
            token_provider if token_provider is not None else self._default_token_provider  # type: ignore[assignment]
        )
        self._timeout_s = float(timeout_s)
        self._transport = transport

    def _send(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None = None,
    ) -> tuple[int, tp.Mapping[str, str], bytes]:
        try:
            if self._transport is not None:
                status, response_headers, payload = self._transport(method, url, headers, body)
            else:
                status, response_headers, payload = _urllib_transport(
                    method, url, headers, body, timeout_s=self._timeout_s
                )
        except StorageBackendError:
            raise
        except Exception:
            raise StorageBackendError(f"GCS {method} request failed due to transport error") from None
        if len(payload) > _MAX_API_RESPONSE_BYTES:
            raise StorageBackendError("GCS API response exceeded size limit")
        return status, response_headers, payload

    def _auth_headers(self, *, refresh: bool = False) -> dict[str, str]:
        if refresh and self._default_token_provider is not None:
            self._default_token_provider.invalidate()
        try:
            token = self._token_provider()
        except StorageAuthError:
            raise
        except Exception:
            raise StorageAuthError("could not obtain a GCS access token") from None
        if not _valid_bearer_token(token):
            raise StorageAuthError("GCS token provider returned an invalid access token")
        return {"Authorization": f"Bearer {token}"}

    def _authed_request(
        self,
        method: str,
        url: str,
        *,
        extra_headers: dict[str, str] | None = None,
        body: bytes | None = None,
    ) -> tuple[int, tp.Mapping[str, str], bytes]:
        headers = {**self._auth_headers(), **(extra_headers or {})}
        status, resp_headers, payload = self._send(method, url, headers, body)
        if status == 401 and self._default_token_provider is not None:
            headers = {**self._auth_headers(refresh=True), **(extra_headers or {})}
            status, resp_headers, payload = self._send(method, url, headers, body)
        return status, resp_headers, payload

    def bucket_exists(self, bucket: str) -> bool:
        """Return whether ``bucket`` exists and is accessible."""

        return self.get_bucket(bucket) is not None

    def get_bucket(self, bucket: str) -> BucketInfo | None:
        """Inspect a GCS bucket; returns ``None`` on 404 and raises on 403/other errors."""

        norm_bucket = validate_bucket_name(bucket)
        url = f"{_GCS_API_ROOT}/b/{quote(norm_bucket, safe='')}"
        status, _, payload = self._authed_request("GET", url)
        if status == 404:
            return None
        if status == 403:
            raise BucketNameConflictError(
                f"GCS bucket {norm_bucket!r} is inaccessible or owned by another project (HTTP 403)"
            )
        if status != 200:
            raise StorageBackendError(f"GCS bucket lookup for {norm_bucket!r} returned HTTP {status}")
        return _parse_bucket_info(norm_bucket, payload, created=False)

    def create_bucket(
        self,
        bucket: str,
        *,
        project: str,
        location: str,
        storage_class: str = "STANDARD",
        uniform_access: bool = True,
    ) -> BucketInfo:
        """Create a GCS bucket via the JSON API with uniform bucket-level access."""

        norm_bucket = validate_bucket_name(bucket)
        norm_project = validate_project_id(project)
        norm_location = validate_location(location)
        norm_class = storage_class.strip().upper()
        if norm_class not in _VALID_STORAGE_CLASSES:
            raise ValueError(f"unsupported GCS storage_class {storage_class!r}")
        if not isinstance(uniform_access, bool):
            raise ValueError("uniform_access must be a boolean")

        query = urlencode({"project": norm_project})
        url = f"{_GCS_API_ROOT}/b?{query}"
        request_body = json.dumps(
            {
                "name": norm_bucket,
                "location": norm_location.upper(),
                "storageClass": norm_class,
                "iamConfiguration": {
                    "uniformBucketLevelAccess": {"enabled": uniform_access},
                },
            },
            separators=(",", ":"),
        ).encode("utf-8")
        status, _, payload = self._authed_request(
            "POST",
            url,
            extra_headers={"Content-Type": "application/json"},
            body=request_body,
        )
        if status in {200, 201}:
            return _parse_bucket_info(norm_bucket, payload, created=True)
        if status == 409:
            # Handle multi-host creation race: if the bucket is now ours in the
            # expected location, succeed cleanly without duplicating errors.
            try:
                existing = self.get_bucket(norm_bucket)
            except BucketNameConflictError:
                existing = None
            if existing is not None and existing.location == norm_location:
                return existing
            raise BucketNameConflictError(f"GCS bucket name {norm_bucket!r} is already taken (HTTP 409)")
        if status == 403:
            raise BucketNameConflictError(f"GCS bucket creation for {norm_bucket!r} was denied (HTTP 403)")
        raise StorageBackendError(f"GCS bucket creation for {norm_bucket!r} returned HTTP {status}")


def _parse_bucket_info(expected_name: str, payload: bytes, *, created: bool) -> BucketInfo:
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise StorageBackendError(f"GCS returned malformed JSON for bucket {expected_name!r}") from None
    if not isinstance(decoded, dict):
        raise StorageBackendError(f"GCS returned non-object metadata for bucket {expected_name!r}")
    name = decoded.get("name")
    raw_location = decoded.get("location")
    if not isinstance(name, str) or not isinstance(raw_location, str):
        raise StorageBackendError(f"GCS metadata for bucket {expected_name!r} omitted name or location")
    norm_name = validate_bucket_name(name)
    norm_location = validate_location(raw_location)
    project_number = decoded.get("projectNumber")
    storage_class = decoded.get("storageClass")
    iam = decoded.get("iamConfiguration")
    ubla_enabled = True
    if isinstance(iam, dict):
        ubla = iam.get("uniformBucketLevelAccess")
        if isinstance(ubla, dict) and isinstance(ubla.get("enabled"), bool):
            ubla_enabled = ubla["enabled"]
    return BucketInfo(
        name=norm_name,
        location=norm_location,
        project_number=str(project_number) if project_number is not None else None,
        storage_class=str(storage_class) if isinstance(storage_class, str) else "STANDARD",
        uniform_access=ubla_enabled,
        created=created,
    )


__all__ = (
    "BucketInfo",
    "GcloudRunner",
    "GcpContext",
    "GcsBucketBackend",
    "MetadataFetcher",
    "StorageBackend",
    "TokenProvider",
    "Transport",
    "clear_gcp_context_cache",
    "default_gcp_token_provider",
    "detect_gcp_context",
    "select_preferred_gcp_project",
)
