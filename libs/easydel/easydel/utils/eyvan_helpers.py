# Copyright 2026 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Internal, zero-overhead runtime integration helpers between EasyDeL and Eyvan.

All functions in this module keep top-level imports strictly within the Python
standard library so importing :mod:`easydel` (or importing this module directly)
never pulls in ``eyvan.sdk``, ``fastapi``, or ``uvicorn``, never initializes JAX
backends, and never spawns any background daemon or thread.

Storage helpers lazily import :mod:`eyvan.storage` (stdlib-only) on call.
Preemption / reclaim helpers lazily delegate to :mod:`eyvan` (or its stdlib-only
:mod:`eyvan.worker.runtime.preemption` module) when available and fall back to
pure-stdlib file/signal handling when an older ``eyvan`` is installed.
This module also hosts the process/context-scoped auto-tracking toggle
(:func:`set_eyvan_auto_track_enabled` / :func:`is_eyvan_auto_track_enabled`)
shared by EasyDeL trainers and eSurge monitoring without introducing any new
``os.environ`` knobs.

Public compute-dispatch and Eyvan SDK APIs (such as ``eyvan.init`` and
``@eyvan.execute``) live solely in :mod:`eyvan` and are not re-exported by
EasyDeL.
"""

from __future__ import annotations

import contextvars
import dataclasses
import json
import os
import signal
import sys
import tempfile
import threading
import time
import typing as tp
import urllib.parse
from datetime import UTC, datetime
from pathlib import Path

_RECLAIM_NOTICE_ENV = "EYVAN_RECLAIM_NOTICE_PATH"
_CHECKPOINT_RECEIPT_ENV = "EYVAN_CHECKPOINT_RECEIPT_PATH"
_RESUME_CHECKPOINT_ENV = "EYVAN_RESUME_CHECKPOINT_URI"
_WORKLOAD_ID_ENV = "EYVAN_WORKLOAD_ID"
_PROGRESS_URL_ENV = "EYVAN_PROGRESS_URL"
_PROGRESS_TOKEN_ENV = "EYVAN_MANAGED_RESULT_TOKEN"
_DEFAULT_RECLAIM_GRACE_SECONDS = 600
_VALID_RECLAIM_POLICIES = frozenset({"requeue", "stop"})

_PREEMPTION_LOCK = threading.Lock()
_SIGNAL_PREEMPTION_NOTICE: PreemptionNotice | None = None
_SIGNAL_CALLBACKS: list[tp.Callable[[PreemptionNotice], tp.Any]] = []
_SIGNAL_HANDLERS_INSTALLED: bool = False
_PREV_SIGNAL_HANDLERS: dict[int, tp.Any] = {}

_EYVAN_AUTO_TRACK_DEFAULT: bool = True
_EYVAN_AUTO_TRACK_VAR: contextvars.ContextVar[bool | None] = contextvars.ContextVar(
    "easydel_eyvan_auto_track_enabled",
    default=None,
)


class _EyvanAutoTrackToggle:
    """Handle returned by :func:`set_eyvan_auto_track_enabled` supporting direct and ``with`` usage."""

    __slots__ = ("_prev_default", "_reset", "_token", "_update_global", "enabled")

    def __init__(
        self,
        *,
        token: contextvars.Token[bool | None],
        enabled: bool,
        update_global: bool,
        prev_default: bool,
    ) -> None:
        self._token = token
        self.enabled = enabled
        self._update_global = update_global
        self._prev_default = prev_default
        self._reset = False

    def reset(self) -> None:
        """Restore the previous auto-tracking setting (idempotent)."""
        global _EYVAN_AUTO_TRACK_DEFAULT
        if self._reset:
            return
        self._reset = True
        if self._update_global:
            _EYVAN_AUTO_TRACK_DEFAULT = self._prev_default
        _EYVAN_AUTO_TRACK_VAR.reset(self._token)

    def __enter__(self) -> bool:
        return self.enabled

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.reset()


def is_eyvan_auto_track_enabled() -> bool:
    """Return whether implicit Eyvan auto-tracking (when ``use_eyvan_track=None``) is allowed.

    Defaults to ``True`` outside test isolation. Explicit ``use_eyvan_track=True``,
    ``eyvan_track_uri=...``, ``enable_eyvan_track=True``, and Eyvan workload
    environment variables still activate tracking even when this returns ``False``.

    Returns:
        ``True`` if implicit Eyvan auto-tracking is enabled in the current context.
    """
    override = _EYVAN_AUTO_TRACK_VAR.get()
    if override is not None:
        return bool(override)
    return _EYVAN_AUTO_TRACK_DEFAULT


def set_eyvan_auto_track_enabled(
    enabled: bool,
    *,
    global_default: bool = False,
) -> _EyvanAutoTrackToggle:
    """Enable or disable implicit Eyvan auto-tracking in code (no ``os.environ`` knob).

    Works both imperatively (``set_eyvan_auto_track_enabled(False)``) and as a
    scoped context manager (``with set_eyvan_auto_track_enabled(True): ...``).

    Args:
        enabled: Whether implicit Eyvan auto-tracking is enabled.
        global_default: When ``True``, also updates the process-wide fallback
            seen by newly spawned threads (used by pytest isolation fixtures)
            and restores it on ``reset()`` / context exit.

    Returns:
        A toggle handle whose ``reset()`` or ``with`` exit restores the prior state.

    Raises:
        TypeError: If ``enabled`` is not a ``bool``.
    """
    global _EYVAN_AUTO_TRACK_DEFAULT
    if not isinstance(enabled, bool):
        raise TypeError("set_eyvan_auto_track_enabled expects a boolean `enabled` argument.")
    prev_default = _EYVAN_AUTO_TRACK_DEFAULT
    if global_default:
        _EYVAN_AUTO_TRACK_DEFAULT = enabled
    token = _EYVAN_AUTO_TRACK_VAR.set(enabled)
    return _EyvanAutoTrackToggle(
        token=token,
        enabled=enabled,
        update_global=global_default,
        prev_default=prev_default,
    )


def _ensure_local_eyvan_store_writable(
    uri: str | None,
    *,
    project: str,
    org: str = "default",
) -> None:
    """Verify a local Eyvan tracking directory is creatable and writable before starting a run.

    Remote URIs (for example ``gs://...``) return immediately without touching the
    local filesystem. Local paths and ``file://`` URIs are created and checked for
    write access so read-only ``HOME`` / container filesystems fail fast once
    instead of spawning a background writer thread that fails on every flush.

    Args:
        uri: Resolved Eyvan tracking URI or ``None`` for ``~/.eyvan/tracking/<org>/<project>``.
        project: Sanitized Eyvan project slug.
        org: Eyvan organization segment (default ``"default"``).

    Raises:
        PermissionError: If the target local directory exists but is not writable.
        OSError: If creating the target local directory fails.
    """
    if uri is None:
        target = Path("~/.eyvan/tracking").expanduser() / org / project
    elif uri.startswith("file://"):
        parsed = urllib.parse.urlparse(uri)
        target = Path(urllib.parse.unquote(parsed.path)).expanduser()
    elif "://" in uri:
        return
    else:
        target = Path(uri).expanduser()

    target.mkdir(parents=True, exist_ok=True)
    if not target.is_dir() or not os.access(target, os.W_OK | os.X_OK):
        raise PermissionError(f"Eyvan tracking store directory is not writable: {target}")


def _require_eyvan_storage():
    """Lazily import ``eyvan.storage`` (stdlib-only) with a helpful error if missing."""
    try:
        import eyvan.storage as eyvan_storage
    except ImportError as exc:
        raise ImportError(
            "Eyvan storage helpers require `eyvan` to be installed (e.g. `uv sync` in the EasyDeL workspace "
            "or `pip install eyvan`)."
        ) from exc
    return eyvan_storage


def get_storage(
    *,
    region: str | None = None,
    zone: str | None = None,
    project: str | None = None,
    bucket: str | None = None,
    create: bool | None = None,
    persist: bool = True,
    **kwargs: tp.Any,
) -> tp.Any:
    """Resolve the regional Eyvan GCS bucket handle (``eyvan_<region>``).

    Lazily delegates to :func:`eyvan.storage.get_storage` without importing
    ``eyvan.sdk`` or server dependencies.

    Args:
        region: Explicit GCP region (e.g. ``"us-central2"``). Auto-detected when omitted.
        zone: Explicit GCP zone (e.g. ``"us-central2-b"``); implies its parent region.
        project: Optional GCP project ID override.
        bucket: Optional explicit GCS bucket name override.
        create: Consent-by-default creation policy (``True`` creates immediately,
            ``False`` fails if absent, ``None`` prompts on TTY or uses config).
        persist: Persist resolved bucket configuration into ``~/.eyvan/storage.json``.
        **kwargs: Additional keyword arguments forwarded to :func:`eyvan.storage.get_storage`.

    Returns:
        Resolved :class:`eyvan.storage.StorageHandle`.
    """
    storage = _require_eyvan_storage()
    return storage.get_storage(
        region=region,
        zone=zone,
        project=project,
        bucket=bucket,
        create=create,
        persist=persist,
        **kwargs,
    )


def storage_path(
    *parts: str,
    region: str | None = None,
    zone: str | None = None,
    project: str | None = None,
    bucket: str | None = None,
    create: bool | None = None,
    persist: bool = True,
    **kwargs: tp.Any,
) -> str:
    """Resolve regional Eyvan storage and join ``*parts`` into a canonical ``gs://`` URI.

    Lazily delegates to :func:`eyvan.storage.storage_path` without importing
    ``eyvan.sdk`` or server dependencies.

    Args:
        *parts: Relative object path segments appended under the regional bucket
            (e.g. ``storage_path("checkpoints", "qwen3-8b")``).
        region: Explicit GCP region (e.g. ``"us-central2"``). Auto-detected when omitted.
        zone: Explicit GCP zone (e.g. ``"us-central2-b"``).
        project: Optional GCP project ID override.
        bucket: Optional explicit GCS bucket name override.
        create: Consent-by-default creation policy.
        persist: Persist resolved bucket configuration into ``~/.eyvan/storage.json``.
        **kwargs: Additional keyword arguments forwarded to :func:`eyvan.storage.storage_path`.

    Returns:
        Canonical ``gs://`` URI string.
    """
    storage = _require_eyvan_storage()
    return storage.storage_path(
        *parts,
        region=region,
        zone=zone,
        project=project,
        bucket=bucket,
        create=create,
        persist=persist,
        **kwargs,
    )


def get_monitoring_storage(
    *,
    project: str | None = None,
    location: str | None = None,
    region: str | None = None,
    zone: str | None = None,
    bucket: str | None = None,
    prefix: str | None = None,
    create: bool | None = None,
    persist: bool = True,
    **kwargs: tp.Any,
) -> tp.Any:
    """Resolve the default Eyvan monitoring GCS bucket handle used for run tracking.

    Lazily delegates to :func:`eyvan.storage.get_monitoring_storage`.

    Args:
        project: Optional GCP project ID override.
        location: Optional GCS bucket location (region or multi-region).
        region: Optional GCP region alias for ``location``.
        zone: Optional GCP zone whose parent region selects the location.
        bucket: Optional explicit monitoring bucket override.
        prefix: Optional tracking path prefix inside the bucket (default ``"tracking"``).
        create: Consent-by-default creation policy.
        persist: Persist resolved bucket configuration into ``~/.eyvan/storage.json``.
        **kwargs: Additional keyword arguments forwarded to :func:`eyvan.storage.get_monitoring_storage`.

    Returns:
        Resolved :class:`eyvan.storage.StorageHandle`.
    """
    storage = _require_eyvan_storage()
    return storage.get_monitoring_storage(
        project=project,
        location=location,
        region=region,
        zone=zone,
        bucket=bucket,
        prefix=prefix,
        create=create,
        persist=persist,
        **kwargs,
    )


def monitoring_path(
    *parts: str,
    project: str | None = None,
    location: str | None = None,
    region: str | None = None,
    zone: str | None = None,
    bucket: str | None = None,
    prefix: str | None = None,
    create: bool | None = None,
    persist: bool = True,
    **kwargs: tp.Any,
) -> str:
    """Resolve the Eyvan monitoring bucket and join ``*parts`` into a canonical ``gs://`` URI.

    Lazily delegates to :func:`eyvan.storage.monitoring_path`.

    Args:
        *parts: Relative object path segments appended under the monitoring bucket.
        project: Optional GCP project ID override.
        location: Optional GCS bucket location.
        region: Optional GCP region alias for ``location``.
        zone: Optional GCP zone.
        bucket: Optional explicit monitoring bucket override.
        prefix: Optional tracking path prefix inside the bucket.
        create: Consent-by-default creation policy.
        persist: Persist resolved bucket configuration into ``~/.eyvan/storage.json``.
        **kwargs: Additional keyword arguments forwarded to :func:`eyvan.storage.monitoring_path`.

    Returns:
        Canonical ``gs://`` URI string.
    """
    storage = _require_eyvan_storage()
    return storage.monitoring_path(
        *parts,
        project=project,
        location=location,
        region=region,
        zone=zone,
        bucket=bucket,
        prefix=prefix,
        create=create,
        persist=persist,
        **kwargs,
    )


def configure_storage(
    *,
    project: str | None = None,
    region: str | None = None,
    zone: str | None = None,
    bucket: str | None = None,
    monitoring_bucket: str | None = None,
    monitoring_location: str | None = None,
    monitoring_region: str | None = None,
    monitoring_prefix: str | None = None,
    auto_create: bool | None = None,
    create: bool = False,
    **kwargs: tp.Any,
) -> tp.Any:
    """Persist Eyvan regional and monitoring storage configuration into ``~/.eyvan/storage.json``.

    Lazily delegates to :func:`eyvan.storage.configure_storage`.

    Args:
        project: Optional GCP project ID to record.
        region: Optional regional bucket GCP region.
        zone: Optional regional bucket GCP zone.
        bucket: Optional explicit regional bucket name.
        monitoring_bucket: Optional explicit monitoring bucket name.
        monitoring_location: Optional monitoring bucket location.
        monitoring_region: Optional monitoring bucket region alias.
        monitoring_prefix: Optional monitoring tracking prefix.
        auto_create: Optional automatic bucket creation policy for non-interactive jobs.
        create: If ``True``, ensure the buckets exist in GCS before saving.
        **kwargs: Additional keyword arguments forwarded to :func:`eyvan.storage.configure_storage`.

    Returns:
        Updated :class:`eyvan.storage.StorageConfig`.
    """
    storage = _require_eyvan_storage()
    return storage.configure_storage(
        project=project,
        region=region,
        zone=zone,
        bucket=bucket,
        monitoring_bucket=monitoring_bucket,
        monitoring_location=monitoring_location,
        monitoring_region=monitoring_region,
        monitoring_prefix=monitoring_prefix,
        auto_create=auto_create,
        create=create,
        **kwargs,
    )


@dataclasses.dataclass(frozen=True, slots=True)
class PreemptionNotice:
    """Structured Eyvan graceful reclaim / preemption notice delivered inside a job.

    Mirrors :class:`eyvan.PreemptionNotice` so EasyDeL callers and trainers work
    identically whether a newer ``eyvan`` with native preemption exports or an
    older ``eyvan`` package is installed.
    """

    reason: str = "reclaim"
    requested_by: str | None = None
    deadline_at: str | None = None
    deadline_epoch_s: float | None = None
    grace_seconds: int = _DEFAULT_RECLAIM_GRACE_SECONDS
    policy: str = "requeue"
    checkpoint_uri: str | None = None
    workload_id: str | None = None

    @property
    def remaining_seconds(self) -> float:
        """Return non-negative seconds remaining until the reclaim checkpoint deadline."""
        if self.deadline_epoch_s is not None:
            return max(0.0, float(self.deadline_epoch_s) - time.time())
        if self.deadline_at:
            parsed = _parse_iso_timestamp(self.deadline_at)
            if parsed is not None:
                return max(0.0, parsed - time.time())
        return float(max(int(self.grace_seconds), 0))


def _parse_iso_timestamp(value: str) -> float | None:
    """Parse an ISO-8601 timestamp string into POSIX epoch seconds, or ``None`` on error."""
    text = value.strip()
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return float(dt.timestamp())
    except ValueError:
        return None


def _coerce_optional_str(value: tp.Any) -> str | None:
    """Return a stripped non-empty string or ``None``."""
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _parse_preemption_notice_payload(
    payload: tp.Mapping[str, tp.Any],
    *,
    environment: tp.Mapping[str, str] | None = None,
) -> PreemptionNotice:
    """Construct a :class:`PreemptionNotice` from a decoded JSON mapping."""
    environ = os.environ if environment is None else environment
    reason = _coerce_optional_str(payload.get("reason")) or "reclaim"
    requested_by = _coerce_optional_str(payload.get("requested_by"))
    deadline_at = _coerce_optional_str(payload.get("deadline_at"))

    deadline_epoch_s: float | None = None
    raw_epoch = payload.get("deadline_epoch_s")
    if isinstance(raw_epoch, (int, float)) and not isinstance(raw_epoch, bool):
        epoch_f = float(raw_epoch)
        if epoch_f == epoch_f and epoch_f not in (float("inf"), float("-inf")):
            deadline_epoch_s = epoch_f
    if deadline_epoch_s is None and deadline_at is not None:
        deadline_epoch_s = _parse_iso_timestamp(deadline_at)

    grace_seconds = _DEFAULT_RECLAIM_GRACE_SECONDS
    raw_grace = payload.get("grace_seconds")
    if isinstance(raw_grace, (int, float)) and not isinstance(raw_grace, bool):
        grace_seconds = max(int(raw_grace), 0)

    raw_policy = _coerce_optional_str(payload.get("policy"))
    policy = raw_policy if raw_policy in _VALID_RECLAIM_POLICIES else "requeue"
    checkpoint_uri = _coerce_optional_str(payload.get("checkpoint_uri"))
    workload_id = _coerce_optional_str(payload.get("workload_id")) or _coerce_optional_str(environ.get(_WORKLOAD_ID_ENV))

    return PreemptionNotice(
        reason=reason,
        requested_by=requested_by,
        deadline_at=deadline_at,
        deadline_epoch_s=deadline_epoch_s,
        grace_seconds=grace_seconds,
        policy=policy,
        checkpoint_uri=checkpoint_uri,
        workload_id=workload_id,
    )


def _serialize_preemption_notice(notice: tp.Any) -> dict[str, tp.Any]:
    """Serialize any Eyvan or fallback preemption notice object into a JSON-safe dict."""
    return {
        "reason": _coerce_optional_str(getattr(notice, "reason", None)) or "reclaim",
        "requested_by": _coerce_optional_str(getattr(notice, "requested_by", None)),
        "deadline_at": _coerce_optional_str(getattr(notice, "deadline_at", None)),
        "deadline_epoch_s": getattr(notice, "deadline_epoch_s", None),
        "grace_seconds": int(getattr(notice, "grace_seconds", _DEFAULT_RECLAIM_GRACE_SECONDS) or 0),
        "policy": _coerce_optional_str(getattr(notice, "policy", None)) or "requeue",
        "checkpoint_uri": _coerce_optional_str(getattr(notice, "checkpoint_uri", None)),
        "workload_id": _coerce_optional_str(getattr(notice, "workload_id", None)),
    }


def _read_notice_file(environment: tp.Mapping[str, str]) -> PreemptionNotice | None:
    """Read and parse the atomic Eyvan reclaim notice JSON file, returning ``None`` on absence or error."""
    raw_path = _coerce_optional_str(environment.get(_RECLAIM_NOTICE_ENV))
    if raw_path is None:
        return None
    try:
        raw_text = Path(raw_path).read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw_text)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    return _parse_preemption_notice_payload(data, environment=environment)


def _get_eyvan_preemption_callable(name: str) -> tp.Callable[..., tp.Any] | None:
    """Lazily resolve a preemption protocol callable from ``eyvan`` without pulling in ``eyvan.sdk``."""
    try:
        import eyvan
    except ImportError:
        return None
    candidate = getattr(eyvan, name, None)
    if callable(candidate):
        return candidate
    try:
        from eyvan.worker.runtime import preemption as eyvan_preemption
    except ImportError:
        return None
    candidate = getattr(eyvan_preemption, name, None)
    return candidate if callable(candidate) else None


def _uninstall_eyvan_preemption_module(preemption_mod: tp.Any, *, on_main_thread: bool) -> None:
    """Safely uninstall or clear live notice state on ``eyvan.worker.runtime.preemption``.

    On the main thread, delegates directly to ``preemption_mod.uninstall_preemption_handler()``.
    On a background thread, clears ``_signal_notice`` and ``_callbacks`` under ``_install_lock``
    without discarding ``_installed_signals`` / ``_previous_handlers`` so a subsequent main-thread
    uninstall can still restore the original handlers cleanly (even with older ``eyvan`` builds).
    """
    if on_main_thread:
        uninstall = getattr(preemption_mod, "uninstall_preemption_handler", None)
        if callable(uninstall):
            uninstall()
        return

    install_lock = getattr(preemption_mod, "_install_lock", None)
    has_internal_state = (
        hasattr(preemption_mod, "_signal_notice")
        and hasattr(preemption_mod, "_installed_signals")
        and hasattr(preemption_mod, "_previous_handlers")
    )
    if has_internal_state and hasattr(install_lock, "__enter__") and hasattr(install_lock, "__exit__"):
        with install_lock:
            preemption_mod._signal_notice = None
            callbacks = getattr(preemption_mod, "_callbacks", None)
            if isinstance(callbacks, list):
                callbacks.clear()
        return

    uninstall = getattr(preemption_mod, "uninstall_preemption_handler", None)
    if callable(uninstall):
        uninstall()


def uninstall_preemption_handler() -> None:
    """Restore previous ``SIGUSR1`` / ``SIGTERM`` handlers and clear live signal notice state.

    Delegates lazily to Eyvan's ``uninstall_preemption_handler`` when available
    and also restores any fallback handlers recorded in ``_PREV_SIGNAL_HANDLERS``
    while clearing process-local signal notices and callbacks so subsequent
    trainers in the same process never observe a stale reclaim notice.

    Safe and idempotent to call multiple times or from a non-main thread. When
    called from a background thread, live signal notices and callbacks are
    cleared without discarding saved main-thread original handlers so a later
    main-thread call can still restore them cleanly.
    """
    global _SIGNAL_PREEMPTION_NOTICE, _SIGNAL_HANDLERS_INSTALLED
    on_main_thread = threading.current_thread() is threading.main_thread()

    with _PREEMPTION_LOCK:
        _SIGNAL_PREEMPTION_NOTICE = None
        _SIGNAL_CALLBACKS.clear()
        if on_main_thread:
            for signum, prev in list(_PREV_SIGNAL_HANDLERS.items()):
                try:
                    signal.signal(signum, prev)
                except (OSError, ValueError):
                    pass
            _PREV_SIGNAL_HANDLERS.clear()
            _SIGNAL_HANDLERS_INSTALLED = False

    eyvan_fn = _get_eyvan_preemption_callable("uninstall_preemption_handler")
    if eyvan_fn is not None:
        if not on_main_thread and getattr(eyvan_fn, "__module__", None) == "eyvan.worker.runtime.preemption":
            preemption_mod = sys.modules.get("eyvan.worker.runtime.preemption")
            if preemption_mod is not None and eyvan_fn is getattr(preemption_mod, "uninstall_preemption_handler", None):
                _uninstall_eyvan_preemption_module(preemption_mod, on_main_thread=False)
                return
        eyvan_fn()


def _reset_fallback_preemption_state() -> None:
    """Reset module-level fallback and Eyvan signal state (intended for test isolation)."""
    on_main_thread = threading.current_thread() is threading.main_thread()
    try:
        uninstall_preemption_handler()
    except Exception:
        pass

    preemption_mod = sys.modules.get("eyvan.worker.runtime.preemption")
    if preemption_mod is not None:
        try:
            _uninstall_eyvan_preemption_module(preemption_mod, on_main_thread=on_main_thread)
        except Exception:
            pass


def preemption_notice(
    environment: tp.Mapping[str, str] | None = None,
) -> tp.Any:
    """Return the active Eyvan graceful reclaim / preemption notice, or ``None``.

    Delegates lazily to :func:`eyvan.preemption_notice` when available and falls
    back cleanly to reading ``EYVAN_RECLAIM_NOTICE_PATH`` and the process-local
    signal flag installed by :func:`install_preemption_handler`. Never raises on
    missing or malformed notice files and performs zero XLA compilation.

    Args:
        environment: Optional environment mapping (defaults to ``os.environ``).

    Returns:
        Active :class:`PreemptionNotice` (or ``eyvan.PreemptionNotice``) if a
        reclaim / preemption notice is active, otherwise ``None``.
    """
    eyvan_fn = _get_eyvan_preemption_callable("preemption_notice")
    if eyvan_fn is not None:
        if environment is not None:
            return eyvan_fn(environment=environment)
        return eyvan_fn()

    environ = os.environ if environment is None else environment
    file_notice = _read_notice_file(environ)
    if file_notice is not None:
        return file_notice
    if environment is None:
        return _SIGNAL_PREEMPTION_NOTICE
    return None


def reclaim_requested(
    environment: tp.Mapping[str, str] | None = None,
) -> bool:
    """Return ``True`` when an Eyvan reclaim / preemption notice is active.

    Fast O(1) host-only check suitable for calling at training step boundaries
    with zero XLA compilation.

    Args:
        environment: Optional environment mapping (defaults to ``os.environ``).

    Returns:
        ``True`` if a graceful reclaim / preemption has been requested.
    """
    eyvan_fn = _get_eyvan_preemption_callable("reclaim_requested")
    if eyvan_fn is not None:
        if environment is not None:
            return bool(eyvan_fn(environment=environment))
        return bool(eyvan_fn())
    return preemption_notice(environment=environment) is not None


def _fallback_signal_handler(signum: int, frame: tp.Any) -> None:
    """Record a live fallback :class:`PreemptionNotice` on ``SIGUSR1`` / ``SIGTERM`` without aborting mid-step."""
    global _SIGNAL_PREEMPTION_NOTICE
    file_notice = _read_notice_file(os.environ)
    if file_notice is not None:
        notice = file_notice
    else:
        try:
            sig_name = signal.Signals(signum).name
        except ValueError:
            sig_name = str(signum)
        now = time.time()
        deadline_epoch = now + float(_DEFAULT_RECLAIM_GRACE_SECONDS)
        deadline_iso = datetime.fromtimestamp(deadline_epoch, tz=UTC).isoformat()
        notice = PreemptionNotice(
            reason=f"signal:{sig_name}",
            requested_by=None,
            deadline_at=deadline_iso,
            deadline_epoch_s=deadline_epoch,
            grace_seconds=_DEFAULT_RECLAIM_GRACE_SECONDS,
            policy="requeue",
            checkpoint_uri=None,
            workload_id=_coerce_optional_str(os.environ.get(_WORKLOAD_ID_ENV)),
        )

    _SIGNAL_PREEMPTION_NOTICE = notice
    callbacks = tuple(_SIGNAL_CALLBACKS)
    for cb in callbacks:
        try:
            cb(notice)
        except Exception:
            pass

    prev = _PREV_SIGNAL_HANDLERS.get(signum)
    if callable(prev) and prev is not _fallback_signal_handler:
        try:
            prev(signum, frame)
        except Exception:
            pass


def install_preemption_handler(
    callback: tp.Callable[[tp.Any], tp.Any] | None = None,
) -> None:
    """Install an idempotent ``SIGUSR1`` / ``SIGTERM`` graceful preemption signal handler.

    Delegates lazily to :func:`eyvan.install_preemption_handler` when available
    and otherwise installs a safe, chaining stdlib handler on the main thread so
    an Eyvan reclaim signal records a live :class:`PreemptionNotice` instead of
    aborting training mid-step. Safe and non-fatal to call from any thread or
    multiple times.

    Args:
        callback: Optional callable invoked with the recorded notice on signal receipt.
    """
    eyvan_fn = _get_eyvan_preemption_callable("install_preemption_handler")
    if eyvan_fn is not None:
        if callback is not None:
            eyvan_fn(callback=callback)
        else:
            eyvan_fn()
        return

    global _SIGNAL_HANDLERS_INSTALLED
    with _PREEMPTION_LOCK:
        if callback is not None and callback not in _SIGNAL_CALLBACKS:
            _SIGNAL_CALLBACKS.append(callback)
        if _SIGNAL_HANDLERS_INSTALLED:
            return
        if threading.current_thread() is not threading.main_thread():
            return
        for sig_name in ("SIGUSR1", "SIGTERM"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                prev = signal.getsignal(sig)
                if prev is not _fallback_signal_handler:
                    _PREV_SIGNAL_HANDLERS[int(sig)] = prev
                    signal.signal(sig, _fallback_signal_handler)
            except (OSError, ValueError):
                continue
        _SIGNAL_HANDLERS_INSTALLED = True


def acknowledge_checkpoint(
    uri: str,
    *,
    step: int | None = None,
    environment: tp.Mapping[str, str] | None = None,
) -> Path | None:
    """Acknowledge a saved reclaim checkpoint so Eyvan can hand it to the resumed attempt.

    Delegates lazily to :func:`eyvan.acknowledge_checkpoint` when available.
    Otherwise atomically writes a ``0o600`` JSON receipt to
    ``EYVAN_CHECKPOINT_RECEIPT_PATH`` (when set) and best-effort reports
    checkpoint progress via :func:`eyvan.report_progress` if an in-job progress
    channel is active.

    Args:
        uri: Durable checkpoint URI or path saved before exiting.
        step: Optional global training step associated with the checkpoint.
        environment: Optional environment mapping (defaults to ``os.environ``).

    Returns:
        Path to the written receipt file, or ``None`` when no receipt path is configured.
    """
    eyvan_fn = _get_eyvan_preemption_callable("acknowledge_checkpoint")
    if eyvan_fn is not None:
        if environment is not None:
            return eyvan_fn(uri, step=step, environment=environment)
        return eyvan_fn(uri, step=step)

    environ = os.environ if environment is None else environment
    receipt_raw = _coerce_optional_str(environ.get(_CHECKPOINT_RECEIPT_ENV))
    written_path: Path | None = None

    if receipt_raw is not None:
        target = Path(receipt_raw)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint_uri": str(uri),
            "step": int(step) if step is not None else None,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        fd, tmp_name = tempfile.mkstemp(prefix=".eyvan-ckpt-receipt-", dir=str(target.parent))
        tmp_path = Path(tmp_name)
        try:
            try:
                os.fchmod(fd, 0o600)
            except (AttributeError, OSError):
                pass
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, target)
            try:
                os.chmod(target, 0o600)
            except OSError:
                pass
            written_path = target
        except Exception:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    if environ.get(_PROGRESS_URL_ENV) and environ.get(_PROGRESS_TOKEN_ENV):
        try:
            from eyvan.worker.runtime.progress import report_progress

            report_progress(
                step=max(int(step), 0) if step is not None else None,
                message=f"Reclaim checkpoint saved: {uri}",
                environment=environ,
            )
        except Exception:
            pass

    return written_path


def resume_checkpoint_uri(
    environment: tp.Mapping[str, str] | None = None,
) -> str | None:
    """Return the Eyvan resume checkpoint URI for a rescheduled attempt, or ``None``.

    Delegates lazily to :func:`eyvan.resume_checkpoint_uri` when available.
    Otherwise checks ``EYVAN_RESUME_CHECKPOINT_URI`` first, then falls back to
    reading ``checkpoint_uri`` from ``EYVAN_CHECKPOINT_RECEIPT_PATH`` if present.

    Args:
        environment: Optional environment mapping (defaults to ``os.environ``).

    Returns:
        Checkpoint URI/path string to resume from, or ``None`` if none is set.
    """
    eyvan_fn = _get_eyvan_preemption_callable("resume_checkpoint_uri")
    if eyvan_fn is not None:
        if environment is not None:
            return eyvan_fn(environment=environment)
        return eyvan_fn()

    environ = os.environ if environment is None else environment
    direct = _coerce_optional_str(environ.get(_RESUME_CHECKPOINT_ENV))
    if direct is not None:
        return direct

    receipt_raw = _coerce_optional_str(environ.get(_CHECKPOINT_RECEIPT_ENV))
    if receipt_raw is None:
        return None
    try:
        data = json.loads(Path(receipt_raw).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if isinstance(data, dict):
        return _coerce_optional_str(data.get("checkpoint_uri"))
    return None


__all__ = (
    "PreemptionNotice",
    "acknowledge_checkpoint",
    "configure_storage",
    "get_monitoring_storage",
    "get_storage",
    "install_preemption_handler",
    "is_eyvan_auto_track_enabled",
    "monitoring_path",
    "preemption_notice",
    "reclaim_requested",
    "resume_checkpoint_uri",
    "set_eyvan_auto_track_enabled",
    "storage_path",
    "uninstall_preemption_handler",
)
