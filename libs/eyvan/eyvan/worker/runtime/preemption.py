# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""In-job graceful preemption and reclaim notice protocol.

This module runs *inside* a training or serving workload process on the same
critical path as :mod:`eyvan.worker.runtime.progress` and
:mod:`eyvan.worker.runtime.in_job`. It intentionally imports **only** the Python
standard library so ``import eyvan`` and ``eyvan.preemption_notice()`` never
pull in FastAPI, Pydantic, httpx, or server dependencies.

Workflow during a graceful reclaim (e.g. a 10-minute owner-reclaim window):

1. The job calls :func:`install_preemption_handler` once at startup.
2. Eyvan writes an atomic JSON notice to ``EYVAN_RECLAIM_NOTICE_PATH`` (mode
   ``0600``) and delivers ``SIGUSR1`` (or ``SIGTERM``) to the workload process
   group.
3. The handler records the live :class:`PreemptionNotice` (enriched from the
   notice file when present) instead of terminating Python mid-step.
4. At the next safe step boundary, the job checks :func:`reclaim_requested` or
   :func:`preemption_notice`, saves a durable checkpoint, calls
   :func:`acknowledge_checkpoint` (which atomically writes
   ``EYVAN_CHECKPOINT_RECEIPT_PATH`` and reports progress when configured), and
   exits cleanly before ``deadline_epoch_s``.
5. When Eyvan reschedules the job onto fresh capacity, the new attempt reads
   :func:`resume_checkpoint_uri` (from ``EYVAN_RESUME_CHECKPOINT_URI`` or the
   receipt file) and resumes seamlessly.
"""

from __future__ import annotations

import json
import math
import os
import signal
import tempfile
import threading
import time
import typing as tp
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType

#: Path to the atomic JSON file written by the worker runtime when a graceful
#: reclaim or preemption begins.
RECLAIM_NOTICE_PATH_ENV = "EYVAN_RECLAIM_NOTICE_PATH"

#: Path to the atomic JSON receipt written by the job after saving its
#: checkpoint so the control plane can persist the URI for the next attempt.
CHECKPOINT_RECEIPT_PATH_ENV = "EYVAN_CHECKPOINT_RECEIPT_PATH"

#: Checkpoint URI injected into the rescheduled attempt by the control plane.
RESUME_CHECKPOINT_URI_ENV = "EYVAN_RESUME_CHECKPOINT_URI"

#: Valid reclaim settlement policies.
_VALID_POLICIES = frozenset({"requeue", "stop"})

# Module-level state updated by the installed signal handler. Keep mutations
# simple reference replacements so the signal handler never takes a blocking
# lock and cannot deadlock with main-thread code.
_signal_notice: PreemptionNotice | None = None
_callbacks: list[tp.Callable[[PreemptionNotice], tp.Any]] = []
_installed_signals: set[int] = set()
_previous_handlers: dict[int, tp.Any] = {}
_install_lock = threading.Lock()


def _parse_iso_epoch(raw: str | None) -> float | None:
    """Parse an ISO-8601 timestamp into POSIX epoch seconds, or ``None``."""

    if not raw or not isinstance(raw, str):
        return None
    normalized = raw.strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _optional_str(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _non_negative_int(value: object, *, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float) and math.isfinite(value):
        return max(int(value), 0)
    if isinstance(value, str):
        try:
            return max(int(value.strip()), 0)
        except ValueError:
            return default
    return default


def _atomic_write_json_0600(path: Path, payload: tp.Mapping[str, object]) -> Path:
    """Write ``payload`` atomically with owner-only ``0600`` permissions."""

    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return path


@dataclass(frozen=True, slots=True)
class PreemptionNotice:
    """Structured notice delivered to a running workload on graceful reclaim."""

    reason: str
    requested_by: str | None = None
    deadline_at: str | None = None
    deadline_epoch_s: float | None = None
    grace_seconds: int = 0
    policy: str = "requeue"
    checkpoint_uri: str | None = None
    workload_id: str | None = None

    @property
    def remaining_seconds(self) -> float:
        """Seconds remaining before the worker escalates to forced termination."""

        if self.deadline_epoch_s is not None and math.isfinite(self.deadline_epoch_s):
            return max(0.0, self.deadline_epoch_s - time.time())
        parsed = _parse_iso_epoch(self.deadline_at)
        if parsed is not None:
            return max(0.0, parsed - time.time())
        return float(max(self.grace_seconds, 0))

    def to_dict(self) -> dict[str, object]:
        """Serialize into the canonical on-disk notice JSON representation."""

        return {
            "reason": self.reason,
            "requested_by": self.requested_by,
            "deadline_at": self.deadline_at,
            "deadline_epoch_s": self.deadline_epoch_s,
            "grace_seconds": self.grace_seconds,
            "policy": self.policy,
            "checkpoint_uri": self.checkpoint_uri,
            "workload_id": self.workload_id,
        }

    @classmethod
    def from_dict(cls, data: tp.Mapping[str, object]) -> PreemptionNotice | None:
        """Decode one notice JSON object without raising on partial or invalid fields."""

        raw_reason = _optional_str(data.get("reason"))
        if raw_reason is None:
            return None
        raw_policy = _optional_str(data.get("policy")) or "requeue"
        policy = raw_policy if raw_policy in _VALID_POLICIES else "requeue"
        deadline_at = _optional_str(data.get("deadline_at"))
        deadline_epoch_s = _optional_float(data.get("deadline_epoch_s"))
        if deadline_epoch_s is None and deadline_at is not None:
            deadline_epoch_s = _parse_iso_epoch(deadline_at)
        return cls(
            reason=raw_reason,
            requested_by=_optional_str(data.get("requested_by")),
            deadline_at=deadline_at,
            deadline_epoch_s=deadline_epoch_s,
            grace_seconds=_non_negative_int(data.get("grace_seconds"), default=0),
            policy=policy,
            checkpoint_uri=_optional_str(data.get("checkpoint_uri")),
            workload_id=_optional_str(data.get("workload_id")),
        )


def write_reclaim_notice(path: str | os.PathLike[str], notice: PreemptionNotice | tp.Mapping[str, object]) -> Path:
    """Atomically write a reclaim notice JSON file with ``0600`` permissions."""

    payload = notice.to_dict() if isinstance(notice, PreemptionNotice) else dict(notice)
    return _atomic_write_json_0600(Path(path), payload)


def _read_notice_file(environ: tp.Mapping[str, str]) -> PreemptionNotice | None:
    raw_path = environ.get(RECLAIM_NOTICE_PATH_ENV, "").strip()
    if not raw_path:
        return None
    try:
        raw_text = Path(raw_path).read_text(encoding="utf-8")
        document = json.loads(raw_text)
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(document, dict):
        return None
    notice = PreemptionNotice.from_dict(document)
    if notice is None:
        return None
    if notice.workload_id is None:
        workload_id = _optional_str(environ.get("EYVAN_WORKLOAD_ID"))
        if workload_id is not None:
            return PreemptionNotice(
                reason=notice.reason,
                requested_by=notice.requested_by,
                deadline_at=notice.deadline_at,
                deadline_epoch_s=notice.deadline_epoch_s,
                grace_seconds=notice.grace_seconds,
                policy=notice.policy,
                checkpoint_uri=notice.checkpoint_uri,
                workload_id=workload_id,
            )
    return notice


def read_checkpoint_receipt(path: str | os.PathLike[str]) -> dict[str, tp.Any] | None:
    """Read and validate one checkpoint receipt file, or return ``None`` if absent/invalid."""

    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(document, dict):
        return None
    uri = _optional_str(document.get("checkpoint_uri"))
    if uri is None:
        return None
    return document


def preemption_notice(environment: tp.Mapping[str, str] | None = None) -> PreemptionNotice | None:
    """Return the active reclaim/preemption notice, or ``None`` if none is active.

    Reads the atomic JSON notice file at ``EYVAN_RECLAIM_NOTICE_PATH`` when
    present and falls back to the signal-recorded notice captured by
    :func:`install_preemption_handler`. Never raises on missing or malformed
    notice files.
    """

    environ = os.environ if environment is None else environment
    file_notice = _read_notice_file(environ)
    if file_notice is not None:
        return file_notice
    return _signal_notice


def reclaim_requested(environment: tp.Mapping[str, str] | None = None) -> bool:
    """Fast O(1) check for whether a graceful reclaim or preemption has been requested."""

    if _signal_notice is not None:
        return True
    environ = os.environ if environment is None else environment
    raw_path = environ.get(RECLAIM_NOTICE_PATH_ENV, "").strip()
    if not raw_path:
        return False
    return _read_notice_file(environ) is not None


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal {signum}"


def _handle_preemption_signal(signum: int, frame: FrameType | None) -> None:
    """Record a live notice on ``SIGUSR1``/``SIGTERM`` instead of killing Python mid-step."""

    global _signal_notice
    enriched = _read_notice_file(os.environ)
    if enriched is not None:
        notice = enriched
    elif _signal_notice is not None:
        notice = _signal_notice
    else:
        sig_name = _signal_name(signum)
        workload_id = _optional_str(os.environ.get("EYVAN_WORKLOAD_ID"))
        notice = PreemptionNotice(
            reason=f"received {sig_name}",
            requested_by=None,
            deadline_at=None,
            deadline_epoch_s=None,
            grace_seconds=0,
            policy="requeue",
            checkpoint_uri=_optional_str(os.environ.get("EYVAN_CHECKPOINT_URI")),
            workload_id=workload_id,
        )
    _signal_notice = notice
    for callback in tuple(_callbacks):
        try:
            callback(notice)
        except Exception:
            pass
    previous = _previous_handlers.get(signum)
    if callable(previous) and previous is not _handle_preemption_signal:
        previous(signum, frame)


def install_preemption_handler(
    callback: tp.Callable[[PreemptionNotice], tp.Any] | None = None,
) -> None:
    """Install a safe, idempotent, chaining ``SIGUSR1``/``SIGTERM`` preemption handler.

    When a graceful reclaim signal arrives, this records a live
    :class:`PreemptionNotice` (enriched from ``EYVAN_RECLAIM_NOTICE_PATH`` if
    present), invokes any registered ``callback(notice)``, and chains to any
    previously registered callable Python handler instead of terminating the
    interpreter mid-step.

    Calling this function multiple times is idempotent and never double-wraps
    signal handlers. Calling it from a non-main thread still registers
    ``callback`` without raising.
    """

    with _install_lock:
        if callback is not None and callback not in _callbacks:
            _callbacks.append(callback)
        if threading.current_thread() is not threading.main_thread():
            return
        targets: list[int] = [signal.SIGTERM]
        sigusr1 = getattr(signal, "SIGUSR1", None)
        if isinstance(sigusr1, int):
            targets.append(sigusr1)
        for signum in targets:
            if signum in _installed_signals:
                continue
            try:
                previous = signal.getsignal(signum)
                signal.signal(signum, _handle_preemption_signal)
            except (OSError, ValueError):
                continue
            _previous_handlers[signum] = previous
            _installed_signals.add(signum)


def uninstall_preemption_handler() -> None:
    """Restore previous signal handlers and clear live signal state (primarily for tests)."""

    global _signal_notice
    with _install_lock:
        if threading.current_thread() is threading.main_thread():
            for signum in tuple(_installed_signals):
                previous = _previous_handlers.get(signum, signal.SIG_DFL)
                try:
                    signal.signal(signum, previous)
                except (OSError, ValueError):
                    pass
            _installed_signals.clear()
            _previous_handlers.clear()
        _callbacks.clear()
        _signal_notice = None


def acknowledge_checkpoint(
    uri: str,
    *,
    step: int | None = None,
    environment: tp.Mapping[str, str] | None = None,
) -> Path | None:
    """Record a durable checkpoint URI so Eyvan can resume the next attempt from it.

    Atomically writes ``{"checkpoint_uri": uri, "step": step, "recorded_at": ...}``
    (mode ``0600``) to ``EYVAN_CHECKPOINT_RECEIPT_PATH`` when set, and also
    reports progress on the in-job progress channel if active.

    Args:
        uri: Non-empty durable checkpoint URI (for example ``gs://bucket/ckpt-120``).
        step: Optional training step associated with the saved checkpoint.
        environment: Environment mapping to read (defaults to ``os.environ``).

    Returns:
        The written receipt :class:`~pathlib.Path`, or ``None`` when
        ``EYVAN_CHECKPOINT_RECEIPT_PATH`` is unset.

    Raises:
        ValueError: If ``uri`` is empty or ``step`` is negative.
    """

    cleaned = uri.strip() if isinstance(uri, str) else ""
    if not cleaned:
        raise ValueError("checkpoint uri must be a non-empty string")
    if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
        raise ValueError("checkpoint step must be a non-negative integer")

    environ = os.environ if environment is None else environment
    written_path: Path | None = None
    raw_receipt_path = environ.get(CHECKPOINT_RECEIPT_PATH_ENV, "").strip()
    if raw_receipt_path:
        payload: dict[str, object] = {
            "checkpoint_uri": cleaned,
            "step": step,
            "recorded_at": datetime.now(UTC).isoformat(),
        }
        written_path = _atomic_write_json_0600(Path(raw_receipt_path), payload)

    if environ.get("EYVAN_PROGRESS_URL") and environ.get("EYVAN_MANAGED_RESULT_TOKEN"):
        try:
            from eyvan.worker.runtime.progress import report_progress

            report_progress(
                step=step,
                message=f"checkpoint saved: {cleaned}",
                environment=environ,
            )
        except Exception:
            pass
    return written_path


def resume_checkpoint_uri(environment: tp.Mapping[str, str] | None = None) -> str | None:
    """Return the checkpoint URI this attempt should resume from, if any.

    Checks ``EYVAN_RESUME_CHECKPOINT_URI`` first, then falls back to reading
    ``checkpoint_uri`` from ``EYVAN_CHECKPOINT_RECEIPT_PATH`` when present.
    """

    environ = os.environ if environment is None else environment
    direct = environ.get(RESUME_CHECKPOINT_URI_ENV, "").strip()
    if direct:
        return direct
    raw_receipt_path = environ.get(CHECKPOINT_RECEIPT_PATH_ENV, "").strip()
    if not raw_receipt_path:
        return None
    receipt = read_checkpoint_receipt(raw_receipt_path)
    if receipt is None:
        return None
    return _optional_str(receipt.get("checkpoint_uri"))


__all__ = (
    "CHECKPOINT_RECEIPT_PATH_ENV",
    "RECLAIM_NOTICE_PATH_ENV",
    "RESUME_CHECKPOINT_URI_ENV",
    "PreemptionNotice",
    "acknowledge_checkpoint",
    "install_preemption_handler",
    "preemption_notice",
    "read_checkpoint_receipt",
    "reclaim_requested",
    "resume_checkpoint_uri",
    "uninstall_preemption_handler",
    "write_reclaim_notice",
)
