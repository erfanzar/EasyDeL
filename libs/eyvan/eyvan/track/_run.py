# Copyright (C) 2026 Erfan Zare Chavoshi
# This file is part of Eyvan.
#
# SPDX-License-Identifier: AGPL-3.0-or-later

"""The in-job run-tracking client behind :mod:`eyvan.track`.

Two rules from :mod:`eyvan.worker.runtime.progress` apply here unchanged:

* **Standard library only.** This runs inside training jobs.
* **A monitoring channel never breaks the job it monitors.** Storage and
  transport failures are absorbed: a segment that cannot be uploaded is spilled
  to a local directory and retried, and anything that has to be dropped is
  counted and published in a ``stats`` record, so loss is visible rather than
  silent. Bad *arguments* still raise, because they are the caller's bug and
  fail the same way on every call.

Records are buffered in memory and written by a daemon thread as immutable
gzip NDJSON segments (see :mod:`eyvan.track._format`) every few seconds, or
sooner when a batch fills.
"""

from __future__ import annotations

import atexit
import bisect
import itertools
import json
import math
import os
import re
import secrets
import socket
import sys
import threading
import time
import typing as tp
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from eyvan.track import _format as fmt
from eyvan.track import _rich
from eyvan.track._names import generate_run_name
from eyvan.track._store import TrackingStore, TrackingStoreError, store_for_uri


class _UnavailableTrackingStore:
    """Fallback store used when a local path cannot be opened at ``start()`` time.

    Preserves the core guarantee that a monitoring channel never crashes the job
    it monitors: writes fail cleanly inside the background writer, spilling if a
    spill directory is writable and otherwise counting dropped records.
    """

    def __init__(self, root: str, reason: str) -> None:
        self._root = root
        self._reason = reason

    @property
    def root(self) -> str:
        return self._root

    def put(self, key: str, data: bytes) -> None:
        del key, data
        raise TrackingStoreError(f"tracking store {self._root!r} is unavailable: {self._reason}")

    def get(self, key: str) -> bytes:
        del key
        raise TrackingStoreError(f"tracking store {self._root!r} is unavailable: {self._reason}")

    def list(self, prefix: str = "") -> list[str]:
        del prefix
        raise TrackingStoreError(f"tracking store {self._root!r} is unavailable: {self._reason}")


def _safe_hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "unknown"


#: Local store used when a run starts outside an Eyvan job with no ``uri``.
DEFAULT_LOCAL_STORE = "~/.eyvan/tracking"
#: Where segments that could not be uploaded wait for a retry or ``eyvan track sync``.
DEFAULT_SPILL_DIR = "~/.eyvan/tracking-spill"
#: Organization segment used for laptop runs, matching the plane's default org.
DEFAULT_ORG = "default"

_KEY = re.compile(r"[A-Za-z0-9_./-]{1,128}")
_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_NAME = 256
_MAX_TAGS = 64
_MAX_JSON_DEPTH = 32
_STATS_INTERVAL_S = 60.0
_DEFAULT_FINISH_TIMEOUT_S = 30.0

Clock: tp.TypeAlias = tp.Callable[[], float]


@dataclass(frozen=True, slots=True)
class TrackLimits:
    """Per-run bounds that keep user code from growing tracking without limit.

    Attributes:
        max_keys: Distinct scalar keys accepted; values for further new keys
            are dropped and counted under ``dropped.keys``.
        max_records_per_s: ``log`` calls accepted per wall-clock second per
            rank; excess calls are counted under ``dropped.rate``.
        max_queued: Records buffered in memory before new ones are dropped and
            counted under ``dropped.queue``.
        batch_records: Buffered scalar records that trigger an early flush.
    """

    max_keys: int = 2000
    max_records_per_s: int = 500
    max_queued: int = 50_000
    batch_records: int = 1000
    #: Distinct histogram keys accepted; further new keys count under ``dropped.keys``.
    max_hist_keys: int = 256
    #: ``episode`` calls accepted per wall-clock second per process (actors are chatty).
    max_episodes_per_s: int = 2000
    #: Distinct table keys and, separately, media keys; further new keys count under ``dropped.keys``.
    max_media_keys: int = 256
    #: Total distinct blob bytes (media, videos, table blobs) one writer uploads;
    #: blobs past it are dropped and counted under ``dropped.queue``.
    max_blob_bytes: int = 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in (
            "max_keys",
            "max_records_per_s",
            "max_queued",
            "batch_records",
            "max_hist_keys",
            "max_episodes_per_s",
            "max_media_keys",
            "max_blob_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """Outcome of :func:`replay_spill`."""

    replayed: int
    failed: int
    errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _Spilled:
    marker: Path
    payload: Path
    key: str
    records: int


@dataclass(slots=True)
class _Upload:
    key: str
    data: bytes
    records: int
    blob: bool = False
    done: bool = False
    retained: bool = False


@dataclass(slots=True)
class _UploadBatch:
    payloads: list[_Upload]
    final: bool = False
    done: threading.Event = field(default_factory=threading.Event)
    clean: bool = True
    expired: bool = False
    terminal: str | None = None
    finalized: bool = False
    wait_for: tuple[_UploadBatch, ...] = ()
    retain_lock: tp.Any = field(default_factory=threading.Lock)


_MAX_EXTRA = 32
_MAX_ACTOR = 64
_MAX_ENV = 128
_MAX_CHECKPOINT = 256


def _empty_buffers() -> dict[str, list[dict[str, object]]]:
    return {stream: [] for stream in fmt.STREAMS}


def _optional_text(value: object, name: str, limit: int) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, str | int):
        raise TypeError(f"{name} must be a string")
    text = str(value)
    if not text or len(text) > limit:
        raise ValueError(f"{name} must be 1-{limit} characters")
    return text


def _optional_count(value: object, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        number = int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if number != value and not hasattr(value, "item"):
        raise TypeError(f"{name} must be an integer")
    if number < 0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _flatten(values: object) -> list[float]:
    """Flatten nested sequences (or anything with ``tolist``) into floats."""

    tolist = getattr(values, "tolist", None)
    if callable(tolist):
        values = tolist()
    if isinstance(values, int | float) and not isinstance(values, bool):
        return [float(values)]
    out: list[float] = []
    stack: list[object] = [values]
    while stack:
        item = stack.pop()
        if isinstance(item, list | tuple):
            stack.extend(reversed(item))
        elif isinstance(item, bool):
            out.append(1.0 if item else 0.0)
        elif isinstance(item, int | float):
            out.append(float(item))
        elif isinstance(item, str | bytes) or not hasattr(item, "__iter__"):
            out.append(_number("histogram", item))
        else:
            stack.extend(reversed(list(item)))  # type: ignore[call-overload]
    return out


def _edges(lo: float, hi: float, bins: int) -> list[float]:
    width = hi - lo
    # Weighted interpolation avoids overflowing hi - lo for opposite extremes.
    edges = [
        lo + width * (i / bins) if math.isfinite(width) else lo * (1 - i / bins) + hi * (i / bins)
        for i in range(bins + 1)
    ]
    edges[0], edges[-1] = lo, hi
    # Very narrow explicit ranges may contain fewer representable floats than
    # requested bins. Coalesce rounded duplicates rather than publish bad edges.
    return list(dict.fromkeys(edges))


def _bin_values(values: object, bins: int, value_range: tuple[float, float] | None) -> dict[str, object]:
    """Count against one canonical edge array for Python iterables and arrays alike."""

    flat = _flatten(values)
    finite_values = [v for v in flat if math.isfinite(v)]
    lo, hi = _resolve_range(
        value_range, min(finite_values) if finite_values else None, max(finite_values) if finite_values else None, bins
    )
    edges = _edges(lo, hi, bins)
    counts = [0] * (len(edges) - 1)
    under = over = 0
    for value in finite_values:
        if value < lo:
            under += 1
        elif value > hi:
            over += 1
        else:
            counts[min(len(counts) - 1, bisect.bisect_right(edges, value) - 1)] += 1
    stats = _moments(finite_values) if finite_values else (None, None, None, None)
    return {
        "edges": edges,
        "counts": counts,
        "under": under,
        "over": over,
        "nonfinite": len(flat) - len(finite_values),
        **dict(zip(("min", "max", "mean", "std"), stats, strict=True)),
    }


def _resolve_range(
    value_range: tuple[float, float] | None, low: float | None, high: float | None, bins: int
) -> tuple[float, float]:
    if value_range is not None:
        return value_range
    if low is None or high is None:
        return 0.0, 1.0
    if low == high:
        padding = max(0.5, math.ulp(low) * bins)
        lo, hi = low - padding, high + padding
        if not math.isfinite(lo):
            lo = low
        if not math.isfinite(hi):
            hi = high
        return lo, hi
    return low, high


def _moments(values: list[float]) -> tuple[float, float, float, float]:
    scale = max(abs(v) for v in values)
    if not scale:
        return min(values), max(values), 0.0, 0.0
    scaled = [v / scale for v in values]
    mean = math.fsum(scaled) / len(scaled)
    variance = math.fsum((v - mean) ** 2 for v in scaled) / len(scaled)
    # Population std and mean cannot exceed max(abs(values)); clamp rounding.
    return min(values), max(values), max(-1.0, min(1.0, mean)) * scale, min(1.0, math.sqrt(variance)) * scale


def _check_key(key: object, what: str = "metric key") -> str:
    if not isinstance(key, str) or not _KEY.fullmatch(key):
        raise ValueError(f"invalid {what} {key!r}: use 1-128 characters from letters, digits, '_', '.', '/', '-'")
    return key


def _number(key: str, value: object) -> float:
    """Convert one logged value to a float, accepting numpy/JAX/torch scalars."""

    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(f"value for {key!r} must be a number or a 0-d array, got {type(value).__name__}") from exc


def _jsonable(value: object, depth: int = 0) -> object:
    """Return a strict-JSON-safe copy of a config or summary value."""

    if depth > _MAX_JSON_DEPTH:
        return repr(value)[:1000]
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return fmt.encode_value(value)
    if isinstance(value, tp.Mapping):
        return {str(k): _jsonable(v, depth + 1) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_jsonable(item, depth + 1) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item(), depth + 1)
        except Exception:
            pass
    return repr(value)[:1000]


def _env_int(environment: tp.Mapping[str, str], names: tuple[str, ...], default: int, ceiling: int) -> int:
    for name in names:
        raw = environment.get(name)
        if raw is None:
            continue
        try:
            value = int(raw)
        except ValueError:
            continue
        if 0 <= value <= ceiling:
            return value
    return default


def _fresh_run_id(name: str | None) -> str:
    slug = _SLUG_UNSAFE.sub("-", name or "run").strip("-._")[:80] or "run"
    return fmt.validate_run_id(f"{slug}-{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}")


_UNCAUGHT = False
_HOOK_INSTALLED = False
_HOOK_LOCK = threading.Lock()


def _install_excepthook() -> None:
    """Chain ``sys.excepthook`` once so an exit after an uncaught error finishes as failed."""

    global _HOOK_INSTALLED
    with _HOOK_LOCK:
        if _HOOK_INSTALLED:
            return
        previous = sys.excepthook

        def hook(exc_type: type[BaseException], exc: BaseException, traceback: tp.Any) -> None:
            global _UNCAUGHT
            _UNCAUGHT = True
            previous(exc_type, exc, traceback)

        sys.excepthook = hook
        _HOOK_INSTALLED = True


class Run:
    """One tracked run as seen by this process (one attempt of one rank).

    Create it with :func:`eyvan.track.start`. All methods are thread-safe.
    Scalars, histograms, completions, tables and media follow ``ranks``;
    episodes and rollouts independently follow ``episode_ranks``. Inactive
    methods validate arguments but record nothing, without counting inactivity
    as loss. After :meth:`finish`, recording calls are unconditional no-ops.
    """

    def __init__(
        self,
        *,
        run_id: str,
        name: str,
        group: str | None,
        tags: tuple[str, ...],
        store: TrackingStore,
        spill_dir: Path,
        attempt: int,
        rank: int,
        workload_id: str | None,
        active: bool,
        limits: TrackLimits,
        flush_interval_s: float,
        clock: Clock,
        episodes_active: bool | None = None,
    ) -> None:
        self.id = run_id
        self.name = name
        self.group = group
        self.tags = tags
        self.attempt = attempt
        self.rank = rank
        self.workload_id = workload_id
        self.active = active
        #: Episodes may be recorded on ranks that record nothing else (actors).
        self.episodes_active = active if episodes_active is None else episodes_active
        self._store = store
        self._spill_dir = spill_dir
        self._limits = limits
        self._flush_interval_s = flush_interval_s
        self._clock = clock
        self._writer = secrets.token_hex(6)
        self._lock = threading.Lock()
        self._spill_lock = threading.Lock()
        self._batches: deque[_UploadBatch] = deque()
        self._inflight: list[_UploadBatch] = []
        self._upload_pending = 0
        self._buffers: dict[str, list[dict[str, object]]] = _empty_buffers()
        self._pending = 0
        self._keys: set[str] = set()
        self._hist_keys: set[str] = set()
        self._hist_rate = (-1, 0)
        self._episode_rate = (-1, 0)
        self._rates: dict[str, tuple[int, int]] = {}
        self._rollout_calls = 0
        self._completion_groups = 0
        self._table_keys: set[str] = set()
        self._media_keys: set[str] = set()
        self._blob_queue: list[tuple[str, bytes]] = []
        self._blob_shas: set[str] = set()
        self._blob_total = 0
        self._config: dict[str, object] = {}
        self._next_step = 0
        self._rate_second = -1
        self._rate_count = 0
        self._records = 0
        self._spilled = 0
        self._dropped = {"keys": 0, "rate": 0, "invalid": 0, "queue": 0}
        self._stats_dirty = False
        self._last_stats = clock()
        self._seq = 0
        self._spill_backlog: list[_Spilled] = []
        self._finished = False
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        if not self._writing:
            return
        if active:
            self._buffers[fmt.STREAM_META].append(
                {
                    "op": fmt.META_START,
                    "t": clock(),
                    "fmt": fmt.FORMAT_VERSION,
                    "name": name,
                    "group": group,
                    "tags": list(tags),
                    "workload_id": workload_id,
                    "attempt": attempt,
                    "rank": rank,
                    "host": _safe_hostname(),
                    "pid": os.getpid(),
                }
            )
            self._pending = 1
        _install_excepthook()
        atexit.register(self._at_exit)
        self._thread = threading.Thread(target=self._loop, name=f"eyvan-track-{run_id}", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ public

    @property
    def finished(self) -> bool:
        """Return whether :meth:`finish` has run."""

        return self._finished

    @property
    def _writing(self) -> bool:
        return self.active or self.episodes_active

    @property
    def store_root(self) -> str:
        """Return the scope root this run writes under."""

        return self._store.root

    def log(
        self,
        values: tp.Mapping[str, object],
        step: int | None = None,
        *,
        axes: tp.Mapping[str, object] | None = None,
    ) -> None:
        """Record scalar values at ``step``.

        Args:
            values: Metric names to numbers (Python, numpy, JAX or torch 0-d
                values). NaN and infinity are kept and shown, not dropped.
            step: Non-negative step. Omitted, it is one past the largest step
                logged so far by this process.
            axes: Extra x-axis values for this record, such as
                ``{"env_steps": frames}``. A non-finite axis value is dropped
                and counted under ``dropped.invalid``.

        Raises:
            TypeError: If ``values`` or ``axes`` is not a mapping, or a value
                is not numeric.
            ValueError: If a key is malformed or ``step`` is negative.
        """

        if self._finished:
            return
        if not isinstance(values, tp.Mapping):
            raise TypeError("values must be a mapping of metric names to numbers")
        numbers = {_check_key(key): _number(key, value) for key, value in values.items()}
        if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
            raise ValueError("step must be a non-negative integer")
        x: dict[str, float] = {}
        invalid_axes = 0
        if axes is not None:
            if not isinstance(axes, tp.Mapping):
                raise TypeError("axes must be a mapping of axis names to numbers")
            for axis, raw in axes.items():
                value = _number(_check_key(axis, "axis name"), raw)
                if math.isfinite(value):
                    x[axis] = value
                else:
                    invalid_axes += 1
        if not self.active or self._finished:
            return
        flush_now = False
        with self._lock:
            if self._finished:
                return
            self._stats_dirty = True
            self._dropped["invalid"] += invalid_axes
            now = self._clock()
            if step is None:
                step = self._next_step
            self._next_step = max(self._next_step, step + 1)
            second = int(now)
            if second != self._rate_second:
                self._rate_second, self._rate_count = second, 0
            if self._rate_count >= self._limits.max_records_per_s:
                self._dropped["rate"] += 1
                return
            self._rate_count += 1
            kept: dict[str, float | str] = {}
            for key, value in numbers.items():
                if key not in self._keys:
                    if len(self._keys) >= self._limits.max_keys:
                        self._dropped["keys"] += 1
                        continue
                    self._keys.add(key)
                kept[key] = fmt.encode_value(value)
            if not kept:
                return
            if not self._queue_ok_locked():
                return
            record: dict[str, object] = {"step": step, "t": now, "v": kept}
            if x:
                record["x"] = x
            buffer = self._buffers[fmt.STREAM_SCALARS]
            buffer.append(record)
            self._pending += 1
            self._records += 1
            flush_now = len(buffer) >= self._limits.batch_records
        if flush_now:
            self._wake.set()

    def histogram(
        self,
        key: str,
        values: object = None,
        *,
        step: int | None = None,
        bins: int = 64,
        range: tuple[float, float] | None = None,  # noqa: A002 - mirrors numpy.histogram
        counts: tp.Sequence[object] | None = None,
        edges: tp.Sequence[object] | None = None,
        axes: tp.Mapping[str, object] | None = None,
    ) -> None:
        """Record a distribution at ``step``, binned here so only counts travel.

        Pass raw ``values`` (any nested sequence, iterable or array with
        ``tolist``), or pre-binned ``counts`` with their ``edges``. Values
        outside ``range`` are counted as under/over; NaN and infinity are
        counted, never binned. Narrow ranges with fewer representable floats
        than requested bins use fewer bins rather than duplicate edges.

        Raises:
            TypeError: For non-numeric values or malformed arguments.
            ValueError: For a bad key, step, bin count, range or edges.
        """

        if self._finished:
            return
        _check_key(key, "histogram key")
        if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
            raise ValueError("step must be a non-negative integer")
        if (values is None) == (counts is None):
            raise ValueError("pass either values or counts (with edges), not both")
        if counts is not None:
            if edges is None:
                raise ValueError("counts require edges")
            edge_list = [_number("edges", edge) for edge in edges]
            count_list = [_optional_count(count, "counts") for count in counts]
            if not 1 <= len(count_list) <= fmt.MAX_HIST_BINS or len(edge_list) != len(count_list) + 1:
                raise ValueError(f"need 1-{fmt.MAX_HIST_BINS} counts and exactly one more edge than counts")
            if any(not math.isfinite(e) for e in edge_list) or any(b <= a for a, b in itertools.pairwise(edge_list)):
                raise ValueError("edges must be finite and strictly increasing")
            binned: dict[str, object] = {
                "edges": edge_list,
                "counts": count_list,
                "under": 0,
                "over": 0,
                "nonfinite": 0,
                "min": None,
                "max": None,
                "mean": None,
                "std": None,
            }
        else:
            if isinstance(bins, bool) or not isinstance(bins, int) or not 1 <= bins <= fmt.MAX_HIST_BINS:
                raise ValueError(f"bins must be an integer from 1 through {fmt.MAX_HIST_BINS}")
            value_range = None
            if range is not None:
                lo, hi = (_number("range", bound) for bound in range)
                if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
                    raise ValueError("range must be two finite numbers with low < high")
                value_range = (lo, hi)
            binned = _bin_values(values, bins, value_range)
        x = self._axes(axes)
        if not self.active or self._finished:
            return
        with self._lock:
            if self._finished:
                return
            self._stats_dirty = True
            self._dropped["invalid"] += x[1]
            now = self._clock()
            if step is None:
                step = max(self._next_step - 1, 0)
            second = int(now)
            window, used = self._hist_rate
            if second != window:
                window, used = second, 0
            if used >= self._limits.max_records_per_s:
                self._dropped["rate"] += 1
                return
            self._hist_rate = (window, used + 1)
            if key not in self._hist_keys:
                if len(self._hist_keys) >= self._limits.max_hist_keys:
                    self._dropped["keys"] += 1
                    return
                self._hist_keys.add(key)
            record: dict[str, object] = {"key": key, "step": step, "t": now, **binned}
            record["n"] = sum(tp.cast(list[int], binned["counts"]))
            if x[0]:
                record["x"] = x[0]
            self._enqueue_locked(fmt.STREAM_HIST, record)

    def episode(
        self,
        ret: object,
        length: int | None = None,
        *,
        success: bool | None = None,
        kind: str = "train",
        actor: str | int | None = None,
        env: str | None = None,
        seed: int | None = None,
        policy_version: int | None = None,
        learner_version: int | None = None,
        checkpoint: str | None = None,
        step: int | None = None,
        axes: tp.Mapping[str, object] | None = None,
        extra: tp.Mapping[str, object] | None = None,
    ) -> None:
        """Record one finished RL episode.

        Accepted from every rank by default, so actors on other hosts can
        report their own episodes. ``step`` defaults to the learner step this
        process last logged, which places the episode on training curves and
        decides whether a later attempt replayed it.

        Args:
            ret: Episode return (any numeric; NaN/inf kept).
            length: Episode length in environment steps.
            success: Task success, for success-rate views.
            kind: ``"train"`` or ``"eval"``.
            actor: Actor id, for per-actor views.
            env: Environment name.
            seed: Environment seed.
            policy_version: Policy version that generated the episode.
            learner_version: Learner version at the time it was reported, for
                actor lag (``learner_version - policy_version``).
            checkpoint: Checkpoint name, for evaluation tables.
            step: Learner step to attach the episode to.
            axes: Extra x-axis values such as ``{"env_steps": frames}``.
            extra: Up to 32 additional numeric fields.

        Raises:
            TypeError: For non-numeric values or wrong field types.
            ValueError: For an unknown kind, negative counts or bad names.
        """

        if self._finished:
            return
        value = _number("ret", ret)
        if kind not in fmt.EPISODE_KINDS:
            raise ValueError(f"kind must be one of {', '.join(fmt.EPISODE_KINDS)}")
        if success is not None and not isinstance(success, bool):
            raise TypeError("success must be a boolean")
        if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
            raise ValueError("step must be a non-negative integer")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise TypeError("seed must be an integer")
        fields: dict[str, object] = {
            "length": _optional_count(length, "length"),
            "success": success,
            "kind": kind,
            "actor": _optional_text(actor, "actor", _MAX_ACTOR),
            "env": _optional_text(env, "env", _MAX_ENV),
            "seed": seed,
            "policy_version": _optional_count(policy_version, "policy_version"),
            "learner_version": _optional_count(learner_version, "learner_version"),
            "checkpoint": _optional_text(checkpoint, "checkpoint", _MAX_CHECKPOINT),
        }
        extras: dict[str, float | str] = {}
        if extra is not None:
            if not isinstance(extra, tp.Mapping):
                raise TypeError("extra must be a mapping of names to numbers")
            if len(extra) > _MAX_EXTRA:
                raise ValueError(f"extra accepts at most {_MAX_EXTRA} fields")
            extras = {_check_key(k, "extra field"): fmt.encode_value(_number(k, v)) for k, v in extra.items()}
        x = self._axes(axes)
        if not self.episodes_active or self._finished:
            return
        with self._lock:
            if self._finished:
                return
            self._stats_dirty = True
            self._dropped["invalid"] += x[1]
            now = self._clock()
            second = int(now)
            window, used = self._episode_rate
            if second != window:
                window, used = second, 0
            if used >= self._limits.max_episodes_per_s:
                self._dropped["rate"] += 1
                return
            self._episode_rate = (window, used + 1)
            if step is None and self._next_step > 0:
                step = self._next_step - 1
            record: dict[str, object] = {"t": now, "step": step, "ret": fmt.encode_value(value), **fields}
            record["x"] = x[0]
            record["extra"] = extras
            self._enqueue_locked(fmt.STREAM_EPISODES, record)

    def rollout(
        self,
        steps: object = None,
        *,
        tracks: tp.Mapping[str, object] | None = None,
        actions: object = None,
        action_names: object = None,
        every: int = 1,
        episode_id: str | int | None = None,
        actor: str | int | None = None,
        env: str | None = None,
        kind: str = "train",
        policy_version: int | None = None,
        video: object = None,
        fps: float | None = None,
        video_type: str = "video/mp4",
        step: int | None = None,
        axes: tp.Mapping[str, object] | None = None,
    ) -> bool:
        """Record a sampled trajectory: per-step tracks, actions and optional video.

        Pass ``steps`` as a list of mappings (``{"reward": r, "value": v,
        "action": a}``), or ``tracks`` (name -> sequence) and/or ``actions``.
        Only one call in ``every`` is kept, so this can be called for every
        episode and stays cheap; validation still runs on every call.

        Returns:
            Whether this call was sampled and queued.

        Raises:
            TypeError: For non-numeric values or wrong field types.
            ValueError: For mismatched lengths, bad names, unknown kinds or media types.
        """

        if self._finished:
            return False
        if isinstance(every, bool) or not isinstance(every, int) or every < 1:
            raise ValueError("every must be a positive integer")
        if kind not in fmt.EPISODE_KINDS:
            raise ValueError(f"kind must be one of {', '.join(fmt.EPISODE_KINDS)}")
        if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
            raise ValueError("step must be a non-negative integer")
        body = _rich.rollout_body(steps, tracks, actions, action_names)
        fields: dict[str, object] = {
            "episode_id": _rich.label(episode_id, "episode_id"),
            "actor": _optional_text(actor, "actor", _MAX_ACTOR),
            "env": _optional_text(env, "env", _MAX_ENV),
            "kind": kind,
            "policy_version": _optional_count(policy_version, "policy_version"),
        }
        fps_value: float | None = None
        if fps is not None:
            fps_value = _number("fps", fps)
            if not (math.isfinite(fps_value) and fps_value > 0):
                raise ValueError("fps must be a positive finite number")
        media: tuple[bytes, str] | None = None
        if video is not None:
            ctype = _rich.media_type(video, None, default=video_type)
            if not ctype.startswith("video/"):
                raise ValueError("video must be a video media type")
            media = self._read_media(video, ctype, active=self.episodes_active)
            if media is None:
                return False
        x = self._axes(axes)
        if not self.episodes_active or self._finished:
            return False
        with self._lock:
            if self._finished:
                return False
            self._stats_dirty = True
            self._dropped["invalid"] += x[1]
            self._rollout_calls += 1
            if (self._rollout_calls - 1) % every:
                return False
            now = self._clock()
            if not self._rate_ok_locked("rollout", self._limits.max_records_per_s, now):
                return False
            if not self._queue_ok_locked():
                return False
            ref = self._enqueue_blob_locked(*media) if media is not None else None
            if media is not None and ref is None:
                return False
            if fields["episode_id"] is None:
                fields["episode_id"] = f"{self.attempt}-{self.rank}-{self._rollout_calls}"
            record: dict[str, object] = {
                "t": now,
                "step": self._default_step_locked(step),
                **fields,
                **body,
                "video": ref,
                "fps": fps_value,
                "x": x[0],
            }
            return self._enqueue_locked(fmt.STREAM_ROLLOUTS, record)

    def completions(
        self,
        prompt: str,
        responses: tp.Sequence[str],
        *,
        rewards: object = None,
        advantages: object = None,
        group: str | int | None = None,
        answer: str | None = None,
        tokens: object = None,
        kl: object = None,
        truncated: object = None,
        policy_version: int | None = None,
        step: int | None = None,
    ) -> None:
        """Record one LLM-RL completion group (GRPO-style).

        ``rewards`` is either one total per response or a mapping of reward
        components (each one value per response) whose sum is the total.
        When ``advantages`` is omitted they are group-normalized here:
        ``(r - mean) / std``, and all zero for a group with no reward variance.
        Texts longer than 32768 characters are cut and counted under
        ``dropped.invalid``. Like scalars, completions follow ``ranks`` (not
        ``episode_ranks``): nonzero ranks with ``ranks="zero"`` intentionally
        record nothing and do not count that inactivity as a drop.

        Raises:
            TypeError: For wrong field types.
            ValueError: For an empty or oversized group or mismatched lengths.
        """

        if self._finished:
            return
        if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
            raise ValueError("step must be a non-negative integer")
        body, clipped = _rich.completions_body(
            prompt,
            responses,
            rewards=rewards,
            advantages=advantages,
            answer=answer,
            tokens=tokens,
            kl=kl,
            truncated=truncated,
        )
        group_id = _rich.label(group, "group")
        version = _optional_count(policy_version, "policy_version")
        if not self.active or self._finished:
            return
        with self._lock:
            if self._finished:
                return
            self._stats_dirty = True
            self._dropped["invalid"] += clipped
            now = self._clock()
            if not self._rate_ok_locked("completions", self._limits.max_records_per_s, now):
                return
            self._completion_groups += 1
            record: dict[str, object] = {
                "t": now,
                "step": self._default_step_locked(step),
                "group": group_id or f"g{self.attempt}-{self.rank}-{self._completion_groups}",
                "policy_version": version,
                **body,
            }
            self._enqueue_locked(fmt.STREAM_COMPLETIONS, record)

    def table(
        self,
        key: str,
        rows: object,
        *,
        columns: tp.Sequence[str] | None = None,
        step: int | None = None,
    ) -> None:
        """Record a small table (at most 1000 rows, 64 columns) at ``step``.

        Rows are sequences (with ``columns``) or mappings (columns inferred).
        Cells may be numbers, strings, booleans, ``None`` or a BlobRef from
        :meth:`blob`. Long strings are cut to 4096 characters and counted.

        Raises:
            TypeError: For unsupported cell types.
            ValueError: For a bad key, too many rows/columns or ragged rows.
        """

        if self._finished:
            return
        _rich.check_name(key, "table key")
        if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
            raise ValueError("step must be a non-negative integer")
        names, cells, clipped = _rich.table_body(rows, columns)
        if not self.active or self._finished:
            return
        with self._lock:
            if self._finished:
                return
            self._stats_dirty = True
            self._dropped["invalid"] += clipped
            now = self._clock()
            if not self._rate_ok_locked("table", self._limits.max_records_per_s, now):
                return
            if not self._key_ok_locked(self._table_keys, key):
                return
            record = {"t": now, "step": self._default_step_locked(step), "key": key, "columns": names, "rows": cells}
            self._enqueue_locked(fmt.STREAM_TABLES, record)

    def media(
        self,
        key: str,
        data_or_path: object,
        *,
        content_type: str | None = None,
        caption: str | None = None,
        step: int | None = None,
    ) -> dict[str, object] | None:
        """Upload an image, video or audio file and record it under ``key``.

        Bytes are stored once by SHA-256, so logging the same file again costs
        nothing. Only allowlisted media types are accepted.

        Returns:
            The BlobRef, or ``None`` when nothing was recorded (inactive rank,
            finished run, or a cap was hit — the cap is counted).

        Raises:
            TypeError: For data that is neither bytes nor a path.
            ValueError: For a bad key, unsupported type, or a file over 64 MiB.
        """

        if self._finished:
            return None
        _rich.check_name(key, "media key")
        if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
            raise ValueError("step must be a non-negative integer")
        text = None if caption is None else _rich.clip_text(caption, "caption", _rich.MAX_CAPTION)
        media = self._read_media(data_or_path, content_type, active=self.active)
        if media is None:
            return None
        data, ctype = media
        with self._lock:
            if self._finished:
                return
            self._stats_dirty = True
            if text is not None:
                self._dropped["invalid"] += int(text[1])
            now = self._clock()
            if not self._rate_ok_locked("media", self._limits.max_records_per_s, now):
                return None
            if not self._key_ok_locked(self._media_keys, key):
                return None
            if not self._queue_ok_locked():
                return None
            ref = self._enqueue_blob_locked(data, ctype)
            if ref is None:
                return None
            record = {
                "t": now,
                "step": self._default_step_locked(step),
                "key": key,
                "blob": ref,
                "caption": None if text is None else text[0],
            }
            self._enqueue_locked(fmt.STREAM_MEDIA, record)
            return dict(ref)

    def blob(self, data_or_path: object, *, content_type: str | None = None) -> dict[str, object] | None:
        """Upload media bytes without a media record, for use as a :meth:`table` cell.

        Returns:
            The BlobRef, or ``None`` when inactive or over the byte budget.
        """

        if self._finished:
            return None
        media = self._read_media(data_or_path, content_type, active=self.active)
        if media is None:
            return None
        data, ctype = media
        with self._lock:
            if self._finished:
                return
            self._stats_dirty = True
            ref = self._enqueue_blob_locked(data, ctype)
            return None if ref is None else dict(ref)

    def config(self, values: tp.Mapping[str, object]) -> None:
        """Merge ``values`` into the run configuration (later values win).

        Raises:
            TypeError: If ``values`` is not a mapping.
        """

        if self._finished:
            return
        if not isinstance(values, tp.Mapping):
            raise TypeError("config must be a mapping")
        safe = _jsonable(dict(values))
        if not self.active or self._finished:
            return
        assert isinstance(safe, dict)
        with self._lock:
            if self._finished:
                return
            self._config.update(safe)
            self._append_meta_locked({"op": fmt.META_CONFIG, "t": self._clock(), "config": dict(self._config)})

    def summary(self, **values: object) -> None:
        """Set run-level summary values such as ``best_return``; later values win."""

        if self._finished:
            return
        safe = _jsonable(values)
        if not self.active or self._finished:
            return
        with self._lock:
            if self._finished:
                return
            self._append_meta_locked({"op": fmt.META_SUMMARY, "t": self._clock(), "summary": safe})

    def flush(self, timeout: float | None = None) -> bool:
        """Write everything buffered now.

        Returns:
            ``True`` when every buffered record reached the store, ``False``
            when some had to be spilled or dropped. Never raises for storage
            failures.
        """

        if not self._writing or self._finished:
            return True
        return self._flush(force_stats=False, timeout=timeout)

    def finish(self, status: str = "finished", *, timeout: float = _DEFAULT_FINISH_TIMEOUT_S) -> bool:
        """Record a terminal status, flush, and stop the background writer.

        Calling it again is a no-op.

        Args:
            status: ``"finished"``, ``"failed"`` or ``"aborted"``.
            timeout: Seconds to wait for the final upload before spilling.

        Returns:
            Whether the final flush reached the store.

        Raises:
            ValueError: If ``status`` is not a terminal status.
        """

        if self._finished:
            return True
        if status not in fmt.FINISH_STATUSES:
            raise ValueError(f"status must be one of {', '.join(fmt.FINISH_STATUSES)}")
        with self._lock:
            if self._finished:
                return True
            self._finished = True
        if not self._writing:
            return True
        try:
            atexit.unregister(self._at_exit)
        except Exception:
            pass
        # The one deadline covers existing uploads and the final data + stats
        # writes. Store I/O is ALWAYS performed by the daemon writer, not here.
        try:
            return self._flush(force_stats=False, timeout=timeout, final=True, status=status)
        finally:
            self._stop.set()
            self._wake.set()

    def stats(self) -> dict[str, object]:
        """Return this writer's cumulative counters (records, drops, spills)."""

        with self._lock:
            return {"records": self._records, "dropped": dict(self._dropped), "spilled": self._spilled}

    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object) -> None:
        if exc_type is None:
            status = "finished"
        elif issubclass(exc_type, KeyboardInterrupt):
            status = "aborted"
        else:
            status = "failed"
        self.finish(status)

    def __repr__(self) -> str:
        return f"Run(id={self.id!r}, attempt={self.attempt}, rank={self.rank}, active={self.active})"

    # ---------------------------------------------------------------- internals

    @staticmethod
    def _axes(axes: tp.Mapping[str, object] | None) -> tuple[dict[str, float], int]:
        """Validate extra x-axes; returns finite axes and the count of dropped non-finite ones."""

        if axes is None:
            return {}, 0
        if not isinstance(axes, tp.Mapping):
            raise TypeError("axes must be a mapping of axis names to numbers")
        kept: dict[str, float] = {}
        invalid = 0
        for axis, raw in axes.items():
            value = _number(_check_key(axis, "axis name"), raw)
            if math.isfinite(value):
                kept[axis] = value
            else:
                invalid += 1
        return kept, invalid

    def _read_media(self, source: object, content_type: str | None, *, active: bool) -> tuple[bytes, str] | None:
        ctype = _rich.media_type(source, content_type)
        if not active or self._finished:
            return None
        try:
            return _rich.read_media(source, ctype)
        except OSError:
            with self._lock:
                if not self._finished:
                    self._dropped["invalid"] += 1
                    self._stats_dirty = True
            return None

    def _queue_ok_locked(self) -> bool:
        if self._pending + self._upload_pending >= self._limits.max_queued:
            self._dropped["queue"] += 1
            self._stats_dirty = True
            return False
        return True

    def _enqueue_locked(self, stream: str, record: dict[str, object]) -> bool:
        if not self._queue_ok_locked():
            return False
        buffer = self._buffers[stream]
        buffer.append(record)
        self._pending += 1
        self._records += 1
        if len(buffer) >= self._limits.batch_records:
            self._wake.set()
        return True

    def _append_meta_locked(self, record: dict[str, object]) -> None:
        if not self._queue_ok_locked():
            return
        self._buffers[fmt.STREAM_META].append(record)
        self._pending += 1

    def _stats_record_locked(self) -> dict[str, object]:
        self._stats_dirty = False
        self._last_stats = self._clock()
        return {
            "op": fmt.META_STATS,
            "t": self._last_stats,
            "records": self._records,
            "dropped": dict(self._dropped),
            "spilled": self._spilled,
        }

    def _segment_key_locked(self, stream: str) -> str:
        name = fmt.SegmentName(attempt=self.attempt, rank=self.rank, writer=self._writer, seq=self._seq)
        self._seq += 1
        return fmt.segment_key(self.id, stream, name)

    def _drain(self, *, force_stats: bool) -> _UploadBatch:
        with self._lock:
            due = self._stats_dirty and self._clock() - self._last_stats >= _STATS_INTERVAL_S
            if force_stats or due:
                self._buffers[fmt.STREAM_META].append(self._stats_record_locked())
            payloads = [_Upload(key, data, 0, blob=True) for key, data in self._blob_queue]
            for stream, records in self._buffers.items():
                if records:
                    # Reserve sequence numbers in the SAME critical section as
                    # the drain, not when an upload eventually gets scheduled.
                    key = self._segment_key_locked(stream)
                    payloads.append(_Upload(key, fmt.encode_segment(records), len(records)))
            self._blob_queue = []
            self._buffers = _empty_buffers()
            self._pending = 0
            batch = _UploadBatch(payloads)
            self._upload_pending += sum(max(1, payload.records) for payload in payloads)
            self._inflight.append(batch)
            return batch

    def _release_batch_locked(self, batch: _UploadBatch) -> None:
        if batch in self._inflight:
            self._inflight.remove(batch)
            self._upload_pending -= sum(max(1, payload.records) for payload in batch.payloads)

    def _flush(
        self, *, force_stats: bool, timeout: float | None, final: bool = False, status: str | None = None
    ) -> bool:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        batch = self._drain(force_stats=force_stats)
        batch.final, batch.terminal = final, status
        with self._lock:
            closed = self._finished and not final
            if final:
                batch.wait_for = tuple(pending for pending in self._inflight if pending is not batch)
            if not batch.expired and not closed:
                self._batches.append(batch)
            waiting = [*batch.wait_for, batch] if final else [batch]
        if closed:
            # finish may have won the race after flush's initial guard. Never
            # submit to a writer that has already completed its final drain.
            self._expire_batch(batch)
            return not batch.payloads
        self._wake.set()
        clean = True
        for pending in waiting:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not pending.done.wait(remaining):
                self._expire_batch(pending)
            clean = pending.clean and clean
        return clean

    def _retain(self, batch: _UploadBatch, payload: _Upload) -> None:
        # A timeout and the writer may both discover the same failed put. Only
        # one of them owns its spill/counters; the original bytes/key never change.
        if payload.done or payload.retained:
            return
        payload.retained = True
        if self._spill_bytes(payload.key, payload.data, payload.records):
            return
        if payload.blob:
            digest = payload.key.rsplit("/", 1)[-1]
            with self._lock:
                self._blob_shas.discard(digest)
                self._blob_total -= len(payload.data)

    def _final_payload(self, batch: _UploadBatch) -> _Upload | None:
        with self._lock:
            if not (self.active or self._stats_dirty or self._records):
                return None
            records = [self._stats_record_locked()]
            if self.active and batch.terminal is not None:
                records.append({"op": fmt.META_FINISH, "t": self._clock(), "status": batch.terminal})
            return _Upload(self._segment_key_locked(fmt.STREAM_META), fmt.encode_segment(records), len(records))

    def _expire_batch(self, batch: _UploadBatch) -> None:
        with batch.retain_lock:
            if batch.done.is_set():
                return
            batch.expired, batch.clean = True, False
            for payload in batch.payloads:
                self._retain(batch, payload)
            # This snapshot follows all data upload/spill outcomes observed at
            # the deadline, including losses from the last failed data put.
            if batch.final and not batch.finalized:
                batch.finalized = True
                if (payload := self._final_payload(batch)) is not None:
                    self._retain(batch, payload)
            with self._lock:
                self._release_batch_locked(batch)
                if batch in self._batches:
                    self._batches.remove(batch)
            batch.done.set()

    def _process_batch(self, batch: _UploadBatch) -> None:
        try:
            if not batch.expired:
                batch.clean = self._retry_backlog() and batch.clean
            for payload in batch.payloads:
                if batch.expired:
                    break
                try:
                    self._store.put(payload.key, payload.data)
                except Exception:
                    with batch.retain_lock:
                        batch.clean = False
                        self._retain(batch, payload)
                else:
                    payload.done = True
            with batch.retain_lock:
                final = self._final_payload(batch) if batch.final and not batch.expired else None
                if final is not None:
                    batch.finalized = True
                    with self._lock:
                        batch.payloads.append(final)
                        self._upload_pending += max(1, final.records)
            if final is not None:
                # Do not hold retain_lock over put: a deadline must be able to
                # retain this final metadata even when that put never returns.
                try:
                    self._store.put(final.key, final.data)
                except Exception:
                    with batch.retain_lock:
                        batch.clean = False
                        self._retain(batch, final)
                else:
                    final.done = True
        finally:
            with self._lock:
                self._release_batch_locked(batch)
            batch.done.set()

    def _enqueue_blob_locked(self, data: bytes, content_type: str) -> dict[str, object] | None:
        """Queue ``data`` for upload once per digest; ``None`` when the byte budget is spent."""

        ref = _rich.blob_ref(data, content_type)
        digest = tp.cast(str, ref["sha256"])
        if digest in self._blob_shas:
            return ref
        if (
            self._blob_total + len(data) > self._limits.max_blob_bytes
            or len(self._blob_queue) + self._upload_pending >= self._limits.max_queued
        ):
            self._dropped["queue"] += 1
            return None
        self._blob_shas.add(digest)
        self._blob_total += len(data)
        self._blob_queue.append((fmt.blob_key(self.id, digest), data))
        self._wake.set()
        return ref

    def _rate_ok_locked(self, name: str, limit: int, now: float) -> bool:
        second = int(now)
        window, used = self._rates.get(name, (-1, 0))
        if second != window:
            window, used = second, 0
        if used >= limit:
            self._dropped["rate"] += 1
            return False
        self._rates[name] = (window, used + 1)
        return True

    def _key_ok_locked(self, keys: set[str], key: str) -> bool:
        if key not in keys:
            if len(keys) >= self._limits.max_media_keys:
                self._dropped["keys"] += 1
                return False
            keys.add(key)
        return True

    def _default_step_locked(self, step: int | None) -> int | None:
        if step is None and self._next_step > 0:
            return self._next_step - 1
        return step

    def _spill_bytes(self, key: str, data: bytes, records: int) -> bool:
        stem = key.replace("/", "__")
        payload = self._spill_dir / f"{stem}.seg"
        marker = self._spill_dir / f"{stem}.json"
        try:
            with self._spill_lock:
                self._spill_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
                payload.write_bytes(data)
                marker.write_text(json.dumps({"store": self._store.root, "key": key, "records": records}))
        except Exception:
            self._count_lost(max(1, records))
            return False
        with self._lock:
            self._spill_backlog.append(_Spilled(marker=marker, payload=payload, key=key, records=records))
            # Evicted descriptors remain durably on disk for `eyvan track sync`.
            del self._spill_backlog[: max(0, len(self._spill_backlog) - min(256, self._limits.max_queued))]
            self._spilled += records
            self._stats_dirty = True
        return True

    def _retry_backlog(self) -> bool:
        while True:
            with self._lock:
                if not self._spill_backlog:
                    return True
                entry = self._spill_backlog[0]
            try:
                self._store.put(entry.key, entry.payload.read_bytes())
            except FileNotFoundError:
                # Already replayed by `eyvan track sync`.
                pass
            except Exception:
                return False
            with self._lock:
                if entry in self._spill_backlog:
                    self._spill_backlog.remove(entry)
            for path in (entry.marker, entry.payload):
                try:
                    path.unlink()
                except OSError:
                    pass

    def _count_lost(self, records: int) -> None:
        with self._lock:
            self._dropped["queue"] += records
            self._stats_dirty = True

    def _loop(self) -> None:
        while True:
            self._wake.wait(self._flush_interval_s)
            self._wake.clear()
            while True:
                with self._lock:
                    batch = next(
                        (pending for pending in self._batches if all(prior.done.is_set() for prior in pending.wait_for)),
                        None,
                    )
                    if batch is not None:
                        self._batches.remove(batch)
                    finished = self._finished
                if batch is None:
                    break
                self._process_batch(batch)
            if self._stop.is_set():
                return
            if not finished:
                self._process_batch(self._drain(force_stats=False))

    def _at_exit(self) -> None:
        try:
            self.finish("failed" if _UNCAUGHT else "finished", timeout=10.0)
        except Exception:
            pass


def start(
    name: str | None = None,
    *,
    project: str | None = None,
    group: str | None = None,
    tags: tp.Iterable[str] = (),
    config: tp.Mapping[str, object] | None = None,
    run_id: str | None = None,
    uri: str | None = None,
    spill_dir: str | os.PathLike[str] | None = None,
    ranks: str = "zero",
    episode_ranks: str = "all",
    limits: TrackLimits | None = None,
    flush_interval_s: float = 5.0,
    store: TrackingStore | None = None,
    environment: tp.Mapping[str, str] | None = None,
    clock: Clock | None = None,
    auto_name: bool | None = None,
) -> Run:
    """Start (or resume) a tracked run.

    Inside an Eyvan workload the run id is derived from the workload id, so a
    retry or a Spot preemption resumes the same run, and the scope root comes
    from ``EYVAN_TRACKING_URI``. Outside Eyvan, when ``uri`` and ``store`` are
    omitted (and ``environment`` is not overridden), Eyvan checks for a
    configured default monitoring bucket in ``~/.eyvan/storage.json`` before
    falling back to ``~/.eyvan/tracking/default/<project>``.

    Args:
        name: Human-readable run name. When omitted on a fresh standalone run
            (no explicit ``run_id`` or ``EYVAN_WORKLOAD_ID``), a readable
            W&B-style name is generated via :func:`generate_run_name`.
        project: Scope name used when ``uri`` and ``EYVAN_TRACKING_URI`` are unset.
        group: Optional group such as a sweep name.
        tags: Short labels.
        config: Initial configuration, same as calling :meth:`Run.config`.
        run_id: Explicit run id; overrides the workload-derived id.
        uri: Scope root (``gs://…``, ``file://…`` or a path) to write under.
        spill_dir: Local directory for segments that could not be uploaded.
        ranks: ``"zero"`` records only on rank 0; ``"all"`` records on every rank.
        episode_ranks: Independent policy for :meth:`Run.episode` and
            :meth:`Run.rollout`, even with ``ranks="all"``. Defaults to ``"all"``
            so actors on any rank can report episodes/rollouts. Completions
            follow ``ranks`` instead.
        limits: Per-run bounds; defaults to :class:`TrackLimits`.
        flush_interval_s: Seconds between background flushes.
        store: Explicit store, for embedding and tests; overrides ``uri``.
        environment: Environment to read, for tests. Defaults to ``os.environ``.
        clock: Wall-clock source, for tests. Defaults to ``time.time``.
        auto_name: Controls automatic W&B-style naming when ``name`` is ``None``.
            ``None`` (default) generates a name for fresh runs with no explicit
            ``run_id`` or workload id; ``True`` always generates a name when
            ``name`` is ``None``; ``False`` preserves ``name = run_id``.

    Returns:
        The run handle.

    Raises:
        ValueError: For malformed names, ids, tags, ``ranks`` or store URIs.
    """

    env = os.environ if environment is None else environment
    if ranks not in {"zero", "all"}:
        raise ValueError('ranks must be "zero" or "all"')
    if episode_ranks not in {"zero", "all"}:
        raise ValueError('episode_ranks must be "zero" or "all"')
    if name is not None and (not isinstance(name, str) or not name.strip() or len(name) > _MAX_NAME):
        raise ValueError(f"name must be a non-empty string of at most {_MAX_NAME} characters")
    if group is not None and (not isinstance(group, str) or not group.strip() or len(group) > _MAX_NAME):
        raise ValueError(f"group must be a non-empty string of at most {_MAX_NAME} characters")
    if auto_name is not None and not isinstance(auto_name, bool):
        raise ValueError("auto_name must be a boolean or None")
    tag_list = tuple(tags)
    if len(tag_list) > _MAX_TAGS or any(not isinstance(tag, str) or not tag or len(tag) > 64 for tag in tag_list):
        raise ValueError(f"tags must be at most {_MAX_TAGS} non-empty strings of at most 64 characters")
    if not isinstance(flush_interval_s, int | float) or isinstance(flush_interval_s, bool) or flush_interval_s <= 0:
        raise ValueError("flush_interval_s must be a positive number")
    workload_id = env.get("EYVAN_WORKLOAD_ID") or None
    effective_name = name
    if effective_name is None:
        use_auto_name = (run_id is None and workload_id is None) if auto_name is None else auto_name
        if use_auto_name:
            effective_name = generate_run_name()
    if run_id is not None:
        resolved_id = fmt.validate_run_id(run_id)
    elif workload_id is not None:
        resolved_id = fmt.run_id_for_workload(workload_id)
    else:
        resolved_id = _fresh_run_id(effective_name)
    attempt = _env_int(env, ("EYVAN_ATTEMPT",), 1, fmt.MAX_ATTEMPT)
    rank = _env_int(env, ("EYVAN_DISTRIBUTED_PROCESS_ID", "JAX_PROCESS_ID"), 0, fmt.MAX_RANK)
    if store is None:
        root = uri or env.get(fmt.TRACKING_URI_ENV)
        from_default_storage = False
        if not root:
            scope = fmt.validate_run_id(project or "default")
            if environment is None:
                try:
                    from eyvan.storage import resolve_default_tracking_uri

                    root = resolve_default_tracking_uri(project=scope, org=DEFAULT_ORG)
                except OSError:
                    root = None
                from_default_storage = root is not None
            if not root:
                root = f"{DEFAULT_LOCAL_STORE}/{DEFAULT_ORG}/{scope}"
        if from_default_storage and root.startswith("gs://"):
            from eyvan.storage import default_gcp_token_provider
            from eyvan.track._gcs import GcsTrackingStore

            store = GcsTrackingStore(root, token_provider=default_gcp_token_provider())
        else:
            try:
                store = store_for_uri(root)
            except (OSError, RuntimeError) as exc:
                store = _UnavailableTrackingStore(root, str(exc))
    raw_spill = spill_dir if spill_dir is not None else DEFAULT_SPILL_DIR
    try:
        spill_root = Path(raw_spill).expanduser()
    except (OSError, RuntimeError):
        spill_root = Path(raw_spill)
    run = Run(
        run_id=resolved_id,
        name=effective_name or resolved_id,
        group=group,
        tags=tag_list,
        store=store,
        spill_dir=spill_root / resolved_id,
        attempt=attempt,
        rank=rank,
        workload_id=workload_id,
        active=ranks == "all" or rank == 0,
        limits=limits or TrackLimits(),
        flush_interval_s=float(flush_interval_s),
        clock=clock or time.time,
        episodes_active=episode_ranks == "all" or rank == 0,
    )
    if config is not None:
        run.config(config)
    return run


def replay_spill(spill_dir: str | os.PathLike[str], *, store_uri: str | None = None) -> ReplayResult:
    """Upload segments a run spilled locally, deleting each one once stored.

    Replaying is idempotent: a segment that already reached the store with the
    same bytes counts as replayed. Failures are collected, never raised.

    Args:
        spill_dir: A spill directory (or a parent of several).
        store_uri: Scope root to upload to instead of the one each segment
            recorded when it was spilled.

    Returns:
        Counts of replayed and failed segments, and their error messages.
    """

    root = Path(spill_dir).expanduser()
    if not root.exists():
        return ReplayResult(replayed=0, failed=0)
    stores: dict[str, TrackingStore] = {}
    replayed = failed = 0
    errors: list[str] = []
    for marker in sorted(root.rglob("*.json")):
        payload = marker.with_suffix(".seg")
        try:
            meta = json.loads(marker.read_text())
            key = meta["key"]
            target = store_uri or meta["store"]
            if not isinstance(key, str) or not isinstance(target, str):
                raise ValueError("spill marker is missing its key or store")
            store = stores.get(target)
            if store is None:
                store = stores[target] = store_for_uri(target)
            store.put(key, payload.read_bytes())
        except Exception as exc:
            failed += 1
            errors.append(f"{marker.name}: {exc}")
            continue
        replayed += 1
        for path in (marker, payload):
            try:
                path.unlink()
            except OSError:
                pass
    return ReplayResult(replayed=replayed, failed=failed, errors=tuple(errors))


__all__ = (
    "DEFAULT_LOCAL_STORE",
    "DEFAULT_SPILL_DIR",
    "ReplayResult",
    "Run",
    "TrackLimits",
    "replay_spill",
    "start",
)
