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

"""eSurge Monitoring and Observability System.

This module provides comprehensive monitoring capabilities for the eSurge engine,
including Prometheus metrics export for production monitoring, Eyvan experiment
tracking (``eyvan.track``), and a Rich-based console monitor for development and
debugging.

Classes:
    PrometheusMetrics: Prometheus metrics exporter with standard metrics.
    EyvanMetricsExporter: Eyvan run-tracking exporter for eSurge metrics.
    RichConsoleMonitor: Rich console-based live monitoring dashboard.
    eSurgeMonitoringServer: Combined monitoring server for Prometheus and/or Eyvan export.

Functions:
    start_monitoring_server: Start the global monitoring server.
    start_eyvan_tracking: Start an owner-scoped Eyvan metrics tracking loop.
    start_console_monitor: Start the global console monitor.
    stop_monitoring: Stop all monitoring services (or release one owner's leases).

Constants:
    PROMETHEUS_AVAILABLE: Whether prometheus_client is installed.
    EYVAN_TRACK_AVAILABLE: Whether eyvan.track is installed.
    RICH_AVAILABLE: Whether rich library is installed.

Example:
    >>> from easydel.inference.esurge.monitoring import (
    ...     start_monitoring_server,
    ...     stop_monitoring
    ... )
    >>>
    >>> # Start monitoring with Prometheus endpoint
    >>> server = start_monitoring_server(prometheus_port=8000)
    >>> print("Metrics at http://localhost:8000/metrics")
    >>>
    >>> # Later, stop monitoring
    >>> stop_monitoring()

Note:
    For production monitoring, use Prometheus with Grafana dashboards and/or
    Eyvan tracking. The console monitor is intended for development and debugging.
"""

from __future__ import annotations

import logging
import math
import os
import re
import threading
import time
import typing as tp
import weakref

from .metrics import MetricsCollector, get_metrics_collector

try:
    from prometheus_client import REGISTRY, Counter, Gauge, Histogram, Info, start_http_server

    _prometheus_available = True
except ImportError:
    REGISTRY = None
    Counter = None
    Gauge = None
    Histogram = None
    Info = None
    start_http_server = None
    _prometheus_available = False

PROMETHEUS_AVAILABLE: bool = _prometheus_available

try:
    import eyvan.track as _eyvan_track

    _eyvan_track_available = True
except ImportError:
    _eyvan_track = None
    _eyvan_track_available = False

EYVAN_TRACK_AVAILABLE: bool = _eyvan_track_available

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    _rich_available = True
except ImportError:
    Console = None
    Layout = None
    Live = None
    Panel = None
    Table = None
    Text = None
    _rich_available = False

RICH_AVAILABLE: bool = _rich_available

_DEFAULT_EYVAN_TAGS: tuple[str, ...] = ("easydel", "jax", "esurge", "inference")
_EYVAN_KEY_UNSAFE = re.compile(r"[^A-Za-z0-9_./-]+")
_EYVAN_PROJECT_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_EYVAN_MAX_KEY = 128
_EYVAN_MAX_PREFIX = 64
_EYVAN_MAX_NAME = 256
_EYVAN_MAX_TAG = 64
_EYVAN_MAX_TAGS = 64


def _load_eyvan_auto_helpers():
    """Lazily import shared Eyvan auto-config helpers from ``training_configurations``."""
    from easydel.trainers import training_configurations as tc

    return tc


def _resolve_esurge_eyvan_uri(
    *,
    explicit_uri: str | None,
    project: str,
    org: str = "default",
) -> str | None:
    """Resolve the Eyvan tracking store URI for eSurge (explicit -> env -> default monitoring bucket -> None)."""
    tc = _load_eyvan_auto_helpers()
    return tc._resolve_eyvan_default_tracking_uri(
        explicit_uri=explicit_uri,
        project=project,
        org=org,
    )


def _resolve_esurge_eyvan_run_name(
    *,
    explicit_name: str | None = None,
    model_name: str | None = None,
    model_type: str | None = None,
    seed: int | None = None,
) -> str:
    """Resolve a human-readable eSurge Eyvan run name (explicit verbatim or W&B-style auto name <=64 chars)."""
    tc = _load_eyvan_auto_helpers()
    return tc._resolve_eyvan_run_name(
        explicit_name=explicit_name,
        model_name=model_name or model_type,
        trainer_slug="esurge",
        seed=seed,
    )


def _auto_detect_esurge_eyvan_tags(
    *,
    model_name: str | None = None,
    model_type: str | None = None,
    user_tags: tp.Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Auto-detect framework, eSurge inference, model, and hardware tags merged with user tags."""
    tc = _load_eyvan_auto_helpers()
    auto: list[str] = list(_DEFAULT_EYVAN_TAGS)
    model_slug = tc._extract_eyvan_model_slug(model_type, prefer_model_type=False) or tc._extract_eyvan_model_slug(
        model_name, prefer_model_type=False
    )
    if model_slug:
        auto.append(model_slug)
    auto.extend(tc._detect_eyvan_hardware_tags())
    return tc._merge_eyvan_tags(auto, user_tags)


def _collect_system_info() -> dict[str, str]:
    """Collect string-valued system metadata for monitoring exporters.

    Returns:
        Mapping of label name to string value (Python/JAX versions and, when
        the JAX backend is available, backend name and device counts).
    """
    import platform

    info: dict[str, str] = {"python_version": platform.python_version()}
    try:
        import jax

        info["jax_version"] = str(jax.__version__)
        try:
            info["jax_backend"] = str(jax.default_backend())
            info["device_count"] = str(jax.device_count())
            info["process_count"] = str(jax.process_count())
        except Exception:
            pass
    except Exception:
        pass
    return info


def _observe_new_entries(deque, cursor, observe_fn):
    """Observe entries after a retained object cursor, or all if evicted.

    Args:
        deque: Bounded deque of immutable metrics snapshots.
        cursor: Last observed snapshot, or ``None`` on the first call.
        observe_fn: Callable invoked once per newly retained snapshot.

    Returns:
        The last snapshot itself, keeping its identity alive until the
        next poll, or ``None`` for an empty deque.
    """
    if not deque:
        return None
    pending = []
    found = False
    for item in deque:
        if found:
            observe_fn(item)
        elif item is cursor:
            found = True
        else:
            pending.append(item)
    if not found:
        for item in pending:
            observe_fn(item)
    return deque[-1]


def _normalize_eyvan_prefix(prefix: str) -> str:
    """Sanitize and normalize a metric key prefix for Eyvan tracking."""
    raw = str(prefix).strip()
    if not raw:
        return ""
    keep_sep = raw[-1] in {"/", "_", "-", "."}
    cleaned = _EYVAN_KEY_UNSAFE.sub("_", raw).strip("/")
    if not cleaned:
        return ""
    if not keep_sep:
        cleaned = cleaned.rstrip("_")
        if not cleaned:
            return ""
    if len(cleaned) > _EYVAN_MAX_PREFIX:
        cleaned = cleaned[:_EYVAN_MAX_PREFIX]
    if not cleaned.endswith(("/", "_", "-", ".")):
        cleaned = f"{cleaned}/"
    return cleaned


def _sanitize_eyvan_key(key: str) -> str:
    """Coerce a metric or histogram name into a valid Eyvan key."""
    cleaned = _EYVAN_KEY_UNSAFE.sub("_", str(key).strip()).strip("/")
    if not cleaned:
        return "metric"
    if len(cleaned) > _EYVAN_MAX_KEY:
        cleaned = cleaned[:_EYVAN_MAX_KEY]
    return cleaned


def _sanitize_eyvan_text(value: str | None, *, limit: int = _EYVAN_MAX_NAME) -> str | None:
    """Normalize an optional human-readable Eyvan label (name/group)."""
    if value is None:
        return None
    cleaned = str(value).strip()
    if not cleaned:
        return None
    return cleaned[:limit]


def _sanitize_eyvan_project(project: str | None) -> str:
    """Normalize an Eyvan local project identifier to a valid scope segment."""
    raw = "esurge" if project is None else str(project).strip()
    cleaned = _EYVAN_PROJECT_UNSAFE.sub("-", raw).strip("-._")[:_EYVAN_MAX_KEY]
    if not cleaned or not cleaned[0].isalnum():
        return "esurge"
    return cleaned


def _sanitize_eyvan_tags(tags: tp.Iterable[str] | None) -> tuple[str, ...]:
    """Normalize and deduplicate Eyvan run tags within Eyvan's bounds."""
    if tags is None:
        return _DEFAULT_EYVAN_TAGS
    result: list[str] = []
    seen: set[str] = set()
    for raw in tags:
        cleaned = str(raw).strip()[:_EYVAN_MAX_TAG]
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
        if len(result) >= _EYVAN_MAX_TAGS:
            break
    return tuple(result)


def _effective_esurge_eyvan_rank(env: tp.Mapping[str, str] | None = None) -> int:
    """Return the effective Eyvan distributed process rank matching ``eyvan.track.start``."""
    src = os.environ if env is None else env
    for key in ("EYVAN_DISTRIBUTED_PROCESS_ID", "JAX_PROCESS_ID"):
        raw = (src.get(key) or "").strip()
        if raw:
            try:
                return max(0, int(raw))
            except ValueError:
                pass
    try:
        import jax

        return max(0, int(jax.process_index()))
    except Exception:
        return 0


def _resolve_eyvan_rank_environment() -> dict[str, str] | None:
    """Return an environment override with ``JAX_PROCESS_ID`` on nonzero JAX ranks.

    Eyvan's ``ranks="zero"`` reads ``EYVAN_DISTRIBUTED_PROCESS_ID`` or
    ``JAX_PROCESS_ID``. When running on a multi-host JAX slice where neither
    variable is set in ``os.environ``, pass a copy with ``JAX_PROCESS_ID`` set
    to ``jax.process_index()`` so nonzero hosts stay inactive without mutating
    ``os.environ``.
    """
    if (os.environ.get("EYVAN_DISTRIBUTED_PROCESS_ID") or "").strip() or (
        os.environ.get("JAX_PROCESS_ID") or ""
    ).strip():
        return None
    try:
        import jax

        proc_idx = int(jax.process_index())
    except Exception:
        return None
    if proc_idx > 0:
        env = dict(os.environ)
        env["JAX_PROCESS_ID"] = str(proc_idx)
        return env
    return None


class PrometheusMetrics:
    """Prometheus metrics exporter for eSurge.

    Provides standard Prometheus metrics for monitoring eSurge engine
    performance, including request latency, throughput, scheduler stats,
    and cache utilization.

    Metrics exported:
        - esurge_requests_total: Total request count by status
        - esurge_request_duration_seconds: Request latency histogram
        - esurge_time_to_first_token_seconds: TTFT histogram
        - esurge_tokens_generated_total: Total tokens generated
        - esurge_tokens_per_second: Current throughput gauge
        - esurge_waiting_requests: Requests in queue
        - esurge_running_requests: Active requests
        - esurge_scheduled_tokens: Tokens in current batch
        - esurge_preempted_requests_total: Preemption count
        - esurge_schedule_duration_seconds: Scheduler latency
        - esurge_model_execution_duration_seconds: Model forward pass time
        - esurge_batch_size: Current batch size
        - esurge_cache_pages_total: Total cache pages
        - esurge_cache_pages_used: Used cache pages
        - esurge_cache_hit_rate: Cache hit ratio

    Attributes:
        prefix: Prefix for all metric names.
        requests_total: Counter for total requests by status.
        request_duration: Histogram for request latency.
        time_to_first_token: Histogram for TTFT.
        tokens_generated_total: Counter for generated tokens.
        tokens_per_second: Gauge for current throughput.
        waiting_requests: Gauge for queue depth.
        running_requests: Gauge for active requests.
        scheduled_tokens: Gauge for batch token count.
        preempted_requests_total: Counter for preemptions.
        schedule_duration: Histogram for scheduler latency.
        model_execution_duration: Histogram for model latency.
        batch_size: Gauge for batch size.
        cache_pages_total: Gauge for total cache pages.
        cache_pages_used: Gauge for used cache pages.
        cache_hit_rate: Gauge for cache hit ratio.
        system_info: Info metric for system metadata.

    Example:
        >>> metrics = PrometheusMetrics(prefix="myapp_")
        >>> metrics.tokens_per_second.set(150.0)
        >>> metrics.requests_total.labels(status="completed").inc()

    Raises:
        ImportError: If prometheus_client is not installed.
    """

    def __init__(self, prefix: str = "esurge_"):
        """Initialize Prometheus metrics.

        Args:
            prefix: Prefix for all metric names. Defaults to "esurge_".

        Raises:
            ImportError: If prometheus_client is not available.
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("prometheus_client not available. Install with: pip install prometheus-client")

        self.prefix = prefix
        # Track every collector incrementally so a mid-init registration error
        # unregisters already-created collectors from the global REGISTRY.
        self._collectors: list = []

        def _track(collector):
            self._collectors.append(collector)
            return collector

        try:
            self.requests_total = _track(Counter(f"{prefix}requests_total", "Total number of requests", ["status"]))

            self.request_duration = _track(
                Histogram(
                    f"{prefix}request_duration_seconds",
                    "Request duration in seconds",
                    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
                )
            )

            self.time_to_first_token = _track(
                Histogram(
                    f"{prefix}time_to_first_token_seconds",
                    "Time to first token in seconds",
                    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
                )
            )

            self.tokens_generated_total = _track(
                Counter(f"{prefix}tokens_generated_total", "Total number of tokens generated")
            )

            self.tokens_per_second = _track(Gauge(f"{prefix}tokens_per_second", "Current tokens per second throughput"))

            self.waiting_requests = _track(
                Gauge(f"{prefix}waiting_requests", "Number of requests waiting to be scheduled")
            )

            self.running_requests = _track(Gauge(f"{prefix}running_requests", "Number of currently running requests"))

            self.scheduled_tokens = _track(
                Gauge(f"{prefix}scheduled_tokens", "Number of tokens scheduled in current batch")
            )

            self.preempted_requests_total = _track(
                Counter(f"{prefix}preempted_requests_total", "Total number of preempted requests")
            )

            self.schedule_duration = _track(
                Histogram(
                    f"{prefix}schedule_duration_seconds",
                    "Scheduler operation duration in seconds",
                    buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
                )
            )

            self.model_execution_duration = _track(
                Histogram(
                    f"{prefix}model_execution_duration_seconds",
                    "Model execution duration in seconds",
                    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
                )
            )

            self.batch_size = _track(Gauge(f"{prefix}batch_size", "Current batch size"))

            self.cache_pages_total = _track(Gauge(f"{prefix}cache_pages_total", "Total number of cache pages"))

            self.cache_pages_used = _track(Gauge(f"{prefix}cache_pages_used", "Number of used cache pages"))

            self.cache_hit_rate = _track(Gauge(f"{prefix}cache_hit_rate", "Cache hit rate (0-1)"))

            self.system_info = _track(Info(f"{prefix}system_info", "System information"))
            self.system_info.info(self._collect_system_info())
        except BaseException:
            self.close()
            raise

        # Monotonic cursor: total requests seen so far (completed + failed).
        # Used to process only new completions from the deque without
        # double-counting, even after the deque wraps around.
        self._completions_cursor: int = 0
        # Keep the actual cursor objects alive across bounded-deque eviction;
        # an integer id alone can be recycled for a genuinely new snapshot.
        self._last_scheduler_entry = None
        self._last_runner_entry = None

    @staticmethod
    def _collect_system_info() -> dict[str, str]:
        """Collect string-valued system metadata for the ``system_info`` metric.

        Returns:
            Mapping of label name to string value (Python/JAX versions and, when
            the JAX backend is available, backend name and device counts).
        """
        return _collect_system_info()

    def close(self) -> None:
        """Unregister every collector from the default Prometheus registry.

        After ``close`` the metric names are free again, so a new
        :class:`PrometheusMetrics` with the same prefix can be created (for
        example when monitoring is stopped and started again). Idempotent.
        """
        for collector in self._collectors:
            try:
                REGISTRY.unregister(collector)
            except KeyError:
                pass
        self._collectors = []

    @staticmethod
    def _observe_new_entries(deque, cursor, observe_fn):
        """Observe entries after a retained object cursor, or all if evicted.

        Args:
            deque: Bounded deque of immutable metrics snapshots.
            cursor: Last observed snapshot, or ``None`` on the first call.
            observe_fn: Callable invoked once per newly retained snapshot.

        Returns:
            The last snapshot itself, keeping its identity alive until the
            next poll, or ``None`` for an empty deque.
        """
        return _observe_new_entries(deque, cursor, observe_fn)

    def _observe_scheduler_entry(self, entry) -> None:
        """Fold one new scheduler snapshot into the schedule histogram and preemption counter.

        Args:
            entry: A ``SchedulerMetrics`` record not yet observed.
        """
        self.schedule_duration.observe(entry.schedule_time)
        if entry.num_preempted_requests > 0:
            self.preempted_requests_total.inc(entry.num_preempted_requests)

    def update_from_metrics_collector(self, collector: MetricsCollector) -> None:
        """Update Prometheus metrics from the metrics collector.

        Fetches the latest metrics from the MetricsCollector and updates
        all Prometheus gauges, counters, and histograms accordingly.

        Only newly-completed requests (since the last call) are processed
        so that counters and histograms are not double-counted.

        Args:
            collector: MetricsCollector instance to read metrics from.

        Note:
            This method is thread-safe and acquires the collector's lock.
        """
        system_metrics = collector.get_system_metrics()

        with collector._lock:
            live_tps = 0.0
            scheduler_reports_running = (
                bool(collector.scheduler_metrics) and collector.scheduler_metrics[-1].num_running_requests > 0
            )
            if collector.runner_metrics and (scheduler_reports_running or not collector.scheduler_metrics):
                live_tps = collector.runner_metrics[-1].tokens_per_second
            if live_tps > 0:
                self.tokens_per_second.set(live_tps)
            elif system_metrics.average_throughput > 0:
                self.tokens_per_second.set(float(system_metrics.average_throughput))
            else:
                self.tokens_per_second.set(0.0)

            total_now = collector.counters["total_completed"] + collector.counters["total_failed"]
            num_new = total_now - self._completions_cursor
            if num_new > len(collector.completed_requests):
                # More completed than the deque holds — we lost some, take all
                num_new = len(collector.completed_requests)
            new_requests = list(collector.completed_requests)[-num_new:] if num_new > 0 else []
            self._completions_cursor = total_now

            for req_metrics in new_requests:
                if req_metrics.error:
                    self.requests_total.labels(status="failed").inc()
                else:
                    self.requests_total.labels(status="completed").inc()

                if req_metrics.total_latency:
                    self.request_duration.observe(req_metrics.total_latency)

                if req_metrics.time_to_first_token:
                    self.time_to_first_token.observe(req_metrics.time_to_first_token)

                self.tokens_generated_total.inc(req_metrics.generated_tokens)

            if collector.scheduler_metrics:
                latest_scheduler = collector.scheduler_metrics[-1]
                # Gauges: always set to latest value
                self.waiting_requests.set(latest_scheduler.num_waiting_requests)
                self.running_requests.set(latest_scheduler.num_running_requests)
                self.scheduled_tokens.set(latest_scheduler.num_scheduled_tokens)
                self._last_scheduler_entry = self._observe_new_entries(
                    collector.scheduler_metrics,
                    self._last_scheduler_entry,
                    self._observe_scheduler_entry,
                )

            if collector.runner_metrics:
                latest_runner = collector.runner_metrics[-1]
                # Gauge: always set to latest
                self.batch_size.set(latest_runner.batch_size)
                self._last_runner_entry = self._observe_new_entries(
                    collector.runner_metrics,
                    self._last_runner_entry,
                    lambda rm: self.model_execution_duration.observe(rm.execution_time),
                )

            if collector.cache_metrics:
                latest_cache = collector.cache_metrics[-1]
                self.cache_pages_total.set(latest_cache.total_pages)
                self.cache_pages_used.set(latest_cache.used_pages)
                self.cache_hit_rate.set(latest_cache.cache_hit_rate)


class EyvanMetricsExporter:
    """Eyvan experiment-tracking exporter for eSurge inference metrics.

    Streams scalars, distribution histograms, and a terminal summary from a
    :class:`MetricsCollector` into an :mod:`eyvan.track` run without double-
    counting completions, scheduler snapshots, or runner snapshots across polls.

    Args:
        name: Human-readable run name.
        project: Local Eyvan project scope when ``uri`` and
            ``EYVAN_TRACKING_URI`` are unset. Defaults to ``"esurge"``.
        group: Optional Eyvan run group (for example a serving benchmark sweep).
        tags: Optional iterable of tags. Defaults to
            ``("EasyDeL", "eSurge", "Inference")``.
        uri: Optional Eyvan tracking store URI (``gs://...``, ``file://...``,
            or a local directory path). Falls back to ``EYVAN_TRACKING_URI`` or
            Eyvan's local default store.
        run_id: Optional explicit Eyvan run id.
        flush_interval_s: Seconds between background Eyvan segment flushes.
        prefix: Metric key prefix. Defaults to ``"esurge/"``.
        config: Optional extra configuration metadata merged with system info.
        run: Optional pre-built ``eyvan.track.Run`` (or compatible object).
            When provided, this exporter writes into it without owning its
            lifecycle (``close()`` writes summary and flushes, without calling
            ``run.finish()``).

    Raises:
        ImportError: If ``run`` is ``None`` and ``eyvan.track`` is unavailable.
        ValueError: If ``flush_interval_s`` is not a positive finite number.
    """

    def __init__(
        self,
        name: str | None = None,
        project: str | None = None,
        group: str | None = None,
        tags: tp.Iterable[str] | None = None,
        uri: str | None = None,
        run_id: str | None = None,
        flush_interval_s: float = 5.0,
        prefix: str = "esurge/",
        config: tp.Mapping[str, tp.Any] | None = None,
        run: tp.Any | None = None,
    ) -> None:
        if (
            isinstance(flush_interval_s, bool)
            or not isinstance(flush_interval_s, int | float)
            or not math.isfinite(flush_interval_s)
            or flush_interval_s <= 0
        ):
            raise ValueError("flush_interval_s must be a positive finite number")

        self.prefix = _normalize_eyvan_prefix(prefix)
        self.flush_interval_s = float(flush_interval_s)

        if run is not None:
            self._run = run
            self._owns_run = False
            if config is not None and hasattr(run, "config"):
                run.config(dict(config))
        else:
            if not EYVAN_TRACK_AVAILABLE or _eyvan_track is None:
                raise ImportError("eyvan.track not available. Install eyvan to enable Eyvan metrics tracking.")

            merged_config: dict[str, tp.Any] = dict(_collect_system_info())
            if config is not None:
                merged_config.update(dict(config))

            resolved_project = _sanitize_eyvan_project(project)
            cfg_model_name = merged_config.get("model_name")
            cfg_model_type = merged_config.get("model_type")
            model_name_str = cfg_model_name if isinstance(cfg_model_name, str) else None
            model_type_str = cfg_model_type if isinstance(cfg_model_type, str) else None

            resolved_name = _resolve_esurge_eyvan_run_name(
                explicit_name=name,
                model_name=model_name_str,
                model_type=model_type_str,
            )
            resolved_uri = _resolve_esurge_eyvan_uri(
                explicit_uri=uri,
                project=resolved_project,
                org="default",
            )
            resolved_tags = _auto_detect_esurge_eyvan_tags(
                model_name=model_name_str,
                model_type=model_type_str,
                user_tags=tags,
            )

            start_kwargs: dict[str, tp.Any] = {
                "name": _sanitize_eyvan_text(resolved_name),
                "project": resolved_project,
                "group": _sanitize_eyvan_text(group),
                "tags": resolved_tags,
                "config": merged_config,
                "run_id": run_id,
                "uri": resolved_uri,
                "ranks": "zero",
                "episode_ranks": "zero",
                "flush_interval_s": self.flush_interval_s,
            }
            env_override = _resolve_eyvan_rank_environment()
            if env_override is not None:
                start_kwargs["environment"] = env_override
            if _effective_esurge_eyvan_rank(env_override) == 0:
                from easydel.utils.eyvan_helpers import _ensure_local_eyvan_store_writable

                _ensure_local_eyvan_store_writable(resolved_uri, project=resolved_project, org="default")

            self._run = _eyvan_track.start(**start_kwargs)
            self._owns_run = True

        self._lock = threading.Lock()
        self._closed = False
        self._step = 0
        self._start_monotonic = time.monotonic()
        self._completions_cursor = 0
        self._last_scheduler_entry = None
        self._last_runner_entry = None
        self._preempted_total = 0
        self._latest_summary: dict[str, tp.Any] = {}
        self._on_close_callbacks: list[tp.Callable[[], None]] = []

    @property
    def run(self) -> tp.Any:
        """Underlying ``eyvan.track.Run`` handle."""
        return self._run

    @property
    def run_id(self) -> str | None:
        """Run identifier assigned by Eyvan, if available."""
        value = getattr(self._run, "id", None)
        return str(value) if value is not None else None

    @property
    def store_root(self) -> str | None:
        """Canonical tracking store root URI, if available."""
        value = getattr(self._run, "store_root", None)
        return str(value) if value is not None else None

    @property
    def active(self) -> bool:
        """Whether the underlying Eyvan run actively records on this rank."""
        return bool(getattr(self._run, "active", True))

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has completed."""
        return self._closed

    @property
    def step(self) -> int:
        """Next monotonic step index to be logged."""
        return self._step

    @property
    def owns_run(self) -> bool:
        """Whether this exporter created (and will finish) the underlying run."""
        return self._owns_run

    def _add_close_callback(self, callback: tp.Callable[[], None]) -> None:
        run_now = False
        with self._lock:
            if self._closed:
                run_now = True
            else:
                self._on_close_callbacks.append(callback)
        if run_now:
            try:
                callback()
            except Exception:
                pass

    def update_from_metrics_collector(self, collector: MetricsCollector) -> dict[str, float]:
        """Read the latest eSurge metrics and emit scalars and delta histograms to Eyvan.

        Only completions, scheduler snapshots, and runner snapshots observed
        since the previous poll contribute to delta histograms and the
        cumulative preemption counter.

        Args:
            collector: :class:`MetricsCollector` instance to sample from.

        Returns:
            Dictionary of scalar metric keys and float values logged at this
            step, or an empty dict if the exporter is already closed.

        Note:
            Thread-safe. Lock ordering is always ``self._lock`` followed by
            ``collector._lock``, and ``collector._lock`` is released before
            writing to the Eyvan run.
        """
        with self._lock:
            if self._closed:
                return {}

            has_locked_sys = hasattr(collector, "_compute_system_metrics_locked")
            system_metrics = None if has_locked_sys else collector.get_system_metrics()

            with collector._lock:
                if has_locked_sys:
                    system_metrics = collector._compute_system_metrics_locked()

                live_tps = 0.0
                scheduler_reports_running = (
                    bool(collector.scheduler_metrics) and collector.scheduler_metrics[-1].num_running_requests > 0
                )
                if collector.runner_metrics and (scheduler_reports_running or not collector.scheduler_metrics):
                    live_tps = float(collector.runner_metrics[-1].tokens_per_second)
                if live_tps > 0:
                    tokens_per_second = live_tps
                elif system_metrics.average_throughput > 0:
                    tokens_per_second = float(system_metrics.average_throughput)
                else:
                    tokens_per_second = 0.0

                total_completed = int(collector.counters["total_completed"])
                total_failed = int(collector.counters["total_failed"])
                total_tokens_generated = int(collector.counters["total_tokens_generated"])
                active_requests = len(collector.request_metrics)

                total_now = total_completed + total_failed
                if total_now < self._completions_cursor:
                    # Collector was reset between polls.
                    self._completions_cursor = 0
                num_new = total_now - self._completions_cursor
                if num_new > len(collector.completed_requests):
                    num_new = len(collector.completed_requests)
                new_requests = list(collector.completed_requests)[-num_new:] if num_new > 0 else []
                self._completions_cursor = total_now

                latest_scheduler = collector.scheduler_metrics[-1] if collector.scheduler_metrics else None
                new_scheduler_entries: list[tp.Any] = []
                if collector.scheduler_metrics:
                    self._last_scheduler_entry = _observe_new_entries(
                        collector.scheduler_metrics,
                        self._last_scheduler_entry,
                        new_scheduler_entries.append,
                    )
                    for entry in new_scheduler_entries:
                        if entry.num_preempted_requests > 0:
                            self._preempted_total += int(entry.num_preempted_requests)
                preempted_total = self._preempted_total

                latest_runner = collector.runner_metrics[-1] if collector.runner_metrics else None
                new_runner_entries: list[tp.Any] = []
                if collector.runner_metrics:
                    self._last_runner_entry = _observe_new_entries(
                        collector.runner_metrics,
                        self._last_runner_entry,
                        new_runner_entries.append,
                    )

                latest_cache = collector.cache_metrics[-1] if collector.cache_metrics else None

            p = self.prefix
            cache_hit_rate = float(latest_cache.cache_hit_rate) if latest_cache is not None else 0.0
            cache_utilization = (
                float(latest_cache.used_pages) / float(latest_cache.total_pages)
                if latest_cache is not None and latest_cache.total_pages > 0
                else 0.0
            )
            queue_size = float(latest_scheduler.num_waiting_requests) if latest_scheduler is not None else 0.0
            running_requests = float(latest_scheduler.num_running_requests) if latest_scheduler is not None else 0.0

            raw_scalars: dict[str, float] = {
                f"{p}system/requests_per_second": float(system_metrics.requests_per_second),
                f"{p}system/average_latency_s": float(system_metrics.average_latency),
                f"{p}system/average_ttft_s": float(system_metrics.average_ttft),
                f"{p}system/average_throughput_tok_s": float(system_metrics.average_throughput),
                f"{p}system/tokens_per_second": float(tokens_per_second),
                f"{p}system/total_requests_completed": float(total_completed),
                f"{p}system/total_requests_failed": float(total_failed),
                f"{p}system/total_tokens_generated": float(total_tokens_generated),
                f"{p}system/active_requests": float(active_requests),
                f"{p}system/queue_size": queue_size,
                f"{p}system/running_requests": running_requests,
                f"{p}system/cache_hit_rate": cache_hit_rate,
                f"{p}system/cache_utilization": cache_utilization,
            }

            if latest_scheduler is not None:
                raw_scalars.update(
                    {
                        f"{p}scheduler/waiting_requests": float(latest_scheduler.num_waiting_requests),
                        f"{p}scheduler/running_requests": float(latest_scheduler.num_running_requests),
                        f"{p}scheduler/scheduled_tokens": float(latest_scheduler.num_scheduled_tokens),
                        f"{p}scheduler/batch_size": float(latest_scheduler.batch_size),
                        f"{p}scheduler/schedule_time_s": float(latest_scheduler.schedule_time),
                        f"{p}scheduler/preempted_requests_total": float(preempted_total),
                    }
                )

            if latest_runner is not None:
                raw_scalars.update(
                    {
                        f"{p}runner/execution_time_s": float(latest_runner.execution_time),
                        f"{p}runner/batch_size": float(latest_runner.batch_size),
                        f"{p}runner/num_tokens": float(latest_runner.num_tokens),
                        f"{p}runner/tokens_per_second": float(latest_runner.tokens_per_second),
                    }
                )

            if latest_cache is not None:
                raw_scalars.update(
                    {
                        f"{p}cache/total_pages": float(latest_cache.total_pages),
                        f"{p}cache/used_pages": float(latest_cache.used_pages),
                        f"{p}cache/free_pages": float(latest_cache.free_pages),
                        f"{p}cache/utilization": cache_utilization,
                        f"{p}cache/cache_hit_rate": cache_hit_rate,
                    }
                )

            scalars = {_sanitize_eyvan_key(k): float(v) for k, v in raw_scalars.items()}

            step = self._step
            self._step += 1
            elapsed_s = max(0.0, time.monotonic() - self._start_monotonic)
            axes = {"elapsed_s": elapsed_s}

            self._latest_summary = {
                "total_requests_completed": total_completed,
                "total_requests_failed": total_failed,
                "total_tokens_generated": total_tokens_generated,
                "preempted_requests_total": preempted_total,
                "average_latency_s": float(system_metrics.average_latency),
                "average_ttft_s": float(system_metrics.average_ttft),
                "average_throughput_tok_s": float(system_metrics.average_throughput),
                "tokens_per_second": float(tokens_per_second),
                "requests_per_second": float(system_metrics.requests_per_second),
                "cache_hit_rate": cache_hit_rate,
                "cache_utilization": cache_utilization,
                "polls": self._step,
                "elapsed_s": elapsed_s,
            }

            self._run.log(scalars, step=step, axes=axes)

            histograms: tuple[tuple[str, list[float]], ...] = (
                (
                    "hist/request_duration_s",
                    [
                        float(req.total_latency)
                        for req in new_requests
                        if req.total_latency is not None and math.isfinite(float(req.total_latency))
                    ],
                ),
                (
                    "hist/ttft_s",
                    [
                        float(req.time_to_first_token)
                        for req in new_requests
                        if req.time_to_first_token is not None and math.isfinite(float(req.time_to_first_token))
                    ],
                ),
                (
                    "hist/tokens_per_second",
                    [
                        float(req.tokens_per_second)
                        for req in new_requests
                        if req.tokens_per_second is not None
                        and float(req.tokens_per_second) > 0
                        and math.isfinite(float(req.tokens_per_second))
                    ],
                ),
                (
                    "hist/schedule_duration_s",
                    [
                        float(entry.schedule_time)
                        for entry in new_scheduler_entries
                        if math.isfinite(float(entry.schedule_time))
                    ],
                ),
                (
                    "hist/model_execution_duration_s",
                    [
                        float(entry.execution_time)
                        for entry in new_runner_entries
                        if math.isfinite(float(entry.execution_time))
                    ],
                ),
            )
            for suffix, values in histograms:
                if values:
                    self._run.histogram(_sanitize_eyvan_key(f"{p}{suffix}"), values, step=step, axes=axes)

            return scalars

    def close(self, status: str = "finished") -> None:
        """Write final summary metrics and flush/finish the Eyvan run.

        Only runs owned by this exporter (created internally via
        ``eyvan.track.start``) are finished via ``run.finish(status=status)``;
        externally supplied runs receive the final summary and are flushed
        without being finished. Idempotent.

        Args:
            status: Terminal Eyvan status (``"finished"``, ``"failed"``, or
                ``"aborted"``).
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            summary_payload = dict(self._latest_summary)
            summary_payload.setdefault("total_requests_completed", 0)
            summary_payload.setdefault("total_requests_failed", 0)
            summary_payload.setdefault("total_tokens_generated", 0)
            summary_payload.setdefault("preempted_requests_total", self._preempted_total)
            summary_payload["polls"] = self._step
            summary_payload["elapsed_s"] = max(0.0, time.monotonic() - self._start_monotonic)
            run = self._run
            owns_run = self._owns_run
            callbacks = list(self._on_close_callbacks)
            self._on_close_callbacks.clear()

        for callback in callbacks:
            try:
                callback()
            except Exception:
                pass

        if run is None:
            return

        try:
            if hasattr(run, "summary") and not getattr(run, "finished", False):
                run.summary(**summary_payload)
        except Exception:
            logging.debug("Failed to record final Eyvan summary", exc_info=True)

        if owns_run:
            try:
                if hasattr(run, "finish"):
                    run.finish(status=status)
            finally:
                writer_thread = getattr(run, "_thread", None)
                if (
                    isinstance(writer_thread, threading.Thread)
                    and writer_thread.is_alive()
                    and writer_thread is not threading.current_thread()
                ):
                    writer_thread.join(timeout=2.0)
        else:
            try:
                if hasattr(run, "flush") and not getattr(run, "finished", False):
                    run.flush()
            except Exception:
                logging.debug("Failed to flush external Eyvan run", exc_info=True)

    def __enter__(self) -> EyvanMetricsExporter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        if exc_type is None:
            status = "finished"
        elif issubclass(exc_type, KeyboardInterrupt):
            status = "aborted"
        else:
            status = "failed"
        self.close(status=status)


class RichConsoleMonitor:
    """Rich console-based live monitoring for eSurge.

    Provides a terminal-based dashboard with real-time metrics display
    using the Rich library. Shows system metrics, scheduler status,
    model runner performance, cache utilization, and recent requests.

    The monitor runs in a background thread and updates the display
    at a configurable refresh rate.

    Attributes:
        console: Rich Console instance for rendering.
        refresh_rate: Update interval in seconds.
        running: Whether the monitor is currently running.
        layout: Rich Layout for organizing the display.

    Example:
        >>> monitor = RichConsoleMonitor(refresh_rate=1.0)
        >>> monitor.start()  # Blocks until Ctrl+C
        >>> # Or run in background:
        >>> monitor.running = True
        >>> monitor._thread.start()

    Raises:
        ImportError: If rich library is not installed.
    """

    def __init__(self, refresh_rate: float = 1.0):
        """Initialize console monitor.

        Args:
            refresh_rate: Update rate in seconds. Defaults to 1.0.

        Raises:
            ImportError: If rich is not available.
        """
        if not RICH_AVAILABLE:
            raise ImportError("rich not available. Install with: pip install rich")

        self.console = Console()
        self.refresh_rate = refresh_rate
        self.running = False
        self._thread: threading.Thread | None = None

        self.layout = Layout()
        self.layout.split_column(Layout(name="header", size=3), Layout(name="main"), Layout(name="footer", size=3))

        self.layout["main"].split_row(Layout(name="left"), Layout(name="right"))

    def _create_system_metrics_table(self, collector: MetricsCollector) -> Table:  # pyright: ignore[reportInvalidTypeForm]
        """Create system metrics table.

        Args:
            collector: MetricsCollector to read metrics from.

        Returns:
            Rich Table with system-level metrics.
        """
        table = Table(title="System Metrics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        system_metrics = collector.get_system_metrics()

        table.add_row("Requests/sec", f"{system_metrics.requests_per_second:.2f}")
        table.add_row("Avg Latency", f"{system_metrics.average_latency:.3f}s")
        table.add_row("Avg TTFT", f"{system_metrics.average_ttft:.3f}s")
        table.add_row("Avg Throughput", f"{system_metrics.average_throughput:.1f} tok/s")
        table.add_row("Total Completed", str(system_metrics.total_requests_completed))
        table.add_row("Total Failed", str(system_metrics.total_requests_failed))
        table.add_row("Total Tokens", str(system_metrics.total_tokens_generated))

        return table

    def _create_scheduler_metrics_table(self, collector: MetricsCollector) -> Table:  # pyright: ignore[reportInvalidTypeForm]
        """Create scheduler metrics table.

        Args:
            collector: MetricsCollector to read metrics from.

        Returns:
            Rich Table with scheduler statistics.
        """
        table = Table(title="Scheduler Status", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")

        with collector._lock:
            if collector.scheduler_metrics:
                latest = collector.scheduler_metrics[-1]
                table.add_row("Waiting Requests", str(latest.num_waiting_requests))
                table.add_row("Running Requests", str(latest.num_running_requests))
                table.add_row("Scheduled Tokens", str(latest.num_scheduled_tokens))
                table.add_row("Batch Size", str(latest.batch_size))
                table.add_row("Schedule Time", f"{latest.schedule_time * 1000:.2f}ms")
            else:
                table.add_row("No Data", "Available")

        return table

    def _create_runner_metrics_table(self, collector: MetricsCollector) -> Table:  # pyright: ignore[reportInvalidTypeForm]
        """Create runner metrics table.

        Args:
            collector: MetricsCollector to read metrics from.

        Returns:
            Rich Table with model runner performance.
        """
        table = Table(title="Model Runner", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        with collector._lock:
            if collector.runner_metrics:
                latest = collector.runner_metrics[-1]
                table.add_row("Execution Time", f"{latest.execution_time * 1000:.2f}ms")
                table.add_row("Batch Size", str(latest.batch_size))
                table.add_row("Tokens", str(latest.num_tokens))
                table.add_row("Tokens/sec", f"{latest.tokens_per_second:.1f}")
            else:
                table.add_row("No Data", "Available")

        return table

    def _create_cache_metrics_table(self, collector: MetricsCollector) -> Table:  # pyright: ignore[reportInvalidTypeForm]
        """Create cache metrics table.

        Args:
            collector: MetricsCollector to read metrics from.

        Returns:
            Rich Table with cache utilization statistics.
        """
        table = Table(title="Cache Status", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="blue")

        with collector._lock:
            if collector.cache_metrics:
                latest = collector.cache_metrics[-1]
                utilization = (latest.used_pages / latest.total_pages * 100) if latest.total_pages > 0 else 0
                table.add_row("Total Pages", str(latest.total_pages))
                table.add_row("Used Pages", str(latest.used_pages))
                table.add_row("Free Pages", str(latest.free_pages))
                table.add_row("Utilization", f"{utilization:.1f}%")
                table.add_row("Hit Rate", f"{latest.cache_hit_rate:.2%}")
            else:
                table.add_row("No Data", "Available")

        return table

    def _create_recent_requests_table(self, collector: MetricsCollector) -> Table:  # pyright: ignore[reportInvalidTypeForm]
        """Create recent requests table.

        Args:
            collector: MetricsCollector to read metrics from.

        Returns:
            Rich Table with the 5 most recent request summaries.
        """
        table = Table(title="Recent Requests", show_header=True)
        table.add_column("Request ID", style="dim")
        table.add_column("Status", style="bold")
        table.add_column("Latency", style="green")
        table.add_column("TTFT", style="yellow")
        table.add_column("Tokens", style="blue")

        with collector._lock:
            recent = list(collector.completed_requests)[-5:]  # Last 5 requests

            for req in recent:
                status = "v" if not req.error else "x"
                status_style = "green" if not req.error else "red"

                table.add_row(
                    req.request_id[:8] + "...",
                    Text(status, style=status_style),
                    f"{req.total_latency:.3f}s" if req.total_latency else "N/A",
                    f"{req.time_to_first_token:.3f}s" if req.time_to_first_token else "N/A",
                    str(req.generated_tokens),
                )

        return table

    def _update_layout(self) -> None:
        """Update the layout with current metrics.

        Fetches metrics from the global collector and rebuilds all
        display tables.
        """
        collector = get_metrics_collector()
        if not collector:
            self.layout["header"].update(Panel("No metrics collector initialized", style="red"))
            return

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.layout["header"].update(Panel(f"eSurge Live Monitor - {timestamp}", style="bold blue"))

        left_layout = Layout()
        left_layout.split_column(
            Layout(self._create_system_metrics_table(collector)), Layout(self._create_recent_requests_table(collector))
        )
        self.layout["left"].update(left_layout)

        right_layout = Layout()
        right_layout.split_column(
            Layout(self._create_scheduler_metrics_table(collector)),
            Layout(self._create_runner_metrics_table(collector)),
            Layout(self._create_cache_metrics_table(collector)),
        )
        self.layout["right"].update(right_layout)

        self.layout["footer"].update(Panel("Press Ctrl+C to stop monitoring", style="dim"))

    def start(self, blocking: bool = False) -> None:
        """Start the live console monitor.

        Starts a background daemon thread that continuously updates the
        display.  By default the call returns immediately so the engine
        can continue processing requests.  Pass ``blocking=True`` to
        block until Ctrl+C (useful when running the monitor standalone).

        Args:
            blocking: If True, block the calling thread until interrupted.
        """
        if self.running:
            return

        self.running = True

        def _monitor_loop():
            with Live(self.layout, console=self.console, refresh_per_second=1 / self.refresh_rate):
                while self.running:
                    try:
                        self._update_layout()
                        time.sleep(self.refresh_rate)
                    except KeyboardInterrupt:
                        break

        self._thread = threading.Thread(target=_monitor_loop, daemon=True)
        self._thread.start()

        if blocking:
            try:
                self._thread.join()
            except KeyboardInterrupt:
                self.stop()

    def stop(self) -> None:
        """Stop the console monitor.

        Signals the background thread to stop and waits for it to finish.
        """
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)


class eSurgeMonitoringServer:
    """Monitoring server for Prometheus export and/or Eyvan metrics tracking.

    Provides a unified interface for running Prometheus metrics export,
    :class:`EyvanMetricsExporter` polling, or both concurrently with
    independent polling intervals.

    Attributes:
        prometheus_port: Port for the Prometheus HTTP endpoint.
        dashboard_port: Deprecated, kept for compatibility.
        update_interval: Interval between Prometheus metrics updates.
        enable_prometheus: Whether Prometheus export is enabled.
        prometheus_metrics: PrometheusMetrics instance if enabled and available.
        eyvan_exporter: Optional EyvanMetricsExporter polled by this server.
        eyvan_update_interval: Interval between Eyvan metric updates.
        running: Whether the server is currently running.

    Example:
        >>> server = eSurgeMonitoringServer(
        ...     prometheus_port=8000,
        ...     update_interval=1.0
        ... )
        >>> server.start()
        >>> # Metrics available at http://localhost:8000/metrics
        >>> server.stop()
    """

    def __init__(
        self,
        prometheus_port: int = 8000,
        dashboard_port: int | None = None,
        metrics_prefix: str = "esurge_",
        update_interval: float = 1.0,
        enable_prometheus: bool = True,
        eyvan_exporter: EyvanMetricsExporter | None = None,
        eyvan_update_interval: float | None = None,
    ):
        """Initialize monitoring server.

        Args:
            prometheus_port: Port for Prometheus metrics endpoint.
            dashboard_port: Deprecated; retained for compatibility (ignored).
            metrics_prefix: Prefix for Prometheus metrics.
            update_interval: Update interval in seconds for Prometheus polling.
            enable_prometheus: Whether to expose the Prometheus HTTP endpoint
                and register Prometheus collectors.
            eyvan_exporter: Optional :class:`EyvanMetricsExporter` to poll from
                the server's background updater thread.
            eyvan_update_interval: Optional polling interval in seconds for
                ``eyvan_exporter`` (defaults to ``update_interval``).
        """
        if (
            isinstance(update_interval, bool)
            or not isinstance(update_interval, int | float)
            or not math.isfinite(update_interval)
            or update_interval <= 0
        ):
            raise ValueError("update_interval must be a positive finite number")
        if eyvan_update_interval is not None and (
            isinstance(eyvan_update_interval, bool)
            or not isinstance(eyvan_update_interval, int | float)
            or not math.isfinite(eyvan_update_interval)
            or eyvan_update_interval <= 0
        ):
            raise ValueError("eyvan_update_interval must be a positive finite number")

        self.prometheus_port = prometheus_port
        self.dashboard_port = dashboard_port
        self.update_interval = float(update_interval)
        self.enable_prometheus = bool(enable_prometheus)
        self.eyvan_exporter = eyvan_exporter
        self.eyvan_update_interval = (
            float(eyvan_update_interval) if eyvan_update_interval is not None else float(update_interval)
        )

        self.metrics_prefix = metrics_prefix
        self.prometheus_metrics = None
        if self.enable_prometheus and PROMETHEUS_AVAILABLE:
            self.prometheus_metrics = PrometheusMetrics(metrics_prefix)

        self.running = False
        self._update_thread: threading.Thread | None = None
        self._update_stop_event = threading.Event()
        self._http_server = None
        self._http_thread: threading.Thread | None = None

    def _update_metrics_loop(self, stop_event: threading.Event | None = None) -> None:
        """Poll the collector until this updater's stop event is set.

        Args:
            stop_event: Per-start lifecycle event; defaults to the current event.
        """
        if stop_event is None:
            stop_event = self._update_stop_event
        next_prom_at = 0.0
        next_eyvan_at = 0.0
        while not stop_event.is_set():
            now = time.monotonic()
            prom_metrics = self.prometheus_metrics if self.enable_prometheus else None
            eyvan_exporter = (
                self.eyvan_exporter if self.eyvan_exporter is not None and not self.eyvan_exporter.closed else None
            )
            prom_active = prom_metrics is not None
            eyvan_active = eyvan_exporter is not None

            collector = None
            try:
                collector = get_metrics_collector()
            except Exception as e:
                logging.error(f"Error fetching metrics collector: {e}")

            if prom_active and now >= next_prom_at:
                try:
                    if collector:
                        prom_metrics.update_from_metrics_collector(collector)
                except Exception as e:
                    logging.error(f"Error updating Prometheus metrics: {e}")
                finally:
                    next_prom_at = time.monotonic() + self.update_interval

            if eyvan_active and now >= next_eyvan_at:
                try:
                    if collector:
                        eyvan_exporter.update_from_metrics_collector(collector)
                except Exception as e:
                    logging.error(f"Error updating Eyvan metrics: {e}")
                finally:
                    next_eyvan_at = time.monotonic() + self.eyvan_update_interval

            after = time.monotonic()
            waits: list[float] = []
            if prom_active:
                waits.append(max(0.0, next_prom_at - after))
            if eyvan_active:
                waits.append(max(0.0, next_eyvan_at - after))
            sleep_for = min(waits) if waits else self.update_interval
            stop_event.wait(sleep_for)

    def start_prometheus_server(self) -> None:
        """Start Prometheus metrics server.

        Starts the HTTP server for Prometheus scraping on the configured port.
        """
        if not self.enable_prometheus:
            return
        if not PROMETHEUS_AVAILABLE:
            logging.warning("Prometheus client not available, skipping Prometheus server")
            return

        if self._http_server is not None:
            return

        # Keep the (server, thread) pair so ``stop`` can release the listening port.
        self._http_server, self._http_thread = start_http_server(self.prometheus_port)
        logging.info(f"Prometheus metrics server started on port {self.prometheus_port}")
        logging.info(f"Metrics available at: http://localhost:{self.prometheus_port}/metrics")

    def start(self) -> None:
        """Start the monitoring server.

        Starts the Prometheus HTTP server (when enabled and available) and the
        background metrics update thread for Prometheus and/or Eyvan export.
        """
        if self.running:
            return

        if self.enable_prometheus and self.prometheus_metrics is None and PROMETHEUS_AVAILABLE:
            # ``stop`` released the previous metric set; build a fresh one on restart.
            self.prometheus_metrics = PrometheusMetrics(self.metrics_prefix)

        self.running = True

        try:
            if self.enable_prometheus:
                self.start_prometheus_server()
            self._update_stop_event = threading.Event()
            self._update_thread = threading.Thread(
                target=self._update_metrics_loop, args=(self._update_stop_event,), daemon=True
            )
            self._update_thread.start()
        except BaseException:
            # Failed binds and updater startup must release metric names and
            # any listener already opened, just like an ordinary stop.
            self.stop()
            raise

        logging.info("eSurge monitoring server started")

    def stop(self) -> None:
        """Stop the monitoring server.

        Signals the update thread to stop and waits for it to finish, flushes
        and closes any attached :class:`EyvanMetricsExporter`, shuts down the
        Prometheus HTTP listener (freeing its port), and unregisters the
        Prometheus metric set so :meth:`start` can be called again on the same
        port.
        """
        self.running = False
        self._update_stop_event.set()
        if self._update_thread:
            if self._update_thread.is_alive():
                self._update_thread.join(timeout=2.0)
            self._update_thread = None

        if self.eyvan_exporter is not None and not self.eyvan_exporter.closed:
            try:
                collector = get_metrics_collector()
                if collector:
                    self.eyvan_exporter.update_from_metrics_collector(collector)
            except Exception:
                logging.debug("Failed final Eyvan metrics poll on server stop", exc_info=True)
            try:
                self.eyvan_exporter.close()
            except Exception:
                logging.debug("Failed to close EyvanMetricsExporter on server stop", exc_info=True)

        if self._http_server is not None:
            try:
                self._http_server.shutdown()
                self._http_server.server_close()
            except Exception:
                logging.debug("Failed to shut down Prometheus HTTP server", exc_info=True)
            self._http_server = None
        if self._http_thread is not None:
            self._http_thread.join(timeout=1.0)
            self._http_thread = None

        if self.prometheus_metrics is not None:
            self.prometheus_metrics.close()
            self.prometheus_metrics = None

        logging.info("eSurge monitoring server stopped")


class _EyvanTrackerHandle:
    """Owner-scoped background polling loop driving an :class:`EyvanMetricsExporter`."""

    def __init__(self, exporter: EyvanMetricsExporter, update_interval: float) -> None:
        self.exporter = exporter
        self.update_interval = float(update_interval)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stopped = False
        self._state_lock = threading.Lock()
        self.exporter._add_close_callback(self._stop_event.set)

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            if self.exporter.closed:
                break
            try:
                collector = get_metrics_collector()
                if collector:
                    self.exporter.update_from_metrics_collector(collector)
            except Exception as e:
                logging.error(f"Error updating Eyvan metrics: {e}")
            self._stop_event.wait(self.update_interval)

    def start(self) -> None:
        with self._state_lock:
            if self._stopped or self.exporter.closed or not self.exporter.active:
                return
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._poll_loop,
                name="esurge-eyvan-metrics",
                daemon=True,
            )
            self._thread.start()

    def stop(self, status: str = "finished") -> None:
        with self._state_lock:
            self._stopped = True
            self._stop_event.set()
            thread = self._thread
            self._thread = None

        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)

        if not self.exporter.closed:
            try:
                collector = get_metrics_collector()
                if collector:
                    self.exporter.update_from_metrics_collector(collector)
            except Exception:
                logging.debug("Failed final Eyvan metrics poll on tracker stop", exc_info=True)
            try:
                self.exporter.close(status=status)
            except Exception:
                logging.debug("Failed to close EyvanMetricsExporter on tracker stop", exc_info=True)


_monitoring_server: eSurgeMonitoringServer | None = None
_console_monitor: RichConsoleMonitor | None = None
_monitoring_lock = threading.RLock()
# Each successful start acquires one idempotent lease per owner. ``None`` is
# the legacy process-wide owner; stack shutdown must not release that lease.
# Non-None owners are tracked via weakref.finalize so abandoned engines are
# never pinned in module globals and automatically release their leases on GC.
_monitoring_server_owners: dict[int | None, tuple[object, weakref.finalize | None] | None] = {}
_console_monitor_owners: dict[int | None, tuple[object, weakref.finalize | None] | None] = {}
# Eyvan tracking is strictly per-owner so separate engines never cross-pollinate
# runs or stop each other's trackers.
_eyvan_trackers: dict[int | None, tuple[object, _EyvanTrackerHandle, weakref.finalize | None]] = {}


def _detach_owner_lease(entry: tuple[object, weakref.finalize | None] | None) -> None:
    if entry is not None and entry[1] is not None:
        entry[1].detach()


def _detach_eyvan_entry(entry: tuple[object, _EyvanTrackerHandle, weakref.finalize | None] | None) -> None:
    if entry is not None and entry[2] is not None:
        entry[2].detach()


def _release_gc_owner_lease(owner_key: int, token: object, service: str) -> None:
    """Release an owner's lease automatically when the owner object is garbage-collected."""
    global _monitoring_server, _console_monitor

    eyvan_handle: _EyvanTrackerHandle | None = None
    with _monitoring_lock:
        if service == "server":
            entry = _monitoring_server_owners.get(owner_key)
            if entry is None or entry[0] is not token:
                return
            _monitoring_server_owners.pop(owner_key, None)
            if not _monitoring_server_owners and _monitoring_server is not None:
                _monitoring_server.stop()
                _monitoring_server = None
        elif service == "console":
            entry = _console_monitor_owners.get(owner_key)
            if entry is None or entry[0] is not token:
                return
            _console_monitor_owners.pop(owner_key, None)
            if not _console_monitor_owners and _console_monitor is not None:
                _console_monitor.stop()
                _console_monitor = None
        elif service == "eyvan":
            eyvan_entry = _eyvan_trackers.get(owner_key)
            if eyvan_entry is None or eyvan_entry[0] is not token:
                return
            _eyvan_trackers.pop(owner_key, None)
            eyvan_handle = eyvan_entry[1]

    if eyvan_handle is not None:
        eyvan_handle.stop()


def _acquire_owner_lease(
    owners: dict[int | None, tuple[object, weakref.finalize | None] | None],
    owner: object | None,
    service: str,
) -> None:
    """Record an idempotent owner lease without holding a strong reference to ``owner``."""
    if owner is None:
        owners[None] = None
        return
    owner_key = id(owner)
    existing = owners.get(owner_key)
    if existing is not None:
        _tok, existing_fin = existing
        if existing_fin is None or existing_fin.alive:
            return
    token = object()
    try:
        fin = weakref.finalize(owner, _release_gc_owner_lease, owner_key, token, service)
        fin.atexit = False
    except TypeError:
        fin = None
    owners[owner_key] = (token, fin)


def start_monitoring_server(
    prometheus_port: int = 8000,
    dashboard_port: int | None = None,
    update_interval: float = 1.0,
    *,
    enable_prometheus: bool = True,
    eyvan_exporter: EyvanMetricsExporter | None = None,
    eyvan_update_interval: float | None = None,
    owner: object | None = None,
) -> eSurgeMonitoringServer:
    """Start the global monitoring server.

    Creates and starts the monitoring server singleton. Subsequent calls
    return the existing instance.

    Args:
        prometheus_port: Port for Prometheus metrics endpoint.
        dashboard_port: Deprecated; retained for compatibility (ignored).
        update_interval: Update interval in seconds.
        enable_prometheus: Whether to enable Prometheus HTTP export.
        eyvan_exporter: Optional :class:`EyvanMetricsExporter` attached to the
            server when creating it.
        eyvan_update_interval: Optional polling interval in seconds for
            ``eyvan_exporter``.
        owner: Optional identity acquiring a shared service lease. Release it
            with ``stop_monitoring(owner=owner)``. Repeated starts by the same
            owner are idempotent. ``None`` retains process-wide ownership.

    Returns:
        The eSurgeMonitoringServer instance.

    Example:
        >>> server = start_monitoring_server(prometheus_port=9090)
        >>> print(f"Metrics at http://localhost:9090/metrics")
    """
    global _monitoring_server

    with _monitoring_lock:
        if _monitoring_server is None:
            server_kwargs: dict[str, tp.Any] = {
                "prometheus_port": prometheus_port,
                "dashboard_port": dashboard_port,
                "update_interval": update_interval,
            }
            if not enable_prometheus:
                server_kwargs["enable_prometheus"] = False
            if eyvan_exporter is not None:
                server_kwargs["eyvan_exporter"] = eyvan_exporter
            if eyvan_update_interval is not None:
                server_kwargs["eyvan_update_interval"] = eyvan_update_interval
            server = eSurgeMonitoringServer(**server_kwargs)
            # Publish only after startup succeeds so a failed bind cannot pin the
            # global service to an unusable port or configuration.
            server.start()
            _monitoring_server = server
        else:
            _monitoring_server.start()
        _acquire_owner_lease(_monitoring_server_owners, owner, "server")
        return _monitoring_server


def start_eyvan_tracking(
    name: str | None = None,
    *,
    project: str | None = None,
    group: str | None = None,
    tags: tp.Iterable[str] | None = None,
    uri: str | None = None,
    run_id: str | None = None,
    update_interval: float = 5.0,
    flush_interval_s: float = 5.0,
    prefix: str = "esurge/",
    config: tp.Mapping[str, tp.Any] | None = None,
    run: tp.Any | None = None,
    owner: object | None = None,
) -> EyvanMetricsExporter:
    """Start an owner-scoped Eyvan metrics tracking exporter and polling thread.

    Unlike the process-wide Prometheus HTTP server, each ``owner`` receives its
    own isolated :class:`EyvanMetricsExporter` and background polling thread so
    multiple engines never interfere with one another's runs. Releasing that
    owner via ``stop_monitoring(owner=owner)`` stops, performs a final poll,
    writes the run summary, and finishes only that owner's run. Repeated calls
    for the same live owner are idempotent and return its existing exporter.

    Args:
        name: Human-readable Eyvan run name.
        project: Local Eyvan project scope (default ``"esurge"``).
        group: Optional Eyvan run group.
        tags: Optional iterable of run tags.
        uri: Optional Eyvan store URI (``gs://...``, ``file://...``, or path).
        run_id: Optional explicit Eyvan run id.
        update_interval: Polling interval in seconds between collector reads.
        flush_interval_s: Seconds between background Eyvan segment flushes.
        prefix: Metric key prefix (default ``"esurge/"``).
        config: Optional engine configuration metadata to merge into run config.
        run: Optional pre-built ``eyvan.track.Run`` handle.
        owner: Optional owner object for scoped lifecycle cleanup (and automatic
            GC finalization without pinning ``owner``). ``None`` uses the
            process-wide slot.

    Returns:
        The active :class:`EyvanMetricsExporter` instance.

    Raises:
        ImportError: If ``run`` is ``None`` and ``eyvan.track`` is unavailable.
        ValueError: If ``update_interval`` or ``flush_interval_s`` is invalid.
    """
    if (
        isinstance(update_interval, bool)
        or not isinstance(update_interval, int | float)
        or not math.isfinite(update_interval)
        or update_interval <= 0
    ):
        raise ValueError("update_interval must be a positive finite number")

    owner_key = None if owner is None else id(owner)
    stale_handle: _EyvanTrackerHandle | None = None

    with _monitoring_lock:
        existing = _eyvan_trackers.get(owner_key)
        if existing is not None:
            _tok, existing_handle, existing_fin = existing
            owner_alive = owner is None or existing_fin is None or existing_fin.alive
            if owner_alive and not existing_handle.exporter.closed:
                existing_handle.start()
                return existing_handle.exporter
            _detach_eyvan_entry(_eyvan_trackers.pop(owner_key, None))
            stale_handle = existing_handle

    if stale_handle is not None:
        stale_handle.stop()

    exporter = EyvanMetricsExporter(
        name=name,
        project=project,
        group=group,
        tags=tags,
        uri=uri,
        run_id=run_id,
        flush_interval_s=flush_interval_s,
        prefix=prefix,
        config=config,
        run=run,
    )
    handle = _EyvanTrackerHandle(exporter=exporter, update_interval=float(update_interval))
    try:
        handle.start()
    except BaseException:
        handle.stop()
        raise

    token = object()
    fin: weakref.finalize | None = None
    if owner is not None:
        try:
            fin = weakref.finalize(owner, _release_gc_owner_lease, owner_key, token, "eyvan")
            fin.atexit = False
        except TypeError:
            fin = None

    replaced_handle: _EyvanTrackerHandle | None = None
    existing_exporter: EyvanMetricsExporter | None = None
    with _monitoring_lock:
        existing = _eyvan_trackers.get(owner_key)
        if existing is not None:
            _tok, existing_handle, existing_fin = existing
            owner_alive = owner is None or existing_fin is None or existing_fin.alive
            if owner_alive and not existing_handle.exporter.closed:
                # Another thread won the race for the same owner; discard ours.
                if fin is not None:
                    fin.detach()
                existing_exporter = existing_handle.exporter
            else:
                _detach_eyvan_entry(_eyvan_trackers.pop(owner_key, None))
                replaced_handle = existing_handle
                _eyvan_trackers[owner_key] = (token, handle, fin)
        else:
            _eyvan_trackers[owner_key] = (token, handle, fin)

    if existing_exporter is not None:
        handle.stop()
        return existing_exporter
    if replaced_handle is not None:
        replaced_handle.stop()
    return exporter


def start_console_monitor(refresh_rate: float = 1.0, *, owner: object | None = None) -> RichConsoleMonitor:
    """Start the global console monitor.

    Creates and starts the console monitor singleton. Subsequent calls
    return the existing instance.

    Args:
        refresh_rate: Display refresh rate in seconds.
        owner: Optional identity acquiring an idempotent shared service lease,
            released with ``stop_monitoring(owner=owner)``. ``None`` retains
            process-wide ownership.

    Returns:
        The RichConsoleMonitor instance.

    Example:
        >>> monitor = start_console_monitor(refresh_rate=0.5)
        >>> # Press Ctrl+C to stop
    """
    global _console_monitor

    with _monitoring_lock:
        if _console_monitor is None:
            monitor = RichConsoleMonitor(refresh_rate=refresh_rate)
            monitor.start()
            _console_monitor = monitor
        else:
            _console_monitor.start()
        _acquire_owner_lease(_console_monitor_owners, owner, "console")
        return _console_monitor


def stop_monitoring(*, owner: object | None = None) -> None:
    """Stop all monitoring services, or release only one owner's leases.

    With no owner, stops all monitoring services (Prometheus server, console
    monitor, and every active Eyvan tracker) and clears global state
    (process-wide cleanup). With an owner, releases only that owner's leases
    and stops only that owner's Eyvan tracker; shared services (Prometheus,
    console) are stopped only when their last lease is released. An owner that
    never started a service cannot stop it.

    Args:
        owner: Identity passed to the start functions, or ``None`` to stop all
            services regardless of ownership.

    Example:
        >>> start_monitoring_server()
        >>> # ... do work ...
        >>> stop_monitoring()  # Cleanup
    """
    global _monitoring_server, _console_monitor

    eyvan_handles_to_stop: list[_EyvanTrackerHandle] = []

    with _monitoring_lock:
        owner_key = None if owner is None else id(owner)
        stop_server = owner is None or owner_key in _monitoring_server_owners
        stop_console = owner is None or owner_key in _console_monitor_owners
        if owner is None:
            for entry in _monitoring_server_owners.values():
                _detach_owner_lease(entry)
            for entry in _console_monitor_owners.values():
                _detach_owner_lease(entry)
            for eyvan_entry in _eyvan_trackers.values():
                _detach_eyvan_entry(eyvan_entry)
                eyvan_handles_to_stop.append(eyvan_entry[1])
            _monitoring_server_owners.clear()
            _console_monitor_owners.clear()
            _eyvan_trackers.clear()
        else:
            _detach_owner_lease(_monitoring_server_owners.pop(owner_key, None))
            _detach_owner_lease(_console_monitor_owners.pop(owner_key, None))
            eyvan_entry = _eyvan_trackers.pop(owner_key, None)
            if eyvan_entry is not None:
                _detach_eyvan_entry(eyvan_entry)
                eyvan_handles_to_stop.append(eyvan_entry[1])

        if stop_server and not _monitoring_server_owners and _monitoring_server is not None:
            _monitoring_server.stop()
            _monitoring_server = None

        if stop_console and not _console_monitor_owners and _console_monitor is not None:
            _console_monitor.stop()
            _console_monitor = None

    for handle in eyvan_handles_to_stop:
        handle.stop()
