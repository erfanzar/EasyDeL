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
including Prometheus metrics export for production monitoring and a Rich-based
console monitor for development and debugging.

Classes:
    PrometheusMetrics: Prometheus metrics exporter with standard metrics.
    RichConsoleMonitor: Rich console-based live monitoring dashboard.
    eSurgeMonitoringServer: Combined monitoring server for metrics and visualization.

Functions:
    start_monitoring_server: Start the global monitoring server.
    start_console_monitor: Start the global console monitor.
    stop_monitoring: Stop all monitoring services.

Constants:
    PROMETHEUS_AVAILABLE: Whether prometheus_client is installed.
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
    For production monitoring, use Prometheus with Grafana dashboards.
    The console monitor is intended for development and debugging.
"""

from __future__ import annotations

import logging
import threading
import time

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

        self.requests_total = Counter(f"{prefix}requests_total", "Total number of requests", ["status"])

        self.request_duration = Histogram(
            f"{prefix}request_duration_seconds",
            "Request duration in seconds",
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )

        self.time_to_first_token = Histogram(
            f"{prefix}time_to_first_token_seconds",
            "Time to first token in seconds",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        )

        self.tokens_generated_total = Counter(f"{prefix}tokens_generated_total", "Total number of tokens generated")

        self.tokens_per_second = Gauge(f"{prefix}tokens_per_second", "Current tokens per second throughput")

        self.waiting_requests = Gauge(f"{prefix}waiting_requests", "Number of requests waiting to be scheduled")

        self.running_requests = Gauge(f"{prefix}running_requests", "Number of currently running requests")

        self.scheduled_tokens = Gauge(f"{prefix}scheduled_tokens", "Number of tokens scheduled in current batch")

        self.preempted_requests_total = Counter(
            f"{prefix}preempted_requests_total", "Total number of preempted requests"
        )

        self.schedule_duration = Histogram(
            f"{prefix}schedule_duration_seconds",
            "Scheduler operation duration in seconds",
            buckets=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
        )

        self.model_execution_duration = Histogram(
            f"{prefix}model_execution_duration_seconds",
            "Model execution duration in seconds",
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        )

        self.batch_size = Gauge(f"{prefix}batch_size", "Current batch size")

        self.cache_pages_total = Gauge(f"{prefix}cache_pages_total", "Total number of cache pages")

        self.cache_pages_used = Gauge(f"{prefix}cache_pages_used", "Number of used cache pages")

        self.cache_hit_rate = Gauge(f"{prefix}cache_hit_rate", "Cache hit rate (0-1)")

        self.system_info = Info(f"{prefix}system_info", "System information")
        self.system_info.info(self._collect_system_info())

        # Every collector registered in the default registry, so ``close`` can
        # release the metric names and a later instance can register them again.
        self._collectors = [
            value for value in vars(self).values() if isinstance(value, (Counter, Gauge, Histogram, Info))
        ]

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
        import platform

        import jax

        info = {
            "python_version": platform.python_version(),
            "jax_version": str(jax.__version__),
        }
        try:
            info["jax_backend"] = str(jax.default_backend())
            info["device_count"] = str(jax.device_count())
            info["process_count"] = str(jax.process_count())
        except Exception:
            pass
        return info

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
    """Monitoring server for Prometheus export and console monitoring.

    Provides a unified interface for starting Prometheus metrics export
    and optionally a console-based monitoring display.

    Attributes:
        prometheus_port: Port for the Prometheus HTTP endpoint.
        dashboard_port: Deprecated, kept for compatibility.
        update_interval: Interval between metrics updates.
        prometheus_metrics: PrometheusMetrics instance if available.
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
    ):
        """Initialize monitoring server.

        Args:
            prometheus_port: Port for Prometheus metrics endpoint.
            dashboard_port: Deprecated; retained for compatibility (ignored).
            metrics_prefix: Prefix for Prometheus metrics.
            update_interval: Update interval in seconds.
        """
        self.prometheus_port = prometheus_port
        self.dashboard_port = dashboard_port
        self.update_interval = update_interval

        self.metrics_prefix = metrics_prefix
        self.prometheus_metrics = None
        if PROMETHEUS_AVAILABLE:
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
        while not stop_event.is_set():
            try:
                collector = get_metrics_collector()
                if collector and self.prometheus_metrics:
                    self.prometheus_metrics.update_from_metrics_collector(collector)
            except Exception as e:
                logging.error(f"Error updating metrics: {e}")
            stop_event.wait(self.update_interval)

    def start_prometheus_server(self) -> None:
        """Start Prometheus metrics server.

        Starts the HTTP server for Prometheus scraping on the configured port.
        """
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

        Starts both the Prometheus HTTP server and the background
        metrics update thread.
        """
        if self.running:
            return

        if self.prometheus_metrics is None and PROMETHEUS_AVAILABLE:
            # ``stop`` released the previous metric set; build a fresh one on restart.
            self.prometheus_metrics = PrometheusMetrics(self.metrics_prefix)

        self.running = True

        try:
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

        Signals the update thread to stop and waits for it to finish, shuts
        down the Prometheus HTTP listener (freeing its port) and unregisters
        the metric set, so :meth:`start` can be called again on the same port.
        """
        self.running = False
        self._update_stop_event.set()
        if self._update_thread:
            if self._update_thread.is_alive():
                self._update_thread.join(timeout=1.0)
            self._update_thread = None

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


_monitoring_server: eSurgeMonitoringServer | None = None
_console_monitor: RichConsoleMonitor | None = None
_monitoring_lock = threading.RLock()
# Each successful start acquires one idempotent lease per owner. ``None`` is
# the legacy process-wide owner; stack shutdown must not release that lease.
_monitoring_server_owners: dict[int | None, object | None] = {}
_console_monitor_owners: dict[int | None, object | None] = {}


def start_monitoring_server(
    prometheus_port: int = 8000,
    dashboard_port: int | None = None,
    update_interval: float = 1.0,
    *,
    owner: object | None = None,
) -> eSurgeMonitoringServer:
    """Start the global monitoring server.

    Creates and starts the monitoring server singleton. Subsequent calls
    return the existing instance.

    Args:
        prometheus_port: Port for Prometheus metrics endpoint.
        dashboard_port: Deprecated; retained for compatibility (ignored).
        update_interval: Update interval in seconds.
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
            server = eSurgeMonitoringServer(
                prometheus_port=prometheus_port,
                dashboard_port=dashboard_port,
                update_interval=update_interval,
            )
            # Publish only after startup succeeds so a failed bind cannot pin the
            # global service to an unusable port or configuration.
            server.start()
            _monitoring_server = server
        else:
            _monitoring_server.start()
        _monitoring_server_owners[None if owner is None else id(owner)] = owner
        return _monitoring_server


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
        _console_monitor_owners[None if owner is None else id(owner)] = owner
        return _console_monitor


def stop_monitoring(*, owner: object | None = None) -> None:
    """Stop all monitoring services, or release only one owner's leases.

    With no owner, stops both services and clears the global instances
    (process-wide cleanup). With an owner, releases only
    that owner's leases; a service is stopped and cleared only when its last
    lease is released. An owner that never started a service cannot stop it.

    Args:
        owner: Identity passed to the start functions, or ``None`` to stop all
            services regardless of ownership.

    Example:
        >>> start_monitoring_server()
        >>> # ... do work ...
        >>> stop_monitoring()  # Cleanup
    """
    global _monitoring_server, _console_monitor

    with _monitoring_lock:
        owner_key = None if owner is None else id(owner)
        stop_server = owner is None or owner_key in _monitoring_server_owners
        stop_console = owner is None or owner_key in _console_monitor_owners
        if owner is None:
            _monitoring_server_owners.clear()
            _console_monitor_owners.clear()
        else:
            _monitoring_server_owners.pop(owner_key, None)
            _console_monitor_owners.pop(owner_key, None)

        if stop_server and not _monitoring_server_owners and _monitoring_server is not None:
            _monitoring_server.stop()
            _monitoring_server = None

        if stop_console and not _console_monitor_owners and _console_monitor is not None:
            _console_monitor.stop()
            _console_monitor = None
