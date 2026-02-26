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
    from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server

    _prometheus_available = True
except ImportError:
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

        # Request metrics
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

        # Scheduler metrics
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

        # Model runner metrics
        self.model_execution_duration = Histogram(
            f"{prefix}model_execution_duration_seconds",
            "Model execution duration in seconds",
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
        )

        self.batch_size = Gauge(f"{prefix}batch_size", "Current batch size")

        # Cache metrics
        self.cache_pages_total = Gauge(f"{prefix}cache_pages_total", "Total number of cache pages")

        self.cache_pages_used = Gauge(f"{prefix}cache_pages_used", "Number of used cache pages")

        self.cache_hit_rate = Gauge(f"{prefix}cache_hit_rate", "Cache hit rate (0-1)")

        # System info
        self.system_info = Info(f"{prefix}system_info", "System information")

    def update_from_metrics_collector(self, collector: MetricsCollector) -> None:
        """Update Prometheus metrics from the metrics collector.

        Fetches the latest metrics from the MetricsCollector and updates
        all Prometheus gauges, counters, and histograms accordingly.

        Args:
            collector: MetricsCollector instance to read metrics from.

        Note:
            This method is thread-safe and acquires the collector's lock.
        """
        system_metrics = collector.get_system_metrics()

        # Update system-level metrics
        self.tokens_per_second.set(system_metrics.average_throughput)

        # Update from recent completed requests
        with collector._lock:
            # Update request metrics from recent completions
            recent_requests = list(collector.completed_requests)[-10:]  # Last 10 requests

            for req_metrics in recent_requests:
                if req_metrics.error:
                    self.requests_total.labels(status="failed").inc()
                else:
                    self.requests_total.labels(status="completed").inc()

                if req_metrics.total_latency:
                    self.request_duration.observe(req_metrics.total_latency)

                if req_metrics.time_to_first_token:
                    self.time_to_first_token.observe(req_metrics.time_to_first_token)

                self.tokens_generated_total.inc(req_metrics.generated_tokens)

            # Update scheduler metrics
            if collector.scheduler_metrics:
                latest_scheduler = collector.scheduler_metrics[-1]
                self.waiting_requests.set(latest_scheduler.num_waiting_requests)
                self.running_requests.set(latest_scheduler.num_running_requests)
                self.scheduled_tokens.set(latest_scheduler.num_scheduled_tokens)
                self.schedule_duration.observe(latest_scheduler.schedule_time)

            # Update runner metrics
            if collector.runner_metrics:
                latest_runner = collector.runner_metrics[-1]
                self.model_execution_duration.observe(latest_runner.execution_time)
                self.batch_size.set(latest_runner.batch_size)

            # Update cache metrics
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

        # Layout for the console display
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

        # Header
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.layout["header"].update(Panel(f"eSurge Live Monitor - {timestamp}", style="bold blue"))

        # Left column
        left_layout = Layout()
        left_layout.split_column(
            Layout(self._create_system_metrics_table(collector)), Layout(self._create_recent_requests_table(collector))
        )
        self.layout["left"].update(left_layout)

        # Right column
        right_layout = Layout()
        right_layout.split_column(
            Layout(self._create_scheduler_metrics_table(collector)),
            Layout(self._create_runner_metrics_table(collector)),
            Layout(self._create_cache_metrics_table(collector)),
        )
        self.layout["right"].update(right_layout)

        # Footer
        self.layout["footer"].update(Panel("Press Ctrl+C to stop monitoring", style="dim"))

    def start(self) -> None:
        """Start the live console monitor.

        Starts a background thread that continuously updates the display.
        Blocks the main thread until Ctrl+C is pressed.
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

        # Initialize Prometheus metrics if available
        self.prometheus_metrics = None
        if PROMETHEUS_AVAILABLE:
            self.prometheus_metrics = PrometheusMetrics(metrics_prefix)

        self.running = False
        self._update_thread: threading.Thread | None = None

    def _update_metrics_loop(self) -> None:
        """Background thread to update Prometheus metrics.

        Continuously polls the MetricsCollector and updates Prometheus
        metrics at the configured interval.
        """
        while self.running:
            try:
                collector = get_metrics_collector()
                if collector and self.prometheus_metrics:
                    self.prometheus_metrics.update_from_metrics_collector(collector)
                time.sleep(self.update_interval)
            except Exception as e:
                logging.error(f"Error updating metrics: {e}")
                time.sleep(self.update_interval)

    def start_prometheus_server(self) -> None:
        """Start Prometheus metrics server.

        Starts the HTTP server for Prometheus scraping on the configured port.
        """
        if not PROMETHEUS_AVAILABLE:
            logging.warning("Prometheus client not available, skipping Prometheus server")
            return

        start_http_server(self.prometheus_port)
        logging.info(f"Prometheus metrics server started on port {self.prometheus_port}")
        logging.info(f"Metrics available at: http://localhost:{self.prometheus_port}/metrics")

    def start(self) -> None:
        """Start the monitoring server.

        Starts both the Prometheus HTTP server and the background
        metrics update thread.
        """
        if self.running:
            return

        self.running = True

        # Start Prometheus server
        self.start_prometheus_server()

        # Start metrics update thread
        self._update_thread = threading.Thread(target=self._update_metrics_loop, daemon=True)
        self._update_thread.start()

        logging.info("eSurge monitoring server started")

    def stop(self) -> None:
        """Stop the monitoring server.

        Signals the update thread to stop and waits for it to finish.
        """
        self.running = False
        if self._update_thread:
            self._update_thread.join(timeout=1.0)

        logging.info("eSurge monitoring server stopped")


# Global monitoring instances
_monitoring_server: eSurgeMonitoringServer | None = None
_console_monitor: RichConsoleMonitor | None = None


def start_monitoring_server(
    prometheus_port: int = 8000,
    dashboard_port: int | None = None,
    update_interval: float = 1.0,
) -> eSurgeMonitoringServer:
    """Start the global monitoring server.

    Creates and starts the monitoring server singleton. Subsequent calls
    return the existing instance.

    Args:
        prometheus_port: Port for Prometheus metrics endpoint.
        dashboard_port: Deprecated; retained for compatibility (ignored).
        update_interval: Update interval in seconds.

    Returns:
        The eSurgeMonitoringServer instance.

    Example:
        >>> server = start_monitoring_server(prometheus_port=9090)
        >>> print(f"Metrics at http://localhost:9090/metrics")
    """
    global _monitoring_server

    if _monitoring_server is None:
        _monitoring_server = eSurgeMonitoringServer(
            prometheus_port=prometheus_port,
            dashboard_port=dashboard_port,
            update_interval=update_interval,
        )

    _monitoring_server.start()
    return _monitoring_server


def start_console_monitor(refresh_rate: float = 1.0) -> RichConsoleMonitor:
    """Start the global console monitor.

    Creates and starts the console monitor singleton. Subsequent calls
    return the existing instance.

    Args:
        refresh_rate: Display refresh rate in seconds.

    Returns:
        The RichConsoleMonitor instance.

    Example:
        >>> monitor = start_console_monitor(refresh_rate=0.5)
        >>> # Press Ctrl+C to stop
    """
    global _console_monitor

    if _console_monitor is None:
        _console_monitor = RichConsoleMonitor(refresh_rate=refresh_rate)

    _console_monitor.start()
    return _console_monitor


def stop_monitoring() -> None:
    """Stop all monitoring services.

    Stops both the monitoring server and console monitor if they are running,
    and clears the global instances.

    Example:
        >>> start_monitoring_server()
        >>> # ... do work ...
        >>> stop_monitoring()  # Cleanup
    """
    global _monitoring_server, _console_monitor

    if _monitoring_server:
        _monitoring_server.stop()
        _monitoring_server = None

    if _console_monitor:
        _console_monitor.stop()
        _console_monitor = None
