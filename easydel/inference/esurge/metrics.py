# Copyright 2025 The EasyDeL Author @erfanzar (Erfan Zare Chavoshi).
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

"""eSurge Metrics Collection System."""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from jax import numpy as jnp


@dataclass
class RequestMetrics:
    """Metrics for a single request."""

    request_id: str
    start_time: float
    end_time: float | None = None
    first_token_time: float | None = None
    prompt_tokens: int = 0
    generated_tokens: int = 0
    total_tokens: int = 0
    tokens_per_second: float = 0.0
    time_to_first_token: float | None = None
    total_latency: float | None = None
    finish_reason: str | None = None
    error: str | None = None


@dataclass
class SchedulerMetrics:
    """Metrics for scheduler operations."""

    timestamp: float
    num_waiting_requests: int = 0
    num_running_requests: int = 0
    num_scheduled_tokens: int = 0
    num_preempted_requests: int = 0
    batch_size: int = 0
    schedule_time: float = 0.0


@dataclass
class ModelRunnerMetrics:
    """Metrics for model runner operations."""

    timestamp: float
    execution_time: float = 0.0
    batch_size: int = 0
    num_tokens: int = 0
    tokens_per_second: float = 0.0
    memory_usage: dict[str, Any] | None = None


@dataclass
class CacheMetrics:
    """Metrics for KV cache operations."""

    timestamp: float
    total_pages: int = 0
    used_pages: int = 0
    free_pages: int = 0
    cache_hit_rate: float = 0.0
    page_allocation_rate: float = 0.0
    page_free_rate: float = 0.0


@dataclass
class SystemMetrics:
    """System-wide metrics summary."""

    timestamp: float
    total_requests_completed: int = 0
    total_requests_failed: int = 0
    total_tokens_generated: int = 0
    average_latency: float = 0.0
    average_ttft: float = 0.0
    average_throughput: float = 0.0
    requests_per_second: float = 0.0


class MetricsCollector:
    """Centralized metrics collection and logging system for eSurge."""

    def __init__(
        self,
        log_file: str | None = None,
        log_interval: float = 10.0,
        history_size: int = 1000,
        enable_detailed_logging: bool = True,
    ):
        """Initialize the metrics collector.

        Args:
            log_file: Path to metrics log file (JSON format)
            log_interval: Interval in seconds to log summary metrics
            history_size: Number of metrics records to keep in memory
            enable_detailed_logging: Whether to log detailed per-request metrics
        """
        self.log_file = log_file
        self.log_interval = log_interval
        self.history_size = history_size
        self.enable_detailed_logging = enable_detailed_logging

        # Thread-safe data structures
        self._lock = Lock()

        # Metrics storage
        self.request_metrics: dict[str, RequestMetrics] = {}
        self.completed_requests: deque[RequestMetrics] = deque(maxlen=history_size)
        self.scheduler_metrics: deque[SchedulerMetrics] = deque(maxlen=history_size)
        self.runner_metrics: deque[ModelRunnerMetrics] = deque(maxlen=history_size)
        self.cache_metrics: deque[CacheMetrics] = deque(maxlen=history_size)

        # Counters and aggregates
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)

        # Last log time
        self.last_log_time = time.time()

        # Setup logging
        self.logger = logging.getLogger("eSurge.metrics")
        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def start_request(self, request_id: str, prompt_tokens: int = 0) -> None:
        """Start tracking metrics for a new request."""
        with self._lock:
            self.request_metrics[request_id] = RequestMetrics(
                request_id=request_id,
                start_time=time.time(),
                prompt_tokens=prompt_tokens,
                total_tokens=prompt_tokens,
            )
            self.counters["total_requests"] += 1

    def record_first_token(self, request_id: str) -> None:
        """Record when the first token is generated for a request."""
        with self._lock:
            if request_id in self.request_metrics:
                current_time = time.time()
                metrics = self.request_metrics[request_id]
                metrics.first_token_time = current_time
                metrics.time_to_first_token = current_time - metrics.start_time

    def add_generated_tokens(self, request_id: str, num_tokens: int) -> None:
        """Add generated tokens to a request's metrics."""
        with self._lock:
            if request_id in self.request_metrics:
                metrics = self.request_metrics[request_id]
                metrics.generated_tokens += num_tokens
                metrics.total_tokens = metrics.prompt_tokens + metrics.generated_tokens

    def complete_request(
        self,
        request_id: str,
        finish_reason: str | None = None,
        error: str | None = None,
    ) -> None:
        """Complete tracking for a request."""
        with self._lock:
            if request_id not in self.request_metrics:
                return

            metrics = self.request_metrics[request_id]
            metrics.end_time = time.time()
            metrics.total_latency = metrics.end_time - metrics.start_time
            metrics.finish_reason = finish_reason
            metrics.error = error

            # Calculate tokens per second
            if metrics.generated_tokens > 0 and metrics.time_to_first_token is not None:
                generation_time = metrics.total_latency - metrics.time_to_first_token
                if generation_time > 0:
                    metrics.tokens_per_second = metrics.generated_tokens / generation_time

            # Move to completed requests
            self.completed_requests.append(metrics)
            del self.request_metrics[request_id]

            # Update counters
            if error:
                self.counters["total_failed"] += 1
            else:
                self.counters["total_completed"] += 1
                self.counters["total_tokens_generated"] += metrics.generated_tokens

    def record_scheduler_metrics(
        self,
        num_waiting: int,
        num_running: int,
        num_scheduled_tokens: int,
        num_preempted: int = 0,
        batch_size: int = 0,
        schedule_time: float = 0.0,
    ) -> None:
        """Record scheduler performance metrics."""
        with self._lock:
            metrics = SchedulerMetrics(
                timestamp=time.time(),
                num_waiting_requests=num_waiting,
                num_running_requests=num_running,
                num_scheduled_tokens=num_scheduled_tokens,
                num_preempted_requests=num_preempted,
                batch_size=batch_size,
                schedule_time=schedule_time,
            )
            self.scheduler_metrics.append(metrics)

    def record_runner_metrics(
        self,
        execution_time: float,
        batch_size: int,
        num_tokens: int,
        memory_usage: dict[str, Any] | None = None,
    ) -> None:
        """Record model runner performance metrics."""
        with self._lock:
            tokens_per_second = num_tokens / execution_time if execution_time > 0 else 0
            metrics = ModelRunnerMetrics(
                timestamp=time.time(),
                execution_time=execution_time,
                batch_size=batch_size,
                num_tokens=num_tokens,
                tokens_per_second=tokens_per_second,
                memory_usage=memory_usage,
            )
            self.runner_metrics.append(metrics)

    def record_cache_metrics(
        self,
        total_pages: int,
        used_pages: int,
        cache_hit_rate: float = 0.0,
        page_allocation_rate: float = 0.0,
        page_free_rate: float = 0.0,
    ) -> None:
        """Record KV cache metrics."""
        with self._lock:
            metrics = CacheMetrics(
                timestamp=time.time(),
                total_pages=total_pages,
                used_pages=used_pages,
                free_pages=total_pages - used_pages,
                cache_hit_rate=cache_hit_rate,
                page_allocation_rate=page_allocation_rate,
                page_free_rate=page_free_rate,
            )
            self.cache_metrics.append(metrics)

    def get_system_metrics(self, window_seconds: float = 60.0) -> SystemMetrics:
        """Get aggregated system metrics for the specified time window."""
        current_time = time.time()
        cutoff_time = current_time - window_seconds

        with self._lock:
            # Filter recent completed requests
            recent_requests = [req for req in self.completed_requests if req.end_time and req.end_time >= cutoff_time]

            if not recent_requests:
                return SystemMetrics(timestamp=current_time)

            # Calculate aggregates
            total_completed = len(recent_requests)
            total_failed = len([req for req in recent_requests if req.error])
            total_tokens = sum(req.generated_tokens for req in recent_requests)

            latencies = [req.total_latency for req in recent_requests if req.total_latency]
            ttfts = [req.time_to_first_token for req in recent_requests if req.time_to_first_token]
            throughputs = [req.tokens_per_second for req in recent_requests if req.tokens_per_second > 0]

            avg_latency = jnp.mean(jnp.array(latencies)) if latencies else 0.0
            avg_ttft = jnp.mean(jnp.array(ttfts)) if ttfts else 0.0
            avg_throughput = jnp.mean(jnp.array(throughputs)) if throughputs else 0.0
            requests_per_second = total_completed / window_seconds

            return SystemMetrics(
                timestamp=current_time,
                total_requests_completed=total_completed,
                total_requests_failed=total_failed,
                total_tokens_generated=total_tokens,
                average_latency=avg_latency,
                average_ttft=avg_ttft,
                average_throughput=avg_throughput,
                requests_per_second=requests_per_second,
            )

    def log_summary(self, force: bool = False) -> None:
        """Log a summary of current metrics."""
        current_time = time.time()

        if not force and current_time - self.last_log_time < self.log_interval:
            return

        with self._lock:
            system_metrics = self.get_system_metrics()

            # Get latest metrics from each category
            latest_scheduler = self.scheduler_metrics[-1] if self.scheduler_metrics else None
            latest_runner = self.runner_metrics[-1] if self.runner_metrics else None
            latest_cache = self.cache_metrics[-1] if self.cache_metrics else None

            summary = {
                "timestamp": current_time,
                "system": asdict(system_metrics),
                "scheduler": asdict(latest_scheduler) if latest_scheduler else None,
                "runner": asdict(latest_runner) if latest_runner else None,
                "cache": asdict(latest_cache) if latest_cache else None,
                "active_requests": len(self.request_metrics),
            }

            if self.enable_detailed_logging and self.logger:
                self.logger.info(f"METRICS_SUMMARY: {json.dumps(summary)}")

            self.last_log_time = current_time

    def export_metrics(self, file_path: str, format: str = "json") -> None:  # noqa
        """Export all metrics to a file."""
        with self._lock:
            data = {
                "timestamp": time.time(),
                "system_metrics": asdict(self.get_system_metrics()),
                "completed_requests": [asdict(req) for req in self.completed_requests],
                "scheduler_metrics": [asdict(m) for m in self.scheduler_metrics],
                "runner_metrics": [asdict(m) for m in self.runner_metrics],
                "cache_metrics": [asdict(m) for m in self.cache_metrics],
                "counters": dict(self.counters),
            }

        path = Path(file_path)
        if format.lower() == "json":
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def reset_metrics(self) -> None:
        """Reset all metrics and counters."""
        with self._lock:
            self.request_metrics.clear()
            self.completed_requests.clear()
            self.scheduler_metrics.clear()
            self.runner_metrics.clear()
            self.cache_metrics.clear()
            self.counters.clear()
            self.timers.clear()


# Global metrics collector instance
_global_metrics_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector | None:
    """Get the global metrics collector instance."""
    return _global_metrics_collector


def initialize_metrics(
    log_file: str | None = None,
    log_interval: float = 10.0,
    history_size: int = 1000,
    enable_detailed_logging: bool = True,
) -> MetricsCollector:
    """Initialize the global metrics collector."""
    global _global_metrics_collector
    _global_metrics_collector = MetricsCollector(
        log_file=log_file,
        log_interval=log_interval,
        history_size=history_size,
        enable_detailed_logging=enable_detailed_logging,
    )
    return _global_metrics_collector


def log_metrics_summary() -> None:
    """Log a summary of current metrics if collector is initialized."""
    if _global_metrics_collector:
        _global_metrics_collector.log_summary()
