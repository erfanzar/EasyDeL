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

"""JAX device memory monitoring helpers.

This module exposes a few related utilities for inspecting and tracking HBM
usage on the local JAX devices:

* :class:`SMPMemoryMonitor` -- single-process monitor with a background
  polling thread that records per-device memory snapshots into an in-memory
  history buffer.
* :class:`DeviceStats` -- pickleable dataclass payload for transporting
  per-device memory analytics between hosts.
* :class:`MemoryMonitorServer` / :class:`MemoryMonitorClient` -- a tiny
  TCP server/client pair for collecting :class:`DeviceStats` from many
  workers in a SPMD/cluster setup.

Pandas is used opportunistically in :meth:`SMPMemoryMonitor.get_summary`;
when it isn't installed the module gracefully falls back to plain Python
lists.
"""

import logging
import pickle
import queue
import socket
import socketserver
import threading
import time
import typing as tp
from dataclasses import field
from datetime import datetime

import jax
from eformer.pytree import auto_pytree

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("MemoryMonitor")

try:
    import pandas as pd  # pyright: ignore[reportMissingTypeStubs]

    _pandas_available = True
except ImportError:
    pd = None
    _pandas_available = False


class SMPMemoryMonitor:
    """Simple memory monitor for JAX devices.

    Tracks memory usage across all local JAX devices, maintains a history
    of measurements, and provides reporting utilities. Can run in the
    background to log periodic snapshots.

    Attributes:
        check_interval: Seconds between automatic checks.
        quiet: If True, suppresses printed output during monitoring.
        running: Whether background monitoring is active.
        history: List of memory analysis dictionaries.
    """

    def __init__(self, check_interval: int = 60, quiet: bool = False):
        """Configure the monitor without starting any background thread.

        Args:
            check_interval: Number of seconds the background thread sleeps
                between successive sweeps over ``jax.local_devices()``. Has
                no effect until :meth:`start_monitoring` is called.
            quiet: When ``True`` the background loop calls
                :meth:`check_all_devices` (silent), recording history and
                logging only WARNING-level events. When ``False`` (default)
                each cycle additionally prints the formatted status block
                produced by :meth:`print_current_status`.
        """
        self.check_interval = check_interval
        self.quiet = quiet
        self.running = False
        self.history = []
        self._monitor_thread = None

    def analyze_device(self, device_stats: dict, dev) -> dict:
        """Analyze memory stats for a single device.

        Args:
            device_stats: Raw stats dict returned by ``device.memory_stats()``.
                Must contain ``bytes_limit``, ``bytes_in_use``,
                ``peak_bytes_in_use`` and ``num_allocs``.
            dev: The JAX ``Device`` the stats belong to (used only for its
                ``str()`` representation).

        Returns:
            A dict containing rounded GB usage, percent-utilization fields,
            allocation count, and a ``"OK"``/``"WARNING"`` status string.
        """
        bytes_limit = device_stats["bytes_limit"]
        current_usage = device_stats["bytes_in_use"]
        peak_usage = device_stats["peak_bytes_in_use"]

        analysis = {
            "timestamp": datetime.now(),
            "device_id": str(dev),
            "memory_used_gb": round(current_usage / 1e9, 2),
            "memory_limit_gb": round(bytes_limit / 1e9, 2),
            "utilization_pct": round((current_usage / bytes_limit) * 100, 2),
            "peak_usage_gb": round(peak_usage / 1e9, 2),
            "peak_utilization_pct": round((peak_usage / bytes_limit) * 100, 2),
            "num_allocations": device_stats["num_allocs"],
            "status": "OK" if current_usage / bytes_limit < 0.75 else "WARNING",
        }

        return analysis

    def start_monitoring(self):
        """Spawn a daemon thread that records memory snapshots periodically.

        Sets :attr:`running` to ``True`` and starts a daemon thread running
        :meth:`_monitor_loop`. The method returns immediately; call
        :meth:`stop_monitoring` to cleanly halt the loop. Calling this
        method while monitoring is already running starts an additional
        thread (avoid).
        """
        self.running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        print(f"Started monitoring memory every {self.check_interval} seconds")

    def stop_monitoring(self):
        """Signal the monitor thread to exit and wait for it to join.

        Flips :attr:`running` to ``False`` so the next iteration of
        :meth:`_monitor_loop` exits, then blocks on the thread until it
        finishes. Safe to call when monitoring was never started.
        """
        self.running = False
        if self._monitor_thread:
            self._monitor_thread.join()
        print("Stopped memory monitoring")

    def _monitor_loop(self):
        """Background thread body that polls all devices on a fixed cadence.

        Runs until :attr:`running` is set to ``False``. Each iteration calls
        :meth:`check_all_devices` (silent) or :meth:`print_current_status`
        (verbose) depending on :attr:`quiet`, then sleeps for
        :attr:`check_interval` seconds.
        """
        while self.running:
            if self.quiet:
                self.check_all_devices()
            else:
                self.print_current_status()
            time.sleep(self.check_interval)

    def check_all_devices(self) -> list[dict]:
        """Snapshot every local JAX device and append the analyses to ``history``.

        Iterates over ``jax.local_devices()``, calls ``device.memory_stats()``
        on each, runs :meth:`analyze_device` on the result, appends the dict
        to :attr:`history` (truncated to the most recent 1000 entries), and
        prints a one-line WARNING for any device whose status is
        ``"WARNING"``. Devices that raise are logged but skipped so a single
        bad device does not abort the sweep.

        Returns:
            list[dict]: The freshly produced analyses, in the same order as
            ``jax.local_devices()``. Failed devices are omitted.
        """
        results = []
        for device in jax.local_devices():
            try:
                stats = device.memory_stats()
                analysis = self.analyze_device(stats, device)
                self.history.append(analysis)
                results.append(analysis)

                if len(self.history) > 1000:
                    self.history = self.history[-1000:]

                if analysis["status"] == "WARNING":
                    print(f"WARNING: High memory usage on {analysis['device_id']}: {analysis['utilization_pct']}%")

            except Exception as e:
                print(f"Error checking device {device}: {e}")

        return results

    def get_summary(self, format: str = "auto") -> list[dict] | tp.Any:  # noqa
        """Return the recorded history sorted newest-first.

        Args:
            format: Output container selector. ``"pandas"`` forces a pandas
                ``DataFrame`` (raises :class:`ImportError` when pandas is
                not installed); ``"list"`` forces a plain list of dicts;
                ``"auto"`` (default) returns a ``DataFrame`` when pandas is
                available and falls back to the list form otherwise.

        Returns:
            list[dict] | pandas.DataFrame: Either a list of analysis dicts
            (matching :meth:`analyze_device` output) or a DataFrame of the
            same data, ordered by ``timestamp`` descending. An empty list
            (or empty DataFrame) is returned when no entries have been
            recorded yet.

        Raises:
            ImportError: If ``format == "pandas"`` and pandas is not
                installed in the environment.
        """
        if not self.history:
            return [] if format == "list" else pd.DataFrame() if _pandas_available else []

        if format == "pandas" and not _pandas_available:
            raise ImportError("Pandas is not available. Install pandas or use format='list'")

        if format == "pandas" or (format == "auto" and _pandas_available):
            return pd.DataFrame(self.history).sort_values("timestamp", ascending=False)

        return sorted(self.history, key=lambda x: x["timestamp"], reverse=True)

    def print_current_status(self):
        """Run a fresh sweep and pretty-print one block per device to stdout.

        Calls :meth:`check_all_devices` and then prints a multi-line block
        per result containing device id, status, used/limit GB, utilization
        percent, peak usage, and active allocation count. Used by the
        non-quiet variant of :meth:`_monitor_loop` and intended for ad-hoc
        debugging — it is not parsing-stable.
        """
        results = self.check_all_devices()
        print("\nCurrent Memory Status:")
        print("-" * 50)

        for r in results:
            print(f"\nDevice: {r['device_id']}")
            print(f"Status: {r['status']}")
            print(f"Memory Used: {r['memory_used_gb']} GB / {r['memory_limit_gb']} GB")
            print(f"Utilization: {r['utilization_pct']}%")
            print(f"Peak Usage: {r['peak_usage_gb']} GB ({r['peak_utilization_pct']}%)")
            print(f"Active Allocations: {r['num_allocations']}")

    def get_device_history(self, device_id: str | None = None) -> list[dict]:
        """Return the recorded history, optionally filtered to one device.

        Args:
            device_id: When supplied, only entries whose ``device_id`` field
                equals this string (set by :meth:`analyze_device` to
                ``str(device)``) are returned. ``None`` (default) returns
                the full history.

        Returns:
            list[dict]: A list of analysis dicts in insertion order
            (oldest first); the list is the underlying buffer when no
            filter is given, so callers should not mutate it.
        """
        if device_id:
            return [entry for entry in self.history if entry["device_id"] == device_id]
        return self.history

    def print_history_summary(self, n_entries: int = 5):
        """Print the ``n_entries`` most recent history entries newest-first.

        Args:
            n_entries: Maximum number of recent entries to print. ``5`` by
                default; pass a larger value to inspect a longer window.
                When ``self.history`` is shorter than ``n_entries`` only the
                available entries are printed.
        """
        if not self.history:
            print("No history available")
            return

        print("\nRecent Memory Usage History:")
        print("-" * 50)

        sorted_history = sorted(self.history, key=lambda x: x["timestamp"], reverse=True)
        for entry in sorted_history[:n_entries]:
            print(f"\nTimestamp: {entry['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Device: {entry['device_id']}")
            print(f"Memory Used: {entry['memory_used_gb']} GB ({entry['utilization_pct']}%)")
            print(f"Status: {entry['status']}")


@auto_pytree
class DeviceStats:
    """Wire-format snapshot of one device's memory state for cluster monitoring.

    Produced by :meth:`MemoryMonitorClient.analyze_memory` once per polling
    cycle and pickled to a :class:`MemoryMonitorServer`. Each instance
    represents a single ``(host, device, timestamp)`` measurement: clients
    emit a stream of these so the server can build a multi-host time series
    of HBM utilisation and react to pressure events.

    Attributes:
        device_id (str): String form of the JAX device the snapshot describes
            (typically ``"TpuDevice(...)"`` or ``"GpuDevice(...)"``); used as
            the primary grouping key on the server side.
        hostname (str): ``socket.gethostname()`` of the reporting client; lets
            the server distinguish two devices that happen to share a string
            id across different physical hosts.
        timestamp (datetime): Wall-clock time the snapshot was taken on the
            client. The server orders trends by this field rather than by
            arrival time.
        utilization_percent (float): ``bytes_in_use / bytes_limit * 100``
            rounded to 2dp. Above 90 the server logs a critical warning.
        peak_utilization_percent (float): Same ratio computed against
            ``peak_bytes_in_use`` — useful for spotting transient spikes
            that a sampling-based monitor would miss.
        fragmentation_ratio (float): ``largest_free_block_bytes / total_free``
            (rounded to 4dp). Values near ``1`` mean the free pool is one
            contiguous block; values near ``0`` indicate severe fragmentation
            that may starve large allocations.
        allocation_efficiency (float): ``avg_alloc_size / largest_alloc_size``
            (rounded to 4dp). Diagnoses whether memory is held by many small
            objects (low ratio) or by a few large tensors (ratio close to 1).
        memory_pressure (str): Coarse label derived from ``utilization_percent``:
            one of ``"Low"`` (≤50), ``"Moderate"`` (≤75), ``"High"`` (≤90)
            or ``"Critical"`` (>90).
        raw_stats (dict[str, Any]): Verbatim copy of the JAX
            ``device.memory_stats()`` dict the snapshot was derived from,
            preserved so the server can compute additional metrics without
            another round-trip to the client.
    """

    device_id: str = field(
        metadata={"help": "The ID of the device."},
    )
    hostname: str = field(
        metadata={"help": "The hostname of the machine."},
    )
    timestamp: datetime = field(
        metadata={"help": "The timestamp of the statistics."},
    )
    utilization_percent: float = field(
        metadata={"help": "The utilization percentage of the device."},
    )
    peak_utilization_percent: float = field(
        metadata={"help": "The peak utilization percentage of the device."},
    )
    fragmentation_ratio: float = field(
        metadata={"help": "The fragmentation ratio of the device memory."},
    )
    allocation_efficiency: float = field(
        metadata={"help": "The allocation efficiency of the device memory."},
    )
    memory_pressure: str = field(
        metadata={"help": "The memory pressure status (e.g., 'low', 'medium', 'high')."},
    )
    raw_stats: dict[str, tp.Any] = field(
        default_factory=dict,
        metadata={"help": "A dictionary containing the raw statistics from the device."},
    )


class MemoryMonitorServer:
    """TCP server that collects device memory statistics from remote clients.

    Receives serialized ``DeviceStats`` objects over TCP, stores them, and
    analyzes trends for high utilization or fragmentation warnings.

    Attributes:
        host: Address to bind the server socket to.
        port: Port to listen on.
        stats_queue: Thread-safe queue for incoming stats.
        running: Whether the server is active.
        data_store: In-memory list of received ``DeviceStats``.
        lock: Threading lock for ``data_store`` access.
    """

    def __init__(self, host="0.0.0.0", port=5000):
        """Initialize the server but do not bind to the socket yet.

        Args:
            host: Address to listen on. Defaults to all interfaces.
            port: TCP port to listen on. Defaults to 5000.
        """
        self.host = host
        self.port = port
        self.stats_queue = queue.Queue()
        self.running = False
        self.data_store = []
        self.lock = threading.Lock()

    def start(self):
        """Bind the server socket and spawn the receive/process daemon threads.

        Two daemon threads are started: ``_run_server`` (accepts incoming
        TCP connections and pushes pickled :class:`DeviceStats` payloads
        onto :attr:`stats_queue`) and ``_process_data`` (drains the queue
        into :attr:`data_store` under :attr:`lock` and runs trend analysis
        for warnings). The method returns immediately after both threads
        are running.
        """
        self.running = True

        server_thread = threading.Thread(target=self._run_server)
        server_thread.daemon = True
        server_thread.start()

        process_thread = threading.Thread(target=self._process_data)
        process_thread.daemon = True
        process_thread.start()

        logger.info(f"Memory monitor server started on {self.host}:{self.port}")

    def _run_server(self):
        """Run the TCP server loop on the configured host/port.

        Blocks the calling thread; intended to be run from a daemon thread
        spawned by :meth:`start`.
        """

        class RequestHandler(socketserver.BaseRequestHandler):
            """Per-connection handler that decodes one pickled payload."""

            def handle(self):
                """Read a pickled :class:`DeviceStats` and enqueue it.

                Errors are logged and swallowed so that one bad client does
                not bring the server down.
                """
                try:
                    data = pickle.loads(self.request.recv(4096))
                    self.server.stats_queue.put(data)
                except Exception as e:
                    logger.error(f"Error handling request: {e}")

        class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
            """Threaded TCP server that exposes the outer queue to handlers."""

            stats_queue = self.stats_queue
            allow_reuse_address = True

        with ThreadedTCPServer((self.host, self.port), RequestHandler) as server:
            server.serve_forever()

    def _process_data(self):
        """Drain the stats queue, persist entries and emit warnings.

        Loops until ``self.running`` becomes ``False`` and per-iteration
        errors are logged but do not stop the loop.
        """
        while self.running:
            try:
                stats = self.stats_queue.get(timeout=1)
                with self.lock:
                    self.data_store.append(stats)

                    self._cleanup_old_data()
                self._analyze_trends(stats)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing data: {e}")

    def _cleanup_old_data(self):
        """Cap :attr:`data_store` at 1000 entries to bound server memory.

        Called from :meth:`_process_data` after each enqueue so the in-memory
        buffer stays bounded regardless of cluster size. Older entries are
        discarded silently — durable archival is the consumer's job.
        """
        if len(self.data_store) > 1000:
            self.data_store = self.data_store[-1000:]

    def _analyze_trends(self, stats: DeviceStats):
        """Analyze memory usage trends and log warnings.

        Emits warnings for high utilization (>90%) and high fragmentation
        (largest-free-block ratio < 0.5).

        Args:
            stats: A single :class:`DeviceStats` snapshot to inspect.
        """
        if stats.utilization_percent > 90:
            logger.warning(
                f"Critical memory usage on {stats.hostname} ({stats.device_id}): {stats.utilization_percent}%"
            )
        elif stats.fragmentation_ratio < 0.5:
            logger.warning(
                f"High memory fragmentation on {stats.hostname} ({stats.device_id}): {stats.fragmentation_ratio}"
            )

    def get_device_stats(self, device_id=None):
        """Get statistics for all devices or a specific device.

        Args:
            device_id: When provided, filters the data store to entries with
                a matching ``device_id``. When ``None`` returns everything.

        Returns:
            A list of :class:`DeviceStats` objects (a snapshot of the
            internal store under the lock).
        """
        with self.lock:
            if device_id:
                return [s for s in self.data_store if s.device_id == device_id]
            return self.data_store


class MemoryMonitorClient:
    """Client that monitors local JAX device memory and reports to a server.

    Periodically collects memory statistics from all local JAX devices,
    packages them as ``DeviceStats``, and sends them to a
    ``MemoryMonitorServer`` over TCP.

    Attributes:
        server_host: Hostname or IP of the monitoring server.
        server_port: Port of the monitoring server.
        interval: Seconds between monitoring cycles.
        running: Whether monitoring is active.
        hostname: Local hostname for identification.
    """

    def __init__(self, server_host, server_port=5000, interval=60):
        """Configure the client without starting any background thread.

        Args:
            server_host: Hostname or IP address of the monitoring server.
            server_port: TCP port of the monitoring server. Defaults to 5000.
            interval: Polling interval in seconds. Defaults to 60.
        """
        self.server_host = server_host
        self.server_port = server_port
        self.interval = interval
        self.running = False
        self.hostname = socket.gethostname()

    def analyze_memory(self, memory_stats: dict[str, tp.Any]) -> DeviceStats:
        """Analyze memory statistics for a single device.

        Args:
            memory_stats: Raw stats dict from ``device.memory_stats()``;
                expected keys include ``bytes_limit``, ``bytes_in_use``,
                ``peak_bytes_in_use``, ``largest_free_block_bytes``,
                ``num_allocs``, and ``largest_alloc_size``.

        Returns:
            A :class:`DeviceStats` summarizing utilization, peak utilization,
            fragmentation, allocation efficiency, and memory pressure.
        """
        bytes_limit = memory_stats["bytes_limit"]
        current_usage = memory_stats["bytes_in_use"]
        peak_usage = memory_stats["peak_bytes_in_use"]

        utilization = (current_usage / bytes_limit) * 100
        peak_utilization = (peak_usage / bytes_limit) * 100

        largest_free = memory_stats["largest_free_block_bytes"]
        total_free = bytes_limit - current_usage
        fragmentation_ratio = largest_free / total_free if total_free > 0 else 1.0

        num_allocs = memory_stats["num_allocs"]
        avg_alloc_size = current_usage / num_allocs if num_allocs > 0 else 0
        largest_alloc = memory_stats["largest_alloc_size"]
        allocation_efficiency = avg_alloc_size / largest_alloc if largest_alloc > 0 else 0

        memory_pressure = (
            "Critical" if utilization > 90 else "High" if utilization > 75 else "Moderate" if utilization > 50 else "Low"
        )

        return DeviceStats(
            device_id=str(jax.local_devices()[0]),
            hostname=self.hostname,
            timestamp=datetime.now(),
            utilization_percent=round(utilization, 2),
            peak_utilization_percent=round(peak_utilization, 2),
            fragmentation_ratio=round(fragmentation_ratio, 4),
            allocation_efficiency=round(allocation_efficiency, 4),
            memory_pressure=memory_pressure,
            raw_stats=memory_stats,
        )

    def start_monitoring(self):
        """Start monitoring memory usage.

        Spawns a daemon thread running :meth:`_monitor_loop` and returns
        immediately.
        """
        self.running = True
        monitor_thread = threading.Thread(target=self._monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        logger.info(f"Started monitoring on {self.hostname}")

    def stop_monitoring(self):
        """Stop monitoring memory usage by signalling the monitor loop to exit."""
        self.running = False

    def _monitor_loop(self):
        """Main monitoring loop.

        Polls every local JAX device, sends snapshots to the server, and
        sleeps ``self.interval`` seconds between iterations until
        ``self.running`` becomes ``False``.
        """
        while self.running:
            try:
                for device in jax.local_devices():
                    memory_stats = device.memory_stats()
                    stats = self.analyze_memory(memory_stats)
                    self._send_stats(stats)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            time.sleep(self.interval)

    def _send_stats(self, stats: DeviceStats):
        """Send statistics to the server.

        Args:
            stats: A :class:`DeviceStats` payload pickled and sent over a new
                short-lived TCP connection. Errors are logged and swallowed.
        """
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((self.server_host, self.server_port))
                sock.send(pickle.dumps(stats))
        except Exception as e:
            logger.error(f"Error sending stats to server: {e}")




if __name__ == "__main__":
    print(SMPMemoryMonitor().check_all_devices())
