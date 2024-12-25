# Copyright 2023 The EASYDEL Author @erfanzar (Erfan Zare Chavoshi).
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

import logging
import pickle
import queue
import socket
import socketserver
import threading
import time
import typing as tp
from dataclasses import dataclass
from datetime import datetime

import jax

logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
	handlers=[logging.FileHandler("memory_monitor.log"), logging.StreamHandler()],
)
logger = logging.getLogger("MemoryMonitor")

try:
	import pandas as pd

	PANDAS_AVAILABLE = True
except ImportError:
	PANDAS_AVAILABLE = False


class SMPMemoryMonitor:
	def __init__(self, check_interval: int = 60):
		"""
		Initialize the memory monitor.

		Args:
		    check_interval: How often to check memory in seconds (default: 60)
		"""
		self.check_interval = check_interval
		self.running = False
		self.history = []
		self._monitor_thread = None

	def analyze_device(self, device_stats: tp.Dict, dev) -> tp.Dict:
		"""
		Analyze memory stats for a single device.
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
		"""Start automatic memory monitoring."""
		self.running = True
		self._monitor_thread = threading.Thread(target=self._monitor_loop)
		self._monitor_thread.daemon = True
		self._monitor_thread.start()
		print(f"Started monitoring memory every {self.check_interval} seconds")

	def stop_monitoring(self):
		"""Stop automatic memory monitoring."""
		self.running = False
		if self._monitor_thread:
			self._monitor_thread.join()
		print("Stopped memory monitoring")

	def _monitor_loop(self):
		"""Internal monitoring loop."""
		while self.running:
			self.check_all_devices()
			time.sleep(self.check_interval)

	def check_all_devices(self) -> tp.List[tp.Dict]:
		"""
		Check memory usage on all available devices.
		Returns list of analysis results.
		"""
		results = []
		for device in jax.devices():
			try:
				stats = device.memory_stats()
				analysis = self.analyze_device(stats, device)
				self.history.append(analysis)
				results.append(analysis)

				if len(self.history) > 1000:
					self.history = self.history[-1000:]

				if analysis["status"] == "WARNING":
					print(
						f"WARNING: High memory usage on {analysis['device_id']}: "
						f"{analysis['utilization_pct']}%"
					)

			except Exception as e:
				print(f"Error checking device {device}: {e}")

		return results

	def get_summary(
		self, format: str = "auto"
	) -> tp.Union[tp.List[tp.Dict], "pd.DataFrame"]:
		"""
		Get a summary of memory usage history.

		Args:
		    format: Output format - 'pandas' (force pandas DataFrame),
		           'list' (force list), or 'auto' (use pandas if available)

		Returns:
		    Either pandas DataFrame or list of dictionaries depending on format
		    and pandas availability
		"""
		if not self.history:
			return [] if format == "list" else pd.DataFrame() if PANDAS_AVAILABLE else []

		if format == "pandas" and not PANDAS_AVAILABLE:
			raise ImportError("Pandas is not available. Install pandas or use format='list'")

		if format == "pandas" or (format == "auto" and PANDAS_AVAILABLE):
			return pd.DataFrame(self.history).sort_values("timestamp", ascending=False)

		return sorted(self.history, key=lambda x: x["timestamp"], reverse=True)

	def print_current_status(self):
		"""
		Print current memory status for all devices.
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

	def get_device_history(self, device_id: tp.Optional[str] = None) -> tp.List[tp.Dict]:
		"""
		Get memory history for a specific device or all devices.

		Args:
		    device_id: tp.Optional device ID to filter by

		Returns:
		    tp.List of history entries for the specified device(s)
		"""
		if device_id:
			return [entry for entry in self.history if entry["device_id"] == device_id]
		return self.history

	def print_history_summary(self, n_entries: int = 5):
		"""
		Print a summary of recent memory usage without using pandas.

		Args:
		    n_entries: Number of most recent entries to show
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


@dataclass
class DeviceStats:
	device_id: str
	hostname: str
	timestamp: datetime
	utilization_percent: float
	peak_utilization_percent: float
	fragmentation_ratio: float
	allocation_efficiency: float
	memory_pressure: str
	raw_stats: tp.Dict[str, tp.Any]


class MemoryMonitorServer:
	def __init__(self, host="0.0.0.0", port=5000):
		self.host = host
		self.port = port
		self.stats_queue = queue.Queue()
		self.running = False
		self.data_store = []
		self.lock = threading.Lock()

	def start(self):
		"""Start the monitoring server"""
		self.running = True

		server_thread = threading.Thread(target=self._run_server)
		server_thread.daemon = True
		server_thread.start()

		process_thread = threading.Thread(target=self._process_data)
		process_thread.daemon = True
		process_thread.start()

		logger.info(f"Memory monitor server started on {self.host}:{self.port}")

	def _run_server(self):
		class RequestHandler(socketserver.BaseRequestHandler):
			def handle(self):
				try:
					data = pickle.loads(self.request.recv(4096))
					self.server.stats_queue.put(data)
				except Exception as e:
					logger.error(f"Error handling request: {e}")

		class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
			stats_queue = self.stats_queue
			allow_reuse_address = True

		with ThreadedTCPServer((self.host, self.port), RequestHandler) as server:
			server.serve_forever()

	def _process_data(self):
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
		"""Keep only recent data to prevent memory issues"""
		if len(self.data_store) > 1000:
			self.data_store = self.data_store[-1000:]

	def _analyze_trends(self, stats: DeviceStats):
		"""Analyze memory usage trends and log warnings"""
		if stats.utilization_percent > 90:
			logger.warning(
				f"Critical memory usage on {stats.hostname} ({stats.device_id}): {stats.utilization_percent}%"
			)
		elif stats.fragmentation_ratio < 0.5:
			logger.warning(
				f"High memory fragmentation on {stats.hostname} ({stats.device_id}): {stats.fragmentation_ratio}"
			)

	def get_device_stats(self, device_id=None):
		"""Get statistics for all devices or a specific device"""
		with self.lock:
			if device_id:
				return [s for s in self.data_store if s.device_id == device_id]
			return self.data_store


class MemoryMonitorClient:
	def __init__(self, server_host, server_port=5000, interval=60):
		self.server_host = server_host
		self.server_port = server_port
		self.interval = interval
		self.running = False
		self.hostname = socket.gethostname()

	def analyze_memory(self, memory_stats: tp.Dict[str, tp.Any]) -> DeviceStats:
		"""Analyze memory statistics for a single device"""
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
			"Critical"
			if utilization > 90
			else "High"
			if utilization > 75
			else "Moderate"
			if utilization > 50
			else "Low"
		)

		return DeviceStats(
			device_id=str(jax.devices()[0]),
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
		"""Start monitoring memory usage"""
		self.running = True
		monitor_thread = threading.Thread(target=self._monitor_loop)
		monitor_thread.daemon = True
		monitor_thread.start()
		logger.info(f"Started monitoring on {self.hostname}")

	def stop_monitoring(self):
		"""Stop monitoring memory usage"""
		self.running = False

	def _monitor_loop(self):
		"""Main monitoring loop"""
		while self.running:
			try:
				for device in jax.devices():
					memory_stats = device.memory_stats()
					stats = self.analyze_memory(memory_stats)
					self._send_stats(stats)
			except Exception as e:
				logger.error(f"Error in monitoring loop: {e}")
			time.sleep(self.interval)

	def _send_stats(self, stats: DeviceStats):
		"""Send statistics to the server"""
		try:
			with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
				sock.connect((self.server_host, self.server_port))
				sock.send(pickle.dumps(stats))
		except Exception as e:
			logger.error(f"Error sending stats to server: {e}")


def start_server():
	server = MemoryMonitorServer()
	server.start()
	return server


def start_client(server_host):
	client = MemoryMonitorClient(server_host)
	client.start_monitoring()
	return client


if __name__ == "__main__":
	print(SMPMemoryMonitor().check_all_devices())
