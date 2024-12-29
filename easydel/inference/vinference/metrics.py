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


import gc
import threading
import time
from dataclasses import dataclass

import jax
import psutil

try:
	from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server
except ModuleNotFoundError:
	Counter, Gauge, Histogram, Info, start_http_server = [None] * 5


@dataclass
class ModelMetadata:
	batch_size: int
	sequence_length: int
	dtype: str
	platfrom: str


class vInferenceMetrics:
	def __init__(self, model_name: str):
		model_name = model_name.replace("-", "_").replace(".", "_")
		self.model_name = model_name

		# Basic request metrics
		self.inference_requests = Counter(
			f"{model_name}_model_inference_requests_total",
			"Total number of inference requests",
			["model_name", "status"],
		)

		self.inference_latency = Histogram(
			f"{model_name}_model_inference_latency",
			"Time spent processing inference request",
			["model_name", "stage"],  # stages: preprocessing, inference, postprocessing
			buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
		)

		# Queue metrics
		self.queue_size = Gauge(
			f"{model_name}_model_queue_size",
			"Current number of requests in queue",
			["model_name"],
		)

		# Memory metrics
		self.jax_memory_used = Gauge(
			f"{model_name}_jax_memory_used_bytes",
			"Current JAX memory usage",
			["device_id", "memory_type"],  # memory_type: used, peak
		)

		self.host_memory_used = Gauge(
			f"{model_name}_host_memory_used_bytes",
			"Host memory usage",
			["type"],  # type: total, available, used
		)

		# Model-specific metrics
		self.token_throughput = Counter(
			f"{model_name}_model_token_throughput_total",
			"Total number of tokens processed",
			["model_name", "operation"],
		)

		self.generation_length = Histogram(
			f"{model_name}_model_generation_length",
			"Distribution of generation lengths",
			["model_name"],
			buckets=(
				16,
				32,
				64,
				128,
				256,
				512,
				1024,
				2048,
				4096,
				8192,
				16384,
				32768,
			),
		)

		# Compilation metrics
		self.compilation_time = Histogram(
			f"{model_name}_model_compilation_time_seconds",
			"Time spent on JAX compilation",
			["model_name", "function_name"],
		)

		# Model metadata
		self.model_info = Info(
			f"{model_name}_model_metadata",
			"Model configuration information",
		)

		# Start monitoring threads
		self._start_memory_monitoring()

	def _start_memory_monitoring(self):
		def monitor_memory():
			while True:
				# JAX memory monitoring
				for device in jax.devices():
					memory_stats = device.memory_stats()
					if memory_stats:  # Some devices might not support memory stats
						self.jax_memory_used.labels(
							device_id=str(device.id), memory_type="used"
						).set(memory_stats["bytes_in_use"])
						self.jax_memory_used.labels(
							device_id=str(device.id), memory_type="peak"
						).set(memory_stats.get("peak_bytes_in_use", 0))

				# Host memory monitoring
				memory = psutil.virtual_memory()
				self.host_memory_used.labels(type="total").set(memory.total)
				self.host_memory_used.labels(type="available").set(memory.available)
				self.host_memory_used.labels(type="used").set(memory.used)

				time.sleep(1)

		threading.Thread(target=monitor_memory, daemon=True).start()

	def record_model_metadata(self, metadata: ModelMetadata):
		"""Record static model information"""
		self.model_info.info(
			{
				"model_name": self.model_name,
				"batch_size": str(metadata.batch_size),
				"sequence_length": str(metadata.sequence_length),
				"dtype": metadata.dtype,
				"platfrom": metadata.platfrom,
			}
		)

	def track_compilation(self, function_name: str):
		"""Decorator to track compilation time of JAX functions"""

		def decorator(func):
			def wrapper(*args, **kwargs):
				start_time = time.time()
				result = func(*args, **kwargs)
				compilation_time = time.time() - start_time
				self.compilation_time.labels(
					model_name=self.model_name, function_name=function_name
				).observe(compilation_time)
				return result

			return wrapper

		return decorator

	def measure_inference_first_step(self, func):
		"""Decorator to measure inference metrics"""

		def wrapper(*args, **kwargs):
			# Track queue size
			self.queue_size.labels(model_name=self.model_name).inc()

			try:
				with self.first_step_inference_latency.labels(
					model_name=self.model_name, stage="preprocessing"
				).time():
					pass
				start_time = time.time()
				result = func(*args, **kwargs)
				inference_time = time.time() - start_time

				self.first_step_inference_latency.labels(
					model_name=self.model_name, stage="inference"
				).observe(inference_time)
				self.inference_requests.labels(
					model_name=self.model_name, status="success"
				).inc()

				return result

			except Exception as e:
				self.inference_requests.labels(model_name=self.model_name, status="error").inc()
				raise e

			finally:
				self.queue_size.labels(model_name=self.model_name).dec()
				gc.collect()

		return wrapper

	def measure_inference_afterward(self, func):
		"""Decorator to measure inference metrics"""

		def wrapper(*args, **kwargs):
			self.queue_size.labels(model_name=self.model_name).inc()
			try:
				with self.afterward_inference_latency.labels(
					model_name=self.model_name, stage="preprocessing"
				).time():
					pass
				start_time = time.time()
				result = func(*args, **kwargs)
				inference_time = time.time() - start_time
				self.afterward_inference_latency.labels(
					model_name=self.model_name, stage="inference"
				).observe(inference_time)
				return result
			except Exception as e:
				self.inference_requests.labels(model_name=self.model_name, status="error").inc()
				raise e

			finally:
				self.queue_size.labels(model_name=self.model_name).dec()
				gc.collect()

		return wrapper


if __name__ == "__main__":
	metrics = vInferenceMetrics("test")
	start_http_server(7860)
	while True:
		# Continuously update or collect metrics
		pass
