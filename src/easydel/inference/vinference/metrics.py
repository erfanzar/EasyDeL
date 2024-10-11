import gc
import threading
import time
from dataclasses import dataclass

import jax
import psutil
from prometheus_client import Counter, Gauge, Histogram, Info


@dataclass
class ModelMetadata:
	batch_size: int
	sequence_length: int
	dtype: str
	device: str


class vInferenceMetrics:
	def __init__(self, model_name: str):
		self.model_name = model_name

		# Basic request metrics
		self.inference_requests = Counter(
			"model_inference_requests_total",
			"Total number of inference requests",
			["model_name", "status"],
		)

		# Latency metrics
		self.inference_latency = Histogram(
			"model_inference_latency_seconds",
			"Time spent processing inference requests",
			["model_name", "stage"],  # stages: preprocessing, inference, postprocessing
			buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
		)

		# Queue metrics
		self.queue_size = Gauge(
			"model_queue_size", "Current number of requests in queue", ["model_name"]
		)

		# Memory metrics
		self.jax_memory_used = Gauge(
			"jax_memory_used_bytes",
			"Current JAX memory usage",
			["device_id", "memory_type"],  # memory_type: used, peak
		)

		self.host_memory_used = Gauge(
			"host_memory_used_bytes",
			"Host memory usage",
			["type"],  # type: total, available, used
		)

		# Model-specific metrics
		self.token_throughput = Counter(
			"model_token_throughput_total",
			"Total number of tokens processed",
			["model_name", "operation"],  # operation: input, output
		)

		self.generation_length = Histogram(
			"model_generation_length",
			"Distribution of generation lengths",
			["model_name"],
			buckets=(10, 32, 64, 128, 256, 512, 1024),
		)

		# Compilation metrics
		self.compilation_time = Histogram(
			"model_compilation_time_seconds",
			"Time spent on JAX compilation",
			["model_name", "function_name"],
		)

		# Model metadata
		self.model_info = Info("model_metadata", "Model configuration information")

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
				"device": metadata.device,
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

	async def measure_inference(self, func):
		"""Decorator to measure inference metrics"""

		async def wrapper(*args, **kwargs):
			# Track queue size
			self.queue_size.labels(model_name=self.model_name).inc()

			try:
				# Measure preprocessing time
				with self.inference_latency.labels(
					model_name=self.model_name, stage="preprocessing"
				).time():
					# Your preprocessing logic here
					pass

				# Measure inference time
				start_time = time.time()
				result = await func(*args, **kwargs)
				inference_time = time.time() - start_time

				self.inference_latency.labels(
					model_name=self.model_name, stage="inference"
				).observe(inference_time)

				# Record success
				self.inference_requests.labels(
					model_name=self.model_name, status="success"
				).inc()

				return result

			except Exception as e:
				# Record failure
				self.inference_requests.labels(model_name=self.model_name, status="error").inc()
				raise e

			finally:
				self.queue_size.labels(model_name=self.model_name).dec()
				gc.collect()
		return wrapper
