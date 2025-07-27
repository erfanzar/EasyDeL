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


import gc
import threading
import time
from dataclasses import fields

import jax
from eformer.pytree import auto_pytree

try:
    import psutil  # type:ignore
    from prometheus_client import Counter, Gauge, Histogram, Info, start_http_server  # type:ignore
except ModuleNotFoundError:
    Counter, Gauge, Histogram, Info, start_http_server, psutil = [None] * 6


@auto_pytree
class ModelMetadata:
    """
    A dataclass to hold basic metadata about the loaded model and runtime environment.

    This information can be useful for logging, monitoring, or debugging purposes.

    Attributes:
        batch_size: The batch size used for inference.
        sequence_length: The maximum sequence length the model is configured for.
        dtype: The data type (e.g., 'float16', 'bfloat16') used for model parameters/computation.
        platform: The JAX platform being used (e.g., 'cpu', 'gpu', 'tpu').
    """

    batch_size: int
    sequence_length: int
    dtype: str
    platfrom: str

    def __repr__(self):
        cls_name = self.__class__.__name__
        field_lines = [f"    {f.name}: {getattr(self, f.name)!r}".replace("\n", "\n    ") for f in fields(self)]
        return f"{cls_name}(\n" + "\n".join(field_lines) + "\n)"

    __str__ = __repr__


class vInferenceMetrics:
    """
    Manages and exposes Prometheus metrics for monitoring the vInference engine.

    This class initializes various Prometheus metric objects (Counters, Gauges, Histograms,
    Info) to track key performance indicators and resource usage of a specific model
    during inference.

    It provides decorators (`track_compilation`, `measure_inference_first_step`, `measure_inference_afterward`)
    to easily instrument functions and record relevant metrics.

    It also starts background threads for monitoring system resources like JAX device
    memory and host memory if running in a local environment.

    Attributes:
        model_name (str): The sanitized name of the model used in metric labels.
        inference_requests (Counter): Tracks the total number of inference requests,
            labeled by status (success/error).
        inference_latency (Histogram): Measures the latency distribution of different
            inference stages (preprocessing, inference, postprocessing).
        queue_size (Gauge): Tracks the current number of requests waiting in the queue.
        jax_memory_used (Gauge): Tracks the current and peak JAX memory usage per device.
        host_memory_used (Gauge): Tracks the host system's memory usage (total, available, used).
        token_throughput (Counter): Counts the total number of tokens processed, labeled by
            the operation (e.g., 'prefill', 'decode').
        generation_length (Histogram): Measures the distribution of generated sequence lengths.
        compilation_time (Histogram): Measures the time spent on JAX function compilation,
            labeled by the function name.
        model_info (Info): Stores static model configuration metadata.
    """

    def __init__(self, model_name: str):
        """
        Initializes the vInferenceMetrics instance.

        Args:
            model_name: The name of the model being monitored. This name will be
                sanitized (replacing '-' and '.' with '_') and used in metric names
                and labels.
        """
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
            buckets=(16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768),
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
        if jax.device_count() == jax.local_device_count():
            self._start_memory_monitoring()  # Fixes 181 (currently)

    def _start_memory_monitoring(self):
        """
        Starts a background thread to periodically monitor JAX device memory and host memory.

        This method is typically called during initialization if the process is running
        with local JAX devices.

        The monitoring thread runs indefinitely, querying `jax.local_devices()` for
        memory statistics and `psutil.virtual_memory()` for host statistics, updating
        the corresponding Prometheus Gauges (`jax_memory_used`, `host_memory_used`)
        every second.
        """

        def monitor_memory():
            while True:
                # JAX memory monitoring
                for device in jax.local_devices():
                    memory_stats = device.memory_stats()
                    if memory_stats:  # Some devices might not support memory stats
                        self.jax_memory_used.labels(device_id=str(device.id), memory_type="used").set(
                            memory_stats["bytes_in_use"]
                        )
                        self.jax_memory_used.labels(device_id=str(device.id), memory_type="peak").set(
                            memory_stats.get("peak_bytes_in_use", 0)
                        )

                # Host memory monitoring
                memory = psutil.virtual_memory()
                self.host_memory_used.labels(type="total").set(memory.total)
                self.host_memory_used.labels(type="available").set(memory.available)
                self.host_memory_used.labels(type="used").set(memory.used)

                time.sleep(1)

        threading.Thread(target=monitor_memory, daemon=True).start()

    def record_model_metadata(self, metadata: ModelMetadata):
        """
        Records static model metadata using the Prometheus Info metric.

        Args:
            metadata: A `ModelMetadata` object containing information like batch size,
                sequence length, dtype, and platform.
        """
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
        """
        Returns a decorator to measure and record the compilation time of a JAX function.

        Usage:
        ```python
        metrics = vInferenceMetrics("my_model")


        @metrics.track_compilation("my_compiled_function")
        @ed.jit
        def my_compiled_function(x):
          # ... JAX computation ...
          return x * 2
        ```

        Args:
            function_name: A descriptive name for the function being compiled,
                used as a label in the `compilation_time` metric.

        Returns:
            A decorator function.
        """

        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                compilation_time = time.time() - start_time
                self.compilation_time.labels(
                    model_name=self.model_name,
                    function_name=function_name,
                ).observe(compilation_time)
                return result

            return wrapper

        return decorator

    def measure_inference_first_step(self, func):
        """
        Returns a decorator to measure metrics specifically for the *first* step
        of an inference process (often corresponds to prefill).

        This decorator:
        - Increments/decrements the `queue_size` gauge upon entry/exit.
        - Measures the latency of the decorated function using the
          `first_step_inference_latency` histogram (labeled with stage='inference').
          (Note: Assumes a `self.first_step_inference_latency` histogram exists, which
           might need to be added if different latency tracking for the first step is desired).
        - Increments the `inference_requests` counter (status='success' or 'error').
        - Performs garbage collection upon exit.

        Args:
            func: The function representing the first inference step to be measured.

        Returns:
            A decorator function.
        """

        def wrapper(*args, **kwargs):
            self.queue_size.labels(model_name=self.model_name).inc()

            try:
                with self.first_step_inference_latency.labels(model_name=self.model_name, stage="preprocessing").time():
                    pass
                start_time = time.time()
                result = func(*args, **kwargs)
                inference_time = time.time() - start_time

                self.first_step_inference_latency.labels(model_name=self.model_name, stage="inference").observe(
                    inference_time
                )
                self.inference_requests.labels(model_name=self.model_name, status="success").inc()

                return result

            except Exception as e:
                self.inference_requests.labels(model_name=self.model_name, status="error").inc()
                raise e

            finally:
                self.queue_size.labels(model_name=self.model_name).dec()
                gc.collect()

        return wrapper

    def measure_inference_afterward(self, func):
        """
        Returns a decorator to measure metrics for subsequent inference steps
        (after the first step, often corresponding to decode steps).

        This decorator behaves similarly to `measure_inference_first_step` but uses
        a separate latency histogram (`afterward_inference_latency`).

        It:
        - Increments/decrements the `queue_size` gauge upon entry/exit.
        - Measures the latency of the decorated function using the
          `afterward_inference_latency` histogram (labeled with stage='inference').
          (Note: Assumes a `self.afterward_inference_latency` histogram exists, which
           might need to be added for separate afterward step latency tracking).
        - Increments the `inference_requests` counter (only on errors, success is likely
          tracked elsewhere or per-token).
        - Performs garbage collection upon exit.

        Args:
            func: The function representing a subsequent inference step to be measured.

        Returns:
            A decorator function.
        """

        def wrapper(*args, **kwargs):
            self.queue_size.labels(model_name=self.model_name).inc()
            try:
                with self.afterward_inference_latency.labels(model_name=self.model_name, stage="preprocessing").time():
                    pass
                start_time = time.time()
                result = func(*args, **kwargs)
                inference_time = time.time() - start_time
                self.afterward_inference_latency.labels(model_name=self.model_name, stage="inference").observe(
                    inference_time
                )
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
