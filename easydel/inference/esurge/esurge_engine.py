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

"""eSurge Engine - High-level interface for efficient text generation."""

from __future__ import annotations

import threading
import time
import typing
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import jax
from jax import numpy as jnp
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from easydel.inference.sampling_params import SamplingParams
from easydel.utils.helpers import get_logger

from .engine_types import EngineCoreOutputs
from .metrics import get_metrics_collector, initialize_metrics
from .request import EngineRequest, EngineRequestStatus
from .runners import eSurgeRunner
from .scheduler import Scheduler

if typing.TYPE_CHECKING:
    from easydel import AutoEasyDeLModelForCausalLM

logger = get_logger("eSurgeEngine")


@dataclass
class CompletionOutput:
    """Output of a single completion."""

    index: int
    text: str
    token_ids: list[int]
    cumulative_logprob: float | None = None
    logprobs: list[dict[int, float]] | None = None
    finish_reason: str | None = None


@dataclass
class RequestOutput:
    """Output of a generation request with comprehensive metrics.

    This class provides a consistent output format for all eSurge generation methods,
    including real-time metrics like tokens per second and accumulated text.
    """

    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    outputs: list[CompletionOutput]
    finished: bool = False
    metrics: dict[str, Any] | None = None
    accumulated_text: str = ""
    tokens_per_second: float = 0.0
    num_generated_tokens: int = 0
    time_spent_generating: float = 0.0
    first_token_time: float | None = None
    processing_time: float = 0.0

    def get_text(self) -> str:
        """Get the generated text from the first output."""
        return self.outputs[0].text if self.outputs else ""

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the generation including all metrics."""
        return {
            "request_id": self.request_id,
            "text": self.get_text(),
            "tokens_per_second": self.tokens_per_second,
            "num_generated_tokens": self.num_generated_tokens,
            "time_spent_generating": self.time_spent_generating,
            "finished": self.finished,
            "finish_reason": self.outputs[0].finish_reason if self.outputs else None,
        }


class eSurge:
    """High-level engine interface for text generation with eSurge."""

    def __init__(
        self,
        model: str | AutoEasyDeLModelForCausalLM,
        tokenizer: str | PreTrainedTokenizerBase | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        max_model_len: int = 8192,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int | None = None,
        hbm_utilization: float = 0.85,
        page_size: int = 128,
        enable_prefix_caching: bool = True,
        auto_shard_model: bool = True,
        sharding_axis_dims: tuple[int, ...] = (1, 1, 1, -1, 1),
        compile_runner: bool = True,
        runner_verbose: bool = False,
        use_combined_forward: bool = False,
        use_aot_forward: bool = True,
        esurge_name: str | None = None,
        **kwargs,
    ):
        """Initialize eSurge engine.

        Args:
            model: Model name/path or preloaded EasyDeL model
            tokenizer: Tokenizer name/path or preloaded tokenizer
            dtype: Data type for model computations
            max_model_len: Maximum sequence length
            max_num_seqs: Maximum number of sequences to process
            max_num_batched_tokens: Maximum tokens in a batch
            hbm_utilization: HBM memory utilization factor
            page_size: Page size for KV cache
            enable_prefix_caching: Enable prefix caching optimization
            auto_shard_model: Enable automatic model sharding
            sharding_axis_dims: Sharding dimensions
            **kwargs: Additional model configuration
        """

        from easydel import AutoEasyDeLModelForCausalLM, EasyDeLBaseConfigDict
        from easydel.layers.attention import AttentionMechanisms

        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.page_size = page_size

        if isinstance(model, str):
            self.model = AutoEasyDeLModelForCausalLM.from_pretrained(
                model,
                dtype=dtype,
                param_dtype=dtype,
                precision=jax.lax.Precision.DEFAULT,
                auto_shard_model=auto_shard_model,
                sharding_axis_dims=sharding_axis_dims,
                config_kwargs=EasyDeLBaseConfigDict(
                    attn_mechanism=kwargs.get("attn_mechanism", AttentionMechanisms.PAGED_ATTENTION),
                    attn_dtype=dtype,
                    kvdtype=dtype,
                    freq_max_position_embeddings=max_model_len,
                    mask_max_position_embeddings=max_model_len,
                    **kwargs.get("config_kwargs", {}),
                ),
                **{k: v for k, v in kwargs.items() if k not in ["attn_mechanism", "config_kwargs"]},
            )
        else:
            self.model = model

        if tokenizer is None:
            if isinstance(model, str):
                self.tokenizer = AutoTokenizer.from_pretrained(model)
            else:
                raise ValueError("Tokenizer must be provided when using preloaded model")
        elif isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

        self._monitoring_server = None
        self._dashboard = None
        self._dashboard_thread = None
        self._dashboard_urls = None
        self._monitoring_initialized = False
        self._esurge_name = esurge_name

        self.runner = eSurgeRunner(
            model=self.model,
            hbm_utilization=hbm_utilization,
            page_size=page_size,
            max_model_len=max_model_len,
            max_num_seqs=max_num_seqs,
            use_combined_forward=use_combined_forward,
            use_aot_forward=use_aot_forward,
            verbose=runner_verbose,
        )

        if compile_runner:
            self.runner.compile()
        self.scheduler = Scheduler.from_runner(
            self.runner,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_prefix_caching=enable_prefix_caching,
        )

        self._request_counter = 0
        self._active_requests: dict[str, dict] = {}
        self._request_outputs: dict[str, RequestOutput] = {}

    def _calculate_model_size(self, graphstate) -> str:
        """Calculates the model size in billions of parameters."""
        try:
            num_params = sum(n.size for n in jax.tree_util.tree_flatten(graphstate)[0])
            size_in_billions = num_params / 1e9
            return f"{size_in_billions:.2f}"
        except Exception:
            return "unknown"

    def _get_model_name(self, model) -> str:
        """Generates a model name string including type and size."""
        model_type = self._get_model_type(model)
        model_size = self._calculate_model_size(model.graphstate)
        return f"{model_type}-{model_size}b"

    def _get_model_type(self, model) -> str:
        """Extracts the model type from the model's configuration."""
        return getattr(model.config, "model_type", "unknown").lower()

    @cached_property
    def esurge_name(self) -> str:
        """Returns a standardized name for the esruge and its model."""
        return self._esurge_name or self._get_model_name(self.model)

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
        request_id: str | list[str] | None = None,
        use_tqdm: bool = True,
    ) -> list[RequestOutput]:
        """Generate completions for prompts.

        Args:
            prompts: Single prompt or list of prompts
            sampling_params: Sampling parameters for generation
            request_id: Optional request ID(s)
            use_tqdm: Show progress bar

        Returns:
            List of RequestOutput objects
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        if request_id is None:
            request_ids = [self._generate_request_id() for _ in prompts]
        elif isinstance(request_id, str):
            request_ids = [request_id]
        else:
            request_ids = request_id

        if sampling_params is None:
            sampling_params = SamplingParams(max_tokens=128)

        for prompt, req_id in zip(prompts, request_ids, strict=False):
            self._add_request(req_id, prompt, sampling_params)

        outputs = []
        if use_tqdm:
            from tqdm import tqdm

            pbar = tqdm(total=len(prompts), desc="Generating")

        completed = set()
        while len(completed) < len(prompts):
            scheduler_output = self.scheduler.schedule()
            if scheduler_output.scheduled_new_reqs or scheduler_output.scheduled_cached_reqs:
                model_output = self.runner.execute_model(scheduler_output)
                engine_outputs = self.scheduler.update_from_output(scheduler_output, model_output)

                if engine_outputs:
                    self._process_engine_outputs(engine_outputs)

                for req_id in request_ids:
                    if req_id not in completed and req_id in self._request_outputs:
                        output = self._request_outputs[req_id]
                        if output.finished:
                            completed.add(req_id)
                            outputs.append(output)
                            if use_tqdm:
                                pbar.update(1)

        if use_tqdm:
            pbar.close()

        return outputs

    def stream(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
    ) -> Iterator[RequestOutput]:
        """Stream generation results for a single prompt or first prompt in list.

        Args:
            prompts: Input prompt(s) - if list, only first is used
            sampling_params: Sampling parameters
            request_id: Optional request ID

        Yields:
            RequestOutput objects as generation progresses
        """
        if isinstance(prompts, list):
            if len(prompts) == 0:
                raise ValueError("Empty prompt list provided")
            prompt = prompts[0]
        else:
            prompt = prompts

        if request_id is None:
            request_id = self._generate_request_id()

        if sampling_params is None:
            sampling_params = SamplingParams(max_tokens=128)

        self._add_request(request_id, prompt, sampling_params)

        while request_id not in self._request_outputs or not self._request_outputs[request_id].finished:
            scheduler_output = self.scheduler.schedule()
            if scheduler_output.scheduled_new_reqs or scheduler_output.scheduled_cached_reqs:
                model_output = self.runner.execute_model(scheduler_output)
                engine_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
                if engine_outputs:
                    self._process_engine_outputs(engine_outputs)
                if request_id in self._request_outputs:
                    yield self._request_outputs[request_id]

    def _add_request(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
    ) -> None:
        """Add a new request to the scheduler."""
        tokenizer_output = self.tokenizer(prompt, return_tensors=None)
        token_ids = tokenizer_output["input_ids"]
        if isinstance(token_ids[0], list):
            token_ids = token_ids[0]

        eos_token_id = self.tokenizer.eos_token_id

        engine_request = EngineRequest(
            request_id=request_id,
            prompt_token_ids=token_ids,
            sampling_params=sampling_params,
            eos_token_id=eos_token_id,
        )

        self.scheduler.add_request(engine_request)
        self._active_requests[request_id] = {
            "prompt": prompt,
            "prompt_token_ids": token_ids,
            "generated_tokens": [],
            "last_decoded_index": 0,
            "start_time": time.time(),
            "first_token_time": None,
        }

        metrics_collector = get_metrics_collector()
        if metrics_collector:
            metrics_collector.start_request(request_id, len(token_ids))

        self._request_outputs[request_id] = RequestOutput(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=token_ids,
            outputs=[
                CompletionOutput(
                    index=0,
                    text="",
                    token_ids=[],
                )
            ],
            finished=False,
            accumulated_text="",
            tokens_per_second=0.0,
            num_generated_tokens=0,
            time_spent_generating=0.0,
            first_token_time=None,
            processing_time=0.0,
        )

    def _generate_request_id(self) -> str:
        """Generate unique request ID."""
        self._request_counter += 1
        return f"req-{uuid.uuid4().hex}-{self._request_counter}"

    def abort_request(self, request_id: str) -> None:
        """Abort a specific request."""
        if request_id in self.scheduler.requests:
            self.scheduler.requests[request_id].status = EngineRequestStatus.FINISHED_ABORTED

    @property
    def num_pending_requests(self) -> int:
        """Get number of pending requests."""
        return len(self.scheduler.waiting)

    @property
    def num_running_requests(self) -> int:
        """Get number of running requests."""
        return len(self.scheduler.running)

    def _process_engine_outputs(self, engine_outputs: dict[int, EngineCoreOutputs]) -> None:
        """Process engine outputs and update request outputs."""
        for client_outputs in engine_outputs.values():
            for engine_output in client_outputs.outputs:
                request_id = engine_output.request_id

                if request_id not in self._request_outputs:
                    continue

                request_data = self._active_requests.get(request_id)
                if not request_data:
                    continue

                new_tokens = engine_output.new_token_ids
                if new_tokens:
                    request_data["generated_tokens"].extend(new_tokens)

                    current_time = time.time()
                    if request_data["first_token_time"] is None and len(request_data["generated_tokens"]) > 0:
                        request_data["first_token_time"] = current_time - request_data["start_time"]

                        metrics_collector = get_metrics_collector()
                        if metrics_collector:
                            metrics_collector.record_first_token(request_id)

                    metrics_collector = get_metrics_collector()
                    if metrics_collector:
                        metrics_collector.add_generated_tokens(request_id, len(new_tokens))

                    accumulated_text = self.tokenizer.decode(
                        request_data["generated_tokens"],
                        skip_special_tokens=True,
                    )

                    last_index = request_data["last_decoded_index"]
                    new_text = self.tokenizer.decode(
                        request_data["generated_tokens"][last_index:],
                        skip_special_tokens=True,
                    )
                    request_data["last_decoded_index"] = len(request_data["generated_tokens"])

                    output = self._request_outputs[request_id]
                    output.outputs[0].text = new_text
                    output.outputs[0].token_ids = request_data["generated_tokens"].copy()

                    output.accumulated_text = accumulated_text
                    output.num_generated_tokens = len(request_data["generated_tokens"])

                    elapsed = current_time - request_data["start_time"]
                    output.processing_time = elapsed
                    output.time_spent_generating = elapsed
                    output.first_token_time = request_data["first_token_time"]

                    if request_data["first_token_time"] is not None and output.num_generated_tokens > 0:
                        generation_time = elapsed - request_data["first_token_time"]
                        output.tokens_per_second = (
                            output.num_generated_tokens / generation_time if generation_time > 0 else 0
                        )
                    else:
                        output.tokens_per_second = 0

                if engine_output.finished:
                    output = self._request_outputs[request_id]
                    output.finished = True
                    output.outputs[0].finish_reason = (
                        str(engine_output.finish_reason) if engine_output.finish_reason else None
                    )

                    if not new_tokens:
                        output.outputs[0].text = ""

                    elapsed = time.time() - request_data["start_time"]
                    num_prompt_tokens = len(request_data["prompt_token_ids"])
                    num_generated_tokens = len(request_data["generated_tokens"])

                    output.processing_time = elapsed
                    output.time_spent_generating = elapsed
                    output.num_generated_tokens = num_generated_tokens
                    output.first_token_time = request_data.get("first_token_time")

                    if output.first_token_time is not None and num_generated_tokens > 0:
                        generation_time = elapsed - output.first_token_time
                        output.tokens_per_second = num_generated_tokens / generation_time if generation_time > 0 else 0
                    else:
                        output.tokens_per_second = 0

                    output.metrics = {
                        "prompt_tokens": num_prompt_tokens,
                        "generated_tokens": num_generated_tokens,
                        "total_tokens": num_prompt_tokens + num_generated_tokens,
                        "processing_time": elapsed,
                        "first_token_time": output.first_token_time,
                        "tokens_per_second": output.tokens_per_second,
                    }

                    metrics_collector = get_metrics_collector()
                    if metrics_collector:
                        metrics_collector.complete_request(
                            request_id,
                            finish_reason=output.outputs[0].finish_reason,
                        )

    def start_monitoring(
        self,
        dashboard_port: int = 11481,
        prometheus_port: int = 11184,
        dashboard_host: str = "0.0.0.0",
        enable_prometheus: bool = True,
        enable_dashboard: bool = True,
        enable_console: bool = False,
        log_file: str | None = None,
        log_interval: float = 10.0,
        history_size: int = 1000,
        enable_detailed_logging: bool = True,
    ) -> dict[str, str]:
        """Start comprehensive monitoring for this eSurge instance.

        Args:
            dashboard_port: Port for web dashboard
            prometheus_port: Port for Prometheus metrics endpoint
            dashboard_host: Host for dashboard server
            enable_prometheus: Enable Prometheus metrics server
            enable_dashboard: Enable web dashboard
            enable_console: Enable rich console monitor
            log_file: Path to metrics log file
            log_interval: Interval for summary logging (seconds)
            history_size: Number of metrics to keep in memory
            enable_detailed_logging: Enable detailed per-request logging

        Returns:
            Dictionary with URLs of started services
        """
        if self._monitoring_initialized:
            logger.info("Monitoring already initialized for this eSurge instance")
            return self._dashboard_urls

        logger.info("Starting eSurge monitoring services...")

        if not get_metrics_collector():
            initialize_metrics(
                log_file=log_file,
                log_interval=log_interval,
                history_size=history_size,
                enable_detailed_logging=enable_detailed_logging,
            )
            logger.info(" Metrics collection initialized")

        urls = {}

        if enable_prometheus:
            try:
                from .monitoring import start_monitoring_server

                self._monitoring_server = start_monitoring_server(
                    prometheus_port=prometheus_port,
                    update_interval=1.0,
                )
                urls["prometheus"] = f"http://{dashboard_host}:{prometheus_port}/metrics"
                logger.info(f" Prometheus metrics: {urls['prometheus']}")
            except ImportError:
                logger.info(" Prometheus monitoring unavailable (install prometheus-client)")
            except Exception as e:
                logger.info(f" Failed to start Prometheus server: {e}")

        if enable_dashboard:
            try:
                from .dashboard import create_dashboard

                self._dashboard = create_dashboard(
                    host=dashboard_host,
                    port=dashboard_port,
                    debug=False,
                )

                def run_dashboard():
                    self._dashboard.run(log_level="warning")

                self._dashboard_thread = threading.Thread(target=run_dashboard, daemon=True)
                self._dashboard_thread.start()

                urls["dashboard"] = f"http://{dashboard_host}:{dashboard_port}"
                urls["health"] = f"http://{dashboard_host}:{dashboard_port}/health"
                urls["api"] = f"http://{dashboard_host}:{dashboard_port}/api/metrics"

                logger.info(f" Web dashboard: {urls['dashboard']}")
                logger.info(f" Health check: {urls['health']}")

            except ImportError:
                logger.info(" Web dashboard unavailable (install fastapi uvicorn)")
            except Exception as e:
                logger.info(f" Failed to start dashboard: {e}")

        if enable_console:
            try:
                from .monitoring import start_console_monitor

                logger.info(" Starting console monitor...")
                start_console_monitor(refresh_rate=1.0)
            except ImportError:
                logger.info(" Console monitor unavailable (install rich)")
            except Exception as e:
                logger.info(f" Failed to start console monitor: {e}")

        self._monitoring_initialized = True

        if urls:
            logger.info(" Monitoring services started successfully!")
            logger.info(" Metrics will be automatically collected during inference")
            if enable_dashboard:
                logger.info(f" Open {urls['dashboard']} to view real-time metrics")
        else:
            logger.info(" No monitoring services were started successfully")
        self._dashboard_urls = urls
        return urls

    def stop_monitoring(self) -> None:
        """Stop all monitoring services for this eSurge instance."""
        if not self._monitoring_initialized:
            logger.info("No monitoring services to stop")
            return

        logger.info("Stopping eSurge monitoring services...")

        if self._monitoring_server:
            try:
                self._monitoring_server.stop()
                logger.info(" Prometheus server stopped")
            except Exception as e:
                logger.info(f" Error stopping Prometheus server: {e}")
            self._monitoring_server = None

        if self._dashboard_thread and self._dashboard_thread.is_alive():
            try:
                logger.info(" Dashboard server will stop with process")
            except Exception as e:
                logger.info(f" Error stopping dashboard: {e}")
            self._dashboard_thread = None
            self._dashboard = None

        self._monitoring_initialized = False
        logger.info(" Monitoring services stopped")

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get current metrics summary for this eSurge instance."""
        metrics_collector = get_metrics_collector()
        if not metrics_collector:
            return {"error": "Metrics collection not initialized"}

        system_metrics = metrics_collector.get_system_metrics()
        return {
            "requests_per_second": system_metrics.requests_per_second,
            "average_latency": system_metrics.average_latency,
            "average_ttft": system_metrics.average_ttft,
            "average_throughput": system_metrics.average_throughput,
            "total_completed": system_metrics.total_requests_completed,
            "total_failed": system_metrics.total_requests_failed,
            "total_tokens": system_metrics.total_tokens_generated,
            "active_requests": len(self._active_requests),
            "queue_size": self.num_pending_requests,
            "running_requests": self.num_running_requests,
        }

    @property
    def monitoring_active(self) -> bool:
        """Check if monitoring is currently active."""
        return self._monitoring_initialized

    def __del__(self):
        """Cleanup monitoring services when eSurge instance is destroyed."""
        if self._monitoring_initialized:
            try:
                self.stop_monitoring()
            except Exception:
                pass
