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

"""eSurge Engine - High-Performance Inference Engine for EasyDeL.

This module provides the eSurge engine, a high-performance text generation system
built on JAX that offers efficient batched inference with advanced features like
continuous batching and comprehensive monitoring.

Key Components:
    - eSurge: Main engine class for text generation
    - RequestOutput: Container for generation results and metrics
    - CompletionOutput: Individual completion within a batch

Features:
    - **Continuous Batching**: Background scheduler thread processes requests
      continuously for optimal throughput.
    - **Context Management**: Automatic prompt truncation and token reservation
      with configurable strategies.
    - **Streaming Support**: Real-time token streaming with delta updates.
    - **Monitoring**: Built-in Prometheus metrics and console monitor (Grafana-ready).

Usage Example:
    >>> from easydel.inference.esurge import eSurge
    >>> from easydel.inference.sampling_params import SamplingParams
    >>>
    >>> # Initialize engine
    >>> engine = eSurge(
    ...     model="model-name",
    ...     max_model_len=8192,
    ...     reserve_tokens=800
    ... )
    >>>
    >>> # Stream generation
    >>> for output in engine.stream("Tell me about AI"):
    ...     print(output.delta_text, end="", flush=True)
    >>>
    >>> # Batch generation
    >>> outputs = engine.generate(
    ...     ["Question 1?", "Question 2?"],
    ...     SamplingParams(max_tokens=100, temperature=0.7)
    ... )

Technical Details:
    The engine uses a multi-threaded architecture with:
    - Main thread: Handles API calls and request submission
    - Scheduler thread: Continuously processes queued requests
    - JAX computation: Executes model forward passes
"""

from __future__ import annotations

import copy
import os
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import typing
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import flax
import flax.nnx
import jax
from eformer.loggings import get_logger
from jax import numpy as jnp
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from easydel.inference.sampling_params import SamplingParams
from easydel.utils import Registry
from easydel.workers.esurge.pipeline import DetokenizerResult, WorkerManager

from .engine_types import EngineCoreOutputs
from .metrics import get_metrics_collector, initialize_metrics, log_metrics_summary
from .multimodal import MultiModalManager
from .request import EngineRequest, EngineRequestStatus
from .runners import eSurgeRunner
from .scheduler import Scheduler, SchedulerOutput
from .utils import truncate_tokens

if typing.TYPE_CHECKING:
    from easydel.infra import EasyDeLBaseModule

logger = get_logger("eSurgeEngine")

# Configuration constants
DEFAULT_DETOKENIZER_MAX_STATES = 1 << 16  # 65536 states for streaming decode
DEFAULT_PAGE_SIZE_GPU_MIN = 256  # Minimum efficient page size for GPU
DEFAULT_DECODE_INTERVAL_TOKENS = 4  # Decode every N tokens
DEFAULT_DECODE_INTERVAL_SECS = 0.02  # Or decode every N seconds (20ms)
# Default to fail-fast (1) so benchmark runs don't spin for hours on fatal errors.
# Set `EASURGE_MAX_SCHEDULER_ERRORS=10` (or higher) to restore retry behavior.
MAX_CONSECUTIVE_SCHEDULER_ERRORS = int(os.environ.get("EASURGE_MAX_SCHEDULER_ERRORS", "1"))
WORKER_DRAIN_MAX_RETRIES = 3  # Maximum retry attempts for worker drain
WORKER_DRAIN_INITIAL_DELAY = 0.1  # Initial retry delay in seconds


def _set_requested_new(sp, n: int):
    if hasattr(sp, "max_tokens"):
        sp.max_tokens = int(n)
    if hasattr(sp, "max_new_tokens"):
        sp.max_new_tokens = int(n)


@dataclass
class CompletionOutput:
    """Output of a single completion.

    Represents the generated output for a single completion within a batch request.
    Contains the generated text, token IDs, and optional probability information.

    Attributes:
        index: Position of this completion in the batch (0-indexed).
        text: The generated text string.
        token_ids: List of token IDs that were generated.
        cumulative_logprob: Cumulative log probability of the generated sequence.
        logprobs: Per-token log probabilities as dict mapping token_id to logprob.
        finish_reason: Reason for completion termination ('stop', 'length', 'eos_token', etc.).
    """

    index: int
    text: str
    token_ids: list[int]
    cumulative_logprob: float | None = None
    logprobs: list[dict[int, float]] | None = None
    finish_reason: str | None = None


@dataclass
class RequestOutput:
    """Output of a generation request with comprehensive metrics.

    Contains the complete output for a generation request including generated text,
    performance metrics, and streaming support fields. Used for both batch and
    streaming generation modes.

    Attributes:
        request_id: Unique identifier for this request.
        prompt: Original prompt text.
        prompt_token_ids: Tokenized prompt as list of token IDs.
        outputs: List of CompletionOutput objects (one per n in sampling params).
        finished: Whether generation has completed.
        metrics: Dictionary of performance metrics (tokens, timing, etc.).
        accumulated_text: Full generated text accumulated so far.
        delta_text: Only the latest decoded text chunk (for streaming).
        tokens_per_second: Current generation throughput.
        num_generated_tokens: Total number of tokens generated.
        time_spent_generating: Total time spent in generation.
        first_token_time: Time to first token (TTFT) in seconds.
        processing_time: Total processing time including queuing.
        update_seq: Sequence number incremented on any update.
        delta_seq: Sequence number incremented only when delta_text changes.
    """

    request_id: str
    prompt: str | list[str]
    prompt_token_ids: list[list[int]] | list[int]
    outputs: list[CompletionOutput]
    finished: bool = False
    metrics: dict[str, Any] | None = None

    accumulated_text: str = ""  # full text so far
    delta_text: str = ""  # only the latest decoded chunk
    tokens_per_second: float = 0.0
    num_generated_tokens: int = 0
    time_spent_generating: float = 0.0
    first_token_time: float | None = None
    processing_time: float = 0.0

    update_seq: int = 0
    delta_seq: int = 0

    def get_text(self) -> str:
        """Get the generated text from the first completion output.

        Returns:
            Generated text string, or empty string if no outputs.
        """
        return self.outputs[0].text if self.outputs else ""

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the request output.

        Returns:
            Dictionary containing key metrics: request_id, text, throughput,
            token count, timing, completion status and finish reason.
        """
        return {
            "request_id": self.request_id,
            "text": self.get_text(),
            "tokens_per_second": self.tokens_per_second,
            "num_generated_tokens": self.num_generated_tokens,
            "time_spent_generating": self.time_spent_generating,
            "finished": self.finished,
            "finish_reason": self.outputs[0].finish_reason if self.outputs else None,
        }


@Registry.register("serve", "esurge")
class eSurge:
    """High-level engine interface for text generation with eSurge.

    eSurge is a high-performance inference engine built on JAX that provides:
    - Efficient batched inference with paged attention
    - Continuous batching with background scheduling
    - Streaming generation with delta text tracking
    - Comprehensive monitoring and metrics
    - Thread-safe request handling
    - Dynamic context management with automatic prompt truncation

    The engine runs a background scheduler thread that continuously processes
    requests from the queue, enabling high throughput and low latency.

    Key Features:
        - **Context Management**: Automatically manages context length with configurable
          truncation strategies and token reservation.
        - **Streaming Support**: Efficient incremental decoding with configurable
          intervals for optimal performance.
        - **Monitoring**: Built-in Prometheus metrics and console monitoring (visualize with Grafana).

    Example:
        >>> # Initialize engine
        >>> engine = eSurge(
        ...     model="model-name",
        ...     max_model_len=8192,
        ...     reserve_tokens=800  # Reserve tokens for generation
        ... )
        >>> engine.initiate()
        >>>
        >>> # Generate with streaming
        >>> for output in engine.stream("Tell me a story"):
        ...     print(output.delta_text, end="", flush=True)
    """

    def __init__(
        self,
        model: str | EasyDeLBaseModule,
        tokenizer: str | PreTrainedTokenizerBase | None = None,
        dtype: jnp.dtype = jnp.bfloat16,
        max_model_len: int = 8192,
        min_input_pad: int = 16,
        min_token_pad: int | None = None,
        max_num_seqs: int = 256,
        max_num_seq_buckets: list[int] | None = None,
        max_num_batched_tokens: int | None = None,
        hbm_utilization: float = 0.85,
        page_size: int = 128,
        use_aot_forward: bool = True,
        enable_prefix_caching: bool = True,
        auto_shard_model: bool = True,
        sharding_axis_dims: tuple[int, ...] = (1, 1, 1, -1, 1),
        compile_runner: bool = True,
        runner_verbose: bool = False,
        overlap_execution: bool = False,
        sampler_metrics: bool = False,
        esurge_name: str | None = None,
        reserve_tokens: int | None = None,
        auto_truncate_prompt: bool = True,
        auto_cap_new_tokens: bool = True,
        strict_context: bool = False,
        truncate_mode: typing.Literal["left", "right", "middle"] = "left",
        prefer_preserve_prompt: bool = True,
        decode_truncated_prompt: bool = True,
        destroy_pages_on_pause: bool = True,
        detokenizer_max_states: int = DEFAULT_DETOKENIZER_MAX_STATES,
        tokenizer_endpoint: str | None = None,
        detokenizer_endpoint: str | None = None,
        sampling_params_callback: typing.Callable[[SamplingParams, dict[str, typing.Any]], SamplingParams | None]
        | None = None,
        extra_eos_token_ids: list[int] | None = None,
        silent_mode: bool = False,
        # Vision-language model support
        processor: Any | None = None,
        resolution_buckets: list[tuple[int, int]] | None = None,
        vision_cache_capacity_mb: int = 1024,
        **kwargs,
    ):
        """Initialize the eSurge engine.

        Args:
            model: Model path (HuggingFace hub) or preloaded EasyDeL model instance.
            tokenizer: Deprecated alias for `processor`. Tokenizer path or instance.
            dtype: JAX dtype for model computations (default: bfloat16).
            max_model_len: Maximum sequence length the model can handle.
            min_input_pad: Minimum padding for input sequences.
            min_token_pad: Optional minimum token bucket size for compilation. When
                smaller than `min_input_pad`, this allows decode steps like `tok=1/b1`
                instead of `tok=1/b16`, at the cost of more compilation variants.
            max_num_seqs: Maximum number of concurrent sequences.
            max_num_seq_buckets: Optional explicit request-capacity buckets used for
                compilation (e.g., [1, 2, 4, 8, 16, 32]). When provided, the runner
                compiles these bucket sizes and selects the smallest that can fit
                the current active batch.
            max_num_batched_tokens: Maximum tokens per batch (auto-computed if None).
            hbm_utilization: Target HBM memory utilization (0.0-1.0).
            page_size: Page size for paged attention KV cache. Recommended >=256 for GPUs.
            enable_prefix_caching: Enable caching of common prefixes for efficiency.
            auto_shard_model: Automatically shard model across devices.
            sharding_axis_dims: Sharding configuration for model parallelism.
            compile_runner: JIT pre-compile the runner for better performance.
            runner_verbose: Enable verbose logging in runner.
            esurge_name: Optional custom name for this engine instance.
            reserve_tokens: Safety margin tokens reserved from max_model_len to prevent
                OOM or Scheduler errors. Defaults to max_model_len // 10.
            auto_truncate_prompt: Allow automatic prompt truncation when it exceeds
                the available context budget.
            auto_cap_new_tokens: Automatically cap max_new_tokens to fit within
                the model's context window.
            strict_context: If True, raise errors on context violations instead of
                auto-fixing. Use for strict validation.
            truncate_mode: Strategy for prompt truncation:
                - "left": Remove tokens from the beginning
                - "right": Remove tokens from the end
                - "middle": Remove tokens from the middle
            prefer_preserve_prompt: When True, prioritize preserving the prompt by
                capping new tokens first before truncating the prompt.
            decode_truncated_prompt: Re-decode truncated prompt to update the text
                representation when truncation occurs.
            overlap_execution: Enable double-buffered model execution to overlap
                scheduler work with device execution (experimental).
            sampler_metrics: Enable the lightweight sampler JIT for recording
                token log-probabilities on-device.
            detokenizer_max_states: Maximum number of streaming decode states
                the detokenizer worker will keep resident (power-of-two recommended).
            destroy_pages_on_pause: When True, destroying the ragged KV cache upon
                pause() to free memory, and lazily reinitializing it on resume().
            tokenizer_endpoint: ZMQ endpoint of the external tokenizer worker.
            detokenizer_endpoint: ZMQ endpoint of the external detokenizer worker.
            sampling_params_callback: Optional callable that can inspect/modify
                the SamplingParams for each submitted request. Receives a cloned
                SamplingParams instance and request metadata dict containing
                "request_id", "prompt", and "engine". May return a new instance
                or mutate the provided one in-place.
            extra_eos_token_ids: Additional EOS token IDs beyond the tokenizer's default.
                These will be treated as end-of-sequence tokens for all requests unless
                overridden in SamplingParams.
            silent_mode: If True, suppress informational eSurge engine logs.
            processor: Unified text/multimodal processor. Can be a tokenizer or an
                HF processor (with an embedded tokenizer). If None, falls back to
                `tokenizer` and then to loading from `model` when `model` is a string.
            **kwargs: Additional configuration passed to model loading.

        Raises:
            ValueError: If tokenizer not provided and cannot be inferred, or if
                configuration parameters are invalid.
        """
        from easydel import AutoEasyDeLModelForCausalLM, EasyDeLBaseConfigDict
        from easydel.layers.attention import AttentionMechanisms

        self.silent_mode = silent_mode
        self._info = logger.info if not self.silent_mode else lambda *args, **kwargs: None

        if reserve_tokens is None:
            reserve_tokens = max_num_seqs

        if max_model_len <= reserve_tokens:
            raise ValueError(f"Configuration error: max_model_len={max_model_len} <= reserve_tokens={reserve_tokens}")

        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self.page_size = page_size
        if kwargs.pop("use_combined_forward", None) is not None:
            logger.warning("`use_combined_forward` is deprecated (the fused step will be used now).")
        # `processor` is the unified interface for text + multimodal workflows.
        # Backward-compat: if `processor` isn't provided, fall back to `tokenizer`.
        if tokenizer is not None and processor is not None and tokenizer is not processor:
            logger.warning("Both `tokenizer` and `processor` were provided; `processor` will be used for multimodal.")

        processor_obj: Any | None = processor if processor is not None else tokenizer
        processor_source: str | None = None

        if processor_obj is None:
            if isinstance(model, str):
                processor_source = model
                processor_obj = AutoTokenizer.from_pretrained(model)
            else:
                raise ValueError("Processor must be provided when using a preloaded model.")
        elif isinstance(processor_obj, str):
            processor_source = processor_obj
            processor_obj = AutoTokenizer.from_pretrained(processor_obj)
        else:
            processor_source = getattr(processor_obj, "name_or_path", None)

        tokenizer_obj: PreTrainedTokenizerBase | None = None
        if isinstance(processor_obj, PreTrainedTokenizerBase):
            tokenizer_obj = processor_obj
        else:
            maybe_tok = getattr(processor_obj, "tokenizer", None)
            if isinstance(maybe_tok, PreTrainedTokenizerBase):
                tokenizer_obj = maybe_tok

        if tokenizer_obj is None:
            if isinstance(tokenizer, PreTrainedTokenizerBase):
                tokenizer_obj = tokenizer
            elif isinstance(tokenizer, str):
                processor_source = processor_source or tokenizer
                tokenizer_obj = AutoTokenizer.from_pretrained(tokenizer)
            else:
                source = processor_source or (model if isinstance(model, str) else None)
                if source is None:
                    raise ValueError(
                        "Tokenizer must be provided (or inferable from processor) when using a preloaded model."
                    )
                tokenizer_obj = AutoTokenizer.from_pretrained(source)

        tokenizer_source = (
            getattr(tokenizer_obj, "name_or_path", None)
            or processor_source
            or (model if isinstance(model, str) else None)
        )
        if tokenizer_source is None:
            raise ValueError("Could not infer a tokenizer source for tokenizer/detokenizer workers.")

        self.processor = processor_obj
        self.tokenizer = tokenizer_obj

        # Vision-language model support
        self._multimodal_manager: MultiModalManager | None = None
        if self.processor is not None:
            self._multimodal_manager = MultiModalManager(
                processor=self.processor,
                model=None if isinstance(model, str) else model,
                resolution_buckets=resolution_buckets,
                cache_capacity_mb=vision_cache_capacity_mb,
                enable_cache=True,
            )

        self._monitoring_server = None
        self._monitoring_urls: dict[str, str] | None = None
        self._monitoring_initialized = False
        self._grafana_container_name: str | None = None
        self._grafana_container_id: str | None = None
        self._grafana_process: subprocess.Popen | None = None
        self._grafana_temp_dir: str | None = None
        self._grafana_url: str | None = None
        self._esurge_name = esurge_name
        self._scheduler_running = False
        self.destroy_pages_on_pause = destroy_pages_on_pause
        self._kv_cache_valid = True
        self._paused = False
        self._sampling_params_callback = sampling_params_callback

        # Detokenizer cleanup tracking
        self._failed_detokenizer_resets: set[str] = set()
        self._detokenizer_cleanup_threshold = 100  # Clean up after this many failures

        tokenizer_endpoint = tokenizer_endpoint or os.environ.get("EASURGE_TOKENIZER_ENDPOINT")
        detokenizer_endpoint = detokenizer_endpoint or os.environ.get("EASURGE_DETOKENIZER_ENDPOINT")

        self._worker_manager = WorkerManager(tokenizer_source)
        self._tokenizer_client, self._detokenizer_client = self._worker_manager.start(
            detokenizer_max_states=detokenizer_max_states,
            tokenizer_endpoint=tokenizer_endpoint,
            detokenizer_endpoint=detokenizer_endpoint,
        )
        self._tokenizer_endpoint = self._worker_manager.tokenizer_endpoint
        self._detokenizer_endpoint = self._worker_manager.detokenizer_endpoint

        if isinstance(model, str):
            model = AutoEasyDeLModelForCausalLM.from_pretrained(
                model,
                dtype=dtype,
                param_dtype=dtype,
                precision=jax.lax.Precision.DEFAULT,
                auto_shard_model=auto_shard_model,
                sharding_axis_dims=sharding_axis_dims,
                config_kwargs=EasyDeLBaseConfigDict(
                    attn_mechanism=kwargs.get("attn_mechanism", AttentionMechanisms.RAGGED_PAGE_ATTENTION_V3),
                    attn_dtype=dtype,
                    kvdtype=dtype,
                    freq_max_position_embeddings=max_model_len,
                    mask_max_position_embeddings=max_model_len,
                    **kwargs.get("config_kwargs", {}),
                ),
                **{k: v for k, v in kwargs.items() if k not in ["attn_mechanism", "config_kwargs"]},
            )

        if self._multimodal_manager is not None and self._multimodal_manager.model is None:
            self._multimodal_manager.model = model

        # Profiling state
        self._profiling_active = False
        self._profiling_steps_remaining = 0
        self._profiling_output_dir: str | None = None
        self._profiling_host_level: int | None = None
        self._profiling_python_level: int | None = None
        self._possible_name = self._get_model_name(model)

        self.runner = eSurgeRunner(
            model=model.esurge_compatible_model,
            hbm_utilization=hbm_utilization,
            page_size=page_size,
            max_model_len=max_model_len,
            min_input_pad=min_input_pad,
            max_num_seqs=max_num_seqs,
            max_num_seq_buckets=max_num_seq_buckets,
            min_token_pad=min_token_pad,
            use_aot_forward=use_aot_forward,
            verbose=runner_verbose,
            enable_overlap_execution=overlap_execution,
            enable_sampler_metrics=sampler_metrics,
        )
        self._overlap_execution = overlap_execution
        if compile_runner:
            self.runner.compile()

        self.scheduler = Scheduler.from_runner(
            self.runner,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_prefix_caching=enable_prefix_caching,
        )
        self._scheduler_max_num_batched_tokens = max_num_batched_tokens
        self._scheduler_enable_prefix_caching = enable_prefix_caching

        # Streaming decode cadence
        self.decode_interval_tokens = DEFAULT_DECODE_INTERVAL_TOKENS
        self.decode_interval_secs = DEFAULT_DECODE_INTERVAL_SECS

        # State
        self._request_counter = 0
        self._active_requests: dict[str, dict] = {}
        self._request_outputs: dict[str, RequestOutput] = {}

        # Per-request events to support many concurrent streams
        self._request_events: dict[str, threading.Event] = {}
        self.reserve_tokens = reserve_tokens
        self.auto_truncate_prompt = auto_truncate_prompt
        self.auto_cap_new_tokens = auto_cap_new_tokens
        self.strict_context = strict_context
        self.truncate_mode = truncate_mode
        self.prefer_preserve_prompt = prefer_preserve_prompt
        self.decode_truncated_prompt = decode_truncated_prompt
        self.extra_eos_token_ids = extra_eos_token_ids or []
        # Locks and signals
        self._scheduler_lock = threading.RLock()
        self._request_lock = threading.RLock()
        self._output_lock = threading.RLock()
        self._counter_lock = threading.Lock()
        self._output_event = threading.Event()  # kept for generate()

        # Scheduler thread
        self._scheduler_thread: threading.Thread | None = None
        self._scheduler_running = False
        self._scheduler_exception: BaseException | None = None
        self._scheduler_exception_tb: str | None = None

        self.initiate()

    def _calculate_model_size(self, graphstate) -> str:
        try:
            num_params = sum(n.size for n in jax.tree_util.tree_flatten(graphstate)[0])
            return f"{num_params / 1e9:.2f}"
        except Exception:
            return "unknown"

    def _get_model_type(self, model) -> str:
        return getattr(model.config, "model_type", "unknown").lower()

    def _get_model_name(self, model) -> str:
        model_type = self._get_model_type(model)
        model_size = self._calculate_model_size(model.graphstate)
        return f"{model_type}-{model_size}b"

    @cached_property
    def esurge_name(self) -> str:
        return self._esurge_name or self._possible_name

    def set_sampling_params_callback(
        self,
        callback: typing.Callable[[SamplingParams, dict[str, typing.Any]], SamplingParams | None] | None,
    ) -> None:
        """Register or clear the sampling-params callback.

        Args:
            callback: Callable receiving a cloned SamplingParams and metadata
                dict (``request_id``, ``prompt``, ``engine``). Return a new
                SamplingParams, mutate the provided one, or return None to
                keep the original values. Pass None to disable the callback.
        """

        self._sampling_params_callback = callback

    def _abort_scheduler_due_to_error(self, exc: BaseException) -> None:
        # Record the failure so waiting callers can raise immediately.
        self._scheduler_exception = exc
        self._scheduler_exception_tb = traceback.format_exc()

        # Stop the scheduler and wake up any waiters (generate/stream/chat).
        self._scheduler_running = False
        self._output_event.set()
        with self._request_lock:
            events = list(self._request_events.values())
        for ev in events:
            ev.set()

    def _raise_if_scheduler_failed(self) -> None:
        exc = self._scheduler_exception
        if exc is None:
            return
        tb = self._scheduler_exception_tb
        if tb:
            raise RuntimeError(f"eSurge scheduler crashed: {exc}\n{tb}") from exc
        raise RuntimeError(f"eSurge scheduler crashed: {exc}") from exc

    def initiate(self) -> None:
        """Start the background scheduler thread.

        Initiates a daemon thread that continuously runs the scheduler loop,
        processing requests from the queue and updating outputs. This must
        be called before using generate() or stream() methods.

        The scheduler thread will:
        1. Schedule requests from the waiting queue
        2. Execute model forward passes
        3. Update request outputs with generated tokens
        4. Signal waiting threads when updates are available
        """
        with self._scheduler_lock:
            if self._scheduler_running:
                self._info("Scheduler loop is already running")
                return

            if self.runner.executor_manager.kv_pages is None:
                self.runner.initialize_kv_cache()
                self._kv_cache_valid = True

            # Clear any previous crash state before starting a fresh scheduler thread.
            self._scheduler_exception = None
            self._scheduler_exception_tb = None

            def _scheduler_loop():
                self._info("Starting background scheduler loop")
                consecutive_errors = 0
                max_consecutive_errors = MAX_CONSECUTIVE_SCHEDULER_ERRORS

                if not self._overlap_execution:
                    while self._scheduler_running:
                        try:
                            with self._scheduler_lock:
                                scheduler_output = self.scheduler.schedule()
                            model_output = self.runner.execute_model(scheduler_output)
                            with self._scheduler_lock:
                                engine_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
                            if engine_outputs:
                                self._process_engine_outputs(engine_outputs)
                            # Reset error counter on success
                            consecutive_errors = 0
                        except KeyboardInterrupt:
                            self._info("Scheduler loop interrupted by user")
                            break
                        except Exception as e:
                            consecutive_errors += 1
                            traceback.print_exc()
                            logger.error(
                                "Scheduler loop error (attempt %d/%d): %s",
                                consecutive_errors,
                                max_consecutive_errors,
                                e,
                            )

                            if consecutive_errors >= max_consecutive_errors:
                                logger.critical(
                                    f"Scheduler loop encountered {consecutive_errors} consecutive errors. "
                                    "Stopping scheduler to prevent resource exhaustion."
                                )
                                self._abort_scheduler_due_to_error(e)
                                break
                            time.sleep(0.01)
                    self._info("Background scheduler loop stopped")
                    return

                pending_future: tuple[Any, SchedulerOutput] | None = None
                prefetched_schedule: SchedulerOutput | None = None

                def _can_prefetch_next(current: SchedulerOutput) -> bool:
                    # Only prefetch when the current batch is guaranteed not to
                    # generate new output tokens. That keeps scheduler state
                    # deterministic (no token-dependent stop conditions) while
                    # overlapping schedule work with device execution.
                    try:
                        for rid in current.num_scheduled_tokens:
                            req = self.scheduler.requests.get(rid)
                            if req is None:
                                continue
                            if req.num_computed_tokens >= req.num_tokens:
                                return False
                    except Exception:
                        return False
                    return True

                while self._scheduler_running:
                    try:
                        if pending_future is not None:
                            future, prev_sched_out = pending_future

                            # Opportunistically prefetch the next schedule while the
                            # current batch is still running on device (prefill-only).
                            if prefetched_schedule is None and _can_prefetch_next(prev_sched_out):
                                with self._scheduler_lock:
                                    prefetched_schedule = self.scheduler.schedule()

                            self._drain_runner_future(future, prev_sched_out)
                            pending_future = None

                        if prefetched_schedule is not None:
                            scheduler_output = prefetched_schedule
                            prefetched_schedule = None
                        else:
                            with self._scheduler_lock:
                                scheduler_output = self.scheduler.schedule()
                        future = self.runner.execute_model_async(scheduler_output)
                        pending_future = (future, scheduler_output)
                        # Reset error counter on success
                        consecutive_errors = 0
                    except KeyboardInterrupt:
                        self._info("Scheduler loop interrupted by user")
                        break
                    except Exception as e:
                        consecutive_errors += 1
                        traceback.print_exc()
                        logger.error(
                            "Scheduler loop error (attempt %d/%d): %s",
                            consecutive_errors,
                            max_consecutive_errors,
                            e,
                        )

                        if consecutive_errors >= max_consecutive_errors:
                            logger.critical(
                                f"Scheduler loop encountered {consecutive_errors} consecutive errors. "
                                "Stopping scheduler to prevent resource exhaustion."
                            )
                            self._abort_scheduler_due_to_error(e)
                            break
                        time.sleep(0.01)

                if pending_future is not None:
                    try:
                        self._drain_runner_future(*pending_future)
                    except Exception as e:
                        traceback.print_exc()
                        logger.error("Error processing pending batch: %s", e)

                self._info("Background scheduler loop stopped")

            self._scheduler_running = True
            self._scheduler_thread = threading.Thread(target=_scheduler_loop, daemon=True)
            self._scheduler_thread.start()
            self._info("Background scheduler initiated")
            self._paused = False

    def terminate(self) -> None:
        """Stop the background scheduler thread.

        Gracefully shuts down the scheduler loop and waits for the thread
        to terminate. Should be called when the engine is no longer needed
        to free resources.
        """
        with self._scheduler_lock:
            if not self._scheduler_running:
                self._info("Scheduler loop is not running")
                return
            self._info("Stopping background scheduler loop...")
            self._scheduler_running = False
            if self._scheduler_thread:
                self._scheduler_thread.join(timeout=5.0)
                if self._scheduler_thread.is_alive():
                    logger.warning("Scheduler thread did not stop gracefully")
                self._scheduler_thread = None
            self._info("Background scheduler terminated")
            if self._profiling_active:
                try:
                    self.stop_profiling()
                except Exception:
                    logger.debug("Profiler stop encountered an error", exc_info=True)
            if hasattr(self.runner, "shutdown"):
                try:
                    self.runner.shutdown()
                except Exception:
                    logger.debug("Runner shutdown encountered an error", exc_info=True)
            # Clear runner buffers if idle to avoid stale state on next start.
            self._reset_runner_state_if_idle("terminate")

    def pause(self) -> None:
        """Pause the background scheduler without clearing queued state."""
        if not self._scheduler_running:
            self._info("Scheduler loop already paused or not running")
            self._paused = True
            return

        self._info("Pausing eSurge scheduler loop...")
        self.terminate()
        self._paused = True
        self._drain_pipeline_workers("pause")
        if self.destroy_pages_on_pause:
            if self.num_running_requests > 0 or self.num_pending_requests > 0:
                logger.warning(
                    f"Active or pending requests detected; skipping KV cache destruction (num running requests "
                    f"{self.num_running_requests} | num pending requests {self.num_pending_requests})."
                )
            else:
                self.runner.destroy_kv_cache()
                self._kv_cache_valid = False
                self._log_cache_event("kv_cache_destroyed", {"reason": "pause"})
        # Always try to clear runner buffers when idle to avoid stale state.
        self._reset_runner_state_if_idle("pause")

    def resume(self) -> None:
        """Resume the scheduler if it was paused."""
        if self._scheduler_running:
            self._info("Scheduler loop already running")
            return
        self._info("Resuming eSurge scheduler loop...")
        self._drain_pipeline_workers("resume")
        if self.destroy_pages_on_pause and not self._kv_cache_valid:
            self.runner.initialize_kv_cache()
            self._kv_cache_valid = True
            self._log_cache_event("kv_cache_reinitialized", {"reason": "resume"})
        self.initiate()

    def _format_chat_prompt(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
        chat_template: str | None = None,
        tools: list[dict] | None = None,
    ) -> str:
        """Format chat messages into a prompt string using the tokenizer's chat template.

        Converts a list of chat messages into a formatted prompt string that can be
        passed to the model for generation. Uses the tokenizer's built-in chat template
        or a custom template if provided.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
                Roles can be 'system', 'user', 'assistant', etc.
            add_generation_prompt: Whether to add the generation prompt token/string
                at the end to indicate the model should generate a response.
            chat_template: Optional custom chat template to override the tokenizer's
                default template. Should be a Jinja2 template string.
            tools: Optional list of tool/function definitions that the model can use.
                Format depends on the specific model's tool calling conventions.

        Returns:
            Formatted prompt string ready for tokenization and generation.

        Example:
            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "What is 2+2?"}
            ... ]
            >>> prompt = engine._format_chat_prompt(messages)
            >>> # Returns formatted string like: "<|system|>You are a helpful assistant.<|user|>What is 2+2?<|assistant|>"

        Note:
            The exact format depends on the tokenizer's chat template. Different models
            use different special tokens and formatting conventions.
        """
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            chat_template=chat_template,
        )

    def _tokenize_prompt(self, request_id: str, prompt: str) -> list[int]:
        return self._tokenizer_client.tokenize(request_id, prompt)

    def _prepare_prompt_segments(self, prompt: typing.Any) -> list[str]:
        if isinstance(prompt, list):
            return [segment if isinstance(segment, str) else str(segment) for segment in prompt]
        return [prompt if isinstance(prompt, str) else str(prompt)]

    def _filter_eos_tokens(self, tokens: list[int]) -> list[int]:
        """Remove eos tokens from a token list before decoding."""
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if eos_token_id is None:
            return tokens
        eos_ids = eos_token_id if isinstance(eos_token_id, (list, tuple, set)) else [eos_token_id]
        eos_set = {int(tid) for tid in eos_ids if tid is not None}
        if not eos_set:
            return tokens
        return [tok for tok in tokens if tok not in eos_set]

    def _tokenize_prompt_segments(self, prompt: typing.Any) -> list[list[int]]:
        segments = self._prepare_prompt_segments(prompt)
        token_segments: list[list[int]] = []
        for segment in segments:
            try:
                encoded = self.tokenizer(
                    segment,
                    add_special_tokens=False,
                    return_attention_mask=False,
                )
                ids = encoded.get("input_ids", [])
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
            except Exception:
                ids = []
            token_segments.append([int(tok) for tok in ids])
        return token_segments

    def _decode_with_pipeline(
        self,
        request_id: str,
        generated_tokens: list[int],
        *,
        finished: bool,
        skip_special_tokens: bool = True,
    ) -> DetokenizerResult:
        tokens_for_decode = self._filter_eos_tokens(generated_tokens)
        return self._detokenizer_client.decode(
            request_id,
            tokens_for_decode,
            finished=finished,
            skip_special_tokens=skip_special_tokens,
        )

    @staticmethod
    def _to_python_scalar(value: Any) -> Any:
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return value

    def _sanitize_metrics_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {k: self._to_python_scalar(v) for k, v in payload.items()}

    def _kv_cache_metadata(self) -> dict[str, Any]:
        metadata = getattr(getattr(self.runner, "executor_manager", None), "metadata", None)
        details: dict[str, Any] = {
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "page_size": self.page_size,
        }
        if metadata is not None:
            for attr in ("num_pages", "page_size", "max_model_length", "hbm_utilization"):
                value = getattr(metadata, attr, None)
                if value is not None:
                    details[attr] = self._to_python_scalar(value)
        return details

    def _record_cache_event(self, event: str, payload: dict[str, Any]) -> None:
        metrics_collector = get_metrics_collector()
        if metrics_collector:
            metrics_collector.record_cache_event(event, payload)

    def _log_cache_event(self, event: str, extra: dict[str, Any] | None = None) -> None:
        payload = self._kv_cache_metadata()
        if extra:
            payload.update(extra)
        sanitized = self._sanitize_metrics_payload(payload)
        self._info("KV cache %s: %s", event, sanitized)
        self._record_cache_event(event, sanitized)

    def _drain_pipeline_workers(self, reason: str) -> None:
        """Drain tokenizer/detokenizer workers with retry logic.

        Args:
            reason: Reason for draining (for logging).
        """
        manager = getattr(self, "_worker_manager", None)
        if not manager:
            return

        max_retries = WORKER_DRAIN_MAX_RETRIES
        retry_delay = WORKER_DRAIN_INITIAL_DELAY

        for attempt in range(max_retries):
            try:
                manager.drain_workers()
                self._info("Drained tokenizer/detokenizer workers (%s)", reason)
                return
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        "Failed to drain workers (attempt %d/%d): %s. Retrying in %.2fs...",
                        attempt + 1,
                        max_retries,
                        e,
                        retry_delay,
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(
                        "Failed to drain tokenizer/detokenizer workers after %d attempts during %s",
                        max_retries,
                        reason,
                        exc_info=True,
                    )

    def _clone_sampling_params(self, sampling_params: SamplingParams) -> SamplingParams:
        try:
            return copy.deepcopy(sampling_params)
        except Exception:
            logger.exception("Failed to clone sampling params; using original instance")
            return sampling_params

    def _prepare_sampling_params_for_request(
        self,
        template: SamplingParams,
        *,
        request_id: str,
        prompt: str,
    ) -> SamplingParams:
        params = self._clone_sampling_params(template)
        callback = self._sampling_params_callback
        if callback is None:
            return params

        metadata = {"request_id": request_id, "prompt": prompt, "engine": self}
        try:
            result = callback(params, metadata)
            if result is None:
                return params
            return result
        except Exception:
            logger.exception("Sampling params callback failed; falling back to unmodified parameters")
            return params

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
        request_id: str | list[str] | None = None,
        use_tqdm: bool = True,
    ) -> list[RequestOutput]:
        """Generate completions for one or more prompts (blocking).

        Synchronous batch generation that waits for all completions to finish
        before returning. Suitable for batch processing scenarios where you need
        all results at once.

        Args:
            prompts: Single prompt string or list of prompts to generate from.
            sampling_params: Generation parameters controlling temperature, top_p,
                max_tokens, etc. Defaults to SamplingParams(max_tokens=128) if None.
            request_id: Optional request ID(s) for tracking. Auto-generated if None.
                Can be a single string (for single prompt) or list of strings.
            use_tqdm: Show progress bar for batch generation. Useful for tracking
                progress with multiple prompts.

        Returns:
            List of RequestOutput objects containing:
                - Generated text in the `text` field
                - Token IDs in the `token_ids` field
                - Performance metrics (tokens/sec, latency, etc.)
                - Finish reason ('stop', 'length', 'eos_token')

        Raises:
            RuntimeError: If background scheduler is not running. Call initiate() first.
            ValueError: If prompts and request_ids have mismatched lengths.

        Example:
            >>> # Single prompt generation
            >>> outputs = engine.generate(
            ...     "What is AI?",
            ...     SamplingParams(max_tokens=100, temperature=0.7)
            ... )
            >>> print(outputs[0].get_text())
            >>>
            >>> # Batch generation with progress bar
            >>> prompts = ["Question 1?", "Question 2?", "Question 3?"]
            >>> outputs = engine.generate(prompts, use_tqdm=True)
            >>> for i, output in enumerate(outputs):
            ...     print(f"Prompt {i}: {output.get_text()[:50]}...")
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        if request_id is None:
            request_ids = [self._generate_request_id() for _ in prompts]
        elif isinstance(request_id, str):
            request_ids = [request_id]
        else:
            request_ids = request_id

        base_sampling_params = sampling_params or SamplingParams(max_tokens=128)

        for prompt, req_id in zip(prompts, request_ids, strict=False):
            prompt_tokens = self._tokenize_prompt(req_id, prompt)
            effective_params = self._prepare_sampling_params_for_request(
                base_sampling_params,
                request_id=req_id,
                prompt=prompt,
            )
            self._add_request(req_id, prompt, effective_params, prompt_token_ids=prompt_tokens)

        outputs = []
        pbar = None
        if use_tqdm:
            from tqdm import tqdm

            pbar = tqdm(total=len(prompts), desc="Generating")

        completed = set()

        if not self._scheduler_running:
            self._raise_if_scheduler_failed()
            raise RuntimeError("Background scheduler is not running. Call initiate() first.")

        while len(completed) < len(prompts):
            self._output_event.wait(timeout=0.1)
            self._output_event.clear()
            self._raise_if_scheduler_failed()
            with self._output_lock:
                for req_id in request_ids:
                    if req_id not in completed and req_id in self._request_outputs:
                        output = self._request_outputs[req_id]
                        if output.finished:
                            completed.add(req_id)
                            outputs.append(output)
                            if pbar:
                                pbar.update(1)

        if pbar:
            pbar.close()

        # Cleanup per-request events (outputs are preserved for post-hoc access)
        with self._request_lock:
            for output in outputs:
                rid = output.request_id
                self._request_events.pop(rid, None)
                n_samples = len(output.outputs)
                if n_samples > 1:
                    for sample_idx in range(n_samples):
                        self._request_events.pop(f"{rid}-{sample_idx}", None)
        return outputs

    def stream(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
    ) -> Iterator[RequestOutput]:
        """Stream generation output as tokens are produced.

        Yields RequestOutput objects incrementally as new tokens are generated,
        enabling real-time streaming of generated text. Perfect for interactive
        applications and chat interfaces.

        Args:
            prompts: Single prompt string or list with one prompt. For multiple
                prompts, use generate() instead.
            sampling_params: Generation parameters controlling temperature, top_p,
                max_tokens, etc. Defaults to SamplingParams(max_tokens=128).
            request_id: Optional request ID for tracking. Auto-generated if None.

        Yields:
            RequestOutput objects with incremental updates:
                - delta_text: Only the newly generated text since last yield
                - accumulated_text: Full text generated so far
                - finished: True when generation is complete
                - tokens_per_second: Current generation throughput
                - num_generated_tokens: Total tokens generated so far

        Raises:
            ValueError: If empty prompt list provided.
            RuntimeError: If scheduler not running or request setup fails.

        Example:
            >>> # Basic streaming
            >>> for output in engine.stream("Tell me a story"):
            ...     if output.delta_text:
            ...         print(output.delta_text, end="", flush=True)
            ...     if output.finished:
            ...         break
            >>>
            >>> # Monitor generation speed
            >>> for output in engine.stream("Long prompt here..."):
            ...     if output.delta_text:
            ...         print(output.delta_text, end="")
            ...     if output.num_generated_tokens % 10 == 0:
            ...         print(f"\n[{output.tokens_per_second:.1f} tok/s]", end="")
        """
        if isinstance(prompts, list):
            if len(prompts) == 0:
                raise ValueError("Empty prompt list provided")
            prompt = prompts[0]
        else:
            prompt = prompts

        if request_id is None:
            request_id = self._generate_request_id()

        base_sampling_params = sampling_params or SamplingParams(max_tokens=128)

        prompt_tokens = self._tokenize_prompt(request_id, prompt)
        effective_params = self._prepare_sampling_params_for_request(
            base_sampling_params,
            request_id=request_id,
            prompt=prompt,
        )
        self._add_request(request_id, prompt, effective_params, prompt_token_ids=prompt_tokens)

        if not self._scheduler_running:
            self._raise_if_scheduler_failed()
            raise RuntimeError("Background scheduler is not running. Call initiate() first.")

        with self._request_lock:
            req_event = self._request_events.get(request_id)
        if req_event is None:
            raise RuntimeError("Request event missing")

        last_update_seq = -1

        try:
            while True:
                req_event.wait(timeout=1.0)
                req_event.clear()
                self._raise_if_scheduler_failed()

                snapshot = None
                with self._output_lock:
                    ro = self._request_outputs.get(request_id)
                    if ro is None:
                        break

                    if ro.update_seq != last_update_seq:
                        # Snapshot without holding the lock during yield
                        outputs_copy = []
                        for comp in ro.outputs:
                            outputs_copy.append(
                                CompletionOutput(
                                    index=comp.index,
                                    text=comp.text,
                                    token_ids=list(comp.token_ids),
                                    cumulative_logprob=comp.cumulative_logprob,
                                    logprobs=[dict(lp) for lp in comp.logprobs] if comp.logprobs else None,
                                    finish_reason=comp.finish_reason,
                                )
                            )

                        snapshot = RequestOutput(
                            request_id=ro.request_id,
                            prompt=ro.prompt,
                            prompt_token_ids=list(ro.prompt_token_ids),
                            outputs=outputs_copy,
                            finished=ro.finished,
                            metrics=dict(ro.metrics) if ro.metrics is not None else None,
                            accumulated_text=ro.accumulated_text,
                            delta_text=ro.delta_text,
                            tokens_per_second=ro.tokens_per_second,
                            num_generated_tokens=ro.num_generated_tokens,
                            time_spent_generating=ro.time_spent_generating,
                            first_token_time=ro.first_token_time,
                            processing_time=ro.processing_time,
                            update_seq=ro.update_seq,
                        )
                        last_update_seq = ro.update_seq

                if snapshot is not None:
                    yield snapshot
                    if snapshot.finished:
                        break
        finally:
            with self._output_lock:
                ro = self._request_outputs.get(request_id)
                n_samples = len(ro.outputs) if ro is not None else 0
                finished = ro.finished if ro is not None else True
            if finished:
                with self._request_lock:
                    self._request_events.pop(request_id, None)
                    if n_samples > 1:
                        for sample_idx in range(n_samples):
                            self._request_events.pop(f"{request_id}-{sample_idx}", None)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        stream: bool = False,
        chat_template: str | None = None,
    ):
        """High-level chat interface compatible with vLLM and OpenAI APIs.

        Provides a convenient chat-based interface for conversational AI applications.
        Automatically formats messages using the model's chat template and handles
        both streaming and non-streaming responses. Supports multimodal content
        (images and videos) for vision-language models.

        Args:
            messages: List of message dictionaries representing the conversation history.
                Each message must have 'role' and 'content' keys. Content can be:
                - A string for text-only messages
                - A list of content items for multimodal messages (OpenAI format)

                Text-only example:
                    [{"role": "user", "content": "Hello!"}]

                Multimodal example:
                    [{"role": "user", "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": "Describe this image"}
                    ]}]

            tools: Optional list of tool/function definitions for function calling.
                Format should match the model's expected tool schema.
            sampling_params: Generation parameters controlling temperature, top_p,
                max_tokens, etc. Defaults to SamplingParams(max_tokens=128) if None.
            request_id: Optional unique identifier for tracking this request.
                Auto-generated if None.
            stream: If True, returns an iterator yielding incremental RequestOutput
                objects with delta_text for real-time streaming. If False, returns
                a single RequestOutput with the complete response.
            chat_template: Optional custom Jinja2 template to override the tokenizer's
                default chat template. Useful for models with non-standard formats.

        Returns:
            - If stream=False: Single RequestOutput object containing the complete
              assistant response with all metrics and generated text.
            - If stream=True: Iterator[RequestOutput] yielding incremental updates
              with delta_text containing newly generated text chunks.

        Raises:
            ValueError: If messages format is invalid or empty, or if multimodal
                content is provided but no processor is configured.
            RuntimeError: If scheduler is not running or tokenizer lacks chat template.

        Example:
            >>> # Text-only chat
            >>> messages = [
            ...     {"role": "system", "content": "You are a helpful assistant."},
            ...     {"role": "user", "content": "Explain quantum computing"}
            ... ]
            >>> response = engine.chat(messages)
            >>> print(response.get_text())
            >>>
            >>> # Multimodal chat with images (requires processor)
            >>> from PIL import Image
            >>> image = Image.open("photo.jpg")
            >>> messages = [
            ...     {"role": "user", "content": [
            ...         {"type": "image", "image": image},
            ...         {"type": "text", "text": "What's in this image?"}
            ...     ]}
            ... ]
            >>> response = engine.chat(messages)
            >>> print(response.get_text())
            >>>
            >>> # Streaming multimodal chat
            >>> for chunk in engine.chat(messages, stream=True):
            ...     print(chunk.delta_text, end="", flush=True)

        Note:
            For multimodal support, you must configure the engine with a processor
            during initialization: eSurge(..., processor=AutoProcessor.from_pretrained(...))
        """
        has_multimodal = self._messages_have_multimodal_content(messages)
        if has_multimodal:
            return self._chat_multimodal(
                messages=messages,
                tools=tools,
                sampling_params=sampling_params,
                request_id=request_id,
                stream=stream,
                chat_template=chat_template,
            )
        else:
            prompt = self._format_chat_prompt(
                messages,
                tools=tools,
                add_generation_prompt=True,
                chat_template=chat_template,
            )
            if stream:
                return self.stream(prompt, sampling_params=sampling_params, request_id=request_id)
            else:
                outs = self.generate(
                    prompt,
                    sampling_params=sampling_params,
                    request_id=request_id,
                    use_tqdm=False,
                )
                return outs[0]

    def _messages_have_multimodal_content(self, messages: list[dict]) -> bool:
        """Check if messages contain multimodal content (images/videos)."""
        for message in messages:
            content = message.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        item_type = item.get("type", "")
                        if item_type in ("image", "image_url", "input_image", "video", "video_url", "input_video"):
                            return True
                        if any(k in item for k in ("image", "image_url", "video", "video_url")):
                            return True
        return False

    def _chat_multimodal(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        stream: bool = False,
        chat_template: str | None = None,
    ):
        """Handle multimodal chat with images/videos."""
        if self._multimodal_manager is None:
            raise ValueError(
                "Multimodal content detected but no processor configured. "
                "Initialize eSurge with: processor=<tokenizer-or-processor> (e.g. AutoProcessor/AutoTokenizer)."
            )

        if request_id is None:
            request_id = self._generate_request_id()

        base_sampling_params = sampling_params or SamplingParams(max_tokens=128)

        images, videos = self._multimodal_manager.extract_media_from_messages(messages)

        pixel_values, image_grid_thw = self._multimodal_manager.process_images(images)
        pixel_values_videos, video_grid_thw = self._multimodal_manager.process_videos(videos)

        # Create mm_features for caching and batching support
        mm_features = []
        if images:
            mm_features.extend(self._multimodal_manager.process_images_to_features(images))
        if videos:
            mm_features.extend(self._multimodal_manager.process_videos_to_features(videos))

        prompt_token_ids = self._multimodal_manager.tokenize_multimodal(
            messages=messages,
            images=images,
            videos=videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )

        prompt = self._format_chat_prompt(
            messages,
            tools=tools,
            add_generation_prompt=True,
            chat_template=chat_template,
        )

        effective_params = self._prepare_sampling_params_for_request(
            base_sampling_params,
            request_id=request_id,
            prompt=prompt,
        )

        # Add request with vision data
        self._add_request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=effective_params,
            prompt_token_ids=prompt_token_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            mm_features=mm_features,
        )

        if not self._scheduler_running:
            self._raise_if_scheduler_failed()
            raise RuntimeError("Background scheduler is not running. Call initiate() first.")

        if stream:
            return self._stream_multimodal_request(request_id)
        else:
            return self._wait_for_request(request_id)

    def _stream_multimodal_request(self, request_id: str) -> Iterator[RequestOutput]:
        """Stream output for a multimodal request."""
        with self._request_lock:
            req_event = self._request_events.get(request_id)
        if req_event is None:
            raise RuntimeError("Request event missing")

        last_update_seq = -1

        try:
            while True:
                req_event.wait(timeout=1.0)
                req_event.clear()
                self._raise_if_scheduler_failed()

                snapshot = None
                with self._output_lock:
                    ro = self._request_outputs.get(request_id)
                    if ro is None:
                        break

                    if ro.update_seq != last_update_seq:
                        outputs_copy = []
                        for comp in ro.outputs:
                            outputs_copy.append(
                                CompletionOutput(
                                    index=comp.index,
                                    text=comp.text,
                                    token_ids=list(comp.token_ids),
                                    cumulative_logprob=comp.cumulative_logprob,
                                    logprobs=[dict(lp) for lp in comp.logprobs] if comp.logprobs else None,
                                    finish_reason=comp.finish_reason,
                                )
                            )

                        snapshot = RequestOutput(
                            request_id=ro.request_id,
                            prompt=ro.prompt,
                            prompt_token_ids=list(ro.prompt_token_ids),
                            outputs=outputs_copy,
                            finished=ro.finished,
                            metrics=dict(ro.metrics) if ro.metrics is not None else None,
                            accumulated_text=ro.accumulated_text,
                            delta_text=ro.delta_text,
                            tokens_per_second=ro.tokens_per_second,
                            num_generated_tokens=ro.num_generated_tokens,
                            time_spent_generating=ro.time_spent_generating,
                            first_token_time=ro.first_token_time,
                            processing_time=ro.processing_time,
                            update_seq=ro.update_seq,
                        )
                        last_update_seq = ro.update_seq

                if snapshot is not None:
                    yield snapshot
                    if snapshot.finished:
                        break
        finally:
            with self._output_lock:
                ro = self._request_outputs.get(request_id)
                n_samples = len(ro.outputs) if ro is not None else 0
                finished = ro.finished if ro is not None else True
            if finished:
                with self._request_lock:
                    self._request_events.pop(request_id, None)
                    if n_samples > 1:
                        for sample_idx in range(n_samples):
                            self._request_events.pop(f"{request_id}-{sample_idx}", None)

    def _wait_for_request(self, request_id: str) -> RequestOutput:
        """Wait for a request to complete and return the output."""
        with self._request_lock:
            req_event = self._request_events.get(request_id)
        if req_event is None:
            raise RuntimeError("Request event missing")

        output: RequestOutput | None = None
        while True:
            req_event.wait(timeout=1.0)
            req_event.clear()
            self._raise_if_scheduler_failed()
            with self._output_lock:
                output = self._request_outputs.get(request_id)
                if output is not None and output.finished:
                    break

        # Request is finished; cleanup per-request events (output is preserved)
        n_samples = len(output.outputs) if output is not None else 0
        with self._request_lock:
            self._request_events.pop(request_id, None)
            if n_samples > 1:
                for sample_idx in range(n_samples):
                    self._request_events.pop(f"{request_id}-{sample_idx}", None)

        assert output is not None
        return output

    def update_model_weights(
        self,
        model: EasyDeLBaseModule | None = None,
        *,
        graphdef=None,
        graphstate=None,
        graphother=None,
        restart_scheduler: bool = True,
    ) -> None:
        """Hot-swap the underlying model weights/graphs.

        The engine must be idle (no pending or running requests) before calling
        this method. It temporarily stops the scheduler loop, refreshes runner
        state, rebuilds the scheduler, and optionally restarts background serving.

        Args:
            model: Optional EasyDeLBaseModule carrying the new weights.
            graphdef: Optional graphdef override.
            graphstate: Optional graphstate override.
            graphother: Optional graphother override.
            restart_scheduler: Restart the scheduler thread if it was previously
                running (default: True).

        Raises:
            RuntimeError: If there are active or pending requests.
            ValueError: If no model/graph data is provided.
        """

        if self.num_running_requests > 0 or self.num_pending_requests > 0:
            raise RuntimeError("Cannot update model weights while requests are active or pending")

        if model is None and graphdef is None and graphstate is None and graphother is None:
            raise ValueError("No new model or graph components provided for update")

        was_running = self._scheduler_running
        if was_running:
            self.terminate()

        self._drain_pipeline_workers("update_model_weights")

        if model is None:
            model = flax.nnx.merge(graphdef, graphstate, graphother)
        if graphstate is None:
            graphstate = model.graphstate
        if graphother is None:
            graphother = model.graphother
        graphdef = model.esurge_graphdef

        self.runner.update_model_weights(
            graphdef=graphdef,
            graphstate=graphstate,
            graphother=graphother,
            reset_state=True,
        )
        self._kv_cache_valid = self.runner.executor_manager.kv_pages is not None
        cache_event = "kv_cache_reinitialized" if self._kv_cache_valid else "kv_cache_destroyed"
        self._log_cache_event(cache_event, {"reason": "update_model_weights"})

        with self._request_lock, self._output_lock:
            self._active_requests.clear()
            self._request_outputs.clear()
        self._request_events.clear()

        self.scheduler = Scheduler.from_runner(
            self.runner,
            max_num_batched_tokens=self._scheduler_max_num_batched_tokens,
            enable_prefix_caching=self._scheduler_enable_prefix_caching,
        )

        if restart_scheduler and was_running:
            self.initiate()

    def _add_request(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
        prompt_token_ids: list[int] | None = None,
        # Vision-language model data (optional)
        pixel_values: Any | None = None,
        image_grid_thw: Any | None = None,
        pixel_values_videos: Any | None = None,
        video_grid_thw: Any | None = None,
        mm_features: list | None = None,
    ) -> None:
        """Add a new request to the scheduler queue with intelligent context management.

        Internal method that tokenizes the prompt, applies context length management
        policies, creates request tracking structures, and adds the request to the
        scheduler for processing. Handles prompt truncation and token reservation
        to ensure generation fits within model constraints.

        Args:
            request_id: Unique identifier for the request.
            prompt: Text prompt to generate from. May be truncated based on
                context management settings.
            sampling_params: Generation parameters including max_tokens/max_new_tokens.

        Context Management:
            The method implements a sophisticated context management strategy:
            1. Calculates available token budget (max_model_len - reserve_tokens)
            2. If prompt exceeds budget:
               - Truncates based on truncate_mode (left/right/middle)
               - Or raises error if strict_context=True
            3. Adjusts max_new_tokens to fit within remaining context
            4. Prioritizes based on prefer_preserve_prompt setting

        Truncation Strategies:
            - "left": Removes tokens from beginning (keeps recent context)
            - "right": Removes tokens from end (keeps initial context)
            - "middle": Removes tokens from middle (keeps both ends)

        Note:
            This method ensures that prompt_len + max_new_tokens + reserve_tokens
            never exceeds max_model_len, preventing OOM errors during generation.
        """

        # ---- Config knobs ----
        max_model_len = int(self.runner.max_model_len)

        def _get_requested_new(sp):
            if hasattr(sp, "max_tokens") and sp.max_tokens is not None:
                return int(sp.max_tokens)
            if hasattr(sp, "max_new_tokens") and sp.max_new_tokens is not None:
                return int(sp.max_new_tokens)
            return None

        requested_new_raw = _get_requested_new(sampling_params)
        auto_infer_new_tokens = requested_new_raw is None
        requested_new = int(requested_new_raw) if requested_new_raw is not None else 0
        original_requested_new = requested_new if not auto_infer_new_tokens else -1

        token_ids_source = (
            prompt_token_ids if prompt_token_ids is not None else self._tokenize_prompt(request_id, prompt)
        )
        token_ids = list(token_ids_source)
        prompt_len = len(token_ids)

        max_prompt_budget = max(0, max_model_len - self.reserve_tokens)
        truncated = False
        tokens_dropped = 0

        if prompt_len > max_prompt_budget:
            if not self.auto_truncate_prompt and self.strict_context:
                raise ValueError(
                    f"Prompt too long: length={prompt_len} > budget={max_prompt_budget} "
                    f"(model_max={max_model_len}, reserve={self.reserve_tokens})."
                )
            new_tokens, dropped = truncate_tokens(token_ids, max_prompt_budget, self.truncate_mode)
            token_ids = new_tokens
            prompt_len = len(token_ids)
            truncated = dropped > 0
            tokens_dropped += dropped
            logger.warn(
                f"Truncated prompt by {dropped} tokens to fit model budget "
                f"(mode={self.truncate_mode}, new_len={prompt_len}, budget={max_prompt_budget})."
            )

        if auto_infer_new_tokens:
            requested_new = max(0, max_model_len - prompt_len - self.reserve_tokens)
            _set_requested_new(sampling_params, requested_new)
            logger.debug(
                "Auto-inferred max_tokens=%s for request %s (prompt_len=%s, reserve=%s, model_max=%s).",
                requested_new,
                request_id,
                prompt_len,
                self.reserve_tokens,
                max_model_len,
            )

        allowed_new_if_keep_prompt = max(0, max_model_len - prompt_len)

        if requested_new > allowed_new_if_keep_prompt:
            do_cap_first = self.prefer_preserve_prompt or not self.auto_truncate_prompt

            if do_cap_first:
                if self.auto_cap_new_tokens:
                    logger.warn(
                        f"Capping max_new_tokens from {requested_new} to {allowed_new_if_keep_prompt} "
                        f"to preserve prompt (prompt_len={prompt_len}, reserve={self.reserve_tokens}, "
                        f"model_max={max_model_len})."
                    )
                    requested_new = allowed_new_if_keep_prompt
                    _set_requested_new(sampling_params, requested_new)
                else:
                    if self.strict_context:
                        raise ValueError(
                            f"Requested max_new_tokens={requested_new} exceeds allowed={allowed_new_if_keep_prompt} "
                            f"for prompt_len={prompt_len}."
                        )
                    logger.warn(
                        f"auto_cap_new_tokens disabled but strict_context=False; "
                        f"capping new tokens to {allowed_new_if_keep_prompt}."
                    )
                    requested_new = allowed_new_if_keep_prompt
                    _set_requested_new(sampling_params, requested_new)
            else:
                target_prompt_budget = max(0, max_model_len - requested_new - self.reserve_tokens)
                if target_prompt_budget == 0 and requested_new > 0:
                    if self.auto_cap_new_tokens:
                        logger.warn(
                            f"Requested max_new_tokens={requested_new} leaves no room for prompt; "
                            f"capping to {allowed_new_if_keep_prompt} to preserve prompt."
                        )
                        requested_new = allowed_new_if_keep_prompt
                        _set_requested_new(sampling_params, requested_new)
                    else:
                        if self.strict_context:
                            raise ValueError("Requested output too large; would require dropping entire prompt.")
                        requested_new = allowed_new_if_keep_prompt
                        _set_requested_new(sampling_params, requested_new)
                else:
                    if prompt_len > target_prompt_budget:
                        new_tokens, dropped = truncate_tokens(token_ids, target_prompt_budget, self.truncate_mode)
                        token_ids = new_tokens
                        prompt_len = len(token_ids)
                        truncated = truncated or dropped > 0
                        tokens_dropped += dropped
                        self._info(
                            f"Truncated prompt by {dropped} tokens (mode={self.truncate_mode}) "
                            f"to honor requested max_new_tokens={requested_new}. "
                            f"New prompt_len={prompt_len}, target_prompt_budget={target_prompt_budget}."
                        )

        allowed_new_final = max(0, max_model_len - prompt_len - self.reserve_tokens)
        if requested_new > allowed_new_final:
            if self.strict_context and not self.auto_cap_new_tokens:
                raise ValueError(
                    f"After adjustments, requested_new={requested_new} still exceeds allowed={allowed_new_final}."
                )
            logger.warn(
                f"Final cap: max_new_tokens {requested_new} -> {allowed_new_final} "
                f"(prompt_len={prompt_len}, reserve={self.reserve_tokens}, model_max={max_model_len})."
            )
            requested_new = allowed_new_final
            _set_requested_new(sampling_params, requested_new)

        prompt_for_engine = prompt
        if truncated and self.decode_truncated_prompt:
            try:
                prompt_for_engine = self.tokenizer.decode(token_ids, skip_special_tokens=False)
            except Exception:
                prompt_for_engine = prompt
                logger.warn("Failed to decode truncated prompt; keeping original prompt text.")

        start_ts = time.perf_counter()
        ev = threading.Event()

        with self._request_lock:
            self._request_events[request_id] = ev
            self._active_requests[request_id] = {
                "prompt": prompt_for_engine,
                "prompt_token_ids": token_ids,
                "generated_tokens": [],
                "last_decoded_index": 0,
                "start_time": start_ts,
                "first_token_time": None,
                "last_decode_time": start_ts,
                "truncated": truncated,
                "tokens_dropped": tokens_dropped,
                "requested_new_tokens_original": original_requested_new,
                "requested_new_tokens_final": requested_new,
                "reserve_tokens": self.reserve_tokens,
                "max_model_len": max_model_len,
            }

        metrics_collector = get_metrics_collector()
        if metrics_collector:
            metrics_collector.start_request(request_id, len(token_ids))

        prompt_token_segments = self._tokenize_prompt_segments(prompt_for_engine)

        # Handle n > 1 sampling: create multiple EngineRequest objects
        n_samples = getattr(sampling_params, "n", 1) or 1

        # Create n CompletionOutput objects for the RequestOutput
        completion_outputs = [CompletionOutput(index=i, text="", token_ids=[]) for i in range(n_samples)]

        with self._output_lock:
            self._request_outputs[request_id] = RequestOutput(
                request_id=request_id,
                prompt=prompt_for_engine,
                prompt_token_ids=prompt_token_segments,
                outputs=completion_outputs,
                finished=False,
                accumulated_text="",
                delta_text="",
                tokens_per_second=0.0,
                num_generated_tokens=0,
                time_spent_generating=0.0,
                first_token_time=None,
                processing_time=0.0,
                update_seq=0,
                delta_seq=0,
            )

        # Prepare EOS token IDs: merge tokenizer default with extra_eos_token_ids
        eos_token_ids = []
        if self.tokenizer.eos_token_id is not None:
            if isinstance(self.tokenizer.eos_token_id, list):
                eos_token_ids.extend(self.tokenizer.eos_token_id)
            else:
                eos_token_ids.append(self.tokenizer.eos_token_id)
        eos_token_ids.extend(self.extra_eos_token_ids)

        # Use the first EOS token as the primary one for backwards compatibility
        primary_eos_token_id = eos_token_ids[0] if eos_token_ids else None

        # Add all EOS tokens to sampling_params.stop_token_ids if not already present
        if eos_token_ids:
            current_stop_ids = set(sampling_params.stop_token_ids)
            for eos_id in eos_token_ids:
                if eos_id not in current_stop_ids:
                    sampling_params.stop_token_ids.append(eos_id)
                    sampling_params._all_stop_token_ids.add(eos_id)

        # Create n EngineRequest objects for parallel sampling
        mm_features_cache_key_only = None
        if mm_features and n_samples > 1:
            mm_features_cache_key_only = []
            for feat in mm_features:
                try:
                    mm_features_cache_key_only.append(
                        type(feat)(
                            mm_hash=getattr(feat, "mm_hash", ""),
                            modality=getattr(feat, "modality", "image"),
                            pixel_values=None,
                            grid_thw=None,
                        )
                    )
                except Exception:
                    # Worst-case fallback: keep the original feature object.
                    # This preserves correctness at the cost of extra memory.
                    mm_features_cache_key_only.append(feat)

        for sample_idx in range(n_samples):
            if n_samples == 1:
                # For n=1, use the original request_id
                child_request_id = request_id
                parent_id = None
            else:
                # For n>1, create child request IDs
                child_request_id = f"{request_id}-{sample_idx}"
                parent_id = request_id

                # Create tracking entries for child requests
                # IMPORTANT: Create a fresh dict for each sample to avoid sharing mutable objects
                with self._request_lock:
                    self._request_events[child_request_id] = self._request_events[request_id]
                    self._active_requests[child_request_id] = {
                        "prompt": prompt_for_engine,
                        "prompt_token_ids": token_ids,
                        "generated_tokens": [],  # Fresh list for each sample
                        "last_decoded_index": 0,
                        "start_time": start_ts,
                        "first_token_time": None,
                        "last_decode_time": start_ts,
                        "truncated": truncated,
                        "tokens_dropped": tokens_dropped,
                        "requested_new_tokens_original": original_requested_new,
                        "requested_new_tokens_final": requested_new,
                        "reserve_tokens": self.reserve_tokens,
                        "max_model_len": max_model_len,
                        "sample_index": sample_idx,
                        "parent_request_id": request_id,
                    }

            with self._scheduler_lock:
                self.scheduler.add_request(
                    EngineRequest(
                        request_id=child_request_id,
                        prompt_token_ids=token_ids,
                        sampling_params=sampling_params,
                        eos_token_id=primary_eos_token_id,
                        parent_request_id=parent_id,
                        sample_index=sample_idx,
                        # Vision-language model data (only for first sample to save memory)
                        pixel_values=pixel_values if sample_idx == 0 else None,
                        image_grid_thw=image_grid_thw if sample_idx == 0 else None,
                        pixel_values_videos=pixel_values_videos if sample_idx == 0 else None,
                        video_grid_thw=video_grid_thw if sample_idx == 0 else None,
                        # Keep multimodal cache keys for all samples so n>1 can share
                        # KV-prefix pages via prefix caching without duplicating pixel buffers.
                        mm_features=mm_features if sample_idx == 0 else mm_features_cache_key_only,
                    )
                )

        self._info(
            f"Queued request {request_id}: prompt_len={prompt_len}, "
            f"max_tokens={requested_new}, n={n_samples}, reserve={self.reserve_tokens}, "
            f"model_max={max_model_len}, dropped={tokens_dropped}"
        )

    def _generate_request_id(self) -> str:
        """Generate a unique request ID with overflow protection.

        Uses UUID for uniqueness and a counter for ordering. The counter
        uses modulo arithmetic to prevent unbounded growth in long-running
        services.

        Returns:
            Unique request ID with format 'req-{uuid}-{counter}'.
        """
        with self._counter_lock:
            self._request_counter = (self._request_counter + 1) % (1 << 32)  # Reset after ~4 billion requests
            return f"req-{uuid.uuid4().hex}-{self._request_counter}"

    def abort_request(self, request_id: str) -> None:
        """Abort an in-progress request.

        Marks the request as aborted and signals any waiting threads.
        The request will be removed from the scheduler queue if still waiting.

        Args:
            request_id: ID of the request to abort.
        """
        detokenizer_reset_ids: set[str] = set()
        parent_request_id = request_id
        sample_index = 0
        metrics_collector = get_metrics_collector()

        # Acquire all locks atomically to prevent race conditions
        with self._scheduler_lock, self._request_lock, self._output_lock:
            rd = self._active_requests.get(request_id)
            if rd is not None:
                parent_request_id = rd.get("parent_request_id", request_id)
                sample_index = int(rd.get("sample_index", 0) or 0)

            # Resolve scheduler-side IDs to abort (n=1: request_id; n>1: children of parent)
            abort_ids: set[str] = set()
            if request_id in self.scheduler.requests:
                abort_ids.add(request_id)
                parent_request_id = self.scheduler.requests[request_id].parent_request_id or parent_request_id
            abort_ids.update(
                rid
                for rid, req in self.scheduler.requests.items()
                if getattr(req, "parent_request_id", None) == request_id
            )

            if abort_ids:
                self.scheduler.finish_requests(abort_ids, EngineRequestStatus.FINISHED_ABORTED)
                detokenizer_reset_ids |= abort_ids

            # Clean up active request tracking (outputs are preserved)
            for rid in abort_ids:
                self._active_requests.pop(rid, None)
            self._active_requests.pop(parent_request_id, None)

            # Update output state
            ro = self._request_outputs.get(parent_request_id)
            if ro is not None:
                if request_id == parent_request_id:
                    ro.finished = True
                    for output in ro.outputs:
                        output.finish_reason = "abort"
                    if metrics_collector:
                        metrics_collector.complete_request(parent_request_id, finish_reason="abort")
                else:
                    if 0 <= sample_index < len(ro.outputs):
                        ro.outputs[sample_index].finish_reason = "abort"
                    ro.finished = all(output.finish_reason is not None for output in ro.outputs)
                    if ro.finished and metrics_collector:
                        metrics_collector.complete_request(parent_request_id, finish_reason="abort")
                ro.update_seq += 1

            # Get event while still holding lock (streaming uses parent event)
            ev = self._request_events.get(parent_request_id)

            if not detokenizer_reset_ids:
                detokenizer_reset_ids.add(request_id)

        # Reset detokenizer state (outside locks to avoid blocking)
        for rid in detokenizer_reset_ids:
            try:
                self._detokenizer_client.reset(rid)
                # Remove from failed set if it was there
                self._failed_detokenizer_resets.discard(rid)
            except Exception:
                logger.debug("Failed to reset detokenizer state for %s", rid, exc_info=True)
                # Track failed reset
                self._failed_detokenizer_resets.add(rid)

        # Trigger cleanup if threshold reached
        if len(self._failed_detokenizer_resets) >= self._detokenizer_cleanup_threshold:
            self._cleanup_detokenizer_state()

        # Notify waiters
        if ev:
            ev.set()
        self._output_event.set()
        log_metrics_summary()

    def _cleanup_detokenizer_state(self) -> None:
        """Attempt to clean up failed detokenizer states.

        Retries resetting detokenizer state for all tracked failed requests.
        Clears successfully reset requests from the tracking set.
        """
        if not self._failed_detokenizer_resets:
            return

        self._info(
            "Attempting to clean up %d failed detokenizer states",
            len(self._failed_detokenizer_resets),
        )

        successfully_reset = set()
        for request_id in list(self._failed_detokenizer_resets):
            try:
                self._detokenizer_client.reset(request_id)
                successfully_reset.add(request_id)
            except Exception:
                # Still failing, keep in set
                pass

        # Remove successfully reset requests
        self._failed_detokenizer_resets -= successfully_reset

        if successfully_reset:
            self._info("Successfully cleaned up %d detokenizer states", len(successfully_reset))
        if self._failed_detokenizer_resets:
            logger.warning(
                "%d detokenizer states still failed to reset",
                len(self._failed_detokenizer_resets),
            )

    @property
    def num_pending_requests(self) -> int:
        """Get the number of requests waiting in queue.

        Returns:
            Count of requests in the waiting queue.
        """
        with self._scheduler_lock:
            return len(self.scheduler.waiting)

    @property
    def num_running_requests(self) -> int:
        """Get the number of actively running requests.

        Returns:
            Count of requests currently being processed.
        """
        with self._scheduler_lock:
            return len(self.scheduler.running)

    def _reset_runner_state_if_idle(self, reason: str) -> None:
        """Reset runner buffers when there are no active/pending requests."""
        if not hasattr(self.runner, "reset_state"):
            return
        if self.num_running_requests > 0 or self.num_pending_requests > 0:
            logger.warning(
                "Skipping runner state reset during %s because there are active or pending requests "
                "(running=%d, pending=%d)",
                reason,
                self.num_running_requests,
                self.num_pending_requests,
            )
            return
        try:
            self.runner.reset_state()
            self._info("Runner state reset (%s)", reason)
        except Exception:
            logger.debug("Runner state reset encountered an error during %s", reason, exc_info=True)

    def start_profiling(
        self,
        output_dir: str,
        num_batches: int = 10,
        host_tracer_level: int | None = None,
        python_tracer_level: int | None = None,
    ) -> None:
        """Start a JAX profiler trace for the next ``num_batches`` scheduler updates."""
        if self._profiling_active:
            raise RuntimeError("A profiling session is already active")
        if num_batches <= 0:
            raise ValueError("num_batches must be positive")

        profiler_options = jax.profiler.ProfileOptions()
        if host_tracer_level is not None:
            profiler_options.host_tracer_level = host_tracer_level
        if python_tracer_level is not None:
            profiler_options.python_tracer_level = python_tracer_level

        jax.profiler.start_trace(output_dir, profiler_options=profiler_options)
        self._profiling_active = True
        self._profiling_steps_remaining = num_batches
        self._profiling_output_dir = output_dir
        self._profiling_host_level = host_tracer_level
        self._profiling_python_level = python_tracer_level
        self._info(
            "Started profiler trace -> %s (batches=%d, host_tracer_level=%s, python_tracer_level=%s)",
            output_dir,
            num_batches,
            host_tracer_level,
            python_tracer_level,
        )

    def stop_profiling(self) -> None:
        """Stop the active JAX profiler trace, if any."""
        if not self._profiling_active:
            return
        try:
            jax.profiler.stop_trace()
            self._info("Stopped profiler trace -> %s", self._profiling_output_dir)
        finally:
            self._profiling_active = False
            self._profiling_steps_remaining = 0
            self._profiling_output_dir = None
            self._profiling_host_level = None
            self._profiling_python_level = None

    def _drain_runner_future(self, future, scheduler_output: SchedulerOutput) -> None:
        model_output = self.runner.wait_for_execution(future)
        with self._scheduler_lock:
            engine_outputs = self.scheduler.update_from_output(scheduler_output, model_output)
        if engine_outputs:
            self._process_engine_outputs(engine_outputs)
        self._handle_profiling_step()

    def _handle_profiling_step(self) -> None:
        if not self._profiling_active:
            return
        if self._profiling_steps_remaining > 0:
            self._profiling_steps_remaining -= 1
        if self._profiling_steps_remaining <= 0:
            self.stop_profiling()

    def _process_engine_outputs(self, engine_outputs: dict[int, EngineCoreOutputs]) -> None:
        """Process engine outputs and update request outputs (thread-safe).

        Core method that processes tokens from the model, performs incremental
        decoding, updates metrics, and signals waiting threads. Uses interval-based
        decoding to reduce tokenizer overhead during streaming.

        Args:
            engine_outputs: Dictionary mapping client IDs to engine outputs containing
                new tokens, completion status, and metadata.

        Processing Flow:
            1. Extracts new tokens from engine outputs
            2. Performs interval-based decoding (every 4 tokens or 20ms)
            3. Updates accumulated and delta text fields
            4. Tracks performance metrics (TTFT, tokens/sec)
            5. Handles request completion with final token flush
            6. Signals per-request events for streaming consumers

        Thread Safety:
            Uses request_lock and output_lock to ensure atomic updates across
            multiple concurrent requests and streaming consumers.
        """
        metrics_collector = get_metrics_collector()

        # Update both request_data and public outputs atomically
        with self._request_lock, self._output_lock:
            for client_outputs in engine_outputs.values():
                for engine_output in client_outputs.outputs:
                    request_id = engine_output.request_id
                    rd = self._active_requests.get(request_id)
                    if rd is None:
                        continue

                    # Handle n>1 sampling: get parent request and sample index
                    parent_request_id = rd.get("parent_request_id", request_id)
                    sample_index = rd.get("sample_index", 0)
                    ro = self._request_outputs.get(parent_request_id)
                    if ro is None:
                        continue

                    text_changed = False
                    new_tokens = engine_output.new_token_ids
                    now = time.perf_counter()
                    elapsed = now - rd["start_time"]
                    if new_tokens:
                        rd["generated_tokens"].extend(new_tokens)
                        num_generated = len(rd["generated_tokens"])
                        decodable_tokens = self._filter_eos_tokens(rd["generated_tokens"])
                        num_decodable = len(decodable_tokens)

                        if rd["first_token_time"] is None and num_generated > 0:
                            rd["first_token_time"] = now - rd["start_time"]
                            if metrics_collector and ro.first_token_time is None:
                                metrics_collector.record_first_token(parent_request_id)

                        if metrics_collector:
                            metrics_collector.add_generated_tokens(parent_request_id, len(new_tokens))

                        last_idx = rd["last_decoded_index"]
                        should_decode = (
                            num_decodable - last_idx >= self.decode_interval_tokens
                            or (now - rd.get("last_decode_time", now)) >= self.decode_interval_secs
                        )
                        if should_decode and num_decodable > last_idx:
                            pipeline_result = self._decode_with_pipeline(
                                request_id,
                                decodable_tokens,
                                finished=False,
                            )
                            rd["last_decoded_index"] = pipeline_result.last_decoded_index
                            rd["last_decode_time"] = now

                            # Update the specific sample's completion output
                            comp = ro.outputs[sample_index]
                            comp.text = pipeline_result.accumulated_text
                            comp.token_ids = list(rd["generated_tokens"])

                            # For backwards compatibility, set ro.accumulated_text to first sample's text
                            if sample_index == 0:
                                ro.accumulated_text = pipeline_result.accumulated_text
                                ro.delta_text = pipeline_result.delta_text

                            if pipeline_result.delta_text:
                                ro.delta_seq += 1
                                text_changed = True

                        ro.num_generated_tokens = len(rd["generated_tokens"])

                        elapsed = now - rd["start_time"]
                        ro.processing_time = elapsed
                        ro.time_spent_generating = elapsed
                        ro.first_token_time = rd["first_token_time"]

                        if rd["first_token_time"] is not None and num_generated > 0:
                            generation_time = elapsed - rd["first_token_time"]
                            ro.tokens_per_second = num_generated / generation_time if generation_time > 0 else 0.0
                        else:
                            ro.tokens_per_second = 0.0

                        ro.num_generated_tokens = num_generated

                    if engine_output.finished:
                        comp = ro.outputs[sample_index]
                        comp.finish_reason = (
                            str(engine_output.finish_reason) if engine_output.finish_reason else "finished"
                        )

                        # For n>1, mark RequestOutput as finished only when ALL samples are done
                        n_samples = len(ro.outputs)
                        if n_samples == 1:
                            ro.finished = True
                        else:
                            # Check if all samples have finish_reason set
                            all_finished = all(output.finish_reason is not None for output in ro.outputs)
                            ro.finished = all_finished

                        num_generated = len(rd["generated_tokens"])
                        decodable_tokens = self._filter_eos_tokens(rd["generated_tokens"])
                        num_decodable = len(decodable_tokens)
                        last_idx = rd["last_decoded_index"]
                        if num_decodable > last_idx:
                            pipeline_result = self._decode_with_pipeline(
                                request_id,
                                decodable_tokens,
                                finished=True,
                            )
                            rd["last_decoded_index"] = pipeline_result.last_decoded_index

                            # Update the specific sample's completion output
                            comp.text = pipeline_result.accumulated_text
                            comp.token_ids = list(rd["generated_tokens"])

                            # For backwards compatibility, set ro.accumulated_text to first sample's text
                            if sample_index == 0:
                                ro.accumulated_text = pipeline_result.accumulated_text
                                ro.delta_text = pipeline_result.delta_text

                            if pipeline_result.delta_text:
                                ro.delta_seq += 1
                                text_changed = True

                        num_prompt_tokens = (
                            len(rd["prompt_token_ids"])
                            if "prompt_token_ids" in rd
                            else sum(len(seg) for seg in ro.prompt_token_ids)
                        )
                        num_generated_tokens = len(rd["generated_tokens"])

                        ro.processing_time = elapsed
                        ro.time_spent_generating = elapsed
                        ro.num_generated_tokens = num_generated_tokens
                        ro.first_token_time = rd.get("first_token_time")

                        if ro.first_token_time is not None and num_generated_tokens > 0:
                            generation_time = elapsed - ro.first_token_time
                            ro.tokens_per_second = num_generated_tokens / generation_time if generation_time > 0 else 0.0
                        else:
                            ro.tokens_per_second = 0.0

                        ro.metrics = {
                            "prompt_tokens": num_prompt_tokens,
                            "generated_tokens": num_generated_tokens,
                            "total_tokens": num_prompt_tokens + num_generated_tokens,
                            "processing_time": elapsed,
                            "first_token_time": ro.first_token_time,
                            "tokens_per_second": ro.tokens_per_second,
                        }

                        if metrics_collector and ro.finished:
                            finish_reason = comp.finish_reason
                            if any(out.finish_reason == "abort" for out in ro.outputs):
                                finish_reason = "abort"
                            elif any(out.finish_reason == "length" for out in ro.outputs):
                                finish_reason = "length"
                            elif any(out.finish_reason == "stop" for out in ro.outputs):
                                finish_reason = "stop"
                            metrics_collector.complete_request(
                                parent_request_id,
                                finish_reason=finish_reason,
                            )
                        try:
                            self._detokenizer_client.reset(request_id)
                        except Exception:
                            logger.debug("Failed to reset detokenizer state for %s", request_id, exc_info=True)
                        self._active_requests.pop(request_id, None)
                        if ro.finished and parent_request_id != request_id:
                            self._active_requests.pop(parent_request_id, None)
                    ro.update_seq += 1
                    if text_changed or engine_output.finished:
                        # Signal the parent request event
                        ev = self._request_events.get(parent_request_id)
                        if ev:
                            ev.set()

        self._output_event.set()

    def _prepare_grafana_provisioning(
        self,
        datasource_name: str,
        datasource_uid: str,
        datasource_url: str,
    ) -> str:
        """Create temporary provisioning config for Grafana."""
        provisioning_root = tempfile.mkdtemp(prefix="esurge_grafana_")
        datasources_dir = os.path.join(provisioning_root, "datasources")
        dashboards_dir = os.path.join(provisioning_root, "dashboards")
        os.makedirs(datasources_dir, exist_ok=True)
        os.makedirs(dashboards_dir, exist_ok=True)

        datasource_config = f"""apiVersion: 1
datasources:
  - name: "{datasource_name}"
    uid: "{datasource_uid}"
    type: prometheus
    access: proxy
    url: "{datasource_url}"
    isDefault: true
    editable: true
    jsonData:
      timeInterval: "1s"
"""
        with open(os.path.join(datasources_dir, "esurge-prometheus.yaml"), "w", encoding="utf-8") as f:
            f.write(datasource_config)

        provider_config = """apiVersion: 1
providers:
  - name: "esurge-autoprovisioned"
    type: file
    disableDeletion: false
    updateIntervalSeconds: 30
    options:
      path: /etc/grafana/provisioning/dashboards
"""
        with open(os.path.join(dashboards_dir, "provider.yaml"), "w", encoding="utf-8") as f:
            f.write(provider_config)

        return provisioning_root

    def _start_local_grafana_service(
        self,
        provisioning_root: str,
        grafana_host: str | None,
        grafana_port: int,
        grafana_admin_user: str,
        grafana_admin_password: str,
        allow_anonymous: bool,
    ) -> str | None:
        """Start Grafana using a locally installed grafana-server binary."""
        if self._grafana_process:
            return self._grafana_url

        grafana_exe = shutil.which("grafana-server")
        if not grafana_exe:
            return None

        data_dir = os.path.join(provisioning_root, "data")
        plugins_dir = os.path.join(provisioning_root, "plugins")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(plugins_dir, exist_ok=True)

        env = os.environ.copy()
        env.update(
            {
                "GF_PATHS_PROVISIONING": provisioning_root,
                "GF_PATHS_DATA": data_dir,
                "GF_PATHS_PLUGINS": plugins_dir,
                "GF_SECURITY_ADMIN_USER": grafana_admin_user,
                "GF_SECURITY_ADMIN_PASSWORD": grafana_admin_password,
                "GF_SERVER_HTTP_PORT": str(grafana_port),
            }
        )
        if allow_anonymous:
            env["GF_AUTH_ANONYMOUS_ENABLED"] = "true"
            env["GF_AUTH_ANONYMOUS_ORG_ROLE"] = "Admin"

        possible_homepaths = ["/usr/share/grafana", "/usr/local/share/grafana"]
        cmd = [grafana_exe, "server"]
        homepath = next((p for p in possible_homepaths if os.path.isdir(p)), None)
        if homepath:
            cmd.extend(["--homepath", homepath])

        try:
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            self._info(" Failed to start local Grafana server: %s", exc)
            return None

        self._grafana_process = proc
        self._grafana_temp_dir = provisioning_root
        self._grafana_url = f"http://{grafana_host or 'localhost'}:{grafana_port}"
        self._info(" Grafana started (local binary) at %s", self._grafana_url)
        return self._grafana_url

    def _start_docker_grafana_service(
        self,
        provisioning_root: str,
        grafana_host: str | None,
        grafana_port: int,
        grafana_image: str,
        grafana_admin_user: str,
        grafana_admin_password: str,
        allow_anonymous: bool,
        datasource_url: str,
    ) -> str | None:
        """Start Grafana using Docker."""
        if self._grafana_container_name:
            return self._grafana_url

        docker_exe = shutil.which("docker")
        if not docker_exe:
            return None

        container_name = f"esurge-grafana-{uuid.uuid4().hex[:8]}"
        cmd = [
            docker_exe,
            "run",
            "--rm",
            "-d",
            "--name",
            container_name,
            "-p",
            f"{grafana_port}:3000",
            "-v",
            f"{provisioning_root}:/etc/grafana/provisioning",
            "--add-host",
            "host.docker.internal:host-gateway",
            "-e",
            f"GF_SECURITY_ADMIN_USER={grafana_admin_user}",
            "-e",
            f"GF_SECURITY_ADMIN_PASSWORD={grafana_admin_password}",
        ]
        if allow_anonymous:
            cmd.extend(["-e", "GF_AUTH_ANONYMOUS_ENABLED=true", "-e", "GF_AUTH_ANONYMOUS_ORG_ROLE=Admin"])
        cmd.append(grafana_image)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            err_output = exc.stderr.strip() if exc.stderr else str(exc)
            self._info(" Failed to start Grafana automatically: %s", err_output)
            return None
        except Exception as exc:
            self._info(" Failed to start Grafana automatically: %s", exc)
            return None

        self._grafana_container_name = container_name
        self._grafana_container_id = result.stdout.strip() or container_name
        self._grafana_temp_dir = provisioning_root
        self._grafana_url = f"http://{grafana_host or 'localhost'}:{grafana_port}"
        self._info(" Grafana started (Docker) at %s (datasource -> %s)", self._grafana_url, datasource_url)
        return self._grafana_url

    def _start_grafana_service(
        self,
        prometheus_url: str | None,
        grafana_host: str | None,
        grafana_port: int,
        grafana_image: str,
        grafana_admin_user: str,
        grafana_admin_password: str,
        allow_anonymous: bool,
        datasource_name: str,
        datasource_uid: str | None,
        datasource_url: str | None,
        use_docker: bool,
    ) -> str | None:
        """Attempt to launch Grafana wired to the Prometheus endpoint."""
        if self._grafana_container_name or self._grafana_process:
            return self._grafana_url

        if not prometheus_url:
            self._info(" Grafana autostart skipped: Prometheus URL unavailable")
            return None

        datasource_uid = datasource_uid or "esurge-prometheus"
        datasource_url = datasource_url or prometheus_url
        docker_datasource_url = (
            datasource_url.replace("0.0.0.0", "host.docker.internal")
            .replace("localhost", "host.docker.internal")
            .replace("127.0.0.1", "host.docker.internal")
        )

        provisioning_root = self._prepare_grafana_provisioning(
            datasource_name=datasource_name,
            datasource_uid=datasource_uid,
            datasource_url=docker_datasource_url if use_docker else datasource_url,
        )

        # Try local grafana-server first
        local_url = self._start_local_grafana_service(
            provisioning_root=provisioning_root,
            grafana_host=grafana_host,
            grafana_port=grafana_port,
            grafana_admin_user=grafana_admin_user,
            grafana_admin_password=grafana_admin_password,
            allow_anonymous=allow_anonymous,
        )
        if local_url:
            return local_url

        if not use_docker:
            shutil.rmtree(provisioning_root, ignore_errors=True)
            self._info(" Grafana autostart skipped: local server unavailable and Docker disabled")
            return None

        docker_url = self._start_docker_grafana_service(
            provisioning_root=provisioning_root,
            grafana_host=grafana_host,
            grafana_port=grafana_port,
            grafana_image=grafana_image,
            grafana_admin_user=grafana_admin_user,
            grafana_admin_password=grafana_admin_password,
            allow_anonymous=allow_anonymous,
            datasource_url=docker_datasource_url,
        )
        if docker_url:
            return docker_url

        shutil.rmtree(provisioning_root, ignore_errors=True)
        return None

    def _stop_grafana_service(self) -> None:
        """Stop the Grafana container if it was started by the engine."""
        container = self._grafana_container_id or self._grafana_container_name
        docker_exe = shutil.which("docker") if container else None
        if container and docker_exe:
            try:
                subprocess.run(
                    [docker_exe, "stop", container],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
                self._info(" Grafana container stopped")
            except Exception:
                self._info(" Failed to stop Grafana container %s", container)

        if self._grafana_temp_dir:
            shutil.rmtree(self._grafana_temp_dir, ignore_errors=True)

        self._grafana_container_name = None
        self._grafana_container_id = None
        if self._grafana_process:
            try:
                self._grafana_process.terminate()
                self._grafana_process.wait(timeout=2)
                self._info(" Grafana process stopped")
            except Exception:
                self._info(" Failed to stop Grafana process gracefully")
            self._grafana_process = None
        self._grafana_temp_dir = None
        self._grafana_url = None

    def start_monitoring(
        self,
        dashboard_port: int | None = None,
        prometheus_port: int = 11184,
        dashboard_host: str | None = None,
        enable_prometheus: bool = True,
        enable_dashboard: bool | None = None,
        enable_console: bool = False,
        log_file: str | None = None,
        log_interval: float = 10.0,
        history_size: int = 1000,
        enable_detailed_logging: bool = True,
        start_grafana: bool = True,
        grafana_port: int = 3000,
        grafana_host: str | None = None,
        grafana_image: str = "grafana/grafana-oss:latest",
        grafana_use_docker: bool = False,
        grafana_admin_user: str = "admin",
        grafana_admin_password: str = "admin",
        grafana_allow_anonymous: bool = True,
        grafana_datasource_name: str = "eSurge Prometheus",
        grafana_datasource_uid: str | None = None,
        grafana_datasource_url: str | None = None,
    ) -> dict[str, str]:
        """Start Prometheus-based monitoring for the engine.

        Initializes the Prometheus metrics exporter, optional console monitor,
        and (by default) tries to auto-start a Grafana instance with a
        pre-provisioned Prometheus data source (local grafana-server first,
        optionally Docker if enabled).

        Args:
            dashboard_port: Deprecated; no longer used (kept for compatibility).
            prometheus_port: Port for Prometheus metrics endpoint.
            dashboard_host: Deprecated; no longer used (kept for compatibility).
            enable_prometheus: Start Prometheus metrics server.
            enable_dashboard: Deprecated; no longer used (kept for compatibility).
            enable_console: Start console monitor with rich display.
            log_file: Optional file path for metrics logging.
            log_interval: Interval in seconds between metric logs.
            history_size: Number of historical metrics to retain.
            enable_detailed_logging: Enable detailed metric logging.
            start_grafana: Auto-start Grafana (via Docker) pointed at the Prometheus endpoint.
            grafana_port: Host port to expose Grafana.
            grafana_host: Hostname to use when reporting Grafana URL (defaults to localhost).
            grafana_image: Docker image for Grafana (used when grafana_use_docker=True).
            grafana_use_docker: Allow falling back to Docker for Grafana when local server is unavailable.
            grafana_admin_user: Admin username for Grafana.
            grafana_admin_password: Admin password for Grafana.
            grafana_allow_anonymous: Allow anonymous admin access to Grafana (for quick local use).
            grafana_datasource_name: Display name for the auto-provisioned Prometheus data source.
            grafana_datasource_uid: UID for the Prometheus data source (auto-generated if None).
            grafana_datasource_url: Override URL for the Prometheus data source inside Grafana.

        Returns:
            Dictionary of service URLs:
            - 'prometheus': Prometheus metrics endpoint
            - 'grafana': Grafana UI (when auto-start succeeds)
        """
        if self._monitoring_initialized:
            if start_grafana and not self._grafana_container_name:
                existing_urls = self._monitoring_urls or {}
                prometheus_url = existing_urls.get("prometheus")
                grafana_url = self._start_grafana_service(
                    prometheus_url=prometheus_url,
                    grafana_host=grafana_host or dashboard_host,
                    grafana_port=grafana_port,
                    grafana_image=grafana_image,
                    grafana_admin_user=grafana_admin_user,
                    grafana_admin_password=grafana_admin_password,
                    allow_anonymous=grafana_allow_anonymous,
                    datasource_name=grafana_datasource_name,
                    datasource_uid=grafana_datasource_uid,
                    datasource_url=grafana_datasource_url,
                    use_docker=grafana_use_docker,
                )
                if grafana_url:
                    existing_urls["grafana"] = grafana_url
                    self._monitoring_urls = existing_urls
            self._info("Monitoring already initialized for this eSurge instance")
            return self._monitoring_urls or {}

        self._info("Starting eSurge monitoring services (Prometheus exporter)...")

        if not get_metrics_collector():
            initialize_metrics(
                log_file=log_file,
                log_interval=log_interval,
                history_size=history_size,
                enable_detailed_logging=enable_detailed_logging,
            )
            self._info(" Metrics collection initialized")

        urls: dict[str, str] = {}

        if enable_prometheus:
            try:
                from .monitoring import start_monitoring_server

                self._monitoring_server = start_monitoring_server(prometheus_port=prometheus_port, update_interval=1.0)
                host_for_logs = dashboard_host or "0.0.0.0"
                urls["prometheus"] = f"http://{host_for_logs}:{prometheus_port}/metrics"
                self._info(f" Prometheus metrics: {urls['prometheus']}")
                self._info(" Point Grafana at the Prometheus endpoint to visualize eSurge metrics.")
            except ImportError:
                self._info(" Prometheus monitoring unavailable (install prometheus-client)")
            except Exception as e:
                self._info(f" Failed to start Prometheus server: {e}")
        elif start_grafana:
            self._info(" Grafana autostart skipped because Prometheus exporter is disabled")

        if enable_dashboard or dashboard_port or dashboard_host:
            self._info(
                " The built-in web dashboard has been removed. "
                "Use Prometheus + Grafana (or another Prometheus UI) for charts."
            )

        if start_grafana and enable_prometheus:
            grafana_url = self._start_grafana_service(
                prometheus_url=urls.get("prometheus"),
                grafana_host=grafana_host or dashboard_host,
                grafana_port=grafana_port,
                grafana_image=grafana_image,
                grafana_admin_user=grafana_admin_user,
                grafana_admin_password=grafana_admin_password,
                allow_anonymous=grafana_allow_anonymous,
                datasource_name=grafana_datasource_name,
                datasource_uid=grafana_datasource_uid,
                datasource_url=grafana_datasource_url,
                use_docker=grafana_use_docker,
            )
            if grafana_url:
                urls["grafana"] = grafana_url
                self._info(f" Grafana UI: {grafana_url}")

        if enable_console:
            try:
                from .monitoring import start_console_monitor

                self._info(" Starting console monitor...")
                start_console_monitor(refresh_rate=1.0)
            except ImportError:
                self._info(" Console monitor unavailable (install rich)")
            except Exception as e:
                self._info(f" Failed to start console monitor: {e}")

        self._monitoring_initialized = True
        if urls:
            self._info(" Monitoring services started successfully!")
            self._info(" Metrics will be automatically collected during inference")
        else:
            self._info(" No monitoring services were started successfully")
        self._monitoring_urls = urls
        return urls

    def stop_monitoring(self) -> None:
        """Stop all monitoring services.

        Gracefully shuts down Prometheus server and console monitor
        if they are running.
        """
        if not self._monitoring_initialized:
            self._info("No monitoring services to stop")
            return
        self._info("Stopping eSurge monitoring services...")

        if self._monitoring_server:
            try:
                self._monitoring_server.stop()
                self._info(" Prometheus server stopped")
            except Exception as e:
                self._info(f" Error stopping Prometheus server: {e}")
            self._monitoring_server = None

        self._stop_grafana_service()

        self._monitoring_initialized = False
        self._monitoring_urls = None
        self._info(" Monitoring services stopped")

    def get_metrics_summary(self) -> dict[str, Any]:
        """Get current performance metrics summary.

        Returns:
            Dictionary containing:
            - requests_per_second: Current request throughput
            - average_latency: Average request latency
            - average_ttft: Average time to first token
            - average_throughput: Average tokens/second
            - total_completed: Total completed requests
            - total_failed: Total failed requests
            - total_tokens: Total tokens generated
            - active_requests: Currently active requests
            - queue_size: Pending requests in queue
            - running_requests: Currently running requests
        """
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
        return self._monitoring_initialized

    def __del__(self):
        if getattr(self, "_scheduler_running", False):
            try:
                self.terminate()
            except Exception:
                pass
        if self._monitoring_initialized:
            try:
                self.stop_monitoring()
            except Exception:
                pass
        if getattr(self, "_profiling_active", False):
            try:
                self.stop_profiling()
            except Exception:
                pass
        if hasattr(self, "_worker_manager"):
            try:
                self._worker_manager.shutdown()
            except Exception:
                pass
        if hasattr(self, "runner"):
            try:
                self.runner.shutdown()
            except Exception:
                pass

    def __repr__(self):
        attrs = [
            f"name={self.esurge_name!r}",
            f"max_model_len={self.max_model_len}",
            f"max_num_seqs={self.max_num_seqs}",
            f"page_size={self.page_size}",
            f"reserve_tokens={self.reserve_tokens}",
            f"auto_truncate_prompt={self.auto_truncate_prompt}",
            f"auto_cap_new_tokens={self.auto_cap_new_tokens}",
            f"strict_context={self.strict_context}",
            f"truncate_mode={self.truncate_mode!r}",
            f"prefer_preserve_prompt={self.prefer_preserve_prompt}",
            f"decode_truncated_prompt={self.decode_truncated_prompt}",
            f"extra_eos_token_ids={self.extra_eos_token_ids}",
            f"scheduler_running={self._scheduler_running}",
        ]
        return "eSurge(\n  " + ",\n  ".join(attrs) + "\n)"
