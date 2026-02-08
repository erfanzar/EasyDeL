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

import os
import subprocess
import threading
import time
import typing
from collections import deque
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import jax
from eformer.common_types import NOT_GIVEN, _Empty
from eformer.loggings import get_logger
from jax import numpy as jnp
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from easydel.inference.sampling_params import SamplingParams
from easydel.utils import Registry
from easydel.workers.esurge.pipeline import WorkerManager

from .mixins import (
    EngineIOMixin,
    EngineLifecycleMixin,
    EngineMonitoringMixin,
    EngineParsingMixin,
    EngineRequestsMixin,
    EngineUtilsMixin,
)
from .multimodal import MultiModalManager
from .runners import eSurgeRunner
from .scheduler import Scheduler

if typing.TYPE_CHECKING:
    from easydel.infra import EasyDeLBaseModule

logger = get_logger("eSurgeEngine")

# Configuration constants
DEFAULT_DETOKENIZER_MAX_STATES = 1 << 16  # 65536 states for streaming decode
DEFAULT_PAGE_SIZE_GPU_MIN = 256  # Minimum efficient page size for GPU
DEFAULT_DECODE_INTERVAL_TOKENS = 16  # Decode every N tokens
DEFAULT_DECODE_INTERVAL_SECS = 0.04  # Or decode every N seconds (20ms)
# Default to fail-fast (1) so benchmark runs don't spin for hours on fatal errors.
# Set `EASURGE_MAX_SCHEDULER_ERRORS=10` (or higher) to restore retry behavior.
MAX_CONSECUTIVE_SCHEDULER_ERRORS = int(os.environ.get("EASURGE_MAX_SCHEDULER_ERRORS", "5"))
WORKER_DRAIN_MAX_RETRIES = 3  # Maximum retry attempts for worker drain
WORKER_DRAIN_INITIAL_DELAY = 0.1  # Initial retry delay in seconds
SamplingCallable = typing.Callable[[SamplingParams, dict[str, typing.Any]], SamplingParams | None] | None


def _set_requested_new(sp, n: int):
    """Set the max_tokens or max_new_tokens attribute on a SamplingParams object.

    Args:
        sp: SamplingParams instance to modify.
        n: Number of tokens to set.
    """
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
    tool_calls: list | None = None
    reasoning_content: str | None = None


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

    tool_calls: list | None = None
    delta_tool_calls: list | None = None
    reasoning_content: str | None = None
    delta_reasoning_content: str | None = None

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
class eSurge(EngineMonitoringMixin, EngineParsingMixin, EngineRequestsMixin, EngineIOMixin, EngineLifecycleMixin, EngineUtilsMixin):
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
        max_num_batched_tokens: int | None | _Empty = NOT_GIVEN,
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
        max_request_outputs: int | None = 1000,
        idle_reset_seconds: float | None = None,
        idle_reset_min_interval: float = 60.0,
        sampling_params_callback: SamplingCallable = None,
        extra_eos_token_ids: list[int] | None = None,
        silent_mode: bool = False,
        processor: Any | None = None,
        resolution_buckets: list[tuple[int, int]] | None = None,
        vision_cache_capacity_mb: int = 1024,
        tool_parser: str | None = None,
        reasoning_parser: str | None = None,
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
            max_request_outputs: Maximum number of completed RequestOutput objects
                to retain in memory for post-hoc access. Set to None for unlimited
                retention or <=0 to disable retention entirely.
            idle_reset_seconds: If set, automatically pause/resume the engine after
                this many seconds of continuous idleness (no running/pending requests).
                Useful for clearing stale state under long-running workloads.
            idle_reset_min_interval: Minimum seconds between idle resets to avoid
                reset loops when traffic is sparse.
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
            tool_parser: Name of the tool-call parser to use for automatic function-call
                extraction (e.g., "hermes", "mistral", "llama3_json"). When set, the
                engine runs the parser in ``_process_engine_outputs()`` so that
                ``RequestOutput.tool_calls`` / ``RequestOutput.delta_tool_calls`` are
                populated directly. If None, tool-call detection is left to the
                API server layer. See ``ToolParserManager`` for available parsers.
            reasoning_parser: Name of the reasoning parser to use for extracting
                chain-of-thought content (e.g., "deepseek_r1", "qwen3", "mistral").
                When set, the engine separates reasoning from content so that
                ``RequestOutput.reasoning_content`` / ``RequestOutput.delta_reasoning_content``
                are populated directly. If None, no reasoning extraction is performed.
                See ``ReasoningParserManager`` for available parsers.
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

        # Idle reset configuration
        self._idle_reset_seconds = float(idle_reset_seconds) if idle_reset_seconds else None
        self._idle_reset_min_interval = float(idle_reset_min_interval)
        self._idle_reset_last_activity = time.time()
        self._idle_reset_last_reset = 0.0
        self._idle_monitor_event = threading.Event()
        self._idle_monitor_thread: threading.Thread | None = None

        # Tool calling and reasoning parser initialization
        self.tool_parser_name = tool_parser
        self.reasoning_parser_name = reasoning_parser
        self._tool_parser_class = None
        self._reasoning_parser_class = None

        if tool_parser:
            try:
                from easydel.inference.tools import ToolParserManager

                self._tool_parser_class = ToolParserManager.get_tool_parser(tool_parser)
                if not silent_mode:
                    logger.info("Initialized tool parser: %s", tool_parser)
            except KeyError:
                logger.warning("Tool parser '%s' not found, function calling disabled", tool_parser)

        if reasoning_parser:
            try:
                from easydel.inference.reasoning import ReasoningParserManager

                self._reasoning_parser_class = ReasoningParserManager.get_reasoning_parser(reasoning_parser)
                if not silent_mode:
                    logger.info("Initialized reasoning parser: %s", reasoning_parser)
            except KeyError:
                logger.warning("Reasoning parser '%s' not found, reasoning disabled", reasoning_parser)

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
            backend = jax.default_backend()
            preferred_attn_mechanism = (
                AttentionMechanisms.UNIFIED_ATTENTION
                if backend == "gpu"
                else AttentionMechanisms.RAGGED_PAGE_ATTENTION_V3
            )
            user_provided_attn = "attn_mechanism" in kwargs
            requested_attn = kwargs.get("attn_mechanism") if user_provided_attn else None
            if requested_attn is None:
                user_provided_attn = False

            if user_provided_attn:
                attn_value = (
                    requested_attn.value
                    if isinstance(requested_attn, AttentionMechanisms)
                    else str(requested_attn) if requested_attn is not None else None
                )
                if backend == "gpu" and attn_value in (
                    AttentionMechanisms.RAGGED_PAGE_ATTENTION_V2.value,
                    AttentionMechanisms.RAGGED_PAGE_ATTENTION_V3.value,
                ):
                    logger.warning(
                        "GPU backend detected: `unified_attention` is preferred for eSurge inference; "
                        f"got attn_mechanism={attn_value!r}."
                    )
                elif backend != "gpu" and attn_value == AttentionMechanisms.PAGED_FLASH_ATTENTION.value:
                    logger.warning(
                        "Paged flash attention is CUDA-only; non-GPU backends are not supported. "
                        f"got attn_mechanism={attn_value!r}."
                    )
                elif backend == "tpu" and attn_value == AttentionMechanisms.UNIFIED_ATTENTION.value:
                    logger.warning(
                        "TPU backend detected: `ragged_page_attention_v3` is preferred for eSurge inference; "
                        f"got attn_mechanism={attn_value!r}."
                    )
                elif backend == "tpu" and attn_value == AttentionMechanisms.RAGGED_PAGE_ATTENTION_V2.value:
                    logger.warning(
                        "TPU backend detected: `ragged_page_attention_v3` is preferred for eSurge inference; "
                        f"got attn_mechanism={attn_value!r}."
                    )

            model = AutoEasyDeLModelForCausalLM.from_pretrained(
                model,
                dtype=dtype,
                param_dtype=dtype,
                precision=jax.lax.Precision.DEFAULT,
                auto_shard_model=auto_shard_model,
                sharding_axis_dims=sharding_axis_dims,
                config_kwargs=EasyDeLBaseConfigDict(
                    attn_mechanism=requested_attn if user_provided_attn else preferred_attn_mechanism,
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
        if max_num_batched_tokens is NOT_GIVEN and jax.default_backend() == "gpu":
            max_num_batched_tokens = min(max(2048, max_num_seqs), max_model_len)
            logger.info(
                f"GPU backend detected and `max_num_batched_tokens` was not provided; defaulting to {max_num_batched_tokens} tokens/step. "
                "Pass an explicit int to override, or pass `None` to disable this auto-default "
                "(falls back to `max_model_len`)."
            )
        elif max_num_batched_tokens is NOT_GIVEN and jax.default_backend() == "tpu":
            max_num_batched_tokens = min(max(8192, max_num_seqs), max_model_len)
            logger.info(
                f"TPU backend detected and `max_num_batched_tokens` was not provided; defaulting to {max_num_batched_tokens} tokens/step. "
                "Pass an explicit int to override, or pass `None` to disable this auto-default "
                "(falls back to `max_model_len`)."
            )
        elif max_num_batched_tokens is NOT_GIVEN:
            max_num_batched_tokens = None

        if compile_runner:
            # Limit compilation to the scheduler's per-step token budget when provided.
            # This avoids compiling long-context token buckets (e.g. 32K/64K) when
            # the scheduler will only ever emit smaller batches (e.g. 512/2048).
            self.runner.compile(max_num_batched_tokens=max_num_batched_tokens)

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
        self._max_request_outputs = None if max_request_outputs is None else int(max_request_outputs)
        self._finished_request_ids: deque[str] = deque()

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
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)

        self.__eos_ids = eos_token_id if isinstance(eos_token_id, (list, tuple, set)) else [eos_token_id]
        self.__eos_set = {int(tid) for tid in self.__eos_ids if tid is not None}
        # Publicly-named aliases for mixins/helpers to avoid class-name mangling.
        self._eos_ids = self.__eos_ids
        self._eos_set = self.__eos_set

        self.initiate()

    def _calculate_model_size(self, graphstate) -> str:
        """Calculate the model size in billions of parameters.

        Args:
            graphstate: The model's graph state containing parameter arrays.

        Returns:
            String representation of model size in billions (e.g., "7.00").
            Returns "unknown" if calculation fails.
        """
        try:
            num_params = sum(n.size for n in jax.tree_util.tree_flatten(graphstate)[0])
            return f"{num_params / 1e9:.2f}"
        except Exception:
            return "unknown"

    def _get_model_type(self, model) -> str:
        """Get the model type from its configuration.

        Args:
            model: The EasyDeL model instance.

        Returns:
            Lowercase model type string (e.g., "llama", "mistral", "unknown").
        """
        return getattr(model.config, "model_type", "unknown").lower()

    def _get_model_name(self, model) -> str:
        """Generate a human-readable model name.

        Args:
            model: The EasyDeL model instance.

        Returns:
            String in format "{model_type}-{size}b" (e.g., "llama-7.00b").
        """
        model_type = self._get_model_type(model)
        model_size = self._calculate_model_size(model.graphstate)
        return f"{model_type}-{model_size}b"

    @cached_property
    def esurge_name(self) -> str:
        """Get the engine's display name.

        Returns:
            Custom name if provided during initialization, otherwise an
            auto-generated name based on model type and size.
        """
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

























































    def __del__(self):
        """Destructor that cleans up resources.

        Attempts to gracefully terminate all running services including:
        - Background scheduler thread
        - Monitoring services (Prometheus, Grafana)
        - Profiler trace
        - Worker processes (tokenizer/detokenizer)
        - Model runner
        """
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
        """Return a detailed string representation of the engine.

        Returns:
            Multi-line string with all key configuration parameters.
        """
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
