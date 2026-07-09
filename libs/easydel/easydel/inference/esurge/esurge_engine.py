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
    >>> from easydel.inference.esurge import eSurge, eSurgeContextConfig, eSurgeRuntimeConfig
    >>> from easydel.inference.sampling_params import SamplingParams
    >>>
    >>> # Initialize engine
    >>> engine = eSurge(
    ...     model="model-name",
    ...     runtime=eSurgeRuntimeConfig.from_dict(max_model_len=8192),
    ...     context=eSurgeContextConfig.from_dict(reserve_tokens=800),
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
import queue
import subprocess
import threading
import time
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import jax
from jax import numpy as jnp
from spectrax.common_types import NOT_GIVEN

from easydel.axis import register_attention_data_parallel_axis
from easydel.inference.sampling_params import SamplingParams
from easydel.inference.speculative import DrafterProtocol
from easydel.utils import Registry

if typing.TYPE_CHECKING:
    from easydel.modules.auto.auto_modeling import PreTrainedLoading
from easydel.workers.esurge.pipeline import WorkerManager

from .config import (
    eSurgeCacheRuntimeConfig,
    eSurgeConfig,
    eSurgeContextConfig,
    eSurgeDistributedConfig,
    eSurgeDrafterConfig,
    eSurgeParsingConfig,
    eSurgeRuntimeConfig,
    eSurgeVisionConfig,
    eSurgeWorkerConfig,
)
from .distributed import make_config_fingerprint
from .distributed.coordinator import create_step_coordinator
from .engine import build_engine_assets
from .engine.output_pipeline import OutputPipeline
from .engine.registry import RequestRegistry
from .logger import logger
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


def _normalize_data_parallelism_axis(axis: str) -> str:
    """Normalize and validate a data-parallel axis name.

    Strips whitespace and ensures the result is non-empty.

    Args:
        axis: Raw axis name string to normalize.

    Returns:
        The stripped, validated axis name.

    Raises:
        ValueError: If the axis name is empty after stripping.
    """
    axis_name = str(axis).strip()
    if not axis_name:
        raise ValueError("`data_parallelism_axis` must be a non-empty string.")
    return axis_name


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
    raw_text: str | None = None


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
    raw_accumulated_text: str = ""  # decoded text before reasoning/tool separation
    raw_delta_text: str = ""  # latest raw decoded chunk before separation
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
        if self.accumulated_text:
            return self.accumulated_text
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
class eSurge(
    EngineMonitoringMixin,
    EngineParsingMixin,
    EngineRequestsMixin,
    EngineIOMixin,
    EngineLifecycleMixin,
    EngineUtilsMixin,
):
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
        ...     runtime=eSurgeRuntimeConfig.from_dict(max_model_len=8192),
        ...     context=eSurgeContextConfig.from_dict(reserve_tokens=800),
        ... )
        >>> engine.initiate()
        >>>
        >>> # Generate with streaming
        >>> for output in engine.stream("Tell me a story"):
        ...     print(output.delta_text, end="", flush=True)
    """

    def __init__(
        self,
        *,
        model: str | EasyDeLBaseModule,
        processor: Any | None = None,
        loading_kwargs: PreTrainedLoading | typing.Mapping[str, Any] | None = None,
        config: eSurgeConfig | typing.Mapping[str, Any] | None = None,
        runtime: eSurgeRuntimeConfig | typing.Mapping[str, Any] | None = None,
        cache: eSurgeCacheRuntimeConfig | typing.Mapping[str, Any] | None = None,
        context: eSurgeContextConfig | typing.Mapping[str, Any] | None = None,
        workers: eSurgeWorkerConfig | typing.Mapping[str, Any] | None = None,
        parsing: eSurgeParsingConfig | typing.Mapping[str, Any] | None = None,
        vision: eSurgeVisionConfig | typing.Mapping[str, Any] | None = None,
        distributed: eSurgeDistributedConfig | typing.Mapping[str, Any] | None = None,
        drafter: DrafterProtocol | bool | typing.Mapping[str, Any] | None = None,
        drafter_config: eSurgeDrafterConfig | typing.Mapping[str, Any] | None = None,
    ):
        """Initialize the eSurge engine.

        The engine accepts sectioned configs (each a ``TypedDict``-backed
        ``ConfigDict`` from :mod:`easydel.inference.esurge.config`). Each config
        argument may be passed as the typed object, a plain mapping with the
        same field names, or ``None`` (in which case all defaults are used).

        Args:
            model (str | EasyDeLBaseModule): Model id/path to load, or an
                already-loaded EasyDeL module.
            processor (Any | None): Unified text/multimodal processor. May be a
                tokenizer or an HF processor. When omitted for a string ``model``,
                it is auto-loaded from the same id/path.
            loading_kwargs (PreTrainedLoading | Mapping[str, Any] | None):
                Optional pretrained-loader kwargs forwarded to
                ``AutoEasyDeLModelForCausalLM.from_pretrained`` when ``model`` is
                a string id/path.
            config (eSurgeConfig | Mapping[str, Any] | None): Aggregate of all
                sections in one object (e.g. from an eLarge YAML ``esurge:``
                block). Mutually exclusive with the per-section arguments
                below; sections absent from the aggregate use their defaults.
            runtime (eSurgeRuntimeConfig | Mapping[str, Any] | None): Runtime
                and execution config. Fields (see :class:`eSurgeRuntimeConfig`):

                - ``esurge_name`` (str | None): Optional human-readable engine name.
                - ``kernel_tile_policy`` (KernelTilePolicy): GDN/Pallas kernel
                  tile policy.
                - ``max_model_len`` (int): Maximum sequence length (prompt + new).
                - ``min_input_pad`` (int): Minimum padded input length per step.
                - ``min_token_pad`` (int | None): Minimum padded total token count;
                  normal decode buckets are floored at 16 tokens.
                - ``max_num_seqs`` (int): Maximum number of concurrent sequences.
                - ``max_num_seq_buckets`` (list[int] | tuple[int, ...] | None):
                  Optional bucket sizes for batch padding.
                - ``async_scheduling`` (bool): Enable async scheduler thread.
                - ``max_num_batched_tokens`` (int | None | _Empty): Per-step
                  token budget. ``NOT_GIVEN`` triggers backend-specific defaults.
                - ``use_aot_forward`` (bool): Compile the forward pass ahead-of-time.
                - ``compile_runner`` (bool): Pre-compile runner buckets at startup.
                - ``runner_verbose`` (bool): Verbose runner logging.
                - ``overlap_execution`` (bool): Overlap host/device work across
                  steps.
                - ``sampler_metrics`` (bool): Emit per-step sampler metrics.
                - ``long_prefill_token_threshold`` (int | None): Threshold for
                  splitting long prefills.
                - ``enable_window_aware_runtime_cap`` (bool): Cap total tokens by
                  attention sliding window.
                - ``mpmd_scheduler`` (MpMdSchedulers | None): Optional MPMD scheduler.

                Consumed via ``Unpack[eSurgeRuntimeConfig]`` in
                :meth:`eSurgeRuntimeConfig.from_dict`.
            cache (eSurgeCacheRuntimeConfig | Mapping[str, Any] | None):
                KV-cache config. Fields (see :class:`eSurgeCacheRuntimeConfig`):

                - ``hbm_utilization`` (float): Fraction of HBM allocated to the
                  KV cache, in ``(0, 1]``.
                - ``page_size`` (int): KV-cache page size in tokens.
                - ``enable_prefix_caching`` (bool): Enable prefix-caching reuse.
                - ``max_cache_tokens`` (int | None): Optional hard cap on total
                  cached tokens.
                - ``cache_capacity_margin`` (float): Safety margin in ``(0, 1]``.
                - ``data_parallelism_axis`` (str): Mesh axis used for KV-page DP.
                - ``destroy_pages_on_pause`` (bool): Free pages on engine pause.

                Consumed via ``Unpack[eSurgeCacheRuntimeConfig]``.
            context (eSurgeContextConfig | Mapping[str, Any] | None):
                Context-window handling config. Fields
                (see :class:`eSurgeContextConfig`):

                - ``reserve_tokens`` (int | None): Tokens reserved for generation;
                  defaults to ``max_num_seqs`` when ``None``.
                - ``auto_truncate_prompt`` (bool): Truncate prompts that exceed
                  the limit.
                - ``auto_cap_new_tokens`` (bool): Cap ``max_new_tokens`` to fit
                  the remaining context window.
                - ``strict_context`` (bool): Raise instead of truncating when
                  context overflows.
                - ``truncate_mode`` (Literal["left", "right", "middle"]): Strategy
                  for prompt truncation.
                - ``prefer_preserve_prompt`` (bool): Prefer cutting generation
                  budget over the prompt.
                - ``decode_truncated_prompt`` (bool): Re-decode truncated prompt
                  text for accurate echoing.

                Consumed via ``Unpack[eSurgeContextConfig]``.
            workers (eSurgeWorkerConfig | Mapping[str, Any] | None):
                Tokenizer/detokenizer worker config. Fields
                (see :class:`eSurgeWorkerConfig`):

                - ``detokenizer_max_states`` (int): Max concurrent detokenizer
                  streaming states.
                - ``tokenizer_endpoint`` (str | None): Out-of-process tokenizer
                  endpoint URL.
                - ``detokenizer_endpoint`` (str | None): Out-of-process
                  detokenizer endpoint URL.
                - ``worker_startup_timeout`` (float | None): Startup timeout, secs.
                - ``max_request_outputs`` (int | None): Cap of retained finished
                  outputs.
                - ``idle_reset_seconds`` (float | None): Auto-reset cache after
                  N idle seconds; ``None`` disables.
                - ``idle_reset_min_interval`` (float): Minimum spacing (secs)
                  between idle resets.

                Consumed via ``Unpack[eSurgeWorkerConfig]``.
            parsing (eSurgeParsingConfig | Mapping[str, Any] | None):
                Tool/reasoning/sampling parser config. Fields
                (see :class:`eSurgeParsingConfig`):

                - ``sampling_params_callback`` (Callable | None): Optional hook
                  to mutate ``SamplingParams`` per request.
                - ``extra_eos_token_ids`` (list[int] | None): Extra EOS ids
                  appended to the tokenizer's defaults.
                - ``extra_stops`` (str | list[str] | None): Extra stop strings.
                - ``ignore_stop_strings_in_reasoning`` (bool): Suppress stop
                  strings inside reasoning blocks.
                - ``silent_mode`` (bool): Silence info-level logging.
                - ``tool_parser`` (Any | None): Tool parser name; ``None``
                  triggers auto-detection.
                - ``reasoning_parser`` (Any | None): Reasoning parser name;
                  ``None`` triggers auto-detection.

                Consumed via ``Unpack[eSurgeParsingConfig]``.
            vision (eSurgeVisionConfig | Mapping[str, Any] | None):
                Multimodal/vision config. Fields
                (see :class:`eSurgeVisionConfig`):

                - ``resolution_buckets`` (list[tuple[int, int]] | None): Discrete
                  vision resolutions for caching/precompilation.
                - ``vision_cache_capacity_mb`` (int): Vision encoder cache size.
                - ``compile_vision_encoder`` (bool): Precompile/use a bucketed
                  JIT helper for multimodal vision features when supported.
                - ``vision_patch_buckets`` (list[int] | None): Optional raw
                  patch-count buckets for vision precompile.

                Consumed via ``Unpack[eSurgeVisionConfig]``.
            distributed (eSurgeDistributedConfig | Mapping[str, Any] | None):
                Multi-host step-coordination config. Fields
                (see :class:`eSurgeDistributedConfig`):

                - ``coordination`` (Literal["replicated", "zmq"]):
                  ``"replicated"`` (default) assumes an outer driver calls the
                  engine identically on every host; ``"zmq"`` builds the
                  leader/worker step-replication plane needed for
                  single-ingress serving on a pod.
                - ``distributed_auth_token`` (str | None): Shared auth token;
                  required for ``coordination="zmq"``.
                - ``distributed_leader_addr`` (str | None): Leader host/IP;
                  defaults to the JAX coordinator host, then DNS discovery.
                - ``distributed_service_name`` (str | None): DNS / discovery
                  service name (worker-side leader lookup).
                - ``distributed_control_port`` (int): Control-plane TCP port.
                - ``distributed_control_bind_host`` (str): Leader bind host.
                - ``distributed_step_timeout_s`` / ``distributed_connect_timeout_s``
                  / ``distributed_ready_timeout_s`` (float): Failure-detection
                  and startup budgets.
                - ``distributed_heartbeat_interval_s`` /
                  ``distributed_heartbeat_timeout_s`` (float): Liveness beacons.
                - ``distributed_verify_digest_interval`` (int): Sampled-token
                  digest cross-check every K steps (0 disables).
                - ``distributed_max_inflight_steps`` (int): Leader/worker step
                  skew bound.

                Consumed via ``Unpack[eSurgeDistributedConfig]``.
            drafter (DrafterProtocol | bool | Mapping[str, Any] | None):
                Explicit drafter object, ``True`` for auto drafter construction,
                or a config mapping treated as ``drafter_config`` for concise
                calls.
            drafter_config (eSurgeDrafterConfig | Mapping[str, Any] | None):
                Declarative drafter settings. When enabled, eSurge calls
                ``model.drafter(method=..., num_draft_tokens=..., ...)`` after
                the target model is loaded. An assistant drafter model is
                configured via ``drafter_config={"method": "gemma4_assistant",
                "assistant_model": ...}``.

        Raises:
            ValueError: If processor/tokenizer cannot be inferred, if a
                ``runtime``/``cache``/``distributed`` field violates an invariant
                (positive numbers, valid mode strings), if ``max_model_len <=
                reserve_tokens``, or if ``coordination="zmq"`` is requested
                without a ``distributed_auth_token``.
        """
        from easydel.modules.auto.auto_modeling import PreTrainedLoading

        loading_data = dict(loading_kwargs or {})
        configured_model = loading_data.pop("pretrained_model_name_or_path", None)
        if configured_model is not None and configured_model != model:
            logger.warning(
                "`loading_kwargs.pretrained_model_name_or_path` is ignored; use the top-level `model` argument."
            )
        if processor is None:
            processor = loading_data.pop("processor", None)
        else:
            loading_data.pop("processor", None)
        loading_data.pop("tokenizer", None)
        loading_data["pretrained_model_name_or_path"] = model
        loading_data["processor"] = processor
        self.loading_kwargs = PreTrainedLoading.coerce_config(loading_data)

        if isinstance(drafter, Mapping):
            if drafter_config is not None:
                raise ValueError("Pass either `drafter` as a config mapping or `drafter_config`, not both.")
            drafter_config = drafter
            drafter = None

        if config is not None:
            sections = {
                "runtime": runtime,
                "cache": cache,
                "context": context,
                "workers": workers,
                "parsing": parsing,
                "vision": vision,
                "distributed": distributed,
                "drafter_config": drafter_config,
            }
            conflicting = sorted(name for name, value in sections.items() if value is not None)
            if conflicting:
                raise ValueError(
                    f"Pass either `config` or per-section arguments, not both (got both `config` and {conflicting})."
                )
            aggregate = eSurgeConfig.coerce_config(config)
            runtime = aggregate.runtime
            cache = aggregate.cache
            context = aggregate.context
            workers = aggregate.workers
            parsing = aggregate.parsing
            vision = aggregate.vision
            distributed = aggregate.distributed
            drafter_config = aggregate.drafter

        self.runtime_config = eSurgeRuntimeConfig.coerce_config(runtime)
        self.cache_config = eSurgeCacheRuntimeConfig.coerce_config(cache)
        self.context_config = eSurgeContextConfig.coerce_config(context)
        self.worker_config = eSurgeWorkerConfig.coerce_config(workers)
        self.parsing_config = eSurgeParsingConfig.coerce_config(parsing)
        self.vision_config = eSurgeVisionConfig.coerce_config(vision)
        self.distributed_config = eSurgeDistributedConfig.coerce_config(distributed)
        self.drafter_config = eSurgeDrafterConfig.coerce_config(drafter_config)

        # Mutable engine-level override; also settable later via
        # :meth:`set_sampling_params_callback`.
        self._sampling_params_callback = self.parsing_config.sampling_params_callback

        # Locals only for values that get transformed (resolved, normalized, or
        # mutated). Pure config field reads use ``self.X_config.field`` directly.
        dtype = self.loading_kwargs.dtype if self.loading_kwargs.dtype is not None else jnp.bfloat16

        self._info = logger.info if not self.parsing_config.silent_mode else lambda *args, **kwargs: None

        reserve_tokens = self.context_config.reserve_tokens
        if reserve_tokens is None:
            reserve_tokens = self.runtime_config.max_num_seqs
        self.reserve_tokens = reserve_tokens

        if self.runtime_config.max_model_len <= reserve_tokens:
            raise ValueError(
                f"Configuration error: max_model_len={self.runtime_config.max_model_len} "
                f"<= reserve_tokens={reserve_tokens}"
            )

        self.data_parallelism_axis = _normalize_data_parallelism_axis(self.cache_config.data_parallelism_axis)
        register_attention_data_parallel_axis(self.data_parallelism_axis)

        assets = build_engine_assets(
            model=model,
            processor=processor,
            loading_kwargs=self.loading_kwargs,
            runtime_config=self.runtime_config,
            parsing_config=self.parsing_config,
            drafter_config=self.drafter_config,
            drafter=drafter,
            data_parallelism_axis=self.data_parallelism_axis,
            dtype=dtype,
        )
        model = assets.model
        self.processor = assets.processor
        self.tokenizer = assets.tokenizer
        self._apply_data_parallel_axis_to_model(model)

        # Vision-language model support
        self._multimodal_manager: MultiModalManager | None = None
        if self.processor is not None:
            self._multimodal_manager = MultiModalManager(
                processor=self.processor,
                model=model,
                resolution_buckets=self.vision_config.resolution_buckets,
                cache_capacity_mb=self.vision_config.vision_cache_capacity_mb,
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
        self._prometheus_process: subprocess.Popen | None = None
        self._prometheus_temp_dir: str | None = None
        self._scheduler_running = False
        self._kv_cache_valid = True
        self._paused = False

        # Detokenizer cleanup tracking
        self._failed_detokenizer_resets: set[str] = set()
        self._detokenizer_cleanup_threshold = 100  # Clean up after this many failures

        # Idle reset state (config lives on self.worker_config)
        self._idle_reset_last_activity = time.time()
        self._idle_reset_last_reset = 0.0
        self._idle_monitor_event = threading.Event()
        self._idle_monitor_thread: threading.Thread | None = None

        tokenizer_endpoint = self.worker_config.tokenizer_endpoint or os.environ.get("EASURGE_TOKENIZER_ENDPOINT")
        detokenizer_endpoint = self.worker_config.detokenizer_endpoint or os.environ.get("EASURGE_DETOKENIZER_ENDPOINT")

        self._worker_manager = WorkerManager(
            assets.tokenizer_source, startup_timeout=self.worker_config.worker_startup_timeout
        )
        self._tokenizer_client, self._detokenizer_client = self._worker_manager.start(
            detokenizer_max_states=self.worker_config.detokenizer_max_states,
            tokenizer_endpoint=tokenizer_endpoint,
            detokenizer_endpoint=detokenizer_endpoint,
        )
        self._tokenizer_endpoint = self._worker_manager.tokenizer_endpoint
        self._detokenizer_endpoint = self._worker_manager.detokenizer_endpoint
        self._worker_startup_timeout = self._worker_manager._startup_timeout

        self.tool_parser = assets.tool_parser_name
        self.reasoning_parser_name = assets.reasoning_parser_name
        self._tool_parser_class = assets.tool_parser_class
        self._reasoning_parser_class = assets.reasoning_parser_class

        drafter = assets.drafter
        self.drafter = drafter
        runner_async_scheduling = bool(self.runtime_config.async_scheduling)
        if drafter is not None and runner_async_scheduling:
            logger.warning("Disabling async scheduling for runner-native speculative decoding.")
            runner_async_scheduling = False

        max_num_batched_tokens = self.runtime_config.max_num_batched_tokens
        if max_num_batched_tokens is NOT_GIVEN and jax.default_backend() == "gpu":
            max_num_batched_tokens = min(max(2048, self.runtime_config.max_num_seqs), self.runtime_config.max_model_len)
            logger.info(
                f"GPU backend detected and `max_num_batched_tokens` was not provided; defaulting to {max_num_batched_tokens} tokens/step. "
                "Pass an explicit int to override, or pass `None` to disable this auto-default "
                "(falls back to `max_model_len`)."
            )
        elif max_num_batched_tokens is NOT_GIVEN and jax.default_backend() == "tpu":
            max_num_batched_tokens = min(max(8192, self.runtime_config.max_num_seqs), self.runtime_config.max_model_len)
            logger.info(
                f"TPU backend detected and `max_num_batched_tokens` was not provided; defaulting to {max_num_batched_tokens} tokens/step. "
                "Pass an explicit int to override, or pass `None` to disable this auto-default "
                "(falls back to `max_model_len`)."
            )
        elif max_num_batched_tokens is NOT_GIVEN:
            max_num_batched_tokens = None

        # Profiling state
        self._profiling_active = False
        self._profiling_steps_remaining = 0
        self._profiling_output_dir: str | None = None
        self._profiling_host_level: int | None = None
        self._profiling_python_level: int | None = None
        self._possible_name = assets.model_name

        self.runtime_config.async_scheduling = bool(self.runtime_config.async_scheduling)

        self.runner = eSurgeRunner(
            model=model.esurge_compatible_model,
            hbm_utilization=self.cache_config.hbm_utilization,
            page_size=self.cache_config.page_size,
            max_cache_tokens=self.cache_config.max_cache_tokens,
            cache_capacity_margin=self.cache_config.cache_capacity_margin,
            kernel_tile_policy=self.runtime_config.kernel_tile_policy,
            max_model_len=self.runtime_config.max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_window_aware_runtime_cap=self.runtime_config.enable_window_aware_runtime_cap,
            min_input_pad=self.runtime_config.min_input_pad,
            max_num_seqs=self.runtime_config.max_num_seqs,
            max_num_seq_buckets=self.runtime_config.max_num_seq_buckets,
            async_scheduling=runner_async_scheduling,
            min_token_pad=self.runtime_config.min_token_pad,
            use_aot_forward=self.runtime_config.use_aot_forward,
            verbose=self.runtime_config.runner_verbose,
            enable_overlap_execution=self.runtime_config.overlap_execution,
            enable_sampler_metrics=self.runtime_config.sampler_metrics,
            mpmd_scheduler=self.runtime_config.mpmd_scheduler,
            pp_microbatch_count=self.runtime_config.pp_microbatch_count,
            pp_microbatch_size=self.runtime_config.pp_microbatch_size,
            compile_vision_encoder=self.vision_config.compile_vision_encoder,
            vision_patch_buckets=self.vision_config.vision_patch_buckets,
            drafter=drafter,
        )

        if self.runtime_config.compile_runner:
            # Limit compilation to the scheduler's per-step token budget when provided.
            # This avoids compiling long-context token buckets (e.g. 32K/64K) when
            # the scheduler will only ever emit smaller batches (e.g. 512/2048).
            self.runner.compile(max_num_batched_tokens=max_num_batched_tokens)

        long_prefill_token_threshold = self.runtime_config.long_prefill_token_threshold
        if long_prefill_token_threshold is None and self.runner.pipeline_plan.is_enabled:
            long_prefill_token_threshold = int(self.runtime_config.min_input_pad)
            logger.debug(
                "PP inference enabled; defaulting long_prefill_token_threshold to min_input_pad=%d.",
                long_prefill_token_threshold,
            )

        self.scheduler = Scheduler.from_runner(
            self.runner,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_prefix_caching=self.cache_config.enable_prefix_caching,
            async_scheduling=runner_async_scheduling,
            long_prefill_token_threshold=long_prefill_token_threshold,
            num_speculative_tokens=self.runner.num_speculative_tokens,
        )
        self._scheduler_max_num_batched_tokens = max_num_batched_tokens

        # Streaming decode cadence
        self.decode_interval_tokens = DEFAULT_DECODE_INTERVAL_TOKENS
        self.decode_interval_secs = DEFAULT_DECODE_INTERVAL_SECS

        # Request-scoped state lives in the registry; the attribute names
        # below alias its containers/locks for the mixins until each concern
        # moves into its own component.
        self._request_counter = 0
        self._registry = RequestRegistry(max_outputs=self.worker_config.max_request_outputs)
        self._active_requests = self._registry.records
        self._request_outputs = self._registry.outputs
        self._request_events = self._registry.events
        self._max_request_outputs = self._registry.max_outputs
        self._finished_request_ids = self._registry.finished_ids
        self._request_lock = self._registry.request_lock
        self._output_lock = self._registry.output_lock
        self._output_event = self._registry.output_event  # kept for generate()
        self._output_pipeline = OutputPipeline(
            process=self._process_engine_outputs,
            on_fatal=self._on_output_worker_fatal,
        )
        self._parser_stop_queue: queue.SimpleQueue[dict[str, str]] = queue.SimpleQueue()

        self.extra_eos_token_ids = self.parsing_config.extra_eos_token_ids or []
        self.extra_stops = self._normalize_stop_sequences(self.parsing_config.extra_stops)
        # Locks and signals
        self._scheduler_lock = threading.RLock()
        self._counter_lock = threading.Lock()

        # Scheduler thread
        self._scheduler_thread: threading.Thread | None = None
        self._scheduler_running = False
        self._scheduler_exception: BaseException | None = None
        self._scheduler_exception_tb: str | None = None

        self._generation_config_dict = assets.generation_config_dict
        self._generation_config_eos_token_ids = assets.generation_config_eos_ids
        self.__eos_ids = list(assets.eos_token_ids)
        self.__eos_set = set(assets.eos_token_ids)
        self._primary_eos_token_id = self.__eos_ids[0] if self.__eos_ids else None
        # Publicly-named aliases for mixins/helpers to avoid class-name mangling.
        self._eos_ids = self.__eos_ids
        self._eos_set = self.__eos_set

        needs_fingerprint = str(self.distributed_config.coordination or "replicated") == "zmq" and (
            jax.process_count() > 1
        )
        if needs_fingerprint:
            distributed_config = {
                "max_model_len": self.runtime_config.max_model_len,
                "max_num_seqs": self.runtime_config.max_num_seqs,
                "page_size": self.cache_config.page_size,
                "data_parallelism_axis": self.data_parallelism_axis,
                "max_num_batched_tokens": (
                    int(self.scheduler.max_num_scheduled_tokens)
                    if self.scheduler.max_num_scheduled_tokens is not None
                    else None
                ),
                "enable_window_aware_runtime_cap": self.runtime_config.enable_window_aware_runtime_cap,
                "scheduler_policy": str(
                    self.scheduler.policy.value if hasattr(self.scheduler.policy, "value") else self.scheduler.policy
                ),
            }
            self._distributed_config_fingerprint = make_config_fingerprint(distributed_config)
        else:
            self._distributed_config_fingerprint = None

        self._worker_stop_event = threading.Event()
        self._worker_replay_thread: threading.Thread | None = None
        self._step_coordinator = create_step_coordinator(
            self.runner,
            distributed_config=self.distributed_config,
            config_fingerprint=self._distributed_config_fingerprint,
        )
        self.distributed_role = "leader" if self._step_coordinator.is_leader else "worker"
        self.distributed_rank = int(self._step_coordinator.rank)
        self.distributed_world_size = int(self._step_coordinator.world_size)

        self.initiate()

    @cached_property
    def esurge_name(self) -> str:
        """Get the engine's display name.

        Returns:
            Custom name if provided during initialization, otherwise an
            auto-generated name based on model type and size.
        """
        return self.runtime_config.esurge_name or self._possible_name

    @property
    def max_model_len(self) -> int:
        """Maximum sequence length (prompt + generation) supported per request."""
        return self.runtime_config.max_model_len

    @property
    def max_num_seqs(self) -> int:
        """Maximum number of concurrently running requests."""
        return self.runtime_config.max_num_seqs

    @property
    def silent_mode(self) -> bool:
        """Whether info-level engine logging is suppressed."""
        return bool(self.parsing_config.silent_mode)

    @property
    def ignore_stop_strings_in_reasoning(self) -> bool:
        """Whether stop strings are suppressed inside reasoning blocks."""
        return bool(self.parsing_config.ignore_stop_strings_in_reasoning)

    @property
    def distributed_mode(self) -> bool:
        """Whether the multi-host step-coordination plane is active."""
        return int(getattr(self._step_coordinator, "world_size", 1) or 1) > 1

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

    def _apply_data_parallel_axis_to_model(self, model: EasyDeLBaseModule) -> None:
        """Keep model partition axes unchanged.

        Request-level DP is represented in scheduler, cache, and batch-preparer
        metadata. It must not be folded into the model's tensor/MLP/vocab axes;
        those remain the normal TP/FSDP/SP sharding policy.

        Args:
            model (EasyDeLBaseModule): Loaded model whose partition axes are
                intentionally left untouched. Argument is dropped after the
                no-op to keep the call site uniform with subclass overrides.
        """
        del model

    def _instantiate_reasoning_parser_for_metadata(self):
        """Build a short-lived reasoning parser instance for token metadata lookups.

        Returns:
            A reasoning-parser instance, or ``None`` if no parser class is
            registered, the tokenizer is missing, or instantiation raises.
        """
        if self._reasoning_parser_class is None or self.tokenizer is None:
            return None
        try:
            return self._reasoning_parser_class(self.tokenizer)
        except Exception:
            return None

    def _resolve_reasoning_boundary_token(self, attr_name: str) -> str | None:
        """Resolve a reasoning boundary token from parser metadata when available.

        Args:
            attr_name (str): Attribute name on the parser to read (e.g.
                ``"start_token"`` or ``"end_token"``).

        Returns:
            The token string if present and non-empty, otherwise ``None``.
        """
        parser = self._instantiate_reasoning_parser_for_metadata()
        return self._find_str_attr(parser, attr_name)

    def _find_str_attr(self, parser, attr_name: str) -> str | None:
        """Search parser, its delegate, and the parser class for a non-empty string attribute.

        Args:
            parser: Reasoning-parser instance to inspect first. May be ``None``.
            attr_name (str): Attribute name to look up on each candidate.

        Returns:
            The first non-empty string found, or ``None``.
        """
        candidates = (parser, getattr(parser, "_delegate", None), self._reasoning_parser_class)
        for candidate in candidates:
            token = getattr(candidate, attr_name, None)
            if isinstance(token, str) and token:
                return token
        return None

    def _resolve_reasoning_boundary_token_id(self, attr_name: str, token_attr_name: str) -> int | None:
        """Resolve a reasoning boundary token ID from parser metadata or tokenizer vocab.

        Args:
            attr_name (str): Attribute name holding a token id on the parser
                (e.g. ``"_start_token_id"``).
            token_attr_name (str): Attribute name holding the corresponding
                string token (e.g. ``"start_token"``); used as a fallback to
                look up the id via the tokenizer's vocabulary.

        Returns:
            The token id as ``int``, or ``None`` if it cannot be resolved.
        """
        parser = self._instantiate_reasoning_parser_for_metadata()
        candidates = (parser, getattr(parser, "_delegate", None))
        for candidate in candidates:
            token_id = getattr(candidate, attr_name, None)
            if isinstance(token_id, int):
                return token_id
        # Reuse the already-instantiated parser instead of creating a second one.
        token = self._find_str_attr(parser, token_attr_name)
        if token is None or self.tokenizer is None:
            return None
        try:
            vocab = self.tokenizer.get_vocab()
        except Exception:
            vocab = None
        if isinstance(vocab, dict):
            token_id = vocab.get(token)
            if isinstance(token_id, int):
                return token_id
        return None

    @property
    def think_start_token(self) -> str | None:
        """Reasoning-start token for the active reasoning parser, if any."""
        return self._resolve_reasoning_boundary_token("start_token")

    @property
    def think_end_token(self) -> str | None:
        """Reasoning-end token for the active reasoning parser, if any."""
        return self._resolve_reasoning_boundary_token("end_token")

    @property
    def think_start_token_id(self) -> int | None:
        """Tokenizer ID for :attr:`think_start_token`, if resolvable."""
        return self._resolve_reasoning_boundary_token_id("_start_token_id", "start_token")

    @property
    def think_end_token_id(self) -> int | None:
        """Tokenizer ID for :attr:`think_end_token`, if resolvable."""
        return self._resolve_reasoning_boundary_token_id("_end_token_id", "end_token")

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
        if getattr(self, "_step_coordinator", None) is not None:
            try:
                self._step_coordinator.shutdown("engine deleted")
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
            f"max_model_len={self.runtime_config.max_model_len}",
            f"max_num_seqs={self.runtime_config.max_num_seqs}",
            f"page_size={self.cache_config.page_size}",
            f"enable_window_aware_runtime_cap={self.runtime_config.enable_window_aware_runtime_cap}",
            f"data_parallelism_axis={self.data_parallelism_axis!r}",
            f"reserve_tokens={self.reserve_tokens}",
            f"auto_truncate_prompt={self.context_config.auto_truncate_prompt}",
            f"auto_cap_new_tokens={self.context_config.auto_cap_new_tokens}",
            f"strict_context={self.context_config.strict_context}",
            f"truncate_mode={self.context_config.truncate_mode!r}",
            f"prefer_preserve_prompt={self.context_config.prefer_preserve_prompt}",
            f"decode_truncated_prompt={self.context_config.decode_truncated_prompt}",
            f"extra_eos_token_ids={self.extra_eos_token_ids}",
            f"extra_stops={self.extra_stops!r}",
            f"coordination={self.distributed_config.coordination!r}",
            f"distributed_role={self.distributed_role!r}",
            f"distributed_rank={self.distributed_rank}",
            f"distributed_world_size={self.distributed_world_size}",
            f"scheduler_running={self._scheduler_running}",
        ]
        return "eSurge(\n  " + ",\n  ".join(attrs) + "\n)"
