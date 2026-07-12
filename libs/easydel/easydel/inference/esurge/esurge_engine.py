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
    - Scheduler thread (:class:`~.engine.loop.EngineLoop`): Continuously
      processes queued requests
    - Output worker (:class:`~.engine.output_pipeline.OutputPipeline`):
      Detokenizes and parses outputs off the hot path
    - JAX computation: Executes model forward passes

    The engine class itself is a composition root over explicit components in
    :mod:`easydel.inference.esurge.engine` — request admission, the scheduler
    loop, the output pipeline, the idle watchdog, and the monitoring stack.
"""

from __future__ import annotations

import os
import queue
import signal
import threading
import time
import traceback
import typing
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Literal, overload

import jax
import spectrax as spx
from jax import numpy as jnp
from spectrax.common_types import NOT_GIVEN

from easydel.axis import register_attention_data_parallel_axis
from easydel.inference.sampling_params import SamplingParams
from easydel.inference.speculative import DrafterProtocol
from easydel.utils import Registry

if typing.TYPE_CHECKING:
    from easydel.modules.auto.auto_modeling import PreTrainedLoading
from easydel.workers.esurge.pipeline import WorkerManager

from ..stream_protocol import (
    RequestOutputLike,
    StreamEventFrame,
    compute_stream_delta_text,
    iter_chat_completion_stream_responses,
    iter_responses_stream_frames,
)
from ..typed_models import ResponsesFinalizationOptions
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
from .distributed.request_plane import create_request_plane
from .distributed.wire import EngineSpec
from .engine import build_engine_assets
from .engine.admission import RequestAdmission, clone_sampling_params
from .engine.chat_templating import format_chat_prompt, normalize_stop_sequences
from .engine.idle import IdleMonitor
from .engine.loop import EngineLoop
from .engine.monitoring_stack import MonitoringStack
from .engine.output_pipeline import OutputPipeline
from .engine.registry import RequestRegistry
from .logger import logger
from .metrics import get_metrics_collector, log_metrics_summary
from .multimodal import MultiModalManager
from .request import EngineRequestStatus
from .runners import eSurgeRunner
from .runners.host_sync import mark_host_payloads_replicated
from .scheduler import Scheduler

if typing.TYPE_CHECKING:
    from easydel.infra import EasyDeLBaseModule

    from ..openai_api_modules import ChatCompletionStreamResponse
    from .request import EngineRequest

# Configuration constants


DEFAULT_DETOKENIZER_MAX_STATES = 1 << 16  # 65536 states for streaming decode
DEFAULT_PAGE_SIZE_GPU_MIN = 256  # Minimum efficient page size for GPU
DEFAULT_DECODE_INTERVAL_TOKENS = 16  # Decode every N tokens
DEFAULT_DECODE_INTERVAL_SECS = 0.04  # Or decode every N seconds (20ms)
WORKER_DRAIN_MAX_RETRIES = 3  # Maximum retry attempts for worker drain
WORKER_DRAIN_INITIAL_DELAY = 0.1  # Initial retry delay in seconds
_SCHEDULER_HEARTBEAT_WARN_S = float(os.environ.get("EASURGE_HEARTBEAT_WARN_S", "120"))
_SCHEDULER_HEARTBEAT_WARN_INTERVAL_S = float(os.environ.get("EASURGE_HEARTBEAT_WARN_INTERVAL_S", "30"))
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

    The class is a composition root: request admission lives in
    :class:`~.engine.admission.RequestAdmission`, the scheduler thread in
    :class:`~.engine.loop.EngineLoop`, output parsing in
    :class:`~.engine.output_pipeline.OutputPipeline`, the idle watchdog in
    :class:`~.engine.idle.IdleMonitor`, and Prometheus/Grafana process
    management in :class:`~.engine.monitoring_stack.MonitoringStack`.

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

        self._monitoring = MonitoringStack(info=self._info)
        self._engine_loop: EngineLoop | None = None
        self._scheduler_running = False
        self._kv_cache_valid = True
        self._paused = False

        # Detokenizer cleanup tracking
        self._failed_detokenizer_resets: set[str] = set()
        self._detokenizer_cleanup_threshold = 100  # Clean up after this many failures

        # Idle reset watchdog (config lives on self.worker_config)
        self._idle_monitor = IdleMonitor(
            on_idle_reset=self._perform_idle_reset,
            idle_reset_seconds=self.worker_config.idle_reset_seconds,
            min_interval=self.worker_config.idle_reset_min_interval,
            is_running=lambda: self._scheduler_running,
            is_busy=self._has_active_work,
            info=self._info,
        )

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

        # Request-scoped state lives in the registry; the attribute names
        # below alias its containers/locks for the engine-resident methods
        # (generate/stream/abort) that mutate request state directly.
        self._registry = RequestRegistry(max_outputs=self.worker_config.max_request_outputs)
        self._active_requests = self._registry.records
        self._request_outputs = self._registry.outputs
        self._request_events = self._registry.events
        self._max_request_outputs = self._registry.max_outputs
        self._finished_request_ids = self._registry.finished_ids
        self._request_lock = self._registry.request_lock
        self._output_lock = self._registry.output_lock
        self._output_event = self._registry.output_event  # kept for generate()
        self._parser_stop_queue: queue.SimpleQueue[dict[str, str]] = queue.SimpleQueue()

        self.extra_eos_token_ids = self.parsing_config.extra_eos_token_ids or []
        self.extra_stops = normalize_stop_sequences(self.parsing_config.extra_stops)
        # Locks and signals
        self._scheduler_lock = threading.RLock()

        # Scheduler thread crash state
        self._scheduler_exception: BaseException | None = None
        self._scheduler_exception_tb: str | None = None

        self._generation_config_dict = assets.generation_config_dict
        self._generation_config_eos_token_ids = assets.generation_config_eos_ids
        self.__eos_ids = list(assets.eos_token_ids)
        self.__eos_set = set(assets.eos_token_ids)
        self._primary_eos_token_id = self.__eos_ids[0] if self.__eos_ids else None
        # Publicly-named aliases for helpers to avoid class-name mangling.
        self._eos_ids = self.__eos_ids
        self._eos_set = self.__eos_set

        # Output pipeline: detokenize/parse off the scheduler thread.
        self._output_pipeline = OutputPipeline(
            registry=self._registry,
            detokenizer_client=self._detokenizer_client,
            eos_token_ids=self.__eos_ids,
            decode_interval_tokens=DEFAULT_DECODE_INTERVAL_TOKENS,
            decode_interval_secs=DEFAULT_DECODE_INTERVAL_SECS,
            on_stop_strings=self._enqueue_parser_stop_requests,
            on_activity=self._touch_activity,
            on_fatal=self._on_output_worker_fatal,
        )

        # Request admission: prompt -> scheduler-ready EngineRequests.
        self._admission = RequestAdmission(
            registry=self._registry,
            scheduler_submit=self._submit_scheduler_requests,
            tokenizer_client=self._tokenizer_client,
            tokenizer=self.tokenizer,
            context_config=self.context_config,
            reserve_tokens=self.reserve_tokens,
            max_model_len=int(self.runner.max_model_len),
            tool_parser_class=self._tool_parser_class,
            reasoning_parser_class=self._reasoning_parser_class,
            sampling_params_callback=lambda: self._sampling_params_callback,
            generation_config_dict=self._generation_config_dict,
            primary_eos_token_id=self._primary_eos_token_id,
            eos_token_ids=self.__eos_ids,
            extra_stops=self.extra_stops,
            ignore_stop_strings_in_reasoning=bool(self.parsing_config.ignore_stop_strings_in_reasoning),
            on_activity=self._touch_activity,
            info=self._info,
            callback_engine=self,
        )

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
            engine_spec=EngineSpec(
                esurge_name=str(self.esurge_name or ""),
                max_model_len=int(self.runner.max_model_len),
                max_num_seqs=int(self.runtime_config.max_num_seqs),
                reserve_tokens=int(self.reserve_tokens),
                eos_token_ids=[int(tid) for tid in self._eos_ids],
                # Must be a loadable tokenizer id/path — remote clients
                # bootstrap their own tokenizer from it. The display name
                # (_possible_name) is only a fallback.
                tokenizer_source=str(getattr(assets, "tokenizer_source", None) or self._possible_name or ""),
                page_size=int(self.cache_config.page_size),
            ),
        )
        if hasattr(self._step_coordinator, "set_stats_provider"):
            # Queue-load snapshots piggyback on client heartbeats so DP
            # routers can join-shortest-queue without polling.
            self._step_coordinator.set_stats_provider(self._queue_stats_snapshot)
        # Unified request plane: ingress on any rank. The owner (leader)
        # schedules; origin ranks forward admissions upstream and render the
        # returned token deltas through their own output pipeline.
        self._request_plane = create_request_plane(
            self._step_coordinator,
            submit_outputs=self._output_pipeline.submit,
            scheduler_submit=self._schedule_requests_direct,
            abort_request=self.abort_request,
            apply_stop_strings=self._enqueue_parser_stop_requests,
            next_arrival_stamp=self._admission.next_arrival_stamp,
            alive=lambda: self._generation_alive,
            admit_timeout_s=float(self.distributed_config.distributed_admit_timeout_s),
            info=self._info,
        )
        self.distributed_role = "leader" if self._step_coordinator.is_leader else "worker"
        self.distributed_rank = int(self._step_coordinator.rank)
        self.distributed_world_size = int(self._step_coordinator.world_size)
        if self.distributed_world_size > 1:
            # The step-replication plane makes host payloads identical by
            # construction; skip the per-step broadcast_one_to_all collectives.
            mark_host_payloads_replicated(True)

        self.initiate()

    @property
    def _scheduler_running(self) -> bool:
        """Whether the background scheduler loop is running.

        Backed by the current :class:`EngineLoop`'s running flag; before a
        loop exists (or on ZMQ worker ranks, which never spawn one) a plain
        boolean fallback attribute is used.
        """
        loop = getattr(self, "_engine_loop", None)
        if loop is not None:
            return bool(loop.running)
        return bool(getattr(self, "_scheduler_running_fallback", False))

    @_scheduler_running.setter
    def _scheduler_running(self, value: bool) -> None:
        loop = getattr(self, "_engine_loop", None)
        if loop is not None:
            loop.running = bool(value)
        else:
            self._scheduler_running_fallback = bool(value)

    def _queue_stats_snapshot(self):
        """Best-effort scheduler-load snapshot for request-plane clients.

        Runs on the coordinator IO thread WITHOUT the scheduler lock —
        routing tolerates slightly stale or torn reads, and taking the lock
        here could stall the step stream behind a heartbeat.
        """
        from .distributed.wire import QueueStats

        try:
            scheduler = self.scheduler
            pending = 0
            for request in list(scheduler.running):
                pending += max(0, int(request.num_prompt_tokens) - int(request.num_computed_tokens))
            for request in list(scheduler.waiting):
                pending += int(request.num_prompt_tokens)
            return QueueStats(
                num_waiting=len(scheduler.waiting),
                num_running=len(scheduler.running),
                pending_prefill_tokens=int(pending),
            )
        except Exception:
            return QueueStats()

    @property
    def _generation_alive(self) -> bool:
        """Whether requests can make progress on this rank.

        True when the local scheduler loop runs (owner / single host), or
        when this is a worker rank whose replay thread is serving the
        owner's step stream — requests admitted here are forwarded through
        the request plane and rendered locally, so generate/stream waiters
        are live even though no local scheduler thread exists.
        """
        if self._scheduler_running:
            return True
        thread = getattr(self, "_worker_replay_thread", None)
        return thread is not None and thread.is_alive()

    @property
    def _scheduler_thread(self) -> threading.Thread | None:
        """The scheduler loop's daemon thread (or ``None``)."""
        loop = getattr(self, "_engine_loop", None)
        if loop is not None:
            return loop.thread
        return getattr(self, "_scheduler_thread_fallback", None)

    @_scheduler_thread.setter
    def _scheduler_thread(self, value: threading.Thread | None) -> None:
        loop = getattr(self, "_engine_loop", None)
        if loop is not None:
            loop._thread = value
        else:
            self._scheduler_thread_fallback = value

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
        monitoring = getattr(self, "_monitoring", None)
        if monitoring is not None and monitoring.monitoring_active:
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

    def _format_chat_prompt(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
        chat_template: str | None = None,
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Render chat messages through :func:`format_chat_prompt` with this engine's tokenizer.

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            add_generation_prompt: Whether to append the generation prompt marker.
            chat_template: Optional custom Jinja2 template overriding the
                tokenizer's default.
            tools: Optional tool/function definitions.
            chat_template_kwargs: Extra keyword arguments forwarded to
                ``tokenizer.apply_chat_template``.

        Returns:
            Formatted prompt string ready for tokenization and generation.
        """
        return format_chat_prompt(
            self.tokenizer,
            messages,
            add_generation_prompt=add_generation_prompt,
            chat_template=chat_template,
            tools=tools,
            chat_template_kwargs=chat_template_kwargs,
        )

    def _tokenize_prompt(self, request_id: str, prompt: str) -> list[int]:
        """Tokenize a prompt string using the worker pipeline.

        Args:
            request_id: Request ID for tracking in the tokenizer worker.
            prompt: Text prompt to tokenize.

        Returns:
            List of token IDs.
        """
        return self._tokenizer_client.tokenize(request_id, prompt)

    @staticmethod
    def _compute_snapshot_delta_text(current_text: str, previous_text: str, fallback_delta: str) -> str:
        """Compute a safe streaming delta from accumulated text snapshots.

        Prefer exact prefix-diff semantics and avoid replaying stale fallback
        chunks when text has not advanced. If prefix alignment is lost, attempt
        suffix-prefix overlap recovery before falling back.

        Args:
            current_text: New accumulated text snapshot.
            previous_text: Accumulated text snapshot recorded at the
                previous yield.
            fallback_delta: Delta to surface when prefix alignment cannot
                be re-established between the two snapshots.

        Returns:
            Delta string safe to emit to the streaming consumer
            (guaranteed monotonic with respect to ``previous_text``).
        """
        return compute_stream_delta_text(current_text, previous_text, fallback_delta)

    def _has_active_work(self) -> bool:
        """Whether any request is running, pending, or still tracked."""
        return self.num_running_requests != 0 or self.num_pending_requests != 0 or bool(self._active_requests)

    def _perform_idle_reset(self) -> None:
        """Idle-watchdog reset action: drain via :meth:`pause` then :meth:`resume`."""
        self.pause()
        self.resume()

    def _touch_activity(self) -> None:
        """Mark the engine as active so the idle-reset watchdog re-arms."""
        self._idle_monitor.touch()

    def _start_idle_monitor(self) -> None:
        """Start the idle-reset watchdog (no-op when disabled / already running)."""
        self._idle_monitor.start()

    def _stop_idle_monitor(self) -> None:
        """Stop the idle-reset watchdog thread."""
        self._idle_monitor.stop()

    @property
    def decode_interval_tokens(self) -> int:
        """Streaming decode cadence: decode every N tokens."""
        return self._output_pipeline.decode_interval_tokens

    @decode_interval_tokens.setter
    def decode_interval_tokens(self, value: int) -> None:
        self._output_pipeline.decode_interval_tokens = int(value)

    @property
    def decode_interval_secs(self) -> float:
        """Streaming decode cadence: or decode every N seconds."""
        return self._output_pipeline.decode_interval_secs

    @decode_interval_secs.setter
    def decode_interval_secs(self, value: float) -> None:
        self._output_pipeline.decode_interval_secs = float(value)

    def _schedule_requests_direct(self, scheduler_requests: list[EngineRequest]) -> None:
        """Enqueue requests on the local scheduler under the scheduler lock.

        The raw scheduling primitive: no plane routing, no coalescing. The
        request plane's owner half injects this as its ``scheduler_submit``
        so plane-managed admissions cannot re-enter the plane.

        Args:
            scheduler_requests: Fully-formed :class:`EngineRequest` objects.
        """
        with self._scheduler_lock:
            for scheduler_request in scheduler_requests:
                self.scheduler.add_request(scheduler_request)

    def _submit_scheduler_requests(self, scheduler_requests: list[EngineRequest]) -> None:
        """Route admission-built requests to whichever scheduler owns this rank.

        Ranks that do not own scheduling (request-plane origins) forward the
        batch to the owner over the plane, blocking on its acknowledgement
        with no engine locks held; local registry state was already
        registered by admission, so the origin renders the returned deltas
        itself. The owner routes its own admissions through the plane's
        coalescing table (identical lockstep admissions from other ranks
        attach to one scheduled request). Single-host ranks enqueue on the
        local scheduler directly.

        Args:
            scheduler_requests: Requests produced by
                :meth:`RequestAdmission.add_request`.
        """
        plane = getattr(self, "_request_plane", None)
        if plane is None:
            self._schedule_requests_direct(scheduler_requests)
        elif plane.owns_scheduling:
            plane.admit_local(scheduler_requests)
        else:
            plane.submit_remote(scheduler_requests)

    def _add_request(
        self,
        request_id: str,
        prompt: str,
        sampling_params: SamplingParams,
        prompt_token_ids: list[int] | None = None,
        tool_parser_request: Any | None = None,
        defer_scheduler_enqueue: bool = False,
        # Vision-language model data (optional)
        pixel_values: Any | None = None,
        image_grid_thw: Any | None = None,
        pixel_values_videos: Any | None = None,
        video_grid_thw: Any | None = None,
        mm_features: list | None = None,
    ) -> list[EngineRequest] | None:
        """Admit a request via :meth:`RequestAdmission.add_request` (see there for details)."""
        return self._admission.add_request(
            request_id,
            prompt,
            sampling_params,
            prompt_token_ids=prompt_token_ids,
            tool_parser_request=tool_parser_request,
            defer_scheduler_enqueue=defer_scheduler_enqueue,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            mm_features=mm_features,
        )

    def _generate_request_id(self) -> str:
        """Allocate a fresh request id (see :meth:`RequestAdmission.generate_request_id`)."""
        return self._admission.generate_request_id()

    def _prepare_sampling_params_for_request(
        self,
        template: SamplingParams,
        *,
        request_id: str,
        prompt: str,
    ) -> SamplingParams:
        """Clone + finalize sampling params (see :meth:`RequestAdmission.prepare_sampling_params_for_request`)."""
        return self._admission.prepare_sampling_params_for_request(template, request_id=request_id, prompt=prompt)

    _TOOL_TOKEN_PATTERNS = ("tool_call", "tool_response", "tool")

    def _prepare_chat_sampling_params(
        self,
        sampling_params: SamplingParams | None,
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | None = None,
    ) -> SamplingParams:
        """Prepare sampling params with tool-token awareness.

        When no tools are active, tool-related special tokens are suppressed
        via ``logit_bias`` and ``skip_special_tokens`` is enabled. When tools
        are present, the protocol tokens are left untouched so the model can
        emit tool calls.

        Args:
            sampling_params: Caller-supplied sampling params; ``None``
                produces a fresh :class:`SamplingParams` with engine defaults.
            tools: Tool definitions for the current chat call; presence
                gates whether tool-token suppression is applied.
            tool_choice: Optional ``"auto"`` / ``"none"`` / ``"required"``
                directive (reserved for downstream parsers).

        Returns:
            A fresh :class:`SamplingParams` clone configured for the
            chat-with-tools (or chat-without-tools) regime.
        """
        if sampling_params is None:
            sampling_params = SamplingParams()
        else:
            sampling_params = clone_sampling_params(sampling_params)

        if tools:
            sampling_params.skip_special_tokens = False
            sampling_params.logit_bias = None
            return sampling_params

        vocab: dict[str, int] = {}
        if hasattr(self, "tokenizer") and hasattr(self.tokenizer, "get_vocab"):
            try:
                vocab = self.tokenizer.get_vocab()
            except Exception:
                pass

        tool_token_ids: list[int] = []
        for token_text, token_id in vocab.items():
            for pattern in self._TOOL_TOKEN_PATTERNS:
                if pattern in token_text.lower():
                    tool_token_ids.append(token_id)
                    break

        if tool_token_ids:
            merged_logit_bias = dict(sampling_params.logit_bias or {})
            for tid in tool_token_ids:
                merged_logit_bias[tid] = min(float(merged_logit_bias.get(tid, 0.0)), -100.0)
            sampling_params.logit_bias = merged_logit_bias
        sampling_params.skip_special_tokens = True
        return sampling_params

    @staticmethod
    def _to_python_scalar(value: Any) -> Any:
        """Convert a value to a Python scalar if possible.

        Args:
            value: Value to convert (may be a JAX/numpy array).

        Returns:
            Python scalar if conversion possible, original value otherwise.
        """
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return value

    def _sanitize_metrics_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Sanitize a metrics payload by converting arrays to scalars.

        Args:
            payload: Dictionary of metric values.

        Returns:
            Dictionary with all values converted to Python scalars.
        """
        return {k: self._to_python_scalar(v) for k, v in payload.items()}

    def _kv_cache_metadata(self) -> dict[str, Any]:
        """Get current KV cache configuration metadata.

        Returns:
            Dictionary with cache configuration including max_model_len,
            max_num_seqs, page_size, and executor-specific attributes.
        """
        metadata = getattr(getattr(self.runner, "executor_manager", None), "metadata", None)
        details: dict[str, Any] = {
            "max_model_len": self.runtime_config.max_model_len,
            "max_num_seqs": self.runtime_config.max_num_seqs,
            "page_size": self.cache_config.page_size,
        }
        if metadata is not None:
            for attr in ("num_pages", "page_size", "max_model_length", "hbm_utilization"):
                value = getattr(metadata, attr, None)
                if value is not None:
                    details[attr] = self._to_python_scalar(value)
        return details

    def _record_cache_event(self, event: str, payload: dict[str, Any]) -> None:
        """Record a cache event to the metrics collector.

        Args:
            event: Event name (e.g., "kv_cache_destroyed", "kv_cache_reinitialized").
            payload: Event details dictionary.
        """
        metrics_collector = get_metrics_collector()
        if metrics_collector:
            metrics_collector.record_cache_event(event, payload)

    def _log_cache_event(self, event: str, extra: dict[str, Any] | None = None) -> None:
        """Log a KV cache event with metadata.

        Args:
            event: Event name for logging.
            extra: Additional event details to include.
        """
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

    def abort_request(self, request_id: str) -> None:
        """Cancel a request and release every resource it holds.

        Atomic abort under all three engine locks (scheduler, request,
        output) so no other thread can observe a half-aborted state. The
        method:

        1. Resolves whether ``request_id`` is the parent or one of the
           ``n>1`` sample children, and gathers the full set of scheduler-
           side ids that must be terminated.
        2. Calls :meth:`Scheduler.finish_requests` with
           ``FINISHED_ABORTED`` to evict the request rows and free their
           pages.
        3. Marks the parent ``RequestOutput`` (and the appropriate sample
           slot for child aborts) as finished with ``finish_reason="abort"``.
        4. Resets the streaming detokenizer state for each terminated id;
           failures here are absorbed and re-tried later by
           :meth:`_cleanup_detokenizer_state`.
        5. Wakes any thread blocked in :meth:`generate` / :meth:`stream`
           / :meth:`_wait_for_request` so it can observe the new finished
           state.

        Args:
            request_id: Identifier of the request to abort. May be the
                parent id (terminates all sample children) or a child id
                of the form ``"{parent}-{sample_idx}"`` (terminates only
                that sample, marking the parent finished once every sample
                has terminated).

        State on exit: the request is no longer in
        ``_active_requests`` / ``_request_events``, the scheduler has
        released its row and pages, and the output object reflects the
        abort. A best-effort log line records the before/after queue
        counts for postmortem.
        """
        plane = getattr(self, "_request_plane", None)
        if plane is not None:
            if plane.owns_scheduling:
                # Detach this caller from any coalesced group it subscribes
                # to; when the group was scheduled under a neutral id and
                # this was its last subscriber, the plane aborts those rows.
                plane.notify_local_abort(request_id)
            else:
                # Origin rank: the scheduled rows live on the owner; tell it
                # to free them, then run the local registry/output cleanup
                # below (the scheduler-side steps no-op — the ids are not
                # scheduled here) so waiters observe the abort immediately.
                plane.notify_abort(request_id)
        detokenizer_reset_ids: set[str] = set()
        parent_request_id = request_id
        sample_index = 0
        metrics_collector = get_metrics_collector()
        before_running = 0
        before_waiting = 0
        after_running = 0
        after_waiting = 0
        abort_ids: set[str] = set()
        rd_present = False
        ro_present = False

        # Acquire all locks atomically to prevent race conditions
        with self._scheduler_lock, self._request_lock, self._output_lock:
            before_running = len(self.scheduler.running)
            before_waiting = len(self.scheduler.waiting)
            rd = self._active_requests.get(request_id)
            rd_present = rd is not None
            if rd is not None:
                parent_request_id = rd.parent_request_id or request_id
                sample_index = int(rd.sample_index or 0)

            # Resolve scheduler-side IDs to abort (n=1: request_id; n>1: children of parent)
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

            # Registry-side ids to clean up. On request-plane origin ranks the
            # children were scheduled on the owner, so the scheduler yields no
            # ids here — the local registry is the authority for cleanup.
            cleanup_ids = set(abort_ids)
            cleanup_ids.update(
                rid
                for rid, record in self._active_requests.items()
                if getattr(record, "parent_request_id", None) == request_id
            )
            detokenizer_reset_ids |= cleanup_ids

            # Clean up active request tracking (output retention honors max_request_outputs).
            for rid in cleanup_ids:
                self._active_requests.pop(rid, None)
            self._active_requests.pop(parent_request_id, None)
            for rid in cleanup_ids:
                if rid != parent_request_id:
                    self._request_events.pop(rid, None)

            # Update output state
            ro = self._request_outputs.get(parent_request_id)
            ro_present = ro is not None
            n_samples = len(ro.outputs) if ro is not None else 0
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
            after_running = len(self.scheduler.running)
            after_waiting = len(self.scheduler.waiting)

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
        if abort_ids:
            logger.warning(
                "Aborted request %s: matched_scheduler_ids=%s parent_request_id=%s "
                "scheduler_before(run=%s,wait=%s) scheduler_after(run=%s,wait=%s)",
                request_id,
                len(abort_ids),
                parent_request_id,
                before_running,
                before_waiting,
                after_running,
                after_waiting,
            )
        else:
            logger.warning(
                "Abort requested for %s but no live scheduler requests matched. "
                "parent_request_id=%s active_request_present=%s output_present=%s "
                "scheduler_before(run=%s,wait=%s) scheduler_after(run=%s,wait=%s)",
                request_id,
                parent_request_id,
                rd_present,
                ro_present,
                before_running,
                before_waiting,
                after_running,
                after_waiting,
            )
        log_metrics_summary()
        if ro is not None and ro.finished:
            with self._request_lock:
                self._request_events.pop(parent_request_id, None)
                if n_samples > 1:
                    for sample_idx in range(n_samples):
                        self._request_events.pop(f"{parent_request_id}-{sample_idx}", None)
            if self._max_request_outputs is not None:
                with self._output_lock:
                    self._track_finished_output(parent_request_id)

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
        """Number of requests admitted to the scheduler but not yet running.

        Acquires the scheduler lock to take a consistent snapshot of
        ``scheduler.waiting``. Surfaced for monitoring dashboards and the
        idle-reset watchdog (which only frees state when both pending and
        running counts are zero).

        Returns:
            Length of the scheduler's waiting queue at this instant.
        """
        with self._scheduler_lock:
            return len(self.scheduler.waiting)

    @property
    def num_running_requests(self) -> int:
        """Number of requests currently holding KV pages and being decoded.

        Mirrors :attr:`num_pending_requests` but reads
        ``scheduler.running``. The two together represent the engine's
        in-flight workload — both must drop to zero before
        :meth:`update_model_weights` or :meth:`release_model_state` will
        proceed.

        Returns:
            Length of the scheduler's running queue at this instant.
        """
        with self._scheduler_lock:
            return len(self.scheduler.running)

    @staticmethod
    def _is_nonrecoverable_scheduler_error(exc: BaseException) -> bool:
        """Decide whether a scheduler exception should bypass the retry budget.

        Most exceptions are absorbed up to ``MAX_CONSECUTIVE_SCHEDULER_ERRORS``
        retries, since a transient failure (rate-limited HBM allocator, a
        flaky tokenizer worker, etc.) often clears on the next iteration.
        Some failures, however, leave the engine in an inconsistent state
        that the next iteration will only worsen — for instance a DP-local
        page-table invariant violation indicates the scheduler and runner
        disagree on which DP shard owns a request, and retrying drives
        further pages onto the wrong shard. Those flavours are matched here
        and force an immediate abort via
        :meth:`_abort_scheduler_due_to_error`.

        Args:
            exc: Exception raised by the scheduler loop body.

        Returns:
            ``True`` for failures that should *not* be retried (currently:
            ``ValueError`` whose message contains
            ``"Non-DP-local page IDs detected"`` or ``"Distributed step
            synchronization failure"``); ``False`` otherwise.
        """
        from .distributed.coordinator import StepCoordinationError

        if isinstance(exc, StepCoordinationError):
            return True
        msg = str(exc)
        return isinstance(exc, ValueError) and (
            "Non-DP-local page IDs detected" in msg or "Distributed step synchronization failure" in msg
        )

    @staticmethod
    def _model_overrides_esurge_graphdef(model: EasyDeLBaseModule) -> bool:
        """Whether the model class supplies its own ``esurge_graphdef`` override.

        Some wrapper models (e.g. embedding-augmented LMs) cannot be
        rebuilt by eSurge's default ``_esurge_graphdef_from_graphdef``
        because their construction depends on runtime state that no longer
        exists at hot-swap time. They opt out by exposing a class-level
        ``esurge_graphdef`` attribute / property; weight-update logic
        consults this flag to decide whether to delegate eSurge graph
        construction to the underlying compatible model instead.

        Args:
            model: Loaded EasyDeL module to inspect.

        Returns:
            ``True`` iff the model's *class* (not the instance) has an
            ``esurge_graphdef`` entry in its dict — checking the class is
            important because most concrete models inherit a default that
            should not register as an override.
        """
        return type(model).__dict__.get("esurge_graphdef") is not None

    @classmethod
    def _split_graph_components_for_weight_update(cls, model: EasyDeLBaseModule):
        """Run :meth:`split_module` on the right model for an eSurge hot-swap.

        Engines that wrap their model in a non-eSurge-compatible class need
        their ``esurge_compatible_model`` (the inner LM that actually carries
        the executable graph) split, not the wrapper. This helper resolves
        the correct module, calls ``split_module()`` on it, and returns the
        full split tuple along with the resolved module so callers don't
        have to repeat the wrapper-detection logic.

        Args:
            model: Wrapped or bare module supplied to :meth:`update_model_weights`.

        Returns:
            Tuple ``(split_model, graphdef, graphstate, graphother)`` where
            ``split_model`` is either ``model.esurge_compatible_model``
            (when the wrapper opts out via :meth:`_model_overrides_esurge_graphdef`)
            or ``model`` itself, and the remaining three are the components
            produced by :meth:`split_module` on that module.
        """
        split_model = model
        if cls._model_overrides_esurge_graphdef(model):
            try:
                split_model = model.esurge_compatible_model
            except Exception:
                split_model = model
        split_graphdef, split_graphstate, split_graphother = split_model.split_module()
        return split_model, split_graphdef, split_graphstate, split_graphother

    def _apply_parser_stop_requests_locked(self, stop_string_finishes: dict[str, str]) -> None:
        """Apply parser-detected stop strings while the scheduler lock is held.

        Text parsing runs on the output worker, not on the generation thread.
        When that worker discovers a stop string it must still tell the
        scheduler to free the request, but it should never grab
        ``_scheduler_lock`` itself. The worker enqueues a tiny stop-signal
        packet and the generation loop drains it at scheduler-safe points.

        Args:
            stop_string_finishes: Mapping from request id to matched stop
                string. The caller must hold ``_scheduler_lock``.
        """
        if not stop_string_finishes:
            return
        for rid, stop_reason in stop_string_finishes.items():
            request = self.scheduler.requests.get(rid)
            if request is not None:
                request.stop_reason = stop_reason
        self.scheduler.finish_requests(stop_string_finishes.keys(), EngineRequestStatus.FINISHED_STOPPED)

    def _enqueue_parser_stop_requests(self, stop_string_finishes: dict[str, str]) -> None:
        """Queue parser-detected stop strings for the generation thread.

        This is intentionally non-blocking with respect to ``_scheduler_lock``.
        It keeps reasoning/tool/stop-string parsing from contending with the
        scheduler loop. Lightweight tests that do not define
        ``_parser_stop_queue`` fall back to the old inline behaviour.

        Args:
            stop_string_finishes: Mapping from request id to the matched
                stop string; routed either onto ``_parser_stop_queue``
                (when present) or applied inline under ``_scheduler_lock``.
        """
        if not stop_string_finishes:
            return
        plane = getattr(self, "_request_plane", None)
        if plane is not None:
            if plane.owns_scheduling:
                # Coalesced groups scheduled under neutral ids (remote-first
                # attach) need their local sample ids mapped back onto the
                # scheduler's ids; everything else passes through unchanged.
                stop_string_finishes = plane.translate_local_stop_hits(stop_string_finishes)
            else:
                # Origin rank: the request is scheduled on the owner; forward
                # the stop hit so the owner frees the rows. Any locally-owned
                # rest (none today on origin ranks) falls through below.
                stop_string_finishes = plane.forward_stop_hits(stop_string_finishes)
                if not stop_string_finishes:
                    return
        stop_queue = getattr(self, "_parser_stop_queue", None)
        if stop_queue is None:
            with self._scheduler_lock:
                self._apply_parser_stop_requests_locked(dict(stop_string_finishes))
            return
        stop_queue.put(dict(stop_string_finishes))

    def _drain_parser_stop_requests_locked(self) -> None:
        """Drain queued parser stop signals on the generation thread.

        Called immediately before ``scheduler.schedule()`` while
        ``_scheduler_lock`` is already held. Merging all currently queued
        packets keeps the scheduler transition tiny and deterministic.
        """
        stop_queue = getattr(self, "_parser_stop_queue", None)
        if stop_queue is None:
            return
        merged: dict[str, str] = {}
        while True:
            try:
                merged.update(stop_queue.get_nowait())
            except queue.Empty:
                break
        self._apply_parser_stop_requests_locked(merged)

    @classmethod
    def _resolve_graphdef_for_weight_update(cls, model: EasyDeLBaseModule, split_graphdef=None):
        """Resolve the graphdef to use for a model-weight refresh.

        Some wrapper types override ``esurge_graphdef`` to delegate eSurge graph
        construction to an underlying base LM because the wrapper itself cannot
        be lazily rebuilt from config. Respect that override instead of forcing
        ``_esurge_graphdef_from_graphdef(...)`` on the wrapper graphdef.
        """
        if cls._model_overrides_esurge_graphdef(model):
            try:
                compatible_model = model.esurge_compatible_model
            except Exception:
                compatible_model = None
            if compatible_model is not None:
                return split_graphdef if split_graphdef is not None else compatible_model.graphdef
            return model.esurge_graphdef

        base_graphdef = split_graphdef if split_graphdef is not None else model.graphdef
        return model._esurge_graphdef_from_graphdef(base_graphdef)

    def _abort_scheduler_due_to_error(self, exc: BaseException) -> None:
        """Record a fatal scheduler error and wake all waiting callers.

        Args:
            exc: The exception that caused the scheduler failure.

        Note:
            This method records the exception and traceback, stops the scheduler,
            and signals all waiting events so that blocking calls (generate/stream/chat)
            can raise the error immediately.
        """
        # Record the failure so waiting callers can raise immediately.
        with self._request_lock:
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
        """Check for scheduler failure and raise if one occurred.

        Raises:
            RuntimeError: If the scheduler encountered a fatal error, with
                the original exception and traceback included.
        """
        exc = self._scheduler_exception
        if exc is None:
            # No crash, but check for heartbeat staleness (possible hang).
            self._check_scheduler_heartbeat()
            return
        tb = self._scheduler_exception_tb
        if tb:
            raise RuntimeError(f"eSurge scheduler crashed: {exc}\n{tb}") from exc
        raise RuntimeError(f"eSurge scheduler crashed: {exc}") from exc

    def _install_signal_diagnostics(self) -> None:
        """Install signal handlers that log engine state before exit.

        Registers handlers for SIGTERM and SIGUSR1 (where available) to
        dump scheduler/request state before the process is killed. This
        makes OOM-kills and external termination diagnosable.
        """

        def _dump_state(signum, frame):
            """Signal handler that logs engine state and re-raises with the default handler.

            Captures running/pending request counts and the last scheduler
            heartbeat age into a CRITICAL log line so post-mortem diagnosis
            of OOM-kills and external SIGTERMs has actionable context. After
            logging it restores the default signal handler and re-sends the
            signal so the process actually terminates instead of swallowing
            the kill.
            """
            sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
            try:
                running = getattr(self, "num_running_requests", "?")
                pending = getattr(self, "num_pending_requests", "?")
                sched_alive = getattr(self, "_scheduler_running", "?")
                heartbeat = getattr(self, "_scheduler_heartbeat", None)
                hb_age = f"{time.monotonic() - heartbeat:.1f}s ago" if heartbeat else "never"
                logger.critical(
                    "eSurge received signal %s — dumping state before exit: "
                    "scheduler_running=%s running_reqs=%s pending_reqs=%s "
                    "last_heartbeat=%s",
                    sig_name,
                    sched_alive,
                    running,
                    pending,
                    hb_age,
                )
            except Exception:
                logger.critical("eSurge received signal %s (state dump failed)", signum)
            # Re-raise with default handler so the process actually terminates.
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)

        for sig in (signal.SIGTERM,):
            try:
                prev = signal.getsignal(sig)
                # Only install if the current handler is the default.
                if prev in (signal.SIG_DFL, None):
                    signal.signal(sig, _dump_state)
            except (OSError, ValueError):
                pass  # Not main thread or signal not available

    def _update_scheduler_heartbeat(self) -> None:
        """Stamp the scheduler-loop progress watermark with the current monotonic time.

        Called twice per iteration of the scheduler loop (once after
        ``schedule()`` lands, once after ``update_from_output()`` lands)
        so :meth:`_check_scheduler_heartbeat` can distinguish a hung loop
        from one that is busy on a long step.
        """
        self._scheduler_heartbeat = time.monotonic()

    def _check_scheduler_heartbeat(self) -> None:
        """Emit a WARNING when the scheduler heartbeat is staler than the threshold.

        Polled from :meth:`_raise_if_scheduler_failed` (which is invoked
        on every poll of :meth:`generate` / :meth:`stream`). Compares the
        last heartbeat against
        :data:`_SCHEDULER_HEARTBEAT_WARN_S`; when stale, logs at WARNING
        with the age and rate-limits subsequent warnings to one per
        :data:`_SCHEDULER_HEARTBEAT_WARN_INTERVAL_S` so a hang doesn't
        flood the logs. No-op when the scheduler isn't running or no
        heartbeat has been recorded yet.
        """
        if not getattr(self, "_scheduler_running", False):
            return
        heartbeat = getattr(self, "_scheduler_heartbeat", None)
        if heartbeat is None:
            return
        now = time.monotonic()
        age = now - heartbeat
        if age > _SCHEDULER_HEARTBEAT_WARN_S:
            last_warn = getattr(self, "_scheduler_heartbeat_last_warn", 0.0)
            if now - last_warn < _SCHEDULER_HEARTBEAT_WARN_INTERVAL_S:
                return
            self._scheduler_heartbeat_last_warn = now
            logger.warning(
                "Scheduler heartbeat stale: last update %.1fs ago (threshold=%.0fs). Possible hang or deadlock.",
                age,
                _SCHEDULER_HEARTBEAT_WARN_S,
            )

    def _track_finished_output(self, request_id: str) -> None:
        """Track and evict completed RequestOutput objects to cap memory usage.

        Manages the finished request history using a FIFO eviction policy.
        When the number of retained outputs exceeds max_request_outputs,
        the oldest outputs are removed to free memory.

        Args:
            request_id: ID of the request that just finished.
        """
        max_outputs = self._max_request_outputs
        if max_outputs is None:
            return
        if max_outputs <= 0:
            self._request_outputs.pop(request_id, None)
            return
        if request_id in self._finished_request_ids:
            return
        self._finished_request_ids.append(request_id)
        while len(self._finished_request_ids) > max_outputs:
            old_id = self._finished_request_ids.popleft()
            if old_id == request_id:
                continue
            self._request_outputs.pop(old_id, None)

    def _on_output_worker_fatal(self, exc: BaseException, tb: str) -> None:
        """Record an output-worker failure and wake every blocked caller.

        Runs on the output-worker thread. Records the exception on
        ``self._scheduler_exception``, flips ``_scheduler_running`` off, and
        wakes every blocked request event plus the global output event so
        generate/stream callers exit their wait loops with the captured
        traceback rather than hanging.

        Args:
            exc: The exception raised inside output processing.
            tb: Its formatted traceback.
        """
        self._scheduler_exception = exc
        self._scheduler_exception_tb = tb
        self._scheduler_running = False
        self._registry.signal_all()

    def _run_worker_replay_loop(self) -> None:
        """Body of the ZMQ worker replay thread.

        Replays the leader's step stream until Shutdown/stop; any failure is
        recorded on the engine's crash-state fields and wakes every blocked
        caller so the process dies loudly instead of desyncing silently.
        """
        try:
            self._step_coordinator.run_worker_loop(self._worker_stop_event)
        except Exception as exc:
            traceback.print_exc()
            logger.critical("Worker replay loop failed: %s", exc)
            self._scheduler_exception = exc
            self._scheduler_exception_tb = traceback.format_exc()
            self._registry.signal_all()

    def _stop_worker_replay_loop(self) -> None:
        """Cooperatively stop the ZMQ worker replay thread, if running."""
        stop_event = getattr(self, "_worker_stop_event", None)
        thread = getattr(self, "_worker_replay_thread", None)
        if stop_event is None or thread is None:
            return
        plane = getattr(self, "_request_plane", None)
        if plane is not None and not plane.owns_scheduling:
            # The replay thread is the origin's receive path; admissions
            # blocked on owner acks can never complete once it stops.
            plane.fail_pending("worker replay loop stopped")
        stop_event.set()
        if thread.is_alive():
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning("Worker replay thread did not stop gracefully")
                return
        if getattr(self, "_worker_replay_thread", None) is thread:
            self._worker_replay_thread = None

    def _start_engine_output_worker(self) -> None:
        """Start the background parser/output worker.

        The scheduler loop is the hot generation path. It should apply model
        tokens to scheduler state and immediately return to launching work.
        Detokenization, reasoning/tool parsing, stream-delta assembly, and
        request-event wakeups are handled by the :class:`OutputPipeline`
        worker so output processing overlaps the next device step while
        preserving per-token ordering.
        """
        self._output_pipeline.start()

    def _enqueue_engine_outputs(self, engine_outputs) -> None:
        """Queue engine outputs for asynchronous parsing, or process inline.

        When this rank owns scheduling for remote origins, the bundle is
        also teed to the request plane (one bool read plus one queue put on
        the hot path) so origins receive their raw token deltas.

        Args:
            engine_outputs: Scheduler-produced output bundle from
                ``update_from_output``.
        """
        plane = getattr(self, "_request_plane", None)
        if plane is not None and plane.owns_scheduling and plane.needs_tee:
            plane.tee_outputs(engine_outputs)
        self._output_pipeline.submit(engine_outputs)

    def _stop_engine_output_worker(self) -> None:
        """Drain and stop the asynchronous parser/output worker."""
        self._output_pipeline.stop()

    def initiate(self) -> None:
        """Start (or wake) the background scheduler so requests can flow.

        Two flavours of behaviour, decided by the step coordinator:

        * **Worker rank** (``coordination="zmq"``) — connects to the leader,
          ensures KV pages are allocated, and spawns the replay thread that
          executes the leader's step stream; no scheduler thread is spawned
          because the leader owns scheduling.
        * **Leader rank or single-host** — re-allocates KV pages if they
          were previously destroyed, clears any prior crash state,
          installs SIGTERM diagnostics, and constructs + starts an
          :class:`EngineLoop` that runs the main control loop on a daemon
          thread.

        The scheduler-loop body itself comes in two shapes depending on
        ``runtime_config.overlap_execution`` — see
        :meth:`EngineLoop._run_sync` and :meth:`EngineLoop._run_overlap`.

        State on exit: ``_scheduler_running=True``, idle monitor armed,
        ``_paused=False``. No-op if the loop is already running.
        """
        with self._scheduler_lock:
            step_coordinator = getattr(self, "_step_coordinator", None)
            if step_coordinator is not None and not step_coordinator.is_leader:
                # ZMQ worker rank: no scheduler thread; a dedicated replay
                # thread executes the leader's step stream so device dispatch
                # order matches the leader's call order exactly.
                if self.runner.executor_manager.kv_pages is None:
                    self.runner.initialize_kv_cache()
                    self._kv_cache_valid = True
                replay_thread = getattr(self, "_worker_replay_thread", None)
                if replay_thread is not None and replay_thread.is_alive():
                    self._info("Worker replay loop is already running")
                    return
                step_coordinator.start()
                self._scheduler_exception = None
                self._scheduler_exception_tb = None
                self._scheduler_running = False
                # Worker ranks render remotely-admitted request outputs
                # locally (request plane), so they need the output worker too.
                self._start_engine_output_worker()
                self._worker_stop_event.clear()
                self._worker_replay_thread = threading.Thread(
                    target=self._run_worker_replay_loop,
                    name="eSurgeWorkerReplay",
                    daemon=True,
                )
                self._worker_replay_thread.start()
                self._touch_activity()
                self._start_idle_monitor()
                self._paused = False
                self._info("Distributed worker replay loop is running (scheduler loop disabled).")
                return

            if self._scheduler_running:
                self._info("Scheduler loop is already running")
                return

            if self.runner.executor_manager.kv_pages is None:
                self.runner.initialize_kv_cache()
                self._kv_cache_valid = True

            if step_coordinator is not None:
                # Leader (or single host, where this is a no-op): block until
                # every worker rank has connected and finished initializing.
                step_coordinator.start()

            request_plane = getattr(self, "_request_plane", None)
            if request_plane is not None and request_plane.owns_scheduling:
                request_plane.ensure_started()

            self._start_engine_output_worker()

            # Clear any previous crash state before starting a fresh scheduler thread.
            self._scheduler_exception = None
            self._scheduler_exception_tb = None
            self._scheduler_heartbeat = None

            # Install signal diagnostics (best-effort, main thread only).
            try:
                self._install_signal_diagnostics()
            except Exception:
                pass

            self._engine_loop = EngineLoop(
                scheduler=self.scheduler,
                runner=self.runner,
                coordinator=step_coordinator,
                scheduler_lock=self._scheduler_lock,
                emit_outputs=self._enqueue_engine_outputs,
                drain_parser_stops=self._drain_parser_stop_requests_locked,
                on_fatal=self._abort_scheduler_due_to_error,
                heartbeat=self._update_scheduler_heartbeat,
                handle_profiling_step=self._handle_profiling_step,
                is_nonrecoverable=self._is_nonrecoverable_scheduler_error,
                overlap_execution=bool(self.runtime_config.overlap_execution),
                async_scheduling=bool(self.runtime_config.async_scheduling),
                runner_verbose=bool(getattr(self.runtime_config, "runner_verbose", False)),
                info=self._info,
            )
            self._engine_loop.start()
            self._info("Background scheduler initiated")
            self._touch_activity()
            self._start_idle_monitor()
            self._paused = False

    def terminate(self) -> None:
        """Tear down the scheduler loop and release subordinate resources.

        Engine-stop path. Steps:

        1. Stop the idle-reset watchdog so it cannot fire mid-shutdown.
        2. Ask the :class:`EngineLoop` to stop under ``_scheduler_lock``,
           then release the lock before joining its thread. The release is
           important because the scheduler may need the same lock to apply the
           final runner output before it can exit. Workers in distributed mode
           are also asked to shut down their control servers here.
        3. If a profiler trace is active, stop it via
           :meth:`stop_profiling` (best-effort; failures are demoted to
           DEBUG).
        4. Forward to :meth:`eSurgeRunner.shutdown` so resident PP-stage
           worker threads inside the :class:`ModelStepExecutor` are
           joined.
        5. Clear runner buffers via :meth:`_reset_runner_state_if_idle`
           when the engine is genuinely idle, so a subsequent
           :meth:`initiate` starts from a clean slate.

        State on exit: the scheduler thread is gone, the runner has no
        in-flight state, and no background timers are active. Idempotent
        — calling it on an already-terminated engine logs an info line
        and returns.
        """
        self._stop_idle_monitor()
        self._stop_worker_replay_loop()
        engine_loop: EngineLoop | None = None
        with self._scheduler_lock:
            if not self._scheduler_running:
                self._stop_engine_output_worker()
                self._info("Scheduler loop is not running")
                return
            self._info("Stopping background scheduler loop...")
            engine_loop = self._engine_loop
            self._scheduler_running = False

        if engine_loop is not None:
            engine_loop.join(timeout=5.0)
            loop_thread = engine_loop.thread
            if loop_thread is not None and loop_thread.is_alive():
                logger.warning("Scheduler thread did not stop gracefully")

        self._stop_engine_output_worker()
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
        step_coordinator = getattr(self, "_step_coordinator", None)
        if step_coordinator is not None:
            try:
                step_coordinator.shutdown("engine terminate")
            except Exception:
                logger.debug("Step coordinator shutdown encountered an error", exc_info=True)
        request_plane = getattr(self, "_request_plane", None)
        if request_plane is not None:
            try:
                if request_plane.owns_scheduling:
                    request_plane.shutdown()
                else:
                    request_plane.fail_pending("engine terminated")
            except Exception:
                logger.debug("Request plane shutdown encountered an error", exc_info=True)
        # Clear runner buffers if idle to avoid stale state on next start.
        self._reset_runner_state_if_idle("terminate")

    def pause(self) -> None:
        """Stop the scheduler loop while preserving request bookkeeping.

        Suspended state is intentionally non-destructive for requests:
        the waiting and running queues survive, and active streaming
        clients merely stop receiving updates until :meth:`resume`.
        Internally this terminates the scheduler thread (so any
        in-flight model step is drained), drains the resident PP worker
        pool, and — when ``cache_config.destroy_pages_on_pause`` is set
        — frees the KV pages to reclaim HBM (skipped if requests are
        still active to avoid corrupting their state). Finally,
        :meth:`_reset_runner_state_if_idle` clears the runner's
        sequence buffer if the engine is genuinely idle.

        Sets ``_paused=True``. Idempotent — pausing an already-paused
        engine logs an info line and returns.
        """
        if not self._scheduler_running:
            self._info("Scheduler loop already paused or not running")
            self._paused = True
            return

        self._info("Pausing eSurge scheduler loop...")
        self.terminate()
        self._paused = True
        self._drain_pipeline_workers("pause")
        if self.cache_config.destroy_pages_on_pause:
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
        """Reopen the engine for inference after a :meth:`pause`.

        Drains any leftover PP-worker state, reinitializes the KV cache
        when ``destroy_pages_on_pause`` had freed it, and delegates to
        :meth:`initiate` to spin a fresh scheduler thread. Requests that
        were waiting before the pause are immediately considered for
        scheduling on the next iteration. Idempotent for already-running
        engines.
        """
        if self._scheduler_running:
            self._info("Scheduler loop already running")
            return
        self._info("Resuming eSurge scheduler loop...")
        self._drain_pipeline_workers("resume")
        if self.cache_config.destroy_pages_on_pause and not self._kv_cache_valid:
            self.runner.initialize_kv_cache()
            self._kv_cache_valid = True
            self._log_cache_event("kv_cache_reinitialized", {"reason": "resume"})
        self.initiate()

    def release_model_state(self, *, clear_compiled_cache: bool = False) -> None:
        """Release runner-held model weights/state references to reduce memory.

        The engine remains reusable. Call `update_model_weights(...)` before
        serving again.

        Args:
            clear_compiled_cache: Whether to clear compiled model/sampler caches.
        """
        if self.num_running_requests > 0 or self.num_pending_requests > 0:
            logger.warning(
                "Skipping model-state release because requests are active or pending (running=%d, pending=%d).",
                self.num_running_requests,
                self.num_pending_requests,
            )
            return

        if self._scheduler_running:
            self.pause()
        else:
            self._drain_pipeline_workers("release_model_state")

        self.runner.release_model_state(clear_compiled_cache=clear_compiled_cache)
        self._paused = True

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

        step_coordinator = getattr(self, "_step_coordinator", None)
        is_worker_rank = step_coordinator is not None and not step_coordinator.is_leader

        # A ZMQ worker rank runs a replay thread, not the scheduler loop, so
        # `_scheduler_running` is always False there; gate on the liveness
        # property that covers both flavours so the worker actually tears
        # down and rejoins the (restarted) leader across the swap.
        was_running = self._generation_alive
        if was_running:
            self.terminate()
            if is_worker_rank and step_coordinator is not None:
                # Worker `terminate()` stops the replay loop but returns
                # before the step-coordinator shutdown (there is no scheduler
                # loop to drain), leaving the DEALER connected to the
                # pre-restart leader. Force it closed so the re-`initiate()`
                # below re-handshakes with the leader — which resets its
                # authed set and ledger when the leader rank re-`start()`s in
                # lockstep — instead of replaying on a stale connection whose
                # frames are dropped as unauthenticated.
                try:
                    step_coordinator.shutdown("update_model_weights")
                except Exception:
                    logger.debug("Worker step coordinator shutdown failed", exc_info=True)

        self._drain_pipeline_workers("update_model_weights")

        split_graphdef = None
        split_graphstate = None
        split_graphother = None
        using_compatible_split = False
        if model is not None and (graphdef is None or graphstate is None or graphother is None):
            using_compatible_split = self._model_overrides_esurge_graphdef(model)
            _split_model, split_graphdef, split_graphstate, split_graphother = (
                self._split_graph_components_for_weight_update(model)
            )

        if model is None:
            model = spx.bind(graphdef, graphstate.overlay(graphother))
        if using_compatible_split and graphdef is None:
            if graphstate is None:
                graphstate = split_graphstate
            if graphother is None:
                graphother = split_graphother
        else:
            if graphstate is None:
                graphstate = split_graphstate
            if graphother is None:
                graphother = split_graphother
        if graphdef is None:
            graphdef = self._resolve_graphdef_for_weight_update(model, split_graphdef)

        self.runner.update_model_weights(
            graphdef=graphdef,
            graphstate=graphstate,
            graphother=graphother,
            reset_state=True,
        )
        if not self.runner.executor_manager.has_compiled_variants():
            # Compilation uses the current KV pages as a template for sharding/shape.
            # Ensure pages exist when coming back from a released model state.
            if self.runner.executor_manager.kv_pages is None:
                self.runner.initialize_kv_cache()
            self.runner.compile(max_num_batched_tokens=self._scheduler_max_num_batched_tokens)
        self._kv_cache_valid = self.runner.executor_manager.kv_pages is not None
        cache_event = "kv_cache_reinitialized" if self._kv_cache_valid else "kv_cache_destroyed"
        self._log_cache_event(cache_event, {"reason": "update_model_weights"})

        with self._request_lock, self._output_lock:
            self._active_requests.clear()
            self._request_outputs.clear()
            self._finished_request_ids.clear()
            self._request_events.clear()

        request_plane = getattr(self, "_request_plane", None)
        if request_plane is not None and request_plane.owns_scheduling:
            request_plane.reset()

        self.scheduler = Scheduler.from_runner(
            self.runner,
            max_num_batched_tokens=self._scheduler_max_num_batched_tokens,
            enable_prefix_caching=self.cache_config.enable_prefix_caching,
        )

        if restart_scheduler and was_running:
            self.initiate()

    def _reset_runner_state_if_idle(self, reason: str) -> None:
        """Reset runner buffers when there are no active/pending requests.

        Args:
            reason: Reason for the reset (for logging).
        """
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
        """Start a JAX profiler trace for the next num_batches scheduler updates.

        Enables JAX profiling to capture performance traces that can be visualized
        in TensorBoard or the Chrome trace viewer.

        Args:
            output_dir: Directory to write profiler output files.
            num_batches: Number of scheduler batches to trace before auto-stopping.
            host_tracer_level: Optional host tracer level (1-4). Higher levels
                capture more detail but add overhead.
            python_tracer_level: Optional Python tracer level for function-level
                profiling.

        Raises:
            RuntimeError: If a profiling session is already active.
            ValueError: If num_batches is not positive.
        """
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
        """Stop the active JAX profiler trace, if any.

        Safe to call even if profiling is not active - will no-op.
        Resets all profiling state.
        """
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

    def _handle_profiling_step(self) -> None:
        """Tick down the profiler step budget and auto-stop the trace at zero.

        Called from both scheduler-loop paths (sync and overlap) after
        each scheduler iteration completes. When a profiler trace is not
        active, returns immediately. Otherwise decrements
        ``_profiling_steps_remaining`` and, if it reaches zero, finalizes
        the trace via :meth:`stop_profiling` so users get a self-bounded
        profile without having to remember to stop it manually.
        """
        if not self._profiling_active:
            return
        if self._profiling_steps_remaining > 0:
            self._profiling_steps_remaining -= 1
        if self._profiling_steps_remaining <= 0:
            self.stop_profiling()


    def _ensure_scheduler_running(self, *, context: str) -> None:
        """Fail fast when the scheduler is not actually running.

        Streaming/blocking entry points spin in a wait loop on
        ``self._output_event``; if the scheduler was stopped quietly
        (manual :meth:`pause` / :meth:`terminate`) the loop would hang
        forever waiting for output that will never come. This helper
        runs once at the start of those entry points: when the scheduler
        is alive it returns immediately, when a recorded crash exists
        :meth:`_raise_if_scheduler_failed` re-raises that exception with
        the original traceback, and otherwise it raises a fresh
        :class:`RuntimeError` mentioning the calling ``context`` so logs
        say which API path discovered the dead scheduler.

        Args:
            context: Short label (``"generate-start"``,
                ``"stream-start"``, …) embedded into the error message
                so log scrapers can identify the originating call.
        """
        if self._generation_alive:
            return
        self._raise_if_scheduler_failed()
        raise RuntimeError(f"Background scheduler is not running ({context}).")

    def _recover_orphaned_request(self, request_id: str) -> bool:
        """Force-abort a request whose child samples vanished without a final output.

        ``n>1`` parallel sampling registers one scheduler-side child per
        sample. When a fatal scheduler error or out-of-order finish event
        leaves a child request gone from both ``scheduler.requests`` and
        ``self._active_requests`` *and* the parent's
        :class:`RequestOutput` still has unresolved samples (no
        ``finish_reason`` set), generate/stream loops would otherwise
        wait forever. This helper detects that state under careful
        triple-locking — releasing each lock between checks so a real
        completion can race in and avoid the abort — and only fires
        :meth:`abort_request` when every cross-check still confirms the
        orphan condition. Logs a WARNING when an abort is forced.

        Args:
            request_id: Parent request id that the calling generate/
                stream loop is currently waiting on.

        Returns:
            ``True`` when an abort was issued (caller should restart its
            poll loop), ``False`` when the request appears healthy.
        """
        with self._output_lock:
            ro = self._request_outputs.get(request_id)
            if ro is None or ro.finished:
                return False
            n_samples = len(ro.outputs)
            unresolved_child_ids: list[str] = []
            if n_samples <= 1:
                if ro.outputs and ro.outputs[0].finish_reason is None:
                    unresolved_child_ids.append(request_id)
            else:
                for sample_idx, comp in enumerate(ro.outputs):
                    if comp.finish_reason is None:
                        unresolved_child_ids.append(f"{request_id}-{sample_idx}")

        if not unresolved_child_ids:
            return False

        with self._scheduler_lock:
            if any(child_id in self.scheduler.requests for child_id in unresolved_child_ids):
                return False

        with self._request_lock:
            if any(child_id in self._active_requests for child_id in unresolved_child_ids):
                return False

        # Re-check under output lock to avoid aborting if completion just arrived.
        with self._output_lock:
            ro_now = self._request_outputs.get(request_id)
            if ro_now is None or ro_now.finished:
                return False
            if len(ro_now.outputs) != n_samples:
                return False
            if n_samples <= 1:
                still_pending = bool(ro_now.outputs and ro_now.outputs[0].finish_reason is None)
            else:
                still_pending = any(comp.finish_reason is None for comp in ro_now.outputs)
            if not still_pending:
                return False

        logger.warning(
            "Recovering orphaned request %s: unresolved child samples are missing from scheduler/active state. "
            "Forcing abort to avoid an indefinite wait.",
            request_id,
        )
        self.abort_request(request_id)
        return True

    @staticmethod
    def _build_tool_parser_request(
        *,
        prompt: str,
        tools: list[dict] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ):
        """Synthesize a stripped-down :class:`ChatCompletionRequest` for tool parsers.

        Some tool parsers consult the request's tool schema and
        ``tool_choice`` selector to disambiguate model output (e.g.
        forced-function calls vs free-form). Internal entry points like
        :meth:`generate` and :meth:`stream` only have a flat prompt
        string, so this helper assembles a synthetic
        :class:`ChatCompletionRequest` carrying a single user message
        plus the engine-normalized tool list / choice. Returns ``None``
        when no tools and no choice were supplied — the parser will then
        fall back to its tool-free behaviour. Each tool dict is
        converted into the OpenAI ``{"type": "function", "function":
        {...}}`` wrapper shape the parsers expect, regardless of
        whether the caller passed the wrapped or unwrapped form.

        Args:
            prompt: Already-formatted prompt string used as the synthetic
                user message content.
            tools: Optional list of tool definitions; pydantic models
                are accepted and ``model_dump``-ed.
            tool_choice: Optional ``"auto"`` / ``"none"`` / ``"required"``
                literal or full ``{"type": "function", ...}`` selector.

        Returns:
            A validated :class:`ChatCompletionRequest`, or ``None`` when
            both inputs were ``None``.
        """
        if tools is None and tool_choice is None:
            return None

        from easydel.inference.openai_api_modules import ChatCompletionRequest

        normalized_tools: list[dict[str, Any]] | None = None
        if tools is not None:
            normalized_tools = []
            for tool in tools:
                if hasattr(tool, "model_dump"):
                    raw_tool = tool.model_dump(exclude_none=True)
                elif isinstance(tool, dict):
                    raw_tool = dict(tool)
                else:
                    continue

                function_payload = raw_tool.get("function")
                if isinstance(function_payload, dict):
                    normalized_tools.append(
                        {
                            "type": str(raw_tool.get("type") or "function"),
                            "function": function_payload,
                        }
                    )
                    continue

                if isinstance(raw_tool.get("name"), str):
                    normalized_tools.append(
                        {
                            "type": "function",
                            "function": raw_tool,
                        }
                    )

        return ChatCompletionRequest.model_validate(
            {
                "model": "dummy",
                "messages": [{"role": "user", "content": prompt}],
                "tools": normalized_tools,
                "tool_choice": tool_choice,
            }
        )

    def generate(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
        request_id: str | list[str] | None = None,
        use_tqdm: bool = True,
        tool_parser_request: Any | None = None,
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
        elif len(prompts) == 0:
            raise ValueError("Empty prompt list provided")

        if request_id is None:
            request_ids = [self._generate_request_id() for _ in prompts]
        elif isinstance(request_id, str):
            if len(prompts) != 1:
                raise ValueError("request_id must be a list when providing multiple prompts.")
            request_ids = [request_id]
        else:
            request_ids = list(request_id)
            if len(request_ids) != len(prompts):
                raise ValueError("Length of request_id list must match number of prompts.")

        base_sampling_params = sampling_params or SamplingParams(max_tokens=128)

        defer_batch_enqueue = len(prompts) > 1
        deferred_scheduler_requests = []
        for prompt, req_id in zip(prompts, request_ids, strict=True):
            prompt_tokens = self._tokenize_prompt(req_id, prompt)
            effective_params = self._prepare_sampling_params_for_request(
                base_sampling_params,
                request_id=req_id,
                prompt=prompt,
            )
            maybe_scheduler_requests = self._add_request(
                req_id,
                prompt,
                effective_params,
                prompt_token_ids=prompt_tokens,
                tool_parser_request=tool_parser_request,
                defer_scheduler_enqueue=defer_batch_enqueue,
            )
            if maybe_scheduler_requests:
                deferred_scheduler_requests.extend(maybe_scheduler_requests)

        if deferred_scheduler_requests:
            self._submit_scheduler_requests(deferred_scheduler_requests)
            logger.info(
                "Queued generate batch: parent_requests=%d scheduler_requests=%d",
                len(prompts),
                len(deferred_scheduler_requests),
            )

        outputs = []
        pbar = None
        if use_tqdm:
            from tqdm import tqdm

            pbar = tqdm(total=len(prompts), desc="Generating")

        completed = set()

        self._ensure_scheduler_running(context="generate-start")

        while len(completed) < len(prompts):
            self._output_event.wait(timeout=0.1)
            self._output_event.clear()
            self._raise_if_scheduler_failed()
            if not self._generation_alive:
                pending_ids = [rid for rid in request_ids if rid not in completed]
                for rid in pending_ids:
                    self.abort_request(rid)
                raise RuntimeError("Background scheduler stopped while waiting for generation outputs.")
            for req_id in request_ids:
                if req_id not in completed:
                    self._recover_orphaned_request(req_id)
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

        # Cleanup per-request events (outputs retained per max_request_outputs).
        with self._request_lock:
            for output in outputs:
                rid = output.request_id
                self._request_events.pop(rid, None)
                n_samples = len(output.outputs)
                if n_samples > 1:
                    for sample_idx in range(n_samples):
                        self._request_events.pop(f"{rid}-{sample_idx}", None)
        if self._max_request_outputs is not None:
            with self._output_lock:
                for output in outputs:
                    self._track_finished_output(output.request_id)
        return outputs

    def stream(
        self,
        prompts: str | list[str],
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        tool_parser_request: Any | None = None,
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
        self._add_request(
            request_id,
            prompt,
            effective_params,
            prompt_token_ids=prompt_tokens,
            tool_parser_request=tool_parser_request,
        )

        self._ensure_scheduler_running(context="stream-start")

        with self._request_lock:
            req_event = self._request_events.get(request_id)
        if req_event is None:
            raise RuntimeError("Request event missing")

        last_update_seq = -1
        last_accumulated_text = ""
        last_raw_accumulated_text = ""
        last_accumulated_reasoning = ""
        try:
            while True:
                req_event.wait(timeout=1.0)
                req_event.clear()
                self._raise_if_scheduler_failed()
                if not self._generation_alive:
                    self.abort_request(request_id)
                    raise RuntimeError("Background scheduler stopped while streaming request.")
                if self._recover_orphaned_request(request_id):
                    continue

                snapshot = None
                with self._output_lock:
                    ro = self._request_outputs.get(request_id)
                    if ro is None:
                        continue

                    if ro.update_seq != last_update_seq:
                        # Snapshot without holding the lock during yield
                        finished_snapshot = bool(ro.finished)
                        outputs_copy = []
                        for comp in ro.outputs:
                            token_ids = list(comp.token_ids) if finished_snapshot else comp.token_ids
                            logprobs = (
                                [dict(lp) for lp in comp.logprobs]
                                if finished_snapshot and comp.logprobs
                                else comp.logprobs
                            )
                            outputs_copy.append(
                                CompletionOutput(
                                    index=comp.index,
                                    text=comp.text,
                                    token_ids=token_ids,
                                    cumulative_logprob=comp.cumulative_logprob,
                                    logprobs=logprobs,
                                    finish_reason=comp.finish_reason,
                                    tool_calls=comp.tool_calls,
                                    reasoning_content=comp.reasoning_content,
                                    raw_text=comp.raw_text,
                                )
                            )

                        snapshot_delta = self._compute_snapshot_delta_text(
                            ro.accumulated_text,
                            last_accumulated_text,
                            ro.delta_text,
                        )
                        snapshot_raw_delta = self._compute_snapshot_delta_text(
                            ro.raw_accumulated_text,
                            last_raw_accumulated_text,
                            ro.raw_delta_text or "",
                        )
                        snapshot_reasoning_delta = self._compute_snapshot_delta_text(
                            ro.reasoning_content or "",
                            last_accumulated_reasoning,
                            ro.delta_reasoning_content or "",
                        )

                        snapshot = RequestOutput(
                            request_id=ro.request_id,
                            prompt=ro.prompt,
                            prompt_token_ids=list(ro.prompt_token_ids),
                            outputs=outputs_copy,
                            finished=ro.finished,
                            metrics=dict(ro.metrics) if ro.metrics is not None else None,
                            accumulated_text=ro.accumulated_text,
                            delta_text=snapshot_delta,
                            raw_accumulated_text=ro.raw_accumulated_text,
                            raw_delta_text=snapshot_raw_delta,
                            tokens_per_second=ro.tokens_per_second,
                            num_generated_tokens=ro.num_generated_tokens,
                            time_spent_generating=ro.time_spent_generating,
                            first_token_time=ro.first_token_time,
                            processing_time=ro.processing_time,
                            update_seq=ro.update_seq,
                            delta_seq=ro.delta_seq,
                            tool_calls=ro.tool_calls,
                            delta_tool_calls=ro.delta_tool_calls,
                            reasoning_content=ro.reasoning_content,
                            delta_reasoning_content=snapshot_reasoning_delta or None,
                        )
                        last_update_seq = ro.update_seq
                        last_accumulated_text = ro.accumulated_text
                        last_raw_accumulated_text = ro.raw_accumulated_text
                        last_accumulated_reasoning = ro.reasoning_content or ""

                if snapshot is not None:
                    if (
                        not snapshot.finished
                        and snapshot.num_generated_tokens == 0
                        and not snapshot.delta_text
                        and not snapshot.raw_delta_text
                        and not snapshot.delta_reasoning_content
                        and not snapshot.delta_tool_calls
                    ):
                        continue
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
                if self._max_request_outputs is not None:
                    with self._output_lock:
                        self._track_finished_output(request_id)
            else:
                self.abort_request(request_id)

    @overload
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        *,
        stream: Literal[False] = ...,
        chat_template: str | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> RequestOutput:
        """Overload: blocking chat call returning a single :class:`RequestOutput`."""
        ...

    @overload
    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        *,
        stream: Literal[True],
        chat_template: str | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> Iterator[RequestOutput]:
        """Overload: streaming chat call yielding incremental :class:`RequestOutput`s."""
        ...

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        stream: bool = False,
        chat_template: str | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> RequestOutput | Iterator[RequestOutput]:
        """High-level chat interface compatible with OpenAI APIs.

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
            in the model loading config:
            eSurge(model="model-id", processor=AutoProcessor.from_pretrained(...))
        """
        has_multimodal = self._messages_have_multimodal_content(messages)

        if has_multimodal:
            return self._chat_multimodal(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                sampling_params=sampling_params,
                request_id=request_id,
                stream=stream,
                chat_template=chat_template,
                chat_template_kwargs=chat_template_kwargs,
            )
        else:
            base_sampling_params = self._prepare_chat_sampling_params(
                sampling_params or SamplingParams(max_tokens=128),
                tools=tools,
                tool_choice=tool_choice if isinstance(tool_choice, str) else None,
            )
            prompt = self._format_chat_prompt(
                messages,
                tools=tools,
                add_generation_prompt=True,
                chat_template=chat_template,
                chat_template_kwargs=chat_template_kwargs,
            )
            tool_parser_request = self._build_tool_parser_request(
                prompt=prompt,
                tools=tools,
                tool_choice=tool_choice,
            )

            if stream:
                return self.stream(
                    prompt,
                    sampling_params=base_sampling_params,
                    request_id=request_id,
                    tool_parser_request=tool_parser_request,
                )
            else:
                outs = self.generate(
                    prompt,
                    sampling_params=base_sampling_params,
                    request_id=request_id,
                    use_tqdm=False,
                    tool_parser_request=tool_parser_request,
                )
                return outs[0]

    def iter_chat_completion_stream(
        self,
        *,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        chat_template: str | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> Iterator[ChatCompletionStreamResponse]:
        """Bridge :meth:`chat` snapshots into OpenAI ``ChatCompletionStreamResponse`` frames.

        Drives :meth:`chat` in streaming mode and pipes its
        :class:`RequestOutput` snapshots through
        :func:`iter_chat_completion_stream_responses`, which translates
        each delta (text, tool calls, reasoning) into the OpenAI Chat
        Completions stream chunk shape (``chat.completion.chunk``).
        Callers can plug the resulting iterator straight into an SSE
        endpoint without further translation.

        Args:
            model: Model id echoed back into every stream frame's
                ``model`` field.
            messages: Conversation messages forwarded to :meth:`chat`.
            tools, tool_choice, sampling_params, request_id,
                chat_template, chat_template_kwargs: Same semantics as
                :meth:`chat`.

        Yields:
            ``ChatCompletionStreamResponse`` frames, one per snapshot,
            terminating with the final frame carrying ``finish_reason``.
        """

        outputs = self.chat(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            sampling_params=sampling_params,
            request_id=request_id,
            stream=True,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
        )
        yield from iter_chat_completion_stream_responses(
            typing.cast(Iterator[RequestOutputLike], outputs),
            model=model,
        )

    def iter_responses_stream(
        self,
        *,
        response_id: str,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        include_reasoning_summary: bool = False,
        final_response_overrides: ResponsesFinalizationOptions | None = None,
        created_at: int | None = None,
        chat_template: str | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> Iterator[StreamEventFrame]:
        """Bridge :meth:`chat` snapshots into OpenAI Responses-API stream frames.

        Sister to :meth:`iter_chat_completion_stream`, but emits the
        richer Responses-API event protocol
        (``response.created``, ``response.output_item.delta``,
        ``response.output_item.reasoning_summary.delta``, …). Wraps
        :meth:`chat` in streaming mode and pipes its snapshots through
        :func:`iter_responses_stream_frames`, which keeps track of
        response/output ids and emits both the per-event frames and the
        terminal ``response.completed`` envelope.

        Args:
            response_id: Stable response id echoed in every emitted frame.
            model: Model id echoed in the response envelope.
            include_reasoning_summary: When ``True``, also emits
                ``reasoning_summary`` items derived from the model's
                reasoning content.
            final_response_overrides: Optional finalization tweaks
                applied to the terminating frame (status,
                ``incomplete_details``, etc.).
            created_at: Optional fixed ``created_at`` epoch; ``None``
                uses the iteration's current time.
            messages, tools, tool_choice, sampling_params, request_id,
                chat_template, chat_template_kwargs: Same semantics as
                :meth:`chat`.

        Yields:
            :class:`StreamEventFrame` events compatible with the OpenAI
            Responses-API SSE protocol.
        """

        outputs = self.chat(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            sampling_params=sampling_params,
            request_id=request_id,
            stream=True,
            chat_template=chat_template,
            chat_template_kwargs=chat_template_kwargs,
        )
        yield from iter_responses_stream_frames(
            typing.cast(Iterator[RequestOutputLike], outputs),
            response_id=response_id,
            model=model,
            include_reasoning_summary=include_reasoning_summary,
            final_response_overrides=final_response_overrides,
            created_at=created_at,
        )

    def _messages_have_multimodal_content(self, messages: list[dict]) -> bool:
        """Detect whether any chat turn carries image or video content.

        Walks every message's ``content`` list looking for typed parts
        whose ``type`` matches the recognized media types
        (``"image"``, ``"image_url"``, ``"input_image"``, ``"video"``,
        ``"video_url"``, ``"input_video"``) or that contain bare media
        keys. Drives the routing decision in :meth:`chat` between the
        pure-text fast path and the multimodal helper
        :meth:`_chat_multimodal`.

        Args:
            messages: Chat messages exactly as the user supplied them
                (no normalization required).

        Returns:
            ``True`` if any part contains image / video content,
            ``False`` for text-only conversations.
        """
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
        tool_choice: str | dict[str, Any] | None = None,
        sampling_params: SamplingParams | None = None,
        request_id: str | None = None,
        stream: bool = False,
        chat_template: str | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> RequestOutput | Iterator[RequestOutput]:
        """Handle multimodal chat with images/videos.

        Internal method that processes vision-language model requests.

        Args:
            messages: Chat messages with multimodal content.
            tools: Optional tool definitions.
            sampling_params: Generation parameters.
            request_id: Optional request ID.
            stream: Whether to stream output.
            chat_template: Optional custom chat template.

        Returns:
            RequestOutput (non-streaming) or Iterator[RequestOutput] (streaming).

        Raises:
            ValueError: If no processor is configured for multimodal content.
        """
        if self._multimodal_manager is None:
            raise ValueError(
                "Multimodal content detected but no processor configured. "
                "Initialize eSurge with: processor=<tokenizer-or-processor> (e.g. AutoProcessor/AutoTokenizer)."
            )

        if request_id is None:
            request_id = self._generate_request_id()

        base_sampling_params = self._prepare_chat_sampling_params(
            sampling_params or SamplingParams(max_tokens=128),
            tools=tools,
            tool_choice=tool_choice if isinstance(tool_choice, str) else None,
        )

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
            chat_template_kwargs=chat_template_kwargs,
        )

        effective_params = self._prepare_sampling_params_for_request(
            base_sampling_params,
            request_id=request_id,
            prompt=prompt,
        )
        tool_parser_request = self._build_tool_parser_request(
            prompt=prompt,
            tools=tools,
            tool_choice=tool_choice,
        )

        # Add request with vision data
        self._add_request(
            request_id=request_id,
            prompt=prompt,
            sampling_params=effective_params,
            prompt_token_ids=prompt_token_ids,
            tool_parser_request=tool_parser_request,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            mm_features=mm_features,
        )

        self._ensure_scheduler_running(context="multimodal-chat-start")

        if stream:
            return self._stream_multimodal_request(request_id)
        else:
            return self._wait_for_request(request_id)

    def _stream_multimodal_request(self, request_id: str) -> Iterator[RequestOutput]:
        """Stream :class:`RequestOutput` snapshots for an already-admitted multimodal request.

        Mirror of :meth:`stream` for the multimodal path: the caller has
        already submitted the request via :meth:`_chat_multimodal`, so
        this helper only owns the read loop. On every wakeup it
        snapshots the live ``_request_outputs`` entry (under
        ``_output_lock``), copies the per-sample fields into a fresh
        :class:`RequestOutput`, computes monotonically-correct
        ``delta_text`` / ``raw_delta_text`` /
        ``delta_reasoning_content`` against the previous snapshot, and
        yields the result. Exits when ``finished`` flips true; in the
        ``finally`` block, cleans up per-request events and either
        records the completed output for retention or aborts the
        request when the consumer abandons mid-stream.

        Args:
            request_id: Parent request id returned by
                :meth:`_chat_multimodal`.

        Yields:
            New :class:`RequestOutput` snapshots whose ``delta_*`` fields
            describe what changed since the last yield.

        Raises:
            RuntimeError: If the per-request event is missing (indicates
                the engine never registered this id) or the scheduler
                stops mid-stream.
        """
        with self._request_lock:
            req_event = self._request_events.get(request_id)
        if req_event is None:
            raise RuntimeError("Request event missing")

        last_update_seq = -1
        last_accumulated_text = ""
        last_raw_accumulated_text = ""
        last_accumulated_reasoning = ""
        try:
            while True:
                req_event.wait(timeout=1.0)
                req_event.clear()
                self._raise_if_scheduler_failed()
                if not self._generation_alive:
                    self.abort_request(request_id)
                    raise RuntimeError("Background scheduler stopped while streaming multimodal request.")
                if self._recover_orphaned_request(request_id):
                    continue

                snapshot = None
                with self._output_lock:
                    ro = self._request_outputs.get(request_id)
                    if ro is None:
                        continue

                    if ro.update_seq != last_update_seq:
                        finished_snapshot = bool(ro.finished)
                        outputs_copy = []
                        for comp in ro.outputs:
                            token_ids = list(comp.token_ids) if finished_snapshot else comp.token_ids
                            logprobs = (
                                [dict(lp) for lp in comp.logprobs]
                                if finished_snapshot and comp.logprobs
                                else comp.logprobs
                            )
                            outputs_copy.append(
                                CompletionOutput(
                                    index=comp.index,
                                    text=comp.text,
                                    token_ids=token_ids,
                                    cumulative_logprob=comp.cumulative_logprob,
                                    logprobs=logprobs,
                                    finish_reason=comp.finish_reason,
                                    tool_calls=comp.tool_calls,
                                    reasoning_content=comp.reasoning_content,
                                    raw_text=comp.raw_text,
                                )
                            )

                        snapshot_delta = self._compute_snapshot_delta_text(
                            ro.accumulated_text,
                            last_accumulated_text,
                            ro.delta_text,
                        )
                        snapshot_raw_delta = self._compute_snapshot_delta_text(
                            ro.raw_accumulated_text,
                            last_raw_accumulated_text,
                            ro.raw_delta_text or "",
                        )
                        snapshot_reasoning_delta = self._compute_snapshot_delta_text(
                            ro.reasoning_content or "",
                            last_accumulated_reasoning,
                            ro.delta_reasoning_content or "",
                        )

                        snapshot = RequestOutput(
                            request_id=ro.request_id,
                            prompt=ro.prompt,
                            prompt_token_ids=list(ro.prompt_token_ids),
                            outputs=outputs_copy,
                            finished=ro.finished,
                            metrics=dict(ro.metrics) if ro.metrics is not None else None,
                            accumulated_text=ro.accumulated_text,
                            delta_text=snapshot_delta,
                            raw_accumulated_text=ro.raw_accumulated_text,
                            raw_delta_text=snapshot_raw_delta,
                            tokens_per_second=ro.tokens_per_second,
                            num_generated_tokens=ro.num_generated_tokens,
                            time_spent_generating=ro.time_spent_generating,
                            first_token_time=ro.first_token_time,
                            processing_time=ro.processing_time,
                            update_seq=ro.update_seq,
                            delta_seq=ro.delta_seq,
                            tool_calls=ro.tool_calls,
                            delta_tool_calls=ro.delta_tool_calls,
                            reasoning_content=ro.reasoning_content,
                            delta_reasoning_content=snapshot_reasoning_delta or None,
                        )
                        last_update_seq = ro.update_seq
                        last_accumulated_text = ro.accumulated_text
                        last_raw_accumulated_text = ro.raw_accumulated_text
                        last_accumulated_reasoning = ro.reasoning_content or ""

                if snapshot is not None:
                    if (
                        not snapshot.finished
                        and snapshot.num_generated_tokens == 0
                        and not snapshot.delta_text
                        and not snapshot.raw_delta_text
                        and not snapshot.delta_reasoning_content
                        and not snapshot.delta_tool_calls
                    ):
                        continue
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
            else:
                self.abort_request(request_id)

    def _wait_for_request(self, request_id: str) -> RequestOutput:
        """Block until a multimodal request finishes and return the final output.

        Sibling of :meth:`_stream_multimodal_request` for non-streaming
        callers. Loops on the per-request event with a one-second
        timeout, periodically calling
        :meth:`_recover_orphaned_request` so child-sample disappearances
        cannot stall the wait, and exits as soon as the underlying
        ``RequestOutput`` is finished. On exit, removes the per-request
        events (parent and per-sample) and records the output for
        retention bookkeeping when ``max_request_outputs`` is set.

        Args:
            request_id: Parent request id returned by
                :meth:`_chat_multimodal`.

        Returns:
            The final :class:`RequestOutput` with every sample's text /
            tool calls / reasoning fully populated.

        Raises:
            RuntimeError: If the per-request event is missing or the
                scheduler stops while waiting.
        """
        with self._request_lock:
            req_event = self._request_events.get(request_id)
        if req_event is None:
            raise RuntimeError("Request event missing")

        output: RequestOutput | None = None
        while True:
            req_event.wait(timeout=1.0)
            req_event.clear()
            self._raise_if_scheduler_failed()
            if not self._generation_alive:
                self.abort_request(request_id)
                raise RuntimeError("Background scheduler stopped while waiting for request completion.")
            if self._recover_orphaned_request(request_id):
                continue
            with self._output_lock:
                output = self._request_outputs.get(request_id)
                if output is not None and output.finished:
                    break

        # Request is finished; cleanup per-request events (retention honors max_request_outputs).
        n_samples = len(output.outputs) if output is not None else 0
        with self._request_lock:
            self._request_events.pop(request_id, None)
            if n_samples > 1:
                for sample_idx in range(n_samples):
                    self._request_events.pop(f"{request_id}-{sample_idx}", None)

        assert output is not None
        if self._max_request_outputs is not None:
            with self._output_lock:
                self._track_finished_output(request_id)
        return output


    def start_monitoring(self, **kwargs) -> dict[str, str]:
        """Start Prometheus-based monitoring for the engine.

        Delegates to :meth:`MonitoringStack.start_monitoring`; see there for
        the full keyword-argument reference (Prometheus port, Grafana
        auto-start and provisioning, console monitor, metric logging).

        Args:
            **kwargs: Forwarded verbatim to
                :meth:`MonitoringStack.start_monitoring`.

        Returns:
            Dictionary of service URLs (``prometheus``, ``grafana`` when
            started).
        """
        return self._monitoring.start_monitoring(**kwargs)

    def stop_monitoring(self) -> None:
        """Stop all monitoring services.

        Gracefully shuts down Prometheus server and console monitor
        if they are running.
        """
        self._monitoring.stop_monitoring()

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
        """Check if monitoring services are currently active.

        Returns:
            True if monitoring has been initialized and is running.
        """
        return self._monitoring.monitoring_active
