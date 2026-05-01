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
import subprocess
import threading
import time
import typing
from collections import deque
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import jax
from jax import numpy as jnp
from spectrax.common_types import NOT_GIVEN
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from easydel.axis import register_attention_data_parallel_axis
from easydel.inference.sampling_params import SamplingParams
from easydel.utils import Registry

if typing.TYPE_CHECKING:
    from easydel.modules.auto.auto_modeling import PreTrainedLoading
from easydel.workers.esurge.pipeline import WorkerManager

from .config import (
    eSurgeCacheRuntimeConfig,
    eSurgeContextConfig,
    eSurgeDistributedConfig,
    eSurgeParsingConfig,
    eSurgeRuntimeConfig,
    eSurgeVisionConfig,
    eSurgeWorkerConfig,
)
from .distributed import DistributedController, make_config_fingerprint, resolve_distributed_role
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
    from easydel.inference.reasoning.abstract_reasoning import ReasoningParserName
    from easydel.inference.tools.abstract_tool import ToolParserName
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
MLA_RAGGED_ATTN_MECHANISM = "multi_latent_ragged_page_attention_v2"
_MLA_RAGGED_ATTN_MECHANISMS = {
    "multi_latent_ragged_page_attention_v1",
    "multi_latent_ragged_page_attention_v2",
}


def _set_requested_new(sp, n: int):  # pyright: ignore[reportUnusedFunction]
    """Set the generation length on a SamplingParams object.

    Attempts to set both ``max_tokens`` and ``max_new_tokens`` attributes
    when they exist, ensuring compatibility across different SamplingParams
    variants.

    Args:
        sp: A SamplingParams-like object whose token-limit fields will be
            mutated in place.
        n: The desired number of new tokens to generate.
    """
    if hasattr(sp, "max_tokens"):
        sp.max_tokens = int(n)
    if hasattr(sp, "max_new_tokens"):
        sp.max_new_tokens = int(n)


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


def _normalize_attn_mechanism_value(attn_mechanism: Any) -> str | None:
    """Normalize an attention mechanism identifier to a plain string.

    Handles enum-like objects (with a ``.value`` attribute) as well as
    raw strings.

    Args:
        attn_mechanism: An attention mechanism identifier. May be ``None``,
            a string, or an enum with a ``value`` attribute.

    Returns:
        The string representation of the mechanism, or ``None`` if the
        input is ``None``.
    """
    if attn_mechanism is None:
        return None
    if hasattr(attn_mechanism, "value"):
        attn_mechanism = attn_mechanism.value
    return str(attn_mechanism)


def _text_config_uses_mla(text_config: Any) -> bool:
    """Detect whether a text config indicates Multi-Latent Attention (MLA).

    Uses several heuristics in order of priority:
    1. Explicit ``attn_mechanism`` or ``mla_attn_mechanism`` matching the
       MLA ragged attention constant.
    2. An ``attention_type`` attribute equal to ``"mla"``.
    3. A callable or boolean ``is_mla`` attribute.
    4. Presence of ``kv_lora_rank`` (or ``kv_lora_dim``) together with
       ``qk_rope_head_dim`` or ``qk_nope_head_dim``.

    Args:
        text_config: A model text configuration object (e.g.,
            ``PretrainedConfig``). May be ``None``.

    Returns:
        ``True`` if the config signals MLA usage, ``False`` otherwise.
    """
    if text_config is None:
        return False

    attn_mechanism = _normalize_attn_mechanism_value(getattr(text_config, "attn_mechanism", None))
    if attn_mechanism in _MLA_RAGGED_ATTN_MECHANISMS:
        return True
    mla_attn_mechanism = _normalize_attn_mechanism_value(getattr(text_config, "mla_attn_mechanism", None))
    if mla_attn_mechanism in _MLA_RAGGED_ATTN_MECHANISMS:
        return True

    attention_type = getattr(text_config, "attention_type", None)
    if attention_type is not None and str(attention_type).lower() == "mla":
        return True

    is_mla_attr = getattr(text_config, "is_mla", None)
    try:
        if callable(is_mla_attr):
            if bool(is_mla_attr()):
                return True
        elif is_mla_attr is not None and bool(is_mla_attr):
            return True
    except Exception:
        pass

    kv_lora_rank = getattr(text_config, "kv_lora_rank", None)
    if kv_lora_rank is None:
        kv_lora_rank = getattr(text_config, "kv_lora_dim", None)

    qk_rope_head_dim = getattr(text_config, "qk_rope_head_dim", None)
    qk_nope_head_dim = getattr(text_config, "qk_nope_head_dim", None)

    if kv_lora_rank is None or (qk_rope_head_dim is None and qk_nope_head_dim is None):
        return False

    try:
        return int(kv_lora_rank) > 0
    except Exception:
        return True


def _detect_mla_attention_mix(model: Any, text_config: Any = None) -> tuple[bool, bool]:
    """Detect whether a model contains MLA and/or non-MLA attention blocks.

    Traverses all ``UnifiedAttention`` sub-modules in *model* and inspects
    each one's ``attention_type``. Falls back to config-level heuristics
    via ``_text_config_uses_mla`` when no attention modules are found.

    Args:
        model: An EasyDeL model instance to inspect.
        text_config: Optional text configuration used as a fallback when
            module traversal yields no results.

    Returns:
        A ``(has_mla, has_non_mla)`` tuple of booleans indicating
        whether MLA and/or standard attention blocks were detected.
    """
    has_mla_attention = False
    has_non_mla_attention = False

    try:
        from easydel.layers.attention import UnifiedAttention
        from easydel.utils.traversals import iter_module_search

        for _, module in iter_module_search(model, UnifiedAttention):
            attention_type = str(getattr(module, "attention_type", "standard")).lower()
            if attention_type == "mla":
                has_mla_attention = True
            else:
                has_non_mla_attention = True
            if has_mla_attention and has_non_mla_attention:
                break
    except Exception:
        pass

    if not has_mla_attention and not has_non_mla_attention and _text_config_uses_mla(text_config):
        has_mla_attention = True

    return has_mla_attention, has_non_mla_attention


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

    @staticmethod
    def _auto_detect_tool_parser(
        *,
        tokenizer: PreTrainedTokenizerBase | None,
        model_type: str | None,
    ) -> ToolParserName | None:
        """Infer the tool parser from tokenizer/template hints and model type."""

        from easydel.inference.tools.auto_detect import detect_tool_parser

        detected = detect_tool_parser(model_type=model_type, tokenizer=tokenizer)
        return detected or None

    @staticmethod
    def _auto_detect_reasoning_parser_name(
        *,
        tokenizer: PreTrainedTokenizerBase | None,
        model_type: str | None,
    ) -> ReasoningParserName | None:
        """Infer the reasoning parser from tokenizer/template hints and model type."""

        from easydel.inference.reasoning.auto_detect import detect_reasoning_parser

        detected = detect_reasoning_parser(model_type=model_type, tokenizer=tokenizer)
        return detected or None

    def __init__(
        self,
        *,
        model: str | EasyDeLBaseModule,
        processor: Any | None = None,
        tokenizer: str | PreTrainedTokenizerBase | None = None,
        loading_kwargs: PreTrainedLoading | typing.Mapping[str, Any] | None = None,
        runtime: eSurgeRuntimeConfig | typing.Mapping[str, Any] | None = None,
        cache: eSurgeCacheRuntimeConfig | typing.Mapping[str, Any] | None = None,
        context: eSurgeContextConfig | typing.Mapping[str, Any] | None = None,
        workers: eSurgeWorkerConfig | typing.Mapping[str, Any] | None = None,
        parsing: eSurgeParsingConfig | typing.Mapping[str, Any] | None = None,
        vision: eSurgeVisionConfig | typing.Mapping[str, Any] | None = None,
        distributed: eSurgeDistributedConfig | typing.Mapping[str, Any] | None = None,
    ):
        """Initialize the eSurge engine.

        Args:
            model: Model id/path to load, or an already loaded EasyDeL model.
            processor: Unified text/multimodal processor. Can be a tokenizer or
                HF processor. When omitted for a string model, it is loaded from
                the model id/path.
            tokenizer: Deprecated alias/fallback for `processor`.
            loading_kwargs: Optional pretrained-loader kwargs used only when
                `model` is a string model id/path.
            runtime: Runtime and execution config.
            cache: KV-cache config.
            context: Context-window handling config.
            workers: Tokenizer/detokenizer worker config.
            parsing: Tool/reasoning parser config.
            vision: Multimodal config.
            distributed: Distributed serving config.

        Raises:
            ValueError: If processor/tokenizer cannot be inferred, or if
                configuration is invalid.
        """
        from easydel.infra import EasyDeLBaseConfigDict
        from easydel.layers.attention import AttentionMechanisms
        from easydel.modules.auto import AutoEasyDeLModelForCausalLM
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
        if tokenizer is None:
            tokenizer = loading_data.pop("tokenizer", None)
        else:
            loading_data.pop("tokenizer", None)
        loading_data["pretrained_model_name_or_path"] = model
        loading_data["processor"] = processor
        loading_data["tokenizer"] = tokenizer
        self.loading_kwargs = PreTrainedLoading.coerce_config(loading_data)
        self.runtime_config = eSurgeRuntimeConfig.coerce_config(runtime)
        self.cache_config = eSurgeCacheRuntimeConfig.coerce_config(cache)
        self.context_config = eSurgeContextConfig.coerce_config(context)
        self.worker_config = eSurgeWorkerConfig.coerce_config(workers)
        self.parsing_config = eSurgeParsingConfig.coerce_config(parsing)
        self.vision_config = eSurgeVisionConfig.coerce_config(vision)
        self.distributed_config = eSurgeDistributedConfig.coerce_config(distributed)

        # Backward-compatible public aliases.  The sectioned configs above are
        # the source of truth, but server/eval adapters still read these names.
        self.silent_mode = bool(self.parsing_config.silent_mode)
        self.max_model_len = self.runtime_config.max_model_len
        self.max_num_seqs = self.runtime_config.max_num_seqs
        self.page_size = self.cache_config.page_size
        self.enable_window_aware_runtime_cap = self.runtime_config.enable_window_aware_runtime_cap
        self.distributed_mode = bool(self.distributed_config.distributed_mode)
        self._overlap_execution = self.runtime_config.overlap_execution
        self._scheduler_enable_prefix_caching = self.cache_config.enable_prefix_caching
        self._min_input_pad = self.runtime_config.min_input_pad
        self._max_num_seqs = self.runtime_config.max_num_seqs
        self._max_num_batched_tokens = self.runtime_config.max_num_batched_tokens
        self._hbm_utilization = self.cache_config.hbm_utilization
        self._page_size = self.cache_config.page_size
        self._enable_prefix_caching = self.cache_config.enable_prefix_caching
        self._runner_verbose = self.runtime_config.runner_verbose
        self._decode_truncated_prompt = self.context_config.decode_truncated_prompt
        self._destroy_pages_on_pause = self.cache_config.destroy_pages_on_pause
        self._sampling_params_callback = self.parsing_config.sampling_params_callback
        self.ignore_stop_strings_in_reasoning = bool(self.parsing_config.ignore_stop_strings_in_reasoning)

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
        self._distributed_controller: DistributedController | None = None

        if self.distributed_config.distributed_mode:
            if not self.distributed_config.distributed_auth_token:
                raise ValueError("`distributed_auth_token` must be provided when distributed_mode=True.")
            if self.distributed_config.distributed_world_size is None:
                raise ValueError("`distributed_world_size` must be provided when distributed_mode=True.")
            if self.runtime_config.overlap_execution:
                raise ValueError(
                    "`overlap_execution=True` is not supported with distributed_mode=True. "
                    "Use overlap_execution=False for lockstep multi-host serving."
                )

            world_size = self.distributed_config.distributed_world_size
            if world_size <= 0:
                raise ValueError(f"`distributed_world_size` must be positive, got {world_size}.")

            rank = (
                self.distributed_config.distributed_rank
                if self.distributed_config.distributed_rank is not None
                else int(jax.process_index())
            )
            if rank < 0 or rank >= world_size:
                raise ValueError(
                    f"`distributed_rank` out of range: rank={rank}, world_size={world_size}. "
                    "Ensure JAX distributed init and DNS membership agree."
                )

            self.distributed_role = resolve_distributed_role(self.distributed_config.distributed_role, rank)
            self.distributed_world_size = world_size
            self.distributed_rank = rank
        else:
            self.distributed_role = "leader"
            self.distributed_world_size = 1
            self.distributed_rank = 0

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

        self._worker_manager = WorkerManager(tokenizer_source, startup_timeout=self.worker_config.worker_startup_timeout)
        self._tokenizer_client, self._detokenizer_client = self._worker_manager.start(
            detokenizer_max_states=self.worker_config.detokenizer_max_states,
            tokenizer_endpoint=tokenizer_endpoint,
            detokenizer_endpoint=detokenizer_endpoint,
        )
        self._tokenizer_endpoint = self._worker_manager.tokenizer_endpoint
        self._detokenizer_endpoint = self._worker_manager.detokenizer_endpoint
        self._worker_startup_timeout = self._worker_manager._startup_timeout

        if isinstance(model, str):
            backend = jax.default_backend()
            preferred_attn_mechanism = (
                AttentionMechanisms.UNIFIED_ATTENTION
                if backend == "gpu"
                else AttentionMechanisms.RAGGED_PAGE_ATTENTION_V3
            )
            user_config_kwargs = dict(self.loading_kwargs.config_kwargs or {})
            user_provided_attn = "attn_mechanism" in user_config_kwargs
            requested_attn = user_config_kwargs.get("attn_mechanism") if user_provided_attn else None
            if requested_attn is None:
                user_provided_attn = False

            if user_provided_attn:
                attn_value = (
                    requested_attn.value
                    if isinstance(requested_attn, AttentionMechanisms)
                    else str(requested_attn)
                    if requested_attn is not None
                    else None
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

            sharding_axis_names_resolved = tuple(self.loading_kwargs.sharding_axis_names)
            if self.data_parallelism_axis not in sharding_axis_names_resolved:
                logger.warning(
                    "`data_parallelism_axis=%r` not found in `sharding_axis_names=%r`; "
                    "KV page sharding will behave as unsharded for that axis.",
                    self.data_parallelism_axis,
                    sharding_axis_names_resolved,
                )

            user_config_kwargs.pop("attn_mechanism", None)

            loading = self.loading_kwargs.to_dict()
            loading["dtype"] = dtype
            loading["param_dtype"] = dtype
            loading["precision"] = jax.lax.Precision.DEFAULT
            loading["sharding_axis_dims"] = tuple(self.loading_kwargs.sharding_axis_dims)
            loading["sharding_axis_names"] = sharding_axis_names_resolved
            loading["config_kwargs"] = EasyDeLBaseConfigDict(
                attn_mechanism=requested_attn if user_provided_attn else preferred_attn_mechanism,
                attn_dtype=dtype,
                kvdtype=dtype,
                freq_max_position_embeddings=self.runtime_config.max_model_len,
                mask_max_position_embeddings=self.runtime_config.max_model_len,
                **user_config_kwargs,
            )

            model = AutoEasyDeLModelForCausalLM.from_pretrained(**loading)
            text_config = model.config.get_text_config()
            has_mla_attention, has_non_mla_attention = _detect_mla_attention_mix(model, text_config)

            _num_heads = getattr(text_config, "num_attention_heads", 0) or 0
            _mla_kernel_compatible = int(_num_heads) > 0

            if has_mla_attention and _mla_kernel_compatible:
                attn_value = _normalize_attn_mechanism_value(getattr(text_config, "attn_mechanism", None))
                mla_compatible = attn_value in _MLA_RAGGED_ATTN_MECHANISMS
                if jax.default_backend() == "gpu":
                    mla_compatible = mla_compatible or attn_value in {
                        AttentionMechanisms.UNIFIED_ATTENTION.value,
                        AttentionMechanisms.PAGED_FLASH_ATTENTION.value,
                    }
                if not mla_compatible:
                    if has_non_mla_attention:
                        logger.warning(
                            "Mixed MLA and non-MLA full-attention layers detected, "
                            "but forcing all inference layers to "
                            f"{MLA_RAGGED_ATTN_MECHANISM!r}."
                        )
                    logger.info(
                        "MLA architecture detected; forcing inference attention mechanism to "
                        f"{MLA_RAGGED_ATTN_MECHANISM!r}."
                    )
                    compat_graphdef = model.new_graphdef(
                        recursive_update=True,
                        attn_mechanism=MLA_RAGGED_ATTN_MECHANISM,
                        decode_attn_mechanism=MLA_RAGGED_ATTN_MECHANISM,
                        mla_attn_mechanism=MLA_RAGGED_ATTN_MECHANISM,
                    )
                    model = model.merge_module(compat_graphdef, model.graphstate, model.graphother)
            elif has_mla_attention and not _mla_kernel_compatible:
                fallback_attn = (
                    AttentionMechanisms.UNIFIED_ATTENTION
                    if jax.default_backend() == "gpu"
                    else AttentionMechanisms.RAGGED_PAGE_ATTENTION_V3
                )
                logger.info(
                    f"MLA architecture detected but num_attention_heads <= 0; falling back to {fallback_attn.value!r}."
                )
                compat_graphdef = model.new_graphdef(
                    recursive_update=True,
                    attn_mechanism=fallback_attn,
                    decode_attn_mechanism=fallback_attn,
                )
                model = model.merge_module(compat_graphdef, model.graphstate, model.graphother)

        self._apply_data_parallel_axis_to_model(model)

        if self._multimodal_manager is not None and self._multimodal_manager.model is None:
            self._multimodal_manager.model = model

        detected_model_type = getattr(getattr(model, "config", None), "model_type", None)
        tool_parser = self.parsing_config.tool_parser
        if tool_parser is None:
            tool_parser = self._auto_detect_tool_parser(
                tokenizer=self.tokenizer,
                model_type=detected_model_type,
            )
        reasoning_parser = self.parsing_config.reasoning_parser
        if reasoning_parser is None:
            reasoning_parser = self._auto_detect_reasoning_parser_name(
                tokenizer=self.tokenizer,
                model_type=detected_model_type,
            )

        self.tool_parser = tool_parser
        self.reasoning_parser_name = reasoning_parser
        self._tool_parser_class = None
        self._reasoning_parser_class = None

        if tool_parser:
            try:
                from easydel.inference.tools import ToolParserManager

                self._tool_parser_class = ToolParserManager.get_tool_parser(tool_parser)
                if not self.parsing_config.silent_mode:
                    logger.info("Initialized tool parser: %s", tool_parser)
            except KeyError:
                logger.warning("Tool parser '%s' not found, function calling disabled", tool_parser)

        if reasoning_parser:
            try:
                from easydel.inference.reasoning import ReasoningParserManager

                self._reasoning_parser_class = ReasoningParserManager.get_reasoning_parser(reasoning_parser)
                if not self.parsing_config.silent_mode:
                    logger.info("Initialized reasoning parser: %s", reasoning_parser)
            except KeyError:
                logger.warning("Reasoning parser '%s' not found, reasoning disabled", reasoning_parser)

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
        self._possible_name = self._get_model_name(model)

        self.runner = eSurgeRunner(
            model=model.esurge_compatible_model,
            hbm_utilization=self.cache_config.hbm_utilization,
            page_size=self.cache_config.page_size,
            pipeline_inference=self.runtime_config.pipeline_inference,
            max_cache_tokens=self.cache_config.max_cache_tokens,
            cache_capacity_margin=self.cache_config.cache_capacity_margin,
            kernel_tile_policy=self.runtime_config.kernel_tile_policy,
            max_model_len=self.runtime_config.max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_window_aware_runtime_cap=self.runtime_config.enable_window_aware_runtime_cap,
            min_input_pad=self.runtime_config.min_input_pad,
            max_num_seqs=self.runtime_config.max_num_seqs,
            max_num_seq_buckets=self.runtime_config.max_num_seq_buckets,
            async_scheduling=self.runtime_config.async_scheduling,
            min_token_pad=self.runtime_config.min_token_pad,
            use_aot_forward=self.runtime_config.use_aot_forward,
            bind_graphstate_for_aot=self.runtime_config.bind_graphstate_for_aot,
            verbose=self.runtime_config.runner_verbose,
            enable_overlap_execution=self.runtime_config.overlap_execution,
            enable_sampler_metrics=self.runtime_config.sampler_metrics,
            mpmd_scheduler=self.runtime_config.mpmd_scheduler,
        )

        if self.runtime_config.compile_runner:
            # Limit compilation to the scheduler's per-step token budget when provided.
            # This avoids compiling long-context token buckets (e.g. 32K/64K) when
            # the scheduler will only ever emit smaller batches (e.g. 512/2048).
            self.runner.compile(max_num_batched_tokens=max_num_batched_tokens)

        self.scheduler = Scheduler.from_runner(
            self.runner,
            max_num_batched_tokens=max_num_batched_tokens,
            enable_prefix_caching=self.cache_config.enable_prefix_caching,
            async_scheduling=self.runtime_config.async_scheduling,
            long_prefill_token_threshold=self.runtime_config.long_prefill_token_threshold,
        )
        self._scheduler_max_num_batched_tokens = max_num_batched_tokens

        # Streaming decode cadence
        self.decode_interval_tokens = DEFAULT_DECODE_INTERVAL_TOKENS
        self.decode_interval_secs = DEFAULT_DECODE_INTERVAL_SECS

        # State
        self._request_counter = 0
        self._active_requests: dict[str, dict] = {}
        self._request_outputs: dict[str, RequestOutput] = {}
        self._max_request_outputs = self.worker_config.max_request_outputs
        self._finished_request_ids: deque[str] = deque()

        # Per-request events to support many concurrent streams
        self._request_events: dict[str, threading.Event] = {}
        self.extra_eos_token_ids = self.parsing_config.extra_eos_token_ids or []
        self.extra_stops = self._normalize_stop_sequences(self.parsing_config.extra_stops)
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

        generation_config = getattr(model, "generation_config", None)
        if isinstance(generation_config, dict):
            self._generation_config_dict = dict(generation_config)
        elif generation_config is not None and hasattr(generation_config, "to_dict"):
            try:
                self._generation_config_dict = generation_config.to_dict()
            except Exception:
                self._generation_config_dict = {}
                logger.debug("Failed to serialize model generation_config; continuing without it", exc_info=True)
        else:
            self._generation_config_dict = {}

        def _coerce_token_ids(raw: Any) -> list[int]:
            if raw is None:
                return []
            candidates = raw if isinstance(raw, (list, tuple, set)) else [raw]
            token_ids: list[int] = []
            for candidate in candidates:
                if candidate is None:
                    continue
                try:
                    token_id = int(candidate)
                except (TypeError, ValueError):
                    logger.debug("Ignoring non-integer EOS token candidate: %r", candidate)
                    continue
                token_ids.append(token_id)
            return token_ids

        tokenizer_eos_ids = _coerce_token_ids(getattr(self.tokenizer, "eos_token_id", None))
        generation_config_eos_ids = _coerce_token_ids(self._generation_config_dict.get("eos_token_id"))
        engine_extra_eos_ids = _coerce_token_ids(self.extra_eos_token_ids)

        combined_eos_ids: list[int] = []
        seen_eos_ids: set[int] = set()
        for token_id in [*tokenizer_eos_ids, *generation_config_eos_ids, *engine_extra_eos_ids]:
            if token_id in seen_eos_ids:
                continue
            seen_eos_ids.add(token_id)
            combined_eos_ids.append(token_id)

        self._generation_config_eos_token_ids = generation_config_eos_ids
        self.__eos_ids = combined_eos_ids
        self.__eos_set = set(combined_eos_ids)
        self._primary_eos_token_id = self.__eos_ids[0] if self.__eos_ids else None
        # Publicly-named aliases for mixins/helpers to avoid class-name mangling.
        self._eos_ids = self.__eos_ids
        self._eos_set = self.__eos_set

        if self.distributed_config.distributed_mode:
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
            self._distributed_controller = DistributedController(
                enabled=True,
                role=self.distributed_role,
                rank=self.distributed_rank,
                world_size=self.distributed_world_size,
                service_name=self.distributed_config.distributed_service_name,
                control_port=self.distributed_config.distributed_control_port,
                control_bind_host=self.distributed_config.distributed_control_bind_host,
                advertise_addr=self.distributed_config.distributed_advertise_addr,
                auth_token=self.distributed_config.distributed_auth_token,
                step_timeout_s=self.distributed_config.distributed_step_timeout_s,
                connect_timeout_s=self.distributed_config.distributed_connect_timeout_s,
                verify_sampling_digest=self.distributed_config.distributed_verify_sampling_digest,
                config_fingerprint=self._distributed_config_fingerprint,
                execute_step=self._distributed_execute_step,
            )
        else:
            self._distributed_config_fingerprint = None

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
        return self.runtime_config.esurge_name or self._possible_name

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

        eSurge's KV-page parallelism uses the dedicated ``ATTN_DP`` semantic
        axis registered during engine setup. Rewriting the model's standard
        data-parallel axis here can alias DP with EP and break MoE shard maps.
        """
        del model

    def _distributed_execute_step(self, scheduler_output):
        """Execute a single scheduler step on worker ranks via control-plane RPC."""

        with self._scheduler_lock:
            return self.runner.execute_model(scheduler_output)

    def _instantiate_reasoning_parser_for_metadata(self):
        """Build a short-lived reasoning parser instance for token metadata lookups."""
        if self._reasoning_parser_class is None or self.tokenizer is None:
            return None
        try:
            return self._reasoning_parser_class(self.tokenizer)
        except Exception:
            return None

    def _resolve_reasoning_boundary_token(self, attr_name: str) -> str | None:
        """Resolve a reasoning boundary token from parser metadata when available."""
        parser = self._instantiate_reasoning_parser_for_metadata()
        return self._find_str_attr(parser, attr_name)

    def _find_str_attr(self, parser, attr_name: str) -> str | None:
        """Search parser, its delegate, and the parser class for a non-empty string attribute."""
        candidates = (parser, getattr(parser, "_delegate", None), self._reasoning_parser_class)
        for candidate in candidates:
            token = getattr(candidate, attr_name, None)
            if isinstance(token, str) and token:
                return token
        return None

    def _resolve_reasoning_boundary_token_id(self, attr_name: str, token_attr_name: str) -> int | None:
        """Resolve a reasoning boundary token ID from parser metadata or tokenizer vocab."""
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
        if getattr(self, "_distributed_controller", None) is not None:
            try:
                self._distributed_controller.shutdown()
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
            f"distributed_mode={self.distributed_config.distributed_mode}",
            f"distributed_role={self.distributed_role!r}",
            f"distributed_rank={self.distributed_rank}",
            f"distributed_world_size={self.distributed_world_size}",
            f"scheduler_running={self._scheduler_running}",
        ]
        return "eSurge(\n  " + ",\n  ".join(attrs) + "\n)"
