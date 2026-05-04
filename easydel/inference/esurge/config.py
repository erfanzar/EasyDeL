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

"""Configuration classes for the eSurge inference engine.

Each class is a ``TypedDict`` decorated with :func:`easydel.typings.typed_config`.
That keeps a single source of truth for the schema:

- Type checkers see a ``TypedDict`` (so ``Unpack[eSurgeRuntimeConfig]`` works in ``**kwargs``).
- At runtime, ``Cls.from_dict(**kwargs)`` returns a ``ConfigDict`` instance — a
  ``dict`` subclass with attribute access, ``to_dict()``, and ``replace(**)``.

Example:
    >>> from easydel.inference.esurge.config import eSurgeRuntimeConfig
    >>> runtime = eSurgeRuntimeConfig.from_dict(max_num_seqs=16, max_model_len=8192)
    >>> runtime.max_num_seqs       # attribute access
    16
    >>> runtime["max_num_seqs"]    # dict access
    16
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypedDict, Unpack

from spectrax.common_types import NOT_GIVEN, _Empty

from easydel.operations.kernels._gdn_policy import KernelTilePolicy, normalize_kernel_tile_policy
from easydel.typings import typed_config

if TYPE_CHECKING:
    from easydel.infra.etils import MpMdSchedulers
else:
    MpMdSchedulers = object

LONG_PREFILL_TRS: int = 2048
PPMicrobatchPolicy = int | Literal["auto"] | None


def _normalize_pp_microbatch_policy(value: Any, *, field_name: str) -> PPMicrobatchPolicy:
    """Normalize PP microbatch runtime knobs.

    ``"auto"`` preserves the built-in policy, ``None`` or ``0`` disables the
    wavefront path, and a positive integer pins either count or rows per
    microbatch depending on the field being normalized.
    """
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "auto":
            return "auto"
        if lowered in {"none", "off", "disable", "disabled"}:
            return None
        try:
            value = int(lowered)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be 'auto', None, 0, or a positive integer; got {value!r}") from exc
    value = int(value)
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative, got {value}")
    return None if value == 0 else value


def _validate_esurge_runtime_config(self):
    """Validate and normalize an :class:`eSurgeRuntimeConfig` after construction.

    Wired via ``post_init`` of the ``@typed_config`` decorator. Mutates ``self``
    in place to canonicalize ``kernel_tile_policy``.

    Args:
        self: The newly-built ``eSurgeRuntimeConfig`` (a dict subclass) whose
            fields are validated and normalized.

    Raises:
        ValueError: If any of ``max_model_len``, ``min_input_pad``,
            ``min_token_pad``, ``max_num_seqs``, ``max_num_batched_tokens``, or
            ``long_prefill_token_threshold`` violates positivity /
            non-negativity invariants, or if the kernel-tile policy string is
            invalid.
    """
    if self.max_model_len <= 0:
        raise ValueError(f"max_model_len must be positive, got {self.max_model_len}")
    if self.min_input_pad <= 0:
        raise ValueError(f"min_input_pad must be positive, got {self.min_input_pad}")
    if self.min_token_pad is not None and self.min_token_pad <= 0:
        raise ValueError(f"min_token_pad must be positive when specified, got {self.min_token_pad}")
    if self.max_num_seqs <= 0:
        raise ValueError(f"max_num_seqs must be positive, got {self.max_num_seqs}")
    if self.max_num_batched_tokens is not NOT_GIVEN and self.max_num_batched_tokens is not None:
        if self.max_num_batched_tokens <= 0:
            raise ValueError(f"max_num_batched_tokens must be positive, got {self.max_num_batched_tokens}")
    if self.long_prefill_token_threshold is not None and self.long_prefill_token_threshold < 0:
        raise ValueError(f"long_prefill_token_threshold must be non-negative, got {self.long_prefill_token_threshold}")
    self.pp_microbatch_count = _normalize_pp_microbatch_policy(
        self.pp_microbatch_count,
        field_name="pp_microbatch_count",
    )
    self.pp_microbatch_size = _normalize_pp_microbatch_policy(
        self.pp_microbatch_size,
        field_name="pp_microbatch_size",
    )
    if self.pp_microbatch_count not in ("auto", None) and self.pp_microbatch_size not in ("auto", None):
        raise ValueError("Only one of pp_microbatch_count or pp_microbatch_size may be set to a positive integer.")
    self.kernel_tile_policy = normalize_kernel_tile_policy(self.kernel_tile_policy)


@typed_config(
    defaults={
        "max_model_len": 8192,
        "esurge_name": None,
        "kernel_tile_policy": "auto",
        "min_input_pad": 16,
        "min_token_pad": None,
        "max_num_seqs": 256,
        "max_num_seq_buckets": None,
        "async_scheduling": True,
        "max_num_batched_tokens": NOT_GIVEN,
        "use_aot_forward": True,
        "compile_runner": True,
        "runner_verbose": False,
        "overlap_execution": True,
        "sampler_metrics": False,
        "long_prefill_token_threshold": None,
        "enable_window_aware_runtime_cap": False,
        "mpmd_scheduler": None,
        "pp_microbatch_count": "auto",
        "pp_microbatch_size": "auto",
    },
    post_init=_validate_esurge_runtime_config,
)
class eSurgeRuntimeConfig(TypedDict, total=False):
    """Runtime and execution configuration for the eSurge inference engine.

    Carries every knob that controls how the engine drives the model forward
    pass: bucket layout for batch / token padding, AOT vs JIT compilation,
    overlap of host scheduling with device dispatch, the optional MPMD
    pipeline-parallel mode, and the sliding-window-aware cap that derives the
    runtime concurrency limit from live KV demand.

    All fields are optional (``NotRequired``) at the type level; defaults are
    injected by the ``@typed_config`` decorator. Use
    :meth:`eSurgeRuntimeConfig.from_dict` (or pass an instance directly to
    :class:`eSurge`) to construct one. The post-init validator
    :func:`_validate_esurge_runtime_config` enforces positivity / range
    invariants on the integer fields and normalizes ``kernel_tile_policy`` to
    its canonical literal.

    Attributes:
        esurge_name: Optional human-readable engine name embedded into
            Prometheus metric labels and dashboard headers; useful when
            multiple engines run in the same process. ``None`` falls back
            to the model's repo id.
        kernel_tile_policy: Tile-shape selection policy for the Pallas/GDN
            inference kernels. ``"auto"`` defers to per-backend heuristics;
            other values pin a specific tile recipe (see
            :func:`normalize_kernel_tile_policy`).
        max_model_len: Hard upper bound on per-request total sequence length
            (prompt tokens + generated tokens). Used to size the page-table
            allocation and clamp individual request budgets. Must be positive.
        min_input_pad: Minimum element of the request-count bucket ladder.
            Smaller request batches are padded up to this floor before
            looking up a compiled executable; raising it reduces the number
            of compiled buckets but increases padding overhead.
        min_token_pad: Optional floor on the *token-count* bucket ladder.
            ``None`` defers to ``min_input_pad``. Set this explicitly when the
            request-count floor and token-count floor should differ. Must be
            positive when set.
        max_num_seqs: Hard ceiling on concurrent in-flight sequences. The
            actual runtime concurrency may be smaller when KV pages are
            scarce.
        max_num_seq_buckets: Explicit list of bucket sizes for the
            request-count axis. ``None`` builds an exponential ladder from
            ``min_input_pad`` up to ``max_num_seqs``. Explicit buckets may be
            larger than ``max_num_seqs`` when matching another serving
            runtime's static request padding; scheduler admission remains
            capped by ``max_num_seqs``.
        async_scheduling: When ``True``, the scheduler runs on a background
            thread so it can produce the next batch while the device finishes
            the previous one. Disable for deterministic step ordering or when
            debugging scheduler-side races. PP MPMD keeps the requested value;
            stage-to-stage sampled-token handoff belongs in the runner/runtime,
            not in config policy that silently changes user intent.
        max_num_batched_tokens: Per-scheduler-step token budget. ``NOT_GIVEN``
            keeps the framework default (auto-sized from the cache metadata);
            ``None`` falls back to ``max_model_len``. Must be positive when
            an explicit value is provided.
        use_aot_forward: When ``True``, the model forward is lowered and
            compiled ahead of time per ``(num_tokens, padded_num_reqs)``
            bucket. When ``False``, ``spx.jit`` traces lazily on first use.
            AOT yields lower per-step host overhead but longer cold-start.
        compile_runner: When ``True``, runner-side helper kernels and
            bucketed model executables are pre-compiled at engine start.
            Setting to ``False`` defers compilation to the first matching
            request, which trades cold-start time for first-request latency.
        runner_verbose: Emit per-step runner log lines (perf counters,
            bucket selection, pipeline timings) at INFO instead of DEBUG.
        overlap_execution: When ``True``, the lifecycle loop dispatches the
            next scheduler step while the previous device step is still in
            flight. Mutually exclusive with multi-host distributed mode
            (the lockstep control plane requires deterministic step ordering).
            With ``async_scheduling`` enabled, the async-handle lifecycle loop
            is used for both TP/SPMD and PP decode; the runner decides per step
            whether device-resident sampled-token handoff is safe, otherwise it
            drains before launching the next step.
        sampler_metrics: When ``True``, the sampler emits per-step
            log-probability tensors so downstream code can record token-level
            metrics. Adds an extra D2H copy each step.
        long_prefill_token_threshold: Token count above which a single
            prompt is split into multiple chunked-prefill steps to avoid
            blocking decode requests. ``None`` disables chunked prefill;
            value must be non-negative.
        enable_window_aware_runtime_cap: When ``True``, the runner derives
            the runtime request cap from per-attention-type page demand
            inferred from the live cache metadata, instead of trusting the
            metadata's heuristic ``get_max_num_seqs()``. Useful for hybrid
            sliding-window models.
        mpmd_scheduler: Optional pre-built ``MpMdSchedulers`` instance.
            When provided, scheduled training-style MPMD runs can reuse the
            same schedule object inside the inference forward pass. ``None``
            uses the forward-only marker-cluster path.
        pp_microbatch_count: Expert PP decode wavefront knob. ``"auto"``
            keeps the built-in policy, ``None`` / ``0`` disables wavefront
            microbatching, and a positive integer pins the maximum number of
            decode microbatches to launch per active window.
        pp_microbatch_size: Expert PP decode wavefront knob. ``"auto"``
            keeps the built-in policy, ``None`` / ``0`` disables wavefront
            microbatching, and a positive integer pins rows per microbatch.
            Mutually exclusive with a positive ``pp_microbatch_count``.
    """

    esurge_name: NotRequired[str | None]
    kernel_tile_policy: NotRequired[KernelTilePolicy]
    max_model_len: NotRequired[int]
    min_input_pad: NotRequired[int]
    min_token_pad: NotRequired[int | None]
    max_num_seqs: NotRequired[int]
    max_num_seq_buckets: NotRequired[list[int] | tuple[int, ...] | None]
    async_scheduling: NotRequired[bool]
    max_num_batched_tokens: NotRequired[int | None | _Empty]
    use_aot_forward: NotRequired[bool]
    compile_runner: NotRequired[bool]
    runner_verbose: NotRequired[bool]
    overlap_execution: NotRequired[bool]
    sampler_metrics: NotRequired[bool]
    long_prefill_token_threshold: NotRequired[int | None]
    enable_window_aware_runtime_cap: NotRequired[bool]
    mpmd_scheduler: NotRequired[MpMdSchedulers | None]
    pp_microbatch_count: NotRequired[PPMicrobatchPolicy]
    pp_microbatch_size: NotRequired[PPMicrobatchPolicy]

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None = None,
        **kwargs: Unpack["eSurgeRuntimeConfig"],
    ) -> "eSurgeRuntimeConfig":
        """Build an :class:`eSurgeRuntimeConfig` from a mapping and/or kwargs.

        Args:
            data: Optional source mapping. Keys override class defaults; ``kwargs``
                override ``data``.
            **kwargs: Any field of :class:`eSurgeRuntimeConfig` (consumed via
                ``Unpack[eSurgeRuntimeConfig]``) — see the class docstring for
                the full schema.

        Returns:
            A validated ``eSurgeRuntimeConfig`` (``ConfigDict`` subclass) with
            attribute and dict access.
        """
        ...


def _validate_esurge_cache_runtime_config(self):
    """Validate an :class:`eSurgeCacheRuntimeConfig` after construction.

    Wired via ``post_init`` of the ``@typed_config`` decorator.

    Args:
        self: The newly-built ``eSurgeCacheRuntimeConfig`` instance whose
            fields are checked.

    Raises:
        ValueError: If ``hbm_utilization`` or ``cache_capacity_margin`` is not
            in ``(0, 1]``, ``page_size`` is non-positive, or
            ``max_cache_tokens`` is non-positive when set.
    """
    if not (0.0 < float(self.hbm_utilization) <= 1.0):
        raise ValueError(f"hbm_utilization must be in (0, 1], got {self.hbm_utilization}")
    if self.page_size <= 0:
        raise ValueError(f"page_size must be positive, got {self.page_size}")
    if self.max_cache_tokens is not None and self.max_cache_tokens <= 0:
        raise ValueError(f"max_cache_tokens must be positive when specified, got {self.max_cache_tokens}")
    if not (0.0 < float(self.cache_capacity_margin) <= 1.0):
        raise ValueError(f"cache_capacity_margin must be in (0, 1], got {self.cache_capacity_margin}")


@typed_config(
    defaults={
        "hbm_utilization": 0.85,
        "page_size": 128,
        "enable_prefix_caching": True,
        "max_cache_tokens": None,
        "cache_capacity_margin": 0.92,
        "data_parallelism_axis": "dp",
        "destroy_pages_on_pause": True,
    },
    post_init=_validate_esurge_cache_runtime_config,
)
class eSurgeCacheRuntimeConfig(TypedDict, total=False):
    """KV-cache, prefix-cache, and paging configuration for the eSurge engine.

    Drives the static page allocation that backs the engine's paged-attention
    kernels. The number of allocated pages is ultimately determined by
    ``hbm_utilization`` (a fraction of free HBM after weights/activations) and
    optionally clamped by ``max_cache_tokens``; both are then scaled down by
    ``cache_capacity_margin`` to leave headroom for transient buffers. Prefix
    caching reuses pages across requests when prompt prefixes match, and the
    DP axis name controls how those pages are sharded across data-parallel
    ranks.

    Validated by :func:`_validate_esurge_cache_runtime_config` on construction.

    Attributes:
        hbm_utilization: Target fraction of available HBM (after model weights)
            to dedicate to the KV cache. Must be in ``(0, 1]``. Higher values
            give more concurrent sequences at the cost of activation headroom.
        page_size: Tokens stored per KV-cache page. Larger pages cut page-table
            indirection cost and improve attention kernel throughput, smaller
            pages reduce internal fragmentation. Must be positive.
        enable_prefix_caching: When ``True``, identical prompt prefixes hit
            already-resident pages and skip re-prefill, saving both compute
            and HBM. Disable for benchmarking or when prompts are
            cache-busting.
        max_cache_tokens: Optional absolute upper bound on the total tokens
            the page pool may hold. ``None`` lets HBM utilization decide.
            Must be positive when set; the page count is rounded up to fit
            this bound and then clamped by ``cache_capacity_margin``.
        cache_capacity_margin: Multiplicative safety factor in ``(0, 1]``
            applied to the HBM-derived page count. Reserves headroom for
            non-KV allocations (activations, sampler scratch, profiler
            traces). Lower values squeeze more pages out of HBM at higher
            risk of OOM.
        data_parallelism_axis: Name of the mesh axis used to shard KV pages
            across data-parallel ranks. The same axis is used by
            :mod:`easydel.inference.esurge.core.dp_sharding` to derive
            per-shard page bounds.
        destroy_pages_on_pause: When ``True``, calling :meth:`eSurge.pause`
            frees the entire page pool to reclaim HBM. Useful when the engine
            shares HBM with other workloads. The pages are reallocated by
            :meth:`eSurge.resume`.
    """

    hbm_utilization: NotRequired[float]
    page_size: NotRequired[int]
    enable_prefix_caching: NotRequired[bool]
    max_cache_tokens: NotRequired[int | None]
    cache_capacity_margin: NotRequired[float]
    data_parallelism_axis: NotRequired[str]
    destroy_pages_on_pause: NotRequired[bool]

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None = None,
        **kwargs: Unpack["eSurgeCacheRuntimeConfig"],
    ) -> "eSurgeCacheRuntimeConfig":
        """Build an :class:`eSurgeCacheRuntimeConfig` from a mapping/kwargs.

        Args:
            data: Optional source mapping; keys override defaults, kwargs
                override ``data``.
            **kwargs: Any field of :class:`eSurgeCacheRuntimeConfig` consumed
                via ``Unpack[eSurgeCacheRuntimeConfig]``.

        Returns:
            A validated ``eSurgeCacheRuntimeConfig`` instance.
        """
        ...


@typed_config(
    defaults={
        "reserve_tokens": None,
        "auto_truncate_prompt": True,
        "auto_cap_new_tokens": True,
        "strict_context": False,
        "truncate_mode": "left",
        "prefer_preserve_prompt": True,
        "decode_truncated_prompt": True,
    },
)
class eSurgeContextConfig(TypedDict, total=False):
    """Context-window-overflow handling for the eSurge engine.

    Centralizes the policy the engine uses when a request's
    ``len(prompt_token_ids) + sampling_params.max_tokens`` would exceed
    ``max_model_len``. The defaults silently truncate-from-the-left and cap
    ``max_tokens`` so most clients "just work"; flip ``strict_context`` to
    surface the overflow as a ``ValueError`` instead.

    Attributes:
        reserve_tokens: Lower bound on tokens kept available for generation
            after prompt truncation. ``None`` falls back to ``max_num_seqs``,
            which is a conservative default that guarantees at least one
            decode step per concurrent slot.
        auto_truncate_prompt: When ``True``, prompts longer than the
            allowable budget are truncated according to ``truncate_mode``.
            When ``False``, an over-long prompt either raises (when
            ``strict_context``) or causes the request to fail at scheduling
            time.
        auto_cap_new_tokens: When ``True``, the requested ``max_new_tokens``
            is silently shrunk to ``max_model_len - len(prompt)`` (after any
            prompt truncation). When ``False``, an over-budget completion
            request is rejected.
        strict_context: When ``True``, both ``auto_truncate_prompt`` and
            ``auto_cap_new_tokens`` are disabled and overflow surfaces as
            ``ValueError`` so the caller can decide how to handle it.
        truncate_mode: Direction of prompt truncation when
            ``auto_truncate_prompt`` triggers. ``"left"`` drops the oldest
            tokens (recommended for chat), ``"right"`` drops the newest,
            ``"middle"`` removes a contiguous middle slice while preserving
            both ends.
        prefer_preserve_prompt: Tiebreaker when both prompt and generation
            budget could be shrunk. When ``True``, prompt tokens are
            preserved and ``max_new_tokens`` is reduced first; when
            ``False``, the prompt is truncated first.
        decode_truncated_prompt: When ``True``, the (possibly truncated)
            prompt is re-decoded so request metadata and ``echo`` output
            reflect what the model actually saw. When ``False``, the
            original prompt string is reported back unchanged.
    """

    reserve_tokens: NotRequired[int | None]
    auto_truncate_prompt: NotRequired[bool]
    auto_cap_new_tokens: NotRequired[bool]
    strict_context: NotRequired[bool]
    truncate_mode: NotRequired[Literal["left", "right", "middle"]]
    prefer_preserve_prompt: NotRequired[bool]
    decode_truncated_prompt: NotRequired[bool]

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None = None,
        **kwargs: Unpack["eSurgeContextConfig"],
    ) -> "eSurgeContextConfig":
        """Build an :class:`eSurgeContextConfig` from a mapping/kwargs.

        Args:
            data: Optional source mapping; keys override defaults.
            **kwargs: Any field of :class:`eSurgeContextConfig` consumed via
                ``Unpack[eSurgeContextConfig]``.

        Returns:
            A validated ``eSurgeContextConfig`` instance.
        """
        ...


@typed_config(
    defaults={
        "detokenizer_max_states": 1 << 16,
        "tokenizer_endpoint": None,
        "detokenizer_endpoint": None,
        "worker_startup_timeout": None,
        "max_request_outputs": 1000,
        "idle_reset_seconds": None,
        "idle_reset_min_interval": 60.0,
    },
)
class eSurgeWorkerConfig(TypedDict, total=False):
    """Tokenizer / detokenizer worker pool and idle-reset configuration.

    Tokenization and streaming detokenization run in helper threads (or
    out-of-process workers when an endpoint is configured) so they don't
    serialize on the GIL with the scheduler thread. This config sizes those
    helpers and the in-memory output retention buffer, and controls the
    optional automatic-reset that fires after the engine is idle for a while.

    Attributes:
        detokenizer_max_states: Maximum number of concurrent streaming
            detokenizer states the worker pool may keep alive (one per
            in-flight request). Pool spills over to LRU eviction once the
            limit is reached. Default is intentionally generous (``2**16``)
            because each state is small.
        tokenizer_endpoint: HTTP/RPC endpoint of an external tokenizer
            worker. ``None`` runs tokenization in-process. Useful for
            offloading expensive tokenizers (e.g. SentencePiece) off the
            scheduler thread or the host process entirely.
        detokenizer_endpoint: Same as ``tokenizer_endpoint`` but for the
            detokenizer side, which is invoked once per generated chunk.
        worker_startup_timeout: Seconds to wait for an external worker to
            become healthy before the engine fails to initiate. ``None``
            uses the framework default. Increase for slow cold-starts (e.g.
            container image pulls).
        max_request_outputs: Maximum number of finished ``RequestOutput``
            objects retained in the engine's ring buffer for late polling
            via the OpenAI-compatible API. ``None`` keeps all outputs
            until the engine shuts down (only safe for short-lived
            processes).
        idle_reset_seconds: Number of seconds without scheduler activity
            after which the engine automatically destroys KV pages and
            resets runner buffers. ``None`` disables the watchdog.
        idle_reset_min_interval: Minimum seconds between two consecutive
            idle resets, even if the engine repeatedly crosses the idle
            threshold. Prevents thrashing on bursty workloads.
    """

    detokenizer_max_states: NotRequired[int]
    tokenizer_endpoint: NotRequired[str | None]
    detokenizer_endpoint: NotRequired[str | None]
    worker_startup_timeout: NotRequired[float | None]
    max_request_outputs: NotRequired[int | None]
    idle_reset_seconds: NotRequired[float | None]
    idle_reset_min_interval: NotRequired[float]

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None = None,
        **kwargs: Unpack["eSurgeWorkerConfig"],
    ) -> "eSurgeWorkerConfig":
        """Build an :class:`eSurgeWorkerConfig` from a mapping/kwargs.

        Args:
            data: Optional source mapping; keys override defaults.
            **kwargs: Any field of :class:`eSurgeWorkerConfig` consumed via
                ``Unpack[eSurgeWorkerConfig]``.

        Returns:
            A validated ``eSurgeWorkerConfig`` instance.
        """
        ...


@typed_config(
    defaults={
        "sampling_params_callback": None,
        "extra_eos_token_ids": None,
        "extra_stops": None,
        "ignore_stop_strings_in_reasoning": True,
        "silent_mode": False,
        "tool_parser": None,
        "reasoning_parser": None,
    },
)
class eSurgeParsingConfig(TypedDict, total=False):
    """Tool-call / reasoning-block parser configuration and sampling hook.

    Connects the engine to the OpenAI-compatible serving layer: tool / reasoning
    parsers translate raw model output into structured ``tool_calls`` and
    ``reasoning_content`` fields, and ``extra_*`` knobs let the caller extend
    the model's stop conditions without subclassing the tokenizer. The
    ``sampling_params_callback`` provides a single hook for per-request
    customization (e.g. injecting a logits processor) that runs before the
    request is admitted.

    Attributes:
        sampling_params_callback: Optional ``(SamplingParams, request_metadata)
            -> SamplingParams`` callable invoked once per submitted request.
            Lets callers mutate (or replace) the per-request sampling params
            after they have been validated but before the scheduler sees
            them. ``None`` skips the hook.
        extra_eos_token_ids: Additional integer token ids appended to the
            tokenizer's default EOS set when checking for stop conditions.
            ``None`` keeps just the tokenizer defaults.
        extra_stops: Additional string or list of strings to treat as stop
            sequences after detokenization. Takes effect on top of any
            request-level ``stop`` field.
        ignore_stop_strings_in_reasoning: When ``True``, stop strings are
            suppressed while the model is still inside a reasoning block
            (delimited by the configured reasoning parser). Prevents
            premature termination when the chain-of-thought happens to
            contain the stop token.
        silent_mode: When ``True``, drops engine info-level log output. The
            scheduler still logs warnings and errors. Useful for embedding
            the engine inside a larger CLI.
        tool_parser: Name of the tool parser (registered in
            :mod:`easydel.inference.tool_parsers`) or ``None`` to auto-detect
            from the model's chat template metadata.
        reasoning_parser: Name of the reasoning parser (registered in
            :mod:`easydel.inference.reasoning_parsers`) or ``None`` to
            auto-detect.
    """

    sampling_params_callback: NotRequired[Any]
    extra_eos_token_ids: NotRequired[list[int] | None]
    extra_stops: NotRequired[str | list[str] | None]
    ignore_stop_strings_in_reasoning: NotRequired[bool]
    silent_mode: NotRequired[bool]
    tool_parser: NotRequired[Any | None]
    reasoning_parser: NotRequired[Any | None]

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None = None,
        **kwargs: Unpack["eSurgeParsingConfig"],
    ) -> "eSurgeParsingConfig":
        """Build an :class:`eSurgeParsingConfig` from a mapping/kwargs.

        Args:
            data: Optional source mapping; keys override defaults.
            **kwargs: Any field of :class:`eSurgeParsingConfig` consumed via
                ``Unpack[eSurgeParsingConfig]``.

        Returns:
            A validated ``eSurgeParsingConfig`` instance.
        """
        ...


@typed_config(
    defaults={
        "resolution_buckets": None,
        "vision_cache_capacity_mb": 1024,
    },
)
class eSurgeVisionConfig(TypedDict, total=False):
    """Multimodal / vision-encoder configuration for VLM-capable engines.

    Vision encoders are typically run once per request before the language
    model attends over the resulting image features. To keep AOT compilation
    feasible we discretize input resolutions into a small set of buckets and
    cache the encoded features keyed by image hash + bucket.

    Attributes:
        resolution_buckets: Explicit list of ``(height, width)`` pairs used
            both for vision-encoder pre-compilation and as the keys of the
            feature cache. Inputs are resized / padded to the smallest
            bucket that fits. ``None`` disables bucketing entirely (each
            unique resolution traces a fresh executable).
        vision_cache_capacity_mb: Maximum size of the vision-feature cache,
            in megabytes. The cache is LRU-evicted; setting this to ``0``
            effectively disables caching, which can help when the working
            set of images is much larger than HBM can hold.
    """

    resolution_buckets: NotRequired[list[tuple[int, int]] | None]
    vision_cache_capacity_mb: NotRequired[int]

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None = None,
        **kwargs: Unpack["eSurgeVisionConfig"],
    ) -> "eSurgeVisionConfig":
        """Build an :class:`eSurgeVisionConfig` from a mapping/kwargs.

        Args:
            data: Optional source mapping; keys override defaults.
            **kwargs: Any field of :class:`eSurgeVisionConfig` consumed via
                ``Unpack[eSurgeVisionConfig]``.

        Returns:
            A validated ``eSurgeVisionConfig`` instance.
        """
        ...


@typed_config(
    defaults={
        "distributed_mode": False,
        "distributed_role": "auto",
        "distributed_service_name": None,
        "distributed_world_size": None,
        "distributed_rank": None,
        "distributed_control_port": 19666,
        "distributed_control_bind_host": "0.0.0.0",
        "distributed_advertise_addr": None,
        "distributed_auth_token": None,
        "distributed_step_timeout_s": 30.0,
        "distributed_connect_timeout_s": 15.0,
        "distributed_verify_sampling_digest": True,
    },
)
class eSurgeDistributedConfig(TypedDict, total=False):
    """Multi-host distributed-serving configuration.

    Wires the engine's lockstep control plane: a leader rank owns the
    scheduler and dispatches each step to one or more worker ranks via the
    HTTP control server in :mod:`easydel.inference.esurge.distributed`.
    Workers run the same forward pass on their shard and return a sampling
    digest that the leader cross-checks before committing the step.

    Attributes:
        distributed_mode: Master switch enabling the multi-host control
            plane. When ``False`` all other distributed_* fields are ignored
            and the engine runs single-process (still supports SPMD/MPMD
            within the local process).
        distributed_role: Role of this rank: ``"leader"`` runs the scheduler
            and serves requests, ``"worker"`` runs the runner only and waits
            for dispatch. ``"auto"`` infers the role from
            ``distributed_rank`` (rank 0 → leader, others → worker).
        distributed_service_name: Service / DNS name used by workers to
            discover the leader. ``None`` skips discovery and requires the
            leader address to be configured out-of-band.
        distributed_world_size: Total number of participating ranks.
            Required when ``distributed_mode=True``; may be left ``None``
            when role is auto-resolved from ``jax.process_count()``.
        distributed_rank: This process's rank id. ``None`` falls back to
            ``jax.process_index()`` so SPMD launchers work without extra
            configuration.
        distributed_control_port: TCP port the control plane binds (leader)
            or connects to (worker). Default ``19666``.
        distributed_control_bind_host: Interface the control server binds
            on. Default ``"0.0.0.0"`` accepts connections on all
            interfaces; restrict for security.
        distributed_advertise_addr: Address the leader advertises to workers
            during discovery. ``None`` uses the auto-detected outbound
            address; set explicitly when running behind NAT or in
            container networks.
        distributed_auth_token: Shared bearer token required on every
            control-plane RPC. ``None`` disables auth (only safe on
            isolated networks).
        distributed_step_timeout_s: Per-step RPC timeout in seconds. The
            leader aborts a step when no quorum of workers responds before
            this deadline. Default ``30.0``.
        distributed_connect_timeout_s: Timeout for the initial worker→leader
            connect handshake at engine startup. Default ``15.0``.
        distributed_verify_sampling_digest: When ``True`` (default), every
            step round-trips a cryptographic digest of the sampled tokens
            between ranks and aborts if they disagree. Set ``False`` only
            for benchmarking / development.
    """

    distributed_mode: NotRequired[bool]
    distributed_role: NotRequired[Literal["auto", "leader", "worker"]]
    distributed_service_name: NotRequired[str | None]
    distributed_world_size: NotRequired[int | None]
    distributed_rank: NotRequired[int | None]
    distributed_control_port: NotRequired[int]
    distributed_control_bind_host: NotRequired[str]
    distributed_advertise_addr: NotRequired[str | None]
    distributed_auth_token: NotRequired[str | None]
    distributed_step_timeout_s: NotRequired[float]
    distributed_connect_timeout_s: NotRequired[float]
    distributed_verify_sampling_digest: NotRequired[bool]

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None = None,
        **kwargs: Unpack["eSurgeDistributedConfig"],
    ) -> "eSurgeDistributedConfig":
        """Build an :class:`eSurgeDistributedConfig` from a mapping/kwargs.

        Args:
            data: Optional source mapping; keys override defaults.
            **kwargs: Any field of :class:`eSurgeDistributedConfig` consumed
                via ``Unpack[eSurgeDistributedConfig]``.

        Returns:
            A validated ``eSurgeDistributedConfig`` instance.
        """
        ...
