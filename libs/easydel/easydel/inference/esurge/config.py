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

import itertools
import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Literal, NotRequired, TypeAlias, TypedDict, Unpack

from ejkernel.modules.operations import KernelTilePolicy, normalize_kernel_tile_policy
from spectrax.common_types import NOT_GIVEN, _Empty

from easydel.typings import typed_config

if TYPE_CHECKING:
    from easydel.infra.etils import MpMdSchedulers
else:
    MpMdSchedulers: type[object] = object

LONG_PREFILL_TRS: int = 2048
ESURGE_MIN_TOKEN_PAD: int = 16
PPMicrobatchPolicy: TypeAlias = int | Literal["auto"] | None


def token_bucket_floor(max_num_seqs: int, limit: int = ESURGE_MIN_TOKEN_PAD) -> int:
    """Largest power-of-two decode bucket not exceeding concurrency or limit."""
    capped = min(int(max_num_seqs), int(limit))
    if capped <= 0:
        raise ValueError(f"max_num_seqs must be positive, got {max_num_seqs}")
    return 1 << (capped.bit_length() - 1)


def normalize_token_bucket_minimum(value: int, max_model_len: int) -> int:
    """Round a token-bucket minimum to a supported power of two.

    ``WindowPlanner`` builds a power-of-two ladder.  Explicit non-power-of-two
    padding minima therefore need normalization too, not only minima derived
    from request concurrency.  If rounding upward would exceed the model
    length, use its largest representable power-of-two bucket.
    """
    value = min(int(value), int(max_model_len))
    if value <= 0:
        raise ValueError(f"token bucket minimum must be positive, got {value}")
    rounded = 1 << (value - 1).bit_length()
    if rounded > int(max_model_len):
        rounded = 1 << (int(max_model_len).bit_length() - 1)
    return rounded


def _normalize_pp_microbatch_policy(value: Any, *, field_name: str) -> PPMicrobatchPolicy:
    """Normalize PP microbatch runtime knobs.

    ``"auto"`` preserves the built-in policy, ``None`` or ``0`` disables the
    wavefront path, and a positive integer pins either count or rows per
    microbatch depending on the field being normalized.

    Args:
        value: Raw user-supplied value. Accepts ``None``, ``"auto"`` /
            ``"none"`` / ``"off"`` / ``"disable"`` (case insensitive), a
            numeric string, or any non-negative integer.
        field_name: Name of the config field being normalized, used in
            error messages.

    Returns:
        ``"auto"`` to defer to the built-in policy, ``None`` to disable
        wavefront microbatching, or a positive integer to pin the
        microbatch knob.

    Raises:
        ValueError: If ``value`` is a string that cannot be coerced to
            one of the recognized literals or a non-negative integer, or
            if a negative integer is supplied.
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
    if self.min_token_pad is not None:
        # A decode step carries at most one token per active request. Never
        # force its smallest token graph above the configured concurrency.
        token_floor = token_bucket_floor(self.max_num_seqs)
        self.min_token_pad = normalize_token_bucket_minimum(
            max(int(self.min_token_pad), token_floor), self.max_model_len
        )
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
            positive when set. Decode token buckets are clamped to at least
            16 tokens for normal model lengths, matching the TPU serving path
            used by decode benchmarks.
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
            bucket. When ``False``, the runner still warms the configured
            ``spx.jit`` buckets during ``compile()`` and stores those warmed
            callables. AOT yields lower per-step host overhead but longer
            cold-start.
        compile_runner: When ``True``, runner-side helper kernels and
            bucketed model executables are pre-compiled at engine start.
            Setting to ``False`` skips this startup compile; callers must run
            ``runner.compile()`` before serving buckets that will be requested.
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
        "compile_vision_encoder": True,
        "vision_patch_buckets": None,
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
        compile_vision_encoder: When ``True``, eSurge precompiles a bucketed
            JIT helper for the VLM vision tower / projector and uses it during
            multimodal prefill. Models with non-JIT-safe vision paths fall back
            to the eager path.
        vision_patch_buckets: Optional raw patch-count buckets for the vision
            precompile helper. ``None`` derives powers-of-two buckets from the
            runtime token budget and the model's spatial merge size.
    """

    resolution_buckets: NotRequired[list[tuple[int, int]] | None]
    vision_cache_capacity_mb: NotRequired[int]
    compile_vision_encoder: NotRequired[bool]
    vision_patch_buckets: NotRequired[list[int] | None]

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
        "coordination": "replicated",
        "distributed_service_name": None,
        "distributed_leader_addr": None,
        "distributed_control_port": 19666,
        "distributed_control_bind_host": "0.0.0.0",
        "distributed_auth_token": None,
        "distributed_step_timeout_s": 30.0,
        "distributed_connect_timeout_s": 600.0,
        "distributed_ready_timeout_s": 600.0,
        "distributed_heartbeat_interval_s": 1.0,
        "distributed_heartbeat_timeout_s": 5.0,
        "distributed_verify_digest_interval": 64,
        "distributed_max_inflight_steps": 4,
        "distributed_admit_timeout_s": 120.0,
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
        coordination: Multi-host step-coordination pattern.
            ``"replicated"`` (default) assumes an outer driver calls the
            engine identically on every host in deterministic lockstep — the
            trainer-rollout pattern; the engine adds no control plane.
            ``"zmq"`` builds the leader/worker step-replication plane
            (single-ingress serving): rank 0 owns the scheduler and mirrors
            its runner-call stream to every worker over ZeroMQ, which is
            required whenever requests arrive at one host only (e.g. the
            HTTP API server on a pod). Ignored when
            ``jax.process_count() == 1``.
        distributed_service_name: Service / DNS name used by workers to
            discover the leader. ``None`` skips discovery and requires the
            leader address to be configured out-of-band.
        distributed_control_port: TCP port the control plane binds (leader)
            or connects to (worker). Default ``19666``.
        distributed_control_bind_host: Interface the control server binds
            on. Default ``"0.0.0.0"`` accepts connections on all
            interfaces; restrict for security.
        distributed_auth_token: Shared bearer token required on every
            control-plane RPC, and identical on every host. **Mandatory
            whenever** ``coordination="zmq"``: the step coordinator refuses to
            start without one rather than running the plane unauthenticated,
            because that plane mirrors the leader's runner calls and anything
            that can reach the control port can drive the model. ``None`` is
            accepted only for ``coordination="replicated"``, which has no
            control plane to authenticate.
        distributed_step_timeout_s: Per-step RPC timeout in seconds. The
            leader aborts a step when no quorum of workers responds before
            this deadline. Default ``30.0``.
        distributed_connect_timeout_s: Timeout for the initial worker→leader
            connect handshake at engine startup. Default ``600.0``, generous
            for the same reason ``distributed_ready_timeout_s`` is: the leader
            cannot answer a hello until it has loaded its weights, and that is
            minutes for any model worth serving on several hosts. The previous
            ``15.0`` assumed a leader that is listening almost immediately,
            which describes no real startup -- it failed a 2x v5p-16 multislice
            run with ``no HelloOk from leader within 15.0s`` while rank 0 was
            still reading 15GB off disk. Workers boot independently on a
            multislice allocation, so the skew between a worker's first attempt
            and the leader being ready is a startup race rather than a fault.
            A dead leader is caught by the heartbeat once the plane is up, so
            waiting longer here costs nothing but a slower report of a genuine
            failure.
        distributed_leader_addr: Explicit leader host/IP for the
            ``coordination="zmq"`` plane. ``None`` falls back to the JAX
            distributed coordinator's host (correct for standard pod
            launches), then DNS discovery via ``distributed_service_name``.
        distributed_ready_timeout_s: How long the ``coordination="zmq"``
            leader waits at startup for every worker to hello + ready.
            Compilation happens before ready, so this defaults generously
            (``600.0`` seconds).
        distributed_heartbeat_interval_s: Liveness beacon cadence on the
            ``coordination="zmq"`` plane. Default ``1.0``.
        distributed_heartbeat_timeout_s: Peer silence tolerated before the
            pod aborts. Default ``5.0``.
        distributed_verify_digest_interval: Every K steps the
            ``coordination="zmq"`` plane cross-checks a sampled-token digest
            between hosts, off the step critical path. ``0`` disables.
            Default ``64``.
        distributed_max_inflight_steps: Bound on leader/worker step skew for
            the ``coordination="zmq"`` plane; the leader stops dispatching
            when the slowest worker falls this many steps behind. Default
            ``4``.
        distributed_admit_timeout_s: How long a non-leader rank blocks on
            the owner's acknowledgement when forwarding a request admission
            through the unified request plane. The first admissions overlap
            owner-side warmup compilation, so this defaults generously
            (``120.0`` seconds).
    """

    coordination: NotRequired[Literal["replicated", "zmq"]]
    distributed_service_name: NotRequired[str | None]
    distributed_leader_addr: NotRequired[str | None]
    distributed_control_port: NotRequired[int]
    distributed_control_bind_host: NotRequired[str]
    distributed_auth_token: NotRequired[str | None]
    distributed_step_timeout_s: NotRequired[float]
    distributed_connect_timeout_s: NotRequired[float]
    distributed_ready_timeout_s: NotRequired[float]
    distributed_heartbeat_interval_s: NotRequired[float]
    distributed_heartbeat_timeout_s: NotRequired[float]
    distributed_verify_digest_interval: NotRequired[int]
    distributed_max_inflight_steps: NotRequired[int]
    distributed_admit_timeout_s: NotRequired[float]

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


DraftSchedule: TypeAlias = list[tuple[int, int, int]]
"""Normalized dynamic draft-token schedule: ``[(start, end, k), ...]``.

Each entry maps the inclusive live-concurrency range ``[start, end]`` (the
number of requests scheduled in a step) to the number of draft tokens ``k``
proposed per verify window at that concurrency. ``k == 0`` disables
speculative drafting for the range. This mirrors vLLM's
``num_speculative_tokens_per_batch_size`` schema.
"""


def normalize_draft_schedule(
    schedule: Sequence[Sequence[int]],
    *,
    max_num_draft_tokens: int | None = None,
) -> DraftSchedule:
    """Validate and normalize a ``num_draft_tokens_per_batch_size`` schedule.

    Args:
        schedule: Iterable of ``(start, end, k)`` triples (tuples or lists,
            e.g. straight out of YAML). ``start``/``end`` are the inclusive
            batch-size (live-concurrency) bounds of the range; ``k`` is the
            draft-token count for that range (``0`` = speculation off).
        max_num_draft_tokens: When given, every ``k`` must be ``<=`` this
            static ``num_draft_tokens`` (compiled buckets and the drafter
            cache are sized for the static maximum).

    Returns:
        The normalized schedule: a list of int 3-tuples sorted by ``start``.

    Raises:
        ValueError: If an entry is not a length-3 sequence of ints, if
            ``start < 1`` or ``end < start``, if ``k < 0`` or
            ``k > max_num_draft_tokens``, or if two ranges overlap.
    """
    entries: list[tuple[int, int, int]] = []
    for raw_entry in schedule:
        entry = tuple(raw_entry)
        if len(entry) != 3:
            raise ValueError(
                f"num_draft_tokens_per_batch_size entries must be (start_batch_size, end_batch_size, "
                f"num_draft_tokens) triples, got {raw_entry!r}"
            )
        start, end, k = (int(v) for v in entry)
        if start < 1:
            raise ValueError(f"num_draft_tokens_per_batch_size range start must be >= 1, got {start}")
        if end < start:
            raise ValueError(f"num_draft_tokens_per_batch_size range end must be >= start, got ({start}, {end})")
        if k < 0:
            raise ValueError(f"num_draft_tokens_per_batch_size draft count must be >= 0, got {k}")
        if max_num_draft_tokens is not None and k > int(max_num_draft_tokens):
            raise ValueError(
                f"num_draft_tokens_per_batch_size draft count {k} exceeds num_draft_tokens="
                f"{int(max_num_draft_tokens)}; the static num_draft_tokens is the compiled maximum."
            )
        entries.append((start, end, k))
    entries.sort(key=lambda item: item[0])
    for (prev_start, prev_end, _prev_k), (next_start, next_end, _next_k) in itertools.pairwise(entries):
        if next_start <= prev_end:
            raise ValueError(
                "num_draft_tokens_per_batch_size ranges must not overlap: "
                f"[{prev_start}, {prev_end}] overlaps [{next_start}, {next_end}]"
            )
    return entries


def resolve_num_draft_tokens(
    schedule: DraftSchedule | None,
    *,
    batch_size: int,
    num_draft_tokens: int,
) -> int:
    """Resolve the live draft-token count ``k`` for a step's concurrency.

    Range bounds are inclusive on both ends. Batch sizes not covered by any
    range (gaps, or beyond the last range) fall back to the static
    ``num_draft_tokens`` — a schedule only overrides where it speaks.

    Args:
        schedule: A normalized schedule (see :func:`normalize_draft_schedule`)
            or ``None`` for static behavior.
        batch_size: Live concurrency (number of requests scheduled this step);
            clamped up to ``1``.
        num_draft_tokens: The static draft-token count (gap fallback and
            upper clamp).

    Returns:
        The resolved ``k`` in ``[0, num_draft_tokens]``.
    """
    static_k = max(0, int(num_draft_tokens))
    if not schedule:
        return static_k
    live = max(1, int(batch_size))
    for start, end, k in schedule:
        if start <= live <= end:
            return min(int(k), static_k)
    return static_k


def default_auto_draft_schedule(num_draft_tokens: int) -> DraftSchedule:
    """Default dynamic schedule for the plug-and-play ``drafter=True`` path.

    Measured on the current inline-MTP head (Qwen3.6-27B GDN hybrid, v5p-8,
    decode-only spec-on/spec-off ratios: batch-1 1.25x, n=8 0.98x, n=16 0.93x,
    n=32 0.92x): drafting wins clearly at very low concurrency and turns into
    a net loss well before saturation, so the auto path drafts only up to
    concurrency 4 and switches speculation OFF above that. The thresholds are
    NOT universal — they are tuned for the inline-MTP drafter measured above
    and are fully overridable via
    ``eSurgeDrafterConfig.num_draft_tokens_per_batch_size``.

    Args:
        num_draft_tokens: The static draft-token count ``K`` used for the
            low-concurrency range.

    Returns:
        ``[(1, 4, K), (5, 2**30, 0)]``.
    """
    return [(1, 4, max(0, int(num_draft_tokens))), (5, 1 << 30, 0)]


def _validate_esurge_drafter_config(self):
    """Validate and normalize an :class:`eSurgeDrafterConfig` after construction.

    Wired via ``post_init`` of the ``@typed_config`` decorator. Mutates ``self``
    in place: canonicalizes ``method`` to a slug (lowercase, non-alphanumeric
    runs collapsed to underscores), reconciles ``enabled`` with ``method``
    (a recognized disable-token clears the method and disables drafting, while
    an enabled config with no method defaults ``method`` to ``"auto"``), coerces
    ``layer_mapping`` entries to ``int``, normalizes
    ``num_draft_tokens_per_batch_size`` via :func:`normalize_draft_schedule`,
    and normalizes ``kwargs`` to a plain ``dict`` (``None`` becomes ``{}``).

    Args:
        self: The newly-built ``eSurgeDrafterConfig`` (a dict subclass) whose
            ``enabled``, ``method``, ``num_draft_tokens``,
            ``num_draft_tokens_per_batch_size``, ``layer_mapping``, and
            ``kwargs`` fields are validated and normalized in place.

    Raises:
        ValueError: If ``num_draft_tokens`` is not positive, or if
            ``num_draft_tokens_per_batch_size`` has malformed/overlapping
            ranges or a draft count above ``num_draft_tokens``.
        TypeError: If ``kwargs`` is neither ``None`` nor a mapping.
    """
    method = self.method
    enabled = bool(self.enabled)
    if method is not None:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(method).strip().lower()).strip("_")
        if normalized in {"", "none", "off", "false", "disable", "disabled"}:
            method = None
            enabled = False
        else:
            method = normalized
            enabled = True

    if enabled and method is None:
        method = "auto"

    num_draft_tokens = int(self.num_draft_tokens)
    if num_draft_tokens <= 0:
        raise ValueError(f"num_draft_tokens must be positive, got {num_draft_tokens}")

    if self.num_draft_tokens_per_batch_size is not None:
        self.num_draft_tokens_per_batch_size = normalize_draft_schedule(
            self.num_draft_tokens_per_batch_size,
            max_num_draft_tokens=num_draft_tokens,
        )

    if self.layer_mapping is not None:
        self.layer_mapping = [int(layer_idx) for layer_idx in self.layer_mapping]

    if self.kwargs is None:
        self.kwargs = {}
    elif not isinstance(self.kwargs, Mapping):
        raise TypeError(f"kwargs must be a mapping or None, got {type(self.kwargs).__name__}")
    else:
        self.kwargs = dict(self.kwargs)

    self.enabled = enabled
    self.method = method
    self.num_draft_tokens = num_draft_tokens


@typed_config(
    defaults={
        "enabled": False,
        "method": None,
        "num_draft_tokens": 4,
        "num_draft_tokens_per_batch_size": None,
        "assistant_model": None,
        "target_embed_module": None,
        "layer_mapping": None,
        "kwargs": None,
    },
    post_init=_validate_esurge_drafter_config,
)
class eSurgeDrafterConfig(TypedDict, total=False):
    """Speculative drafter construction config for eSurge.

    This is the declarative form of ``model.drafter(...)``. Passing it to
    :class:`eSurge` lets the engine build the drafter from the loaded model
    instead of requiring callers to instantiate a drafter class by hand.

    Examples:
        >>> eSurgeDrafterConfig.from_dict(method="mtp", num_draft_tokens=4)
        >>> eSurgeDrafterConfig.from_dict(
        ...     method="gemma4_assistant",
        ...     assistant_model=assistant,
        ...     num_draft_tokens=4,
        ... )

    Attributes:
        enabled: Enable drafter construction. A non-empty ``method`` also
            enables the config. ``False`` / ``method=None`` disables drafting.
        method: Drafter family passed to ``model.drafter``. Supported runtime
            values depend on the model, currently ``"auto"``, ``"mtp"``, and
            ``"gemma4_assistant"``.
        num_draft_tokens: Draft tokens proposed per verify window (default 4,
            the measured sweet spot for inline-MTP drafting with KV
            persistence).
        num_draft_tokens_per_batch_size: Optional dynamic draft-token
            schedule — vLLM's ``num_speculative_tokens_per_batch_size``. A
            list of inclusive ``(start_batch_size, end_batch_size,
            num_draft_tokens)`` ranges resolved against the live concurrency
            (requests scheduled per step): e.g. ``[(1, 4, 4), (5, 2**30, 0)]``
            drafts 4 tokens up to concurrency 4 and turns speculation off
            above it. ``k == 0`` disables drafting for the range; batch sizes
            in gaps fall back to the static ``num_draft_tokens``. Every ``k``
            must be ``<= num_draft_tokens`` and ranges must not overlap.
            ``None`` (default) keeps the static ``num_draft_tokens`` at every
            concurrency. Note the engine's plug-and-play ``drafter=True``
            path installs :func:`default_auto_draft_schedule` when this is
            unset.
        assistant_model: Optional standalone assistant model or model id/path
            for assistant-style drafting.
        target_embed_module: Optional target embedding module forwarded to
            assistant drafters.
        layer_mapping: Optional assistant-layer to target-layer mapping.
        kwargs: Extra keyword arguments forwarded to ``model.drafter``.
    """

    enabled: NotRequired[bool]
    method: NotRequired[str | None]
    num_draft_tokens: NotRequired[int]
    num_draft_tokens_per_batch_size: NotRequired[list[tuple[int, int, int]] | None]
    assistant_model: NotRequired[Any | None]
    target_embed_module: NotRequired[Any | None]
    layer_mapping: NotRequired[list[int] | tuple[int, ...] | None]
    kwargs: NotRequired[dict[str, Any] | None]

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None = None,
        **kwargs: Unpack["eSurgeDrafterConfig"],
    ) -> "eSurgeDrafterConfig":
        """Build an :class:`eSurgeDrafterConfig` from a mapping/kwargs.

        Args:
            data: Optional source mapping; keys override defaults, and
                ``kwargs`` override ``data``.
            **kwargs: Any field of :class:`eSurgeDrafterConfig` consumed
                via ``Unpack[eSurgeDrafterConfig]``.

        Returns:
            A validated ``eSurgeDrafterConfig`` instance.
        """
        ...


def _validate_esurge_config(self) -> None:
    """Fill absent sections with their all-defaults instances."""
    if self.runtime is None:
        self.runtime = eSurgeRuntimeConfig.coerce_config(None)
    if self.cache is None:
        self.cache = eSurgeCacheRuntimeConfig.coerce_config(None)
    if self.context is None:
        self.context = eSurgeContextConfig.coerce_config(None)
    if self.workers is None:
        self.workers = eSurgeWorkerConfig.coerce_config(None)
    if self.parsing is None:
        self.parsing = eSurgeParsingConfig.coerce_config(None)
    if self.vision is None:
        self.vision = eSurgeVisionConfig.coerce_config(None)
    if self.distributed is None:
        self.distributed = eSurgeDistributedConfig.coerce_config(None)
    if self.drafter is None:
        self.drafter = eSurgeDrafterConfig.coerce_config(None)


@typed_config(
    defaults={
        "runtime": None,
        "cache": None,
        "context": None,
        "workers": None,
        "parsing": None,
        "vision": None,
        "distributed": None,
        "drafter": None,
    },
    post_init=_validate_esurge_config,
)
class eSurgeConfig(TypedDict, total=False):
    """Aggregate of every eSurge config section under one object.

    One mapping — e.g. the ``esurge:`` block of an eLarge YAML — coerces into
    all eight sections in a single call. Sections omitted from the input are
    filled with their defaults, and nested mappings are promoted to their
    typed sections recursively::

        >>> cfg = eSurgeConfig.from_dict(
        ...     runtime={"max_model_len": 8192, "max_num_seqs": 16},
        ...     cache={"page_size": 128},
        ... )
        >>> cfg.runtime.max_num_seqs
        16
        >>> engine = eSurge(model="model-id", config=cfg)

    Attributes:
        runtime: Runtime/execution section (:class:`eSurgeRuntimeConfig`).
        cache: KV-cache section (:class:`eSurgeCacheRuntimeConfig`).
        context: Context-window section (:class:`eSurgeContextConfig`).
        workers: Tokenizer/detokenizer worker section (:class:`eSurgeWorkerConfig`).
        parsing: Tool/reasoning/sampling parser section (:class:`eSurgeParsingConfig`).
        vision: Multimodal section (:class:`eSurgeVisionConfig`).
        distributed: Multi-host serving section (:class:`eSurgeDistributedConfig`).
        drafter: Speculative-decoding drafter section (:class:`eSurgeDrafterConfig`).
    """

    runtime: NotRequired[eSurgeRuntimeConfig | None]
    cache: NotRequired[eSurgeCacheRuntimeConfig | None]
    context: NotRequired[eSurgeContextConfig | None]
    workers: NotRequired[eSurgeWorkerConfig | None]
    parsing: NotRequired[eSurgeParsingConfig | None]
    vision: NotRequired[eSurgeVisionConfig | None]
    distributed: NotRequired[eSurgeDistributedConfig | None]
    drafter: NotRequired[eSurgeDrafterConfig | None]

    if TYPE_CHECKING:

        @classmethod
        def from_dict(
            cls,
            data: Mapping[str, Any] | None = None,
            **kwargs: Unpack[eSurgeConfig],
        ) -> eSurgeConfig:
            """Build an aggregate config from a mapping and/or keyword sections.

            Args:
                data: Optional mapping of section name to section
                    mapping/instance.
                **kwargs: Section overrides by name.

            Returns:
                A coerced ``eSurgeConfig`` with every section populated.
            """
            ...

        @classmethod
        def coerce_config(cls, value: eSurgeConfig | Mapping[str, Any] | None = None) -> eSurgeConfig:
            """Coerce ``value`` into an ``eSurgeConfig`` (idempotent).

            Args:
                value: An existing instance, a plain mapping, or ``None``
                    for all-defaults.

            Returns:
                A validated ``eSurgeConfig`` instance.
            """
            ...
